//! M2 存储引擎集成测试（README 16.4 / 17 之 M2）。
//!
//! 对着里程碑那一行逐条验收：`崩溃恢复与加密测试全过；磁盘无明文；万行数据查询正确`。
//!
//! 临时库一律放在 `%TEMP%\mightbe_m2\<tag>`，用例之间不复用目录，失败可复现。

use mightbe_store::api::{ColumnDef, Row, Slot};
use mightbe_store::crypto::KdfParams;
use mightbe_store::mdb::Database;
use mightbe_store::page::PAGE_SIZE;
use mightbe_store::table::col;
use mightbe_store::table::{
    count_rows, delete_row, encode_row, get_row, insert_row, scan_table,
};
use std::path::{Path, PathBuf};

const PW: &[u8] = b"correct horse battery staple";

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join("mightbe_m2").join(tag);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
}

fn col(name: &str, ty: u8) -> ColumnDef {
    ColumnDef {
        name: name.to_string(),
        col_type: ty,
        primary_key: false,
    }
}

/// 建一张 `docs(id, title, score)` 表；返回实例。表本身在 `create_table` 内已提交。
fn seeded(d: &Path) -> Database {
    let mut db = Database::create(d, PW, &KdfParams::fast()).expect("建库");
    db.create_table(
        "docs",
        vec![col("id", col::I64), col("title", col::TEXT), col("score", col::F64)],
    )
    .expect("建表");
    db
}

/// 建库 + 写 200 行 + 关库。
fn seed_small(d: &Path) {
    let mut db = seeded(d);
    let mut def = db.table_def("docs").expect("表");
    db.begin().unwrap();
    for i in 0..200u64 {
        insert_row(
            &mut db,
            &mut def,
            &Row {
                slots: vec![
                    Slot::I64(i as i64),
                    Slot::Text(format!("alpha-beta-gamma-{i}")),
                    Slot::F64(i as f64 * 0.5),
                ],
            },
        )
        .unwrap();
    }
    db.commit().unwrap();
    db.close().expect("关库");
}

/// 全表扫描，取每行的整数槽，按 rid 有序返回。
fn ints(db: &mut Database, table: &str) -> Vec<i64> {
    let def = db.table_def(table).expect("表应存在");
    scan_table(db, &def)
        .expect("扫描")
        .into_iter()
        .map(|(_, row)| match &row.slots[0] {
            Slot::I64(v) => *v,
            other => panic!("第 0 槽不是整数而是 {other:?}"),
        })
        .collect()
}

/// 全表内容按 rid 顺序编码成字节串，用于"轮换/恢复前后逐字节一致"这类比对。
fn dumped(db: &mut Database, table: &str) -> Vec<Vec<u8>> {
    let def = db.table_def(table).expect("表应存在");
    scan_table(db, &def)
        .expect("扫描")
        .into_iter()
        .map(|(_, row)| encode_row(&row))
        .collect()
}

// ─────────────────────────── Catalog 与 CRUD ───────────────────────────

#[test]
fn catalog_survives_reopen() {
    let d = tmp("catalog");
    {
        let db = seeded(&d);
        db.close().expect("关库");
    }

    let mut db = Database::open(&d, PW).expect("重开");
    let def = db.table_def("docs").expect("Catalog 应被保住");
    assert_eq!(def.name, "docs");
    assert_eq!(def.next_rid, 1);
    assert_eq!(def.columns.len(), 3);
    assert_eq!(def.column_index("score"), Some(2));
    assert!(db.table_def("nope").is_err(), "不存在的表要报错，而不是返回空");
    db.close().expect("关库");
}

#[test]
fn ten_k_rows_scan_back_after_reopen() {
    let d = tmp("tenk");
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..10_000u64 {
            let row = Row {
                slots: vec![
                    Slot::I64(i as i64),
                    Slot::Text(format!("doc-title-{i}")),
                    Slot::F64(i as f64 * 0.5),
                ],
            };
            let rid = insert_row(&mut db, &mut def, &row).expect("插入");
            assert_eq!(rid, i + 1, "rid 必须连续自增，既不跳号也不重复");
        }
        db.commit().expect("提交");
        assert!(
            db.n_pages() > 5,
            "一万行应该撑出多条页链，实际只有 {} 页",
            db.n_pages()
        );
    } // 刻意不 close：落盘的那一版必须能独立打开，否则后面全白测

    let mut db = Database::open(&d, PW).expect("重开");
    let def = db.table_def("docs").expect("表");
    assert_eq!(count_rows(&mut db, &def).unwrap(), 10_000, "万行查询行数");

    let rows = scan_table(&mut db, &def).expect("全表扫描");
    assert_eq!(rows.len(), 10_000);
    for (rid, row) in &rows {
        let i = rid - 1;
        match (&row.slots[0], &row.slots[1], &row.slots[2]) {
            (Slot::I64(v), Slot::Text(t), Slot::F64(f)) => {
                assert_eq!(*v, i as i64, "第 {rid} 行的 id");
                assert_eq!(t, &format!("doc-title-{i}"), "第 {rid} 行的标题");
                assert!((*f - i as f64 * 0.5).abs() < 1e-6, "第 {rid} 行的分数");
            }
            other => panic!("槽位类型不对: {other:?}"),
        }
    }
    assert!(
        rows.windows(2).all(|w| w[0].0 < w[1].0),
        "扫描结果必须按 rid 有序"
    );

    for rid in [1u64, 4_321, 7_777, 10_000] {
        let row = get_row(&mut db, &def, rid)
            .expect("点查")
            .unwrap_or_else(|| panic!("rid {rid} 应该查得到"));
        assert_eq!(row.slots[1], Slot::Text(format!("doc-title-{}", rid - 1)));
    }
    assert!(get_row(&mut db, &def, 10_001).expect("点查").is_none());

    assert!(delete_row(&mut db, &def, 10_000).expect("删除"));
    assert!(!delete_row(&mut db, &def, 10_000).expect("重复删除应返回 false"));
    assert_eq!(count_rows(&mut db, &def).unwrap(), 9_999);

    db.drop_table("docs").expect("删表");
    assert!(db.table_def("docs").is_err(), "删表后 Catalog 里不该还有它");
    db.close().expect("关库");
}

// ─────────────────────────── 加密与完整性 ───────────────────────────

#[test]
fn one_flipped_bit_is_never_silently_absorbed() {
    let d = tmp("tamper");
    seed_small(&d);
    let data = d.join("data.mdb");

    // 1) 翻密文页 —— 开库当场失败（AEAD 先拦，CRC 是第二道）
    // 每一步都从**原始字节**重来：翻位是叠加的，留着上一步的损伤，
    // 第二步就变成"头页也坏了"，测的就不是数据页那条路径了。
    let orig = std::fs::read(&data).expect("读原始数据文件");
    let block = 80 + 3 * (PAGE_SIZE as u64 + 16); // 一个页块 = 页 + 16B AEAD 标签

    let mut b = orig.clone();
    b[80 + 5] ^= 0x01; // 文件头页（第 0 页）的密文区
    std::fs::write(&data, &b).expect("写回");
    let err = Database::open(&d, PW).err().expect("头块被翻一位后必须开不了库");
    let msg = format!("{err}");
    assert!(
        msg.contains("口令错误") || msg.contains("MDB"),
        "错误要指向口令/格式，而不是别的地方: {msg}"
    );

    // 2) 翻数据页 —— 开库也许还过得去（只读了元数据页），但碰到那一页必须炸
    let mut b = orig;
    b[(block + 64) as usize] ^= 0x01; // 第 3 页（表堆）的密文区
    std::fs::write(&data, &b).expect("写回");

    let mut db = Database::open(&d, PW).expect("只坏了一页元数据，开库应当仍可以");
    let def = db.table_def("docs").expect("Catalog 没受影响");
    let err = scan_table(&mut db, &def)
        .expect_err("碰到被翻过的页必须报错，绝不能读出脏数据");
    let msg = format!("{err}");
    assert!(
        msg.contains("解密失败") || msg.contains("校验"),
        "报错要指向解密/校验，而不是 panic: {msg}"
    );
}

#[test]
fn wrong_password_cannot_open_and_leaks_nothing() {
    let d = tmp("wrongpw");
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        insert_row(
            &mut db,
            &mut def,
            &Row { slots: vec![Slot::Text("inner-secret-8848".into())] },
        )
        .unwrap();
        db.commit().unwrap();
        db.close().unwrap();
    }

    let err = Database::open(&d, b"not-the-password")
        .err()
        .expect("错口令必须开不开库");
    let msg = format!("{err}");
    assert!(!msg.contains("inner-secret-8848"), "错误信息泄露了明文: {msg}");
    assert!(!msg.contains("alpha"), "错误信息泄露了库内容: {msg}");
    Database::open(&d, PW).expect("正确口令应能开库").close().unwrap();
}

#[test]
fn corpus_plaintext_absent_on_disk() {
    let d = tmp("noplaintext");
    let marker = "zq7q-grammar-corpus-token-2026";
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..64u64 {
            insert_row(
                &mut db,
                &mut def,
                &Row {
                    slots: vec![
                        Slot::I64(i as i64),
                        Slot::Text(format!("{marker}-{i}")),
                        Slot::F64(1.0),
                    ],
                },
            )
            .unwrap();
        }
        db.commit().unwrap();
        db.close().expect("关库");
    }

    let mut checked = 0usize;
    for entry in std::fs::read_dir(&d).expect("列目录") {
        let p = entry.expect("条目").path();
        let body = std::fs::read(&p).expect("读文件");
        assert!(
            !body.windows(marker.len()).any(|w| w == marker.as_bytes()),
            "{:?} 里出现了语料明文串",
            p.file_name().unwrap()
        );
        checked += 1;
    }
    assert!(checked >= 1, "一个文件都没检查到");

    // 顺带核对尺寸：4 个整页（头块 / 位图 / Catalog / 表根），不出现半页或额外尾块
    let size = std::fs::metadata(d.join("data.mdb")).expect("尺寸").len();
    assert_eq!(
        size,
        80 + 4 * (PAGE_SIZE as u64 + 16),
        "尺寸应等于 80 + n·(页 + 标签)"
    );
}

// ─────────────────────────── 崩溃恢复 ───────────────────────────

#[test]
fn crash_at_any_wal_point_keeps_only_committed_rows() {
    let d = tmp("kill");
    let wal = d.join("data.mlog");
    let committed_end;
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");

        db.begin().unwrap();
        for i in 1..=100u64 {
            insert_row(&mut db, &mut def, &Row { slots: vec![Slot::I64(i as i64)] }).unwrap();
        }
        db.commit().unwrap();
        committed_end = std::fs::metadata(&wal).expect("WAL 长度").len() as usize;
        assert!(committed_end > 0, "提交后日志应被截断回超级帧");

        // 第二笔：写到一半就断电（不 commit、不 close，直接 drop）
        db.begin().unwrap();
        for i in 101..=140u64 {
            insert_row(&mut db, &mut def, &Row { slots: vec![Slot::I64(i as i64)] }).unwrap();
        }
    }

    let wal_bytes = std::fs::read(&wal).expect("读 WAL");
    let total = wal_bytes.len();
    assert!(total > committed_end, "第二笔应该在日志里留下没提交的帧");

    // 在"已提交区段之后"的任意位置截断，都是一次合法的掉电时刻
    for frac in [0.0f64, 0.17, 0.5, 0.83] {
        let cut = committed_end + (((total - committed_end) as f64) * frac) as usize;
        std::fs::write(&wal, &wal_bytes[..cut]).expect("截断");

        let mut db = Database::open(&d, PW).expect("重开应成功");
        assert!(!db.wal_has_records(), "恢复完成后日志必须清空，不能留尾巴");
        assert_eq!(
            ints(&mut db, "docs"),
            (1..=100).collect::<Vec<i64>>(),
            "崩溃点 {cut}: 已提交的 100 行要在、未提交的 40 行一个都不许出现"
        );
        let def = db.table_def("docs").expect("表");
        assert_eq!(def.next_rid, 101, "next_rid 不得被未提交事务推进");
    }

    // 完整日志（一处都不截）同样只认已提交的部分
    std::fs::write(&wal, &wal_bytes).expect("还原日志");
    let mut db = Database::open(&d, PW).expect("重开");
    assert_eq!(ints(&mut db, "docs"), (1..=100).collect::<Vec<i64>>());
    db.close().unwrap();
}

#[test]
fn rollback_discards_the_transaction() {
    let d = tmp("rollback");
    let mut db = seeded(&d);
    let mut def = db.table_def("docs").expect("表");

    db.begin().unwrap();
    for i in 1..=10u64 {
        insert_row(&mut db, &mut def, &Row { slots: vec![Slot::I64(i as i64)] }).unwrap();
    }
    db.rollback().expect("回滚");
    assert!(!db.in_txn());
    assert!(ints(&mut db, "docs").is_empty(), "回滚后一行都不该留下");
    db.close().unwrap();

    let mut db = Database::open(&d, PW).expect("重开");
    assert!(ints(&mut db, "docs").is_empty(), "回滚不该被重放出来");
    db.close().unwrap();
}

#[test]
fn tampered_wal_frame_is_discarded() {
    let d = tmp("tamper_wal");
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 1..=20u64 {
            insert_row(&mut db, &mut def, &Row { slots: vec![Slot::I64(i as i64)] }).unwrap();
        }
        db.commit().unwrap();
        // 不 close：日志原样留在磁盘上，正是恢复要处理的情形
    }

    let wal = d.join("data.mlog");
    let mut bytes = std::fs::read(&wal).expect("读日志");
    let n = bytes.len();
    bytes[n - 12] ^= 0xff; // 撕掉最后一帧
    std::fs::write(&wal, &bytes).expect("写回");

    let mut db = Database::open(&d, PW).expect("重开");
    assert!(
        ints(&mut db, "docs").is_empty(),
        "日志被撕坏时重放必须整体止步，不能半套用"
    );
}

// ─────────────────────────── 密钥轮换 / 备份恢复 ───────────────────────────

#[test]
fn rotate_key_preserves_every_row_byte_for_byte() {
    let d = tmp("rotate");
    let before;
    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..500u64 {
            insert_row(
                &mut db,
                &mut def,
                &Row {
                    slots: vec![
                        Slot::I64(i as i64),
                        Slot::Text(format!("title-{i}")),
                        Slot::F64(i as f64 * 0.25),
                    ],
                },
            )
            .unwrap();
        }
        db.commit().unwrap();
        before = dumped(&mut db, "docs");
        let old_epoch = db.super_block().epoch;

        db.rotate_key(b"brand-new-password").expect("轮换");
        assert_ne!(
            db.super_block().epoch,
            old_epoch,
            "epoch 必须前进，作废旧 nonce 空间"
        );
        assert_ne!(db.super_block().kek_salt, old_epoch, "salt 必须换掉");
        db.close().expect("关库");
    }

    let mut db = Database::open(&d, b"brand-new-password").expect("新口令应能开库");
    assert_eq!(dumped(&mut db, "docs"), before, "解密后的内容必须逐字节一致");
    db.close().expect("关库");

    assert!(
        Database::open(&d, PW).is_err(),
        "旧口令必须立刻失效"
    );
}

#[test]
fn backup_and_restore_roundtrip() {
    let d = tmp("backup_src");
    let dst = d.join("backup");
    let restored = d.join("restored");
    let expect: Vec<Vec<u8>> = (0..300u64)
        .map(|i| {
            encode_row(&Row {
                slots: vec![
                    Slot::I64(i as i64),
                    Slot::Text(format!("backup-row-{i}")),
                    Slot::F64(2.0),
                ],
            })
        })
        .collect();

    {
        let mut db = seeded(&d);
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for (i, bytes) in expect.iter().enumerate() {
            let row = mightbe_store::table::decode_row(bytes).expect("解码emplate");
            let rid = insert_row(&mut db, &mut def, &row).unwrap();
            assert_eq!(rid, i as u64 + 1);
        }
        db.commit().unwrap();
        db.backup(&dst).expect("备份");
        db.close().expect("关库");
    }

    Database::restore_from(&dst, &restored).expect("恢复");
    let mut db = Database::open(&restored, PW).expect("恢复后的库应能打开");
    assert_eq!(dumped(&mut db, "docs"), expect);
    db.create_table("fresh", vec![col("x", col::I64)])
        .expect("恢复后仍可写");
    db.close().unwrap();

    assert!(
        Database::create(&d, PW, &KdfParams::fast()).is_err(),
        "库已存在时不能重复建"
    );
}


// ─────────────────────────── 回归：万行之外的脆弱路径 ───────────────────────────
// 这几条专门钉住历史上反复出现的"幽灵 bug"：
//  - 页链指针被压成"没人会读的 Cell"（只会断在根页）；
//  - free_space 差 4 字节目录项的"假装装得下"；
//  - remove_cell 不回退 first_free 导致的空间泄漏；
//  - 多次开合后 WAL/页链状态的不一致。

#[test]
fn large_mixed_rows_reopen_consistent() {
    let d = tmp("mixed");
    let n = 5_000u64;
    {
        let mut db = Database::create(&d, PW, &KdfParams::fast()).expect("建库");
        db.create_table(
            "docs",
            vec![
                col("id", col::I64),
                col("t", col::TEXT),
                col("f", col::F64),
                col("b", col::BYTES),
                col("v", col::VECTOR),
                col("n", col::NULL),
            ],
        )
        .expect("建表");
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..n {
            let row = Row {
                slots: vec![
                    Slot::I64(i as i64),
                    Slot::Text(format!("mixed-{i}-payload-{}-end", i * 7)),
                    Slot::F64(i as f64 * 1.5),
                    Slot::Bytes((0..((i % 64) + 1)).map(|x| x as u8 ^ 0x5a).collect()),
                    Slot::Vector(vec![i as f32, (i % 11) as f32, 2.5]),
                    Slot::Null,
                ],
            };
            insert_row(&mut db, &mut def, &row).expect("插入");
        }
        db.commit().expect("提交");
        assert!(db.n_pages() > 5, "混合类型也应撑出多条页链");
    }
    let mut db = Database::open(&d, PW).expect("重开");
    let def = db.table_def("docs").expect("表");
    assert_eq!(count_rows(&mut db, &def).unwrap(), n as usize, "混合类型行数");
    let rows = scan_table(&mut db, &def).expect("扫描");
    assert_eq!(rows.len(), n as usize);
    assert!(
        rows.windows(2).all(|w| w[0].0 < w[1].0),
        "扫描必须按 rid 有序"
    );
    // 抽样核对：逐槽内容（覆盖全部 codec 分支）
    for rid in [1u64, 1234, 2500, 4999, 5000] {
        let row = get_row(&mut db, &def, rid)
            .expect("点查")
            .unwrap_or_else(|| panic!("rid {rid} 应存在"));
        let i = rid - 1;
        assert_eq!(row.slots[0], Slot::I64(i as i64), "rid {rid} id");
        assert_eq!(
            row.slots[1],
            Slot::Text(format!("mixed-{i}-payload-{}-end", i * 7)),
            "rid {rid} text"
        );
        assert_eq!(row.slots[2], Slot::F64(i as f64 * 1.5), "rid {rid} f64");
        assert_eq!(
            row.slots[3],
            Slot::Bytes((0..((i % 64) + 1)).map(|x| x as u8 ^ 0x5a).collect()),
            "rid {rid} bytes"
        );
        assert_eq!(
            row.slots[4],
            Slot::Vector(vec![i as f32, (i % 11) as f32, 2.5]),
            "rid {rid} vector"
        );
        assert_eq!(row.slots[5], Slot::Null, "rid {rid} null");
    }
    db.close().expect("关库");
}

#[test]
fn reopen_cycles_preserve_all_rows() {
    let d = tmp("cycles");
    let cycles = 20u64;
    let per = 100u64;
    for c in 0..cycles {
        let mut db = if c == 0 {
            Database::create(&d, PW, &KdfParams::fast()).expect("建库")
        } else {
            Database::open(&d, PW).expect("重开已有库")
        };
        if c == 0 {
            db.create_table("docs", vec![col("id", col::I64)]).expect("建表");
        }
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..per {
            let rid_base = c * per + i + 1;
            insert_row(
                &mut db,
                &mut def,
                &Row { slots: vec![Slot::I64(rid_base as i64)] },
            )
            .expect("插入");
        }
        db.commit().expect("提交");
        db.close().expect("关库");
    }
    let mut db = Database::open(&d, PW).expect("最终重开");
    assert_eq!(
        ints(&mut db, "docs"),
        (1..=(cycles * per)).map(|x| x as i64).collect::<Vec<i64>>(),
        "20 轮开合后所有行都该在"
    );
    db.close().unwrap();
}

#[test]
fn delete_then_reinsert_uses_freed_space() {
    let d = tmp("delre");
    let n = 2_000u64;
    {
        let mut db = Database::create(&d, PW, &KdfParams::fast()).expect("建库");
        db.create_table("docs", vec![col("id", col::I64), col("t", col::TEXT)])
            .expect("建表");
        let mut def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        for i in 0..n {
            insert_row(
                &mut db,
                &mut def,
                &Row {
                    slots: vec![
                        Slot::I64(i as i64),
                        Slot::Text(format!("row-{i}-with-some-padding-text")),
                    ],
                },
            )
            .unwrap();
        }
        db.commit().unwrap();
    }
    {
        let mut db = Database::open(&d, PW).expect("重开");
        let def = db.table_def("docs").expect("表");
        db.begin().unwrap();
        let mut removed = 0;
        for i in 0..n {
            if i % 2 == 0 {
                assert!(
                    delete_row(&mut db, &def, i + 1).expect("删除"),
                    "rid {} 应存在",
                    i + 1
                );
                removed += 1;
            }
        }
        assert_eq!(removed, n / 2);
        db.commit().unwrap();
        db.close().unwrap();
    }
    let mut db = Database::open(&d, PW).expect("再重开");
    let mut def = db.table_def("docs").expect("表");
    assert_eq!(
        count_rows(&mut db, &def).unwrap(),
        (n / 2) as usize,
        "删后只剩奇数行"
    );
    // 在回收空间里再插 1000 行，绝不能冒 'page full' panic
    db.begin().unwrap();
    for i in 0..(n / 2) {
        let rid = insert_row(
            &mut db,
            &mut def,
            &Row {
                slots: vec![
                    Slot::I64(1_000_000 + i as i64),
                    Slot::Text(format!("reinserted-{i}")),
                ],
            },
        )
        .expect("回收空间再插入");
        assert_eq!(rid, n + i + 1, "rid 应接着之前的序号");
    }
    db.commit().unwrap();
    assert_eq!(
        count_rows(&mut db, &def).unwrap(),
        n as usize,
        "删 1000 + 插 1000 应回到总量"
    );
    db.close().unwrap();
}
