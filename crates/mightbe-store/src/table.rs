//! Catalog 与表堆（README 4.4 / 11.2）。
//!
//! Catalog 是一段二进制快照，寄存在第 2 页（[`crate::mdb::PAGE_CATALOG`]）的
//! 第一个 Cell 里；表本身则是"页链 + Cell"的简单堆：
//!
//! ```text
//! root ──► [cell0: next page id][row][row]… ──► 下一页 ──► …
//! ```
//!
//! 每个表页的第一个 Cell 固定存链路指针（8 字节），行数据从第二个 Cell 起排。
//! 行按 `rid`（表内自增）寻址，不做额外索引——M2 范围内「万行顺序扫描正确」
//! 是验收口径，索引属于 M9。

use mightbe_core::{MtbError, MtbResult};

use crate::api::{ColumnDef, Row, Slot, TableDef};
use crate::mdb::{Database, PAGE_CATALOG};

/// 列类型标签。**与行内编码用的槽位标签同源**——`encode_row`/`decode_row` 里那套
/// 数字就是这里这些，因此 Catalog 里的 `col_type` 既是给用户看的，也是解码时的
/// 权威判据，不存在两份互不一致的编号。
pub mod col {
    pub const NULL: u8 = 0;
    pub const BOOL: u8 = 1;
    pub const I64: u8 = 2;
    pub const F64: u8 = 3;
    pub const TEXT: u8 = 4;
    pub const BYTES: u8 = 5;
    pub const VECTOR: u8 = 6;
}

// ───────────────────────── Catalog 编解码 ─────────────────────────

/// Catalog 二进制格式（小端、长度前缀，不引 serde）。
pub fn encode_catalog(tables: &[TableDef]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(tables.len() as u32).to_le_bytes());
    for t in tables {
        push_str(&mut out, &t.name);
        out.extend_from_slice(&t.root_page.to_le_bytes());
        out.extend_from_slice(&t.next_rid.to_le_bytes());
        out.extend_from_slice(&(t.columns.len() as u16).to_le_bytes());
        for c in &t.columns {
            push_str(&mut out, &c.name);
            out.push(c.col_type as u8);
            out.push(if c.primary_key { 1 } else { 0 });
        }
    }
    out
}

/// 解析 Catalog。残缺或越界一律报错——宁可拒绝开库，也不要沉默地丢表。
pub fn decode_catalog(bytes: &[u8]) -> MtbResult<Vec<TableDef>> {
    if bytes.len() < 4 {
        return Ok(Vec::new());
    }
    let n = u32::from_le_bytes(bytes[..4].try_into().expect("4 字节")) as usize;
    let mut p = 4usize;
    let mut out = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let name = take_str(bytes, &mut p)?;
        let root = take_u64(bytes, &mut p)?;
        let next_rid = take_u64(bytes, &mut p)?;
        let ncols = take_u16(bytes, &mut p)? as usize;
        let mut cols = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            let cname = take_str(bytes, &mut p)?;
            let col_type = take_u8(bytes, &mut p)?;
            let pk = take_u8(bytes, &mut p)?;
            cols.push(ColumnDef {
                name: cname,
                col_type,
                primary_key: pk != 0,
            });
        }
        out.push(TableDef {
            name,
            root_page: root,
            next_rid,
            columns: cols,
        });
    }
    Ok(out)
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn take_str(bytes: &[u8], p: &mut usize) -> MtbResult<String> {
    let len = take_u16(bytes, p)? as usize;
    let end = *p + len;
    if end > bytes.len() {
        return Err(MtbError::coded(MtbError::STORE + 60, "Catalog 越界"));
    }
    let s = String::from_utf8_lossy(&bytes[*p..end]).into_owned();
    *p = end;
    Ok(s)
}

fn take_u8(bytes: &[u8], p: &mut usize) -> MtbResult<u8> {
    if *p + 1 > bytes.len() {
        return Err(MtbError::coded(MtbError::STORE + 61, "Catalog 越界"));
    }
    let v = bytes[*p];
    *p += 1;
    Ok(v)
}

fn take_u16(bytes: &[u8], p: &mut usize) -> MtbResult<u16> {
    if *p + 2 > bytes.len() {
        return Err(MtbError::coded(MtbError::STORE + 62, "Catalog 越界"));
    }
    let v = u16::from_le_bytes(bytes[*p..*p + 2].try_into().expect("2 字节"));
    *p += 2;
    Ok(v)
}

fn take_u64(bytes: &[u8], p: &mut usize) -> MtbResult<u64> {
    if *p + 8 > bytes.len() {
        return Err(MtbError::coded(MtbError::STORE + 63, "Catalog 越界"));
    }
    let v = u64::from_le_bytes(bytes[*p..*p + 8].try_into().expect("8 字节"));
    *p += 8;
    Ok(v)
}

// ───────────────────────── Catalog 访问 ─────────────────────────

impl Database {
    /// 读 Catalog 快照。
    pub fn catalog(&mut self) -> MtbResult<Vec<TableDef>> {
        let page = self.page(PAGE_CATALOG)?;
        let blob = page.cell(0).unwrap_or_default();
        decode_catalog(&blob)
    }

    /// 写 Catalog 快照（同一事务内，先 WAL 后落盘）。
    fn save_catalog(&mut self, tables: &[TableDef]) -> MtbResult<()> {
        let blob = encode_catalog(tables);
        if blob.len() > crate::page::PAGE_SIZE - crate::page::PAGE_HEADER - 4 {
            return Err(MtbError::coded(
                MtbError::STORE + 64,
                "Catalog 超过单页容量".to_string(),
            ));
        }
        let mut page = self.page(PAGE_CATALOG)?;
        // 整体重写：Catalog 是快照，没有增量更新的必要
        while page.n_cells() > 0 {
            page.remove_cell(page.n_cells() - 1);
        }
        if page.push_cell(&blob).is_none() {
            return Err(MtbError::coded(
                MtbError::STORE + 65,
                "Catalog 放不进页".to_string(),
            ));
        }
        page.seal();
        self.put_page(page);
        Ok(())
    }

    /// 建表：分配根页，写 Catalog。
    pub fn create_table(&mut self, name: &str, columns: Vec<ColumnDef>) -> MtbResult<TableDef> {
        if name.is_empty() {
            return Err(MtbError::coded(MtbError::STORE + 66, "表名不能为空"));
        }
        let mut tables = self.catalog()?;
        if tables.iter().any(|t| t.name == name) {
            return Err(MtbError::coded(
                MtbError::TABLE + 1,
                format!("表 {name} 已存在"),
            ));
        }
        if !self.in_txn() {
            self.begin()?;
        }
        let root = self.alloc_page()?;
        let def = TableDef {
            name: name.to_string(),
            root_page: root,
            next_rid: 1,
            columns,
        };
        let mut root_page = self.page(root)?;
        root_page.set_page_type(crate::page::PT_TABLE);
        // 链路指针占位：第 0 个 Cell 存"下一页页号"，0 表示链尾
        root_page.push_cell(&0u64.to_le_bytes());
        root_page.seal();
        self.put_page(root_page);

        tables.push(def.clone());
        self.save_catalog(&tables)?;
        // 本次调用是原子的：库里要么有这张表，要么一张都没有。
        // 提交顺带把 Catalog 与根页一起 checkpoint 到数据文件。
        self.commit()?;
        Ok(def)
    }

    /// 取表定义（不存在时报 2xxx 表错误码）。
    pub fn table_def(&mut self, name: &str) -> MtbResult<TableDef> {
        self.catalog()?
            .into_iter()
            .find(|t| t.name == name)
            .ok_or_else(|| {
                MtbError::coded(MtbError::TABLE + 2, format!("表 {name} 不存在"))
            })
    }

    /// 删表：清 Catalog、释放根页与整条页链。
    pub fn drop_table(&mut self, name: &str) -> MtbResult<()> {
        if !self.in_txn() {
            self.begin()?;
        }
        let def = self.table_def(name)?;
        let mut tables = self.catalog()?;
        tables.retain(|t| t.name != name);
        self.save_catalog(&tables)?;

        let mut id = Some(def.root_page);
        while let Some(cur) = id {
            let page = match self.page(cur) {
                Ok(p) => p,
                Err(_) => break,
            };
            id = page.cell(0).and_then(|c| {
                if c.len() == 8 {
                    Some(u64::from_le_bytes(c.try_into().expect("8 字节")))
                } else {
                    None
                }
            });
            self.drop_page(cur);
        }
        self.commit()?;
        Ok(())
    }
}

// ───────────────────────── 行编解码 ─────────────────────────

/// 行 → 字节。类型标签在前，变长段带长度前缀。
pub fn encode_row(row: &Row) -> Vec<u8> {
    let mut out = vec![row.slots.len() as u8];
    for s in &row.slots {
        match s {
            Slot::Null => out.push(0),
            Slot::Bool(b) => {
                out.push(1);
                out.push(if *b { 1 } else { 0 });
            }
            Slot::I64(v) => {
                out.push(2);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Slot::F64(v) => {
                out.push(3);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Slot::Text(t) => {
                out.push(4);
                push_blob(&mut out, t.as_bytes());
            }
            Slot::Bytes(b) => {
                out.push(5);
                push_blob(&mut out, b);
            }
            Slot::Vector(v) => {
                out.push(6);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    out
}

/// 字节 → 行。长度不符/未知标签都报错，绝不产出半截行。
pub fn decode_row(bytes: &[u8]) -> MtbResult<Row> {
    let Some(&n) = bytes.first() else {
        return Err(MtbError::coded(MtbError::STORE + 70, "空行编码"));
    };
    let mut p = 1usize;
    let mut slots = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let tag = take_u8(bytes, &mut p)?;
        slots.push(match tag {
            0 => Slot::Null,
            1 => Slot::Bool(take_u8(bytes, &mut p)? != 0),
            // 定长槽：读走 8 字节后必须把游标推过去，否则下一个"标签"会读进
            // 载荷的第一个字节（-7 的小端首字节是 0xf9，会被当成标签 249）。
            2 => {
                let v = i64::from_le_bytes(
                    bytes[p..p + 8].try_into().map_err(|_| bad_row())?,
                );
                p += 8;
                Slot::I64(v)
            }
            3 => {
                let v = f64::from_le_bytes(
                    bytes[p..p + 8].try_into().map_err(|_| bad_row())?,
                );
                p += 8;
                Slot::F64(v)
            }
            // 变长槽的布局是「标签 → u32 长度 → 数据」，`take_u8` 已经把标签
            // 消费掉，此刻 `p` 正指向长度前缀——直接读即可，不能再多跳一个字节。
            4 => Slot::Text(String::from_utf8_lossy(&take_blob(bytes, &mut p)?).into_owned()),
            5 => Slot::Bytes(take_blob(bytes, &mut p)?),
            6 => {
                let n = u32::from_le_bytes(
                    bytes[p..p + 4].try_into().map_err(|_| bad_row())?,
                ) as usize;
                p += 4;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if p + 4 > bytes.len() {
                        return Err(bad_row());
                    }
                    v.push(f32::from_le_bytes(
                        bytes[p..p + 4].try_into().expect("4 字节"),
                    ));
                    p += 4;
                }
                Slot::Vector(v)
            }
            other => return Err(MtbError::coded(MtbError::STORE + 71, format!("未知槽位标签 {other}"))),
        });
    }
    Ok(Row { slots })
}

fn bad_row() -> MtbError {
    MtbError::coded(MtbError::STORE + 72, "行编码被截断")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Row, Slot};

    /// 行编解码必须严格镜像：编码两次、解回来一致，变长槽的长度前缀不能被吃错。
    #[test]
    fn row_codec_roundtrip() {
        let cases = vec![
            Row {
                slots: vec![Slot::I64(-7), Slot::F64(2.5), Slot::Null],
            },
            Row {
                slots: vec![
                    Slot::I64(0),
                    Slot::Text("backup-row-0".into()),
                    Slot::F64(2.0),
                ],
            },
            Row {
                slots: vec![
                    Slot::Bytes(vec![1, 2, 3]),
                    Slot::Vector(vec![1.0, -2.5, 3.25]),
                ],
            },
        ];
        for r in cases.iter() {
            let b = encode_row(r);
            let d = decode_row(&b).expect("应能解回来");
            assert_eq!(d.slots, r.slots, "解回来应当与原件一致");
            assert_eq!(encode_row(&d), b, "同解同编，编码必须确定性");
        }
    }
}

fn push_blob(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn take_blob(bytes: &[u8], p: &mut usize) -> MtbResult<Vec<u8>> {
    if *p + 4 > bytes.len() {
        return Err(bad_row());
    }
    let len = u32::from_le_bytes(bytes[*p..*p + 4].try_into().expect("4 字节")) as usize;
    *p += 4;
    let end = *p + len;
    if end > bytes.len() {
        return Err(bad_row());
    }
    let v = bytes[*p..end].to_vec();
    *p = end;
    Ok(v)
}

// ───────────────────────── 表堆 ─────────────────────────

/// 行在页内的编码：`rid (8B) || 行字节`。
fn encode_cell(rid: u64, row: &Row) -> Vec<u8> {
    let mut out = rid.to_le_bytes().to_vec();
    out.extend_from_slice(&encode_row(row));
    out
}

fn decode_cell(bytes: &[u8]) -> MtbResult<(u64, Row)> {
    if bytes.len() < 8 {
        return Err(bad_row());
    }
    let rid = u64::from_le_bytes(bytes[..8].try_into().expect("8 字节"));
    let row = decode_row(&bytes[8..])?;
    Ok((rid, row))
}

/// 页内还能不能再塞进一个 `len` 字节的 Cell。
///
/// `push_cell` 除了数据本身还要占一个 4 字节目录项，所以判据是 `len + 4` 而不是
/// `len`。只比 `free_space()` 和 `len`，剩余空间恰好落在 (len, len+4) 区间里的那几行
/// 会撞上"fit_page 说这页装得下、push_cell 却拒绝"——差的那 4 字节就是目录项。
fn can_hold(page: &crate::page::Page, len: usize) -> bool {
    page.free_space() >= len + 4
}

/// 沿页链找能放下一行的页；不够就在链尾新挂一页（并回填上一页的链路指针）。
fn fit_page(db: &mut Database, def: &TableDef, needs: usize) -> MtbResult<u64> {
    let mut cur = def.root_page;
    loop {
        let page = db.page(cur)?;
        if can_hold(&page, needs) {
            return Ok(cur);
        }
        if let Some(next) = next_of(&page).filter(|&n| n != 0) {
            cur = next;
            continue;
        }
        let new_id = db.alloc_page()?;
        let mut tail = db.page(cur)?;
        tail.set_page_type(crate::page::PT_TABLE);
        // 链路指针固定占住 Cell 0，所以这里必须**改写**它，不能再 `push_cell`
        // 追加一条：多压出来的那条 Cell 没人会去读（扫描只看 Cell 0），
        // 于是新页虽然挂上去了，整条链却依旧断在当前页，后面每页都只写得进一行。
        tail.overwrite(0, &new_id.to_le_bytes())
            .map_err(|_| MtbError::coded(MtbError::STORE + 74, "表页链路指针写入失败"))?;
        tail.seal();
        db.put_page(tail);
        // 新页必须自带一个"下一页=0"的占位 Cell 占住槽 0：表页约定第 0 个 Cell
        // 是链路指针，若让它空着，首行就会顶到槽 0，扫描时的 `skip(1)` 会漏掉整页。
        let mut fresh = db.page(new_id)?;
        fresh.push_cell(&0u64.to_le_bytes());
        fresh.seal();
        db.put_page(fresh);
        return Ok(new_id);
    }
}

/// 插入一行，返回新 `rid`。
pub fn insert_row(db: &mut Database, def: &mut TableDef, row: &Row) -> MtbResult<u64> {
    let rid = def.next_rid;
    let cell = encode_cell(rid, row);
    let page_id = fit_page(db, def, cell.len())?;
    let mut page = db.page(page_id)?;
    // 报错要把现场写全：`fit_page` 说这页装得下、`push_cell` 却拒绝，
    // 说明两者看的是同一页的不同版本，没有这些数字根本无从下手。
    page.push_cell(&cell).ok_or_else(|| {
        MtbError::coded(
            MtbError::STORE + 73,
            format!(
                "行放不进页（页 {page_id}：需要 {} 字节，剩余 {} 字节，已有 {} 格）",
                cell.len(),
                page.free_space(),
                page.n_cells()
            ),
        )
    })?;
    page.seal();
    db.put_page(page);

    def.next_rid += 1;
    let mut tables = db.catalog()?;
    if let Some(t) = tables.iter_mut().find(|t| t.name == def.name) {
        t.next_rid = def.next_rid;
    }
    db.save_catalog(&tables)?;
    Ok(rid)
}

/// 按 rid 取行。
pub fn get_row(db: &mut Database, def: &TableDef, rid: u64) -> MtbResult<Option<Row>> {
    let mut cur = Some(def.root_page);
    while let Some(id) = cur {
        let page = db.page(id)?;
        let n = page.n_cells().saturating_sub(1); // 首 Cell 是链路指针
        for i in 1..=n {
            if let Some(bytes) = page.cell(i) {
                if let Ok((r, row)) = decode_cell(&bytes) {
                    if r == rid {
                        return Ok(Some(row));
                    }
                }
            }
        }
        cur = page.cell(0).and_then(|c| {
            if c.len() == 8 {
                let v = u64::from_le_bytes(c.try_into().expect("8 字节"));
                if v == 0 { None } else { Some(v) }
            } else {
                None
            }
        });
    }
    Ok(None)
}

/// 全表顺序扫描（火山模型 `SeqScan` 的底子）。
pub fn scan_table(db: &mut Database, def: &TableDef) -> MtbResult<Vec<(u64, Row)>> {
    let mut out = Vec::new();
    let mut cur = Some(def.root_page);
    while let Some(id) = cur {
        let page = db.page(id)?;
        for bytes in page.cells().skip(1) {
            if let Ok(pair) = decode_cell(&bytes) {
                out.push(pair);
            }
        }
        cur = page.cell(0).and_then(|c| {
            if c.len() == 8 {
                let v = u64::from_le_bytes(c.try_into().expect("8 字节"));
                if v == 0 { None } else { Some(v) }
            } else {
                None
            }
        });
    }
    out.sort_by_key(|(rid, _)| *rid);
    Ok(out)
}

/// 按 rid 删除一行。
pub fn delete_row(db: &mut Database, def: &TableDef, rid: u64) -> MtbResult<bool> {
    let mut cur = Some(def.root_page);
    while let Some(id) = cur {
        let page = db.page(id)?;
        let n = page.n_cells().saturating_sub(1);
        let mut hit = None;
        for i in 1..=n {
            if let Some(bytes) = page.cell(i) {
                if let Ok((r, _)) = decode_cell(&bytes) {
                    if r == rid {
                        hit = Some(i);
                        break;
                    }
                }
            }
        }
        if let Some(i) = hit {
            let mut page = db.page(id)?;
            let _ = page.cell(i); // 先读走，再删
            page.remove_cell(i);
            page.seal();
            db.put_page(page);
            return Ok(true);
        }
        cur = page.cell(0).and_then(|c| {
            if c.len() == 8 {
                let v = u64::from_le_bytes(c.try_into().expect("8 字节"));
                if v == 0 { None } else { Some(v) }
            } else {
                None
            }
        });
    }
    Ok(false)
}

/// 表内行数（不含链路指针 Cell）。
pub fn count_rows(db: &mut Database, def: &TableDef) -> MtbResult<usize> {
    Ok(scan_table(db, def)?.len())
}

// ───────────────────────── 链式取页的小工具 ─────────────────────────

/// 取表页的下一页链指针。
pub fn next_of(page: &crate::page::Page) -> Option<u64> {
    page.cell(0)
        .filter(|c| c.len() == 8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8 字节")))
}
