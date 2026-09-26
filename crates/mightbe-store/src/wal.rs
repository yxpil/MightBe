//! 预写日志与崩溃恢复（README 4.4 / 11.2 / 16.4）。
//!
//! # 帧格式
//!
//! ```text
//! ┌──────────────┬──────────────┬────────────────────────────────┐
//! │ magic "MBW1" │ next_counter │ crc32(header)                  │ ← 16B 超级帧
//! ├──────────────┴──────────────┴────────────────────────────────┤
//! │ frame: [u32 total_len][u32 crc32][AEAD(payload)]             │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 提交语义
//!
//! 事务提交时把 `Commit` 帧追加到末尾、fsync，随后**立即把文件截断到提交帧之后**。
//! 于是不变量成立：
//!
//! > WAL 里剩下的每一帧都属于已提交事务。
//!
//! 崩溃发生在任何一点，重放都只需"从头部顺序 Apply 到底"；
//! 落在最后一个 `Commit` 之后的字节（可能是半帧）一律视为不存在。
//!
//! 光靠截断还不够：提交之后紧接着写入的新事务帧，被打断时同样带着**合法的**
//! CRC 与 AEAD tag，只从文件头部顺序扫是分不出新旧的。所以重放还认 `Commit`
//! 边界——只有走到 `Commit` 的那些帧会收进 [`Replay`]，之后累积的一律作废。
//!
//! WAL 载荷同样走 AEAD（DEK），所以磁盘上不存在"已提交事务回滚"这种可被伪造的状态。

use mightbe_core::{MtbError, MtbResult};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::crypto::MasterKeys;
use crate::page::PAGE_SIZE;

const WAL_MAGIC: &[u8; 4] = b"MBW1";
const SUPER_LEN: u64 = 16;

const K_BEGIN: u8 = 0;
const K_COMMIT: u8 = 1;
const K_PUT_PAGE: u8 = 2;
const K_DROP_PAGE: u8 = 3;

/// 一条 WAL 记录（明文形式，落盘时整体封装）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    Begin,
    Commit,
    /// 整页重做图：崩溃后按序覆盖目标页。
    PutPage { page_id: u64, image: Vec<u8> },
    DropPage { page_id: u64 },
}

impl Record {
    fn kind(&self) -> u8 {
        match self {
            Record::Begin => K_BEGIN,
            Record::Commit => K_COMMIT,
            Record::PutPage { .. } => K_PUT_PAGE,
            Record::DropPage { .. } => K_DROP_PAGE,
        }
    }

    /// 编码为明文记录体（`kind` + 参数）。
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.kind());
        match self {
            Record::Begin | Record::Commit => {}
            Record::PutPage { page_id, image } => {
                out.extend_from_slice(&page_id.to_be_bytes());
                out.extend_from_slice(image);
            }
            Record::DropPage { page_id } => {
                out.extend_from_slice(&page_id.to_be_bytes());
            }
        }
        out
    }

    /// 从明文记录体还原。长度不足一律报错——宁可拒绝打开，也不要读到半截记录。
    pub fn decode(plain: &[u8]) -> MtbResult<Self> {
        let Some(&kind) = plain.first() else {
            return Err(MtbError::coded(MtbError::STORE + 20, "空 WAL 记录"));
        };
        let rest = &plain[1..];
        match kind {
            K_BEGIN if rest.is_empty() => Ok(Record::Begin),
            K_COMMIT if rest.is_empty() => Ok(Record::Commit),
            K_PUT_PAGE if rest.len() == 8 + PAGE_SIZE => {
                let mut pid = [0u8; 8];
                pid.copy_from_slice(&rest[..8]);
                let image = rest[8..].to_vec();
                Ok(Record::PutPage {
                    page_id: u64::from_be_bytes(pid),
                    image,
                })
            }
            K_DROP_PAGE if rest.len() == 8 => {
                let mut pid = [0u8; 8];
                pid.copy_from_slice(rest);
                Ok(Record::DropPage {
                    page_id: u64::from_be_bytes(pid),
                })
            }
            other => Err(MtbError::coded(
                MtbError::STORE + 21,
                format!("未知/残缺的 WAL 记录类型 {other}"),
            )),
        }
    }
}

/// 可追加的 WAL 句柄。
pub struct Wal {
    path: std::path::PathBuf,
    file: std::fs::File,
    counter: u64,
}

impl Wal {
    /// 建立（或覆盖）一个空 WAL。
    pub fn create(path: &std::path::Path, keys: &MasterKeys) -> MtbResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;
        let mut s = Self {
            path: path.to_path_buf(),
            file,
            counter: 0,
        };
        s.write_super(keys, true)?;
        Ok(s)
    }

    /// 打开既有 WAL，读回计数器；文件损坏/过短时按空 WAL 处理（调用方负责重放）。
    pub fn open(path: &std::path::Path, keys: &MasterKeys) -> MtbResult<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut head = [0u8; SUPER_LEN as usize];
        match file.read_exact(&mut head) {
            Ok(()) => {
                let mut h = crc32fast::Hasher::new();
                h.update(&head[..12]);
                if h.finalize() != u32::from_le_bytes(head[12..16].try_into().unwrap()) {
                    // 超级帧 CRC 不符：多半是"上次提交前就崩了"留下的空壳，按空 WAL 走
                    return Wal::create(path, keys);
                }
                let counter = u64::from_le_bytes(head[4..12].try_into().unwrap());
                let _ = keys; // 计数器不加密，但要能被任何人读到（只是没有语义）
                Ok(Self {
                    path: path.to_path_buf(),
                    file,
                    counter,
                })
            }
            Err(_) => Wal::create(path, keys),
        }
    }

    fn write_super(&mut self, keys: &MasterKeys, fsync: bool) -> MtbResult<()> {
        let _ = keys;
        let mut head = [0u8; SUPER_LEN as usize];
        head[..4].copy_from_slice(WAL_MAGIC);
        head[4..12].copy_from_slice(&self.counter.to_le_bytes());
        let mut h = crc32fast::Hasher::new();
        h.update(&head[..12]);
        head[12..16].copy_from_slice(&h.finalize().to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&head)?;
        if fsync {
            self.file.sync_all()?;
        }
        Ok(())
    }

    /// 追加一帧（不 fsync，供同一个事务内批量写）。
    pub fn append(&mut self, keys: &MasterKeys, rec: &Record) -> MtbResult<()> {
        let plain = rec.encode();
        // aad 绑定“这条记录属于 WAL 流的第 counter 帧”；counter 在 WAL 内单调递增
        let sealed = keys.seal(b"mdb/wal", self.counter, &plain)?;
        self.counter += 1;

        let mut payload = sealed;
        let total = 8 + payload.len();
        if total > u32::MAX as usize {
            return Err(MtbError::coded(MtbError::STORE + 22, "WAL 帧过长"));
        }
        let mut crc = crc32fast::Hasher::new();
        crc.update(&payload);
        let mut frame = Vec::with_capacity(total);
        frame.extend_from_slice(&(total as u32).to_le_bytes());
        frame.extend_from_slice(&crc.finalize().to_le_bytes());
        frame.append(&mut payload);

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&frame)?;
        self.write_super(keys, false)
    }

    /// 追加 `Commit` 帧并 fsync —— 这是"事务已提交"的唯一权威标记。
    pub fn commit(&mut self, keys: &MasterKeys) -> MtbResult<()> {
        self.append(keys, &Record::Commit)?;
        self.write_super(keys, true)?;
        self.file.sync_all()?;
        // 截断到当前末尾：残留字节一律属于未提交事务（或崩溃撕裂的半帧）。
        // 必须先回到真末尾再量长度——上一次 `write_super` 把光标留在了超级帧
        // 之后，直接 `stream_position` 会把整条日志削掉。
        let end = self.file.seek(SeekFrom::End(0))?;
        self.file.set_len(end)?;
        Ok(())
    }

    /// 丢弃日志（checkpoint 成功后调用）。
    pub fn clear(&mut self) -> MtbResult<()> {
        self.file.set_len(SUPER_LEN)?;
        self.file.seek(SeekFrom::Start(SUPER_LEN))?;
        self.counter = 0;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 是否有可重放的帧（供"是否要做恢复"的判断）。
    pub fn is_empty(&self) -> bool {
        self.len() <= SUPER_LEN
    }

    fn len(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(SUPER_LEN)
    }
}

/// 重放结果：按帧序给出的整页写入（**仅含已提交事务**）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Replay {
    pub puts: Vec<(u64, Vec<u8>)>,
    pub drops: Vec<u64>,
    /// 实际被解析到的帧数（一帧都读不到时为 0）。
    pub frames: usize,
}

/// 重放结果的消费说明：
///
/// - [`Replay::puts`] 按帧序给出，应用时**后写覆盖先写**；
/// - [`Replay::drops`] 全部设在写入之后——先落页再落删除，顺序反了会复活数据。
///
/// 这里刻意不提供 `apply(put, drop)` 这类回调接口：两个回调都要可变借用同一个
/// 数据库，闭包表达不了，最后只会逼出 `RefCell` 或者重复借用报错。
/// 调用方（[`crate::mdb::Database::recover`]）直接按下表顺序串行处理即可。

/// 从头扫描并重放一个 WAL。
///
/// 遇到第一处坏帧就停下并返回已解析部分：这既覆盖了"提交后立刻崩溃"的正常情况，
/// 也覆盖"写入中途断电留下半帧"的情况——两者都必须被丢弃，而不是被猜出来。
pub fn replay(path: &std::path::Path, keys: &MasterKeys) -> MtbResult<Replay> {
    use std::io::BufReader;
    let file = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(_) => return Ok(Replay::default()),
    };
    let mut out = Replay::default();
    let mut reader = BufReader::new(file);
    // 最前面那 16 字节是超级帧（magic + 计数器 + CRC），不是帧头。
    // 不跳过的话第一帧会被读成几亿字节的诡异长度，整条日志当场作废。
    if let Err(e) = reader.seek(SeekFrom::Start(SUPER_LEN)) {
        return Err(e.into());
    }
    // 帧序号一律从 0 重新数，不采信超级帧里那个计数器。
    //
    // 理由是提交时截断过文件，于是"文件里第 i 帧"必然是用 nonce 序号 i 封的：
    // 若反过来按超级帧的计数起步，一旦上次崩溃多写了几帧，序号就整体错位，
    // 日志会被整段判为损坏。超级帧的计数器只用于继续追加，不参与重放。
    let mut counter: u64 = 0;
    // 当前事务累积的操作。只有遇到 `Commit` 才转正——没提交的事务在崩溃时
    // 一个字节都不该被重放出来，哪怕它的帧写得和已提交的完全一样。
    let mut pending_puts: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut pending_drops: Vec<u64> = Vec::new();

    loop {
        let mut lenbuf = [0u8; 4];
        match reader.read_exact(&mut lenbuf) {
            Ok(()) => {}
            // EOF：正常结束
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let total = u32::from_le_bytes(lenbuf) as usize;
        if total < 8 {
            break; // 半帧：停止，丢弃后续一切
        }
        let mut frame = vec![0u8; total];
        frame[..4].copy_from_slice(&lenbuf);
        if let Err(e) = reader.read_exact(&mut frame[4..]) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break; // 撕裂的尾部
            }
            return Err(e.into());
        }
        let payload = &frame[8..];
        let mut h = crc32fast::Hasher::new();
        h.update(payload);
        let stored = u32::from_le_bytes(frame[4..8].try_into().unwrap());
        if h.finalize() != stored {
            break; // CRC 不符 → 说明这一帧及其之后都是崩溃残留
        }
        let plain = match keys.open(b"mdb/wal", counter, payload) {
            Some(p) => p,
            None => break, // 解不开 → 同样停止（可能换了密钥或数据被改写）
        };
        counter += 1;
        // 序号必须逐帧 +1：这一步拦住"上次崩溃多写了一帧、超级帧仍是旧值"
        // 那种状态——宁可停止重放，也不能让两条记录共用同一个 nonce。
        // AES-GCM 下 nonce 复用等于把认证密钥交给攻击者，属于必须拒绝的情况。
        out.frames += 1;
        match Record::decode(&plain)? {
            Record::PutPage { page_id, image } => pending_puts.push((page_id, image)),
            Record::DropPage { page_id } => pending_drops.push(page_id),
            Record::Begin => {}
            Record::Commit => {
                out.puts.extend(std::mem::take(&mut pending_puts));
                out.drops.extend(std::mem::take(&mut pending_drops));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{KdfParams, MasterKeys};
    use crate::page::Page;

    fn keys() -> MasterKeys {
        MasterKeys::derive(
            b"pw",
            &[1u8; 16],
            &KdfParams::fast(),
            &[2u8; 16],
            [3u8; 16],
        )
        .expect("派生")
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn append_then_replay_roundtrip() {
        let k = keys();
        let p = tmp("mb_wal_roundtrip.wal");
        let mut w = Wal::create(&p, &k).expect("创建");
        w.append(&k, &Record::Begin).expect("追加");
        w.append(
            &k,
            &Record::PutPage {
                page_id: 4,
                image: Page::new(4, crate::page::PT_DATA).into_raw(),
            },
        )
        .expect("追加页");
        w.append(&k, &Record::DropPage { page_id: 9 }).expect("追加删除");
        w.commit(&k).expect("提交");

        let r = replay(&p, &k).expect("重放");
        assert_eq!(r.frames, 4, "Begin + 2 条 + Commit");
        // Commit 之后已截断，重放出的正是已提交的两条
        assert_eq!(r.drops, vec![9]);
        assert!(r.puts.iter().any(|(id, _)| *id == 4));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn torn_tail_is_discarded() {
        let k = keys();
        let p = tmp("mb_wal_torn.wal");
        let mut w = Wal::create(&p, &k).expect("创建");
        w.append(&k, &Record::PutPage { page_id: 1, image: vec![7u8; PAGE_SIZE] }).expect("追加");
        w.commit(&k).expect("提交");
        // 模拟“提交后写入下一条事务、写到一半断电”
        w.append(&k, &Record::PutPage { page_id: 2, image: vec![8u8; PAGE_SIZE] }).expect("追加");
        let f = OpenOptions::new().write(true).open(&p).expect("打开");
        let len = f.metadata().expect("长度").len();
        f.set_len(len - 200).expect("截断成半帧");
        drop(f);

        let r = replay(&p, &k).expect("重放");
        let ids: Vec<u64> = r.puts.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![1], "未提交事务必须整条消失");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn uncommitted_tail_is_not_replayed() {
        let k = keys();
        let p = tmp("mb_wal_uncommitted.wal");
        let mut w = Wal::create(&p, &k).expect("创建");
        // 第一笔：已提交
        w.append(&k, &Record::PutPage { page_id: 1, image: vec![7u8; PAGE_SIZE] }).expect("追加");
        w.commit(&k).expect("提交");
        // 第二笔：写到一半就断电（文件里带了完整且合法的帧）
        w.append(&k, &Record::PutPage { page_id: 2, image: vec![8u8; PAGE_SIZE] }).expect("追加");
        w.append(&k, &Record::PutPage { page_id: 3, image: vec![9u8; PAGE_SIZE] }).expect("追加");

        let r = replay(&p, &k).expect("重放");
        let ids: Vec<u64> = r.puts.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![1], "未提交事务的一帧都不能漏出去");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn tampered_frame_is_discarded() {
        let k = keys();
        let p = tmp("mb_wal_tamper.wal");
        let mut w = Wal::create(&p, &k).expect("创建");
        w.append(&k, &Record::PutPage { page_id: 1, image: vec![7u8; PAGE_SIZE] }).expect("追加");
        w.commit(&k).expect("提交");
        w.append(&k, &Record::PutPage { page_id: 2, image: vec![9u8; PAGE_SIZE] }).expect("追加");

        let mut bytes = std::fs::read(&p).expect("读取文件");
        let n = bytes.len();
        bytes[n - 30] ^= 0xff; // 改最后一帧的载荷
        std::fs::write(&p, &bytes).expect("写回");

        let r = replay(&p, &k).expect("重放");
        let ids: Vec<u64> = r.puts.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![1], "被篡改的帧之后全部作废");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn record_codec_rejects_truncated_body() {
        assert!(Record::decode(&[K_PUT_PAGE]).is_err(), "缺页号与图像");
        assert!(Record::decode(&[K_PUT_PAGE, 0, 0]).is_err(), "长度不足");
        assert!(Record::decode(&[0x7f]).is_err(), "未知类型");
        assert!(Record::decode(&[K_COMMIT]).is_ok());
    }

    #[test]
    fn clear_resets_to_empty() {
        let k = keys();
        let p = tmp("mb_wal_clear.wal");
        let mut w = Wal::create(&p, &k).expect("创建");
        w.append(&k, &Record::PutPage { page_id: 1, image: vec![0u8; PAGE_SIZE] }).expect("追加");
        w.commit(&k).expect("提交");
        w.clear().expect("清空");
        assert!(w.is_empty());
        let r = replay(&p, &k).expect("重放");
        assert!(r.puts.is_empty(), "清空后不该有东西可重放");
        std::fs::remove_file(&p).ok();
    }
}
