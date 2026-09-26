//! MDB 页引擎（README 4.4 / 11.1 / 11.2）。
//!
//! 定长页是整库唯一的存储单元：文件长度恒为 `PAGE_SIZE` 的整数倍，
//! 每个页被整块 AEAD 加密后落盘（见 [`crate::crypto`]），所以这里的
//! `raw` 一律是**明文**页，密文只在 `Database` 的读写边界上存在。
//!
//! 页内布局（32 字节头 + 向下收缩的 Cell 目录 + 尾部分配区）：
//!
//! ```text
//! ┌────────────────┬─────────────────────┬──────────────────────┐
//! │ header (32B)   │ cell dir (4·n)      │ cells ↓ (尾部分配)    │
//! └────────────────┴─────────────────────┴──────────────────────┘
//! ```
//!
//! CRC32 覆盖除自身字段外的整页，是 AEAD 之外的第二道校验（README 11.1）：
//! 即使密钥被泄露，位翻转依然会被这里拦下。

use crc32fast::Hasher;
use mightbe_core::{MtbError, MtbResult};

/// 页大小（字节）。8 KiB：比 4 KiB 少一倍页故障，又不至于让 WAL 单帧过大。
pub const PAGE_SIZE: usize = 8192;

/// 页头长度。
pub const PAGE_HEADER: usize = 32;

/// 空闲页标记（位图页里为 0）。
pub const PT_FREE: u8 = 0;
/// 普通数据页。
pub const PT_DATA: u8 = 1;
/// 元数据页（位图 / 系统表根）。
pub const PT_META: u8 = 2;
/// 表堆根页。
pub const PT_TABLE: u8 = 3;

const MAGIC: &[u8; 4] = b"MBP1";

const OFF_MAGIC: usize = 0;
const OFF_PAGE_ID: usize = 4;
const OFF_TYPE: usize = 12;
const OFF_FLAGS: usize = 13;
const OFF_N_CELLS: usize = 14;
const OFF_FIRST_FREE: usize = 16;
/// 保留位。占位用：页头布局一旦定下就不再改动，
/// 将来加标志位直接往这里放，不必挪动 CRC 与目录的位置。
#[allow(dead_code)]
const OFF_RESERVED: usize = 18;
const OFF_CRC: usize = 24;

/// 就地读写的小端工具：页布局全是裸字节，避免引入字节序依赖。
trait Le {
    fn ld(b: &[u8]) -> Self;
    fn st(self, b: &mut [u8]);
}

impl Le for u16 {
    fn ld(b: &[u8]) -> Self {
        u16::from_le_bytes([b[0], b[1]])
    }
    fn st(self, b: &mut [u8]) {
        b.copy_from_slice(&self.to_le_bytes());
    }
}

impl Le for u32 {
    fn ld(b: &[u8]) -> Self {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn st(self, b: &mut [u8]) {
        b.copy_from_slice(&self.to_le_bytes());
    }
}

impl Le for u64 {
    fn ld(b: &[u8]) -> Self {
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn st(self, b: &mut [u8]) {
        b.copy_from_slice(&self.to_le_bytes());
    }
}

/// 一个明文页。容量恒为 `PAGE_SIZE`；改动内容后必须 [`Page::seal`]。
#[derive(Clone)]
pub struct Page {
    raw: Vec<u8>,
}

impl Page {
    /// 造一个指定 id、全零、类型为 `ty` 的页。
    pub fn new(page_id: u64, ty: u8) -> Self {
        let mut raw = vec![0u8; PAGE_SIZE];
        raw[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(MAGIC);
        page_id.st(&mut raw[OFF_PAGE_ID..OFF_PAGE_ID + 8]);
        raw[OFF_TYPE] = ty;
        // 尾部分配区从页尾开始
        (PAGE_SIZE as u16).st(&mut raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]);
        Self { raw }
    }

    /// 包装一段已满页的缓冲（读路径用）。**不做** CRC 校验——那是调用方的事。
    pub fn from_raw(raw: Vec<u8>) -> MtbResult<Self> {
        if raw.len() != PAGE_SIZE {
            return Err(MtbError::coded(
                MtbError::STORE + 1,
                format!("页长 {} 不是 {PAGE_SIZE}", raw.len()),
            ));
        }
        if &raw[OFF_MAGIC..OFF_MAGIC + 4] != MAGIC {
            return Err(MtbError::coded(
                MtbError::STORE + 2,
                "页头 magic 不符（文件不是 MDB，或被截断/覆盖）".to_string(),
            ));
        }
        Ok(Self { raw })
    }

    pub fn page_id(&self) -> u64 {
        u64::ld(&self.raw[OFF_PAGE_ID..OFF_PAGE_ID + 8])
    }

    pub fn set_page_id(&mut self, id: u64) {
        id.st(&mut self.raw[OFF_PAGE_ID..OFF_PAGE_ID + 8]);
    }

    pub fn page_type(&self) -> u8 {
        self.raw[OFF_TYPE]
    }

    pub fn set_page_type(&mut self, ty: u8) {
        self.raw[OFF_TYPE] = ty;
    }

    pub fn flags(&self) -> u8 {
        self.raw[OFF_FLAGS]
    }

    pub fn set_flags(&mut self, f: u8) {
        self.raw[OFF_FLAGS] = f;
    }

    pub fn n_cells(&self) -> u16 {
        u16::ld(&self.raw[OFF_N_CELLS..OFF_N_CELLS + 2])
    }

    /// CRC32 字段的当前值（未必与内容相符）。
    pub fn crc32(&self) -> u32 {
        u32::ld(&self.raw[OFF_CRC..OFF_CRC + 4])
    }

    /// 覆盖整页的 CRC32 —— **不含 `[CRC..CRC+4)` 这 4 个字节**。
    ///
    /// 所以 `seal` 先把这 4 字节清零再算，`crc_ok` 才能对上；
    /// 若把自身字段也算进去，签完名的页永远校验不过。
    fn body_crc(&self) -> u32 {
        let mut h = Hasher::new();
        h.update(&self.raw[..OFF_CRC]);
        h.update(&self.raw[OFF_CRC + 4..]);
        h.update(&[0u8; 4]); // 占位：xor 进去等于把 CRC 字段当 0
        h.finalize()
    }

    /// 写入 CRC 字段。任何改变页内容的操作之后都要重新调用。
    ///
    /// 先算后写：计算时把 CRC 字段位置当 0，因此重复调用结果稳定（幂等）。
    pub fn seal(&mut self) {
        self.raw[OFF_CRC..OFF_CRC + 4].fill(0);
        let crc = self.body_crc();
        self.raw[OFF_CRC..OFF_CRC + 4].copy_from_slice(&crc.to_le_bytes());
    }

    /// 校验 CRC。位翻转必然在此暴露（AEAD 之外的第二道防线）。
    pub fn crc_ok(&self) -> bool {
        self.body_crc() == self.crc32()
    }

    /// 裸页内容（加密前 / 写盘前用）。
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// 页内容的变长视图（位图页、自建元数据等直接改字节）。
    /// 改完必须重新 [`Page::seal`]。
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.raw
    }

    /// 取走裸页内容（WAL 重放、ROTATE KEY 重新加密等场景）。
    pub fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    // ───────────────────────── Cell ─────────────────────────

    fn dir_at(&self, i: u16) -> (usize, usize) {
        let off = PAGE_HEADER + usize::from(i) * 4;
        let start = u16::ld(&self.raw[off..off + 2]) as usize;
        let len = u16::ld(&self.raw[off + 2..off + 4]) as usize;
        (start, len)
    }

    fn dir_set(&mut self, i: u16, start: usize, len: usize) {
        let off = PAGE_HEADER + usize::from(i) * 4;
        debug_assert!(start <= u16::MAX as usize && len <= u16::MAX as usize);
        (start as u16).st(&mut self.raw[off..off + 2]);
        (len as u16).st(&mut self.raw[off + 2..off + 4]);
    }

    /// 尾部剩余字节数。
    pub fn free_space(&self) -> usize {
        let first_free = u16::ld(&self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]) as usize;
        let dir = PAGE_HEADER + usize::from(self.n_cells()) * 4;
        first_free.saturating_sub(dir)
    }

    /// 尝试在尾部放入一段 Cell 数据；空间不足返回 `None`（调用方该换页了）。
    pub fn push_cell(&mut self, data: &[u8]) -> Option<u16> {
        if data.len() > u16::MAX as usize {
            return None;
        }
        let first_free = u16::ld(&self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]) as usize;
        // 目录从页头往下长、数据从页尾往上长，两条路迟早撞上。
        // 留 4 字节给新目录项，撞上没有空间就是"页满了"。
        let dir_end = PAGE_HEADER + usize::from(self.n_cells()) * 4;
        if data.len() + 4 > first_free.saturating_sub(dir_end) {
            return None;
        }
        let start = first_free - data.len();
        self.raw[start..start + data.len()].copy_from_slice(data);
        let n = self.n_cells();
        self.dir_set(n, start, data.len());
        (start as u16).st(&mut self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]);
        self.raw[OFF_N_CELLS..OFF_N_CELLS + 2].copy_from_slice(&(n + 1).to_le_bytes());
        Some(start as u16)
    }

    /// 读第 `i` 个 Cell 的副本。
    pub fn cell(&self, i: u16) -> Option<Vec<u8>> {
        if i >= self.n_cells() {
            return None;
        }
        let (start, len) = self.dir_at(i);
        Some(self.raw[start..start + len].to_vec())
    }

    /// 原地改写第 `i` 个 Cell，长度必须一致。
    /// 变长数据请先取走再用 [`Page::push_cell`] 重写。
    pub fn overwrite(&mut self, i: u16, data: &[u8]) -> MtbResult<()> {
        let (start, len) = self.dir_at(i);
        if data.len() != len {
            return Err(MtbError::coded(
                MtbError::STORE + 3,
                format!("覆盖长度 {} 与原长 {len} 不一致", data.len()),
            ));
        }
        self.raw[start..start + len].copy_from_slice(data);
        Ok(())
    }

    /// 删除第 `i` 个 Cell。
    ///
    /// 页内 Cell 按地址从页尾向下紧凑排放：cell 0 在最高地址，末位 cell 在最低地址
    /// `first_free`，目录记录每个 cell 的 (start, len)。删除中间的 cell 不能把末位直接
    /// `copy_from_slice` 过去——变长 cell 长度不同会越界。正确做法是把被删 cell 之下
    /// 的所有 cell 整体上移 `len` 字节补洞，目录项随之平移，再把分配指针回退到新的最低 cell。
    /// 调用方负责先 [`Page::cell`] 读走内容——本方法只搬字节。
    pub fn remove_cell(&mut self, i: u16) {
        let n = self.n_cells();
        if i >= n {
            return;
        }
        let last = n - 1;
        if i != last {
            let (a0, a1) = self.dir_at(i);
            let first_free = u16::ld(&self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]) as usize;
            // 洞 = [a0, a0+a1)；其下所有 cell 落在 [first_free, a0)，整体上移 a1 字节补洞。
            debug_assert!(a0 >= first_free);
            let tail = self.raw[first_free..a0].to_vec();
            self.raw[first_free + a1..a0 + a1].copy_from_slice(&tail);
            // 目录项平移：原槽 k → 新槽 k-1，起点加 a1。
            for k in (i as usize + 1)..=(last as usize) {
                let (s, l) = self.dir_at(k as u16);
                self.dir_set((k - 1) as u16, s + a1, l);
            }
        }
        // 收缩目录并回退分配指针到新的最低 cell。
        let new_n = last; // = n - 1
        let off = PAGE_HEADER + usize::from(last) * 4;
        self.raw[off..off + 4].fill(0);
        self.raw[OFF_N_CELLS..OFF_N_CELLS + 2]
            .copy_from_slice(&(new_n as u16).to_le_bytes());
        if new_n == 0 {
            (PAGE_SIZE as u16).st(&mut self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]);
        } else {
            let (s_last, _) = self.dir_at((new_n - 1) as u16);
            (s_last as u16).st(&mut self.raw[OFF_FIRST_FREE..OFF_FIRST_FREE + 2]);
        }
    }

    /// 遍历全部 Cell 内容。
    pub fn cells(&self) -> impl Iterator<Item = Vec<u8>> + '_ {
        (0..self.n_cells()).filter_map(|i| self.cell(i))
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.page_id())
            .field("type", &self.page_type())
            .field("cells", &self.n_cells())
            .field("free", &self.free_space())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_roundtrip_and_crc() {
        let mut p = Page::new(7, PT_TABLE);
        p.push_cell(b"hello").expect("放得下");
        p.push_cell(b"world!!").expect("放得下");
        p.seal();
        assert!(p.crc_ok());

        assert_eq!(p.n_cells(), 2);
        assert_eq!(p.cell(0).unwrap(), b"hello".to_vec());
        assert_eq!(p.cell(1).unwrap(), b"world!!".to_vec());

        // 改内容后原 CRC 立刻失效，seal 后恢复
        p.overwrite(0, b"HELLO").expect("同长覆盖");
        assert!(!p.crc_ok());
        p.seal();
        assert!(p.crc_ok());

        // 位翻转：CRC 必须察觉
        let mut q = p.clone();
        let idx = 3000usize;
        q.raw[idx] ^= 0x01;
        assert!(!q.crc_ok(), "单 bit 翻转应被 CRC 抓住");
    }

    #[test]
    fn seal_is_idempotent() {
        let mut p = Page::new(1, PT_DATA);
        p.push_cell(&[1, 2, 3]);
        p.seal();
        let first = p.crc32();
        p.seal();
        p.seal();
        assert_eq!(p.crc32(), first, "重复 seal 不得改写 CRC");
        assert!(p.crc_ok());
    }

    #[test]
    fn remove_cell_is_compact_and_shrink() {
        let mut p = Page::new(2, PT_DATA);
        for i in 0..4u8 {
            p.push_cell(&[i; 1]).expect("放得下");
        }
        assert!(p.cell(1).unwrap() == vec![1u8]);
        let before = p.free_space();
        p.remove_cell(1);
        assert_eq!(p.n_cells(), 3);
        // 压实语义：删掉槽 1 后，其下方的 cell 整体上移补洞，相对顺序保持
        assert_eq!(p.cell(0).unwrap(), vec![0u8]);
        assert_eq!(p.cell(1).unwrap(), vec![2u8]);
        assert_eq!(p.cell(2).unwrap(), vec![3u8]);
        assert!(p.cell(3).is_none(), "目录必须收缩");
        assert!(p.free_space() > before, "删除后应还出空间");

        // 再删末位，分配指针应正确回退到新的最低 cell
        p.remove_cell(2);
        assert_eq!(p.n_cells(), 2);
        assert_eq!(p.cell(0).unwrap(), vec![0u8]);
        assert_eq!(p.cell(1).unwrap(), vec![2u8]);
        assert!(p.cell(2).is_none());
    }

    #[test]
    fn push_cell_reports_full_page() {
        let mut p = Page::new(3, PT_DATA);
        let big = vec![7u8; PAGE_SIZE - PAGE_HEADER - 4];
        assert!(p.push_cell(&big).is_some(), "恰好放下");
        assert!(p.push_cell(&[1]).is_none(), "已满应拒绝");
        assert_eq!(p.n_cells(), 1);
    }

    #[test]
    fn from_raw_rejects_bad_input() {
        assert!(Page::from_raw(vec![0u8; PAGE_SIZE]).is_err(), "magic 不符");
        assert!(Page::from_raw(vec![0u8; 100]).is_err(), "长度不符");
        assert!(Page::from_raw(Page::new(9, PT_DATA).into_raw()).is_ok());
    }
}
