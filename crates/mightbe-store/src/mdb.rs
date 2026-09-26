//! MDB 文件容器（README 4.4 / 11.1 / 11.2 / 16.4）。
//!
//! 一个"库"就是一个目录，里面两个文件：
//!
//! ```text
//! data.mdb   定长页（每页 8 KiB，整块 AEAD 加密）
//! data.mlog  预写日志（帧级 AEAD，提交即截断尾部）
//! ```
//!
//! ### `data.mdb` 布局
//!
//! | 偏移 | 内容 |
//! |---|---|
//! | 0 | 超级块（80 B，**明文**：magic/版本/页大小/KDF 参数/salt/nonce/epoch + CRC32） |
//! | 80 | 第 0 页：KEK 密封的文件头（整块 AEAD） |
//! | 80 + n·8192 | 第 n 页：DEK 密封的数据页 |
//!
//! 超级块是明文的，因为**推导 KEK 必须先知道 salt**——盐和 nonce 本身不是秘密，
//! 而把口令挡在 AEAD 之后，"口令错/文件被改一个 bit ⇒ 打开即失败"这条验收才成立。
//! 于是双校验落地：超级块 CRC32 拦结构性损坏，头块/页块的 AEAD 拦密钥错误与位翻转。
//!
//! ### 页号分工
//!
//! | 页号 | 内容 |
//! |---|---|
//! | 0 | 文件头（Catalog 快照等） |
//! | 1 | 分配位图 |
//! | 2 | Catalog（表定义） |
//! | 3+ | 表堆等用户数据 |
//!
//! 崩溃恢复三步：开库 → 重放 WAL → 重放结果 checkpoint 回数据文件并把日志清空。

use mightbe_core::{MtbError, MtbResult};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::crypto::{KdfParams, MasterKeys};
use crate::page::{Page, PAGE_SIZE};
use crate::wal::{replay, Record, Wal};

/// 文件头里的固定页角色。
pub const PAGE_HEADER_PAGE: u64 = 0;
pub const PAGE_BITMAP: u64 = 1;
pub const PAGE_CATALOG: u64 = 2;
pub const PAGE_FIRST_USER: u64 = 3;

/// 格式版本。版本不符一律拒绝打开，不做静默降级。
pub const MDB_VERSION: u32 = 1;

/// AEAD 的附加数据标签：不同用途绝不共用 nonce 空间。
const AAD_HEADER: &[u8] = b"mdb/hdr";

/// 超级块长度，也是第 0 页的起始偏移。
const SUPER_LEN: u64 = 80;

const S_MAGIC: usize = 0;
const S_VERSION: usize = 4;
const S_PAGE_SIZE: usize = 8;
const S_KDF_M: usize = 12;
const S_KDF_T: usize = 16;
const S_KDF_P: usize = 20;
const S_KEK_SALT: usize = 24; // 16B
const S_DEK_NONCE: usize = 40; // 16B
const S_EPOCH: usize = 56; // 16B
const S_CRC: usize = 72;

// 头块明文布局。整页会被当作一个  使用，因此前 32 字节必须留给页头
// （magic / page_id / type / n_cells / first_free / crc）——把 version 写到偏移 0
// 会直接盖掉页的 magic，解出来就再也  不过。
const H_VERSION: usize = 32;
const H_KEK_SALT: usize = 36; // 16B
const H_DEK_NONCE: usize = 52; // 16B
const H_EPOCH: usize = 68; // 16B
const H_KDF_M: usize = 84;
const H_KDF_T: usize = 88;
const H_KDF_P: usize = 92;
const H_NEXT_HINT: usize = 104;
const H_N_PAGES: usize = 112;
const H_CATALOG_LEN: usize = 120;
const H_CATALOG_BLOB: usize = 124;

/// 超级块（明文）：开库最先生效的那一小段元信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperBlock {
    pub version: u32,
    pub page_size: u32,
    pub kdf: KdfParams,
    pub kek_salt: [u8; 16],
    pub dek_nonce: [u8; 16],
    pub epoch: [u8; 16],
}

impl SuperBlock {
    fn encode(&self) -> [u8; SUPER_LEN as usize] {
        let mut b = [0u8; SUPER_LEN as usize];
        b[S_MAGIC..S_MAGIC + 4].copy_from_slice(b"MDB1");
b[S_VERSION..S_VERSION + 4].copy_from_slice(&self.version.to_le_bytes());
b[S_PAGE_SIZE..S_PAGE_SIZE + 4].copy_from_slice(&self.page_size.to_le_bytes());
b[S_KDF_M..S_KDF_M + 4].copy_from_slice(&self.kdf.m_cost_kib.to_le_bytes());
b[S_KDF_T..S_KDF_T + 4].copy_from_slice(&self.kdf.t_cost.to_le_bytes());
b[S_KDF_P..S_KDF_P + 4].copy_from_slice(&self.kdf.p_cost.to_le_bytes());
        b[S_KEK_SALT..S_KEK_SALT + 16].copy_from_slice(&self.kek_salt);
        b[S_DEK_NONCE..S_DEK_NONCE + 16].copy_from_slice(&self.dek_nonce);
        b[S_EPOCH..S_EPOCH + 16].copy_from_slice(&self.epoch);
        let mut h = crc32fast::Hasher::new();
        h.update(&b[..S_CRC]);
        b[S_CRC..S_CRC + 4].copy_from_slice(&h.finalize().to_le_bytes());
        b
    }

    /// 解析超级块。CRC 不符 / magic 不符都返回 `None`。
    fn decode(raw: &[u8]) -> Option<Self> {
        if raw.len() < SUPER_LEN as usize || &raw[S_MAGIC..S_MAGIC + 4] != b"MDB1" {
            return None;
        }
        let mut h = crc32fast::Hasher::new();
        h.update(&raw[..S_CRC]);
        let want = u32::from_le_bytes(raw[S_CRC..S_CRC + 4].try_into().ok()?);
        if h.finalize() != want {
            return None;
        }
        let mut salt = [0u8; 16];
        let mut nonce = [0u8; 16];
        let mut epoch = [0u8; 16];
        salt.copy_from_slice(&raw[S_KEK_SALT..S_KEK_SALT + 16]);
        nonce.copy_from_slice(&raw[S_DEK_NONCE..S_DEK_NONCE + 16]);
        epoch.copy_from_slice(&raw[S_EPOCH..S_EPOCH + 16]);
        Some(Self {
            version: u32::from_le_bytes(raw[S_VERSION..S_VERSION + 4].try_into().ok()?),
            page_size: u32::from_le_bytes(raw[S_PAGE_SIZE..S_PAGE_SIZE + 4].try_into().ok()?),
            kdf: KdfParams {
                m_cost_kib: u32::from_le_bytes(raw[S_KDF_M..S_KDF_M + 4].try_into().ok()?),
                t_cost: u32::from_le_bytes(raw[S_KDF_T..S_KDF_T + 4].try_into().ok()?),
                p_cost: u32::from_le_bytes(raw[S_KDF_P..S_KDF_P + 4].try_into().ok()?),
            },
            kek_salt: salt,
            dek_nonce: nonce,
            epoch,
        })
    }
}

/// 文件头（明文布局；落盘前整页被 KEK 封装）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    pub version: u32,
    pub kek_salt: [u8; 16],
    pub dek_nonce: [u8; 16],
    pub epoch: [u8; 16],
    pub kdf: KdfParams,
    pub next_hint: u64,
    pub n_pages: u64,
    pub catalog: Vec<u8>,
}

impl FileHeader {
    fn to_page(&self) -> Page {
        let mut page = Page::new(PAGE_HEADER_PAGE, crate::page::PT_META);
        let r = page.as_bytes_mut();
r[H_VERSION..H_VERSION + 4].copy_from_slice(&self.version.to_le_bytes());
        r[H_KEK_SALT..H_KEK_SALT + 16].copy_from_slice(&self.kek_salt);
        r[H_DEK_NONCE..H_DEK_NONCE + 16].copy_from_slice(&self.dek_nonce);
        r[H_EPOCH..H_EPOCH + 16].copy_from_slice(&self.epoch);
r[H_KDF_M..H_KDF_M + 4].copy_from_slice(&self.kdf.m_cost_kib.to_le_bytes());
r[H_KDF_T..H_KDF_T + 4].copy_from_slice(&self.kdf.t_cost.to_le_bytes());
r[H_KDF_P..H_KDF_P + 4].copy_from_slice(&self.kdf.p_cost.to_le_bytes());
r[H_NEXT_HINT..H_NEXT_HINT + 8].copy_from_slice(&self.next_hint.to_le_bytes());
r[H_N_PAGES..H_N_PAGES + 8].copy_from_slice(&self.n_pages.to_le_bytes());
        let len = self.catalog.len().min(PAGE_SIZE - H_CATALOG_BLOB);
r[H_CATALOG_LEN..H_CATALOG_LEN + 4].copy_from_slice(&(len as u32).to_le_bytes());
        r[H_CATALOG_BLOB..H_CATALOG_BLOB + len].copy_from_slice(&self.catalog[..len]);
        page
    }

    fn from_page(page: &Page) -> Option<Self> {
        let r = page.as_bytes();
        let rd = |off: usize, n: usize| -> Option<Vec<u8>> {
            if off + n > r.len() {
                None
            } else {
                Some(r[off..off + n].to_vec())
            }
        };
        let mut salt = [0u8; 16];
        let mut dek = [0u8; 16];
        let mut epoch = [0u8; 16];
        salt.copy_from_slice(&rd(H_KEK_SALT, 16)?[..]);
        dek.copy_from_slice(&rd(H_DEK_NONCE, 16)?[..]);
        epoch.copy_from_slice(&rd(H_EPOCH, 16)?[..]);
        let len = u32::from_le_bytes(rd(H_CATALOG_LEN, 4)?.try_into().ok()?) as usize;
        if len > PAGE_SIZE - H_CATALOG_BLOB {
            return None;
        }
        Some(Self {
            version: u32::from_le_bytes(rd(H_VERSION, 4)?.try_into().ok()?),
            kek_salt: salt,
            dek_nonce: dek,
            epoch,
            kdf: KdfParams {
                m_cost_kib: u32::from_le_bytes(rd(H_KDF_M, 4)?.try_into().ok()?),
                t_cost: u32::from_le_bytes(rd(H_KDF_T, 4)?.try_into().ok()?),
                p_cost: u32::from_le_bytes(rd(H_KDF_P, 4)?.try_into().ok()?),
            },
            next_hint: u64::from_le_bytes(rd(H_NEXT_HINT, 8)?.try_into().ok()?),
            n_pages: u64::from_le_bytes(rd(H_N_PAGES, 8)?.try_into().ok()?),
            catalog: rd(H_CATALOG_BLOB, len)?,
        })
    }
}

/// 一个打开的 MDB 库。
///
/// 内存里持有一份明文页缓存；**脏页绝不直接写数据文件**——先按 write-ahead 写 WAL，
/// 提交时才落盘。因此任何时刻被 kill，重放日志即可还原到已提交状态。
pub struct Database {
    dir: PathBuf,
    data: std::fs::File,
    wal: Wal,
    keys: MasterKeys,
    super_block: SuperBlock,
    /// 明文页缓存（含未落盘的脏页）
    cache: HashMap<u64, Page>,
    dirty: HashSet<u64>,
    next_page: u64,
    n_pages: u64,
    txn_open: bool,
    header_dirty: bool,
}

impl Database {
    // ───────────────────────── 开 / 建 ─────────────────────────

    /// 建立新库。目录不存在时自动创建。
    pub fn create(dir: &Path, password: &[u8], kdf: &KdfParams) -> MtbResult<Self> {
        std::fs::create_dir_all(dir)?;
        let data_path = Self::data_path(dir);
        if data_path.exists() {
            return Err(MtbError::coded(
                MtbError::STORE + 30,
                format!("库已存在: {}", dir.display()),
            ));
        }
        let kek_salt = new_salt();
        let dek_nonce = new_salt();
        let epoch = new_salt();
        let keys = MasterKeys::derive(password, &kek_salt, kdf, &dek_nonce, epoch)?;

        let data = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&data_path)?;
        let wal = Wal::create(&Self::wal_path(dir), &keys)?;

        let mut db = Self {
            dir: dir.to_path_buf(),
            data,
            wal,
            keys,
            super_block: SuperBlock {
                version: MDB_VERSION,
                page_size: PAGE_SIZE as u32,
                kdf: *kdf,
                kek_salt,
                dek_nonce,
                epoch,
            },
            cache: HashMap::new(),
            dirty: HashSet::new(),
            next_page: PAGE_FIRST_USER,
            n_pages: PAGE_FIRST_USER,
            txn_open: false,
            header_dirty: false,
        };

        db.cache.insert(PAGE_BITMAP, Page::new(PAGE_BITMAP, crate::page::PT_META));
        db.dirty.insert(PAGE_BITMAP);
        db.cache.insert(PAGE_CATALOG, Page::new(PAGE_CATALOG, crate::page::PT_META));
        db.dirty.insert(PAGE_CATALOG);
        db.begin()?;
        db.write_header()?;
        db.commit()?;
        Ok(db)
    }

    /// 打开既有库；若有 WAL 则先做崩溃恢复。
    pub fn open(dir: &Path, password: &[u8]) -> MtbResult<Self> {
        let data_path = Self::data_path(dir);
        if !data_path.exists() {
            return Err(MtbError::coded(
                MtbError::STORE + 31,
                format!("库不存在: {}", dir.display()),
            ));
        }
        let data = OpenOptions::new().read(true).write(true).open(&data_path)?;
        let on_disk_pages = data
            .metadata()
            .map(|m| (m.len() - SUPER_LEN) / BLOCK_SIZE)
            .unwrap_or(0);

        // salt 在明面的超级块里，所以 KEK 可以先把派生出来，再去解头块。
        //
        // 超级块独占文件的前 80 字节，第 0 页的密文块从 80 开始——两者不能混读：
        // 拿页块的前 80 字节去解超级块，等价于把头块密文当成盐，开库必然失败。
        let raw_head = read_block(&data, PAGE_HEADER_PAGE)?;
        let raw_super = read_super(&data)?;
        let super_block = SuperBlock::decode(&raw_super).ok_or_else(|| {
            MtbError::coded(
                MtbError::STORE + 32,
                "MDB 超级块损坏（magic 不符或 CRC 校验失败）".to_string(),
            )
        })?;
        if super_block.version != MDB_VERSION {
            return Err(MtbError::coded(
                MtbError::STORE + 33,
                format!(
                    "MDB 版本 {} 与宿主支持的 {} 不符",
                    super_block.version, MDB_VERSION
                ),
            ));
        }
        if super_block.page_size != PAGE_SIZE as u32 {
            return Err(MtbError::coded(
                MtbError::STORE + 34,
                format!("页大小 {} 与宿主不一致", super_block.page_size),
            ));
        }
        let keys = MasterKeys::derive(
            password,
            &super_block.kek_salt,
            &super_block.kdf,
            &super_block.dek_nonce,
            super_block.epoch,
        )?;

        let header = {
            let plain = keys
                .open(AAD_HEADER, PAGE_HEADER_PAGE, &raw_head)
                .ok_or_else(|| {
                    MtbError::coded(
                        MtbError::STORE + 35,
                        "口令错误或文件头被篡改：无法解开 MDB（不泄露任何明文）".to_string(),
                    )
                })?;
            let page = Page::from_raw(plain)?;
            if !page.crc_ok() {
                return Err(MtbError::coded(
                    MtbError::STORE + 36,
                    "文件头 CRC 校验失败".to_string(),
                ));
            }
            FileHeader::from_page(&page).ok_or_else(|| {
                MtbError::coded(MtbError::STORE + 37, "文件头格式无法识别".to_string())
            })?
        };

        let wal = Wal::open(&Self::wal_path(dir), &keys)?;
        let mut db = Self {
            dir: dir.to_path_buf(),
            data,
            wal,
            keys,
            super_block,
            cache: HashMap::new(),
            dirty: HashSet::new(),
            next_page: header.next_hint.max(PAGE_FIRST_USER),
            n_pages: header.n_pages.max(on_disk_pages),
            txn_open: false,
            header_dirty: false,
        };
        db.recover()?;
        Ok(db)
    }

    /// 崩溃恢复：重放 WAL → checkpoint → 清空日志。
    fn recover(&mut self) -> MtbResult<()> {
        let r = replay(self.wal.path(), &self.keys)?;
        if r.puts.is_empty() && r.drops.is_empty() {
            return Ok(());
        }
        // 顺序套用：后写覆盖先写，删除在写入之后生效。
        // 拆成两个串行循环（而不是把两个闭包交给 `Replay::apply`）是因为二者
        // 都要可变借用同一个 `Database`，闭包表达不了这种重叠借用。
        for (id, img) in &r.puts {
            if let Ok(p) = Page::from_raw(img.clone()) {
                self.cache.insert(*id, p);
                self.dirty.insert(*id);
            }
        }
        for id in &r.drops {
            self.cache.remove(id);
            self.dirty.remove(id);
            let mut bm = self.page_or_zero(PAGE_BITMAP);
            clear_bit(bm.as_bytes_mut(), *id);
            bm.seal();
            self.cache.insert(PAGE_BITMAP, bm);
            self.dirty.insert(PAGE_BITMAP);
        }
        self.checkpoint()?;
        self.wal.clear()?;
        Ok(())
    }

    fn page_or_zero(&self, id: u64) -> Page {
        self.cache
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Page::new(id, crate::page::PT_DATA))
    }

    // ───────────────────────── 页访问 ─────────────────────────

    /// 取页：优先缓存，其次读盘 → 解密 → 校验 CRC 与页号。
    pub fn page(&mut self, id: u64) -> MtbResult<Page> {
        if let Some(p) = self.cache.get(&id) {
            return Ok(p.clone());
        }
        let raw = read_block(&self.data, id)?;
        let plain = self.keys.open_page(id, &raw).map_err(|e| match e {
            MtbError::Coded { code, .. } if code == MtbError::STORE + 14 => MtbError::coded(
                MtbError::STORE + 40,
                format!("页 {id} 解密失败（密钥不符或密文被篡改）"),
            ),
            other => other,
        })?;
        let page = Page::from_raw(plain)?;
        if !page.crc_ok() {
            return Err(MtbError::coded(
                MtbError::STORE + 41,
                format!("页 {id} CRC 校验失败"),
            ));
        }
        if page.page_id() != id {
            return Err(MtbError::coded(
                MtbError::STORE + 42,
                format!("页 {id} 的内容却自称页 {}", page.page_id()),
            ));
        }
        self.cache.insert(id, page.clone());
        Ok(page)
    }

    /// 写页：write-ahead —— 先记 WAL，再进脏页集合。
    pub fn put_page(&mut self, page: Page) {
        let id = page.page_id();
        if self.txn_open {
            self.log_page(&page);
        }
        self.cache.insert(id, page);
        self.dirty.insert(id);
    }

    /// 往日志里落一条整页重做记录。
    ///
    /// 日志写不进去就没有提交的意义，直接让调用方看到原因，
    /// 而不是"静默丢掉一条重做记录"。
    fn log_page(&mut self, page: &Page) {
        self.wal
            .append(
                &self.keys,
                &Record::PutPage {
                    page_id: page.page_id(),
                    image: page.as_bytes().to_vec(),
                },
            )
            .expect("写 WAL 失败");
    }

    /// 删页：记一条 `DropPage` 并重定位位图。
    pub fn drop_page(&mut self, id: u64) {
        if id < PAGE_FIRST_USER {
            return; // 固定页不可删
        }
        if self.txn_open {
            self.wal
                .append(&self.keys, &Record::DropPage { page_id: id })
                .expect("写 WAL 失败");
        }
        self.cache.remove(&id);
        self.dirty.remove(&id);
        let mut bm = self.page_or_zero(PAGE_BITMAP);
        clear_bit(bm.as_bytes_mut(), id);
        bm.seal();
        self.cache.insert(PAGE_BITMAP, bm);
        self.dirty.insert(PAGE_BITMAP);
    }

    /// 分配一页：在位图里找第一个空位。
    ///
    /// 位图覆盖前 65536 页（512 MiB）；用尽时给出明确错误，而不是静默失败。
    pub fn alloc_page(&mut self) -> MtbResult<u64> {
        let mut bm = self.page(PAGE_BITMAP)?;
        let raw = bm.as_bytes().to_vec();
        let Some(id) = find_free_bit(&raw) else {
            return Err(MtbError::coded(
                MtbError::STORE + 43,
                "分配位图已满（单库上限 65536 页 / 512 MiB）".to_string(),
            ));
        };
        set_bit(bm.as_bytes_mut(), id);
        bm.seal();
        self.cache.insert(PAGE_BITMAP, bm);
        self.dirty.insert(PAGE_BITMAP);

        let mut fresh = Page::new(id, crate::page::PT_DATA);
        fresh.seal();
        // 空白新页也要进日志：它是页链的一环，重放时缺了它整条链就断了。
        if self.txn_open {
            self.log_page(&fresh);
        }
        self.cache.insert(id, fresh);
        self.dirty.insert(id);
        self.next_page = self.next_page.max(id + 1);
        self.n_pages = self.n_pages.max(id + 1);
        self.header_dirty = true;
        Ok(id)
    }

    // ───────────────────────── 事务 ─────────────────────────

    pub fn begin(&mut self) -> MtbResult<()> {
        if self.txn_open {
            return Err(MtbError::coded(MtbError::STORE + 44, "已有事务在进行"));
        }
        self.wal.append(&self.keys, &Record::Begin)?;
        self.txn_open = true;
        Ok(())
    }

    /// 提交：WAL 落盘并截断到提交点，随后 checkpoint。
    pub fn commit(&mut self) -> MtbResult<()> {
        if !self.txn_open {
            return Err(MtbError::coded(MtbError::STORE + 45, "没有进行中的事务"));
        }
        self.wal.commit(&self.keys)?;
        self.txn_open = false;
        self.checkpoint()?;
        Ok(())
    }

    /// 回滚：丢弃脏页缓存。未提交的 WAL 尾部在下次开库重放时自然被跳过
    /// （它们位于最后一个 `Commit` 之后）。
    pub fn rollback(&mut self) -> MtbResult<()> {
        if !self.txn_open {
            return Err(MtbError::coded(MtbError::STORE + 46, "没有进行中的事务"));
        }
        self.cache.clear();
        self.dirty.clear();
        self.txn_open = false;
        Ok(())
    }

    pub fn in_txn(&self) -> bool {
        self.txn_open
    }

    // ───────────────────────── 落盘 ─────────────────────────

    /// 把脏页写回数据文件；提交与显式 checkpoint 都会调用。
    pub fn checkpoint(&mut self) -> MtbResult<()> {
        let ids: Vec<u64> = self.dirty.iter().copied().collect();
        for id in ids {
            let Some(page) = self.cache.get(&id) else {
                continue;
            };
            let ct = self.keys.seal_page(id, page.as_bytes())?;
            write_block(&mut self.data, id, &ct)?;
        }
        if self.header_dirty {
            let page = self.header_page()?;
            let ct = self.seal_header(&page)?;
            write_block(&mut self.data, PAGE_HEADER_PAGE, &ct)?;
            self.header_dirty = false;
        }
        self.dirty.clear();
        self.data.sync_all()?;
        Ok(())
    }

    /// 关库：落盘 + 清空日志，保证下次打开不必恢复。
    pub fn close(mut self) -> MtbResult<()> {
        if self.txn_open {
            self.rollback()?;
        }
        self.checkpoint()?;
        self.wal.clear()?;
        Ok(())
    }

    /// 头块专用封装。
    ///
    /// 用途标签必须是 `mdb/hdr` 而不是页块的 `mdb/page`：`aad` 参与 nonce 派生，
    /// 头块与第 0 页一旦共用同一条 nonce 空间，两处密文就会互相顶掉。
    /// 写、读、轮换三处都必须走这里，不能各写各的。
    fn seal_header(&self, page: &Page) -> MtbResult<Vec<u8>> {
        self.keys
            .seal(AAD_HEADER, PAGE_HEADER_PAGE, page.as_bytes())
    }

    /// 组装当前文件头页（含 Catalog 快照），并以 KEK 重新密封。
    fn header_page(&self) -> MtbResult<Page> {
        let catalog = self
            .cache
            .get(&PAGE_CATALOG)
            .and_then(|p| p.cell(0))
            .unwrap_or_default();
        let h = FileHeader {
            version: MDB_VERSION,
            kek_salt: self.super_block.kek_salt,
            dek_nonce: self.super_block.dek_nonce,
            epoch: *self.keys.epoch(),
            kdf: self.super_block.kdf,
            next_hint: self.next_page,
            n_pages: self.n_pages,
            catalog,
        };
        let mut page = h.to_page();
        page.set_page_id(PAGE_HEADER_PAGE);
        page.seal();
        Ok(page)
    }

    /// 写超级块（salt/nonce 等，明文）+ 头块（密文）。
    fn write_header(&mut self) -> MtbResult<()> {
        let sb = self.super_block;
        write_super(&mut self.data, &sb)?;
        let page = self.header_page()?;
        let ct = self.seal_header(&page)?;
        write_block(&mut self.data, PAGE_HEADER_PAGE, &ct)?;
        Ok(())
    }

    // ───────────────────────── 密钥轮换 ─────────────────────────

    /// 轮换密钥（README 11.1 / 16.4）：换 salt、nonce、epoch 与口令，
    /// 整库按新 DEK 重写一遍。
    ///
    /// 解密后的内容逐字节不变（这正是"轮换前后数据一致"的验收口径）；
    /// epoch 前进同时作废旧 nonce 空间，杜绝轮换途中的密文复用。
    pub fn rotate_key(&mut self, new_password: &[u8]) -> MtbResult<()> {
        if self.txn_open {
            return Err(MtbError::coded(
                MtbError::STORE + 47,
                "轮换密钥前请先提交事务",
            ));
        }
        if !self.wal.is_empty() {
            self.checkpoint()?;
            self.wal.clear()?;
        }
        let mut epoch = *self.keys.epoch();
        epoch[0] = epoch[0].wrapping_add(1);
        if epoch == *self.keys.epoch() {
            epoch[1] ^= 0xA5;
        }
        let kek_salt = new_salt();
        let dek_nonce = new_salt();
        let mut sb = self.super_block;
        sb.kek_salt = kek_salt;
        sb.dek_nonce = dek_nonce;
        sb.epoch = epoch;
        let new_keys = MasterKeys::derive(
            new_password,
            &kek_salt,
            &sb.kdf,
            &dek_nonce,
            epoch,
        )?;
        let old_keys = std::mem::replace(&mut self.keys, new_keys);
        self.super_block = sb;

        // 1. 逐页：旧 DEK 解 → 新 DEK 封（头块单独处理）
        let ids: Vec<u64> = {
            let mut v: Vec<u64> = self.cache.keys().copied().collect();
            v.extend(0..self.n_pages);
            v.sort_unstable();
            v.dedup();
            v.retain(|id| *id < self.n_pages && *id != PAGE_HEADER_PAGE);
            v
        };
        for id in ids {
            let plain = self
                .cache
                .get(&id)
                .map(|p| p.as_bytes().to_vec())
                .or_else(|| {
                    let ct = read_block_owned(&self.data, id).ok()?;
                    old_keys.open_page(id, &ct).ok()
                });
            let Some(plain) = plain else {
                continue; // 还没分配过的空洞页
            };
            let ct = self.keys.seal_page(id, &plain)?;
            write_block(&mut self.data, id, &ct)?;
        }

        // 2. 头块（用新 KEK 重新密封）+ 明面超级块
        let page = self.header_page()?;
        let ct = self.seal_header(&page)?;
        write_block(&mut self.data, PAGE_HEADER_PAGE, &ct)?;
        let sb = self.super_block;
        write_super(&mut self.data, &sb)?;
        self.data.sync_all()?;
        Ok(())
    }

    // ───────────────────────── 备份 / 恢复 ─────────────────────────

    /// 备份：整目录快照。有未落盘的日志时先 checkpoint。
    pub fn backup(&mut self, dst: &Path) -> MtbResult<()> {
        if self.txn_open {
            return Err(MtbError::coded(
                MtbError::STORE + 49,
                "备份前请先提交事务",
            ));
        }
        if !self.wal.is_empty() {
            self.checkpoint()?;
            self.wal.clear()?;
        }
        std::fs::create_dir_all(dst)?;
        std::fs::copy(Self::data_path(&self.dir), Self::data_path(dst))?;
        copy_wal(&Self::wal_path(&self.dir), &Self::wal_path(dst))?;
        Ok(())
    }

    /// 恢复：把备份的数据文件盖回来。
    pub fn restore_from(src: &Path, dst: &Path) -> MtbResult<()> {
        if !Self::data_path(src).exists() {
            return Err(MtbError::coded(
                MtbError::STORE + 50,
                format!("备份不存在: {}", src.display()),
            ));
        }
        std::fs::create_dir_all(dst)?;
        std::fs::copy(Self::data_path(src), Self::data_path(dst))?;
        // 日志也要一起盖回来：只换数据文件、却留下新库那份日志，等于把恢复
        // 抹掉了。备份里必然带着一份空日志，缺了它 `Wal::open` 直接 NotFound。
        copy_wal(&Self::wal_path(src), &Self::wal_path(dst))?;
        Ok(())
    }

    // ───────────────────────── 杂项 ─────────────────────────

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn data_file(&self) -> PathBuf {
        Self::data_path(&self.dir)
    }

    pub fn wal_file(&self) -> PathBuf {
        Self::wal_path(&self.dir)
    }

    pub fn keys(&self) -> &MasterKeys {
        &self.keys
    }

    pub fn super_block(&self) -> &SuperBlock {
        &self.super_block
    }

    pub fn n_pages(&self) -> u64 {
        self.n_pages
    }

    pub fn data_len(&self) -> u64 {
        self.data.metadata().map(|m| m.len()).unwrap_or(0)
    }

    /// WAL 是否还有可重放的帧。
    pub fn wal_has_records(&self) -> bool {
        !self.wal.is_empty()
    }

    /// 直接读回第 `id` 页的密文（"磁盘上到底有什么"这类断言用）。
    pub fn raw_page(&self, id: u64) -> MtbResult<Vec<u8>> {
        read_block(&self.data, id)
    }

    fn data_path(dir: &Path) -> PathBuf {
        dir.join("data.mdb")
    }

    fn wal_path(dir: &Path) -> PathBuf {
        dir.join("data.mlog")
    }
}

/// 把源 WAL 搬到目标目录；源不存在时留一个空壳。
///
/// 0 字节的 WAL 是合法的——`Wal::open` 读不满超级帧会按"新建"处理；
/// 但**文件缺失**不是，那会直接 `NotFound`。备份能不能独立打开就取决于这点。
fn copy_wal(src: &std::path::Path, dst: &std::path::Path) -> MtbResult<()> {
    let _ = std::fs::remove_file(dst);
    match std::fs::copy(src, dst) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::File::create(dst)?;
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

// ─────────────────────── 位图 ───────────────────────

/// 进程内自增序号：保证同一毫秒内连开两次库、连做两次 ROTATE 也拿到不同的盐。
static SALT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 生成 16 字节盐。
///
/// 盐不需要保密，只需要**不重复**：时钟给出粗粒度唯一性，序号补上同一纳秒内的
/// 多次调用。xorshift 一巡打散成 16 字节（不是简单截断，避免出现大量前导零）。
fn new_salt() -> [u8; 16] {
    use std::sync::atomic::Ordering;
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0) as u64;
    let seq = SALT_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut x = ms ^ 0x9E_37_79_B9_7F_4A_7C_15 ^ (seq << 41) ^ (ms << 17);
    let mut s = [0u8; 16];
    for slot in s.iter_mut() {
        x = x
            .wrapping_mul(0x9E_37_79_B9_7F_4A_7C_15)
            .wrapping_add(0x85_EB_CA_6B_9E_37_79_B9);
        x ^= x >> 29;
        *slot = (x >> 33) as u8;
    }
    s
}

/// 位图从页头之后开始按位排布；返回第一个空位的页号。
fn find_free_bit(raw: &[u8]) -> Option<u64> {
    let base = crate::page::PAGE_HEADER;
    for (i, &b) in raw.iter().enumerate().skip(base) {
        if b == 0xff {
            continue;
        }
        for k in 0..8u32 {
            if b & (1 << k) == 0 {
                let id = ((i - base) as u64) * 8 + k as u64;
                // 固定页（头/位图/Catalog）永不参与分配：位图从头扫，
                // 不跳过它们的话第一张表的根页就会顶到文件头页上。
                if id < PAGE_FIRST_USER {
                    continue;
                }
                return Some(id);
            }
        }
    }
    None
}

fn set_bit(raw: &mut [u8], id: u64) {
    if let Some(b) = raw.get_mut(crate::page::PAGE_HEADER + (id / 8) as usize) {
        *b |= 1 << (id % 8) as u8;
    }
}

fn clear_bit(raw: &mut [u8], id: u64) {
    if let Some(b) = raw.get_mut(crate::page::PAGE_HEADER + (id / 8) as usize) {
        *b &= !(1 << (id % 8) as u8);
    }
}

// ───────────────────────── 裸块读写 ─────────────────────────

/// GCM 的认证标签长度。密文恒为 `明文 + 标签`，所以一个页块必然是
/// `PAGE_SIZE + 16` 字节——想让文件长度正好是页大小的整数倍是做不到的，
/// 除非把标签塞进别处（那等于自造第二个头，风险更大）。
/// 索性按固定块长对齐：`块 n` 占 `SUPER_LEN + n·BLOCK_SIZE`。
const TAG_LEN: usize = 16;

/// 一个页块在文件里的字节数。
pub const BLOCK_SIZE: u64 = PAGE_SIZE as u64 + TAG_LEN as u64;

/// 第 0 页紧跟在 80 字节超级块之后；其余页块等距排开。
fn block_offset(page_id: u64) -> u64 {
    if page_id == 0 {
        SUPER_LEN
    } else {
        SUPER_LEN + page_id * BLOCK_SIZE
    }
}

/// 第 0 页之前有一段 80 字节的明面超级块，因此"页 n"的落盘位置不是 `n·PAGE_SIZE`
/// 而是 `block_offset(n)`——切换读写函数时最容易踩的坑就是这里漏掉 `SUPER_LEN`。
fn read_block(mut f: &std::fs::File, page_id: u64) -> MtbResult<Vec<u8>> {
    // 读的是"页 + 标签"这么长的一整块。按页长读会吞掉下一块的前 16 字节，
    // 第 0 页尤其致命：偏移 80 处的头块密文会被误当成超级块。
    let mut buf = vec![0u8; BLOCK_SIZE as usize];
    f.seek(SeekFrom::Start(block_offset(page_id)))?;
    f.read_exact(&mut buf).map_err(|e| {
        MtbError::coded(
            MtbError::STORE + 51,
            format!("读页 {page_id} 失败（文件被截断？）: {e}"),
        )
    })?;
    Ok(buf)
}

/// 读明面超级块：文件最前面那 80 字节，与页块读写完全分离。
fn read_super(mut f: &std::fs::File) -> MtbResult<Vec<u8>> {
    let mut buf = vec![0u8; SUPER_LEN as usize];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut buf).map_err(|e| {
        MtbError::coded(
            MtbError::STORE + 52,
            format!("读超级块失败（文件被截断？）: {e}"),
        )
    })?;
    Ok(buf)
}

fn read_block_owned(f: &std::fs::File, page_id: u64) -> MtbResult<Vec<u8>> {
    read_block(f, page_id)
}

fn write_block(f: &mut std::fs::File, page_id: u64, block: &[u8]) -> MtbResult<()> {
    debug_assert_eq!(block.len(), PAGE_SIZE + TAG_LEN, "块长度必须等于页 + 标签");
    f.seek(SeekFrom::Start(block_offset(page_id)))?;
    f.write_all(block)?;
    Ok(())
}

fn write_super(f: &mut std::fs::File, sb: &SuperBlock) -> MtbResult<()> {
        let b = sb.encode();
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&b)?;
        Ok(())
    }
