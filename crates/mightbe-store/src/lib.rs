//! # mightbe-store
//! MightBe 存储引擎（README 4.4 / 11）。
//!
//! 模块边界：只依赖 mightbe-core；对外仅暴露 [`api`]。
//!
//! 实际结构（M2 落地）：
//! - `page`      8KiB 页布局与 Slot/Cell 分配
//! - `crypto`    MDB 加密（Argon2id KEK/DEK、AES-256-GCM 页加密、nonce 派生）
//! - `wal`       预写日志（帧级 AEAD，提交即截断尾部）
//! - `mdb`       文件容器（超级块 / 页 / 位图 / 事务 / 恢复 / 轮换 / 备份）
//! - `table`     Catalog 编解码与表堆（CREATE / INSERT / SELECT / DELETE）
//!
//! 尚未落地：`meta`（系统表）、`bufferpool`（LRU 明文缓存）、`backup`（留给
//! `mdb::Database::backup` 这一层薄封装，等 sql 层接进来再补齐）。
//!
//! 依赖方向：`store → core`，不依赖 sql / nlp / reason / net / plugin。

pub mod api;
pub mod crypto;
pub mod mdb;
pub mod page;
pub mod table;
pub mod wal;

pub use api::*;
