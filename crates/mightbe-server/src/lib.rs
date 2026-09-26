//! # mightbe-server
//! MightBe 服务层（README 4.6）：装配、TCP 行协议、会话、优雅停机。
//!
//! 模块边界：`server -> sql -> {net, reason} -> {nlp, store} -> core`。
//! 本 crate 是唯一做依赖注入的地方——各内部 crate 以 trait object 装配，无全局单例。
//!
//! 规划结构（M8 填充）：
//! - [`config`]   `config/mightbe.toml` 的解析与默认值兜底
//! - `assembly`   Backend → Store → NlpPipeline → NetworkRegistry → SqlEngine 的启动装配
//! - `protocol`   行分隔文本协议的编解码（`OK` / 数据行 / `END` / `ERR`）
//! - `session`    连接会话、`USE` 切库、语句分发
//! - `jobs`       TRAIN/LEARN 后台任务登记表与 `SHOW TRAINING` 进度
//! - `shutdown`   SIGINT → 停止接收 → 检查点 → WAL flush

pub mod api;
pub mod config;

pub use api::*;
