//! # mightbe-net
//! MightBe 网络运行时（README 4.2 / 5 / 7）。
//!
//! 模块边界：依赖 core/plugin/nlp/store（禁止反向依赖）；对外仅暴露 [`api`]。
//!
//! 规划结构（M4 填充）：
//! - `spec`      TOML 头格式/层/Head/train/assoc/reason 配置解析与校验
//! - `registry`  多网络多版本管理 + arc-swap 快照
//! - `builder`   配置 → 层实例装配（内置/复合/插件三条路径）
//! - `weights`   参数寻址 `(net, layer, param)` 与迁移
//! - `trainer`   后台 TRAIN 任务、checkpoint 落 `.mbm`

pub mod api;

pub use api::*;
