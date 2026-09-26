//! # mightbe-cli
//! MightBe 命令行客户端（README 4.7）。
//!
//! 与 server 同仓库但物理隔离：本 crate 不依赖任何内部 crate，只依赖线协议，
//! 以此证明"客户端不需要理解引擎内部"。

pub mod render;
pub mod wire;
