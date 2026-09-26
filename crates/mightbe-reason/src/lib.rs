//! # mightbe-reason
//! MightBe 推理层（README 4.8 / 8）：联想检索、多跳合理推理、字段/表关系学习与规则挖掘。
//!
//! 模块边界：依赖 core/nlp/net/store（只读视图，禁止反向依赖）；对外仅暴露 [`api`]。
//!
//! 规划结构（M6 填充）：
//! - `graph`      统一推理图只读视图（词/文档/值/字段/表节点 + 10 类边）
//! - `search`     束搜索 + 扩散激活 + 环剪枝 + Noisy-OR 汇聚
//! - `verifier`   网络 Head 成对验证重排
//! - `calibrate`  温度缩放 / 保序回归（参数存 `mb_calibration`）
//! - `evidence`   文档/字段证据回链
//! - `schema`     画像（profile）→ blocking → 特征 → 关系网络 → `mb_field_relations`
//! - `rules`      LEARN RULES 路径模式挖掘（8.9）

pub mod api;

pub use api::*;
