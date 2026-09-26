//! # mightbe-nlp
//! MightBe NLP 管线（README 4.3 / 8.1）。
//!
//! 模块边界：只依赖 mightbe-core；对外仅暴露 [`api`]。
//!
//! 三层管线（全部数据驱动，无外部 NLP 库）：
//! ```text
//! 原文 ──► normalize ──► tokenizer ──► vocab ──► 特征编码器
//!                                         ├─► keyword (TF-IDF / TextRank)
//!                                         └─► cooccur (共现 → PMI)
//! ```

pub mod api;
pub mod cooccur;
pub mod index;
pub mod keyword;
pub mod normalize;
pub mod tokenizer;
pub mod vocab;

pub use api::*;
