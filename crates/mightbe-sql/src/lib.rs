//! # mightbe-sql
//! MightBe SQL 方言：词法分析、递归下降解析、火山执行器、SQL 数学函数与向量检索入口。
//!
//! 模块边界：`sql -> {net, reason} -> {nlp, store} -> core`。
//! 错误码分带见 README 第 9 章（1xxx 语法 / 2xxx 表列 / 3xxx 网络配置插件 /
//! 4xxx 训练 / 5xxx 存储 / 6xxx NLP / 7xxx 推理 / 8xxx 关系学习）。

pub mod api;
