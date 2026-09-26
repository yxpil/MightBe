//! mightbe-sql 对外 API（README 4.5 / 9）。
//! 其他模块只允许 `use mightbe_sql::api::*`。

use mightbe_core::{MtbError, MtbResult};
use mightbe_store::Row;

/// SQL 引擎抽象。
pub trait SqlEngine: Send + Sync {
    /// 解析并执行一条语句（以 `;` 结尾）。返回结果集或受影响行数。
    fn execute(&self, session: SessionId, sql: &str) -> MtbResult<QueryOutcome>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(u64);

impl SessionId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub enum QueryOutcome {
    /// SELECT 结果集
    Rows { cols: Vec<String>, rows: Vec<Row> },
    /// INSERT/UPDATE/ALTER 等
    Affected { n: u64, info: String },
    /// TRAIN/LEARN 等长任务
    Job { job_id: String, info: String },
    /// 弃判/无结论等零行语义（README 7xxx 错误码的正面用法）
    Empty { info: String },
}

impl QueryOutcome {
    /// 语法错误统一构造入口（错误码区段见 README 第 9 章）
    pub fn syntax_err(msg: impl Into<String>) -> MtbError {
        MtbError::coded(1001, msg)
    }
}

/// 执行计划节点（火山模型）：`next() -> Option<Row>`。
pub trait PlanNode {
    fn next(&mut self) -> MtbResult<Option<Row>>;
}
