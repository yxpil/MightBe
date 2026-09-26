//! mightbe-reason 对外 API（README 4.8 / 8）。
//! 其他模块只允许 `use mightbe_reason::api::*`。

use mightbe_core::MtbResult;

/// 推理器：只读模块（业务数据零写入；关系结果写系统表是唯一写权限）。
pub trait Reasoner: Send + Sync {
    /// 网络前向（分类/回归/投影头）
    fn infer(&self, net: &str, text: &str) -> MtbResult<InferReport>;

    /// 联想检索（1 跳）
    fn associate(&self, net: &str, word: &str, k: usize) -> MtbResult<Vec<AssocHit>>;

    /// 多跳合理推理
    fn reason(&self, net: &str, query: &str, opts: ReasonOpts) -> MtbResult<ReasonReport>;
}

/// 关系学习器（字段/表关系）。
pub trait SchemaLearner: Send + Sync {
    fn learn_relations(&self, scope: LearnScope) -> MtbResult<JobId>;
    fn learn_rules(&self, scope: LearnScope) -> MtbResult<JobId>;
    fn field_relations(&self, db: &str) -> MtbResult<Vec<FieldRelation>>;
    fn table_relations(&self, db: &str) -> MtbResult<Vec<TableRelation>>;
}

#[derive(Debug, Clone)]
pub struct InferReport {
    /// 分类头的 softmax 分布（校准后）
    pub labels: Vec<Scored>,
    /// 投影头向量（若有）
    pub projection: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Scored {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct AssocHit {
    pub word: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct ReasonOpts {
    /// 每项语义见配置 `[network.reason]`（README 第 5 章）
    pub max_hops: u8,
    pub beam_width: usize,
    pub branch_factor: usize,
    pub depth_decay: f32,
    pub min_edge: f32,
    pub abstain_below: f32,
    pub need_evidence: bool,
    pub verifier_on: bool,
    /// 搜索域（README 8.8）
    pub scopes: Vec<Scope>,
}

impl Default for ReasonOpts {
    fn default() -> Self {
        Self {
            max_hops: 3,
            beam_width: 16,
            branch_factor: 8,
            depth_decay: 0.8,
            min_edge: 0.05,
            abstain_below: 0.25,
            need_evidence: true,
            verifier_on: true,
            scopes: vec![Scope::Word],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// 词/文档/标签
    Word,
    /// 字段/高频值
    Field,
    /// 表
    Table,
}

#[derive(Debug, Clone)]
pub struct ReasonReport {
    pub query: String,
    pub conclusions: Vec<Conclusion>,
    pub diagnostics: SearchDiagnostics,
}

#[derive(Debug, Clone)]
pub struct Conclusion {
    pub node: String,
    pub node_type: String,
    pub confidence: f32,
    /// 支撑路径（文本形式的推理链）
    pub paths: Vec<String>,
    /// 证据文档 id 与片段
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone)]
pub struct Evidence {
    pub source: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Default)]
pub struct SearchDiagnostics {
    pub hops_explored: usize,
    pub edges_pruned: usize,
    pub best_confidence: f32,
    pub blocked_edge_type: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LearnScope {
    pub full: bool,
}

pub type JobId = String;

#[derive(Debug, Clone)]
pub struct FieldRelation {
    pub left: String,
    pub right: String,
    pub kind: String,
    pub confidence: f32,
    pub support: u64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TableRelation {
    pub left: String,
    pub right: String,
    pub bridges: Vec<String>,
    pub confidence: f32,
}
