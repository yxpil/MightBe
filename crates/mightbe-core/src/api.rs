//! mightbe-core 对外 API：只暴露 trait 与 DTO。
//! 其他模块（net/nlp/reason/sql/server）只允许 `use mightbe_core::api::*`。
//! 对应 README 4.1 节。
//!
//! 本文件的类型即模块边界的全部内容：`Tensor` / `Var` / `Graph` / `Backend` /
//! `Layer` / `Optimizer` 及其 DTO。具体算子实现位于同 crate 的 `tensor`、`ops`、
//! `backend`、`layers`、`optim` 等私有模块。

use std::error::Error as StdError;

/// 全框架统一错误类型。模块前缀用于定位来源。
#[derive(Debug, thiserror::Error)]
pub enum MtbError {
    #[error("[{code}] {message}")]
    Coded { code: u16, message: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("shape mismatch: expected {expected}, got {got}")]
    Shape { expected: String, got: String },

    #[error("config invalid: {0}")]
    Config(String),

    #[error("device {device} unavailable: {reason}")]
    DeviceUnavailable { device: String, reason: String },

    #[error("plugin abi mismatch: {0}")]
    PluginAbi(String),

    #[error("{0}")]
    Other(String),
}

/// 错误码区段（README 第 9 章）。统一从这里取，避免各处硬编码。
impl MtbError {
    /// 1xxx 语法
    pub const SYNTAX: u16 = 1000;
    /// 2xxx 表/列
    pub const TABLE: u16 = 2000;
    /// 3xxx 网络/配置/插件
    pub const NETWORK: u16 = 3000;
    pub const CONFIG_INVALID: u16 = 3001;
    pub const PLUGIN_ABI: u16 = 3101;
    pub const PLUGIN_SYM_MISSING: u16 = 3102;
    pub const PLUGIN_SMOKE_FAIL: u16 = 3103;
    /// 4xxx 训练
    pub const TRAIN: u16 = 4000;
    /// 5xxx 存储
    pub const STORE: u16 = 5000;
    /// 6xxx NLP
    pub const NLP: u16 = 6000;
    /// 7xxx 推理（7001 弃判 / 7002 证据不足）
    pub const REASON: u16 = 7000;
    /// 8xxx 关系学习
    pub const SCHEMA: u16 = 8000;

    pub fn coded(code: u16, msg: impl Into<String>) -> Self {
        Self::Coded { code, message: msg.into() }
    }
}

pub type MtbResult<T> = Result<T, MtbError>;

// ───────────────────────── Shape / Tensor ─────────────────────────

/// 形状描述。MVP 一律行主序，strides 由 `Tensor` 自己维护。
pub type Shape = Vec<usize>;

/// 行主序多维张量。Phase 1 仅 CPU f32。
///
/// 定义见 `crate::tensor`；此处同名词再导出，保证 `api::*` 是外部模块唯一入口。
pub use crate::tensor::Tensor;

// ───────────────────────── 计算图 ─────────────────────────

pub use crate::autograd::{Graph, Op, Var};

/// 一个前向/反向的计算会话。训练步结束后整体丢弃（内存随 step 释放）。
pub type Session = Graph;

// ───────────────────────── Backend ─────────────────────────

/// 计算后端（naive / simd-avx2 / cuda:0 / npu:0）。见 README 4.1。
pub trait Backend: Send + Sync {
    fn id(&self) -> &str;

    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], g: &GemmShapes) -> MtbResult<()>;

    fn conv1d(
        &self,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        out: &mut [f32],
        p: &ConvParams,
    ) -> MtbResult<()>;

    fn elementwise(&self, out: &mut [f32], op: ElemOp) -> MtbResult<()>;

    /// 沿 `axis` 归约：`x` 是形状 `shape` 的行主序数据，`out` 长度须为归约后元素数。
    fn reduce(
        &self,
        x: &[f32],
        shape: &[usize],
        out: &mut [f32],
        axis: usize,
        r: ReduceKind,
    ) -> MtbResult<()>;

    fn embedding_lookup(
        &self,
        emb: &[f32],
        ids: &[u32],
        out: &mut [f32],
        dim: usize,
    ) -> MtbResult<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShapes {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvParams {
    /// 输入时间步
    pub in_len: usize,
    /// 输入通道数
    pub in_ch: usize,
    /// 卷积核宽度
    pub kernel: usize,
    /// 卷积核个数（输出通道数）
    pub filters: usize,
    pub stride: usize,
    /// same 时左右各补 (kernel-1)/2；valid 为 0
    pub padding: usize,
    pub dilation: usize,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            in_len: 0,
            in_ch: 0,
            kernel: 1,
            filters: 0,
            stride: 1,
            padding: 0,
            dilation: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElemOp {
    Neg,
    Sigmoid,
    Tanh,
    Relu,
    Gelu,
    Exp,
    Log,
    Sqrt,
    Square,
    Scale(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
}

// ───────────────────────── Layer ─────────────────────────

/// 层统一接口（内置层 / 复合层 / 插件层共用）。见 README 4.1.1。
///
/// 前向时层把自己的权重注册进 `graph` 作为参数节点（可训练叶），
/// 因此 `Var` 只是 arena 下标，反向结束后图随 step 丢弃。
pub trait Layer: Send {
    /// 命名参数，热更新按名迁移权重的锚点。
    fn name(&self) -> &str;

    /// 参数名（按声明顺序），形如 `weight` / `bias`。
    fn param_names(&self) -> Vec<String>;

    /// 外部权重迁移（ALTER / 检查点恢复）落到这里。
    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()>;

    /// 取当前参数快照（落检查点用）。
    fn dump_params(&self) -> Vec<(String, Tensor)>;

    /// 前向。`args` 为本层输入；返回本层输出（支持多入多出 DAG）。
    fn forward(
        &mut self,
        args: &[&Var],
        ctx: &LayerCtx,
        graph: &mut Graph,
    ) -> MtbResult<Vec<Var>>;

    /// 形状推导（装载期静态校验用，不跑前向）。
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>>;

    /// 可序列化的结构描述（拓扑哈希用）。
    fn describe(&self) -> LayerSpec;
}

#[derive(Debug, Clone, Default)]
pub struct LayerCtx {
    /// true=训练（dropout/状态更新），false=推理
    pub training: bool,
    /// RNN 上一时刻隐状态（h_t-1）
    pub seq_state: Option<Vec<f32>>,
}

/// 参数寻址：按 `(层名, 参数名)` 唯一（README 4.1 优化器段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamRef {
    pub layer: String,
    pub name: String,
    pub shape: Shape,
}

impl ParamRef {
    pub fn new(layer: impl Into<String>, name: impl Into<String>, shape: Shape) -> Self {
        Self {
            layer: layer.into(),
            name: name.into(),
            shape,
        }
    }

    /// 参数全名 `layer.name`，与检查点里的权重条目一一对应。
    pub fn full(&self) -> String {
        format!("{}.{}", self.layer, self.name)
    }
}

/// 层的结构描述（拓扑哈希输入）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    pub kind: String,
    pub name: String,
    /// 规范化后的参数 JSON（键排序），供拓扑哈希使用
    pub json: String,
}

impl LayerSpec {
    pub fn new(kind: impl Into<String>, name: impl Into<String>, json: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
            json: json.into(),
        }
    }
}

/// 层种类枚举（配置驱动，禁止在代码中硬编码网络结构）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Dense,
    Embedding,
    Rnn,
    Gru,
    Lstm,
    Conv1d,
    MaxPool1d,
    AvgPool1d,
    GlobalMaxPool1d,
    GlobalAvgPool1d,
    Dropout,
    LayerNorm,
    Flatten,
    Reshape,
    Concat,
    Activation,
    /// 配置内自定义子层 DAG（4.1.1 路径 A）
    Composite,
    /// 未在枚举内的名字 = 插件层（4.1.1 路径 C），由 registry 解析
    Plugin,
}

// ───────────────────────── 优化器 ─────────────────────────

/// 优化器（sgd / momentum / adam）。参数按 `(网络名, 层名, 参数名)` 寻址。
pub trait Optimizer: Send {
    fn id(&self) -> &str;

    /// `params` 每项为 `(参数全名, 参数, 该参数梯度)`。
    fn step(&mut self, params: &mut [(String, &mut Tensor, &Tensor)]);
}

// ───────────────────────── 损失 ─────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossKind {
    CrossEntropy,
    Mse,
}

// ───────────────────────── Math ops ─────────────────────────

/// SQL 层与训练共用的数学函数实现（README 8.10）。单一实现、两处使用。
pub mod math {
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub fn norm2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// 余弦相似度；零向量返回 0.0（不产生 NaN 毒化检索结果）
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (na, nb) = (norm2(a), norm2(b));
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
    }
}

pub use math::*;

/// Box<dyn Error> 桥接，供 server/sql 复用
pub type BoxErr = Box<dyn StdError + Send + Sync>;
