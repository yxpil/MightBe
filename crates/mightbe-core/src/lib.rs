//! # mightbe-core
//! MightBe 核心引擎（README 4.1）。
//!
//! 模块边界：本 crate 除 `mightbe-plugin`（零依赖的 ABI 布局定义叶子）外不依赖任何
//! 内部 crate；对外仅暴露 [`api`]。
//! 约定：网络结构、超参、词表等数据**不得**出现在本 crate 源码中（README 第 13 章）。
//!
//! 规划结构（按里程碑填充）：
//! - `tensor`      Tensor/Var 定义与算子
//! - `autograd`    反向模式自动微分
//! - `layers`      内置层库（dense/embedding/rnn/gru/lstm/conv1d/pool/norm/...）
//! - `composite`   复合层展开器（配置式自定义层）
//! - `dsl`         Cell DSL（受限门控方程解析）
//! - `plugin`      层插件宿主（自实现动态库加载 + 稳定 C ABI）
//! - `ops::math`   数学函数（与 SQL 层共用）
//! - `backend`     计算后端（naive -> simd -> gpu/npu feature gate）
//! - `optim`       优化器（sgd/momentum/adam）

pub mod api;
pub mod autograd;
pub mod backend;
pub mod composite;
pub mod dsl;
pub mod init;
pub mod layers;
pub mod optim;
pub mod plugin;
pub mod tensor;

pub use api::*;
pub use backend::{select_backend, NaiveBackend};
pub use composite::{topo_sort, Composite, SubSpec};
pub use dsl::{CellSpec, ParamKind};
pub use init::{make as init_tensor, Init};
pub use layers::{
    avgpool, global_avgpool, global_maxpool, maxpool, Activation, Cell, Concat, Conv1d, Dense,
    Dropout, Embedding, Flatten, GlobalPool1d, LayerNorm, Permute, Pool1d, Reshape, Rnn,
};
pub use optim::{build as build_optimizer, Adam, OptState, OptimKind, Sgd, SgdMomentum};
pub use plugin::{LoadedLibrary, PluginInstance, PluginLayer};
pub use tensor::strides_of;
