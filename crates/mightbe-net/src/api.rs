//! mightbe-net 对外 API（README 4.2 / 5 / 7）。
//! 其他模块只允许 `use mightbe_net::api::*`。

use mightbe_core::MtbResult;
use std::sync::Arc;

/// 网络运行时抽象。
pub trait NetworkRuntime: Send + Sync {
    /// 读配置（TOML）→ 校验头格式与层拓扑 → 注册网络（版本 1）
    fn load_network(&self, path: &str) -> MtbResult<NetworkId>;

    /// ALTER 提交流程（7.2）：合并 spec → 校验 → 权重迁移 → 冒烟 → 原子切换
    fn alter(&self, id: NetworkId, op: AlterOp) -> MtbResult<Version>;

    /// 取当前不可变快照（推理/检查点/推理图用）
    fn snapshot(&self, id: NetworkId) -> Arc<NetworkSnapshot>;

    /// 单步训练（训练线程内调用；权重与动量按参数名迁移）
    fn train_step(&self, id: NetworkId, batch: &[TrainingExample]) -> MtbResult<f32>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetworkId(u64);

impl NetworkId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(pub u32);

/// 不可变网络快照：推理全程持有，ALTER 不影响在途请求。
pub struct NetworkSnapshot {
    pub version: Version,
    pub topology_json: String,
}

/// ALTER 操作（SQL `ALTER NETWORK`）。
#[derive(Debug, Clone)]
pub enum AlterOp {
    AddLayer { layer: LayerSpecDraft, after: String },
    DropLayer { name: String },
    SetParams { json: String },
    /// 外部配置文件已改好，重新读入
    Reload,
    Restore { version: Version },
}

/// 层定义草稿（来自配置或 SQL 字面量）。
#[derive(Debug, Clone)]
pub struct LayerSpecDraft {
    pub kind: String,
    pub name: String,
    pub json: String,
}

/// 训练样本（来自数据源表 + 头格式编码）。
#[derive(Debug, Clone, Default)]
pub struct TrainingExample {
    pub token_ids: Vec<u32>,
    pub tfidf: Vec<(u32, f32)>,
    pub label: Option<i64>,
    pub dense: Vec<f32>,
}
