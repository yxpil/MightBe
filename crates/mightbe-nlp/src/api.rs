//! mightbe-nlp 对外 API（README 4.3 / 8.1）。
//! 其他模块只允许 `use mightbe_nlp::api::*`。

use crate::index::NlpIndex;
use crate::tokenizer::{parse_stopwords, Tokenizer};
use mightbe_core::MtbResult;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

/// 默认共现窗口。
pub const DEFAULT_COOC_WINDOW: usize = 5;
/// 默认 TextRank 阻尼。
pub const DEFAULT_DAMPING: f64 = 0.85;
/// 默认 PageRank 迭代次数。
pub const DEFAULT_ITERS: usize = 40;
/// 默认每文档关键词数。
pub const DEFAULT_TOP_K: usize = 10;

/// NLP 管线全部配置（可 TOML/JSON 序列化；README 第 13 章约定数据不写在源码里）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlpConfig {
    /// 规范化配置。
    pub normalize: NormalizeConfig,
    /// 停用词（显式列表；也常由 `config/nlp/stopwords.*.txt` 载入）。
    pub stopwords: Vec<String>,
    /// 词表最小文档频率。
    pub vocab_min_freq: u32,
    /// 词表用户 token 上限。
    pub vocab_max_size: u32,
    /// 共现滑窗大小（token 数）。
    pub cooc_window: usize,
    /// TextRank PageRank 阻尼系数。
    pub textrank_damping: f64,
    /// TextRank 迭代次数。
    pub textrank_iters: usize,
    /// 每文档关键词返回上限。
    pub keyword_top_k: usize,
}

impl Default for NlpConfig {
    fn default() -> Self {
        Self {
            normalize: NormalizeConfig::default(),
            stopwords: builtin_stopwords(),
            vocab_min_freq: 1,
            vocab_max_size: 20_000,
            cooc_window: DEFAULT_COOC_WINDOW,
            textrank_damping: DEFAULT_DAMPING,
            textrank_iters: DEFAULT_ITERS,
            keyword_top_k: DEFAULT_TOP_K,
        }
    }
}

/// 内置最小停用词集合（外部文件缺失时的回退）。
pub fn builtin_stopwords() -> Vec<String> {
    [
        "的", "了", "和", "是", "在", "我", "有", "也", "就", "不", "都", "这", "那", "与", "及",
        "等", "它", "我们", "你们", "他们", "因为", "所以", "但是", "如果", "之", "其", "被", "把",
        "对", "于", "以", "从", "到", "过", "更", "最", "很", "吗", "呢", "吧", "the", "is", "a",
        "an", "of", "to", "and", "or", "in", "on", "for", "with", "this", "that",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// 从配置目录载入停用词（合并 `stopwords.zh.txt` 与 `stopwords.en.txt`）。
///
/// 目录不存在或文件缺失不报错，退化为内置集合。
pub fn load_stopwords_from_dir(dir: &Path) -> Vec<String> {
    let mut contents: Vec<String> = Vec::new();
    for name in ["stopwords.zh.txt", "stopwords.en.txt"] {
        let p = dir.join(name);
        if let Ok(text) = std::fs::read_to_string(&p) {
            contents.push(text);
        }
    }
    if contents.is_empty() {
        return builtin_stopwords();
    }
    let set = parse_stopwords(&contents);
    set.into_iter().collect()
}

/// NLP 管线抽象。M0 仅有类型面；M3 给出完整实现。
pub trait NlpPipeline: Send + Sync {
    /// 规范化（NFKC、大小写、空白）。
    fn normalize(&self, text: &str) -> MtbResult<String>;

    /// 分词（拉丁状态机 + CJK 字/bigram）。
    fn tokenize(&self, text: &str) -> MtbResult<Vec<String>>;

    /// 入库后由后台任务异步执行：词表增量、DF/关键词/共现边更新。
    fn enqueue_document(&self, doc_id: u64, text: &str) -> MtbResult<()>;
}

/// 默认 NLP 管线实现（包装 [`NlpIndex`]，用互斥锁提供 `&self` 可变语义）。
pub struct DefaultNlp {
    inner: Mutex<NlpIndex>,
    tokenizer: Tokenizer,
    normalize: NormalizeConfig,
}

impl DefaultNlp {
    /// 用配置构造。
    pub fn new(config: NlpConfig) -> Self {
        let tk = Tokenizer::new(config.stopwords.clone(), config.normalize.lowercase);
        let norm = config.normalize.clone();
        Self {
            inner: Mutex::new(NlpIndex::new(config)),
            tokenizer: tk,
            normalize: norm,
        }
    }

    /// 从配置目录（含 `stopwords.*.txt`）构造。
    pub fn open(config_dir: &Path, mut config: NlpConfig) -> MtbResult<Self> {
        config.stopwords = load_stopwords_from_dir(config_dir);
        Ok(Self::new(config))
    }

    /// 取内部索引（用于查询 `mb_*` 表）。
    pub fn index(&self) -> std::sync::MutexGuard<'_, NlpIndex> {
        self.inner.lock().unwrap()
    }
}

impl NlpPipeline for DefaultNlp {
    fn normalize(&self, text: &str) -> MtbResult<String> {
        let n = crate::normalize::Normalizer::new(self.normalize.clone());
        Ok(n.normalize(text))
    }

    fn tokenize(&self, text: &str) -> MtbResult<Vec<String>> {
        let norm = crate::normalize::Normalizer::new(self.normalize.clone()).normalize(text);
        Ok(self.tokenizer.tokenize(&norm))
    }

    fn enqueue_document(&self, doc_id: u64, text: &str) -> MtbResult<()> {
        self.inner.lock().unwrap().add_document(doc_id, text);
        Ok(())
    }
}

/// 文档特征编码结果（喂给网络输入头）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocFeatures {
    /// token_ids 头：定长/变长 token id 序列。
    pub token_ids: Vec<u32>,
    /// tfidf 头：稀疏向量 (index, value)。
    pub tfidf: Vec<(u32, f32)>,
}

/// 关键词打分结果（写 `mb_keywords`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keyword {
    pub doc_id: u64,
    pub word: String,
    /// tfidf | textrank 两种算法的得分。
    pub score: f32,
    pub algo: String,
}

/// 共现联想边（写 `mb_assoc`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssocEdge {
    pub left: String,
    pub right: String,
    /// 共现 PMI（未归一化）。
    pub pmi: f32,
    pub cooc_count: u64,
}

// 便于上层按名字引用子模块类型
pub use crate::normalize::{NormForm, NormalizeConfig};
pub use crate::vocab::{PAD_ID, UNK_ID, VocabRow};

/// 便捷构造：默认配置 + 内置停用词。
pub fn default_pipeline() -> DefaultNlp {
    DefaultNlp::new(NlpConfig::default())
}
