//! 语料索引：把 NLP 各子模块编排成一个可查询的"系统表"集合（README 4.3 / 8）。
//!
//! 对应 INSERT 触发链（8.1）：
//! ```text
//! normalize → tokenize → 词表增量合并 → 写 mb_vocab / 更新 mb_df
//!   → TF-IDF + 可选 TextRank → 写 mb_keywords
//!   → 共现滑窗统计 → UPSERT mb_assoc
//! ```
//!
//! 本模块产出与 `mb_*` 系统表同形状的可查询记录（内存态；落盘到 store 由 M5 完成）：
//! - `mb_vocab`   词表行（id / token / df）
//! - `mb_df`      文档频率（id / df）
//! - `mb_keywords` 关键词打分（doc_id / word / score / algo）
//! - `mb_assoc`   共现联想边（word_a / word_b / pmi / cooc）

use crate::api::{AssocEdge, DocFeatures, Keyword, NlpConfig};
use crate::cooccur::Cooccurrence;
use crate::keyword::{textrank, tfidf_scores};
use crate::normalize::Normalizer;
use crate::tokenizer::Tokenizer;
use crate::vocab::{VocabRow, Vocabulary};
use std::collections::HashMap;

/// 一个文档在索引中的归一化表示。
struct DocTokens {
    id: u64,
    raw: String,
}

/// 语料索引。
pub struct NlpIndex {
    normalizer: Normalizer,
    tokenizer: Tokenizer,
    config: NlpConfig,

    /// 累积的文档（doc_id → 已分词文本）。
    stored: Vec<DocTokens>,
    /// 是否已重新构建派生数据。
    dirty: bool,

    // ── 派生数据（rebuild 后填充）──
    vocab: Vocabulary,
    /// doc_id → 编码后的 token id 序列。
    doc_tokens: HashMap<u64, Vec<u32>>,
    /// token id → 文档频率。
    df: HashMap<u32, u64>,
    /// 共现统计。
    cooc: Cooccurrence,
    /// TextRank 全局重要度（id → score）。
    textrank: HashMap<u32, f64>,
    /// `mb_keywords` 行。
    keywords: Vec<Keyword>,
    /// `mb_assoc` 行。
    assoc: Vec<AssocEdge>,
}

impl NlpIndex {
    /// 用配置构造空索引。
    pub fn new(config: NlpConfig) -> Self {
        let normalizer = Normalizer::new(config.normalize.clone());
        let tokenizer = Tokenizer::new(config.stopwords.clone(), config.normalize.lowercase);
        Self {
            normalizer,
            tokenizer,
            config,
            stored: Vec::new(),
            dirty: true,
            vocab: Vocabulary::default(),
            doc_tokens: HashMap::new(),
            df: HashMap::new(),
            cooc: Cooccurrence::new(),
            textrank: HashMap::new(),
            keywords: Vec::new(),
            assoc: Vec::new(),
        }
    }

    /// 追加一个文档（对应 INSERT 后台任务的入队）。置脏，等待 [`rebuild`]。
    pub fn add_document(&mut self, doc_id: u64, raw_text: &str) -> &mut Self {
        let norm = self.normalizer.normalize(raw_text);
        let toks = self.tokenizer.tokenize(&norm);
        // 用空格把已分词条还原为一条可重分的规范文本，便于后续重建时复算
        self.stored.push(DocTokens {
            id: doc_id,
            raw: toks.join(" "),
        });
        self.dirty = true;
        self
    }

    /// 重新构建派生数据（词表 / df / 关键词 / 共现）。脏时才真正重算。
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }
        // 1. 每文档去重 token 集合 → 拟合词表
        let per_doc_unique: Vec<Vec<String>> = self
            .stored
            .iter()
            .map(|d| {
                d.raw
                    .split(' ')
                    .filter(|s| !s.is_empty())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect()
            })
            .collect();
        let vocab = Vocabulary::fit(
            &per_doc_unique,
            self.config.vocab_min_freq,
            self.config.vocab_max_size,
        );

        // 2. 编码所有文档，统计 df
        let mut doc_tokens: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut df: HashMap<u32, u64> = HashMap::new();
        let mut streams: Vec<Vec<u32>> = Vec::with_capacity(self.stored.len());
        for d in &self.stored {
            let toks: Vec<String> = d.raw.split(' ').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            let ids = vocab.encode(&toks);
            // df：每文档内每个用户 token 只计一次
            let mut seen = std::collections::HashSet::new();
            for &id in &ids {
                if id < 2 {
                    continue;
                }
                doc_tokens.entry(d.id).or_default().push(id);
                if seen.insert(id) {
                    *df.entry(id).or_insert(0) += 1;
                }
            }
            if let Some(stream) = doc_tokens.get(&d.id) {
                streams.push(stream.clone());
            }
        }
        let n_docs = self.stored.len() as u64;

        // 3. 共现统计
        let cooc = Cooccurrence::count(&streams, self.config.cooc_window);

        // 4. TextRank：在词表节点上跑加权 PageRank
        let nodes: Vec<u32> = vocab.rows().iter().map(|r| r.id).collect();
        let edges: HashMap<(u32, u32), f64> = cooc
            .pairs
            .iter()
            .map(|(&(a, b), &c)| ((a, b), c as f64))
            .collect();
        let tr = textrank(&edges, &nodes, self.config.textrank_damping, self.config.textrank_iters);

        // 5. 关键词：tfidf + textrank
        let mut keywords: Vec<Keyword> = Vec::new();
        for d in &self.stored {
            let ids = doc_tokens.get(&d.id).cloned().unwrap_or_default();
            if ids.is_empty() {
                continue;
            }
            // tfidf
            let tfidf = tfidf_scores(&ids, &df, n_docs);
            // textrank：取本文档 token 的全局重要度
            let mut tr_doc: Vec<(u32, f64)> = ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .filter_map(|id| tr.get(&id).map(|&s| (id, s)))
                .collect();
            tr_doc.sort_by(|a, b| { let o = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal); if o == std::cmp::Ordering::Equal { a.0.cmp(&b.0) } else { o } });

            let topk = self.config.keyword_top_k;
            let mut tfidf_sorted: Vec<(u32, f64)> = tfidf.into_iter().collect();
            tfidf_sorted.sort_by(|a, b| { let o = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal); if o == std::cmp::Ordering::Equal { a.0.cmp(&b.0) } else { o } });
            for (id, sc) in tfidf_sorted.into_iter().take(topk) {
                if sc <= 0.0 {
                    continue;
                }
                keywords.push(Keyword {
                    doc_id: d.id,
                    word: vocab.token(id).to_string(),
                    score: sc as f32,
                    algo: "tfidf".to_string(),
                });
            }
            for (id, sc) in tr_doc.iter().take(topk) {
                keywords.push(Keyword {
                    doc_id: d.id,
                    word: vocab.token(*id).to_string(),
                    score: *sc as f32,
                    algo: "textrank".to_string(),
                });
            }
        }

        // 6. 共现边 → mb_assoc
        let pmi_list = cooc.pmi();
        let mut assoc: Vec<AssocEdge> = pmi_list
            .into_iter()
            .map(|((a, b), pmi, c)| AssocEdge {
                left: vocab.token(a).to_string(),
                right: vocab.token(b).to_string(),
                pmi: pmi as f32,
                cooc_count: c,
            })
            .collect();
        assoc.sort_by(|x, y| { let o = y.pmi.partial_cmp(&x.pmi).unwrap_or(std::cmp::Ordering::Equal); if o == std::cmp::Ordering::Equal { (x.left.as_str(), x.right.as_str()).cmp(&(y.left.as_str(), y.right.as_str())) } else { o } });

        self.vocab = vocab;
        self.doc_tokens = doc_tokens;
        self.df = df;
        self.cooc = cooc;
        self.textrank = tr;
        self.keywords = keywords;
        self.assoc = assoc;
        self.dirty = false;
    }

    /// 确保派生数据已构建。
    fn ensure_built(&mut self) {
        if self.dirty {
            self.rebuild();
        }
    }

    /// 文档总数。
    pub fn doc_count(&self) -> u64 {
        self.stored.len() as u64
    }

    /// 把一段原始文本编码为 token id 序列（规范化 + 分词 + 词表映射）。
    pub fn encode_text(&self, raw: &str) -> Vec<u32> {
        let norm = self.normalizer.normalize(raw);
        let toks = self.tokenizer.tokenize(&norm);
        self.vocab.encode(&toks)
    }

    /// 一个文档的特征编码（喂给网络输入头）。要求先 [`rebuild`]。
    pub fn features(&mut self, doc_id: u64) -> DocFeatures {
        self.ensure_built();
        let ids = self.doc_tokens.get(&doc_id).cloned().unwrap_or_default();
        let tfidf = tfidf_scores(&ids, &self.df, self.doc_count())
            .into_iter()
            .filter(|(_, v)| *v > 0.0)
            .map(|(id, v)| (id, v as f32))
            .collect();
        DocFeatures {
            token_ids: ids,
            tfidf,
        }
    }

    /// 查询某文档的关键词（按算法过滤，取前 `top_k`，按分降序）。
    pub fn keywords(&mut self, doc_id: u64, top_k: usize, algo: Option<&str>) -> Vec<Keyword> {
        self.ensure_built();
        let mut out: Vec<Keyword> = self
            .keywords
            .iter()
            .filter(|k| k.doc_id == doc_id)
            .filter(|k| algo.is_none() || k.algo == algo.unwrap())
            .cloned()
            .collect();
        out.sort_by(|a, b| { let o = b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal); if o == std::cmp::Ordering::Equal { a.word.as_str().cmp(b.word.as_str()) } else { o } });
        out.truncate(top_k);
        out
    }

    /// 联想查询：返回与 `word` 共现的相邻词及 PMI（降序，取前 `top_k`）。
    pub fn associate(&mut self, word: &str, top_k: usize) -> Vec<AssocEdge> {
        self.ensure_built();
        let mut out: Vec<AssocEdge> = self
            .assoc
            .iter()
            .filter(|e| e.left == word || e.right == word)
            .map(|e| {
                if e.left == word {
                    e.clone()
                } else {
                    AssocEdge {
                        left: e.right.clone(),
                        right: e.left.clone(),
                        pmi: e.pmi,
                        cooc_count: e.cooc_count,
                    }
                }
            })
            .collect();
        out.sort_by(|a, b| b.pmi.partial_cmp(&a.pmi).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top_k);
        out
    }

    /// `mb_vocab` 全部行。
    pub fn vocab_rows(&mut self) -> Vec<VocabRow> {
        self.ensure_built();
        self.vocab.rows()
    }

    /// `mb_df` 全部行（id, df），按 id 升序（确定性输出）。
    pub fn df_rows(&mut self) -> Vec<(u32, u64)> {
        self.ensure_built();
        let mut v: Vec<(u32, u64)> = self.df.iter().map(|(&id, &c)| (id, c)).collect();
        v.sort_by_key(|&(id, _)| id);
        v
    }

    /// `mb_keywords` 全部行。
    pub fn keyword_rows(&mut self) -> Vec<Keyword> {
        self.ensure_built();
        self.keywords.clone()
    }

    /// `mb_assoc` 全部行（已按 PMI 降序）。
    pub fn assoc_rows(&mut self) -> Vec<AssocEdge> {
        self.ensure_built();
        self.assoc.clone()
    }

    /// 把 `mb_*` 记录序列化为 JSON（便于落盘 / 调试；落 store 由 M5 负责）。
    pub fn dump_json(&mut self) -> String {
        self.ensure_built();
        use serde_json::json;
        let mut df_sorted: Vec<(u32, u64)> = self.df.iter().map(|(&id, &c)| (id, c)).collect();
        df_sorted.sort_by_key(|&(id, _)| id);
        let value = json!({
            "vocab": self.vocab.rows(),
            "df": df_sorted,
            "keywords": self.keywords,
            "assoc": self.assoc,
        });
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
    }
}
