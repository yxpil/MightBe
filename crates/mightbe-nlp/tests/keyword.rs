//! M3 NLP 集成测试（README 4.3 / 8.1 / 里程碑表）。
//!
//! 验收口径：
//! 1. 中文样例关键词人工抽检合理（主题词应进入关键词表，停用词不应）；
//! 2. `mb_*` 系统表可查（vocab / df / keywords / assoc 都能取出记录）。

use mightbe_nlp::api::*;
use mightbe_nlp::index::NlpIndex;
use std::path::Path;

/// 工作区根 `config/nlp` 目录（相对于 crate 目录 `crates/mightbe-nlp`）。
fn config_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/nlp")
}

/// 三段中文语料，主题分别是 人工智能 / 机器学习 / 深度学习。
fn corpus() -> Vec<(u64, &'static str)> {
    vec![
        (
            1,
            "人工智能正在改变世界，机器学习是人工智能的核心技术。",
        ),
        (
            2,
            "机器学习依赖大量数据，深度学习是机器学习的分支。",
        ),
        (
            3,
            "深度学习推动了计算机视觉和自然语言处理的发展。",
        ),
    ]
}

#[test]
fn vocab_excludes_stopwords_and_keeps_topical() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    idx.rebuild();
    let rows = idx.vocab_rows();
    let tokens: Vec<&str> = rows.iter().map(|r| r.token.as_str()).collect();
    // 停用词不应入表
    assert!(!tokens.contains(&"的"), "停用词 '的' 不应进入词表: {:?}", tokens);
    assert!(!tokens.contains(&"是"));
    // 主题词应入表
    assert!(tokens.contains(&"学习"), "主题词 '学习' 应进入词表");
    assert!(tokens.contains(&"机器"));
    assert!(tokens.contains(&"深度"));
}

#[test]
fn df_counts_documents_not_occurrences() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    idx.rebuild();
    let df = idx.df_rows();
    let df_map: std::collections::HashMap<u32, u64> = df.into_iter().collect();
    let vocab = idx.vocab_rows();
    let id_of = |w: &str| vocab.iter().find(|r| r.token == w).map(|r| r.id).unwrap();
    // "学习" 出现在全部 3 篇 → df = 3
    let learn = id_of("学习");
    assert_eq!(df_map[&learn], 3);
    // "人工" 仅第 1 篇 → df = 1
    let ai = id_of("人工");
    assert_eq!(df_map[&ai], 1);
}

#[test]
fn keywords_are_topical_and_queryable() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    // mb_keywords 可查：doc1 的 tfidf 关键词应包含其独有主题词之一
    let kws = idx.keywords(1, 10, Some("tfidf"));
    assert!(!kws.is_empty(), "doc1 应有关键词");
    let words: Vec<&str> = kws.iter().map(|k| k.word.as_str()).collect();
    let has_topical = ["人工", "智能", "技术", "核心", "改变", "世界"]
        .iter()
        .any(|w| words.contains(w));
    assert!(
        has_topical,
        "doc1 tfidf 关键词应包含主题词，实际: {:?}",
        words
    );
    // 关键词不应是停用词
    assert!(!words.contains(&"的"));
    // 关键词带算法标签
    assert!(kws.iter().all(|k| k.algo == "tfidf"));
    // 两种算法都产出
    let tr = idx.keywords(1, 10, Some("textrank"));
    assert!(tr.iter().all(|k| k.algo == "textrank"));
}

#[test]
fn associate_returns_cooccurrence_neighbors() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    let neighbors = idx.associate("学习", 5);
    assert!(
        !neighbors.is_empty(),
        "与 '学习' 共现的邻接词不应为空（mb_assoc 可查）"
    );
    // 至少一条边共现次数 > 0
    assert!(neighbors.iter().any(|e| e.cooc_count > 0));
    // pmi 应为有限值
    assert!(neighbors.iter().all(|e| e.pmi.is_finite()));
}

#[test]
fn min_freq_filters_rare_tokens() {
    let mut cfg = NlpConfig::default();
    cfg.vocab_min_freq = 2;
    let mut idx = NlpIndex::new(cfg);
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    idx.rebuild();
    let tokens: Vec<String> = idx.vocab_rows().iter().map(|r| r.token.clone()).collect();
    // "人工" 仅 1 篇，被 min_freq=2 过滤
    assert!(!tokens.contains(&"人工".to_string()));
    // "学习" 3 篇，保留
    assert!(tokens.contains(&"学习".to_string()));
}

#[test]
fn stopwords_loaded_from_config_dir() {
    let cfg = NlpConfig::default();
    let pipe = DefaultNlp::open(&config_dir(), cfg).expect("配置目录应可加载");
    // 配置目录的停用词文件应覆盖内置集合：'的' 必须被过滤
    let toks = pipe.tokenize("我的学习成果").unwrap();
    assert!(!toks.contains(&"的".to_string()));
    assert!(toks.contains(&"学习".to_string()));
}

#[test]
fn pipeline_trait_enqueue_then_query() {
    let pipe = default_pipeline();
    for (id, text) in corpus() {
        pipe.enqueue_document(id, text).unwrap();
    }
    let mut idx = pipe.index();
    let kws = idx.keywords(2, 5, Some("tfidf"));
    assert!(!kws.is_empty());
    // mb_assoc 可查
    let _ = idx.assoc_rows();
    let _ = idx.vocab_rows();
}

#[test]
fn dump_json_is_valid() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    let json = idx.dump_json();
    let v: serde_json::Value = serde_json::from_str(&json).expect("应为合法 JSON");
    assert!(v.get("vocab").is_some());
    assert!(v.get("keywords").is_some());
    assert!(v.get("assoc").is_some());
}


// ─────────────────────────── 回归：确定性与退化输入 ───────────────────────────
// 历史 bug：tfidf 经 HashMap 迭代后顺序不确定，导致存储的关键词随运行抖动；
// 空语料 / 全停用词语料曾可能 panic。

#[test]
fn determinism_two_runs_identical() {
    let build = || {
        let mut idx = NlpIndex::new(NlpConfig::default());
        for (id, text) in corpus() {
            idx.add_document(id, text);
        }
        idx.rebuild();
        idx.dump_json()
    };
    let a = build();
    let b = build();
    assert_eq!(a, b, "两次独立构建的索引必须逐字节一致（tfidf 排序确定性）");

    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in corpus() {
        idx.add_document(id, text);
    }
    idx.rebuild();
    let k1 = idx.keywords(1, 10, Some("tfidf"));
    let k2 = idx.keywords(1, 10, Some("tfidf"));
    assert_eq!(k1, k2, "同一文档的关键词必须确定性产出");
}

#[test]
fn empty_corpus_yields_empty_index() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    idx.rebuild(); // 没有任何文档
    assert!(idx.vocab_rows().is_empty(), "空语料词表应空");
    assert!(idx.df_rows().is_empty());
    assert!(idx.keywords(1, 10, None).is_empty());
    assert!(idx.associate("学习", 5).is_empty());
    let json = idx.dump_json();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["vocab"].as_array().map(|a| a.len()).unwrap_or(1), 0);
}

#[test]
fn all_stopwords_doc_does_not_panic() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    idx.add_document(1, "的是在和为了与及或但");
    idx.rebuild();
    let kws = idx.keywords(1, 10, None);
    assert!(
        kws.is_empty() || kws.iter().all(|k| !["的", "是", "在", "和"].contains(&k.word.as_str())),
        "全停用词文档不应产出停用词关键词"
    );
}

#[test]
fn large_corpus_rebuild_is_complete_and_deterministic() {
    let docs: Vec<(u64, String)> = (0..200u64)
        .map(|i| {
            let mut s = String::from("神经网络 是 一种 深度学习 模型 ");
            if i % 2 == 1 {
                s.push_str("猫 喜欢 睡觉 ");
            }
            (i + 1, s)
        })
        .collect();

    let mut idx = NlpIndex::new(NlpConfig::default());
    for (id, text) in &docs {
        idx.add_document(*id, text);
    }
    idx.rebuild();
    let vocab = idx.vocab_rows();
    let id_of = |w: &str| vocab.iter().find(|r| r.token == w).map(|r| r.id);
    // 分词是"字 + bigram"，'神经网络' 跨 4 字不会成为单 token；
    // 而 '深度学习' 中的 bigram '学习' 出现在全部 200 篇，用它验证 df。
    let learn = id_of("学习").expect("学习 应入表（深度学习 的 bigram）");
    let df = idx.df_rows();
    let df_map: std::collections::HashMap<u32, u64> = df.into_iter().collect();
    assert_eq!(df_map[&learn], 200, "'学习' 应出现在全部 200 篇");

    let mut idx2 = NlpIndex::new(NlpConfig::default());
    for (id, text) in &docs {
        idx2.add_document(*id, text);
    }
    idx2.rebuild();
    assert_eq!(idx.dump_json(), idx2.dump_json(), "大规模语料也必须确定性");
}
