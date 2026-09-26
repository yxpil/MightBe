//! 关键词打分：TF-IDF 与 TextRank（README 4.3）。
//!
//! - TF-IDF：IDF 由全库文档频率（对应 `mb_df`）计算；
//! - TextRank：在词共现图上跑加权 PageRank 迭代，得到词的重要度。
//!
//! 两个算法都返回 `(token_id -> score)`，由 index 层翻译成 `mb_keywords` 行。

use std::collections::HashMap;

/// 单文档 TF-IDF 打分。
///
/// `doc_tokens` 为该文档已编码的 token id（应过滤 pad/unk/停用词）；`df` 为全库
/// 文档频率表（id -> 出现文档数）；`n_docs` 为全库文档总数。
///
/// 采用 `tf = 词频 / 文档长度`、`idf = ln(n_docs / df)`，返回 `token_id -> tfidf`。
pub fn tfidf_scores(
    doc_tokens: &[u32],
    df: &HashMap<u32, u64>,
    n_docs: u64,
) -> HashMap<u32, f64> {
    if n_docs == 0 || doc_tokens.is_empty() {
        return HashMap::new();
    }
    let mut tf: HashMap<u32, f64> = HashMap::new();
    for &t in doc_tokens {
        *tf.entry(t).or_insert(0.0) += 1.0;
    }
    let len = doc_tokens.len() as f64;
    let nd = n_docs as f64;
    let mut out = HashMap::with_capacity(tf.len());
    for (t, c) in tf {
        let d = (*df.get(&t).unwrap_or(&1u64)).max(1) as f64;
        let idf = (nd / d).ln();
        let score = (c / len) * idf;
        out.insert(t, score);
    }
    out
}

/// 加权 PageRank（TextRank 的核心迭代）。
///
/// `edges` 为无向边 `(a, b) -> weight`；`nodes` 为参与排序的全部节点 id。
/// 返回 `node_id -> 重要度`（已归一化到概率分布）。
///
/// 采用随机冲浪模型：`PR(i) = (1-d)/N + d * Σ_j (w_ij / out(i)) * PR(j)`，
/// 其中 `out(i) = Σ_k w_ik`。
pub fn textrank(
    edges: &HashMap<(u32, u32), f64>,
    nodes: &[u32],
    damping: f64,
    iters: usize,
) -> HashMap<u32, f64> {
    let n = nodes.len();
    if n == 0 {
        return HashMap::new();
    }
    // 建邻接表（无向）与出边权重和
    let mut idx: HashMap<u32, usize> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        idx.insert(node, i);
    }
    let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut out_w = vec![0.0f64; n];
    for (&(a, b), &w) in edges {
        if let (Some(&ia), Some(&ib)) = (idx.get(&a), idx.get(&b)) {
            neighbors[ia].push((ib, w));
            neighbors[ib].push((ia, w));
            out_w[ia] += w;
            out_w[ib] += w;
        }
    }
    let mut pr = vec![1.0 / n as f64; n];
    let teleport = (1.0 - damping) / n as f64;
    for _ in 0..iters {
        let mut next = vec![teleport; n];
        for i in 0..n {
            if out_w[i] > 0.0 && pr[i] > 0.0 {
                let share = damping * pr[i] / out_w[i];
                for &(j, w) in &neighbors[i] {
                    next[j] += share * w;
                }
            }
        }
        pr = next;
    }
    nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, pr[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfidf_rewards_rare_terms() {
        // 全库 3 篇；"学习" 在全部 3 篇（df=3），"机器" 只在 1 篇（df=1）
        let mut df = HashMap::new();
        df.insert(10u32, 3u64);
        df.insert(11u32, 1u64);
        // 一篇文档里两个词各出现 1 次
        let doc = vec![10u32, 11u32];
        let s = tfidf_scores(&doc, &df, 3);
        // "机器"(11) idf 更高 → 得分应高于 "学习"(10)
        assert!(s[&11] > s[&10]);
    }

    #[test]
    fn textrank_ranks_hub_higher() {
        // 节点 1 与 2、3、4 都相连；节点 5 只与 1 相连 → 1 应是中枢，得分最高
        let mut edges = HashMap::new();
        edges.insert((1u32, 2u32), 1.0);
        edges.insert((1u32, 3u32), 1.0);
        edges.insert((1u32, 4u32), 1.0);
        edges.insert((1u32, 5u32), 1.0);
        let nodes = vec![1u32, 2, 3, 4, 5];
        let pr = textrank(&edges, &nodes, 0.85, 50);
        assert!(pr[&1] > pr[&5]);
        // 概率分布求和约为 1
        let sum: f64 = pr.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
