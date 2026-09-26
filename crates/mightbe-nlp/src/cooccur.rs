//! 共现统计与共现 PMI（README 4.3 / 8.3）。
//!
//! 冷启动阶段用滑动窗口统计 token 共现，得到 PMI（点互信息），作为联想图的
//! `cooc` 边权重来源。训练后可由 Embedding 余弦替代（M5 的事）；本模块只负责
//! 数据驱动的统计侧。

use std::collections::HashMap;

/// 共现统计结果。
#[derive(Debug, Clone, Default)]
pub struct Cooccurrence {
    /// 无序词对 `(min_id, max_id)` → 共现（按窗口计一次）次数。
    pub pairs: HashMap<(u32, u32), u64>,
    /// 每个 token id → 出现在多少个窗口中（边际计数，用于 PMI 边际概率）。
    pub window_freq: HashMap<u32, u64>,
    /// 总窗口数（跨所有文档）。
    pub total_windows: u64,
}

impl Cooccurrence {
    /// 空结果。
    pub fn new() -> Self {
        Self::default()
    }

    /// 在一批文档的 token id 流上做滑动窗口共现统计。
    ///
    /// - `docs`：每个文档已编码为 token id 序列（应已过滤 pad/unk 与停用词）。
    /// - `window`：窗口大小（token 数）。小于 2 时退化为不做共现。
    ///
    /// 每个窗口内，对每个**去重** token 累加 `window_freq`；对每个**去重无序对**累加
    /// `pairs` 一次。这样 `P(a,b)=pairs/N`、`P(a)=window_freq(a)/N` 可直接用于 PMI。
    pub fn count(docs: &[Vec<u32>], window: usize) -> Self {
        let mut co = Cooccurrence::new();
        if window < 2 {
            return co;
        }
        for doc in docs {
            if doc.len() < 2 {
                continue;
            }
            let n = doc.len();
            let last = n.saturating_sub(window);
            for start in 0..=last {
                let end = (start + window).min(n);
                let w: Vec<u32> = doc[start..end].to_vec();
                let uniq: Vec<u32> = w.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
                for &t in &uniq {
                    *co.window_freq.entry(t).or_insert(0) += 1;
                }
                for i in 0..uniq.len() {
                    for j in (i + 1)..uniq.len() {
                        let (a, b) = if uniq[i] < uniq[j] {
                            (uniq[i], uniq[j])
                        } else {
                            (uniq[j], uniq[i])
                        };
                        *co.pairs.entry((a, b)).or_insert(0) += 1;
                    }
                }
                co.total_windows += 1;
            }
        }
        co
    }

    /// 计算所有词对的 PMI（带平滑，避免除零 / 取 log(0)）。
    ///
    /// `PMI(a,b) = log( P(a,b) / (P(a) * P(b)) )`，其中各概率按窗口归一。
    /// 返回 `(left_id, right_id, pmi, cooc_count)` 列表，按 PMI 降序。
    pub fn pmi(&self) -> Vec<((u32, u32), f64, u64)> {
        if self.total_windows == 0 {
            return Vec::new();
        }
        let n = self.total_windows as f64;
        let mut out = Vec::with_capacity(self.pairs.len());
        for (&pair, &c) in &self.pairs {
            let (a, b) = pair;
            let pa = *self.window_freq.get(&a).unwrap_or(&0) as f64 / n;
            let pb = *self.window_freq.get(&b).unwrap_or(&0) as f64 / n;
            let pab = c as f64 / n;
            // 平滑：分母加极小值，避免 0
            let denom = (pa * pb).max(1e-12);
            let pmi = (pab / denom).ln();
            out.push((pair, pmi, c));
        }
        out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_pairs() {
        // 文档 token id 序列：1 2 3 1 2，窗口 3
        let docs = vec![vec![1u32, 2, 3, 1, 2]];
        let co = Cooccurrence::count(&docs, 3);
        // 窗口: [1,2,3],[2,3,1],[3,1,2] = 3 窗
        assert_eq!(co.total_windows, 3);
        assert_eq!(co.window_freq[&1], 3); // 1 出现在全部 3 窗
        assert_eq!(co.window_freq[&2], 3);
        assert_eq!(co.window_freq[&3], 3); // 3 在 start0/1/2 三个窗口都出现
        // 对 (1,2) 出现在全部 3 窗
        assert_eq!(co.pairs[&(1, 2)], 3);
    }

    #[test]
    fn pmi_positive_for_associated() {
        // 词 A,B 总是共现，C 几乎独立 → A,B 的 PMI 应明显高于 A,C
        let docs = vec![
            vec![1u32, 2, 1, 2, 3, 3, 3, 3],
            vec![1u32, 2, 1, 2, 3, 3, 3, 3],
        ];
        let co = Cooccurrence::count(&docs, 2);
        let pmis: HashMap<(u32, u32), f64> = co.pmi().into_iter().map(|(p, v, _)| (p, v)).collect();
        // (1,2) 总是相邻共现 → 高 PMI；(1,3) 弱 → 低 PMI
        assert!(pmis[&(1, 2)] > pmis.get(&(1, 3)).copied().unwrap_or(f64::NEG_INFINITY));
    }
}
