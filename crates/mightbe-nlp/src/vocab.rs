//! 词表：token ↔ id，含 `<pad>`/`<unk>`，min_freq / max_size（README 4.3）。
//!
//! 词表是"增量合并"的：每来一个文档就更新文档频率（df），跨过 `min_freq`
//! 阈值且未超 `max_size` 的 token 即被分配 id（`<pad>`=0, `<unk>`=1 固定保留）。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 填充符：序列补零。
pub const PAD_ID: u32 = 0;
/// 未知符：未登录词。
pub const UNK_ID: u32 = 1;
/// 词表前两个固定槽位之后的用户 token 起始 id。
const FIRST_TOKEN_ID: u32 = 2;

/// `mb_vocab` 系统表的一行（持久化 / 可查询用）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VocabRow {
    /// token 的全局 id。
    pub id: u32,
    /// token 文本。
    pub token: String,
    /// 文档频率（出现过该 token 的文档数）。
    pub df: u32,
}

/// 词表。
#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// 每个 id（>= FIRST_TOKEN_ID）的文档频率。
    doc_freq: Vec<u32>,
    min_freq: u32,
    max_size: u32,
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new(1, 1 << 20)
    }
}

impl Vocabulary {
    /// 构造空词表。`min_freq` 为纳入词表的最小文档频率；`max_size` 为用户 token 上限。
    pub fn new(min_freq: u32, max_size: u32) -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: vec![String::new(), String::new()], // [0]=pad, [1]=unk 占位
            doc_freq: vec![0, 0],
            min_freq,
            max_size,
        }
    }

    /// 返回当前用户 token 数量（不含 pad/unk）。
    pub fn len(&self) -> usize {
        self.id_to_token.len() - 2
    }

    /// 词表是否为空（仅含 pad/unk）。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 容量上限（用户 token）。
    pub fn max_size(&self) -> u32 {
        self.max_size
    }

    /// 最小文档频率阈值。
    pub fn min_freq(&self) -> u32 {
        self.min_freq
    }

    /// 取 token 的 id；未登录返回 [`UNK_ID`]。
    pub fn id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    /// 由 id 取 token；越界返回空串。
    pub fn token(&self, id: u32) -> &str {
        self.id_to_token.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }

    /// 是否未登录词（id == UNK）。
    pub fn is_unk(&self, id: u32) -> bool {
        id == UNK_ID
    }

    /// 全量拟合：根据一批文档（每文档一个去重 token 列表）一次性重建词表。
    ///
    /// 比逐文档 [`observe_document`] 更准：df 按真实文档频率过滤，按 df 降序 + 字典序分配 id。
    pub fn fit(documents: &[Vec<String>], min_freq: u32, max_size: u32) -> Self {
        // 1. 统计 df
        let mut df: HashMap<String, u32> = HashMap::new();
        for doc in documents {
            for tok in doc.iter().collect::<std::collections::HashSet<_>>() {
                *df.entry(tok.clone()).or_insert(0) += 1;
            }
        }
        // 2. 过滤 + 排序（df 降序，字典序升序以保证确定性）
        let mut cand: Vec<(String, u32)> = df
            .into_iter()
            .filter(|(_, c)| *c >= min_freq)
            .collect();
        cand.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if (cand.len() as u32) > max_size {
            cand.truncate(max_size as usize);
        }
        // 3. 分配 id
        let mut v = Self::new(min_freq, max_size);
        for (tok, count) in cand {
            let id = v.id_to_token.len() as u32;
            v.token_to_id.insert(tok.clone(), id);
            v.id_to_token.push(tok);
            v.doc_freq.push(count);
        }
        v
    }

    /// 把一段词条编码为 id 序列（未登录词用 [`UNK_ID`]）。
    pub fn encode(&self, tokens: &[String]) -> Vec<u32> {
        tokens.iter().map(|t| self.id(t)).collect()
    }

    /// 把 id 序列解码为 token（遇到 pad/unk 仍照常输出占位文本）。
    pub fn decode(&self, ids: &[u32]) -> Vec<String> {
        ids.iter().map(|&id| self.token(id).to_string()).collect()
    }

    /// 导出 `mb_vocab` 形状的全部行（含 pad/unk 之外的用户 token）。
    pub fn rows(&self) -> Vec<VocabRow> {
        self.id_to_token
            .iter()
            .enumerate()
            .skip(FIRST_TOKEN_ID as usize)
            .map(|(id, tok)| VocabRow {
                id: id as u32,
                token: tok.clone(),
                df: self.doc_freq[id],
            })
            .collect()
    }

    /// 文档频率（按 id）。未入表返回 0。
    pub fn df(&self, id: u32) -> u32 {
        self.doc_freq.get(id as usize).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Vec<String>> {
        vec![
            vec!["机器".into(), "学习".into(), "人工".into(), "智能".into()],
            vec!["机器".into(), "学习".into(), "深度".into()],
            vec!["深度".into(), "学习".into()],
        ]
    }

    #[test]
    fn fit_assigns_ids_and_filters() {
        let v = Vocabulary::fit(&docs(), 2, 100);
        // "学习" 出现 3 次 → df 3；"机器" 2；"深度" 2；"人工"/"智能" 各 1 被过滤
        assert_eq!(v.id("学习"), FIRST_TOKEN_ID); // df 最高，排第一
        assert!(v.df(v.id("学习")) == 3);
        assert!(v.id("人工") == UNK_ID); // 被 min_freq=2 过滤
        assert!(v.len() == 3); // 学习、机器、深度
    }

    #[test]
    fn encode_decode_roundtrip() {
        let v = Vocabulary::fit(&docs(), 1, 100);
        let ids = v.encode(&["机器".into(), "未知词".into()]);
        assert_eq!(ids[0], v.id("机器"));
        assert_eq!(ids[1], UNK_ID);
        let toks = v.decode(&ids);
        assert_eq!(toks[0], "机器");
    }

    #[test]
    fn max_size_cap() {
        let v = Vocabulary::fit(&docs(), 1, 1);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn rows_shape() {
        let v = Vocabulary::fit(&docs(), 1, 100);
        let rows = v.rows();
        assert!(rows.iter().all(|r| r.id >= FIRST_TOKEN_ID));
        assert!(rows.iter().any(|r| r.token == "学习" && r.df == 3));
    }
}
