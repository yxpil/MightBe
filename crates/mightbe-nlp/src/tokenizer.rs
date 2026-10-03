//! 分词器（从零实现，无外部 NLP 库）—— README 4.3 / 8.1。
//!
//! 设计（与 M2 同约束：不引外部 NLP 框架）：
//! - 拉丁/数字：手写状态机，最大 `[A-Za-z0-9]` 连续段作为一个词条；
//! - 中日韩：单字成词 + 相邻二元（bigram）组合候选，由词频在词表内动态过滤；
//! - 停用词 / 标点：读取 `config/nlp/stopwords.*.txt` 外部文件（见 [`crate::api::NlpConfig`]）。

use std::collections::HashSet;

/// 判断一个字符是否属于 CJK / 假名 / 谚文 等需要"字 + bigram"处理的表意文字范围。
pub fn is_cjk(c: char) -> bool {
    let u = c as u32;
    (0x3400..=0x4DBF).contains(&u) // CJK 扩展 A
        || (0x4E00..=0x9FFF).contains(&u) // CJK 统一表意文字
        || (0xF900..=0xFAFF).contains(&u) // CJK 兼容表意文字
        || (0x3040..=0x30FF).contains(&u) // 平假名 + 片假名
        || (0xAC00..=0xD7AF).contains(&u) // 谚文音节
}

/// 分词器。
#[derive(Debug, Clone)]
pub struct Tokenizer {
    stopwords: HashSet<String>,
    lowercase: bool,
}

impl Tokenizer {
    /// 用显式停用词集合构造（离线测试、内建回退都走这里）。
    pub fn new(stopwords: impl IntoIterator<Item = String>, lowercase: bool) -> Self {
        Self {
            stopwords: stopwords.into_iter().collect(),
            lowercase,
        }
    }

    /// 把停用词集合追加进现有实例（配置目录下多个文件可叠加）。
    pub fn extend_stopwords(&mut self, words: impl IntoIterator<Item = String>) {
        self.stopwords.extend(words);
    }

    /// 判断 token 是否应被过滤（命中停用词）。
    pub fn is_stopword(&self, tok: &str) -> bool {
        self.stopwords.contains(tok)
    }

    /// 对**已规范化**的文本分词。
    ///
    /// 返回词条序列：拉丁词 + CJK 单字 + CJK bigram（已过滤停用词与纯标点）。
    /// 空字符串 / 纯空白返回空向量。
    pub fn tokenize(&self, normalized: &str) -> Vec<String> {
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        let mut out: Vec<String> = Vec::new();
        let mut latin = String::new();

        let flush_latin = |buf: &mut String, out: &mut Vec<String>, lower: bool, sw: &HashSet<String>| {
            if buf.is_empty() {
                return;
            }
            let mut w = std::mem::take(buf);
            if lower {
                w = w.to_ascii_lowercase();
            }
            if !sw.contains(&w) {
                out.push(w);
            }
        };

        let mut i = 0usize;
        while i < n {
            let c = chars[i];
            if c.is_ascii_alphanumeric() {
                latin.push(c);
                i += 1;
                continue;
            }
            // 非拉丁：先把累积的拉丁词 flush
            flush_latin(&mut latin, &mut out, self.lowercase, &self.stopwords);
            if is_cjk(c) {
                let uni = c.to_string();
                if !self.stopwords.contains(&uni) {
                    out.push(uni.clone());
                }
                // bigram：与下一个 CJK 字组合
                if i + 1 < n && is_cjk(chars[i + 1]) {
                    let bi = format!("{}{}", c, chars[i + 1]);
                    if !self.stopwords.contains(&bi) {
                        out.push(bi);
                    }
                }
            }
            // 其它（标点 / 空白 / 其它脚本）直接跳过
            i += 1;
        }
        flush_latin(&mut latin, &mut out, self.lowercase, &self.stopwords);
        out
    }
}

/// 从多个停用词文件内容（每行一个词）解析为集合。
///
/// 规则：`#` 开头为注释；空行忽略；行内首尾空白裁掉；空串丢弃。
pub fn parse_stopwords(contents: &[String]) -> HashSet<String> {
    let mut set = HashSet::new();
    for c in contents {
        for line in c.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            set.insert(line.to_string());
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk() -> Tokenizer {
        Tokenizer::new(
            ["的", "了", "the", "is"].iter().map(|s| s.to_string()),
            true,
        )
    }

    #[test]
    fn cjk_unigram_and_bigram() {
        let t = tk();
        let toks = t.tokenize("机器学习");
        // 单字：机、器、学、习；bigram：机器、器学、学习
        assert!(toks.contains(&"机器".to_string()));
        assert!(toks.contains(&"学习".to_string()));
        assert!(toks.contains(&"机".to_string()));
        assert!(toks.contains(&"习".to_string()));
    }

    #[test]
    fn stopword_filtered() {
        let t = tk();
        let toks = t.tokenize("我的学习");
        assert!(!toks.contains(&"的".to_string()));
        assert!(toks.contains(&"学习".to_string()));
    }

    #[test]
    fn latin_state_machine() {
        let t = tk();
        let toks = t.tokenize("RNN is great");
        assert!(toks.contains(&"rnn".to_string()));
        assert!(!toks.contains(&"is".to_string())); // 停用词
        assert!(toks.contains(&"great".to_string()));
    }

    #[test]
    fn mixed_script() {
        let t = tk();
        let toks = t.tokenize("AI技术革命");
        assert!(toks.contains(&"ai".to_string()));
        assert!(toks.contains(&"技术".to_string()));
        assert!(toks.contains(&"革命".to_string()));
    }

    #[test]
    fn empty_input() {
        let t = tk();
        assert!(t.tokenize("   ").is_empty());
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn parse_stopwords_file() {
        let set = parse_stopwords(&["# 注释\n的\n了\n".to_string(), "the\nis\n".to_string()]);
        assert!(set.contains("的"));
        assert!(set.contains("the"));
        assert_eq!(set.len(), 4);
    }

    // ── 注入安全：正文是不可信输入。分词器只认 [A-Za-z0-9] 与 CJK，
    //    标签尖括号、路径分隔符都只是标点/分隔符，绝不被当成结构解释。 ──

    #[test]
    fn html_tags_are_punctuation_not_tokens() {
        let t = tk();
        let toks = t.tokenize("<script>alert('xss')</script>");
        // 标签符号本身不进入词表；脚本内容被降级为普通拉丁词
        assert!(toks.iter().all(|w| !w.contains('<') && !w.contains('>')), "{toks:?}");
        assert!(toks.contains(&"alert".to_string()));
        assert!(toks.contains(&"xss".to_string()));
    }

    #[test]
    fn path_traversal_is_lexical_only_no_semantics() {
        let t = tk();
        // 正文里出现路径穿越串：点号/反斜杠当分隔符，绝不解释成文件路径
        let toks = t.tokenize("../../etc/passwd ..\\..\\windows\\system32");
        assert!(toks.contains(&"etc".to_string()));
        assert!(toks.contains(&"passwd".to_string()));
        assert!(toks.contains(&"system32".to_string()));
        assert!(
            toks.iter().all(|w| !w.starts_with('.') && !w.contains('/') && !w.contains('\\')),
            "没有任何 token 携带路径分隔符: {toks:?}"
        );
    }
}
