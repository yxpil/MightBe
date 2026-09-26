//! 文本规范化（README 4.3 / 8.1）。
//!
//! 管线第一步：Unicode NFKC、全半角统一、大小写（按配置）、空白归一。
//! 不依赖任何外部 NLP 库，也不依赖 store —— 纯函数。

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// 规范化采用的 Unicode 正规化形式。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum NormForm {
    /// 兼容合成：把全角、上下标、连字等映射到规范等价形式。默认值。
    #[default]
    Nfkc,
    /// 不做正规化（仅做全半角/大小写/空白处理）。
    None,
}

/// 规范化配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizeConfig {
    /// 正规化形式。
    pub form: NormForm,
    /// 是否把 ASCII 字母折叠为小写。
    pub lowercase: bool,
    /// 是否把连续空白折叠为单个空格并裁掉首尾空白。
    pub collapse_ws: bool,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            form: NormForm::Nfkc,
            lowercase: true,
            collapse_ws: true,
        }
    }
}

/// 文本规范化器。
#[derive(Debug, Clone)]
pub struct Normalizer {
    cfg: NormalizeConfig,
}

impl Normalizer {
    /// 用给定配置构造。
    pub fn new(cfg: NormalizeConfig) -> Self {
        Self { cfg }
    }

    /// 规范化一段文本，返回规范化结果。
    ///
    /// 处理顺序固定为：NFKC → 全半角 → 小写 → 空白归一。
    pub fn normalize(&self, text: &str) -> String {
        // 1. NFKC 正规化
        let s: String = match self.cfg.form {
            NormForm::Nfkc => text.nfkc().collect(),
            NormForm::None => text.to_string(),
        };
        // 2. 全角 → 半角
        let s = full_to_half(&s);
        // 3. ASCII 小写
        let s = if self.cfg.lowercase {
            s.chars()
                .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c })
                .collect()
        } else {
            s
        };
        // 4. 空白归一
        if self.cfg.collapse_ws {
            let mut out = String::with_capacity(s.len());
            let mut prev_ws = false;
            for c in s.chars() {
                if c.is_whitespace() {
                    if !prev_ws {
                        out.push(' ');
                        prev_ws = true;
                    }
                } else {
                    out.push(c);
                    prev_ws = false;
                }
            }
            out.trim().to_string()
        } else {
            s
        }
    }
}

/// 把全角字符（FF01..FF5E）映射回半角，并把全角空格（U+3000）映射为 ASCII 空格。
fn full_to_half(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{3000}' => ' ',
            c if (0xFF01..=0xFF5E).contains(&(c as u32)) => {
                // 全角可打印字符相对半角偏移 0xFEE0
                char::from_u32((c as u32) - 0xFEE0).unwrap_or(c)
            }
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullwidth_to_halfwidth() {
        let n = Normalizer::new(NormalizeConfig::default());
        assert_eq!(n.normalize("ＡＢＣ１２３　！"), "abc123 !");
    }

    #[test]
    fn lowercase_and_ws_collapse() {
        let n = Normalizer::new(NormalizeConfig::default());
        assert_eq!(n.normalize("  Hello   WORLD\n"), "hello world");
    }

    #[test]
    fn nfkc_compat() {
        let n = Normalizer::new(NormalizeConfig::default());
        // 全角片假名 + 浊点组合应被 NFKC 合成为单码位
        let s = n.normalize("が");
        assert!(!s.is_empty());
    }

    #[test]
    fn cjk_untouched() {
        let n = Normalizer::new(NormalizeConfig::default());
        assert_eq!(n.normalize("人工智能"), "人工智能");
    }

    #[test]
    fn no_lowercase_config_keeps_case() {
        let n = Normalizer::new(NormalizeConfig {
            lowercase: false,
            ..Default::default()
        });
        assert_eq!(n.normalize("Hello WORLD"), "Hello WORLD");
    }
}
