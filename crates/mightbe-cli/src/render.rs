//! 结果表格化渲染：等宽对齐，CJK 全角字符按 2 列宽计算。

use crate::wire::Reply;
use std::fmt::Write as _;

/// 终端显示宽度。东亚全角/假名/常用汉字区间按 2 计，其余按 1 计。
pub fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn char_width(c: char) -> usize {
    let u = c as u32;
    let wide = matches!(u,
        0x1100..=0x115F       // 韩文字母
        | 0x2E80..=0x303E     // CJK 部首、标点
        | 0x3041..=0x33FF     // 假名、CJK 兼容
        | 0x3400..=0x4DBF     // CJK 扩展 A
        | 0x4E00..=0x9FFF     // CJK 基本汉字
        | 0xA000..=0xA4CF     // 彝文
        | 0xAC00..=0xD7A3     // 韩文音节
        | 0xF900..=0xFAFF     // CJK 兼容表意
        | 0xFE30..=0xFE6F     // CJK 兼容标点
        | 0xFF00..=0xFF60     // 全角形字符
        | 0xFFE0..=0xFFE6);
    if wide || (0x20000..=0x3FFFD).contains(&u) {
        2
    } else {
        1
    }
}

fn pad(cell: &str, width: usize) -> String {
    let mut s = String::with_capacity(cell.len() + width);
    s.push_str(cell);
    for _ in display_width(cell)..width {
        s.push(' ');
    }
    s
}

/// 渲染 [`Reply`] 为可直接打印的文本（含结尾换行）。
pub fn render(reply: &Reply) -> String {
    let mut out = String::new();
    match reply {
        Reply::Error { code, message } => {
            let _ = writeln!(out, "错误 {code}: {message}");
        }
        Reply::Ack { info, ms } => {
            for line in info {
                let _ = writeln!(out, "{line}");
            }
            let _ = writeln!(out, "OK ({ms} ms)");
        }
        Reply::Rows { cols, rows, ms } => {
            let n = cols.len();
            let mut widths: Vec<usize> = cols.iter().map(|c| display_width(c)).collect();
            for row in rows {
                for (i, cell) in row.iter().enumerate().take(n) {
                    widths[i] = widths[i].max(display_width(cell));
                }
            }
            let _ = writeln!(out, "| {} |", join(cols, &widths));
            let rule: Vec<String> = widths.iter().map(|w| "-".repeat(w + 2)).collect();
            let _ = writeln!(out, "+{}+", rule.join("+"));
            for row in rows {
                // 服务器给了少于列数的行时补空，避免错位
                let mut cells: Vec<&str> = row.iter().map(String::as_str).collect();
                cells.resize(n, "");
                let _ = writeln!(out, "| {} |", join(&cells, &widths));
            }
            let _ = writeln!(out, "{} 行 ({ms} ms)", rows.len());
        }
    }
    out
}

fn join<'a>(cells: &[impl AsRef<str>], widths: &[usize]) -> String {
    cells
        .iter()
        .enumerate()
        .map(|(i, c)| pad(c.as_ref(), widths[i]))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_counts_as_two_columns() {
        assert_eq!(display_width("所有权"), 6);
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("a所"), 3);
    }

    #[test]
    fn rows_align_across_scripts() {
        let reply = Reply::Rows {
            cols: vec!["id".into(), "title".into()],
            rows: vec![
                vec!["1".into(), "所有权笔记".into()],
                vec!["22".into(), "Rust".into()],
            ],
            ms: 3,
        };
        let text = render(&reply);
        let lines: Vec<&str> = text.lines().collect();
        let w = display_width(lines[0]);
        assert_eq!(w, display_width(lines[1]), "分隔线与表头等宽");
        assert_eq!(w, display_width(lines[2]), "CJK 行与 ASCII 行等宽");
        assert!(lines[4].ends_with("2 行 (3 ms)"));
    }

    #[test]
    fn short_row_is_padded() {
        let reply = Reply::Rows {
            cols: vec!["a".into(), "b".into(), "c".into()],
            rows: vec![vec!["1".into()]],
            ms: 0,
        };
        let line = render(&reply).lines().nth(2).unwrap().to_string();
        assert_eq!(line, "| 1 |   |   |");
    }

    #[test]
    fn error_renders_code() {
        let reply = Reply::Error { code: 7001, message: "弃判".into() };
        assert_eq!(render(&reply), "错误 7001: 弃判\n");
    }
}
