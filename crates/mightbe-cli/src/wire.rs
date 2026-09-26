//! MightBe TCP 线协议（README 4.6）的客户端侧编解码。
//!
//! 帧格式（UTF-8，`\n` 分隔）：
//! ```text
//! OK <col1>|<col2>|…          结果集列名（无列名时仅 "OK"）
//! <v1>|<v2>|…                  数据行，行数任意
//! INFO <自由文本>              可选，ACK/长任务的附加说明
//! END (rows=<N>, ms=<X>)       终止帧
//!
//! ERR <code> <message>         错误响应，单帧
//! ```
//! 请求侧为一条以 `;` 结尾的语句 + `\n`。
//!
//! 本模块与 `mightbe-server` 的编码端刻意保持独立：cli 只依赖协议（README 4.7）。

use std::io::{self, BufRead, Write};

#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    /// SELECT / SHOW 一类带列名的结果集
    Rows {
        cols: Vec<String>,
        rows: Vec<Vec<String>>,
        ms: u128,
    },
    /// INSERT / ALTER / CREATE 等回执
    Ack {
        info: Vec<String>,
        ms: u128,
    },
    /// 服务器错误（错误码见 README 第 9 章）
    Error { code: u16, message: String },
}

impl Reply {
    pub fn ms(&self) -> u128 {
        match self {
            Reply::Rows { ms, .. } | Reply::Ack { ms, .. } => *ms,
            Reply::Error { .. } => 0,
        }
    }
}

/// 发送一条语句。`;` 是协议的一部分，缺失时自动补上。
pub fn write_request(w: &mut impl Write, stmt: &str) -> io::Result<()> {
    let trimmed = stmt.trim_end();
    writeln!(w, "{}", if trimmed.ends_with(';') { trimmed.to_string() } else { format!("{trimmed};") })?;
    w.flush()
}

/// 读取一个完整响应帧。连接关闭且无数据时返回 `None`。
pub fn read_reply(r: &mut impl BufRead) -> io::Result<Option<Reply>> {
    let mut head = String::new();
    if r.read_line(&mut head)? == 0 {
        return Ok(None);
    }
    let head = head.trim_end();
    if let Some(rest) = head.strip_prefix("ERR") {
        return Ok(Some(parse_error(rest)?));
    }
    let Some(cols_part) = head.strip_prefix("OK") else {
        return Err(protocol_err(format!("期望 OK / ERR 帧，收到 {head:?}")));
    };

    let mut cols = split_cells(cols_part.trim_start());
    if cols.len() == 1 && cols[0].is_empty() {
        cols.clear();
    }

    let mut info = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let ms;
    loop {
        let mut line = String::new();
        if r.read_line(&mut line)? == 0 {
            return Err(protocol_err("连接在 END 帧之前关闭".to_string()));
        }
        let line = line.trim_end();
        if let Some(tail) = line.strip_prefix("END") {
            ms = parse_end(tail)?;
            break;
        }
        if let Some(text) = line.strip_prefix("INFO") {
            info.push(text.trim_start().to_string());
            continue;
        }
        rows.push(split_cells(line));
    }

    Ok(Some(if cols.is_empty() {
        Reply::Ack { info, ms }
    } else {
        Reply::Rows { cols, rows, ms }
    }))
}

fn parse_error(rest: &str) -> io::Result<Reply> {
    let rest = rest.trim_start();
    let (code, message) = match rest.split_once(' ') {
        Some((c, m)) => (c, m),
        None => ("0", rest),
    };
    Ok(Reply::Error {
        code: code.parse().map_err(|_| {
            protocol_err(format!("错误码不是无符号整数: {code:?}"))
        })?,
        message: message.to_string(),
    })
}

/// `END (rows=N, ms=X)` → X
fn parse_end(tail: &str) -> io::Result<u128> {
    let inner = tail
        .trim_start()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| protocol_err(format!("END 帧格式错误: {tail:?}")))?;
    let mut rows = None;
    let mut ms = None;
    for kv in inner.split(',') {
        let Some((k, v)) = kv.trim().split_once('=') else {
            return Err(protocol_err(format!("END 帧字段错误: {kv:?}")));
        };
        let v = v.trim();
        match k.trim() {
            "rows" => rows = Some(v),
            "ms" => ms = Some(v),
            other => return Err(protocol_err(format!("END 帧未知字段 {other}"))),
        }
    }
    if rows.is_none() {
        return Err(protocol_err("END 帧缺少 rows".to_string()));
    }
    Ok(ms
        .unwrap_or("0")
        .parse()
        .map_err(|_| protocol_err(format!("ms 不是整数: {:?}", ms.unwrap_or(""))))?)
}

fn split_cells(line: &str) -> Vec<String> {
    line.split('|').map(|c| c.to_string()).collect()
}

fn protocol_err(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// 交互模式下判断一条输入是否已构成完整语句：以 `;` 结束，且该 `;` 不在单引号字符串内。
/// 字符串内 `''` 转义在这里天然表现为"闭合 + 重开"，因此逐字符翻转即可。
pub fn statement_complete(src: &str) -> bool {
    let mut in_string = false;
    for c in src.chars() {
        match c {
            '\'' => in_string = !in_string,
            ';' if !in_string => return true,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(wire: &str) -> io::Result<Option<Reply>> {
        read_reply(&mut io::BufReader::new(wire.as_bytes()))
    }

    #[test]
    fn result_set_roundtrip() {
        let wire = "OK id|title\n1|所有权笔记\n2|借用检查\nEND (rows=2, ms=7)\n";
        assert_eq!(
            read(wire).unwrap().unwrap(),
            Reply::Rows {
                cols: vec!["id".into(), "title".into()],
                rows: vec![
                    vec!["1".into(), "所有权笔记".into()],
                    vec!["2".into(), "借用检查".into()]
                ],
                ms: 7
            }
        );
    }

    #[test]
    fn ack_with_info() {
        let wire = "OK\nINFO affected=1\nEND (rows=0, ms=0)\n";
        assert_eq!(
            read(wire).unwrap().unwrap(),
            Reply::Ack { info: vec!["affected=1".into()], ms: 0 }
        );
    }

    #[test]
    fn error_frame() {
        let wire = "ERR 7001 证据不足，拒绝结论\n";
        assert_eq!(
            read(wire).unwrap().unwrap(),
            Reply::Error { code: 7001, message: "证据不足，拒绝结论".into() }
        );
    }

    #[test]
    fn empty_cell_row_keeps_layout() {
        let wire = "OK a|b|c\n1||3\nEND (rows=1, ms=1)\n";
        let Reply::Rows { rows, .. } = read(wire).unwrap().unwrap() else {
            panic!("期望结果集")
        };
        assert_eq!(rows[0], vec!["1".to_string(), String::new(), "3".to_string()]);
    }

    #[test]
    fn truncated_stream_is_error() {
        assert!(read("OK a\n1\n").is_err());
    }

    #[test]
    fn closed_connection_is_none() {
        assert!(read("").unwrap().is_none());
    }

    #[test]
    fn request_always_terminated_by_semicolon() {
        let mut buf = Vec::new();
        write_request(&mut buf, "SELECT 1").unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "SELECT 1;\n");

        let mut buf = Vec::new();
        write_request(&mut buf, "SELECT 1;\n").unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "SELECT 1;\n");
    }

    #[test]
    fn semicolon_inside_string_does_not_terminate() {
        assert!(!statement_complete("SELECT 'a;b'"));
        assert!(statement_complete("INSERT INTO t VALUES ('a;b');"));
        assert!(!statement_complete("SELECT ';'"));
        assert!(statement_complete("SELECT '';"));
    }

    #[test]
    fn bad_end_frame_rejected() {
        assert!(read("OK a\nEND (rowz=1, ms=0)\n").is_err());
    }
}
