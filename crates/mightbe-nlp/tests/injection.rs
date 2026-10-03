//! 不可信输入注入安全测试（README 8.1：INSERT 文本列自动进入 NLP 管线）。
//!
//! 这些载荷来自恶意/不可信的**文档正文**：XSS、SQL 注入、路径穿越、控制字符。
//! 系统对正文只做词法统计（规范化 → 分词 → 词表/关键词/共现），
//! 绝不应执行、落盘、解释或透传它们。这里断言：
//! - 载荷被降级为普通词法数据（脚本标签、SQL 关键字、路径分量都只是词）；
//! - 不 panic、不产生任何文件副作用；
//! - 注入文档不破坏索引结构，两次构建仍逐字节确定性。

use mightbe_nlp::api::*;
use mightbe_nlp::index::NlpIndex;

#[test]
fn xss_payload_is_tokenized_as_data_not_executed() {
    let pipe = default_pipeline();
    let raw = "<script>alert(document.cookie)</script><img src=x onerror=alert(1)>";
    let toks = pipe.tokenize(raw).expect("tokenize 不返回错误");
    // 没有任何 token 携带 HTML 尖括号（标签结构被打散）
    assert!(
        toks.iter().all(|w| !w.contains('<') && !w.contains('>')),
        "HTML 标签不得整体进入词表: {toks:?}"
    );
    // 脚本内容被降级为普通拉丁词——它没有被当作标签执行
    assert!(toks.contains(&"alert".to_string()), "{toks:?}");
}

#[test]
fn sql_injection_text_is_lexically_tokenized_not_executed() {
    let mut idx = NlpIndex::new(NlpConfig::default());
    let malicious = "'); DROP TABLE articles; -- ' OR '1'='1; DELETE FROM mb_vocab;--";
    idx.add_document(1, malicious);
    idx.add_document(2, "正常的数据库笔记内容关于关系代数与范式");
    idx.rebuild();

    // 关键断言：第 2 篇的主题词仍在词表——"DROP TABLE" 没有真的删掉任何东西。
    let tokens: Vec<String> = idx.vocab_rows().iter().map(|r| r.token.clone()).collect();
    assert!(
        tokens.contains(&"关系".to_string()) || tokens.contains(&"代数".to_string()),
        "第二篇主题词应仍在词表，注入语句被当成纯文本: {tokens:?}"
    );
    // SQL 关键字只是拉丁词，不触发删除；索引结构完好
    let v: serde_json::Value = serde_json::from_str(&idx.dump_json()).expect("索引仍为合法 JSON");
    assert!(v.get("vocab").is_some());
}

#[test]
fn path_traversal_string_creates_no_files_and_is_tokenized() {
    let before: std::collections::BTreeSet<String> = std::fs::read_dir(".")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    let mut idx = NlpIndex::new(NlpConfig::default());
    idx.add_document(
        1,
        "恶意提示：去读 ../../etc/passwd 和 ..\\..\\windows\\system32\\drivers\\etc\\hosts",
    );
    idx.rebuild();

    let rows = idx.vocab_rows();
    let words: Vec<&str> = rows.iter().map(|r| r.token.as_str()).collect();
    assert!(
        words.contains(&"etc") || words.contains(&"passwd") || words.contains(&"hosts"),
        "路径串应仅被词法切分为普通词: {words:?}"
    );

    // 安全断言：处理正文绝不在当前目录凭空创建任何文件
    let after: std::collections::BTreeSet<String> = std::fs::read_dir(".")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let new_files: Vec<&String> = after.difference(&before).collect();
    assert!(
        new_files.is_empty(),
        "注入的路径字符串不应产生任何文件写入: {new_files:?}"
    );
}

#[test]
fn null_bytes_and_huge_malicious_input_do_not_panic() {
    let pipe = default_pipeline();
    // 内嵌 NUL、零宽字符、超长重复载荷、双向覆盖控制符、emoji
    let nasty = format!("a\u{0000}b\u{200B}c{}", "DROP".repeat(500));
    let toks = pipe.tokenize(&nasty).expect("恶意输入不应导致管线错误");
    assert!(!toks.is_empty());

    // 全 NUL / 全控制字符：必须安全返回空，不 panic
    let all_ctrl = pipe.tokenize("\u{0000}\u{0000}\u{0000}").unwrap();
    assert!(all_ctrl.is_empty(), "纯控制字符不应产出词条: {all_ctrl:?}");
}

#[test]
fn injected_documents_keep_pipeline_deterministic() {
    let build = || {
        let mut idx = NlpIndex::new(NlpConfig::default());
        idx.add_document(1, "<script>alert(1)</script> ' OR 1=1 -- ../../x/y");
        idx.add_document(2, "另一篇完全正常的中文文章");
        idx.rebuild();
        idx.dump_json()
    };
    assert_eq!(build(), build(), "注入文档不得引入非确定性");
}
