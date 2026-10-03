# MightBe 测试说明

- 测试完成：是（2026-10-04）
- 测试日期：2026-10-04
- 测试内容：单元覆盖 mightbe-core（张量/自动微分/层/优化器/DSL/composite）与 mightbe-nlp（规范化/分词/词表/共现/关键词）；集成覆盖 XOR/TextCNN/GRU 训练收敛与插件加载→前向→反向→热替换链路；注入测试覆盖 NLP 对 XSS/SQL 片段/路径穿越/NUL 字节的处理；钩子测试覆盖插件层未注册层拒绝、失败钩子隔离兄弟钩子、NUL 配置拒绝、层名路径穿越仅作符号键。
- 运行命令：`cargo test`（workspace）
- 测试框架：Rust #[cfg(test)]
- 模型：豆包（Doubao）生成

本仓库是 Cargo workspace（9 个内部 crate + 1 个测试用 cdylib `tests/mbp-demo`）。
Rust 集成测试按惯例放在**各 crate 自己的 `tests/` 目录**（workspace 根没有根 package，无法在仓库根挂集成测试 target），单元测试留在各 `src/*.rs` 内的 `#[cfg(test)] mod tests`。

## 测试放在哪里

| 位置 | 类型 | 覆盖 |
|---|---|---|
| `crates/mightbe-core/src/*.rs` | 单元 | 张量/自动微分/层/优化器/DSL/composite/插件宿主 的数值与梯度对拍 |
| `crates/mightbe-core/tests/train_m1.rs` | 集成 | XOR / TextCNN / GRU 训练收敛 |
| `crates/mightbe-core/tests/plugin_chain.rs` | 集成 | 插件加载→前向→反向→RELOAD 热替换→panic 隔离全链路 |
| `crates/mightbe-core/tests/plugin_hooks.rs` | **集成（钩子）** | 未注册层拒绝、失败钩子不污染兄弟钩子、NUL 配置拒绝、层名路径穿越仅作符号键 |
| `crates/mightbe-nlp/src/*.rs` | 单元 | 规范化/分词/词表/共现/关键词 |
| `crates/mightbe-nlp/tests/keyword.rs` | 集成 | 微型语料 TF-IDF/TextRank、`mb_*` 表可查、确定性 |
| `crates/mightbe-nlp/tests/injection.rs` | **集成（注入）** | XSS / SQL 注入 / 路径穿越 / 控制字符 不可信文档的安全处理 |
| `crates/mightbe-store/src/*.rs` + `tests/mdb.rs` | 单元+集成 | 页/WAL/加密/崩溃恢复/篡改拒绝/备份恢复 |
| `crates/mightbe-server/src/config.rs` | 单元 | 配置解析、未知键拒绝 |
| `crates/mightbe-cli/src/{wire,render}.rs` | 单元 | 文本协议帧解析、表格渲染 |

## 怎么运行

```powershell
# 全量（首次会下载依赖并编译，约 1 分钟）
cargo test --workspace

# 单 crate
cargo test -p mightbe-core
cargo test -p mightbe-nlp
cargo test -p mightbe-store
cargo test -p mightbe-server
cargo test -p mightbe-cli

# 只跑某类集成测试
cargo test -p mightbe-core --test plugin_hooks     # 钩子/插件装载侧
cargo test -p mightbe-nlp  --test injection       # 注入安全
cargo test -p mightbe-core --test plugin_chain    # 插件全链路
```

> 注：`plugin_chain.rs` / `plugin_hooks.rs` 会在测试进程内调 `cargo build -p mbp-demo`
> 产出 cdylib，再用 `LoadLibraryW`/`dlopen` 跨 ABI 拉起（产物落在 `target/mtb-plugin-build`）。
> 首次运行多花几秒，之后按 mtime 增量。

## 预期结果（本地基线）

全量 `cargo test --workspace` 应**全部通过、0 失败**：

- mightbe-core 单元：93（含本次新增的插件宿主相关用例）
- mightbe-nlp 单元：23
- mightbe-store 单元：18
- mightbe-server 单元：5
- mightbe-cli 单元：13
- 集成：`train_m1` 4 + `plugin_chain` 3 + `plugin_hooks` 4 + nlp `keyword` 12 + nlp `injection` 5 + store `mdb` 13

### 本次补强新增（相对原有 176 个用例）

- **单元测试 +4**（均在 `mightbe-nlp/src`）：
  - `normalize::tests::xss_payload_is_text_not_executed`
  - `normalize::tests::embedded_nul_and_control_chars_are_neutralized_not_crashing`
  - `tokenizer::tests::html_tags_are_punctuation_not_tokens`
  - `tokenizer::tests::path_traversal_is_lexical_only_no_semantics`
- **集成测试 +9**：
  - 注入安全（`crates/mightbe-nlp/tests/injection.rs`，5 个）：
    `xss_payload_is_tokenized_as_data_not_executed`、
    `sql_injection_text_is_lexically_tokenized_not_executed`、
    `path_traversal_string_creates_no_files_and_is_tokenized`、
    `null_bytes_and_huge_malicious_input_do_not_panic`、
    `injected_documents_keep_pipeline_deterministic`。
    断言：脚本标签/SQL 片段/路径串都被降级为普通词法 token（不进尖括号、不解释为路径），
    不 panic、不在工作目录产生文件、两次构建逐字节一致。
  - 钩子/插件（`crates/mightbe-core/tests/plugin_hooks.rs`，4 个）：
    `unregistered_layer_name_is_rejected_not_dispatched`（未注册钩子被 3xxx 错误码拒绝、不跨 ABI 派发）、
    `failing_hook_does_not_break_sibling_hook`（一个钩子 panic 被隔离后，兄弟钩子输出逐位不变）、
    `config_with_interior_nul_is_rejected_safely`（配置夹带 NUL → Config 错误而非崩溃）、
    `layer_name_with_path_traversal_is_symbol_key_only`（层名夹 `../` 仅作符号键、不二次加载文件、库仍健康）。

## 测试风格约定

- 数值用例以解析解/手工小例对拍，梯度用有限差分梯度检查；
- 插件/加密等跨边界用例断言**错误码区段**与**隔离性**，不只是"返回 Err"；
- 注入用例断言载荷被当作**数据**处理（词表/文件系统/索引结构不变），而非仅仅"没崩"。
