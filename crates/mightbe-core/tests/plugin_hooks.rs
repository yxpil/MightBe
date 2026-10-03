//! 钩子/插件（Layer Plugin SDK）装载侧的「拒绝」与「隔离」集成测试。
//!
//! 对应钩子机制的四类断言：
//! 1. **未注册/越权钩子被拒绝**：符号表里没有的层名必须带错误码返回，绝不跨 ABI 派发；
//! 2. **失败隔离**：同一进程里一个钩子（插件层）调用失败/panic，不影响兄弟钩子实例；
//! 3. **配置注入**：钩子配置夹带内嵌 NUL 必须被安全拒绝，而非 panic/越界；
//! 4. **层名注入**：层名里夹带路径穿越，只被当作符号表的键，绝不被拼成文件路径二次加载。
//!
//! 插件不是打桩：与 `plugin_chain.rs` 一样，本文件真的在测试进程里 `cargo build -p mbp-demo`
//! 产出动态库再跨 ABI 拉起。构建走独立 `--target-dir`，避开外层 target 锁。

use std::path::{Path, PathBuf};
use std::process::Command;

use mightbe_core::api::MtbError;
use mightbe_core::plugin::LoadedLibrary;
use mightbe_core::tensor::Tensor;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace 根存在")
}

fn build_dir() -> PathBuf {
    workspace_root().join("target").join("mtb-plugin-build")
}

fn dll_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

/// 构建并返回 mbp-demo 产物路径（cargo 按 mtime 自判增量）。
fn built_dll() -> PathBuf {
    let dir = build_dir();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = Command::new(cargo)
        .args(["build", "--offline", "-p", "mbp-demo", "--target-dir"])
        .arg(&dir)
        .current_dir(workspace_root())
        .output()
        .expect("cargo 可执行");
    assert!(
        out.status.success(),
        "构建 mbp-demo 失败：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir.join("debug").join(dll_name("mbp_demo"))
}

/// 已注册层名集合（来自 mbp-demo：scale_tanh / boom）。
fn registered() -> &'static [&'static str] {
    &["scale_tanh", "boom"]
}

#[test]
fn unregistered_layer_name_is_rejected_not_dispatched() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");
    // 符号表里确实没有这个名字
    for name in registered() {
        assert!(lib.layer_names().iter().any(|n| n == name));
    }
    // 请求一个从未导出的钩子层 → 必须带错误码拒绝，绝不尝试跨 ABI 调用
    let e = lib.create("totally_unknown_hook", "{}").unwrap_err();
    match e {
        MtbError::Coded { code, .. } => {
            // 未注册层落在 3xxx 段（NETWORK=3000），不是 3102 缺符号（库本身是好的）
            assert!((3000..4000).contains(&code), "应为 3xxx 段拒绝码: {e:?}");
        }
        other => panic!("未注册钩子必须以错误返回，不得派发: {other:?}"),
    }
}

#[test]
fn failing_hook_does_not_break_sibling_hook() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");

    // 一个健康钩子 + 一个会炸钩子，同库共存
    let healthy = lib
        .smoke("scale_tanh", r#"{"in":3,"out":2,"scale":1}"#, &[vec![1, 3]])
        .expect("健康钩子冒烟");
    let bomber = lib
        .create("boom", r#"{"trip":1,"width":3}"#)
        .expect("建会炸钩子实例");

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    // 健康钩子的基线输出（无权重槽 → 全零前向，确定性）
    let base = healthy.forward(&[&x]).expect("健康钩子前向");

    // 会炸钩子 panic，被 SDK/宿主隔离成错误码
    assert!(
        bomber.forward(&[&x]).is_err(),
        "trip=1 的钩子应当失败"
    );

    // 兄弟钩子不受影响：输出与失败前逐位一致
    let after = healthy.forward(&[&x]).expect("兄弟钩子在失败后仍可用");
    assert_eq!(base.data, after.data, "一个钩子的失败不得污染兄弟钩子的输出");
}

#[test]
fn config_with_interior_nul_is_rejected_safely() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");
    // 配置 JSON 里塞内嵌 NUL：CString::new 失败，必须转成 Config 错误而非 panic/越界
    let e = lib
        .create("scale_tanh", "{\"in\":3,\"out\":2,\"scale\":1\u{0000}}")
        .unwrap_err();
    match e {
        MtbError::Config(_) => {}
        other => panic!("含内嵌 NUL 的配置应被安全拒绝为 Config 错误: {other:?}"),
    }
}

#[test]
fn layer_name_with_path_traversal_is_symbol_key_only() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");
    // 层名里夹带 ../ —— 它只是符号表的键，绝不该被拼成文件路径再 LoadLibrary
    let e = lib.create("../../etc/passwd", "{}").unwrap_err();
    // 未注册 → 带错误码拒绝；关键是库句柄本身仍健康（open 已成功）
    assert!(
        matches!(e, MtbError::Coded { .. } | MtbError::Config(_) | MtbError::Other(_)),
        "路径穿越层名应按未注册符号拒绝: {e:?}"
    );
    // 合法层照常可用：证明这次"恶意层名"没有破坏库状态
    let ok = lib
        .smoke("boom", r#"{"trip":0,"width":2}"#, &[vec![2]])
        .expect("合法层冒烟");
    assert_eq!(ok.layer_name(), "boom");
}
