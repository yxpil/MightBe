//! 路径 C（Layer Plugin）全链路集成测试（README 16.3）。
//!
//! 覆盖：构建 cdylib → 加载（3101/3102 校验）→ 冒烟（3103）→ 图内前向 → 反向 →
//! RELOAD 热替换（旧实现仍在途）→ 插件 panic 被隔离成错误码且宿主线程存活。
//!
//! 插件不是打桩：本文件真的在测试进程外调 `cargo build -p mbp-demo` 产出动态库，
//! 再用 `LoadLibraryW`/`dlopen` 拉起来跨 ABI 调用。构建走独立的 `--target-dir`，
//! 避免与外层 `cargo test` 抢同一把 target 锁。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use mightbe_core::api::{Layer, LayerCtx, MtbError};
use mightbe_core::autograd::Graph;
use mightbe_core::init::Init;
use mightbe_core::plugin::{LoadedLibrary, PluginInstance, PluginLayer};
use mightbe_core::tensor::Tensor;

/// `scale=1` 的实现版本。
const CFG_A: &str = r#"{"in":3,"out":2,"scale":1}"#;
/// 同一份二进制、`scale=2` 的实现版本——RELOAD 换的是实现，不是文件。
const CFG_B: &str = r#"{"in":3,"out":2,"scale":2}"#;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace 根存在")
}

/// 独立 target 目录：测试进程里再起 cargo，必须避开外层 `target/` 的锁。
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

/// 构建并返回 mbp-demo 产物路径。
///
/// 这里**不做**"只构建一次"的缓存：一旦缓存，改完插件源码后跑测试仍会拿到旧二进制，
/// 报错会假装成宿主的问题。交给 cargo 自己按 mtime 判断即可，产物已在时几乎零开销。
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
    let dll = dir.join("debug").join(dll_name("mbp_demo"));
    assert!(
        dll.exists(),
        "未找到 {:?}，目录内容 {:?}",
        dll,
        std::fs::read_dir(dir.join("debug"))
            .map(|d| d
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>())
            .unwrap_or_default()
    );
    dll
}

/// 把产物复制成一个"版本文件"，模拟 RELOAD 换库。
fn versioned_dll(tag: &str) -> PathBuf {
    let dir = build_dir().join("versions");
    std::fs::create_dir_all(&dir).expect("创建 versions 目录");
    let dst = dir.join(dll_name(&format!("mbp_demo_{tag}")));
    std::fs::copy(built_dll(), &dst).expect("复制插件产物");
    dst
}

// ───────────────────────── 小工具 ─────────────────────────

fn training_ctx() -> LayerCtx {
    LayerCtx {
        training: true,
        seq_state: None,
    }
}

fn ones_like(shape: &[usize]) -> Tensor {
    Tensor::from_vec(vec![1.0; shape.iter().product::<usize>()], shape.to_vec())
        .expect("全 1 张量")
}

/// `(m,k) × (k,n)` 的朴素参考实现。
fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// 行主序 `m×k` 转置为 `k×m`；层数少、规模小，朴素写法就够。
fn transpose_ref(a: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut t = vec![0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            t[c * m + r] = a[r * k + c];
        }
    }
    t
}

fn near(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "长度不一致");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "第 {i} 个元素差异过大：{x} vs {y}（tol={tol}）"
        );
    }
}

// ───────────────────────── 测试 ─────────────────────────

#[test]
fn plugin_load_smoke_forward_backward() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");
    assert_eq!(lib.api_version(), 1);
    assert_eq!(lib.layer_names(), &["scale_tanh".to_string(), "boom".to_string()]);

    // 冒烟：零输入前向形状相符
    lib.smoke("scale_tanh", CFG_A, &[vec![2, 3]])
        .expect("scale_tanh 冒烟");
    lib.smoke("boom", r#"{"trip":0,"width":4}"#, &[vec![1]])
        .expect("boom 冒烟");

    // 3103：形状推导与前向不符必须拦在装载期
    let bad = lib.smoke("scale_tanh", CFG_A, &[vec![2, 5]]);
    match bad {
        Err(MtbError::Coded { code, .. }) => {
            assert_eq!(code, MtbError::PLUGIN_SMOKE_FAIL, "应为 3103")
        }
        other => panic!("期望 3103，得到 {other:?}"),
    }

    // 权重声明与静态形状推导
    let inst: Arc<PluginInstance> = lib.smoke("scale_tanh", CFG_A, &[vec![2, 3]]).unwrap();
    let decls = inst.param_decls().expect("参数声明");
    assert_eq!(decls.len(), 1, "scale_tanh 只有 w");
    assert_eq!(decls[0].0, "w");
    assert_eq!(decls[0].1, vec![2, 3]);
    assert_eq!(
        inst.infer_shapes(&[vec![2usize, 3]]).expect("形状推导"),
        vec![2usize, 2]
    );

    // 建层：权重形状由插件声明，实际分配在宿主
    let mut layer =
        PluginLayer::new("pl", inst.clone(), CFG_A, &Init::Normal { mean: 0.0, std: 1.0 }, 7)
            .expect("建插件层");
    assert_eq!(layer.param_names(), vec!["w".to_string()]);
    let in_shape: Vec<usize> = vec![2, 3];
    // 返回的是「每个输出一个形状」，插件层 v1 单输出
    assert_eq!(
        layer.infer_shapes(&[&in_shape]).unwrap(),
        vec![vec![2usize, 2]]
    );

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let w = layer.dump_params()[0].1.clone();
    assert_eq!(w.shape, vec![2, 3]);

    // 图内前向：与解析参考对拍。
    // 插件实现的是 `pre = x @ Wᵀ`（`W: [out, in]`），所以参考里要给的是 `Wᵀ: [in, out]`——
    // 直接把 `w.data` 当 `[in, out]` 用会连错轴，得到的是另一个矩阵。
    let wt = transpose_ref(&w.data, 2, 3);
    let expect: Vec<f32> = matmul_ref(&x.data, &wt, 2, 3, 2)
        .iter()
        .map(|&v| tanh_ref(v))
        .collect();
    let mut g = Graph::new();
    let xv = g.constant(x.clone());
    let outs = layer.forward(&[&xv], &training_ctx(), &mut g).expect("插件层前向");
    assert_eq!(outs.len(), 1, "插件层 v1 单输出");
    let out = outs[0];
    near(&g.value(out).data, &expect, 1e-5);

    // 反向：梯度回传到宿主侧的 w
    //
    // 插件实现的是 `y = tanh(scale · (x @ Wᵀ))`，故 `dL/dW[c][p] = Σ_r (1 - y[r][c]²)·x[r][p]`
    // （`1 - tanh²` 就是 `sech²`，因此无需反解 pre 就能拿到正确梯度）。
    g.backward(&[out]).expect("反向");
    let wvar = g.param("pl.w", w.clone());
    let grad = g.grad(wvar).expect("w 应有梯度");
    let yv = g.value(out).data.clone();
    let mut grad_ref = vec![0f32; 2 * 3];
    for r in 0..2 {
        for p in 0..3 {
            let xv = x.data[r * 3 + p];
            for c in 0..2 {
                grad_ref[c * 3 + p] += (1.0 - yv[r * 2 + c] * yv[r * 2 + c]) * xv;
            }
        }
    }
    near(&grad.data, &grad_ref, 1e-4);

    assert_eq!(layer.param_shapes().len(), 1);
    assert!(layer.library_path().exists());
}

#[test]
fn reload_swaps_implementation_and_old_snapshot_still_serves() {
    let v1 = versioned_dll("v1");
    let v2 = versioned_dll("v2");

    // 两个版本文件各自持有独立句柄：热切换时旧句柄不得被 FreeLibrary
    let lib_a = LoadedLibrary::open(&v1).expect("加载 v1");
    let lib_b = LoadedLibrary::open(&v2).expect("加载 v2");

    // 冒烟实例（实例不带权重，只验实现可用）
    let smoke_a = lib_a.smoke("scale_tanh", CFG_A, &[vec![1, 3]]).expect("v1 冒烟");
    let smoke_b = lib_b.smoke("scale_tanh", CFG_B, &[vec![1, 3]]).expect("v2 冒烟");

    // 权重在宿主侧按声明形状分配：用同一颗种子，两份实现拿到的就是同一份 `w`。
    let probe = PluginLayer::new(
        "probe",
        smoke_a.clone(),
        CFG_A,
        &Init::Normal { mean: 0.0, std: 1.0 },
        11,
    )
    .expect("建探针层只为取权重");
    let w = probe.dump_params()[0].1.clone();
    assert_eq!(w.shape, vec![2, 3]);

    let inst_a = lib_a.create("scale_tanh", CFG_A).expect("v1 实例");
    let inst_b = lib_b.create("scale_tanh", CFG_B).expect("v2 实例");

    let x = Tensor::from_vec(vec![0.5, -0.25, 2.0], vec![1, 3]).unwrap();
    // 宿主契约：数据输入在前、参数在后，顺序必须和前向一致
    let y_a = inst_a.forward(&[&x, &w]).expect("v1 前向");
    let y_b = inst_b.forward(&[&x, &w]).expect("v2 前向");
    assert_eq!(y_a.shape, y_b.shape);
    assert!(
        y_a.data.iter().any(|v| v.abs() > 1e-6),
        "权重槽若没传进去，两边都会退化成 tanh(0)，这条用例就白写了"
    );

    // 同一份权重、不同实现：`y = tanh(k · pre)`，故 `y_b = tanh(2·atanh(y_a))`
    let expect_b: Vec<f32> = y_a
        .data
        .iter()
        .map(|&v| {
            let pre = 0.5 * ((1.0 + v) / (1.0 - v)).ln();
            tanh_ref(2.0 * pre)
        })
        .collect();
    near(&y_b.data, &expect_b, 1e-4);

    // 反向在替换之后依然可用，梯度槽与输入槽一一对应
    let g = ones_like(&y_a.shape);
    let grads = inst_a
        .backward(&g, &[&x, &w], &y_a)
        .expect("旧实例反向");
    assert_eq!(grads.len(), 2, "数据输入 + 参数各一个梯度槽");
    assert_eq!(grads[0].shape, x.shape);
    assert_eq!(grads[1].shape, w.shape);

    // 替换发生：新实例被丢弃；旧实例（在途请求）继续可用，库不被卸载
    drop(inst_b);
    let y_a_again = inst_a.forward(&[&x, &w]).expect("替换后旧实例仍在途");
    near(&y_a.data, &y_a_again.data, 1e-9);

    // 旧库句柄仍可再建实例
    let inst_a2 = lib_a.create("scale_tanh", CFG_A).expect("v1 再建实例");
    near(
        &inst_a2.forward(&[&x, &w]).expect("新实例前向").data,
        &y_a_again.data,
        1e-9,
    );

    drop(smoke_b);
    // 旧的冒烟实例先死，新实例再死，库句柄才可能归零——归零路径不得 panic
    drop(inst_a2);
    drop(smoke_a);
}

/// `tanh` 的解析参考。
///
/// 注意别在分式外面再套一层 `.tanh()`：`(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` 本身就是 tanh，
/// 套两层会把饱和区（pre≈7）的 0.9999997 变成 `tanh(1)=0.7616`，对拍直接假失败。
fn tanh_ref(v: f32) -> f32 {
    let up = v.exp();
    let down = (-v).exp();
    (up - down) / (up + down)
}

#[test]
fn plugin_panic_is_isolated_as_error_code() {
    let lib = LoadedLibrary::open(built_dll()).expect("加载 mbp-demo");

    // 未 trip 的实例正常前向：width=3 把 [2] 展开成 [2,3]
    let ok_inst = lib.smoke("boom", r#"{"trip":0,"width":3}"#, &[vec![2]]).expect("boom 冒烟");
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let y = ok_inst.forward(&[&x]).expect("正常前向");
    assert_eq!(y.shape, vec![2, 3]);
    assert_eq!(y.data.len(), 6, "输入重复 width 次");

    // trip 的实例在前向里 panic：必须转成错误码，不允许跨 extern "C" 展开
    let boom = lib.create("boom", r#"{"trip":1,"width":3}"#).expect("建 boom 实例");
    let e = boom.forward(&[&x]).expect_err("trip=1 应当失败");
    assert!(
        matches!(e, MtbError::Coded { .. }),
        "插件 panic 必须落成错误码：{e:?}"
    );

    // 宿主进程存活：紧接着的合法调用照常返回
    let y2 = ok_inst.forward(&[&x]).expect("panic 之后仍可用");
    near(&y2.data, &y.data, 1e-9);
}
