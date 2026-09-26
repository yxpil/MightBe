//! 计算后端：MVP 只实现 `naive`（纯 Rust、零依赖、任何 CPU 可编译）。
//!
//! SIMD / CUDA / NPU 走 feature gate（README 4.1 后端表），默认构建不含它们。

use crate::api::{Backend, ConvParams, ElemOp, GemmShapes, MtbResult, ReduceKind};
use crate::tensor::Tensor;

/// MVP 基线后端：纯 Rust + 缓存友好分块。
pub struct NaiveBackend;

impl NaiveBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn arc() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self)
    }
}

impl Default for NaiveBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for NaiveBackend {
    fn id(&self) -> &str {
        "naive"
    }

    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        g: &GemmShapes,
    ) -> MtbResult<()> {
        let (m, k, n) = (g.m, g.k, g.n);
        check_len(a.len(), m * k, "matmul A")?;
        check_len(b.len(), k * n, "matmul B")?;
        check_len(out.len(), m * n, "matmul out")?;
        // 分块 ikj：内层对 out 与 B 的一行都是顺序访问，且 A 的元素只读一次。
        const BM: usize = 64;
        const BN: usize = 64;
        for i0 in (0..m).step_by(BM) {
            let i1 = (i0 + BM).min(m);
            for j0 in (0..n).step_by(BN) {
                let j1 = (j0 + BN).min(n);
                for i in i0..i1 {
                    out[i * n + j0..i * n + j1].fill(0.0);
                    let orow = &mut out[i * n + j0..i * n + j1];
                    for p in 0..k {
                        let av = a[i * k + p];
                        if av == 0.0 {
                            continue;
                        }
                        let brow = &b[p * n + j0..p * n + j1];
                        for (o, &bv) in orow.iter_mut().zip(brow) {
                            *o += av * bv;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// 单样本一维卷积：`x` 为通道优先 `(in_ch, in_len)`，`w` 为 `(filters, in_ch*kernel)`，
    /// `out` 为 `(filters, out_len)`。批维度由调用方（autograd）逐样本切分。
    fn conv1d(
        &self,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        out: &mut [f32],
        p: &ConvParams,
    ) -> MtbResult<()> {
        let (in_len, in_ch, kernel, filters) = (p.in_len, p.in_ch, p.kernel, p.filters);
        let (stride, dilation, padding) = (p.stride.max(1), p.dilation.max(1), p.padding);
        let win = in_ch * kernel;
        check_len(x.len(), in_len * in_ch, "conv1d input")?;
        check_len(w.len(), win * filters, "conv1d filter")?;
        check_len(bias.len(), filters, "conv1d bias")?;

        let out_len = (in_len + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
        check_len(out.len(), out_len * filters, "conv1d out")?;

        for oc in 0..filters {
            let wbase = oc * win;
            for o in 0..out_len {
                let base_in = (o * stride) as isize - padding as isize;
                let mut acc = bias[oc];
                for c in 0..in_ch {
                    let xrow = &x[c * in_len..c * in_len + in_len];
                    let wrow = &w[wbase + c * kernel..wbase + c * kernel + kernel];
                    for (kk, &wv) in wrow.iter().enumerate() {
                        let i = base_in + kk as isize * dilation as isize;
                        if i < 0 || i >= in_len as isize {
                            continue;
                        }
                        acc += xrow[i as usize] * wv;
                    }
                }
                out[oc * out_len + o] = acc;
            }
        }
        Ok(())
    }

    fn elementwise(&self, out: &mut [f32], op: ElemOp) -> MtbResult<()> {
        for o in out.iter_mut() {
            *o = apply_elem(*o, op);
        }
        Ok(())
    }

    fn reduce(
        &self,
        x: &[f32],
        shape: &[usize],
        out: &mut [f32],
        axis: usize,
        r: ReduceKind,
    ) -> MtbResult<()> {
        if axis >= shape.len() {
            return Err(crate::api::MtbError::Shape {
                expected: format!("axis < {}", shape.len()),
                got: format!("axis = {axis}"),
            });
        }
        let dim = shape[axis].max(1);
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        check_len(x.len(), outer * dim * inner, "reduce input")?;
        check_len(out.len(), outer * inner, "reduce out")?;
        for o in 0..outer {
            for j in 0..inner {
                let start = (o * dim * inner) + j;
                let mut acc = match r {
                    ReduceKind::Max => f32::NEG_INFINITY,
                    _ => 0.0,
                };
                for k in 0..dim {
                    let v = x[start + k * inner];
                    match r {
                        ReduceKind::Max => acc = acc.max(v),
                        _ => acc += v,
                    }
                }
                out[o * inner + j] = if r == ReduceKind::Mean {
                    acc / dim as f32
                } else {
                    acc
                };
            }
        }
        Ok(())
    }

    fn embedding_lookup(
        &self,
        emb: &[f32],
        ids: &[u32],
        out: &mut [f32],
        dim: usize,
    ) -> MtbResult<()> {
        if dim == 0 || emb.len() % dim != 0 {
            return Err(crate::api::MtbError::Shape {
                expected: format!("词表行数 × {dim}"),
                got: format!("{} 个权重", emb.len()),
            });
        }
        let vocab = emb.len() / dim;
        check_len(out.len(), ids.len() * dim, "emb out")?;
        for (r, &id) in ids.iter().enumerate() {
            let row = (id as usize).checked_mul(dim).ok_or_else(|| {
                crate::api::MtbError::Shape {
                    expected: format!("id < {vocab}"),
                    got: format!("id = {id}"),
                }
            })?;
            if row + dim > emb.len() {
                return Err(crate::api::MtbError::Shape {
                    expected: format!("id < {vocab}"),
                    got: format!("id = {id}"),
                });
            }
            out[r * dim..r * dim + dim].copy_from_slice(&emb[row..row + dim]);
        }
        Ok(())
    }
}

pub fn apply_elem(x: f32, op: ElemOp) -> f32 {
    match op {
        ElemOp::Neg => -x,
        ElemOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ElemOp::Tanh => x.tanh(),
        ElemOp::Relu => if x > 0.0 { x } else { 0.0 },
        ElemOp::Gelu => 0.5 * x * (1.0 + ((0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())),
        ElemOp::Exp => x.exp(),
        ElemOp::Log => x.ln(),
        ElemOp::Sqrt => x.sqrt(),
        ElemOp::Square => x * x,
        ElemOp::Scale(s) => x * s,
    }
}

fn check_len(got: usize, want: usize, what: &str) -> MtbResult<()> {
    if got != want {
        return Err(crate::api::MtbError::Shape {
            expected: format!("{what}: {want}"),
            got: format!("{what}: {got}"),
        });
    }
    Ok(())
}

/// 设备选择（README 4.1 "设备选择与约束"）。MVP 只有 naive。
pub fn select_backend(device: &str) -> MtbResult<std::sync::Arc<dyn Backend>> {
    match device {
        "" | "cpu" | "naive" => Ok(NaiveBackend::arc()),
        other if other.starts_with("cuda") || other.starts_with("npu") => {
            Err(crate::api::MtbError::DeviceUnavailable {
                device: other.to_string(),
                reason: "GPU/NPU 后端在 M9 之前不可用（feature gate）".into(),
            })
        }
        other => Err(crate::api::MtbError::DeviceUnavailable {
            device: other.to_string(),
            reason: "未知后端".into(),
        }),
    }
}

/// 后端统一入口：把 `Tensor` 上的 matmul 交给后端执行。
/// [`crate::autograd::Graph`] 内部走的正是同一份内核，这里留给层实现与测试直接调用。
pub fn backend_matmul(
    backend: &dyn Backend,
    a: &Tensor,
    b: &Tensor,
) -> MtbResult<Tensor> {
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];
    let mut out = vec![0f32; m * n];
    backend.matmul(&a.data, &b.data, &mut out, &GemmShapes { m, k, n })?;
    Tensor::from_vec(out, vec![m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_graph() {
        let b = NaiveBackend;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let bb = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        b.matmul(&a, &bb, &mut out, &GemmShapes { m: 2, k: 2, n: 2 })
            .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv1d_same_padding_shape() {
        let b = NaiveBackend;
        let in_ch = 1;
        let kernel = 3;
        let in_len = 5;
        let x = vec![0.0f32; in_len * in_ch];
        let w = vec![0.0f32; in_ch * kernel * 1];
        let bias = vec![0.0f32; 1];
        let params = ConvParams {
            in_len,
            in_ch,
            kernel,
            filters: 1,
            stride: 1,
            padding: 1,
            dilation: 1,
        };
        let out_len = (in_len + 2 - (kernel - 1) - 1) / 1 + 1;
        let mut out = vec![0.0f32; out_len];
        b.conv1d(&x, &w, &bias, &mut out, &params).unwrap();
        assert_eq!(out.len(), in_len, "padding=same 时输出长度应等于输入");
    }

    #[test]
    fn embedding_lookup_rows() {
        let b = NaiveBackend;
        let emb: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let mut out = vec![0.0; 2 * 3];
        b.embedding_lookup(&emb, &[1, 2], &mut out, 3).unwrap();
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut bad = vec![0.0; 3];
        assert!(b.embedding_lookup(&emb, &[3], &mut bad, 3).is_err(), "越界 id 必须报错");
    }

    #[test]
    fn reduce_matches_axis_semantics() {
        let b = NaiveBackend;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2, 3)
        let mut cols = vec![0.0; 2];
        b.reduce(&x, &[2, 3], &mut cols, 1, ReduceKind::Sum).unwrap();
        assert_eq!(cols, vec![6.0, 15.0]);
        let mut rows = vec![0.0; 3];
        b.reduce(&x, &[2, 3], &mut rows, 0, ReduceKind::Mean).unwrap();
        assert_eq!(rows, vec![2.5, 3.5, 4.5]);
        let mut mx = vec![0.0; 3];
        b.reduce(&x, &[2, 3], &mut mx, 0, ReduceKind::Max).unwrap();
        assert_eq!(mx, vec![4.0, 5.0, 6.0]);
        assert!(b.reduce(&x, &[2, 3], &mut rows, 2, ReduceKind::Sum).is_err());
    }

    #[test]
    fn elementwise_is_in_place() {
        let b = NaiveBackend;
        let mut v = vec![-1.0, 0.0, 2.0];
        b.elementwise(&mut v, ElemOp::Relu).unwrap();
        assert_eq!(v, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn only_naive_backend_in_mvp() {
        assert!(select_backend("cpu").is_ok());
        assert!(select_backend("cuda:0").is_err());
    }
}
