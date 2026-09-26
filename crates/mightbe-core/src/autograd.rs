//! 反向模式自动微分（README 4.1）。
//!
//! 设计要点（对应 README 4.1 "自动微分" 段）：
//! - 前向只往 arena 里追加节点，[`Var`] 是 arena 下标（`Copy`、无 `Rc`、无环）；
//! - 节点缓存自己的前向输出，反向时无需回溯重算前向；
//! - [`Graph::backward`] 是纯计算：拓扑排序后逐节点写梯度；
//! - 训练步结束后图随 `Graph` 一起 drop，参数梯度由调用方取走后更新。

use std::sync::Arc;

use crate::api::{Backend, ConvParams, ElemOp, GemmShapes, MtbError, MtbResult, ReduceKind, Shape, Tensor};
use std::fmt;

/// 计算图节点句柄。只是 `usize` 下标，可 `Copy`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Var(pub usize);

/// 常量输入（dropout 掩码、整型 id）的哨兵父节点：反向遇到即跳过。
pub(crate) const CONST_NODE: usize = usize::MAX;

/// [`Op::Permute`] 能装下的最大秩（MVP 覆盖 batch/通道/时间/多头四维；更高阶先 reshape 降阶）。
pub const MAX_PERM_RANK: usize = 4;

/// 算子种类。枚举即"框架支持的全部可微运算"。
///
/// 含 f32 载荷（`Pow`、`Conv1d`），故只实现 `PartialEq`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Identity,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    MatMul,
    Transpose2d,
    Sigmoid,
    Tanh,
    Relu,
    Gelu,
    /// 沿最后一轴做数值稳定 softmax
    SoftmaxLast,
    LogSoftmaxLast,
    SumAxis(usize),
    MeanAxis(usize),
    MaxAxis(usize),
    Reshape,
    Flatten,
    /// 轴置换：`axes[i]` 是输出第 i 轴取自输入的哪一轴
    Permute([usize; MAX_PERM_RANK]),
    Concat(usize),
    /// 沿轴切片 `start..end`，反向把梯度放回原区间
    Slice { axis: usize, start: usize, end: usize },
    /// 一维卷积（权重为独立可训练输入），展开 `ConvParams`
    Conv1d(ConvParams),
    /// 一维最大池化
    MaxPool1d { kernel: usize, stride: usize },
    /// 一维平均池化
    MeanPool1d { kernel: usize, stride: usize },
    /// 幂 `x^p`（p 为常数指数，LayerNorm 的 1/sqrt 用）
    Pow(f32),
    /// Dropout：第 2 个输入是 0/1 掩码常量，反向乘回掩码
    Dropout,
    /// 嵌入查表 `weight[ids]`，反向散回权重行
    Embedding,
    /// 均方损失（标量）
    Mse,
    /// 交叉熵损失（标量），标签为整型输入
    CrossEntropy,
    /// 插件层（README 4.1.1 路径 C）：反向经节点的 [`Node::custom`] 回调，
    /// 枚举本身不携带任何状态。
    Plugin,
}

/// 外部算子（插件）的反向回调。
///
/// 只依赖节点自身缓存的输入/输出与前向值，因此不需要 `&Graph`，
/// 可以和 `local_grads` 一样在倒序遍历中就地求梯度。
pub trait CustomBackward: Send + Sync + fmt::Debug {
    /// 返回各输入的梯度；长度不得超过 `inputs`，空张量表示该槽位无梯度。
    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        out: &Tensor,
    ) -> MtbResult<Vec<Tensor>>;
}

/// 非 f32 输入（如 token id）。
#[derive(Debug, Clone)]
pub enum Input {
    F(Arc<Tensor>),
    I(Vec<u32>),
}

/// 输入引用：父节点下标 + 数据。父节点下标在 push 时确定，反向无需查找。
#[derive(Debug, Clone)]
struct InputRef {
    node: usize,
    data: Input,
}

#[derive(Debug, Clone)]
struct Node {
    op: Op,
    inputs: Vec<InputRef>,
    out: Arc<Tensor>,
    grad: Option<Tensor>,
    /// 可训练参数的名；空串表示常量
    param_name: String,
    trainable: bool,
    /// `Op::Plugin` 节点的反向回调；其余算子恒为 `None`
    custom: Option<Arc<dyn CustomBackward>>,
}

impl Node {
    fn input_f(&self, i: usize) -> MtbResult<&Tensor> {
        match &self.inputs.get(i).ok_or(MtbError::Other("输入越界".into()))?.data {
            Input::F(t) => Ok(t.as_ref()),
            Input::I(_) => Err(MtbError::Other("节点输入不是 f32".into())),
        }
    }

    fn input_i(&self, i: usize) -> MtbResult<&[u32]> {
        match &self.inputs.get(i).ok_or(MtbError::Other("输入越界".into()))?.data {
            Input::I(v) => Ok(v),
            Input::F(_) => Err(MtbError::Other("节点输入不是整型".into())),
        }
    }

    fn parent(&self, i: usize) -> usize {
        self.inputs[i].node
    }
}

// ───────────────────────── 形状工具 ─────────────────────────

/// 行主序多下标计数器。广播路径需要按多下标走访输出张量，
/// 用一次性分配的计数器代替"每元素构造一个下标 Vec"。
struct Odometer {
    shape: Vec<usize>,
    idx: Vec<usize>,
    done: bool,
}

impl Odometer {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            idx: vec![0usize; shape.len()],
            done: shape.is_empty() || shape.iter().any(|&d| d == 0),
        }
    }

    fn cur(&self) -> &[usize] {
        &self.idx
    }

    /// 前进一格；返回 `false` 表示已遍历完最后一个下标。
    fn step(&mut self) -> bool {
        if self.done {
            return false;
        }
        for i in (0..self.shape.len()).rev() {
            self.idx[i] += 1;
            if self.idx[i] < self.shape[i] {
                return true;
            }
            self.idx[i] = 0;
        }
        self.done = true;
        false
    }
}

/// 把"输出张量的多下标"映射到"更窄输入张量的扁平下标"：轴右对齐，`dim == 1` 恒取 0。
fn offset_for(idx: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    let skip = idx.len().saturating_sub(shape.len());
    let mut s = 0usize;
    for (i, &d) in shape.iter().enumerate() {
        s += (idx[skip + i] % d.max(1)) * strides[i];
    }
    s
}

/// 右对齐广播形状：`[2,3]` 与 `[3]` → `[2,3]`，各轴取较大者。
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Shape {
    let rank = a.len().max(b.len());
    let mut out = Shape::new();
    for i in 0..rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        assert!(
            da == db || da == 1 || db == 1,
            "不可广播: {a:?} 与 {b:?}（第 {i} 轴从右起 {da} vs {db}）"
        );
        out.push(da.max(db));
    }
    out.reverse();
    out
}

/// `permute` 的正/反向共用下标换算：输出多下标 → 输入扁平下标。
fn perm_src(in_strides: &[usize], axes: &[usize], idx: &[usize]) -> usize {
    let mut s = 0usize;
    for (i, &ax) in axes.iter().enumerate() {
        s += idx[i] * in_strides[ax];
    }
    s
}

/// 把"最后一轴"语义下的 (rows, cols) 从任意秩抽出来。
fn last_axis_layout(shape: &[usize]) -> (usize, usize) {
    if shape.len() <= 1 {
        (1, shape.first().copied().unwrap_or(1).max(1))
    } else {
        (shape[0], shape[shape.len() - 1])
    }
}

// ───────────────────────── Graph ─────────────────────────

pub struct Graph {
    nodes: Vec<Node>,
    /// 参数名 → 节点下标。同名参数在一个图里只有一个节点，
    /// 因此 RNN 跨时间步、双向两方向共享权重不会退化成"每步一个副本"。
    params: std::collections::HashMap<String, usize>,
    /// 数值核派发目标（README 4.1：所有数值运算统一经 `Backend` trait）。
    backend: std::sync::Arc<dyn Backend>,
}

impl Default for Graph {
    fn default() -> Self {
        Graph::new()
    }
}

impl Graph {
    /// 默认后端为 `naive`；换设备用 [`Graph::with_backend`]。
    pub fn new() -> Self {
        Self::with_backend(crate::backend::NaiveBackend::arc())
    }

    pub fn with_backend(backend: std::sync::Arc<dyn Backend>) -> Self {
        Self {
            nodes: Vec::new(),
            params: std::collections::HashMap::new(),
            backend,
        }
    }

    /// 当前后端 id（`SHOW NETWORKS` 与测试对拍用）。
    pub fn backend_id(&self) -> &str {
        self.backend.id()
    }

    /// 后端内核出错即图构建非法（形状在前向入口已断言过），故直接 panic 暴露 bug。
    fn kernel<T>(&self, r: MtbResult<T>) -> T {
        r.unwrap_or_else(|e| panic!("[{}] 后端内核调用失败: {e}", self.backend.id()))
    }

    /// 节点上的前向值。
    pub fn value(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].out
    }

    /// 节点上的梯度（反向后才有）。
    pub fn grad(&self, v: Var) -> Option<&Tensor> {
        self.nodes[v.0].grad.as_ref()
    }

    /// 节点是否为可训练参数。
    pub fn is_trainable(&self, v: Var) -> bool {
        self.nodes[v.0].trainable
    }

    /// 参数全名（可训练节点才有意义）。
    pub fn param_name(&self, v: Var) -> &str {
        &self.nodes[v.0].param_name
    }

    /// 叶子节点的值（常量 / 参数）。
    pub fn leaf(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].out
    }

    /// 节点值的共享句柄：算子入图时只增引用计数，不复制数据。
    fn shared(&self, v: Var) -> &Arc<Tensor> {
        &self.nodes[v.0].out
    }

    /// 注册一个常量叶（非训练）。
    pub fn constant(&mut self, value: Tensor) -> Var {
        self.push_leaf(String::new(), value, false)
    }

    /// 注册一个可训练参数叶。名字形如 `layer.param`，是权重迁移的锚点。
    /// 同名参数在一个图内只登记一次，第二次调用返回既有节点。
    pub fn param(&mut self, name: impl Into<String>, value: Tensor) -> Var {
        let name = name.into();
        if let Some(&idx) = self.params.get(&name) {
            return Var(idx);
        }
        let var = self.push_leaf(name.clone(), value, true);
        self.params.insert(name, var.0);
        var
    }

    fn push_leaf(&mut self, param_name: String, value: Tensor, trainable: bool) -> Var {
        self.nodes.push(Node {
            op: Op::Identity,
            inputs: vec![],
            out: Arc::new(value),
            grad: None,
            param_name,
            trainable,
            custom: None,
        });
        Var(self.nodes.len() - 1)
    }

    fn push(&mut self, op: Op, inputs: Vec<InputRef>, out: impl Into<Arc<Tensor>>) -> Var {
        self.nodes.push(Node {
            op,
            inputs,
            out: out.into(),
            grad: None,
            param_name: String::new(),
            trainable: false,
            custom: None,
        });
        Var(self.nodes.len() - 1)
    }

    /// 入图一个外部算子节点：前向值由调用方算好，反向经 `cb` 回调（README 4.1.1 路径 C）。
    fn push_custom(
        &mut self,
        inputs: Vec<InputRef>,
        out: Tensor,
        cb: Arc<dyn CustomBackward>,
    ) -> Var {
        self.nodes.push(Node {
            op: Op::Plugin,
            inputs,
            out: Arc::new(out),
            grad: None,
            param_name: String::new(),
            trainable: false,
            custom: Some(cb),
        });
        Var(self.nodes.len() - 1)
    }

    /// 元素级二元算子（含广播）。反向由 `local_grads` 按 `op` 分派，此处只算前向值。
    fn binary<F>(&mut self, a: Var, b: Var, op: Op, f: F) -> Var
    where
        F: Fn(f32, f32) -> f32,
    {
        let ta = self.shared(a).clone();
        let tb = self.shared(b).clone();
        let out_shape = broadcast_shape(&ta.shape, &tb.shape);
        let n = out_shape.iter().product();
        let mut out_data = vec![0f32; n];
        if ta.shape == tb.shape {
            // 最常见路径：同形状逐元素，无需下标换算
            for (o, (x, y)) in out_data.iter_mut().zip(ta.data.iter().zip(tb.data.iter())) {
                *o = f(*x, *y);
            }
        } else {
            let mut od = Odometer::new(&out_shape);
            for oi in 0..n {
                let (oa, ob) = {
                    let idx = od.cur();
                    (
                        offset_for(idx, &ta.shape, &ta.strides),
                        offset_for(idx, &tb.shape, &tb.strides),
                    )
                };
                out_data[oi] = f(ta.data[oa], tb.data[ob]);
                od.step();
            }
        }
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![
            InputRef { node: a.0, data: Input::F(ta) },
            InputRef { node: b.0, data: Input::F(tb) },
        ];
        self.push(op, inputs, out)
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Add, |x, y| x + y)
    }
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Sub, |x, y| x - y)
    }
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Mul, |x, y| x * y)
    }
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Div, |x, y| x / y)
    }

    /// 逐元素函数；新节点的 op 决定反向，前向核走后端派发。
    pub fn unary(&mut self, op: Op, a: Var) -> Var {
        let ta = self.shared(a).clone();
        let elem = match op {
            Op::Neg => ElemOp::Neg,
            Op::Sigmoid => ElemOp::Sigmoid,
            Op::Tanh => ElemOp::Tanh,
            Op::Relu => ElemOp::Relu,
            Op::Gelu => ElemOp::Gelu,
            _ => unreachable!("unary: 算子 {op:?} 不是逐元素运算"),
        };
        let mut out_data = ta.data.clone();
        self.kernel(self.backend.elementwise(&mut out_data, elem));
        let out = Tensor::from_vec(out_data, ta.shape.clone()).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(op, inputs, out)
    }

    pub fn neg(&mut self, a: Var) -> Var {
        self.unary(Op::Neg, a)
    }
    pub fn sigmoid(&mut self, a: Var) -> Var {
        self.unary(Op::Sigmoid, a)
    }
    pub fn tanh(&mut self, a: Var) -> Var {
        self.unary(Op::Tanh, a)
    }
    pub fn relu(&mut self, a: Var) -> Var {
        self.unary(Op::Relu, a)
    }
    pub fn gelu(&mut self, a: Var) -> Var {
        self.unary(Op::Gelu, a)
    }

    /// 数值稳定 softmax（最后一轴；一维输入视为单行）。
    pub fn softmax_last(&mut self, a: Var) -> Var {
        let ta = self.shared(a).clone();
        let (rows, cols) = last_axis_layout(&ta.shape);
        let n = rows * cols;
        let mut out_data = vec![0f32; n];
        for r in 0..rows {
            let base = r * cols;
            let max_v = ta.data[base..base + cols]
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let mut sum = 0f32;
            for i in 0..cols {
                let e = (ta.data[base + i] - max_v).exp();
                out_data[base + i] = e;
                sum += e;
            }
            for i in 0..cols {
                out_data[base + i] /= sum;
            }
        }
        let out = Tensor::from_vec(out_data, ta.shape.clone()).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::SoftmaxLast, inputs, out)
    }

    /// log_softmax（最后一轴），交叉熵的数值稳定实现基础。
    pub fn log_softmax_last(&mut self, a: Var) -> Var {
        let ta = self.shared(a).clone();
        let (rows, cols) = last_axis_layout(&ta.shape);
        let mut out_data = vec![0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let max_v = ta.data[base..base + cols]
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let mut sum = 0f32;
            for i in 0..cols {
                sum += (ta.data[base + i] - max_v).exp();
            }
            let log_norm = max_v + sum.ln();
            for i in 0..cols {
                out_data[base + i] = ta.data[base + i] - log_norm;
            }
        }
        let out = Tensor::from_vec(out_data, ta.shape.clone()).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::LogSoftmaxLast, inputs, out)
    }

    /// 二维矩阵乘 `(m,k) x (k,n) -> (m,n)`，行主序。
    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let ta = self.shared(a).clone();
        let tb = self.shared(b).clone();
        let (m, k) = (ta.shape[0], ta.shape[1]);
        let n = tb.shape[1];
        assert_eq!(tb.shape[0], k, "matmul: 内维不匹配 {} × {:?}", k, tb.shape);
        let mut out_data = vec![0f32; m * n];
        self.kernel(
            self.backend
                .matmul(&ta.data, &tb.data, &mut out_data, &GemmShapes { m, k, n }),
        );
        let out = Tensor::from_vec(out_data, vec![m, n]).unwrap();
        let inputs = vec![
            InputRef { node: a.0, data: Input::F(ta) },
            InputRef { node: b.0, data: Input::F(tb) },
        ];
        self.push(Op::MatMul, inputs, out)
    }

    /// 二维转置 `(r,c) -> (c,r)`。
    pub fn transpose2d(&mut self, a: Var) -> Var {
        let ta = self.shared(a).clone();
        let (r, c) = (ta.shape[0], ta.shape[1]);
        let mut out_data = vec![0f32; ta.data.len()];
        for i in 0..r {
            for j in 0..c {
                out_data[j * r + i] = ta.data[i * c + j];
            }
        }
        let out = Tensor::from_vec(out_data, vec![c, r]).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::Transpose2d, inputs, out)
    }

    /// 沿指定轴规约（sum / mean / max）。
    pub fn reduce_axis(&mut self, op: Op, a: Var, axis: usize) -> Var {
        let ta = self.shared(a).clone();
        if ta.shape.is_empty() {
            // 标量：恒等
            let out = ta.clone();
            let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
            return self.push(op, inputs, out);
        }
        let rank = ta.shape.len();
        let axis = axis.min(rank - 1);
        let outer: usize = ta.shape[..axis].iter().product();
        let inner: usize = ta.shape[axis + 1..].iter().product();
        let kind = match op {
            Op::SumAxis(_) => ReduceKind::Sum,
            Op::MeanAxis(_) => ReduceKind::Mean,
            Op::MaxAxis(_) => ReduceKind::Max,
            other => unreachable!("reduce_axis: 非规约算子 {other:?}"),
        };
        let mut out_data = vec![0f32; outer * inner];
        self.kernel(self.backend.reduce(&ta.data, &ta.shape, &mut out_data, axis, kind));
        let mut out_shape = ta.shape.clone();
        out_shape[axis] = 1;
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(op, inputs, out)
    }

    pub fn sum_axis(&mut self, a: Var, axis: usize) -> Var {
        self.reduce_axis(Op::SumAxis(axis), a, axis)
    }
    pub fn mean_axis(&mut self, a: Var, axis: usize) -> Var {
        self.reduce_axis(Op::MeanAxis(axis), a, axis)
    }
    pub fn max_axis(&mut self, a: Var, axis: usize) -> Var {
        self.reduce_axis(Op::MaxAxis(axis), a, axis)
    }

    /// 形状重排（元素数不变）。
    pub fn reshape(&mut self, a: Var, shape: Shape) -> Var {
        let ta = self.shared(a).clone();
        let out = ta.reshape(shape).expect("reshape: 元素数不匹配");
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::Reshape, inputs, out)
    }

    /// 压平成一维。
    pub fn flatten(&mut self, a: Var) -> Var {
        let ta = self.shared(a).clone();
        let shape = vec![ta.data.len()];
        let out = Tensor::from_vec(ta.data.clone(), shape).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::Flatten, inputs, out)
    }

    /// 轴置换，如 `(batch, seq, ch) -> (batch, ch, seq)` 走 `[0, 2, 1]`。
    pub fn permute(&mut self, a: Var, axes: &[usize]) -> Var {
        let ta = self.shared(a).clone();
        let rank = ta.shape.len();
        assert!(rank <= MAX_PERM_RANK, "permute: 秩 {rank} 超过上限 {MAX_PERM_RANK}");
        assert_eq!(axes.len(), rank, "permute: 轴数组长度须等于秩 {rank}");
        let mut seen = [false; MAX_PERM_RANK];
        for &ax in axes {
            assert!(ax < rank && !seen[ax], "permute: 轴 {ax} 越界或重复");
            seen[ax] = true;
        }
        let mut out_shape = vec![0usize; rank];
        for (i, &ax) in axes.iter().enumerate() {
            out_shape[i] = ta.shape[ax];
        }
        let mut out_data = vec![0f32; ta.data.len()];
        let mut od = Odometer::new(&out_shape);
        for (_oi, o) in out_data.iter_mut().enumerate() {
            let src = perm_src(&ta.strides, axes, od.cur());
            *o = ta.data[src];
            od.step();
        }
        let mut arr = [0usize; MAX_PERM_RANK];
        arr[..rank].copy_from_slice(axes);
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::Permute(arr), inputs, out)
    }

    /// 沿轴拼接，各输入除拼接轴外形状须一致。
    pub fn concat(&mut self, axis: usize, args: &[Var]) -> Var {
        assert!(!args.is_empty(), "concat: 至少需要一个输入");
        let tas: Vec<Arc<Tensor>> = args.iter().map(|&v| self.shared(v).clone()).collect();
        let rank = tas[0].shape.len();
        let axis = axis.min(rank.saturating_sub(1));
        let mut out_shape = tas[0].shape.clone();
        let mut total = 0usize;
        for t in &tas {
            assert_eq!(t.shape.len(), rank, "concat: 秩不一致");
            for (i, d) in t.shape.iter().enumerate() {
                if i != axis {
                    assert_eq!(*d, out_shape[i], "concat: 非拼接轴形状不一致");
                }
            }
            total += t.shape[axis];
        }
        out_shape[axis] = total;
        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = out_shape[axis + 1..].iter().product();
        let mut out_data = vec![0f32; outer * total * inner];
        let mut cursor = 0usize;
        for t in &tas {
            let slice = t.shape[axis];
            let t_inner: usize = t.shape[axis + 1..].iter().product();
            assert_eq!(inner, t_inner, "concat: 尾部维不一致");
            for o in 0..outer {
                for s in 0..slice {
                    let dst = ((o * total) + (cursor + s)) * inner;
                    let src = (o * slice + s) * inner;
                    out_data[dst..dst + inner]
                        .copy_from_slice(&t.data[src..src + inner]);
                }
            }
            cursor += slice;
        }
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = args
            .iter()
            .zip(tas.into_iter())
            .map(|(&v, t)| InputRef { node: v.0, data: Input::F(t) })
            .collect();
        self.push(Op::Concat(axis), inputs, out)
    }

    /// 沿轴切片 `start..end`（RNN 展开逐时刻取输入用）。
    pub fn slice_axis(&mut self, a: Var, axis: usize, start: usize, end: usize) -> Var {
        let ta = self.shared(a).clone();
        let rank = ta.shape.len();
        let axis = axis.min(rank.saturating_sub(1));
        let dim = ta.shape[axis];
        assert!(start <= end && end <= dim, "slice: 区间 [start,end) 越界");
        let outer: usize = ta.shape[..axis].iter().product();
        let inner: usize = ta.shape[axis + 1..].iter().product();
        let mut out_data = Vec::with_capacity((end - start) * outer * inner);
        for o in 0..outer {
            for s in start..end {
                let src = (o * dim + s) * inner;
                out_data.extend_from_slice(&ta.data[src..src + inner]);
            }
        }
        let mut out_shape = ta.shape.clone();
        out_shape[axis] = end - start;
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![InputRef { node: a.0, data: Input::F(ta) }];
        self.push(Op::Slice { axis, start, end }, inputs, out)
    }

    /// 一维卷积：`x` 为 `(in_ch, in_len)` 或 `(batch, in_ch, in_len)`（行主序，通道优先），
    /// 权重 `(filters, in_ch*kernel)` + 偏置 `(filters,)`，按 `ConvParams` 对齐。
    pub fn conv1d(&mut self, x: Var, w: Var, bias: Var, p: ConvParams) -> Var {
        let tx = self.shared(x).clone();
        let tw = self.shared(w).clone();
        let tb = self.shared(bias).clone();
        let (in_ch, in_len, kernel, filters) = (p.in_ch, p.in_len, p.kernel, p.filters);
        let (stride, dilation, padding) = (p.stride.max(1), p.dilation.max(1), p.padding);
        let batched = tx.shape.len() == 3;
        let batch = if batched { tx.shape[0] } else { 1 };
        assert_eq!(
            tx.data.len(),
            batch * in_ch * in_len,
            "conv1d: 输入长度与 (batch, in_ch, in_len) 不符"
        );
        assert_eq!(tw.data.len(), filters * in_ch * kernel, "conv1d: 权重形状不符");
        assert_eq!(tb.data.len(), filters, "conv1d: 偏置长度应为 filters");

        let out_len = (in_len + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
        let shape: Shape =
            if batched { vec![batch, filters, out_len] } else { vec![filters, out_len] };
        // 后端按单样本 (in_ch, in_len) 计算，批维度在此切分
        let mut out_data = vec![0f32; batch * filters * out_len];
        let per_in = in_ch * in_len;
        let per_out = filters * out_len;
        for b in 0..batch {
            let xo = &tx.data[b * per_in..b * per_in + per_in];
            let yo = &mut out_data[b * per_out..b * per_out + per_out];
            self.kernel(self.backend.conv1d(xo, &tw.data, &tb.data, yo, &p));
        }
        let out = Tensor::from_vec(out_data, shape).unwrap();
        let inputs = vec![
            InputRef { node: x.0, data: Input::F(tx) },
            InputRef { node: w.0, data: Input::F(tw) },
            InputRef { node: bias.0, data: Input::F(tb) },
        ];
        self.push(Op::Conv1d(p), inputs, out)
    }

    /// 一维最大池化（最后一轴）。
    pub fn maxpool1d(&mut self, x: Var, kernel: usize, stride: usize) -> Var {
        let tx = self.shared(x).clone();
        let rank = tx.shape.len();
        let axis = rank.saturating_sub(1);
        let dim = tx.shape[axis];
        let kernel = kernel.max(1);
        let stride = stride.max(1);
        let out_len = if dim < kernel { 0 } else { (dim - kernel) / stride + 1 };
        let outer: usize = tx.shape[..axis].iter().product();
        let inner: usize = tx.shape[axis + 1..].iter().product();
        let mut out_data = Vec::with_capacity(out_len * outer * inner);
        for o in 0..outer {
            for s in 0..out_len {
                let base = (o * dim + s * stride) * inner;
                for i in 0..inner {
                    let mut m = f32::NEG_INFINITY;
                    for kk in 0..kernel {
                        m = m.max(tx.data[base + kk * inner + i]);
                    }
                    out_data.push(m);
                }
            }
        }
        let mut out_shape = tx.shape.clone();
        out_shape[axis] = out_len;
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![InputRef { node: x.0, data: Input::F(tx) }];
        self.push(Op::MaxPool1d { kernel, stride }, inputs, out)
    }

    /// 一维平均池化（最后一轴）。
    pub fn meanpool1d(&mut self, x: Var, kernel: usize, stride: usize) -> Var {
        let tx = self.shared(x).clone();
        let rank = tx.shape.len();
        let axis = rank.saturating_sub(1);
        let dim = tx.shape[axis];
        let kernel = kernel.max(1);
        let stride = stride.max(1);
        let out_len = if dim < kernel { 0 } else { (dim - kernel) / stride + 1 };
        let outer: usize = tx.shape[..axis].iter().product();
        let inner: usize = tx.shape[axis + 1..].iter().product();
        let mut out_data = Vec::with_capacity(out_len * outer * inner);
        for o in 0..outer {
            for s in 0..out_len {
                let base = (o * dim + s * stride) * inner;
                for i in 0..inner {
                    let mut m = 0f32;
                    for kk in 0..kernel {
                        m += tx.data[base + kk * inner + i];
                    }
                    out_data.push(m / kernel as f32);
                }
            }
        }
        let mut out_shape = tx.shape.clone();
        out_shape[axis] = out_len;
        let out = Tensor::from_vec(out_data, out_shape).unwrap();
        let inputs = vec![InputRef { node: x.0, data: Input::F(tx) }];
        self.push(Op::MeanPool1d { kernel, stride }, inputs, out)
    }

    /// 幂运算 `x^p`，`p` 为常数（不参与梯度）。
    pub fn pow(&mut self, x: Var, p: f32) -> Var {
        let tx = self.shared(x).clone();
        let out_data: Vec<f32> = tx.data.iter().map(|&v| v.powf(p)).collect();
        let out = Tensor::from_vec(out_data, tx.shape.clone()).unwrap();
        let inputs = vec![InputRef { node: x.0, data: Input::F(tx) }];
        self.push(Op::Pow(p), inputs, out)
    }

    /// Dropout：`mask` 由层在训练模式下生成后作为常量输入传入（已含 1/(1-p) 缩放）。
    /// 掩码不是图的一部分，故第 2 个输入的 `node` 指向自身，反向时跳过。
    pub fn dropout(&mut self, x: Var, mask: &[f32]) -> Var {
        let tx = self.shared(x).clone();
        assert_eq!(mask.len(), tx.data.len(), "dropout: 掩码长度与输入不一致");
        let out_data: Vec<f32> = tx.data.iter().zip(mask.iter()).map(|(a, m)| a * m).collect();
        let out = Tensor::from_vec(out_data, tx.shape.clone()).unwrap();
        let mask_t = Arc::new(Tensor::from_vec(mask.to_vec(), tx.shape.clone()).unwrap());
        let inputs = vec![
            InputRef { node: x.0, data: Input::F(tx) },
            InputRef { node: CONST_NODE, data: Input::F(mask_t) },
        ];
        self.push(Op::Dropout, inputs, out)
    }

    /// 嵌入查表 `weight(vocab, dim)[ids] -> (n, dim)`。
    pub fn embedding(&mut self, weight: Var, ids: &[u32]) -> Var {
        let tw = self.shared(weight).clone();
        let (vocab, dim) = (tw.shape[0], tw.shape[1]);
        assert!(
            ids.iter().all(|&id| (id as usize) < vocab),
            "embedding: token id 超出词表 {vocab}"
        );
        let mut out_data = vec![0f32; ids.len() * dim];
        self.kernel(self.backend.embedding_lookup(&tw.data, ids, &mut out_data, dim));
        let out = Tensor::from_vec(out_data, vec![ids.len(), dim]).unwrap();
        let inputs = vec![
            InputRef { node: weight.0, data: Input::F(tw) },
            InputRef { node: CONST_NODE, data: Input::I(ids.to_vec()) },
        ];
        self.push(Op::Embedding, inputs, out)
    }

    /// 均方损失：返回标量 Var。
    pub fn mse_loss(&mut self, pred: Var, target: Var) -> Var {
        let tp = self.shared(pred).clone();
        let tt = self.shared(target).clone();
        assert_eq!(tp.shape, tt.shape, "mse: 形状不一致");
        let n = tp.data.len().max(1);
        let loss: f32 = tp
            .data
            .iter()
            .zip(tt.data.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        let out = Tensor::from_vec(vec![loss], vec![]).unwrap();
        self.push(
            Op::Mse,
            vec![
                InputRef { node: pred.0, data: Input::F(tp) },
                InputRef { node: target.0, data: Input::F(tt) },
            ],
            out,
        )
    }

    /// 交叉熵损失：logits `(n, C)` + `n` 个类别 id → 标量 Var。
    /// log_softmax 只在前向里就地计算，不进图（反向由 logits 直接重算 softmax）。
    pub fn cross_entropy_loss(&mut self, logits: Var, labels: &[u32]) -> Var {
        let tl = self.shared(logits).clone();
        let (rows, cols) = (tl.shape[0], tl.shape[1]);
        assert_eq!(labels.len(), rows, "cross_entropy: 标签数不匹配");
        let mut loss = 0f32;
        for i in 0..rows {
            let cls = labels[i] as usize;
            assert!(cls < cols, "cross_entropy: 类别下标越界");
            let row = &tl.data[i * cols..i * cols + cols];
            let max_v = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let sum: f32 = row.iter().map(|x| (x - max_v).exp()).sum();
            loss -= row[cls] - max_v - sum.ln();
        }
        let out = Tensor::from_vec(vec![loss / rows.max(1) as f32], vec![]).unwrap();
        self.push(
            Op::CrossEntropy,
            vec![
                InputRef { node: logits.0, data: Input::F(tl) },
                InputRef { node: CONST_NODE, data: Input::I(labels.to_vec()) },
            ],
            out,
        )
    }

    /// 入图一个插件层节点：`out` 是插件前向已算好的值，`cb` 负责反向。
    ///
    /// 输入按调用方给定的顺序与 `cb.backward` 的返回一一对齐，
    /// 因此把可训练参数混排在输入里即可复用既有的梯度累积路径。
    pub fn plugin_op(&mut self, inputs: &[Var], out: Tensor, cb: Arc<dyn CustomBackward>) -> Var {
        let refs = inputs
            .iter()
            .map(|&v| InputRef { node: v.0, data: Input::F(self.shared(v).clone()) })
            .collect();
        self.push_custom(refs, out, cb)
    }

    // ── 反向 ──

    /// 对 `outputs` 做反向，梯度写回各节点。
    pub fn backward(&mut self, outputs: &[Var]) -> MtbResult<()> {
        for &out in outputs {
            let node = &self.nodes[out.0];
            let seed = match &node.grad {
                Some(t) if t.data.len() == 1 => t.data[0],
                _ => 1.0,
            };
            let n = node.out.data.len().max(1);
            let shape = node.out.shape.clone();
            self.nodes[out.0].grad = Some(Tensor::from_vec(vec![seed; n], shape).unwrap());
        }

        // 节点按下标追加，父节点下标恒小于子节点 —— 倒序遍历即拓扑序。
        let n = self.nodes.len();
        for i in (0..n).rev() {
            // 梯度张量按引用传给 local_grads（返回 Owned 后借用即结束），避免每节点一次深拷贝
            let grads = match self.nodes[i].grad.as_ref() {
                Some(g) if !self.nodes[i].inputs.is_empty() => {
                    Self::local_grads(&self.nodes[i], g)?
                }
                _ => continue,
            };
            for (k, grad) in grads.into_iter().enumerate() {
                if grad.is_empty() {
                    continue;
                }
                let pi = self.nodes[i].parent(k);
                match self.nodes[pi].grad.as_mut() {
                    Some(existing) => existing.add_inplace(&grad)?,
                    None => self.nodes[pi].grad = Some(grad),
                }
            }
        }
        Ok(())
    }

    /// 某节点输出的梯度对各个输入的局部梯度。
    ///
    /// 只读节点自己缓存的前向输入/输出，因此不需要 `&Graph`。
    fn local_grads(node: &Node, grad_out: &Tensor) -> MtbResult<Vec<Tensor>> {
        let op = node.op;
        let outs: &[f32] = &grad_out.data;

        // 逐元素 + 广播的通用反向。
        let elem_back = |a: &Tensor, b: &Tensor,
                         fda: fn(f32, f32) -> f32,
                         fdb: fn(f32, f32) -> f32| -> (Tensor, Tensor) {
            let out_shape = broadcast_shape(&a.shape, &b.shape);
            let mut da = vec![0f32; a.data.len()];
            let mut db = vec![0f32; b.data.len()];
            let mut od = Odometer::new(&out_shape);
            for oi in 0..out_shape.iter().product::<usize>() {
                let (oa, ob) = {
                    let idx = od.cur();
                    (
                        offset_for(idx, &a.shape, &a.strides),
                        offset_for(idx, &b.shape, &b.strides),
                    )
                };
                let (va, vb) = (a.data[oa], b.data[ob]);
                let go = outs[oi];
                da[oa] += go * fda(va, vb);
                db[ob] += go * fdb(va, vb);
                od.step();
            }
            (
                Tensor::from_vec(da, a.shape.clone()).unwrap(),
                Tensor::from_vec(db, b.shape.clone()).unwrap(),
            )
        };

        match op {
            Op::Add => {
                let (a, b) = (node.input_f(0)?, node.input_f(1)?);
                let (ga, gb) = elem_back(a, b, |_, _| 1.0, |_, _| 1.0);
                Ok(vec![ga, gb])
            }
            Op::Sub => {
                let (a, b) = (node.input_f(0)?, node.input_f(1)?);
                let (ga, gb) = elem_back(a, b, |_, _| 1.0, |_, _| -1.0);
                Ok(vec![ga, gb])
            }
            Op::Mul => {
                let (a, b) = (node.input_f(0)?, node.input_f(1)?);
                let (ga, gb) = elem_back(a, b, |_, y| y, |x, _| x);
                Ok(vec![ga, gb])
            }
            Op::Div => {
                let (a, b) = (node.input_f(0)?, node.input_f(1)?);
                let (ga, gb) = elem_back(a, b, |_, y| 1.0 / y, |x, y| -x / (y * y));
                Ok(vec![ga, gb])
            }
            Op::Neg => {
                let a = node.input_f(0)?;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    d[j] = -outs[j];
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Sigmoid => {
                let a = node.input_f(0)?;
                let y = &node.out;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    let v = y.data[j];
                    d[j] = outs[j] * v * (1.0 - v);
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Tanh => {
                let a = node.input_f(0)?;
                let y = &node.out;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    d[j] = outs[j] * (1.0 - y.data[j] * y.data[j]);
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Relu => {
                let a = node.input_f(0)?;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    d[j] = if a.data[j] > 0.0 { outs[j] } else { 0.0 };
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Gelu => {
                let a = node.input_f(0)?;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    let x = a.data[j];
                    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
                    let t = inner.tanh();
                    let dt = 0.7978845608028654 * (1.0 + 0.044715 * 3.0 * x * x);
                    d[j] = outs[j] * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dt);
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::SoftmaxLast => {
                let a = node.input_f(0)?;
                let p = &node.out;
                let (rows, cols) = last_axis_layout(&a.shape);
                let mut d = vec![0f32; a.data.len()];
                for r in 0..rows {
                    let base = r * cols;
                    for i in 0..cols {
                        for j in 0..cols {
                            if i == j {
                                d[base + i] +=
                                    outs[base + j] * p.data[base + i] * (1.0 - p.data[base + i]);
                            } else {
                                d[base + i] -= outs[base + j] * p.data[base + i] * p.data[base + j];
                            }
                        }
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::LogSoftmaxLast => {
                let a = node.input_f(0)?;
                let lse = &node.out;
                let (rows, cols) = last_axis_layout(&a.shape);
                let mut d = vec![0f32; a.data.len()];
                for r in 0..rows {
                    let base = r * cols;
                    let mut s = 0f32;
                    for j in 0..cols {
                        s += outs[base + j];
                    }
                    for j in 0..cols {
                        // node.out 存的是 log(p)，反向需要 p 本身
                        let p = lse.data[base + j].exp();
                        d[base + j] = outs[base + j] - s * p;
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::MatMul => {
                let a = node.input_f(0)?;
                let b = node.input_f(1)?;
                let (m, k) = (a.shape[0], a.shape[1]);
                let n = b.shape[1];
                let mut ga = vec![0f32; m * k];
                let mut gb = vec![0f32; k * n];
                for i in 0..m {
                    for j in 0..n {
                        let g = outs[i * n + j];
                        for p in 0..k {
                            ga[i * k + p] += g * b.data[p * n + j];
                            gb[p * n + j] += g * a.data[i * k + p];
                        }
                    }
                }
                Ok(vec![
                    Tensor::from_vec(ga, vec![m, k]).unwrap(),
                    Tensor::from_vec(gb, vec![k, n]).unwrap(),
                ])
            }
            Op::Transpose2d => {
                let a = node.input_f(0)?;
                let (r, c) = (a.shape[0], a.shape[1]);
                let mut d = vec![0f32; a.data.len()];
                // 前向是 t[j][i] = a[i][j]，故梯度按同一置换反向流动：
                // dL/da[i][j] = dL/dt[j][i]（outs 的形状是转置后的 (c,r)）
                for i in 0..r {
                    for j in 0..c {
                        d[i * c + j] = outs[j * r + i];
                    }
                }
                Ok(vec![Tensor::from_vec(d, vec![r, c]).unwrap()])
            }
            Op::SumAxis(axis) | Op::MeanAxis(axis) => {
                let a = node.input_f(0)?;
                let mut d = vec![0f32; a.data.len()];
                if a.shape.is_empty() {
                    return Ok(vec![grad_out.reshape(a.shape.clone()).unwrap()]);
                }
                let rank = a.shape.len();
                let axis = axis.min(rank - 1);
                let dim = a.shape[axis];
                let inner: usize = a.shape[axis + 1..].iter().product();
                for (oi, &o) in outs.iter().enumerate() {
                    let outer_idx = oi / inner.max(1);
                    let inner_idx = oi % inner.max(1);
                    for k in 0..dim {
                        d[(outer_idx * dim + k) * inner + inner_idx] += o;
                    }
                }
                if matches!(op, Op::MeanAxis(_)) && dim > 0 {
                    for x in d.iter_mut() {
                        *x /= dim as f32;
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::MaxAxis(axis) => {
                let a = node.input_f(0)?;
                if a.shape.is_empty() {
                    return Ok(vec![grad_out.reshape(a.shape.clone()).unwrap()]);
                }
                let mut d = vec![0f32; a.data.len()];
                let rank = a.shape.len();
                let axis = axis.min(rank - 1);
                let dim = a.shape[axis];
                let inner: usize = a.shape[axis + 1..].iter().product();
                let outer: usize = a.shape[..axis].iter().product();
                for (o, i) in (0..outer).flat_map(|o| (0..inner).map(move |i| (o, i))) {
                    // 前向保留窗口内首个最大值，反向也只把梯度交给那一个位置
                    let mut best = 0usize;
                    let mut bv = f32::NEG_INFINITY;
                    for k in 0..dim {
                        let v = a.data[(o * dim + k) * inner + i];
                        if v > bv {
                            bv = v;
                            best = k;
                        }
                    }
                    d[(o * dim + best) * inner + i] += outs[o * inner + i];
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Reshape | Op::Flatten | Op::Identity => {
                let a = node.input_f(0)?;
                Ok(vec![grad_out.reshape(a.shape.clone()).unwrap()])
            }
            Op::Permute(axes) => {
                let a = node.input_f(0)?;
                let rank = a.shape.len();
                let ax = &axes[..rank];
                let mut d = vec![0f32; a.data.len()];
                let mut od = Odometer::new(&node.out.shape);
                for &go in outs.iter() {
                    let src = perm_src(&a.strides, ax, od.cur());
                    d[src] += go;
                    od.step();
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Concat(axis) => {
                let out_shape = &node.out.shape;
                let inner: usize = out_shape[axis + 1..].iter().product();
                let total: usize = out_shape[axis];
                let outer: usize = out_shape[..axis].iter().product();
                let mut cursor = 0usize;
                let mut res = Vec::new();
                for k in 0..node.inputs.len() {
                    let t = node.input_f(k)?;
                    let slice = t.shape[axis];
                    let mut g = vec![0f32; t.data.len()];
                    for o in 0..outer {
                        for s in 0..slice {
                            let dst = ((o * total) + (cursor + s)) * inner;
                            let src = (o * slice + s) * inner;
                            g[src..src + inner].copy_from_slice(&outs[dst..dst + inner]);
                        }
                    }
                    res.push(Tensor::from_vec(g, t.shape.clone()).unwrap());
                    cursor += slice;
                }
                Ok(res)
            }
            Op::Slice { axis, start, end } => {
                let a = node.input_f(0)?;
                let rank = a.shape.len();
                let axis = axis.min(rank.saturating_sub(1));
                let dim = a.shape[axis];
                let outer: usize = a.shape[..axis].iter().product();
                let inner: usize = a.shape[axis + 1..].iter().product();
                let mut d = vec![0f32; a.data.len()];
                let mut w = 0usize;
                for o in 0..outer {
                    for s in start..end {
                        let dst = (o * dim + s) * inner;
                        d[dst..dst + inner].copy_from_slice(&outs[w..w + inner]);
                        w += inner;
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Conv1d(p) => {
                let x = node.input_f(0)?;
                let w = node.input_f(1)?;
                let bias = node.input_f(2)?;
                let (in_ch, in_len, kernel, filters) = (p.in_ch, p.in_len, p.kernel, p.filters);
                let (stride, dilation, padding) = (p.stride.max(1), p.dilation.max(1), p.padding);
                let out_len = node.out.shape[node.out.shape.len() - 1];
                let batched = x.shape.len() == 3;
                let batch = if batched { x.shape[0] } else { 1 };
                let win = in_ch * kernel;

                let mut dx = vec![0f32; x.data.len()];
                let mut dw = vec![0f32; w.data.len()];
                let mut db = vec![0f32; bias.data.len()];
                for b in 0..batch {
                    for oc in 0..filters {
                        for o in 0..out_len {
                            let g = outs[(b * filters + oc) * out_len + o];
                            db[oc] += g;
                            let base_in = (o * stride) as isize - padding as isize;
                            for c in 0..in_ch {
                                let xrow = (b * in_ch + c) * in_len;
                                let wrow = oc * win + c * kernel;
                                for kk in 0..kernel {
                                    let i = base_in + kk as isize * dilation as isize;
                                    if i < 0 || i >= in_len as isize {
                                        continue;
                                    }
                                    let xi = xrow + i as usize;
                                    dx[xi] += g * w.data[wrow + kk];
                                    dw[wrow + kk] += g * x.data[xi];
                                }
                            }
                        }
                    }
                }
                Ok(vec![
                    Tensor::from_vec(dx, x.shape.clone()).unwrap(),
                    Tensor::from_vec(dw, w.shape.clone()).unwrap(),
                    Tensor::from_vec(db, bias.shape.clone()).unwrap(),
                ])
            }
            Op::MaxPool1d { kernel, stride } => {
                let a = node.input_f(0)?;
                let rank = a.shape.len();
                let axis = rank.saturating_sub(1);
                let dim = a.shape[axis];
                let outer: usize = a.shape[..axis].iter().product();
                let inner: usize = a.shape[axis + 1..].iter().product();
                let out_len = node.out.shape[axis];
                let mut d = vec![0f32; a.data.len()];
                let mut w = 0usize;
                for o in 0..outer {
                    for s in 0..out_len {
                        let base = (o * dim + s * stride) * inner;
                        for i in 0..inner {
                            // 每个内部下标各有自己的最大值位置
                            let mut best = f32::NEG_INFINITY;
                            let mut best_at = 0usize;
                            for kk in 0..kernel {
                                let at = base + kk * inner + i;
                                let v = a.data[at];
                                if v > best {
                                    best = v;
                                    best_at = at;
                                }
                            }
                            d[best_at] += outs[w + i];
                        }
                        w += inner;
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::MeanPool1d { kernel, stride } => {
                let a = node.input_f(0)?;
                let rank = a.shape.len();
                let axis = rank.saturating_sub(1);
                let dim = a.shape[axis];
                let outer: usize = a.shape[..axis].iter().product();
                let inner: usize = a.shape[axis + 1..].iter().product();
                let out_len = node.out.shape[axis];
                let mut d = vec![0f32; a.data.len()];
                let mut w = 0usize;
                for o in 0..outer {
                    for s in 0..out_len {
                        let base = (o * dim + s * stride) * inner;
                        for kk in 0..kernel {
                            for i in 0..inner {
                                d[base + kk * inner + i] += outs[w + i] / kernel as f32;
                            }
                        }
                        w += inner;
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Pow(p) => {
                let a = node.input_f(0)?;
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    if p == -0.5 {
                        // 特殊化：1/sqrt 的导数，数值上更稳
                        let v = a.data[j].max(1e-12);
                        d[j] = outs[j] * (-0.5) * node.out.data[j] / v;
                    } else {
                        d[j] = outs[j] * p * a.data[j].powf(p - 1.0);
                    }
                }
                Ok(vec![Tensor::from_vec(d, a.shape.clone()).unwrap()])
            }
            Op::Dropout => {
                let a = node.input_f(0)?;
                let mask = node.input_f(1)?;
                let mut d = vec![0f32; a.data.len()];
                for (j, o) in outs.iter().enumerate() {
                    d[j] = o * mask.data[j];
                }
                Ok(vec![
                    Tensor::from_vec(d, a.shape.clone()).unwrap(),
                    // 掩码是常量：返回空张量，反向累加时直接跳过
                    Tensor::from_vec(vec![], vec![0]).unwrap(),
                ])
            }
            Op::Embedding => {
                let w = node.input_f(0)?;
                let ids = node.input_i(1)?;
                let dim = w.shape[1];
                let mut gw = vec![0f32; w.data.len()];
                for (r, &id) in ids.iter().enumerate() {
                    let row = (id as usize) * dim;
                    for i in 0..dim {
                        gw[row + i] += outs[r * dim + i];
                    }
                }
                Ok(vec![Tensor::from_vec(gw, w.shape.clone()).unwrap()])
            }
            Op::Mse => {
                let a = node.input_f(0)?;
                let b = node.input_f(1)?;
                let n = a.data.len().max(1);
                let mut d = vec![0f32; a.data.len()];
                for j in 0..d.len() {
                    d[j] = outs[0] * 2.0 * (a.data[j] - b.data[j]) / n as f32;
                }
                let shape = a.shape.clone();
                let da = Tensor::from_vec(d, shape.clone()).unwrap();
                // 对目标取相反数：目标通常是常量，但复合层里可能是另一条预测分支
                let db = Tensor::from_vec(da.data.iter().map(|x| -x).collect(), shape).unwrap();
                Ok(vec![da, db])
            }
            Op::Plugin => {
                let cb = node.custom.as_ref().ok_or_else(|| {
                    MtbError::Other("Op::Plugin 节点缺少反向回调".into())
                })?;
                let mut ins: Vec<&Tensor> = Vec::with_capacity(node.inputs.len());
                for k in 0..node.inputs.len() {
                    ins.push(node.input_f(k)?);
                }
                let grads = cb.backward(grad_out, &ins, &node.out)?;
                if grads.len() > node.inputs.len() {
                    return Err(MtbError::Shape {
                        expected: format!("{}", node.inputs.len()),
                        got: format!("{}", grads.len()),
                    });
                }
                Ok(grads)
            }
            Op::CrossEntropy => {
                // d/d logits = softmax(logits) - onehot，再除以样本数（前向取的是均值）
                let logits = node.input_f(0)?;
                let labels = node.input_i(1)?;
                let (rows, cols) = (logits.shape[0], logits.shape[1]);
                let mut d = vec![0f32; rows * cols];
                for r in 0..rows {
                    let base = r * cols;
                    let row = &logits.data[base..base + cols];
                    let max_v = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                    let mut sum = 0f32;
                    for j in 0..cols {
                        let e = (row[j] - max_v).exp();
                        d[base + j] = e;
                        sum += e;
                    }
                    for j in 0..cols {
                        d[base + j] /= sum;
                    }
                    d[base + labels[r] as usize] -= 1.0;
                }
                let s = outs[0] / rows.max(1) as f32;
                for x in d.iter_mut() {
                    *x *= s;
                }
                Ok(vec![
                    Tensor::from_vec(d, vec![rows, cols]).unwrap(),
                    Tensor::from_vec(vec![], vec![0]).unwrap(),
                ])
            }
        }
    }

    /// 遍历全部可训练参数：`(节点下标, 参数全名, 参数, 梯度)`。
    /// 本步未被使用的参数梯度为 `None`，由训练循环决定跳过或按零处理。
    pub fn trainable(&self) -> impl Iterator<Item = (usize, &str, &Tensor, Option<&Tensor>)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.trainable)
            .map(|(i, n)| (i, n.param_name.as_str(), n.out.as_ref(), n.grad.as_ref()))
    }

    /// 节点总数（测试与可观测性用）。
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 有限差分梯度检查（README 16.1 要求的测试项）。
    /// `build` 接收图与权重，返回标量损失。
    /// 把任意形状的预测归约成标量损失（有限差分检查需要标量）。
    fn sum_all(g: &mut Graph, v: Var) -> Var {
        let n = g.value(v).data.len();
        let flat = g.reshape(v, vec![n]);
        g.sum_axis(flat, 0)
    }

    fn grad_check<F: Fn(&mut Graph, Var) -> Var>(build: F) {
        let tp = Tensor::from_vec(
            (0..6).map(|i| (i as f32 + 1.0) * 0.13).collect(),
            vec![2, 3],
        )
        .unwrap();
        grad_check_on(&tp, build);
    }

    /// 同 [`grad_check`]，但被检查参数的初值/形状由调用方给出。
    fn grad_check_on<F: Fn(&mut Graph, Var) -> Var>(tp: &Tensor, build: F) {
        let mut g = Graph::new();
        let w = g.param("w", tp.clone());
        let out = build(&mut g, w);
        let loss = sum_all(&mut g, out);
        g.backward(&[loss]).unwrap();
        let analytic = g.grad(w).unwrap().data.clone();

        let mut numerical = tp.data.clone();
        let loss_of = |data: Vec<f32>| -> f32 {
            let mut gg = Graph::new();
            let ww = gg
                .param("w", Tensor::from_vec(data, tp.shape.clone()).unwrap());
            let o = build(&mut gg, ww);
            let l = sum_all(&mut gg, o);
            gg.value(l).scalar().unwrap()
        };
        for i in 0..tp.data.len() {
            let eps = 1e-3f32;
            let mut plus = tp.data.clone();
            plus[i] += eps;
            let mut minus = tp.data.clone();
            minus[i] -= eps;
            numerical[i] = (loss_of(plus) - loss_of(minus)) / (2.0 * eps);
        }
        for i in 0..analytic.len() {
            let (a, n) = (analytic[i], numerical[i]);
            // 相对容差为主，零附近用绝对下限（f32 中心差分的舍入误差量级）
            let tol = 2e-2f32 * a.abs().max(n.abs()).max(5e-2);
            assert!(
                (a - n).abs() < tol,
                "梯度检查失败: idx={i} analytic={a} numerical={n}"
            );
        }
    }

    #[test]
    fn matmul_grad_check() {
        grad_check(|g, w| {
            let x = g.constant(Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2, 2],
            ).unwrap());
            let h = g.matmul(x, w);
            g.sum_axis(h, 0)
        });
    }

    #[test]
    fn broadcast_bias_grad_check() {
        grad_check(|g, w| {
            let x = g.constant(Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
            ).unwrap());
            // 取 w 的第一行当偏置：广播反向 + 切片反向 + reshape 反向一起验
            let row0 = g.slice_axis(w, 0, 0, 1);
            let bias = g.reshape(row0, vec![3]);
            let h = g.add(x, bias);
            g.sum_axis(h, 1)
        });
    }

    #[test]
    fn activation_grad_check() {
        grad_check(|g, w| {
            let x = g.constant(Tensor::from_vec(
                vec![0.5, -1.5, 2.0, 1.0],
                vec![2, 2],
            ).unwrap());
            let h = g.matmul(x, w);
            let a = g.gelu(h);
            g.sum_axis(a, 1)
        });
    }

    #[test]
    fn cross_entropy_is_numerically_stable() {
        let mut g = Graph::new();
        let logits = g.constant(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 100.0, 100.0, 100.0],
            vec![2, 3],
        ).unwrap());
        let loss = g.cross_entropy_loss(logits, &[0, 2]);
        let v = g.value(loss).scalar().unwrap();
        assert!(v.is_finite() && v > 0.0, "loss={v}");
        g.backward(&[loss]).unwrap();
        let grad = g.grad(logits).unwrap();
        assert_eq!(grad.shape, vec![2, 3]);
        assert!(grad.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut g = Graph::new();
        let a = g.constant(Tensor::from_vec(
            vec![0.5, -1.0, 2.0, 10.0, 0.0, 0.0],
            vec![2, 3],
        ).unwrap());
        let s = g.softmax_last(a);
        let data = g.value(s).data.clone();
        for r in 0..2 {
            let sum: f32 = data[r * 3..r * 3 + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sum={sum}");
        }
    }

    #[test]
    fn embedding_grad_scatters_to_rows() {
        let mut g = Graph::new();
        let w = g.param("w", Tensor::from_vec(
            (0..8).map(|i| (i + 1) as f32).collect(),
            vec![4, 2],
        ).unwrap());
        let emb = g.embedding(w, &[0, 2, 3]);
        let loss = g.sum_axis(emb, 1);
        g.backward(&[loss]).unwrap();
        let gw = g.grad(w).unwrap().data.clone();
        for j in [0usize, 2, 3] {
            let row_sum: f32 = (0..2).map(|i| gw[j * 2 + i]).sum();
            assert!((row_sum - 2.0).abs() < 1e-5, "row {j} grad={row_sum}");
        }
        let row1 = &gw[2..4];
        assert!(row1.iter().all(|x| x.abs() < 1e-9), "未命中的行梯度应为 0");
    }

    #[test]
    fn concat_grad_is_split_back() {
        let mut g = Graph::new();
        let a = g.constant(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap());
        let b = g.constant(Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]).unwrap());
        let c = g.concat(0, &[a, b]);
        let loss = g.sum_axis(c, 0);
        g.backward(&[loss]).unwrap();
        assert_eq!(g.value(a).data, vec![1.0, 2.0]);
        assert_eq!(g.grad(a).unwrap().data, vec![1.0, 1.0]);
        assert_eq!(g.grad(b).unwrap().data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn mean_axis_grad_divides_by_dim() {
        let mut g = Graph::new();
        let a = g.constant(Tensor::from_vec((0..6).map(|i| i as f32).collect(), vec![2, 3]).unwrap());
        let m = g.mean_axis(a, 1);
        g.backward(&[m]).unwrap();
        let d = g.grad(a).unwrap().data.clone();
        for x in d {
            assert!((x - 1.0 / 3.0).abs() < 1e-6, "mean 梯度应为 1/dim，得到 {x}");
        }
    }

    // ─────────────── README 16.1：其余算子的梯度检查 ───────────────

    fn conv_params() -> ConvParams {
        ConvParams { in_len: 5, in_ch: 2, kernel: 3, filters: 2, stride: 1, padding: 1, dilation: 1 }
    }

    #[test]
    fn conv1d_weight_grad_check() {
        let w = Tensor::from_vec(
            (0..12).map(|i| (i as f32 - 5.0) * 0.21).collect(),
            vec![2, 2, 3],
        )
        .unwrap();
        grad_check_on(&w, |g, wp| {
            let x = g.constant(Tensor::from_vec(
                (0..10).map(|i| (i as f32) * 0.31 - 1.2).collect(),
                vec![2, 5],
            ).unwrap());
            let b = g.constant(Tensor::zeros(vec![2]));
            g.conv1d(x, wp, b, conv_params())
        });
    }

    #[test]
    fn conv1d_bias_grad_check() {
        let b = Tensor::from_vec(vec![0.7, -0.3], vec![2]).unwrap();
        grad_check_on(&b, |g, bp| {
            let x = g.constant(Tensor::from_vec(
                (0..10).map(|i| (i as f32) * 0.31 - 1.2).collect(),
                vec![1, 2, 5],
            ).unwrap());
            let w = g.constant(Tensor::from_vec(
                (0..12).map(|i| (i as f32 - 5.0) * 0.21).collect(),
                vec![2, 2, 3],
            ).unwrap());
            g.conv1d(x, w, bp, conv_params())
        });
    }

    #[test]
    fn maxpool1d_grad_check() {
        // 每窗口内取值互不相同，避免"并列最大值"造成的不可导点
        let w = Tensor::from_vec(vec![1.0, 4.0, 2.0, 3.0, 6.0, 5.0], vec![2, 3]).unwrap();
        grad_check_on(&w, |g, xp| g.maxpool1d(xp, 2, 2));
    }

    #[test]
    fn meanpool1d_grad_check() {
        let w = Tensor::from_vec((0..8).map(|i| i as f32 * 0.4 - 1.0).collect(), vec![2, 4])
            .unwrap();
        grad_check_on(&w, |g, xp| g.meanpool1d(xp, 3, 1));
    }

    #[test]
    fn dropout_grad_check_uses_given_mask() {
        let w = Tensor::from_vec(vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0], vec![2, 3]).unwrap();
        grad_check_on(&w, |g, xp| g.dropout(xp, &[2.0, 0.0, 2.0, 2.0, 0.0, 2.0]));
    }

    #[test]
    fn elementwise_binary_grad_check() {
        grad_check(|g, w| {
            let x = g.constant(Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3]).unwrap());
            let neg = g.sub(x, w);
            let scaled = g.mul(neg, x);
            g.div(scaled, x)
        });
    }

    #[test]
    fn transpose2d_grad_check() {
        grad_check(|g, w| {
            let t = g.transpose2d(w);
            let h = g.matmul(w, t);
            g.sum_axis(h, 1)
        });
    }

    #[test]
    fn slice_axis_grad_check() {
        grad_check(|g, w| {
            let s = g.slice_axis(w, 1, 1, 3);
            g.add(s, s)
        });
    }

    #[test]
    fn pow_grad_check() {
        let w = Tensor::from_vec(vec![0.6, 1.1, 1.7, 2.3, 0.9, 1.4], vec![2, 3]).unwrap();
        grad_check_on(&w, |g, xp| {
            let p = g.pow(xp, 1.7);
            let s = g.sum_axis(p, 1);
            g.pow(s, 0.5)
        });
    }

    #[test]
    fn relu_and_max_axis_grad_check() {
        // 有正有负才真正打到 relu 的折点两侧；行内取值互异保证 max 唯一
        let w = Tensor::from_vec(vec![-1.4, 0.8, 2.1, 0.3, -0.6, 1.2], vec![2, 3]).unwrap();
        grad_check_on(&w, |g, xp| {
            let r = g.relu(xp);
            let m = g.max_axis(r, 1);
            let s = g.sum_axis(r, 1);
            g.add(m, s)
        });
    }

    #[test]
    fn log_softmax_grad_check() {
        grad_check(|g, w| g.log_softmax_last(w));
    }

    #[test]
    fn mse_loss_grad_check() {
        grad_check(|g, w| {
            let target = g.constant(Tensor::from_vec(
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                vec![2, 3],
            ).unwrap());
            g.mse_loss(w, target)
        });
    }

    #[test]
    fn sigmoid_and_sums_grad_check() {
        grad_check(|g, w| {
            let s = g.sigmoid(w);
            let byrow = g.sum_axis(s, 1);
            let bycol = g.sum_axis(s, 0);
            let flat = g.reshape(bycol, vec![3]);
            g.add(byrow, flat)
        });
    }

    #[test]
    fn permute_grad_check() {
        let w = Tensor::from_vec((0..24).map(|i| i as f32 * 0.13 - 1.1).collect(), vec![2, 3, 4])
            .unwrap();
        grad_check_on(&w, |g, xp| {
            let p = g.permute(xp, &[0, 2, 1]);
            let t = g.max_axis(p, 2);
            g.mean_axis(t, 0)
        });
    }

    #[test]
    fn permute_roundtrip_is_value_preserving() {
        let mut g = Graph::new();
        let a = g.constant(
            Tensor::from_vec((0..6).map(|i| i as f32).collect(), vec![2, 3]).unwrap(),
        );
        let p = g.permute(a, &[1, 0]);
        let back = g.permute(p, &[1, 0]);
        assert_eq!(g.value(a).data, g.value(back).data);
        assert_eq!(g.value(p).shape, vec![3, 2]);
    }

    #[test]
    #[should_panic(expected = "越界或重复")]
    fn permute_rejects_bad_axes() {
        let mut g = Graph::new();
        let a = g.constant(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        g.permute(a, &[0, 0]);
    }

    /// LayerNorm 的算子链（mean/sub/pow/倒数开方）整体过一遍有限差分。
    #[test]
    fn layernorm_chain_grad_check() {
        grad_check(|g, w| {
            let mu = g.mean_axis(w, 1);
            let d = g.sub(w, mu);
            let d2 = g.mul(d, d);
            let var = g.mean_axis(d2, 1);
            let eps = g.constant(Tensor::from_vec(vec![0.5], vec![1, 1]).unwrap());
            let shifted = g.add(var, eps);
            let inv = g.pow(shifted, -0.5);
            g.mul(d, inv)
        });
    }

    // ─────────────── 反向语义的定点回归（曾经出错的三处） ───────────────

    #[test]
    fn max_axis_grads_route_only_to_argmax() {
        let mut g = Graph::new();
        let a = g.constant(
            Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 0.0, 4.0], vec![2, 3]).unwrap(),
        );
        let m = g.max_axis(a, 1);
        g.backward(&[m]).unwrap();
        assert_eq!(
            g.grad(a).unwrap().data,
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "max 的梯度只能落到每行的最大值位置"
        );
    }

    #[test]
    fn log_softmax_total_grad_is_zero() {
        // 每行 log_softmax 之和是常数（Σ 梯度 = Σ(1 - p) = cols - 1 ... 只对全 1 上游），
        // 真正的不变量是：行内梯度之和为 0（输出对输入平移不变）。
        let mut g = Graph::new();
        let a = g.constant(
            Tensor::from_vec(vec![0.4, -1.2, 2.1, 0.3, -0.6, 1.2], vec![2, 3]).unwrap(),
        );
        let l = g.log_softmax_last(a);
        g.backward(&[l]).unwrap();
        let d = &g.grad(a).unwrap().data;
        for r in 0..2 {
            let s: f32 = d[r * 3..r * 3 + 3].iter().sum();
            assert!(s.abs() < 1e-5, "行 {r} 梯度之和应为 0，得到 {s}");
        }
    }

    #[test]
    fn transpose_grad_follows_same_permutation() {
        let mut g = Graph::new();
        let a = g.constant(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
        );
        let t = g.transpose2d(a);
        g.backward(&[t]).unwrap();
        assert_eq!(
            g.grad(a).unwrap().data,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "转置的反向是同一置换，梯度应逐元素回到原位"
        );
    }

    #[test]
    #[should_panic(expected = "不可广播")]
    fn incompatible_broadcast_is_rejected() {
        let mut g = Graph::new();
        let a = g.constant(Tensor::from_vec(vec![1.0; 6], vec![2, 3]).unwrap());
        let b = g.constant(Tensor::from_vec(vec![1.0; 2], vec![2]).unwrap());
        g.add(a, b);
    }

    /// 计数包装后端：用来证明 Graph 的数值核真的派发到所配置的 `Backend`
    /// （README 4.1 "所有数值运算都经统一 Backend trait 派发"）。
    struct CountingBackend {
        matmul: std::sync::atomic::AtomicUsize,
        conv1d: std::sync::atomic::AtomicUsize,
        elementwise: std::sync::atomic::AtomicUsize,
        reduce: std::sync::atomic::AtomicUsize,
        embedding: std::sync::atomic::AtomicUsize,
    }

    impl CountingBackend {
        fn new() -> Self {
            Self {
                matmul: Default::default(),
                conv1d: Default::default(),
                elementwise: Default::default(),
                reduce: Default::default(),
                embedding: Default::default(),
            }
        }

        fn bump(c: &std::sync::atomic::AtomicUsize) {
            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn count(c: &std::sync::atomic::AtomicUsize) -> usize {
            c.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl Backend for CountingBackend {
        fn id(&self) -> &str {
            "counting"
        }
        fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], g: &GemmShapes) -> MtbResult<()> {
            Self::bump(&self.matmul);
            crate::backend::NaiveBackend.matmul(a, b, out, g)
        }
        fn conv1d(
            &self,
            x: &[f32],
            w: &[f32],
            bias: &[f32],
            out: &mut [f32],
            p: &ConvParams,
        ) -> MtbResult<()> {
            Self::bump(&self.conv1d);
            crate::backend::NaiveBackend.conv1d(x, w, bias, out, p)
        }
        fn elementwise(&self, out: &mut [f32], op: ElemOp) -> MtbResult<()> {
            Self::bump(&self.elementwise);
            crate::backend::NaiveBackend.elementwise(out, op)
        }
        fn reduce(
            &self,
            x: &[f32],
            shape: &[usize],
            out: &mut [f32],
            axis: usize,
            r: ReduceKind,
        ) -> MtbResult<()> {
            Self::bump(&self.reduce);
            crate::backend::NaiveBackend.reduce(x, shape, out, axis, r)
        }
        fn embedding_lookup(
            &self,
            emb: &[f32],
            ids: &[u32],
            out: &mut [f32],
            dim: usize,
        ) -> MtbResult<()> {
            Self::bump(&self.embedding);
            crate::backend::NaiveBackend.embedding_lookup(emb, ids, out, dim)
        }
    }

    #[test]
    fn graph_dispatches_all_kernels_through_backend() {
        let counted = std::sync::Arc::new(CountingBackend::new());
        let mut g = Graph::with_backend(counted.clone());
        assert_eq!(g.backend_id(), "counting");

        let w = g.param("e.weight", Tensor::from_vec((0..8).map(|i| i as f32 * 0.1).collect(), vec![4, 2]).unwrap());
        let ids = [0u32, 2, 3, 1];
        let emb = g.embedding(w, &ids);
        let act = g.gelu(emb);
        let proj = g.param(
            "p.weight",
            Tensor::from_vec((0..4).map(|i| i as f32 * 0.25).collect(), vec![2, 2]).unwrap(),
        );
        let mm = g.matmul(act, proj);
        let cw = g.param("c.weight", Tensor::from_vec((0..6).map(|i| i as f32 * 0.1).collect(), vec![1, 2, 3]).unwrap());
        let cb = g.constant(Tensor::zeros(vec![1]));
        let cv = g.conv1d(mm, cw, cb, ConvParams { in_len: 4, in_ch: 2, kernel: 3, filters: 1, stride: 1, padding: 1, dilation: 1 });
        let reduced = g.mean_axis(cv, 2);
        let loss = g.sum_axis(reduced, 0);
        g.backward(&[loss]).unwrap();

        for (name, count) in [
            ("matmul", CountingBackend::count(&counted.matmul)),
            ("conv1d", CountingBackend::count(&counted.conv1d)),
            ("elementwise", CountingBackend::count(&counted.elementwise)),
            ("reduce", CountingBackend::count(&counted.reduce)),
            ("embedding", CountingBackend::count(&counted.embedding)),
        ] {
            assert!(count >= 1, "{name} 未经过后端派发（count={count}）");
        }
        assert!(CountingBackend::count(&counted.embedding) == 1);
    }
}
