//! 内置层库（README 4.1 表）。
//!
//! 全部从零实现、CPU、可微：层的推进一律通过 `crate::autograd::Graph` 的算子，
//! 因此**任何内置层都不需要自己写反向传播**，与复合层/插件层共用同一接口。
//!
//! 形状约定：
//! - `dense` / `layernorm` / `dropout` / `activation`：`(batch, features)`
//! - `embedding`：一维 token id 序列 → `(n, dim)`
//! - `rnn` / `gru` / `lstm`：`(batch, seq, features)`
//! - `conv1d` / `*pool1d`：`(batch, channels, len)`

use crate::api::{
    Layer, LayerCtx, LayerSpec, MtbError, MtbResult, Shape, Tensor,
};
use crate::autograd::{Graph, Var};
use crate::dsl::{CellSpec, ParamKind};
use crate::init::{rng_from, Init};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

/// 把张量里的数值当作 token id。
pub fn as_ids(t: &Tensor) -> Vec<u32> {
    t.data.iter().map(|x| *x as u32).collect()
}

fn spec(kind: &str, name: &str, json: &str) -> LayerSpec {
    LayerSpec::new(kind, name, json)
}

// ───────────────────────── 公共基类字段 ─────────────────────────

/// 层内统一的参数槽：`(名字, 张量)`。
pub type Params = Vec<(String, Tensor)>;

fn push_param(layer: &str, out: &mut Params, n: &str, t: Tensor) {
    out.push((format!("{layer}.{n}"), t));
}

/// `bind_params` 的键既接受短名 `weight` 也接受检查点里的全名 `dense.weight`。
/// 约定（README 4.1）：`dump_params` 输出全名，外部迁移按全名回灌。
fn short_key<'a>(layer: &str, key: &'a str) -> &'a str {
    key.strip_prefix(&format!("{layer}.")).unwrap_or(key)
}

/// `(batch, …)` → `(batch, 其余维乘积)`：接 dense 前的标准压平形状。
fn flat_2d(shape: &[usize]) -> Shape {
    let b = shape.first().copied().unwrap_or(1);
    vec![b, shape.iter().skip(1).product()]
}

// ───────────────────────── Dense ─────────────────────────

/// `y = activation(W·x + b)`，形状 `(m, in) -> (m, out)`。
pub struct Dense {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Option<&'static str>,
    pub weight: Tensor,
    pub bias: Tensor,
    pub init: Init,
    pub seed: u64,
}

impl Dense {
    pub fn new(
        name: &str,
        in_dim: usize,
        out_dim: usize,
        activation: Option<&'static str>,
        init: Init,
        seed: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            in_dim,
            out_dim,
            activation,
            init,
            seed,
            weight: init.make(&[in_dim, out_dim], seed),
            bias: Tensor::zeros(vec![out_dim]),
        }
    }

    /// 换用另一套初始化策略（配置 `init = "..."` 覆盖默认值时调用）。
    pub fn reinit(&mut self) {
        self.weight = self.init.make(&[self.in_dim, self.out_dim], self.seed);
        self.bias = Tensor::zeros(vec![self.out_dim]);
    }
}

impl Layer for Dense {
    fn name(&self) -> &str {
        &self.name
    }

    fn param_names(&self) -> Vec<String> {
        vec!["weight".into(), "bias".into()]
    }

    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()> {
        for (n, t) in params {
            match short_key(&self.name, n) {
                "weight" => self.weight = t.clone(),
                "bias" => self.bias = t.clone(),
                _ => {}
            }
        }
        Ok(())
    }

    fn dump_params(&self) -> Params {
        vec![
            (format!("{}.weight", self.name), self.weight.clone()),
            (format!("{}.bias", self.name), self.bias.clone()),
        ]
    }

    fn forward(
        &mut self,
        args: &[&Var],
        _ctx: &LayerCtx,
        graph: &mut Graph,
    ) -> MtbResult<Vec<Var>> {
        let x = *args[0];
        let w = graph.param(format!("{}.weight", self.name), self.weight.clone());
        let b = graph.param(format!("{}.bias", self.name), self.bias.clone());
        let h = graph.matmul(x, w);
        let y = graph.add(h, b);
        let out = match self.activation {
            Some("relu") => graph.relu(y),
            Some("sigmoid") => graph.sigmoid(y),
            Some("tanh") => graph.tanh(y),
            Some("gelu") => graph.gelu(y),
            Some(other) => {
                return Err(MtbError::Config(format!(
                    "dense: 未知激活 {other:?}（可用 relu/sigmoid/tanh/gelu）"
                )))
            }
            None => y,
        };
        Ok(vec![out])
    }

    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let m = inputs[0][0];
        Ok(vec![vec![m, self.out_dim]])
    }

    fn describe(&self) -> LayerSpec {
        spec(
            "dense",
            &self.name,
            &format!("{{\"in\":{},\"out\":{}}}", self.in_dim, self.out_dim),
        )
    }
}

// ───────────────────────── Embedding ─────────────────────────

/// 查表层：参数即词向量矩阵（README 4.1 说"联想能力的来源"）。
pub struct Embedding {
    pub name: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub weight: Tensor,
    pub trainable: bool,
}

impl Embedding {
    pub fn new(name: &str, vocab_size: usize, dim: usize, seed: u64) -> Self {
        Self {
            name: name.to_string(),
            vocab_size,
            dim,
            weight: Init::Xavier.make(&[vocab_size, dim], seed),
            trainable: true,
        }
    }
}

impl Layer for Embedding {
    fn name(&self) -> &str {
        &self.name
    }
    fn param_names(&self) -> Vec<String> {
        vec!["weight".into()]
    }
    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()> {
        for (n, t) in params {
            if short_key(&self.name, n) == "weight" {
                self.weight = t.clone();
            }
        }
        Ok(())
    }
    fn dump_params(&self) -> Params {
        vec![(format!("{}.weight", self.name), self.weight.clone())]
    }
    fn forward(&mut self, args: &[&Var], _ctx: &LayerCtx, graph: &mut Graph) -> MtbResult<Vec<Var>> {
        let ids_var = *args[0];
        let ids_shape = graph.value(ids_var).shape.clone();
        let ids = as_ids(graph.value(ids_var));
        let w = graph.param(format!("{}.weight", self.name), self.weight.clone());
        let flat = graph.embedding(w, &ids);
        // 输入是 (batch, seq) 时还原成 (batch, seq, dim)，供 conv1d 前置使用
        Ok(vec![match ids_shape.as_slice() {
            [b, s] => graph.reshape(flat, vec![*b, *s, self.dim]),
            _ => flat,
        }])
    }
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        Ok(vec![match inputs[0].as_slice() {
            [b, s] => vec![*b, *s, self.dim],
            _ => vec![inputs[0][0], self.dim],
        }])
    }
    fn describe(&self) -> LayerSpec {
        spec("embedding", &self.name, &format!("{{\"dim\":{}}}", self.dim))
    }
}

// ───────────────────────── 循环层 ─────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Simple,
    Gru,
    Lstm,
    /// 由 Cell DSL 的方程展开而来（见 [`crate::dsl`]，构造走 [`Rnn::custom`]）
    Custom,
}

/// 循环层：`(batch, seq, features) -> (batch, seq, units)` 或末时刻隐状态。
#[derive(Debug)]
pub struct Rnn {
    pub name: String,
    pub input_dim: usize,
    pub units: usize,
    pub cell: Cell,
    pub bidirectional: bool,
    pub return_sequences: bool,
    pub seed: u64,
    pub params: Params,
    /// `Cell::Custom` 的方程组；内置细胞为 None。
    spec: Option<Arc<CellSpec>>,
    /// 装载期定形好的 DSL 参数表 `(参数名, 角色)`，双向时按 `<名>r` 再复制一套。
    cell_params: Vec<(String, ParamKind)>,
}

impl Rnn {
    /// 内置细胞（simple / gru / lstm）。`Cell::Custom` 没有方程可用，
    /// 必须由 [`Rnn::custom`] 构造——误用会在前向时明确报错，不会静默按 GRU 跑。
    pub fn new(name: &str, input_dim: usize, units: usize, cell: Cell, seed: u64) -> Self {
        let mut rnn = Self {
            name: name.to_string(),
            input_dim,
            units,
            cell,
            bidirectional: false,
            return_sequences: true,
            seed,
            params: Vec::new(),
            spec: None,
            cell_params: Vec::new(),
        };
        rnn.allocate();
        rnn
    }

    /// DSL 自定义细胞（README 4.1.1 路径 B）：`equations` 在网络装载期解析，
    /// 语法或形状错误直接以 1xxx 错误码返回，不留下半成品层。
    pub fn custom(
        name: &str,
        input_dim: usize,
        units: usize,
        equations: &str,
        seed: u64,
    ) -> MtbResult<Self> {
        let spec = CellSpec::parse(equations)?;
        let cell_params = spec.params(input_dim, units)?;
        let mut rnn = Self {
            name: name.to_string(),
            input_dim,
            units,
            cell: Cell::Custom,
            bidirectional: false,
            return_sequences: true,
            seed,
            params: Vec::new(),
            spec: Some(Arc::new(spec)),
            cell_params,
        };
        rnn.allocate();
        Ok(rnn)
    }

    /// 双向：反向方向另有一套权重（`<layer>.w0r` 等），二者独立训练。
    pub fn bidirectional(mut self) -> Self {
        if !self.bidirectional {
            self.bidirectional = true;
            self.allocate();
        }
        self
    }

    /// 只输出末时刻隐状态。
    pub fn last_state(mut self) -> Self {
        self.return_sequences = false;
        self
    }

    /// 内置细胞把同方向所有门合并为一次 matmul（README 4.1 表）：simple 1、GRU 3、LSTM 4。
    /// DSL 细胞每门各有参数、不合并，故为 0。
    pub fn gates(&self) -> usize {
        match self.cell {
            Cell::Simple => 1,
            Cell::Gru => 3,
            Cell::Lstm => 4,
            Cell::Custom => 0,
        }
    }

    /// 分配（含双向化时重新分配）参数槽。两种细胞都按 `<层名>.<参数名>` 存键，
    /// 因此 checkpoint / 热更新 / 权重迁移不需要为 DSL 层特判。
    fn allocate(&mut self) {
        let u = self.units;
        let d = self.input_dim;
        let init = Init::Xavier;
        self.params.clear();
        for dir in 0..(1 + self.bidirectional as usize) {
            let suffix = if dir == 0 { "" } else { "r" };
            let base = self.seed + (dir as u64) * 1000;
            if self.cell == Cell::Custom {
                for (i, (n, kind)) in self.cell_params.iter().enumerate() {
                    let shape = CellSpec::param_shape(u, *kind);
                    let t = init.make(&shape, base + 3 * (i as u64));
                    push_param(&self.name, &mut self.params, &format!("{n}{suffix}"), t);
                }
                continue;
            }
            // w: (in, gates*u)、u0: (u, gates*u)、b0: (gates*u)
            let wide = self.gates() * u;
            let w = init.make(&[d, wide], base);
            let uh = init.make(&[u, wide], base + 1);
            let b = Tensor::zeros(vec![wide]);
            push_param(&self.name, &mut self.params, &format!("w0{suffix}"), w);
            push_param(&self.name, &mut self.params, &format!("u0{suffix}"), uh);
            push_param(&self.name, &mut self.params, &format!("b0{suffix}"), b);
        }
    }

    /// 参数全名；`rev` 选择反向方向那套权重。
    fn key(&self, n: &str, rev: bool) -> String {
        format!("{}.{}{}", self.name, n, if rev { "r" } else { "" })
    }

    fn p(&self, n: &str, rev: bool) -> Tensor {
        let key = self.key(n, rev);
        self.params
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| Tensor::zeros(vec![]))
    }

    /// 单个时间步：内置细胞把门预激活合成一次 `x·W + h·U + b` 再沿特征轴切门；
    /// DSL 细胞把方程展开成同一批图算子。
    ///
    /// 返回 `(h, c)`；非 LSTM 细胞的 `c` 恒等于 `h`，只为让 `forward` 统一处理状态。
    fn step(
        &self,
        x_t: Var,
        prev: Option<(Var, Var)>,
        rev: bool,
        g: &mut Graph,
    ) -> MtbResult<(Var, Var)> {
        let u = self.units;
        let batch = g.value(x_t).shape.first().copied().unwrap_or(1);
        let (h_prev, c_prev) = match prev {
            Some(p) => p,
            None => {
                let h = g.constant(Tensor::zeros(vec![batch, u]));
                let c = g.constant(Tensor::zeros(vec![batch, u]));
                (h, c)
            }
        };
        if let Some(spec) = &self.spec {
            // 图内键用带方向后缀的全名：跨时间步、跨方向复用同一节点（README 4.1 参数寻址）
            let look = |n: &str| (self.key(n, rev), self.p(n, rev));
            return spec.step(g, x_t, h_prev, c_prev, look);
        }
        if self.cell == Cell::Custom {
            return Err(MtbError::Config(format!(
                "rnn 层 {} 是 custom 细胞但没有 equations，请用 Rnn::custom 构造",
                self.name
            )));
        }
        let w = g.param(self.key("w0", rev), self.p("w0", rev));
        let uh = g.param(self.key("u0", rev), self.p("u0", rev));
        let b = g.param(self.key("b0", rev), self.p("b0", rev));
        let xw = g.matmul(x_t, w);
        let hh = g.matmul(h_prev, uh);
        let xwh = g.add(xw, hh);
        let pre = g.add(xwh, b);
        let gates: Vec<Var> = (0..self.gates())
            .map(|i| g.slice_axis(pre, 1, i * u, (i + 1) * u))
            .collect();

        Ok(match self.cell {
            Cell::Simple => {
                let h = g.tanh(gates[0]);
                (h, h)
            }
            Cell::Gru => {
                // 两门 GRU：z 更新门、r 重置门。重置门只作用于历史项，
                // 所以候选态取 tanh(W_c·x + B_c + r ⊙ (U_c·h_prev))，与 DSL 手写 GRU 同式。
                let z = g.sigmoid(gates[0]);
                let r = g.sigmoid(gates[1]);
                let cx = g.slice_axis(xw, 1, 2 * u, 3 * u);
                let ch = g.slice_axis(hh, 1, 2 * u, 3 * u);
                let cb = g.slice_axis(b, 1, 2 * u, 3 * u);
                let gated = g.mul(r, ch);
                let xcb = g.add(cx, cb);
                let cand_pre = g.add(xcb, gated);
                let cand = g.tanh(cand_pre);
                let one = g.constant(Tensor::ones(vec![u]));
                let keep = g.sub(one, z);
                let kept = g.mul(keep, h_prev);
                let updated = g.mul(z, cand);
                let h = g.add(kept, updated);
                (h, h)
            }
            Cell::Lstm => {
                // 四门 LSTM：i 输入门、f 遗忘门、o 输出门、候选细胞态
                let i = g.sigmoid(gates[0]);
                let f = g.sigmoid(gates[1]);
                let o = g.sigmoid(gates[2]);
                let cand = g.tanh(gates[3]);
                let forget = g.mul(f, c_prev);
                let added = g.mul(i, cand);
                let c = g.add(forget, added);
                let tanh_c = g.tanh(c);
                let h = g.mul(o, tanh_c);
                (h, c)
            }
            // 无 equations 的 custom 细胞在上面已拦截，合并门只服务内置细胞
            Cell::Custom => unreachable!(),
        })
    }
}

impl Layer for Rnn {
    fn name(&self) -> &str {
        &self.name
    }
    fn param_names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(k, _)| short_key(&self.name, k).to_string())
            .collect()
    }
    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()> {
        for (n, t) in params {
            let key = format!("{}.{}", self.name, short_key(&self.name, n));
            if let Some(slot) = self.params.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = t.clone();
            }
        }
        Ok(())
    }
    fn dump_params(&self) -> Params {
        self.params.clone()
    }
    fn forward(&mut self, args: &[&Var], ctx: &LayerCtx, graph: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *args[0];
        let shape = graph.value(x).shape.clone();
        if shape.len() != 3 {
            return Err(MtbError::Shape {
                expected: "(batch, seq, features)".into(),
                got: format!("{shape:?}"),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);
        let u = self.units;

        // `seq_state` 是跨批次续算用的上一批末隐状态（推理侧）；细胞态一律从 0 起。
        let mut states_fwd: Option<(Var, Var)> = None;
        if let Some(v) = &ctx.seq_state {
            if v.len() != u {
                return Err(MtbError::Shape {
                    expected: format!("{} 维隐状态", u),
                    got: format!("{} 维", v.len()),
                });
            }
            let h0 = graph.constant(Tensor::from_vec(v.clone(), vec![1, u])?);
            let c0 = graph.constant(Tensor::zeros(vec![1, u]));
            states_fwd = Some((h0, c0));
        }
        let mut states_rev: Option<(Var, Var)> = None;

        let mut outs_fwd: Vec<Var> = Vec::new();
        let mut outs_rev: Vec<Var> = Vec::new();
        for t in 0..seq {
            let xt = graph.slice_axis(x, 1, t, t + 1);
            let xt = graph.reshape(xt, vec![batch, self.input_dim]);
            let (h, c) = self.step(xt, states_fwd, false, graph)?;
            states_fwd = Some((h, c));
            if self.return_sequences {
                outs_fwd.push(graph.reshape(h, vec![batch, 1, u]));
            }
        }
        if self.bidirectional {
            for t in (0..seq).rev() {
                let xt = graph.slice_axis(x, 1, t, t + 1);
                let xt = graph.reshape(xt, vec![batch, self.input_dim]);
                // 反向方向有独立的一套权重（见 allocate 的 `r` 后缀）
                let (h, c) = self.step(xt, states_rev, true, graph)?;
                states_rev = Some((h, c));
                if self.return_sequences {
                    outs_rev.push(graph.reshape(h, vec![batch, 1, u]));
                }
            }
            // 逆序遍历得到的是 t = seq-1..0，时间轴要回到 0..seq-1
            outs_rev.reverse();
        }

        // 输出装配：
        // - return_sequences：逐时刻 (batch,1,units) 沿时间轴拼成 (batch, seq, units)
        // - 否则：只取末时刻 (batch, units)；双向再沿特征轴拼成 (batch, 2*units)
        let out = if self.return_sequences {
            let f = graph.concat(1, &outs_fwd);
            if outs_rev.is_empty() {
                f
            } else {
                let r = graph.concat(1, &outs_rev);
                graph.concat(2, &[f, r])
            }
        } else {
            let f = match states_fwd {
                Some((h, _)) => h,
                None => graph.constant(Tensor::zeros(vec![batch, u])),
            };
            match states_rev {
                Some((h, _)) => graph.concat(1, &[f, h]),
                None => f,
            }
        };
        Ok(vec![out])
    }
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let (b, s) = (inputs[0][0], inputs[0][1]);
        let units = if self.bidirectional {
            self.units * 2
        } else {
            self.units
        };
        let out = if self.return_sequences {
            vec![b, s, units]
        } else {
            vec![b, units]
        };
        Ok(vec![out])
    }
    fn describe(&self) -> LayerSpec {
        let kind = match self.cell {
            Cell::Simple => "rnn",
            Cell::Gru => "gru",
            Cell::Lstm => "lstm",
            Cell::Custom => "rnn_custom",
        };
        let mut json = format!(
            "{{\"units\":{},\"bi\":{},\"rs\":{}",
            self.units, self.bidirectional, self.return_sequences
        );
        // 方程组进描述：两套不同 equations 的 custom 层必须有不同拓扑哈希（README 6 节）
        if let Some(spec) = &self.spec {
            json.push_str(&format!(",\"equations\":\"{}\"", spec.canonical()));
        }
        json.push('}');
        spec(kind, &self.name, &json)
    }
}

// ───────────────────────── 卷积 / 池化 ─────────────────────────

/// 一维卷积：`(batch, in_ch, len) -> (batch, filters, out_len)`。
pub struct Conv1d {
    pub name: String,
    pub in_ch: usize,
    pub filters: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub activation: Option<&'static str>,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Conv1d {
    pub fn new(
        name: &str,
        in_ch: usize,
        filters: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        activation: Option<&'static str>,
        seed: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            in_ch,
            filters,
            kernel,
            stride,
            padding,
            dilation,
            activation,
            weight: Init::Xavier.make(&[filters, in_ch * kernel], seed),
            bias: Tensor::zeros(vec![filters]),
        }
    }
}

impl Layer for Conv1d {
    fn name(&self) -> &str {
        &self.name
    }
    fn param_names(&self) -> Vec<String> {
        vec!["weight".into(), "bias".into()]
    }
    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()> {
        for (n, t) in params {
            match short_key(&self.name, n) {
                "weight" => self.weight = t.clone(),
                "bias" => self.bias = t.clone(),
                _ => {}
            }
        }
        Ok(())
    }
    fn dump_params(&self) -> Params {
        vec![
            (format!("{}.weight", self.name), self.weight.clone()),
            (format!("{}.bias", self.name), self.bias.clone()),
        ]
    }
    fn forward(&mut self, args: &[&Var], _ctx: &LayerCtx, graph: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *args[0];
        let len = graph.value(x).shape[2];
        let w = graph.param(format!("{}.weight", self.name), self.weight.clone());
        let b = graph.param(format!("{}.bias", self.name), self.bias.clone());
        let params = crate::api::ConvParams {
            in_len: len,
            in_ch: self.in_ch,
            kernel: self.kernel,
            filters: self.filters,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
        };
        let y = graph.conv1d(x, w, b, params);
        let out = match self.activation {
            Some("relu") => graph.relu(y),
            Some("tanh") => graph.tanh(y),
            Some("sigmoid") => graph.sigmoid(y),
            Some("gelu") => graph.gelu(y),
            None => y,
            Some(other) => {
                return Err(MtbError::Config(format!(
                    "conv1d: 未知激活 {other:?}"
                )))
            }
        };
        Ok(vec![out])
    }
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let b = inputs[0][0];
        let len = inputs[0][2];
        let out_len =
            (len + 2 * self.padding - self.dilation * (self.kernel - 1) - 1) / self.stride + 1;
        Ok(vec![vec![b, self.filters, out_len]])
    }
    fn describe(&self) -> LayerSpec {
        spec("conv1d", &self.name, &format!(
            "{{\"filters\":{},\"kernel\":{},\"stride\":{},\"pad\":{}}}",
            self.filters, self.kernel, self.stride, self.padding
        ))
    }
}

fn pool_layer(kind: &str, name: String, kernel: usize, stride: usize) -> impl Layer + Send {
    Pool1d { kind: kind.to_string(), name, kernel, stride }
}

/// 定长池化 `(batch, ch, len) -> (batch, ch, out_len)`。
pub struct Pool1d {
    pub kind: String,
    pub name: String,
    pub kernel: usize,
    pub stride: usize,
}

impl Layer for Pool1d {
    fn name(&self) -> &str {
        &self.name
    }
    fn param_names(&self) -> Vec<String> {
        vec![]
    }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> {
        Ok(())
    }
    fn dump_params(&self) -> Params {
        vec![]
    }
    fn forward(&mut self, args: &[&Var], _ctx: &LayerCtx, graph: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *args[0];
        let out = if self.kind == "max" {
            graph.maxpool1d(x, self.kernel, self.stride)
        } else {
            graph.meanpool1d(x, self.kernel, self.stride)
        };
        Ok(vec![out])
    }
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let b = inputs[0][0];
        let len = inputs[0][2];
        let out_len = (len - self.kernel) / self.stride + 1;
        Ok(vec![vec![b, inputs[0][1], out_len]])
    }
    fn describe(&self) -> LayerSpec {
        spec(&format!("{}pool1d", self.kind), &self.name, &format!("{{\"kernel\":{}}}", self.kernel))
    }
}

/// 全局池化 `(batch, ch, len) -> (batch, ch)`。
pub struct GlobalPool1d {
    pub kind: String,
    pub name: String,
}

impl Layer for GlobalPool1d {
    fn name(&self) -> &str {
        &self.name
    }
    fn param_names(&self) -> Vec<String> {
        vec![]
    }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> {
        Ok(())
    }
    fn dump_params(&self) -> Params {
        vec![]
    }
    fn forward(&mut self, args: &[&Var], _ctx: &LayerCtx, graph: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *args[0];
        let shape = graph.value(x).shape.clone();
        let axis = shape.len() - 1;
        let pooled = if self.kind == "max" {
            graph.max_axis(x, axis)
        } else {
            graph.mean_axis(x, axis)
        };
        // 归约保留长度为 1 的轴，这里压成 (batch, ch) 与 infer_shapes 对齐
        Ok(vec![graph.reshape(pooled, vec![shape[0], shape[1]])])
    }
    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        Ok(vec![vec![inputs[0][0], inputs[0][1]]])
    }
    fn describe(&self) -> LayerSpec {
        spec(&format!("global{}pool1d", self.kind), &self.name, "{}")
    }
}

/// 便捷：构造定长池化层。
pub fn maxpool(name: &str, kernel: usize, stride: usize) -> Box<dyn Layer> {
    Box::new(pool_layer("max", name.to_string(), kernel, stride))
}
/// 便捷：构造定长平均池化层。
pub fn avgpool(name: &str, kernel: usize, stride: usize) -> Box<dyn Layer> {
    Box::new(pool_layer("avg", name.to_string(), kernel, stride))
}
/// 便捷：构造全局最大池化层。
pub fn global_maxpool(name: &str) -> Box<dyn Layer> {
    Box::new(GlobalPool1d { kind: "max".into(), name: name.to_string() })
}
/// 便捷：构造全局平均池化层。
pub fn global_avgpool(name: &str) -> Box<dyn Layer> {
    Box::new(GlobalPool1d { kind: "avg".into(), name: name.to_string() })
}

// ───────────────────────── 结构层 ─────────────────────────

pub struct Flatten {
    pub name: String,
}
impl Layer for Flatten {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let shape = flat_2d(g.value(*a[0]).shape());
        Ok(vec![g.reshape(*a[0], shape)])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> {
        Ok(vec![flat_2d(i[0])])
    }
    fn describe(&self) -> LayerSpec { spec("flatten", &self.name, "{}") }
}

pub struct Reshape {
    pub name: String,
    pub shape: Shape,
}
impl Layer for Reshape {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        Ok(vec![g.reshape(*a[0], self.shape.clone())])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let want: usize = self.shape.iter().product();
        let got: usize = i[0].iter().product();
        if want != got {
            return Err(MtbError::Shape {
                expected: format!("{want} elements"),
                got: format!("{got} elements"),
            });
        }
        Ok(vec![self.shape.clone()])
    }
    fn describe(&self) -> LayerSpec { spec("reshape", &self.name, "{}") }
}

pub struct Concat {
    pub name: String,
    pub axis: usize,
}
impl Layer for Concat {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let vs: Vec<Var> = a.iter().map(|v| **v).collect();
        Ok(vec![g.concat(self.axis, &vs)])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let mut out = i[0].clone();
        out[self.axis] = i.iter().map(|s| s[self.axis]).sum();
        Ok(vec![out])
    }
    fn describe(&self) -> LayerSpec { spec("concat", &self.name, "{}") }
}

/// 轴置换层：文本卷积常用 `[0, 2, 1]` 把 `(batch, seq, ch)` 转成通道的行主序。
pub struct Permute {
    pub name: String,
    pub axes: Vec<usize>,
}
impl Permute {
    pub fn new(name: &str, axes: &[usize]) -> Self {
        Self { name: name.to_string(), axes: axes.to_vec() }
    }
}
impl Layer for Permute {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *a[0];
        Ok(vec![g.permute(x, &self.axes)])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let src = &i[0];
        if self.axes.len() != src.len() {
            return Err(MtbError::Shape {
                expected: format!("{} 个轴", src.len()),
                got: format!("{} 个轴", self.axes.len()),
            });
        }
        let mut out = Shape::new();
        for &ax in &self.axes {
            out.push(*src.get(ax).ok_or(MtbError::coded(
                MtbError::TABLE,
                format!("permute: 轴 {ax} 越界"),
            ))?);
        }
        Ok(vec![out])
    }
    fn describe(&self) -> LayerSpec {
        let axes = self
            .axes
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",");
        spec("permute", &self.name, &format!("{{\"axes\":[{axes}]}}"))
    }
}

/// 独立激活层。
pub struct Activation {
    pub name: String,
    pub kind: &'static str,
}
impl Layer for Activation {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *a[0];
        let out = match self.kind {
            "relu" => g.relu(x),
            "sigmoid" => g.sigmoid(x),
            "tanh" => g.tanh(x),
            "gelu" => g.gelu(x),
            "softmax" => g.softmax_last(x),
            other => return Err(MtbError::Config(format!("activation: 未知 {other:?}"))),
        };
        Ok(vec![out])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> { Ok(vec![i[0].clone()]) }
    fn describe(&self) -> LayerSpec { spec("activation", &self.name, "{}") }
}

/// Dropout：掩码在层内生成，训练/推理双模式（README 4.1 表）。
pub struct Dropout {
    pub name: String,
    pub rate: f32,
    /// 随每次前向推进，保证同一个 batch 不会被反复套同一张掩码
    rng: ChaCha8Rng,
}
impl Dropout {
    pub fn new(name: &str, rate: f32, seed: u64) -> Self {
        Self { name: name.to_string(), rate, rng: rng_from(seed) }
    }
}
impl Layer for Dropout {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec![] }
    fn bind_params(&mut self, _p: &[(String, Tensor)]) -> MtbResult<()> { Ok(()) }
    fn dump_params(&self) -> Params { vec![] }
    fn forward(&mut self, a: &[&Var], ctx: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *a[0];
        if !ctx.training || self.rate <= 0.0 {
            return Ok(vec![x]);
        }
        let n = g.value(x).data.len();
        let keep = 1.0 - self.rate;
        let mut mask = vec![0f32; n];
        for m in mask.iter_mut() {
            *m = if self.rng.gen::<f32>() < keep { 1.0 / keep } else { 0.0 };
        }
        Ok(vec![g.dropout(x, &mask)])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> { Ok(vec![i[0].clone()]) }
    fn describe(&self) -> LayerSpec {
        spec("dropout", &self.name, &format!("{{\"rate\":{}}}", self.rate))
    }
}

/// LayerNorm：统计量在线计算（README 4.1 表），全由基础算子组合而成。
pub struct LayerNorm {
    pub name: String,
    pub eps: f32,
    pub gamma: Tensor,
    pub beta: Tensor,
}
impl LayerNorm {
    pub fn new(name: &str, dim: usize, eps: f32) -> Self {
        Self {
            name: name.to_string(),
            eps,
            gamma: Tensor::ones(vec![dim]),
            beta: Tensor::zeros(vec![dim]),
        }
    }
}
impl Layer for LayerNorm {
    fn name(&self) -> &str { &self.name }
    fn param_names(&self) -> Vec<String> { vec!["gamma".into(), "beta".into()] }
    fn dump_params(&self) -> Params {
        vec![
            (format!("{}.gamma", self.name), self.gamma.clone()),
            (format!("{}.beta", self.name), self.beta.clone()),
        ]
    }
    fn bind_params(&mut self, p: &[(String, Tensor)]) -> MtbResult<()> {
        for (n, t) in p {
            match short_key(&self.name, n) {
                "gamma" => self.gamma = t.clone(),
                "beta" => self.beta = t.clone(),
                _ => {}
            }
        }
        Ok(())
    }
    fn forward(&mut self, a: &[&Var], _c: &LayerCtx, g: &mut Graph) -> MtbResult<Vec<Var>> {
        let x = *a[0];
        let rank = g.value(x).shape.len();
        let axis = rank - 1;
        let gamma = g.param(format!("{}.gamma", self.name), self.gamma.clone());
        let beta = g.param(format!("{}.beta", self.name), self.beta.clone());
        let mu = g.mean_axis(x, axis);
        let d = g.sub(x, mu);
        let d2 = g.mul(d, d);
        let var = g.mean_axis(d2, axis);
        let eps = g.constant(Tensor::from_vec(vec![self.eps], vec![1]).unwrap());
        let var = g.add(var, eps);
        // 1/sqrt(var)
        let inv = g.pow(var, -0.5);
        let normed = g.mul(d, inv);
        let scaled = g.mul(normed, gamma);
        Ok(vec![g.add(scaled, beta)])
    }
    fn infer_shapes(&self, i: &[&Shape]) -> MtbResult<Vec<Shape>> { Ok(vec![i[0].clone()]) }
    fn describe(&self) -> LayerSpec {
        spec("layernorm", &self.name, &format!("{{\"eps\":{}}}", self.eps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LayerCtx;

    fn ctx(training: bool) -> LayerCtx {
        LayerCtx { training, seq_state: None }
    }

    #[test]
    fn dense_forwards_and_produces_grads() {
        let mut dense = Dense::new("d", 3, 2, Some("relu"), Init::Zeros, 1);
        dense.bind_params(&[
            ("weight".into(), Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![3, 2]).unwrap()),
            ("bias".into(), Tensor::from_vec(vec![0.5, -0.5], vec![2]).unwrap()),
        ]).unwrap();
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
        let args = [&x];
        let out = dense.forward(&args, &ctx(true), &mut g).unwrap();
        let y = g.value(out[0]).data.clone();
        assert_eq!(y, vec![1.5, 1.5], "W·x+b 应为 [1.5, 1.5]");
        assert_eq!(g.value(out[0]).shape, vec![1, 2]);
        // 反向必须能一路回到权重节点（dense 的 weight 是可训练参数）
        g.backward(&[out[0]]).unwrap();
        let mut found = false;
        for (_, name, _, grad) in g.trainable() {
            if name == "d.weight" {
                found = true;
                let grad = grad.expect("已使用的参数应有梯度");
                assert_eq!(grad.shape, vec![3, 2], "权重梯度形状应与权重一致");
                assert_eq!(
                    grad.data,
                    vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                    "dL/dW 每列都应是输入向量"
                );
            }
        }
        assert!(found, "dense 应注册可训练参数 d.weight");
    }

    #[test]
    fn gru_forwards_over_sequence() {
        let mut rnn = Rnn::new("enc", 4, 3, Cell::Gru, 7);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![0.1f32; 2 * 5 * 4], vec![2, 5, 4]).unwrap());
        let args = [&x];
        let out = rnn.forward(&args, &ctx(false), &mut g).unwrap();
        let shape = g.value(out[0]).shape.clone();
        assert_eq!(shape, vec![2, 5, 3], "return_sequences 默认应保留时间维");
    }

    #[test]
    fn gru_return_sequences_false() {
        let mut rnn = Rnn::new("enc", 4, 3, Cell::Gru, 7);
        rnn.return_sequences = false;
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![0.1f32; 2 * 5 * 4], vec![2, 5, 4]).unwrap());
        let args = [&x];
        let out = rnn.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out[0]).shape, vec![2, 3]);
    }

    #[test]
    fn lstm_forwards() {
        let mut rnn = Rnn::new("enc", 3, 4, Cell::Lstm, 11);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![0.2f32; 1 * 3 * 3], vec![1, 3, 3]).unwrap());
        let args = [&x];
        let out = rnn.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out[0]).shape, vec![1, 3, 4]);
    }

    // ───────────── Cell DSL 路径（README 4.1.1 路径 B） ─────────────

    /// README 示例方程组：手写 GRU，应与内置 GRU 数值一致。
    const GRU_EQS: &str = "
        z = sigmoid(Wz*x + Uz*h_prev + Bz)
        r = sigmoid(Wr*x + Ur*h_prev + Br)
        h_cand = tanh(W*x + r*(U*h_prev) + B)
        h = (1-z)*h_prev + z*h_cand
    ";

    /// 取合并门参数里第 `i` 个门那一段：二维按列切，一维（偏置）按段切。
    fn gate_slice(t: &Tensor, units: usize, i: usize) -> Tensor {
        let (lo, hi) = (i * units, (i + 1) * units);
        if t.shape.len() == 1 {
            return Tensor::from_vec(t.data[lo..hi].to_vec(), vec![units]).unwrap();
        }
        let (rows, wide) = (t.shape[0], t.shape[1]);
        let mut d = Vec::with_capacity(rows * units);
        for r in 0..rows {
            d.extend_from_slice(&t.data[r * wide + lo..r * wide + hi]);
        }
        Tensor::from_vec(d, vec![rows, units]).unwrap()
    }

    /// 装载期对拍基线：相同权重下 DSL 手写 GRU 与内置 GRU 前向逐时刻一致。
    #[test]
    fn dsl_gru_matches_builtin_gru() {
        let (d, u, batch, seq) = (3usize, 4usize, 2usize, 5usize);
        let builtin = Rnn::new("bi", d, u, Cell::Gru, 21);
        let dump = builtin.dump_params();
        let w0 = dump.iter().find(|(k, _)| k == "bi.w0").map(|(_, v)| v.clone()).unwrap();
        let u0 = dump.iter().find(|(k, _)| k == "bi.u0").map(|(_, v)| v.clone()).unwrap();
        let b0 = dump.iter().find(|(k, _)| k == "bi.b0").map(|(_, v)| v.clone()).unwrap();

        let mut custom = Rnn::custom("cu", d, u, GRU_EQS, 99).unwrap();
        // 内置合并顺序是 0=z、1=r、2=候选态，逐门搬到 DSL 的独立参数槽
        custom
            .bind_params(&[
                ("Wz".into(), gate_slice(&w0, u, 0)),
                ("Uz".into(), gate_slice(&u0, u, 0)),
                ("Bz".into(), gate_slice(&b0, u, 0)),
                ("Wr".into(), gate_slice(&w0, u, 1)),
                ("Ur".into(), gate_slice(&u0, u, 1)),
                ("Br".into(), gate_slice(&b0, u, 1)),
                ("W".into(), gate_slice(&w0, u, 2)),
                ("U".into(), gate_slice(&u0, u, 2)),
                ("B".into(), gate_slice(&b0, u, 2)),
            ])
            .unwrap();

        let data: Vec<f32> = (0..batch * seq * d)
            .map(|i| (i % 7) as f32 * 0.25 - 0.5)
            .collect();
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(data, vec![batch, seq, d]).unwrap());
        let args = [&x];
        let mut b = builtin;
        let out_b = b.forward(&args, &ctx(false), &mut g).unwrap();
        let out_c = custom.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out_b[0]).shape, vec![batch, seq, u]);
        assert_eq!(g.value(out_c[0]).shape, vec![batch, seq, u]);

        let mut max_diff = 0f32;
        for (l, r) in g.value(out_b[0]).data.iter().zip(&g.value(out_c[0]).data) {
            max_diff = max_diff.max((l - r).abs());
        }
        assert!(max_diff < 1e-5, "DSL GRU 与内置 GRU 前向应一致，最大偏差 {max_diff}");
    }

    #[test]
    fn custom_cell_backprops_through_layer_params() {
        let mut rnn = Rnn::custom("enc", 2, 3, GRU_EQS, 5).unwrap();
        let mut g = Graph::new();
        let x = g.constant(
            Tensor::from_vec((0..8).map(|i| i as f32 * 0.1).collect(), vec![1, 4, 2]).unwrap(),
        );
        let args = [&x];
        let out = rnn.forward(&args, &ctx(true), &mut g).unwrap();
        g.backward(&[out[0]]).unwrap();
        let trained: Vec<String> = g
            .trainable()
            .filter(|(_, _, _, grad)| grad.is_some())
            .map(|(_, n, _, _)| n.to_string())
            .collect();
        for n in [
            "enc.Wz", "enc.Uz", "enc.Bz", "enc.Wr", "enc.Ur", "enc.Br", "enc.W", "enc.U", "enc.B",
        ] {
            assert!(trained.iter().any(|k| k == n), "{n} 应收到梯度，实际 {trained:?}");
        }
    }

    #[test]
    fn custom_cell_rejects_bad_equations_at_load() {
        let e = Rnn::custom("enc", 2, 3, "z = sigmoid(Wz*x)", 1).unwrap_err();
        assert!(format!("{e}").contains("h ="), "装载期就该拒绝没有 h 的方程组，实际 {e}");
    }

    #[test]
    fn custom_cell_without_equations_errors_instead_of_gru() {
        let mut rnn = Rnn::new("enc", 2, 3, Cell::Custom, 1);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![0.1f32; 8], vec![1, 4, 2]).unwrap());
        let args = [&x];
        let e = rnn.forward(&args, &ctx(false), &mut g).unwrap_err();
        assert!(format!("{e}").contains("equations"), "应明确报错而非静默按 GRU 跑，实际 {e}");
    }

    /// 两层 custom 只用同名参数（都叫 `Wz`）：图内键必须带层名前缀，否则权重会被合并共享。
    #[test]
    fn two_custom_layers_keep_their_own_params() {
        let mut a = Rnn::custom("a", 2, 3, GRU_EQS, 3).unwrap();
        let mut b = Rnn::custom("b", 2, 3, GRU_EQS, 4).unwrap();
        let mut g = Graph::new();
        let x = g.constant(
            Tensor::from_vec((0..6).map(|i| i as f32 * 0.1).collect(), vec![1, 3, 2]).unwrap(),
        );
        let args = [&x];
        let oa = a.forward(&args, &ctx(true), &mut g).unwrap();
        let ob = b.forward(&args, &ctx(true), &mut g).unwrap();
        assert_ne!(
            g.value(oa[0]).data,
            g.value(ob[0]).data,
            "两套独立初值的 custom 层输出不应相同（相同即参数节点撞名）"
        );
        g.backward(&[oa[0], ob[0]]).unwrap();
        let trained: Vec<String> = g
            .trainable()
            .filter(|(_, _, _, grad)| grad.is_some())
            .map(|(_, n, _, _)| n.to_string())
            .collect();
        assert!(trained.iter().any(|k| k == "a.Wz") && trained.iter().any(|k| k == "b.Wz"),
            "两个层各自的 Wz 都应可训练，实际 {trained:?}");
    }

    #[test]
    fn bidirectional_custom_cell_allocates_reverse_weights() {
        let rnn = Rnn::custom("enc", 2, 3, GRU_EQS, 3).unwrap().bidirectional();
        let names = rnn.param_names();
        assert!(names.iter().any(|n| n == "Wz"), "应有正向 Wz");
        assert!(names.iter().any(|n| n == "Wzr"), "反向方向应另有一套 Wzr");
        assert_eq!(names.len(), 18, "9 个参数 × 2 个方向");
        let d = rnn.describe().json;
        assert!(d.contains("equations") && d.contains("h=(((1-z)*h_prev)+(z*h_cand))"),
            "方程组必须进拓扑描述，否则不同 custom 结构会撞哈希：{d}");
    }

    #[test]
    fn conv1d_same_padding_preserves_length() {
        let mut conv = Conv1d::new("c", 1, 2, 3, 1, 1, 1, None, 5);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![0.0f32; 1 * 1 * 7], vec![1, 1, 7]).unwrap());
        let args = [&x];
        let out = conv.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out[0]).shape, vec![1, 2, 7], "padding=same 应保持长度");
    }

    #[test]
    fn global_pool_squeezes_time() {
        let mut gp = global_maxpool("p");
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec((0..12).map(|i| i as f32).collect::<Vec<f32>>(), vec![2, 3, 2]).unwrap());
        let args = [&x];
        let out = gp.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out[0]).shape, vec![2, 3]);
        assert_eq!(g.value(out[0]).data, vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn layernorm_normalizes_rows() {
        let mut ln = LayerNorm::new("n", 4, 1e-5);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(
            (0..4).map(|i| i as f32).collect::<Vec<_>>().iter().cloned().collect(),
            vec![1, 4],
        ).unwrap());
        let args = [&x];
        let out = ln.forward(&args, &ctx(false), &mut g).unwrap();
        let d = g.value(out[0]).data.clone();
        let mean = d.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "LayerNorm 后均值应≈0，得到 {mean}");
    }

    #[test]
    fn dropout_is_identity_in_inference() {
        let mut d = Dropout::new("drop", 0.5, 3);
        let mut g = Graph::new();
        let x = g.constant(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap());
        let args = [&x];
        let out = d.forward(&args, &ctx(false), &mut g).unwrap();
        assert_eq!(g.value(out[0]).data, vec![1.0, 2.0, 3.0, 4.0], "推理模式不应丢元素");
    }
}
