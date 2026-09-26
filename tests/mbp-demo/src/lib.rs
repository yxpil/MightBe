//! 测试用 cdylib 插件（README 16.3：加载 → 前向 → 反向 → RELOAD 热替换 → panic 隔离）。
//!
//! 刻意只依赖 `mightbe-plugin` 这一个叶子、不链接宿主，因此能真实地跨 ABI 被加载。
//!
//! 导出两个层：
//! - `scale_tanh`：可训练层，`y = tanh(scale · (x @ Wᵀ))`，权重 `w` 由宿主分配，
//!   `scale` 由配置带入——同一份二进制换个配置就是另一种"实现"，正好用来演示
//!   "RELOAD 换实现、旧实例仍在途" 的热替换语义；
//! - `boom`：`{"trip":1}` 时在前向里 panic，用来验证插件 panic 被隔离成错误码
//!   且宿主线程存活。
//!
//! 配置解析刻意不引 serde_json（不在 README 15 节白名单内）：只认本文件用到的几个键。

use mightbe_plugin::{elem, Arg, Host, MbLayer, Out};

/// 取 `"key": <value>` 中 value 的字面量（去引号、去空白）。
fn field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let idx = json.find(&needle)? + needle.len();
    // 跳过键与值之间的 ':'，再跳过空白
    let after = &json[idx..];
    let colon = after.find(':')?;
    let val = after[colon + 1..].trim_start();
    match val.chars().next()? {
        '"' => {
            let end = val[1..].find('"')? + 1;
            Some(&val[1..end])
        }
        c if c.is_ascii_digit() || c == '-' || c == '+' => {
            let n = val[1..]
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
                .count();
            Some(&val[..1 + n])
        }
        _ => None,
    }
}

fn f32_of(json: &str, key: &str, default: f32) -> f32 {
    field(json, key)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default)
}

fn usize_of(json: &str, key: &str, default: usize) -> usize {
    field(json, key)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn at<'a>(args: &'a [Arg<'a>], k: usize) -> Arg<'a> {
    args[k]
}

fn ndim_of(a: &Arg) -> usize {
    a.dims.len()
}

fn dim_of(a: &Arg, axis: usize) -> usize {
    a.dims[axis] as usize
}

/// 行主序 `m×k` 转置为 `k×m`。
fn transpose(g: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut t = vec![0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            t[c * m + r] = g[r * k + c];
        }
    }
    t
}

// ───────────────────────── scale_tanh ─────────────────────────

/// `y = tanh(scale · (x @ Wᵀ))`，`x: [b, in]`，`W: [out, in]` → `y: [b, out]`。
struct ScaleTanh {
    in_dim: usize,
    out_dim: usize,
    scale: f32,
}

fn scale_in_place(x: &mut [f32], k: f32) {
    if k != 1.0 {
        for v in x.iter_mut() {
            *v *= k;
        }
    }
}

impl MbLayer for ScaleTanh {
    fn new(config_json: &str) -> Result<Self, String> {
        Ok(Self {
            in_dim: usize_of(config_json, "in", 1),
            out_dim: usize_of(config_json, "out", 1),
            scale: f32_of(config_json, "scale", 1.0),
        })
    }

    fn params(&self) -> Vec<(String, Vec<u32>)> {
        vec![(
            "w".to_string(),
            vec![self.out_dim as u32, self.in_dim as u32],
        )]
    }

    /// 装载期静态形状推导：宿主只传**数据**输入，参数形状不在其中。
    fn infer_shapes(&self, inputs: &[Vec<u32>]) -> Result<Vec<u32>, String> {
        match inputs {
            [shape] if shape.len() == 2 => {
                if shape[1] as usize != self.in_dim {
                    return Err(format!(
                        "输入第二维 {} 与配置 in={} 不符",
                        shape[1], self.in_dim
                    ));
                }
                Ok(vec![shape[0], self.out_dim as u32])
            }
            other => Err(format!("期望 [batch, {}]，得到 {other:?}", self.in_dim)),
        }
    }

    fn forward(&self, args: &[Arg], host: &Host) -> Result<Out, String> {
        if args.is_empty() {
            return Err("缺少数据输入".to_string());
        }
        let x = at(args, 0);
        if ndim_of(&x) != 2 {
            return Err(format!("数据输入期望 2 维，得到 {} 维", ndim_of(&x)));
        }
        let b = dim_of(&x, 0);
        let i = dim_of(&x, 1);
        if i != self.in_dim {
            return Err(format!("输入第二维 {i} 与配置 in={} 不符", self.in_dim));
        }
        // 权重在 args 尾部；冒烟只传数据输入时它不在，此时按零权重处理——
        // 冒烟校验的是"形状推导与前向一致"，不关心数值。
        let (wd, wdata) = if args.len() >= 2 {
            (args[1].dims.to_vec(), args[1].data.to_vec())
        } else {
            (
                vec![self.out_dim as u32, self.in_dim as u32],
                vec![0f32; self.out_dim * self.in_dim],
            )
        };
        let out_dim = wd.get(0).copied().unwrap_or_default() as usize;
        let w_in = wd.get(1).copied().unwrap_or_default() as usize;
        if w_in != self.in_dim || out_dim != self.out_dim {
            return Err(format!(
                "权重形状 [{}] 与配置 in={} out={} 不符",
                wd.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                self.in_dim,
                self.out_dim
            ));
        }
        // 权重是 `[out, in]` 行主序，而 `pre = x @ Wᵀ` 需要 `[in, out]`
        let wt = transpose(&wdata, out_dim, i);
        let mut y = host.matmul(x.data, &wt, b, i, out_dim)?;
        scale_in_place(&mut y, self.scale);
        host.elementwise(&mut y, elem::TANH)?;
        Out::new(y, vec![b as u32, self.out_dim as u32])
    }

    /// 反向：`args` 与前向同序（缺权重槽时按零权重复用前向的兜底），梯度同序回传。
    ///
    /// `y = tanh(scale · pre)` ⇒ `dL/dpre = dL/dy · (1 - y²)`，这个恒等式对任意
    /// `scale` 都成立，因此不必反解 `pre` 就能拿到正确的权重梯度。
    fn backward(
        &self,
        grad_out: &Arg,
        args: &[Arg],
        out: &Arg,
        host: &Host,
    ) -> Result<Vec<Option<Out>>, String> {
        let x = at(args, 0);
        let (wd, wdata) = if args.len() >= 2 {
            (args[1].dims.to_vec(), args[1].data.to_vec())
        } else {
            (
                vec![self.out_dim as u32, self.in_dim as u32],
                vec![0f32; self.out_dim * self.in_dim],
            )
        };
        let out_dim = wd.get(0).copied().unwrap_or_default() as usize;
        let i = dim_of(&x, 1);
        let b = dim_of(&x, 0);
        if i != self.in_dim || out_dim != self.out_dim {
            return Err(format!("形状不符：x=[{b},{i}] 权重=[{}]", wd.len()));
        }
        if grad_out.data.len() != b * out_dim || out.data.len() != b * out_dim {
            return Err(format!(
                "grad_out/ out 长度与 [{b},{out_dim}] 不符：{} / {}",
                grad_out.data.len(),
                out.data.len()
            ));
        }
        let mut ghat = grad_out.data.to_vec();
        for (gh, y) in ghat.iter_mut().zip(out.data.iter()) {
            *gh *= 1.0 - y * y;
        }
        // dW = ĝᵀ @ x，`[out, i]`
        let gt = transpose(&ghat, b, out_dim);
        let dw = host.matmul(&gt, x.data, out_dim, b, i)?;
        // dx = ĝ @ W，`[b, i]`（前向里用的就是 Wᵀ，转置回来即可）
        let wt = transpose(&wdata, out_dim, i);
        let dx = host.matmul(&ghat, &wt, b, out_dim, i)?;
        Ok(vec![
            Some(Out::new(dx, vec![b as u32, i as u32])?),
            Some(Out::new(dw, vec![out_dim as u32, i as u32])?),
        ])
    }
}

// ───────────────────────── boom ─────────────────────────

/// 冒烟必过、`{"trip":1}` 时前向 panic 的对照层。无参数。
struct Boom {
    trip: bool,
    width: usize,
}

impl MbLayer for Boom {
    fn new(config_json: &str) -> Result<Self, String> {
        Ok(Self {
            trip: field(config_json, "trip").map(|s| s == "1").unwrap_or(false),
            width: usize_of(config_json, "width", 1),
        })
    }

    fn params(&self) -> Vec<(String, Vec<u32>)> {
        Vec::new()
    }

    fn infer_shapes(&self, inputs: &[Vec<u32>]) -> Result<Vec<u32>, String> {
        if inputs.len() != 1 {
            return Err(format!("期望 [batch]，得到 {inputs:?}"));
        }
        Ok(vec![inputs[0][0], self.width as u32])
    }

    fn forward(&self, args: &[Arg], _host: &Host) -> Result<Out, String> {
        if self.trip {
            // 故意跨 extern "C" 展开：SDK 侧的 catch_unwind 必须接住它。
            panic!("boom: 插件按配置要求现炸");
        }
        let x = at(args, 0);
        let b = if ndim_of(&x) == 0 { 1 } else { dim_of(&x, 0) };
        Out::new(
            x.data.repeat(self.width),
            vec![b as u32, self.width as u32],
        )
    }

    fn backward(
        &self,
        _grad_out: &Arg,
        _args: &[Arg],
        _out: &Arg,
        _host: &Host,
    ) -> Result<Vec<Option<Out>>, String> {
        // 本层不参与训练：不返回梯度槽内容，宿主填空张量。
        Ok(vec![None])
    }
}

// ───────────────────────── 导出 ─────────────────────────

mightbe_plugin::mb_export! {
    api = mightbe_plugin::MB_ABI_VERSION,
    layers = [
        "scale_tanh" => ScaleTanh,
        "boom"       => Boom,
    ],
}
