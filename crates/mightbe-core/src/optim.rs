//! 优化器：SGD（含 momentum）/ Adam（默认）。
//!
//! 参数按 `(网络名, 层名, 参数名)` 寻址，动量状态按**参数全名**保存，
//! 因此热更新迁移权重时，名字对得上的参数自动带走动量（README 7.2 训练侧）。

use crate::api::{MtbResult, Optimizer, Tensor};
use std::collections::HashMap;

/// 优化器种类（配置 `[network.train] optimizer` 的取值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimKind {
    Sgd,
    Momentum,
    Adam,
}

/// 动量 / Adam 的共享状态容器，按参数全名索引。
#[derive(Debug, Default)]
pub struct OptState {
    pub mom: HashMap<String, Vec<f32>>,
    pub vel: HashMap<String, Vec<f32>>,
    pub step: u64,
}

/// 纯 SGD（无动量）。
pub struct Sgd {
    pub lr: f32,
    pub kind: &'static str,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr, kind: "sgd" }
    }
}

impl Optimizer for Sgd {
    fn id(&self) -> &str {
        self.kind
    }
    fn step(&mut self, params: &mut [(String, &mut Tensor, &Tensor)]) {
        for (_, p, g) in params {
            for (pv, gv) in p.data.iter_mut().zip(g.data.iter()) {
                *pv -= self.lr * gv;
            }
        }
    }
}

/// SGD + 动量。
pub struct SgdMomentum {
    pub lr: f32,
    pub momentum: f32,
    state: OptState,
}

impl SgdMomentum {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            state: OptState::default(),
        }
    }
}

impl Optimizer for SgdMomentum {
    fn id(&self) -> &str {
        "momentum"
    }
    fn step(&mut self, params: &mut [(String, &mut Tensor, &Tensor)]) {
        self.state.step += 1;
        for (name, p, g) in params {
            let entry = self.state.mom.entry(name.clone()).or_insert_with(|| vec![0.0; p.data.len()]);
            if entry.len() != p.data.len() {
                entry.resize(p.data.len(), 0.0);
            }
            for ((pv, gv), mv) in
                p.data.iter_mut().zip(g.data.iter()).zip(entry.iter_mut())
            {
                *mv = self.momentum * *mv + gv;
                *pv -= self.lr * *mv;
            }
        }
    }
}

/// Adam（默认优化器）。
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    state: OptState,
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            lr: 0.002,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            state: OptState::default(),
        }
    }
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }
}

impl Optimizer for Adam {
    fn id(&self) -> &str {
        "adam"
    }
    fn step(&mut self, params: &mut [(String, &mut Tensor, &Tensor)]) {
        self.state.step += 1;
        let t = self.state.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for (name, p, g) in params {
            let m = self.state.mom.entry(name.clone()).or_insert_with(|| vec![0.0; p.data.len()]);
            let v = self.state.vel.entry(name.clone()).or_insert_with(|| vec![0.0; p.data.len()]);
            if m.len() != p.data.len() {
                m.resize(p.data.len(), 0.0);
                v.resize(p.data.len(), 0.0);
            }
            for ((pv, gv), (m_i, v_i)) in p.data.iter_mut().zip(g.data.iter()).zip(m.iter_mut().zip(v.iter_mut()))
            {
                *m_i = self.beta1 * *m_i + (1.0 - self.beta1) * gv;
                *v_i = self.beta2 * *v_i + (1.0 - self.beta2) * gv * gv;
                let mhat = *m_i / bc1;
                let vhat = *v_i / bc2;
                *pv -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
        }
    }
}

/// 统一入口：按配置字符串构造优化器。
pub fn build(kind: OptimKind, lr: f32, momentum: f32) -> Box<dyn Optimizer> {
    match kind {
        OptimKind::Sgd => Box::new(Sgd::new(lr)),
        OptimKind::Momentum => Box::new(SgdMomentum::new(lr, momentum)),
        OptimKind::Adam => Box::new(Adam::new(lr)),
    }
}

impl OptimKind {
    pub fn parse(s: &str) -> MtbResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sgd" => Ok(OptimKind::Sgd),
            "momentum" | "sgdm" => Ok(OptimKind::Momentum),
            "adam" => Ok(OptimKind::Adam),
            other => Err(crate::api::MtbError::Config(format!(
                "optimizer: 未知优化器 {other:?}（可用 sgd/momentum/adam）"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Tensor as T;

    #[test]
    fn sgd_moves_param_by_lr() {
        let mut o = Sgd::new(0.1);
        let mut p = T::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let g = T::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        o.step(&mut [("w".into(), &mut p, &g)]);
        assert_eq!(p.data, vec![0.9, 0.9, 0.9]);
    }

    #[test]
    fn adam_is_bounded_and_stateful() {
        let mut o = Adam::default();
        let mut p = T::from_vec(vec![0.0, 0.0], vec![2]).unwrap();
        let g = T::from_vec(vec![2.0, -2.0], vec![2]).unwrap();
        for _ in 0..3 {
            o.step(&mut [("w".into(), &mut p, &g)]);
        }
        assert_eq!(p.data.len(), 2);
        assert!(p.data.iter().all(|x| x.is_finite()));
        // 步长量级应与 lr 同阶
        assert!(p.data[0].abs() < 0.01 && p.data[1].abs() < 0.01);
    }

    #[test]
    fn momentum_accumulates_across_steps() {
        let mut o = SgdMomentum::new(0.1, 0.9);
        let mut p = T::from_vec(vec![0.0], vec![1]).unwrap();
        let g = T::from_vec(vec![1.0], vec![1]).unwrap();
        o.step(&mut [("k".into(), &mut p, &g)]);
        assert_eq!(p.data, vec![-0.1], "首步动量为 1，等价于纯 SGD");
        o.step(&mut [("k".into(), &mut p, &g)]);
        assert_eq!(p.data, vec![-0.29], "第二步应叠加 0.9 倍旧动量");
    }

    #[test]
    fn distinct_params_keep_independent_state() {
        let mut o = SgdMomentum::new(0.1, 0.9);
        let mut a = T::from_vec(vec![0.0], vec![1]).unwrap();
        let mut b = T::from_vec(vec![0.0], vec![1]).unwrap();
        let g = T::from_vec(vec![1.0], vec![1]).unwrap();
        o.step(&mut [("a".into(), &mut a, &g.clone()), ("b".into(), &mut b, &g)]);
        assert_eq!(a.data, b.data, "同一步内不同名字的参数互不干扰");
        let ga = T::from_vec(vec![3.0], vec![1]).unwrap();
        o.step(&mut [("a".into(), &mut a, &ga)]);
        assert_ne!(a.data, b.data, "梯度不同则轨迹必须分开");
    }

    #[test]
    fn parse_optim_kinds() {
        assert_eq!(OptimKind::parse("ADAM").unwrap(), OptimKind::Adam);
        assert!(OptimKind::parse("rmsprop").is_err());
    }
}
