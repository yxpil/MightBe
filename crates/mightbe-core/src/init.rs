//! 权重初始化（README 5 节 `init = "..."` 字段的实现）。
//!
//! 全部走可复现的 seeded RNG：同一 `seed` + 同一形状 + 同一策略 ⇒ 同一权重，
//! 保证训练可复现（README 4.2 `seed` 字段）。

use crate::api::{MtbError, MtbResult, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// 可复现随机数发生器。
pub fn rng_from(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// 初始化策略。配置里的字符串形式见 [`Init::parse`]。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// `uniform:<a>:<b>` — [a, b) 均匀分布（默认 -0.1..0.1）
    Uniform { a: f32, b: f32 },
    /// `normal:<mean>:<std>`
    Normal { mean: f32, std: f32 },
    /// `xavier` — 按 (fan_in + fan_out) 缩放
    Xavier,
    /// `zeros` / `ones`
    Zeros,
    Ones,
}

impl Default for Init {
    fn default() -> Self {
        Init::Uniform { a: -0.1, b: 0.1 }
    }
}

impl Init {
    pub fn parse(s: &str) -> MtbResult<Self> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("uniform:") {
            let parts: Vec<f32> = rest.split(':').map(|x| x.trim().parse()).collect::<Result<_, _>>()
                .map_err(|_| MtbError::Config(format!("init: 无法解析 uniform: {rest}")))?;
            if parts.len() != 2 {
                return Err(MtbError::Config(format!("init: uniform 需要 2 个参数: {s}")));
            }
            return Ok(Init::Uniform { a: parts[0], b: parts[1] });
        }
        if let Some(rest) = s.strip_prefix("normal:") {
            let parts: Vec<f32> = rest.split(':').map(|x| x.trim().parse()).collect::<Result<_, _>>()
                .map_err(|_| MtbError::Config(format!("init: 无法解析 normal: {rest}")))?;
            if parts.len() != 2 {
                return Err(MtbError::Config(format!("init: normal 需要 2 个参数: {s}")));
            }
            return Ok(Init::Normal { mean: parts[0], std: parts[1] });
        }
        match s {
            "xavier" | "glorot" => Ok(Init::Xavier),
            "zeros" => Ok(Init::Zeros),
            "ones" => Ok(Init::Ones),
            "" => Ok(Init::default()),
            other => Err(MtbError::Config(format!("init: 未知策略 {other:?}"))),
        }
    }

    /// 生成形状为 `shape` 的初始权重。
    pub fn make(&self, shape: &[usize], seed: u64) -> Tensor {
        let n = shape.iter().product::<usize>().max(1);
        let mut rng = rng_from(seed);
        let mut data = Vec::with_capacity(n);
        match self {
            Init::Zeros => data.resize(n, 0.0),
            Init::Ones => data.resize(n, 1.0),
            Init::Uniform { a, b } => {
                for _ in 0..n {
                    data.push(rng.gen_range(*a..*b) as f32);
                }
            }
            Init::Normal { mean, std } => {
                // Box–Muller：两个均匀分布合成一个正态分布，避免引入 rand_distr。
                let mut i = 0;
                while i < n {
                    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
                    let u2: f32 = rng.gen_range(0.0..1.0);
                    let r = (-2.0 * u1.ln()).sqrt();
                    data.push(mean + std * (r * (2.0 * std::f32::consts::PI * u2).cos()));
                    if i + 1 < n {
                        data.push(mean + std * (r * (2.0 * std::f32::consts::PI * u2).sin()));
                    }
                    i += 2;
                }
            }
            Init::Xavier => {
                // Glorot 均匀分布：端点只与 fan_in + fan_out 有关，故与轴的先后顺序无关。
                let fan_out = *shape.last().unwrap_or(&1);
                let fan_in: usize = shape[..shape.len().saturating_sub(1)].iter().product::<usize>().max(1);
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                for _ in 0..n {
                    data.push(rng.gen_range(-limit..limit) as f32);
                }
            }
        }
        Tensor::from_vec(data, shape.to_vec()).expect("init: 形状与长度不一致")
    }
}

/// 便捷入口：按形状与策略造权重。
pub fn make(shape: &[usize], init: &Init, seed: u64) -> Tensor {
    init.make(shape, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strategies() {
        assert_eq!(Init::parse("xavier").unwrap(), Init::Xavier);
        assert!(Init::parse("uniform:-0.1:0.1").unwrap() == Init::Uniform { a: -0.1, b: 0.1 });
        assert!(Init::parse("").is_ok());
        assert!(Init::parse("bogus").is_err());
    }

    #[test]
    fn same_seed_is_reproducible() {
        let a = Init::Xavier.make(&[4, 8], 42);
        let b = Init::Xavier.make(&[4, 8], 42);
        let c = Init::Xavier.make(&[4, 8], 43);
        assert_eq!(a.data, b.data, "同种子必须一致（训练可复现）");
        assert_ne!(a.data, c.data);
    }

    #[test]
    fn uniform_bounds_hold() {
        let t = Init::Uniform { a: -0.1, b: 0.1 }.make(&[64], 7);
        assert_eq!(t.len(), 64);
        assert!(t.data.iter().all(|x| (-0.1..0.1).contains(x)));
    }

    #[test]
    fn xavier_follows_glorot_bound() {
        let t = Init::Xavier.make(&[16, 32], 5);
        let limit = (6.0f32 / 48.0).sqrt();
        assert!(t.data.iter().all(|x| x.abs() <= limit));
        // 端点只依赖 fan_in + fan_out：转置形状给出同一分布
        let u = Init::Xavier.make(&[32, 16], 5);
        assert_eq!(t.data, u.data);
    }

    #[test]
    fn normal_is_not_uniform() {
        let t = Init::Normal { mean: 0.0, std: 1.0 }.make(&[256], 11);
        assert_eq!(t.len(), 256);
        assert!(t.data.iter().all(|x| x.is_finite()));
        assert!(t.data.iter().any(|x| x.abs() > 0.1), "正态分布应出现远离均值样本");
    }
}
