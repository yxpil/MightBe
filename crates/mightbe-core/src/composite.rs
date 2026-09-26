//! 路径 A：Composite 复合层（README 4.1.1）——零 Rust 代码的配置式自定义层。
//!
//! 一个 composite 层内部是一张子层 DAG：每个子层用 `from = ["..."]` 声明输入来自
//! 哪个端口（`@input` = 本层输入，其余为子层名）。框架按拓扑序展开进全局 autograd
//! 图，`forward` 里不做任何手工反向。

use crate::api::{
    Layer, LayerCtx, LayerSpec, MtbError, MtbResult, Shape,
};
use crate::autograd::{Graph, Var};
use std::collections::BTreeMap;

/// 子层声明。字段值统一用字符串，便于直接来自 TOML。
#[derive(Debug, Clone)]
pub struct SubSpec {
    pub name: String,
    /// 层类型：`conv1d` / `globalmaxpool1d` / `concat` / ...
    pub kind: String,
    /// 其余参数以 `key = "value"` 形式保留（供具体层解析）
    pub params: BTreeMap<String, String>,
    /// 输入来源列表
    pub from: Vec<String>,
}

impl SubSpec {
    pub fn num(&self, key: &str, default: usize) -> usize {
        self.params
            .get(key)
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    }
    pub fn opt_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }
}

/// 复合层：只持有子层定义与实际子层实例。
pub struct Composite {
    pub name: String,
    pub subs: Vec<SubSpec>,
    /// 输出端口名；缺省取最后一个子层
    pub output: Option<String>,
    pub layers: Vec<(String, Box<dyn Layer>)>,
}

impl Composite {
    pub fn new(name: &str, subs: Vec<SubSpec>) -> Self {
        Self {
            name: name.to_string(),
            subs,
            output: None,
            layers: Vec::new(),
        }
    }

    /// 由外部（net crate）装配子层实例。
    pub fn attach(&mut self, name: &str, layer: Box<dyn Layer>) {
        self.layers.push((name.to_string(), layer));
    }

    fn index_of(&self, name: &str) -> MtbResult<usize> {
        self.layers
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| MtbError::Config(format!("composite: 未注册的子层 {name:?}")))
    }

}

impl Layer for Composite {
    fn name(&self) -> &str {
        &self.name
    }

    fn param_names(&self) -> Vec<String> {
        self.layers.iter().flat_map(|(_, l)| l.param_names()).collect()
    }

    fn bind_params(&mut self, params: &[(String, crate::api::Tensor)]) -> MtbResult<()> {
        for (full, t) in params {
            // 参数名形如 `textcnn.c3.weight`，按前缀派发给对应子层
            let rest = full.strip_prefix(&format!("{}.", self.name))
                .ok_or_else(|| MtbError::Config(format!("composite: 参数 {full} 不属于本层")))?;
            let sub_name = rest.split('.').next().unwrap_or("").to_string();
            let idx = self.index_of(&sub_name)?;
            self.layers[idx].1.bind_params(&[(rest[sub_name.len() + 1..].to_string(), t.clone())])?;
        }
        Ok(())
    }

    fn dump_params(&self) -> Vec<(String, crate::api::Tensor)> {
        let mut out = Vec::new();
        for (_, l) in &self.layers {
            for (k, v) in l.dump_params() {
                out.push((format!("{}.{}", self.name, k), v));
            }
        }
        out
    }

    fn forward(
        &mut self,
        args: &[&Var],
        ctx: &LayerCtx,
        graph: &mut Graph,
    ) -> MtbResult<Vec<Var>> {
        // `name -> 该子层的输出`；子层必须按拓扑序展开（README 4.1.1）
        let order = topo_sort(&self.subs, self.output.as_deref())?;
        let mut port: BTreeMap<String, Vec<Var>> = BTreeMap::new();
        port.insert("@input".to_string(), args.iter().copied().copied().collect());

        for &si in &order {
            let sub = self.subs[si].clone();
            let resolved: Vec<Var> = sub
                .from
                .iter()
                .map(|src| {
                    port.get(src)
                        .and_then(|v| v.first().copied())
                        .ok_or_else(|| {
                            MtbError::Config(format!(
                                "composite: 子层 {} 的输入端口 {} 不存在（或有多路输出）",
                                sub.name, src
                            ))
                        })
                })
                .collect::<MtbResult<Vec<_>>>()?;
            let refs: Vec<&Var> = resolved.iter().collect();
            let idx = self.index_of(&sub.name)?;
            let outs = self.layers[idx].1.forward(&refs, ctx, graph)?;
            port.insert(sub.name.clone(), outs);
        }

        let out_name = self.output.clone().unwrap_or_else(|| {
            self.subs.last().map(|s| s.name.clone()).unwrap_or_default()
        });
        let outs = port
            .get(&out_name)
            .cloned()
            .ok_or_else(|| MtbError::Config(format!("composite: 输出端口 {out_name} 不存在")))?;
        Ok(outs)
    }

    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let order = topo_sort(&self.subs, self.output.as_deref())?;
        let mut port: BTreeMap<String, Vec<Shape>> = BTreeMap::new();
        port.insert(
            "@input".to_string(),
            inputs.iter().map(|s| (*s).clone()).collect(),
        );
        for &si in &order {
            let sub = &self.subs[si];
            let ins: Vec<&Shape> = sub
                .from
                .iter()
                .map(|s| {
                    port.get(s)
                        .and_then(|v| v.first())
                        .ok_or_else(|| {
                            MtbError::Config(format!("composite: 输入端口 {s} 不存在"))
                        })
                })
                .collect::<MtbResult<Vec<_>>>()?;
            let idx = self.index_of(&sub.name)?;
            let outs = self.layers[idx].1.infer_shapes(&ins)?;
            port.insert(sub.name.clone(), outs);
        }
        let out_name = self.output.clone().unwrap_or_else(|| {
            self.subs.last().map(|s| s.name.clone()).unwrap_or_default()
        });
        port.get(&out_name).cloned().ok_or_else(|| {
            MtbError::Config(format!("composite: 输出端口 {out_name} 不存在"))
        })
    }

    fn describe(&self) -> LayerSpec {
        // 键排序后拼接：保证拓扑哈希稳定（README 6 节）
        let mut parts = Vec::new();
        for s in &self.subs {
            let mut keys: Vec<&String> = s.params.keys().collect();
            keys.sort();
            let kv: Vec<String> = keys
                .iter()
                .map(|k| format!("{k}={}", s.params[*k]))
                .collect();
            parts.push(format!(
                "{}:({})<{}>",
                s.name,
                s.kind,
                kv.join(",")
            ));
        }
        LayerSpec::new("composite", &self.name, &parts.join("|"))
    }
}

/// 拓扑排序 + 环检测（README 5 节校验项：composite 的 `from` 必须存在且 DAG 无环）。
pub fn topo_sort(subs: &[SubSpec], output: Option<&str>) -> MtbResult<Vec<usize>> {
    if subs.is_empty() {
        return Err(MtbError::Config("composite: 没有任何子层".into()));
    }
    let names: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
    let out_name = output.unwrap_or_else(|| names.last().copied().unwrap_or(""));
    let mut state = vec![0u8; subs.len()]; // 0=未访问 1=在栈 2=已完成

    fn visit(
        i: usize,
        subs: &[SubSpec],
        names: &[&str],
        state: &mut Vec<u8>,
        order: &mut Vec<usize>,
    ) -> MtbResult<()> {
        match state[i] {
            1 => {
                return Err(MtbError::Config(format!(
                    "composite: 子层 {} 存在环",
                    names[i]
                )))
            }
            2 => return Ok(()),
            _ => {}
        }
        state[i] = 1;
        for src in &subs[i].from {
            if src == "@input" || src == "@state" || src == "@output" {
                continue;
            }
            let idx = names
                .iter()
                .position(|n| *n == src)
                .ok_or_else(|| {
                    MtbError::Config(format!("composite: 子层 {} 引用了不存在的 {}", names[i], src))
                })?;
            visit(idx, subs, names, state, order)?;
        }
        state[i] = 2;
        order.push(i);
        Ok(())
    }

    let mut sorted = Vec::new();
    let end = names.iter().position(|n| *n == out_name).unwrap_or(subs.len() - 1);
    visit(end, subs, &names, &mut state, &mut sorted)?;
    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_sort_respects_dependencies() {
        let subs = vec![
            SubSpec {
                name: "cat".into(),
                kind: "concat".into(),
                params: BTreeMap::new(),
                from: vec!["p3".into(), "p4".into()],
            },
            SubSpec {
                name: "p3".into(),
                kind: "globalmaxpool1d".into(),
                params: BTreeMap::new(),
                from: vec!["c3".into()],
            },
            SubSpec {
                name: "c3".into(),
                kind: "conv1d".into(),
                params: BTreeMap::new(),
                from: vec!["@input".into()],
            },
            SubSpec {
                name: "p4".into(),
                kind: "globalmaxpool1d".into(),
                params: BTreeMap::new(),
                from: vec!["c3".into()],
            },
        ];
        let order = topo_sort(&subs, Some("cat")).unwrap();
        let pos: Vec<&str> = order.iter().map(|&i| subs[i].name.as_str()).collect();
        assert_eq!(pos, vec!["c3", "p3", "p4", "cat"], "必须先于依赖者被展开");
    }

    #[test]
    fn cycle_is_rejected() {
        let subs = vec![
            SubSpec {
                name: "a".into(),
                kind: "dense".into(),
                params: BTreeMap::new(),
                from: vec!["b".into()],
            },
            SubSpec {
                name: "b".into(),
                kind: "dense".into(),
                params: BTreeMap::new(),
                from: vec!["a".into()],
            },
        ];
        assert!(topo_sort(&subs, Some("a")).is_err());
    }

    #[test]
    fn missing_port_is_rejected() {
        let subs = vec![SubSpec {
            name: "a".into(),
            kind: "dense".into(),
            params: BTreeMap::new(),
            from: vec!["ghost".into()],
        }];
        assert!(topo_sort(&subs, Some("a")).is_err());
    }

    #[test]
    fn describe_is_deterministic() {
        let subs = vec![SubSpec {
            name: "c3".into(),
            kind: "conv1d".into(),
            params: [("filters".to_string(), "64".to_string()), ("kernel".to_string(), "3".to_string())].into_iter().collect(),
            from: vec!["@input".into()],
        }];
        let c = Composite::new("textcnn", subs);
        let a = c.describe().json.clone();
        let b = c.describe().json.clone();
        assert_eq!(a, b, "拓扑描述必须稳定（拓扑哈希用）");
        assert!(a.contains("c3"));
    }
}
