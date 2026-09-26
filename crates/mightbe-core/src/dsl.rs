//! Cell DSL（README 4.1.1 路径 B）：在配置里用受限门控方程定义 RNN 单元。
//!
//! 语法（刻意最小化，保证"前馈门控方程"必然有界、可微）：
//! - 每行一条 `name = expr`，`#` 起注释；
//! - `expr := term (('+' | '-') term)*`，`term := factor ('*' factor)*`，
//!   `factor := number | ident | '-' factor | '(' expr ')' | func '(' expr ')'`；
//! - 白名单函数只有 `sigmoid` / `tanh` / `relu`；
//! - 保留变量 `x`（当前输入）、`h_prev`、`c_prev`（上一时刻状态）；
//! - 大写字母（或 `_`）打头的其余标识符一律是**待训练参数**：出现在 `*` 一侧的是矩阵
//!   （`fan_in × units`，fan_in 由被乘侧宽度推出），单独出现在 `+`/`-` 上的是偏置（`units`）；
//! - 小写打头且未定义的名字直接报错——先定义后使用，写错变量名不会凭空长出参数。
//!
//! 门的位置有语义：`r*(U*h_prev)`（重置门乘在投影之后，与内置 GRU 同式）与
//! `U*(r*h_prev)`（乘在投影之前，另一族 GRU）数值不等价，两种都合法、由方程作者选。
//!
//! 形状约定与 [`crate::layers::Rnn`] 一致：批维度在最外，状态量都是 `(batch, units)`。
//! 解析发生在**网络装载期**，语法/形状错误一律落在 1xxx（语法）错误码段，训练不会带着坏方程启动。
//! 权重归属仍在层里：[`CellSpec::step`] 通过调用方给的 `look` 取「图内键 + 值」，
//! 因此 DSL 层与内置层在热更新 / checkpoint / 拓扑哈希上完全同构。

use crate::api::{Graph, MtbError, MtbResult, Shape, Tensor, Var};
use std::collections::{HashMap, HashSet};

/// 参数在 DSL 里扮演的角色（决定初始化形状）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// `(fan_in, units)` 线性核
    Matrix { fan_in: usize },
    /// `(units,)` 偏置
    Bias,
}

/// 白名单激活。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Func {
    Sigmoid,
    Tanh,
    Relu,
}

impl Func {
    fn name(self) -> &'static str {
        match self {
            Func::Sigmoid => "sigmoid",
            Func::Tanh => "tanh",
            Func::Relu => "relu",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "sigmoid" => Some(Func::Sigmoid),
            "tanh" => Some(Func::Tanh),
            "relu" => Some(Func::Relu),
            _ => None,
        }
    }
}

/// 一条方程右部。标识符按 [`is_param_name`] 分成两类：大写字母打头的是待训练参数
/// （运行时经 `look` 取权重），其余必须是保留变量或前一方程已定义的中间变量。
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    /// `*`：一侧是参数时按矩阵乘解释，否则逐元素乘
    Mul(Box<Expr>, Box<Expr>),
    Call(Func, Box<Expr>),
    Name(String),
    Const(f32),
}

/// 一条赋值方程。
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub target: String,
    pub rhs: Expr,
}

/// 解析后的单元定义。
#[derive(Debug, Clone, PartialEq)]
pub struct CellSpec {
    pub equations: Vec<Equation>,
}

/// 保留变量名。
const RESERVED: [&str; 3] = ["x", "h_prev", "c_prev"];

/// 参数名规则（README 4.1.1）：大写字母或 `_` 打头即待训练参数，其余是变量。
fn is_param_name(s: &str) -> bool {
    matches!(s.chars().next(), Some(c) if c.is_ascii_uppercase() || c == '_')
}

fn syntax(msg: impl Into<String>) -> MtbError {
    MtbError::coded(MtbError::SYNTAX, msg)
}

impl CellSpec {
    /// 解析方程串（README 4.1.1 的 `equations` 字段）。
    pub fn parse(src: &str) -> MtbResult<CellSpec> {
        let mut equations = Vec::new();
        for (lineno, raw) in src.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let eq = Equation::parse(line).map_err(|e| match e {
                MtbError::Coded { code, message } => {
                    MtbError::coded(code, format!("第 {} 行: {message}", lineno + 1))
                }
                other => other,
            })?;
            equations.push(eq);
        }
        let spec = CellSpec { equations };
        spec.validate()?;
        Ok(spec)
    }

    /// 结构校验：方程非空、lhs 不重复、变量先定义后使用、必须以 `h` 收尾（`c` 可选）。
    fn validate(&self) -> MtbResult<()> {
        if self.equations.is_empty() {
            return Err(syntax("equations 为空，无法定义 Cell"));
        }
        let mut defined: HashSet<&str> = RESERVED.iter().copied().collect();
        let mut targets: HashSet<&str> = HashSet::new();
        for eq in &self.equations {
            check_uses(&eq.rhs, &defined)?;
            if !targets.insert(eq.target.as_str()) {
                return Err(syntax(format!("变量 {} 被重复赋值", eq.target)));
            }
            defined.insert(eq.target.as_str());
        }
        if !defined.contains("h") {
            return Err(syntax("方程组必须以 `h = ...` 结束（Cell 的隐状态）"));
        }
        Ok(())
    }

    /// 装载期收集全部待训练参数：`(参数名, 角色)`，按名字典序输出以便 checkpoint 对齐。
    pub fn params(&self, input_dim: usize, units: usize) -> MtbResult<Vec<(String, ParamKind)>> {
        let mut defined: HashSet<String> = RESERVED.iter().map(|s| s.to_string()).collect();
        let mut kinds: HashMap<String, ParamKind> = HashMap::new();
        for eq in &self.equations {
            collect_params(&eq.rhs, &defined, input_dim, units, &mut kinds)?;
            defined.insert(eq.target.clone());
        }
        let mut names: Vec<String> = kinds.keys().cloned().collect();
        names.sort();
        Ok(names
            .into_iter()
            .map(|n| {
                let k = kinds.remove(&n).expect("刚收集到");
                (n, k)
            })
            .collect())
    }

    /// 参数形状视图，层的 `Init::make` 直接用。
    pub fn param_shape(units: usize, kind: ParamKind) -> Shape {
        match kind {
            ParamKind::Matrix { fan_in } => vec![fan_in, units],
            ParamKind::Bias => vec![units],
        }
    }

    /// 规范化文本（无空白、全括号）：拓扑哈希与 `describe()` 用它区分不同方程组。
    pub fn canonical(&self) -> String {
        self.equations
            .iter()
            .map(|e| format!("{}={}", e.target, e.rhs))
            .collect::<Vec<String>>()
            .join(";")
    }

    /// 展开一个时间步：把方程翻译成图上的算子序列。
    ///
    /// `look(参数名) -> (图内键, 值)` 由调用方（`Rnn` 层）提供：权重仍归层所有、
    /// 图内键带层名前缀，因此同一网络里两个 DSL 层的同名参数不会共享节点，
    /// DSL 层与内置层在热更新/迁移上也完全同构。
    pub fn step<F>(
        &self,
        g: &mut Graph,
        x_t: Var,
        h_prev: Var,
        c_prev: Var,
        look: F,
    ) -> MtbResult<(Var, Var)>
    where
        F: Fn(&str) -> (String, Tensor),
    {
        let mut env: HashMap<String, Var> = HashMap::new();
        env.insert("x".into(), x_t);
        env.insert("h_prev".into(), h_prev);
        env.insert("c_prev".into(), c_prev);
        for eq in &self.equations {
            let v = self.eval(g, &eq.rhs, &look, &env)?;
            env.insert(eq.target.clone(), v);
        }
        let h = *env
            .get("h")
            .ok_or_else(|| syntax("方程组没有产出 h"))?;
        let c = env.get("c").copied().unwrap_or(h);
        Ok((h, c))
    }

    fn eval<F>(
        &self,
        g: &mut Graph,
        e: &Expr,
        look: &F,
        env: &HashMap<String, Var>,
    ) -> MtbResult<Var>
    where
        F: Fn(&str) -> (String, Tensor),
    {
        Ok(match e {
            Expr::Const(c) => g.constant(Tensor::from_vec(vec![*c], vec![]).expect("标量")),
            Expr::Name(n) => match env.get(n.as_str()) {
                Some(v) => *v,
                None => {
                    let (key, t) = look(n);
                    g.param(key, t)
                }
            },
            Expr::Add(a, b) => {
                let va = self.eval(g, a, look, env)?;
                let vb = self.eval(g, b, look, env)?;
                g.add(va, vb)
            }
            Expr::Sub(a, b) => {
                let va = self.eval(g, a, look, env)?;
                let vb = self.eval(g, b, look, env)?;
                g.sub(va, vb)
            }
            Expr::Mul(a, b) => {
                // 某一侧是参数名即矩阵乘；否则逐元素乘
                if let Some((name, other)) = matmul_side(a, b, env) {
                    let v = self.eval(g, other, look, env)?;
                    let (key, t) = look(name);
                    let w = g.param(key, t);
                    return Ok(g.matmul(v, w));
                }
                let va = self.eval(g, a, look, env)?;
                let vb = self.eval(g, b, look, env)?;
                g.mul(va, vb)
            }
            Expr::Call(f, a) => {
                let v = self.eval(g, a, look, env)?;
                match f {
                    Func::Sigmoid => g.sigmoid(v),
                    Func::Tanh => g.tanh(v),
                    Func::Relu => g.relu(v),
                }
            }
        })
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Add(a, b) => write!(f, "({a}+{b})"),
            Expr::Sub(a, b) => write!(f, "({a}-{b})"),
            Expr::Mul(a, b) => write!(f, "({a}*{b})"),
            Expr::Call(func, a) => write!(f, "{}({a})", func.name()),
            Expr::Name(n) => f.write_str(n),
            Expr::Const(c) => write!(f, "{c}"),
        }
    }
}

/// 若 `a * b` 的一侧是参数名，返回 `(参数名, 另一侧)`。
fn matmul_side<'a>(
    a: &'a Expr,
    b: &'a Expr,
    env: &HashMap<String, Var>,
) -> Option<(&'a str, &'a Expr)> {
    let is_param = |e: &Expr| match e {
        Expr::Name(n) => is_param_name(n) && !env.contains_key(n.as_str()),
        _ => false,
    };
    match (a, b) {
        (Expr::Name(n), _) if is_param(a) => Some((n.as_str(), b)),
        (_, Expr::Name(n)) if is_param(b) => Some((n.as_str(), a)),
        _ => None,
    }
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// 变量先用后定义检查（README 4.1.1 校验项：小写名必须是保留变量或已定义的中间变量）。
fn check_uses(e: &Expr, defined: &HashSet<&str>) -> MtbResult<()> {
    match e {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            check_uses(a, defined)?;
            check_uses(b, defined)
        }
        Expr::Call(_, a) => check_uses(a, defined),
        Expr::Name(n) if !is_param_name(n) && !defined.contains(n.as_str()) => {
            Err(undefined_var(n))
        }
        _ => Ok(()),
    }
}

fn undefined_var(n: &str) -> MtbError {
    syntax(format!(
        "变量 {n} 未定义：需先赋值后使用（待训练参数要大写字母或 `_` 打头）"
    ))
}

/// 静态收集参数：矩阵的 fan_in 由被乘侧宽度推出，独立出现的按偏置处理。
fn collect_params(
    e: &Expr,
    defined: &HashSet<String>,
    input_dim: usize,
    units: usize,
    kinds: &mut HashMap<String, ParamKind>,
) -> MtbResult<()> {
    match e {
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            collect_params(a, defined, input_dim, units, kinds)?;
            collect_params(b, defined, input_dim, units, kinds)
        }
        Expr::Mul(a, b) => {
            let param_side = |x: &Expr| match x {
                Expr::Name(n) if is_param_name(n) && !defined.contains(n) => Some(n.clone()),
                _ => None,
            };
            match (param_side(a), param_side(b)) {
                (Some(n), None) | (None, Some(n)) => {
                    let other = if param_side(a).is_some() { &**b } else { &**a };
                    let w = width_of(other, defined, input_dim, units)?;
                    declare(kinds, &n, ParamKind::Matrix { fan_in: w })?;
                    collect_params(other, defined, input_dim, units, kinds)
                }
                (Some(x), Some(y)) => {
                    Err(syntax(format!("参数 {x} 与 {y} 相乘无法推出形状")))
                }
                (None, None) => {
                    collect_params(a, defined, input_dim, units, kinds)?;
                    collect_params(b, defined, input_dim, units, kinds)
                }
            }
        }
        Expr::Call(_, a) => collect_params(a, defined, input_dim, units, kinds),
        Expr::Name(n) if defined.contains(n) => Ok(()),
        // 参数名且不在乘法里出现 → 偏置
        Expr::Name(n) if is_param_name(n) => declare(kinds, n, ParamKind::Bias),
        Expr::Name(n) => Err(undefined_var(n)),
        Expr::Const(_) => Ok(()),
    }
}

/// 表达式的特征宽度（状态量一律 units，`x` 是 input_dim）。
fn width_of(
    e: &Expr,
    defined: &HashSet<String>,
    input_dim: usize,
    units: usize,
) -> MtbResult<usize> {
    match e {
        Expr::Name(n) if n == "x" => Ok(input_dim),
        Expr::Name(n) if defined.contains(n) => Ok(units),
        // 乘法的另一侧不会是参数（上一步已判），其余未定义名按 units 处理
        Expr::Name(_) => Ok(units),
        Expr::Const(_) => Ok(units),
        Expr::Add(a, _) | Expr::Sub(a, _) | Expr::Mul(a, _) | Expr::Call(_, a) => {
            width_of(a, defined, input_dim, units)
        }
    }
}

fn declare(
    kinds: &mut HashMap<String, ParamKind>,
    name: &str,
    kind: ParamKind,
) -> MtbResult<()> {
    match kinds.get(name) {
        None => {
            kinds.insert(name.to_string(), kind);
            Ok(())
        }
        Some(prev) if *prev == kind => Ok(()),
        Some(prev) => Err(syntax(format!(
            "参数 {name} 形状冲突：{prev:?} 与 {kind:?}"
        ))),
    }
}

impl Equation {
    fn parse(line: &str) -> MtbResult<Equation> {
        let (lhs, rhs) = match line.split_once('=') {
            Some(p) => p,
            None => return Err(syntax("每条方程必须形如 `name = expr`")),
        };
        let target = lhs.trim();
        if !is_ident(target) {
            return Err(syntax(format!("非法赋值目标 {target:?}")));
        }
        if rhs.contains('=') {
            return Err(syntax("一行只允许一个赋值"));
        }
        let mut p = Parser::new(rhs)?;
        let expr = p.parse_expr()?;
        p.expect_end()?;
        Ok(Equation { target: target.to_string(), rhs: expr })
    }
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ───────────────────────── 词法 ─────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Num(f32),
    Plus,
    Minus,
    Star,
    Open,
    Close,
}

fn tokenize(src: &str) -> MtbResult<Vec<Tok>> {
    let mut out = Vec::new();
    let b = src.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        match c {
            ' ' | '\t' | '\r' | '\n' => i += 1,
            '+' => { out.push(Tok::Plus); i += 1 }
            '-' => { out.push(Tok::Minus); i += 1 }
            '*' => { out.push(Tok::Star); i += 1 }
            '(' => { out.push(Tok::Open); i += 1 }
            ')' => { out.push(Tok::Close); i += 1 }
            d if d.is_ascii_digit() || d == '.' => {
                let s = i;
                while i < b.len() && ((b[i] as char).is_ascii_digit() || b[i] as char == '.') {
                    i += 1;
                }
                let num: f32 = src[s..i]
                    .parse()
                    .map_err(|_| syntax(format!("数字字面量 {:?} 非法", &src[s..i])))?;
                out.push(Tok::Num(num));
            }
            a if a.is_ascii_alphabetic() || a == '_' => {
                let s = i;
                while i < b.len()
                    && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_')
                {
                    i += 1;
                }
                out.push(Tok::Name(src[s..i].to_string()));
            }
            other => return Err(syntax(format!("非法字符 {other:?}"))),
        }
    }
    Ok(out)
}

// ───────────────────────── 语法分析（递归下降） ─────────────────────────

struct Parser {
    toks: Vec<Tok>,
    at: usize,
}

impl Parser {
    fn new(src: &str) -> MtbResult<Parser> {
        Ok(Parser { toks: tokenize(src)?, at: 0 })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at)
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expect_end(&self) -> MtbResult<()> {
        match self.peek() {
            None => Ok(()),
            Some(other) => Err(syntax(format!("多余的内容 {other:?}"))),
        }
    }

    fn parse_expr(&mut self) -> MtbResult<Expr> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.at += 1;
                    lhs = Expr::Add(Box::new(lhs), Box::new(self.parse_term()?));
                }
                Some(Tok::Minus) => {
                    self.at += 1;
                    lhs = Expr::Sub(Box::new(lhs), Box::new(self.parse_term()?));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> MtbResult<Expr> {
        let mut lhs = self.parse_factor()?;
        while self.eat(&Tok::Star) {
            lhs = Expr::Mul(Box::new(lhs), Box::new(self.parse_factor()?));
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> MtbResult<Expr> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.at += 1;
                Ok(Expr::Const(n))
            }
            Some(Tok::Open) => {
                self.at += 1;
                let inner = self.parse_expr()?;
                if !self.eat(&Tok::Close) {
                    return Err(syntax("括号未闭合"));
                }
                Ok(inner)
            }
            Some(Tok::Minus) => {
                self.at += 1;
                let inner = self.parse_factor()?;
                Ok(Expr::Sub(Box::new(Expr::Const(0.0)), Box::new(inner)))
            }
            Some(Tok::Name(name)) => {
                self.at += 1;
                if self.eat(&Tok::Open) {
                    let f = Func::parse(&name)
                        .ok_or_else(|| syntax(format!("函数 {name:?} 不在白名单")))?;
                    let arg = self.parse_expr()?;
                    if !self.eat(&Tok::Close) {
                        return Err(syntax("函数调用缺少右括号"));
                    }
                    return Ok(Expr::Call(f, Box::new(arg)));
                }
                Ok(Expr::Name(name))
            }
            Some(Tok::Plus) | Some(Tok::Star) | Some(Tok::Close) => {
                Err(syntax("表达式不完整"))
            }
            None => Err(syntax("表达式不完整")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// README 4.1.1 示例：手写 GRU。
    const GRU: &str = "
        z = sigmoid(Wz*x + Uz*h_prev + Bz)
        r = sigmoid(Wr*x + Ur*h_prev + Br)
        h_cand = tanh(W*x + r*(U*h_prev) + B)
        h = (1-z)*h_prev + z*h_cand
    ";

    #[test]
    fn gru_equations_parse() {
        let spec = CellSpec::parse(GRU).unwrap();
        assert_eq!(spec.equations.len(), 4);
        assert_eq!(spec.equations[3].target, "h");
    }

    #[test]
    fn gru_declares_nine_params() {
        let spec = CellSpec::parse(GRU).unwrap();
        let p = spec.params(5, 8).unwrap();
        assert_eq!(
            p,
            vec![
                ("B".into(), ParamKind::Bias),
                ("Br".into(), ParamKind::Bias),
                ("Bz".into(), ParamKind::Bias),
                ("U".into(), ParamKind::Matrix { fan_in: 8 }),
                ("Ur".into(), ParamKind::Matrix { fan_in: 8 }),
                ("Uz".into(), ParamKind::Matrix { fan_in: 8 }),
                ("W".into(), ParamKind::Matrix { fan_in: 5 }),
                ("Wr".into(), ParamKind::Matrix { fan_in: 5 }),
                ("Wz".into(), ParamKind::Matrix { fan_in: 5 }),
            ],
            "README 说这个例子应有 9 个参数，且 U* 吃状态、W* 吃输入"
        );
    }

    #[test]
    fn canonical_form_is_whitespace_free_and_ordered() {
        let spec = CellSpec::parse(GRU).unwrap();
        assert_eq!(
            spec.canonical(),
            "z=sigmoid((((Wz*x)+(Uz*h_prev))+Bz));\
             r=sigmoid((((Wr*x)+(Ur*h_prev))+Br));\
             h_cand=tanh((((W*x)+(r*(U*h_prev)))+B));\
             h=(((1-z)*h_prev)+(z*h_cand))",
            "规范化文本是拓扑哈希的输入，必须稳定且无空白"
        );
    }

    #[test]
    fn missing_h_assignment_is_rejected() {
        let e = CellSpec::parse("z = sigmoid(Wz*x)\ng = tanh(z)").unwrap_err();
        assert!(format!("{e}").contains("h ="), "应提示缺少 h，实际 {e}");
    }

    #[test]
    fn empty_program_is_rejected() {
        let e = CellSpec::parse("   \n# 只有注释\n").unwrap_err();
        assert!(format!("{e}").contains("为空"), "实际 {e}");
    }

    #[test]
    fn double_assignment_is_rejected() {
        let e = CellSpec::parse("h = x\nh = x").unwrap_err();
        assert!(format!("{e}").contains("重复赋值"), "实际 {e}");
    }

    #[test]
    fn undefined_variable_is_rejected() {
        // 打错一个字母的变量名不能被静默当成参数建权重
        let e = CellSpec::parse("z = sigmoid(Wz*x)\nh = tanh(zz + z)").unwrap_err();
        assert!(format!("{e}").contains("zz 未定义"), "实际 {e}");
    }

    #[test]
    fn unknown_function_is_rejected() {
        let e = CellSpec::parse("h = gelush(x)").unwrap_err();
        assert!(format!("{e}").contains("白名单"), "实际 {e}");
    }

    #[test]
    fn unbalanced_paren_is_rejected() {
        let e = CellSpec::parse("h = sigmoid(Wz*x").unwrap_err();
        assert!(format!("{e}").contains("括号"), "实际 {e}");
    }

    #[test]
    fn second_equals_is_rejected() {
        let e = CellSpec::parse("h = z = x").unwrap_err();
        assert!(format!("{e}").contains("一个赋值"), "实际 {e}");
    }

    #[test]
    fn bad_param_reuse_is_rejected() {
        // W 既当矩阵又当偏置用，形状无法自洽；冲突在装载期收集参数时报出
        let spec = CellSpec::parse("h = tanh(W*x) + W").unwrap();
        let e = spec.params(3, 4).unwrap_err();
        assert!(format!("{e}").contains("形状冲突"), "实际 {e}");
    }

    #[test]
    fn two_params_multiplied_is_rejected() {
        let spec = CellSpec::parse("h = tanh(W * V)").unwrap();
        let e = spec.params(3, 4).unwrap_err();
        assert!(format!("{e}").contains("无法推出形状"), "实际 {e}");
    }

    #[test]
    fn line_number_is_reported() {
        let e = CellSpec::parse("h = sigmoid(x)\nbad line here").unwrap_err();
        assert!(format!("{e}").contains("第 2 行"), "实际 {e}");
    }

    #[test]
    fn comment_and_blank_lines_are_skipped() {
        let spec = CellSpec::parse("# 只有一行\n\nh = x  # 结尾\n").unwrap();
        assert_eq!(spec.equations.len(), 1);
    }

    #[test]
    fn illegal_char_is_rejected() {
        let e = CellSpec::parse("h = x $ 2").unwrap_err();
        assert!(format!("{e}").contains("非法字符"), "实际 {e}");
    }

    #[test]
    fn unary_minus_and_grouping_work() {
        let spec = CellSpec::parse("h = (0 - x) + (1 - 2)").unwrap();
        let const_of = |e: &Expr| match e {
            Expr::Const(c) => *c,
            other => panic!("期望常量，实际 {other:?}"),
        };
        match &spec.equations[0].rhs {
            Expr::Add(l, r) => {
                assert!(matches!(**l, Expr::Sub(_, _)), "一元负号应展开成减法");
                let (a, b) = match &**r {
                    Expr::Sub(a, b) => (a, b),
                    other => panic!("右部应是减法，实际 {other:?}"),
                };
                assert_eq!(const_of(a), 1.0);
                assert_eq!(const_of(b), 2.0);
            }
            other => panic!("期望加法，实际 {other:?}"),
        }
    }

    /// DSL 展开必须能在图上跑通并把梯度一路送回全部参数。
    #[test]
    fn custom_cell_runs_and_backprops() {
        let spec = CellSpec::parse(GRU).unwrap();
        let (d, u) = (3usize, 4usize);
        let params: HashMap<String, Tensor> = spec
            .params(d, u)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, (n, k))| {
                let sh = CellSpec::param_shape(u, k);
                let data: Vec<f32> = (0..sh.iter().product::<usize>())
                    .map(|j| 0.07 * (j + i) as f32 - 0.2)
                    .collect();
                (n, Tensor::from_vec(data, sh).unwrap())
            })
            .collect();
        let look =
            |n: &str| (format!("enc.{n}"), params.get(n).cloned().unwrap_or_else(Tensor::default));

        let mut g = Graph::new();
        let x = g.constant(
            Tensor::from_vec((0..6).map(|i| i as f32 * 0.2).collect(), vec![2, d]).unwrap(),
        );
        let h0 = g.constant(Tensor::zeros(vec![2, u]));
        let c0 = g.constant(Tensor::zeros(vec![2, u]));
        let (h, c) = spec.step(&mut g, x, h0, c0, look).unwrap();
        assert_eq!(g.value(h).shape, vec![2, u]);
        assert_eq!(g.value(c).shape, vec![2, u]);
        assert!(g.value(h).data.iter().all(|v| v.is_finite()));

        g.backward(&[h]).unwrap();
        let mut trained: Vec<String> = g
            .trainable()
            .filter(|(_, _, _, grad)| grad.is_some())
            .map(|(_, name, _, _)| name.to_string())
            .collect();
        trained.sort();
        assert_eq!(
            trained,
            vec![
                "enc.B", "enc.Br", "enc.Bz", "enc.U", "enc.Ur", "enc.Uz", "enc.W", "enc.Wr",
                "enc.Wz"
            ],
            "9 个参数都应收到梯度，且图内键由调用方命名（多 custom 层不撞名）"
        );
    }
}
