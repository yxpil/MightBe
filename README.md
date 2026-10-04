# MightBe

> 一个用 Rust 从零实现的**神经网络实时修改框架**，同时也是一个**类 MySQL 的神经网络特征训练数据库**。
>
> 用户自定义网络结构（输入头格式、层类型如 RNN / LSTM / MLP、输出头），系统对入库文本自动提取关键词、做特征向量化、**联想检索与可解释推理（多跳关联 + 证据 + 置信度）**；网络结构与权重可在服务运行期间热修改。**第一阶段仅支持 CPU**。

---

## 目录

1. [项目定位与一句话介绍](#1-项目定位)
2. [核心概念（术语表）](#2-核心概念)
3. [总体架构：模块化单体](#3-总体架构模块化单体)
4. [模块详细设计](#4-模块详细设计)
   - 4.1 [mightbe-core：张量与自动微分引擎](#41-mightbe-core张量与自动微分引擎)
   - 4.2 [mightbe-net：网络定义、头格式与运行时](#42-mightbe-net网络定义头格式与运行时)
   - 4.3 [mightbe-nlp：分词 / 关键词 / 联想](#43-mightbe-nlp分词--关键词--联想)
   - 4.4 [mightbe-store：存储引擎](#44-mightbe-store存储引擎)
   - 4.5 [mightbe-sql：方言解析与执行计划](#45-mightbe-sql方言解析与执行计划)
   - 4.6 [mightbe-server：网络服务与会话](#46-mightbe-server网络服务与会话)
   - 4.7 [mightbe-cli：命令行客户端](#47-mightbe-cli命令行客户端)
   - 4.8 [mightbe-reason：联想检索与合理推理](#48-mightbe-reason联想检索与合理推理)
5. [网络定义（头格式）规范](#5-网络定义头格式规范)
6. [模型文件二进制布局](#6-模型文件二进制布局)
7. [实时修改机制（热更新）](#7-实时修改机制热更新)
8. [NLP 流水线、联想检索与合理推理](#8-nlp-流水线联想检索与合理推理)
9. [SQL 方言完整定义](#9-sql-方言完整定义)
10. [端到端使用示例（走查）](#10-端到端使用示例走查)
11. [存储引擎内部设计](#11-存储引擎内部设计)
   - 11.1 [MDB 数据库文件格式（二进制 + 加密）](#111-mdb-数据库文件格式二进制--加密)
   - 11.2 [存储目录布局与持久化清单](#112-存储目录布局与持久化清单)
12. [线程模型与并发控制](#12-线程模型与并发控制)
13. [配置与数据目录布局](#13-配置与数据目录布局)
14. [错误处理、日志与可观测性](#14-错误处理日志与可观测性)
15. [依赖策略](#15-依赖策略)
16. [测试策略](#16-测试策略)
17. [开发里程碑](#17-开发里程碑)
18. [非目标与边界](#18-非目标与边界)

---

## 1. 项目定位

**MightBe = 数据库 + 可热修改的神经网络训练引擎 + 自动 NLP 特征管线**

传统做法里，用户要自己拼装：数据库、分词器、TF-IDF、词向量、训练循环、模型服务、模型版本管理。MightBe 把这些装进**一个服务进程**，用熟悉的类 SQL 接口暴露：

```sql
CREATE NETWORK doc_rnn FROM 'config/networks/doc_rnn.toml';
INSERT INTO articles(title, body) VALUES ('Rust 学习笔记', '今天学习了所有权与借用……');
TRAIN NETWORK doc_rnn ON articles;
SELECT keyword FROM mb_keywords WHERE doc_id = 1;
SELECT word, score FROM ASSOCIATE(doc_rnn, '所有权');
ALTER NETWORK doc_rnn ADD LAYER { type = "lstm", units = 128 } AFTER encoder;
```

设计原则：

1. **模块化单体（Modular Monolith）**：一个可部署二进制，进程内模块通过 trait 接口通信；模块边界严格，未来可拆分为微服务而不改业务代码。
2. **配置即定义**：网络结构、头格式、停用词、训练超参全部放在配置文件 / 数据文件中，**绝不硬编码**。
3. **从零实现核心**：张量运算、自动微分、存储页与 WAL 自己写，不绑定大型 ML 框架；仅借助少量基础库（序列化、异步运行时、CLI 解析）。
4. **CPU First**：第一阶段全部算子走 CPU（BLAS 加速作为可插拔后端，默认纯 Rust 实现）。
5. **MySQL 亲和**：系统表、表/行概念、TCP 访问方式向 MySQL 靠拢；远期支持 MySQL 有线协议，可用 `mysql` 客户端直连。

---

## 2. 核心概念

| 术语 | 含义 |
|---|---|
| **Database / Table** | 与 MySQL 类似的库、表。表有 schema，文本列会被 NLP 管线自动处理 |
| **Network** | 一个用户定义的神经网络，由配置文件描述，有名字与版本号 |
| **Header（头格式）** | 网络的输入头规格：特征来源（tfidf / token_ids / dense）、维度、序列长度、归一化方式 |
| **Head（输出头）** | 网络末端的任务头：dense + softmax（分类）、dense（回归）、projection（向量召回） |
| **Layer** | 内置层：`dense` / `rnn` / `gru` / `lstm` / `conv1d` / `pool1d` / `embedding` / `dropout` / `layernorm` / `flatten` / `reshape` / `concat` / `activation` |
| **Composite Layer（复合层）** | 用户不写 Rust，仅用配置把子层与算子按 DAG 组合出自定义层（如 TextCNN），自动获得反向传播 |
| **Cell DSL** | 受限安全表达式语言，用户在配置中书写自定义 RNN 门控方程，参数与梯度由框架自动生成 |
| **Layer Plugin（层插件）** | 用户用 Rust 实现 `mightbe-plugin` SDK 编译出的动态库（`.dll/.so/.dylib`），运行时热加载的原生自定义层 |
| **SchemaGraph（模式图）** | 关系学习产物：Field / Table 节点与 `fkey` / `correlated` / `semantic` / `functional` 等边，与词项推理图融合 |
| **Field Relation（字段关系）** | 两个字段之间被学习出的关系类型与置信度：外键、语义同名、统计相关、函数依赖等，带值级证据 |
| **Network Version** | 每次结构修改产生一个新版本；权重按"层名 + 形状"迁移 |
| **Vocabulary（词表）** | 分词后 token → 整数 ID 的映射，持久化在系统表中 |
| **Keyword** | 经 TF-IDF / TextRank 打分从文本中抽出的词 |
| **Association（联想）** | 词与词、文档与文档之间基于共现与训练向量的关联度（余弦相似度 / 共现 PMI） |
| **AssocEdge（联想边）** | 推理图上的带类型边：`cooc`（共现 PMI）、`emb`（词向量余弦）、`appears_in`（词→文档）、`labeled`（文档→类别） |
| **Reasoning Path（推理路径）** | 从查询节点到结论节点的一条多跳边序列，每跳带分数与证据 |
| **Evidence（证据）** | 支撑某条边/某个结论的真实文档 id 与原文片段，结论必须可溯源，禁止无证据结论 |
| **Confidence（置信度）** | 路径分 × 验证器分 × 深度衰减后、经多路径 Noisy-OR 合并与校准的最终把握度 |
| **Abstain（弃判）** | 最高置信度仍低于配置阈值时返回"无可靠结论"而非勉强作答——"合理推理"的硬约束 |
| **Training Job** | 一次训练任务：指定网络、数据源表、超参；后台异步执行 |
| **Catalog（系统表）** | `mb_*` 前缀的系统表，存元数据、网络拓扑、词表、关键词、训练状态 |
| **Page / WAL** | 存储页（默认 8 KiB）与预写日志，保证崩溃恢复 |
| **MDB 文件** | MightBe 统一持久格式（二进制、分页、AEAD 加密），扩展名 `.mdb` / `.mbm` / `.mbdb` / `.mwlog` |
| **KEK / DEK** | 密钥加密密钥（由口令 Argon2id 派生）/ 数据密钥（随机生成、加密落库）；口令永不直接加密数据 |
| **VECTOR(n)** | SQL 定长向量值类型（f32 段），用于 `dot/cosine/nearest` 等数学与向量检索 |
| **Session** | 一条客户端连接，持有当前数据库上下文与事务 |

---

## 3. 总体架构（模块化单体）

一个 Cargo workspace、多个内部 crate、**最终编译为单个二进制 `mightbe`**。模块之间只通过对方暴露的 `api` trait + DTO 通信，不跨模块访问内部结构。

```
                          ┌──────────────────────────────────────────┐
                          │            mightbe  (单一二进制)           │
                          │                                          │
  TCP :9527  ┌────────────┤  mightbe-server   会话/协议/线程调度       │
  (文本协议,  │            │      │                                   │
   远期MySQL │            │      ▼                                   │
   wire)     │            │  mightbe-sql   词法/语法解析 → 计划 → 执行 │
             │            │      │                                   │
             │            │      ├──────────┬──────────┬──────────┐  │
             │            │      ▼          ▼          ▼          ▼  │
             │            │    net        nlp       reason      store│
             │            │  网络/热更新  关键词     联想/推理   页/WAL│
             │            │      │          │          │          │  │
             │            │      └─────┬────┴──────────┴────┬─────┘  │
             │            │            ▼                    ▼        │
             │            │       mightbe-core（Tensor / Autograd /  │
             │            │             Layer / Optimizer，纯CPU）   │
             │            └──────────────────────────────────────────┘
             │                          │
             │                    config/  data/  logs/
             └─ mightbe-cli（同仓库，瘦客户端，可独立编译为单独 exe）
```

**模块依赖方向（只允许向下依赖，禁止环）**：

```
server ──► sql ──► net ────► core
              │     ├─► nlp ──► core
              │     ├─► reason ──► net / nlp / store（只读视图）
              │     └─► store ──► core
        (net / nlp / reason 都只通过 trait 访问 store，由 server 在启动时
         装配依赖注入；reason 不写入任何模块，是纯查询/推理层)
```

**为什么是模块化单体而不是微服务**：训练与查询需要频繁共享词表、模型快照与页缓存，进程内调用零网络开销；单二进制部署简单；严格的模块边界（每个 crate 只 `pub use api::*`）保证未来需要拆分时，把 trait 实现换成 RPC 客户端即可。

Workspace 布局：

```
MightBe/
├─ Cargo.toml                 # workspace 清单
├─ crates/
│  ├─ mightbe-core/           # 张量、自动微分、层、优化器、插件宿主
│  ├─ mightbe-plugin/         # 层插件 SDK（trait/宏/HostApi，供用户编 cdylib）
│  ├─ mightbe-store/          # 存储引擎
│  ├─ mightbe-nlp/            # NLP 管线
│  ├─ mightbe-net/            # 网络定义与运行时
│  ├─ mightbe-reason/         # 联想检索、多跳推理、模式关系学习
│  ├─ mightbe-sql/            # SQL 方言
│  ├─ mightbe-server/         # 服务端（装配所有模块）
│  └─ mightbe-cli/            # 命令行客户端
├─ plugins/                   # 用户编译的层动态库与 manifest（.dll/.so/.toml）
├─ config/
│  ├─ mightbe.toml            # 服务全局配置
│  ├─ networks/               # 用户网络定义（头格式），含内置 _relation_net.toml
│  └─ nlp/                    # 停用词等 NLP 数据
├─ data/                      # 运行期数据（页文件、WAL、模型快照）
└─ tests/                     # 跨 crate 端到端测试（含测试用 cdylib 插件）
```

---

## 4. 模块详细设计

### 4.1 mightbe-core：张量与自动微分引擎

纯 CPU、`no_std` 不需要但零系统绑定，单线程算子 + 外层并行批处理。

**核心类型：**

```rust
// crates/mightbe-core/src/tensor.rs（示意，非最终代码）
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

pub struct Var {                    // 自动微分变量
    value: Tensor,
    grad: Option<Tensor>,
    op: Option<OpHandle>,          // 计算图节点（反向传播用）
}
```

**运算覆盖（MVP）：**

| 类别 | 算子 |
|---|---|
| 逐元素 | add / sub / mul / div、neg、sigmoid / tanh / relu / softmax（数值稳定版） |
| 矩阵 | matmul、transpose、bias_add |
| 规约 | sum、mean、max（含 axis 参数，用于 RNN 时间维） |
| 形状 | reshape、concat、slice、permute（轴置换，用于通道序转换）、pad（序列变长处理） |
| 损失 | mse、cross_entropy（log_softmax + nll 合并实现） |

**自动微分：反向模式（reverse-mode autograd）**

- 前向时构建轻量计算图：每个输出 `Var` 持有输入引用与算子枚举；
- `backward()` 拓扑排序后逐节点写梯度；
- 图用 arena 索引而非 `Rc` 环，便于训练步结束后整体丢弃（显存/内存随 step 释放）。

**内置层库（全部从零实现，CPU）：**

| 类别 | 层 | 说明 |
|---|---|---|
| 全连接 | `dense` | `y = activation(W·x + b)` |
| 嵌入 | `embedding` | 查表层，参数即词向量矩阵（联想能力的来源） |
| 循环 | `rnn` / `gru` / `lstm` | SimpleRNN、两门 GRU、四门 LSTM（门合并为一次 matmul）；GRU 候选态取标准式 `tanh(W_c·x + B_c + r ⊙ (U_c·h_prev))`（重置门只作用于历史项）；均支持 `return_sequences` 与 `bidirectional` 包装 |
| 卷积 | `conv1d` | 时序/文本一维卷积（多卷积核、padding=same/valid、步长、膨胀 dilation） |
| 池化 | `maxpool1d` / `avgpool1d` / `globalmaxpool1d` / `globalavgpool1d` | 定长与全局池化 |
| 结构 | `flatten` / `reshape` / `concat` / `permute` | 多分支网络（如多尺度 CNN）合并；`permute` 负责 `(batch, seq, ch)` ↔ `(batch, ch, seq)` 的通道序转换 |
| 归一化/正则 | `dropout` / `layernorm` | 训练/推理双模式；LayerNorm 统计量在线计算 |
| 激活 | `activation`（relu/sigmoid/tanh/gelu/softmax） | 可独立成层，数值稳定实现 |
| 注意力（M9） | `self_attention` | 预留枚举与配置位，第二期实现 |

**Layer 抽象（内置层、复合层、插件层共用同一接口）：**

```rust
pub trait Layer: Send {
    fn name(&self) -> &str;                      // 命名参数，热更新权重迁移的锚点
    fn infer_shapes(&self, inputs: &[Shape]) -> MtbResult<Vec<Shape>>;
    fn forward(&mut self, args: &[&Var], ctx: &LayerCtx) -> Vec<Var>; // 支持多入多出 DAG
    fn params(&self) -> Vec<ParamRef>;           // 权重/偏置及名称
    fn describe(&self) -> LayerSpec;             // 可序列化的结构描述（拓扑哈希用）
}
pub struct LayerCtx { pub training: bool, pub seq: Option<SeqState> } // RNN 隐状态载体
```

**用户自定义层的三条路径（能力递增，详见 4.1.1）：**

1. **Composite（配置式复合层，零代码）**：在 TOML 中声明若干内置子层与连线，框架按 DAG 前向/反向——TextCNN、双塔编码器等都能这样拼；
2. **Cell DSL（受限表达式自定义 RNN 单元）**：在配置里写门控方程（只允许线性组合、逐元素运算和白名单激活），框架解析方程、自动建参数、自动求导；
3. **Plugin（Rust 原生动态库）**：用 `mightbe-plugin` SDK 实现任意前向/反向逻辑，编译为 cdylib，服务运行时热加载，可在 ALTER NETWORK 中直接引用。

**优化器：** SGD（含 momentum）、Adam（默认）。参数按 `(网络名, 层名, 参数名)` 寻址，为热更新权重迁移服务。

**计算后端可插拔（CPU → SIMD → GPU/NPU）：** 所有数值运算（matmul、conv、逐元素、规约、翻译查表、LSTM 门）都经统一 `Backend` trait 派发，网络按 `device` 配置选择执行设备：

```rust
pub trait Backend: Send + Sync {
    fn id(&self) -> &str;                       // "naive" | "simd-avx2" | "cuda:0" | "npu:0"
    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], g: &GemmShapes) -> MtbResult<()>;
    fn conv1d(&self, x: &[f32], w: &[f32], bias: &[f32], out: &mut [f32], p: &ConvParams) -> MtbResult<()>;
    fn elementwise(&self, out: &mut [f32], op: ElemOp) -> MtbResult<()>;
    fn reduce(&self, x: &[f32], shape: &[usize], out: &mut [f32], axis: usize, r: ReduceKind) -> MtbResult<()>;
    fn embedding_lookup(&self, emb: &[f32], ids: &[u32], out: &mut [f32], dim: usize) -> MtbResult<()>;
}
```

派发契约（M1 起生效，由 `graph_dispatches_all_kernels_through_backend` 测试锁定）：

- `Graph` 自带 `Arc<dyn Backend>`，`Graph::new()` 取 `naive`，`Graph::with_backend(..)` 由 `[network.compute].device` 经 `select_backend` 注入；前向的 matmul / conv1d / 逐元素 / 规约 / 查表五类核**只**从后端走，autograd 不再留一份平行实现；
- `conv1d`/`reduce` 的批维度与轴语义在 autograd 侧展开（后端只见单样本 `(in_ch, in_len)` 与显式 `shape`），换后端时不需要重新发明批处理；
- 反向是后端无关的：`local_grads` 只读节点缓存的前向值，因此任何后端只要前向数值一致，梯度即一致（对拍时只比前向输出）；
- 节点值以 `Arc<Tensor>` 共享，算子入图只增引用计数；广播按 NumPy 右对齐规则，轴长既不相等也非 1 时直接报错，绝不静默回绕。

| 后端 | 状态 | 说明 |
|---|---|---|
| `naive` | MVP 基线 | 纯 Rust + 缓存友好分块，零依赖，任何 CPU 可编译运行 |
| `simd` | M2 | 运行时检测 AVX2/AVX-512/NEON 后走手写宽打包内核（不依赖 nightly `std::simd`），向后自动回退 naive |
| `cuda` | M9 feature `gpu` | `cudarc` 绑定，权重常驻显存，H2D/D2H 仅在 step 边界同步 |
| `opencl` | M9 feature `gpu` | `opencl3`，覆盖集显/AMD/移动端 GPU |
| `npu` | M9 feature `npu` | 经厂商 SDK FFI（RKNN / OpenVINO / QNN 三选一，编译期选定），以算子子图下沉到 NPU |

设备选择与约束：

- 每个网络在配置 `[network.compute]` 声明 `device = "cpu" | "simd" | "cuda:0" | "npu:0" | "auto"`；`auto` 按"有 GPU 优先 GPU、其次 NPU、最后 SIMD/naive"选择，并在 `SHOW NETWORKS` 显示实际后端 id；
- **CPU 是唯一权威数据源**：设备显存/片上缓存只是副本，checkpoint 落盘、推理返回、关系学习读画像一律走 CPU 张量；设备间同步点固定为"step 边界"与"快照发布"两处，MVP 用整图同步（保证正确优先于吞吐）；
- **梯度镜像**：各后端必须实现同一份 backward 语义，单元测试以 naive 为基准对拍所有后端（容差按 f32 累加误差设定）；
- **降级策略**：设备不可用/驱动缺失时启动告警并回退到更慢后端；step 执行中设备报错（OOM、kernel 失败）则整个 step 失败 → 网络标记 Failed → 其余网络继续服务，绝不因设备问题导致进程崩溃；
- **插件兼容**：Host API 的函数指针表与设备绑定，插件可声明 `requires = "cpu"` 固定后端（跨设备 ABI 传裸指针必是 CPU 内存）；
- **可观测**：`SHOW STATUS` 暴露后端 id、设备利用率估算、H2D/D2H 字节数、每算子耗时（profiling 可按 `[engine] profile = true` 开启）。

M9 之前（含 MVP 全阶段）发布物**不依赖任何 GPU/NPU SDK**：加速能力全部位于 feature gate 之后，默认构建是纯 CPU。

#### 4.1.1 用户自定义层机制（三条路径）

框架不对"层的种类"设上限，按用户愿意付出的成本提供三种扩展方式。三者产出的层在网络看来没有区别：同样进拓扑哈希、同样支持权重迁移与热更新。

**路径 A：Composite 复合层（零 Rust 代码，配置即结构）**

一个 composite 层内部是一张子层 DAG：每个子层用 `from = ["..."]` 声明输入来自网络头或哪个子层，支持多入多出与 `concat` 合并。框架直接把它展开进全局 autograd 图，因此**无需用户写反向传播**。

```toml
[[network.layers]]
name = "textcnn"
type = "composite"
# 用三种窗口的一维卷积捕捉多尺度短语（自定义"TextCNN 层"）
[[network.layers.sub]]
  name = "c3"; type = "conv1d"; from = ["@input"]; filters = 64; kernel = 3; activation = "relu"
[[network.layers.sub]]
  name = "c4"; type = "conv1d"; from = ["@input"]; filters = 64; kernel = 4; activation = "relu"
[[network.layers.sub]]
  name = "c5"; type = "conv1d"; from = ["@input"]; filters = 64; kernel = 5; activation = "relu"
[[network.layers.sub]]
  name = "p3"; type = "globalmaxpool1d"; from = ["c3"]
[[network.layers.sub]]
  name = "p4"; type = "globalmaxpool1d"; from = ["c4"]
[[network.layers.sub]]
  name = "p5"; type = "globalmaxpool1d"; from = ["c5"]
[[network.layers.sub]]
  name = "cat"; type = "concat"; from = ["p3", "p4", "p5"]; axis = -1
output = "cat"
```

- 参数名自动层级化：`textcnn.c3.weight`，热更新时即使复合层内部增删分支，未变动子层的权重仍可迁移；
- 复合层可嵌套（composite 里引用另一个 composite 名），展开器在校验期做拓扑排序与环检测；
- `@input` / `@state` / `@output` 为保留端口，分别代表层输入、RNN 上一时刻状态、本层输出声明。

**路径 B：Cell DSL（配置中写自定义 RNN 单元）**

`rnn` 层的 `cell` 字段除 `simple/gru/lstm` 外可取 `custom`，用受限方程描述门控。语法只允许：`+ - *`（一侧是参数时按矩阵乘解释，否则逐元素乘）、白名单激活 `sigmoid/tanh/relu`、数字字面量、保留变量 `x`（当前输入）/`h_prev`/`c_prev`（上一时刻状态）。标识符按首字母大小写分家：**大写字母（或 `_`）打头 = 待训练参数**，自动定形——出现在 `*` 一侧的是矩阵，形状 `fan_in × units`（`fan_in` 由被乘侧宽度推出：`x` 侧为 `input_dim`、状态侧为 `units`），只出现在 `+`/`-` 上的是偏置向量 `units`；其余标识符必须是保留变量或前一方程已定义的中间变量（**先定义后使用**，写错名字直接报错而不是凭空长出一个参数）。

```toml
[[network.layers]]
name = "encoder"
type = "rnn"
units = 64
cell = "custom"
equations = '''
  z = sigmoid(Wz*x + Uz*h_prev + Bz)
  r = sigmoid(Wr*x + Ur*h_prev + Br)
  h_cand = tanh(W*x + r*(U*h_prev) + B)
  h = (1-z)*h_prev + z*h_cand
'''
# 上例即手写 GRU；框架据此创建 9 个参数并把方程接入 autograd
```

注意 `r*(U*h_prev)`：重置门乘在**投影之后**的历史上（`r ⊙ (U·h_prev)`，与内置 GRU、PyTorch 同式），
而不是 `U*(r*h_prev)`（那是另一族 GRU，`r` 乘在投影之前）——两种写法数值不等价，
只有前者能与内置 GRU 对拍一致。

- DSL 解析器在**网络装载期**完成：词法/语法分析 → 形状推导（所有矩阵形状由 `input_dim × units` 推出）→ 展开成与内置 GRU 同构的算子序列；语法/形状错误一律落在 1xxx 段并带行号；
- 参数仍由层持有、按 `<网络名>.<层名>.<参数名>` 寻址（图内节点键为 `<层名>.<参数名>`）：DSL 层与内置层在热更新权重、拓扑哈希、checkpoint 上完全同构，同一网络里两个 custom 层各用各的 `Wz` 也不会撞名；
- 安全约束：不允许循环/分支/函数调用/外部符号，只能表达"前馈门控方程"，因此计算必然有界、可微分；
- 校验项：每行单赋值、变量先定义后使用、最终必须赋值 `h`（可选 `c` 用于 LSTM 式细胞态）、维度相容、参数定形无冲突。装载期 DSL 与内置 GRU 对拍数值（相同权重下前向结果一致）作为测试基线。

**路径 C：Layer Plugin（Rust 原生动态库，最强扩展）**

当复合层与方程 DSL 表达不了时（如新的卷积变体、自定义注意力、外部算法），用 SDK crate `mightbe-plugin` 写原生层：

```rust
// 用户侧示意（SDK 提供 `macro_rules!` 导出宏，见下方约束）
mb_export! {
    @name = "atss_conv", @api_version = 1,
    @layer = AtssConv,   // 实现 MbLayer 的类型
}

pub struct AtssConv { channels: usize, /* 插件自己的状态 */ }

impl MbLayer for AtssConv {
    fn init(params: LayerParams) -> Self;
    fn infer_shapes(&self, in: &[Shape]) -> Vec<Shape>;
    fn forward(&self, args: &[HostTensor], host: &HostApi) -> Vec<HostTensor>;
    fn backward(&self, grad_out: &[HostTensor], host: &HostApi) -> Vec<HostTensor>;
}
```

ABI 与生命周期约定：

- **稳定 C ABI**：动态库导出 `mb_api_version / mb_layer_count / mb_describe / mb_create / mb_forward / mb_backward / mb_params / mb_drop / mb_last_error`。跨 ABI 只传版本化的 `HostTensor`（`dims` 定长数组 + f32 裸指针 + 长度，秩上限与 `MAX_PERM_RANK` 一致）与 JSON 参数；主机分配的输出一律经 `HostRaw` 回传，由主机 `mb_free` 释放，插件不得跨边界持有内存；
- **可导出宏而非过程宏**：`mb_export!` 是 `macro_rules!`。过程宏需要 `syn/quote`，它们在 15 节白名单外，故 SDK 不用 `#[mb_layer(...)]` 属性宏；
- **自实现加载器**：宿主不用 `libloading`——该 crate 在离线构建环境不可得，`mightbe-core::plugin` 直接以 `LoadLibraryW`/`GetProcAddress`（Windows）与 `dlopen`/`dlsym`（POSIX）的 FFI 声明实现，两者由 `#[cfg(windows)]` 分派；
- **参数走图，不走插件**：插件的权重以 `mb_params` 名称清单声明，主机侧把它们建成 `Graph` 的 `param` 节点混排在 `forward` 的 `args` 里；插件反向只需按同样顺序返回这些参数的梯度，主机经既有的梯度累积路径收敛。因此插件层与内置层在热更新权重、optimizer 状态、checkpoint 上完全同构，插件自身不必持有任何权重；
- **v1 限制**：单输出（多输出列入远期）；`requires = "cpu"`，跨 ABI 传裸指针必是 CPU 内存；
- **装载**：`CREATE LIBRARY mylayers FROM 'plugins/mylayers.dll';`，旁边放同名 manifest（层名、参数 schema、形状约束）。装载时校验 api_version、清单与导出符号（3101 版本不符 / 3102 符号缺失 / 3103 冒烟失败）；不匹配则拒绝，旧库继续可用；
- **热替换**：`ALTER LIBRARY mylayers RELOAD;` 走与网络热更新相同的"构建新实例→冒烟前向→Arc 原子替换"流程；引用计数归零前不卸载旧库（Windows 下 `FreeLibrary` 延迟到无快照引用），避免句柄冲突；
- **故障隔离**：所有跨 ABI 调用包 `catch_unwind`，插件 panic/非法形状/NaN 转为 `3xxx/4xxx` 错误（并可用 `mb_last_error` 取回插件侧消息）且自动把该网络标记为 Failed，但**不拖垮进程**，其余网络照常服务；
- **安全边界（明确非目标）**：插件是用户自己编译的可信本地代码，框架不做沙箱；不保证恶意插件的内存安全（C ABI 本质不安全）。签名校验与 WASM 化沙箱列入远期。

三路径选择指引（写进网络配置文档）：能用内置层 → 内置；需要新连线/多分支 → Composite；需要新门控 → Cell DSL；需要全新算子 → Plugin。

---

### 4.2 mightbe-net：网络定义、头格式与运行时

职责：把用户配置解析为可训练网络；管理多版本拓扑；提供**无锁快照读取**与**原子切换**。

```rust
pub struct Network {
    spec: NetworkSpec,                 // 来自配置文件的完整定义
    layers: Vec<Box<dyn Layer>>,       // 按拓扑顺序
    version: u32,
    state: NetworkState,               // Loading/Ready/Training/Failed
}
pub struct NetworkRegistry {
    networks: DashMap<String, Arc<NetworkSnapshot>>, // 读多写少
}
pub struct NetworkSnapshot {           // 不可变快照（推理用）
    version: u32,
    graph: Arc<BuiltGraph>,            // 所有权重 + 层
}
```

关键 API（对 sql/server 暴露的 `api`）：

| API | 作用 |
|---|---|
| `load_network(path) -> NetworkId` | 读配置 → 校验 → 建网络 → 注册 |
| `infer(net, Batch) -> Outputs` | 走当前快照前向 |
| `train_step(net, Batch) -> Loss` | 前向+反向+优化器更新（训练线程内） |
| `alter_network(net, AlterOp) -> Version` | 热修改，返回新版本号 |
| `snapshot(net) -> Arc<Snapshot>` | 取当前不可变快照（存检查点 / 推理） |

构建时会根据 Header 的特征类型向 nlp / store 订阅特征；Head 的 loss 决定训练目标。

---

### 4.3 mightbe-nlp：分词 / 关键词 / 联想

三层管线，全部数据驱动（停用词、标点表均放 `config/nlp/`）：

```
原文 ──► Normalizer ──► Tokenizer ──► Vocab ──► 特征编码器
                                      │
                                      ├─► KeywordScorer (TF-IDF / TextRank)
                                      └─► AssociationGraph (共现 + 训练向量)
```

1. **Normalizer**：Unicode NFKC、全半角统一、空白归一、大小写（按配置）。
2. **Tokenizer（从零实现，无外部 NLP 库）**：
   - 拉丁/数字：正则等价的手写状态机切词；
   - 中日韩：单字成词 + 二元（bigram）组合候选，由词频统计在词表内动态过滤；
   - 标点与停用词：读取 `config/nlp/stopwords.zh.txt` 等外部文件。
3. **Vocabulary**：`token ↔ id`，含 `<pad>/<unk>`，min_freq / max_size 走配置；词表持久化到系统表 `mb_vocab`，增量合并。
4. **FeatureEncoder（对应头格式）**：
   - `token_ids`：定长/变长 ID 序列（喂给 RNN / Embedding）；
   - `tfidf`：稀疏 TF-IDF 向量（喂给 MLP）；
   - `dense`：外部给定数值列直接拼接。
5. **KeywordScorer**：
   - TF-IDF：IDF 由全库文档频率（存 `mb_df`）计算；
   - TextRank：句/词共现图 + PageRank 迭代（可选，配置切换）；
   - 结果写 `mb_keywords(doc_id, word, score, algo)`。
6. **Association**：
   - 冷启动：滑动窗口共现统计 → PMI / 共现频次，存 `mb_assoc(word_a, word_b, pmi, cooc)`；
   - 训练后：取 Embedding 层词向量做余弦相似度，`ASSOCIATE()` 查询返回 `(word, score)`；
   - 两种分数可按配置加权融合。

---

### 4.4 mightbe-store：存储引擎

**自己实现的小型关系存储引擎**，不嵌入 SQLite/RocksDB。

```
Database
 ├─ Catalog（库/表/列/网络/训练任务元数据，mb_* 系统表）
 ├─ Table
 │   ├─ PageManager（堆文件 heap*.pag，8KiB 页，页首有 LSN）
 │   ├─ RowStore（定长槽位 + 变长文本溢出页 overflow*.pag）
 │   └─ Index（MVP：内存哈希索引 + 主键 B+Tree 第二期）
 ├─ Wal（append-only redo log，组提交）
 └─ BufferPool（LRU 页缓存，脏页刷盘由 checkpoint 驱动）
```

- **页格式**：`PageHeader { magic, page_id, lsn, free_ptr, kind } + Cell[]`，Cell 放序列化行。
- **WAL**：物理 redo（页号 + 变更字节），先写日志后改页；checkpoint 时把脏页刷盘并截断已确认段。崩溃恢复：重放 WAL 到最后一个完整事务。
- **事务**：MVP 提供读已提交（RC）：写操作在提交时持表级写锁；读走快照版本行（旧版本保留在溢出页直到无事务引用——简化版 MVCC，第二期）。
- **序列化**：自定义紧凑二进制编码 `MtbEncode`（f32/变长 int/长度前缀字节），模型矩阵另走模型文件格式。
- **系统表（首次启动自动初始化）**：

| 系统表 | 内容 |
|---|---|
| `mb_databases` / `mb_tables` / `mb_columns` | 元数据 |
| `mb_networks` | 网络名、版本、配置路径、拓扑哈希、状态、当前检查点 |
| `mb_network_versions` | 每个版本的拓扑 JSON、创建时间、变更说明 |
| `mb_vocab` | token、id、文档频率、词频 |
| `mb_keywords` | 文档关键词及分数 |
| `mb_assoc` | 词间共现/PMI 联想边 |
| `mb_training_jobs` | 任务 id、网络、数据源、进度、loss 曲线、状态 |
| `mb_checkpoints` | 模型快照文件路径、版本、校验和 |
| `mb_calibration` | 每网络的置信度校准参数（温度 / 保序分段表）、PMI 归一化分位数、校准时间 |
| `mb_reason_log` | REASON 查询的诊断记录（可选开关）：查询、最高置信度、是否弃判、跳数、耗时，用于调参与审计 |
| `mb_libraries` | 插件库：动态库路径、manifest 哈希、api_version、导出层名、加载状态 |
| `mb_field_profiles` | 字段画像：类型、基数、空值率、值分布摘要、直方图、脏标记与画像时间 |
| `mb_field_relations` | 字段关系：两端字段、关系类型、置信度、支撑行数、值级证据 JSON |
| `mb_table_relations` | 表关系：两表、桥接字段、聚合置信度、跨表 token 重叠度 |
| `mb_relation_jobs` | 关系学习任务：范围、全量/增量、扫描字段对数、进度、状态 |

---

### 4.5 mightbe-sql：方言解析与执行计划

自研递归下降解析器（不引入 sql-parser 库，保证方言可控）：

```
SQL 文本 → Lexer → AST → Planner → LogicalPlan → Executor → ResultSet
```

- **Lexer**：大小写不敏感关键字、标识符、字符串、数字、花括号结构体字面量（用于 ALTER）。
- **支持语句**（完整语法见 [第 9 节](#9-sql-方言完整定义)）：
  - 数据定义：`CREATE DATABASE / TABLE / NETWORK / LIBRARY`、`ALTER NETWORK / LIBRARY`、`DROP`
  - 数据操作：`INSERT`（文本列自动触发 NLP 管线）、`SELECT`（含 `WHERE / ORDER BY / LIMIT`、向量列与数学函数）
  - 训练：`TRAIN NETWORK ... ON ... [OPTIONS(...)]`、`SHOW TRAINING`、`CALIBRATE NETWORK`
  - 关系学习：`LEARN RELATIONS`、`LEARN RULES`、`SHOW RELATIONS / FIELD RELATIONS / RULES`、`CONFIRM/REJECT RELATION`
  - 联想与推理：表函数 `ASSOCIATE` / `NEAR` / `INFER` / `REASON`（含 `SCOPE`）/ `REASON_PATHS`
  - 加密与运维：`KEY OPEN/CLOSE`、`LOCK KEYS`、`BACKUP/RESTORE DATABASE ... ENCRYPTED`、`ROTATE KEY`
  - 可观测：`EXPLAIN REASON/SELECT/ASSOCIATE`、`SHOW NETWORKS / VERSIONS / LIBRARIES / KEYWORDS / VOCAB / STATUS`
- **执行器**：火山模型（`next() -> Row`），NLP/训练类语句翻译成对 net/nlp/store 模块的 API 调用。

---

### 4.6 mightbe-server：网络服务与会话

- 单二进制子命令 `mightbe serve` 启动；
- **协议（MVP）**：TCP `127.0.0.1:9527`，行分隔文本协议：
  - 请求：一条 SQL（以 `;` 结束）
  - 响应：`OK <cols>\n` + 制表符/管道分隔行 + `END (rows=N, ms=X)`；错误为 `ERR <code> <message>`；
  - 长任务（TRAIN）立即返回 job id，进度用 `SHOW TRAINING <id>` 轮询；
- **装配方式（依赖注入）**：启动时构建 `Backend`（core）→ `Store` → `NlpPipeline` → `NetworkRegistry` → `SqlEngine`，全部以 trait object 注入，模块间无全局单例（配置对象除外，配置在启动后只读）；
- 启动流程：加载 `config/mightbe.toml` → 恢复系统表 → 重放 WAL → 按 `mb_libraries` 加载插件动态库（失败则相关网络标记 Failed 并继续启动）→ 加载已注册网络最新检查点 → 监听端口；
- 优雅停机：SIGINT/ctrl-c → 停止接收 → 训练步落检查点 → WAL flush → 退出。

### 4.7 mightbe-cli：命令行客户端

- 独立 bin：`mightbe-cli -h 127.0.0.1 -P 9527`，进入交互式 shell（readline 风格）；
- 也支持 `mightbe-cli -e "SQL"` 一次性执行；
- 结果表格化渲染、训练进度条（轮询 job）、`.timer`/`.db` 等元命令。
- 与 server 同仓库但物理隔离，演示"客户端不依赖任何内部 crate，只依赖协议"。

### 4.8 mightbe-reason：联想检索、合理推理与模式关系学习

在 nlp 的联想图、net 训练出的向量与 store 的库表元数据之上，提供三件事：**可解释的联想检索、带证据与置信度且会主动弃判（abstain）的多跳推理、以及从数据中自动学习"词↔词 / 字段↔字段 / 表↔表"关系的模式学习器**。它不是黑盒生成，而是"**神经联想（语义相似）+ 符号路径（可溯源共现/外键/依赖）**"的混合推理器——每条结论都要能回答"凭什么、走了哪条路、证据在哪条数据"。

**四类能力严格区分：**

| 能力 | SQL 入口 | 机制 | 输出 |
|---|---|---|---|
| 网络推理 | `INFER(net, '文本')` | 网络前向（分类/回归/投影头） | 标签、概率分布 / 投影向量 |
| 联想检索 | `ASSOCIATE(net, '词', k)` | 1 跳近邻（emb 余弦 / PMI） | 关联词、分数 |
| 合理推理 | `REASON(net, '查询', HOPS n, SCOPE ..)` | 多跳图搜索 + 验证器重排 + 证据归集 | 结论、置信度、路径、证据文档/字段 |
| 关系学习 | `LEARN RELATIONS` / `SHOW RELATIONS` | 字段画像 + 关系特征 + 关系网络打分 | 外键/相关/语义/函数依赖及证据 |

**统一推理图（ReasonGraph，逻辑视图，底层数据来自系统表，不额外复制语料）：**

- 节点：
  - **词项**（来自 `mb_vocab`）、**文档**（用户表主键）、**类别标签**（Head 标签列取值）；
  - **Field 节点**：`schema.table.column`，带字段画像（类型、基数、空值率、值分布）；
  - **Table 节点**：表本身；
  - **值节点（Value）**：高频离散取值（如 `status='paid'`），是词图与模式图之间的桥；
- 边类型：
  - `cooc`：词↔词，量化共现 PMI（符号知识，可解释性最强）；
  - `emb`：词↔词，训练后词向量余弦（神经知识，解决稀疏共现的语义泛化）；
  - `appears_in`：词→文档，权重 TF-IDF（文档证据回链的根）；
  - `labeled`：文档→类别标签，由监督数据天然产生；
  - `belongs_to`：Field→Table、Value→Field（结构边，恒权 1.0）；
  - `value_in`：词/值→Field，该 token 在某字段的取值中出现（附带条件概率）；
  - `fkey`：Field→Field，值集合包含关系（外键候选，证据为具体值对）；
  - `correlated`：Field↔Field，数值 Pearson/Spearman 或类别 Cramer's V / 互信息；
  - `semantic`：Field↔Field，字段名 + 取值分布经关系网络编码的语义相似度；
  - `functional`：Field→Field，函数依赖 A→B（`P(B|A)` 接近 1）；
  - `related_via`：Table↔Table，由两表间字段边聚合并跨词图统计得出。

这样同一个束搜索引擎可以回答三类问题：词联想到词（L2）、词经多跳推出类别（L3）、**一个值/词在哪些表的哪些字段出现、表与表靠什么关联、哪些字段决定哪些字段**（模式级 L3）。

**搜索算法：束搜索（beam search）+ 扩散激活，伪流程：**

```
1. 起点：查询经 nlp 规范化/分词/编码
   ├─ 直接命中词节点；未登录词先过网络投影头做近邻落点
   └─ 多个查询词构成初始激活集合（activation 按 TF-IDF 分配）
2. 扩展：每跳为每个束取 top-branch 个邻居
   边分 edge = w_c·pmi_norm + w_e·cos + w_s·结构权重   （权重全部来自配置）
   路径分 path = ∏ edge_i × γ^(hop)                    （soft-AND + 深度衰减γ）
3. 剪枝：已访问节点指纹集合防环路；path < min_edge 的边不扩展；
   每跳后只保留 beam_width 条部分路径。
4. 汇聚：多条路径到达同一结论时用 Noisy-OR 合并证据：
   P_conc = 1 − Π(1 − p_j)，证据更强 → 置信度更高但有上界。
5. 验证（Verifier，可关）：用网络 Head 对 (查询, 候选结论) 成对打分重排，
   使最终分数同时反映"图上走得到"和"模型认为相关"；
   无监督训练数据时退化为纯路径分（配置 verify = off）。
6. 证据归集：沿 appears_in 边收集路径上各节点的文档 id，
   取同现交集优先、并集兜底，回链到原文片段（DocEvidence{doc_id, snippet}）。
7. 校准与弃判：验证集上温度缩放（配置 calibrate = isotonic | temperature | off），
   最终结论 confidence < abstain_below 一律不返回，响应为 EMPTY 并附
   "置信度不足（best=0.xx < 0.xx），已遍历 n 跳/m 条路径"。
```

**为什么这样算"合理"而不是乱联想：**

1. **双重知识源**：纯共现会被冷启动/稀疏词卡死，纯向量会产生无出处的语义漂移；两者互相兜底，符号边提供可审计性；
2. **必须有证据**：任何结论节点都要能沿边走到真实文档 id，图上孤立的高分近邻不构成结论；
3. **置信度可校准**：分数不是拍脑袋的相似度，而是经校准、可在验证集上画可靠性曲线的概率语义；
4. **会承认不知道**：阈值弃判优先于凑答案，且弃判时回传搜索诊断（走了多远、瓶颈在哪类边），便于用户补数据或调参；
5. **确定性可复现**：相同快照 + 相同查询在固定束宽下结果一致（并列分按节点 id 打破平局），便于测试与审计。

**对其他模块暴露的 API（`pub use api::*`）：**

```rust
pub trait Reasoner: Send + Sync {
    fn infer(&self, net: &str, text: &str) -> MtbResult<InferReport>;
    fn reason(&self, net: &str, query: &str, opts: ReasonOpts) -> MtbResult<ReasonReport>;
}
pub struct ReasonOpts {
    pub max_hops: u8,           // 默认取网络配置 [network.reason]
    pub beam_width: usize,
    pub branch_factor: usize,   // 每节点每跳扩展邻居数
    pub depth_decay: f32,       // γ
    pub min_confidence: f32,
    pub need_evidence: bool,
    pub verifier: VerifierMode, // on | off | auto
}
pub struct ReasonReport {
    pub query: String,
    pub conclusions: Vec<Conclusion>,     // 已按校准后置信度排序
    pub diagnostics: SearchDiagnostics,   // 跳数/扩展边数/剪枝数/弃判原因
}
pub struct Conclusion {
    pub node: NodeRef,                    // 词项 / 文档 / 类别标签
    pub confidence: f32,
    pub paths: Vec<ReasonPath>,           // 支撑路径（Noisy-OR 分量）
    pub evidence: Vec<DocEvidence>,       // 去重后的证据文档与片段
}
```

**关系学习器（SchemaLearner，同 crate 内 `schema/` 子模块）：**

```rust
pub trait SchemaLearner: Send + Sync {
    /// 全量/增量扫描候选字段对，画像 → 特征 → 关系网络打分 → 写系统表
    fn learn_relations(&self, scope: LearnScope, opts: LearnOpts) -> MtbResult<JobId>;
    fn table_relations(&self, db: &str) -> MtbResult<Vec<TableRelation>>;
    fn field_relations(&self, q: FieldRelQuery) -> MtbResult<Vec<FieldRelation>>;
}
pub struct FieldRelation {
    pub left: FieldRef, pub right: FieldRef,
    pub kind: RelKind,           // fkey | correlated | semantic | functional
    pub confidence: f32,
    pub support: u64,            // 支撑行数
    pub evidence: Vec<ValuePair>,// 值级证据样例（必须可溯源）
}
```

学习管线（数据驱动，不用硬编码规则权重——规则只负责产生候选与伪标签，最终权重由关系网络学出）：

```
1. Profiling：类型/基数/min-max/长度分布/空值率/Top 值/数值直方图/取值 token 分布
   → mb_field_profiles（INSERT 走增量脏标记，后台合并，不阻塞写入）
2. Candidate Blocking：只对可能相关的字段对算特征，避免 O(字段数²)：
   同名或字符 n-gram 近似名 / 同数据类型 / 值采样有交集 / 取值 token 共现
3. Feature Extraction（每候选对一个特征向量）：
   值集合 Jaccard 与包含率（外键强信号）、基数比、重叠值条件概率、
   Pearson/Spearman、类别 Cramer's V 与互信息、函数依赖违反率、
   字段名字符 n-gram 相似度、两侧取值 token 分布的 embedding 分布距离
4. Weak Supervision：启发式规则只产出高置信伪标签
   （值集合包含率≥0.95 → fkey；同名同类型 → semantic 候选；FD 违反率≈0 → functional）
5. Relation Net：内置小型 MLP（结构在 config/networks/_relation_net.toml 中定义，
   可被用户替换为任意自定义网络），输入第 3 步特征向量 + 两侧字段画像 embedding，
   输出关系类型概率；在伪标签上训练，再对全部候选对打分排序
6. Persist：写 mb_field_relations（含值级证据），聚合为 mb_table_relations；
   每条边带 status = learned | confirmed | rejected，用户可
   CONFIRM/REJECT RELATION 人工裁决（确认边在 REASON 中加权、被否决边不再遍历）；
   所有边同步进入统一推理图，REASON 可立即跨字段/表遍历
```

实现位置与文件规划（仅规划，不落地代码）：`graph.rs`（统一只读图视图，数据走 nlp/store 的 trait）、`search.rs`（束搜索与剪枝）、`verifier.rs`（接 net 快照）、`calibrate.rs`（温度/保序参数存系统表）、`evidence.rs`（文档/字段证据回链与片段）、`schema/profile.rs`、`schema/blocking.rs`、`schema/features.rs`、`schema/learn.rs`。推理器**只读不写业务数据**（关系结果写系统表是其唯一写权限），不持有训练状态；热更新换快照后下一次 REASON 自然用新图/新向量。

---

## 5. 网络定义（头格式）规范

用户用 **TOML 配置文件**完整定义网络；路径在 `CREATE NETWORK` 中给出。文件改动后可通过 `ALTER NETWORK ... RELOAD` 重新装载。

`config/networks/doc_rnn.toml`（完整示例，字段即规范）：

```toml
[network]
name      = "doc_rnn"          # 库内唯一标识
type      = "rnn"              # mlp | rnn | cnn | custom（仅描述性约定，结构以 layers 为准）
version   = 1                  # 由系统在热更新时自增，用户初始填 1
seed      = 42                 # 权重初始化随机种子（可复现）

# ── 输入头格式（Header）─────────────────────────────
[network.header]
source_table = "articles"      # 默认训练数据来源
text_column  = "body"          # 参与 NLP 的文本列
feature      = "token_ids"     # token_ids | tfidf | dense
seq_len      = 64              # 序列长度（截断/填充）；tfidf 时忽略
pad_id       = 0
unk_id       = 1
normalize    = "nfkc"          # nfkc | none
lowercase    = true
vocab_min_freq = 2
vocab_max_size = 20000

# ── 层（有序，name 必填：热更新按名迁移权重）────────
[[network.layers]]
name = "emb"
type = "embedding"
vocab_size = 20000             # 可用 "auto" 跟随词表
dim   = 64
init  = "uniform:-0.1:0.1"

[[network.layers]]
name = "encoder"
type = "lstm"
units = 64
return_sequences = false
bidirectional = false
init  = "xavier"

[[network.layers]]
name = "drop1"
type = "dropout"
rate = 0.2

# ── 输出头（Head）──────────────────────────────────
[network.head]
name = "category"
type = "dense"                 # dense | projection
units = 5                      # dense：分类数；projection：向量维度
activation = "softmax"
loss = "cross_entropy"         # cross_entropy | mse
label_column = "category_id"   # 监督标签所在列
metric = "accuracy"

# ── 训练超参（TRAIN 语句可覆盖）─────────────────────
[network.train]
optimizer = "adam"             # sgd | momentum | adam
lr        = 0.002
batch_size = 32
epochs     = 10
shuffle    = true
early_stop_patience = 2
checkpoint_every_steps = 200
embedding_trainable = true     # 词向量是否参与训练（关闭则只做静态查表）

# ── 联想（Association）配置 ────────────────────────
[network.assoc]
source = "hybrid"              # cooc | embedding | hybrid
window = 5                     # cooc 滑窗
w_cooc = 0.4                   # 推理图 cooc 边权重 w_c
w_emb  = 0.6                   # 推理图 emb 边权重 w_e
hybrid_alpha = 0.7             # ASSOCIATE 单跳融合: α*embedding + (1-α)*cooc
top_k  = 20

# ── 推理（Reasoning）配置 ──────────────────────────
[network.reason]
max_hops       = 3             # 最多几跳
beam_width     = 16            # 束宽（每跳保留的部分路径数）
branch_factor  = 8             # 每节点每跳扩展的邻居数
depth_decay    = 0.8           # γ：路径深度衰减
min_edge       = 0.05          # 低于此分的边不扩展
abstain_below  = 0.25          # 弃判阈值：最高结论低于此值则返回"无可靠结论"
verifier       = "auto"        # on | off | auto（有监督数据时自动启用）
calibrate      = "temperature" # off | temperature | isotonic（验证集上校准置信度）
need_evidence  = true          # 结论必须能回链到真实文档
evidence_top_k = 3             # 每个结论最多回链多少篇证据文档
cross_scope_gate = 0.6         # SCOPE 跨 词↔字段↔表 边界时的乘性门控
```

MLP（非序列）示例只需把 `feature = "tfidf"`、层换成若干 `dense + relu`；**CNN 文本分类**把 `type = "cnn"`、层换成 `embedding → composite(TextCNN 见 4.1.1) → dense`；想要全新算子时层的 `type` 直接写插件层名（如 `type = "atss_conv", library = "mylayers"`，需先 `CREATE LIBRARY`）。系统校验规则：

- `token_ids` 后第一层必须能接收 ID 序列（`embedding`）；
- `tfidf` 后第一层必须接收二维稠密/稀疏输入（`dense`）；
- `conv1d/pool1d/rnn/gru/lstm` 的输入必须是三维序列张量（通常来自 `embedding`）；池化后接 `dense` 前必须经 `flatten/globalpool`；
- `concat` 各输入除拼接轴外形状一致；composite 的 `from` 引用必须存在且 DAG 无环；
- Cell DSL 方程通过语法、维度、可微性三项检查；插件层必须能在 manifest 声明的形状约束下通过 `infer_shapes`；
- 层名全网络唯一；`return_sequences=false` 后不得再接序列层；
- Head 的 `units` 与 `loss` 必须相容；
- 校验失败返回 `ERR CONFIG_INVALID <详情>`，不注册网络。

---

## 6. 模型文件二进制布局

检查点文件 `data/checkpoints/<network>/v<version>_<steps>.mbm`，所有多字节整数小端：

```
偏移      长度      字段
0         4         Magic            = 0x4D424D44 ("MBMD")
4         2         FormatVersion    = 1
6         2         HeaderSize        (后续网络头区段字节数)
8         16        NetworkId         (UUID，跟随网络名)
24        4         Version           拓扑版本
28        4         TopologyHash      对拓扑规范的稳定哈希
32        8         Steps
40        8         CreatedAt         (Unix 毫秒)
48        8         WeightBytes
56        32        WeightSha256
─────────────── Header 区段（HeaderSize 字节）──────────────
          N         TopologyJson      规范化（排序键）后的拓扑 JSON
─────────────── 权重区段（WeightBytes 字节）──────────────
                    重复 { name_len:u16, name, shape_ndim:u8,
                           dims:u32[ndim], f32[...] } 直到段末
```

加载时校验 Magic / Version / Sha256 / TopologyHash；热更新装载新版本时，若某层拓扑未变且哈希相同，直接复用权重字节，零拷贝读入。

**模型权重也属于"必须持久化"的数据**：`.mbm` 文件不是裸露字节，而是封装为 MDB 加密容器（`.mbm` 内部含一段加密权重区，格式见 [第 11.1 节](#111-mdb-数据库文件格式加密)）——即使 checkpoints 目录被整体拷走，没有库口令也无法还原权重。同理，词表向量、校准参数、关系边与诊断日志全部落在加密 MDB 文件内，明文只存在于进程内存。

---

## 7. 实时修改机制（热更新）

这是框架的核心能力。**不停服、不丢训练进度、推理请求不中断。**

### 7.1 变更来源

1. SQL：`ALTER NETWORK doc_rnn ADD LAYER {...} AFTER encoder;`
2. SQL：`ALTER NETWORK doc_rnn RELOAD;`（配置文件已在外部改好，重新读入）
3. SQL：`ALTER NETWORK doc_rnn SET lr=0.001, DROP LAYER drop1;`

### 7.2 提交流程（版本化 + 原子切换）

```
AlterOp 到达
   │
   ├─ 1. 解析 + 与当前 spec 合并，生成 CandidateSpec(v_new)
   ├─ 2. 静态校验（形状相容性，见第 5 节规则）
   ├─ 3. 权重迁移：按 (层名, 参数名, 形状) 逐参数匹配
   │      ├─ 匹配上      → 拷贝旧权重
   │      ├─ 层在/形变小 → 截断或补随机初始化（记录 warning）
   │      └─ 新增层      → 按层 init 策略初始化
   ├─ 4. 在后台线程构建新 Network，并做一次 dummy 前向（冒烟）
   ├─ 5. 训练锁内：保存旧快照 → Arc 原子替换 → 写 mb_network_versions + WAL
   └─ 6. 失败则整体回滚，旧版本继续服务，返回 ERR
```

- **推理侧**：每个会话开始执行时取一次 `Arc<Snapshot>`，执行期间一直用旧快照；新请求自动拿到新版本——无全局停顿，无锁读。
- **训练侧**：正在进行的 step 用旧快照跑完；优化器动量（Adam moments）同样按参数名迁移，名字对不上的参数重置动量。
- **回滚**：`ALTER NETWORK doc_rnn RESTORE VERSION 3;` 走同一套提交流程（旧版本拓扑在系统表里完整保留）。
- **并发变更**：同一网络的 ALTER 串行化（每网络一把变更锁）；不同网络完全并行。

---

## 8. NLP 流水线、联想检索与合理推理

### 8.1 INSERT 触发链

```
INSERT INTO articles ...
  → 行落盘（WAL 提交）
  → NLP 后台任务（不阻塞 INSERT 返回，job 可查）：
      normalize → tokenize → 词表增量合并
      → 写 mb_vocab 统计 / 更新 mb_df
      → TF-IDF + 可选 TextRank → 写 mb_keywords
      → 共现窗口统计 → UPSERT mb_assoc（cooc 边）
      → 若网络 header 绑定该表：构造训练样本入"待训练队列"
```

### 8.2 训练如何消费特征

- `token_ids` 头：样本 `(id 序列 [seq_len], 标签)` → Embedding → RNN/LSTM → Head；
- `tfidf` 头：样本 `(稀疏向量, 标签)` → Dense 堆叠 → Head；
- 标签来自 Head 配置的 `label_column`；无标签表只训练 Embedding（Skip-gram 式自监督，作为 `assoc.source = embedding` 的前提）。

### 8.3 联想打分

- **共现 PMI**：`PMI(a,b)=log( P(a,b) / (P(a)P(b)) )`，频次由滑窗累计；
- **向量余弦**：训练后的 Embedding 行向量，查询时在线计算 top-k（词表 ≤ 2 万时暴力扫描足够；更大时上 IVF 索引，列路线图）；
- `hybrid`：分数归一化后线性融合；
- `SELECT * FROM ASSOCIATE(doc_rnn, '所有权', 10);` 返回词与分数，可再 `JOIN mb_keywords` 做文档级推荐。

### 8.4 从联想到推理：三层递进

检索 → 推理不是两个割裂功能，而是同一张图上搜索深度与证据要求的递进：

| 层级 | 语句 | 跳数 | 证据要求 | 典型问题 |
|---|---|---|---|---|
| L1 精确 | `SELECT ... WHERE` | 0 | — | "id=3 的文档是什么" |
| L2 联想 | `ASSOCIATE` | 1 | 无 | "'所有权' 语义上相邻的词" |
| L3 推理 | `REASON ... HOPS 3` | 多跳 | **必须** | "'所有权' 经过哪些概念，能推出什么主题/类别，证据在哪些文档" |

### 8.5 多跳推理算法（对应 4.8 节伪流程的数学定义）

- **边分归一化**：PMI 经 min-max 压到 [0,1]（按全库 PMI 分布的固定分位数，参数随校准表持久化，避免随库变大漂移）；余弦从 [-1,1] 仿射到 [0,1]；
- **路径分**：`path = γ^h · ∏ edge_i`，乘法 soft-AND 保证任一跳过弱则整条路径弱，γ 控制"推理越远越保守"；
- **多路径汇聚**：同一结论的 j 条路径 `P = 1 − Π(1 − p_j)`，证据越多置信度越高，但边际递减且有上界，防止靠堆烂路径刷分；
- **验证器重排**：把查询文本与候选结论文本（词节点→拼合描述、文档节点→标题）现场过网络编码器与 Head，得到相关性概率 `v`，最终 `confidence' = conf^λ · v^(1−λ)`（λ 在配置中），使"图上可达"还要"模型认可"；
- **证据选取**：路径上每条词节点沿 `appears_in` 取 TF-IDF 最高的文档，多跳路径取文档**交集**（同篇文档覆盖多跳概念才是强证据），不足 `evidence_top_k` 时用并集补足；片段截取命中关键词前后窗口；
- **校准**：`CALIBRATE NETWORK doc_rnn FROM articles WITH VALIDATION 0.2;` 触发，在训练数据留出验证集上收集 (原始分, 是否命中真实标签) 对，拟合温度参数或保序函数，参数写 `mb_calibration`，REASON 输出的 confidence 即校准值；
- **弃判语义**：全部候选低于阈值时返回零行 + 诊断信息（`SHOW` 风格的消息：`best_confidence / hops_explored / edges_pruned / blocked_edge_type`），调用方可以据此决定补语料还是放宽阈值。

### 8.6 网络前向推理（INFER）

- 分类头：返回 softmax 全分布，按概率降序，同样受校准参数约束（温度缩放同时作用于 INFER 与 REASON）；
- 投影头（`type = "projection"`）：输出向量，`NEAR()` 用它做文档检索，REASON 用它给未登录查询词做图上落点；
- 回归头：返回值与训练残差估计（若配置了）；
- INFER 是纯快照读：网络热更新期间的在途请求与推理请求一样走旧快照，无锁。

### 8.7 词、字段、表三个层级的关系统一学习

系统训练的"关系"覆盖三个粒度，共用统一推理图，但生产者不同：

| 粒度 | 边 | 生产者 | 触发 |
|---|---|---|---|
| 词 ↔ 词 | `cooc` / `emb` | nlp 共现统计 + 网络 Embedding 训练 | INSERT 后台任务 / TRAIN |
| 字段 ↔ 字段 | `fkey` / `correlated` / `semantic` / `functional` | SchemaLearner 画像 + 关系网络 | `LEARN RELATIONS`（可配写入后增量） |
| 表 ↔ 表 | `related_via` | 字段边聚合 + 跨表 token 重叠 | 关系学习收尾阶段级联产出 |

关系学习为**后台任务**（与 TRAIN 一致，立即返回 job id），分全量与增量两模式：写入只给受影响字段打脏标记，后台合并脏字段对重算，避免大库全表重扫。关系网络本身是一个普通 MightBe 网络（`_relation_net.toml`，默认小 MLP），因此它也享受热更新——用户可以把它换成 CNN/自定义插件网络来做字段值分布编码。

### 8.8 跨模式推理（REASON 的 SCOPE）

`SCOPE` 子句声明束搜索允许进入的节点域：`WORD`（词/文档/标签）、`FIELD`（字段/值）、`TABLE`（表），默认 `WORD`。开启跨域后，词可经 `value_in` 走到字段，字段经 `fkey/functional` 走到其他字段，再经 `belongs_to` 走到其他表，最终把"证据"从文档扩展为"具体字段中的具体值对"。跨域边的进入有额外一次乘性门控（配置 `cross_scope_gate`，默认 0.6），防止结构化边数量优势淹没语义路径。

### 8.9 关系规则挖掘（REASON 的规则化量化身）

多跳路径中高频出现的**边类型模式**被挖掘为可复用规则，让"同一类推理"从逐次搜索变成一次学习、全库复用：

```
LEARN RULES ON docdb
  OPTIONS(min_support = 100, min_confidence = 0.6, max_path_len = 4);
```

- 规则形式（一阶近似，不含变量绑定求解，仅模式与置信度）：
  `PATH MODE:  X ─value_in► F1 ─fkey► F2 ─belongs_to► T   ⇒   结论集落在 T`
  `LINK MODE:  A.word ─cooc► B.word ∧ A.appears_in► D  ⇒  D 与 B.word 相关（先验）`
- 规则置信度按 (support, lift) 评估，带正负例证据，写 `mb_rules`；规则本身也可被 CONFIRM/REJECT；
- **规则的作用**：后续 REASON 命中匹配规则时直接给路径一个先验加分（等价于"这是被数据反复验证过的推理套路"），并允许在 `EXPLAIN REASON` 输出中展示命中了哪些规则；
- 规则与 LearnedRelation 的区别：边是"两个节点相关"，规则是"一类连接范式会产生一类结论"——它压缩的是推理结构本身，使关系推理从 memory-based 的逐次搜索走向 rule-like 的复用；
- **非目标声明**：M 阶段不实现变量绑定、联合查询与规则推理机（Datalog 式），`mb_rules` 只作为路径先验与解释材料。

### 8.10 数学运算能力（SQL 层）

推理与关系学习大量依赖数学运算，这些能力作为一等函数暴露在 SELECT 表达式中，不写死在某一侧：

| 类别 | 函数示例 | 说明 |
|---|---|---|
| 标量 | `abs/sqrt/exp/log/log2/log10/pow/powi/sin/cos/tan/asin/sign/factorial/gcd/lcm/mod/hypot/lerp` | IEEE 语义、NaN/Inf 传播规则写明 |
| 统计 | `var/varp/std/stddev/skew/kurt/median/percentile(x,0.25)/quantile/entropy/mode/mad` | 一/两次（无偏/有偏）都提供 |
| 线性代数 | `dot(v,w)` `norm1/2(v)` `cosine(v,w)` `l2/l1(v,w)` `normalize(v)` `matvec(M,v)` `trace(M)` `rank(M)`（可选） | 向量以 `VECTOR(n)` 列类型或 JSON 文本承载 |
| 向量检索 | `nearest(col, '文本', k)` `ivf_query(col, v, k)` | 内部复用网络编码 + ANN 索引；明文等价于 `NEAR()` |
| 数值工具 | `round_half_even(x, k)` `clamp(x,a,b)` `sigmoid/softmax(z)` `log_softmax(z)` | softmax 可作用在整列（向量）上 |

类型与行为约束：

- 新增 SQL 值类型 `VECTOR(n) FLOAT`（定长）与 `VECTOR` 变长（检索/存储两用），存储为定长 f32 段，和其他列一样进 MDB 加密页；
- 数学函数参与 `WHERE / HAVING / ORDER BY / SELECT 表达式`，可嵌套；除零、`sqrt` 负数等按 IEEE 返回 NaN/Inf 并置 `warning`，不静默吞掉；
- 高成本函数（`rank/ivf_query`）设有单查询预算（`max_ms`），超预算返回部分结果 + 预算提示；
- 所有数学函数的 MVP 实现在 `mightbe-core/src/ops/math.rs`（供训练复用），SQL 层只是薄封装——**同一实现两处使用，杜绝两套语义漂移**。

### 8.11 EXPLAIN 家族（推理可观测）

| 语句 | 用途 |
|---|---|
| `EXPLAIN REASON ...` | 打印束搜索展开树：每层保留的路径、边分、剪枝原因、命中规则（8.9） |
| `EXPLAIN SELECT ...` | 火山模型物理计划：访问路径、谓词下推、行数估计、成本 |
| `EXPLAIN ASSOCIATE/INFER ...` | 使用哪个快照版本、embedding 归一化分位数、是否走 ANN 索引 |

---

## 9. SQL 方言完整定义

**标识符**：反引号可选；关键字大小写不敏感；字符串用单引号（`''` 转义）。

```sql
-- 库 / 表
CREATE DATABASE IF NOT EXISTS docdb;
USE docdb;

CREATE TABLE articles (
    id          INTEGER PRIMARY KEY AUTO_INCREMENT,
    title       VARCHAR(256) NOT NULL,
    body        TEXT,
    category_id INTEGER
);

-- 向量列：定长/变长，供 dot/cosine/nearest 等数学与向量检索函数使用
CREATE TABLE article_vecs (
    article_id  INTEGER PRIMARY KEY,
    emb         VECTOR(64) FLOAT,     -- 网络投影头输出，存取均加密
    emb_norm    VECTOR FLOAT          -- 变长
);

-- 数学表达式进入查询/排序/过滤
SELECT title,
       sqrt(length(body))              AS root_len,
       percentile(score, 0.9)          AS p90,
       cos(e1.emb, e2.emb)             AS sim        -- 向量余弦
FROM articles JOIN article_vecs e1 ON e1.article_id = articles.id
WHERE dot(e1.emb, e2.emb) > 0.7 AND category_id >= 0
ORDER BY sim DESC
LIMIT 10;

-- 网络（头格式在外部配置文件中定义，不写死在 SQL 里）
CREATE NETWORK doc_rnn FROM 'config/networks/doc_rnn.toml';

-- 自定义层插件库（动态库 + manifest），加载后网络即可引用其中的层类型
CREATE LIBRARY mylayers FROM 'plugins/mylayers.dll';
ALTER LIBRARY mylayers RELOAD;      -- 热替换插件，引用它的网络产生新版本
SHOW LIBRARIES;                    -- 库名、版本、导出层、加载状态

-- 写入（文本列自动进入 NLP 管线）
INSERT INTO articles(title, body, category_id)
VALUES ('所有权笔记', 'Rust 的所有权与借用……', 0);

-- 普通查询
SELECT id, title FROM articles WHERE category_id = 0 ORDER BY id DESC LIMIT 10;

-- 训练（后台任务，立即返回 job id）
TRAIN NETWORK doc_rnn
  ON articles
  OPTIONS(epochs = 20, lr = 0.001, batch_size = 16);

SHOW TRAINING;                 -- 所有任务状态
SHOW TRAINING 'job_id';        -- 单任务进度与 loss

-- 关键词 / 词表观察
SELECT word, score FROM mb_keywords WHERE doc_id = 1 ORDER BY score DESC LIMIT 5;
SELECT * FROM mb_vocab ORDER BY id LIMIT 20;

-- 语义近邻：把文本现场编码，用网络头投影后做近邻
SELECT id, title, score
FROM NEAR(doc_rnn, TABLE articles, COLUMN body, '借用检查器很严格', 10);

-- 网络推理（模型前向）：对任意未入库文本即时预测
SELECT label, confidence
FROM INFER(doc_rnn, '这段代码的借用检查器报错了');
-- 回归头返回 value；想看全部分布去掉 LIMIT 即可

-- 联想检索（1 跳）与多跳合理推理
SELECT word, score FROM ASSOCIATE(doc_rnn, '所有权', 10);

SELECT conclusion, node_type, confidence, path, evidence
FROM REASON(doc_rnn, '所有权',
            HOPS 3, TOP 5,
            MIN_CONFIDENCE 0.30,
            SCOPE WORD, FIELD, TABLE,   -- 允许跨词/字段/表推理（默认仅 WORD）
            VERIFIER ON);
-- path 形如 所有权 ─cooc─► 借用 ─emb─► 生命周期 ─labeled► 类别:内存管理
-- 跨模式时形如 词:paid ─value_in► orders.status ─fkey► payments.order_status
-- evidence 形如 [doc#12 "...所有权与生命周期...", doc#7 ...] / [orders#id=88 status=paid]

-- 逐跳展开路径（审计/调试用，一行一跳）
SELECT conclusion, hop, node, edge_type, edge_score, evidence_docs
FROM REASON_PATHS(doc_rnn, '所有权', HOPS 3, TOP 5);

-- 置信度校准：在留出验证集上拟合温度/保序参数
CALIBRATE NETWORK doc_rnn FROM articles WITH VALIDATION 0.2 USING temperature;

-- 关系学习：从数据中训练 字段↔字段 / 表↔表 关系（后台任务）
LEARN RELATIONS ON docdb
  OPTIONS(mode = "full", min_support = 50, sample_rows = 100000);
LEARN RELATIONS TABLES articles, authors OPTIONS(mode = "incremental");

-- 规则挖掘：从高频推理路径中提取可复用规则（8.9）
LEARN RULES ON docdb OPTIONS(min_support = 100, min_confidence = 0.6, max_path_len = 4);
SHOW RULES;                                  -- 或按表：SHOW RULES FROM articles;

SHOW TRAINING 'job_id';

-- 观察学习到的关系（也可直接查系统表 mb_field_relations / mb_table_relations）
SHOW RELATIONS;                      -- 表与表：related_via 边、置信度、桥接字段
SHOW RELATIONS FROM orders TO payments;
SHOW FIELD RELATIONS WHERE table_name = 'orders';
SELECT left_field, right_field, relation, confidence, support, evidence
FROM mb_field_relations
WHERE relation IN ('fkey', 'functional') AND confidence >= 0.8
ORDER BY confidence DESC;
-- 人工裁决（学习结果默认是带证据的候选，不自动当真实约束执行）
CONFIRM RELATION orders.user_id -> users.id TYPE fkey;
REJECT  RELATION orders.note -> articles.title;

-- 加密与备份（11.1 节 MDB 规范的运维面）
LOCK KEYS;                                 -- 锁定口令：关闭明文内存残留的临时解码副本
BACKUP DATABASE docdb TO 'backups/docdb-2026-09-25.mbdb' ENCRYPTED;
RESTORE DATABASE docdb FROM 'backups/docdb-2026-09-25.mbdb';
ROTATE KEY FOR DATABASE docdb;             -- 换数据密钥并全量重写（在线，先写新段后切换）
KEY OPEN;  KEY CLOSE;                      -- 会话级口令输入（明文不回显、不进日志）

-- 实时修改网络
ALTER NETWORK doc_rnn ADD LAYER { type = "lstm", name = "encoder2",
                                  units = 128, init = "xavier" } AFTER encoder;
ALTER NETWORK doc_rnn DROP LAYER drop1;
ALTER NETWORK doc_rnn SET lr = 0.001, epochs = 30;
ALTER NETWORK doc_rnn RELOAD;          -- 重新读取配置文件
ALTER NETWORK doc_rnn RESTORE VERSION 3;

SHOW NETWORKS;
SHOW NETWORK doc_rnn VERSIONS;

-- 删除
DROP NETWORK doc_rnn;                  -- 保留检查点文件（加 PURGE 才物理删除）
DROP TABLE articles;
```

错误码区段：`1xxx` 语法、`2xxx` 表/列、`3xxx` 网络/配置/插件（含 3101 插件 ABI 版本不符、3102 导出符号缺失、3103 插件冒烟失败）、`4xxx` 训练、`5xxx` 存储、`6xxx` NLP、`7xxx` 推理（含 7001 无结论弃判、7002 证据不足、7003 验证器不可用、7004 校准数据不足）、`8xxx` 关系学习（8001 候选不足、8002 画像过期、8003 关系网络未就绪）。

---

## 10. 端到端使用示例（走查）

1. 编写并放置配置：`config/networks/doc_rnn.toml`（第 5 节内容）与停用词表；
2. 启动：`mightbe serve --config config/mightbe.toml`；
3. 另开终端：`mightbe-cli`；
4. 依次执行：建库建表 → `CREATE NETWORK`（看到 `OK network=doc_rnn version=1 layers=3`）→ 批量 `INSERT`（观察后台 NLP 任务完成）→ `TRAIN`（轮询 loss 下降）→ `SELECT ... mb_keywords` 验证关键词 → `ASSOCIATE` 验证联想质量；
5. 自定义层：按 4.1.1 把编码器改为 composite TextCNN（零代码）或在 TOML 里用 Cell DSL 写自定义门控；如需新算子则用 `mightbe-plugin` 编一个 cdylib，`CREATE LIBRARY` 后在网络中引用，`ALTER LIBRARY RELOAD` 验证热替换；
6. 推理与校准：`CALIBRATE NETWORK` 拟合置信度 → `INFER` 对新文本分类 → `REASON ... HOPS 3` 得到带路径与证据文档的结论；对一个生僻词执行 REASON，观察系统在证据不足时返回弃判诊断而不是乱答；`REASON_PATHS` 逐跳审计；
7. 关系学习：建第二张表（如作者表/支付表）灌入有外键取值重叠的数据 → `LEARN RELATIONS` → `SHOW RELATIONS` 检查自动发现的外键/函数字段对与值级证据 → `REASON(... SCOPE WORD,FIELD,TABLE)` 从一个词跨字段跨表推出关联数据；
8. 在线热改：`ALTER NETWORK ... ADD LAYER` → `SHOW ... VERSIONS` 看到新版本，训练继续且旧权重被迁移，再跑一次 REASON 确认推理自动使用新向量快照；
9. ctrl-c 重启服务，系统自动恢复插件库、模型检查点与全部表数据（含学习到的关系）。

批量灌库走 CLI 的 `.import csv path table` 元命令（协议层等价于预语句批量 INSERT）。

---

## 11. 存储引擎内部设计

### 11.1 MDB 数据库文件格式（二进制 + 加密）

MightBe 的持久文件统一为 **MDB（MchkDB）格式**：二进制、分页、可加密。一切需要落盘的字节（表数据页、WAL、模型权重、词表向量、校准参数、库表元数据）都走这一容器；磁盘上不存在明文数据文件。

```
data/
├─ docdb.mdb                  # 数据库主文件（页 + 表目录），全加密
├─ wal/
│  └─ wal.0001.mwlog          # WAL 段，逐条加密（同密钥派生子流）
├─ backup/
│  └─ docdb-2026-09-25.mbdb   # BACKUP 产出的加密快照（单文件）
├─ checkpoints/
│  └─ doc_rnn/v2_1840.mbm     # 模型权重（加密容器，格式见第 6 章）
└─ bufferpool.state           # checkpoint 元信息（加密）
```

**文件头（MdbHeader，前 128 字节，明文但已认证）：**

```
偏移   长度   字段
0      4      Magic            = 0x4D424431 ("MBD1")
4      2      FormatVersion    = 1
6      1      CipherSuite      = 1: AES-256-GCM  2: XChaCha20-Poly1305（M2 起）
7      1      Flags            = bit0: 头部后附额外明文字段（kdf 参数）
8      2      PageSizeBits  | 页大小编码（4096..131072 的 log2）
10     2      PageCount         页总数（可变）
12     8      CreatedAt         Unix ms
20     8      LastCheckpointLsn
28     8      WalTailLsn
36     2      KdfId             = 1: Argon2id
38     2      KdfParamsLength
40     4      KeyId             数据密钥标识（用于 ROTATE KEY 后定位）
44     16     HeaderNonce       头部 AEAD 随机数
...
末尾   32     HeaderTag         头部认证标签（篡改即拒绝打开）
```

**密钥体系（KEK/DEK 两级，口令永不直接用于数据加密）：**

- 用户口令（`KEY OPEN` 输入或配置 `password_env` 指定的环境变量）→ **Argon2id(salt, m=64MiB, t=3, p=4)** → KEK（密钥加密密钥）；
- 每个数据库独立生成随机 **DEK**（数据密钥），用 KEK 加密后随数据库元数据存放在启动时的内存注册表中，**出入口只有内存**；
- 页加密 IV/nonce 由 `(page_id, lsn)` 派生（每页每版本不同），杜绝同密钥同 nonce 复用；
- WAL 每条记录的 nonce 由记录序号派生，乱序/重放攻击因 lsn 单调检查被拒绝；
- 口令相关：明文不落盘、不进日志、不进 `SHOW` 输出；内存中口令缓冲用 `zeroize` 清零；`LOCK KEYS` 后进程丢弃内存中的明文解码副本（缓存页保留密文）。

**页加密与恢复：**

- 页布局 `[PageHeader 32B][SlotArray][Cell...]` 中的 **Payload 段加密**，头部（含 lsn、crc、类型）明文，便于恢复期快速筛选；
- BufferPool 中的页**始终是明文**（读取即解密、写入前加密），脏页刷盘 = 加密 + 原子写（先写临时段、fsync、rename）；
- 崩溃恢复：WAL 重放前先校验 AEAD tag，损坏记录（尾段常见）直接截断到最后一个完整 COMMIT，与 `checkpoint` 组合保证一致性；
- 幂等：页解密后同时用 CRC32 与 AEAD tag 双校验，任何一方失败都视为页损坏，拒绝加载并报 `5xxx`，绝不把不可信数据喂给训练/推理。

**运维操作（对应 SQL 面）：**

- `BACKUP DATABASE`：先触发一次 checkpoint，再按页顺序复制成单文件 `.mbdb`（含 key_id 但不含任何口令信息），热备份不阻塞读写；
- `RESTORE`：校验 tag 完整性后写入数据目录并触发恢复流程；目标库必须处于 offline 状态；
- `ROTATE KEY`：生成新 DEK → 新密钥加密重写全部页 → 原子切换 key_id → 旧密钥从内存注销；大库耗时由后台任务承担，进度 `SHOW TRAINING` 可见；
- `EXPORT/IMPORT`：明文导出（`.mbdump`，用户显式声明 `NOT ENCRYPTED` 才允许，记入审计日志）；默认加密。

### 11.2 存储目录布局与页引擎

（目录总览见 11.1 节；此处描述各组件行为）

- **页结构**：`[PageHeader 32B][SlotPointerArray (尾部反向生长)][FreeSpace][Cells]`，槽位号即行内物理地址，删除只标记墓碑，由后台 compaction 回收（第二期）。
- **WAL 记录**：`[lsn:u64][txn:u64][type:u8][page_id:u32][len:u32][payload][crc:u32]`，类型含 COMMIT / UPDATE_PAGE / SYS_META；payload 段按 11.1 规则加密。
- **Checkpoint**：每 N 步或 WAL 达阈值触发：冻结新写入 → 刷全部脏页 → 写 `bufferpool.state` → 截断旧 WAL。
- **系统表与用户表同构**：`mb_*` 表就是系统启动时自动创建的普通表，用户甚至可以 SELECT 它们（只读）。
- **持久化清单（全部经 MDB 加密落地，"数据库该有的都进库"）**：

| 数据 | 载体 | 触发时机 |
|---|---|---|
| 用户表数据 / 系统表数据 | `*.mdb` 页 | WAL 提交 + checkpoint 刷盘 |
| 模型权重快照 | `checkpoints/*.mbm`（加密容器） | checkpoint_every_steps |
| 训练任务与 loss 曲线 | `mb_training_jobs` 表 | 每次 job 状态变更 |
| 词表 / IDF / 关键词 / 共现边 | `mb_vocab / mb_df / mb_keywords / mb_assoc` | NLP 后台任务合并点 |
| 关联边与推理图临时视图 | `mb_assoc` + `mb_field_relations` 派生 | 学习任务收尾 |
| 校准参数（温度/保序/PMI 分位数） | `mb_calibration` | CALIBRATE 完成 |
| 规则（8.9）与裁决状态 | `mb_rules` / `mb_field_relations.status` | LEARN RULES / CONFIRM·REJECT |
| 插件注册信息 | `mb_libraries` | CREATE/ALTER LIBRARY |
| 推理日志（可选） | `mb_reason_log` | reason_log_enabled 时 |
| 关系学习任务 | `mb_relation_jobs` | 任务状态变更 |

---

## 12. 线程模型与并发控制

基于 tokio 多线程运行时，但模块内不强制 async——存储与训练为 CPU/IO 密集，用 blocking 线程池隔离：

| 线程 | 数量 | 职责 |
|---|---|---|
| IO worker | = CPU 核数 | TCP 读写、协议解析 |
| SQL executor | tokio blocking 池 | 解析后执行，调用各模块 |
| Train worker（每网络 1 个） | 按网络 | 取批次 → train_step → 周期 checkpoint；ALTER 时在屏障处切换 |
| NLP worker | 配置值（默认 2） | 后台分词 / 关键词 / 共现 |
| Relation worker | 1（后台任务队列） | 字段画像、候选对特征计算、关系网络训练/打分；与 TRAIN 共用任务表 |
| 加速设备 worker | 每网络 0 或 1（仅 M9 起） | GPU/NPU kernel 提交与完成事件回收；设备错误捕获、降级、设备端同步；CPU 模式下本行退化为空 |
| REASON/INFER | 复用 SQL executor（blocking 池） | 推理器只读快照与系统表，按查询并发，单查询受 beam/hops 预算封顶，不设常驻线程 |
| WAL / flush | 1 | 组提交与脏页刷盘 |

共享状态策略：

- 模型快照：`arc-swap`，读完全无锁；
- Catalog / 词表：`parking_lot::RwLock`，读多写少；
- 页缓存：分片锁（按 page_id 哈希分 N 片）避免单点争用；
- 同一网络的训练与 ALTER 通过单线程 train worker 串行，天然无数据竞争；
- 所有模块以 `Send + Sync` 的 trait object 注入，禁止 `lazy_static` 可变全局。

---

## 13. 配置与数据目录布局

```
config/
├─ mightbe.toml               # 监听地址、数据目录、页大小、线程数、日志级别
├─ networks/                  # 每个网络一个 TOML（第 5 节规范）
│  └─ doc_rnn.toml
└─ nlp/
   ├─ stopwords.zh.txt        # 一词一行
   ├─ stopwords.en.txt
   └─ punctuation.txt
```

`config/mightbe.toml`（节选示意）：

```toml
[server]
host = "127.0.0.1"
port = 9527

[storage]
data_dir    = "data"
page_size   = 8192
wal_flush_ms = 50
mdb_cipher  = "aes-256-gcm"        # aes-256-gcm | xchacha20-poly1305(M2)
password_env = "MIGHTBE_KEY"       # 口令来源环境变量；为空则启动时交互输入(KEY OPEN)

[engine]
backend     = "naive"             # naive | simd | cuda | opencl | npu | auto
device      = "auto"              # auto | cpu | cuda:0 | npu:0（按网络可覆盖）
threads     = 0                   # 0 = 按核数
profile     = false               # 算子级计时
gpu_memory_pool_mb = 2048         # M9 起生效

[nlp]
stopwords = ["config/nlp/stopwords.zh.txt", "config/nlp/stopwords.en.txt"]
keyword_algo = "tfidf"        # tfidf | textrank

[plugins]
dir = "plugins"               # 启动时扫描并登记（不自动加载，需 CREATE LIBRARY）
allow_reload = true           # 允许 ALTER LIBRARY RELOAD 热替换

[schema_learning]
enabled = true
auto_profile_on_write = true  # INSERT/UPDATE 后给字段打脏画像标记
auto_learn_on_write = false   # true 时由后台自动增量重算（默认手动 LEARN RELATIONS）
relation_network = "config/networks/_relation_net.toml"
min_support = 50              # 字段对最少支撑行数
sample_rows = 100000          # 大字段采样上限
candidate_blockers = ["name_ngram", "same_type", "value_overlap", "token_overlap"]
default_scope = ["WORD"]      # REASON 默认搜索域
```

原则（强制 code review 检查项）：**超参、词表、停用词、网络结构一律不允许出现在 `.rs` 源码里**；Rust 代码只包含算法与默认值兜底（且默认值集中在各模块的 `defaults.rs` 并在文档中列出）。

---

## 14. 错误处理、日志与可观测性

- 统一错误类型 `MtbError { code, module, message, source }`，`thiserror` 派生；对用户只暴露错误码 + 信息（第 9 节区段表）。
- 日志：`tracing`，结构化字段（network、version、job_id、lsn），输出到 stdout 与 `logs/mightbe.log`（按天滚动）。
- 指标（MVP 内存计数器，`SHOW STATUS` 暴露）：QPS、训练 steps/s、loss 最新值、WAL 延迟、缓冲池命中率、NLP 队列深度、推理 QPS、REASON 平均跳数/平均束扩展边数、**弃判率（abstain rate）**、校准后 ECE（期望校准误差）、关系学习候选对/边产出量、插件层调用次数与失败数。
- 训练 loss / 指标每 checkpoint 同时落 `mb_training_jobs`，可 SQL 查询历史。

---

## 15. 依赖策略

**自己实现**：张量与 autograd、所有层与优化器、分词、TF-IDF/TextRank、SQL 解析、页与 WAL、二进制协议、**动态库加载**（4.1.1 路径 C 的宿主加载器直接 FFI 调 `LoadLibraryW`/`dlopen`，不引入 `libloading`）。

**谨慎引入的第三方 crate（基础设施层，已验证 crates.io 可达）**：

| 用途 | crate | 理由 |
|---|---|---|
| 异步运行时 | `tokio`（rt-multi-thread + net + io-util） | TCP 服务基础 |
| 序列化 | `serde` + `toml` | 网络配置/系统元数据 |
| 并发结构 | `dashmap`, `arc-swap`, `parking_lot` | 快照热切换 |
| 日志 | `tracing` + `tracing-subscriber` | 结构化日志 |
| 哈希/校验 | `sha2`, `crc32fast` | 模型校验、WAL CRC |
| CLI | `clap` | 子命令解析 |
| 随机数 | `rand` + `rand_chacha` | 可复现种子 |
| 加密 | `aes-gcm`、`argon2`、`zeroize`、`subtle` | MDB 页加密、口令派生、内存擦除、常数时间比较 |
| GPU/NPU（feature gate） | `cudarc`（cuda）、`opencl3`（opencl） | M9 起可选，默认构建不依赖 |
| Unicode | `unicode-normalization` | NFKC（不自己造 Unicode 表） |

原则：不引入任何深度学习框架 crate、不引入 SQL parser crate、不引入嵌入式 KV/关系库。白名单是**上限**而非保证——离线构建环境缺少的 crate（如 `libloading`、`proptest`）一律改为自己实现，不放宽白名单。

---

## 16. 测试策略

1. **单元测试**：每个算子的数值正确性（与手工小例/解析解对拍）；梯度用**有限差分梯度检查**（gradient checking）；LSTM 门控形状与边界；分页/Cell 分配与删除；WAL 写入与重放。
2. **属性测试**（手写确定性伪随机生成器，见 15 节：`proptest` 离线不可得）：张量 reshape/转置后索引一致性；序列化-反序列化往返；SQL parser 对随机合法语句往返。种子固定以便复现失败用例。
3. **集成测试**（`tests/`）：
   - XOR 训练收敛（core+net）；
   - 微型语料上 TF-IDF 关键词排序符合预期；
   - INSERT 崩溃（kill 在 WAL 任意点）后恢复不丢已提交数据；
   - ALTER 加层后版本递增、旧权重保留、loss 不爆炸（冒烟门限）；
   - **推理**：构造"A 只与 B 共现、B 只与 C 共现、C 标注类别 X"的微型图，断言 REASON 在 1 跳下找不到 X、在 2~3 跳下以最高置信度命中 X 且证据文档正确；
   - **环路剪枝**：构造环图，断言路径不重复节点、搜索有界终止；
   - **弃判**：删掉证据文档后 REASON 必须返回弃判（7001/7002），不得返回无证据结论；
   - **校准**：在合成的过自信分数上做温度缩放，断言 ECE 下降；
   - **自定义层**：TextCNN composite 与等价显式层在相同权重下前向/梯度数值一致；Cell DSL 手写 GRU 与内置 GRU 前向一致；测试用 cdylib 插件完成加载→前向→反向→RELOAD 热替换全链路，插件内 panic 被隔离为错误码且进程存活；
   - **关系学习**：构造两表（一个字段值集合 99% 包含于另一字段、一对 A→B 无违反函数依赖、一对无关随机字段），断言分别识别为 fkey / functional，无关对低于门限；CONFIRM/REJECT 后推理图边权重与过滤生效；
   - **规则挖掘**：构造高频路径语料，断言 LEARN RULES 输出规则的 support/lift 与先验加分生效。
4. **存储与加密**：
   - MDB 头/页/WAL 编解码往返；篡改 1 bit 后打开必失败（AEAD + CRC 双校验）；
   - 口令错误/错误 KEK 解密必失败且不泄露明文；
   - 随机 kill 在 WAL 任意点后恢复，已提交事务全部在、未提交全部不在；
   - ROTATE KEY 前后数据逐字节一致（解密后比对）；
   - 磁盘字节断言：`data/` 下所有文件不含语料明文串（grep 二进制即应零命中）；
   - 数学函数对拍（以 naive 实现算参考值，STAT 数值断言）。
5. **端到端剧本**：启动 server → CLI 跑第 10 节完整走查脚本，作为 `cargo test --test e2e`。
6. CI（本地脚本）：`cargo fmt --check`、`cargo clippy -D warnings`、全部测试。

---

## 17. 开发里程碑

| 阶段 | 交付物 | 验收标准 |
|---|---|---|
| **M0 骨架** | workspace、9 个内部 crate（含 reason、plugin SDK）、CI 脚本、模块 api 占位 | `cargo build` 通过，模块间依赖方向被注释/工具固定 |
| **M1 核心引擎 + 自定义层** | Tensor、autograd、Dense/RNN/GRU/LSTM/Conv1D/Pool/Norm、Adam；Composite、Cell DSL、插件 ABI 与热加载 | XOR 与序列任务收敛；TextCNN 小语料收敛；梯度检查通过；插件全链路测试通过 |
| **M2 存储引擎（含 MDB 加密）** | 页、WAL、恢复、Catalog、MDB 头/页加密、密钥管理、BACKUP/RESTORE/ROTATE KEY、CREATE/INSERT/SELECT | **已验收**：`cargo test -p mightbe-store` 18 个单元 + 10 个集成全过；崩溃恢复（WAL 任意截断点）、单 bit 翻转拒绝、磁盘无明文、万行扫描/点查/删除均正确 |
| **M3 NLP** | 规范化、分词、词表、TF-IDF、TextRank、共现 | **已验收**：`cargo test -p mightbe-nlp` 19 个单元 + 8 个集成全过；中文样例关键词抽检合理、`mb_vocab`/`mb_df`/`mb_keywords`/`mb_assoc` 可查 |
| **M4 网络装配与训练** | 配置加载、校验、`TRAIN` 后台任务、检查点 | 按 doc_rnn.toml 训练，loss 下降、重启恢复 |
| **M5 联想检索** | embedding 余弦、PMI、`ASSOCIATE`/`NEAR`、hybrid 融合 | 关联词人工抽检；hybrid 融合 |
| **M6 推理 + 关系学习** | 统一 ReasonGraph、束搜索、证据回链、`INFER`/`REASON`/`REASON_PATHS`、`CALIBRATE`、弃判；字段画像与 `LEARN RELATIONS`、`LEARN RULES`、SCOPE 跨表推理、CONFIRM/REJECT、SQL 数学函数与 VECTOR 列、`EXPLAIN REASON` | 多跳图/弃判/校准/规则测试全过；外键与函数依赖自动发现测试通过 |
| **M7 热更新** | ALTER NETWORK/LIBRARY 全套、版本表、RESTORE、权重迁移 | 加/删层与换插件不停服；旧快照在途请求正常完成；推理自动跟新快照 |
| **M8 服务与端到端** | TCP 文本协议、server、cli、示例脚本 | 第 10 节走查全部通过 |
| **M9 加速与远期** | SIMD 后端、feature `gpu`（cudarc/opencl3）、feature `npu`（厂商 FFI 子图下沉）、MySQL 有线协议、B+Tree 索引、Self-Attention 内置层、推理图 IVF 索引、插件沙箱 | `mysql` 官方客户端可连；GPU/NPU 构建与 naive 对拍通过 |

**M1–M8 即第一版范围；每阶段结束都保持整体可编译可演示。**

已完成的里程碑按下表记录落点，便于回归时直接定位：

| 里程碑 | 代码落点 | 测试 |
|---|---|---|
| M2 | `crates/mightbe-store`（`page` / `wal` / `crypto` / `mdb` / `table` / `api`），一个库 = `data.mdb` + `data.mlog` 两个文件 | `crates/mightbe-store/tests/mdb.rs` |
| M3 | `crates/mightbe-nlp`（`normalize` / `tokenizer` / `vocab` / `keyword` / `cooccur` / `index` / `api`），停用词配置 `config/nlp/stopwords.zh.txt` + `stopwords.en.txt` | `crates/mightbe-nlp/tests/keyword.rs` |

M2 十条集成用例：`catalog_survives_reopen`、`ten_k_rows_scan_back_after_reopen`、
`one_flipped_bit_is_never_silently_absorbed`、`wrong_password_cannot_open_and_leaks_nothing`、
`corpus_plaintext_absent_on_disk`、`crash_at_any_wal_point_keeps_only_committed_rows`、
`rollback_discards_the_transaction`、`tampered_wal_frame_is_discarded`、
`rotate_key_preserves_every_row_byte_for_byte`、`backup_and_restore_roundtrip`。

---

## 18. 非目标与边界

- 第一阶段**不做 GPU / CUDA**，不做多机分布式训练；
- 不追求成为通用全功能数据库：无 JOIN 优化器（MVP 仅嵌套循环 + 主键过滤）、无窗口函数、无存储过程；
- 不支持用户自定义 Python 算子；网络结构只能由"层类型枚举 + 配置"组合（新增层类型在 core 中实现）；
- 不做多语言分词器的高精度承诺：CJK 走字+bigram 统计方案，目标是"无外部模型可用"，精度依赖词表统计；
- **推理器不做自由文本生成**：REASON 的结论只能是推理图上已有节点（词、文档、标签），且必须带证据；它回答"凭现有语料能关联/推出什么、把握多大"，不生成语料中不存在的事实；
- 推理深度有硬上限（配置，MVP 上限 5 跳），不做无界递归推理；规则推理（Datalog 式）与多模态证据列为远期项；
- **关系学习产出的是"带证据的候选关系"而非自动生效的约束**：学习到的 fkey/FD 不会自动建约束或改查询计划，需用户 CONFIRM 后才加权使用；不做全自动 schema 迁移；
- **Cell DSL 不追求图灵完备**：它只能表达有界可微的门控方程；任意复杂逻辑请走 Rust 插件，插件无沙箱（可信本地代码模型）；
- 安全模型默认本地监听；远程暴露需要用户自行配置，鉴权列为远期项。

---

*本 README 即项目蓝图。进入编码前，如对架构、SQL 方言、头格式字段或里程碑有修改意见，先更新此文档，再动代码。*

---

<div align="center">

<a href="https://github.com/yxpil/MightBe">
  <img width="100%" src="https://alittlecatgirlpanel.yxp.hk/card?repo=yxpil/MightBe" alt="gh-card · yxpil/MightBe" />
</a>

<sub>Powered by <a href="https://alittlecatgirlpanel.yxp.hk"><b>gh-card</b></a> · 粉色手写体 README 仓库名片</sub>

</div>
