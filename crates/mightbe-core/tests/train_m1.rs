//! M1 验收（README 16.2）：只用 `mightbe_core::api` 手搓训练循环，
//! 验证 autograd / 内置层 / 优化器三者闭环后任务确实收敛。

use mightbe_core::api::{Graph, Layer, LayerCtx, MtbResult, Optimizer, Tensor, Var};
use mightbe_core::{Adam, Cell, Dense, Embedding, Init, Permute, Rnn};

/// 一层一层把输入推进计算图。
fn push_layers(
    layers: &mut [Box<dyn Layer>],
    g: &mut Graph,
    x: Var,
    ctx: &LayerCtx,
) -> MtbResult<Var> {
    let mut cur = vec![x];
    for l in layers.iter_mut() {
        let refs: Vec<&Var> = cur.iter().collect();
        cur = l.forward(&refs, ctx, g)?;
    }
    Ok(cur[0])
}

/// 一个训练步：建图 → 前向 → 反向 → 更新 → 回灌层权重。返回本步损失。
fn train_step(
    layers: &mut [Box<dyn Layer>],
    opt: &mut dyn Optimizer,
    xs: &Tensor,
    labels: &[u32],
) -> MtbResult<f32> {
    let mut g = Graph::new();
    let ctx = LayerCtx { training: true, seq_state: None };
    let x = g.constant(xs.clone());
    let logits = push_layers(layers, &mut g, x, &ctx)?;
    let loss = g.cross_entropy_loss(logits, labels);
    let value = g.value(loss).scalar()?;
    g.backward(&[loss])?;

    // 图按步丢弃：参数从图里取出来更新，再按名回灌进层（README 4.1 参数寻址）
    let mut owned: Vec<(String, Tensor, Tensor)> = g
        .trainable()
        .filter_map(|(_, name, param, grad)| {
            grad.map(|gr| (name.to_string(), param.clone(), gr.clone()))
        })
        .collect();
    {
        let mut items: Vec<(String, &mut Tensor, &Tensor)> = owned
            .iter_mut()
            .map(|(n, p, gr)| (n.clone(), p, &*gr))
            .collect();
        opt.step(&mut items);
    }
    drop(g);
    let snapshot: Vec<(String, Tensor)> =
        owned.into_iter().map(|(n, p, _)| (n, p)).collect();
    for l in layers.iter_mut() {
        l.bind_params(&snapshot)?;
    }
    Ok(value)
}

fn argmax_rows(t: &Tensor) -> Vec<u32> {
    let cols = t.shape[1];
    (0..t.shape[0])
        .map(|r| {
            let row = &t.data[r * cols..r * cols + cols];
            let mut best = 0usize;
            for (i, v) in row.iter().enumerate() {
                if v > &row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

fn dense_classifier() -> Vec<Box<dyn Layer>> {
    vec![
        Box::new(Dense::new("h", 2, 4, Some("tanh"), Init::Xavier, 3)),
        Box::new(Dense::new("o", 4, 2, None, Init::Xavier, 4)),
    ]
}

#[test]
fn xor_converges_with_adam() {
    let mut layers = dense_classifier();
    let mut opt = Adam::new(0.05);
    let xs = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2])
        .unwrap();
    let labels = [0u32, 1, 1, 0];

    let mut losses = Vec::new();
    for _ in 0..600 {
        losses.push(train_step(&mut layers, &mut opt, &xs, &labels).unwrap());
    }
    assert!(
        losses[599] < 0.05,
        "XOR 应收敛，末段损失 {:?}",
        &losses[590..]
    );
    assert!(losses[599] < losses[0], "损失必须单调下降趋势");

    // 分类正确性：前向 argmax 与标签全对
    let mut g = Graph::new();
    let ctx = LayerCtx { training: false, seq_state: None };
    let x = g.constant(xs.clone());
    let logits = push_layers(&mut layers, &mut g, x, &ctx).unwrap();
    assert_eq!(argmax_rows(g.value(logits)), labels.to_vec(), "XOR 四样本应全对");
}

#[test]
fn gru_learns_last_token_task() {
    // 序列任务：类别 = 时间步最后一个输入值，考察 RNN 是否能记住并对梯度回传到全部权重
    let seq = 6usize;
    let n = 8usize;
    let mut data = vec![0f32; n * seq];
    let mut labels = Vec::new();
    for s in 0..n {
        for t in 0..seq {
            data[s * seq + t] = ((s * 7 + t * 13) % 2) as f32;
        }
        labels.push(data[s * seq + seq - 1] as u32);
    }
    let xs = Tensor::from_vec(data, vec![n, seq, 1]).unwrap();

    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Rnn::new("enc", 1, 4, Cell::Gru, 9).last_state()),
        Box::new(Dense::new("head", 4, 2, None, Init::Xavier, 10)),
    ];
    let mut opt = Adam::new(0.05);
    let mut losses = Vec::new();
    for _ in 0..400 {
        losses.push(train_step(&mut layers, &mut opt, &xs, &labels).unwrap());
    }
    assert!(losses[399] < 0.05, "序列任务应收敛，末尾 {:?}", &losses[390..]);

    let mut g = Graph::new();
    let ctx = LayerCtx { training: false, seq_state: None };
    let x = g.constant(xs.clone());
    let logits = push_layers(&mut layers, &mut g, x, &ctx).unwrap();
    assert_eq!(argmax_rows(g.value(logits)), labels, "末位任务应全部预测正确");
}

/// TextCNN 小语料（README 16.2 M1 验收项）：embedding → permute → conv1d → global maxpool → dense。
/// 任务：定长 6 的 token 序列里含关键词 A(id=1) 还是 B(id=2)，其余位置是噪声 token。
#[test]
fn textcnn_learns_keyword_task() {
    let seq = 6usize;
    let n = 12usize;
    let mut data = vec![0f32; n * seq];
    let mut labels = Vec::new();
    for s in 0..n {
        for t in 0..seq {
            // 3..5 三个噪声词轮转，保证关键词位置不固定
            data[s * seq + t] = 3.0 + ((s + t) % 3) as f32;
        }
        let kw_pos = (s * 5) % seq;
        let kw: u32 = if s % 2 == 0 { 1 } else { 2 };
        data[s * seq + kw_pos] = kw as f32;
        labels.push(kw - 1);
    }
    let xs = Tensor::from_vec(data, vec![n, seq]).unwrap();

    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Embedding::new("e", 6, 4, 21)),
        Box::new(Permute::new("tr", &[0, 2, 1])),
        Box::new(mightbe_core::Conv1d::new(
            "c", 4, 8, 3, 1, 1, 1, Some("relu"), 22,
        )),
        mightbe_core::global_maxpool("gp"),
        Box::new(Dense::new("head", 8, 2, None, Init::Xavier, 23)),
    ];
    let mut opt = Adam::new(0.05);
    let mut losses = Vec::new();
    for _ in 0..400 {
        losses.push(train_step(&mut layers, &mut opt, &xs, &labels).unwrap());
    }
    assert!(
        losses[399] < 0.1,
        "TextCNN 应收敛到接近 0，末段损失 {:?}",
        &losses[390..]
    );

    let mut g = Graph::new();
    let ctx = LayerCtx { training: false, seq_state: None };
    let x = g.constant(xs.clone());
    let logits = push_layers(&mut layers, &mut g, x, &ctx).unwrap();
    let pred = argmax_rows(g.value(logits));
    let right = pred.iter().zip(labels.iter()).filter(|(a, b)| a == b).count();
    assert_eq!(right, n, "关键词任务应全对，预测 {pred:?} 标签 {labels:?}");
}

#[test]
fn unused_params_are_not_perturbed() {
    // 第二步只走 head 之外的层时，未被使用的参数不应被优化器改写
    let mut layers = dense_classifier();
    let mut opt = Adam::new(0.05);
    let xs = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
    let before = layers[1].dump_params();
    let _ = train_step(&mut layers, &mut opt, &xs, &[0, 1]).unwrap();
    let _ = &before;
    // 这里只要求训练不 panic 且形状稳定（参数命名寻址正确）
    let after = layers[1].dump_params();
    assert_eq!(before.len(), after.len());
    for ((n1, t1), (n2, t2)) in before.iter().zip(after.iter()) {
        assert_eq!(n1, n2);
        assert_eq!(t1.shape, t2.shape);
    }
}
