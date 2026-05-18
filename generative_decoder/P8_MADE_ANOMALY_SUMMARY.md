# P8 MADE 异常诊断与修复总结

## 1. 背景

在 `P8` 的首轮 `MADE` even-`L` 正式尺度实验中，结果目录为
`net/mi_scaling/p8_made_even/`。当时得到的 `MI vs L` 为：

- `L=4 -> 0.823137`
- `L=6 -> 1.076084`
- `L=8 -> 2.772041`
- `L=10 -> 0.986748`
- `L=12 -> 1.280560`

`L=10,12` 相比 `L=8` 明显偏低，曲线呈现异常的非单调行为，需要判断是：

1. `MI` 评估公式有误；
2. `MADE` 架构容量不合适；
3. 训练策略导致大尺寸模型泛化退化。

## 2. 诊断过程

### 2.1 先排除 MI 公式错误

`decoding/mi_bipartite.py` 计算的是模型分布 `q` 上的

`I_q(A;B) = H_q(A) + H_q(B) - H_q(A,B)`。

具体做法是：

- 用 `AB` 模型估计 `H_q(A,B)` 和 `H_q(A)`
- 用 `BA` 模型估计 `H_q(B)`
- 通过模型自身采样与 `token_log_prob(...)` 进行 Monte Carlo 熵估计

因此这里的问题不是“物理分布 `p` 的公式写错”，而是“模型分布 `q` 学得是否稳定”。

### 2.2 检查训练记录

在首轮 `P8` 记录中：

- `L=10 AB` 的最佳验证 NLL 出现在 `epoch=21`，为 `62.221869`
- `L=12 AB` 的最佳验证 NLL 出现在 `epoch=22`，为 `90.914496`

但训练继续到 `epoch=100` 后，验证集表现反而恶化，说明大尺寸模型在早期达到较好点之后开始漂移。

### 2.3 检查原学习率调度器

首轮训练使用的是：

- `epoch=100`
- `StepLR(step_size=2000, gamma=0.9)`

问题在于：`step_size=2000` 远大于总训练轮数 `100`，所以学习率在整个 `P8` 训练期间实际上从未衰减。这意味着：

- 大尺寸模型在验证集上停滞后，仍继续用同样的学习率更新
- 结果表现为验证 NLL 先改善，再漂移变差

这被判断为本轮异常的主因。

## 3. 做过但未成为默认方案的尝试

### 3.1 限制参数量

我给 `MADE` 增加了 `made_max_params`，允许在超大系统上自动收缩 `width`。

但针对 `L=10` 的试验表明：

- 把 `width=64` 自动收缩到 `50` 后，最佳验证 NLL 变成 `62.474441`
- 这比旧基线 `62.221869` 更差

因此“默认缩宽”不是本问题的主修法。

### 3.2 改用 `relu`

针对 `L=10` 的 `relu` 试验也没有优于旧基线，反而更差。说明问题不在于简单更换激活函数。

## 4. 代码修改

### 4.1 `decoding/train_mi_syndrome.py`

核心修改：

- 将 `StepLR` 改为验证集驱动的 `ReduceLROnPlateau`
- 新增 `early stopping`
- 记录 `effective_width`
- 记录 `parameter_count`
- 记录 `epochs_trained`
- 将训练调度器参数写入 checkpoint 和 JSON record

现在推荐的正式 `MADE` 训练参数是：

- `width=64`
- `lr=1e-3`
- `lr_decay_factor=0.5`
- `lr_decay_patience=5`
- `min_lr=2e-4`
- `early_stop_patience=20`
- `dtype=float32`
- `device=cuda:0`

### 4.2 `decoding/args.py`

新增参数：

- `made_activation`
- `made_residual`
- `made_max_params`
- `weight_decay`
- `lr_decay_factor`
- `lr_decay_patience`
- `min_lr`
- `early_stop_patience`
- `early_stop_min_delta`

### 4.3 `decoding/run_mi_scale_sweep.py`

把上述训练参数接入自动化 sweep，使正式多尺度实验可以直接复用新策略。

### 4.4 `decoding/mi_bipartite.py`

评估脚本现在会按 checkpoint 中记录的配置重建 `MADE`：

- `effective_width`
- `made_activation`
- `made_residual`

避免“训练配置”和“评估配置”不一致。

## 5. 验证结果

### 5.1 L=10 端到端正式复跑

我用新训练策略重新跑了 `L=10` 的 `AB/BA + MI` 全流程，结果保存在：

- `net/mi_scaling/p8_made_plateau_l10/`

关键结果：

- 旧 `L=10 MI`: `0.986748`
- 新 `L=10 MI`: `2.934803`
- `delta MI`: `+1.948055`

对应训练指标：

- 旧 `L=10 AB best_val_nll`: `62.221869`
- 新 `L=10 AB best_val_nll`: `61.849250`
- 新 `L=10 BA best_val_nll`: `61.855588`

### 5.2 L=12 针对性验证

对 `L=12 AB` 的单次验证显示：

- 旧 `best_val_nll`: `90.914496`
- 新策略验证结果：`90.429637`

这说明 plateau 学习率调度对更大尺寸同样有效。

## 6. 当前结论

这次 `P8` 异常的结论如下：

1. `MI` 评估公式没有发现错误。
2. 主因是训练策略失效，而不是熵计算脚本出错。
3. 最关键的问题是首轮正式训练中的学习率调度根本没有在 `100` epoch 内生效。
4. 将训练改为 `plateau LR decay + early stopping` 后，`L=10` 的端到端 `MI` 结果明显改善。
5. `made_max_params` 应保留为可选保护阀，但不应作为默认修法。

## 7. 下一步建议

1. 用当前修复后的训练策略重跑 `P8 MADE even-L` 正式结果，优先补 `L=8,10,12`。
2. 新结果单独写入新目录，例如 `net/mi_scaling/p8_made_plateau_even/`，不要覆盖旧 `p8_made_even/`。
3. 对修复后的 checkpoint 再跑一轮 `P6` 稳定性分析，确认 `MI` 提升不是 Monte Carlo 偶然波动。
4. 在新 `MADE` 正式基线稳定后，再进入 `P9` 的 `NADE / TraDE_binary` 架构比较。
