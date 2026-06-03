# Stability Checklist

这份清单用于判断某个固定配置在给定 `L` 上是否已经“足够稳定”，以及它是否可以作为更大 `L` 的外推基线。

适用对象：

- `toric code syndrome-only MI`
- `MADE / NADE / TraDE_binary` 的训练稳定性比较
- 固定 `L` 下的多 `train_seed` 实验

---

## 0. Syndrome-Only MI 理论回归门

在运行真实训练或尺度外推前，先跑固定的理论与排序回归门：

```bash
scripts/run_mi_agent_audits.sh
```

该命令会检查：

- pair model 闭式解与自回归熵分解一致
- `p=0` 时互信息为 0
- depolarizing 单 CSS 投影的 `p=3/4` 无偏点互信息为 0
- `AB` prefix 对应 `H(A)`
- `BA` prefix 对应 `H(B)`
- `AB` suffix 没有被误当成 `H(B)`
- 所有输出默认使用自然对数，单位为 nats

通过标志：

```text
MI_AGENT_AUDITS_PASSED
```

---

## 1. 先区分两类稳定性

在做结论前，先明确区分：

- **评估稳定性**
  - 由 `mi_samples` 和 bootstrap 决定
  - 主要看 Monte Carlo 波动
- **训练稳定性**
  - 由 `train_seed`、`lr`、`batch`、`n_train`、学习策略决定
  - 主要看不同训练轨迹是否得到相近 `MI`

如果评估波动已经很小，但训练波动仍很大，则当前主问题仍在训练侧。

---

## 2. 每个 `L` 必须至少汇总的量

对每个固定 `L` 和固定训练配置，至少汇总：

- `MI mean` across `train_seed`
- `std_across_train_seeds`
- `min(MI)`
- `max(MI)`
- `range = max - min`
- `mean bootstrap std`
- `cv = std_across_train_seeds / mean(MI)`

并同时保留训练记录：

- `best_epoch`
- `epochs_trained`
- `best_val_nll`
- `test_nll`

---

## 3. 三级判断标准

### A. 评估已稳定

满足下面两条可认为评估侧已经基本稳定：

- `mean bootstrap std` 明显小于 `std_across_train_seeds`
- 增大 `mi_samples` 后，均值变化很小，只是误差条缩窄

这表示：

- 继续加 `mi_samples` 不是当前主优先级

### B. 可用基线

如果目标是“把这套配置拿去外推到更大 `L`”，建议至少满足：

- `cv <= 0.06`
- `std_across_train_seeds <= 0.15 ~ 0.18`
- `train_seed` 数量至少 `>= 5`
- `best_val_nll` 没有明显异常点
- 不同 `train_seed` 的 `MI` 没有明显双峰或分裂成两团

满足这一层，就可以认为：

- 当前配置已经是“可工作的基线配置”

### C. 正式结果级稳定

如果目标是“把该 `L` 的结果当成正式报告值”，建议更严格：

- `cv <= 0.04 ~ 0.05`
- `std_across_train_seeds <= 0.10`
- `train_seed` 数量至少 `>= 8`
- `std_across_train_seeds / mean_bootstrap_std <= 2`
- 再补新 `train_seed` 后，均值不明显漂移

满足这一层，才适合把均值当成当前阶段的正式结果口径。

---

## 4. 如果不稳定，按什么顺序排查

如果某个 `L` 不稳定，建议按下面顺序调：

1. 先看是否是评估问题
   - bootstrap 是否已经很小
   - `mi_samples` 是否已经足够
2. 再扩大 `train_seed`
   - 避免被少量 seed 误导
3. 先调训练噪声
   - `lr`
   - `batch`
4. 再加数据量
   - `n_train`
5. 最后才改学习策略
   - plateau patience
   - `min_lr`
   - early-stop patience

不建议一开始就同时改：

- `lr`
- `batch`
- `n_train`
- 学习策略

否则无法归因。

---

## 5. 对更大 `L` 的外推原则

当某个 `L` 满足“可用基线”标准后：

- 可以拿它的配置去做更大 `L` 的 pilot

对更大 `L` 的第一轮实验建议：

- 先固定当前最稳配置
- 每个新 `L` 只跑 `train_seed = 1..3`
- 先看是否明显失稳

如果新 `L` 失稳，再按顺序增加：

1. `n_train`
2. `batch`
3. 学习策略复杂度

---

## 6. 最终输出格式

对每个 `L`，建议最终形成一张固定格式记录：

- `L`
- 推荐 `n_train`
- 推荐 `lr`
- 推荐 `batch`
- 推荐学习策略
- `MI mean`
- `std_across_train_seeds`
- `cv`
- `mean bootstrap std`
- 训练 seed 数量
- 当前判断级别
  - `评估已稳定`
  - `可用基线`
  - `正式结果级稳定`

这样后面随着 `L` 增大，就能逐步形成一张按尺寸缩放的稳定训练经验表。
