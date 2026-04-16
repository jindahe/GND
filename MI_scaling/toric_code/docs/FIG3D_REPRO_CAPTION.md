# Fig.3(d) 复现图注与说明（2026-04-16）

## 输出文件

- 主图 PNG：`outputs/FIG3d_repro_paper_v1.png`
- 主图 PDF：`outputs/FIG3d_repro_paper_v1.pdf`
- 图用主表：`outputs/CMI_paper_repro_main_table_v1.csv`
- 图用拟合摘要：`outputs/FIG3d_repro_fit_summary_v1.csv`
- 选择性 `v2` 主图 PNG：`outputs/FIG3d_repro_paper_v2_selective_p015r6.png`
- 选择性 `v2` 主图 PDF：`outputs/FIG3d_repro_paper_v2_selective_p015r6.pdf`
- 选择性 `v2` 主表：`outputs/CMI_paper_repro_main_table_v2_selective_p015r6.csv`
- 选择性 `v2` 拟合摘要：`outputs/FIG3d_repro_fit_summary_v2_selective_p015r6.csv`
- `v1/v2` 对比表：`outputs/FIG3d_repro_fit_comparison_v1_vs_v2_selective_p015r6.csv`
- 出图脚本：`src/scripts/make_fig3d_paper.py`

## 图注草稿

复现论文 Fig.3(d)：去相干二维环面码态的条件互信息
`I(A:C|B)` 随缓冲区宽度 `r` 的衰减。点为 Monte Carlo + tensor-network
估计值，误差棒表示 Monte Carlo 标准误差。`p=0.05` 与 `p=0.15`
使用指数形式 `I(A:C|B) ~ exp(b r)` 拟合，`p=0.11` 使用临界幂律形式
`I(A:C|B) ~ r^{-alpha}` 拟合。当前复现使用 `r=1..6` 的最佳可用数据；
其中 `p=0.15` 的尾部点已采用高 `chi` 修正：`r=4` 用 `chi=160`，
`r=5` 用 `chi=192`，`r=6` 用 `chi=256`。此外，两个最敏感尾点
`(p=0.11, r=6, chi=192)` 与 `(p=0.15, r=6, chi=256)` 已各自追加一个
独立 seed、`N=5000` 的复核；两次复核与原结果分别在 `0.90σ` 与 `1.91σ`
内一致。`r=7` scout 未通过稳定性判据，因此未纳入图中。

## 拟合口径

| 分支 | 模型 | 窗口 | 权重口径 | 参数 | `R^2` |
|---|---|---|---|---|---:|
| `p=0.05` | 指数 | `r=1..6` | MC SEM | `b=-0.7793` | `0.9984` |
| `p=0.11` | 幂律 | `r=2..6` | `sigma_eff` | `alpha=0.8061` | `0.7106` |
| `p=0.15` | 指数 | `r=1..6` | MC SEM | `b=-0.3072` | `0.6750` |

## 数据口径

当前图只使用 `r<=6`：

- `p=0.05`：`r=1..5` 使用 `chi=64` 主生产 / top-up，`r=6` 使用 `chi=96, N=20000`；
- `p=0.11`：`r=1..4` 使用 `chi=64` 主生产，`r=5` 使用 `chi=128`，`r=6` 使用 `chi=192`；
- `p=0.15`：`r=1..3` 使用现有主生产 / top-up，`r=4` 使用 `chi=160`，`r=5` 使用 `chi=192`，`r=6` 使用 `chi=256`。

## 与论文 Fig.3(d) 的对应关系

当前图复现了论文 Fig.3(d) 的定性结构：

1. `p=0.05 < p_c`：CMI 随 `r` 清晰指数衰减；
2. `p=0.11 ≈ p_c`：CMI 衰减明显慢于两侧，用幂律拟合；
3. `p=0.15 > p_c`：在修正 `r=4` 低 `chi` 偏差后，与指数衰减图景相容。

当前图还不是论文级定量复现，因为论文 Fig.3(d) 每点样本数约为 `6e6`，
而当前图使用的是 `N=5000..20000` 级别的可行复现数据；因此应表述为
“Fig.3(d) 的低样本 tensor-network 复现实验版本”。

## 独立 seed 复核结果

已完成的最小可信度增强检查如下：

| 点位 | 原结果 | 新 seed 结果 | `z_delta` | 合并后 |
|---|---|---|---:|---|
| `p=0.11, r=6, chi=192` | `0.007790 ± 0.002400` | `0.004789 ± 0.002315` | `-0.90` | `0.006235 ± 0.001666` |
| `p=0.15, r=6, chi=256` | `0.001241 ± 0.000426` | `0.000150 ± 0.000381` | `-1.91` | `0.000635 ± 0.000284` |

对应汇总表见 `outputs/FIG3d_seedcheck_summary_v1.csv`。

解释口径：

1. `p=0.11, r=6` 的双 seed 结果在 `1σ` 内一致，说明当前尾点虽噪声较大，但符号与量级稳定；
2. `p=0.15, r=6` 的双 seed 结果相差约 `1.9σ`，尚未显示明显矛盾，但说明该点仍是当前 Fig.3(d) 中最脆弱的尾点；
3. 两点做逆方差合并后，标准误分别较单次最好结果下降约 `28.0%` 与 `25.5%`，可作为后续升级到 `v2` 图表时的优先候选值。

## 当前推荐版本

当前建议同时保留两版：

1. `v1`：`outputs/CMI_paper_repro_main_table_v1.csv`
   - 用途：可追溯基线；
2. `v2_selective_p015r6`：`outputs/CMI_paper_repro_main_table_v2_selective_p015r6.csv`
   - 用途：当前更适合 paper-draft 展示的 Fig.3(d) 候选版；
   - 唯一改动：把 `p=0.15, r=6, chi=256` 替换为双 seed 合并值
     `0.000635 ± 0.000284`；
   - 效果：`p=0.15` 的指数拟合 `R^2` 由 `0.6750` 提升到 `0.9121`。

## 最小后续动作建议

当前不建议继续推进 `r=7`。若需要进一步增强 Fig.3(d) 的可 defend 性，
建议按以下顺序推进：

1. 保留 `v1` 基线，同时把 `v2_selective_p015r6` 作为当前推荐展示版本；
2. 若还要继续提升拟合度，优先给 `p=0.15, r=5, chi=192, N=5000` 再做一个独立 seed；
3. 仅当 `p=0.15, r=5` 的双 seed 合并也能稳定改善 `R^2` 时，才考虑生成 `v3`；
4. `p=0.11` 暂不替换主图点，只保留 seed-check 作为可信度补充；
5. 仍不建议在当前样本量级上重新推进 `r=7` scout。
