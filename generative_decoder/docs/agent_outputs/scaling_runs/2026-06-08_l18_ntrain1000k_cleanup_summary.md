# L18 ntrain1000k cleanup summary

Date: 2026-06-08

Scope:

```text
p28_l18_made_depth0_width64_ntrain1000k_rerun
p30_l18_made_depth0_width64_gradclip_warmup_pilot
```

Purpose:

Summarize the completed and partial `L=18`, MADE `depth=0,width=64`,
`n_train=1000000` diagnostics before deleting heavy local training artifacts.

## p28 Result

Configuration:

```text
model: MADE depth=0,width=64
n_train: 1000000
optimizer: Adam
lr: 0.001
weight_decay: 0.0
grad_clip_norm: none
warmup_steps: none
train_seeds: 1,2,3,5
```

Per-seed MI:

| train_seed | MI | bootstrap std | objective failure |
|---:|---:|---:|---|
| 1 | 6.316429 | 0.100381 | yes, AB and BA |
| 2 | 8.736366 | 0.095521 | yes, AB and BA |
| 3 | 5.670700 | 0.099523 | yes, AB and BA |
| 5 | 8.816757 | 0.099568 | yes, AB and BA |

Aggregate:

| n | mean MI | seed std | cv | min | max | mean bootstrap std |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7.385063 | 1.628577 | 0.220523 | 5.670700 | 8.816757 | 0.098748 |

Conclusion:

- Increasing `n_train` to `1000000` under the original optimizer/schedule made
  the `L=18` endpoint less stable.
- All p28 pilot seeds failed the objective late-NLL rule in both AB and BA.
- p28 is diagnostic only and must not be promoted.

Detailed report:

```text
docs/agent_outputs/scaling_runs/2026-06-07_p28_l18_made_depth0_width64_ntrain1000k_rerun.md
```

## p30 Result

Configuration:

```text
model: MADE depth=0,width=64
n_train: 1000000
optimizer: AdamW
lr: 0.0003
weight_decay: 0.0001
grad_clip_norm: 1.0
warmup_steps: 2000
divergence_nll_threshold: 1000
train_seeds attempted: 1,2,3,5
completed MI seeds: 1,2,3
```

Per-seed MI at stop time:

| train_seed | MI | bootstrap std | objective failure |
|---:|---:|---:|---|
| 1 | 8.096962 | 0.111570 | no |
| 2 | 8.695381 | 0.101134 | no |
| 3 | 6.810791 | 0.106690 | no |

Aggregate over completed seeds:

| n | mean MI | seed std | cv | min | max | mean bootstrap std |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 7.867711 | 0.962983 | 0.122397 | 6.810791 | 8.695381 | 0.106465 |

Training diagnostics:

| train_seed | order | test NLL | late-NLL max | objective failure |
|---:|---|---:|---:|---|
| 1 | AB | 182.370199 | 183.457285 | no |
| 1 | BA | 182.287191 | 183.447016 | no |
| 2 | AB | 182.428234 | 183.401617 | no |
| 2 | BA | 182.517746 | 183.405613 | no |
| 3 | AB | 182.430688 | 183.389609 | no |
| 3 | BA | 182.338738 | 183.556496 | no |

Conclusion:

- P30 fixed the p28-style late-NLL divergence for completed seeds.
- P30 did not fix `L=18` MI train-seed spread.
- Completed-seed `cv = 0.122397`, far above the pilot stop gate
  `cv > 0.08`.
- Even the most favorable possible seed-5 value cannot reduce the four-seed
  pilot `cv` to `<= 0.08`.
- P30 is diagnostic only and must not be promoted or extended to seeds
  `4,6,7,8`.

Detailed report:

```text
docs/agent_outputs/scaling_runs/2026-06-08_p30_l18_made_depth0_width64_gradclip_warmup_interim.md
```

## Overall Decision

Do not change the recommended fit.

Keep the current endpoint policy:

```text
L=18 recommended/provisional row: p19_l18_ntrain400k_pilot
endpoint-stable comparison: through L=16
```

Interpretation:

- Larger `n_train` alone did not stabilize the active MADE endpoint.
- Stabilized optimizer/schedule removed objective NLL divergence but still left
  large clean MI seed spread.
- The unresolved problem is architecture/inductive-bias sensitivity, not MI
  bootstrap noise.
- Further work should not continue the `n_train=1000000` MADE
  `depth=0,width=64` branch. The next useful direction is an architecture
  change such as CNN/PixelCNN-style autoregressive modeling.

## Cleanup

The following heavy local artifacts were deleted after this summary was
written:

```text
net/mi_scaling/p28_l18_made_depth0_width64_ntrain1000k_rerun/
net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot/
logs/p28_l18_made_depth0_width64_ntrain1000k_rerun.background_attempt_empty.log
logs/p28_l18_made_depth0_width64_ntrain1000k_rerun.log
logs/p28_l18_made_depth0_width64_ntrain1000k_rerun.tmux_wrapper_attempt.log
logs/p30_l18_made_depth0_width64_gradclip_warmup_pilot.log
code/tor_n8_d2_k2_seed0
```

Approximate deleted repo-local size:

```text
p28 run root: 8.1G
p30 run root: 7.3G
logs: <1M
smoke code artifact: 4K
```
