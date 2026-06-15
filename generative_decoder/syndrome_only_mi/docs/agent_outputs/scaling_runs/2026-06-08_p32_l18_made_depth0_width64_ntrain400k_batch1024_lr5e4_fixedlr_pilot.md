# p32 L18 MADE depth0 width64 fixed-LR pilot

Date: 2026-06-08

Run id:

```text
p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot
```

Purpose:

Rerun the p31 same-scope `L=18`, MADE `depth=0,width=64`,
`n_train=400000`, batch-1024, lr-5e-4 diagnostic after fixing the warmup/LR
scheduler interaction in `decoding/train_mi_syndrome.py`.

Actual launch:

```text
launched: 2026-06-08 16:10:08 Asia/Shanghai
session: p32_l18_fixedlr_pilot
log: logs/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot.log
run_root: net/mi_scaling/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot
completed: 2026-06-08 17:45:06 Asia/Shanghai
```

Actual project command:

```bash
tmux new-session -d -s p32_l18_fixedlr_pilot \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot \
   L_VALUES="18" TRAIN_SEEDS="1 2 3 5" N_TRAIN=400000 DEVICE=cuda:0 \
   DEPTH=0 WIDTH=64 EPOCH=80 BATCH=1024 LR=0.0005 WEIGHT_DECAY=0.00001 \
   GRAD_CLIP_NORM=1.0 WARMUP_STEPS=1000 DIVERGENCE_NLL_THRESHOLD=1000 \
   MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=3 MIN_LR=0.00001 \
   EARLY_STOP_PATIENCE=20 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh \
   > logs/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot.log 2>&1'
```

Launch checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- `scripts/run_codex_gpu.sh "scripts/check_gpu_env.sh"` passed through the
  GPU wrapper on `cuda:0`; CUDA allocation, matmul, and project import checks
  passed on an `NVIDIA H100 PCIe`.
- Prelaunch duplicate checks found no existing p32 log, run root, or training
  process.
- Postlaunch checks confirmed direct project execution of
  `scripts/run_made_mi_ntrain400k.sh` and `decoding/train_mi_syndrome.py`, not
  a nested Codex prompt.

Configuration:

```text
model: MADE
depth: 0
requested_width: 64
effective_width: 64
activation: tanh
residual: false
n_train: 400000
n_val: 2000
n_test: 2000
optimizer: AdamW
lr: 0.0005
weight_decay: 0.00001
grad_clip_norm: 1.0
warmup_steps: 1000
divergence_nll_threshold: 1000
lr_decay_factor: 0.5
lr_decay_patience: 3
min_lr: 0.00001
early_stop_patience: 20
early_stop_min_delta: 0.01
epoch: 80
batch: 1024
mi_samples: 40000
bootstrap_samples: 200
partition: x-mid
p: 0.05
error_model: dep
```

## Per-Seed MI

| train_seed | H(A) | H(B) | H(A,B) | MI | bootstrap mean | bootstrap std | ci95 low | ci95 high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 94.656342 | 94.259132 | 180.800156 | 8.115318 | 8.123715 | 0.103360 | 7.924059 | 8.310284 |
| 2 | 95.348511 | 94.383064 | 181.645218 | 8.086357 | 8.089689 | 0.101965 | 7.892778 | 8.281984 |
| 3 | 95.225296 | 94.611588 | 182.017426 | 7.819458 | 7.829382 | 0.104661 | 7.625083 | 8.036801 |
| 5 | 95.360794 | 95.002838 | 181.853989 | 8.509644 | 8.518421 | 0.108236 | 8.317297 | 8.719769 |

## Training Diagnostics

Failure rule:

```text
failed if late AB or BA train/validation NLL reaches 1e3 or larger
```

| train_seed | order | best epoch | epochs trained | test NLL | late-NLL max | divergence flag | failed late-NLL rule |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | AB | 60 | 80 | 187.028367 | 185.296375 | no | no |
| 1 | BA | 56 | 76 | 187.022563 | 185.313500 | no | no |
| 2 | AB | 51 | 71 | 186.975000 | 185.318461 | no | no |
| 2 | BA | 58 | 78 | 187.145813 | 185.282086 | no | no |
| 3 | AB | 54 | 74 | 186.784383 | 185.011469 | no | no |
| 3 | BA | 56 | 76 | 186.925359 | 185.158227 | no | no |
| 5 | AB | 52 | 72 | 187.061180 | 185.299367 | no | no |
| 5 | BA | 57 | 77 | 186.912383 | 185.209359 | no | no |

All completed p32 pilot seeds pass the objective late-NLL training-failure
rule. The saved JSON records have `divergence.objective_training_failure=false`
for both AB and BA in every completed seed.

The LR histories no longer show p31-style scheduler reductions being reset
back to base LR. They decay monotonically after the first plateau reductions,
ending at `1e-05` for all AB/BA records.

## Aggregate

| L | n | seeds | n_seeds | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 18 | 648 | 1,2,3,5 | 4 | 8.132694244 | 0.284403924 | 0.034970443 | 7.819458008 | 8.509643555 | 0.104555521 | none |

The p32 pilot passes the predeclared continuation gate: no objective failures
and `cv <= 0.06`.

## Comparison With P31 Same Seeds

| source | seeds | mean MI | seed std | cv | min | max | mean bootstrap std | objective failures |
|---|---|---:|---:|---:|---:|---:|---:|---|
| p31 fixed-scope before LR fix | 1,2,3,5 | 8.067623 | 0.968598 | 0.120060 | 6.855270 | 9.218262 | 0.101760 | none |
| p32 fixed-LR pilot | 1,2,3,5 | 8.132694 | 0.284404 | 0.034970 | 7.819458 | 8.509644 | 0.104556 | none |

Same-seed MI changes:

| train_seed | p31 MI | p32 MI | p32 - p31 |
|---:|---:|---:|---:|
| 1 | 8.001343 | 8.115318 | 0.113976 |
| 2 | 9.218262 | 8.086357 | -1.131905 |
| 3 | 6.855270 | 7.819458 | 0.964188 |
| 5 | 8.195618 | 8.509644 | 0.314026 |

P32 stabilizes the four-seed pilot relative to p31 by removing the LR reset
behavior. Its mean remains substantially below the provisional p19 `L=18`
recommended row, so it should not replace p19 until the predeclared 8-seed
aggregate is complete and endpoint policy is updated.

## Decision

- Keep p32 pilot as diagnostic for now.
- Do not change the recommended `L=18` row yet.
- Continue under the same p32 configuration to train seeds `4,6,7,8`.
- After the 8-seed p32 aggregate is complete, re-evaluate objective failures,
  train-seed spread, and endpoint policy before any fit-doc promotion.

## Extension Completion

Extension launch:

```text
extension launched: 2026-06-08 20:28:46 Asia/Shanghai
extension completed: 2026-06-08 22:04:32 Asia/Shanghai
session: p32_l18_fixedlr_extend
mode: append to existing p32 log
```

Extension command:

```bash
tmux new-session -d -s p32_l18_fixedlr_extend \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot \
   L_VALUES="18" TRAIN_SEEDS="4 6 7 8" N_TRAIN=400000 DEVICE=cuda:0 \
   DEPTH=0 WIDTH=64 EPOCH=80 BATCH=1024 LR=0.0005 WEIGHT_DECAY=0.00001 \
   GRAD_CLIP_NORM=1.0 WARMUP_STEPS=1000 DIVERGENCE_NLL_THRESHOLD=1000 \
   MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=3 MIN_LR=0.00001 \
   EARLY_STOP_PATIENCE=20 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh \
   >> logs/p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot.log 2>&1'
```

Extension launch checks:

- No active p32/training process was present before launch.
- `scripts/run_codex_gpu.sh "scripts/check_gpu_env.sh"` passed through the GPU
  wrapper on `cuda:0`; CUDA allocation, matmul, and project import checks
  passed on an `NVIDIA H100 PCIe`.
- Postlaunch checks confirmed direct project execution of
  `scripts/run_made_mi_ntrain400k.sh` and `decoding/train_mi_syndrome.py`, not
  a nested Codex prompt.
- The run reused the p32 run root and `--skip-existing` behavior; completed
  pilot seeds `1,2,3,5` were not overwritten.

## Eight-Seed Per-Seed MI

| train_seed | H(A) | H(B) | H(A,B) | MI | bootstrap mean | bootstrap std | ci95 low | ci95 high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 94.656342 | 94.259132 | 180.800156 | 8.115318 | 8.123715 | 0.103360 | 7.924059 | 8.310284 |
| 2 | 95.348511 | 94.383064 | 181.645218 | 8.086357 | 8.089689 | 0.101965 | 7.892778 | 8.281984 |
| 3 | 95.225296 | 94.611588 | 182.017426 | 7.819458 | 7.829382 | 0.104661 | 7.625083 | 8.036801 |
| 4 | 95.151985 | 94.955765 | 182.634293 | 7.473457 | 7.479919 | 0.101444 | 7.284003 | 7.662968 |
| 5 | 95.360794 | 95.002838 | 181.853989 | 8.509644 | 8.518421 | 0.108236 | 8.317297 | 8.719769 |
| 6 | 94.759247 | 94.451752 | 181.262772 | 7.948227 | 7.950247 | 0.103999 | 7.776013 | 8.157695 |
| 7 | 94.410561 | 94.790672 | 180.209152 | 8.992081 | 8.993506 | 0.102297 | 8.801392 | 9.190878 |
| 8 | 94.949478 | 94.865059 | 181.655975 | 8.158562 | 8.162705 | 0.105105 | 7.967459 | 8.365078 |

## Eight-Seed Training Diagnostics

| train_seed | order | best epoch | epochs trained | test NLL | late-NLL max | divergence flag | failed late-NLL rule |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | AB | 60 | 80 | 187.028367 | 185.296375 | no | no |
| 1 | BA | 56 | 76 | 187.022563 | 185.313500 | no | no |
| 2 | AB | 51 | 71 | 186.975000 | 185.318461 | no | no |
| 2 | BA | 58 | 78 | 187.145813 | 185.282086 | no | no |
| 3 | AB | 54 | 74 | 186.784383 | 185.011469 | no | no |
| 3 | BA | 56 | 76 | 186.925359 | 185.158227 | no | no |
| 4 | AB | 53 | 73 | 186.754305 | 185.026234 | no | no |
| 4 | BA | 50 | 70 | 187.005063 | 184.916953 | no | no |
| 5 | AB | 52 | 72 | 187.061180 | 185.299367 | no | no |
| 5 | BA | 57 | 77 | 186.912383 | 185.209359 | no | no |
| 6 | AB | 56 | 76 | 186.888258 | 185.098445 | no | no |
| 6 | BA | 52 | 72 | 187.105961 | 185.149789 | no | no |
| 7 | AB | 58 | 78 | 186.967781 | 185.197031 | no | no |
| 7 | BA | 59 | 79 | 187.228102 | 185.312484 | no | no |
| 8 | AB | 57 | 77 | 186.747078 | 185.084383 | no | no |
| 8 | BA | 53 | 73 | 187.306969 | 185.276219 | no | no |

All eight p32 seeds pass the objective late-NLL training-failure rule. The
saved JSON records have `divergence.objective_training_failure=false` for both
AB and BA in every completed seed.

## Eight-Seed Aggregates

| subset | seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pilot | 1,2,3,5 | 4 | 8.132694 | 0.284404 | 0.034970 | 7.819458 | 8.509644 | 0.104556 | none |
| extension | 4,6,7,8 | 4 | 8.143082 | 0.634404 | 0.077907 | 7.473457 | 8.992081 | 0.103211 | none |
| all p32 | 1,2,3,4,5,6,7,8 | 8 | 8.137888 | 0.455173 | 0.055933 | 7.473457 | 8.992081 | 0.103883 | none |

The completed p32 8-seed aggregate has no objective failures and remains below
the `cv <= 0.06` usable-baseline gate. The extension block has larger spread
than the pilot block because seed 4 is low and seed 7 is high, but the pilot
and extension means are consistent.

## Endpoint Sensitivity

If the p32 8-seed mean were directly substituted for the current provisional
p19 `L=18` row while keeping `L=10,12,14,16` unchanged, the endpoint-included
fit would shift strongly:

| L18 source | L18 MI | window | 2 alpha | alpha | beta | RSS |
|---|---:|---|---:|---:|---:|---:|
| current p19 provisional | 9.573411 | L=10,12,14,16,18 | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| p32 fixed-LR 8-seed | 8.137888 | L=10,12,14,16,18 | 0.618689 | 0.309344 | -2.496670 | 0.813163 |
| current through L16 |  | L=10,12,14,16 | 0.744148 | 0.372074 | -4.002181 | 0.183562 |

The p32 8-seed mean is also essentially equal to the current recommended
`L=16` mean `8.133990`. This makes p32 a clean and important optimizer
diagnostic, but not a safe silent replacement for the recommended `L=18` row:
the training protocol changed only at the endpoint, and direct substitution
would mix protocols and strongly change the fitted slope.

## Final Decision

- Record p32 as a clean 8-seed fixed-LR diagnostic.
- Do not promote p32 to the recommended `L=18` row yet.
- Do not change the recommended fit values yet.
- The next decision should test protocol consistency before promotion: run the
  same fixed-LR configuration on at least `L=16` as a same-protocol anchor.
- If the fixed-LR `L=16` anchor is consistent with the current `L=16` row,
  then p32 can be reconsidered as a candidate endpoint policy. If it shifts
  `L=16` substantially, build a same-protocol fixed-LR scaling subset rather
  than mixing p32 with the older recommended rows.
