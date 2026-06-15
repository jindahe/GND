# p31 L18 MADE depth0 width64 batch1024 lr5e-4 pilot

Date: 2026-06-08

Run id:

```text
p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot
```

Purpose:

Test whether the active `L=18`, MADE `depth=0,width=64`, `n_train=400000`
endpoint becomes stable with larger batch, lower LR than the original p19
schedule, AdamW weight decay, gradient clipping, and warmup.

Actual project command, reconstructed from the saved log and JSON records:

```bash
env BASE_ROOT=net/mi_scaling/p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot \
  L_VALUES="18" TRAIN_SEEDS="1 2 3 5" N_TRAIN=400000 DEVICE=cuda:0 \
  DEPTH=0 WIDTH=64 EPOCH=80 BATCH=1024 LR=0.0005 WEIGHT_DECAY=0.00001 \
  GRAD_CLIP_NORM=1.0 WARMUP_STEPS=1000 DIVERGENCE_NLL_THRESHOLD=1000 \
  MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=3 MIN_LR=0.00001 \
  EARLY_STOP_PATIENCE=20 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh
```

Artifacts:

```text
net/mi_scaling/p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot/
logs/p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot.log
```

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
| 1 | 96.562164 | 94.143372 | 182.704193 | 8.001343 | 8.002507 | 0.101133 | 7.812861 | 8.185358 |
| 2 | 94.600098 | 95.266739 | 180.648575 | 9.218262 | 9.221417 | 0.099785 | 9.024080 | 9.402480 |
| 3 | 95.807404 | 95.372772 | 184.324905 | 6.855270 | 6.864587 | 0.104663 | 6.656600 | 7.078956 |
| 5 | 94.837479 | 95.470596 | 182.112457 | 8.195618 | 8.194273 | 0.101457 | 8.008009 | 8.380875 |

## Training Diagnostics

Failure rule:

```text
failed if late AB or BA train/validation NLL reaches 1e3 or larger
```

| train_seed | order | best epoch | epochs trained | test NLL | late-NLL max | divergence flag | failed late-NLL rule |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | AB | 36 | 56 | 188.155125 | 188.109297 | no | no |
| 1 | BA | 35 | 55 | 188.336047 | 188.041500 | no | no |
| 2 | AB | 36 | 56 | 188.169359 | 188.204320 | no | no |
| 2 | BA | 35 | 55 | 188.197094 | 187.906586 | no | no |
| 3 | AB | 35 | 55 | 188.359719 | 188.033406 | no | no |
| 3 | BA | 36 | 56 | 188.189430 | 187.943211 | no | no |
| 5 | AB | 36 | 56 | 188.164680 | 188.047039 | no | no |
| 5 | BA | 35 | 55 | 188.060281 | 187.847438 | no | no |

All completed p31 seeds pass the objective late-NLL training-failure rule.
The saved JSON records have `divergence.objective_training_failure=false` for
both AB and BA in every completed seed.

## Aggregate

| L | n | seeds | n_seeds | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 18 | 648 | 1,2,3,5 | 4 | 8.067623138 | 0.968597616 | 0.120059849 | 6.855270386 | 9.218261719 | 0.101759538 | none |

This aggregate fails the pilot spread gate. The mean bootstrap std is about
`0.101760`, much smaller than the train-seed std `0.968598`, so the failure is
training/configuration spread rather than MI Monte Carlo noise.

## LR Scheduler Diagnostic

The p31 LR histories show repeated scheduler reductions followed by resets to
the base LR. For example, p31 seed 1 AB contains:

```text
... 0.0005, 0.00025, 0.0005, 0.0005, 0.0005, 0.00025, 0.0005 ...
```

The cause was the warmup helper in `decoding/train_mi_syndrome.py`: when
`warmup_steps > 0`, it continued writing `args.lr` after warmup because the
warmup scale saturated at `1.0`. That made `ReduceLROnPlateau` reductions
non-persistent.

## Interpretation

- P31 is diagnostic only.
- Do not extend p31 to seeds `4,6,7,8`.
- Do not promote p31 into the recommended fit.
- The large spread is visible in entropy decomposition, especially seed 3's
  high `H(A,B) = 184.324905`, not in bootstrap noise.
- The next same-scope diagnostic must use a fresh run id after fixing the LR
  warmup/scheduler interaction.

## Next Decision

The next predeclared diagnostic remains:

```text
p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot
```

Before p32 launch, rerun `scripts/run_mi_agent_audits.sh` and
`scripts/run_codex_gpu.sh "scripts/check_gpu_env.sh"`.
