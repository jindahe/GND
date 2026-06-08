# Next Steps For Current Syndrome-Only MI Training

This document records only the current unfinished plan. Completed run details
belong in `docs/agent_outputs/scaling_runs/`, seed policy belongs in
`docs/SEED_POLICY.md`, and fit records belong in `docs/MI_FIT_POINTS.csv`,
`docs/MI_FIT_SUMMARY.md`, and `docs/MI_FIT_ANALYSIS.md`.

Do not use this file as a run-history archive. When a run completes, move the
run-specific conclusion into a report and leave here only the next unfinished
decision.

The active fit target remains:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

The current recommended fit inputs are the rows in `docs/MI_FIT_POINTS.csv`
with `include_in_recommended_fit=yes`. Diagnostic rows must not be silently
substituted into the recommended fit.

## Current Fit State

Recommended fit points:

| L | n | run_id | seeds | n_train | MI | seed_std | status |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 32 | `p8_made_plateau_long_468` | 1 |  | 1.074160 | 0.000000 | historical single point |
| 6 | 72 | `p8_made_plateau_long_468` | 1 |  | 1.513966 | 0.000000 | historical single point |
| 8 | 128 | `p18_l8_ntrain400k` | 8 | 400000 | 2.640582 | 0.132981 | usable baseline |
| 10 | 200 | `p26_l10_l12_made_depth0_width64_ntrain400k` | 8 | 400000 | 3.557084 | 0.190860 | usable baseline |
| 12 | 288 | `p26_l10_l12_made_depth0_width64_ntrain400k` | 8 | 400000 | 4.921827 | 0.210903 | usable baseline |
| 14 | 392 | `p17_l14_ntrain400k` | 8 | 400000 | 6.074064 | 0.313907 | usable; p27 confirms low position |
| 16 | 512 | `p16_l16_ntrain400k` | 8 | 400000 | 8.133990 | 0.382952 | usable baseline |
| 18 | 648 | `p19_l18_ntrain400k_pilot` | 8 | 400000 | 9.573411 | 0.574411 | provisional endpoint row |

Endpoint reporting until the next policy change:

- Current endpoint-included window:
  `L=10,12,14,16,18`, `2 alpha = 0.762241`, `alpha = 0.381120`.
- Endpoint-stable comparison window:
  `L=10,12,14,16`, `2 alpha = 0.744148`, `alpha = 0.372074`.
- State that `L=18` is provisional whenever quoting endpoint-included fits.
- Keep p32 fixed-LR `L=18` as diagnostic until a same-protocol anchor is
  completed and promoted by policy.

## Protocol Policy

Current protocol labels:

```text
v1 historical active protocol:
  MADE depth=0,width=64; n_train=400000; batch=512; lr=0.001;
  no warmup; no grad clip; no weight decay; lr_decay_patience=5;
  min_lr=0.0001; early_stop_patience=30.

v2 fixed-LR candidate protocol:
  MADE depth=0,width=64; n_train=400000; batch=1024; lr=0.0005;
  AdamW weight_decay=0.00001; grad_clip_norm=1.0; warmup_steps=1000;
  lr_decay_patience=3; min_lr=0.00001; early_stop_patience=20;
  early_stop_min_delta=0.01; epoch=80.

v3 MADE quality-scaling mainline, design only:
  MADE depth=0,width=64; AdamW lr=0.0005; weight_decay=0.00001;
  grad_clip_norm=1.0; batch=1024; data size and stopping rules are fixed
  functions of n=2L^2 rather than hand-tuned per L.
```

Protocol-fixed rule:

- Compare MI across `L` only within a fixed protocol track.
- If optimizer, batch, learning rate, warmup, clipping, scheduler, data size,
  architecture, or evaluation settings change, record the result as a
  diagnostic row under a new protocol track until promoted by an explicit
  endpoint or scaling-subset policy.
- Do not replace one `L` in the recommended fit with a different-protocol
  result while leaving neighboring `L` values on the old protocol.
- For fixed-`L` multiple MI values, prefer the current recommended-protocol
  clean 8-seed aggregate. Keep clean high-MI and low-MI seeds; exclude only
  objective saved-JSON training failures according to `docs/SEED_POLICY.md`.
- Use bootstrap std to judge MI evaluation noise, but choose protocol results
  primarily by objective training health, held-out NLL consistency,
  entropy-decomposition stability, and train-seed spread.

## Persistent MADE Quality-Scaling Design

Keep this section in `docs/NEXT_STEPS.md` until it is either promoted into a
dedicated protocol document or explicitly retired. Do not delete it merely
because one immediate p33/p32 decision completes.

Protocol id:

```text
v3_made_quality_scaling
```

Purpose:

Define a MADE-only training-quality mainline that can be extended to larger
`L` without ad hoc per-size tuning. This protocol ignores parameter-count
practicality for now and focuses on whether each `q_theta` is trained to a
comparable held-out quality under one predeclared rule set.

Fixed architecture and optimizer:

```text
model: MADE
depth: 0
width: 64
activation: tanh
residual: false
optimizer: AdamW
lr: 0.0005
weight_decay: 0.00001
grad_clip_norm: 1.0
batch: 1024
lr_decay_factor: 0.5
lr_decay_patience: 5
min_lr: 0.000005
early_stop_patience: 30
divergence_nll_threshold: 1000
dtype: float32
partition: x-mid
p: 0.05
error_model: dep
code_seed: 0
error_seed: 51697
split_seed: 0
bootstrap_samples: 200
```

Size-dependent rules:

```text
n = 2 L^2
n_train(L) = max(400000, ceil(800 * n))
n_val(L) = max(5000, ceil(10 * n))
n_test(L) = max(5000, ceil(10 * n))
pilot_max_optimizer_steps = 80000
formal_max_optimizer_steps = 160000
warmup_steps = min(5000, max(1000, ceil(0.05 * max_optimizer_steps)))
early_stop_min_delta = 0.00001 * n
epoch = ceil(max_optimizer_steps * batch / n_train(L))
```

Seed policy for v3:

```text
pilot seeds: 1,2,3,5
formal seeds: 1,2,3,4,5,6,7,8
stability extension: 9,10,11,12,13,14,15,16 only if a predeclared gate
  triggers, such as clean 8-seed cv in (0.06, 0.08].
```

Required v3 reporting:

- Report total and per-token `best_val_nll` and `test_nll` for AB and BA.
- Report per-token generalization gaps:
  `(test_nll - best_val_nll) / n`.
- Report `H(A)`, `H(B)`, `H(A,B)`, MI, bootstrap std, and seed-level entropy
  decomposition spread.
- Choose checkpoints only by validation NLL, never by final MI value.
- Exclude only objective saved-JSON training failures; keep clean high-MI and
  low-MI seeds.

V3 promotion gates:

- Pilot gate: seeds `1,2,3,5` have no objective failures, no AB/BA held-out
  NLL outlier, stable entropy decomposition, and `cv <= 0.06`.
- Formal gate: clean seeds `1..8`, `cv <= 0.06`, bootstrap std well below
  train-seed spread, and no mean drift after pilot extension.
- If clean 8-seed `0.06 < cv <= 0.08`, extend to seeds `9..16` under the same
  v3 protocol before any promotion.
- If clean 8-seed `cv > 0.08`, do not rescue by selecting seeds or changing
  one hyperparameter at that `L`; treat the protocol as unstable there.

V3 ladder, once explicitly launched:

1. Run `L=16` pilot to anchor against existing clean `L=16` behavior.
2. Run `L=18` pilot to compare against p32 fixed-LR endpoint health.
3. Run `L=20` pilot as the first larger-size feasibility check.
4. If the first three pilots are clean, extend them to formal seeds `1..8`.
5. Continue with `L=24`, `L=28`, `L=32`, and only then consider `L=40`.

## Priority 1. Run P33 Fixed-LR Anchor

Goal:

Run a same-protocol v2 fixed-LR anchor at `L=16` before deciding whether the
clean p32 `L=18` endpoint can be interpreted as a candidate endpoint or should
seed a separate same-protocol curve.

This is a protocol-consistency diagnostic, not a recommended fit update by
default.

Prelaunch checks:

```bash
scripts/run_mi_agent_audits.sh
scripts/run_codex_gpu.sh "scripts/check_gpu_env.sh"
```

Required audit marker:

```text
MI_AGENT_AUDITS_PASSED
```

Predeclared run:

```text
run_id: p33_l16_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_anchor
L: 16
architecture: MADE depth=0,width=64
n_train: 400000
n_val: 2000
n_test: 2000
train_seeds: 1,2,3,5
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
status: same-protocol anchor diagnostic only; fresh run root required
```

Launch command template:

```bash
tmux new-session -d -s p33_l16_fixedlr_anchor \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p33_l16_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_anchor \
   L_VALUES="16" TRAIN_SEEDS="1 2 3 5" N_TRAIN=400000 DEVICE=cuda:0 \
   DEPTH=0 WIDTH=64 EPOCH=80 BATCH=1024 LR=0.0005 WEIGHT_DECAY=0.00001 \
   GRAD_CLIP_NORM=1.0 WARMUP_STEPS=1000 DIVERGENCE_NLL_THRESHOLD=1000 \
   MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=3 MIN_LR=0.00001 \
   EARLY_STOP_PATIENCE=20 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh \
   > logs/p33_l16_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_anchor.log 2>&1'
```

After launch, verify:

- canonical log exists and shows `scripts/run_made_mi_ntrain400k.sh` and
  `decoding/train_mi_syndrome.py`, not a nested Codex prompt;
- run root exists;
- no duplicate writer is active.

P33 decision gate:

- If any seed has objective saved-JSON training failure, keep p33 diagnostic
  and do not promote p32.
- If p33 is clean and the `L=16` mean is consistent with current p16 within
  train-seed spread, then p32 can be reconsidered as a clean fixed-LR
  `L=18` endpoint candidate under an explicit endpoint policy.
- If p33 shifts `L=16` materially, do not mix p32 with current recommended
  rows. Either build a same-protocol fixed-LR subset at `L=10,12,14,16,18` or
  start the v3 quality-scaling track.
- Do not change the recommended `L=18` row or fit values before the p33 report
  and explicit endpoint-policy decision.

If p33 passes the pilot gate on seeds `1,2,3,5`, extend the same p33 run root
with `TRAIN_SEEDS="4 6 7 8"` and no other configuration changes. Record the
completed 8-seed anchor before any promotion decision.

## Priority 2. Prepare V3 Launch Helper

Before launching v3 pilots, add or reuse a helper that maps `L` and stage
(`pilot` or `formal`) to:

```text
n_train
n_val
n_test
max_optimizer_steps
epoch
warmup_steps
early_stop_min_delta
train_seeds
run_id
log path
```

The helper must implement the formulas in `v3_made_quality_scaling` exactly so
future v3 runs are generated from the protocol, not hand-tuned per L.

Initial v3 pilot values to verify:

| L | n | n_train | n_val | n_test | max_steps | epoch | warmup | early_stop_min_delta | seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 512 | 409600 | 5120 | 5120 | 80000 | 200 | 4000 | 0.00512 | 1,2,3,5 |
| 18 | 648 | 518400 | 6480 | 6480 | 80000 | 159 | 4000 | 0.00648 | 1,2,3,5 |
| 20 | 800 | 640000 | 8000 | 8000 | 80000 | 128 | 4000 | 0.00800 | 1,2,3,5 |

Do not start v3 training until p33 is reported or the user explicitly redirects
the active plan from v2 endpoint validation to v3 quality-scaling.

## Priority 3. Reporting Rules

- Do not include heavy `net/` artifacts in Git.
- Keep old rows in `docs/MI_FIT_POINTS.csv`; do not delete diagnostic rows.
- Mark only one row per `L` as `include_in_recommended_fit=yes`.
- Any changed recommended row must be accompanied by updates to
  `docs/MI_FIT_SUMMARY.md` and `docs/MI_FIT_ANALYSIS.md`.
- Any failed seed must have the failure signature recorded from saved training
  JSON, not only from MI value.
- Keep CNN/PixelCNN work as future architecture planning unless the user
  explicitly redirects away from MADE.
