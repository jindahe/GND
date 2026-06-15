# p30 L18 MADE depth0 width64 gradclip warmup interim

Date: 2026-06-08

Run id:

```text
p30_l18_made_depth0_width64_gradclip_warmup_pilot
```

Purpose:

Test a stabilized `L=18`, MADE `depth=0,width=64`, `n_train=1000000`
diagnostic after p28 showed severe late-NLL divergence under the unchanged
optimizer and schedule.

Configuration:

```text
model: MADE
depth: 0
requested_width: 64
effective_width: 64
activation: tanh
residual: false
n_train: 1000000
n_val: 2000
n_test: 2000
optimizer: AdamW
lr: 0.0003
weight_decay: 0.0001
grad_clip_norm: 1.0
warmup_steps: 2000
divergence_nll_threshold: 1000
lr_decay_factor: 0.5
lr_decay_patience: 2
min_lr: 0.00001
early_stop_patience: 8
early_stop_min_delta: 0.01
epoch: 60
batch: 512
mi_samples: 40000
bootstrap_samples: 200
partition: x-mid
p: 0.05
error_model: dep
```

Initial launch:

```bash
tmux new-session -d -s p30_l18_gradclip_warmup_pilot \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot \
   L_VALUES="18" TRAIN_SEEDS="1 2 3 5" N_TRAIN=1000000 DEVICE=cuda:0 \
   DEPTH=0 WIDTH=64 EPOCH=60 BATCH=512 LR=0.0003 WEIGHT_DECAY=0.0001 \
   GRAD_CLIP_NORM=1.0 WARMUP_STEPS=2000 DIVERGENCE_NLL_THRESHOLD=1000 \
   MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=2 MIN_LR=0.00001 \
   EARLY_STOP_PATIENCE=8 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh \
   > logs/p30_l18_made_depth0_width64_gradclip_warmup_pilot.log 2>&1'
```

Artifacts:

```text
net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot/
logs/p30_l18_made_depth0_width64_gradclip_warmup_pilot.log
```

Launch checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- CPU smoke test covered `grad_clip_norm`, `warmup_steps`, `max_train_steps`,
  MI evaluation, and JSON records.
- Guard smoke test confirmed `divergence_nll_threshold` records objective
  training failure when exceeded.
- CUDA wrapper check passed before launch on `cuda:0`, host GPU
  `NVIDIA H100 PCIe`.

## Completed Seed 1

Per-seed MI:

| train_seed | H(A) | H(B) | H(A,B) | MI | bootstrap mean | bootstrap std | ci95 low | ci95 high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 93.522827 | 93.567940 | 178.993805 | 8.096962 | 8.105952 | 0.111570 | 7.902224 | 8.300702 |

Training diagnostics:

| train_seed | order | best epoch | epochs trained | test NLL | late-NLL max | divergence flag | failed late-NLL rule |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | AB | 28 | 36 | 182.370199 | 183.457285 | no | no |
| 1 | BA | 30 | 38 | 182.287191 | 183.447016 | no | no |

Seed 1 is a clean diagnostic result. It has no objective late-NLL failure and
no recorded divergence state.

Same-seed comparison:

| source | n_train | MI | AB test NLL | BA test NLL | objective failures |
|---|---:|---:|---:|---:|---|
| `p19_l18_ntrain400k_pilot` | 400000 | 9.065102 | 187.394746 | 187.547609 | none for seed 1 |
| `p28_l18_made_depth0_width64_ntrain1000k_rerun` | 1000000 | 6.316429 | 191.468902 | 192.596582 | AB and BA |
| `p30_l18_made_depth0_width64_gradclip_warmup_pilot` | 1000000 | 8.096962 | 182.370199 | 182.287191 | none for seed 1 |

Entropy difference relative to p19 seed 1:

| quantity | p30 - p19 |
|---|---:|
| H(A) | -1.512276 |
| H(B) | -0.739815 |
| H(A,B) | -1.283951 |
| MI | -0.968140 |

Interpretation:

- P30 seed 1 fixes the p28-style late-NLL divergence on this seed.
- P30 seed 1 has substantially better AB/BA test NLL than p19 seed 1.
- MI is lower than p19 seed 1 because `H(A)+H(B)` drops more than `H(A,B)`.
- The MI drop is much larger than bootstrap noise, so it is a clean
  training/configuration effect rather than MI Monte Carlo noise.

## Interrupted Seed 2

Seed 2 AB completed cleanly and wrote a checkpoint plus JSON record:

```text
net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot/models/made_tor_n648_d18_k2_seed0_er0.05_dep_tseed2_AB_xmid.pt
net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot/models/records/made_tor_n648_d18_k2_seed0_er0.05_dep_tseed2_AB_xmid.json
```

Seed 2 BA stopped before writing a training JSON record because the CUDA
runtime reported:

```text
torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error
```

This is classified as an infrastructure/runtime failure for the interrupted
launch, not an objective training failure. Under the seed policy, objective
late-NLL failure requires saved training JSON evidence.

## Interim Decision

- Do not update the recommended fit.
- Keep p19 as the sole provisional recommended `L=18` row for continuity.
- Resume the p30 pilot from existing artifacts after a fresh CUDA check.
- Reuse completed seed 1 and seed 2 AB artifacts; do not delete or overwrite
  partial artifacts without explicit approval.
- Continue with predeclared pilot seeds `2,3,5` under the same configuration,
  using `--skip-existing` behavior in `scripts/run_made_mi_ntrain400k.sh`.

Resume launch:

```text
launched: 2026-06-08 12:08:06 Asia/Shanghai
session: p30_l18_gradclip_warmup_resume
log: logs/p30_l18_made_depth0_width64_gradclip_warmup_pilot.log
mode: append to existing log
```

Fresh CUDA check before resume passed on `cuda:0`; host GPU was again
`NVIDIA H100 PCIe`, with CUDA allocation, matmul, and project import checks
all passing.

Actual resume command:

```bash
tmux new-session -d -s p30_l18_gradclip_warmup_resume \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p30_l18_made_depth0_width64_gradclip_warmup_pilot \
   L_VALUES="18" TRAIN_SEEDS="2 3 5" N_TRAIN=1000000 DEVICE=cuda:0 \
   DEPTH=0 WIDTH=64 EPOCH=60 BATCH=512 LR=0.0003 WEIGHT_DECAY=0.0001 \
   GRAD_CLIP_NORM=1.0 WARMUP_STEPS=2000 DIVERGENCE_NLL_THRESHOLD=1000 \
   MAX_TRAIN_STEPS=0 LR_DECAY_FACTOR=0.5 LR_DECAY_PATIENCE=2 MIN_LR=0.00001 \
   EARLY_STOP_PATIENCE=8 EARLY_STOP_MIN_DELTA=0.01 scripts/run_made_mi_ntrain400k.sh \
   >> logs/p30_l18_made_depth0_width64_gradclip_warmup_pilot.log 2>&1'
```

Resume launch check:

- `tmux` session `p30_l18_gradclip_warmup_resume` exists.
- Log shows seed 2 AB checkpoint reuse.
- Log shows direct project command for seed 2 BA training.

Stop status:

```text
stopped_by_user: 2026-06-08 13:17:43 Asia/Shanghai
completed_mi_results_at_stop: seeds 1,2,3
partial_at_stop: seed 5 AB had started but wrote no training record or MI result
```

Completed p30 results at stop time:

| train_seed | MI | bootstrap std | objective failure |
|---:|---:|---:|---|
| 1 | 8.096962 | 0.111570 | no |
| 2 | 8.695381 | 0.101134 | no |
| 3 | 6.810791 | 0.106690 | no |

Aggregate over completed seeds `1,2,3`:

| n | mean MI | seed std | cv | min | max | mean bootstrap std |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 7.867711 | 0.962983 | 0.122397 | 6.810791 | 8.695381 | 0.106465 |

This completed-seed aggregate already fails the p30 pilot spread gate. Even
the most favorable possible seed 5 value cannot reduce the four-seed pilot
`cv` to `<= 0.08`, so p30 should not be extended to seeds `4,6,7,8` and
should not be promoted into the recommended fit.

Pilot decision after completion remains:

- If any completed seed has objective late-NLL failure, stop and report as
  diagnostic.
- If pilot `cv > 0.08`, stop and report as diagnostic.
- If `0.06 < cv <= 0.08`, continue only after deciding whether more diagnostic
  cost is justified.
- If no objective failures and `cv <= 0.06`, extend to seeds `4,6,7,8` under
  the same predeclared configuration.
