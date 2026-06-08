# p28 L18 MADE depth0 width64 ntrain1000k pilot

Date: 2026-06-07

Run id:

```text
p28_l18_made_depth0_width64_ntrain1000k_rerun
```

Purpose:

Test whether the provisional `p19_l18_ntrain400k_pilot` endpoint instability
is data-size limited by rerunning the same `L=18`, same seed-block pilot under
the active MADE `depth=0,width=64` protocol with `n_train=1000000`.

Actual launch command:

```bash
tmux new-session -d -s p28_l18_1000k_pilot \
  'cd /home/jinboyu/GND/generative_decoder &&
   env BASE_ROOT=net/mi_scaling/p28_l18_made_depth0_width64_ntrain1000k_rerun L_VALUES="18" TRAIN_SEEDS="1 2 3 5" N_TRAIN=1000000 DEPTH=0 WIDTH=64 scripts/run_made_mi_ntrain400k.sh \
   > logs/p28_l18_made_depth0_width64_ntrain1000k_rerun.log 2>&1'
```

Launch notes:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- `scripts/run_codex_gpu.sh "scripts/check_a100_env.sh"` passed before launch;
  CUDA was available on `cuda:0`, with visible device `NVIDIA H100 PCIe`.
- Bare-background and `tmux + scripts/run_codex_gpu.sh` launch attempts did not
  start the training script cleanly. Diagnostic logs were preserved under
  `logs/`; the completed pilot used direct `tmux` execution of the project
  script.

Run window, Asia/Shanghai:

```text
2026-06-07 18:29:56 to 2026-06-07 21:34:52
```

Artifacts:

```text
net/mi_scaling/p28_l18_made_depth0_width64_ntrain1000k_rerun/
logs/p28_l18_made_depth0_width64_ntrain1000k_rerun.log
```

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
batch: 512
mi_samples: 40000
bootstrap_samples: 200
partition: x-mid
p: 0.05
error_model: dep
```

## Per-Seed MI

| train_seed | H(A) | H(B) | H(A,B) | MI | bootstrap mean | bootstrap std | ci95 low | ci95 high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.025230 | 99.134796 | 192.843597 | 6.316429 | 6.318981 | 0.100381 | 6.149987 | 6.503646 |
| 2 | 97.448799 | 97.052887 | 185.765320 | 8.736366 | 8.741433 | 0.095521 | 8.543916 | 8.916098 |
| 3 | 98.526321 | 98.934387 | 191.790009 | 5.670700 | 5.674284 | 0.099523 | 5.482819 | 5.857574 |
| 5 | 96.839455 | 96.069054 | 184.091751 | 8.816757 | 8.815710 | 0.099568 | 8.629107 | 8.998793 |

## Training Diagnostics

Failure rule:

```text
failed if late AB or BA train/validation NLL reaches 1e3 or larger
```

| train_seed | order | best epoch | test NLL | late-NLL max | failed |
|---:|---|---:|---:|---:|---|
| 1 | AB | 4 | 191.468902 | 6345.759500 | yes |
| 1 | BA | 4 | 192.596582 | 7814.642375 | yes |
| 2 | AB | 4 | 191.616805 | 4835.687125 | yes |
| 2 | BA | 4 | 191.744656 | 5160.663625 | yes |
| 3 | AB | 5 | 191.461527 | 4946.579250 | yes |
| 3 | BA | 5 | 190.501551 | 6126.046500 | yes |
| 5 | AB | 4 | 191.898215 | 4521.904625 | yes |
| 5 | BA | 4 | 191.844918 | 8476.618500 | yes |

All four pilot seeds fail the objective late-NLL rule in both partition orders.
The best epochs are very early (`4` or `5`), followed by severe late-epoch NLL
divergence.

## Aggregate

| L | n | seeds | n_seeds | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 18 | 648 | 1,2,3,5 | 4 | 7.385063171 | 1.628576822 | 0.220523073 | 5.670700073 | 8.816757202 | 0.098748063 | 1,2,3,5 |

This aggregate is diagnostic only. It is not eligible for promotion because it
has objective training failures in every included seed and `cv` is far above
the pilot gate.

## Comparison With p19 Same Seeds

| train_seed | p28 MI | p19 MI | p28 - p19 | p28 bootstrap std | p19 bootstrap std |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.316429 | 9.065102 | -2.748672 | 0.100381 | 0.102899 |
| 2 | 8.736366 | 8.760048 | -0.023682 | 0.095521 | 0.108807 |
| 3 | 5.670700 | 9.347771 | -3.677071 | 0.099523 | 0.102633 |
| 5 | 8.816757 | 10.379700 | -1.562943 | 0.099568 | 0.094562 |

Same-seed p19 aggregate over seeds `1,2,3,5`:

| source | seeds | mean MI | seed std | cv | min | max | mean bootstrap std | objective failures |
|---|---|---:|---:|---:|---:|---:|---:|---|
| p19 ntrain400k | 1,2,3,5 | 9.388154984 | 0.703248130 | 0.074908023 | 8.760047913 | 10.379699707 | 0.102225244 | seed 5 BA |
| p28 ntrain1000k | 1,2,3,5 | 7.385063171 | 1.628576822 | 0.220523073 | 5.670700073 | 8.816757202 | 0.098748063 | all seeds AB/BA |

Increasing `n_train` from `400000` to `1000000` did not stabilize the endpoint
under the active MADE `depth=0,width=64` protocol. It produced lower same-seed
MI for three of four seeds, much larger train-seed spread, and objective
late-NLL failures for all pilot seeds.

## Decision

Stop p28 as diagnostic. Do not launch the planned extension seeds
`4,6,7,8`.

Do not add p28 to `docs/MI_FIT_POINTS.csv`, and do not change the recommended
fit. The p19 `L=18` row remains the sole provisional recommended endpoint for
continuity, while endpoint-sensitive conclusions should continue to report the
through-`L=16` comparison window.

Interpretation:

- The p19 endpoint instability is not resolved by larger `n_train` under the
  current MADE `depth=0,width=64`, batch-512, learning-rate schedule.
- The failure signature points to training/optimization instability, not MI
  bootstrap noise.
- Future endpoint work should use a new predeclared diagnostic plan rather than
  extending p28. Reasonable directions are optimizer/schedule stabilization or
  the separately documented CNN/PixelCNN-style architecture track.
