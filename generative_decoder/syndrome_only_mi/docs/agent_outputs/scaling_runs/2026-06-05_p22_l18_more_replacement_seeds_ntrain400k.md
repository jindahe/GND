# p22 L18 More Replacement Seeds, ntrain400k

Date: 2026-06-05

Run id:

```text
p22_l18_more_replacement_seeds_ntrain400k
```

Command:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p22_l18_more_replacement_seeds_ntrain400k L_VALUES=18 TRAIN_SEEDS='10 11 12 13' scripts/run_p16_l16_ntrain400k.sh"
```

Pre-run checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- CUDA wrapper check passed: PyTorch saw 1 CUDA device and allocated on `cuda:0`.

Run window, Asia/Shanghai:

```text
2026-06-05 15:26:27 to 2026-06-05 16:59:18
```

Artifacts:

```text
net/mi_scaling/p22_l18_more_replacement_seeds_ntrain400k/
logs/p22_l18_more_replacement_seeds_ntrain400k.log
```

## Per-Seed Results

| train_seed | MI | bootstrap std | AB best epoch | BA best epoch | AB test NLL | BA test NLL | AB late NLL max | BA late NLL max | failed by late NLL rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10.672577 | 0.105356 | 39 | 33 | 188.015227 | 187.103773 | 186.982 | 186.034 | no |
| 11 | 9.114868 | 0.103321 | 38 | 22 | 187.022414 | 189.839781 | 186.181 | 4065.804 | yes, BA late divergence |
| 12 | 11.237228 | 0.104360 | 46 | 34 | 188.139719 | 187.368668 | 187.056 | 186.167 | no |
| 13 | 10.789398 | 0.104906 | 38 | 33 | 187.209613 | 187.364402 | 186.203 | 185.956 | no |

Failure rule used here:

```text
Mark a seed as failed if late AB or BA train/val NLL reaches 1e3 or larger.
```

Seed 11 is a training-failure diagnostic row because BA NLL diverged after
epoch 20. Early stopping restored the best checkpoint at epoch 22 and the MI
evaluation completed, but the late-NLL failure signature is present.

## Aggregates

| Aggregate | Seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original `p19` all seeds | 1,2,3,4,5,6,7,8 | 8 | 9.573410988 | 0.574411020 | 0.060000664 | 8.760047913 | 10.379699707 | 0.104420142 |
| `p19` failed seed 5 excluded | 1,2,3,4,6,7,8 | 7 | 9.458226885 | 0.510989725 | 0.054025953 | 8.760047913 | 10.334045410 | 0.105828442 |
| `p19` failed seed 5 and high seed 7 excluded | 1,2,3,4,6,8 | 6 | 9.312257131 | 0.366541101 | 0.039361145 | 8.760047913 | 9.807640076 | 0.106169435 |
| seed 5 replaced by seed 9 | 1,2,3,4,9,6,7,8 | 8 | 9.645092964 | 0.709338037 | 0.073543930 | 8.760047913 | 10.953155518 | 0.105632929 |
| seed 5 replaced by first clean `p22` seed 10 | 1,2,3,4,10,6,7,8 | 8 | 9.610020638 | 0.638858158 | 0.066478334 | 8.760047913 | 10.672576904 | 0.105769326 |
| all clean p19/p21/p22 seeds | 1,2,3,4,6,7,8,9,10,12,13 | 11 | 9.987267928 | 0.844645211 | 0.084572199 | 8.760047913 | 11.237228394 | 0.105425932 |
| `p22` all seeds | 10,11,12,13 | 4 | 10.453517914 | 0.925020301 | 0.088488900 | 9.114868164 | 11.237228394 | 0.104485774 |
| `p22` clean only | 10,12,13 | 3 | 10.899734497 | 0.298057714 | 0.027345411 | 10.672576904 | 11.237228394 | 0.104873940 |

For the replacement-by-p22 aggregate, the predeclared clean replacement rule
used here is:

```text
Choose the lowest train_seed among p22 seeds that pass the objective late-NLL
training-failure rule.
```

This chooses seed 10. It does not choose by observed MI.

## Interpretation

`p22` does not support treating seed 5 as the only source of the L=18 endpoint
problem.

The clean p22 seeds are all high-MI:

```text
seed 10: 10.672577
seed 12: 11.237228
seed 13: 10.789398
```

Seed 9 from `p21` was also clean and high at 10.953156. Therefore clean
high-MI seeds are common in the additional-seed sample.

The all-clean p19/p21/p22 aggregate has:

```text
mean MI = 9.987267928
seed std = 0.844645211
cv = 0.084572199
```

This is less stable than the original provisional p19 endpoint and well above
the `cv <= 0.06` usable-baseline gate. The replacement aggregate using the
first clean p22 seed also remains above the gate:

```text
cv = 0.066478334
```

## Decision

Keep `L=18` provisional. Do not promote a replacement aggregate into
`docs/MI_FIT_POINTS.csv`.

Current evidence favors the heavy-tail interpretation:

- deterministic training failures exist at L=18 (`p19` seed 5, `p22` seed 11)
- clean high-MI seeds are not rare (`p21` seed 9, `p22` seeds 10/12/13)
- replacing failed seed 5 by a clean predefined p22 seed does not stabilize the
  endpoint to `cv <= 0.06`
- using all clean additional seeds increases the spread rather than resolving it

Recommended next step:

Write the formal seed policy before any recommended-fit update. The policy
should exclude only objective training failures, keep clean high-MI seeds, and
report all-seed, failed-excluded, and robust sensitivity aggregates without
choosing replacements by MI.
