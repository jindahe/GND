# p25 L18 Architecture Seed Block, depth1 width8, ntrain400k

Date: 2026-06-06

Run id:

```text
p25_l18_arch_depth1_width8_seeds1to8_ntrain400k
```

Command:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p25_l18_arch_depth1_width8_seeds1to8_ntrain400k L_VALUES=18 TRAIN_SEEDS='1 2 3 4 5 6 7 8' DEPTH=1 WIDTH=8 scripts/run_p16_l16_ntrain400k.sh" 2>&1 | tee logs/p25_l18_arch_depth1_width8_seeds1to8_ntrain400k.log
```

Pre-run checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- CUDA was available on `cuda:0`; PyTorch saw 1 CUDA device and allocated on
  `cuda:0`.

Run window, Asia/Shanghai:

```text
2026-06-06 16:05:24 to 2026-06-06 19:32:49
```

Artifacts:

```text
net/mi_scaling/p25_l18_arch_depth1_width8_seeds1to8_ntrain400k/
logs/p25_l18_arch_depth1_width8_seeds1to8_ntrain400k.log
```

Architecture:

```text
model: MADE
depth: 1
requested_width: 8
effective_width: 8
activation: tanh
residual: false
parameter_count: 33396262
```

## Per-Seed Results

| train_seed | MI | bootstrap std | AB best epoch | BA best epoch | AB test NLL | BA test NLL | AB late NLL max | BA late NLL max | failed by late NLL rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.378075 | 0.103806 | 20 | 21 | 184.691031 | 185.031055 | 187.599328 | 187.807961 | no |
| 2 | 9.724525 | 0.111176 | 18 | 20 | 184.448164 | 184.833750 | 186.954496 | 187.556430 | no |
| 3 | 10.777229 | 0.101371 | 22 | 18 | 185.020949 | 184.224219 | 187.876457 | 186.896418 | no |
| 4 | 9.766930 | 0.107446 | 22 | 18 | 185.072090 | 184.528105 | 188.102988 | 186.862766 | no |
| 5 | 9.342827 | 0.106336 | 20 | 21 | 184.769242 | 184.709957 | 187.526902 | 187.815852 | no |
| 6 | 7.852448 | 0.102802 | 18 | 19 | 184.549656 | 184.422453 | 187.175828 | 187.199582 | no |
| 7 | 9.243690 | 0.106546 | 22 | 20 | 185.153754 | 184.689980 | 188.387477 | 187.653797 | no |
| 8 | 9.553291 | 0.104418 | 19 | 20 | 184.513301 | 184.665703 | 187.108102 | 187.407777 | no |

Failure rule used here:

```text
Mark a seed as failed if late AB or BA train/val NLL reaches 1e3 or larger.
```

The late NLL max is computed from the saved JSON training records under
`models/records/`, using train/val histories from the best epoch through early
stop. No seed in this run triggered the objective training-failure rule.

## Aggregates

| Aggregate | Seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `p25` all seeds | 1,2,3,4,5,6,7,8 | 8 | 9.329876900 | 0.893454491 | 0.095762731 | 7.852447510 | 10.777229309 | 0.105487480 | none |

For comparison:

| Aggregate | mean MI | seed std | cv |
|---|---:|---:|---:|
| `p19` depth0 width64 same seed block | 9.573410988 | 0.574411020 | 0.060001 |
| `p24` depth1 width8 seeds 10/11/12/13 | 9.484302521 | 0.494612000 | 0.052151 |
| `p25` depth1 width8 seeds 1..8 | 9.329876900 | 0.893454491 | 0.095762731 |

## Fit Sensitivity

If `p25` is substituted as the `L=18` endpoint while leaving the other current
recommended points unchanged, selected unweighted OLS windows are:

| Window | Points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|
| current multi-seed large-L | L=10,12,14,16,18 | 0.701495 | 0.350748 | -3.298489 | 0.315635 |
| without L14 | L=10,12,16,18 | 0.701495 | 0.350748 | -3.186394 | 0.064330 |
| L>=12 | L=12,14,16,18 | 0.694769 | 0.347385 | -3.190876 | 0.313825 |
| largest three | L=14,16,18 | 0.813953 | 0.406977 | -5.177275 | 0.124427 |

These are diagnostic windows only. `p25` is not promoted to the recommended fit.

## Interpretation

The `depth=1,width=8` architecture remained clean under the objective late-NLL
failure rule on train seeds 1..8. This supports the earlier `p24` observation
that the smaller hidden width avoids the severe late-NLL divergences seen in
the `depth=0,width=64` `L=18` diagnostics.

However, `p25` did not stabilize the `L=18` endpoint. The 8-seed aggregate has
`cv = 0.095762731`, far above the `cv <= 0.06` usable-baseline gate and above
the current `p19` provisional `cv = 0.060001`. The seed range is wide:
`7.852447510` to `10.777229309`, while bootstrap uncertainty remains near
`0.105`, much smaller than the train-seed spread.

Decision:

- Keep the existing `p19_l18_ntrain400k_pilot` row as the provisional
  recommended `L=18` point for continuity.
- Add `p25` as a diagnostic architecture row with
  `include_in_recommended_fit=no`.
- Treat `L=18` as architecture-sensitive and train-seed-sensitive.
- Do not promote the `depth=1,width=8` architecture endpoint without a new
  endpoint policy or additional diagnostics explaining the large seed spread.

Recommended next step:

Interpret the `p25` seed spread before spending GPU time on `L=10/12` reruns.
The most useful immediate follow-up is a diagnostic comparison of p19 and p25
seed-level entropies, best epochs, and late-NLL histories for the same
train-seed block, especially seeds 3 and 6 where p25 reaches the extrema.
