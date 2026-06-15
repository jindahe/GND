# 2026-06-04 p19 L18 Seed-Split Diagnostic

## Run Context

- `run_id`: `p19_l18_ntrain400k_pilot`
- `target`: toric-code syndrome-only MI, `L=18`
- `n`: `648`
- `model`: `MADE`
- `p`: `0.05`
- `partition`: `x-mid`
- `n_train`: `400000`
- `train_seed_list`: `1, 2, 3, 4, 5, 6, 7, 8`
- `batch`: `512`
- `mi_samples`: `40000`
- `bootstrap_samples`: `200`

This diagnostic executes `P3b` from `docs/NEXT_STEPS.md`: compare the
large-`L` fits under the completed `L=18` aggregate, the aggregate without seed
5, and the aggregate without seeds 5 and 7; inspect seed 5 `BA` training; and
decide the next action without changing the `p19_l18_ntrain400k_pilot` run
protocol.

## Seed-Level MI

| train_seed | MI | bootstrap_std | note |
|---:|---:|---:|---|
| 1 | 9.065102 | 0.102899 |  |
| 2 | 8.760048 | 0.108807 |  |
| 3 | 9.347771 | 0.102633 |  |
| 4 | 9.553818 | 0.109733 |  |
| 5 | 10.379700 | 0.094562 | high MI; abnormal `BA` training record |
| 6 | 9.807640 | 0.105091 |  |
| 7 | 10.334045 | 0.103782 | high MI; no matching training-failure signature |
| 8 | 9.339165 | 0.107854 |  |

Aggregate sensitivity:

| Subset | seeds | mean MI | seed_std | cv | min | max |
|---|---:|---:|---:|---:|---:|---:|
| all seeds | 8 | 9.573411 | 0.574411 | 0.060001 | 8.760048 | 10.379700 |
| without seed 5 | 7 | 9.458227 | 0.510990 | 0.054026 | 8.760048 | 10.334045 |
| without seeds 5 and 7 | 6 | 9.312257 | 0.366541 | 0.039361 | 8.760048 | 9.807640 |

Seed 5 moves the mean by `+0.115184` relative to the without-seed-5 aggregate.
Seeds 5 and 7 together move the mean by `+0.261154` relative to the aggregate
without both high-MI seeds.

## Fit Sensitivity

Fits are unweighted OLS fits of:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

Only the `L=18` aggregate changes across the three scenarios below. All lower
`L` points are the current recommended rows from `docs/MI_FIT_POINTS.csv`.

| Fit window | `L=18` subset | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| `L=8,10,12,14,16,18` | all seeds | 6 | 0.695525 | 0.347763 | -3.125775 | 0.425885 |
| `L=8,10,12,14,16,18` | without seed 5 | 6 | 0.687298 | 0.343649 | -3.038016 | 0.390799 |
| `L=8,10,12,14,16,18` | without seeds 5 and 7 | 6 | 0.676872 | 0.338436 | -2.926801 | 0.364488 |
| `L=10,12,14,16,18` | all seeds | 5 | 0.725849 | 0.362924 | -3.590730 | 0.340066 |
| `L=10,12,14,16,18` | without seed 5 | 5 | 0.714330 | 0.357165 | -3.452509 | 0.322597 |
| `L=10,12,14,16,18` | without seeds 5 and 7 | 5 | 0.699733 | 0.349867 | -3.277345 | 0.315707 |
| `L=10,12,16,18` | all seeds | 4 | 0.725849 | 0.362924 | -3.466458 | 0.031198 |
| `L=10,12,16,18` | without seed 5 | 4 | 0.714330 | 0.357165 | -3.333996 | 0.041694 |
| `L=10,12,16,18` | without seeds 5 and 7 | 4 | 0.699733 | 0.349867 | -3.166131 | 0.068337 |
| `L=12,14,16,18` | all seeds | 4 | 0.731299 | 0.365650 | -3.677944 | 0.338878 |
| `L=12,14,16,18` | without seed 5 | 4 | 0.714022 | 0.357011 | -3.447576 | 0.322593 |
| `L=12,14,16,18` | without seeds 5 and 7 | 4 | 0.692126 | 0.346063 | -3.155636 | 0.313393 |
| `L=14,16,18` | all seeds | 3 | 0.874837 | 0.437418 | -6.070234 | 0.064171 |
| `L=14,16,18` | without seed 5 | 3 | 0.846041 | 0.423020 | -5.647892 | 0.090207 |
| `L=14,16,18` | without seeds 5 and 7 | 3 | 0.809548 | 0.404774 | -5.112670 | 0.129554 |

Main effect on the preferred large-`L` diagnostic window:

| Scenario | `L=18` mean | `L=10,12,14,16,18` 2 alpha | delta vs all seeds |
|---|---:|---:|---:|
| all seeds | 9.573411 | 0.725849 | 0.000000 |
| without seed 5 | 9.458227 | 0.714330 | -0.011518 |
| without seeds 5 and 7 | 9.312257 | 0.699733 | -0.026115 |

Interpretation:

- The all-seed `L=18` point increases the preferred `L>=10` slope by about
  `0.0115` in `2 alpha` relative to removing seed 5.
- Removing both high-MI seeds 5 and 7 lowers the preferred `L>=10` slope by
  about `0.0261` in `2 alpha`.
- The slope remains in the same broad range as the previous `L<=16` large-`L`
  diagnostics, but `L=18` is not yet clean enough to use as a formal endpoint.
- `L=14` remains the dominant low residual; removing `L=18` high seeds does
  not remove the previously observed low-`L14` feature.

## Seed 5 Training Diagnostic

`AB` seed 5 is normal relative to the other `AB` runs. `BA` seed 5 is the
outlier.

| train_seed | order | best_epoch | best_val_nll | test_nll | epochs_trained | last_train_nll | last_val_nll | max_train_nll | max_val_nll |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `AB` | 37 | 185.334109 | 187.004469 | 67 | 176.189685 | 186.159840 | 220.177119 | 205.751414 |
| 5 | `BA` | 19 | 188.233578 | 190.030531 | 49 | 6071.461432 | 6070.282625 | 6673.665762 | 6670.250625 |
| 7 | `AB` | 33 | 185.643570 | 187.384727 | 63 | 175.978092 | 186.300734 | 220.047411 | 205.549445 |
| 7 | `BA` | 33 | 185.414367 | 187.269574 | 63 | 176.022823 | 186.084801 | 220.074702 | 205.548023 |

Seed 5 `BA` late-epoch history:

| History | Last 10 values |
|---|---|
| train NLL | 6400.289656, 6400.266333, 6400.144847, 6400.122353, 6400.105116, 6400.076551, 6400.048697, 6400.021071, 6218.667702, 6071.461432 |
| val NLL | 6393.378125, 6393.317500, 6393.222875, 6393.217375, 6393.180750, 6393.205500, 6393.203250, 6393.193250, 6078.304875, 6070.282625 |

For comparison, seed 7 `BA` late-epoch validation NLL remains near `186.08`,
and all other non-seed-5 `BA` last validation NLL values are near `186.08` to
`186.69`. Seed 5 `BA` is therefore not just a high-MI statistical draw; it has
a distinct training-failure signature.

## Decision

Do not promote `p19_l18_ntrain400k_pilot` to a clean formal `L=18` result.
Keep the all-seed aggregate as the provisional recommended row only because it
is the only completed largest-size point, and report seed-5/seed-7 sensitivity
whenever `L=18` is used in a fit.

Follow-up result:

- Train seed 5 was rerun under the new run id
  `p20_l18_seed5_rerun_ntrain400k`.
- The rerun exactly reproduced the original `p19` seed-5 `BA` failure and
  `MI = 10.379700`.
- The diagnostic row is recorded in `docs/MI_FIT_POINTS.csv` with
  `include_in_recommended_fit=no`.
- Do not overwrite or mix the new result into `p19_l18_ntrain400k_pilot`.
- The next action is to add more `L=18` train seeds under a new run id, for
  example after the completed `p21_l18_replace_seed5_ntrain400k` seed-9
  replacement diagnostic, rather than treating the current seed split as
  resolved.

Completed rerun command:

```bash
env BASE_ROOT=net/mi_scaling/p20_l18_seed5_rerun_ntrain400k \
  L_VALUES=18 \
  TRAIN_SEEDS='5' \
  scripts/run_p16_l16_ntrain400k.sh
```

This command intentionally uses a new `BASE_ROOT` so the original `p19` pilot
remains immutable and auditable.
