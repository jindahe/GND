# Next Steps For Syndrome-Only MI Scaling

This document records the next execution plan after the current toric-code
syndrome-only MI updates at `p = 0.05`.

The active fit target remains:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

Use `docs/MI_FIT_POINTS.csv` rows with
`include_in_recommended_fit=yes` as the current recommended fit inputs.

## Current Recommended Data

| L | n | run_id | seeds | n_train | MI | seed_std | mean_bootstrap_std | status |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 32 | `p8_made_plateau_long_468` | 1 |  | 1.074160 | 0.000000 | 0.021748 | historical single point |
| 6 | 72 | `p8_made_plateau_long_468` | 1 |  | 1.513966 | 0.000000 | 0.032413 | historical single point |
| 8 | 128 | `p8_made_plateau_long_468` | 1 |  | 1.866652 | 0.000000 | 0.044885 | historical single point |
| 10 | 200 | `p9_largeL_ntrain200k` | 8 | 200000 | 3.689559 | 0.265209 | 0.063953 | current multi-seed mean |
| 12 | 288 | `p9_largeL_ntrain200k` | 8 | 200000 | 5.384724 | 0.321396 | 0.073780 | current multi-seed mean |
| 14 | 392 | `p17_l14_ntrain400k` | 8 | 400000 | 6.074064 | 0.313907 | 0.088458 | current multi-seed mean |
| 16 | 512 | `p16_l16_ntrain400k` | 8 | 400000 | 8.133990 | 0.382952 | 0.094499 | current multi-seed mean |

Current strengths:

- The large-`L` range now includes `L=10,12,14,16`.
- `L=14` is no longer missing and has an 8-seed result.
- `L=16` has a cleaner 8-seed `n_train=400k` result than the previous
  `p12_l16_ntrain300k` reference.
- For `L=10/12/14/16`, train-seed spread is the main numerical uncertainty,
  not MI bootstrap noise.

Current weaknesses:

- `L=4/6/8` are historical single-point results and do not have train-seed
  uncertainty.
- `L=14` is statistically usable after extension to 8 seeds, but its mean sits
  below the simple interpolation between the current `L=12` and `L=16` points.
- The largest completed size is still `L=16`, so the asymptotic region is not
  yet strongly constrained.
- `L=10/12` use `n_train=200k`, while `L=14/16` use `n_train=400k`.

## Immediate Priority

### P0. Freeze The Current Record

Before launching more experiments, commit the current lightweight records so
future generated artifacts cannot obscure the present state.

Expected tracked changes:

```text
docs/MI_FIT_POINTS.csv
docs/MI_FIT_SUMMARY.md
docs/agent_outputs/scaling_runs/2026-06-03_p16_l16_ntrain400k_a100.md
docs/agent_outputs/scaling_runs/2026-06-03_p17_l14_ntrain400k_pilot.md
docs/NEXT_STEPS.md
```

Suggested commit message:

```text
Record L14 and updated L16 MI scaling results
```

Rationale:

- Heavy datasets, checkpoints, result JSON files, and plots under `net/` are
  intentionally outside the lightweight GitHub record.
- The CSV and Markdown files are the auditable source for current conclusions.

### P1. Generate A Formal Fit Analysis

Create a dedicated fit-analysis document:

```text
docs/MI_FIT_ANALYSIS.md
```

The analysis should not only report a single line fit. It should compare fit
windows and sensitivity to specific points.

Minimum fit windows:

| Window | Points | Purpose |
|---|---|---|
| all recommended | `L=4,6,8,10,12,14,16` | Full recorded curve |
| bridge and large-L | `L=8,10,12,14,16` | Reduces small-size influence |
| current multi-seed large-L | `L=10,12,14,16` | Uses the best current multi-seed range |
| without L14 | `L=10,12,16` | Measures sensitivity to low `L=14` |
| largest three | `L=12,14,16` | Checks local large-size curvature |

For each window, report:

```text
2 alpha
alpha
beta
RSS
per-point residuals
normalized residuals, when seed_std is available
leave-one-out sensitivity
```

Current unweighted OLS reference values:

| Window | n_points | 2 alpha | alpha | beta | RSS |
|---|---:|---:|---:|---:|---:|
| all recommended, `L=4..16` | 7 | 0.603889 | 0.301944 | -2.076441 | 1.843343 |
| `L>=8` | 5 | 0.745959 | 0.372980 | -3.921712 | 0.396009 |
| `L>=10` | 4 | 0.701132 | 0.350566 | -3.294127 | 0.315629 |
| `L>=12` | 3 | 0.687317 | 0.343658 | -3.091507 | 0.313085 |

Specific diagnostic for `L=14`:

```text
L12 recommended MI = 5.384724
L16 recommended MI = 8.133990
linear interpolation at L14 = 6.759357
observed L14 mean = 6.074064
delta = -0.685293
L14 seed_std = 0.313907
```

Interpretation requirement:

- Do not hide the low relative position of `L=14`.
- Do not drop `L=14` silently.
- Report both with-`L14` and without-`L14` windows.

## Experiment Priorities

### P2. Recheck `L=8`

`L=8` is currently the weakest bridge point because it is a historical
single-point result and has no train-seed spread.

Recommended pilot:

```text
run_id: p18_l8_ntrain400k
L: 8
n: 128
model: MADE
p: 0.05
partition: x-mid
n_train: 400000
n_val: 2000
n_test: 2000
train_seeds: 1,2,3
batch: 512
lr: 1e-3
lr_decay_factor: 0.5
lr_decay_patience: 5
min_lr: 1e-4
early_stop_patience: 30
mi_samples: 40000
bootstrap_samples: 200
```

Suggested command:

```bash
env BASE_ROOT=net/mi_scaling/p18_l8_ntrain400k \
  L_VALUES=8 \
  TRAIN_SEEDS='1 2 3' \
  scripts/run_p16_l16_ntrain400k.sh
```

Decision gates after seeds `1..3`:

- If `cv <= 0.06` and the mean is close to the historical `L=8` point, decide
  whether an 8-seed extension is necessary.
- If the mean differs substantially from historical `L=8` or train-seed spread
  remains high, extend to seeds `4..8`.
- If the result changes the `L=8 -> L=10` slope materially, update
  `docs/MI_FIT_POINTS.csv` and rerun the fit analysis.

### P3. Start An `L=18` Pilot

After the fit analysis and `L=8` pilot are understood, begin probing a larger
size. The current largest completed size is `L=16`, so `L=18` is the next
natural test of the large-`L` trend.

Recommended pilot:

```text
run_id: p19_l18_ntrain400k_pilot
L: 18
n: 648
model: MADE
p: 0.05
partition: x-mid
n_train: 400000
n_val: 2000
n_test: 2000
train_seeds: 1,2,3
batch: 512
lr: 1e-3
lr_decay_factor: 0.5
lr_decay_patience: 5
min_lr: 1e-4
early_stop_patience: 30
mi_samples: 40000
bootstrap_samples: 200
```

Suggested command:

```bash
env BASE_ROOT=net/mi_scaling/p19_l18_ntrain400k_pilot \
  L_VALUES=18 \
  TRAIN_SEEDS='1 2 3' \
  scripts/run_p16_l16_ntrain400k.sh
```

Operational rule:

- If `batch=512` fails due to memory pressure, stop and create a new run id
  such as `p19_l18_ntrain400k_b256`.
- Do not mix different batch sizes or learning configurations under the same
  run id.

Decision gates:

- If `cv <= 0.06`, extend to seeds `4..8`.
- If `cv > 0.06`, inspect training records and seed-level MI before expanding.
- Compare the 3-seed mean against extrapolations from `L=12,14,16`.

### P4. Do Not Immediately Rerun `L=10/12`

`L=10` and `L=12` use `n_train=200k`, while `L=14` and `L=16` use `400k`.
This is a real inconsistency, but it should not be the next experiment unless
the fit analysis shows a clear problem.

Trigger conditions for rerunning `L=10/12` at `400k`:

- Large systematic residuals at `L=10/12` compared with the `L=14/16`
  `400k` trend.
- Fit windows with and without `L=10/12` imply incompatible slopes.
- `L=8` recheck indicates the transition into large-`L` is controlled by
  training configuration rather than finite-size behavior.

If triggered, start with a same-seed pilot:

```text
L: 10,12
n_train: 400000
train_seeds: 1,2,3
```

Only expand to 8 seeds if the pilot changes the existing `p9_largeL_ntrain200k`
means or reduces uncertainty enough to affect the fit.

## Stability Criteria

Use `docs/STABILITY_CHECKLIST.md` for final decisions.

Working thresholds:

| Level | Suggested gate |
|---|---|
| Evaluation stable | `mean bootstrap std` clearly below train-seed std |
| Usable baseline | `cv <= 0.06`, at least 5 seeds, no obvious seed split |
| Formal result | `cv <= 0.04..0.05`, 8 seeds, no mean drift after new seeds |

Current status:

| L | status |
|---:|---|
| 10 | usable 8-seed baseline |
| 12 | usable 8-seed baseline |
| 14 | usable 8-seed baseline, but low relative to L12-L16 interpolation |
| 16 | usable 8-seed baseline and current best L16 point |

## Recommended Execution Order

1. Commit the current lightweight records.
2. Create `docs/MI_FIT_ANALYSIS.md`.
3. Run `L=8, n_train=400k, seeds=1..3`.
4. Decide whether to extend `L=8` to seeds `4..8`.
5. Start `L=18, n_train=400k, seeds=1..3`.
6. Decide whether to extend `L=18` to seeds `4..8`.
7. Rerun `L=10/12` at `400k` only if the fit analysis or `L=8/L18` results
   show that the mixed `n_train` protocol is limiting the conclusion.

## Reporting Rules

When updating the recommended fit:

- Add every completed aggregate to `docs/MI_FIT_POINTS.csv`.
- Mark only one row per `L` as `include_in_recommended_fit=yes`.
- Update `docs/MI_FIT_SUMMARY.md` after changing the CSV.
- Keep run-specific details in `docs/agent_outputs/scaling_runs/`.
- Do not include heavy `net/` artifacts in the lightweight GitHub record.
- Preserve older rows as auditable references rather than deleting them.
