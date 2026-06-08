# 2026-06-04 p21 L18 Replace Seed 5 With Seed 9

## Run Metadata

- `run_id`: `p21_l18_replace_seed5_ntrain400k`
- `target`: toric-code syndrome-only MI, `L=18`
- `replacement policy`: exclude failed `p19` train seed 5 from the replacement
  diagnostic and use train seed 9 as the replacement seed
- `n`: `648`
- `model`: `MADE`
- `p`: `0.05`
- `partition`: `x-mid`
- `n_train`: `400000`
- `n_val`: `2000`
- `n_test`: `2000`
- `train_seed_list`: `9`
- `batch`: `512`
- `lr`: `1e-3`
- `lr_decay_factor`: `0.5`
- `lr_decay_patience`: `5`
- `min_lr`: `1e-4`
- `early_stop_patience`: `30`
- `mi_samples`: `40000`
- `bootstrap_samples`: `200`
- `code_seed`: `0`
- `error_seed`: `51697`
- `split_seed`: `0`

## Command

The replacement seed was launched under a new run id so that the original
`p19_l18_ntrain400k_pilot` and `p20_l18_seed5_rerun_ntrain400k` artifacts
remain immutable:

```bash
env BASE_ROOT=net/mi_scaling/p21_l18_replace_seed5_ntrain400k \
  L_VALUES=18 \
  TRAIN_SEEDS='9' \
  scripts/run_p16_l16_ntrain400k.sh
```

The command was executed through the repository GPU wrapper because the default
sandbox cannot access CUDA:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p21_l18_replace_seed5_ntrain400k L_VALUES=18 TRAIN_SEEDS='9' scripts/run_p16_l16_ntrain400k.sh"
```

The run completed successfully at `2026-06-04T21:07:02+08:00`.

## Seed 9 Result

Seed 9 did not reproduce the seed-5 `BA` training failure. Both `AB` and `BA`
records stayed in the normal NLL range, but the final MI is higher than the
failed seed-5 MI.

| Quantity | Value |
|---|---:|
| `H_q(A,B)` | 179.149780 |
| `H_q(A)` | 94.831764 |
| `H_q(B)` | 95.271172 |
| `I_q(A;B)` | 10.953156 |
| bootstrap mean | 10.962839 |
| bootstrap std | 0.104264 |
| bootstrap ci95 low | 10.789856 |
| bootstrap ci95 high | 11.171152 |

Result JSON:

```text
net/mi_scaling/p21_l18_replace_seed5_ntrain400k/results/made_tor_n648_d18_k2_seed0_er0.05_dep_tseed9_xmid.json
```

## Training Signals

| order | best_epoch | best_val_nll | test_nll | epochs_trained | last_train_nll | last_val_nll | max_train_nll | max_val_nll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `AB` | 42 | 185.803039 | 187.602973 | 72 | 175.496262 | 186.705535 | 220.135676 | 206.151188 |
| `BA` | 37 | 185.639133 | 187.662090 | 67 | 175.416100 | 186.346223 | 220.237247 | 205.611035 |

Unlike `p19/p20` seed 5, the replacement seed's `BA` run did not diverge to
NLL values in the `6e3` range.

## Replacement Aggregate

The replacement diagnostic uses:

```text
p19 seeds: 1,2,3,4,6,7,8
p21 seed: 9
```

This excludes the failed `p19` seed 5 but preserves the original `p19` records
on disk and in the CSV as auditable artifacts.

| Subset | seeds | mean MI | seed_std | cv | min | max | mean bootstrap std |
|---|---:|---:|---:|---:|---:|---:|---:|
| original `p19` all seeds | 8 | 9.573411 | 0.574411 | 0.060001 | 8.760048 | 10.379700 | 0.104420 |
| original `p19` without seed 5 | 7 | 9.458227 | 0.510990 | 0.054026 | 8.760048 | 10.334045 | 0.105828 |
| replacement seed 9 for seed 5 | 8 | 9.645093 | 0.709338 | 0.073544 | 8.760048 | 10.953156 | 0.105633 |
| original `p19` without seeds 5 and 7 | 6 | 9.312257 | 0.366541 | 0.039361 | 8.760048 | 9.807640 | 0.106169 |

The replacement aggregate is worse by the usable-baseline cv gate than the
original `p19` aggregate because seed 9 is a clean-training but higher-MI point.

## Fit Sensitivity

Using the replacement aggregate as `L=18` changes the main large-`L` diagnostic
window as follows:

| Fit window | n_points | 2 alpha | alpha | beta | RSS |
|---|---:|---:|---:|---:|---:|
| all recommended plus replacement `L18` | 8 | 0.625549 | 0.312775 | -2.111524 | 1.383813 |
| `L=8,10,12,14,16,18` | 6 | 0.700646 | 0.350323 | -3.180390 | 0.454099 |
| `L=10,12,14,16,18` | 5 | 0.733017 | 0.366508 | -3.676748 | 0.356295 |
| `L=10,12,16,18` | 4 | 0.733017 | 0.366508 | -3.548892 | 0.029355 |
| `L=12,14,16,18` | 4 | 0.742052 | 0.371026 | -3.821308 | 0.353030 |
| `L=14,16,18` | 3 | 0.892757 | 0.446379 | -6.333068 | 0.050201 |

Compared with the original all-seed `p19` `L=10,12,14,16,18` fit
(`2 alpha = 0.725849`), the replacement seed raises the same-window slope to
`2 alpha = 0.733017`.

## Decision

- Do not physically delete or overwrite the failed `p19` seed-5 artifacts.
  They remain necessary for auditability.
- Do not treat the replacement aggregate as a clean formal `L=18` result. Seed
  9 has clean training, but it increases the replacement aggregate cv to
  `0.073544`, above the usable-baseline gate.
- Record seed 9 and the replacement aggregate as diagnostic rows with
  `include_in_recommended_fit=no`.
- The next robust action is to add more `L=18` replacement/additional seeds
  under a new run id, then decide whether to use a robust estimator, exclude
  failed-training seeds by a predefined rule, or report multiple endpoint
  scenarios.

## Artifacts

Heavy artifacts are intentionally left under `net/` and are not part of the
lightweight GitHub record:

- Results:
  - `net/mi_scaling/p21_l18_replace_seed5_ntrain400k/results/`
- Model checkpoints and training records:
  - `net/mi_scaling/p21_l18_replace_seed5_ntrain400k/models/`
- Datasets:
  - `net/mi_scaling/p21_l18_replace_seed5_ntrain400k/datasets/`
- Seed-level summary:
  - `net/mi_scaling/p21_l18_replace_seed5_ntrain400k/L18_tseed9/`
