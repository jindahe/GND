# 2026-06-04 p20 L18 Seed 5 Rerun

## Run Metadata

- `run_id`: `p20_l18_seed5_rerun_ntrain400k`
- `target`: toric-code syndrome-only MI, `L=18`
- `n`: `648`
- `model`: `MADE`
- `p`: `0.05`
- `partition`: `x-mid`
- `n_train`: `400000`
- `n_val`: `2000`
- `n_test`: `2000`
- `train_seed_list`: `5`
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

The rerun was launched under a new run id so that the original
`p19_l18_ntrain400k_pilot` artifacts remain immutable:

```bash
env BASE_ROOT=net/mi_scaling/p20_l18_seed5_rerun_ntrain400k \
  L_VALUES=18 \
  TRAIN_SEEDS='5' \
  scripts/run_p16_l16_ntrain400k.sh
```

The default sandbox could not access CUDA, so the command was executed through
the repository GPU wrapper:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p20_l18_seed5_rerun_ntrain400k L_VALUES=18 TRAIN_SEEDS='5' scripts/run_p16_l16_ntrain400k.sh"
```

The run completed successfully at `2026-06-04T20:23:03+08:00`.

## Result

The rerun exactly reproduced the original `p19` seed-5 result.

| Quantity | Value |
|---|---:|
| `H_q(A,B)` | 180.838181 |
| `H_q(A)` | 95.477112 |
| `H_q(B)` | 95.740768 |
| `I_q(A;B)` | 10.379700 |
| bootstrap mean | 10.387472 |
| bootstrap std | 0.094562 |
| bootstrap ci95 low | 10.207259 |
| bootstrap ci95 high | 10.574651 |

Result JSON:

```text
net/mi_scaling/p20_l18_seed5_rerun_ntrain400k/results/made_tor_n648_d18_k2_seed0_er0.05_dep_tseed5_xmid.json
```

## Training Signals

| order | best_epoch | best_val_nll | test_nll | epochs_trained | last_train_nll | last_val_nll | max_train_nll | max_val_nll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `AB` | 37 | 185.334109 | 187.004469 | 67 | 176.189685 | 186.159840 | 220.177119 | 205.751414 |
| `BA` | 19 | 188.233578 | 190.030531 | 49 | 6071.461432 | 6070.282625 | 6673.665762 | 6670.250625 |

`AB` again followed the normal seed-5 trajectory. `BA` again diverged after the
early best epoch and ended with late-epoch NLL in the `6e3` range.

## Interpretation

- The seed-5 `BA` failure is deterministic under the current `n_train=400k`,
  `batch=512`, learning schedule, dataset split, and training seed.
- The repeated `I_q(A;B) = 10.379700` means the `p19` seed-5 high MI is not a
  transient artifact from a corrupted checkpoint or one-off execution issue.
- The `p19_l18_ntrain400k_pilot` aggregate should remain provisional and should
  not be promoted to a clean formal `L=18` result.
- This rerun does not justify replacing the `p19` seed-5 value inside the same
  run id, because the reproduced value is identical and the failed training
  signature is part of the diagnostic record.

Recommended next action:

1. A replacement seed was run as `p21_l18_replace_seed5_ntrain400k` using train
   seed 9. It has clean training but higher `MI = 10.953156`, so the replacement
   aggregate is still not clean.
2. Keep reporting fits with all `p19` seeds, without seed 5, and without seeds
   5/7 until the larger `L=18` seed sample resolves the endpoint.
3. Do not rerun `L=10/12` at `400k` before the `L=18` endpoint issue is
   resolved.

## Artifacts

Heavy artifacts are intentionally left under `net/` and are not part of the
lightweight GitHub record:

- Results:
  - `net/mi_scaling/p20_l18_seed5_rerun_ntrain400k/results/`
- Model checkpoints and training records:
  - `net/mi_scaling/p20_l18_seed5_rerun_ntrain400k/models/`
- Datasets:
  - `net/mi_scaling/p20_l18_seed5_rerun_ntrain400k/datasets/`
- Seed-level summary:
  - `net/mi_scaling/p20_l18_seed5_rerun_ntrain400k/L18_tseed5/`
