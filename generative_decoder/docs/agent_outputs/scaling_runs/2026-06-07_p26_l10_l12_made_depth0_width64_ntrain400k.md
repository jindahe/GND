# p26 L10/L12 MADE depth0 width64 ntrain400k

Date: 2026-06-07

Run id:

```text
p26_l10_l12_made_depth0_width64_ntrain400k
```

Command:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p26_l10_l12_made_depth0_width64_ntrain400k L_VALUES='10 12' TRAIN_SEEDS='1 2 3 4 5 6 7 8' DEPTH=0 WIDTH=64 scripts/run_made_mi_ntrain400k.sh"
```

Pre-run checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- CUDA was available through `scripts/run_codex_gpu.sh`; the device was
  `NVIDIA H100 PCIe`.
- The project GPU wrapper passed `scripts/check_a100_env.sh`.

Run window, Asia/Shanghai:

```text
2026-06-06 22:21:26 to 2026-06-07 04:35:54
```

Artifacts:

```text
net/mi_scaling/p26_l10_l12_made_depth0_width64_ntrain400k/
logs/p26_l10_l12_made_depth0_width64_ntrain400k.log
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
batch: 512
mi_samples: 40000
bootstrap_samples: 200
partition: x-mid
p: 0.05
error_model: dep
```

## L10 Per-Seed Results

| train_seed | MI | bootstrap std | AB best epoch | BA best epoch | AB test NLL | BA test NLL | AB late NLL max | BA late NLL max | failed by late NLL rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.430105 | 0.061317 | 51 | 51 | 55.393924 | 55.353111 | 55.203134 | 55.233948 | no |
| 2 | 3.270197 | 0.063158 | 49 | 50 | 55.396676 | 55.306143 | 55.220620 | 55.240510 | no |
| 3 | 3.640820 | 0.062026 | 55 | 49 | 55.381228 | 55.350532 | 55.148973 | 55.149331 | no |
| 4 | 3.421734 | 0.064039 | 52 | 50 | 55.436052 | 55.221045 | 55.169682 | 55.199152 | no |
| 5 | 3.805380 | 0.064141 | 54 | 49 | 55.432790 | 55.351871 | 55.276397 | 55.239847 | no |
| 6 | 3.816570 | 0.064849 | 61 | 47 | 55.319843 | 55.359896 | 55.224842 | 55.227483 | no |
| 7 | 3.561953 | 0.064458 | 46 | 46 | 55.338140 | 55.322105 | 55.234145 | 55.150477 | no |
| 8 | 3.509911 | 0.063211 | 54 | 54 | 55.342832 | 55.427726 | 55.159132 | 55.209089 | no |

## L12 Per-Seed Results

| train_seed | MI | bootstrap std | AB best epoch | BA best epoch | AB test NLL | BA test NLL | AB late NLL max | BA late NLL max | failed by late NLL rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.698509 | 0.074651 | 46 | 44 | 81.000379 | 81.156607 | 81.107295 | 81.101340 | no |
| 2 | 4.864792 | 0.074840 | 50 | 50 | 81.129525 | 81.082666 | 81.106652 | 81.117801 | no |
| 3 | 4.661427 | 0.071895 | 42 | 51 | 81.018156 | 81.043359 | 81.101707 | 81.128330 | no |
| 4 | 4.807858 | 0.070503 | 50 | 47 | 81.089906 | 81.021262 | 81.168260 | 81.162803 | no |
| 5 | 4.933792 | 0.073398 | 51 | 49 | 81.059322 | 81.133473 | 81.159867 | 81.050889 | no |
| 6 | 5.297520 | 0.071714 | 50 | 45 | 81.091668 | 81.049123 | 81.282266 | 81.142623 | no |
| 7 | 5.071011 | 0.079760 | 52 | 46 | 81.162039 | 81.122705 | 81.206236 | 81.096064 | no |
| 8 | 5.039707 | 0.072653 | 46 | 45 | 81.056115 | 81.120604 | 81.069625 | 81.105447 | no |

Failure rule used here:

```text
Mark a seed as failed if late AB or BA train/val NLL reaches 1e3 or larger.
```

The late NLL max is computed from the saved JSON training records under
`models/records/`, using train/val histories from the best epoch through early
stop. No seed in this run triggered the objective training-failure rule.

## Aggregates

| L | n | seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 200 | 1,2,3,4,5,6,7,8 | 8 | 3.557083607 | 0.190859754 | 0.053656246 | 3.270196915 | 3.816570282 | 0.063399924 | none |
| 12 | 288 | 1,2,3,4,5,6,7,8 | 8 | 4.921826839 | 0.210903132 | 0.042850579 | 4.661426544 | 5.297519684 | 0.073676726 | none |

For comparison with the previous recommended `p9_largeL_ntrain200k` bridge
rows:

| L | p9 mean MI | p26 mean MI | delta |
|---:|---:|---:|---:|
| 10 | 3.689559221 | 3.557083607 | -0.132475 |
| 12 | 5.384723663 | 4.921826839 | -0.462897 |

## Fit Sensitivity

After replacing the previous `p9` rows for `L=10` and `L=12`, selected
unweighted OLS windows are:

| Window | Points | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| all recommended | `L=4,6,8,10,12,14,16,18` | 8 | 0.620596 | 0.310298 | -2.140417 | 1.619790 |
| bridge and large-L | `L=8,10,12,14,16,18` | 6 | 0.707816 | 0.353908 | -3.384778 | 0.473119 |
| current multi-seed large-L | `L=10,12,14,16,18` | 5 | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| without L14 | `L=10,12,16,18` | 4 | 0.762241 | 0.381120 | -4.124795 | 0.018041 |
| `L>=12` | `L=12,14,16,18` | 4 | 0.800734 | 0.400367 | -4.835186 | 0.137388 |
| largest three | `L=14,16,18` | 3 | 0.874837 | 0.437418 | -6.070234 | 0.064171 |

The rerun shifts both bridge points downward relative to `p9`, especially
`L=12`. This reduces the apparent low position of `L=14`: with the p26 `L=12`
mean and current `L=16` mean, linear interpolation at `L=14` is `6.527909`,
so the observed `L=14` mean is `-0.453845`, or `-1.445795` L14 seed-standard
deviations, below the interpolation. Before p26, the same diagnostic was
`-2.183111` seed-standard deviations.

## Interpretation

The p26 rerun is clean under the formal late-NLL rule and is now the internally
consistent `n_train=400000`, MADE `depth=0,width=64` bridge baseline for
`L=10` and `L=12`. The new `L=10` mean is close to the previous p9 value, while
the new `L=12` mean is materially lower.

Decision:

- Promote the p26 `L=10` and `L=12` rows to the recommended fit inputs.
- Keep the previous p9 `L=10` and `L=12` rows in `docs/MI_FIT_POINTS.csv` as
  historical diagnostics with `include_in_recommended_fit=no`.
- Treat the updated `L=14` residual as still notable but less severe than under
  the mixed-`n_train` p9 bridge.
- Keep `L=18` provisional; p26 does not resolve the endpoint architecture and
  train-seed sensitivity documented for `L=18`.
