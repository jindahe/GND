# p23 L8 Architecture Ablation, ntrain400k

Date: 2026-06-05

Run id:

```text
p23_l8_arch_ablation_ntrain400k
```

Command:

```bash
bash scripts/run_p23_l8_arch_ablation_ntrain400k.sh
```

Run window, Asia/Shanghai:

```text
2026-06-05 20:42:07 to 2026-06-05 21:55:28
```

Artifacts:

```text
net/mi_scaling/p23_l8_arch_ablation_ntrain400k/
logs/p23_l8_arch_ablation_ntrain400k.log
```

The run shared one `L=8` dataset directory across variants, but used separate
model/result directories per architecture label.

## Variants

| Label | depth | width | parameter_count | seeds |
|---|---:|---:|---:|---|
| baseline | 0 | 64 | 2040318 | 1,2,3 |
| narrow-deep-1 | 1 | 8 | 1272222 | 1,2,3 |
| narrow-deep-2 | 1 | 16 | 4576446 | 1,2,3 |
| narrow-deep-3 | 2 | 8 | 2289294 | 1,2,3 |

## Per-Seed Results

| Label | seed | MI | bootstrap std | AB test NLL | BA test NLL | failed late-NLL rule |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1 | 2.808210 | 0.052290 | 35.154046 | 35.118037 | no |
| baseline | 2 | 2.494656 | 0.051980 | 35.090187 | 35.106752 | no |
| baseline | 3 | 2.570799 | 0.050730 | 35.075117 | 35.078401 | no |
| narrow-deep-1 | 1 | 2.704603 | 0.048586 | 34.657188 | 34.651443 | no |
| narrow-deep-1 | 2 | 2.907337 | 0.049055 | 34.668055 | 34.679126 | no |
| narrow-deep-1 | 3 | 2.765352 | 0.050414 | 34.663024 | 34.589748 | no |
| narrow-deep-2 | 1 | 2.970188 | 0.049520 | 34.742111 | 34.659366 | no |
| narrow-deep-2 | 2 | 2.918804 | 0.050921 | 34.735295 | 34.721356 | no |
| narrow-deep-2 | 3 | 2.678415 | 0.050819 | 34.658223 | 34.672132 | no |
| narrow-deep-3 | 1 | 2.921965 | 0.045788 | 34.735854 | 34.786856 | no |
| narrow-deep-3 | 2 | 2.550920 | 0.049126 | 34.720329 | 34.729289 | no |
| narrow-deep-3 | 3 | 2.516272 | 0.049548 | 34.762835 | 34.769362 | no |

## Aggregate Results

| Label | depth | width | n | mean MI | seed std | cv | min | max | mean bootstrap std | mean AB test NLL | mean BA test NLL | failed seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 64 | 3 | 2.624554952 | 0.163543372 | 0.062312801 | 2.494655609 | 2.808210373 | 0.051666598 | 35.106450 | 35.101063 | 0 |
| narrow-deep-1 | 1 | 8 | 3 | 2.792430878 | 0.104044256 | 0.037259384 | 2.704603195 | 2.907337189 | 0.049351618 | 34.662756 | 34.640106 | 0 |
| narrow-deep-2 | 1 | 16 | 3 | 2.855802536 | 0.155755424 | 0.054539984 | 2.678415298 | 2.970188141 | 0.050419990 | 34.711876 | 34.684285 | 0 |
| narrow-deep-3 | 2 | 8 | 3 | 2.663052241 | 0.224893000 | 0.084449338 | 2.516271591 | 2.921964645 | 0.048153780 | 34.739673 | 34.761836 | 0 |

## Interpretation

All four variants trained cleanly at `L=8`; none triggered the late-NLL failure
rule. The architecture ablation therefore does not reproduce the `L=18`
training-failure mode at this smaller size.

The clearest improvement is `narrow-deep-1`:

```text
depth = 1
width = 8
parameter_count = 1272222
mean MI = 2.792430878
seed_std = 0.104044256
cv = 0.037259384
mean AB test NLL = 34.662756
mean BA test NLL = 34.640106
```

Compared with the baseline `depth=0,width=64`, `narrow-deep-1` has lower test
NLL and lower seed spread:

| Quantity | baseline | narrow-deep-1 | delta |
|---|---:|---:|---:|
| mean MI | 2.624555 | 2.792431 | +0.167876 |
| seed std | 0.163543 | 0.104044 | -0.059499 |
| cv | 0.062313 | 0.037259 | -0.025053 |
| mean AB test NLL | 35.106450 | 34.662756 | -0.443694 |
| mean BA test NLL | 35.101063 | 34.640106 | -0.460958 |

`narrow-deep-2` also has a lower NLL than baseline, but its seed spread remains
close to baseline and its parameter count is more than 3.5x larger than
`narrow-deep-1`. It is not the first architecture to carry forward.

`narrow-deep-3` is not attractive: it has the largest seed spread among the
four variants and does not improve test NLL over the other narrow-deep options.

## Decision

Use `depth=1,width=8` as the next architecture candidate.

Recommended next pilot:

```text
run_id: p24_l18_arch_pilot_depth1_width8_ntrain400k
L: 18
train_seeds: 10,12,13 or a new predeclared seed block
depth: 1
width: 8
n_train: 400000
```

Do not promote any `L=18` endpoint from architecture results until the same
seed policy in `docs/SEED_POLICY.md` is applied.
