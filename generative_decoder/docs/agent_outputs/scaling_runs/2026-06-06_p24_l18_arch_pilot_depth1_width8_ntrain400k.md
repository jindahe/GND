# p24 L18 Architecture Pilot, depth1 width8, ntrain400k

Date: 2026-06-06

Run id:

```text
p24_l18_arch_pilot_depth1_width8_ntrain400k
```

Command:

```bash
scripts/run_codex_gpu.sh "env BASE_ROOT=net/mi_scaling/p24_l18_arch_pilot_depth1_width8_ntrain400k L_VALUES=18 TRAIN_SEEDS='10 11 12 13' DEPTH=1 WIDTH=8 scripts/run_p16_l16_ntrain400k.sh"
```

Pre-run checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- CUDA wrapper check passed: PyTorch saw 1 CUDA device and allocated on
  `cuda:0`.

Run window, Asia/Shanghai:

```text
2026-06-06 13:56:46 to 2026-06-06 15:09:38
```

Artifacts:

```text
net/mi_scaling/p24_l18_arch_pilot_depth1_width8_ntrain400k/
logs/p24_l18_arch_pilot_depth1_width8_ntrain400k.log
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
| 10 | 9.359779 | 0.109731 | 18 | 21 | 184.402188 | 184.896246 | 186.806207 | 187.638563 | no |
| 11 | 9.825012 | 0.109352 | 19 | 18 | 184.690340 | 184.277852 | 186.945902 | 186.559926 | no |
| 12 | 9.914246 | 0.112436 | 22 | 20 | 184.814574 | 184.624980 | 188.151051 | 187.644742 | no |
| 13 | 8.838173 | 0.105649 | 21 | 20 | 184.772121 | 184.749945 | 187.991586 | 187.420574 | no |

Failure rule used here:

```text
Mark a seed as failed if late AB or BA train/val NLL reaches 1e3 or larger.
```

The late NLL max is computed from the saved JSON training records under
`models/records/`, using train/val histories from the best epoch through early
stop. No seed in this pilot triggered the objective training-failure rule.

## Aggregates

| Aggregate | Seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `p24` all seeds | 10,11,12,13 | 4 | 9.484302521 | 0.494612 | 0.052151 | 8.838172913 | 9.914245605 | 0.109292 | none |

For comparison:

| Aggregate | mean MI | seed std | cv |
|---|---:|---:|---:|
| `p22` all seeds, same train_seed block under depth0 width64 | 10.453517914 | 0.925020301 | 0.088489 |
| `p22` clean only | 10.899734497 | 0.298057714 | 0.027345 |
| `p24` all seeds, depth1 width8 | 9.484302521 | 0.494612 | 0.052151 |

## Interpretation

The `depth=1,width=8` architecture removed the objective late-NLL failures for
the tested `L=18` seed block. In particular, train seed 11 no longer shows the
`BA` late divergence that appeared in `p22` under the `depth=0,width=64`
protocol.

The `p24` aggregate has `cv = 0.052151`, lower than the `p22` all-seed
diagnostic `cv = 0.088489` and below the `cv <= 0.06` usable-baseline gate.
It is not below the stricter `cv <= 0.05` formal-update gate, and it is only a
4-seed architecture pilot.

The mean MI also shifts downward relative to the `p22` same-seed diagnostic:

```text
p22 same seeds mean MI = 10.453517914
p24 same seeds mean MI = 9.484302521
delta = -0.969215393
```

This supports architecture sensitivity at `L=18`; it should not be silently
used as a replacement endpoint without a larger predeclared seed block and fit
sensitivity report.

## Decision

Keep `L=18` provisional and keep `p24` as a diagnostic architecture row in
`docs/MI_FIT_POINTS.csv` with `include_in_recommended_fit=no`.

Recommended next step:

Run a larger predeclared `L=18` `depth=1,width=8` seed block, reusing the same
seed policy. A natural next block is train seeds `1 2 3 4 5 6 7 8`, so the new
architecture can be compared directly with the current provisional `p19`
baseline seed set.
