# p27 L14 MADE depth0 width64 seed9to16 ntrain400k

Date: 2026-06-07

Run id:

```text
p27_l14_made_depth0_width64_seed9to16_ntrain400k
```

Formal command:

```bash
env BASE_ROOT=net/mi_scaling/p27_l14_made_depth0_width64_seed9to16_ntrain400k L_VALUES='14' TRAIN_SEEDS='9 10 11 12 13 14 15 16' DEPTH=0 WIDTH=64 scripts/run_made_mi_ntrain400k.sh 2>&1 | tee logs/p27_l14_made_depth0_width64_seed9to16_ntrain400k.log
```

Note: the planned `scripts/run_codex_gpu.sh` invocation was not used for the
formal run because that wrapper starts a nested `codex exec` context in this
environment. A wrapper attempt was stopped before training artifacts were
created and saved as:

```text
logs/p27_l14_made_depth0_width64_seed9to16_ntrain400k.wrapper_attempt.log
```

Pre-run checks:

- `scripts/run_mi_agent_audits.sh` passed with `MI_AGENT_AUDITS_PASSED`.
- Direct CUDA check with `scripts/check_a100_env.sh` passed in the escalated
  shell; the device was `NVIDIA H100 PCIe`.

Run window, Asia/Shanghai:

```text
2026-06-07 12:13:32 to 2026-06-07 16:49:31
```

Artifacts:

```text
net/mi_scaling/p27_l14_made_depth0_width64_seed9to16_ntrain400k/
logs/p27_l14_made_depth0_width64_seed9to16_ntrain400k.log
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

## Per-Seed Results

| train_seed | MI | bootstrap std | AB best epoch | BA best epoch | AB test NLL | BA test NLL | AB late NLL max | BA late NLL max | failed by late NLL rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 6.114063 | 0.088961 | 44 | 45 | 110.630324 | 110.683037 | 111.239066 | 111.464938 | no |
| 10 | 5.341743 | 0.088853 | 38 | 47 | 110.801457 | 110.644418 | 111.297809 | 111.323459 | no |
| 11 | 5.958115 | 0.088728 | 42 | 41 | 110.665793 | 110.562924 | 111.242711 | 111.108074 | no |
| 12 | 5.945919 | 0.089253 | 49 | 39 | 110.851480 | 110.731068 | 111.351797 | 111.452953 | no |
| 13 | 6.195026 | 0.086606 | 46 | 45 | 110.824863 | 110.797033 | 111.346828 | 111.354336 | no |
| 14 | 6.781536 | 0.094517 | 48 | 44 | 110.930117 | 110.628449 | 111.566389 | 111.304611 | no |
| 15 | 6.247059 | 0.090334 | 42 | 43 | 110.738299 | 110.619092 | 111.276854 | 111.251416 | no |
| 16 | 6.441814 | 0.091615 | 43 | 46 | 110.619186 | 110.688236 | 111.203709 | 111.499068 | no |

Failure rule used here:

```text
Mark a seed as failed if late AB or BA train/val NLL reaches 1e3 or larger.
```

No seed in this run triggered the objective late-NLL training-failure rule.

## Aggregates

| L | n | seeds | n | mean MI | seed std | cv | min | max | mean bootstrap std | failed seeds |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 14 | 392 | 9,10,11,12,13,14,15,16 | 8 | 6.128159523 | 0.418272700 | 0.068254212 | 5.341743469 | 6.781536102 | 0.089858636 | none |

Comparison with the current recommended `p17_l14_ntrain400k` row:

| source | seeds | mean MI | seed std | cv | min | max | mean bootstrap std |
|---|---|---:|---:|---:|---:|---:|---:|
| `p17_l14_ntrain400k` | 1..8 | 6.074063778 | 0.313906723 | 0.051679853 | 5.563480377 | 6.552925110 | 0.088458206 |
| `p27_l14_made_depth0_width64_seed9to16_ntrain400k` | 9..16 | 6.128159523 | 0.418272700 | 0.068254212 | 5.341743469 | 6.781536102 | 0.089858636 |
| combined diagnostic | 1..16 | 6.101111650 | 0.358341715 | 0.058733840 | 5.341743469 | 6.781536102 | 0.089158421 |

The p27 mean is `+0.054095745` above p17, only `0.172331` p17
seed-standard deviations. The new seed block is therefore consistent with the
existing p17 mean, but has larger train-seed spread.

## Fit Sensitivity

Using the current recommended `L=12` and `L=16` points, the simple interpolation
at `L=14` is:

```text
(4.921826839 + 8.133990288) / 2 = 6.527908564
```

| L14 source | L14 MI | seed std | delta from interpolation | delta / seed std |
|---|---:|---:|---:|---:|
| current recommended p17 | 6.074064 | 0.313907 | -0.453845 | -1.445795 |
| p27 diagnostic | 6.128160 | 0.418273 | -0.399749 | -0.955714 |
| p17+p27 combined diagnostic | 6.101112 | 0.358342 | -0.426797 | -1.191033 |

Selected unweighted OLS windows if the p17 `L=14` row were replaced only for
sensitivity:

| L14 source | Window | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|
| p17 current | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| p27 diagnostic | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.208478 | 0.158100 |
| p17+p27 combined | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.213888 | 0.176793 |
| p17 current | `L=12,14,16,18` | 0.800734 | 0.400367 | -4.835186 | 0.137388 |
| p27 diagnostic | `L=12,14,16,18` | 0.798029 | 0.399015 | -4.781090 | 0.106868 |
| p17+p27 combined | `L=12,14,16,18` | 0.799382 | 0.399691 | -4.808138 | 0.121616 |

Because `L=14` is centered in the `L=10,12,14,16,18` window, substituting p27
or the combined diagnostic changes the intercept and RSS but not the slope in
that symmetric five-point window.

## Interpretation

p27 confirms that the current `L=14` low position is not an obvious artifact of
the original p17 seed block. The p27 mean is very close to p17 and remains
below the `L=12` to `L=16` interpolation. The combined 16-seed diagnostic also
remains below interpolation.

Decision:

- Keep `p17_l14_ntrain400k` as the current recommended `L=14` row.
- Add p27 as a diagnostic row with `include_in_recommended_fit=no`.
- Add the p17+p27 16-seed aggregate as a diagnostic sensitivity row with
  `include_in_recommended_fit=no`.
- Treat `L=14` as a stable but mildly low finite-size/local-curvature point.
- Move the next active decision back to the provisional `L=18` endpoint policy.
