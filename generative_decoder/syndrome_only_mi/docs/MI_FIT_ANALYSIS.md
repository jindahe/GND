# MI Fit Analysis For Syndrome-Only Scaling

Generated from `docs/MI_FIT_POINTS.csv` recommended rows on 2026-06-07.

The active fit target is:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

All fits below are unweighted ordinary least squares fits of `MI` against `L`.
Residuals are reported as:

```text
residual = observed MI - fitted MI
```

Normalized residuals use `residual / MI_std_across_train_seeds` where a
nonzero train-seed standard deviation is available. They are not reported for
historical single-point rows with zero train-seed spread.

## Recommended Input Points

| L | n | run_id | seeds | n_train | MI | seed_std | bootstrap_std_mean |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 32 | `p8_made_plateau_long_468` | 1 |  | 1.074160 | 0.000000 | 0.021748 |
| 6 | 72 | `p8_made_plateau_long_468` | 1 |  | 1.513966 | 0.000000 | 0.032413 |
| 8 | 128 | `p18_l8_ntrain400k` | 8 | 400000 | 2.640582 | 0.132981 | 0.050881 |
| 10 | 200 | `p26_l10_l12_made_depth0_width64_ntrain400k` | 8 | 400000 | 3.557084 | 0.190860 | 0.063400 |
| 12 | 288 | `p26_l10_l12_made_depth0_width64_ntrain400k` | 8 | 400000 | 4.921827 | 0.210903 | 0.073677 |
| 14 | 392 | `p17_l14_ntrain400k` | 8 | 400000 | 6.074064 | 0.313907 | 0.088458 |
| 16 | 512 | `p16_l16_ntrain400k` | 8 | 400000 | 8.133990 | 0.382952 | 0.094499 |
| 18 | 648 | `p19_l18_ntrain400k_pilot` | 8 | 400000 | 9.573411 | 0.574411 | 0.104420 |

## Fit Summary

| Window | Points | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| all recommended | `L=4,6,8,10,12,14,16,18` | 8 | 0.620596 | 0.310298 | -2.140417 | 1.619790 |
| bridge and large-L | `L=8,10,12,14,16,18` | 6 | 0.707816 | 0.353908 | -3.384778 | 0.473119 |
| current multi-seed large-L | `L=10,12,14,16,18` | 5 | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| without L14 | `L=10,12,16,18` | 4 | 0.762241 | 0.381120 | -4.124795 | 0.018041 |
| `L>=12` | `L=12,14,16,18` | 4 | 0.800734 | 0.400367 | -4.835186 | 0.137388 |
| largest three | `L=14,16,18` | 3 | 0.874837 | 0.437418 | -6.070234 | 0.064171 |

## L14 Diagnostic

The `L=14` point is usable as an 8-seed baseline, but it is low relative to the
simple interpolation between the current `L=12` and `L=16` recommended points.
The p26 `L=12` replacement lowers that interpolation and reduces the residual
relative to the previous p9 bridge row.

| Quantity | Value |
|---|---:|
| `L12` recommended MI | 4.921827 |
| `L16` recommended MI | 8.133990 |
| Linear interpolation at `L14` | 6.527909 |
| Observed `L14` mean | 6.074064 |
| `L14` delta from interpolation | -0.453845 |
| `L14` seed_std | 0.313907 |
| Delta in `L14` seed_std units | -1.445795 |

This low position remains visible, but p26 reduces the discrepancy enough that
`L=14` is no longer the dominant residual in the current `L=10..18` window.
It should not be hidden or silently dropped. The recommended analysis therefore
keeps both with-`L14` and without-`L14` windows in view.

## P27 L14 Stability Diagnostic

The p27 same-architecture diagnostic used MADE `depth=0,width=64`,
`n_train=400k`, and train seeds 9..16. All seeds passed the objective
late-NLL training-failure rule.

| Source | seeds | mean MI | seed_std | cv | min | max | mean bootstrap std |
|---|---|---:|---:|---:|---:|---:|---:|
| current p17 recommended | 1..8 | 6.074064 | 0.313907 | 0.051680 | 5.563480 | 6.552925 | 0.088458 |
| p27 diagnostic | 9..16 | 6.128160 | 0.418273 | 0.068254 | 5.341743 | 6.781536 | 0.089859 |
| p17+p27 combined diagnostic | 1..16 | 6.101112 | 0.358342 | 0.058734 | 5.341743 | 6.781536 | 0.089158 |

The p27 mean is only `+0.054096` above p17, or `0.172331` p17
seed-standard deviations. This confirms that the current `L=14` low position
is stable under an independent same-protocol seed block, although p27 has
larger train-seed spread.

Relative to the current `L=12` to `L=16` interpolation:

| L14 source | L14 MI | seed_std | delta from interpolation | delta / seed_std |
|---|---:|---:|---:|---:|
| p17 current | 6.074064 | 0.313907 | -0.453845 | -1.445795 |
| p27 diagnostic | 6.128160 | 0.418273 | -0.399749 | -0.955714 |
| p17+p27 combined | 6.101112 | 0.358342 | -0.426797 | -1.191033 |

Selected replacement-only fit sensitivities:

| L14 source | Window | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|
| p17 current | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| p27 diagnostic | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.208478 | 0.158100 |
| p17+p27 combined | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.213888 | 0.176793 |
| p17 current | `L=12,14,16,18` | 0.800734 | 0.400367 | -4.835186 | 0.137388 |
| p27 diagnostic | `L=12,14,16,18` | 0.798029 | 0.399015 | -4.781090 | 0.106868 |
| p17+p27 combined | `L=12,14,16,18` | 0.799382 | 0.399691 | -4.808138 | 0.121616 |

Decision:

- Keep `p17_l14_ntrain400k` as the current recommended `L=14` row.
- Keep p27 and the p17+p27 combined aggregate as diagnostic rows with
  `include_in_recommended_fit=no`.
- Treat `L=14` as a stable but mildly low finite-size/local-curvature point.

## P26 L10/L12 Rerun

The p26 rerun replaced the mixed-`n_train` p9 bridge rows with clean
`n_train=400k`, MADE `depth=0,width=64` rows. All seeds passed the objective
late-NLL training-failure rule.

| L | previous p9 MI | p26 MI | delta | p26 seed_std | p26 cv |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.689559 | 3.557084 | -0.132475 | 0.190860 | 0.053656 |
| 12 | 5.384724 | 4.921827 | -0.462897 | 0.210903 | 0.042851 |

The larger `L=12` downward shift raises the preferred large-`L` slope:
`2 alpha` for the `L=10,12,14,16,18` window changes from `0.725849` under p9
to `0.762241` under p26. The matching `without L14` window has the same slope
and a smaller RSS, `0.018041`.

## L18 Diagnostic

The `L=18` pilot completed 8 train seeds with the same `n_train=400k`,
`batch=512`, and learning schedule as the `L=14/16` runs. It is included as a
provisional recommended point because it is the only completed largest-size
aggregate, but its stability is borderline.

A dedicated seed-split diagnostic is recorded in
`docs/agent_outputs/scaling_runs/2026-06-04_p19_l18_seed_split_diagnostic.md`.
The focused seed-5 rerun is recorded in
`docs/agent_outputs/scaling_runs/2026-06-04_p20_l18_seed5_rerun_ntrain400k.md`.
The seed-9 replacement diagnostic is recorded in
`docs/agent_outputs/scaling_runs/2026-06-04_p21_l18_replace_seed5_ntrain400k.md`.
The seed-10/11/12/13 replacement diagnostic is recorded in
`docs/agent_outputs/scaling_runs/2026-06-05_p22_l18_more_replacement_seeds_ntrain400k.md`.
The seed-10/11/12/13 `depth=1,width=8` architecture pilot is recorded in
`docs/agent_outputs/scaling_runs/2026-06-06_p24_l18_arch_pilot_depth1_width8_ntrain400k.md`.
The seed-1..8 `depth=1,width=8` architecture block is recorded in
`docs/agent_outputs/scaling_runs/2026-06-06_p25_l18_arch_depth1_width8_seeds1to8_ntrain400k.md`.
The same-architecture `n_train=1000k` data-size pilot is recorded in
`docs/agent_outputs/scaling_runs/2026-06-07_p28_l18_made_depth0_width64_ntrain1000k_rerun.md`.
The same-architecture p31 batch-1024 lr-5e-4 gradclip warmup diagnostic is
recorded in
`docs/agent_outputs/scaling_runs/2026-06-08_p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot.md`.
The same-scope p32 fixed-LR follow-up is recorded in
`docs/agent_outputs/scaling_runs/2026-06-08_p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot.md`.
The current seed policy is recorded in `docs/SEED_POLICY.md`.

Per-seed MI:

| train_seed | MI | bootstrap_std | note |
|---:|---:|---:|---|
| 1 | 9.065102 | 0.102899 |  |
| 2 | 8.760048 | 0.108807 |  |
| 3 | 9.347771 | 0.102633 |  |
| 4 | 9.553818 | 0.109733 |  |
| 5 | 10.379700 | 0.094562 | high MI; BA training diagnostic |
| 6 | 9.807640 | 0.105091 |  |
| 7 | 10.334045 | 0.103782 | high MI |
| 8 | 9.339165 | 0.107854 |  |

Additional replacement-seed diagnostics:

| train_seed | source | MI | bootstrap_std | note |
|---:|---|---:|---:|---|
| 9 | `p21` | 10.953156 | 0.104264 | clean AB/BA training; high MI |
| 10 | `p22` | 10.672577 | 0.105356 | clean AB/BA training; high MI |
| 11 | `p22` | 9.114868 | 0.103321 | BA late-NLL training failure |
| 12 | `p22` | 11.237228 | 0.104360 | clean AB/BA training; very high MI |
| 13 | `p22` | 10.789398 | 0.104906 | clean AB/BA training; high MI |

Architecture pilot diagnostics:

| train_seed | source | architecture | MI | bootstrap_std | AB test NLL | BA test NLL | failed late-NLL rule |
|---:|---|---|---:|---:|---:|---:|---|
| 10 | `p24` | depth1 width8 | 9.359779 | 0.109731 | 184.402188 | 184.896246 | no |
| 11 | `p24` | depth1 width8 | 9.825012 | 0.109352 | 184.690340 | 184.277852 | no |
| 12 | `p24` | depth1 width8 | 9.914246 | 0.112436 | 184.814574 | 184.624980 | no |
| 13 | `p24` | depth1 width8 | 8.838173 | 0.105649 | 184.772121 | 184.749945 | no |

Architecture seed-block diagnostics:

| train_seed | source | architecture | MI | bootstrap_std | AB test NLL | BA test NLL | failed late-NLL rule |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `p25` | depth1 width8 | 8.378075 | 0.103806 | 184.691031 | 185.031055 | no |
| 2 | `p25` | depth1 width8 | 9.724525 | 0.111176 | 184.448164 | 184.833750 | no |
| 3 | `p25` | depth1 width8 | 10.777229 | 0.101371 | 185.020949 | 184.224219 | no |
| 4 | `p25` | depth1 width8 | 9.766930 | 0.107446 | 185.072090 | 184.528105 | no |
| 5 | `p25` | depth1 width8 | 9.342827 | 0.106336 | 184.769242 | 184.709957 | no |
| 6 | `p25` | depth1 width8 | 7.852448 | 0.102802 | 184.549656 | 184.422453 | no |
| 7 | `p25` | depth1 width8 | 9.243690 | 0.106546 | 185.153754 | 184.689980 | no |
| 8 | `p25` | depth1 width8 | 9.553291 | 0.104418 | 184.513301 | 184.665703 | no |

Aggregate diagnostics:

| Subset | seeds | mean MI | seed_std | cv | min | max |
|---|---:|---:|---:|---:|---:|---:|
| all seeds | 8 | 9.573411 | 0.574411 | 0.060001 | 8.760048 | 10.379700 |
| without seed 5 | 7 | 9.458227 | 0.510990 | 0.054026 | 8.760048 | 10.334045 |
| without seeds 5 and 7 | 6 | 9.312257 | 0.366541 | 0.039361 | 8.760048 | 9.807640 |
| seeds 1..3 pilot | 3 | 9.057640 | 0.293932 | 0.032451 | 8.760048 | 9.347771 |
| seeds 4..8 extension | 5 | 9.882874 | 0.463667 | 0.046916 | 9.339165 | 10.379700 |
| seed 5 replaced by seed 9 | 8 | 9.645093 | 0.709338 | 0.073544 | 8.760048 | 10.953156 |
| seed 5 replaced by first clean p22 seed 10 | 8 | 9.610021 | 0.638858 | 0.066478 | 8.760048 | 10.672577 |
| all clean p19/p21/p22 seeds | 11 | 9.987268 | 0.844645 | 0.084572 | 8.760048 | 11.237228 |
| p22 all seeds | 4 | 10.453518 | 0.925020 | 0.088489 | 9.114868 | 11.237228 |
| p22 clean only | 3 | 10.899734 | 0.298058 | 0.027345 | 10.672577 | 11.237228 |
| p24 depth1 width8 same seeds | 4 | 9.484303 | 0.494612 | 0.052151 | 8.838173 | 9.914246 |
| p25 depth1 width8 seeds 1..8 | 8 | 9.329877 | 0.893454 | 0.095763 | 7.852448 | 10.777229 |
| p28 depth0 width64 ntrain1000k seeds 1,2,3,5 | 4 | 7.385063 | 1.628577 | 0.220523 | 5.670700 | 8.816757 |
| p31 depth0 width64 batch1024 lr5e-4 seeds 1,2,3,5 | 4 | 8.067623 | 0.968598 | 0.120060 | 6.855270 | 9.218262 |
| p32 depth0 width64 batch1024 lr5e-4 fixed-LR seeds 1,2,3,5 | 4 | 8.132694 | 0.284404 | 0.034970 | 7.819458 | 8.509644 |
| p32 depth0 width64 batch1024 lr5e-4 fixed-LR seeds 1..8 | 8 | 8.137888 | 0.455173 | 0.055933 | 7.473457 | 8.992081 |

P32 is the cleanest same-architecture optimizer diagnostic at `L=18`: all
eight seeds pass the saved-JSON late-NLL rule and the 8-seed aggregate has
`cv = 0.055933`. However, its mean is `8.137888`, essentially equal to the
current recommended `L=16` mean `8.133990`.

Selected endpoint fit sensitivities:

| L18 source | L18 MI | Window | 2 alpha | alpha | beta | RSS |
|---|---:|---|---:|---:|---:|---:|
| current p19 provisional | 9.573411 | `L=10,12,14,16,18` | 0.762241 | 0.381120 | -4.219298 | 0.196656 |
| p32 fixed-LR 8-seed | 8.137888 | `L=10,12,14,16,18` | 0.618689 | 0.309344 | -2.496670 | 0.813163 |
| current through L16 |  | `L=10,12,14,16` | 0.744148 | 0.372074 | -4.002181 | 0.183562 |

Do not silently substitute p32 for the recommended `L=18` row. The p32 result
uses a fixed-LR optimizer protocol that has only been tested at the endpoint;
direct substitution would mix training protocols and strongly change the
fitted slope. A same-protocol fixed-LR anchor at `L=16` is the next diagnostic
needed to decide whether p32 should seed a new endpoint policy or a new
same-protocol scaling subset.

L18 endpoint sensitivity for selected fit windows:

| L18 scenario | L18 MI | `2 alpha`, L=10..18 | `beta`, L=10..18 | RSS, L=10..18 | `2 alpha`, without L14 | `beta`, without L14 | RSS, without L14 |
|---|---:|---:|---:|---:|---:|---:|---:|
| current p19 all | 9.573411 | 0.762241 | -4.219298 | 0.196656 | 0.762241 | -4.124795 | 0.018041 |
| p19 failed seed5 excluded | 9.458227 | 0.750723 | -4.081077 | 0.185291 | 0.750723 | -3.992333 | 0.027782 |
| p19 failed5 high7 excluded | 9.312257 | 0.736126 | -3.905913 | 0.186136 | 0.736126 | -3.824468 | 0.053470 |
| seed5 replaced by seed9 | 9.645093 | 0.769409 | -4.305316 | 0.209087 | 0.769409 | -4.207229 | 0.016666 |
| seed5 replaced by first clean p22 seed10 | 9.610021 | 0.765902 | -4.263229 | 0.202491 | 0.765902 | -4.166896 | 0.016889 |
| all clean p19/p21/p22 | 9.987268 | 0.803627 | -4.715926 | 0.325071 | 0.803627 | -4.600730 | 0.059670 |
| p24 depth1 width8 same seeds | 9.484303 | 0.753330 | -4.112367 | 0.186934 | 0.753330 | -4.022320 | 0.024764 |
| p25 depth1 width8 seeds 1..8 | 9.329877 | 0.737888 | -3.927057 | 0.185130 | 0.737888 | -3.844731 | 0.049578 |

The full 8-seed cv is `0.060001`, effectively at but technically just above
the `0.06` usable-baseline gate. Seed 5 is the most concerning point: its `BA`
training record selected best epoch 19, later epochs showed NLL values in the
`6e3` range, and its final MI is the maximum of the set. Seed 7 is also high
but did not show the same obvious training failure signature.

Interpretation:

- Do not treat the `L=18` aggregate as a clean formal result yet.
- Keep it in the fit as the current provisional largest-size point, but report
  sensitivity to seed 5 and seed 7 whenever using it.
- Under the post-p26 bridge rows, the preferred `L=10,12,14,16,18` slope
  changes from `2 alpha = 0.762241` with all `L=18` seeds to `0.750723`
  without seed 5 and `0.736126` without seeds 5 and 7.
- The `p20` seed-5 rerun exactly reproduced the failed `BA` trajectory and
  `MI = 10.379700`, so the issue is deterministic under the current protocol.
- Replacing seed 5 with clean-training seed 9 raises the replacement aggregate
  to `mean(MI) = 9.645093`, `seed_std = 0.709338`, and `cv = 0.073544`.
- The `p22` replacement batch found another failed-`BA` seed, seed 11, and
  three clean high-MI seeds, seeds 10/12/13.
- Replacing failed seed 5 by the first clean p22 seed, seed 10, gives
  `mean(MI) = 9.610021`, `seed_std = 0.638858`, and `cv = 0.066478`, still
  above the `cv <= 0.06` usable-baseline gate.
- The all-clean p19/p21/p22 aggregate gives `mean(MI) = 9.987268`,
  `seed_std = 0.844645`, and `cv = 0.084572`, so clean high-MI seeds are common
  under the current protocol.
- The `p24` `depth=1,width=8` architecture pilot removes late-NLL failures on
  seeds 10/11/12/13 and gives `mean(MI) = 9.484303`, `seed_std = 0.494612`,
  and `cv = 0.052151`. This is below the `cv <= 0.06` usable-baseline gate but
  is only a 4-seed architecture diagnostic and shifts the mean downward relative
  to `p22` same-seed diagnostics.
- The `p25` `depth=1,width=8` predeclared seeds 1..8 block removes objective
  late-NLL failures on the p19-comparable seed set, but gives
  `mean(MI) = 9.329877`, `seed_std = 0.893454`, and `cv = 0.095763`. This is
  above the usable-baseline gate and shows that the new architecture is clean
  but not stable as an endpoint.
- The `p28` same-architecture `n_train=1000k` pilot is not a candidate
  endpoint: all four pilot seeds fail the objective late-NLL rule in both AB
  and BA, `mean(MI) = 7.385063`, and `cv = 0.220523`. Increasing `n_train`
  alone did not solve the active MADE endpoint instability.
- Current evidence favors treating `L=18` as an architecture-sensitive
  provisional endpoint. Do not promote a replacement or architecture aggregate
  to the recommended fit row without a larger predeclared seed block and a new
  explicit endpoint decision.

## Endpoint Policy

Current policy:

- Keep `p19_l18_ntrain400k_pilot` as the single `include_in_recommended_fit=yes`
  `L=18` row because it is the only completed active-architecture 8-seed
  largest-size aggregate.
- Treat that row as provisional, not as a clean formal endpoint.
- Do not promote failed-seed-excluded, replacement-seed, all-clean, or
  `depth=1,width=8` architecture aggregates into the recommended fit at this
  stage. Keep p28 as a failed data-size diagnostic with
  `include_in_recommended_fit=no`.
- Interpret the scaling using two mandatory windows:
  the current provisional endpoint window `L=10,12,14,16,18`, and the
  endpoint-stable through-`L=16` window `L=10,12,14,16`.
- When quoting endpoint-sensitive conclusions, give both values:
  `2 alpha = 0.762241` (`alpha = 0.381120`) for the provisional
  `L=10,12,14,16,18` window, and `2 alpha = 0.744148`
  (`alpha = 0.372074`) for the through-`L=16` window.
- Do not launch another MADE or architecture run automatically. A new `L=18`
  run should only start after a predeclared endpoint plan specifies the
  architecture, seed block, replacement/failure rule, and promotion rule.

Rationale:

The replacement and architecture diagnostics do not point to a clean
substitute endpoint. Excluding the failed p19 seed, replacing it by clean seeds,
or using all clean p19/p21/p22 seeds all changes the endpoint mean and leaves
substantial train-seed spread. The `depth=1,width=8` architecture removes
objective late-NLL failures in p24 and p25, but the predeclared p25 seed block
has `cv = 0.095763`, so it is not a stable replacement architecture. Keeping
p19 as provisional preserves the largest-size evidence while the through-`L=16`
window protects the interpretation from depending on an unstable endpoint.

## L<=16 Window Details

The `L<=16` diagnostics below use the post-p26 recommended rows and omit the
provisional `L=18` endpoint.

| Window | Points | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| all through L16 | `L=4,6,8,10,12,14,16` | 7 | 0.581802 | 0.290901 | -1.830071 | 1.114137 |
| bridge through L16 | `L=8,10,12,14,16` | 5 | 0.675190 | 0.337595 | -3.036769 | 0.373770 |
| multi-seed L10-L16 | `L=10,12,14,16` | 4 | 0.744148 | 0.372074 | -4.002181 | 0.183562 |
| without L14 L<=16 | `L=10,12,16` | 3 | 0.768564 | 0.384282 | -4.197510 | 0.016641 |
| largest three L<=16 | `L=12,14,16` | 3 | 0.803041 | 0.401520 | -4.865945 | 0.137317 |

Interpretation:

- The p26 replacement raises the L10-L16 large-window slope relative to the
  older p9 bridge rows because `L=12` shifted downward more than `L=10`.
- `L=14` remains the local curvature diagnostic in the `L=12,14,16` triplet,
  but the discrepancy is weaker than it was under the p9 `L=12` row.
- The L<=16 windows are useful for checking bridge consistency, but the
  current endpoint decision still depends on how `L=18` is handled.

## Conclusions

- The current multi-seed large-`L` fit including provisional `L=18` gives
  `2 alpha = 0.762241` and `alpha = 0.381120` using `L=10,12,14,16,18`.
- Excluding the low `L=14` point gives the same slope for `L=10,12,16,18`,
  but lowers RSS from `0.196656` to `0.018041`; this keeps `L=14` visible as a
  finite-size or stability diagnostic without silently removing it.
- The p27 same-architecture `L=14` seed block is clean and consistent with p17.
  It confirms the low `L=14` position while increasing the measured seed-spread
  context; it is recorded as a diagnostic and does not replace the recommended
  p17 row.
- Including the rechecked `L=8` bridge point gives a lower bridge-plus-large
  slope, `2 alpha = 0.707816`, than the `L>=10` multi-seed window.
- The all-recommended fit is useful as a recorded full-curve summary, but the
  small historical sizes make it a poor standalone asymptotic estimate.
- The p26 bridge rerun is clean and should remain the recommended `L=10/12`
  input unless a later explicit endpoint policy changes the active training
  architecture.
- `L=18` remains provisional by policy. Keep p19 as the sole recommended
  largest-size row for continuity, but pair every endpoint-sensitive
  interpretation with the endpoint-stable through-`L=16` window. Do not promote
  any diagnostic `L=18` aggregate without a predeclared endpoint plan.
