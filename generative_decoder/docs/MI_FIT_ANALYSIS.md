# MI Fit Analysis For Syndrome-Only Scaling

Generated from `docs/MI_FIT_POINTS.csv` recommended rows on 2026-06-04.

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
| 10 | 200 | `p9_largeL_ntrain200k` | 8 | 200000 | 3.689559 | 0.265209 | 0.063953 |
| 12 | 288 | `p9_largeL_ntrain200k` | 8 | 200000 | 5.384724 | 0.321396 | 0.073780 |
| 14 | 392 | `p17_l14_ntrain400k` | 8 | 400000 | 6.074064 | 0.313907 | 0.088458 |
| 16 | 512 | `p16_l16_ntrain400k` | 8 | 400000 | 8.133990 | 0.382952 | 0.094499 |
| 18 | 648 | `p19_l18_ntrain400k_pilot` | 8 | 400000 | 9.573411 | 0.574411 | 0.104420 |

## Fit Summary

| Window | Points | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| all recommended | `L=4,6,8,10,12,14,16,18` | 8 | 0.622562 | 0.311281 | -2.087630 | 1.315597 |
| bridge and large-L | `L=8,10,12,14,16,18` | 6 | 0.695525 | 0.347763 | -3.125775 | 0.425885 |
| current multi-seed large-L | `L=10,12,14,16,18` | 5 | 0.725849 | 0.362924 | -3.590730 | 0.340066 |
| without L14 | `L=10,12,16,18` | 4 | 0.725849 | 0.362924 | -3.466458 | 0.031198 |
| `L>=12` | `L=12,14,16,18` | 4 | 0.731299 | 0.365650 | -3.677944 | 0.338878 |
| largest three | `L=14,16,18` | 3 | 0.874837 | 0.437418 | -6.070234 | 0.064171 |

## L14 Diagnostic

The `L=14` point is usable as an 8-seed baseline, but it is low relative to the
simple interpolation between the current `L=12` and `L=16` recommended points.

| Quantity | Value |
|---|---:|
| `L12` recommended MI | 5.384724 |
| `L16` recommended MI | 8.133990 |
| Linear interpolation at `L14` | 6.759357 |
| Observed `L14` mean | 6.074064 |
| `L14` delta from interpolation | -0.685293 |
| `L14` seed_std | 0.313907 |
| Delta in `L14` seed_std units | -2.183111 |

This low position is the main reason the with-`L14` large-`L` fit has a lower
slope than the `without L14` fit. It should not be hidden or silently dropped.
The recommended analysis therefore keeps both with-`L14` and without-`L14`
windows in view.

## L18 Diagnostic

The `L=18` pilot completed 8 train seeds with the same `n_train=400k`,
`batch=512`, and learning schedule as the `L=14/16` runs. It is included as a
provisional recommended point because it is the only completed largest-size
aggregate, but its stability is borderline.

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

Aggregate diagnostics:

| Subset | seeds | mean MI | seed_std | cv | min | max |
|---|---:|---:|---:|---:|---:|---:|
| all seeds | 8 | 9.573411 | 0.574411 | 0.060001 | 8.760048 | 10.379700 |
| without seed 5 | 7 | 9.458227 | 0.510990 | 0.054026 | 8.760048 | 10.334045 |
| without seeds 5 and 7 | 6 | 9.312257 | 0.366541 | 0.039361 | 8.760048 | 9.807640 |
| seeds 1..3 pilot | 3 | 9.057640 | 0.293932 | 0.032451 | 8.760048 | 9.347771 |
| seeds 4..8 extension | 5 | 9.882874 | 0.463667 | 0.046916 | 9.339165 | 10.379700 |

The full 8-seed cv is `0.060001`, effectively at but technically just above
the `0.06` usable-baseline gate. Seed 5 is the most concerning point: its `BA`
training record selected best epoch 19, later epochs showed NLL values in the
`6e3` range, and its final MI is the maximum of the set. Seed 7 is also high
but did not show the same obvious training failure signature.

Interpretation:

- Do not treat the `L=18` aggregate as a clean formal result yet.
- Keep it in the fit as the current provisional largest-size point, but report
  sensitivity to seed 5 and seed 7 whenever using it.
- Before rerunning `L=10/12`, inspect `L=18` seed-level behavior and consider
  either a focused seed-5 rerun under a new run id or additional seeds if the
  large-`L` conclusion depends on this point.

## L<=16 Window Details

The detailed residual and leave-one-out tables below are the `L<=16`
diagnostics from before adding the provisional `L=18` point. The current
post-`L18` fit summary is the table above.

### All Recommended Through L16

Fit window: `L=4,6,8,10,12,14,16`

```text
2 alpha = 0.590068391
alpha   = 0.295034196
beta    = -1.827677591
RSS     = 0.960826752
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.074160 | 0.532596 | 0.541564 | 0.000000 | n/a |
| 6 | 1.513966 | 1.712733 | -0.198767 | 0.000000 | n/a |
| 8 | 2.640582 | 2.892870 | -0.252287 | 0.132981 | -1.897173 |
| 10 | 3.689559 | 4.073006 | -0.383447 | 0.265209 | -1.445830 |
| 12 | 5.384724 | 5.253143 | 0.131581 | 0.321396 | 0.409403 |
| 14 | 6.074064 | 6.433280 | -0.359216 | 0.313907 | -1.144340 |
| 16 | 8.133990 | 7.613417 | 0.520574 | 0.382952 | 1.359371 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.644225 | 0.322112 | -2.513658 | 0.413350 | 0.054156 | -0.685981 |
| 6 | 0.580130 | 0.290065 | -1.688541 | 0.905515 | -0.009938 | 0.139137 |
| 8 | 0.584584 | 0.292292 | -1.728956 | 0.883341 | -0.005485 | 0.098721 |
| 10 | 0.590068 | 0.295034 | -1.763770 | 0.789290 | -0.000000 | 0.063908 |
| 12 | 0.587208 | 0.293604 | -1.821957 | 0.939750 | -0.002860 | 0.005721 |
| 14 | 0.608029 | 0.304015 | -1.935442 | 0.780176 | 0.017961 | -0.107765 |
| 16 | 0.538011 | 0.269006 | -1.445924 | 0.454966 | -0.052057 | 0.381754 |

Interpretation:

- The all-point fit is still influenced by finite-size curvature and by the
  historical `L=4/6` points.
- The largest normalized residual in the multi-seed subset is now `L=8`
  at `-1.897173` seed-std units.
- Replacing the historical `L=8` single point with the `p18` 8-seed mean
  reduces the full-window RSS from the previous `1.843343` to `0.960827`.

### Bridge And Large-L

Fit window: `L=8,10,12,14,16`

```text
2 alpha = 0.668566048
alpha   = 0.334283024
beta    = -2.838208771
RSS     = 0.358050016
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 8 | 2.640582 | 2.510320 | 0.130262 | 0.132981 | 0.979559 |
| 10 | 3.689559 | 3.847452 | -0.157892 | 0.265209 | -0.595351 |
| 12 | 5.384724 | 5.184584 | 0.200140 | 0.321396 | 0.622720 |
| 14 | 6.074064 | 6.521716 | -0.447652 | 0.313907 | -1.426067 |
| 16 | 8.133990 | 7.858848 | 0.275142 | 0.382952 | 0.718478 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.701132 | 0.350566 | -3.294127 | 0.315629 | 0.032566 | -0.455919 |
| 10 | 0.657288 | 0.328644 | -2.657760 | 0.322436 | -0.011278 | 0.180449 |
| 12 | 0.668566 | 0.334283 | -2.888244 | 0.307980 | 0.000000 | -0.050035 |
| 14 | 0.700541 | 0.350271 | -3.094010 | 0.071775 | 0.031975 | -0.255801 |
| 16 | 0.599780 | 0.299890 | -2.150353 | 0.168792 | -0.068786 | 0.687856 |

Interpretation:

- Removing `L=8` raises the slope to the current multi-seed large-`L` value,
  so the rechecked bridge point now pulls the bridge-plus-large slope down.
- Removing `L=14` raises `2 alpha` by `0.031975` and reduces RSS to `0.071775`.
- The largest normalized residual is `L=14` at `-1.426067` seed-std units.

### Current Multi-Seed Large-L

Fit window: `L=10,12,14,16`

```text
2 alpha = 0.701131666
alpha   = 0.350565833
beta    = -3.294127418
RSS     = 0.315629238
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.689559 | 3.717189 | -0.027630 | 0.265209 | -0.104182 |
| 12 | 5.384724 | 5.119453 | 0.265271 | 0.321396 | 0.825371 |
| 14 | 6.074064 | 6.521716 | -0.447652 | 0.313907 | -1.426067 |
| 16 | 8.133990 | 7.923979 | 0.210011 | 0.382952 | 0.548401 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.687317 | 0.343658 | -3.091507 | 0.313085 | -0.013815 | 0.202620 |
| 12 | 0.720080 | 0.360040 | -3.635190 | 0.215102 | 0.018948 | -0.341063 |
| 14 | 0.733107 | 0.366553 | -3.549929 | 0.029354 | 0.031975 | -0.255801 |
| 16 | 0.596126 | 0.298063 | -2.104065 | 0.168614 | -0.105006 | 1.190063 |

Interpretation:

- This is the cleanest current multi-seed window, but it contains only four
  sizes and is visibly sensitive to the endpoints.
- `L=14` is the largest normalized residual and is the main source of RSS.
- Dropping `L=14` gives `2 alpha = 0.733107`, while dropping `L=16` gives
  `2 alpha = 0.596126`. This highlights that `L=18` is important for anchoring
  the large-size trend.

### Without L14

Fit window: `L=10,12,16`

```text
2 alpha = 0.733106818
alpha   = 0.366553409
beta    = -3.549928632
RSS     = 0.029354344
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.689559 | 3.781140 | -0.091580 | 0.265209 | -0.345314 |
| 12 | 5.384724 | 5.247353 | 0.137370 | 0.321396 | 0.427418 |
| 16 | 8.133990 | 8.179780 | -0.045790 | 0.382952 | -0.119572 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.687317 | 0.343658 | -2.863076 | 0.000000 | -0.045790 | 0.686852 |
| 12 | 0.740739 | 0.370369 | -3.717826 | 0.000000 | 0.007632 | -0.167897 |
| 16 | 0.847582 | 0.423791 | -4.786263 | 0.000000 | 0.114475 | -1.236334 |

Interpretation:

- This window fits the three remaining points tightly, but it should be treated
  as a sensitivity diagnostic rather than the default result.
- Because there are only three points, each leave-one-out fit is a two-point
  interpolation with zero RSS.
- The slope is higher than the with-`L14` multi-seed fit by `0.031975` in
  `2 alpha`.

### Largest Three

Fit window: `L=12,14,16`

```text
2 alpha = 0.687316656
alpha   = 0.343658328
beta    = -3.091507278
RSS     = 0.313084511
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 12 | 5.384724 | 5.156293 | 0.228431 | 0.321396 | 0.710746 |
| 14 | 6.074064 | 6.530926 | -0.456862 | 0.313907 | -1.455407 |
| 16 | 8.133990 | 7.905559 | 0.228431 | 0.382952 | 0.596501 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1.029963 | 0.514982 | -8.345422 | 0.000000 | 0.342647 | -5.253915 |
| 14 | 0.687317 | 0.343658 | -2.863076 | 0.000000 | -0.000000 | 0.228431 |
| 16 | 0.344670 | 0.172335 | 1.248683 | 0.000000 | -0.342647 | 4.340190 |

Interpretation:

- The largest-three window makes the local curvature around `L=14` explicit:
  `L=14` sits below the line through `L=12` and `L=16`.
- The leave-one-out slopes are unstable because a three-point window reduces to
  two-point fits after omission.
- This window should be used as a curvature diagnostic, not as a standalone
  asymptotic estimate.

## Conclusions

- The current multi-seed large-`L` fit including provisional `L=18` gives
  `2 alpha = 0.725849` and `alpha = 0.362924` using `L=10,12,14,16,18`.
- Excluding the low `L=14` point raises the multi-seed slope diagnostic to
  `2 alpha = 0.725849` and `alpha = 0.362924` for `L=10,12,16,18`, with much
  lower RSS because `L=14` is the main point below the local trend.
- Including the rechecked `L=8` bridge point gives a lower bridge-plus-large
  slope, `2 alpha = 0.695525`, than the `L>=10` multi-seed window. The
  difference is now a real 8-seed diagnostic rather than a historical
  single-point artifact.
- The all-recommended fit is useful as a recorded full-curve summary, but the
  small historical sizes make it a poor standalone asymptotic estimate.
- The next analysis step should focus on `L=18` seed-level stability,
  especially seed 5 and seed 7, before treating the `L=18` point as a clean
  formal result or using it to justify rerunning `L=10/12`.
