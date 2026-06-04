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
| 8 | 128 | `p8_made_plateau_long_468` | 1 |  | 1.866652 | 0.000000 | 0.044885 |
| 10 | 200 | `p9_largeL_ntrain200k` | 8 | 200000 | 3.689559 | 0.265209 | 0.063953 |
| 12 | 288 | `p9_largeL_ntrain200k` | 8 | 200000 | 5.384724 | 0.321396 | 0.073780 |
| 14 | 392 | `p17_l14_ntrain400k` | 8 | 400000 | 6.074064 | 0.313907 | 0.088458 |
| 16 | 512 | `p16_l16_ntrain400k` | 8 | 400000 | 8.133990 | 0.382952 | 0.094499 |

## Fit Summary

| Window | Points | n_points | 2 alpha | alpha | beta | RSS |
|---|---|---:|---:|---:|---:|---:|
| all recommended | `L=4,6,8,10,12,14,16` | 7 | 0.603889 | 0.301944 | -2.076441 | 1.843343 |
| bridge and large-L | `L=8,10,12,14,16` | 5 | 0.745959 | 0.372980 | -3.921712 | 0.396009 |
| current multi-seed large-L | `L=10,12,14,16` | 4 | 0.701132 | 0.350566 | -3.294127 | 0.315629 |
| without L14 | `L=10,12,16` | 3 | 0.733107 | 0.366553 | -3.549929 | 0.029354 |
| largest three | `L=12,14,16` | 3 | 0.687317 | 0.343658 | -3.091507 | 0.313085 |

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

## Window Details

### All Recommended

Fit window: `L=4,6,8,10,12,14,16`

```text
2 alpha = 0.603888580
alpha   = 0.301944290
beta    = -2.076440982
RSS     = 1.843342527
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.074160 | 0.339113 | 0.735046 | 0.000000 | n/a |
| 6 | 1.513966 | 1.546890 | -0.032925 | 0.000000 | n/a |
| 8 | 1.866652 | 2.754668 | -0.888016 | 0.000000 | n/a |
| 10 | 3.689559 | 3.962445 | -0.272886 | 0.265209 | -1.028945 |
| 12 | 5.384724 | 5.170222 | 0.214502 | 0.321396 | 0.667406 |
| 14 | 6.074064 | 6.377999 | -0.303935 | 0.313907 | -0.968235 |
| 16 | 8.133990 | 7.585776 | 0.548214 | 0.382952 | 1.431548 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.677393 | 0.338697 | -3.007500 | 0.834796 | 0.073505 | -0.931059 |
| 6 | 0.602242 | 0.301121 | -2.053394 | 1.841825 | -0.001646 | 0.023047 |
| 8 | 0.584584 | 0.292292 | -1.728956 | 0.883341 | -0.019305 | 0.347485 |
| 10 | 0.603889 | 0.301944 | -2.030960 | 1.756465 | -0.000000 | 0.045481 |
| 12 | 0.599226 | 0.299613 | -2.067115 | 1.787329 | -0.004663 | 0.009326 |
| 14 | 0.619085 | 0.309543 | -2.167622 | 1.714015 | 0.015197 | -0.091181 |
| 16 | 0.549067 | 0.274534 | -1.674417 | 1.282337 | -0.054821 | 0.402024 |

Interpretation:

- The all-point fit is strongly influenced by finite-size curvature and by the
  historical `L=4/6/8` points.
- The largest normalized residual in the multi-seed subset is `L=16`
  at `+1.431548` seed-std units.
- The historical `L=8` point has a large raw residual, but no train-seed spread
  is available for a normalized diagnostic.

### Bridge And Large-L

Fit window: `L=8,10,12,14,16`

```text
2 alpha = 0.745959103
alpha   = 0.372979552
beta    = -3.921711541
RSS     = 0.396009203
```

Residuals:

| L | observed MI | fitted MI | residual | seed_std | normalized residual |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.866652 | 2.045961 | -0.179310 | 0.000000 | n/a |
| 10 | 3.689559 | 3.537879 | 0.151680 | 0.265209 | 0.571925 |
| 12 | 5.384724 | 5.029798 | 0.354926 | 0.321396 | 1.104325 |
| 14 | 6.074064 | 6.521716 | -0.447652 | 0.313907 | -1.426067 |
| 16 | 8.133990 | 8.013634 | 0.120356 | 0.382952 | 0.314285 |

Leave-one-out sensitivity:

| Omitted L | 2 alpha | alpha | beta | RSS | delta 2 alpha | delta beta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.701132 | 0.350566 | -3.294127 | 0.315629 | -0.044827 | 0.627584 |
| 10 | 0.756793 | 0.378397 | -4.095060 | 0.363142 | 0.010834 | -0.173348 |
| 12 | 0.745959 | 0.372980 | -4.010443 | 0.238544 | 0.000000 | -0.088731 |
| 14 | 0.777934 | 0.388967 | -4.177513 | 0.109734 | 0.031975 | -0.255801 |
| 16 | 0.715870 | 0.357935 | -3.620821 | 0.359795 | -0.030089 | 0.300890 |

Interpretation:

- Removing `L=8` lowers the slope to the current multi-seed large-`L` value,
  which shows that the historical bridge point still matters.
- Removing `L=14` raises `2 alpha` by `0.031975` and reduces RSS to `0.109734`.
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

- The current multi-seed large-`L` fit gives `2 alpha = 0.701132` and
  `alpha = 0.350566` using `L=10,12,14,16`.
- Excluding the low `L=14` point raises the multi-seed slope diagnostic to
  `2 alpha = 0.733107` and `alpha = 0.366553`.
- Including the historical `L=8` bridge point gives a steeper bridge-plus-large
  slope, `2 alpha = 0.745959`, but `L=8` has no train-seed uncertainty and
  should be rechecked before it controls interpretation.
- The all-recommended fit is useful as a recorded full-curve summary, but the
  small historical sizes make it a poor standalone asymptotic estimate.
- The next experiments should prioritize the planned `L=8, n_train=400k`
  recheck and then an `L=18, n_train=400k` pilot. These two results directly
  test the bridge point and the large-size extrapolation that currently limit
  the fit.
