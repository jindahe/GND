# MI Fit Summary

This document is the lightweight GitHub record for completed syndrome-only toric-code MI runs used to prepare the boundary-law fit.

The fit form is:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

## Scope

- Final CSV for plotting/fitting: `docs/MI_FIT_POINTS.csv`
- Available completed L values in local records: `4, 6, 8, 10, 12, 14, 16`
- Requested but unavailable L values: `2`
- `L=14` has a completed 8-seed result and is now included in the recommended
  fit, but its mean sits below the simple `L=12` to `L=16` interpolation.
- Large generated artifacts under `net/` are not part of the lightweight GitHub record.
- Use only rows with `include_in_recommended_fit=yes` for the current recommended fit curve.

## Recommended Fit Points

| L | n | run_id | n_points | train_seeds | n_train | MI | seed_std | mean_bootstrap_std | source_type |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| 4 | 32 | p8_made_plateau_long_468 | 1 |  |  | 1.074160 | 0.000000 | 0.021748 | historical_tracked_summary |
| 6 | 72 | p8_made_plateau_long_468 | 1 |  |  | 1.513966 | 0.000000 | 0.032413 | historical_tracked_summary |
| 8 | 128 | p8_made_plateau_long_468 | 1 |  |  | 1.866652 | 0.000000 | 0.044885 | historical_tracked_summary |
| 10 | 200 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 200000 | 3.689559 | 0.265209 | 0.063953 | current_multi_seed_mean |
| 12 | 288 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 200000 | 5.384724 | 0.321396 | 0.073780 | current_multi_seed_mean |
| 14 | 392 | p17_l14_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 6.074064 | 0.313907 | 0.088458 | current_multi_seed_mean |
| 16 | 512 | p16_l16_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 8.133990 | 0.382952 | 0.094499 | current_multi_seed_mean |

## Source Aggregates

The CSV keeps every recovered run-level aggregate so older candidates remain auditable, while the recommended fit is selected by a single yes/no column.

| L | run_id | n_points | train_seeds | MI_mean | seed_std | mean_bootstrap_std | include | notes |
|---:|---|---:|---|---:|---:|---:|---|---|
| 4 | p8_made_even | 1 |  | 0.823137 | 0.000000 | 0.030570 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 4 | p8_made_plateau_even | 1 |  | 1.054483 | 0.000000 | 0.030064 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 4 | p8_made_plateau_long_468 | 1 |  | 1.074160 | 0.000000 | 0.021748 | yes | recommended fit point; source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 4 | p8_made_plateau_longmix_even | 1 |  | 1.074160 | 0.000000 | 0.021748 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 6 | p8_made_even | 1 |  | 1.076084 | 0.000000 | 0.044164 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 6 | p8_made_plateau_even | 1 |  | 1.739309 | 0.000000 | 0.041863 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 6 | p8_made_plateau_long_468 | 1 |  | 1.513966 | 0.000000 | 0.032413 | yes | recommended fit point; source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 6 | p8_made_plateau_longmix_even | 1 |  | 1.513966 | 0.000000 | 0.032413 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 8 | l8_retrain_probe | 1 |  | 1.939175 | 0.000000 | 0.045612 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 8 | p8_made_even | 1 |  | 2.772041 | 0.000000 | 0.067042 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 8 | p8_made_plateau_even | 1 |  | 1.765041 | 0.000000 | 0.069670 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 8 | p8_made_plateau_long_468 | 1 |  | 1.866652 | 0.000000 | 0.044885 | yes | recommended fit point; source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 8 | p8_made_plateau_longmix_even | 1 |  | 1.866652 | 0.000000 | 0.044885 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 10 | p8_made_even | 1 |  | 0.986748 | 0.000000 | 0.079282 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_40k_eval | 1 |  | 3.395239 | 0.000000 | 0.059515 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_even | 1 |  | 3.377934 | 0.000000 | 0.082536 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_longmix_even | 1 |  | 3.395239 | 0.000000 | 0.059515 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 10 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 3.689559 | 0.265209 | 0.063953 | yes | recommended fit point |
| 12 | p8_made_even | 1 |  | 1.280560 | 0.000000 | 0.099664 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_40k_eval | 1 |  | 4.691067 | 0.000000 | 0.069457 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_even | 1 |  | 4.638306 | 0.000000 | 0.098755 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_longmix_even | 1 |  | 4.691067 | 0.000000 | 0.069457 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 12 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 5.384724 | 0.321396 | 0.073780 | yes | recommended fit point |
| 14 | p17_l14_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 6.074064 | 0.313907 | 0.088458 | yes | recommended fit point; clean 8-seed L14 result with cv=0.051680 but mean sits below L12-L16 interpolation |
| 16 | p12_l16_ntrain300k | 8 | 1,2,3,4,5,6,7,8 | 8.114293 | 0.514215 | 0.092206 | no | previous recommended fit point; superseded by p16_l16_ntrain400k 8-seed result |
| 16 | p15_l16_ntrain400k | 2 | 1,2 | 8.123028 | 0.470572 | 0.094529 | no | partial 2-seed comparison only; seed 3 incomplete |
| 16 | p16_l16_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 8.133990 | 0.382952 | 0.094499 | yes | recommended fit point; clean 8-seed ntrain400k result with lower seed spread than p12 |

## Missing L Values

- `L=2`: no current or tracked historical MI summary was found.

## Interpretation Notes

- `L=4/6/8` recommended points are historical tracked summaries recovered from pre-cleanup tracked history after deleting heavy/obsolete run folders.
- `L=10/12` recommended points are 8-seed means from `p9_largeL_ntrain200k`.
- `L=14` recommended point is the 8-seed mean from `p17_l14_ntrain400k`.
- `L=16` recommended point is the 8-seed mean from `p16_l16_ntrain400k`.
- `L=14` remains notable because its 8-seed mean is lower than the simple
  interpolation between the neighboring `L=12` and `L=16` recommended points.
- `p15_l16_ntrain400k` currently has only 2 completed seeds and is kept as a comparison row, not the main fit point.
- `p16_l16_ntrain400k` completed a clean 8-seed rerun of the `400k` direction;
  it slightly increases the mean MI relative to `p12` while reducing seed
  spread, so it supersedes `p12_l16_ntrain300k` as the current `L=16`
  recommended point.
- For multi-seed rows, `seed_std` is the across-train-seed standard deviation and should be treated as the main numerical uncertainty indicator.
