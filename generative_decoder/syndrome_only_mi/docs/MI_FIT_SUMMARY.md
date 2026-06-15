# MI Fit Summary

This document is the lightweight GitHub record for completed syndrome-only toric-code MI runs used to prepare the boundary-law fit.

The fit form is:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1), p = 0.05
```

## Scope

- Final CSV for plotting/fitting: `docs/MI_FIT_POINTS.csv`
- Available completed L values in local records: `4, 6, 8, 10, 12, 14, 16, 18`
- Requested but unavailable L values: `2`
- `L=8` has a completed 8-seed `n_train=400k` recheck and is now included in
  the recommended fit. Its mean is substantially above the previous historical
  single-point recommendation.
- `L=10/12` now use the completed p26 8-seed `n_train=400k` rerun, replacing
  the previous p9 `n_train=200k` bridge rows in the recommended fit.
- `L=14` has a completed 8-seed result and is now included in the recommended
  fit. The p27 same-architecture seed block on seeds 9..16 is clean and has a
  mean consistent with p17, so p27 is kept as a diagnostic row rather than
  replacing the recommended p17 row. The combined 16-seed diagnostic remains
  below the simple `L=12` to `L=16` interpolation.
- `L=18` has a completed 8-seed pilot and is included as a provisional
  recommended fit point. Its cv is at the usable-baseline gate, and seed 5 has
  abnormal training/MI diagnostics. The `p20` seed-5 rerun exactly reproduced
  the failure. The `p21` seed-9 replacement and clean `p22` replacement seeds
  are high-MI, while `p22` seed 11 also shows a `BA` late-NLL training failure.
  The `p24` `depth=1,width=8` architecture pilot removed late-NLL failures on
  seeds 10/11/12/13 and reduced cv to `0.052151`, but the larger predeclared
  `p25` `depth=1,width=8` seed block on seeds 1..8 has no late-NLL failures
  and still gives `cv = 0.095763`. Current evidence favors architecture and
  train-seed sensitivity at `L=18`. The p28 same-architecture `n_train=1000k`
  pilot on seeds 1,2,3,5 failed the late-NLL rule for every seed and had
  `cv = 0.220523`, so increasing `n_train` alone did not resolve the endpoint.
  The p31 same-architecture `n_train=400k`, batch-1024, lr-5e-4, gradclip,
  warmup pilot on seeds 1,2,3,5 had no objective late-NLL failures but still
  failed the spread gate with `cv = 0.120060`; its saved LR histories also
  exposed a warmup bug that reset `ReduceLROnPlateau` reductions back to the
  base LR after warmup.
  The p32 same-scope fixed-LR pilot on seeds 1,2,3,5 has no objective
  late-NLL failures and passes the continuation gate with `cv = 0.034970`;
  the completed 8-seed p32 block has no objective late-NLL failures and
  `cv = 0.055933`. It remains diagnostic because directly substituting only
  the fixed-LR `L=18` endpoint would mix training protocols and lower the
  `L=10,12,14,16,18` fit to `2 alpha = 0.618689`, far below the through-`L=16`
  comparison `2 alpha = 0.744148`.
  Endpoint policy: keep p19 as the sole provisional recommended `L=18` row for
  continuity, but pair endpoint-sensitive conclusions with the through-`L=16`
  window and do not promote any replacement, architecture, or failed data-size
  diagnostic aggregate without a new predeclared endpoint plan.
- Large generated artifacts under `net/` are not part of the lightweight GitHub record.
- Use only rows with `include_in_recommended_fit=yes` for the current recommended fit curve.

## Recommended Fit Points

| L | n | run_id | n_points | train_seeds | n_train | MI | seed_std | mean_bootstrap_std | source_type |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| 4 | 32 | p8_made_plateau_long_468 | 1 |  |  | 1.074160 | 0.000000 | 0.021748 | historical_tracked_summary |
| 6 | 72 | p8_made_plateau_long_468 | 1 |  |  | 1.513966 | 0.000000 | 0.032413 | historical_tracked_summary |
| 8 | 128 | p18_l8_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 2.640582 | 0.132981 | 0.050881 | current_multi_seed_mean |
| 10 | 200 | p26_l10_l12_made_depth0_width64_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 3.557084 | 0.190860 | 0.063400 | current_multi_seed_mean |
| 12 | 288 | p26_l10_l12_made_depth0_width64_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 4.921827 | 0.210903 | 0.073677 | current_multi_seed_mean |
| 14 | 392 | p17_l14_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 6.074064 | 0.313907 | 0.088458 | current_multi_seed_mean |
| 16 | 512 | p16_l16_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 400000 | 8.133990 | 0.382952 | 0.094499 | current_multi_seed_mean |
| 18 | 648 | p19_l18_ntrain400k_pilot | 8 | 1,2,3,4,5,6,7,8 | 400000 | 9.573411 | 0.574411 | 0.104420 | current_multi_seed_mean |

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
| 8 | p8_made_plateau_long_468 | 1 |  | 1.866652 | 0.000000 | 0.044885 | no | previous recommended fit point; source file is deleted in working tree; value recovered from pre-cleanup tracked history; superseded by p18_l8_ntrain400k 8-seed result |
| 8 | p8_made_plateau_longmix_even | 1 |  | 1.866652 | 0.000000 | 0.044885 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 8 | p18_l8_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 2.640582 | 0.132981 | 0.050881 | yes | recommended fit point; clean 8-seed L8 ntrain400k recheck with cv=0.050360; mean is substantially above previous historical L8 point |
| 10 | p8_made_even | 1 |  | 0.986748 | 0.000000 | 0.079282 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_40k_eval | 1 |  | 3.395239 | 0.000000 | 0.059515 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_even | 1 |  | 3.377934 | 0.000000 | 0.082536 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 10 | p8_made_plateau_longmix_even | 1 |  | 3.395239 | 0.000000 | 0.059515 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 10 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 3.689559 | 0.265209 | 0.063953 | no | previous recommended L10 bridge point; superseded by clean p26 ntrain400k rerun |
| 10 | p26_l10_l12_made_depth0_width64_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 3.557084 | 0.190860 | 0.063400 | yes | recommended fit point; clean 8-seed L10 ntrain400k rerun; supersedes p9 L10 ntrain200k bridge row |
| 12 | p8_made_even | 1 |  | 1.280560 | 0.000000 | 0.099664 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_40k_eval | 1 |  | 4.691067 | 0.000000 | 0.069457 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_even | 1 |  | 4.638306 | 0.000000 | 0.098755 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history |
| 12 | p8_made_plateau_longmix_even | 1 |  | 4.691067 | 0.000000 | 0.069457 | no | source file is deleted in working tree; value recovered from pre-cleanup tracked history; duplicates long_468 values for L=4/6/8 and 40k_eval values for L=10/12 |
| 12 | p9_largeL_ntrain200k | 8 | 1,2,3,4,5,6,7,8 | 5.384724 | 0.321396 | 0.073780 | no | previous recommended L12 bridge point; superseded by clean p26 ntrain400k rerun |
| 12 | p26_l10_l12_made_depth0_width64_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 4.921827 | 0.210903 | 0.073677 | yes | recommended fit point; clean 8-seed L12 ntrain400k rerun; supersedes p9 L12 ntrain200k bridge row |
| 14 | p17_l14_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 6.074064 | 0.313907 | 0.088458 | yes | recommended fit point; clean 8-seed L14 result with cv=0.051680 but mean sits below L12-L16 interpolation |
| 14 | p27_l14_made_depth0_width64_seed9to16_ntrain400k | 8 | 9,10,11,12,13,14,15,16 | 6.128160 | 0.418273 | 0.089859 | no | diagnostic L14 stability seed block; clean 8-seed depth0 width64 ntrain400k run; mean is consistent with p17 but seed spread is larger |
| 14 | p17_p27_l14_combined_depth0_width64_ntrain400k | 16 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 | 6.101112 | 0.358342 | 0.089158 | no | combined p17+p27 L14 diagnostic only; keep p17 as recommended row unless an explicit combined-aggregate policy is adopted |
| 16 | p12_l16_ntrain300k | 8 | 1,2,3,4,5,6,7,8 | 8.114293 | 0.514215 | 0.092206 | no | previous recommended fit point; superseded by p16_l16_ntrain400k 8-seed result |
| 16 | p15_l16_ntrain400k | 2 | 1,2 | 8.123028 | 0.470572 | 0.094529 | no | partial 2-seed comparison only; seed 3 incomplete |
| 16 | p16_l16_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 8.133990 | 0.382952 | 0.094499 | yes | recommended fit point; clean 8-seed ntrain400k result with lower seed spread than p12 |
| 18 | p19_l18_ntrain400k_pilot | 8 | 1,2,3,4,5,6,7,8 | 9.573411 | 0.574411 | 0.104420 | yes | provisional recommended fit point; completed 8-seed L18 pilot with cv=0.060001 at the usable-baseline gate; seed 5 BA training showed abnormal late-epoch NLL and seed 5 MI is high |
| 18 | p19_l18_failed_seed5_excluded | 7 | 1,2,3,4,6,7,8 | 9.458227 | 0.510990 | 0.105828 | no | sensitivity aggregate excluding p19 seed 5 due to objective BA late-NLL training failure; not promoted to recommended fit |
| 18 | p19_l18_failed5_high7_excluded | 6 | 1,2,3,4,6,8 | 9.312257 | 0.366541 | 0.106169 | no | robust sensitivity aggregate excluding failed seed 5 and high clean seed 7; diagnostic only because high clean seeds are kept by formal policy |
| 18 | p20_l18_seed5_rerun_ntrain400k | 1 | 5 | 10.379700 | 0.000000 | 0.094562 | no | seed-5 rerun under new run id; exactly reproduces p19 seed-5 MI and BA training failure; diagnostic row only |
| 18 | p21_l18_replace_seed5_ntrain400k | 1 | 9 | 10.953156 | 0.000000 | 0.104264 | no | replacement train seed 9 for failed seed 5; clean AB/BA training but higher MI; diagnostic row only |
| 18 | p21_l18_seed5_replaced_by_seed9 | 8 | 1,2,3,4,9,6,7,8 | 9.645093 | 0.709338 | 0.105633 | no | replacement aggregate excluding failed p19 seed 5 and using p21 seed 9; cv=0.073544 so not promoted to recommended fit |
| 18 | p22_l18_more_replacement_seeds_ntrain400k | 1 | 10 | 10.672577 | 0.000000 | 0.105356 | no | p22 replacement seed 10; clean AB/BA training but high MI; diagnostic row only |
| 18 | p22_l18_more_replacement_seeds_ntrain400k | 1 | 11 | 9.114868 | 0.000000 | 0.103321 | no | p22 replacement seed 11; BA late-NLL training failure reaches 1e3+; diagnostic row only |
| 18 | p22_l18_more_replacement_seeds_ntrain400k | 1 | 12 | 11.237228 | 0.000000 | 0.104360 | no | p22 replacement seed 12; clean AB/BA training but very high MI; diagnostic row only |
| 18 | p22_l18_more_replacement_seeds_ntrain400k | 1 | 13 | 10.789398 | 0.000000 | 0.104906 | no | p22 replacement seed 13; clean AB/BA training but high MI; diagnostic row only |
| 18 | p22_l18_all_seeds | 4 | 10,11,12,13 | 10.453518 | 0.925020 | 0.104486 | no | p22 additional-seed aggregate including failed seed 11; cv=0.088489; diagnostic row only |
| 18 | p22_l18_clean_only | 3 | 10,12,13 | 10.899734 | 0.298058 | 0.104874 | no | p22 clean-only aggregate excluding seed 11 BA late-NLL failure; clean seeds are all high-MI; diagnostic row only |
| 18 | p22_l18_seed5_replaced_by_seed10 | 8 | 1,2,3,4,10,6,7,8 | 9.610021 | 0.638858 | 0.105769 | no | replacement aggregate excluding failed p19 seed 5 and using first clean p22 seed 10 by train-seed order; cv=0.066478 so not promoted |
| 18 | p22_l18_all_clean_p19_p21_p22 | 11 | 1,2,3,4,6,7,8,9,10,12,13 | 9.987268 | 0.844645 | 0.105426 | no | all clean p19/p21/p22 L18 seeds excluding objective training failures p19 seed 5 and p22 seed 11; cv=0.084572 supports heavy-tail interpretation |
| 18 | p24_l18_arch_pilot_depth1_width8_ntrain400k | 4 | 10,11,12,13 | 9.484303 | 0.494612 | 0.109292 | no | depth1 width8 architecture pilot; all seeds pass late-NLL rule; cv=0.052151; diagnostic row only pending larger predeclared seed block |
| 18 | p25_l18_arch_depth1_width8_seeds1to8_ntrain400k | 8 | 1,2,3,4,5,6,7,8 | 9.329877 | 0.893454 | 0.105487 | no | depth1 width8 predeclared p19-comparable seed block; all seeds pass late-NLL rule but cv=0.095763 so not promoted |
| 18 | p28_l18_made_depth0_width64_ntrain1000k_rerun | 4 | 1,2,3,5 | 7.385063 | 1.628577 | 0.098748 | no | ntrain1000k same-architecture pilot; all four seeds fail late-NLL rule in AB and BA; cv=0.220523; stopped without extension |
| 18 | p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot | 4 | 1,2,3,5 | 8.067623 | 0.968598 | 0.101760 | no | batch1024 lr5e-4 gradclip warmup pilot; no objective late-NLL failures but cv=0.120060 and LR warmup bug reset scheduler reductions; diagnostic only |
| 18 | p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot | 4 | 1,2,3,5 | 8.132694 | 0.284404 | 0.104556 | no | fixed warmup/LR scheduler pilot after p31; no objective late-NLL failures and cv=0.034970 passes continuation gate; diagnostic only pending seeds 4,6,7,8 |
| 18 | p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_all8 | 8 | 1,2,3,4,5,6,7,8 | 8.137888 | 0.455173 | 0.103883 | no | fixed warmup/LR scheduler 8-seed block; all seeds pass late-NLL rule and cv=0.055933 but not promoted because endpoint-only protocol substitution would mix training protocols |

## Missing L Values

- `L=2`: no current or tracked historical MI summary was found.

## Interpretation Notes

- `L=4/6` recommended points are historical tracked summaries recovered from pre-cleanup tracked history after deleting heavy/obsolete run folders.
- `L=8` recommended point is the 8-seed mean from `p18_l8_ntrain400k`; it
  supersedes the previous historical single-point `p8_made_plateau_long_468`
  value.
- `L=10/12` recommended points are 8-seed means from
  `p26_l10_l12_made_depth0_width64_ntrain400k`, the clean `n_train=400k`
  rerun under the active MADE `depth=0,width=64` protocol. They supersede the
  previous `p9_largeL_ntrain200k` `n_train=200k` bridge rows.
- `L=14` recommended point is the 8-seed mean from `p17_l14_ntrain400k`.
  The same-architecture p27 diagnostic on seeds 9..16 is clean and has
  `mean(MI) = 6.128160`, `seed_std = 0.418273`, and `cv = 0.068254`.
  Its mean is only `0.054096` above p17, or `0.172331` p17 seed-standard
  deviations. The combined p17+p27 diagnostic has `mean(MI) = 6.101112`,
  `seed_std = 0.358342`, and `cv = 0.058734`. Keep p27 and the combined
  aggregate as diagnostic rows with `include_in_recommended_fit=no`.
- `L=16` recommended point is the 8-seed mean from `p16_l16_ntrain400k`.
- `L=18` recommended point is the provisional 8-seed mean from
  `p19_l18_ntrain400k_pilot`; it should be treated as borderline because
  cv=`0.060001` and seed 5 has abnormal training/MI diagnostics. The dedicated
  diagnostics are recorded in
  `docs/agent_outputs/scaling_runs/2026-06-04_p19_l18_seed_split_diagnostic.md`
  and
  `docs/agent_outputs/scaling_runs/2026-06-04_p20_l18_seed5_rerun_ntrain400k.md`.
  The seed-9 replacement diagnostic is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-04_p21_l18_replace_seed5_ntrain400k.md`.
  The seed-10/11/12/13 replacement diagnostic is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-05_p22_l18_more_replacement_seeds_ntrain400k.md`.
  The `depth=1,width=8` architecture pilot is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-06_p24_l18_arch_pilot_depth1_width8_ntrain400k.md`.
  The larger predeclared `depth=1,width=8` seed block is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-06_p25_l18_arch_depth1_width8_seeds1to8_ntrain400k.md`.
  The p28 same-architecture `n_train=1000k` data-size diagnostic is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-07_p28_l18_made_depth0_width64_ntrain1000k_rerun.md`;
  every pilot seed failed the objective late-NLL rule, so p28 is not a fit
  candidate.
  The p31 same-architecture optimizer diagnostic is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-08_p31_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_clip_pilot.md`;
  it has no objective late-NLL failures but fails the spread gate and revealed
  a warmup/LR-scheduler interaction bug, so it is not a fit candidate.
  The p32 fixed-LR follow-up is recorded in
  `docs/agent_outputs/scaling_runs/2026-06-08_p32_l18_made_depth0_width64_ntrain400k_batch1024_lr5e4_fixedlr_pilot.md`;
  its completed 8-seed block is clean and passes the cv gate, but it is not
  promoted because the fixed-LR change has only been tested at the endpoint.
  The formal seed policy is recorded in `docs/SEED_POLICY.md`: exclude only
  objective training failures, keep clean high-MI seeds, and do not choose
  replacement seeds by observed MI. Under this policy, `p22` supports keeping
  `L=18` provisional rather than promoting a replacement aggregate. `p24`
  supported testing a larger predeclared new-architecture seed block, and
  `p25` shows that the new architecture remains clean but does not stabilize
  the endpoint.
- Endpoint policy for current reporting: keep `p19_l18_ntrain400k_pilot` as
  the single provisional `include_in_recommended_fit=yes` `L=18` row because it
  is the only completed active-architecture 8-seed largest-size aggregate.
  Interpret it alongside the endpoint-stable through-`L=16` window. The
  provisional `L=10,12,14,16,18` window gives `2 alpha = 0.762241`
  (`alpha = 0.381120`), while the through-`L=16` `L=10,12,14,16` window gives
  `2 alpha = 0.744148` (`alpha = 0.372074`). Do not promote
  failed-excluded, replacement, all-clean, or `depth=1,width=8` diagnostic
  aggregates without a new predeclared endpoint policy. Also do not promote
  the p32 fixed-LR `L=18` endpoint alone until same-protocol fixed-LR behavior
  is checked on a neighboring anchor such as `L=16`.
- `L=14` remains notable because its 8-seed mean is lower than the simple
  interpolation between the neighboring `L=12` and `L=16` recommended points,
  though the p26 `L=12` replacement reduces the gap to `-1.445795` L14
  seed-standard deviations. The p27 diagnostic remains below the same
  interpolation by `-0.955714` p27 seed-standard deviations, and the combined
  p17+p27 diagnostic remains below it by `-1.191033` combined seed-standard
  deviations.
- `p15_l16_ntrain400k` currently has only 2 completed seeds and is kept as a comparison row, not the main fit point.
- `p16_l16_ntrain400k` completed a clean 8-seed rerun of the `400k` direction;
  it slightly increases the mean MI relative to `p12` while reducing seed
  spread, so it supersedes `p12_l16_ntrain300k` as the current `L=16`
  recommended point.
- For multi-seed rows, `seed_std` is the across-train-seed standard deviation and should be treated as the main numerical uncertainty indicator.
