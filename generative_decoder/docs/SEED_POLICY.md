# Syndrome-Only MI Seed Policy

This policy applies to toric-code syndrome-only MI scaling runs when
aggregating multiple `train_seed` results at fixed `L` and fixed training
configuration.

The goal is to avoid silently biasing an endpoint downward while still
separating objective training failures from clean high-MI outcomes.

## Training Failure Rule

A seed is marked as a training failure if either `AB` or `BA` training has an
objective late-epoch failure signature in the saved training record.

Current failure signature:

```text
late train or validation NLL >= 1e3
```

The late NLL check is applied to the saved JSON records under
`models/records/`, not inferred from the MI value.

For the current `L=18` runs, this rule marks:

| run_id | train_seed | failed order | reason |
|---|---:|---|---|
| `p19_l18_ntrain400k_pilot` | 5 | `BA` | late `BA` NLL reaches `1e3+` |
| `p22_l18_more_replacement_seeds_ntrain400k` | 11 | `BA` | late `BA` NLL reaches `1e3+` |

## Clean High-MI Seeds

Clean-training seeds are kept even when their MI is high.

High MI alone is not a failure criterion. It is evidence about the seed-level
distribution under the current protocol.

For the current `L=18` diagnostics, the following seeds are clean and high:

| run_id | train_seed | MI |
|---|---:|---:|
| `p21_l18_replace_seed5_ntrain400k` | 9 | 10.953156 |
| `p22_l18_more_replacement_seeds_ntrain400k` | 10 | 10.672577 |
| `p22_l18_more_replacement_seeds_ntrain400k` | 12 | 11.237228 |
| `p22_l18_more_replacement_seeds_ntrain400k` | 13 | 10.789398 |

## Aggregation Rules

Always report these scenarios when a fixed-`L` endpoint has a failed seed:

| Scenario | Rule |
|---|---|
| all attempted seeds | Include every completed MI result, including failed-training diagnostic seeds |
| failed-excluded | Exclude only seeds with objective training-failure signatures |
| robust sensitivity | Use a predeclared robust rule, not an observed-MI rule |
| replacement sensitivity | Replace failed seeds by predeclared train-seed order, not by observed MI |

Replacement seeds must be selected by train-seed order or by a predeclared
count. They must not be selected because their observed MI is low.

When choosing a single replacement seed from a completed replacement batch, use:

```text
lowest train_seed among seeds passing the objective training-failure rule
```

For `p22`, this selects seed 10.

## Reporting Rules

For each aggregate, report:

- seed list
- number of seeds
- mean MI
- sample standard deviation across included seeds
- coefficient of variation
- min and max MI
- mean bootstrap std
- which seeds were excluded and why

Use sample standard deviation for `seed_std`.

Do not update a recommended fit row unless the chosen endpoint policy is stated
in the same report or in a linked policy document.

## Current L18 Decision

The current `L=18` endpoint remains provisional.

The `p22` additional seeds show that clean high-MI seeds are common under the
current protocol. Replacing failed seed 5 by the first clean p22 replacement
seed does not stabilize the endpoint to the `cv <= 0.06` usable-baseline gate,
and the all-clean p19/p21/p22 aggregate has `cv = 0.084572199`.

Therefore, do not promote a new `L=18` recommended row in
`docs/MI_FIT_POINTS.csv` at this stage.
