# Next Steps

This file records only unfinished current work for the repository mainline.

## Current Mainline

The active implementation target is the GND plus outline/L2M workflow:

```text
sample errors -> build [gamma,beta] datasets -> train q_theta(beta,gamma)
-> decode beta conditioned on gamma -> compare true/model cut MI
-> aggregate n_d^min(L)
```

`outline.md` uses CMI terminology, but the implemented quantity is ordinary
bipartite mutual information `I(A:B)`.

Implemented outline cuts:

- `middle`: `I(beta : gamma)`
- `quarter`: `I(beta_1 : beta_2, gamma)`
- `three_quarter`: `I(beta, gamma_1 : gamma_2)`

See `docs/ENTROPY_BACKENDS_FOR_MI.md` for the current plan to make these cuts
extensible through a shared `CutRecord` resolver instead of hard-coded branches.

## Immediate Unfinished Work

1. Promote `gnd/` smoke commands into a small formal CPU run:
   - code: `sur_n13_d3_k1_seed0`
   - target: `beta_gamma`
   - model: MADE
   - outputs under a fresh `net/gnd/<run_id>/`
   - record true-sample and model-sample cut MI JSON files.

2. Add an extensible cut layer for GND MI:
   - introduce a shared `CutRecord` schema with `a_indices`, `b_indices`, and
     reserved `c_indices`
   - support outline cuts, variable block cuts, and custom JSON index-set cuts
   - make `gnd.evaluate_cut_mi` and `gnd.exact_mi` consume the same resolved cut
     records.

3. Define the first formal `n_d^min(L)` policy:
   - capacity key, initially `parameter_count`
   - relative MI tolerance
   - sample sizes for true/model MI estimation
   - seeds and training budget
   - allowed architecture grid.

4. Add focused tests for GND layout and cuts:
   - saved dataset layout matches `[gamma,beta]`
   - `middle`, `quarter`, and `three_quarter` cuts use disjoint A/B indices
   - custom cut specs reject overlapping, empty, duplicate, and out-of-range
     index sets
   - decoder evaluation consumes `gamma` prefix and generates exactly `2k`
     beta bits.

5. Use `python -m gnd.exact_mi` on small codes as the reference true-MI
   backend when validating sampled cut-MI estimates. It is exact but costs
   `4^n`, so large `L` experiments still need the sampled estimator.

6. Decide whether `decoding/training.py` should become a wrapper around
   `gnd.train` or remain as a legacy entry point for code-capacity comparisons.
