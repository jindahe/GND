# Next Steps

This file records only unfinished current work for the repository mainline.
Archived toric syndrome-only MI plans, seed policy, fit records, and run reports
now live under `syndrome_only_mi/docs/`.

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

## Immediate Unfinished Work

1. Promote `gnd/` smoke commands into a small formal CPU run:
   - code: `sur_n13_d3_k1_seed0`
   - target: `beta_gamma`
   - model: MADE
   - outputs under a fresh `net/gnd/<run_id>/`
   - record true-sample and model-sample cut MI JSON files.

2. Define the first formal `n_d^min(L)` policy:
   - capacity key, initially `parameter_count`
   - relative MI tolerance
   - sample sizes for true/model MI estimation
   - seeds and training budget
   - allowed architecture grid.

3. Add focused tests for GND layout and cuts:
   - saved dataset layout matches `[gamma,beta]`
   - `middle`, `quarter`, and `three_quarter` cuts use disjoint A/B indices
   - decoder evaluation consumes `gamma` prefix and generates exactly `2k`
     beta bits.

4. Use `python -m gnd.exact_mi` on small codes as the reference true-MI
   backend when validating sampled cut-MI estimates. It is exact but costs
   `4^n`, so large `L` experiments still need the sampled estimator.

5. Decide whether `decoding/training.py` should become a wrapper around
   `gnd.train` or remain as a legacy entry point for code-capacity comparisons.

## Archived Syndrome-Only MI

The old syndrome-only boundary-law workflow is isolated under
`syndrome_only_mi/`. Its regression gate remains:

```bash
syndrome_only_mi/scripts/run_mi_agent_audits.sh
```

Required marker:

```text
MI_AGENT_AUDITS_PASSED
```

Do not update old recommended fit rows from the GND mainline. If archived
syndrome-only work resumes, use `syndrome_only_mi/docs/SEED_POLICY.md`,
`syndrome_only_mi/docs/STABILITY_CHECKLIST.md`, and the reports under
`syndrome_only_mi/docs/agent_outputs/scaling_runs/`.
