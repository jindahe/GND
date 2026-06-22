# Next Steps

This file records only unfinished current work for the repository mainline.

## Current Mainline

The active implementation target is the GND plus outline/L2M workflow:

```text
sample errors -> build [gamma,beta] datasets -> train q_theta(beta,gamma)
-> decode beta conditioned on gamma -> compare true/model cut MI
-> aggregate n_d^min(L)
```

The outline quantity is ordinary bipartite mutual information `I(A:B)`.

Implemented outline cuts:

- `middle`: `I(beta : gamma)`
- `quarter`: `I(beta_1 : beta_2, gamma)`
- `three_quarter`: `I(beta, gamma_1 : gamma_2)`

See `docs/THEORY_GUIDE.md` for the theory target and scalable-MI rationale.
See `docs/IMPLEMENTATION.md` for the cut-record schema, backend contract, CLI
plan, and tests.
See `docs/MIDDLE_CUT_SCALING.md` for the implemented middle-cut scaling
summary and the toric depolarizing-noise upper-bound check.

## Structured Middle-Cut True MI Baseline

The first structured true-distribution path for the GND middle cut is now in
place:

- `gnd.beta_distribution` computes exact `p(beta)` and `H_true(beta)` by
  Walsh/Fourier inversion over the `2k` logical-sector bits.
- `gnd.sector_partition` defines the common `log Z_b(gamma) -> p(beta|gamma)`
  interface.
- `gnd.partition_backends.brute_force` is the tiny-code exact reference backend
  for `Z_b(gamma)`.
- `gnd.true_middle_mi` estimates
  `I_true(beta:gamma) = H_true(beta) - E_gamma H_true(beta|gamma)`.
- `tests/test_gnd_structured_mi.py` checks the `L=20` `H_true(beta)`
  regression and verifies that the brute-force sector aggregation matches
  `gnd.exact_mi` at `L=2`.

Current calibrated numbers:

```text
toric dep er=0.05, L=20:
H_true(beta) = 2.6445747158582260 nats

existing 10k-sample plug-in middle-cut estimate:
I_plugin(beta:gamma) = 2.6509807221451887 nats

therefore I_plugin - I_true >= 0.0064060062869627 nats
```

The missing large-`L` piece is a scalable backend for
`log Z_b(gamma), b in {0,1}^{2k}`. The immediate implementation target is an
exact variable-elimination / tensor-contraction sector backend, followed by a
toric transfer-matrix backend if generic elimination has too much intermediate
width.

Required calibration before any large-`L` run:

- At `L=2`, the structured backend must match `gnd.exact_mi` for
  `I(beta:gamma)` and match the brute-force sector posterior for tested
  `gamma` values.
- At `L=4`, the backend must run without truncation and report contraction
  diagnostics such as elimination order, maximum intermediate scope width, and
  elapsed time per `gamma`.
- If approximate or truncated contraction is added later, those results remain
  diagnostic until the exact `L=2` and `L=4` gates pass.

Current elimination status:

- `gnd.partition_backends.elimination` implements exact sparse variable
  elimination and contracts all logical sectors for one `gamma` in a single
  pass.
- `python -m gnd.true_middle_mi --backend elimination --gamma-mode exhaustive`
  matches the `L=2` exact reference:

```text
toric dep er=0.05, L=2:
I_true(beta:gamma) = 0.6513339478450818 nats
exact gamma support size = 64
output = net/gnd/structured_true_mi/elimination_20260622/L2_elimination_exhaustive.json
```

- The generic elimination backend is too wide for the first `L=4` one-gamma
  smoke with the current min-fill heuristic: all-sector contraction reached
  33,554,432 intermediate states and hit
  `--elimination-max-intermediate-states 20000000`.
- Do not run `L=20` yet. Next optimize the contraction plan itself: implement
  a toric row/column transfer-matrix backend or a stronger geometry-aware
  elimination order, then re-run the exact `L=4` gate.

## Immediate Unfinished Work

1. Promote `gnd/` smoke commands into a small formal CPU run:
   - code: `sur_n13_d3_k1_seed0`
   - target: `beta_gamma`
   - model: MADE
   - outputs under a fresh `net/gnd/<run_id>/`
   - record true-sample and model-sample cut MI JSON files.

2. Add an extensible cut layer for GND MI:
   - introduce a shared `CutRecord` schema with `a_indices` and `b_indices`
   - support outline cuts, variable block cuts, and custom JSON index-set cuts
   - make `gnd.evaluate_cut_mi` and `gnd.exact_mi` consume the same resolved cut
     records
   - record the estimator/backend and fully resolved variable indices for every
     GND MI output.

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

5. Extend structured true middle-cut MI to large `L`:
   - implement a toric transfer-matrix backend, or improve elimination with a
     geometry-aware contraction order that keeps `L=4` intermediate width
     below the current 33,554,432-state failure;
   - validate the next backend's sector posterior and MI against
     brute-force / `gnd.exact_mi` at `L=2`;
   - re-run an exact `L=4` structured smoke with contraction diagnostics;
   - only after the `L=2` and `L=4` gates pass, attempt `L=20` pilots or move
     to a toric transfer-matrix backend if generic elimination is too wide.

6. Use `python -m gnd.exact_mi` on small codes as the reference true-MI
   backend when validating sampled cut-MI estimates. It is exact but costs
   `4^n`, so large `L` experiments need structured sector-partition backends
   rather than direct enumeration.

7. Decide whether `decoding/training.py` should become a wrapper around
   `gnd.train` or remain as a legacy entry point for code-capacity comparisons.
