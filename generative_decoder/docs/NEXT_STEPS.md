# Next Steps

This file records only unfinished current work for the repository mainline.

## Active Task: Scalable Structured True MI Backend

Current execution target:

```text
build scalable structured true middle-cut backend
-> validate against exact L=2 brute force
-> validate against binary_dense_elimination at L=4
-> implement sparse/compressed transfer for L=10 fixed-gamma pilots
-> only then decide whether L=20 sampled MI is feasible
```

Immediate implementation plan:

1. Add a toric scalable-backend scaffold that reports contraction-plan
   diagnostics before doing expensive contraction:
   - binary X/Z variable layout;
   - row/column frontier width;
   - estimated active state count;
   - exact/truncated flag;
   - reason when a requested `L` is refused.
   Status: implemented as `python -m gnd.plan_true_middle_mi` with the
   `binary_dense` planner. It is a dense-backend dry run, not the final
   transfer backend.
2. Add tests that the planner reports the known calibrated `L=4` width scale
   and refuses dense-style plans for `L=10`/`L=20` unless an explicit scalable
   transfer path is selected.
   Status: implemented in `tests/test_gnd_structured_mi.py`.
3. Implement the first transfer/trellis contraction path behind the existing
   `sector_weights(gamma)` contract.
   Status: implemented as `gnd.partition_backends.toric_row_transfer` for
   small boundary counts using exact dense-character row transfer.
4. Validate the transfer path:
   - L=2 posterior and MI vs brute force / `gnd.exact_mi`;
   - L=4 posterior and entropy vs `binary_dense_elimination` for zero gamma and
     sampled nonzero gamma;
   - L=10 fixed-gamma pilot diagnostics.
   Status: L=2 and L=4 gates passed; L=10 is explicitly refused by the current
   dense-character transfer because it has 1,048,576 boundary states.
5. Next implement sparse/compressed transfer before any L=10 or L=20 sampled MI.
   Keep L=20 disabled until L=10 pilot diagnostics show stable active-state
   growth and acceptable per-gamma runtime.

Current planner diagnostics:

```text
binary_dense planner, toric dep er=0.05:
L=4:  n_physical_bits=64,  max_scope_width=26, max_table_size=67,108,864
L=10: n_physical_bits=400, max_scope_width=72, refused; use transfer/trellis
L=20: n_physical_bits=1600, refused before min-fill; use transfer/trellis

toric_row_transfer dense-character planner:
L=4:  boundary_bits=8,  boundary_states=256, exact dense-character transfer
L=10: boundary_bits=20, boundary_states=1,048,576, refused pending sparse/compressed transfer
```

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
- `gnd.partition_backends.binary_dense_elimination` is the first exact `L=4`
  calibration backend. It represents each Pauli error as binary X/Z bits and
  uses dense NumPy log-tensor elimination, avoiding the sparse Python-dict
  bottleneck in the generic 4-state elimination backend.
- `gnd.partition_backends.toric_row_transfer` is the first toric row-transfer
  backend. It maps saved toric stabilizers by geometry, infers CSS orientation
  from the artifact, and computes exact sector weights by dense-character row
  transfer when the boundary-state count is small enough.
- `gnd.true_middle_mi` estimates
  `I_true(beta:gamma) = H_true(beta) - E_gamma H_true(beta|gamma)`.
- `tests/test_gnd_structured_mi.py` checks the `L=20` `H_true(beta)`
  regression, verifies brute-force sector aggregation against `gnd.exact_mi`
  at `L=2`, verifies binary dense elimination against brute force at `L=2`,
  and runs an exact `L=4` binary dense smoke.

Current calibrated numbers:

```text
toric dep er=0.05, L=20:
H_true(beta) = 2.6445747158582260 nats

existing 10k-sample plug-in middle-cut estimate:
I_plugin(beta:gamma) = 2.6509807221451887 nats

therefore I_plugin - I_true >= 0.0064060062869627 nats
```

The missing large-`L` piece is now a sparse/compressed row-transfer backend for
`log Z_b(gamma), b in {0,1}^{2k}` beyond the exact `L=4` dense-character
calibration path. Do not use dense global elimination or the dense-character
row-transfer path for `L=10` or `L=20`; their state spaces grow too quickly.

Required calibration before any large-`L` run:

- At `L=2`, the structured backend must match `gnd.exact_mi` for
  `I(beta:gamma)` and match the brute-force sector posterior for tested
  `gamma` values.
- At `L=4`, the backend must run without truncation and report contraction
  diagnostics such as elimination order, maximum intermediate scope width, and
  elapsed time per `gamma`.
- If approximate or truncated contraction is added later, those results remain
  diagnostic until matched against the exact `L=2` and `L=4` gates.

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
- `gnd.partition_backends.binary_dense_elimination` passes the `L=2` brute-force
  posterior gate and completes exact `L=4` sector-weight computations. The
  `L=4`, zero-gamma smoke reports:

```text
toric dep er=0.05, L=4:
H_true(beta|gamma=0) = 0.000024038410309570597 nats
max_scope_width = 26 binary variables
max_table_size = 67,108,864 states
```

- A CLI sample smoke also completed:

```text
command backend = binary_dense_elimination
gamma_samples = 4
sample_seed = 123
H_true(beta) = 1.464151961575 nats
mean H_true(beta|gamma) = 0.000885367375 nats
I_true(beta:gamma) = 1.463266594200 nats
MC stderr = 0.000674033698 nats
output = net/gnd/structured_true_mi/binary_dense_l4_sample4_20260622/L4_binary_dense_sample4.json
sector records = net/gnd/structured_true_mi/binary_dense_l4_sample4_20260622/L4_binary_dense_sample4.sectors.jsonl
```

- `gnd.partition_backends.toric_row_transfer` passes the same exact gates:

```text
toric dep er=0.05, L=2:
tested gamma values match brute-force posterior/entropy to <= 1e-10.

toric dep er=0.05, L=4, zero gamma:
H_true(beta|gamma=0) = 0.00002403841029835873 nats
binary_dense reference = 0.000024038410309570597 nats
posterior max abs diff = 6.7e-16
transfer diagnostics: boundary_bits=8, boundary_states=256,
  max_state_count=4096, transfer_mode=dense_character,
  elapsed_seconds ~= 0.22 per gamma on CPU.

toric dep er=0.05, L=4, sample_seed=123, 4 sampled gamma values:
max entropy abs diff vs binary_dense = 1.9e-14 nats
max posterior abs diff vs binary_dense = 1.8e-15
```

- A transfer CLI sample smoke completed and matches the earlier binary-dense
  sample output:

```text
command backend = toric_row_transfer
gamma_samples = 4
sample_seed = 123
H_true(beta) = 1.464151961575 nats
mean H_true(beta|gamma) = 0.000885367375 nats
I_true(beta:gamma) = 1.463266594200 nats
MC stderr = 0.000674033698 nats
output = net/gnd/structured_true_mi/toric_transfer_l4_sample4_20260623/L4_toric_transfer_sample4.json
sector records = net/gnd/structured_true_mi/toric_transfer_l4_sample4_20260623/L4_toric_transfer_sample4.sectors.jsonl
```

- Do not run `L=20` yet. The exact `L=4` calibration gate is now satisfied,
  but current dense-character transfer still has `2^(2L)` boundary states.
  For `L=10` this is 1,048,576 boundary states, so the backend explicitly
  refuses until a sparse/compressed transfer is implemented. `L=20` remains
  disabled until an `L=10` pilot gives acceptable active-state growth and
  runtime diagnostics.

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
   - implement the next sparse/compressed toric row-transfer path over binary
     X/Z error bits and the existing `sector_weights(gamma)` contract;
   - reuse the passed dense-character `L=2`/`L=4` exact gates as calibration;
   - run `L=10` zero-gamma and sampled-gamma pilots only after the sparse path
     reports bounded active-state growth;
   - do not run `L=20` until `L=10` pilots are stable;
   - record boundary width, maximum active state count, elapsed time per gamma,
     and exact/truncated status in every sector record.

6. Use `python -m gnd.exact_mi` on small codes as the reference true-MI
   backend when validating sampled cut-MI estimates. It is exact but costs
   `4^n`, so large `L` experiments need structured sector-partition backends
   rather than direct enumeration.

7. Decide whether `decoding/training.py` should become a wrapper around
   `gnd.train` or remain as a legacy entry point for code-capacity comparisons.
