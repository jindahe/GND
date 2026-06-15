# GND MI Implementation Guide

This document is the concrete implementation guide for GND cut-MI computation,
backend integration, and validation.

## Current Implementation

The current GND path has these components:

- `gnd/datasets.py` stores `beta_gamma` samples in `[gamma,beta]` order and
  records layout metadata.
- `gnd/partitions.py` resolves the current built-in cuts:
  `middle`, `quarter`, and `three_quarter`.
- `gnd/evaluate_cut_mi.py` estimates MI from true dataset samples or generated
  model samples.
- `gnd/exact_mi.py` computes exact true MI for small codes by enumerating
  physical errors.
- `gnd/sweep_nd_min.py` aggregates true/model MI records into `n_d^min(L)`
  tables.

## Cut Record Contract

All backends should consume the same JSON-serializable cut record:

```json
{
  "name": "middle",
  "quantity": "mi",
  "description": "I(beta : gamma)",
  "a_indices": [13, 14],
  "b_indices": [0, 1, 2],
  "layout_order": "gamma_beta",
  "metadata": {
    "family": "outline",
    "beta_split": null,
    "gamma_split": null,
    "ratio": null
  }
}
```

Validation rules:

- `a_indices` and `b_indices` are non-empty.
- `a_indices` and `b_indices` are disjoint.
- No index is duplicated.
- Every index is within the dataset width.
- `quantity` is `mi`.
- `description` is display-only; downstream logic uses indices.
- `metadata` records cut family, split positions, ratios, and geometry labels.

## Built-In Cut Families

Outline cuts:

```text
middle:        A = beta,            B = gamma
quarter:       A = beta_1,          B = beta_2 + gamma
three_quarter: A = beta + gamma_1,  B = gamma_2
```

Useful block diagnostics:

```text
beta_vs_gamma:     A = beta,        B = gamma
beta_prefix_rest:  A = beta[:i],    B = beta[i:] + gamma
gamma_prefix_rest: A = gamma[:i],   B = gamma[i:] + beta
gamma_window_rest: A = gamma[l:r],  B = all variables outside A
```

Custom cuts should be loadable from JSON:

```json
{
  "name": "custom_gamma_window",
  "quantity": "mi",
  "a": {"indices": [0, 1, 2, 3]},
  "b": {"indices": [4, 5, 13, 14]}
}
```

Physical geometry cuts may be added only as resolvers that map geometry and
code metadata into explicit `[gamma,beta]` indices. Existing outline cuts must
not be reinterpreted as physical qubit, plaquette, or real-space regions.

## Backend Outputs

Every MI result should include:

- target distribution label: true samples, model samples, exact enumeration, or
  structured true-distribution backend;
- code and error-model metadata;
- dataset layout order, currently `gamma_beta`;
- fully resolved `a_indices` and `b_indices`;
- entropy terms `H_A`, `H_B`, and `H_AB`;
- final `mi` in nats;
- estimator/backend name;
- sample count, exact support size, or structured-backend accuracy controls;
- bootstrap/statistical uncertainty when applicable.

## Empirical Plug-In Backend

`gnd.evaluate_cut_mi` estimates:

```text
H_hat(A), H_hat(B), H_hat(A,B), I_hat(A:B)
```

from observed discrete states. Input can be:

- held-out true samples from a saved GND dataset;
- generated samples from a trained checkpoint.

This backend is useful for smoke tests and controlled comparisons. It is biased
in sparse high-dimensional regimes, so formal runs must predeclare sample
counts, bootstrap settings, seed policy, and acceptance tolerance.

## Exact Enumeration Backend

`gnd.exact_mi` should:

1. Enumerate physical Pauli errors.
2. Map each error to `(gamma,beta)` using the same convention as
   `gnd/datasets.py`.
3. Accumulate the exact joint distribution.
4. Compute requested MI terms from exact marginals.

Use this as the small-code reference backend. Keep the `--max-exact-errors`
guard because the method scales exponentially in physical error count.

## Structured True-Distribution Backend

For larger systems, derive true MI directly for `p(beta,gamma)`:

- keep the same `beta` convention as `gnd/datasets.py`;
- express the marginals for `H(A)`, `H(B)`, and `H(A,B)` as constrained sums
  over physical errors;
- include logical-sector constraints needed for `beta`, not only syndrome
  constraints for `gamma`;
- support the same `CutRecord` input as empirical and exact backends;
- validate against `gnd.exact_mi` on small codes before using the backend in
  scaling claims.

Possible engines include factor-graph elimination, transfer matrices,
tensor-network contraction, and controlled Monte Carlo entropy estimation.

## CLI Plan

Cut resolver options:

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/example.pt \
  --cut outline:quarter \
  --beta-cut-ratio 0.5
```

Custom cut spec for sample/model MI:

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/example.pt \
  --cut-spec docs/cuts/custom.json
```

The same cut spec for exact enumeration:

```bash
python -m gnd.exact_mi \
  --c-type rep --n 3 --d 3 --k 1 --seed 0 \
  --e-model dep --er 0.05 \
  --cut-spec docs/cuts/custom.json
```

Both CLIs should write the fully resolved `CutRecord` into output JSON.

## Suggested Module Shape

Add a focused cut-resolution module:

```text
gnd/cuts.py
```

Responsibilities:

- define `CutRecord` validation;
- resolve built-in cut families from layout;
- load and validate custom JSON specs;
- expose `all_outline_cuts(layout)` for compatibility.

Suggested functions:

```python
def resolve_cut(layout, spec_or_name, *, beta_cut=None, gamma_cut=None) -> dict:
    ...

def resolve_all_outline_cuts(layout) -> list[dict]:
    ...

def load_cut_spec(path) -> dict:
    ...

def validate_cut_record(cut, *, n_bits) -> dict:
    ...
```

`gnd/partitions.py` can become a compatibility wrapper around `gnd/cuts.py` or
be replaced once all callers are migrated.

## Tests

Add focused tests before formal MI runs:

- saved `beta_gamma` datasets are in `[gamma,beta]` order;
- `middle`, `quarter`, and `three_quarter` match the existing hard-coded
  indices;
- overlapping, out-of-range, empty, and duplicate index sets are rejected;
- custom JSON index-set cuts resolve to the expected `CutRecord`;
- empirical and exact backends agree on a tiny code within sampling error when
  empirical sample count is large;
- `gnd.evaluate_cut_mi` and `gnd.exact_mi` accept the same custom cut spec;
- decoder evaluation consumes a `gamma` prefix and generates exactly `2k`
  `beta` bits.

## Implementation Order

1. Add `gnd/cuts.py` with validation and built-in resolvers.
2. Make `gnd/partitions.py` a compatibility wrapper.
3. Add `--cut-spec`, `--beta-cut-index`, `--beta-cut-ratio`,
   `--gamma-cut-index`, and `--gamma-cut-ratio` to `gnd.evaluate_cut_mi`.
4. Add the same cut options to `gnd.exact_mi`.
5. Add layout, cut-resolution, validation, and CLI smoke tests.
6. Run exact-vs-sampled validation on `rep_n3` or another tiny code.
