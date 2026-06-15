# Extensible Cut MI Plan For GND

This note defines the implementation plan for computing mutual information
across different cuts in the current GND workflow. The target distribution is
always the GND distribution over logical and syndrome variables:

```text
p(beta, gamma) = p(beta_1, beta_2, gamma_1, gamma_2)
```

The quantity currently called CMI in `outline.md` is implemented as ordinary
bipartite mutual information unless a future design adds an explicit
conditioning variable:

```text
I(A:B) = H(A) + H(B) - H(A,B)
```

This document treats MI as a GND `[gamma,beta]` problem.

## Current State

The existing implementation has three useful pieces:

- `gnd/datasets.py` stores samples in `[gamma,beta]` order and records a layout:
  `layout["gamma"] = [0, m)` and `layout["beta"] = [m, m + 2k)`.
- `gnd/partitions.py` hard-codes the three outline cuts:
  `middle`, `quarter`, and `three_quarter`.
- `gnd/evaluate_cut_mi.py` and `gnd/exact_mi.py` share the same cut records and
  compute `H(A) + H(B) - H(AB)` from either samples or exact small-code
  enumeration.

The hard-coded cut names are enough for the first smoke tests, but they are not
the right long-term interface. Scaling studies need to sweep cut families,
positions, ratios, and possibly conditioned variants without adding a new
Python branch for every geometry.

## Target Abstraction

Separate cut selection from entropy estimation.

The cut layer should only answer:

```python
CutSpec -> CutRecord(a_indices, b_indices, optional c_indices, metadata)
```

The estimator layer should only answer:

```python
CutRecord + samples_or_distribution -> MI/CMI record
```

This keeps all backends consistent. Empirical samples, model samples, exact
enumeration, and future factor-graph or neural entropy backends should consume
the same `CutRecord` schema.

## Cut Schema

Use a stable JSON-serializable schema for every cut:

```json
{
  "name": "middle",
  "quantity": "mi",
  "description": "I(beta : gamma)",
  "a_indices": [13, 14],
  "b_indices": [0, 1, 2],
  "c_indices": [],
  "layout_order": "gamma_beta",
  "metadata": {
    "family": "outline",
    "beta_split": null,
    "gamma_split": null,
    "ratio": null
  }
}
```

Required rules:

- `a_indices`, `b_indices`, and `c_indices` must be pairwise disjoint.
- Every index must be inside the dataset width.
- `quantity="mi"` means `I(A:B)`.
- `quantity="cmi"` is reserved for future `I(A:C|B)` and must require
  non-empty `a_indices`, `b_indices`, and `c_indices`.
- `description` is display-only; downstream logic must use indices.
- `metadata` records cut family, split positions, ratios, and geometry labels.

## Built-In Cut Families

### 1. Outline Cuts

These reproduce `outline.md` and remain the default for `--cut all`:

```text
middle:        A = beta,            B = gamma
quarter:       A = beta_1,          B = beta_2 + gamma
three_quarter: A = beta + gamma_1,  B = gamma_2
```

Generalization:

- `quarter` should accept `--beta-cut-index` or `--beta-cut-ratio`.
- `three_quarter` should accept `--gamma-cut-index` or `--gamma-cut-ratio`.
- The default split remains half of the corresponding variable block.

### 2. Block-To-Block Cuts

These are useful diagnostics for checking learned dependencies:

```text
beta_vs_gamma:     A = beta,        B = gamma
beta_prefix_rest:  A = beta[:i],    B = beta[i:] + gamma
gamma_prefix_rest: A = gamma[:i],   B = gamma[i:] + beta
gamma_window_rest: A = gamma[l:r],  B = all variables outside A
```

These cuts are variable-order cuts, not physical-space cuts. They are valid for
any GND dataset because they rely only on the recorded `[gamma,beta]` layout.

### 3. Named Index-Set Cuts

For experiments that need custom cuts, allow explicit index sets:

```json
{
  "name": "custom_gamma_window",
  "quantity": "mi",
  "a": {"indices": [0, 1, 2, 3]},
  "b": {"indices": [4, 5, 13, 14]}
}
```

The CLI should accept a path:

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/example.pt \
  --cut-spec docs/cuts/custom_gamma_window.json \
  --samples 10000 \
  --output-path net/gnd/results/custom_gamma_window.json
```

Custom specs make it possible to prototype new cuts without editing
`gnd/partitions.py`.

### 4. Future Physical Geometry Cuts

If a code generator records geometry metadata mapping stabilizers/logicals to
coordinates, add a resolver that converts geometric regions into variable
indices:

```python
GeometryCutSpec + dataset_meta + code_meta -> CutRecord
```

This must be a separate resolver. The existing outline cuts should not be
reinterpreted as physical qubit, plaquette, or real-space regions.

## Estimator Backends

### Empirical Plug-In

Current backend in `gnd/evaluate_cut_mi.py`.

Input:

- held-out true samples from a dataset, or
- samples generated from a trained model.

Output:

- `H_A`, `H_B`, `H_AB`, `I(A:B)`;
- unique state counts;
- bootstrap statistics when requested.

Use this for smoke tests and controlled comparisons. Record that the estimator
is biased in sparse high-dimensional regimes.

### Exact Error Enumeration

Current backend in `gnd/exact_mi.py`.

Input:

- code parameters, error model, physical error rate, and a `CutRecord`.

Method:

1. enumerate all Pauli errors;
2. map each error to `(gamma,beta)` using the same convention as
   `gnd/datasets.py`;
3. accumulate the exact joint distribution;
4. compute the requested MI terms from exact marginals.

Use this as the reference backend for small codes. The cost is `4^n`, so the
existing `--max-exact-errors` guard should remain.

### Future Distribution-Aware Backend

For larger systems, a scalable true-distribution backend should be derived
directly for `p(beta,gamma)` rather than for a reduced variable set.

Required derivation:

- keep the same `beta` convention as `gnd/datasets.py`;
- express `p(beta_A, gamma_B, ...)` as constrained sums over physical errors;
- add logical commutation constraints to the factor graph;
- support the same `CutRecord` API as empirical and exact backends;
- validate against `gnd/exact_mi.py` on small codes before using it in scaling
  claims.

This backend may use factor graphs, tensor networks, or other structured
probability methods, but its public interface should remain `CutRecord` based.

## CLI Plan

Extend the current CLIs in three small steps.

Step 1: cut resolver options.

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/example.pt \
  --cut outline:quarter \
  --beta-cut-ratio 0.5
```

Step 2: custom cut specs.

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/example.pt \
  --cut-spec docs/cuts/custom.json
```

Step 3: shared cut resolution in exact enumeration.

```bash
python -m gnd.exact_mi \
  --c-type rep --n 3 --d 3 --k 1 --seed 0 \
  --e-model dep --er 0.05 \
  --cut-spec docs/cuts/custom.json
```

Both CLIs should write the fully resolved `CutRecord` into the output JSON, not
only the user-facing cut name.

## Suggested Module Shape

Keep the public surface small:

```text
gnd/cuts.py
```

Responsibilities:

- define `CutRecord` validation;
- resolve built-in cut families from layout;
- load and validate custom JSON specs;
- expose `all_outline_cuts(layout)` for backward compatibility.

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

`gnd/partitions.py` can either become a compatibility wrapper around
`gnd/cuts.py` or be replaced once all callers are migrated.

## Test Plan

Add focused tests before using new cuts in formal runs:

- dataset layout test: saved `beta_gamma` datasets are in `[gamma,beta]` order;
- resolver tests: `middle`, `quarter`, and `three_quarter` match the current
  hard-coded indices;
- validation tests: overlapping, out-of-range, empty, and duplicate indices are
  rejected;
- custom spec test: JSON index sets resolve to the expected `CutRecord`;
- estimator invariance test: empirical and exact backends agree on a tiny code
  within sampling error when empirical sample count is large;
- CLI smoke tests: both `gnd.evaluate_cut_mi` and `gnd.exact_mi` accept the same
  custom cut spec.

## Immediate Implementation Order

1. Add `gnd/cuts.py` with validation and built-in resolvers.
2. Make `gnd/partitions.py` a compatibility wrapper.
3. Add `--cut-spec`, `--beta-cut-index`, `--beta-cut-ratio`,
   `--gamma-cut-index`, and `--gamma-cut-ratio` to `gnd.evaluate_cut_mi`.
4. Add the same cut options to `gnd.exact_mi`.
5. Add tests for layout, cut resolution, and CLI smoke behavior.
6. Run a small exact-vs-sampled validation on `rep_n3` or another tiny code.

After these steps, new cut families can be added as resolver functions without
touching the entropy estimators.
