# GND Bipartite MI Theory Guide

This document is the theory reference for the current GND outline/L2M workflow.
The target is ordinary bipartite mutual information over GND variables, not a
real-space entropy diagnostic.

## Variables And Distribution

For a stabilizer code, each sampled physical error `e` is mapped to:

```text
gamma(e) = syndrome bits
beta(e)  = logical-sector bits
```

The learned and evaluated distribution is:

```text
p(beta, gamma)
```

Dataset tensors use:

```text
x = [gamma, beta]
```

Every MI cut must first resolve to two disjoint index sets in this vector:

```text
A_indices, B_indices
```

## Bipartite Mutual Information

For two disjoint variable sets `A` and `B`, the true-distribution quantity is:

```text
I_p(A:B) = H_p(A) + H_p(B) - H_p(A,B)
```

For a learned autoregressive model `q_theta(beta,gamma)`, the corresponding
model quantity is:

```text
I_q(A:B) = H_q(A) + H_q(B) - H_q(A,B)
```

True and model comparisons are meaningful only when both values use the same
resolved cut, entropy convention, physical error rate, and estimator/backend
policy.

## Outline Cuts

The current outline cuts are variable cuts in the GND representation:

| Name | Side A | Side B | MI |
|---|---|---|---|
| `middle` | `beta` | `gamma` | `I(beta : gamma)` |
| `quarter` | `beta_1` | `beta_2, gamma` | `I(beta_1 : beta_2, gamma)` |
| `three_quarter` | `beta, gamma_1` | `gamma_2` | `I(beta, gamma_1 : gamma_2)` |

The `beta_1/beta_2` and `gamma_1/gamma_2` splits are deterministic index splits
and must be written into result metadata as resolved indices.

## Exact Small-Code Reference

Small systems can be evaluated exactly:

1. Enumerate all physical Pauli errors `e`.
2. Map each error to `(gamma(e), beta(e))`.
3. Accumulate the exact probability mass for `p(gamma,beta)`.
4. Marginalize to `p(A)`, `p(B)`, and `p(A,B)` for the requested cut.
5. Compute `H(A)`, `H(B)`, `H(A,B)`, and `I(A:B)`.

This backend has no sampling error, but its cost grows with the physical error
space. Its role is calibration: it checks variable conventions, cut indices,
sample estimators, and structured true-distribution backends.

## Scaling MI To Larger L

Large-`L` studies should not build the full table for `p(gamma,beta)`. A single
cut only needs three marginals:

```text
p(A), p(B), p(A,B)
```

Each marginal is a constrained sum over physical errors:

```text
p(y_S) = sum_e p(e) 1[f_S(e) = y_S],
```

where `S` is `A`, `B`, or `A,B`, and `f_S(e)` extracts the selected syndrome and
logical-sector bits. For stabilizer codes, these constraints are sparse GF(2)
parity constraints. This turns marginal computation into partition-function
evaluation on a sparse binary factor graph.

The scalable path is:

- exact enumeration for the smallest `L` values;
- structured marginal computation for `H(A)`, `H(B)`, and `H(A,B)`;
- tensor-network, transfer-matrix, variable-elimination, or factor-graph
  contraction when locality controls contraction width;
- controlled Monte Carlo entropy estimation when exact contraction is too
  expensive, with fixed sample budgets, independent seeds, bootstrap
  uncertainty, and bias checks against smaller exact or structured points;
- model-side MI evaluation on the same resolved cut for
  `q_theta(beta,gamma)`.

The relevant scaling cost is the factor-graph treewidth, tensor-network bond
dimension, contraction boundary width, or sample complexity. Near critical
regimes, correlation length can increase these costs, so scaling claims must
report backend convergence controls and statistical uncertainty.

## `n_d^min(L)` Principle

For fixed `L`, code family, physical error rate, cut, and true-MI backend, a
model passes when:

```text
abs(I_q - I_p) / max(abs(I_p), 1e-12) <= tolerance
```

`n_d^min(L)` is the smallest accepted model capacity satisfying this criterion.
Across `L`, keep the protocol fixed:

- code family and physical error rate;
- cut definition;
- true-MI backend or estimator;
- model architecture family;
- training budget and seed policy;
- capacity metric;
- MI tolerance.

Any change to this protocol starts a new comparison track.
