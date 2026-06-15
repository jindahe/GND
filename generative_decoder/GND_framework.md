# Generative Neural Decoder Framework

This document describes the repository implementation of the GND workflow and
the outline/L2M cut-MI extension.

## Variables

For a stabilizer code with `m` syndrome bits and `k` logical qubits, sampled
physical errors are converted into binary configurations:

```text
gamma = syndrome variables, length m
beta  = logical-sector variables, length 2k
alpha = stabilizer/pure-error variables, length m
```

The current GND dataset target is:

```text
[gamma, beta]
```

This matches the GND objective of learning `q_theta(beta, gamma)`. The
`full_config` dataset target preserves `[gamma, beta, alpha]` for diagnostics,
but the main decoder path uses `beta_gamma`.

The dataset layout is saved in every `.pt` artifact:

```json
{
  "gamma": {"start": 0, "stop": m},
  "beta": {"start": m, "stop": m + 2k}
}
```

Downstream cut definitions use this metadata rather than hard-coded slices.

## Training Objective

The GND model learns an autoregressive density:

```text
q_theta(beta, gamma)
```

by minimizing negative log likelihood on samples from the error model. In code,
the sequence is stored as `[gamma, beta]`, so the learned factorization is:

```text
q_theta(gamma, beta)
  = prod_i q_theta(gamma_i | gamma_<i)
    prod_j q_theta(beta_j | gamma, beta_<j)
```

This ordering makes decoder evaluation direct: observed syndrome `gamma` is a
prefix condition, and the model sequentially generates `beta`.

Implemented model families:

- MADE
- NADE
- TraDE binary transformer

Main commands:

```bash
python -m gnd.datasets ...
python -m gnd.train ...
python -m gnd.evaluate_decoder ...
```

## Decoding

Given an observed syndrome `gamma`, decoding proceeds autoregressively:

```text
beta_hat_i = argmax_beta_i q_theta(beta_i | gamma, beta_hat_<i)
```

The predicted logical-sector bits are converted to a logical correction and
combined with the pure error determined by the syndrome. Logical failure is
measured by checking commutation of the recovered error with logical operators.

## Outline Cut MI

`outline.md` asks for CMI, but the formulas are ordinary bipartite mutual
information. This repository therefore implements the requested quantities as
plain `I(A:B)`:

- `middle`: `I(beta : gamma)`
- `quarter`: `I(beta_1 : beta_2, gamma)`
- `three_quarter`: `I(beta, gamma_1 : gamma_2)`

The command is:

```bash
python -m gnd.evaluate_cut_mi ...
```

It supports two source types:

- true samples from a saved GND dataset, interpreted as samples from `p`
- generated samples from a trained checkpoint, interpreted as samples from `q`

The current estimator is an empirical discrete plug-in estimator:

```text
I(A:B) = H(A) + H(B) - H(A,B)
```

It reports the number of unique empirical states for `A`, `B`, and `AB`.
Because this estimator is biased in high-dimensional sparse regimes, formal
large-`L` results should predeclare sample sizes and use the same estimator for
all compared rows.

## L2M And `n_d^min(L)`

L2M motivates comparing the growth of data mutual information against model
history-state or capacity growth. The repository records model capacity in
model-sample MI results and provides a first aggregation tool:

```bash
python -m gnd.sweep_nd_min ...
```

Current capacity key:

```text
parameter_count
```

Other architecture-specific keys may be added later, such as MADE width,
NADE hidden dimension, or transformer history-state proxies. A model is counted
as satisfying a cut at size `L` when:

```text
abs(I_q - I_p) / max(abs(I_p), 1e-12) <= relative_tolerance
```

The smallest passing capacity is reported as `n_d^min(L)` for that cut.

This is a framework-level aggregation rule, not a final physics claim. Formal
studies should define the architecture family, train budget, seeds, estimator,
and tolerance before launching sweeps.

## Exact True MI

For small codes, `python -m gnd.exact_mi` computes the true cut MI exactly by
enumerating all Pauli errors, accumulating the exact distribution of
`(gamma,beta)`, and applying ordinary bipartite MI `H(A)+H(B)-H(A,B)` for the
requested cut. This is not a Monte Carlo estimator and has no sampling bias,
but its cost is exponential: `4^n` error strings. The CLI has a
`--max-exact-errors` guard and should be used as a small-code reference for the
sampled GND workflow.

## Archived Syndrome-Only MI

The old toric syndrome-only AB/BA workflow now lives in `syndrome_only_mi/`.
It remains available for reproducing historical boundary-law runs and for the
ordering audit:

```bash
syndrome_only_mi/scripts/run_mi_agent_audits.sh
```
