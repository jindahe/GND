# L=20 Middle-Cut MI Calculation Report

This report summarizes how the current `L=20` middle-cut mutual information
number was computed, what code paths implement it, and how its uncertainty
should be interpreted.

## Scope

The quantity is the GND outline middle cut:

```text
I(beta : gamma)
```

This is a variable cut in the GND `[gamma,beta]` representation. It is not a
physical real-space cut of qubits, edges, plaquettes, or regions.

The current recorded `L=20` value is for:

```text
code family: toric
L / d:       20
n:           800
k:           2
error model: depolarizing
physical er: 0.05
code seed:   0
samples:     10000
estimator:   empirical_discrete_plugin
log unit:    nats
```

## Current Recorded Value

The current sampled record is:

```text
I_plugin(beta:gamma) = 2.6509807221451887 nats
bootstrap std        = 0.004155610292884161 nats
```

The source artifact is:

```text
net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.csv
```

The `L=20` row records:

```text
H_A  = H_hat(beta)       = 2.6509807221451887
H_B  = H_hat(gamma)      = 9.210340371976178
H_AB = H_hat(beta,gamma) = 9.210340371976178
MI   = H_A + H_B - H_AB  = 2.6509807221451887
```

Since `9.210340371976178 = ln(10000)`, the empirical `gamma` and
`(beta,gamma)` states are saturated at the sample count. This is an important
diagnostic: the plug-in estimator is operating in a high-dimensional sparse
regime.

## Command Used For The Scaling Record

The documented scaling run was:

```bash
python -m gnd.middle_cut_scaling \
  --d-values 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
  --c-type tor \
  --k 2 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --backend sample \
  --samples 10000 \
  --bootstrap-samples 30 \
  --output-json net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.json \
  --output-csv net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.csv
```

The command is recorded in `docs/MIDDLE_CUT_SCALING.md`.

## Implementation Path

### 1. Error Samples

`gnd.middle_cut_scaling.sample_middle_row` reads the toric code metadata,
constructs the depolarizing error model, and samples physical Pauli errors:

```text
errors = error_model.generate_error(code.n, m=10000, seed=sample_seed + index)
```

For `L=20`, `code.n = 800`.

### 2. Mapping Errors To `[gamma,beta]`

The sampled physical errors are mapped into GND variables by
`gnd.exact_mi.error_configs`:

```text
gamma = commute(errors, code.g_stabilizer)
beta  = commute(errors, code.logical_opt)
beta  = beta[:, logical_indices(k)]
sample = concat([gamma,beta])
```

For toric `k=2`, `beta` has `2k = 4` binary variables. The `gamma` block is the
stabilizer-commutation syndrome block.

### 3. Middle Cut Resolution

`gnd.partitions.build_cut(layout, "middle")` resolves the middle cut as:

```text
A = beta
B = gamma
```

Equivalently:

```text
a_indices = layout["beta"]
b_indices = layout["gamma"]
```

### 4. Empirical Plug-In MI

`gnd.evaluate_cut_mi.estimate_plugin_mi` computes empirical entropies from the
sample table:

```text
I_hat(A:B) = H_hat(A) + H_hat(B) - H_hat(A,B)
```

It creates integer IDs for unique rows in `A`, `B`, and `[A,B]`, then computes:

```text
H_hat = -sum_x count(x)/N * log(count(x)/N)
```

The recorded estimator name is:

```text
empirical_discrete_plugin
```

### 5. Bootstrap

The bootstrap standard deviation is computed by resampling the same `10000`
rows with replacement `30` times and recomputing the same plug-in MI. For L=20:

```text
bootstrap std = 0.004155610292884161 nats
```

This is a finite-sample resampling diagnostic for the plug-in estimator. It is
not a full error bar for the true mutual information.

## Relation To The True Middle MI

The structured true-distribution target is:

```text
I_true(beta:gamma) = H_true(beta) - E_gamma H_true(beta|gamma)
```

The current structured implementation has already computed the exact `L=20`
logical-sector marginal entropy:

```text
H_true(beta) = 2.6445747158582260 nats
```

This is computed by `gnd.beta_distribution` using Walsh/Fourier inversion over
the `2k` logical-sector bits, avoiding enumeration of all physical errors.

Since conditional entropy is nonnegative:

```text
I_true(beta:gamma) <= H_true(beta)
```

Therefore, for the current `L=20` sampled plug-in value:

```text
I_plugin(beta:gamma) - I_true(beta:gamma)
  >= I_plugin(beta:gamma) - H_true(beta)
  = 2.6509807221451887 - 2.6445747158582260
  = 0.0064060062869627 nats
```

So the current plug-in estimate overestimates the true MI by at least:

```text
0.0064060062869627 nats
```

The actual overestimate can be larger if `E_gamma H_true(beta|gamma) > 0`.

## Error Interpretation

There are two distinct uncertainties:

1. Plug-in bootstrap fluctuation:

```text
std = 0.004155610292884161 nats
```

A rough normal-approximation 95% resampling half-width is:

```text
1.96 * std = 0.00814 nats
```

This only describes the stability of the empirical plug-in estimator under
resampling of the same sample size.

2. Systematic plug-in bias relative to true MI:

```text
at least 0.0064060062869627 nats upward at L=20
```

This lower bound follows from the exact `H_true(beta)` calculation and the
nonnegativity of conditional entropy.

Because the high-dimensional `gamma` states are saturated at the sample count,
the bootstrap standard deviation should not be treated as a rigorous true-MI
error bar.

## Current Limitation

The repository does not yet have a completed large-`L` structured backend for:

```text
E_gamma H_true(beta|gamma)
```

Existing exact/structured backends have passed small-code gates:

```text
L=2: exact reference / exhaustive checks
L=4: binary dense elimination and dense-character row transfer calibration
```

However, the current dense-character row-transfer path has `2^(2L)` boundary
states and is explicitly disabled for large sizes. The current plan is:

```text
implement sparse/compressed toric row transfer
-> validate L=10 fixed-gamma and sampled-gamma pilots
-> only then attempt L=20 sampled true middle MI
```

Until that backend is available, the best current statement is:

```text
recorded L=20 sampled plug-in MI:
  2.6509807221451887 nats

bootstrap std of the plug-in estimator:
  0.004155610292884161 nats

exact upper bound from H_true(beta):
  I_true(beta:gamma) <= 2.6445747158582260 nats

minimum known plug-in overestimate:
  0.0064060062869627 nats
```

## Relevant Files

```text
docs/MIDDLE_CUT_SCALING.md
docs/NEXT_STEPS.md
gnd/middle_cut_scaling.py
gnd/evaluate_cut_mi.py
gnd/exact_mi.py
gnd/partitions.py
gnd/beta_distribution.py
gnd/true_middle_mi.py
net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.csv
net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.json
```
