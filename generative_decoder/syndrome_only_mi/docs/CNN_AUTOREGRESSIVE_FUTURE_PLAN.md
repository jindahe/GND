# CNN Autoregressive Future Plan For Syndrome-Only MI

This document records a future architecture direction. It is not the active
training protocol for the current recommended MI fit.

The current active training path remains MADE unless `docs/NEXT_STEPS.md`
explicitly says otherwise.

## Motivation

The current MADE models treat the gauge-fixed syndrome as a one-dimensional
sequence. This is convenient for the existing AB/BA entropy decomposition, but
it does not encode the two-dimensional locality of toric-code syndrome data.

At `L=18`, the completed architecture diagnostics show two separate issues:

- `depth=0,width=64` is the historical large-L baseline but can show late-NLL
  instability on some seeds.
- `depth=1,width=8` removes those late-NLL failures in `p24` and `p25`, but
  the larger `p25` seed block still has large train-seed MI spread.

This indicates that lower full-sequence NLL is not sufficient to stabilize the
boundary mutual information endpoint. The MI observable is a difference of
large entropies,

```text
I_q(A;B) = H_q(A) + H_q(B) - H_q(A,B),
```

so small seed-dependent changes in learned spatial correlations can produce a
large MI spread.

## Proposed Future Architecture

Use a locality-aware masked convolutional autoregressive model, similar in
spirit to PixelCNN, while preserving the AB/BA decomposition.

Recommended input representation:

```text
shape: L x L x C
C: syndrome type channels, normally 2 for toric-code CSS syndrome components
```

Recommended model family:

```text
model: masked residual CNN autoregressive model
channels: 32 or 64
residual_blocks: 8 to 12
kernel_size: 3 or 5
dilation_schedule: 1,2,4,8 repeated
activation: gated tanh/sigmoid, SiLU, or GELU
output: Bernoulli logits for gauge-fixed syndrome bits
```

A concrete first implementation target:

```text
n_type: pixelcnn_syndrome
channels: 32
residual_blocks: 8
kernel_size: 3
dilations: 1,2,4,8,1,2,4,8
coordinate_features: yes
partition_order_features: yes
syndrome_type_features: yes
gauge_fixed_valid_mask: yes
```

## Why This Should Scale Better

The current MADE parameter count scales roughly like `O(n_bits^2)` times a
width/depth factor. Since the syndrome length scales like `O(L^2)`, dense MADE
parameter count scales roughly like `O(L^4)`.

For the current implementation at `L=18`, the observed parameter counts are:

```text
depth=0,width=64: 53458438
depth=1,width=8:  33396262
```

A masked residual CNN has parameters controlled mainly by channel count, kernel
size, and number of residual blocks:

```text
O(residual_blocks * kernel_size^2 * channels^2)
```

The activation cost still grows with system area, but the number of trainable
parameters does not explode with `L` in the same way.

This better matches the physics:

- toric-code syndrome correlations are spatial and mostly local at low
  physical error rate;
- the target MI follows a boundary-law scaling form;
- convolution shares local statistical structure across the lattice;
- dilation gives a controlled way to cover several correlation lengths without
  dense all-to-all masked weights.

## Compatibility Requirements

Any CNN autoregressive model must preserve the syndrome-only MI invariants:

- matching AB and BA checkpoints are required;
- AB prefix estimates `H(A)`;
- BA prefix estimates `H(B)`;
- full-sequence likelihood estimates `H(A,B)`;
- partition metadata, code identifiers, error model, physical error rate, and
  seeds must match across paired artifacts.

The implementation must also preserve artifact compatibility with:

```text
syndrome_only_mi/train.py
syndrome_only_mi/bipartite_mi.py
syndrome_only_mi/run_scale_sweep.py
syndrome_only_mi/docs/MI_FIT_POINTS.csv
```

## Validation Plan

Do not start with `L=18`.

Suggested validation ladder:

```text
L=8  -> compare against p18_l8_ntrain400k
L=12 -> compare against current p9 L12 and future 400k rerun
L=16 -> compare against p16_l16_ntrain400k
L=18 -> compare against p19 and p25 diagnostics
```

At each fixed `L`, report:

- mean MI across train seeds;
- sample seed standard deviation;
- coefficient of variation;
- min and max MI;
- mean bootstrap std;
- AB and BA test NLL;
- AB and BA late-NLL max;
- objective failure flags from saved JSON records.

Promotion gates should be at least as strict as the current MADE gates:

```text
usable baseline: cv <= 0.06, no objective late-NLL failures, >=5 train seeds
formal endpoint: cv <= 0.04-0.05, >=8 train seeds, no unexplained mean drift
```

## Status

This is a future direction only. The current near-term scaling work should keep
using MADE with the documented active architecture in `docs/NEXT_STEPS.md`.
