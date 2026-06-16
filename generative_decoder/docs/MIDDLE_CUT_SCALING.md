# Middle Cut MI Scaling

This note records the current implementation and first toric-code check for
the outline middle cut:

```text
I(beta : gamma)
```

The cut is a variable cut in the GND `[gamma,beta]` representation. It is not a
physical real-space cut.

## Implementation

The scaling entry point is:

```bash
python -m gnd.middle_cut_scaling
```

It builds JSON/CSV records for middle-cut MI across code size `L`, fits simple
diagnostic scaling forms, and checks the information-theoretic upper bound:

```text
I(beta : gamma) <= H(beta) <= beta_dim * ln(2)
```

For toric code, `k = 2` and `beta_dim = 2k = 4`, so:

```text
I(beta : gamma) <= 4 ln(2) = 2.772588722239781 nats
```

This is an `O(1)` upper bound in `L` for the middle cut.

The CLI supports two workflows:

- direct true-distribution estimates from code metadata with `--d-values`;
- aggregation of existing `gnd.evaluate_cut_mi` or `gnd.exact_mi` JSON records
  with repeated `--result`.

Direct sampled example:

```bash
python -m gnd.middle_cut_scaling \
  --d-values 2,3,4,5,6 \
  --c-type tor \
  --k 2 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --backend sample \
  --samples 10000 \
  --bootstrap-samples 30 \
  --output-json net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-6_sample10k.json \
  --output-csv net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-6_sample10k.csv
```

Small-code exact example:

```bash
python -m gnd.middle_cut_scaling \
  --d-values 2 \
  --c-type tor \
  --k 2 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --backend exact
```

## Depolarizing Toric L2-30 Check

On June 16, 2026, after generating the missing toric-code instances through
`L = 30`, the direct sampled middle-cut check was extended to every integer
`L = 2..30`:

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

The output records are:

- `net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.json`
- `net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k.csv`
- `net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_saturating_fit.json`
- `net/gnd/scaling/middle_cut_tor_dep_er0.05_L2-30_sample10k_saturating_fit.png`

The new high-`L` sampled points were:

| L | n | MI (nats) | bootstrap std | margin to `4 ln(2)` |
|---|---:|---:|---:|---:|
| 21 | 882 | 2.6602365811 | 0.0044683077 | 0.1123521411 |
| 22 | 968 | 2.6739955464 | 0.0039328741 | 0.0985931759 |
| 23 | 1058 | 2.6891615197 | 0.0042155742 | 0.0834272025 |
| 24 | 1152 | 2.6953407318 | 0.0034491973 | 0.0772479905 |
| 25 | 1250 | 2.7083893505 | 0.0037477865 | 0.0641993717 |
| 26 | 1352 | 2.7116088348 | 0.0030215826 | 0.0609798875 |
| 27 | 1458 | 2.7244774896 | 0.0030937048 | 0.0481112327 |
| 28 | 1568 | 2.7275322282 | 0.0034745716 | 0.0450564940 |
| 29 | 1682 | 2.7364888588 | 0.0026665932 | 0.0360998634 |
| 30 | 1800 | 2.7420079695 | 0.0032106398 | 0.0305807527 |

All `L = 2..30` sampled points satisfy the logical-sector entropy bound. The
largest observed sampled value was `2.7420079695` nats at `L = 30`, leaving a
margin of `0.0305807527` nats below `4 ln(2)`. The mean bootstrap standard
deviation across the 29 sampled points was `0.0067818170` nats.

The preferred finite-range plot uses the fixed upper-bound form:

```text
I(L) = U - f(L)
U = 4 ln(2) = 2.772588722239781 nats
```

Fitting the sampled gap `f(L) = U - I(L)` over `L = 2..30` gives:

```text
f(L) = 3.3287734274 * exp(-0.2823728616 * L^0.8190)
I(L) = 4 ln(2) - 3.3287734274 * exp(-0.2823728616 * L^0.8190)
rmse = 0.0286098451 nats
```

For comparison, a simpler exponential gap fit,
`f(L) = 2.2824876665 * exp(-0.1433788554 * L)`, has
`rmse = 0.0746171073` nats.

These finite-range fits are descriptive only. The scaling-law validation for
the middle cut is the exact bound
`I(beta:gamma) <= H(beta) <= 4 ln(2)` for fixed toric-code `k = 2`, which
implies `O(1)` middle-cut MI in `L`.

As a small exact calibration, the same command with `--backend exact` at
`L = 2` gave:

```text
I(beta : gamma) = 0.65133395 nats
```

The sampled estimator is a plug-in estimator and remains biased in sparse
high-dimensional regimes. The bounded scaling conclusion for the middle cut
does not depend on the fit: it follows from `I(beta:gamma) <= H(beta)` and fixed
toric-code `k = 2`.
