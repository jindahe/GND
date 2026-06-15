# Generative Decoder

This repository contains a generative neural decoder (GND) workflow for
quantum error-correcting codes, plus an archived syndrome-only mutual
information (MI) scaling workflow for toric-code experiments.

The current mainline follows the GND formulation:

1. generate or reuse a code instance
2. sample physical errors from an error model
3. convert errors into logical variables `beta` and syndrome variables `gamma`
4. train an autoregressive density model `q_theta(beta, gamma)`
5. decode by sequentially generating `beta` conditioned on an observed syndrome
6. compare true-sample and model-sample cut MI for the outline/L2M analysis

`outline.md` uses the term CMI, but in this codebase it is interpreted as
ordinary bipartite mutual information `I(A:B)` unless a future document defines
an explicit conditioning variable.

## Repository Layout

- `gnd/`: current GND data, training, decoding, cut-MI, and `n_d^min(L)` tools
- `module/`: shared code definitions, GF(2) algebra, error models, MADE / NADE / TraDE
- `decoding/code_generator.py`: generate and save code instances
- `decoding/training.py`: legacy code-capacity GND-style trainer
- `decoding/forward_decoding.py`: legacy checkpoint decoder evaluation
- `decoding/mwpm.py`: MWPM baseline
- `decoding/bposd.py`: BP+OSD baseline
- `syndrome_only_mi/`: archived syndrome-only toric MI pipeline, including
  its old scripts, docs, logs, and generated MI artifacts
- `scripts/`: GPU/CUDA wrapper utilities used by the current repo
- `docs/`: current project notes and non-syndrome-only plans
- `net/`: generated datasets, checkpoints, result JSON files, and plots

The old syndrome-only entry points have been moved out of top-level
`decoding/` and `scripts/`. New or resumed syndrome-only work should use
`python -m syndrome_only_mi...` or scripts under `syndrome_only_mi/scripts/`.

## GND Quick Start

Generate a small surface-code instance if it does not already exist:

```bash
python decoding/code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
```

Build a `beta_gamma` GND dataset:

```bash
python -m gnd.datasets \
  --c-type sur \
  --n 13 \
  --d 3 \
  --k 1 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --n-train 10000 \
  --n-val 2000 \
  --n-test 2000 \
  --target beta_gamma \
  --output-dir net/gnd/datasets
```

Train a small MADE density model:

```bash
python -m gnd.train \
  --dataset-path net/gnd/datasets/beta_gamma_sur_n13_d3_k1_seed0_er0.05_dep_ntrain10000.pt \
  --save-dir net/gnd/models \
  --n-type made \
  --depth 0 \
  --width 64 \
  --device cpu \
  --dtype float32 \
  --epoch 100 \
  --batch 256 \
  --lr 0.001 \
  --train-seed 1
```

Evaluate the trained checkpoint as a decoder:

```bash
python -m gnd.evaluate_decoder \
  --checkpoint net/gnd/models/made_beta_gamma_sur_n13_d3_k1_seed0_er0.05_dep_tseed1.pt \
  --trials 1000 \
  --device cpu
```

## Outline Cut MI

`outline.md` defines the current MI objective. The distribution is over GND
variables:

```text
p(beta, gamma) = p(beta_1, beta_2, gamma_1, gamma_2)
```

Here `beta` denotes logical variables and `gamma` denotes syndrome variables
after mapping physical errors through the code. These cuts are **not physical
real-space cuts of qubits or plaquettes**, and they are **not syndrome-only
spatial cuts**. The archived syndrome-only AB/BA workflow is a separate
historical pipeline and should not be used as the meaning of the outline cuts.

Implemented cut definitions:

- `middle`: `I(beta : gamma)`
- `quarter`: `I(beta_1 : beta_2, gamma)`, where `beta_1` is the A side and
  `(beta_2, gamma)` is the B side
- `three_quarter`: `I(beta, gamma_1 : gamma_2)`, where `(beta, gamma_1)` is the
  A side and `gamma_2` is the B side

The first step in `outline.md` is to compute these MI values under the true
probability distribution `p(beta, gamma)` at the selected physical error rate.
The second step is to train neural architectures to learn that same
distribution and use the resulting model MI records for `n_d^min(L)`.

Estimate all three from held-out true samples:

```bash
python -m gnd.evaluate_cut_mi \
  --dataset-path net/gnd/datasets/beta_gamma_sur_n13_d3_k1_seed0_er0.05_dep_ntrain10000.pt \
  --split test \
  --cut all \
  --samples 2000 \
  --bootstrap-samples 200 \
  --output-path net/gnd/results/true_cut_mi_sur_d3.json
```

Estimate all three from a trained model:

```bash
python -m gnd.evaluate_cut_mi \
  --checkpoint net/gnd/models/made_beta_gamma_sur_n13_d3_k1_seed0_er0.05_dep_tseed1.pt \
  --cut all \
  --samples 2000 \
  --bootstrap-samples 200 \
  --device cpu \
  --output-path net/gnd/results/model_cut_mi_sur_d3_made_width64.json
```

For small codes, compute exact true-distribution cut MI by exhaustive
Pauli-error enumeration:

```bash
python -m gnd.exact_mi \
  --c-type rep \
  --n 3 \
  --d 3 \
  --k 1 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --cut all \
  --output-path net/gnd/results/exact_cut_mi_rep_d3.json
```

Exact enumeration sums all `4^n` Pauli errors, maps each error to
`(gamma,beta)`, and evaluates `H(A)+H(B)-H(A,B)` from the resulting exact joint
distribution. The `--max-exact-errors` guard refuses large instances before
they launch. This backend is intended for validation and small-code references
rather than large scaling sweeps; `sur_n13` requires enumerating `67,108,864`
errors and therefore needs an explicit larger guard.

The sampled true/model estimator is an empirical discrete plug-in estimator.
It is useful for smoke tests and controlled small problems, but it is biased in
sparse high-dimensional regimes. Formal `n_d^min(L)` studies should predeclare
sample sizes, capacity metrics, and error thresholds.

Aggregate model MI records into `n_d^min(L)` once true and model MI JSON files
exist:

```bash
python -m gnd.sweep_nd_min \
  --true-result net/gnd/results/true_cut_mi_sur_d3.json \
  --model-result net/gnd/results/model_cut_mi_sur_d3_made_width64.json \
  --relative-tolerance 0.10 \
  --capacity-key parameter_count \
  --output-json net/gnd/nd_min/nd_min.json \
  --output-csv net/gnd/nd_min/nd_min.csv
```

## Legacy Code-Capacity Usage

The legacy scripts remain available while the new `gnd/` package matures.

Train a small MADE decoder:

```bash
python decoding/training.py \
  -save True \
  -n_type made \
  -c_type sur \
  -n 13 \
  -d 3 \
  -k 1 \
  -seed 0 \
  -er 0.189 \
  -device cpu \
  -batch 1000 \
  -epoch 1000 \
  -depth 3 \
  -width 20
```

Run forward decoding:

```bash
python decoding/forward_decoding.py \
  -n_type made \
  -c_type sur \
  -n 13 \
  -d 3 \
  -k 1 \
  -seed 0 \
  -e_model dep \
  -device cpu \
  -trials 1000 \
  -er 0.189
```

Run classical baselines:

```bash
python decoding/mwpm.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -trials 1000
python decoding/bposd.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -trials 1000
```

## Syndrome-Only MI Archive

The previous toric-code syndrome-only boundary-law workflow is now isolated in
`syndrome_only_mi/`. Its audit gate is:

```bash
syndrome_only_mi/scripts/run_mi_agent_audits.sh
```

The required marker is:

```text
MI_AGENT_AUDITS_PASSED
```

Archived fit records and run reports live under:

- `syndrome_only_mi/docs/MI_FIT_SUMMARY.md`
- `syndrome_only_mi/docs/MI_FIT_POINTS.csv`
- `syndrome_only_mi/docs/MI_FIT_ANALYSIS.md`
- `syndrome_only_mi/docs/agent_outputs/scaling_runs/`

See `syndrome_only_mi/README.md` for the old AB/BA workflow and command
examples.

## GPU Usage

Codex sessions may have two different execution contexts:

- the default sandboxed shell
- the project GPU wrapper at `scripts/run_codex_gpu.sh`

The default shell may not see the host CUDA device. Check both visibility and
allocation before starting a CUDA run.

Sandbox visibility check:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

Wrapper visibility and allocation check:

```bash
./scripts/run_codex_gpu.sh "python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); x=torch.empty((1,), device=\"cuda:0\"); print(x.device)'"
```

For long-running experiments, pass training configuration explicitly rather
than relying on script defaults. Use `tmux` for long GPU jobs and keep run ids,
commands, and logs in the relevant run report or `docs/NEXT_STEPS.md`.

## Generated Artifacts

Generated checkpoints, datasets, logs, and plots should stay out of Git. Common
output roots are:

- `net/gnd/datasets/`
- `net/gnd/models/`
- `net/gnd/results/`
- `net/gnd/nd_min/`
- `syndrome_only_mi/net/` for archived syndrome-only MI artifacts
- `logs/`

Environment setup notes are documented in `ENVIRONMENT.md`.
