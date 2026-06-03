# Generative Decoder

This repository contains a cleaned code-capacity decoding workflow and a
syndrome-only mutual-information (MI) pipeline for toric-code experiments.

The main syndrome-only MI workflow is:

1. generate or reuse a toric-code instance
2. build `AB` and `BA` syndrome datasets for a spatial bipartition
3. train one autoregressive model per order
4. evaluate `I_q(A;B)` from the two checkpoints
5. aggregate per-`L` results into an `MI vs L` summary

Experiment-specific plans, seed lists, partial-run status, and interpretation
notes belong in `docs/agent_outputs/scaling_runs/` and the JSON records written
next to the artifacts. Keep README as the stable project entry point.

## Current Focus

The active research target is the toric-code syndrome-only boundary-law fit at
fixed physical error rate `p = 0.05`:

```text
I(L) = 2 alpha(p) L + beta(p) + o(1)
```

For the left/right half-system cut on the torus:

- theoretical audits of the `AB/BA` entropy decomposition are complete
- the regression gate is `scripts/run_mi_agent_audits.sh`
- multi-seed large-`L` artifacts live under `net/mi_scaling/`
- run-specific notes live under `docs/agent_outputs/scaling_runs/`
- train-seed spread should be treated as the main numerical uncertainty source

## Fit Data

Completed lightweight MI fit records are kept in:

- `docs/MI_FIT_SUMMARY.md`
- `docs/MI_FIT_POINTS.csv`

Use rows with `include_in_recommended_fit=yes` for the current recommended fit
curve. Generated datasets, checkpoints, per-run result folders, and plots under
`net/` are intentionally excluded from the lightweight GitHub record.

## Repository Layout

- `code/`: pre-generated code instances
- `module/`: core code definitions, GF(2) algebra, error models, MADE / NADE / TraDE
- `decoding/code_generator.py`: generate and save code instances
- `decoding/syndrome_dataset.py`: build syndrome-only datasets
- `decoding/train_mi_syndrome.py`: train syndrome-only autoregressive models
- `decoding/mi_bipartite.py`: evaluate syndrome-only bipartite MI
- `decoding/mi_scale_analysis.py`: aggregate per-`L` MI records
- `decoding/run_mi_scale_sweep.py`: end-to-end code/data/train/evaluate orchestration
- `decoding/training.py`: original code-capacity decoder training path
- `decoding/forward_decoding.py`: load a decoder checkpoint and evaluate logical error rate
- `decoding/mwpm.py`: MWPM baseline
- `decoding/bposd.py`: BP+OSD baseline
- `scripts/`: audit and experiment helper scripts
- `docs/`: derivations, checklists, and experiment notes
- `net/`: generated datasets, checkpoints, result JSON files, and plots

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

Interpretation:

- If the wrapper prints `True`, a positive device count, and `cuda:0`, CUDA is
  usable through the wrapper.
- If the sandbox cannot see CUDA but the wrapper can allocate on CUDA, run CUDA
  jobs through `scripts/run_codex_gpu.sh`.
- If PyTorch can see the GPU but allocation fails, do not start a formal CUDA
  run until the host GPU, driver, or scheduler state is fixed.
- If neither context can see CUDA, use CPU only for small checks.

For long-running experiments, pass training configuration explicitly rather
than relying on script defaults.

## Code-Capacity Usage

Generate a surface-code instance:

```bash
python decoding/code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
```

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

## Syndrome-Only MI Usage

This path estimates model-distribution mutual information `I_q(A;B)` for a
spatial bipartition of syndrome bits.

### Single-L Example

Example: `L=4`, toric code, MADE, CPU.

```bash
python decoding/code_generator.py -c_type tor -n 32 -d 4 -k 2 -seed 0

python decoding/syndrome_dataset.py \
  -c_type tor \
  -n 32 \
  -d 4 \
  -k 2 \
  -seed 0 \
  -e_model dep \
  -er 0.05 \
  -n_train 10000 \
  -n_val 2000 \
  -n_test 2000 \
  -partition_axis x \
  -partition_order AB \
  -dataset_dir net/syndrome_data

python decoding/syndrome_dataset.py \
  -c_type tor \
  -n 32 \
  -d 4 \
  -k 2 \
  -seed 0 \
  -e_model dep \
  -er 0.05 \
  -n_train 10000 \
  -n_val 2000 \
  -n_test 2000 \
  -partition_axis x \
  -partition_order BA \
  -dataset_dir net/syndrome_data

python decoding/train_mi_syndrome.py \
  -c_type tor \
  -n 32 \
  -d 4 \
  -k 2 \
  -seed 0 \
  -e_model dep \
  -er 0.05 \
  -n_type made \
  -partition_axis x \
  -partition_order AB \
  -device cpu \
  -width 64 \
  -lr_decay_factor 0.5 \
  -lr_decay_patience 5 \
  -min_lr 0.0002 \
  -early_stop_patience 20 \
  -save True \
  -save_dir net/syndrome_models

python decoding/train_mi_syndrome.py \
  -c_type tor \
  -n 32 \
  -d 4 \
  -k 2 \
  -seed 0 \
  -e_model dep \
  -er 0.05 \
  -n_type made \
  -partition_axis x \
  -partition_order BA \
  -device cpu \
  -width 64 \
  -lr_decay_factor 0.5 \
  -lr_decay_patience 5 \
  -min_lr 0.0002 \
  -early_stop_patience 20 \
  -save True \
  -save_dir net/syndrome_models

python decoding/mi_bipartite.py \
  -c_type tor \
  -n 32 \
  -d 4 \
  -k 2 \
  -seed 0 \
  -e_model dep \
  -er 0.05 \
  -n_type made \
  -partition_axis x \
  -device cpu \
  -save_dir net/syndrome_models \
  -mi_samples 10000 \
  -bootstrap_samples 200 \
  -mi_output_path net/mi_scaling/results/made_tor_n32_d4_k2_seed0_er0.05_dep_xmid.json
```

Artifacts from this flow:

- code instance: `code/tor_n32_d4_k2_seed0`
- datasets: `net/syndrome_data/tor_n32_d4_k2_seed0_er0.05_dep_AB_xmid.pt` and `..._BA_xmid.pt`
- checkpoints: `net/syndrome_models/made_tor_n32_d4_k2_seed0_er0.05_dep_AB_xmid.pt` and `..._BA_xmid.pt`
- training records: `net/syndrome_models/records/*.json`
- MI result record: `net/mi_scaling/results/*.json`

### Scale Sweep

For a symmetric left/right cut, even `L` values are preferred. Odd `L` values are
supported, but the default `cut = L // 2` gives an unbalanced partition.

Small CPU example:

```bash
python decoding/run_mi_scale_sweep.py \
  --l-values 4 6 8 \
  --device cpu \
  --n-type made \
  --n-train 10000 \
  --n-val 2000 \
  --n-test 2000 \
  --epoch 100 \
  --batch 256 \
  --width 64 \
  --lr 0.001 \
  --lr-decay-factor 0.5 \
  --lr-decay-patience 5 \
  --min-lr 0.0002 \
  --early-stop-patience 20 \
  --mi-samples 10000 \
  --bootstrap-samples 200
```

The sweep script will:

1. generate missing toric-code instances under `code/`
2. build `AB/BA` syndrome datasets
3. train `AB/BA` syndrome-only models
4. evaluate `I_q(A;B)` for each `L`
5. write summaries under the selected `--summary-dir`

`run_mi_scale_sweep.py` has conservative local defaults such as `--device cpu`
and `--width 32`. Formal runs should pass the intended device, architecture,
training, evaluation, and output-directory flags explicitly.

## Theory Notes

For the theoretical interpretation of the fitted quantity, use:

- `docs/toric_syndrome_only_mi_simple_derivation.md`
- `docs/toric_syndrome_only_mi_algebraic_form.md`

Operationally:

- `AB` prefix estimates `H(A)`
- `BA` prefix estimates `H(B)`
- `AB` full-sequence NLL estimates `H(A,B)`
- the fitted constant `beta(p)` is tied to the current independent-generator
  syndrome coordinates and should not be over-interpreted as a topological
  constant from a full-check representation

## Aggregation And Stability

Aggregate existing per-`L` MI JSON files:

```bash
python decoding/mi_scale_analysis.py \
  --result-path path/to/result_L4.json \
  --result-path path/to/result_L6.json \
  --output-json net/mi_scaling/mi_vs_L.json \
  --output-csv net/mi_scaling/mi_vs_L.csv \
  --output-plot net/mi_scaling/mi_vs_L.png
```

Run repeated MI evaluations for convergence checks:

```bash
python decoding/mi_stability_analysis.py \
  --sample-sizes 1000 2000 5000 10000 \
  --eval-seeds 0 1 2 3 4 \
  --c-type tor \
  --n 32 \
  --d 4 \
  --k 2 \
  --seed 0 \
  --e-model dep \
  --er 0.05 \
  --n-type made \
  --device cpu \
  --partition-axis x \
  --save-dir net/mi_scaling/models \
  --bootstrap-samples 200
```

Run-level hyperparameters such as `batch`, `lr`, sample count, seed list, and
output directories are experiment-specific. Record them in JSON artifacts and
matching experiment notes, not as permanent README guidance.

## Experiment Records

The current scripts write machine-readable records:

- `decoding/train_mi_syndrome.py` writes one JSON training record per checkpoint.
- `decoding/mi_bipartite.py` writes one JSON MI result per evaluation when
  `-mi_output_path` is provided.
- `decoding/mi_stability_analysis.py` writes raw and grouped convergence summaries.
- `decoding/mi_scale_analysis.py` writes aggregate `MI vs L` summaries.
- `decoding/run_mi_scale_sweep.py` writes a `sweep_manifest.json` for each sweep.

Each recorded run should preserve:

- code identifiers: `c_type`, `n`, `d`, `k`, `seed`
- physical-noise identifiers: `e_model`, `er`
- model identifiers: `n_type`, architecture hyperparameters, effective width,
  and parameter count
- training hyperparameters: device, dtype, epoch count, batch size, learning
  rate, scheduler settings, and epochs trained
- dataset provenance: dataset path, partition metadata, and sample counts
- final metrics: `best_val_nll`, `test_nll`, or `I_q(A;B)` plus bootstrap statistics
- artifact paths: checkpoint path, result JSON path, summary path, and plot path

Do not rely on ad-hoc notes or shell history as the only experiment record.
Use the JSON files and experiment notes as the source of truth.

## Notes

- `decoding/training.py` writes code-capacity decoder checkpoints under
  `net/code_capacity/`.
- The syndrome-only MI pipeline writes datasets, checkpoints, and summaries
  under `net/syndrome_data/`, `net/syndrome_models/`, `net/mi_scaling/`, or the
  explicit output directories passed on the command line.
- The clean copy preserves the original tensor / Pauli representation and CLI
  flag style.
- `training.py` and `forward_decoding.py` include minor fixes versus the
  original scripts: stable project-root path handling, correct `trade` / `nade`
  branching, and simpler checkpoint handling.
- Environment setup notes are documented in `ENVIRONMENT.md`.
