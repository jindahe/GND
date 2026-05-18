# Generative Decoder (Clean Layout)

This directory is a cleaned and self-contained version of the repository outside
`MI_scaling/`. It preserves the code-capacity decoding workflow and the
syndrome-only mutual-information (MI) pipeline, while removing experimental or
duplicated files that were not part of the main training or inference path.

The current recommended workflow is:

1. generate or reuse a toric-code instance
2. build `AB` and `BA` syndrome datasets for a spatial bipartition
3. train one autoregressive model per order
4. evaluate `I_q(A;B)` from the two checkpoints
5. aggregate multiple `L` values into an `MI vs L` summary

## Kept Scope

- `code/`: pre-generated code instances
- `module/`: core code definitions, GF(2) algebra, error model, MADE / NADE / TraDE
- `decoding/code_generator.py`: generate and save code instances
- `decoding/training.py`: train MADE / NADE / TraDE decoders
- `decoding/forward_decoding.py`: load a trained decoder and evaluate logical error rate
- `decoding/mwpm.py`: MWPM baseline
- `decoding/bposd.py`: BP+OSD baseline

## Removed From This Clean Copy

- duplicated / older model implementations: `module/model.py`, `module/net.py`, `module/graph_generator.py`
- one-off or experimental scripts: `decoding/Block_training.py`, `decoding/Ctrain.py`
- circuit-level and hard-coded simulation scripts: `decoding/cir_*.py`, `decoding/rep_cir.py`
- timing-only helper: `decoding/time.py`
- QCC circuit benchmarking helpers: `module/benchmarkqcc.py`, `module/qcc_circuit.py`

## Codex And GPU Usage

When this repository is used from Codex, there are effectively two execution
modes:

- the default sandboxed shell
- the project GPU wrapper at `scripts/run_codex_gpu.sh`

This distinction matters because `torch.cuda.is_available()` can be `False`
inside the default sandbox even when the host machine does have a usable GPU.

### How To Judge Whether GPU Is Actually Usable

First check the sandboxed environment:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

If that prints `False` and `0`, do not conclude that the host has no GPU yet.
Under Codex, this often only means the current shell cannot see the host CUDA
device.

Then check through the project wrapper:

```bash
./scripts/run_codex_gpu.sh "python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count() if torch.cuda.is_available() else 0)'"
```

Interpret the result as follows:

- wrapper prints `True` and a positive device count: use `-device cuda:0` for training and MI evaluation
- wrapper still prints `False` or `0`: stay on CPU for this session
- sandbox prints `False` but wrapper prints `True`: this is the expected Codex case; the host GPU is usable, but only through the wrapper

### How To Run GPU Commands From Codex

The wrapper runs `codex exec --sandbox danger-full-access` so PyTorch can reach
the host CUDA device:

```bash
./scripts/run_codex_gpu.sh "python3 decoding/run_mi_scale_sweep.py --l-values 4 6 --device cuda:0 ..."
```

For long-running formal experiments, prefer the wrapper and pass the training
configuration explicitly. Do not rely on script defaults to imply a formal GPU
configuration.

## Typical Usage

Generate a surface code:

```bash
cd generative_decoder/decoding
python code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
```

Train a MADE decoder:

```bash
cd generative_decoder/decoding
python training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cpu -batch 1000 -epoch 1000 -depth 3 -width 20
```

Run forward decoding with a saved checkpoint:

```bash
cd generative_decoder/decoding
python forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 1000 -er 0.189
```

Check MWPM or BP+OSD baselines:

```bash
cd generative_decoder/decoding
python mwpm.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -trials 1000
python bposd.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -trials 1000
```

Run a GPU-dependent command through Codex with the project wrapper:

```bash
./scripts/run_codex_gpu.sh "python3 -c 'import torch; print(torch.cuda.is_available())'"
```

## Training Workflow

This section documents the recommended syndrome-only MI path for toric-code
experiments. The target quantity is the model-distribution mutual information
`I_q(A;B)` for a spatial bipartition of syndrome bits.

### Single-L Training

Example: `L=4`, toric code, MADE, CPU.

```bash
python decoding/code_generator.py -c_type tor -n 32 -d 4 -k 2 -seed 0
python decoding/syndrome_dataset.py -c_type tor -n 32 -d 4 -k 2 -seed 0 -e_model dep -er 0.05 -n_train 10000 -n_val 2000 -n_test 2000 -partition_axis x -partition_order AB -dataset_dir net/syndrome_data
python decoding/syndrome_dataset.py -c_type tor -n 32 -d 4 -k 2 -seed 0 -e_model dep -er 0.05 -n_train 10000 -n_val 2000 -n_test 2000 -partition_axis x -partition_order BA -dataset_dir net/syndrome_data
python decoding/train_mi_syndrome.py -c_type tor -n 32 -d 4 -k 2 -seed 0 -e_model dep -er 0.05 -n_type made -partition_axis x -partition_order AB -device cpu -width 64 -lr_decay_factor 0.5 -lr_decay_patience 5 -min_lr 0.0002 -early_stop_patience 20 -save True -save_dir net/syndrome_models
python decoding/train_mi_syndrome.py -c_type tor -n 32 -d 4 -k 2 -seed 0 -e_model dep -er 0.05 -n_type made -partition_axis x -partition_order BA -device cpu -width 64 -lr_decay_factor 0.5 -lr_decay_patience 5 -min_lr 0.0002 -early_stop_patience 20 -save True -save_dir net/syndrome_models
python decoding/mi_bipartite.py -c_type tor -n 32 -d 4 -k 2 -seed 0 -e_model dep -er 0.05 -n_type made -partition_axis x -device cpu -save_dir net/syndrome_models -mi_samples 10000 -bootstrap_samples 200 -mi_output_path net/mi_scaling/results/made_tor_n32_d4_k2_seed0_er0.05_dep_xmid.json
```

For formal `MADE` runs, keep the validation-plateau learning-rate schedule
enabled. In the P6/P7/P8 setting with `100` epochs, the older
`StepLR(step_size=2000)` configuration never reduced the learning rate, which
made large-`L` validation NLL flatten early and then drift upward. The formal
GPU runs in this repository should therefore pass at least:

- `-device cuda:0`
- `-width 64`
- `-batch 256`
- `-lr_decay_factor 0.5`
- `-lr_decay_patience 5`
- `-min_lr 0.0002`
- `-early_stop_patience 20`

Artifacts from the single-L flow:

- code instance: `code/tor_n32_d4_k2_seed0`
- datasets: `net/syndrome_data/tor_n32_d4_k2_seed0_er0.05_dep_AB_xmid.pt` and `..._BA_xmid.pt`
- checkpoints: `net/syndrome_models/made_tor_n32_d4_k2_seed0_er0.05_dep_AB_xmid.pt` and `..._BA_xmid.pt`
- training records: `net/syndrome_models/records/made_tor_n32_d4_k2_seed0_er0.05_dep_AB_xmid.json` and `..._BA_xmid.json`
- MI result record: `net/mi_scaling/results/made_tor_n32_d4_k2_seed0_er0.05_dep_xmid.json`

### Multi-L Sweep

For a symmetric left/right cut, even `L` is usually preferred. Odd `L` is still
supported, but the default `cut = L // 2` produces an unbalanced partition.

Run an end-to-end even-`L` scale sweep:

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
  --lr-decay-factor 0.5 \
  --lr-decay-patience 5 \
  --min-lr 0.0002 \
  --early-stop-patience 20 \
  --mi-samples 10000 \
  --bootstrap-samples 200
```

For a formal Codex GPU sweep, run the same command through
`scripts/run_codex_gpu.sh` and keep the explicit training flags. A typical
formal command is:

```bash
./scripts/run_codex_gpu.sh "python3 decoding/run_mi_scale_sweep.py \
  --l-values 4 6 8 10 12 \
  --device cuda:0 \
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
  --mi-samples 20000 \
  --bootstrap-samples 200"
```

Note that `run_mi_scale_sweep.py` has general-purpose defaults such as
`--device cpu` and `--width 32`. Those defaults are useful for small local
checks, but they should not be treated as the formal `P8` configuration.

This pipeline will:

1. generate missing toric code instances under `code/`
2. build `AB/BA` syndrome datasets
3. train `AB/BA` syndrome-only models
4. evaluate `I_q(A;B)` for each `L`
5. write `MI vs L` summaries under `net/mi_scaling/`

The main P8 outputs are:

- per-L MI result records: `net/mi_scaling/results/*.json`
- aggregated table: `net/mi_scaling/mi_vs_L.csv`
- aggregated JSON summary: `net/mi_scaling/mi_vs_L.json`
- plot: `net/mi_scaling/mi_vs_L.png`
- sweep manifest: `net/mi_scaling/sweep_manifest.json`

### Aggregate Existing Results

If you already have per-L MI JSON files and only want to regenerate the table
and plot:

```bash
python decoding/mi_scale_analysis.py \
  --result-path /tmp/gnd_mi_results_d3.json \
  --result-path /tmp/gnd_mi_results_d4.json \
  --output-json net/mi_scaling/mi_vs_L.json \
  --output-csv net/mi_scaling/mi_vs_L.csv \
  --output-plot net/mi_scaling/mi_vs_L.png
```

### Stability Check

To complete the current `P6` stage, run convergence checks over multiple Monte
Carlo sample sizes and repeated evaluation seeds:

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

This writes:

- per-run MI records under `net/mi_stability/results/`
- raw stability table: `net/mi_stability/mi_stability_raw.csv`
- grouped convergence table: `net/mi_stability/mi_stability_grouped.csv`
- grouped summary JSON: `net/mi_stability/mi_stability_summary.json`
- convergence plot: `net/mi_stability/mi_stability.png`

## Experiment Record Standard

Every training or sweep run should leave machine-readable records. The current
scripts now do this automatically.

### Required record files

- `decoding/train_mi_syndrome.py` writes one JSON training record per checkpoint.
- `decoding/mi_bipartite.py` writes one JSON MI result per evaluation when `-mi_output_path` is provided.
- `decoding/mi_stability_analysis.py` writes raw and grouped convergence summaries for repeated evaluations.
- `decoding/mi_scale_analysis.py` writes one JSON summary for the aggregated `MI vs L` result.
- `decoding/run_mi_scale_sweep.py` writes one `sweep_manifest.json` describing the full sweep configuration and output files.

### Record requirements

Each recorded run should preserve at least:

- code identifiers: `c_type`, `n`, `d`, `k`, `seed`
- physical-noise identifiers: `e_model`, `er`
- model identifiers: `n_type`, architecture hyperparameters, `effective_width`, and `parameter_count`
- training hyperparameters: `device`, `dtype`, `epoch`, `batch`, `lr`, LR scheduler settings, and `epochs_trained`
- dataset provenance: dataset path, partition metadata, sample counts
- final metrics: `best_val_nll`, `test_nll`, or `I_q(A;B)` plus bootstrap statistics
- artifact paths: checkpoint path, result JSON path, summary paths

### Practical rule

Do not rely on ad-hoc notebook notes or shell history as the only experiment
record. Use the JSON files produced by the scripts as the source of truth, and
keep them together with the checkpoints and summary artifacts.

## Notes

- `decoding/training.py` writes code-capacity decoder checkpoints under
  `generative_decoder/net/code_capacity/`.
- The syndrome-only MI pipeline writes datasets, checkpoints, and summaries
  under `net/syndrome_models/`, `net/mi_scaling/`, or whichever explicit output
  directories are passed on the command line.
- The clean copy intentionally preserves the original tensor / Pauli representation and CLI flags.
- `training.py` and `forward_decoding.py` include minor fixes versus the original scripts:
  stable project-root path handling, correct `trade` / `nade` branching, and simpler checkpoint handling.
- Environment and GPU setup notes are documented in `ENVIRONMENT.md`.
