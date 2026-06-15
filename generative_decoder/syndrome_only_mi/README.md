# Syndrome-Only MI Archive

This folder contains the archived toric-code syndrome-only mutual-information
pipeline. It is kept for reproducibility of the boundary-law scaling runs and
for regression checks of the AB/BA entropy decomposition.

The workflow estimates model-distribution mutual information for syndrome bits:

```text
I_q(A;B) = H_q(A) + H_q(B) - H_q(A,B)
```

For a toric-code spatial cut:

- `AB` prefix estimates `H(A)`
- `BA` prefix estimates `H(B)`
- full-sequence likelihood estimates `H(A,B)`
- matching AB and BA datasets/checkpoints are required

The archive audit wrapper is:

```bash
syndrome_only_mi/scripts/run_mi_agent_audits.sh
```

Expected marker:

```text
MI_AGENT_AUDITS_PASSED
```

## Layout

- `dataset.py`: build syndrome-only datasets
- `train.py`: train syndrome-only autoregressive models
- `bipartite_mi.py`: evaluate `I_q(A;B)` from AB/BA checkpoints
- `scale_analysis.py`: aggregate per-`L` MI JSON files
- `stability_analysis.py`: repeated MI convergence checks
- `run_scale_sweep.py`: end-to-end code/data/train/evaluate orchestration
- `audits/`: exact pair-model and ordering regression checks
- `scripts/`: historical run helpers
- `docs/`: fit records, stability policy, seed policy, and run reports

Archive work should call modules directly with `python -m syndrome_only_mi...`
or use scripts under `syndrome_only_mi/scripts/`.

## Single-L CPU Example

Generate a toric-code instance:

```bash
python decoding/code_generator.py -c_type tor -n 32 -d 4 -k 2 -seed 0
```

Build AB and BA datasets:

```bash
python -m syndrome_only_mi.dataset \
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
  -dataset_dir syndrome_only_mi/net/syndrome_data

python -m syndrome_only_mi.dataset \
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
  -dataset_dir syndrome_only_mi/net/syndrome_data
```

Train AB and BA models:

```bash
python -m syndrome_only_mi.train \
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
  -save True \
  -save_dir syndrome_only_mi/net/syndrome_models

python -m syndrome_only_mi.train \
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
  -save True \
  -save_dir syndrome_only_mi/net/syndrome_models
```

Evaluate MI:

```bash
python -m syndrome_only_mi.bipartite_mi \
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
  -save_dir syndrome_only_mi/net/syndrome_models \
  -mi_samples 10000 \
  -bootstrap_samples 200 \
  -mi_output_path syndrome_only_mi/net/mi_scaling/results/made_tor_n32_d4_k2_seed0_er0.05_dep_xmid.json
```

## Scale Sweep

```bash
python -m syndrome_only_mi.run_scale_sweep \
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
  --mi-samples 10000 \
  --bootstrap-samples 200
```

The sweep script generates missing toric-code instances, builds AB/BA datasets,
trains AB/BA models, evaluates `I_q(A;B)`, and writes summaries under the
selected `--summary-dir`.

## Fit And Stability Records

Historical fit state and policy live here:

- `syndrome_only_mi/docs/MI_FIT_SUMMARY.md`
- `syndrome_only_mi/docs/MI_FIT_POINTS.csv`
- `syndrome_only_mi/docs/MI_FIT_ANALYSIS.md`
- `syndrome_only_mi/docs/SEED_POLICY.md`
- `syndrome_only_mi/docs/STABILITY_CHECKLIST.md`
- `syndrome_only_mi/docs/agent_outputs/scaling_runs/`

Generated datasets, checkpoints, plots, and logs under `syndrome_only_mi/net/`
and `syndrome_only_mi/logs/` are not part of the lightweight archive.
