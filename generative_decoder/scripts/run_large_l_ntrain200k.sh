#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_ROOT="${BASE_ROOT:-net/mi_scaling/p9_largeL_ntrain200k}"
DATASET_DIR="${DATASET_DIR:-$BASE_ROOT/datasets}"
MODEL_DIR="${MODEL_DIR:-$BASE_ROOT/models}"
RESULT_DIR="${RESULT_DIR:-$BASE_ROOT/results}"
L_VALUES=(${L_VALUES:-10 12})
TRAIN_SEEDS=(${TRAIN_SEEDS:-1 2 3 4 5 6 7 8})
N_TRAIN="${N_TRAIN:-200000}"
DEVICE="${DEVICE:-cuda:0}"

for L in "${L_VALUES[@]}"; do
  for TSEED in "${TRAIN_SEEDS[@]}"; do
    SUMMARY_DIR="$BASE_ROOT/L${L}_tseed${TSEED}"
    echo "=== START L=${L} tseed=${TSEED} $(date -Is) ==="
    python3 decoding/run_mi_scale_sweep.py \
      --l-values "$L" \
      --device "$DEVICE" \
      --n-type made \
      --seed 0 \
      --error-seed 51697 \
      --split-seed 0 \
      --train-seed "$TSEED" \
      --er 0.05 \
      --e-model dep \
      --partition-axis x \
      --n-train "$N_TRAIN" \
      --n-val 2000 \
      --n-test 2000 \
      --mi-samples 40000 \
      --bootstrap-samples 200 \
      --chunk-size 1000 \
      --epoch 100 \
      --batch 512 \
      --lr 0.001 \
      --weight-decay 0.0 \
      --lr-decay-factor 0.5 \
      --lr-decay-patience 5 \
      --min-lr 0.0001 \
      --log-every 10 \
      --early-stop-patience 30 \
      --early-stop-min-delta 0.0 \
      --dtype float32 \
      --depth 0 \
      --width 64 \
      --dataset-dir "$DATASET_DIR" \
      --model-dir "$MODEL_DIR" \
      --result-dir "$RESULT_DIR" \
      --summary-dir "$SUMMARY_DIR" \
      --skip-existing
    echo "=== END L=${L} tseed=${TSEED} $(date -Is) ==="
  done
done
