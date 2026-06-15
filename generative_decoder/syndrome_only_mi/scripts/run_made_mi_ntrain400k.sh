#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_ROOT="${BASE_ROOT:-syndrome_only_mi/net/mi_scaling/made_mi_ntrain400k}"
DATASET_DIR="${DATASET_DIR:-$BASE_ROOT/datasets}"
MODEL_DIR="${MODEL_DIR:-$BASE_ROOT/models}"
RESULT_DIR="${RESULT_DIR:-$BASE_ROOT/results}"

L_VALUES=(${L_VALUES:-16})
TRAIN_SEEDS=(${TRAIN_SEEDS:-1 2 3})
N_TRAIN="${N_TRAIN:-400000}"
DEVICE="${DEVICE:-cuda:0}"
DEPTH="${DEPTH:-0}"
WIDTH="${WIDTH:-64}"
EPOCH="${EPOCH:-100}"
BATCH="${BATCH:-512}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
DIVERGENCE_NLL_THRESHOLD="${DIVERGENCE_NLL_THRESHOLD:-0.0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
LR_DECAY_FACTOR="${LR_DECAY_FACTOR:-0.5}"
LR_DECAY_PATIENCE="${LR_DECAY_PATIENCE:-5}"
MIN_LR="${MIN_LR:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-30}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"

for L in "${L_VALUES[@]}"; do
  for TSEED in "${TRAIN_SEEDS[@]}"; do
    SUMMARY_DIR="$BASE_ROOT/L${L}_tseed${TSEED}"

    echo "=== START made_mi_ntrain400k L=${L} train_seed=${TSEED} $(date -Is) ==="

    python3 -m syndrome_only_mi.run_scale_sweep \
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
      --epoch "$EPOCH" \
      --batch "$BATCH" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --grad-clip-norm "$GRAD_CLIP_NORM" \
      --warmup-steps "$WARMUP_STEPS" \
      --divergence-nll-threshold "$DIVERGENCE_NLL_THRESHOLD" \
      --max-train-steps "$MAX_TRAIN_STEPS" \
      --lr-decay-factor "$LR_DECAY_FACTOR" \
      --lr-decay-patience "$LR_DECAY_PATIENCE" \
      --min-lr "$MIN_LR" \
      --log-every 10 \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
      --dtype float32 \
      --depth "$DEPTH" \
      --width "$WIDTH" \
      --dataset-dir "$DATASET_DIR" \
      --model-dir "$MODEL_DIR" \
      --result-dir "$RESULT_DIR" \
      --summary-dir "$SUMMARY_DIR" \
      --skip-existing

    echo "=== END made_mi_ntrain400k L=${L} train_seed=${TSEED} $(date -Is) ==="
  done
done
