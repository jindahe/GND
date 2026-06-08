#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-net/mi_scaling/p23_l8_arch_ablation_ntrain400k}"
DATASET_ROOT="${DATASET_ROOT:-$RUN_ROOT/datasets}"
L_VALUES="${L_VALUES:-8}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1 2 3}"
N_TRAIN="${N_TRAIN:-400000}"
DEVICE="${DEVICE:-cuda:0}"

run_variant() {
  local label="$1"
  local depth="$2"
  local width="$3"
  local base_root="$RUN_ROOT/$label"

  echo "=== START p23 label=${label} depth=${depth} width=${width} $(date -Is) ==="

  env \
    BASE_ROOT="$base_root" \
    DATASET_DIR="$DATASET_ROOT" \
    MODEL_DIR="$base_root/models" \
    RESULT_DIR="$base_root/results" \
    L_VALUES="$L_VALUES" \
    TRAIN_SEEDS="$TRAIN_SEEDS" \
    N_TRAIN="$N_TRAIN" \
    DEVICE="$DEVICE" \
    DEPTH="$depth" \
    WIDTH="$width" \
    scripts/run_made_mi_ntrain400k.sh

  echo "=== END p23 label=${label} depth=${depth} width=${width} $(date -Is) ==="
}

run_variant baseline 0 64
run_variant narrow_deep_1 1 8
run_variant narrow_deep_2 1 16
run_variant narrow_deep_3 2 8
