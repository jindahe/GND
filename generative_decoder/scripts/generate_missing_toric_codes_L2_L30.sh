#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

START_L="${START_L:-2}"
END_L="${END_L:-30}"
K="${K:-2}"
SEED="${SEED:-0}"
LOG_DIR="${LOG_DIR:-logs}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"

echo "repo: $ROOT_DIR"
echo "range: L=${START_L}..${END_L}"
echo "k: $K"
echo "seed: $SEED"
echo "dry_run: $DRY_RUN"
echo

for ((L = START_L; L <= END_L; L++)); do
  n=$((2 * L * L))
  artifact="code/tor_n${n}_d${L}_k${K}_seed${SEED}"
  log_path="${LOG_DIR}/generate_tor_L${L}_n${n}_k${K}_seed${SEED}.log"

  if [[ -f "$artifact" ]]; then
    echo "skip L=${L}: exists ${artifact}"
    continue
  fi

  cmd=(python decoding/code_generator.py -c_type tor -n "$n" -d "$L" -k "$K" -seed "$SEED")
  echo "generate L=${L} n=${n}: ${cmd[*]}"
  echo "log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  {
    echo "started_at=$(date -Is)"
    echo "cmd=${cmd[*]}"
    "${cmd[@]}"
    echo "finished_at=$(date -Is)"
  } 2>&1 | tee "$log_path"

  if [[ ! -f "$artifact" ]]; then
    echo "ERROR: expected artifact was not created: ${artifact}" >&2
    exit 1
  fi
done

echo
echo "done"
