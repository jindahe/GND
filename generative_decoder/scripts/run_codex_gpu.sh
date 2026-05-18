#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  cat <<'EOF'
Usage:
  scripts/run_codex_gpu.sh "python3 -c 'import torch; print(torch.cuda.is_available())'"
  scripts/run_codex_gpu.sh python3 -c 'import torch; print(torch.cuda.is_available())'

This wrapper runs `codex exec` with `--sandbox danger-full-access` so CUDA
workloads can access the host GPU.
EOF
  exit 1
fi

if [ "$#" -eq 1 ]; then
  exec codex exec --sandbox danger-full-access "$1"
fi

exec codex exec --sandbox danger-full-access "$*"
