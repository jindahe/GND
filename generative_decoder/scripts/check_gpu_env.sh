#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== HOST ==="
hostname
date -Is

echo "=== GPU / NVIDIA-SMI ==="
nvidia-smi

echo "=== PYTHON / TORCH ==="
python3 - <<'PY'
import sys

import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

for index in range(torch.cuda.device_count()):
    print(f"device {index}: {torch.cuda.get_device_name(index)}")

print("current device:", torch.cuda.current_device())
x = torch.empty((1,), device="cuda:0")
print("allocation:", x.device)

a = torch.randn((2048, 2048), device="cuda:0")
b = torch.randn((2048, 2048), device="cuda:0")
c = a @ b
torch.cuda.synchronize()
print("matmul ok:", float(c[0, 0]))
PY

echo "=== PROJECT IMPORTS ==="
python3 - <<'PY'
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))

import module  # noqa: F401

print("module import ok")
PY
