#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m py_compile scripts/pair_model_benchmark.py scripts/audit_mi_ordering.py
python3 scripts/pair_model_benchmark.py
python3 scripts/audit_mi_ordering.py

echo "MI_AGENT_AUDITS_PASSED"
