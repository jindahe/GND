#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 -m syndrome_only_mi.audits.pair_model_benchmark
python3 -m syndrome_only_mi.audits.audit_mi_ordering

echo "MI_AGENT_AUDITS_PASSED"
