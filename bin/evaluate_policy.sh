#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/runtime.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/runtime.env"
fi

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if [[ "$#" -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash bin/evaluate_policy.sh [evaluator arguments]

Examples:
  bash bin/evaluate_policy.sh --model pi05 --task-suite libero_spatial --output outputs/clean.json
  bash bin/evaluate_policy.sh --model pi05 --task-suite libero_spatial --patch outputs/texture.png \
    --output outputs/attack.json
EOF
  exit 0
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m dmt.rollout.evaluate_policy "$@"
