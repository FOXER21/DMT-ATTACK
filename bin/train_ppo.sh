#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/runtime.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/runtime.env"
fi
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m dmt.training.train_ppo "$@"
