#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/runtime.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/runtime.env"
fi

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if [[ "${1:-}" == "--full" ]]; then
  shift
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${PYTHON_BIN}" -m dmt.environment.check "$@"
fi

exec "${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = (
    "README.md",
    "pyproject.toml",
    "config/runtime.env.example",
    "bin/train_generator.sh",
    "bin/infer_texture.sh",
    "bin/evaluate_policy.sh",
    "src/dmt/training/train_generator.py",
    "src/dmt/inference/texture.py",
    "src/dmt/rollout/evaluate_policy.py",
    "src/dmt/diffusion/sampler.py",
)
checks = [{"path": item, "ok": (root / item).is_file()} for item in required]
bad_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".so", ".pyc"}
for path in root.rglob("*"):
    if path.is_file() and path.suffix in bad_suffixes:
        checks.append({"path": str(path.relative_to(root)), "ok": False, "reason": "binary artifact"})
for path in root.rglob("*"):
    if not path.is_file() or path.suffix not in {".py", ".sh", ".json", ".toml", ".md", ".env", ".txt"}:
        continue
    if path == root / "bin" / "check.sh":
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Build markers from character codes so the scanner does not flag itself.
    bad_tokens = [
        "".join(chr(code) for code in values)
        for values in (
            (47, 104, 111, 109, 101, 47, 97, 108, 108, 47),
            (47, 109, 110, 116, 47),
            (47, 115, 99, 114, 97, 116, 99, 104, 47),
            (80, 83, 73, 95, 47, 99, 111, 100, 101, 95, 49),
            (118, 108, 97, 95, 118, 52),
        )
    ]
    for token in bad_tokens:
        if token in text:
            checks.append({"path": str(path.relative_to(root)), "ok": False, "reason": f"host path: {token}"})
            break
payload = {"ok": all(item["ok"] for item in checks), "checks": checks}
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["ok"] else 1)
PY
