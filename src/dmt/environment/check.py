#!/usr/bin/env python3
"""Check a DMT host without loading multi-gigabyte policy weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from dmt.paths import ROOT, base_scene_path, env_path, model_root, pi0_model_dir, pi05_model_dir, sd_model_dir, tokenizer_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-policy-files", action="store_true", help="Also check Pi0/Pi0.5 weight files.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "environment_check.json")
    return parser.parse_args()


def check_file(checks: list[dict], name: str, path: Path, required: bool = True) -> None:
    checks.append({"name": name, "path": str(path), "required": required, "ok": path.is_file() if required else True})


def check_dir(checks: list[dict], name: str, path: Path, required: bool = True) -> None:
    checks.append({"name": name, "path": str(path), "required": required, "ok": path.is_dir() if required else True})


def main() -> int:
    args = parse_args()
    checks: list[dict] = []
    check_file(checks, "teacher_records", ROOT / "data" / "teacher" / "records.jsonl")
    check_file(checks, "sd_embeddings", ROOT / "data" / "teacher" / "sd_embeddings.npz")
    check_dir(checks, "teacher_latents", ROOT / "data" / "teacher" / "latents")
    check_file(checks, "clean_rollouts", ROOT / "data" / "baselines" / "clean_rollouts.pkl")
    check_file(checks, "gate_thresholds", ROOT / "data" / "gates" / "naturalness.json")
    check_file(checks, "tablecloth_source", ROOT / "data" / "textures" / "tablecloth_gingham.png")
    check_file(checks, "prepared_base_scene", base_scene_path())
    check_file(checks, "sd_model_index", sd_model_dir() / "model_index.json")
    check_file(checks, "tokenizer", tokenizer_dir() / "tokenizer.json")
    if args.include_policy_files:
        check_file(checks, "pi0_weights", pi0_model_dir() / "model.safetensors")
        check_file(checks, "pi05_weights", pi05_model_dir() / "model.safetensors")

    for module in ("torch", "diffusers", "transformers", "safetensors", "PIL", "mujoco", "libero", "lerobot"):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        checks.append({"name": f"import:{module}", "path": module, "required": True, "ok": found})

    payload = {
        "python": sys.version,
        "model_root": str(model_root()),
        "checks": checks,
        "ok": all(item["ok"] for item in checks if item["required"]),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
