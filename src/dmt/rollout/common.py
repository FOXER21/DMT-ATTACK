#!/usr/bin/env python3
"""Shared, dependency-light utilities for DMT rollout reports.

This module deliberately contains no model imports.  It is used by setup,
rollout, naturalness, and table scripts so every artifact carries the same
experiment identifier and provenance representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from dmt.paths import ROOT

# The fresh bundle uses one manifest for both the generator and the rollout
# helpers.  Keep this local so evaluators do not depend on the source project.
MANIFEST_PATH = ROOT / "config" / "protocol.json"

SUITES: tuple[str, ...] = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
PATCH_SIDES = {"Pi0": 57, "Pi0.5": 57}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stable_sample(items: Iterable[Path], count: int, seed: int) -> list[Path]:
    values = sorted({Path(item).resolve() for item in items})
    if count <= 0 or len(values) <= count:
        return values
    return [path for _key, path in sorted((stable_key(str(path), seed), path) for path in values)[:count]]


def experiment_id() -> str:
    return str(read_json(MANIFEST_PATH)["protocol_id"])


def source_record(path: str | Path, *, role: str, source_kind: str, copy_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    record: dict[str, Any] = {
        "role": role,
        "source_kind": source_kind,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_size_bytes": source.stat().st_size,
    }
    if copy_path is not None:
        target = resolve(copy_path)
        record["copied_path"] = rel(target)
        record["copied_sha256"] = sha256_file(target)
    return record


def require_workspace_path(path: str | Path) -> Path:
    value = resolve(path).resolve()
    try:
        value.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Output must be inside the fresh experiment workspace: {value}") from exc
    return value


def paired_casr(clean_episodes: list[dict[str, Any]], attack_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute conditional ASR on paired official task/initial-state episodes."""
    def episode_map(rows: list[dict[str, Any]]) -> dict[tuple[int, int], bool]:
        mapped: dict[tuple[int, int], bool] = {}
        for row in rows:
            key = (int(row["task_id"]), int(row["episode_index"]))
            if key in mapped:
                raise ValueError(f"Duplicate episode in report: {key}")
            mapped[key] = bool(row["success"])
        return mapped

    clean = episode_map(clean_episodes)
    attack = episode_map(attack_episodes)
    if set(clean) != set(attack):
        missing_clean = sorted(set(attack) - set(clean))
        missing_attack = sorted(set(clean) - set(attack))
        raise ValueError(f"Unpaired rollout reports; missing_clean={missing_clean[:3]}, missing_attack={missing_attack[:3]}")
    denominator = sum(clean.values())
    failures = sum(clean[key] and not attack[key] for key in clean)
    return {
        "episodes": len(clean),
        "clean_successes": denominator,
        "attacked_successes": sum(attack.values()),
        "conditional_failures": failures,
        "clean_sr": sum(clean.values()) / len(clean) if clean else None,
        "attack_sr": sum(attack.values()) / len(attack) if attack else None,
        "casr": failures / denominator if denominator else None,
    }


def report_guard(payload: dict[str, Any]) -> None:
    """Reject stale reports and report paths that leak old evaluation outputs."""
    if payload.get("experiment_id") != experiment_id():
        raise ValueError("Report experiment_id does not match this fresh restart")
    text = json.dumps(payload, ensure_ascii=False)
    forbidden = ("/legacy/results/eval/", "/legacy/results/analysis/", "/legacy_workspace/", "/legacy_runtime/")
    if any(token in text for token in forbidden):
        raise ValueError("Report contains a prohibited old evaluation/analysis result path")
