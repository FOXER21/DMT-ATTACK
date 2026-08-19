#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


from dmt.paths import (
    ROOT as BUNDLE_ROOT,
    configure_bundle_pythonpath,
    libero_root,
    path_from_env,
)


RUNTIME_ROOT = BUNDLE_ROOT / "runtime" / "pi_libero"
DEFAULT_CONFIG_DIR = RUNTIME_ROOT / "libero_config"
DEFAULT_DATASETS_DIR = RUNTIME_ROOT / "libero_datasets"
DEFAULT_OUTPUT_JSON = RUNTIME_ROOT / "stack_probe.json"
DEFAULT_FRAME_PATH = RUNTIME_ROOT / "clean_frames" / "libero_spatial_task0_reset.png"


def ensure_local_import_paths() -> None:
    configure_bundle_pythonpath()


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - surfaced in JSON
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "ok": True,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def ensure_libero_config(config_dir: Path, datasets_dir: Path) -> Path:
    import yaml

    package_root = libero_root()
    if package_root is None or not package_root.is_dir():
        raise RuntimeError(
            "LIBERO is unavailable. Install the benchmark runtime or set DMT_LIBERO_ROOT to its libero/libero directory."
        )
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(package_root / "assets"),
    }

    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_libero_smoke(
    suite_name: str,
    task_id: int,
    observation_width: int,
    observation_height: int,
    config_dir: Path,
    frame_path: Path | None,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    suite_ctor = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_ctor()
    task = suite.get_task(task_id)

    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        observation_width=observation_width,
        observation_height=observation_height,
    )

    try:
        observation, info = env.reset()
        pixels = observation["pixels"]
        pixel_summary = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": int(np.min(value)),
                "max": int(np.max(value)),
            }
            for key, value in pixels.items()
        }

        if frame_path is not None:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels["image"]).save(frame_path)

        dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        step_observation, reward, terminated, truncated, step_info = env.step(dummy_action)

        return {
            "suite": suite_name,
            "task_id": task_id,
            "num_tasks_in_suite": len(suite.tasks),
            "task_name": task.name,
            "task_language": task.language,
            "camera_name": list(env.camera_name),
            "task_description": env.task_description,
            "max_episode_steps": env._max_episode_steps,
            "reset_info_keys": sorted(info.keys()),
            "pixel_summary": pixel_summary,
            "frame_path": str(frame_path) if frame_path is not None else None,
            "step_reward": float(reward),
            "step_terminated": bool(terminated),
            "step_truncated": bool(truncated),
            "step_info_keys": sorted(step_info.keys()),
            "post_step_pixel_keys": sorted(step_observation["pixels"].keys()),
            "action_space_shape": list(env.action_space.shape),
            "observation_width": observation_width,
            "observation_height": observation_height,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the local LIBERO + LeRobot stack.")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--observation-width", type=int, default=256)
    parser.add_argument("--observation-height", type=int, default=256)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_local_import_paths()

    config_path = ensure_libero_config(args.config_dir, args.datasets_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(args.config_dir)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    report: dict[str, Any] = {
        "status": "ok",
        "bundle_root": str(BUNDLE_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "libero_config_path": str(config_path),
        "env": {
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "imports": {
            name: import_status(name)
            for name in [
                "torch",
                "transformers",
                "mujoco",
                "robosuite",
                "robomimic",
                "libero",
                "lerobot",
                "lerobot.policies.pi0.modeling_pi0",
            ]
        },
    }

    try:
        report["libero_smoke"] = run_libero_smoke(
            suite_name=args.suite,
            task_id=args.task_id,
            observation_width=args.observation_width,
            observation_height=args.observation_height,
            config_dir=args.config_dir,
            frame_path=args.frame_path,
        )
    except Exception as exc:  # pragma: no cover - surfaced in JSON
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
