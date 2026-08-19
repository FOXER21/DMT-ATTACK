#!/usr/bin/env python3
"""Evaluate a fixed 2D patch on Pi0 or Pi0.5 with official LIBERO init states.

The LeRobot LIBERO wrapper maps ``reset(seed=[episode_index])`` to the
benchmark's fixed init-state index.  This gives Pi0-family rollouts the same
paired task/episode contract used by the Pi evaluator while retaining
each policy's native pre/post-processors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Pi policies are PyTorch-only. Keep Transformers from importing the bundled
# TensorFlow runtime, which is intentionally pinned to a different NumPy ABI.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
from PIL import Image

from dmt.rollout.common import PATCH_SIDES, ROOT, atomic_write_json, experiment_id, rel, sha256_file
from dmt.paths import (
    configure_bundle_pythonpath,
    insert_existing_paths,
    path_from_env,
    pi0_model_dir,
    pi05_model_dir,
    resolve_bundle_path,
    tokenizer_dir,
)


configure_bundle_pythonpath()
DEPS = path_from_env("DMT_EXTRA_DEPS")
LEROBOT = path_from_env("DMT_LEROBOT_SRC")
TOOLS = path_from_env("DMT_TOOLS")
for dependency in (DEPS, LEROBOT, TOOLS):
    if dependency is None:
        continue
    insert_existing_paths([dependency], front=True)


VIEW_KEYS = {"agent": "observation.images.image", "wrist": "observation.images.image2"}
DEFAULT_MODEL = {
    "pi0": pi0_model_dir(),
    "pi05": pi05_model_dir(),
}
DEFAULT_TOKENIZER = tokenizer_dir()


def parse_task_ids(value: str) -> list[int] | None:
    if value == "all":
        return None
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("--task-ids must be all or a non-empty comma separated list")
    return result


def parse_episode_indices(value: str) -> list[int]:
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episode-indices must be START:STOP (stop excluded)") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("--episode-indices must satisfy 0 <= START < STOP")
    return list(range(start, stop))


def parse_views(value: str) -> list[str]:
    views = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(views) - set(VIEW_KEYS))
    if unknown or not views:
        raise argparse.ArgumentTypeError(f"Invalid views: {unknown or value}")
    return views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--task-suite", choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"), required=True)
    parser.add_argument("--task-ids", type=parse_task_ids, default=None)
    parser.add_argument("--episode-indices", type=parse_episode_indices, default=list(range(50)))
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--patch-side", type=int)
    parser.add_argument("--views", type=parse_views, default=["agent", "wrist"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-new-episodes", type=int)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def apply_patch(observation: dict[str, Any], patch: np.ndarray | None, views: list[str]) -> dict[str, Any]:
    if patch is None:
        return observation
    import torch

    output = dict(observation)
    for view in views:
        key = VIEW_KEYS[view]
        value = observation[key]
        height, width = value.shape[-2:]
        patch_height, patch_width = patch.shape[-2:]
        if patch_height > height or patch_width > width:
            raise ValueError(f"Patch {patch_height}x{patch_width} exceeds {key} input {height}x{width}")
        updated = value.clone()
        patch_tensor = torch.as_tensor(patch, device=value.device, dtype=value.dtype)
        updated[:, :, :patch_height, :patch_width] = patch_tensor.unsqueeze(0)
        output[key] = updated
    return output


def save_policy_input(observation: dict[str, Any], directory: Path, label: str, views: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for view in views:
        tensor = observation[VIEW_KEYS[view]][0].detach().float().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(np.clip(np.rint(tensor * 255), 0, 255).astype(np.uint8), mode="RGB").save(directory / f"{label}_{view}_input.png")


class PiRunner:
    def __init__(self, *, model: str, model_dir: Path, tokenizer_dir: Path) -> None:
        self.model_kind = model
        self.model_dir = model_dir.resolve()
        self.tokenizer_dir = tokenizer_dir.resolve()
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None
        self.envs = None
        self.vec_env = None
        self.env_preprocessor = None
        self.env_postprocessor = None
        self.suite: str | None = None
        self.task_id: int | None = None
        self._load_policy()

    def _load_policy(self) -> None:
        import torch
        from safetensors.torch import load_file
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        if self.model_kind == "pi0":
            from dmt.runtime.pi0_libero import ensure_runtime_environment, load_pi0_policy

            ensure_runtime_environment()
            self.policy, self.cfg, _report = load_pi0_policy(self.model_dir, "cuda")
        else:
            from dmt.runtime.pi0_libero import ensure_runtime_environment
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            ensure_runtime_environment()
            cfg = PreTrainedConfig.from_pretrained(self.model_dir)
            cfg.device = "cuda"
            if hasattr(cfg, "compile_model"):
                cfg.compile_model = False
            policy = PI05Policy(cfg)
            state = load_file(str(self.model_dir / "model.safetensors"))
            fixed = policy._fix_pytorch_state_dict_keys(state, cfg)
            missing, unexpected = policy.load_state_dict(
                {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}, strict=False
            )
            if unexpected:
                raise RuntimeError(f"Unexpected Pi0.5 weights: {unexpected[:5]}")
            policy.to("cuda").eval()
            self.policy, self.cfg = policy, cfg
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=self.model_dir,
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": {}},
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer_dir)},
            },
        )
        torch.cuda.synchronize()

    def ensure_env(self, suite: str, task_id: int) -> None:
        if self.vec_env is not None and self.suite == suite and self.task_id == task_id:
            return
        self.close_env()
        from dmt.runtime.pi0_libero import make_libero_envs
        from lerobot.envs.factory import make_env_pre_post_processors

        env_cfg, envs = make_libero_envs(
            suite=suite,
            task_id=task_id,
            observation_width=256,
            observation_height=256,
            control_mode="relative",
            scene_xml=None,
        )
        self.envs = envs
        self.vec_env = envs[suite][task_id]
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.cfg)
        self.suite, self.task_id = suite, task_id

    def prepare(self, raw: dict[str, Any], patch: np.ndarray | None, views: list[str]) -> dict[str, Any]:
        from lerobot.envs.utils import add_envs_task, preprocess_observation

        if self.vec_env is None or self.env_preprocessor is None or self.preprocessor is None:
            raise RuntimeError("Environment is not initialized")
        observation = preprocess_observation(raw)
        observation = add_envs_task(self.vec_env, observation)
        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)
        return apply_patch(observation, patch, views)

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        from lerobot.utils.constants import ACTION

        action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        transition = self.env_postprocessor({ACTION: action})
        return transition[ACTION].to("cpu").numpy()

    def close_env(self) -> None:
        if self.envs is None:
            return
        try:
            from lerobot.envs.utils import close_envs

            close_envs(self.envs)
        except Exception:
            try:
                self.vec_env.close()
            except Exception:
                pass
        self.envs = self.vec_env = None
        self.env_preprocessor = self.env_postprocessor = None
        self.suite = self.task_id = None

    def close(self) -> None:
        self.close_env()
        del self.policy, self.preprocessor, self.postprocessor
        self.policy = self.preprocessor = self.postprocessor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def read_resumed(output: Path, protocol: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    if not resume or not output.is_file():
        return []
    payload = json.loads(output.read_text(encoding="utf-8"))
    existing_protocol = payload.get("protocol")
    if not isinstance(existing_protocol, dict):
        raise ValueError(f"Cannot resume report with a malformed protocol: {output}")
    existing_identity = dict(existing_protocol)
    current_identity = dict(protocol)
    # Public checkpoints can live at a different absolute root on the server.
    existing_identity["model_dir"] = Path(str(existing_identity.get("model_dir", ""))).name
    current_identity["model_dir"] = Path(str(current_identity.get("model_dir", ""))).name
    if payload.get("experiment_id") != experiment_id() or existing_identity != current_identity:
        raise ValueError(f"Cannot resume report with a different protocol: {output}")
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"Malformed resume report: {output}")
    seen = set()
    for row in rows:
        key = (int(row["task_id"]), int(row["episode_index"]))
        if key in seen:
            raise ValueError(f"Duplicate resume episode: {key}")
        seen.add(key)
    return rows


def write_report(output: Path, protocol: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    ordered = sorted(episodes, key=lambda row: (int(row["task_id"]), int(row["episode_index"])))
    payload = {
        "experiment_id": experiment_id(),
        "protocol": protocol,
        "episodes": ordered,
        "success_rate": sum(bool(row["success"]) for row in ordered) / len(ordered) if ordered else None,
    }
    atomic_write_json(output, payload)


def main() -> int:
    args = parse_args()
    import torch
    from libero.libero import benchmark

    model_dir = resolve_bundle_path(args.model_dir or DEFAULT_MODEL[args.model]).resolve()
    tokenizer_path = resolve_bundle_path(args.tokenizer_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    patch_path = None if args.patch is None else resolve_bundle_path(args.patch).resolve()
    if patch_path is not None and not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    patch_side = args.patch_side or PATCH_SIDES["Pi0" if args.model == "pi0" else "Pi0.5"]
    if patch_side <= 0:
        raise ValueError("--patch-side must be positive")
    patch = None
    if patch_path is not None:
        patch = np.asarray(
            Image.open(patch_path).convert("RGB").resize((patch_side, patch_side), Image.Resampling.LANCZOS), dtype=np.float32
        ).transpose(2, 0, 1) / 255.0
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= suite.n_tasks]
    if invalid:
        raise ValueError(f"Task ids outside suite range: {invalid}")
    if max(args.episode_indices) >= 50:
        raise ValueError("LIBERO official benchmark exposes 50 fixed initial states per task")
    output = resolve_bundle_path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": "Pi0" if args.model == "pi0" else "Pi0.5",
        "model_dir": f"models/{model_dir.name}",
        "model_dir_resolved": str(model_dir),
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "episode_indices": args.episode_indices,
        "fixed_initial_states": True,
        "reset_mapping": "LeRobot LiberoEnv reset(seed=[episode_index]) maps seed to official task init-state index then executes 10 stabilization steps",
        "patch_path": None if patch_path is None else rel(patch_path),
        "patch_sha256": None if patch_path is None else sha256_file(patch_path),
        "patch_side": patch_side,
        "relative_area": (patch_side / 256.0) ** 2,
        "position": "top_left_after_policy_preprocessing",
        "views": args.views,
        "seed": args.seed,
    }
    episodes = read_resumed(output, protocol, args.resume)
    completed = {(int(row["task_id"]), int(row["episode_index"])) for row in episodes}
    if args.max_new_episodes is not None and args.max_new_episodes <= 0:
        raise ValueError("--max-new-episodes must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = PiRunner(model=args.model, model_dir=model_dir, tokenizer_dir=tokenizer_path)
    new_episodes = 0
    try:
        for task_id in task_ids:
            runner.ensure_env(args.task_suite, task_id)
            max_steps = int(runner.vec_env.call("_max_episode_steps")[0])
            for episode_index in args.episode_indices:
                if (task_id, episode_index) in completed:
                    continue
                if args.max_new_episodes is not None and new_episodes >= args.max_new_episodes:
                    write_report(output, protocol, episodes)
                    print(json.dumps({"output": rel(output), "status": "partial", "episodes": len(episodes)}))
                    return 0
                # Reset policy noise deterministically for every paired Clean/attack rollout.
                torch.manual_seed(args.seed + task_id * 1000 + episode_index)
                torch.cuda.manual_seed_all(args.seed + task_id * 1000 + episode_index)
                raw, _info = runner.vec_env.reset(seed=[episode_index])
                observation = runner.prepare(raw, patch, args.views)
                if args.verification_dir is not None and task_id == task_ids[0] and episode_index == args.episode_indices[0]:
                    save_policy_input(
                        observation,
                        resolve_bundle_path(args.verification_dir).resolve(),
                        "patched" if patch is not None else "clean",
                        args.views,
                    )
                runner.policy.reset()
                success = False
                steps = 0
                start = time.time()
                while steps < max_steps:
                    env_action = runner.select_action(observation)
                    raw, _reward, terminated, truncated, info = runner.vec_env.step(env_action)
                    steps += 1
                    if "final_info" in info and isinstance(info["final_info"], dict):
                        value = info["final_info"].get("is_success")
                        if value is not None:
                            success = bool(np.asarray(value).reshape(-1)[0])
                    if bool(terminated[0]) or bool(truncated[0]):
                        break
                    observation = runner.prepare(raw, patch, args.views)
                row = {
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": success,
                    "steps": steps,
                    "elapsed_s": round(time.time() - start, 3),
                }
                episodes.append(row)
                completed.add((task_id, episode_index))
                new_episodes += 1
                write_report(output, protocol, episodes)
                print(f"task={task_id} episode={episode_index} success={success} steps={steps}", flush=True)
    finally:
        runner.close()
    write_report(output, protocol, episodes)
    print(json.dumps({"output": rel(output), "episodes": len(episodes), "success_rate": sum(row["success"] for row in episodes) / len(episodes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
