#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from dmt.paths import (
    ROOT as BUNDLE_ROOT,
    configure_bundle_pythonpath,
    insert_existing_paths,
    path_from_env,
    pi0_model_dir,
    tokenizer_dir,
)


RUNTIME_ROOT = BUNDLE_ROOT / "runtime" / "pi_libero"
DEFAULT_CONFIG_DIR = RUNTIME_ROOT / "libero_config"
DEFAULT_DATASETS_DIR = RUNTIME_ROOT / "libero_datasets"
DEFAULT_MODEL_DIR = pi0_model_dir()
DEFAULT_TOKENIZER_DIR = tokenizer_dir()

EMBED_TOKENS_KEY = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "model.paligemma_with_expert.paligemma.lm_head.weight"


def ensure_local_import_paths() -> None:
    configure_bundle_pythonpath()
    extra_deps = path_from_env("DMT_EXTRA_DEPS")
    if extra_deps is not None:
        insert_existing_paths([extra_deps], front=True)


def ensure_runtime_environment(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    ensure_local_import_paths()

    from dmt.runtime.probe import ensure_libero_config

    config_path = ensure_libero_config(config_dir=config_dir, datasets_dir=datasets_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return config_path


def get_cuda_memory_report(device: str = "cuda:0") -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "available": False,
            "device": device,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    return {
        "available": True,
        "device": device,
        "free_bytes": int(free_bytes),
        "used_bytes": int(used_bytes),
        "total_bytes": int(total_bytes),
        "free_gb": round(free_bytes / (1024**3), 3),
        "used_gb": round(used_bytes / (1024**3), 3),
        "total_gb": round(total_bytes / (1024**3), 3),
    }


def choose_device(
    requested_device: str,
    *,
    min_cuda_free_gb: float,
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    requested_device = requested_device.strip().lower()

    if requested_device == "cpu":
        return {
            "requested_device": requested_device,
            "selected_device": "cpu",
            "fallback_used": False,
            "reason": "cpu_requested",
            "cuda_memory": get_cuda_memory_report(),
        }

    cuda_report = get_cuda_memory_report()
    cuda_available = bool(cuda_report.get("available"))
    free_gb = float(cuda_report.get("free_gb", 0.0))

    if requested_device in {"cuda", "cuda:0"}:
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if free_gb < min_cuda_free_gb:
            raise RuntimeError(
                f"CUDA was requested but only {free_gb:.3f} GB is free; "
                f"need at least {min_cuda_free_gb:.3f} GB."
            )
        return {
            "requested_device": requested_device,
            "selected_device": "cuda",
            "fallback_used": False,
            "reason": "cuda_requested",
            "cuda_memory": cuda_report,
        }

    if requested_device != "auto":
        raise ValueError(f"Unsupported device selector: {requested_device}")

    if cuda_available and free_gb >= min_cuda_free_gb:
        return {
            "requested_device": requested_device,
            "selected_device": "cuda",
            "fallback_used": False,
            "reason": "cuda_has_enough_free_memory",
            "cuda_memory": cuda_report,
        }

    if not allow_cpu_fallback:
        raise RuntimeError(
            "Automatic device selection did not find enough free CUDA memory "
            f"({free_gb:.3f} GB free, need {min_cuda_free_gb:.3f} GB), and CPU fallback is disabled."
        )

    return {
        "requested_device": requested_device,
        "selected_device": "cpu",
        "fallback_used": True,
        "reason": "insufficient_cuda_memory_for_pi0",
        "cuda_memory": cuda_report,
    }


def _remap_pi0_state_dict(policy: Any, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    start = time.time()
    original_state_dict = load_file(str(model_dir / "model.safetensors"))
    fixed_state_dict = policy._fix_pytorch_state_dict_keys(original_state_dict, policy.config)

    remapped_state_dict: dict[str, Any] = {}
    prefixed_keys = 0
    for key, value in fixed_state_dict.items():
        new_key = key if key.startswith("model.") else f"model.{key}"
        if new_key != key:
            prefixed_keys += 1
        remapped_state_dict[new_key] = value

    embed_tokens_aliased_from_lm_head = False
    if EMBED_TOKENS_KEY not in remapped_state_dict and LM_HEAD_KEY in remapped_state_dict:
        remapped_state_dict[EMBED_TOKENS_KEY] = remapped_state_dict[LM_HEAD_KEY]
        embed_tokens_aliased_from_lm_head = True

    report = {
        "state_dict_load_s": round(time.time() - start, 3),
        "remapped_key_prefix_count": prefixed_keys,
        "embed_tokens_aliased_from_lm_head": embed_tokens_aliased_from_lm_head,
        "embed_tokens_key": EMBED_TOKENS_KEY,
        "lm_head_key": LM_HEAD_KEY,
    }
    return remapped_state_dict, report


def load_pi0_policy(model_dir: Path, target_device: str) -> tuple[Any, Any, dict[str, Any]]:
    ensure_local_import_paths()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    start = time.time()
    policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
    policy_cfg.pretrained_path = model_dir
    policy_cfg.device = "cpu"
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.use_amp = False

    init_start = time.time()
    policy = PI0Policy(policy_cfg)
    remapped_state_dict, remap_report = _remap_pi0_state_dict(policy=policy, model_dir=model_dir)
    missing_keys, unexpected_keys = policy.load_state_dict(remapped_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "PI0 state dict did not load cleanly.\n"
            + json.dumps(
                {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                },
                indent=2,
            )
        )

    device_move_s = 0.0
    if target_device != "cpu":
        move_start = time.time()
        policy.to(target_device)
        device_move_s = time.time() - move_start

    policy.config.device = target_device
    policy.eval()

    report = {
        "model_dir": str(model_dir),
        "target_device": target_device,
        "policy_init_s": round(time.time() - init_start, 3),
        "device_move_s": round(device_move_s, 3),
        "total_load_s": round(time.time() - start, 3),
        "config_dtype": getattr(policy_cfg, "dtype", None),
        "config_chunk_size": getattr(policy_cfg, "chunk_size", None),
        "config_n_action_steps": getattr(policy_cfg, "n_action_steps", None),
        "compile_model": getattr(policy_cfg, "compile_model", None),
        "gradient_checkpointing": getattr(policy_cfg, "gradient_checkpointing", None),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        **remap_report,
    }
    return policy, policy_cfg, report


def make_policy_processors(policy_cfg: Any, model_dir: Path, tokenizer_dir: Path, device: str) -> tuple[Any, Any]:
    ensure_local_import_paths()

    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
        },
    )
    return preprocessor, postprocessor


def make_libero_envs(
    *,
    suite: str,
    task_id: int,
    observation_width: int,
    observation_height: int,
    control_mode: str = "relative",
    scene_xml: Path | None = None,
) -> tuple[Any, Any]:
    ensure_local_import_paths()

    from gymnasium.vector import SyncVectorEnv
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env

    env_cfg = LiberoEnvConfig(
        task=suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=observation_height,
        observation_width=observation_width,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        control_mode=control_mode,
    )
    if scene_xml is None:
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    else:
        from scene_override_libero import create_scene_override_libero_envs

        envs = create_scene_override_libero_envs(
            task=suite,
            n_envs=1,
            scene_xml=scene_xml,
            gym_kwargs={
                "task_ids": [task_id],
                "obs_type": "pixels_agent_pos",
                "observation_height": observation_height,
                "observation_width": observation_width,
            },
            env_cls=SyncVectorEnv,
            control_mode=control_mode,
            camera_name="agentview_image,robot0_eye_in_hand_image",
        )
    return env_cfg, envs


@dataclass
class Pi0LiberoStack:
    suite: str
    task_id: int
    policy: Any
    policy_cfg: Any
    policy_load_report: dict[str, Any]
    device_report: dict[str, Any]
    env_cfg: Any
    envs: Any
    vec_env: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any

    def close(self) -> None:
        ensure_local_import_paths()
        from lerobot.envs.utils import close_envs

        close_envs(self.envs)


def build_pi0_libero_stack(
    *,
    suite: str,
    task_id: int,
    model_dir: Path,
    tokenizer_dir: Path,
    device: str,
    observation_width: int,
    observation_height: int,
    control_mode: str = "relative",
    scene_xml: Path | None = None,
) -> Pi0LiberoStack:
    ensure_runtime_environment()
    ensure_local_import_paths()

    from lerobot.envs.factory import make_env_pre_post_processors

    policy, policy_cfg, policy_load_report = load_pi0_policy(model_dir=model_dir, target_device=device)
    policy_cfg.device = device

    env_cfg, envs = make_libero_envs(
        suite=suite,
        task_id=task_id,
        observation_width=observation_width,
        observation_height=observation_height,
        control_mode=control_mode,
        scene_xml=scene_xml,
    )
    vec_env = envs[suite][task_id]
    preprocessor, postprocessor = make_policy_processors(
        policy_cfg=policy_cfg,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        device=device,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    return Pi0LiberoStack(
        suite=suite,
        task_id=task_id,
        policy=policy,
        policy_cfg=policy_cfg,
        policy_load_report=policy_load_report,
        device_report={},
        env_cfg=env_cfg,
        envs=envs,
        vec_env=vec_env,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
    )


def reset_and_preprocess(stack: Pi0LiberoStack, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ensure_local_import_paths()

    from lerobot.envs.utils import add_envs_task, preprocess_observation

    raw_observation, reset_info = stack.vec_env.reset(seed=[seed])
    observation = preprocess_observation(raw_observation)
    observation = add_envs_task(stack.vec_env, observation)
    observation = stack.env_preprocessor(observation)
    observation = stack.preprocessor(observation)
    return raw_observation, reset_info, observation


def policy_action_to_env_action(stack: Pi0LiberoStack, action: Any) -> np.ndarray:
    ensure_local_import_paths()

    from lerobot.utils.constants import ACTION

    action = stack.postprocessor(action)
    action_transition = {ACTION: action}
    action_transition = stack.env_postprocessor(action_transition)
    env_action = action_transition[ACTION]
    return env_action.to("cpu").numpy()


def summarize_object(value: Any) -> Any:
    try:
        import torch
    except Exception:  # pragma: no cover - only for defensive fallback
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        summary: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if value.numel() > 0 and value.dtype not in {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            cpu_value = value.detach().to("cpu")
            summary["min"] = float(cpu_value.min().item())
            summary["max"] = float(cpu_value.max().item())
        return summary

    if isinstance(value, np.ndarray):
        summary = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size > 0 and value.dtype.kind in {"f", "i", "u"}:
            summary["min"] = float(value.min())
            summary["max"] = float(value.max())
        return summary

    if isinstance(value, dict):
        return {key: summarize_object(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(not isinstance(item, (dict, list, tuple)) for item in value):
            return list(value)
        return {
            "type": type(value).__name__,
            "length": len(value),
        }

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value

    return str(type(value))


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: summarize_object(value) for key, value in batch.items()}


def extract_camera_frame(raw_observation: dict[str, Any], key: str) -> np.ndarray:
    frame = raw_observation["pixels"][key]
    if frame.ndim == 4:
        return np.asarray(frame[0])
    return np.asarray(frame)


def slugify_task_name(task_name: str) -> str:
    safe_chars = []
    for char in task_name.lower():
        if char.isalnum():
            safe_chars.append(char)
        elif char in {" ", "-", "_"}:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "task"
