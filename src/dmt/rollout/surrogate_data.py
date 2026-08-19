#!/usr/bin/env python3
"""Collect (VAE latent, SR) pairs for surrogate model training.

Uses Pi05Evaluator to reuse π05 across episodes (30s/sample instead of 20min).
Samples in VAE latent space with multi-scale strategy:
  - 70%: small perturbation (sigma=0.3) around current best
  - 30%: large perturbation (sigma=0.8) for exploration

Usage:
  PYTHONPATH=src python -m dmt.rollout.surrogate_data \
      --source data/textures/tablecloth_gingham.png \
      --output outputs/surrogate_v2/ \
      --num-samples 200 --episodes 5
"""
from __future__ import annotations

import argparse, gc, json, math, os, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dmt.paths import (  # noqa: E402
    ROOT,
    configure_runtime,
    env_path,
    pi05_model_dir,
    resolve,
    sd_model_dir,
    tokenizer_dir,
)

PROJECT_ROOT = ROOT
SRC_DIR = ROOT / "src"

configure_runtime()

DEFAULT_SD_MODEL = sd_model_dir()
DEFAULT_PI05_MODEL = pi05_model_dir()
DEFAULT_TOKENIZER_DIR = tokenizer_dir()
_DEPS = env_path("DMT_EXTRA_DEPS")
_LEROBOT = env_path("DMT_LEROBOT_SRC")
_TOOLS = env_path("DMT_TOOLS")


def load_vae(device: str = "cuda"):
    """Load SD2.1 VAE."""
    import torch
    from diffusers import StableDiffusionPipeline
    sd_dir = DEFAULT_SD_MODEL
    single_file = next((f for f in sd_dir.iterdir() if f.suffix == ".safetensors"), None)
    if single_file:
        pipe = StableDiffusionPipeline.from_single_file(
            str(single_file), config=str(sd_dir), torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(str(sd_dir), torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    return pipe.vae


def replace_tablecloth(texture_path: Path, output_dir: Path) -> Path:
    """Generate tablecloth scene XML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    paths = [p for p in (_DEPS, _TOOLS, _LEROBOT, SRC_DIR) if p is not None]
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    subprocess.run([sys.executable, "-m", "dmt.scene.tablecloth",
                     "--image", str(texture_path), "--output-dir", str(output_dir),
                     "--no-render"], env=env, capture_output=True, check=True)
    return output_dir / "tablecloth_scene.xml"


# ═══════════════════════════════════════════════════════════════════════════
# Pi05Evaluator (reused from tablecloth_ddpm_reinforce.py)
# ═══════════════════════════════════════════════════════════════════════════
class Pi05Evaluator:
    """π05 + env loaded once → reused for all evaluations."""

    def __init__(self, suite="libero_spatial", task_id=1):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation, add_envs_task
        from dmt.runtime.pi0_libero import ensure_runtime_environment, make_libero_envs
        from safetensors.torch import load_file

        ensure_runtime_environment()
        cfg = PreTrainedConfig.from_pretrained(DEFAULT_PI05_MODEL)
        cfg.device = "cuda"
        # Disable torch.compile: attn_hook appends to a growing list each forward,
        # which dynamo captures as a `len(stored_attn)` guard → guard fails every call
        # → recompiles the whole Gemma in max-autotune → CPU-bound stall + RSS blowup.
        # This is a frozen inference model with eager attention forced anyway.
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False
        self.policy = PI05Policy(cfg)
        s = load_file(str(DEFAULT_PI05_MODEL / "model.safetensors"))
        f = self.policy._fix_pytorch_state_dict_keys(s, cfg)
        self.policy.load_state_dict({k if k.startswith("model.") else f"model.{k}": v
                                      for k, v in f.items()}, strict=False)
        self.policy.to("cuda").eval()
        self.cfg = cfg
        self.suite = suite
        self.task_id = task_id
        self._vec = None
        self._ep = None; self._ept = None
        self._pr = None; self._po = None
        self._max_steps = None
        self._current_xml = None
        self._current_task = None
        self._preprocess_observation = preprocess_observation
        self._add_envs_task = add_envs_task
        self._make_env_pre_post = make_env_pre_post_processors
        self._make_pre_post = make_pre_post_processors
        self._make_envs = make_libero_envs
        print("Pi05Evaluator loaded (reused across samples)", flush=True)

    def _ensure_env(self, scene_xml: Path, task_id: int = None):
        if task_id is None:
            task_id = self.task_id
        xml_str = str(scene_xml)

        # Same scene + task → reuse env
        if self._current_xml == xml_str and self._current_task == task_id and self._vec is not None:
            return

        if not hasattr(self, '_pp_cache'):
            self._pp_cache = {}

        # Sync GPU before env transition
        import torch; torch.cuda.synchronize()

        # Close old env
        if self._vec is not None:
            try: self._vec.close()
            except Exception: pass
            self._vec = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self.task_id = task_id
        ec, envs = self._make_envs(suite=self.suite, task_id=self.task_id,
                                    scene_xml=scene_xml,
                                    observation_width=256, observation_height=256,
                                    control_mode="relative")
        self._vec = envs[self.suite][task_id]
        self._max_steps = int(self._vec.call("_max_episode_steps")[0])

        if task_id in self._pp_cache:
            self._ep, self._ept, self._pr, self._po = self._pp_cache[task_id]
        else:
            self._ep, self._ept = self._make_env_pre_post(env_cfg=ec, policy_cfg=self.cfg)
            self._pr, self._po = self._make_pre_post(
                policy_cfg=self.cfg, pretrained_path=DEFAULT_PI05_MODEL,
                preprocessor_overrides={
                    "rename_observations_processor": {"rename_map": {}},
                    "device_processor": {"device": "cuda"},
                    "tokenizer_processor": {"tokenizer_name": str(DEFAULT_TOKENIZER_DIR)},
                })
            self._pp_cache[task_id] = (self._ep, self._ept, self._pr, self._po)

        self._current_xml = xml_str; self._current_task = task_id

    # NOTE: In-place texture hot-swap (mjr_uploadTexture) was attempted to avoid per-candidate
    # env rebuilds, but LIBERO's OffScreenRenderEnv renders policy observations through a
    # separate render context — uploading to sim._render_context_offscreen did NOT change the
    # policy-facing image (verified: dark vs light textures gave byte-identical obs). So each
    # distinct texture requires a real env rebuild. Do not re-attempt without confirming which
    # render context feeds the observation pipeline.

    def eval(self, scene_xml: Path, num_episodes: int = 5, start_seed: int = 0,
             detail: bool = False, task_id: int = None) -> dict:
        """Run episodes, reusing policy and env.

        detail=True additionally records per-episode eef_positions and
        action_trace (for reward layer-2 w2/w3 paired divergence).
        task_id overrides the default task_id for this evaluation."""
        self._ensure_env(scene_xml, task_id=task_id)
        return self._run_episodes(num_episodes, start_seed, detail)

    def _run_episodes(self, num_episodes: int, start_seed: int, detail: bool) -> dict:
        from lerobot.utils.constants import ACTION

        def _eef(raw_obs: dict) -> list[float]:
            rs = raw_obs.get("robot_state", {})
            pos = np.asarray(rs.get("eef", {}).get("pos", [[0, 0, 0]])).flatten()[:3]
            return pos.tolist()

        results = []
        for ep_idx in range(num_episodes):
            seed = start_seed + ep_idx
            raw, _ = self._vec.reset(seed=[seed])
            obs = self._preprocess_observation(raw); obs = self._add_envs_task(self._vec, obs)
            obs = self._ep(obs); obs = self._pr(obs)
            self.policy.reset()
            step = 0; success = False
            eef_positions = [_eef(raw)] if detail else None
            action_trace = [] if detail else None
            while step < self._max_steps:
                action = self.policy.select_action(obs)
                action_pp = self._po(action)
                act_trans = {ACTION: action_pp}; act_trans = self._ept(act_trans)
                env_action = act_trans[ACTION].to("cpu").numpy()
                if detail:
                    action_trace.append(env_action[0].tolist())
                raw, reward, terminated, truncated, info = self._vec.step(env_action)
                step += 1
                if detail:
                    eef_positions.append(_eef(raw))
                if "final_info" in info and isinstance(info["final_info"], dict):
                    sv = info["final_info"].get("is_success")
                    if sv is not None: success = bool(np.asarray(sv).flatten()[0])
                if bool(terminated[0]) or bool(truncated[0]): break
                obs = self._preprocess_observation(raw); obs = self._add_envs_task(self._vec, obs)
                obs = self._ep(obs); obs = self._pr(obs)
            rec = {"success": success, "steps": step, "seed": seed}
            if detail:
                rec["eef_positions"] = eef_positions
                rec["action_trace"] = action_trace
            results.append(rec)
        sr = sum(r["success"] for r in results) / len(results)
        return {"success_rate": sr, "episodes": results}


    def close(self):
        if self._vec is not None: self._vec.close(); self._vec = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect (VAE latent, SR) pairs for surrogate training.")
    parser.add_argument("--source", type=Path, required=True,
                        help="Source tablecloth texture.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for dataset.")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of (latent, SR) pairs to collect.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per SR evaluation.")
    parser.add_argument("--sigma-init", type=float, default=0.5,
                        help="Initial perturbation sigma (latent space).")
    parser.add_argument("--sigma-max", type=float, default=3.0,
                        help="Maximum sigma cap.")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Adaptively increase sigma if all SR==1.0.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing dataset.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # configure_runtime() must run before PyTorch so a host can supply a
    # compatible CUDA package tree through DMT_EXTRA_DEPS.
    import torch

    device = args.device
    start_time = time.time()

    # ── Load VAE + encode source ──
    vae = load_vae(device)
    scaling = getattr(vae.config, "scaling_factor", 0.18215)

    source_img = Image.open(args.source).convert("RGB").resize((512, 512))
    source_np = np.array(source_img).astype(np.float32)
    source_tensor = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float16) / 255.0
    source_norm = (source_tensor * 2.0 - 1.0).to(dtype=torch.float16)
    with torch.no_grad():
        z_source = vae.encode(source_norm).latent_dist.sample() * scaling
    z_source = z_source.detach()
    print(f"Source latent: {z_source.shape}, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    # ── Load π05 evaluator ──
    evaluator = Pi05Evaluator()

    # ── Output dirs ──
    args.output.mkdir(parents=True, exist_ok=True)
    tex_dir = args.output / "textures"
    latent_dir = args.output / "latents"
    tex_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = PROJECT_ROOT / "config" / "scene" / "surrogate_data_tmp"

    # Resume
    dataset_path = args.output / "dataset.jsonl"
    existing = 0
    if args.resume and dataset_path.exists():
        with open(dataset_path) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing samples", flush=True)

    # ── Collect with adaptive sampling ──
    rng = np.random.RandomState(42)
    records = []
    sigma = args.sigma_init
    current_center = z_source.clone()
    exploit_counter = 0

    for idx in range(existing, args.num_samples):
        t0 = time.time()

        # Sample perturbation
        eps = torch.randn(1, *current_center.shape[1:], device=device) * sigma
        z_i = current_center + eps

        # Decode to texture
        with torch.no_grad():
            tex_01 = vae.decode((z_i / scaling).to(dtype=torch.float16)).sample
            tex_01 = (tex_01 + 1.0) / 2.0
        tex_np = (tex_01[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if tex_np.shape[0] != 512:
            tex_np = np.array(Image.fromarray(tex_np).resize((512, 512)))

        # Save texture
        tex_path = tex_dir / f"{idx:04d}.png"
        Image.fromarray(tex_np).save(tex_path)

        # Save latent
        z_np = z_i.cpu().float().numpy()
        np.save(latent_dir / f"{idx:04d}.npy", z_np)

        # Generate scene + evaluate SR
        scene_xml = replace_tablecloth(tex_path, tmp_dir / f"s_{idx:04d}")
        result = evaluator.eval(scene_xml, num_episodes=args.episodes,
                                start_seed=idx * args.episodes)
        sr = result["success_rate"]

        elapsed = time.time() - t0
        record = {
            "index": idx, "sigma": round(sigma, 2), "sr": float(sr),
            "tex_path": str(tex_path), "latent_path": str(latent_dir / f"{idx:04d}.npy"),
            "elapsed_s": round(elapsed, 1),
        }
        records.append(record)

        # Append to JSONL
        with open(dataset_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        sr_list = [r["sr"] for r in records]
        print(f"[{idx+1}/{args.num_samples}] sigma={sigma:.2f} sr={sr:.2f} "
              f"({elapsed:.0f}s) | min={min(sr_list):.2f} max={max(sr_list):.2f} "
              f"mean={np.mean(sr_list):.2f}",
              flush=True)

        # ── Adaptive strategy ──
        if args.adaptive:
            # Adaptive sigma: increase if recent 10 samples all SR==1.0
            if len(records) >= 10:
                recent = sr_list[-10:]
                if all(s == 1.0 for s in recent):
                    sigma = min(sigma * 1.5, args.sigma_max)
                    print(f"  ⚠ All recent SR=1.0 → sigma ↑ {sigma:.2f}", flush=True)
                elif any(s < 0.8 for s in recent):
                    sigma = max(sigma * 0.7, args.sigma_init)
                    print(f"  ✓ Found low SR → sigma ↓ {sigma:.2f}", flush=True)

            # If we find SR < 0.8, center around it for next few samples
            if sr < 0.8 and exploit_counter <= 0:
                current_center = z_i.detach().clone()
                exploit_counter = 5  # 5 nearby probes
                sigma = max(sigma * 0.5, 0.3)
                print(f"  ★ Low SR={sr:.2f} found! Centering + 5 exploit probes", flush=True)

            if exploit_counter > 0:
                exploit_counter -= 1
                if exploit_counter == 0:
                    current_center = z_source.clone()
                    sigma = args.sigma_init
                    print(f"  ← Back to source center, sigma={sigma:.2f}", flush=True)

    evaluator.close()

    # ── Summary ──
    sr_list = [r["sr"] for r in records]
    summary = {
        "num_samples": len(records),
        "source": str(args.source),
        "sr_min": float(min(sr_list)),
        "sr_max": float(max(sr_list)),
        "sr_mean": float(np.mean(sr_list)),
        "sr_std": float(np.std(sr_list)),
        "below_08": sum(1 for s in sr_list if s < 0.8),
        "below_09": sum(1 for s in sr_list if s < 0.9),
        "elapsed_s": round(time.time() - start_time, 1),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {summary_path}")
    print(f"  SR: min={summary['sr_min']:.2f} max={summary['sr_max']:.2f} "
          f"mean={summary['sr_mean']:.2f}±{summary['sr_std']:.2f}")
    print(f"  SR<0.8: {summary['below_08']}, SR<0.9: {summary['below_09']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
