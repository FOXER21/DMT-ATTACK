#!/usr/bin/env python3
"""Diffusion sampler: prompt-conditional generation from an external
latent z_T, with typical-set whitening and natural/siglip gates.

Design law (see method.md): the reverse diffusion trajectory is FROZEN. The only
search variable is the entry boundary z_T (+ optional guidance). No intermediate
denoising step is ever supervised. CMA-ES treats generate_from_latent() as a
black box and only scores the FINAL rendered texture.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dmt.paths import configure_runtime  # noqa: E402

configure_runtime()

import math
import numpy as np
import torch
from PIL import Image

# Reuse the natural-pool metrics so generation and gating stay identical.
from dmt.diffusion.natural_pool import (  # noqa: E402
    NEGATIVE_PROMPT,
    PATTERNS,
    PALETTES,
    load_sd_pipeline,
    image_metrics,
    high_frequency_energy,
    total_variation,
    saturation_stats,
    color_entropy,
)

# SD2.1 @ 512x512 latent geometry.
LATENT_C, LATENT_H, LATENT_W = 4, 64, 64
LATENT_DIM = LATENT_C * LATENT_H * LATENT_W  # 16384
SQRT_D = math.sqrt(LATENT_DIM)


def whiten_to_typical_set(u: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """Project a flat search vector u onto the Gaussian typical set.

    A standard normal in R^d concentrates on the shell ||z|| ~ sqrt(d). CMA-ES
    searches in whitened coords; we only rescale the radius into [lo,hi]*sqrt(d)
    so the frozen DDPM always receives an in-distribution z_T (prevents the
    out-of-distribution glitch failure). Direction is preserved; this is an ENTRY
    constraint, NOT intermediate-step supervision.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(u))
    if norm < 1e-8:
        return np.zeros(LATENT_DIM, dtype=np.float32)
    target = float(np.clip(norm, lo * SQRT_D, hi * SQRT_D))
    z = u * (target / norm)
    return z.astype(np.float32)


def latent_radius_ok(u: np.ndarray, lo: float = 0.85, hi: float = 1.15) -> bool:
    """Hard feasibility: reject vectors whose radius is far off the shell even
    before whitening (a sanity guard; whitening already clamps, but a wildly
    off-shell raw sample signals the search drifted)."""
    norm = float(np.linalg.norm(np.asarray(u, dtype=np.float64).reshape(-1)))
    return lo * SQRT_D <= norm <= hi * SQRT_D


def build_prompt(family: str, palette: str | None = None) -> str:
    """Reconstruct a natural tablecloth prompt for a given family."""
    pattern = dict(PATTERNS).get(family, family)
    pal = palette or PALETTES[0]
    return (
        f"photorealistic seamless square {pal} {pattern}, "
        "flat lay, evenly lit, natural fabric weave, repeatable textile pattern, "
        "high quality material scan"
    )


def generate_from_latent(
    pipe,
    prompt: str,
    z_flat: np.ndarray,
    *,
    guidance_scale: float = 7.5,
    num_steps: int = 30,
    device: str = "cuda",
    negative_prompt: str = NEGATIVE_PROMPT,
) -> Image.Image:
    """Run the FROZEN reverse diffusion from an external latent z_T.

    z_flat is a flat (16384,) latent already whitened to the typical set. We
    reshape to (1,4,64,64) and hand it to the standard diffusers pipeline via
    the `latents=` argument; every intermediate step is untouched.
    """
    dtype = pipe.unet.dtype
    z = torch.from_numpy(np.asarray(z_flat, dtype=np.float32)).reshape(
        1, LATENT_C, LATENT_H, LATENT_W
    ).to(device=device, dtype=dtype)
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            latents=z,
        ).images[0]
    return image.convert("RGB").resize((512, 512))


def texture_metrics(img: Image.Image, source: np.ndarray, source_tv: float) -> dict:
    """Compute pool-compatible natural metrics directly from a PIL image."""
    arr = np.asarray(img.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
    diff = arr - source
    mse = float(np.mean(diff ** 2))
    tv = total_variation(arr)
    sat = saturation_stats(arr)
    return {
        "mse_vs_source": mse,
        "tv": tv,
        "tv_ratio": float(tv / max(source_tv, 1e-8)),
        "hf_energy": high_frequency_energy(arr),
        "color_entropy": color_entropy(arr),
        **sat,
    }


def natural_gate(metrics: dict, thresholds: dict) -> tuple[bool, dict]:
    """Hard absolute-threshold natural gate using absolute cutoffs
    calibrated from the natural texture pool, NOT in-batch percentiles, because
    candidates are not a fixed pool). Returns (passed, per-check dict)."""
    checks = {
        "tv_ratio": metrics["tv_ratio"] <= thresholds["tv_ratio_p90"],
        "hf_energy": metrics["hf_energy"] <= thresholds["hf_energy_p90"],
        "saturation_p95": metrics["saturation_p95"] <= thresholds["saturation_p95_p95"],
        "color_entropy": metrics["color_entropy"] >= thresholds["color_entropy_p05"],
    }
    return bool(all(checks.values())), checks


class SiglipScorer:
    """Lazy-loaded π05 SigLIP image-tower scorer. drift = 1 - cos(anchor, adv).

    image-image (NOT image-text): prompt fidelity is measured as drift away from
    a same-prompt anchor image, reusing π05's own perception front-end so the
    score reflects 'does π05 still see this as the same texture class'.
    """

    def __init__(self, model_dir: str, device: str = "cuda", policy=None):
        self.model_dir = model_dir
        self.device = device
        self.policy = policy  # if provided, reuse (avoids loading pi05 twice)

    def _ensure(self):
        if self.policy is not None:
            return
        from dmt.runtime.pi0_libero import load_pi0_policy
        self.policy, _, _ = load_pi0_policy(Path(self.model_dir), self.device)

    def _embed(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
        t = t * 2.0 - 1.0
        with torch.no_grad():
            # Pi0/Pi0.5 expose the frozen SigLIP tower through this local
            # method.  Keeping the call here avoids a source-project helper
            # import that is absent from the portable bundle.
            features = self.policy.model.paligemma_with_expert.embed_image(t)
            return features.float().mean(dim=1)

    def drift(self, anchor: Image.Image, adv: Image.Image) -> float:
        self._ensure()
        ea = self._embed(anchor)
        eb = self._embed(adv)
        return float((1.0 - torch.nn.functional.cosine_similarity(ea, eb, dim=-1)).item())
