#!/usr/bin/env python3
"""Generate a diffusion-natural tablecloth texture pool.

The output manifest records the generated texture pool and its naturalness
diagnostics. This script only creates full diffusion samples; it does not
apply pixel residuals or modify a denoising trajectory.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from dmt.paths import ROOT, configure_runtime, sd_model_dir  # noqa: E402

configure_runtime()

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


DEFAULT_SD_MODEL = sd_model_dir()
PROJECT_ROOT = ROOT
DEFAULT_SOURCE = ROOT / "data/textures/tablecloth_gingham.png"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/natural_pool"

NEGATIVE_PROMPT = (
    "noise, glitch, corrupted, mosaic, pixelated, adversarial pattern, artifacts, "
    "watermark, text, logo, object, bowl, plate, cup, hand, people, room, furniture, "
    "perspective scene, heavy shadow, torn fabric, crumpled fabric"
)

PATTERNS = [
    ("gingham", "gingham cotton tablecloth fabric texture"),
    ("plaid", "classic plaid cotton tablecloth fabric texture"),
    ("woven", "woven linen tablecloth fabric texture with visible fibers"),
    ("floral", "small floral printed cotton tablecloth fabric texture"),
    ("jacquard", "subtle jacquard woven tablecloth fabric texture"),
    ("herringbone", "herringbone woven linen tablecloth fabric texture"),
    ("stripe", "thin striped cotton tablecloth fabric texture"),
    ("paisley", "small paisley printed cotton tablecloth fabric texture"),
]

PALETTES = [
    "red, navy blue, and white",
    "forest green, ivory, and charcoal",
    "teal, white, and muted yellow",
    "cobalt blue, soft gold, and white",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: str | Path) -> str:
    p = resolve(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "sample"


def prompt_catalog() -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for family, pattern in PATTERNS:
        for palette in PALETTES:
            prompt = (
                f"photorealistic seamless square {palette} {pattern}, "
                "flat lay, evenly lit, natural fabric weave, repeatable textile pattern, "
                "high quality material scan"
            )
            prompts.append({"family": family, "palette": palette, "prompt": prompt})
    return prompts


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0


def total_variation(img: np.ndarray) -> float:
    return float(np.mean(np.abs(img[1:] - img[:-1])) + np.mean(np.abs(img[:, 1:] - img[:, :-1])))


def high_frequency_energy(img: np.ndarray) -> float:
    gray = np.tensordot(img, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([2], [0]))
    gray_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    blur = Image.fromarray(gray_u8).filter(ImageFilter.GaussianBlur(radius=2.0))
    blur_f = np.asarray(blur, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(gray - blur_f)))


def saturation_stats(img: np.ndarray) -> dict[str, float]:
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = np.zeros_like(mx)
    mask = mx > 1e-6
    sat[mask] = (mx[mask] - mn[mask]) / mx[mask]
    return {
        "saturation_mean": float(np.mean(sat)),
        "saturation_std": float(np.std(sat)),
        "saturation_p95": float(np.percentile(sat, 95)),
    }


def color_entropy(img: np.ndarray, bins: int = 8) -> float:
    q = np.clip((img * bins).astype(np.int32), 0, bins - 1)
    idx = q[:, :, 0] * bins * bins + q[:, :, 1] * bins + q[:, :, 2]
    hist = np.bincount(idx.reshape(-1), minlength=bins ** 3).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(bins ** 3))


def image_metrics(path: Path, source: np.ndarray, source_tv: float) -> dict[str, float]:
    img = load_rgb(path)
    diff = img - source
    mse = float(np.mean(diff ** 2))
    tv = total_variation(img)
    sat = saturation_stats(img)
    return {
        "mae_vs_source": float(np.mean(np.abs(diff))),
        "mse_vs_source": mse,
        "rmse_vs_source": float(math.sqrt(max(mse, 0.0))),
        "linf_vs_source": float(np.max(np.abs(diff))),
        "tv": tv,
        "tv_ratio": float(tv / max(source_tv, 1e-8)),
        "hf_energy": high_frequency_energy(img),
        "color_entropy": color_entropy(img),
        **sat,
    }


def percentile(values: list[float], value: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(100.0 * np.mean(arr <= value))


def annotate_pool(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = ["tv", "tv_ratio", "hf_energy", "saturation_p95", "color_entropy"]
    stats: dict[str, Any] = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in records]
        stats[key] = {
            "p10": float(np.percentile(vals, 10)),
            "p20": float(np.percentile(vals, 20)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "mean": float(np.mean(vals)),
        }

    for rec in records:
        rec["tv_percentile"] = percentile([float(r["tv"]) for r in records], float(rec["tv"]))
        rec["hf_percentile"] = percentile([float(r["hf_energy"]) for r in records], float(rec["hf_energy"]))
        rec["sat_p95_percentile"] = percentile(
            [float(r["saturation_p95"]) for r in records], float(rec["saturation_p95"])
        )
        rec["entropy_percentile"] = percentile(
            [float(r["color_entropy"]) for r in records], float(rec["color_entropy"])
        )
        rec["natural_gate"] = bool(
            rec["tv_percentile"] <= 90.0
            and rec["hf_percentile"] <= 90.0
            and rec["sat_p95_percentile"] <= 95.0
            and rec["entropy_percentile"] >= 5.0
        )
        rec["natural_score"] = float(
            100.0
            - np.mean([rec["tv_percentile"], rec["hf_percentile"], rec["sat_p95_percentile"]])
        )
    return stats


def load_sd_pipeline(model_dir: Path, device: str):
    import torch
    from diffusers import DDPMScheduler, StableDiffusionPipeline

    sf = next((p for p in model_dir.iterdir() if p.suffix == ".safetensors"), None)
    if sf is not None:
        pipe = StableDiffusionPipeline.from_single_file(str(sf), config=str(model_dir), torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.vae.eval()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    return pipe


def get_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_contact_sheet(records: list[dict[str, Any]], output: Path, *, title: str, max_items: int = 64) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(records, key=lambda r: float(r.get("natural_score", 0.0)), reverse=True)[:max_items]
    if not items:
        return
    thumb = 128
    label_h = 54
    gap = 8
    cols = min(8, max(1, math.ceil(math.sqrt(len(items)))))
    rows = math.ceil(len(items) / cols)
    title_h = 34
    w = cols * thumb + (cols + 1) * gap
    h = title_h + rows * (thumb + label_h) + (rows + 1) * gap
    canvas = Image.new("RGB", (w, h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = get_font(11)
    font_sm = get_font(9)
    draw.text((gap, 9), title, fill=(235, 235, 235), font=get_font(14))
    resample = getattr(Image, "Resampling", Image).LANCZOS

    for idx, rec in enumerate(items):
        row, col = divmod(idx, cols)
        x = gap + col * (thumb + gap)
        y = title_h + gap + row * (thumb + label_h + gap)
        img = Image.open(resolve(rec["path"])).convert("RGB")
        img = ImageOps.fit(img, (thumb, thumb), method=resample)
        canvas.paste(img, (x, y))
        label = f"{rec['name']}"
        label2 = f"score={rec['natural_score']:.1f} gate={int(rec['natural_gate'])}"
        label3 = f"TVp={rec['tv_percentile']:.0f} HFp={rec['hf_percentile']:.0f}"
        draw.text((x, y + thumb + 4), label[:20], fill=(235, 235, 235), font=font)
        draw.text((x, y + thumb + 21), label2, fill=(190, 210, 255), font=font_sm)
        draw.text((x, y + thumb + 36), label3, fill=(190, 190, 190), font=font_sm)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a diffusion-natural texture pool.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SD_MODEL)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sheet-max", type=int, default=64)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU", flush=True)
        args.device = "cpu"

    output_dir = resolve(args.output_dir)
    tex_dir = output_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve(args.source)
    model_dir = resolve(args.model_dir)
    source = load_rgb(source_path)
    source_tv = total_variation(source)
    catalog = prompt_catalog()

    print(json.dumps({
        "output_dir": rel(output_dir),
        "source": rel(source_path),
        "model_dir": str(model_dir),
        "num_samples": args.num_samples,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }, indent=2), flush=True)

    t0 = time.time()
    pipe = load_sd_pipeline(model_dir, args.device)
    records: list[dict[str, Any]] = []

    for idx in range(args.num_samples):
        prompt_info = catalog[idx % len(catalog)]
        sample_seed = int(args.seed + idx)
        generator = torch.Generator(device=args.device).manual_seed(sample_seed)
        name = f"s7_{idx:04d}_{prompt_info['family']}_seed{sample_seed}"
        out_path = tex_dir / f"{name}.png"
        print(f"[{idx + 1}/{args.num_samples}] {name} family={prompt_info['family']}", flush=True)

        with torch.inference_mode():
            image = pipe(
                prompt=prompt_info["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
        image.convert("RGB").resize((512, 512)).save(out_path)
        metrics = image_metrics(out_path, source, source_tv)
        rec = {
            "name": name,
            "path": rel(out_path),
            "shell": f"natural_{prompt_info['family']}",
            "dataset_kind": "natural_texture_pool",
            "family": prompt_info["family"],
            "palette": prompt_info["palette"],
            "prompt": prompt_info["prompt"],
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": sample_seed,
            "model_path": str(model_dir),
            "scheduler": "DDPMScheduler",
            "num_steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "width": int(args.width),
            "height": int(args.height),
            **metrics,
        }
        records.append(rec)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pool_stats = annotate_pool(records)
    records = sorted(records, key=lambda r: (not r["natural_gate"], -float(r["natural_score"]), r["name"]))

    source_metrics = {
        "tv": source_tv,
        "hf_energy": high_frequency_energy(source),
        "color_entropy": color_entropy(source),
        **saturation_stats(source),
    }
    manifest = {
        "dataset_kind": "natural_texture_pool",
        "created_at_unix": round(time.time(), 3),
        "elapsed_s": round(time.time() - t0, 1),
        "source": rel(source_path),
        "model_path": str(model_dir),
        "scheduler": "DDPMScheduler",
        "negative_prompt": NEGATIVE_PROMPT,
        "num_samples": len(records),
        "generation": {
            "seed_start": int(args.seed),
            "num_steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "width": int(args.width),
            "height": int(args.height),
        },
        "natural_gate": {
            "tv_percentile_max": 90.0,
            "hf_percentile_max": 90.0,
            "sat_p95_percentile_max": 95.0,
            "entropy_percentile_min": 5.0,
            "passed": sum(1 for r in records if r["natural_gate"]),
        },
        "source_metrics": source_metrics,
        "pool_stats": pool_stats,
        "prompt_catalog_size": len(catalog),
        "prompts": catalog,
        "candidates": records,
    }

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    sheet_path = output_dir / "contact_sheet_top.png"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "manifest": rel(manifest_path),
        "num_samples": len(records),
        "natural_gate_passed": manifest["natural_gate"]["passed"],
        "best_natural": [
            {
                "name": r["name"],
                "path": r["path"],
                "family": r["family"],
                "natural_score": r["natural_score"],
                "tv_percentile": r["tv_percentile"],
                "hf_percentile": r["hf_percentile"],
                "mse_vs_source": r["mse_vs_source"],
                "tv_ratio": r["tv_ratio"],
            }
            for r in records[:10]
        ],
        "elapsed_s": manifest["elapsed_s"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(records, sheet_path, title="Natural diffusion texture pool", max_items=args.sheet_max)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Manifest: {rel(manifest_path)}", flush=True)
    print(f"Contact sheet: {rel(sheet_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
