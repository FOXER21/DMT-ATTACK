#!/usr/bin/env python3
"""Generate one or more DDPM textures from a trained DMT checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from dmt.paths import ROOT, require_file, resolve, sd_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", type=int, default=None, help="Head index; defaults to score argmax.")
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sd-model", type=Path, default=sd_model_dir())
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--natural-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = require_file(resolve(args.checkpoint), "checkpoint")
    sd_dir = require_file(args.sd_model / "model_index.json", "Stable Diffusion model_index.json").parent

    import torch

    import dmt.diffusion.sampler as sd
    from dmt.training.train_generator import build_generator_model, token_from_prompt

    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    heads = int(payload.get("heads", config.get("heads", 8)))
    bottleneck = int(payload.get("bottleneck", config.get("bottleneck", 128)))
    hidden = int(payload.get("hidden", config.get("hidden", 512)))
    model = build_generator_model(1024, heads, bottleneck, hidden, 1024).to(args.device).eval()
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"warning: ignored unexpected keys: {unexpected[:5]}")
    pipe = sd.load_sd_pipeline(sd_dir, args.device)
    tokens, pooled = token_from_prompt(pipe, args.prompt, args.device)
    with torch.no_grad():
        z_all, scores, _gate = model(tokens, pooled)
    score_values = scores[0].detach().float().cpu().tolist()
    if args.all_heads:
        indices = list(range(heads))
    elif args.head is not None:
        if not 0 <= args.head < heads:
            raise ValueError(f"--head must be in [0, {heads})")
        indices = [args.head]
    else:
        indices = [int(np.argmax(score_values))]

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path = ROOT / "data" / "gates" / "naturalness.json"
    source_path = ROOT / "data" / "textures" / "tablecloth_gingham.png"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8")) if thresholds_path.is_file() else None
    source = None
    source_tv = None
    if source_path.is_file():
        source = np.asarray(sd.Image.open(source_path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
        source_tv = sd.total_variation(source)

    outputs = []
    for index in indices:
        z = z_all[0, index].detach().float().cpu().numpy()
        image = sd.generate_from_latent(
            pipe,
            args.prompt,
            sd.whiten_to_typical_set(z),
            guidance_scale=args.guidance_scale,
            num_steps=args.steps,
            device=args.device,
        )
        target = output if len(indices) == 1 else output.with_name(f"{output.stem}_head{index}{output.suffix or '.png'}")
        image.save(target)
        record = {"head": index, "score": score_values[index], "path": str(target), "natural_gate": None}
        if source is not None and source_tv is not None:
            metrics = sd.texture_metrics(image, source, source_tv)
            gate = None
            if thresholds is not None:
                gate, checks = sd.natural_gate(metrics, thresholds)
            record.update({"metrics": metrics, "natural_gate": gate, "natural_checks": checks if thresholds is not None else None})
            if args.natural_gate and gate is False:
                print(f"warning: head {index} failed naturalness gate", flush=True)
        outputs.append(record)

    report = {
        "checkpoint": str(checkpoint),
        "prompt": args.prompt,
        "heads": heads,
        "scores": score_values,
        "seed": args.seed,
        "outputs": outputs,
        "missing_model_keys": missing,
    }
    report_path = resolve(args.report) if args.report else output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
