#!/usr/bin/env python3
"""One-click tablecloth texture replacement for VLA adversarial attack.

Replaces the tablecloth texture in the LIBERO scene and renders comparison
views.  The tablecloth is a thin visual-only overlay on the table surface,
covering ~50% of the agentview pixels (vs 6% for the picture frame).

The tablecloth:
  - sits 0.1 mm above the table surface (does NOT block objects)
  - uses texuniform="true" + zero reflectance/specular (no rendering attenuation)
  - has conaffinity=0/contype=0 (no physics interaction)

Usage:
  # Visualize with a clean tablecloth
  python -m dmt.scene.tablecloth --image tablecloth.png --output-dir outputs/scenes/tablecloth

  # Replace with adversarial texture
  python -m dmt.scene.tablecloth --image data/textures/tablecloth_gingham.png --output-dir outputs/scenes/tablecloth_adv

  # Batch replace: apply all textures in a directory
  python -m dmt.scene.tablecloth --image-dir data/textures/ --output-dir outputs/scenes/tablecloth_batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dmt.paths import ROOT, base_scene_path

PROJECT_ROOT = ROOT
CONFIG_ROOT = ROOT / "config"
BASE_SCENE_PATH = base_scene_path()


def build_tablecloth_scene(
    base_scene: Path,
    texture_image: Path,
    output_xml: Path,
) -> dict:
    """Replace the table's cube-map texture with a 2D tablecloth texture.

    Instead of adding a separate geom (which causes z-fighting / floating),
    we modify the EXISTING table_visual material to use a 2D image with
    uniform mapping and zero reflectance.  This guarantees the cloth sits
    exactly on the table surface and appears in both agentview and wristview.

    Returns dict with texture and material info.
    """
    tree = ET.parse(str(base_scene.resolve()))
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Scene XML has no <asset> element")

    # Copy source texture
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tex_copy = output_xml.parent / f"tablecloth_tex{texture_image.suffix.lower()}"
    img = Image.open(texture_image).convert("RGB").resize((512, 512))
    img.save(tex_copy)

    # ---- Replace tex-table texture: cube map → 2D ----
    for tex_elem in asset.findall("texture"):
        if tex_elem.get("name") == "tex-table":
            # Remove cube-specific attributes
            for attr in ["type", "file", "width", "height", "gridsize",
                          "rgb1", "rgb2", "builtin"]:
                if attr in tex_elem.attrib:
                    del tex_elem.attrib[attr]
            tex_elem.set("type", "2d")
            tex_elem.set("file", str(tex_copy.resolve()))
            tex_elem.set("width", "512")
            tex_elem.set("height", "512")
            break

    # ---- Update table_texture material for flat rendering ----
    for mat_elem in asset.findall("material"):
        if mat_elem.get("name") == "table_texture":
            mat_elem.set("texture", "tex-table")
            mat_elem.set("reflectance", "0.0")
            mat_elem.set("specular", "0.0")
            mat_elem.set("shininess", "0.0")
            mat_elem.set("texrepeat", "1 1")
            mat_elem.set("texuniform", "true")
            break

    ET.indent(tree, space="  ")
    tree.write(str(output_xml), encoding="utf-8", xml_declaration=True)

    return {
        "texture_path": str(tex_copy.resolve()),
        "texture_name": "tex-table",
        "material_name": "table_texture",
        "scene_xml": str(output_xml.resolve()),
    }


def replace_tablecloth(texture_path: Path, output_dir: Path, base_scene: Path | None = None) -> Path:
    """Create a scene XML using ``texture_path`` and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / "tablecloth_scene.xml"
    build_tablecloth_scene(base_scene or BASE_SCENE_PATH, texture_path, output_xml)
    return output_xml


def render_scene(scene_xml: Path, output_dir: Path, suite: str = "libero_spatial",
                 task_id: int = 1, seed: int = 0) -> dict:
    """Render agentview and wristview from a MuJoCo scene."""
    from dmt.runtime.pi0_libero import ensure_runtime_environment, make_libero_envs

    ensure_runtime_environment()
    ec, envs = make_libero_envs(
        suite=suite, task_id=task_id, scene_xml=scene_xml,
        observation_width=256, observation_height=256, control_mode="relative")
    vec = envs[suite][1]
    obs, _ = vec.reset(seed=[seed])

    av = obs["pixels"]["image"]
    wv = obs["pixels"]["image2"]
    if av.ndim == 4: av = av[0]
    if wv.ndim == 4: wv = wv[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    av_path = output_dir / "agentview.png"
    wv_path = output_dir / "wristview.png"
    Image.fromarray(av).save(av_path)
    Image.fromarray(wv).save(wv_path)

    vec.close()
    return {
        "agentview": str(av_path.resolve()),
        "wristview": str(wv_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click tablecloth texture replacement for VLA attack.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Single image to use as tablecloth texture.")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Directory of images (batch replace all PNG/JPG).")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for scene XML and renders.")
    parser.add_argument("--base-scene", type=Path, default=BASE_SCENE_PATH,
                        help="Base scene XML to modify.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rendering (just generate scene XML).")
    args = parser.parse_args()

    images: list[Path] = []
    if args.image is not None:
        images = [args.image]
    elif args.image_dir is not None:
        img_dir = args.image_dir
        images = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    else:
        print("ERROR: --image or --image-dir required", file=sys.stderr)
        return 1

    results = []
    for img_path in images:
        name = img_path.stem
        out_dir = args.output_dir / name if args.image_dir else args.output_dir
        scene_xml = out_dir / f"tablecloth_scene.xml"

        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"  Output: {out_dir}")

        cloth_info = build_tablecloth_scene(args.base_scene, img_path, scene_xml)
        print(f"  Scene XML: {scene_xml}")

        render_info = {}
        if not args.no_render:
            print("  Rendering...")
            render_info = render_scene(scene_xml, out_dir)
            print(f"  Agentview: {Path(render_info['agentview']).name}")
            print(f"  Wristview: {Path(render_info['wristview']).name}")

        result = {**cloth_info, "render": render_info, "source_image": str(img_path)}
        results.append(result)

    # Summary
    summary_path = args.output_dir / "tablecloth_report.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_scene": str(args.base_scene.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "num_scenes": len(results),
        "results": [{k: str(v) if isinstance(v, Path) else v
                       for k, v in r.items()} for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDone. {len(results)} scenes generated. Report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
