#!/usr/bin/env python3
"""Materialize a portable LIBERO scene from the bundled XML template."""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from dmt.paths import ROOT, base_scene_path, env_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "config" / "scene" / "base_scene.template.xml")
    parser.add_argument("--libero-assets", type=Path, default=env_path("LIBERO_ASSETS_DIR"))
    parser.add_argument("--output", type=Path, default=base_scene_path())
    return parser.parse_args()


def asset_suffix(value: str) -> str | None:
    markers = ("/libero/libero/assets/", "/libero/assets/")
    for marker in markers:
        if marker in value:
            return value.split(marker, 1)[1]
    return None


def main() -> int:
    args = parse_args()
    template = args.template.resolve()
    assets = None if args.libero_assets is None else args.libero_assets.expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    if assets is None or not assets.is_dir():
        raise FileNotFoundError(
            "LIBERO assets directory is required; pass --libero-assets or set LIBERO_ASSETS_DIR"
        )

    tree = ET.parse(template)
    root = tree.getroot()
    rewrites = 0
    unresolved: list[str] = []
    for element in root.iter():
        value = element.attrib.get("file")
        if not value:
            continue
        suffix = asset_suffix(value)
        if suffix is None:
            if os.path.isabs(value) and not Path(value).is_file():
                unresolved.append(value)
            continue
        target = assets / suffix
        if not target.is_file():
            unresolved.append(str(target))
        element.set("file", str(target))
        rewrites += 1

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    report = {
        "template": str(template),
        "output": str(output),
        "libero_assets": str(assets),
        "rewritten_asset_paths": rewrites,
        "unresolved_paths": unresolved,
        "ok": not unresolved,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
