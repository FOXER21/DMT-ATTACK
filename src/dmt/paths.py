"""Portable path and runtime helpers for the DMT package.

All paths are resolved from environment variables or the bundle root.  The
bundle deliberately keeps large policy weights and CUDA runtimes external.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# The package lives under ``src/dmt``; all user-facing paths are relative
# to the repository root rather than to the installed package directory.
ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return None if default is None else default.resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# ``path_from_env`` is the runtime-facing spelling retained as a small,
# explicit alias. Keeping both names makes call sites read naturally without
# maintaining a second path helper module.
path_from_env = env_path


def resolve_bundle_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "models":
        return model_root() / Path(*path.parts[1:])
    return resolve(path)


def model_root() -> Path:
    return env_path("DMT_MODEL_ROOT", ROOT / "models")  # type: ignore[return-value]


def sd_model_dir() -> Path:
    return env_path("DMT_SD_MODEL", model_root() / "stable-diffusion-2-1")  # type: ignore[return-value]


def pi05_model_dir() -> Path:
    return env_path("DMT_PI05_MODEL", model_root() / "pi05_libero")  # type: ignore[return-value]


def pi0_model_dir() -> Path:
    return env_path("DMT_PI0_MODEL", model_root() / "pi0_libero_finetuned_v044")  # type: ignore[return-value]


def tokenizer_dir() -> Path:
    return env_path("DMT_TOKENIZER", model_root() / "gemma-tokenizer")  # type: ignore[return-value]


def base_scene_path() -> Path:
    return env_path("DMT_BASE_SCENE", ROOT / "config" / "scene" / "base_scene.xml")  # type: ignore[return-value]


def scene_root() -> Path:
    return env_path("DMT_SCENE_ROOT", ROOT / "config" / "scene")  # type: ignore[return-value]


def insert_paths(paths: list[Path]) -> None:
    for path in reversed(paths):
        if not path.is_dir():
            continue
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def insert_existing_paths(paths: list[Path], *, front: bool = True) -> None:
    ordered = list(reversed(paths)) if front else list(paths)
    for path in ordered:
        if not path.is_dir():
            continue
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        if front:
            sys.path.insert(0, value)
        else:
            sys.path.append(value)


def configure_runtime() -> None:
    """Add optional bundled/runtime paths without assuming a workstation layout."""

    deps = env_path("DMT_EXTRA_DEPS")
    tools = env_path("DMT_TOOLS")
    lerobot = env_path("DMT_LEROBOT_SRC")
    insert_paths([p for p in (deps, tools, lerobot) if p is not None])
    if deps is not None:
        torch_lib = deps / "torch" / "lib"
        if torch_lib.is_dir():
            old = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{old}" if old else str(torch_lib)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def extra_python_paths() -> list[Path]:
    values = os.environ.get("DMT_EXTRA_PYTHONPATH", "")
    paths = [Path(item).expanduser().resolve() for item in values.split(os.pathsep) if item.strip()]
    paths.extend(
        path
        for path in (env_path("DMT_RUNTIME_DEPS"), env_path("DMT_EXTRA_DEPS"))
        if path is not None
    )
    return paths


def configure_bundle_pythonpath() -> None:
    """Add optional external runtime paths for policy evaluation."""

    insert_existing_paths(extra_python_paths(), front=False)


def libero_root() -> Path | None:
    explicit = env_path("DMT_LIBERO_ROOT")
    if explicit is not None:
        return explicit
    try:
        import libero.libero as libero_package
    except ModuleNotFoundError:
        return None
    return Path(libero_package.__file__).resolve().parent


configure_runtime()
