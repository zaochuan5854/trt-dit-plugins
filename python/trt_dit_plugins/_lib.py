# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Locate and load the bundled plugin shared libraries.

pip-only install: CUDA runtime and TensorRT come from the
``nvidia-cuda-runtime-cu12`` and ``tensorrt-cu12`` wheels. They are preloaded
here by absolute path so no LD_LIBRARY_PATH / PATH setup is needed.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "dit-plugins"

# name -> (lib file, TRT plugin name). lib resolved at load time.
PLUGINS = {
    "int8_attention": "int8_attention",
    "adaln": "adaln",
    "rms_adaln": "rms_adaln",
    "apply_rope": "apply_rope",
    "rms_rope_split_half": "rms_rope_split_half",
    "stochastic_round_fp8": "stochastic_round_fp8",
    "block_sparse_sage2_attn": "block_sparse_sage2_attn",
}

# pip packages searched (first hit wins) for each dependency library.
_PKG_CANDIDATES = ("nvidia.cuda_runtime", "nvidia.cu12", "tensorrt_libs")
_LIB_NAMES = ("libcudart.so.12", "libnvinfer.so.10")
_LIB_NAMES_WIN = ("cudart64_12.dll", "nvinfer_10.dll")

_loaded: dict[str, ctypes.CDLL] = {}


def libdir() -> Path:
    """Directory containing libck_kernels / libtrt_dit_plugins."""
    env = os.environ.get("TRT_DIT_LIBDIR")
    if env:
        return Path(env)
    try:  # installed wheel: trt_dit_plugins/lib/
        from importlib.resources import files

        d = Path(str(files(__package__) / "lib"))
        if d.is_dir():
            return d
    except Exception:
        pass
    # source tree: ../build relative to this file
    d = Path(__file__).resolve().parent.parent.parent / "build"
    return d


def _load_one(directory: Path, stem: str) -> ctypes.CDLL:
    if sys.platform == "win32":
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(directory))
        for ext in (".dll", ".pyd"):
            p = directory / (stem + ext)
            if p.exists():
                return ctypes.CDLL(str(p))
        # also try bare .so name for MSYS-style builds
    names = [f"lib{stem}.so", f"{stem}.so", f"lib{stem}.dylib"]
    for n in names:
        p = directory / n
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        f"{stem} shared library not found in {directory} "
        "(set TRT_DIT_LIBDIR to override)"
    )


def _preload_deps() -> None:
    """CDLL each pip-provided dependency so ours resolve without PATH setup."""
    names = _LIB_NAMES_WIN if sys.platform == "win32" else _LIB_NAMES
    for pkg in _PKG_CANDIDATES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, AttributeError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        base = Path(next(iter(spec.submodule_search_locations)))
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            for sub in (base, base / "bin", base / "lib"):
                if sub.is_dir():
                    try:
                        os.add_dll_directory(str(sub))
                    except Exception:
                        pass
        for name in names:
            try:
                hit = next(base.rglob(name))
            except StopIteration:
                continue
            mode = getattr(ctypes, "RTLD_GLOBAL", 0)
            ctypes.CDLL(str(hit), mode=mode)


def ensure_loaded() -> dict[str, ctypes.CDLL]:
    """Preload pip deps, then dlopen kernels, then plugins. Idempotent."""
    global _loaded
    if not _loaded:
        _preload_deps()
        d = libdir()
        _loaded["ck_kernels"] = _load_one(d, "ck_kernels")
        _loaded["trt_dit_plugins"] = _load_one(d, "trt_dit_plugins")
    return _loaded
