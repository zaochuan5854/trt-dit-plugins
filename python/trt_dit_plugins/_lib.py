# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Locate and load the bundled plugin shared libraries."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "comfy_kitchen"

# name -> (lib file, TRT plugin name). lib resolved at load time.
PLUGINS = {
    "int8_attention": "int8_attention",
    "adaln": "adaln",
    "rms_adaln": "rms_adaln",
    "apply_rope": "apply_rope",
    "rms_rope_split_half": "rms_rope_split_half",
    "stochastic_round_fp8": "stochastic_round_fp8",
}

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


def ensure_loaded() -> dict[str, ctypes.CDLL]:
    """dlopen kernels first, then plugins. Idempotent. No GPU needed."""
    global _loaded
    if not _loaded:
        d = libdir()
        _loaded["ck_kernels"] = _load_one(d, "ck_kernels")
        _loaded["trt_dit_plugins"] = _load_one(d, "trt_dit_plugins")
    return _loaded
