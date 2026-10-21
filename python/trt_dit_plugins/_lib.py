# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Locate and load the bundled plugin shared libraries.

pip-only install: CUDA runtime and TensorRT come from the
``nvidia-cuda-runtime`` and ``tensorrt-cu13`` wheels. They are preloaded
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

# Filled from the TensorRT plugin registry by ensure_loaded(): one source of
# truth, no second hand-maintained op list. Empty until then.
from .select import PluginOp

PLUGINS: dict[PluginOp, str] = {}

# pip packages searched (first hit wins) for each dependency library.
_PKG_CANDIDATES = ("nvidia.cu13", "tensorrt_libs")
_LIB_NAMES = (
    "libcudart.so.13",
    "libnvinfer.so.11",
    "libnvonnxparser.so.11",
    "libnvinfer_plugin.so.11",
)
_LIB_NAMES_WIN = (
    "cudart64_13.dll",
    "nvinfer_11.dll",
    "nvonnxparser_11.dll",
    "nvinfer_plugin_11.dll",
)

_loaded: dict[str, ctypes.CDLL] = {}
_preloaded_deps: list[ctypes.CDLL] = []
_deps_preloaded: bool = False


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
    global _deps_preloaded
    if _deps_preloaded:
        return
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
            try:
                lib = ctypes.CDLL(str(hit), mode=mode)
                _preloaded_deps.append(lib)
            except OSError:
                pass
    _deps_preloaded = True


def ensure_loaded() -> dict[str, ctypes.CDLL]:
    """Preload pip deps, then dlopen kernels, then plugins. Idempotent."""
    global _loaded
    if not _loaded:
        _preload_deps()
        d = libdir()
        _loaded["ck_kernels"] = _load_one(d, "ck_kernels")
        _loaded["trt_dit_plugins"] = _load_one(d, "trt_dit_plugins")
        _refresh_plugins()
    return _loaded


def _refresh_plugins() -> None:
    """Rebuild PLUGINS from the registered creators (same enum names)."""
    try:
        import tensorrt as trt
    except ImportError:  # autotune-only use without TensorRT
        return

    names = {
        c.name
        for c in trt.get_plugin_registry().all_creators_recursive
        if c.plugin_namespace == PLUGIN_NAMESPACE
    }
    PLUGINS.clear()
    for op in PluginOp:
        if op.value in names:
            PLUGINS[op] = op.value


def get_autotune():
    """``convrot_int4_autotune(M, N, K, out_dtype, scratch)`` -> config index.

    Build-time only: the tuner measures kernels, so it must never run from a
    TRT enqueue. ``scratch`` is a CUDA int8 tensor sized by
    :func:`autotune.scratch_bytes`.
    """
    fn = ensure_loaded()["ck_kernels"].convrot_int4_autotune
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int

    def run(m: int, n: int, k: int, out_dtype: int, scratch) -> int:
        import torch

        return int(fn(m, n, k, out_dtype, scratch.data_ptr(), scratch.numel(),
                      torch.cuda.current_stream(scratch.device).cuda_stream))

    return run
