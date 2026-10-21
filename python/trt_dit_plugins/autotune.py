# SPDX-License-Identifier: Apache-2.0
"""Build-time W4A4 GEMM tile-config tuner.

The tuner measures the 7 CUTLASS int4 configs on dummy buffers, so it must
never run inside a TensorRT enqueue. Results are cached per (M, N, K, dtype)
and the plugin takes the index through ``gemm_config``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import _lib

_CACHE = Path(os.environ.get(
    "TRT_DIT_AUTOTUNE_CACHE",
    str(Path.home() / ".cache" / "trt_dit_plugins" / "convrot_w4a4.json"),
))
_DTYPE_CODE = {"float32": 0, "float16": 1, "bfloat16": 2}


def scratch_bytes(m: int, n: int, k: int, out_dtype: str = "bfloat16") -> int:
    """Bytes the tuner needs: A, B, D, xs, ws, plus alignment slack."""
    elem = 4 if out_dtype == "float32" else 2
    return m * k // 2 + n * k // 2 + m * n * elem + 4 * (m + n) + 4 * 256


def cached_config(m: int, n: int, k: int, out_dtype: str = "bfloat16") -> int | None:
    return _read().get(_key(m, n, k, out_dtype))


def autotune_convrot_w4a4(m: int, n: int, k: int, out_dtype: str = "bfloat16",
                          *, refresh: bool = False) -> int:
    """Measure and cache the best CUTLASS config for one W4A4 shape.

    Returns the config index, or ``-1`` when no CUTLASS tile fits (``N % 32
    != 0``); that is also the plugin's "automatic route", so the result can
    be passed straight to ``gemm_config`` and the legacy W4A8 fallback runs.
    """
    key = _key(m, n, k, out_dtype)
    cache = _read()
    if not refresh and key in cache:
        return int(cache[key])

    import torch

    if out_dtype not in _DTYPE_CODE:
        raise ValueError(f"out_dtype must be one of {sorted(_DTYPE_CODE)}")
    run = _lib.get_autotune()
    with torch.cuda.device(torch.cuda.current_device()):
        scratch = torch.empty(scratch_bytes(m, n, k, out_dtype), dtype=torch.int8,
                              device="cuda")
        cfg = run(m, n, k, _DTYPE_CODE[out_dtype], scratch)
    if cfg < 0:
        # CUTLASS int4 tiles need N % 32 == 0; shapes outside that still run
        # through the plugin's legacy fallback, so do not fail the tune.
        return -1
    cache[key] = cfg
    try:
        _write(cache)
    except OSError:
        pass  # measured result is still valid; the next call re-measures
    return cfg


def _key(m: int, n: int, k: int, out_dtype: str) -> str:
    # Configs are tile choices, so a cached index is only valid for this arch.
    import torch

    arch = "sm" + "".join(map(str, torch.cuda.get_device_capability()))
    return f"{m}x{n}x{k}/{out_dtype}/{arch}"


def _read() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _write(cache: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(_CACHE)
