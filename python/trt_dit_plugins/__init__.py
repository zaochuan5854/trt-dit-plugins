# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""TensorRT DiT plugins with comfy-kitchen-compatible names.

GPUs only for execution; import itself needs neither GPU nor torch.
"""

from . import _lib as _lib
from ._lib import PLUGINS, ensure_loaded, libdir

__version__ = "0.1.0"
__all__ = [
    "PLUGIN_VERSION",
    "PLUGINS",
    "ensure_loaded",
    "libdir",
    "int8_attention",
    "adaln",
    "rms_adaln",
    "apply_rope",
    "rms_rope_split_half",
    "stochastic_rounding_fp8",
]

PLUGIN_VERSION = _lib.PLUGIN_VERSION


def __getattr__(name: str):
    if name in (
        "int8_attention",
        "adaln",
        "rms_adaln",
        "apply_rope",
        "rms_rope_split_half",
        "stochastic_rounding_fp8",
    ):
        from . import ops as _ops

        return getattr(_ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
