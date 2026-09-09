# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Single-plugin TensorRT engine cache + execution.

One engine per (plugin, fields, input signature). Engines stay cached,
contexts are created per call. All execution happens on the caller's
torch CUDA stream: no extra synchronisation.
"""
from __future__ import annotations

import struct
import threading
from typing import Any, Sequence

from . import _lib

_TORCH_TO_TRT: dict[str, Any] = {}
_cache: dict[tuple, Any] = {}
_lock = threading.Lock()


def _trt():
    import tensorrt as trt

    return trt


def torch_dtype_to_trt(dtype) -> Any:
    trt = _trt()
    key = str(dtype)
    if key not in _TORCH_TO_TRT:
        import torch

        mapping = {
            str(torch.float32): trt.DataType.FLOAT,
            str(torch.float16): trt.DataType.HALF,
            str(torch.bfloat16): trt.DataType.BF16,
            str(torch.int32): trt.DataType.INT32,
            str(torch.float8_e4m3fn): trt.DataType.FP8,
        }
        if key not in mapping:
            raise TypeError(f"unsupported dtype for TRT plugin I/O: {key}")
        _TORCH_TO_TRT[key] = mapping[key]
    return _TORCH_TO_TRT[key]


def trt_dtype_to_torch(dtype) -> Any:
    import torch

    trt = _trt()
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.FP8: torch.float8_e4m3fn,
    }
    return mapping[dtype]


class _Logger:
    def __init__(self) -> None:
        import tensorrt as trt

        class _L(trt.Logger):
            def log(self, severity, msg):  # noqa: N802
                if severity <= trt.Logger.WARNING:
                    print(f"[trt] {msg}")

        self.impl = _L()


_logger: _Logger | None = None


def _get_logger() -> Any:
    global _logger
    if _logger is None:
        _logger = _Logger()
    return _logger.impl


def _fields_key(fields: dict[str, Any]) -> tuple:
    out = []
    for k in sorted(fields):
        v = fields[k]
        out.append((k, tuple(v) if isinstance(v, (list, tuple)) else v))
    return tuple(out)


def _sig_key(
    name: str, fields: dict[str, Any], inputs: Sequence[tuple[str, Any, tuple[int, ...]]]
) -> tuple:
    return (
        name,
        _lib.PLUGIN_VERSION,
        _lib.PLUGIN_NAMESPACE,
        _fields_key(fields),
        tuple((n, str(d), tuple(s)) for n, d, s in inputs),
    )


def _build_engine(
    name: str,
    fields: dict[str, Any],
    inputs: Sequence[tuple[str, Any, tuple[int, ...]]],
    n_outputs: int,
    out_names: Sequence[str] | None = None,
) -> Any:
    """Build (or fetch from cache) a single-plugin engine."""
    trt = _trt()
    _lib.ensure_loaded()
    key = _sig_key(name, fields, inputs)
    with _lock:
        if key in _cache:
            return _cache[key]
        reg = trt.get_plugin_registry()
        creator = reg.get_creator(name, _lib.PLUGIN_VERSION, _lib.PLUGIN_NAMESPACE)
        if creator is None:
            raise RuntimeError(f"plugin creator not registered: {name}")
        pf = []
        for k in sorted(fields):
            v = fields[k]
            if isinstance(v, float):
                pf.append(
                    trt.PluginField(k, struct.pack("<f", v), trt.PluginFieldType.FLOAT32)
                )
            elif isinstance(v, int):
                pf.append(
                    trt.PluginField(k, struct.pack("<i", v), trt.PluginFieldType.INT32)
                )
            else:
                raise TypeError(f"unsupported plugin field type: {k}={v!r}")
        fc = trt.PluginFieldCollection(pf)
        plug = creator.create_plugin(name, fc, trt.TensorRTPhase.BUILD)
        builder = trt.Builder(_get_logger())
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        tensors = [network.add_input(n, d, s) for n, d, s in inputs]
        layer = network.add_plugin_v3(tensors, [], plug)
        names = list(out_names) if out_names else [f"out{i}" for i in range(n_outputs)]
        for i, oname in enumerate(names):
            layer.get_output(i).name = oname
            network.mark_output(layer.get_output(i))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        engine = builder.build_engine_with_config(network, config)
        if engine is None:
            raise RuntimeError(f"engine build failed: {name} {key}")
        _cache[key] = engine
        return engine


def run_plugin(
    name: str,
    tensors: dict[str, Any],
    out_specs: Sequence[tuple[str, Any, tuple[int, ...]]],
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one plugin layer end to end. Returns {out_name: torch.Tensor}.

    tensors: {input_name: torch CUDA tensor} (insertion order = plugin input
    order — do NOT sort: positions matter, e.g. q before k).
    out_specs: [(out_name, torch dtype, shape)].
    """
    import torch

    fields = fields or {}
    names = list(tensors)  # insertion order, positional for add_plugin_v3
    first = tensors[names[0]]
    if not first.is_cuda:
        raise ValueError("plugin inputs must be CUDA tensors")
    device = first.device
    for n in names:
        t = tensors[n]
        if not t.is_cuda or t.device != device or not t.is_contiguous():
            raise ValueError(f"input {n}: same CUDA device + contiguous required")
    in_sig = [(n, torch_dtype_to_trt(tensors[n].dtype), tuple(tensors[n].shape)) for n in names]
    out_names = [o[0] for o in out_specs]
    engine = _build_engine(name, fields, in_sig, len(out_specs), out_names)
    ctx = engine.create_execution_context()
    for n in names:
        ctx.set_tensor_address(n, tensors[n].data_ptr())
    outs: dict[str, Any] = {}
    for oname, odtype, oshape in out_specs:
        o = torch.empty(oshape, dtype=odtype, device=device)
        ctx.set_tensor_address(oname, o.data_ptr())
        outs[oname] = o
    stream = torch.cuda.current_stream(device).cuda_stream
    if not ctx.execute_async_v3(stream):
        raise RuntimeError(f"enqueue failed: {name}")
    return outs
