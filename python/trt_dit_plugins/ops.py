# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Torch-friendly DiT ops. All run on CUDA via cached TRT engines.

Op names/shapes follow the upstream kernel projects (see NOTICE); eps defaults
match their torch ops. Engines are built once per (op, shapes, fields) and reused.
"""
from __future__ import annotations

from typing import Any

from . import _engine as E


def _require_cuda(*tensors: Any) -> Any:
    import torch

    dev = None
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError("all inputs must be CUDA tensors")
        dev = t.device if dev is None else dev
        if t.device != dev:
            raise ValueError("all inputs must share one CUDA device")
    return dev


def int8_attention(q, k, v):
    """INT8 Q/K/V attention. q,k,v: BF16/FP16/FP32 CUDA [B,H,S,D], D in (64,128,256).

    Output dtype: BF16 for FP32 input, else input dtype.
    """
    import torch

    _require_cuda(q, k, v)
    for t in (q, k, v):
        if t.dim() != 4 or t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("int8_attention expects 4D f32/f16/bf16 tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q,k,v shapes must match")
    otype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    outs = E.run_plugin(
        "int8_attention",
        {"q": q, "k": k, "v": v},
        [("o", otype, tuple(q.shape))],
    )
    return outs["o"]


def _adaln(name, x, scale, shift, eps=1e-6):
    _require_cuda(x, scale, shift)
    if not (x.dim() == 2 and scale.dim() == 2 and shift.dim() == 2):
        raise ValueError(f"{name} expects flat [N,D] inputs")
    outs = E.run_plugin(
        name,
        {"x": x, "scale": scale, "shift": shift},
        [("o", x.dtype, tuple(x.shape))],
        {"eps": float(eps)},
    )
    return outs["o"]


def adaln(x, scale, shift, eps=1e-6):
    """Fused LayerNorm AdaLN: layernorm(x) * (1 + scale) + shift."""
    return _adaln("adaln", x, scale, shift, eps)


def rms_adaln(x, scale, shift, eps=1e-6):
    """Fused RMSNorm AdaLN: rmsnorm(x) * (1 + scale) + shift."""
    return _adaln("rms_adaln", x, scale, shift, eps)


def apply_rope(q, k, freqs):
    """Interleaved RoPE. q,k: [B,H,S,D] f16/bf16; freqs: [..,D/2,2,2]."""
    _require_cuda(q, k, freqs)
    outs = E.run_plugin(
        "apply_rope",
        {"q": q, "k": k, "f": freqs},
        [("qo", q.dtype, tuple(q.shape)), ("ko", k.dtype, tuple(k.shape))],
    )
    return outs["qo"], outs["ko"]


def rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6, rot_dim=0):
    """RMSNorm (full D) + split-half RoPE (rot_dim prefix, 0 = all)."""
    _require_cuda(q, k, freqs, q_scale, k_scale)
    outs = E.run_plugin(
        "rms_rope_split_half",
        {"q": q, "k": k, "f": freqs, "qs": q_scale, "ks": k_scale},
        [("qo", q.dtype, tuple(q.shape)), ("ko", k.dtype, tuple(k.shape))],
        {"epsilon": float(epsilon), "rot_dim": int(rot_dim)},
    )
    return outs["qo"], outs["ko"]


def stochastic_rounding_fp8(x, rng, alias_rng=False):
    """Stochastically round to FP8 E4M3. x: f32/f16/bf16; rng: int32 same numel.

    alias_rng=True skips the rng staging copy (engine must enable the
    kALIASED_PLUGIN_IO_10_03 preview feature; not yet wired here).
    """
    _require_cuda(x, rng)
    import torch

    if rng.dtype != torch.int32 or rng.numel() != x.numel():
        raise ValueError("rng must be int32 with same numel as x")
    outs = E.run_plugin(
        "stochastic_round_fp8",
        {"x": x, "r": rng},
        [("o", torch.float8_e4m3fn, tuple(x.shape))],
        {"alias_rng": int(alias_rng)},
    )
    return outs["o"]


def block_sparse_sage2_attn(q, k, v, mask, scale=0.0, pvthreshd=50.0, attention_sink=0):
    """Block-sparse SageAttention2 (sm89; kernel from SpargeAttn, see NOTICE).
    q,k,v: [B,H,S,D] f16/bf16, D in (64,128), S % 128 == 0;
    mask: int32 [B,H,S//128,S//64] (all-ones = dense).

    scale=0 selects 1/sqrt(D). Returns o with q's shape/dtype.
    """
    _require_cuda(q, k, v, mask)
    import torch

    if q.dim() != 4 or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("block_sparse_sage2_attn expects 4D f16/bf16 q/k/v")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q,k,v shapes must match")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q,k,v dtypes must match")
    B, H, S, D = q.shape
    if mask.dtype != torch.int32 or tuple(mask.shape) != (B, H, S // 128, S // 64):
        raise ValueError("mask must be int32 [B,H,S//128,S//64]")
    outs = E.run_plugin(
        "block_sparse_sage2_attn",
        {"q": q, "k": k, "v": v, "m": mask},
        [("o", q.dtype, tuple(q.shape))],
        {"scale": float(scale), "pvthreshd": float(pvthreshd),
         "attention_sink": int(attention_sink)},
    )
    return outs["o"]
