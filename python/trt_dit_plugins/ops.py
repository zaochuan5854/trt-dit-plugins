# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""Torch-friendly DiT ops. All run on CUDA via cached TRT engines.

Op names/shapes follow the upstream kernel projects (see NOTICE); eps defaults
match their torch ops. Engines are built once per (op, shapes, fields) and reused.
"""
from __future__ import annotations

from typing import Any

from . import _engine as E
from .select import PluginOp


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
        PluginOp.INT8_ATTENTION,
        {"q": q, "k": k, "v": v},
        [("o", otype, tuple(q.shape))],
    )
    return outs["o"]


def sage_attn(q, k, v, fp8_pv=False):
    """Plain dense SageAttention. q: BF16/FP16/FP32 CUDA [B,Hq,Sq,D];
    k,v: same dtype, [B,Hkv,Sk,D] with Hq % Hkv == 0 (GQA ok), D in
    (64,128,256). FP8 inputs are rejected (no kernel consumes them natively).

    Tactic: portable FP16-PV path everywhere (sm80/86/89/90/100/120).
    fp8_pv=True opts into the FP8-PV dense SageAttention2 path, valid only
    on sm89 with D in (64,128), square S % 128 == 0 and FP16/BF16 inputs
    (measured slower than tactic0 for dense shapes; opt-in for experiments).
    Output dtype: BF16 for FP32 input, else input dtype.
    """
    import torch

    _require_cuda(q, k, v)
    for t in (q, k, v):
        if t.dim() != 4 or t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("sage_attn expects 4D f32/f16/bf16 tensors")
    if not (q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3]):
        raise ValueError("sage_attn q/k/v must share B and D")
    if not (k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2]):
        raise ValueError("sage_attn k/v shapes must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("sage_attn Hq must be a multiple of Hkv")
    otype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    outs = E.run_plugin(
        PluginOp.SAGE_ATTN,
        {"q": q, "k": k, "v": v},
        [("o", otype, tuple(q.shape))],
        {"fp8_pv": int(bool(fp8_pv))},
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
    return _adaln(PluginOp.ADALN, x, scale, shift, eps)


def rms_adaln(x, scale, shift, eps=1e-6):
    """Fused RMSNorm AdaLN: rmsnorm(x) * (1 + scale) + shift."""
    return _adaln(PluginOp.RMS_ADALN, x, scale, shift, eps)


def apply_rope(q, k, freqs):
    """Interleaved RoPE. q,k: [B,H,S,D] f16/bf16; freqs: [..,D/2,2,2]."""
    _require_cuda(q, k, freqs)
    outs = E.run_plugin(
        PluginOp.APPLY_ROPE,
        {"q": q, "k": k, "f": freqs},
        [("qo", q.dtype, tuple(q.shape)), ("ko", k.dtype, tuple(k.shape))],
    )
    return outs["qo"], outs["ko"]


def rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6, rot_dim=0):
    """RMSNorm (full D) + split-half RoPE (rot_dim prefix, 0 = all)."""
    _require_cuda(q, k, freqs, q_scale, k_scale)
    outs = E.run_plugin(
        PluginOp.RMS_ROPE_SPLIT_HALF,
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
        PluginOp.STOCHASTIC_ROUND_FP8,
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
        PluginOp.BLOCK_SPARSE_SAGE2_ATTN,
        {"q": q, "k": k, "v": v, "m": mask},
        [("o", q.dtype, tuple(q.shape))],
        {"scale": float(scale), "pvthreshd": float(pvthreshd),
         "attention_sink": int(attention_sink)},
    )
    return outs["o"]


def fused_int8_rope_sage_attn(q_i8, q_scale, k_i8, k_scale, v_i8, v_scale,
                               rms_w_q, rms_w_k, inv_freq):
    """INT8-input RoPE+SageAttn fused DiT-self (D=128 only, S>1024).

    q_i8/k_i8/v_i8: int8 CUDA [B,H,S,128] (identical shapes required).
    q_scale/k_scale/v_scale: per-tensor fp32 scalar (single-element CUDA tensors).
    rms_w_q/rms_w_k: fp32/bf16 CUDA [128] (same dtype). inv_freq: fp32 CUDA [64].
    Returns bf16 [B,H,S,128].
    """
    _require_cuda(q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q, rms_w_k,
                  inv_freq)
    import torch

    for t in (q_i8, k_i8, v_i8):
        if t.dim() != 4 or t.dtype != torch.int8 or t.shape[-1] != 128:
            raise ValueError("fused q/k/v must be 4D int8 with D=128")
    if not (q_i8.shape == k_i8.shape == v_i8.shape):
        raise ValueError("q,k,v shapes must match (DiT-self)")
    if q_i8.shape[2] <= 1024:
        raise ValueError("fused q/k/v need S>1024 (L<=1024 path unsupported)")
    for t in (q_scale, k_scale, v_scale):
        if t.dtype != torch.float32 or t.numel() != 1:
            raise ValueError("input scales must be single-element fp32")
    if rms_w_q.shape != (128,) or rms_w_k.shape != (128,) or rms_w_q.dtype != rms_w_k.dtype:
        raise ValueError("norm scales must be [128] with matching dtype")
    if inv_freq.shape != (64,) or inv_freq.dtype != torch.float32:
        raise ValueError("inv_freq must be fp32 [64]")
    outs = E.run_plugin(
        PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
        {"q": q_i8, "qs": q_scale, "k": k_i8, "ks": k_scale, "v": v_i8,
         "vs": v_scale, "qn": rms_w_q, "kn": rms_w_k, "inv": inv_freq},
        [("o", torch.bfloat16, tuple(q_i8.shape))],
    )
    return outs["o"]
