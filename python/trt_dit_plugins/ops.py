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


def _attn_out_shape(q, out_layout):
    if out_layout == "bhsd":
        return tuple(q.shape)
    if out_layout == "bsc":
        b, h, s, d = q.shape
        return (b, s, h * d)
    raise ValueError('out_layout must be "bhsd" or "bsc"')


def int8_attention(q, k, v, *, out_layout="bhsd"):
    """INT8 Q/K/V attention. q,k,v: BF16/FP16/FP32 CUDA [B,H,S,D], D in (64,128,256).

    out_layout="bsc" returns [B,S,H*D] (attention epilogue store, no Transpose).
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
        [("o", otype, _attn_out_shape(q, out_layout))],
        {"out_layout": int(out_layout == "bsc")},
    )
    return outs["o"]


def sage_attn(q, k, v, fp8_pv=False, *, out_layout="bhsd"):
    """Plain dense SageAttention. q: BF16/FP16/FP32 CUDA [B,Hq,Sq,D];
    k,v: same dtype, [B,Hkv,Sk,D] with Hq % Hkv == 0 (GQA ok), D in
    (64,128,256). FP8 inputs are rejected (no kernel consumes them natively).

    Tactic: portable FP16-PV path everywhere (sm80/86/89/90/100/120).
    fp8_pv=True opts into the FP8-PV dense SageAttention2 path, valid only
    on sm89 with D in (64,128), square S % 128 == 0 and FP16/BF16 inputs
    (measured slower than tactic0 for dense shapes; opt-in for experiments).
    out_layout="bsc" returns [B,Sq,Hq*D].
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
        [("o", otype, _attn_out_shape(q, out_layout))],
        {"fp8_pv": int(bool(fp8_pv)), "out_layout": int(out_layout == "bsc")},
    )
    return outs["o"]


def _adaln(name, x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
           profile=None):
    _require_cuda(x, scale, shift)
    if x.dim() not in (2, 3) or scale.dim() not in (2, 3) or shift.dim() not in (2, 3):
        raise ValueError(f"{name} expects x [N,D]/[B,S,D] and scale/shift [R,D]/[B,1,D]")
    d = x.shape[-1]
    if scale.shape[-1] != d or shift.shape[-1] != d:
        raise ValueError(f"{name} last-dim mismatch")
    if scale.dtype != shift.dtype:
        raise ValueError(f"{name} scale/shift dtype mismatch")
    mod_dtype = scale.dtype
    tensors = {"x": x, "scale": scale, "shift": shift}
    fields = {"eps": float(eps), "has_weight": int(weight is not None),
              "residual_gate": int(residual is not None)}
    if weight is not None:
        if weight.dim() != 1 or weight.numel() != d or weight.dtype != mod_dtype:
            raise ValueError("weight must be [D] in the modulation dtype")
        tensors["weight"] = weight
    if residual is not None:
        if gate is None or residual.shape != x.shape or residual.dtype != x.dtype:
            raise ValueError("residual needs gate and the same shape/dtype as x")
        if gate.dim() not in (2, 3) or gate.shape[-1] != d or gate.dtype != mod_dtype:
            raise ValueError("gate must be [B,D]/[B,1,D] in the modulation dtype")
        tensors["residual"] = residual
        tensors["gate"] = gate
    outs = E.run_plugin(
        name, tensors, [("o", x.dtype, tuple(x.shape))], fields, profile=profile)
    return outs["o"]


def rms_adaln(x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
              profile=None):
    """rmsnorm(x) * (1 + scale) + shift, optional weight / residual+gate."""
    return _adaln(PluginOp.RMS_ADALN, x, scale, shift, eps, weight, residual, gate,
                  profile)


def layer_norm_adaln(x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
                     profile=None):
    """layernorm(x) * (1 + scale) + shift, optional weight / residual+gate."""
    return _adaln(PluginOp.LAYER_NORM_ADALN, x, scale, shift, eps, weight, residual,
                  gate, profile)


def apply_rope(q, k, freqs):
    """Interleaved RoPE. q,k: [B,H,S,D] f16/bf16; freqs: [..,D/2,2,2]."""
    _require_cuda(q, k, freqs)
    outs = E.run_plugin(
        PluginOp.APPLY_ROPE,
        {"q": q, "k": k, "f": freqs},
        [("qo", q.dtype, tuple(q.shape)), ("ko", k.dtype, tuple(k.shape))],
    )
    return outs["qo"], outs["ko"]


def rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6, rot_dim=0,
                        profile=None):
    """RMSNorm (full D) + split-half RoPE (rot_dim prefix, 0 = all).

    ``rot_dim=-1`` skips rotation entirely (RMS-only); ``freqs`` is still
    required by this pair op and may be a 1x1x1x1x2x2 dummy. Use
    ``rms_norm`` instead when q and k have different S (cross-attention).

    ``profile``: {"q": (min, opt, max), ...} makes B/S dynamic, same form as
    ``_engine.run_plugin``.
    """
    _require_cuda(q, k, freqs, q_scale, k_scale)
    outs = E.run_plugin(
        PluginOp.RMS_ROPE_SPLIT_HALF,
        {"q": q, "k": k, "f": freqs, "qs": q_scale, "ks": k_scale},
        [("qo", q.dtype, tuple(q.shape)), ("ko", k.dtype, tuple(k.shape))],
        {"epsilon": float(epsilon), "rot_dim": int(rot_dim)},
        profile=profile,
    )
    return outs["qo"], outs["ko"]


def rms_norm(x, weight, eps=1e-6, layout="bhsd", profile=None):
    """RMSNorm with optional BSHD input and BHSD output.

    layout="bshd": x [B,S,H,D] -> out [B,H,S,D] (fuses the q/k transpose)
    layout="bhsd": x [B,H,S,D] -> out [B,H,S,D]
    weight: [D] in FP32/FP16/BF16 (kernel reads it as the per-dim scale).
    """
    _require_cuda(x, weight)
    if x.dim() != 4 or weight.dim() != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("rms_norm expects x [B,S,H,D] / [B,H,S,D] and weight [D]")
    if layout not in ("bhsd", "bshd"):
        raise ValueError("layout must be 'bhsd' or 'bshd'")
    if layout == "bshd":
        b, s, h, d = x.shape
        oshape = (b, h, s, d)
    else:
        oshape = tuple(x.shape)
    outs = E.run_plugin(
        PluginOp.RMS_NORM,
        {"x": x, "w": weight},
        [("y", x.dtype, oshape)],
        {"eps": float(eps), "in_layout": int(layout == "bshd")},
        profile=profile,
    )
    return outs["y"]


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


def block_sparse_sage2_attn(q, k, v, mask, scale=0.0, pvthreshd=50.0,
                            attention_sink=0, *, out_layout="bhsd"):
    """Block-sparse SageAttention2 (sm89; kernel from SpargeAttn, see NOTICE).
    q,k,v: [B,H,S,D] f16/bf16, D in (64,128), S % 128 == 0;
    mask: int32 [B,H,S//128,S//64] (all-ones = dense).

    scale=0 selects 1/sqrt(D). out_layout="bsc" returns [B,S,H*D].
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
        [("o", q.dtype, _attn_out_shape(q, out_layout))],
        {"scale": float(scale), "pvthreshd": float(pvthreshd),
         "attention_sink": int(attention_sink),
         "out_layout": int(out_layout == "bsc")},
    )
    return outs["o"]


def fused_int8_rope_sage_attn(q_i8, q_scale, k_i8, k_scale, v_i8, v_scale,
                               rms_w_q, rms_w_k, inv_freq, *, out_layout="bhsd"):
    """RoPE+SageAttn fused DiT-self (D=128 only).

    q_i8/k_i8/v_i8: int8 or f32/f16/bf16 CUDA [B,H,S,128] (identical shapes
    and dtype). INT8 inputs are dequantized with the per-tensor scales; float
    inputs skip that stage (scales unused). q_scale/k_scale/v_scale:
    single-element fp32 CUDA tensors. rms_w_q/rms_w_k: fp32/bf16 CUDA [128]
    (same dtype). inv_freq: fp32 CUDA [64]. out_layout="bsc" returns
    bf16 [B,S,H*128]; default bf16 [B,H,S,128].
    """
    _require_cuda(q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q, rms_w_k,
                  inv_freq)
    import torch

    dtypes = (torch.int8, torch.float32, torch.float16, torch.bfloat16)
    for t in (q_i8, k_i8, v_i8):
        if t.dim() != 4 or t.dtype not in dtypes or t.shape[-1] != 128:
            raise ValueError("fused q/k/v must be 4D int8/f32/f16/bf16 with D=128")
    if not (q_i8.shape == k_i8.shape == v_i8.shape):
        raise ValueError("q,k,v shapes must match (DiT-self)")
    if not (q_i8.dtype == k_i8.dtype == v_i8.dtype):
        raise ValueError("fused q,k,v must share one dtype")
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
        [("o", torch.bfloat16, _attn_out_shape(q_i8, out_layout))],
        {"out_layout": int(out_layout == "bsc")},
    )
    return outs["o"]


def fused_rope_sage_attn(q, k, v, rms_w_q, rms_w_k, inv_freq, *, out_layout="bhsd"):
    """Float-input variant: linear output feeds the fused op directly.

    q/k/v f32/f16/bf16 [B,H,S,128] (no QuantizeLinear/calibration needed).
    """
    import torch

    one = torch.ones(1, device=q.device, dtype=torch.float32)
    return fused_int8_rope_sage_attn(q, one, k, one, v, one, rms_w_q, rms_w_k,
                                     inv_freq, out_layout=out_layout)


_EPILOGUE = {"none": 0, "gelu": 1, "silu": 2}


def _convrot_linear(op, x, weight, weight_scale, bias, out_dtype, gemm_config, w4a4,
                    epilogue="none", residual=None, gate=None, out_layout="bsc",
                    qkv_heads=0, head_dim=0, profile=None):
    import torch

    _require_cuda(x, weight, weight_scale, *(() if bias is None else (bias,)))
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("convrot_linear expects f32/f16/bf16 x")
    if weight.dim() != 2 or weight.dtype != torch.int8:
        raise ValueError("weight must be a 2D int8 tensor")
    if weight_scale.dtype != torch.float32 or tuple(weight_scale.shape) != (weight.shape[0],):
        raise ValueError("weight_scale must be fp32 [N]")
    k = weight.shape[1] * (2 if w4a4 else 1)
    if x.shape[-1] != k:
        raise ValueError(f"x K={x.shape[-1]} does not match weight K={k}")
    if k % 256 != 0 or k > 16384:
        raise ValueError("K must be a multiple of 256 and <= 16384")
    if not w4a4 and weight.shape[0] % 4 != 0:
        raise ValueError("N must be a multiple of 4")
    if out_dtype is None:
        otype = x.dtype
    elif out_dtype in (torch.float32, torch.float16, torch.bfloat16):
        otype = out_dtype
    else:
        raise ValueError("out_dtype must be float32/float16/bfloat16 or None")
    if bias is None:
        has_bias = 0
        bias = torch.zeros(weight.shape[0], dtype=otype, device=x.device)
    elif bias.dtype != otype or tuple(bias.shape) != (weight.shape[0],):
        raise ValueError("bias must be [N] in the output dtype")
    else:
        has_bias = 1
    if isinstance(epilogue, str):
        if epilogue not in _EPILOGUE:
            raise ValueError("epilogue must be 'none', 'gelu' or 'silu'")
        ep = _EPILOGUE[epilogue]
    elif epilogue in (0, 1, 2):
        ep = int(epilogue)
    else:
        raise ValueError("epilogue must be 'none', 'gelu' or 'silu'")
    if w4a4 and (ep or residual is not None or gate is not None):
        raise ValueError("W4A4 does not support epilogue/residual/gate")
    if out_layout not in ("bsc", "bhsd"):
        raise ValueError("out_layout must be 'bsc' or 'bhsd'")
    if w4a4 and (out_layout != "bsc" or qkv_heads or head_dim):
        raise ValueError("W4A4 does not support out_layout/qkv_heads/head_dim")
    if head_dim < 0 or qkv_heads < 0:
        raise ValueError("qkv_heads and head_dim must be >= 0")
    tensors = {"x": x, "w": weight, "ws": weight_scale, "b": bias}
    if residual is not None:
        n = weight.shape[0]
        if x.dim() != 3 or gate is None or residual.dim() != 3 \
                or tuple(residual.shape[:2]) != tuple(x.shape[:2]) \
                or residual.shape[-1] != n or residual.dtype != otype \
                or not residual.is_cuda:
            raise ValueError("residual must be [B,S,N] in the output dtype")
        if gate.dtype != otype or gate.dim() not in (2, 3) \
                or gate.shape[-1] != n or gate.shape[0] != x.shape[0] \
                or (gate.dim() == 3 and gate.shape[1] != 1):
            raise ValueError("gate must be [B,N] or [B,1,N] in the output dtype")
        tensors["residual"] = residual
        tensors["gate"] = gate
    n = weight.shape[0]
    if out_layout == "bhsd":
        if x.dim() != 3:
            raise ValueError("bhsd output requires x [B,S,K]")
        if ep or residual is not None or gate is not None:
            raise ValueError("bhsd output does not support epilogue/residual")
        D = head_dim or 128
        H = qkv_heads or n // D
        if D <= 0 or H <= 0 or H * D * (3 if qkv_heads else 1) != n:
            raise ValueError("weight N must be H*D or 3*H*D")
        b, s_, _ = x.shape
        names = ["q", "k", "v"] if qkv_heads else ["y"]
        out_specs = [(name, otype, (b, H, s_, D)) for name in names]
    else:
        names = ["y"]
        out_specs = [("y", otype, (*x.shape[:-1], n))]
    outs = E.run_plugin(
        op,
        tensors,
        out_specs,
        {"group_size": 256, "has_bias": has_bias,
         "out_dtype": -1 if out_dtype is None
         else {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[otype],
         "gemm_config": int(gemm_config), "epilogue": ep,
         "residual_gate": int(residual is not None), "out_layout": 1 if out_layout == "bhsd" else 0,
         "qkv_heads": int(qkv_heads), "head_dim": int(head_dim)},
        profile=profile,
    )
    return tuple(outs[name] for name in names) if qkv_heads else outs["y"]


def convrot_linear(x, weight, weight_scale, bias=None, *, out_dtype=None, gemm_config=-1,
                   epilogue="none", residual=None, gate=None, out_layout="bsc",
                   qkv_heads=0, head_dim=0, profile=None):
    """ConvRot W8A8 linear: y = x @ W.T + bias, optional GELU/SiLU epilogue.

    x: f32/f16/bf16 CUDA [..., K]; weight: int8 [N, K] already ConvRot-rotated;
    weight_scale: fp32 [N]; bias: [N] in the output dtype (None = zeros, the
    kernel still runs the bias epilogue with has_bias=0).
    ``epilogue``: "none" | "gelu" (exact erf) | "silu".
    ``residual``/``gate``: optional [B,S,N] and [B,N]/[B,1,N] in the output
    dtype; result is residual + gate * epilogue(x @ W.T + bias).
    ``profile``: {"x": (min, opt, max)} makes the M dim dynamic so one engine
    serves many token counts (same form as ``_engine.run_plugin``).
    """
    return _convrot_linear(PluginOp.CONVROT_LINEAR, x, weight, weight_scale, bias,
                           out_dtype, gemm_config, False, epilogue, residual, gate,
                           out_layout, qkv_heads, head_dim, profile)


def convrot_linear_w4a4(x, qweight, wscales, bias=None, *, out_dtype=None, gemm_config=-1,
                        epilogue="none", out_layout="bsc", qkv_heads=0, head_dim=0,
                        profile=None):
    """ConvRot W4A4 linear: packed signed INT4 weight [N, K/2], fp32 scales [N].

    ``gemm_config``: -1 = automatic route, else a pinned sm8x tile config.
    ``profile``: {"x": (min, opt, max)} makes the M dim dynamic.
    """
    return _convrot_linear(PluginOp.CONVROT_LINEAR_W4A4, x, qweight, wscales, bias,
                           out_dtype, gemm_config, True, epilogue, None, None, out_layout,
                           qkv_heads, head_dim, profile)
