# SPDX-License-Identifier: Apache-2.0
"""FireQ-style RPN (RoPE-preserving normalization) helpers for DiT-self.

RPN (FireQ Sec.3.2.1, arXiv:2505.20839) bounds post-RoPE norms offline: per
split-half RoPE pair ``f``, ``n_f = max over rows of ||(k_f, k_{f+64})||_2``.
Dividing K pairs by ``n_f`` (folded into the K RMSNorm weights) and
multiplying Q pairs by ``n_f`` (folded into the Q RMSNorm weights) leaves
attention scores ``QK^T`` exactly invariant (diagonal ``N`` cancels), while
post-RoPE pair norms are bounded by construction (FireQ Thm 3.1).

Scope: Anima DiT-self only (D=128, split-half pairs ``(f, f+64)``,
non-causal). The fused kernel needs NO changes: folded weights flow through
the existing ``rms_w_q``/``rms_w_k`` inputs transparently. The online K-mean
(smooth-K) path is kept as-is; RPN does not replace mean subtraction (pair
norms vs channel bias are different mechanisms) until real-data validation.

Calibration input is Fp K activations (e.g. sampler trajectory); folding
targets are the per-channel RMSNorm weights. Importing this module needs
neither GPU nor torch (both imported lazily).
"""

from __future__ import annotations


def compute_rpn_scales(k_calib):
    """Per-pair RPN scales from calibration K.

    ``k_calib``: ``[S,128]`` or ``[B,H,S,128]`` (any float dtype, CPU or CUDA).
    Returns fp32 ``[64]`` with ``n_f = max_rows hypot(k_f, k_{f+64})``,
    clamped to ``>= 1e-12``.
    """
    import torch

    if not isinstance(k_calib, torch.Tensor) or k_calib.dim() not in (2, 4):
        raise ValueError("k_calib must be [S,128] or [B,H,S,128]")
    if k_calib.shape[-1] != 128:
        raise ValueError("RPN helper supports D=128 split-half only")
    k = k_calib.detach().to(torch.float32).reshape(-1, k_calib.shape[-1])
    pair = torch.hypot(k[:, :64], k[:, 64:])  # [N,64] split-half pairs
    n = pair.amax(dim=0).clamp_min(1e-12)
    return n


def fold_rpn_into_norms(rms_w_q, rms_w_k, rpn):
    """Fold pair scales into RMSNorm weights, score-invariant.

    ``rms_w_q``/``rms_w_k``: ``[128]`` (any float dtype/device, must match each
    other); ``rpn``: ``[64]`` from :func:`compute_rpn_scales`.
    Returns ``(rms_w_q_folded, rms_w_k_folded)`` with the same dtype/device:
    K pair channels ``/= n_f``, Q pair channels ``*= n_f``.
    """
    import torch

    for t in (rms_w_q, rms_w_k):
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != (128,):
            raise ValueError("norm weights must be [128]")
    if not isinstance(rpn, torch.Tensor) or tuple(rpn.shape) != (64,):
        raise ValueError("rpn must be [64]")
    if rms_w_q.dtype != rms_w_k.dtype:
        raise ValueError("rms_w_q/rms_w_k dtypes must match")
    work = torch.float64 if rms_w_q.dtype == torch.float64 else torch.float32
    n = rpn.to(dtype=work, device=rms_w_q.device)
    pair_scale = torch.cat([n, n])  # [128]: same scale for both pair halves
    k_folded = (rms_w_k.to(work) / pair_scale).to(rms_w_q.dtype)
    q_folded = (rms_w_q.to(work) * pair_scale).to(rms_w_q.dtype)
    return q_folded, k_folded
