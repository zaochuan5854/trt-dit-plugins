# SPDX-License-Identifier: Apache-2.0
"""Tests for rpn helpers (torch-only; no GPU/TRT needed)."""
import math

import pytest

torch = pytest.importorskip("torch")

from trt_dit_plugins.rpn import compute_rpn_scales, fold_rpn_into_norms


def test_score_invariance():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 256, 128
    q = torch.randn(B, H, S, D, dtype=torch.float64)
    k = torch.randn(B, H, S, D, dtype=torch.float64)
    qn = torch.rand(D, dtype=torch.float64) * 0.5 + 0.75
    kn = torch.rand(D, dtype=torch.float64) * 0.5 + 0.75
    rpn = compute_rpn_scales(k)
    assert rpn.shape == (64,)
    assert bool((rpn > 0).all())
    qf, kf = fold_rpn_into_norms(qn, kn, rpn)
    s_ref = (q * qn) @ (k * kn).transpose(-1, -2)
    s_new = (q * qf.double()) @ (k * kf.double()).transpose(-1, -2)
    assert torch.allclose(s_ref, s_new, rtol=1e-9, atol=1e-9)


def test_rpn_bounds():
    torch.manual_seed(1)
    S, D = 512, 128
    k = torch.randn(S, D, dtype=torch.float32) * 3
    rpn = compute_rpn_scales(k)
    n = rpn
    pair = torch.hypot(k[:, :64], k[:, 64:])
    # After folding K by 1/n, every pair's max L2 norm is <= 1 (up to fp error).
    assert bool(((pair / n).amax(dim=0) <= 1.0 + 1e-5).all())


def test_fold_contract():
    qn = torch.ones(128)
    kn = torch.ones(128) * 2
    rpn = torch.arange(1, 65, dtype=torch.float32)
    qf, kf = fold_rpn_into_norms(qn, kn, rpn)
    assert torch.allclose(qf[:64], rpn)
    assert torch.allclose(qf[64:], rpn)
    assert torch.allclose(kf[:64], 2 / rpn)
    assert torch.allclose(kf[64:], 2 / rpn)
    with pytest.raises(ValueError):
        fold_rpn_into_norms(torch.ones(64), kn, rpn)
    with pytest.raises(ValueError):
        compute_rpn_scales(torch.ones(10, 64))
