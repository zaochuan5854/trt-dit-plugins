# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
"""GPU smoke: python frontend vs eager reference. Needs torch+tensorrt+GPU."""

import torch
import torch.nn.functional as F

import trt_dit_plugins as tdp


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / a.norm() / b.norm())


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    B, H, S, D = 1, 8, 256, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    o = tdp.int8_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).bfloat16()
    c = cos(o, ref)
    print(f"int8_attention cos={c:.5f}", flush=True)
    assert c > 0.99, c

    N = 256
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.2
    sh = torch.randn(N, D, device="cuda", dtype=torch.bfloat16) * 0.2
    o = tdp.adaln(x, sc, sh)
    ref = F.layer_norm(x.float(), (D,), eps=1e-6) * (1 + sc.float()) + sh.float()
    c = cos(o, ref.bfloat16())
    print(f"adaln cos={c:.5f}", flush=True)
    assert c > 0.999, c

    o = tdp.rms_adaln(x, sc, sh)
    ref = (
        x.float() / x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        * (1 + sc.float())
        + sh.float()
    )
    c = cos(o, ref.bfloat16())
    print(f"rms_adaln cos={c:.5f}", flush=True)
    assert c > 0.999, c

    # rope: identity freqs -> output == input (rotation by 0)
    P = D // 2
    freqs = torch.zeros(1, H, S, P, 2, 2, device="cuda", dtype=torch.float32)
    freqs[..., 0, 0] = 1
    freqs[..., 1, 1] = 1
    qo, ko = tdp.apply_rope(q, k, freqs)
    cq, ck = cos(qo, q), cos(ko, k)
    print(f"apply_rope identity cosQ={cq:.5f} cosK={ck:.5f}", flush=True)
    assert cq > 0.999 and ck > 0.999, (cq, ck)

    rng = torch.zeros(N * D, dtype=torch.int32, device="cuda")
    o = tdp.stochastic_rounding_fp8(x.flatten(), rng)
    assert o.dtype == torch.float8_e4m3fn and o.shape == (N * D,), (o.dtype, o.shape)
    assert torch.isfinite(o.float()).all()
    print("stochastic_round_fp8 ok", flush=True)

    print("PY_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
