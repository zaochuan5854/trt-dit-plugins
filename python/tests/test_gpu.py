# SPDX-License-Identifier: Apache-2.0
"""GPU regressions for the runtime contract (skipped without CUDA/libs).

Covers the paths that CPU tests cannot: CUDA-graph capture (R1 pinned H2D),
a deserialized engine that never sees a shape change (workspace-plan
fallback), dynamic optimization profiles (R5), and the fused S<=1024 path
(R3-2).
"""
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

tdp = pytest.importorskip("trt_dit_plugins")
from trt_dit_plugins import _engine  # noqa: E402

try:
    tdp.ensure_loaded()
except Exception as exc:  # noqa: BLE001 - missing built libs / deps
    pytest.skip(f"plugin libraries unavailable: {exc}", allow_module_level=True)

B, H, S, D = 1, 2, 1024, 128


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _qkv(S_=S):
    return [torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)]


def _capture(name, fields, tensors, out_specs):
    """Build + bind outside capture, then capture only execute_async_v3."""
    in_sig = [(n, _engine.torch_dtype_to_trt(t.dtype), tuple(t.shape))
              for n, t in tensors.items()]
    eng = _engine._build_engine(name, fields, in_sig, len(out_specs),
                                [o[0] for o in out_specs])
    ctx = eng.create_execution_context()
    for n, t in tensors.items():
        ctx.set_tensor_address(n, t.data_ptr())
    outs = {}
    for n, dt, shape in out_specs:
        outs[n] = torch.empty(shape, device="cuda", dtype=dt)
        ctx.set_tensor_address(n, outs[n].data_ptr())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            assert ctx.execute_async_v3(side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    g.replay()
    torch.cuda.synchronize()
    return outs


def test_capture_int8_attention():
    q, k, v = _qkv()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()).bfloat16()
    out = _capture(tdp.PluginOp.INT8_ATTENTION, {}, {"q": q, "k": k, "v": v},
                   [("o", torch.bfloat16, (B, H, S, D))])["o"]
    assert _cos(out, ref) > 0.99


def test_capture_sage_pv_threshold():
    sm = torch.cuda.get_device_properties(0)
    fp8 = int(sm.major * 10 + sm.minor == 89)
    q, k, v = _qkv()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()).bfloat16()
    out = _capture(tdp.PluginOp.SAGE_ATTN, {"fp8_pv": fp8},
                   {"q": q, "k": k, "v": v},
                   [("o", torch.bfloat16, (B, H, S, D))])["o"]
    assert _cos(out, ref) > 0.99


def test_deserialized_static_engine_has_no_shape_change():
    q, k, v = _qkv()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()).bfloat16()
    eng = _engine._build_engine(
        tdp.PluginOp.SAGE_ATTN, {},
        [("q", _engine.torch_dtype_to_trt(q.dtype), (B, H, S, D)),
         ("k", _engine.torch_dtype_to_trt(k.dtype), (B, H, S, D)),
         ("v", _engine.torch_dtype_to_trt(v.dtype), (B, H, S, D))],
        1, ["o"])
    import tensorrt as trt

    # deserialize_cuda_engine is not a module-level function in TRT 11.
    eng2 = trt.Runtime(_engine._get_logger()).deserialize_cuda_engine(eng.serialize())
    assert eng2 is not None
    ctx = eng2.create_execution_context()
    out = torch.empty(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    for n, t in (("q", q), ("k", k), ("v", v), ("o", out)):
        ctx.set_tensor_address(n, t.data_ptr())
    assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert _cos(out, ref) > 0.99


def test_dynamic_profile():
    prof = {"q": ((1, H, 128, D), (1, H, 512, D), (1, H, S, D))}
    prof["k"] = prof["v"] = prof["q"]
    for S_ in (128, 512, S):
        q, k, v = _qkv(S_)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()).bfloat16()
        out = _engine.run_plugin(
            tdp.PluginOp.SAGE_ATTN, {"q": q, "k": k, "v": v},
            [("o", torch.bfloat16, (B, H, S_, D))], profile=prof)["o"]
        assert _cos(out, ref) > 0.99


def _rms_rope_ref(q, k, freqs, qs, ks, rot):
    """Reference for split-half RMS+RoPE; rot=0 means RMS-only."""
    def norm(x, w):
        xf = x.float()
        rr = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return xf * rr * w.float()

    nq, nk = norm(q, qs), norm(k, ks)
    if rot == 0:
        return nq.to(q.dtype), nk.to(k.dtype)
    half = rot // 2
    q0, q1 = nq[..., :half], nq[..., half:rot]
    k0, k1 = nk[..., :half], nk[..., half:rot]
    c, s = freqs[..., 0, 0].float(), freqs[..., 1, 0].float()
    qo = torch.cat([q0 * c - q1 * s, q0 * s + q1 * c,
                    nq[..., rot:]], dim=-1)
    ko = torch.cat([k0 * c - k1 * s, k0 * s + k1 * c,
                    nk[..., rot:]], dim=-1)
    return qo.to(q.dtype), ko.to(k.dtype)


def _freqs(B, H, S_, D):
    ang = torch.rand(B, H, S_, D // 2, device="cuda") * 0.05
    f = torch.empty(B, H, S_, D // 2, 2, 2, device="cuda")
    c, s = ang.cos(), ang.sin()
    f[..., 0, 0], f[..., 0, 1] = c, -s
    f[..., 1, 0], f[..., 1, 1] = s, c
    return f


def test_rms_rope_dynamic_b_s():
    H, D = 2, 128
    profile = {
        "q": ((1, H, 256, D), (2, H, 1024, D), (2, H, 4096, D)),
        "k": ((1, H, 256, D), (2, H, 1024, D), (2, H, 4096, D)),
        "f": ((1, 1, 256, D // 2, 2, 2), (1, 1, 1024, D // 2, 2, 2),
              (1, 1, 4096, D // 2, 2, 2)),
    }
    for B, S_ in ((1, 256), (2, 1024), (1, 4096)):
        q = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
        fq = _freqs(1, 1, S_, D)
        qs = torch.rand(D, device="cuda")
        ks = torch.rand(D, device="cuda")
        qo, ko = tdp.rms_rope_split_half(
            q, k, fq, qs, ks, profile=profile)
        rq, rk = _rms_rope_ref(q, k, fq, qs, ks, D)
        assert _cos(qo, rq) > 0.9999
        assert _cos(ko, rk) > 0.9999


def test_rms_rope_rms_only_rot_minus_one():
    B, H, S_, D = 1, 2, 512, 128
    q = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, 2, 2, device="cuda")
    qs = torch.rand(D, device="cuda")
    ks = torch.rand(D, device="cuda")
    qo, ko = tdp.rms_rope_split_half(
        q, k, dummy, qs, ks, rot_dim=-1)
    rq, rk = _rms_rope_ref(q, k, dummy, qs, ks, 0)
    assert _cos(qo, rq) > 0.9999
    assert _cos(ko, rk) > 0.9999


def test_rms_norm_cross_attention_different_s():
    B, H, Sq, Sk, D = 1, 2, 1024, 256, 128
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    wq = torch.rand(D, device="cuda")
    wk = torch.rand(D, device="cuda")

    def ref(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
                * w.float()).to(x.dtype).permute(0, 2, 1, 3).contiguous()

    qo = tdp.rms_norm(q, wq, layout="bshd")
    ko = tdp.rms_norm(k, wk, layout="bshd")
    assert qo.shape == (B, H, Sq, D)
    assert ko.shape == (B, H, Sk, D)
    assert _cos(qo, ref(q, wq)) > 0.9999
    assert _cos(ko, ref(k, wk)) > 0.9999


def test_rms_norm_bhsd_matches_bshd():
    B, H, S_, D = 2, 2, 64, 128
    x = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(D, device="cuda")
    y = tdp.rms_norm(x, w)
    z = tdp.rms_norm(x.transpose(1, 2).contiguous(), w, layout="bshd")
    assert _cos(y, z) > 0.9999


def test_rms_adaln_mixed_mod_dtype():
    B, S_, D = 2, 8, 128
    x = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    sh = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    out = tdp.rms_adaln(x, sc, sh)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
           * (1 + sc.float()) + sh.float()).to(torch.bfloat16)
    assert _cos(out, ref) > 0.999


def test_layer_norm_adaln_3d_residual_gate():
    B, S_, D = 2, 16, 256
    x = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    sh = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    w = torch.randn(D, device="cuda", dtype=torch.float32)
    res = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)

    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    y = (xf - mean) * torch.rsqrt(var + 1e-6) * w.float()
    y = y * (1 + sc.float()) + sh.float()
    ref = (res.float() + gate.float() * y).to(torch.bfloat16)

    out = tdp.layer_norm_adaln(x, sc, sh, weight=w, residual=res, gate=gate)
    assert out.shape == x.shape
    assert _cos(out, ref) > 0.999


def test_rms_adaln_residual_gate_2d():
    N, D = 64, 128
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(N, D, device="cuda", dtype=torch.float32)
    sh = torch.randn(N, D, device="cuda", dtype=torch.float32)
    res = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(N, D, device="cuda", dtype=torch.float32)
    out = tdp.rms_adaln(x, sc, sh, residual=res, gate=gate)
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (res.float() + gate.float() * (y * (1 + sc.float()) + sh.float())).to(torch.bfloat16)
    assert _cos(out, ref) > 0.999


def test_fused_short_seq():
    S_ = 512
    qi = torch.randint(-127, 127, (B, H, S_, D), device="cuda", dtype=torch.int8)
    one = torch.full((1,), 0.02, device="cuda")
    rw = torch.ones(D, device="cuda")
    inv = torch.rand(64, device="cuda") * 0.05
    of = tdp.fused_int8_rope_sage_attn(qi, one, qi, one, qi, one, rw, rw, inv)
    assert of.shape == (B, H, S_, D) and torch.isfinite(of.float()).all()

    ang = torch.arange(S_, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    fq = torch.zeros(1, 1, S_, 64, 2, 2, device="cuda")
    fq[..., 0, 0], fq[..., 1, 0] = ang.cos(), ang.sin()
    fq[..., 0, 1], fq[..., 1, 1] = -ang.sin(), ang.cos()
    qf = (qi.float() * one).bfloat16()
    qr, kr = tdp.rms_rope_split_half(qf, qf, fq, rw, rw)
    ref = tdp.sage_attn(qr, kr, qf)
    assert _cos(of, ref) > 0.99


def test_fused_float_input():
    # R2: linear output (bf16) feeds the fused op with no QuantizeLinear.
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    rw = torch.rand(D, device="cuda") * 0.5 + 0.5
    inv = torch.rand(64, device="cuda") * 0.05
    of = tdp.fused_rope_sage_attn(q, q, q, rw, rw, inv)
    assert of.dtype == torch.bfloat16 and of.shape == q.shape

    ang = torch.arange(S, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    fq = torch.zeros(1, 1, S, 64, 2, 2, device="cuda")
    fq[..., 0, 0], fq[..., 1, 0] = ang.cos(), ang.sin()
    fq[..., 0, 1], fq[..., 1, 1] = -ang.sin(), ang.cos()
    qr, kr = tdp.rms_rope_split_half(q, q, fq, rw, rw)
    assert _cos(of, tdp.sage_attn(qr, kr, q)) > 0.99
