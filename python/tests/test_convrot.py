# SPDX-License-Identifier: Apache-2.0
"""ConvRot W8A8 / W4A4 plugin tests.

CPU-only checks run everywhere onnx-graphsurgeon is installed. The parity
tests need a GPU, the built plugin libraries and comfy-kitchen (the oracle).
"""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
gs = pytest.importorskip("onnx_graphsurgeon")

import trt_dit_plugins as tdp
from trt_dit_plugins import surgeon as S
from trt_dit_plugins.select import PluginOp

F32 = np.float32
I8 = np.int8


def _graph():
    x = gs.Variable("x", dtype=F32, shape=[128, 2048])
    g = gs.Graph(nodes=[], inputs=[x], outputs=[])
    return g, x


def _convrot_node(k=2048, n=256, w4a4=False, name=None):
    g, x = _graph()
    w = gs.Variable("w", dtype=I8, shape=[n, k // 2 if w4a4 else k])
    ws = gs.Variable("ws", dtype=F32, shape=[n])
    b = gs.Variable("b", dtype=F32, shape=[n])
    g.inputs += [w, ws, b]
    y = S.add_convrot_linear(g, x, w, ws, b, w4a4=w4a4, name=name)
    g.outputs = [y]
    return g, y


def test_add_convrot_linear_w8a8():
    g, y = _convrot_node()
    node = g.nodes[0]
    assert (node.op, node.domain) == (PluginOp.CONVROT_LINEAR, "dit-plugins")
    assert node.attrs["group_size"] == 256 and node.attrs["has_bias"] == 1
    assert node.attrs["out_dtype"] == -1 and node.attrs["gemm_config"] == -1
    assert node.attrs["epilogue"] == 0 and node.attrs["residual_gate"] == 0
    assert node.attrs["out_layout"] == 0 and node.attrs["qkv_heads"] == 0
    assert node.attrs["head_dim"] == 0
    assert y.shape == [128, 256] and y.dtype == F32
    assert S.validate_plugin_nodes(g) == []
    onnx.checker.check_model(gs.export_onnx(g))


def test_add_convrot_linear_w4a4():
    g, y = _convrot_node(w4a4=True, name="q")
    node = g.nodes[0]
    assert node.op == PluginOp.CONVROT_LINEAR_W4A4 and node.name == "q"
    assert S.validate_plugin_nodes(g) == []


def test_add_convrot_out_dtype_declares_real_output():
    for xdt in ("float16", "float32", "bfloat16"):
        x = gs.Variable("x", dtype=xdt, shape=[4, 2048])
        w = gs.Variable("w", dtype=I8, shape=[256, 2048])
        ws = gs.Variable("ws", dtype=F32, shape=[256])
        b = gs.Variable("b", dtype="bfloat16", shape=[256])
        g = gs.Graph(nodes=[], inputs=[x, w, ws, b], outputs=[])
        y = S.add_convrot_linear(g, x, w, ws, b, out_dtype=0)  # 0 = BF16
        assert y.dtype == "bfloat16", xdt
        g.outputs = [y]
        assert S.validate_plugin_nodes(g) == []
    with pytest.raises(ValueError, match="out_dtype"):
        S.add_convrot_linear(g, x, w, ws, b, out_dtype=7)


def test_convrot_validation_rejects_bad_dtypes_and_fields():
    g, _ = _convrot_node()
    node = g.nodes[0]
    node.inputs[0].dtype = "float64"
    node.inputs[1].dtype = "float32"
    node.inputs[2].dtype = "float16"
    node.inputs[3].dtype = "float16"
    node.attrs.update(out_dtype=7, has_bias=5, gemm_config=-2)
    errs = S.validate_plugin_nodes(g)
    for frag in ("x dtype", "weight dtype", "weight_scale dtype", "bias dtype",
                 "out_dtype", "has_bias", "gemm_config"):
        assert any(frag in e for e in errs), (frag, errs)


def test_convrot_validation_bias_must_match_output_dtype():
    g, _ = _convrot_node()  # x/bias f32, out_dtype=-1 -> f32
    g.nodes[0].attrs["out_dtype"] = 0  # BF16
    assert any("bias dtype" in e for e in S.validate_plugin_nodes(g))


def test_convrot_validation_rejects_bad_shapes():
    g, _ = _convrot_node(k=2048)
    g.nodes[0].attrs["group_size"] = 512
    assert any("group_size" in e for e in S.validate_plugin_nodes(g))

    g, _ = _convrot_node(k=2048)
    g.nodes[0].inputs[1].shape = [256, 2000]  # not a multiple of 256
    assert any("K=" in e for e in S.validate_plugin_nodes(g))

    g, _ = _convrot_node(k=2048, n=250)  # W8A8 needs N % 4 == 0 (250 % 4 != 2)
    assert any("N=" in e for e in S.validate_plugin_nodes(g))

    g, _ = _convrot_node(k=2048)
    g.nodes[0].inputs[0].shape = [128, 1024]
    assert any("x K=" in e for e in S.validate_plugin_nodes(g))


def _comfy_node(convrot=1, groups=256, out_code=2, bias=True, k=2048, n=256):
    x = gs.Variable("x", dtype="bfloat16", shape=[128, k])
    w = gs.Variable("w", dtype=I8, shape=[n, k])
    ws = gs.Variable("ws", dtype=F32, shape=[n])
    inputs = [x, w, ws]
    graph_inputs = [x]
    if bias:
        b = gs.Variable("b", dtype="bfloat16", shape=[n])
        inputs.append(b)
        graph_inputs.append(b)
    y = gs.Variable("y", dtype="bfloat16", shape=[128, n])
    g = gs.Graph(nodes=[], inputs=graph_inputs + [w, ws], outputs=[y])
    g.nodes.append(gs.Node(op="int8_linear", name="lin", domain="comfy_kitchen",
                           inputs=inputs, outputs=[y],
                           attrs={"convrot": convrot, "convrot_groupsize": groups,
                                  "output_dtype_code": out_code}))
    return g


def test_import_comfy_convrot():
    g = _comfy_node()
    converted, skipped = S.import_comfy_convrot(g, gemm_config=5)
    assert converted == ["lin"] and skipped == []
    node = g.nodes[0]
    assert (node.op, node.domain) == (PluginOp.CONVROT_LINEAR, "dit-plugins")
    assert node.attrs["out_dtype"] == 0 and node.attrs["gemm_config"] == 5
    assert node.attrs["plugin_namespace"] == "dit-plugins"
    assert S.validate_plugin_nodes(g) == []

    # comfy output_dtype_code 0/1 -> plugin fields 2/1 (0=BF16, 2=FP32).
    g = _comfy_node(out_code=0)
    S.import_comfy_convrot(g)
    assert g.nodes[0].attrs["out_dtype"] == 2
    g = _comfy_node(out_code=1)
    S.import_comfy_convrot(g)
    assert g.nodes[0].attrs["out_dtype"] == 1


def test_import_comfy_convrot_fails_open():
    transposed = _comfy_node()
    transposed.nodes[0].attrs["transposed"] = 1
    five_inputs = _comfy_node()
    five_inputs.nodes[0].inputs.append(gs.Variable("rs", dtype=F32, shape=[128]))
    two_outputs = _comfy_node()
    two_outputs.nodes[0].outputs.append(
        gs.Variable("y2", dtype="bfloat16", shape=[128, 256]))
    reasons = (
        _comfy_node(convrot=0),          # plain (non-ConvRot) linear
        transposed,                       # _qdata layout is [K,N]
        _comfy_node(groups=128),          # only group size 256 is supported
        _comfy_node(bias=False),          # the plugin always takes a bias tensor
        _comfy_node(out_code=7),          # unknown output dtype code
        _comfy_node(k=2000),              # K % 256 != 0
        five_inputs,                      # residual_scale not representable
        two_outputs,                      # plugin has exactly one output
    )
    for g in reasons:
        converted, skipped = S.import_comfy_convrot(g)
        assert converted == [] and skipped == ["lin"]
        assert g.nodes[0].domain == "comfy_kitchen"  # untouched
        assert S.validate_plugin_nodes(g) == []


# --- GPU parity against comfy-kitchen (the oracle) ---------------------------

def _torch():
    return pytest.importorskip("torch")


def _hadamard(size, device):
    import torch

    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=torch.float32, device=device)
    h, n = h4, 4
    while n < size:
        h, n = torch.kron(h, h4), n * 4
    return h / size**0.5


def _rotate_weight(w, group=256):
    h = _hadamard(group, w.device).T
    return (w.reshape(w.shape[0], -1, group) @ h).reshape(w.shape)


def _quant_w8(w):
    import torch

    s = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-10)
    return (w / s).round().clamp(-127, 127).to(torch.int8), s.reshape(-1)


def _quant_w4(w):
    import torch

    s = (w.abs().amax(1, keepdim=True) / 7).clamp(min=1e-10)
    q = (w / s).round().clamp(-7, 7).to(torch.int8)
    lo = q[:, 0::2].to(torch.int32) & 0xF
    hi = q[:, 1::2].to(torch.int32) & 0xF
    return (lo | (hi << 4)).to(torch.int8), s.reshape(-1)


def _weights(n=256, k=2048):
    torch = _torch()
    torch.manual_seed(0)
    w0 = torch.randn(n, k, dtype=torch.float32, device="cuda")
    wr = _rotate_weight(w0)
    return w0, _quant_w8(wr), _quant_w4(wr)


def _gpu():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return pytest.importorskip("comfy_kitchen")


def test_convrot_w8a8_parity():
    torch = _torch()
    ck = _gpu()
    _, (w8, ws8), _ = _weights()
    for m in (1, 128):
        x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
        got = tdp.convrot_linear(x, w8, ws8, b)
        want = ck.int8_linear(x, w8, ws8, b, out_dtype=torch.bfloat16, convrot=True)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want), (m, (got.float() - want.float()).abs().max())


def test_convrot_w8a8_out_dtype_parity():
    """Field codes (0=BF16, 1=FP16, 2=FP32) must reach the kernel converted."""
    torch = _torch()
    ck = _gpu()
    _, (w8, ws8), _ = _weights()
    for m in (1, 128):
        x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            b = torch.randn(256, dtype=dt, device="cuda") * 0.1
            got = tdp.convrot_linear(x, w8, ws8, b, out_dtype=dt)
            want = ck.int8_linear(x, w8, ws8, b, out_dtype=dt, convrot=True)
            assert got.dtype == dt and torch.equal(got, want), (m, dt)


def test_convrot_w4a4_parity_and_autotune():
    from trt_dit_plugins import autotune

    torch = _torch()
    ck = _gpu()
    _, _, (w4, ws4) = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    for m in (1, 128):
        x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
        cfg = autotune.autotune_convrot_w4a4(m, 256, 2048, "bfloat16")
        assert 0 <= cfg < 7
        assert autotune.cached_config(m, 256, 2048, "bfloat16") == cfg
        got = tdp.convrot_linear_w4a4(x, w4, ws4, b, gemm_config=cfg)
        want = ck.convrot_w4a4_linear(x, w4, ws4, b)
        assert torch.equal(got, want), (m, (got.float() - want.float()).abs().max())


def test_convrot_enqueue_does_not_sync():
    """The plugin enqueue must return while earlier stream work is still pending.

    Uses a pre-created execution context: ``run_plugin`` makes a fresh one per
    call, and TRT's context allocation can block on its memory pool, which is
    not plugin time.
    """
    import time

    pytest.importorskip("tensorrt")
    torch = _torch()
    _gpu()
    from trt_dit_plugins import _engine

    _, (w8, ws8), _ = _weights()
    b = torch.zeros(256, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(128, 2048, dtype=torch.bfloat16, device="cuda")
    tdp.ensure_loaded()
    sig = [(n, _engine.torch_dtype_to_trt(t.dtype), tuple(t.shape))
           for n, t in (("x", x), ("w", w8), ("ws", ws8), ("b", b))]
    engine = _engine._build_engine(PluginOp.CONVROT_LINEAR, {}, sig, 1, ["y"])
    ctx = engine.create_execution_context()
    for n, t in (("x", x), ("w", w8), ("ws", ws8), ("b", b)):
        ctx.set_tensor_address(n, t.data_ptr())
    out = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    ctx.set_tensor_address("y", out.data_ptr())

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        assert ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()

    with torch.cuda.stream(s):
        gate = torch.randn(4096, 4096, device="cuda")
        for _ in range(8):
            gate = gate @ gate  # tens of ms of work queued ahead of the plugin
        t0 = time.perf_counter()
        ok = ctx.execute_async_v3(s.cuda_stream)
        enqueue_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
    assert ok
    assert enqueue_ms < 20.0, enqueue_ms


def _apply_epilogue(y, epilogue):
    import torch

    if epilogue == "gelu":
        return torch.nn.functional.gelu(y)
    if epilogue == "silu":
        return torch.nn.functional.silu(y)
    return y


@pytest.mark.parametrize("epilogue", ["gelu", "silu"])
def test_convrot_epilogue_parity(epilogue):
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    for shape in ((1, 64, 2048), (2, 16, 2048), (1, 2048)):
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        base = tdp.convrot_linear(x, w8, ws8, b)
        got = tdp.convrot_linear(x, w8, ws8, b, epilogue=epilogue)
        want = _apply_epilogue(base.float(), epilogue).to(torch.bfloat16)
        assert got.shape == want.shape
        assert (got.float() - want.float()).abs().mean() < 0.02


def test_convrot_gelu_residual_gate():
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    for batch, seq in ((1, 64), (3, 16)):
        x = torch.randn(batch, seq, 2048, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(batch, seq, 256, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(batch, 1, 256, dtype=torch.bfloat16, device="cuda")
        base = tdp.convrot_linear(x, w8, ws8, b)
        got = tdp.convrot_linear(x, w8, ws8, b, epilogue="gelu",
                                 residual=res, gate=gate)
        want = res.float() + gate.float() * _apply_epilogue(base.float(), "gelu")
        assert got.shape == (batch, seq, 256)
        assert (got.float() - want).abs().mean() < 0.05


def test_convrot_residual_gate_no_act():
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(2, 8, 2048, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(2, 8, 256, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
    base = tdp.convrot_linear(x, w8, ws8, b)
    got = tdp.convrot_linear(x, w8, ws8, b, residual=res, gate=gate)
    want = res.float() + gate.float().unsqueeze(1) * base.float()
    assert (got.float() - want).abs().mean() < 0.05


def test_convrot_residual_gate_rank_free():
    """rank の異なる residual/gate: plugin 出力は residual の shape を取る."""
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    res5 = torch.randn(1, 1, 64, 64, 256, dtype=torch.bfloat16, device="cuda")
    gate5 = torch.randn(1, 1, 1, 1, 256, dtype=torch.bfloat16, device="cuda")
    for x in (
        torch.randn(1, 4096, 2048, dtype=torch.bfloat16, device="cuda"),
        torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda"),
        torch.randn(1, 1, 64, 64, 2048, dtype=torch.bfloat16, device="cuda"),
    ):
        base = tdp.convrot_linear(x, w8, ws8, b)
        got = tdp.convrot_linear(x, w8, ws8, b, residual=res5, gate=gate5)
        want = res5.float() + gate5.float() * base.reshape(res5.shape).float()
        assert got.shape == res5.shape, got.shape
        assert (got.float() - want).abs().mean() < 0.05


def test_convrot_residual_gate_broadcast_batch():
    """lead numel 1 の gate は batch 間で共有される (per-batch slice しない)."""
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(2, 8, 2048, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(2, 8, 256, dtype=torch.bfloat16, device="cuda")
    base = tdp.convrot_linear(x, w8, ws8, b)
    for gate in (torch.randn(1, 1, 1, 256, dtype=torch.bfloat16, device="cuda"),
                 torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")):
        got = tdp.convrot_linear(x, w8, ws8, b, residual=res, gate=gate)
        want = res.float() + gate.reshape(1, 1, 256).float() * base.float()
        assert got.shape == (2, 8, 256)
        assert (got.float() - want).abs().mean() < 0.05


def test_convrot_residual_gate_epilogue_vector_and_fallback():
    """128-bit epilogue と、`cols % VEC != 0` のスカラ fallback の両方が正しい."""
    torch = _torch()
    _gpu()
    for n in (16, 12):  # 16 -> vector path, 12 -> scalar fallback (VEC=8, bf16)
        _, (w8, ws8), _ = _weights(n=n, k=256)
        b = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1
        x = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(1, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(1, 1, n, dtype=torch.bfloat16, device="cuda")
        base = tdp.convrot_linear(x, w8, ws8, b)
        got = tdp.convrot_linear(x, w8, ws8, b, residual=res, gate=gate)
        want = res.float() + gate.reshape(1, n).float() * base.float()
        assert got.shape == (1, n), (n, got.shape)
        assert (got.float() - want).abs().mean() < 0.05, n


def test_convrot_residual_gate_rejects_bad_shapes():
    torch = _torch()
    _gpu()
    _, (w8, ws8), _ = _weights()
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(2, 8, 2048, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(2, 8, 256, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
    bad_res = torch.randn(2, 7, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="residual rows"):
        tdp.convrot_linear(x, w8, ws8, b, residual=bad_res, gate=gate)
    with pytest.raises(ValueError, match="gate batch"):
        tdp.convrot_linear(x, w8, ws8, b, residual=res,
                           gate=torch.randn(3, 256, dtype=torch.bfloat16,
                                            device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        tdp.convrot_linear(x, w8, ws8, b, residual=res,
                           gate=torch.randn(256, 2, dtype=torch.bfloat16,
                                            device="cuda").t())
    with pytest.raises(ValueError, match="residual requires gate"):
        tdp.convrot_linear(x, w8, ws8, b, residual=res)


def test_convrot_w4a4_rejects_epilogue():
    torch = _torch()
    _gpu()
    qw = torch.randint(-8, 7, (256, 1024), device="cuda", dtype=torch.int8)
    ws = torch.rand(256, device="cuda") * 0.01
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, 8, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="W4A4"):
        tdp.convrot_linear_w4a4(x, qw, ws, b, epilogue="gelu")


def test_convrot_qkv_bhsd_store():
    torch = _torch()
    _gpu()
    B, S_, H, D = 2, 8, 2, 128
    K, C = 2048, H * D
    x = torch.randn(B, S_, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(-127, 127, (3 * C, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(3 * C, device="cuda") * 0.01
    b = torch.randn(3 * C, dtype=torch.bfloat16, device="cuda")

    base = tdp.convrot_linear(x, w, ws, b)
    q, k, v = tdp.convrot_linear(x, w, ws, b, out_layout="bhsd",
                                 qkv_heads=H, head_dim=D)
    assert q.shape == k.shape == v.shape == (B, H, S_, D)
    y = base.reshape(B, S_, 3, H, D)
    for got, idx in ((q, 0), (k, 1), (v, 2)):
        ref = y[:, :, idx].permute(0, 2, 1, 3).contiguous()
        assert (got.float() - ref.float()).abs().mean() < 1e-3

    # Single-tensor BHSD store.
    w1 = w[:C].contiguous()
    b1 = b[:C].contiguous()
    ws1 = ws[:C].contiguous()
    base1 = tdp.convrot_linear(x, w1, ws1, b1)
    got1 = tdp.convrot_linear(x, w1, ws1, b1, out_layout="bhsd", head_dim=D)
    ref1 = base1.reshape(B, S_, H, D).permute(0, 2, 1, 3).contiguous()
    assert got1.shape == (B, H, S_, D)
    assert (got1.float() - ref1.float()).abs().mean() < 1e-3


def test_convrot_bhsd_dynamic_bs():
    torch = _torch()
    _gpu()
    H, D = 2, 128
    K, C = 2048, H * D
    w = torch.randint(-127, 127, (3 * C, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(3 * C, device="cuda") * 0.01
    b = torch.randn(3 * C, dtype=torch.bfloat16, device="cuda")
    prof = {"x": ((1, 1, K), (2, 8, K), (4, 16, K))}
    for batch, seq in ((1, 1), (3, 7), (4, 16)):
        x = torch.randn(batch, seq, K, dtype=torch.bfloat16, device="cuda")
        base = tdp.convrot_linear(x, w, ws, b, profile=prof)
        q, k, v = tdp.convrot_linear(x, w, ws, b, out_layout="bhsd", qkv_heads=H,
                                     head_dim=D, profile=prof)
        y = base.reshape(batch, seq, 3, H, D)
        for got, idx in ((q, 0), (k, 1), (v, 2)):
            ref = y[:, :, idx].permute(0, 2, 1, 3).contiguous()
            assert (got.float() - ref.float()).abs().mean() < 1e-3


def test_convrot_bhsd_m_lt_3_fallback():
    torch = _torch()
    _gpu()
    for batch, seq in ((1, 1), (1, 2)):
        x = torch.randn(batch, seq, 2048, dtype=torch.bfloat16, device="cuda")
        w = torch.randint(-127, 127, (256, 2048), device="cuda", dtype=torch.int8)
        ws = torch.rand(256, device="cuda") * 0.01
        b = torch.randn(256, dtype=torch.bfloat16, device="cuda")
        base = tdp.convrot_linear(x, w, ws, b)
        got = tdp.convrot_linear(x, w, ws, b, out_layout="bhsd", head_dim=128)
        ref = base.reshape(batch, seq, 2, 128).permute(0, 2, 1, 3).contiguous()
        assert (got.float() - ref.float()).abs().mean() < 1e-3


def test_convrot_w4a4_rejects_layout():
    torch = _torch()
    _gpu()
    qw = torch.randint(-8, 7, (256, 1024), device="cuda", dtype=torch.int8)
    ws = torch.rand(256, device="cuda") * 0.01
    b = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, 8, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="W4A4"):
        tdp.convrot_linear_w4a4(x, qw, ws, b, out_layout="bhsd", head_dim=128)


def test_add_convrot_residual_gate_rank_free_node():
    """ONNX 側も plugin と同じ契約: 出力 shape は residual に従う."""
    x = gs.Variable("x", dtype="bfloat16", shape=[1, 4096, 2048])
    w = gs.Variable("w", dtype=I8, shape=[256, 2048])
    ws = gs.Variable("ws", dtype=F32, shape=[256])
    b = gs.Variable("b", dtype="bfloat16", shape=[256])
    res = gs.Variable("res", dtype="bfloat16", shape=[1, 1, 64, 64, 256])
    gate = gs.Variable("gate", dtype="bfloat16", shape=[1, 1, 1, 1, 256])
    g = gs.Graph(nodes=[], inputs=[x, w, ws, b, res, gate], outputs=[])
    y = S.add_convrot_linear(g, x, w, ws, b, residual=res, gate=gate)
    g.outputs = [y]
    assert y.shape == [1, 1, 64, 64, 256]
    node = g.nodes[0]
    assert node.attrs["residual_gate"] == 1 and len(node.inputs) == 6
    assert S.validate_plugin_nodes(g) == []
    onnx.checker.check_model(gs.export_onnx(g))

    bad = gs.Variable("bad", dtype="bfloat16", shape=[1, 1, 63, 64, 256])
    with pytest.raises(ValueError, match="residual rows"):
        S.add_convrot_linear(g, x, w, ws, b, residual=bad, gate=gate)
    bad_gate = gs.Variable("bg", dtype="bfloat16", shape=[2, 1, 1, 256])
    with pytest.raises(ValueError, match="gate rank"):
        S.add_convrot_linear(g, x, w, ws, b, residual=res, gate=bad_gate)
    with pytest.raises(ValueError, match="dtype"):
        S.add_convrot_linear(g, x, w, ws, b, residual=res,
                             gate=gs.Variable("gf", dtype=F32, shape=[1, 256]))


def test_add_convrot_qkv_bhsd_node():
    x = gs.Variable("x", dtype="bfloat16", shape=[2, 8, 2048])
    w = gs.Variable("w", dtype=I8, shape=[768, 2048])
    ws = gs.Variable("ws", dtype=F32, shape=[768])
    b = gs.Variable("b", dtype="bfloat16", shape=[768])
    g = gs.Graph(nodes=[], inputs=[x, w, ws, b], outputs=[])
    q, k, v = S.add_convrot_linear(g, x, w, ws, b, out_layout="bhsd",
                                   qkv_heads=2, head_dim=128)
    g.outputs = [q, k, v]
    assert q.shape == k.shape == v.shape == [2, 2, 8, 128]
    node = g.nodes[0]
    assert node.attrs["out_layout"] == 1 and node.attrs["qkv_heads"] == 2
    assert S.validate_plugin_nodes(g) == []
    onnx.checker.check_model(gs.export_onnx(g))
