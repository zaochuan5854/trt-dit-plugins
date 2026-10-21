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
    assert y.shape == [128, 256] and y.dtype == F32
    assert S.validate_plugin_nodes(g) == []
    onnx.checker.check_model(gs.export_onnx(g))


def test_add_convrot_linear_w4a4():
    g, y = _convrot_node(w4a4=True, name="q")
    node = g.nodes[0]
    assert node.op == PluginOp.CONVROT_LINEAR_W4A4 and node.name == "q"
    assert S.validate_plugin_nodes(g) == []


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
    reasons = (
        _comfy_node(convrot=0),          # plain (non-ConvRot) linear
        transposed,                       # _qdata layout is [K,N]
        _comfy_node(groups=128),          # only group size 256 is supported
        _comfy_node(bias=False),          # the plugin always takes a bias tensor
        _comfy_node(out_code=7),          # unknown output dtype code
        _comfy_node(k=2000),              # K % 256 != 0
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
