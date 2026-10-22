# SPDX-License-Identifier: Apache-2.0
"""Minimal fused block contract test (no Anima checkpoint required).

Composes the 0.6.0 ops in the target layout: AdaLN -> QKV ConvRot (BHSD) ->
RMS+RoPE -> attention -> ConvRot out. The real-checkpoint comparison is the
Reforge handoff gate, so this only pins shapes, composition numerics, and the
absence of layout-transpose glue.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

tdp = pytest.importorskip("trt_dit_plugins")
gs = pytest.importorskip("onnx_graphsurgeon")
from trt_dit_plugins import surgeon as S  # noqa: E402

try:
    tdp.ensure_loaded()
except Exception as exc:  # noqa: BLE001 - missing built libs / deps
    pytest.skip(f"plugin libraries unavailable: {exc}", allow_module_level=True)


C, H, D, K = 2048, 16, 128, 2048
PROFILES = ((1, 256), (1, 4096), (2, 4096))
XPROFILE = {"x": ((1, 256, K), (1, 4096, K), (2, 4096, K))}


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _rms_ref(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
            * w.float()).to(x.dtype)


@pytest.mark.parametrize("B,S", PROFILES)
def test_fused_block_ops_dynamic(B, S):
    torch.manual_seed(0)
    x = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)
    wq = torch.randint(-127, 127, (3 * C, K), device="cuda", dtype=torch.int8)
    wq_s = torch.rand(3 * C, device="cuda") * 0.01
    bq = torch.randn(3 * C, device="cuda", dtype=torch.bfloat16)

    q, k, v = tdp.convrot_linear(
        x, wq, wq_s, bq, out_layout="bhsd", qkv_heads=H, head_dim=D,
        profile=XPROFILE)
    assert q.shape == k.shape == v.shape == (B, H, S, D)
    assert q.dtype == torch.bfloat16

    # Identity rotation (cos=1, sin=0): the rope output must equal RMSNorm,
    # which checks the BHSD input path end to end.
    fq = torch.zeros(1, 1, S, D // 2, 2, 2, device="cuda")
    fq[..., 0, 0] = 1.0
    fq[..., 1, 1] = 1.0
    qs = torch.ones(D, device="cuda")
    ks = torch.ones(D, device="cuda")
    qo, ko = tdp.rms_rope_split_half(
        q, k, fq, qs, ks,
        profile={"q": ((1, H, 256, D), (1, H, 4096, D), (2, H, 4096, D)),
                 "k": ((1, H, 256, D), (1, H, 4096, D), (2, H, 4096, D)),
                 "f": ((1, 1, 256, D // 2, 2, 2),
                       (1, 1, 4096, D // 2, 2, 2),
                       (1, 1, 4096, D // 2, 2, 2))})
    assert qo.shape == ko.shape == (B, H, S, D)
    assert _cos(qo, _rms_ref(q, qs)) > 0.9999
    assert _cos(ko, _rms_ref(k, ks)) > 0.9999

    attn = tdp.int8_attention(qo, ko, v)
    assert attn.shape == (B, H, S, D) and attn.dtype == torch.bfloat16
    if S <= 256:
        ref = torch.nn.functional.scaled_dot_product_attention(
            qo.float(), ko.float(), v.float()).bfloat16()
        assert _cos(attn, ref) > 0.99

    wo = torch.randint(-127, 127, (C, C), device="cuda", dtype=torch.int8)
    wo_s = torch.rand(C, device="cuda") * 0.01
    bo = torch.randn(C, device="cuda", dtype=torch.bfloat16)
    xo = attn.permute(0, 2, 1, 3).reshape(B, S, C).contiguous()
    y = tdp.convrot_linear(xo, wo, wo_s, bo, profile=XPROFILE)
    assert y.shape == (B, S, C)
    assert torch.equal(y, tdp.convrot_linear(xo, wo, wo_s, bo, profile=XPROFILE))


def _graph():
    x = gs.Variable("x", dtype="bfloat16", shape=[1, 256, C])
    scale = gs.Variable("scale", dtype=np.float32, shape=[1, 1, C])
    shift = gs.Variable("shift", dtype=np.float32, shape=[1, 1, C])
    wq = gs.Variable("wq", dtype=np.int8, shape=[3 * C, K])
    wq_s = gs.Variable("wq_s", dtype=np.float32, shape=[3 * C])
    bq = gs.Variable("bq", dtype="bfloat16", shape=[3 * C])
    fq = gs.Variable("fq", dtype=np.float32, shape=[1, 1, 256, D // 2, 2, 2])
    qs = gs.Variable("qs", dtype=np.float32, shape=[D])
    ks = gs.Variable("ks", dtype=np.float32, shape=[D])
    wo = gs.Variable("wo", dtype=np.int8, shape=[C, C])
    wo_s = gs.Variable("wo_s", dtype=np.float32, shape=[C])
    bo = gs.Variable("bo", dtype="bfloat16", shape=[C])

    g = gs.Graph(nodes=[],
                 inputs=[x, scale, shift, wq, wq_s, bq, fq, qs, ks,
                         wo, wo_s, bo],
                 outputs=[])
    g.opset = 23
    xa = S.add_layer_norm_adaln(g, x, scale, shift)
    q, k, v = S.add_convrot_linear(
        g, xa, wq, wq_s, bq, out_layout="bhsd", qkv_heads=H, head_dim=D)
    qo, ko = S.add_rms_rope_split_half(g, q, k, fq, qs, ks)
    attn = S.add_int8_attention(g, qo, ko, v, out_layout="bsc")
    y = S.add_convrot_linear(g, attn, wo, wo_s, bo)
    g.outputs = [y]
    return g


def test_fused_block_graph_has_no_layout_glue():
    g = _graph()
    assert S.validate_plugin_nodes(g) == []
    ops = [n.op for n in g.nodes]
    assert ops.count("convrot_linear") == 2       # QKV + out projection
    assert ops.count("rms_rope_split_half") == 1
    assert ops.count("int8_attention") == 1
    assert ops.count("layer_norm_adaln") == 1
    # Attention stores token-major directly; no layout glue anywhere.
    assert ops.count("Transpose") == 0
    assert not [n for n in g.nodes
                if n.op in ("LayerNormalization", "RMSNormalization", "Mul",
                            "Reshape", "ScatterElements")]
    attn = next(n for n in g.nodes if n.op == "int8_attention")
    assert all(i.inputs and i.inputs[0].op in
               ("rms_rope_split_half", "convrot_linear") for i in attn.inputs)
    gs.export_onnx(g)
