# SPDX-License-Identifier: Apache-2.0
"""CPU-only surgeon tests: no torch, no TensorRT, no GPU."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
gs = pytest.importorskip("onnx_graphsurgeon")

from trt_dit_plugins import surgeon as S

F32 = np.float32


def _check_ok(graph):
    onnx.checker.check_model(gs.export_onnx(graph))
    assert S.validate_plugin_nodes(graph) == []


def test_add_all_plugins():
    B, H, Sv, D = 1, 2, 128, 64
    q = gs.Variable("q", dtype=F32, shape=[B, H, Sv, D])
    k = gs.Variable("k", dtype=F32, shape=[B, H, Sv, D])
    v = gs.Variable("v", dtype=F32, shape=[B, H, Sv, D])
    m = gs.Variable("m", dtype=np.int32, shape=[B, H, Sv // 128, Sv // 64])
    g = gs.Graph(nodes=[], inputs=[q, k, v, m], outputs=[])

    o_attn = S.add_int8_attention(g, q, k, v)
    N, D2 = 8, 32
    x = gs.Variable("x", dtype=F32, shape=[N, D2])
    sc = gs.Variable("sc", dtype=F32, shape=[1, D2])
    sh = gs.Variable("sh", dtype=F32, shape=[1, D2])
    g.inputs += [x, sc, sh]
    o_ada = S.add_adaln(g, x, sc, sh, eps=1e-6)
    o_rms = S.add_rms_adaln(g, x, sc, sh)
    f = gs.Variable("f", dtype=F32, shape=[1, 1, 1, D // 2, 2, 2])
    g.inputs.append(f)
    qo, ko = S.add_apply_rope(g, q, k, f)
    qs = gs.Variable("qs", dtype=F32, shape=[D])
    ks = gs.Variable("ks", dtype=F32, shape=[D])
    g.inputs += [qs, ks]
    qo2, ko2 = S.add_rms_rope_split_half(g, q, k, f, qs, ks, epsilon=1e-6, rot_dim=0)
    r = gs.Variable("r", dtype=np.int32, shape=[N, D2])
    g.inputs.append(r)
    o_fp8 = S.add_stochastic_round_fp8(g, x, r)
    o_sp = S.add_block_sparse_sage2_attn(g, q, k, v, m)
    g.outputs = [o_attn, o_ada, o_rms, qo, ko, qo2, ko2, o_fp8, o_sp]
    g.cleanup().toposort()

    assert S.summarize(g)["plugin_nodes"] == {
        "int8_attention": 1, "adaln": 1, "rms_adaln": 1, "apply_rope": 1,
        "rms_rope_split_half": 1, "stochastic_round_fp8": 1,
        "block_sparse_sage2_attn": 1,
    }
    node = next(n for n in g.nodes if n.op == "rms_rope_split_half")
    assert node.attrs["epsilon"] == pytest.approx(1e-6) and node.attrs["rot_dim"] == 0
    _check_ok(g)


def test_fuse_norm_affine_explicit_anchors():
    # norm producer is arbitrary (Relu) -> proves exporter-independence
    N, D = 8, 32
    x = gs.Variable("x", dtype=F32, shape=[N, D])
    sc = gs.Variable("sc", dtype=F32, shape=[1, D])
    sh = gs.Variable("sh", dtype=F32, shape=[1, D])
    norm = gs.Variable("norm", dtype=F32, shape=[N, D])
    g = gs.Graph(nodes=[gs.Node("Relu", inputs=[x], outputs=[norm])],
                 inputs=[x, sc, sh], outputs=[])
    a = gs.Variable("a", dtype=F32, shape=[N, D])
    o = gs.Variable("o", dtype=F32, shape=[N, D])
    g.nodes.append(gs.Node("Mul", inputs=[norm, sc], outputs=[a]))
    g.nodes.append(gs.Node("Add", inputs=[a, sh], outputs=[o]))
    g.outputs = [o]

    S.fuse_norm_affine(g, norm="norm", scale="sc", shift="sh", out="o",
                       plugin="rms_adaln", eps=1e-6)
    assert [n.op for n in g.nodes if n.domain == S.PLUGIN_DOMAIN] == ["rms_adaln"]
    assert not [n for n in g.nodes if n.op in ("Mul", "Add")]
    assert S.summarize(g)["total"] == 1
    _check_ok(g)


def test_retarget_and_validate_catches_arity():
    x = gs.Variable("x", dtype=F32, shape=[1, 2, 8, 64])
    o = gs.Variable("o", dtype=F32, shape=[1, 2, 8, 64])
    g = gs.Graph(nodes=[gs.Node("MyAttn", inputs=[x, x, x], outputs=[o])],
                 inputs=[x], outputs=[o])
    assert S.retarget_nodes(g, {"MyAttn": "int8_attention"}) == 1
    node = g.nodes[0]
    assert (node.op, node.domain) == ("int8_attention", "dit-plugins")
    assert S.validate_plugin_nodes(g) == []
    node.inputs.pop()  # break arity -> validator must complain
    assert S.validate_plugin_nodes(g) != []


def test_cli_end_to_end(tmp_path):
    N, D = 8, 32
    x = gs.Variable("x", dtype=F32, shape=[N, D])
    sc = gs.Variable("sc", dtype=F32, shape=[1, D])
    sh = gs.Variable("sh", dtype=F32, shape=[1, D])
    norm = gs.Variable("norm", dtype=F32, shape=[N, D])
    o = gs.Variable("o", dtype=F32, shape=[N, D])
    mid = gs.Variable("mid", dtype=F32, shape=[N, D])
    g = gs.Graph(nodes=[gs.Node("Relu", inputs=[x], outputs=[norm]),
                        gs.Node("Mul", inputs=[norm, sc], outputs=[mid]),
                        gs.Node("Add", inputs=[mid, sh], outputs=[o])],
                 inputs=[x, sc, sh], outputs=[o], opset=17)
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    onnx.save(gs.export_onnx(g), str(src))
    rc = S.main([str(src), str(dst), "--fuse-adaln", "norm,sc,sh,o,1e-6"])
    assert rc == 0 and dst.exists()
    out_g = S.load(str(dst))
    assert S.summarize(out_g)["plugin_nodes"] == {"adaln": 1}
    assert S.validate_plugin_nodes(out_g) == []


def _post_rope_graph():
    B, H, Sv, D = 1, 2, 2048, 128
    I8 = np.int8
    qi = gs.Variable("qi", dtype=I8, shape=[B, H, Sv, D])
    ki = gs.Variable("ki", dtype=I8, shape=[B, H, Sv, D])
    vi = gs.Variable("vi", dtype=I8, shape=[B, H, Sv, D])
    qs = gs.Variable("qs", dtype=F32, shape=[1])
    ks = gs.Variable("ks", dtype=F32, shape=[1])
    vs = gs.Variable("vs", dtype=F32, shape=[1])
    qn = gs.Variable("qn", dtype=F32, shape=[D])
    kn = gs.Variable("kn", dtype=F32, shape=[D])
    inv = gs.Variable("inv", dtype=F32, shape=[64])
    qt = gs.Variable("qt", dtype=I8, shape=[B, H, Sv, D])
    kt = gs.Variable("kt", dtype=I8, shape=[B, H, Sv, D])
    fq = gs.Variable("fq", dtype=F32, shape=[1, 1, Sv, D // 2, 2, 2])
    qsc = gs.Variable("qsc", dtype=F32, shape=[D])
    ksc = gs.Variable("ksc", dtype=F32, shape=[D])
    rq = gs.Variable("rq", dtype=F32, shape=[B, H, Sv, D])
    rk = gs.Variable("rk", dtype=F32, shape=[B, H, Sv, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, H, Sv, D])
    out = gs.Variable("out", dtype=F32, shape=[B, H, Sv, D])
    rope = gs.Node("rms_rope_split_half", inputs=[qt, kt, fq, qsc, ksc],
                   outputs=[rq, rk], domain="dit-plugins")
    attn = gs.Node("SageInt8Attn", inputs=[rq, rk, vi], outputs=[ao])
    tail = gs.Node("Relu", inputs=[ao], outputs=[out])
    g = gs.Graph(nodes=[rope, attn, tail],
                 inputs=[qi, qs, ki, ks, vi, vs, qn, kn, inv,
                         qt, kt, fq, qsc, ksc],
                 outputs=[out])
    return g


def test_fuse_dit_self_attn_block():
    g = _post_rope_graph()
    node = S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    assert node.op == "fused_int8_rope_sage_attn"
    assert [i.name for i in node.inputs] == [
        "qi", "qs", "ki", "ks", "vi", "vs", "qn", "kn", "inv"]
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 1}
    assert not [n for n in g.nodes if n.op in ("SageInt8Attn", "rms_rope_split_half")]
    tail = next(n for n in g.nodes if n.op == "Relu")
    assert tail.inputs[0].name.endswith("_fused")
    _check_ok(g)


def test_fuse_dit_self_attn_block_rejects_cross():
    g = _post_rope_graph()
    # rewire K to a different producer -> cross-attn shape, must refuse
    attn = next(n for n in g.nodes if n.op == "SageInt8Attn")
    other = gs.Variable("other", dtype=F32, shape=[1, 2, 2048, 128])
    g.inputs.append(other)
    attn.inputs[1] = other
    with pytest.raises(ValueError):
        S.fuse_dit_self_attn_block(
            g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
            v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")


def test_fuse_dit_self_attn_block_rejects_masked():
    g = _post_rope_graph()
    # 4th (mask) input must refuse: silent mask drop would corrupt numerics
    attn = next(n for n in g.nodes if n.op == "SageInt8Attn")
    mask = gs.Variable("mask", dtype=np.int32, shape=[1, 2, 2048, 2048])
    g.inputs.append(mask)
    attn.inputs.append(mask)
    with pytest.raises(ValueError, match="exactly 3 inputs"):
        S.fuse_dit_self_attn_block(
            g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
            v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")


def test_fuse_output_dtype_follows_attn_out():
    g = _post_rope_graph()
    S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    node = next(n for n in g.nodes if n.op == "fused_int8_rope_sage_attn")
    # qi is INT8 but the plugin emits BF16: output must follow attn_out (F32
    # here), never the INT8 anchor dtype.
    assert node.outputs[0].dtype == F32
    _check_ok(g)


def test_add_dit_self_out_dtype_override():
    B, H, Sv, D = 1, 2, 2048, 128
    I8 = np.int8
    args = [gs.Variable(n, dtype=d, shape=s) for n, d, s in [
        ("qi", I8, [B, H, Sv, D]), ("qs", F32, [1]), ("ki", I8, [B, H, Sv, D]),
        ("ks", F32, [1]), ("vi", I8, [B, H, Sv, D]), ("vs", F32, [1]),
        ("qn", F32, [D]), ("kn", F32, [D]), ("inv", F32, [64])]]
    g = gs.Graph(nodes=[], inputs=args, outputs=[])
    o = S.add_dit_self_fused_attn(g, *args)
    assert o.dtype == np.int8  # default keeps anchor dtype (documented wart)
    o2 = S.add_dit_self_fused_attn(g, *args, out_dtype=F32)
    assert o2.dtype == F32


def test_fused_attn_validation():
    g = _post_rope_graph()
    S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    assert S.validate_plugin_nodes(g) == []
    # break q/k/v shape contract (H differs, S stays valid) -> must complain
    node = next(n for n in g.nodes if n.op == "fused_int8_rope_sage_attn")
    bad = gs.Variable("bad", dtype=np.int8, shape=[1, 3, 2048, 128])
    g.inputs.append(bad)
    node.inputs[2] = bad
    assert S.validate_plugin_nodes(g) != []


def test_fused_attn_validation_rejects_short_seq():
    g = _post_rope_graph()
    S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    assert S.validate_plugin_nodes(g) == []
    # shrink S to the unsupported L<=1024 path -> must complain
    node = next(n for n in g.nodes if n.op == "fused_int8_rope_sage_attn")
    for i in (0, 2, 4):
        node.inputs[i].shape = [1, 2, 1024, 128]
    assert any("want >1024" in e for e in S.validate_plugin_nodes(g))


def test_cli_fuse_dit_self_attn(tmp_path):
    g = _post_rope_graph()
    g.opset = 17
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    onnx.save(gs.export_onnx(g), str(src))
    spec = "qi,qs,ki,ks,vi,vs,qn,kn,inv,ao"
    rc = S.main([str(src), str(dst), "--fuse-dit-self-attn", spec])
    assert rc == 0 and dst.exists()
    out_g = S.load(str(dst))
    assert S.summarize(out_g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 1}
    assert S.validate_plugin_nodes(out_g) == []


def test_add_sage_attn():
    B, Hq, Hkv, Sv, D = 1, 4, 2, 256, 128
    q = gs.Variable("q", dtype=F32, shape=[B, Hq, Sv, D])
    k = gs.Variable("k", dtype=F32, shape=[B, Hkv, Sv, D])
    v = gs.Variable("v", dtype=F32, shape=[B, Hkv, Sv, D])
    g = gs.Graph(nodes=[], inputs=[q, k, v], outputs=[])
    o = S.add_sage_attn(g, q, k, v)
    g.outputs = [o]
    g.cleanup().toposort()
    assert S.summarize(g)["plugin_nodes"] == {"sage_attn": 1}
    assert S.validate_plugin_nodes(g) == []


def test_sage_attn_validation():
    B, H, Sv, D = 1, 2, 256, 128
    q = gs.Variable("q", dtype=F32, shape=[B, H, Sv, D])
    k = gs.Variable("k", dtype=F32, shape=[B, H, Sv, D])
    v = gs.Variable("v", dtype=F32, shape=[B, H, Sv, D])
    g = gs.Graph(nodes=[], inputs=[q, k, v], outputs=[])
    S.add_sage_attn(g, q, k, v)
    assert S.validate_plugin_nodes(g) == []
    # bad D
    g2 = gs.Graph(nodes=[], inputs=[q, k, v], outputs=[])
    S.add_sage_attn(g2, q, k, v)
    g2.nodes[0].inputs[0].shape = [B, H, Sv, 32]
    assert S.validate_plugin_nodes(g2) != []
    # GQA violation
    k3 = gs.Variable("k3", dtype=F32, shape=[B, 3, Sv, D])
    v3 = gs.Variable("v3", dtype=F32, shape=[B, 3, Sv, D])
    g3 = gs.Graph(nodes=[], inputs=[q, k3, v3], outputs=[])
    S.add_sage_attn(g3, q, k3, v3)
    assert S.validate_plugin_nodes(g3) != []


def test_save_caps_ir_version(tmp_path):
    # Newer onnx defaults exceed what the TRT parser reads; save() caps it.
    N, D = 8, 32
    x = gs.Variable("x", dtype=F32, shape=[N, D])
    g = gs.Graph(nodes=[], inputs=[x], outputs=[x], opset=17)
    dst = tmp_path / "capped.onnx"
    S.save(g, str(dst))
    model = onnx.load(str(dst))
    assert model.ir_version <= S.TRT_MAX_IR_VERSION
    onnx.checker.check_model(model)
