# SPDX-License-Identifier: Apache-2.0
"""CPU-only surgeon tests: no torch, no TensorRT, no GPU."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
gs = pytest.importorskip("onnx_graphsurgeon")

from trt_dit_plugins import surgeon as S
from trt_dit_plugins.select import PluginOp

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
                       plugin=PluginOp.RMS_ADALN, eps=1e-6)
    assert [n.op for n in g.nodes if n.domain == S.PLUGIN_DOMAIN] == ["rms_adaln"]
    assert not [n for n in g.nodes if n.op in ("Mul", "Add")]
    assert S.summarize(g)["total"] == 1
    _check_ok(g)


def test_retarget_and_validate_catches_arity():
    x = gs.Variable("x", dtype=F32, shape=[1, 2, 8, 64])
    o = gs.Variable("o", dtype=F32, shape=[1, 2, 8, 64])
    g = gs.Graph(nodes=[gs.Node("MyAttn", inputs=[x, x, x], outputs=[o])],
                 inputs=[x], outputs=[o])
    assert S.retarget_nodes(g, {"MyAttn": PluginOp.INT8_ATTENTION}) == 1
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


def test_cli_refuses_to_save_invalid(tmp_path, capsys):
    g = _post_rope_graph()
    g.opset = 17
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    # break the D contract, then fuse: validation must fail and no file written.
    bad = gs.Variable("bad", dtype=np.int8, shape=[1, 2, 2048, 32])
    g.inputs.append(bad)
    onnx.save(gs.export_onnx(g), str(src))
    spec = "bad,qs,ki,ks,vi,vs,qn,kn,inv,ao"
    rc = S.main([str(src), str(dst), "--fuse-dit-self-attn", spec])
    assert rc == 1 and not dst.exists()
    assert "ERROR" in capsys.readouterr().out


def test_retarget_strips_stale_attrs():
    x = gs.Variable("x", dtype=F32, shape=[1, 2, 8, 64])
    o = gs.Variable("o", dtype=F32, shape=[1, 2, 8, 64])
    g = gs.Graph(nodes=[gs.Node("MyAttn", inputs=[x, x, x], outputs=[o],
                                attrs={"legacy_flag": 1})],
                 inputs=[x], outputs=[o])
    assert S.retarget_nodes(g, {"MyAttn": PluginOp.INT8_ATTENTION}) == 1
    node = g.nodes[0]
    assert node.attrs.get("legacy_flag") is None
    assert S.validate_plugin_nodes(g) == []


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
    g.cleanup().toposort()  # caller prunes (fuse leaves dead nodes for multi-block safety)
    assert node.op == "fused_int8_rope_sage_attn"
    assert [i.name for i in node.inputs] == [
        "qi", "qs", "ki", "ks", "vi", "vs", "qn", "kn", "inv"]
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 1}
    assert not [n for n in g.nodes if n.op in ("SageInt8Attn", "rms_rope_split_half")]
    tail = next(n for n in g.nodes if n.op == "Relu")
    # output tensor keeps its name: downstream consumers are untouched.
    assert tail.inputs[0].name == "ao"
    assert node.outputs[0] is tail.inputs[0]
    _check_ok(g)


def test_fuse_dit_self_attn_block_keeps_graph_output_name():
    g = _post_rope_graph()
    # attn_out as a graph output: the name must survive for I/O bindings.
    ao = next(t for t in g.tensors().values() if t.name == "ao")
    g.outputs = [ao]
    S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    g.cleanup().toposort()
    assert [o.name for o in g.outputs] == ["ao"]
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
    g.cleanup().toposort()
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


def test_fused_attn_validation_ignores_dynamic_symbols():
    g = _post_rope_graph()
    S.fuse_dit_self_attn_block(
        g, q_i8="qi", q_scale="qs", k_i8="ki", k_scale="ks", v_i8="vi",
        v_scale="vs", rms_w_q="qn", rms_w_k="kn", inv_freq="inv", attn_out="ao")
    node = next(n for n in g.nodes if n.op == "fused_int8_rope_sage_attn")
    # dynamic axes with differing symbol names must not false-positive.
    node.inputs[0].shape = [1, 2, "batch", 128]
    node.inputs[2].shape = [1, 2, "unk__0", 128]
    node.inputs[4].shape = [1, 2, None, 128]
    assert S.validate_plugin_nodes(g) == []
    # one dynamic input must not mask a real static mismatch of the others.
    node.inputs[2].shape = [1, 3, 2048, 128]
    node.inputs[4].shape = [1, 2, 2048, 128]
    assert any("shapes differ" in e for e in S.validate_plugin_nodes(g))
    # genuinely differing static shapes still complain.
    node.inputs[0].shape = [1, 2, 2048, 128]
    node.inputs[4].shape = [1, 2, 2048, 128]
    assert any("shapes differ" in e for e in S.validate_plugin_nodes(g))


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


def _baked_freqs(inv, S=8):
    fq = np.zeros((1, 1, S, 64, 2, 2), dtype=np.float32)
    for pos in range(S):
        ang = pos * inv
        fq[0, 0, pos, :, 0, 0] = np.cos(ang)
        fq[0, 0, pos, :, 1, 0] = np.sin(ang)
        fq[0, 0, pos, :, 0, 1] = -np.sin(ang)
        fq[0, 0, pos, :, 1, 1] = np.cos(ang)
    return fq


def test_inv_freq_from_baked_freqs():
    rng = np.random.default_rng(0)
    inv = (rng.random(64).astype(np.float32) * 0.5 + 0.001)
    assert np.allclose(S.inv_freq_from_baked_freqs(_baked_freqs(inv, S=8)), inv, atol=1e-6)
    with pytest.raises(ValueError, match="r=64"):
        S.inv_freq_from_baked_freqs(np.zeros((1, 1, 8, 32, 2, 2), np.float32))
    with pytest.raises(ValueError, match="\\[1,1,S,r,2,2\\]"):
        S.inv_freq_from_baked_freqs(np.zeros((1, 1, 8, 64, 2), np.float32))


def _discover_graph():
    B, H, D = 1, 2, 128
    g = _post_rope_graph()  # Sv=2048 self block (qi..ao)
    # cross: plain attention, no rope upstream
    cq = gs.Variable("cq", dtype=F32, shape=[B, H, 2048, D])
    ck = gs.Variable("ck", dtype=F32, shape=[B, H, 2048, D])
    cv = gs.Variable("cv", dtype=F32, shape=[B, H, 2048, D])
    co = gs.Variable("co", dtype=F32, shape=[B, H, 2048, D])
    g.inputs += [cq, ck, cv]
    g.nodes.append(gs.Node("Attention", inputs=[cq, ck, cv], outputs=[co]))
    # short-S self block (S=512 <= 1024 truth gate)
    st = gs.Variable("st", dtype=np.int8, shape=[B, H, 512, D])
    sk = gs.Variable("sk", dtype=np.int8, shape=[B, H, 512, D])
    sv = gs.Variable("sv", dtype=F32, shape=[B, H, 512, D])
    sf = gs.Variable("sf", dtype=F32, shape=[1, 1, 512, D // 2, 2, 2])
    sq = gs.Variable("sq", dtype=F32, shape=[D])
    skk = gs.Variable("skk", dtype=F32, shape=[D])
    srq = gs.Variable("srq", dtype=F32, shape=[B, H, 512, D])
    srk = gs.Variable("srk", dtype=F32, shape=[B, H, 512, D])
    sao = gs.Variable("sao", dtype=F32, shape=[B, H, 512, D])
    g.inputs += [st, sk, sv, sf, sq, skk]
    g.nodes.append(gs.Node("rms_rope_split_half", inputs=[st, sk, sf, sq, skk],
                           outputs=[srq, srk], domain="dit-plugins"))
    g.nodes.append(gs.Node("Attention", inputs=[srq, srk, sv], outputs=[sao]))
    # malformed: 2-input attention
    m1 = gs.Variable("m1", dtype=F32, shape=[B, H, 512, D])
    m2 = gs.Variable("m2", dtype=F32, shape=[B, H, 512, D])
    mo = gs.Variable("mo", dtype=F32, shape=[B, H, 512, D])
    g.inputs += [m1, m2]
    g.nodes.append(gs.Node("Attention", inputs=[m1, m2], outputs=[mo]))
    return g


def test_discover_dit_self_blocks():
    blocks = {b["attn_out"]: b for b in S.discover_dit_self_blocks(_discover_graph())}
    assert blocks["ao"]["eligible"] and blocks["ao"]["reason"] == "ok"
    assert blocks["ao"]["rms_w_q"] == "qsc" and blocks["ao"]["v"] == "vi"
    assert not blocks["co"]["eligible"] and blocks["co"]["reason"] == "cross-or-unpaired-qk"
    assert not blocks["sao"]["eligible"] and blocks["sao"]["reason"] == "S-512"
    assert not blocks["mo"]["eligible"] and blocks["mo"]["reason"] == "malformed-attn"


def _quantizable_graph():
    rng = np.random.default_rng(1)
    inv = (rng.random(64).astype(np.float32) * 0.5 + 0.001)
    B, H, Sv, D = 1, 2, 2048, 128
    I8 = np.int8
    qt = gs.Variable("qt", dtype=I8, shape=[B, H, Sv, D])
    kt = gs.Variable("kt", dtype=I8, shape=[B, H, Sv, D])
    vi = gs.Variable("vi", dtype=I8, shape=[B, H, Sv, D])
    fq = gs.Constant("fq", values=_baked_freqs(inv, Sv))
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
                 inputs=[qt, kt, vi, qsc, ksc], outputs=[out])
    g.opset = 23  # QuantizeLinear output_dtype (21+) for the anchors below
    return g, inv


def test_quantize_dit_self_anchors():
    g, inv = _quantizable_graph()
    rep = S.quantize_dit_self_anchors(g, {"qt": 0.02, "kt": 0.03, "vi": 0.04})
    assert rep["applied"] == 1 and rep["skipped"] == 0
    a = rep["anchors"]["ao"]
    assert (a["q_i8"], a["q_scale"], a["rms_w_q"], a["inv_freq"]) == \
        ("ao_fused_qi", "ao_fused_qs", "qsc", "fused_shared_inv_freq")
    qls = [n for n in g.nodes if n.op == "QuantizeLinear"]
    assert len(qls) == 3
    assert all(n.attrs["output_dtype"] == 3 for n in qls)
    got = next(t for t in g.tensors().values() if t.name == "fused_shared_inv_freq")
    assert np.allclose(got.values, inv, atol=1e-6)
    # carrier present pre-fusion (transient, pruned by post-fusion cleanup)
    assert any("carrier" in n.name for n in g.nodes)


def test_apply_dit_self_fused():
    g, _ = _quantizable_graph()
    rep = S.apply_dit_self_fused(g, {"qt": 0.02, "kt": 0.03, "vi": 0.04})
    assert rep["applied"] == 1 and rep["by_reason"] == {"ok": 1}
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 1}
    assert not [n for n in g.nodes if n.op in ("SageInt8Attn", "rms_rope_split_half")]
    assert not [n for n in g.nodes if "carrier" in n.name]
    assert S.validate_plugin_nodes(g) == []
    _check_ok(g)


def test_apply_dit_self_fused_missing_scales():
    g, _ = _quantizable_graph()
    rep = S.apply_dit_self_fused(g, {})
    assert rep == {"applied": 0, "skipped": 1, "by_reason": {"missing-scales": 1}}
    assert not [n for n in g.nodes if n.op == "fused_int8_rope_sage_attn"]


def _pre_rope_graph():
    B, H, S, D, Hd = 1, 2, 256, 128, 2
    q = gs.Variable("q", dtype=F32, shape=[B, S, Hd, D])
    k = gs.Variable("k", dtype=F32, shape=[B, S, Hd, D])
    v = gs.Variable("v", dtype=F32, shape=[B, S, Hd, D])
    qs = gs.Variable("qs", dtype=F32, shape=[D])
    ks = gs.Variable("ks", dtype=F32, shape=[D])
    nq = gs.Variable("nq", dtype=F32, shape=[B, S, Hd, D])
    nk = gs.Variable("nk", dtype=F32, shape=[B, S, Hd, D])
    c1 = gs.Constant("c1", values=np.ones((1,), dtype=F32))
    mq = gs.Variable("mq", dtype=F32, shape=[B, S, Hd, D])
    mk = gs.Variable("mk", dtype=F32, shape=[B, S, Hd, D])
    aq = gs.Variable("aq", dtype=F32, shape=[B, H, S, D])
    ak = gs.Variable("ak", dtype=F32, shape=[B, H, S, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, H, S, D])
    out = gs.Variable("out", dtype=F32, shape=[B, H, S, D])
    # cross: norm feeds attention directly (no Mul)
    cx = gs.Variable("cx", dtype=F32, shape=[B, S, Hd, D])
    cxs = gs.Variable("cxs", dtype=F32, shape=[D])
    cn = gs.Variable("cn", dtype=F32, shape=[B, S, Hd, D])
    co = gs.Variable("co", dtype=F32, shape=[B, H, S, D])
    nodes = [
        gs.Node("RMSNormalization", inputs=[q, qs], outputs=[nq]),
        gs.Node("RMSNormalization", inputs=[k, ks], outputs=[nk]),
        gs.Node("Mul", inputs=[nq, c1], outputs=[mq]),
        gs.Node("Mul", inputs=[nk, c1], outputs=[mk]),
        gs.Node("Attention", inputs=[mq, mk, v], outputs=[ao]),
        gs.Node("RMSNormalization", inputs=[cx, cxs], outputs=[cn]),
        gs.Node("Attention", inputs=[cn, cn, v], outputs=[co]),
        gs.Node("Relu", inputs=[ao], outputs=[out]),
    ]
    g = gs.Graph(nodes=nodes,
                 inputs=[q, k, v, qs, ks, cx, cxs], outputs=[out, co])
    g.opset = 23  # RMSNormalization needs opset 23+
    return g


def test_fuse_rope_blocks():
    rng = np.random.default_rng(2)
    g = _pre_rope_graph()
    n = S.fuse_rope_blocks(g, _baked_freqs(rng.random(64).astype(F32) * 0.1 + 0.01, 256))
    assert n == 1
    rope = [x for x in g.nodes if x.op == "rms_rope_split_half"]
    assert len(rope) == 1 and len(rope[0].inputs) == 5
    assert rope[0].attrs["epsilon"] == pytest.approx(1e-6)
    attn = next(x for x in g.nodes if x.op == "Attention" and x.outputs[0].name == "ao")
    assert attn.inputs[0].inputs[0].op == "rms_rope_split_half"
    assert attn.inputs[0].inputs[0].name.endswith("_rmsrope")
    cross = next(x for x in g.nodes if x.op == "Attention" and x.outputs[0].name == "co")
    assert [i.name for i in cross.inputs] == ["cn", "cn", "v"]
    assert S.summarize(g)["plugin_nodes"] == {"rms_rope_split_half": 1}
    _check_ok(g)


def test_fuse_rope_blocks_rejects():
    g = _pre_rope_graph()
    g.nodes = [n for n in g.nodes if n.op != "Attention" or n.outputs[0].name == "co"]
    # only cross left -> zero matches
    with pytest.raises(RuntimeError, match="0 blocks"):
        S.fuse_rope_blocks(g, np.zeros((1, 1, 8, 64, 2, 2), F32))
    with pytest.raises(ValueError, match="\\[1,1,S,r,2,2\\]"):
        S.fuse_rope_blocks(_pre_rope_graph(), np.zeros((8, 64), F32))
    bad = _pre_rope_graph()
    bad_attn = next(n for n in bad.nodes if n.op == "Attention" and n.outputs[0].name == "ao")
    bad_attn.inputs.pop()
    with pytest.raises(ValueError, match="malformed"):
        S.fuse_rope_blocks(bad, np.zeros((1, 1, 8, 64, 2, 2), F32))


def test_fuse_rope_blocks_idempotent():
    rng = np.random.default_rng(3)
    g = _pre_rope_graph()
    fq = _baked_freqs(rng.random(64).astype(F32) * 0.1 + 0.01, 256)
    assert S.fuse_rope_blocks(g, fq) == 1
    # second run: self block already rope-fed, cross still cross -> 0 blocks
    with pytest.raises(RuntimeError, match="0 blocks"):
        S.fuse_rope_blocks(g, fq)
    assert S.summarize(g)["plugin_nodes"] == {"rms_rope_split_half": 1}


def test_fuse_rope_blocks_dynamic_shape():
    g = _pre_rope_graph()
    for name in ("q", "k"):
        t = next(t for t in g.tensors().values() if t.name == name)
        t.shape = [1, "S", 2, 128]
    rng = np.random.default_rng(4)
    assert S.fuse_rope_blocks(g, _baked_freqs(rng.random(64).astype(F32) * 0.1 + 0.01, 256)) == 1
    rope = next(n for n in g.nodes if n.op == "rms_rope_split_half")
    assert rope.inputs[0].shape is None  # unknown dims stay unknown
    _check_ok(g)


def test_apply_sage_attention():
    B, H, Sv, D = 1, 2, 256, 128
    q = gs.Variable("q", dtype=F32, shape=[B, H, Sv, D])
    k = gs.Variable("k", dtype=F32, shape=[B, H, Sv, D])
    v = gs.Variable("v", dtype=F32, shape=[B, H, Sv, D])
    o = gs.Variable("o", dtype=F32, shape=[B, H, Sv, D])
    g = gs.Graph(nodes=[gs.Node("Attention", inputs=[q, k, v], outputs=[o],
                                       name="attn0")],
                 inputs=[q, k, v], outputs=[o])
    assert S.apply_sage_attention(g) == 1
    node = g.nodes[0]
    assert node.op == "SageInt8Attn" and node.domain != S.PLUGIN_DOMAIN
    assert node.attrs == {} and node.name == "attn0_sage"
    assert [i.name for i in node.inputs] == ["q", "k", "v"]
    # NOTE: no _check_ok here — SageInt8Attn is a legacy op with no schema;
    # the structural asserts above are the contract.


def test_apply_sage_attention_rejects():
    B, H, Sv, D = 1, 2, 256, 128
    mk = lambda n, **kw: gs.Graph(
        nodes=[gs.Node("Attention", inputs=[gs.Variable(f"i{j}", dtype=F32, shape=[B, H, Sv, D])
                                            for j in range(kw.pop("nin", 3))],
                       outputs=[gs.Variable("o", dtype=F32, shape=[B, H, Sv, D])], **kw)],
        inputs=[], outputs=[])
    with pytest.raises(ValueError, match="mask"):
        S.apply_sage_attention(mk("m", nin=4))
    with pytest.raises(RuntimeError, match="causal"):
        S.apply_sage_attention(mk("c", attrs={"is_causal": 1}))
    g = gs.Graph(nodes=[], inputs=[], outputs=[])
    with pytest.raises(RuntimeError, match="0 Attention"):
        S.apply_sage_attention(g)


def test_fuse_norm_affine_bf16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    BF16 = ml_dtypes.bfloat16
    N, D = 8, 32

    def mkgraph():
        x = gs.Variable("x", dtype=BF16, shape=[N, D])
        sc = gs.Variable("sc", dtype=BF16, shape=[1, D])
        sh = gs.Variable("sh", dtype=BF16, shape=[1, D])
        norm = gs.Variable("norm", dtype=BF16, shape=[N, D])
        a = gs.Variable("a", dtype=BF16, shape=[N, D])
        o = gs.Variable("o", dtype=BF16, shape=[N, D])
        return gs.Graph(nodes=[gs.Node("Relu", inputs=[x], outputs=[norm]),
                               gs.Node("Mul", inputs=[norm, sc], outputs=[a]),
                               gs.Node("Add", inputs=[a, sh], outputs=[o])],
                        inputs=[x, sc, sh], outputs=[o])

    for plugin in (PluginOp.ADALN, PluginOp.RMS_ADALN):
        g = mkgraph()
        S.fuse_norm_affine(g, norm="norm", scale="sc", shift="sh", out="o",
                           plugin=plugin)
        assert [n.op for n in g.nodes if n.domain == S.PLUGIN_DOMAIN] == [plugin]
        assert S.summarize(g)["total"] == 1
        _check_ok(g)


def test_save_falls_back_to_external_data(tmp_path, monkeypatch):
    import onnx as _onnx

    N, D = 8, 32
    x = gs.Variable("x", dtype=F32, shape=[N, D])
    g = gs.Graph(nodes=[], inputs=[x], outputs=[x], opset=17)
    dst = tmp_path / "big.onnx"
    calls = []

    def flaky(*a, **k):
        calls.append(k)
        if not k:  # first (plain) attempt fails like a >2GB protobuf write
            raise ValueError("exceeds maximum protobuf size of 2GB")
        return None

    monkeypatch.setattr(_onnx, "save", flaky)
    S.save(g, str(dst))
    assert len(calls) == 2
    retry = calls[1]
    assert retry.get("save_as_external_data") is True
    assert retry.get("all_tensors_to_one_file") is True
    # location must be basename-relative, never an absolute path.
    assert retry.get("location") == "big.onnx.data"


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


def test_discover_cache_hit_and_copy():
    S._QUERY_CACHE.clear()
    first = S.discover_dit_self_blocks(_post_rope_graph())
    assert len(S._QUERY_CACHE) == 1
    # structurally identical graph object shares the entry (same input).
    second = S.discover_dit_self_blocks(_post_rope_graph())
    assert second == first and second is not first
    assert len(S._QUERY_CACHE) == 1
    # caller-side mutation must not leak into the cache.
    second[0]["eligible"] = not second[0]["eligible"]
    assert S.discover_dit_self_blocks(_post_rope_graph()) == first


def test_discover_cache_invalidated_by_mutation():
    g = _post_rope_graph()
    assert S.discover_dit_self_blocks(g)[0]["eligible"]
    rope = next(n for n in g.nodes if n.op == "rms_rope_split_half")
    for i in (0, 1):
        rope.inputs[i].shape = [1, 2, 512, 128]
    after = S.discover_dit_self_blocks(g)
    assert not after[0]["eligible"] and after[0]["reason"] == "S-512"


def _rope_fused_self(Sv=256):
    B, Hd, D = 1, 2, 128
    q = gs.Variable("q", dtype=F32, shape=[B, Sv, Hd, D])
    k = gs.Variable("k", dtype=F32, shape=[B, Sv, Hd, D])
    v = gs.Variable("v", dtype=F32, shape=[B, Sv, Hd, D])
    qs = gs.Variable("qs", dtype=F32, shape=[D])
    ks = gs.Variable("ks", dtype=F32, shape=[D])
    nq = gs.Variable("nq", dtype=F32, shape=[B, Sv, Hd, D])
    nk = gs.Variable("nk", dtype=F32, shape=[B, Sv, Hd, D])
    c1 = gs.Constant("c1", values=np.ones((1,), dtype=F32))
    mq = gs.Variable("mq", dtype=F32, shape=[B, Sv, Hd, D])
    mk = gs.Variable("mk", dtype=F32, shape=[B, Sv, Hd, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, Sv, Hd, D])
    out = gs.Variable("out", dtype=F32, shape=[B, Sv, Hd, D])
    g = gs.Graph(nodes=[
        gs.Node("RMSNormalization", inputs=[q, qs], outputs=[nq]),
        gs.Node("RMSNormalization", inputs=[k, ks], outputs=[nk]),
        gs.Node("Mul", inputs=[nq, c1], outputs=[mq]),
        gs.Node("Mul", inputs=[nk, c1], outputs=[mk]),
        gs.Node("Attention", inputs=[mq, mk, v], outputs=[ao], name="attn"),
        gs.Node("Relu", inputs=[ao], outputs=[out]),
    ], inputs=[q, k, v, qs, ks], outputs=[out])
    g.opset = 23
    rng = np.random.default_rng(11)
    inv = (rng.random(64).astype(F32) * 0.5 + 0.001)
    assert S.fuse_rope_blocks(g, _baked_freqs(inv, Sv)) == 1
    return g


def test_apply_attention_plugins_tier2_sage_pin():
    B, H, Sv, D = 1, 2, 256, 128
    q = gs.Variable("q", dtype=F32, shape=[B, H, Sv, D])
    k = gs.Variable("k", dtype=F32, shape=[B, H, Sv, D])
    v = gs.Variable("v", dtype=F32, shape=[B, H, Sv, D])
    o = gs.Variable("o", dtype=F32, shape=[B, H, Sv, D])
    g = gs.Graph(nodes=[gs.Node("Attention", inputs=[q, k, v], outputs=[o],
                                name="attn0")],
                 inputs=[q, k, v], outputs=[o])
    dry = S.apply_attention_plugins(g, dry_run=True)
    assert dry["applied"] == {"SageInt8Attn": 1} and dry["native"] == 0
    assert [n.op for n in g.nodes] == ["Attention"]  # dry-run never mutates
    rep = S.apply_attention_plugins(g)
    assert rep["applied"] == {"SageInt8Attn": 1}
    assert g.nodes[0].op == "SageInt8Attn"
    site = rep["sites"][0]
    assert site["tier"] == 2 and site["emitted"] == "SageInt8Attn"
    assert site["selected"] in ("sage_attn", "int8_attention")
    rep2 = S.apply_attention_plugins(g)
    assert rep2["applied"] == {} and rep2["native"] == 0
    assert sum(n.op == "SageInt8Attn" for n in g.nodes) == 1


def test_apply_attention_plugins_tier1_experimental_fused():
    g, _ = _quantizable_graph()
    sc = {"qt": 0.02, "kt": 0.03, "vi": 0.04}
    dry = S.apply_attention_plugins(g, scales=sc, dry_run=True)
    assert dry["applied"] == {"fused_int8_rope_sage_attn": 1}
    assert S.summarize(g)["plugin_nodes"] == {"rms_rope_split_half": 1}
    rep = S.apply_attention_plugins(g, scales=sc)
    assert rep["applied"] == {"fused_int8_rope_sage_attn": 1}
    assert "experimental" in rep["sites"][0]["reason"]
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 1}
    assert S.validate_plugin_nodes(g) == []
    _check_ok(g)
    rep2 = S.apply_attention_plugins(g, scales=sc, dry_run=True)
    assert rep2["applied"] == {} and rep2["sites"][0]["tier"] == 3


def test_apply_attention_plugins_tier1_strict_with_samples():
    g, _ = _quantizable_graph()
    sc = {"qt": 0.02, "kt": 0.03, "vi": 0.04}
    rng = np.random.default_rng(7)
    samples = {k: (rng.random((1, 2, 2048, 128)).astype(F32) - 0.5) * 0.2
               for k in ("qt", "kt", "vi")}
    rep = S.apply_attention_plugins(g, scales=sc, samples=samples)
    assert rep["applied"] == {"fused_int8_rope_sage_attn": 1}
    assert "experimental" not in rep["sites"][0]["reason"]
    assert S.validate_plugin_nodes(g) == []


def test_apply_attention_plugins_small_s_falls_back_to_sage():
    g = _rope_fused_self(256)
    b = S.discover_dit_self_blocks(g)[0]
    assert not b["eligible"] and b["reason"] == "S-256"
    sc = {b["q"]: 0.05, b["k"]: 0.05, b["v"]: 0.05}
    rep = S.apply_attention_plugins(g, scales=sc)
    assert rep["applied"] == {"SageInt8Attn": 1}
    assert "fused N/A (S-256)" in rep["sites"][0]["reason"]


def test_apply_sage_attention_only_subset():
    B, H, Sv, D = 1, 2, 256, 128
    g = gs.Graph(nodes=[
        gs.Node("Attention",
                inputs=[gs.Variable(f"a{i}", dtype=F32, shape=[B, H, Sv, D])
                        for i in range(3)],
                outputs=[gs.Variable("oa", dtype=F32, shape=[B, H, Sv, D])],
                name="a"),
        gs.Node("Attention",
                inputs=[gs.Variable(f"b{i}", dtype=F32, shape=[B, H, Sv, D])
                        for i in range(3)],
                outputs=[gs.Variable("ob", dtype=F32, shape=[B, H, Sv, D])],
                name="b"),
    ], inputs=[], outputs=[])
    assert S.apply_sage_attention(g, only={"b"}) == 1
    assert {n.name: n.op for n in g.nodes} == \
        {"a": "Attention", "b_sage": "SageInt8Attn"}


def _two_block_graph():
    rng = np.random.default_rng(21)
    inv = (rng.random(64).astype(F32) * 0.5 + 0.001)
    nodes, inputs, outputs = [], [], []
    for p in ("a", "b"):
        qt = gs.Variable(f"{p}qt", dtype=np.int8, shape=[1, 2, 2048, 128])
        kt = gs.Variable(f"{p}kt", dtype=np.int8, shape=[1, 2, 2048, 128])
        vi = gs.Variable(f"{p}vi", dtype=np.int8, shape=[1, 2, 2048, 128])
        fq = gs.Constant(f"{p}fq", values=_baked_freqs(inv, 2048))
        qsc = gs.Variable(f"{p}qsc", dtype=F32, shape=[128])
        ksc = gs.Variable(f"{p}ksc", dtype=F32, shape=[128])
        rq = gs.Variable(f"{p}rq", dtype=F32, shape=[1, 2, 2048, 128])
        rk = gs.Variable(f"{p}rk", dtype=F32, shape=[1, 2, 2048, 128])
        ao = gs.Variable(f"{p}ao", dtype=F32, shape=[1, 2, 2048, 128])
        out = gs.Variable(f"{p}out", dtype=F32, shape=[1, 2, 2048, 128])
        nodes += [
            gs.Node("rms_rope_split_half", inputs=[qt, kt, fq, qsc, ksc],
                    outputs=[rq, rk], domain="dit-plugins"),
            gs.Node("SageInt8Attn", inputs=[rq, rk, vi], outputs=[ao]),
            gs.Node("Relu", inputs=[ao], outputs=[out]),
        ]
        inputs += [qt, kt, vi, qsc, ksc]
        outputs.append(out)
    g = gs.Graph(nodes=nodes, inputs=inputs, outputs=outputs)
    g.opset = 23
    return g


def test_apply_dit_self_fused_multi_block():
    # Regression: quantize-all-then-fuse must not prune block b's anchors
    # while fusing block a (per-fuse cleanup did exactly that).
    g = _two_block_graph()
    sc = {"aqt": 0.02, "akt": 0.03, "avi": 0.04,
          "bqt": 0.05, "bkt": 0.06, "bvi": 0.07}
    rep = S.apply_dit_self_fused(g, sc)
    assert rep["applied"] == 2 and rep["by_reason"] == {"ok": 2}
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 2}
    assert S.validate_plugin_nodes(g) == []
    _check_ok(g)


def test_apply_attention_plugins_multi_block_fused():
    g = _two_block_graph()
    sc = {"aqt": 0.02, "akt": 0.03, "avi": 0.04,
          "bqt": 0.05, "bkt": 0.06, "bvi": 0.07}
    rep = S.apply_attention_plugins(g, scales=sc)
    assert rep["applied"] == {"fused_int8_rope_sage_attn": 2}
    assert S.summarize(g)["plugin_nodes"] == {"fused_int8_rope_sage_attn": 2}
    assert S.validate_plugin_nodes(g) == []
    _check_ok(g)


def test_apply_attention_plugins_fp8_vehicle_excludes_fused():
    # FP8 exports must not emit INT8 anchors (TRT11 rejects INT8+FP8
    # mixed graphs pre-Blackwell): fused is out, sage stays.
    from trt_dit_plugins.select import BasePrecision, GemmKind
    g, _ = _quantizable_graph()
    for n in g.nodes:
        if n.op == "SageInt8Attn":
            n.op = "Attention"  # unconverted site: Tier2 must sage it
    sc = {"qt": 0.02, "kt": 0.03, "vi": 0.04}
    rep = S.apply_attention_plugins(g, scales=sc, gemm=GemmKind.FP8,
                                    base=BasePrecision.BF16, arch="sm89")
    assert rep["applied"] == {"SageInt8Attn": 1}
    assert not [n for n in g.nodes if n.op == "fused_int8_rope_sage_attn"]
    assert not [n for n in g.nodes if n.op == "QuantizeLinear"]
