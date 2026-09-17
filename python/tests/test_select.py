# SPDX-License-Identifier: Apache-2.0
"""CPU-only selector tests: no torch, no TensorRT, no GPU."""

import numpy as np
import pytest

from trt_dit_plugins import select as C
from trt_dit_plugins.select import BasePrecision, QuantKind

gs = pytest.importorskip("onnx_graphsurgeon")

F32 = np.float32
rng = np.random.default_rng(7)


def _compat(x=None, scale=0.02, base="bf16", min_cos=None):
    if x is None:
        x = rng.normal(0, 1, size=(4, 64)).astype(F32)
    return C.simulate_upcast_compat(x, scale, base=base,
                                    quant=QuantKind.INT8, min_cos=min_cos)


def test_enum_coercion():
    assert C.select_attention(base="BF16", S=256, D=64)["op"] == "int8_attention"
    assert C.select_attention(base=BasePrecision.F16, S=256, D=64)["op"] == "int8_attention"
    with pytest.raises(ValueError, match="base precision"):
        C.select_attention(base="fp8", S=256, D=64)
    with pytest.raises(ValueError, match="quant"):
        C.select_attention(base="f16", quant="f16", S=256, D=64)
    with pytest.raises(ValueError, match="norm"):
        C.select_norm(base="f16", norm="alien")


def test_fused_needs_compat_evidence():
    no_evidence = C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                                     Hq=16, Hkv=16, arch="sm89", rope_fusable=True)
    assert no_evidence["op"] is None and "simulation required" in no_evidence["reason"]
    ok = C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                            Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                            compat=_compat())
    assert ok["op"] == "fused_int8_rope_sage_attn"
    assert ok["trt_flags"] == {"network": ["STRONGLY_TYPED"], "builder": []}
    bad = C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                             Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                             compat={"pass": False, "cos": 0.5, "reason": "x"})
    assert bad["op"] is None


def test_fused_scope_min_cos_applies_to_sim_not_table():
    # min_cos above the static table value must NOT kill fused (lossy by
    # design); it gates the measured sim cos instead.
    assert C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              min_cos=0.9995,
                              compat=_compat(min_cos=0.9995))["op"] == \
        "fused_int8_rope_sage_attn"
    gated = C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                               Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                               min_cos=0.99999,
                               compat={"pass": True, "cos": 0.9999,
                                       "reason": "x"})
    assert gated["op"] is None and "sim cos" in gated["reason"]


def test_fused_seq_gates_and_s_bucket():
    assert C.select_attention(base="bf16", quant="int8", S=1024, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["op"] is None
    assert C.select_attention(base="bf16", quant="int8", S=None, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["op"] is None
    assert C.select_attention(base="bf16", quant="int8", S=9217, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["op"] is None
    assert C.select_attention(base="bf16", quant="int8", S=2048, D=64,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["op"] is None
    # S-bucket: exact points report their own cos, unmeasured S is conservative.
    assert C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["cos"] == pytest.approx(0.99855)
    assert C.select_attention(base="bf16", quant="int8", S=2048, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["cos"] == pytest.approx(0.9977)
    assert C.select_attention(base="bf16", quant="int8", S=3000, D=128,
                              Hq=16, Hkv=16, arch="sm89", rope_fusable=True,
                              compat=_compat())["cos"] == pytest.approx(0.9977)


def test_known_bad_regressions():
    for entry in C.KNOWN_BAD:
        r = C.select_attention(**entry["probe"])
        assert r["op"] != entry["op"], entry
        assert r["op"] is None, entry


def test_dense_float_prefers_int8_static():
    r = C.select_attention(base="f16", S=4096, D=128, Hq=16, Hkv=16, arch="sm89")
    assert r["op"] == "int8_attention"


def test_gqa_routes_to_sage():
    r = C.select_attention(base="bf16", S=256, D=128, Hq=4, Hkv=2, arch="sm90")
    assert r["op"] == "sage_attn" and r["fields"] == {}
    bad = C.select_attention(base="bf16", S=256, D=128, Hq=4, Hkv=3, arch="sm90")
    assert bad["op"] is None


def test_causal_and_mask_fall_to_native():
    r = C.select_attention(base="f16", S=4096, D=128, causal=True)
    assert r["op"] is None and r["trt_flags"]["builder"] == ["FP16"]
    m = C.select_attention(base="f16", S=256, D=64, has_mask=True)
    assert m["op"] is None


def test_sparse_needs_block_mask_and_sm89():
    ok = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                            arch="sm89", sparse_mask=True)
    assert ok["op"] == "block_sparse_sage2_attn"
    off = C.select_attention(base="f16", S=4096, D=128, arch="sm100",
                             sparse_mask=True)
    assert off["op"] == "int8_attention"
    odd = C.select_attention(base="f16", S=100, D=128, arch="sm89",
                             sparse_mask=True)
    assert odd["op"] == "int8_attention"


def test_quant_inputs_go_native_with_combined_flags():
    f = C.select_attention(base="f16", quant="fp8", S=4096, D=128, arch="sm89")
    assert f["op"] is None
    assert f["trt_flags"] == {"network": ["STRONGLY_TYPED"],
                              "builder": ["FP16", "FP8"]}
    i = C.select_attention(base="bf16", quant="int8", S=4096, D=128, arch="sm89")
    assert i["op"] is None and i["trt_flags"]["builder"] == ["BF16", "INT8"]


def test_gemm_pins():
    pv = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                            arch="sm89", gemm="fp8")
    assert pv["op"] == "sage_attn" and pv["fields"] == {"fp8_pv": 1}
    no_pv = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                               arch="sm90", gemm="fp8")
    assert no_pv["op"] is None and "FP8" in no_pv["trt_flags"]["builder"]
    plain = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                               arch="sm89", gemm="fp16")
    assert plain["op"] is None and plain["trt_flags"]["builder"] == ["FP16"]
    with pytest.raises(ValueError, match="gemm"):
        C.select_attention(base="f16", S=4096, D=128, gemm="int4")


def test_min_cos_gates_quant_class_only():
    assert C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                              arch="sm89", min_cos=0.9995)["op"] == "int8_attention"
    assert C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                              arch="sm89", min_cos=0.99995)["op"] is None


def test_bench_override_and_incomplete_table():
    full = {("int8_attention", 4096, 128, 8, 8, 89): 0.9,
            ("sage_attn", 4096, 128, 8, 8, 89): 0.5}
    assert C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                              arch="sm89", bench=full)["op"] == "sage_attn"
    partial = {("sage_attn", 4096, 128, 8, 8, 89): 0.5}
    assert C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                              arch="sm89", bench=partial)["op"] == "int8_attention"


def test_bench_key_includes_heads():
    # MHA entry must not leak into a GQA query: miss -> static rank (sage).
    mha_only = {("int8_attention", 4096, 128, 8, 8, 89): 0.9,
                ("sage_attn", 4096, 128, 8, 8, 89): 0.5}
    r = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=2,
                           arch="sm89", bench=mha_only)
    assert r["op"] == "sage_attn" and "static rank" in r["reason"]
    # legacy 4-tuple keys fail fast instead of silently missing.
    with pytest.raises(ValueError, match=r"\(op, S, D, Hq, Hkv, arch\)"):
        C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                           arch="sm89",
                           bench={("sage_attn", 4096, 128, 89): 0.5})


def test_arch_optional_defaults_to_unknown():
    r = C.select_attention(base="bf16", quant="int8", S=4096, D=128,
                           Hq=16, Hkv=16, rope_fusable=True, compat=_compat())
    assert r["op"] == "fused_int8_rope_sage_attn"
    s = C.select_attention(base="f16", S=4096, D=128, Hq=8, Hkv=8,
                           sparse_mask=True)
    assert s["op"] == "int8_attention"


def test_norm_and_rope():
    assert C.select_norm(base="f16", norm="rms")["op"] == "rms_adaln"
    assert C.select_norm(base="bf16", norm="layer")["op"] == "adaln"
    qn = C.select_norm(base="f16", quant="int8", norm="rms")
    assert qn["op"] is None and qn["trt_flags"]["builder"] == ["FP16", "INT8"]
    r = C.select_rope(base="f16", style="split-half", fuse_norm=True,
                      D=128, has_scales=True)
    assert r["op"] == "rms_rope_split_half"
    assert C.select_rope(base="f16", style="split-half", fuse_norm=True,
                         D=128)["op"] is None
    assert C.select_rope(base="bf16", style="interleaved")["op"] == "apply_rope"
    assert C.select_rope(base="bf16", quant="fp8", style="interleaved")["op"] is None


def test_sim_int8_roundtrip():
    x = rng.normal(0, 0.5, size=(8, 128)).astype(F32)
    r = C.simulate_upcast_compat(x, 0.02, base="bf16", quant="int8")
    assert r["pass"] and r["cos"] > 0.999 and r["saturation"] == 0.0
    # saturation is reported, not fatal without a bar.
    wide = C.simulate_upcast_compat(np.full((4, 8), 10.0, F32), 0.01,
                                    base="f32", quant="int8")
    assert wide["saturation"] > 0.0 and wide["pass"]
    nan = C.simulate_upcast_compat(np.full((2, 4), np.nan, dtype=F32), 0.02,
                                   base="f32", quant="int8")
    assert not nan["pass"] and "finite" in nan["reason"]
    with pytest.raises(ValueError, match="positive"):
        C.simulate_upcast_compat(x, 0.0, base="f32", quant="int8")
    with pytest.raises(ValueError, match="scalar"):
        C.simulate_upcast_compat(x, [0.02, 0.03], base="f32", quant="int8")


def test_sim_e4m3_runs():
    x = (rng.normal(0, 0.3, size=(8, 64))).astype(F32)
    r = C.simulate_upcast_compat(x, 0.05, base="f16", quant="fp8")
    assert r["pass"] and r["cos"] > 0.99
    # E4M3 code bounds: max representable is 448.
    q = C._e4m3_nearest(np.array([1000.0, -1000.0, 0.0]))
    assert list(q) == [448.0, -448.0, 0.0]


def test_sim_gpu_crosscheck():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for torch fp8 cast")
    x = torch.rand(256, dtype=torch.float32, device="cuda") * 4 - 2
    got = C._e4m3_nearest(x.cpu().numpy())
    want = x.to(torch.float8_e4m3fn).to(torch.float32).cpu().numpy()
    assert np.array_equal(got, want)


def _site_graph():
    B, H, Sv, D = 1, 2, 2048, 128
    qt = gs.Variable("qt", dtype=F32, shape=[B, H, Sv, D])
    kt = gs.Variable("kt", dtype=F32, shape=[B, H, Sv, D])
    vi = gs.Variable("vi", dtype=F32, shape=[B, H, Sv, D])
    fq = gs.Constant("fq", values=np.zeros((1, 1, Sv, 64, 2, 2), F32))
    qsc = gs.Variable("qsc", dtype=F32, shape=[D])
    ksc = gs.Variable("ksc", dtype=F32, shape=[D])
    rq = gs.Variable("rq", dtype=F32, shape=[B, H, Sv, D])
    rk = gs.Variable("rk", dtype=F32, shape=[B, H, Sv, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, H, Sv, D])
    out = gs.Variable("out", dtype=F32, shape=[B, H, Sv, D])
    gq = gs.Variable("gq", dtype=F32, shape=[1, 4, 256, D])
    gk = gs.Variable("gk", dtype=F32, shape=[1, 2, 256, D])
    gv = gs.Variable("gv", dtype=F32, shape=[1, 2, 256, D])
    go = gs.Variable("go", dtype=F32, shape=[1, 4, 256, D])
    pq = gs.Variable("pq", dtype=F32, shape=[1, 2, 256, D])
    po = gs.Variable("po", dtype=F32, shape=[1, 2, 256, D])
    g = gs.Graph(
        nodes=[gs.Node("rms_rope_split_half", inputs=[qt, kt, fq, qsc, ksc],
                       outputs=[rq, rk], domain="dit-plugins"),
               gs.Node("SageInt8Attn", inputs=[rq, rk, vi], outputs=[ao]),
               gs.Node("Relu", inputs=[ao], outputs=[out]),
               gs.Node("Attention", inputs=[gq, gk, gv], outputs=[go], name="gqa0"),
               gs.Node("int8_attention", inputs=[pq, pq, pq], outputs=[po],
                       domain="dit-plugins", name="done0")],
        inputs=[qt, kt, vi, qsc, ksc, gq, gk, gv, pq], outputs=[out, go, po])
    return g


def _calib():
    sc = {"qt": 0.02, "kt": 0.03, "vi": 0.04}
    sa = {k: rng.normal(0, 0.5, size=(2, 128)).astype(F32) for k in sc}
    return sc, sa


def test_discover_all_tiers():
    sc, sa = _calib()
    sites = {s["attn"]: s for s in C.discover_attention_sites(
        _site_graph(), base="bf16", arch="sm89", scales=sc, samples=sa)}
    t1 = next(s for k, s in sites.items() if s["tier"] == 1)
    assert t1["op"] == "fused_int8_rope_sage_attn"
    assert sites["gqa0"]["tier"] == 2 and sites["gqa0"]["op"] == "sage_attn"
    assert sites["done0"]["tier"] == 3 and sites["done0"]["op"] == "int8_attention"
    # samples missing -> Tier1 pair reports native (simulation required).
    sites2 = C.discover_attention_sites(_site_graph(), base="bf16",
                                        arch="sm89", scales=sc)
    assert all(s["op"] != "fused_int8_rope_sage_attn" for s in sites2)
    t1b = next(s for s in sites2 if s["tier"] == 1)
    assert t1b["op"] is None and "simulation required" in t1b["reason"]


def test_cli_dry_run(tmp_path, capsys):
    import onnx

    from trt_dit_plugins import surgeon as S

    g = _site_graph()
    src = tmp_path / "in.onnx"
    onnx.save(gs.export_onnx(g), str(src))
    sc, sa = _calib()
    scales_p = tmp_path / "scales.json"
    scales_p.write_text('{"qt": 0.02, "kt": 0.03, "vi": 0.04}')
    npz_p = tmp_path / "samples.npz"
    np.savez(str(npz_p), qt=sa["qt"], kt=sa["kt"], vi=sa["vi"])
    rc = C.main([str(src), "--base", "bf16", "--arch", "sm89",
                 "--scales-json", str(scales_p), "--samples-npz", str(npz_p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tier1" in out and "fused_int8_rope_sage_attn" in out
    assert "tier2" in out and "tier3" in out
    # read-only: input untouched (still the 2 pre-existing plugin nodes).
    assert S.summarize(S.load(str(src)))["total"] == 2
