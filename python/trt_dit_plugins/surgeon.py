# SPDX-License-Identifier: Apache-2.0
"""ONNX graph surgery: rewrite subgraphs to dit-plugins nodes.

The TRT ONNX parser imports unrecognized nodes as plugins when the node
op matches a registered plugin name; ``plugin_version`` / ``plugin_namespace``
attrs override the defaults (``"1"`` / ``""``). Every constructor below emits
exactly that, so ``trtexec --onnx out.onnx --plugins libtrt_dit_plugins.so``
just works (or ``tdp.ensure_loaded()`` + the Python builder API).

No auto pattern-matching on purpose: exporter decompositions differ too much
to guess safely. Fusion takes explicit tensor anchors instead::

    fuse_norm_affine(graph, norm="ln_out", scale="blk_scale",
                     shift="blk_shift", out="blk_out", plugin="rms_adaln")
    fuse_dit_self_attn_block(graph, q_i8="q", q_scale="qs", k_i8="k",
                             k_scale="ks", v_i8="v", v_scale="vs",
                             rms_w_q="qn", rms_w_k="kn", inv_freq="inv",
                             attn_out="attn_out")

The fused DiT-self attention op (``fused_int8_rope_sage_attn``, 9 inputs,
1 output) collapses a post-rope-surgery self-attention block
(``rms_rope_split_half`` + ``SageInt8Attn``/``Attention``) into one node.
Same CLI surface: ``--fuse-dit-self-attn
Q_I8,Q_S,K_I8,K_S,V_I8,V_S,QN,KN,INV,ATTN_OUT[,NAME]``.

Needs ``pip install trt-dit-plugins[onnx]``. Importing this module is cheap;
``onnx`` / ``onnx-graphsurgeon`` are imported lazily and only fail when used.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Any

PLUGIN_DOMAIN = "dit-plugins"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "dit-plugins"
# Newer onnx releases default to IR versions the TRT parser cannot read.
# Cap saved models here so output stays TRT-readable regardless of env.
TRT_MAX_IR_VERSION = 10
_uid = itertools.count()

# op -> (n_inputs, n_outputs, field names). Ranks/shapes are checked in
# validate_plugin_nodes where statically known; dtypes resolve at TRT build.
_SPECS: dict[str, tuple[int, int, frozenset[str]]] = {
    "int8_attention": (3, 1, frozenset()),
    "sage_attn": (3, 1, frozenset({"fp8_pv"})),
    "adaln": (3, 1, frozenset({"eps"})),
    "rms_adaln": (3, 1, frozenset({"eps"})),
    "apply_rope": (3, 2, frozenset()),
    "rms_rope_split_half": (5, 2, frozenset({"epsilon", "rot_dim"})),
    "stochastic_round_fp8": (2, 1, frozenset({"alias_rng"})),
    "block_sparse_sage2_attn": (4, 1, frozenset({"scale", "pvthreshd", "attention_sink"})),
    "fused_int8_rope_sage_attn": (9, 1, frozenset()),
}


def _gs():
    try:
        import onnx_graphsurgeon as gs
    except ImportError as e:
        raise ImportError("surgeon needs the onnx extra: pip install trt-dit-plugins[onnx]") from e
    return gs


def _like(ref, name: str | None = None, dtype=None):
    gs = _gs()
    shape = list(ref.shape) if ref.shape is not None else None
    # ONNX forbids empty tensor names; always emit a unique one by default.
    return gs.Variable(name or f"dit_out{next(_uid)}",  # type: ignore[arg-type]
                       dtype=dtype if dtype is not None else ref.dtype, shape=shape)


def _plugin_node(graph, op: str, inputs, outputs, fields: dict[str, Any] | None = None, name: str | None = None):
    gs = _gs()
    attrs: dict[str, Any] = {"plugin_version": PLUGIN_VERSION, "plugin_namespace": PLUGIN_NAMESPACE}
    attrs.update(fields or {})
    node = gs.Node(op=op, name=name or "", attrs=attrs,
                   inputs=list(inputs), outputs=list(outputs), domain=PLUGIN_DOMAIN)
    graph.nodes.append(node)
    return node


# --- 7 plugin constructors (explicit, exporter-independent) ---

def add_int8_attention(graph, q, k, v, name=None):
    """q,k,v: [B,H,S,D] variables. Returns o (same shape/dtype as q)."""
    return _plugin_node(graph, "int8_attention", [q, k, v], [_like(q)], name=name).outputs[0]


def add_sage_attn(graph, q, k, v, fp8_pv=False, name=None):
    """Plain dense SageAttention. q,k,v: [B,H,S,D] f32/f16/bf16 variables
    (GQA: Hq % Hkv == 0 allowed). fp8_pv opts into the FP8-PV dense Sage2
    path (sm89-only, validated at build). Returns o (same shape as q)."""
    fields = {"fp8_pv": int(bool(fp8_pv))} if fp8_pv else None
    return _plugin_node(graph, "sage_attn", [q, k, v], [_like(q)], fields, name=name).outputs[0]


def _add_adaln(op, graph, x, scale, shift, eps=1e-6, name=None):
    return _plugin_node(graph, op, [x, scale, shift], [_like(x)],
                        {"eps": float(eps)}, name=name).outputs[0]


def add_adaln(graph, x, scale, shift, eps=1e-6, name=None):
    """Fused LayerNorm AdaLN: layernorm(x) * (1 + scale) + shift."""
    return _add_adaln("adaln", graph, x, scale, shift, eps, name)


def add_rms_adaln(graph, x, scale, shift, eps=1e-6, name=None):
    """Fused RMSNorm AdaLN: rmsnorm(x) * (1 + scale) + shift."""
    return _add_adaln("rms_adaln", graph, x, scale, shift, eps, name)


def add_apply_rope(graph, q, k, f, name=None):
    """q,k: [B,H,S,D]; f: [fb,f1,f2,D/2,2,2]. Returns (qo, ko)."""
    outs = [_like(q), _like(k)]
    _plugin_node(graph, "apply_rope", [q, k, f], outs, name=name)
    return outs[0], outs[1]


def add_rms_rope_split_half(graph, q, k, f, qs, ks, epsilon=1e-6, rot_dim=0, name=None):
    """qs,ks: [D] scales. Returns (qo, ko). rot_dim=0 means full D."""
    outs = [_like(q), _like(k)]
    _plugin_node(graph, "rms_rope_split_half", [q, k, f, qs, ks], outs,
                 {"epsilon": float(epsilon), "rot_dim": int(rot_dim)}, name=name)
    return outs[0], outs[1]


def add_stochastic_round_fp8(graph, x, r, alias_rng=0, name=None):
    """x: any float; r: INT32 same numel. Output dtype resolves to FP8 at TRT build."""
    return _plugin_node(graph, "stochastic_round_fp8", [x, r], [_like(x)],
                        {"alias_rng": int(alias_rng)}, name=name).outputs[0]


def add_block_sparse_sage2_attn(graph, q, k, v, m, scale=0.0, pvthreshd=50.0,
                                attention_sink=0, name=None):
    """m: INT32 [B,H,S//128,S//64]. scale=0 selects 1/sqrt(D)."""
    return _plugin_node(graph, "block_sparse_sage2_attn", [q, k, v, m], [_like(q)],
                        {"scale": float(scale), "pvthreshd": float(pvthreshd),
                         "attention_sink": int(attention_sink)}, name=name).outputs[0]


def add_dit_self_fused_attn(graph, q_i8, q_scale, k_i8, k_scale, v_i8,
                              v_scale, rms_w_q, rms_w_k, inv_freq, name=None,
                              out_dtype=None):
    """DiT-self int8 RoPE+Sage fused (D=128, S>1024).

    q_i8/k_i8/v_i8: INT8 [B,H,S,128]; q_scale/k_scale/v_scale: FP32 scalar;
    rms_w_q/rms_w_k: [128]; inv_freq: FP32 [64]. Returns BF16 [B,H,S,128].

    ``out_dtype`` overrides the output tensor dtype (pass the graph's BF16
    dtype object); default keeps the q_i8 dtype, which mistypes the output
    for ORT/shape-inference (TRT itself takes BF16 from the plugin).
    """
    return _plugin_node(graph, "fused_int8_rope_sage_attn",
                        [q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q,
                         rms_w_k, inv_freq],
                        [_like(q_i8, dtype=out_dtype)], name=name).outputs[0]


def fuse_dit_self_attn_block(graph, *, q_i8, q_scale, k_i8, k_scale, v_i8,
                             v_scale, rms_w_q, rms_w_k, inv_freq, attn_out,
                             name=None):
    """Third surgery: collapse a post-rope-surgery self-attn block into one
    ``fused_int8_rope_sage_attn`` node (DiT-self, D=128, S>1024).

    Expected pre-shape (rope+sage surgery done; the nine INT8 anchors
    created by the caller, e.g. ``QuantizeLinear`` Q nodes with calibrated
    per-tensor scales — a plain int8 export has NO int8 tensors here, so
    anchors never "already exist")::

        rms_rope_split_half(qt, kt, freqs, qsc, ksc) -> (rq, rk)
        SageInt8Attn/Attention(rq, rk, v) -> attn_out

    All block tensors are explicit anchors (no pattern guessing, same
    philosophy as :func:`fuse_norm_affine`): the nine ``q_i8 ... inv_freq``
    names are the fused node's inputs, ``attn_out`` names the tensor whose
    consumers are rewired to the fused output. The old attention node must
    take both Q and K from the same ``rms_rope_split_half`` node, otherwise
    ``ValueError`` (protects cross-attn / non-rope blocks); it must have
    exactly 3 inputs (a mask would be silently dropped, so masked attention
    is rejected). The fused output takes dtype/shape from ``attn_out`` (the
    plugin emits BF16; copying the INT8 anchor dtype would mistype it).
    Now-dead replaced nodes are pruned. Returns the plugin node.
    """
    tensors = graph.tensors()
    for n in (q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q, rms_w_k,
              inv_freq, attn_out):
        if n not in tensors:
            raise KeyError(f"tensor not in graph: {n!r}")
    anchor_vars = [tensors[n] for n in
                   (q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q,
                    rms_w_k, inv_freq)]
    out_var = tensors[attn_out]
    prods = list(out_var.inputs)
    if len(prods) != 1:
        raise ValueError(f"{attn_out}: want 1 producer, got {len(prods)}")
    attn = prods[0]
    if attn.op not in ("SageInt8Attn", "Attention"):
        raise ValueError(f"{attn_out}: producer must be SageInt8Attn/Attention, "
                         f"got {attn.op!r}")
    if len(attn.inputs) != 3:
        raise ValueError(f"{attn_out}: attention node needs exactly 3 inputs "
                         f"(mask unsupported), got {len(attn.inputs)}")
    rq, rk = attn.inputs[0], attn.inputs[1]
    if not rq.inputs or not rk.inputs or rq.inputs[0] is not rk.inputs[0]:
        raise ValueError(f"{attn_out}: Q/K must share one rms_rope_split_half node")
    rope = rq.inputs[0]
    if rope.op != "rms_rope_split_half":
        raise ValueError(f"{attn_out}: Q/K producer must be rms_rope_split_half, "
                         f"got {rope.op!r}")
    fused_out = _like(out_var, attn_out + "_fused")
    node = _plugin_node(graph, "fused_int8_rope_sage_attn", anchor_vars,
                        [fused_out], name=name or f"fused_{attn_out}")
    for consumer in list(out_var.outputs):
        consumer.inputs = [fused_out if i is out_var else i
                           for i in consumer.inputs]
        if consumer not in fused_out.outputs:
            fused_out.outputs.append(consumer)
    out_var.outputs.clear()
    if attn_out in [o.name for o in graph.outputs]:
        graph.outputs = [fused_out if o is out_var else o
                         for o in graph.outputs]
    for i in attn.inputs:
        if attn in i.outputs:
            i.outputs.remove(attn)
    for dead in (attn, rope):
        dead.outputs = [o for o in dead.outputs if o.outputs]
        if not dead.outputs and dead in graph.nodes:
            graph.nodes.remove(dead)
    graph.cleanup().toposort()
    return node


# --- explicit fusion + retarget ---

def _last_dim(t):
    if t.shape is not None and len(t.shape) > 0:
        d = t.shape[-1]
        return d if isinstance(d, int) else None
    return None


def fuse_norm_affine(graph, *, norm: str, scale: str, shift: str, out: str,
                     plugin: str = "adaln", eps: float = 1e-6):
    """Replace ``norm*(1+scale)+shift`` ending at tensor ``out`` with one plugin node.

    ``norm`` may come from any norm lowering (LayerNorm node, decomposed RMS,
    ...); only the 4 anchor tensor names matter. Dead producers are pruned by
    cleanup unless shared with other live tensors. Returns the plugin node.
    """
    if plugin not in ("adaln", "rms_adaln"):
        raise ValueError(f"plugin must be adaln/rms_adaln, got {plugin!r}")
    tensors = graph.tensors()
    for n in (norm, scale, shift, out):
        if n not in tensors:
            raise KeyError(f"tensor not in graph: {n!r}")
    tn, ts, tsh, to = (tensors[n] for n in (norm, scale, shift, out))
    ds = {_last_dim(t) for t in (tn, ts, tsh) if _last_dim(t) is not None}
    if len(ds) > 1:
        raise ValueError(f"last-dim mismatch in fuse anchors: {norm, scale, shift}")
    node = _plugin_node(graph, plugin, [tn, ts, tsh], [to],
                        {"eps": float(eps)}, name=f"{plugin}_{out}")
    for prod in list(to.inputs):
        if prod is not node and to in prod.outputs:
            prod.outputs.remove(to)
        if prod is not node and prod in to.inputs:
            to.inputs.remove(prod)
    graph.cleanup().toposort()
    return node


def retarget_nodes(graph, mapping: dict[str, str]) -> int:
    """Rename custom ops to plugin ops in place: ``{"MyAttn": "int8_attention"}``.

    Sets domain + plugin_version/namespace; other attrs are forwarded as plugin
    fields by the TRT parser. Returns the number of nodes touched.
    """
    n = 0
    for node in graph.nodes:
        if node.op in mapping and node.domain != PLUGIN_DOMAIN:
            node.op = mapping[node.op]
            node.domain = PLUGIN_DOMAIN
            node.attrs.setdefault("plugin_version", PLUGIN_VERSION)
            node.attrs.setdefault("plugin_namespace", PLUGIN_NAMESPACE)
            n += 1
    if n:
        graph.toposort()
    return n


# --- validation / summary / IO ---

def _shape_of(t, i: int | None):
    if t.shape is None or i is None or i >= len(t.shape):
        return None
    d = t.shape[i]
    return d if isinstance(d, int) else None


def validate_plugin_nodes(graph) -> list[str]:
    """Static checks that don't need a GPU. Returns error strings (empty = ok)."""
    errs: list[str] = []
    for node in graph.nodes:
        if node.domain != PLUGIN_DOMAIN:
            continue
        spec = _SPECS.get(node.op)
        if spec is None:
            errs.append(f"{node.name or '?'}: unknown dit-plugins op {node.op!r}")
            continue
        want_in, want_out, fields = spec
        if len(node.inputs) != want_in:
            errs.append(f"{node.name or node.op}: want {want_in} inputs, got {len(node.inputs)}")
        if len(node.outputs) != want_out:
            errs.append(f"{node.name or node.op}: want {want_out} outputs, got {len(node.outputs)}")
        extra = set(node.attrs) - {"plugin_version", "plugin_namespace"} - fields
        if extra:
            errs.append(f"{node.name or node.op}: unknown fields {sorted(extra)}")
        if node.op == "int8_attention" and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            if d is not None and d not in (64, 128, 256):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128/256")
        if node.op == "sage_attn" and len(node.inputs) == 3:
            d = _shape_of(node.inputs[0], 3)
            if d is not None and d not in (64, 128, 256):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128/256")
            hq = _shape_of(node.inputs[0], 1)
            hkv = _shape_of(node.inputs[1], 1)
            if hq is not None and hkv is not None:
                if hkv == 0 or hq % hkv != 0:
                    errs.append(f"{node.name or node.op}: Hq={hq} must be a multiple of Hkv={hkv}")
        if node.op == "block_sparse_sage2_attn" and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            s = _shape_of(node.inputs[0], 2)
            if d is not None and d not in (64, 128):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128")
            if s is not None and s % 128 != 0:
                errs.append(f"{node.name or node.op}: S={s}, want multiple of 128")
        if node.op == "fused_int8_rope_sage_attn" and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            s = _shape_of(node.inputs[0], 2)
            if d is not None and d != 128:
                errs.append(f"{node.name or node.op}: D={d}, want 128")
            if s is not None and s <= 1024:
                errs.append(f"{node.name or node.op}: S={s}, want >1024")
            if len(node.inputs) == 9:
                # DiT-self contract: q/k/v identical shapes when static.
                shapes = []
                for t in (node.inputs[0], node.inputs[2], node.inputs[4]):
                    shapes.append(tuple(t.shape) if t.shape is not None else None)
                known = [sh for sh in shapes if sh is not None]
                if len(known) == 3 and not (known[0] == known[1] == known[2]):
                    errs.append(f"{node.name or node.op}: q/k/v shapes differ: {known}")
                # Per-tensor input scales: single element when static.
                for idx in (1, 3, 5):
                    t = node.inputs[idx]
                    if t.shape is not None and all(isinstance(x, int) for x in t.shape):
                        n = 1
                        for x in t.shape:
                            n *= x
                        if n != 1:
                            errs.append(f"{node.name or node.op}: input {idx} "
                                        f"numel={n}, want 1")
                # Norm weights [128] and inv_freq [64] when static.
                for idx, want in ((6, (128,)), (7, (128,)), (8, (64,))):
                    t = node.inputs[idx]
                    if (t.shape is not None and
                            all(isinstance(x, int) for x in t.shape) and
                            tuple(t.shape) != want):
                        errs.append(f"{node.name or node.op}: input {idx} shape "
                                    f"{tuple(t.shape)}, want {want}")
    return errs


def summarize(graph) -> dict[str, Any]:
    """Counts of dit-plugins nodes by op."""
    counts: dict[str, int] = {}
    for node in graph.nodes:
        if node.domain == PLUGIN_DOMAIN:
            counts[node.op] = counts.get(node.op, 0) + 1
    return {"plugin_nodes": counts, "total": sum(counts.values()), "all_nodes": len(graph.nodes)}


def load(path: str):
    import onnx
    return _gs().import_onnx(onnx.load(path))


def save(graph, path: str) -> None:
    import warnings

    import onnx
    model = _gs().export_onnx(graph)
    if model.ir_version > TRT_MAX_IR_VERSION:
        warnings.warn(
            f"lowering ir_version {model.ir_version} -> {TRT_MAX_IR_VERSION} "
            f"for TRT parser compatibility")
        model.ir_version = TRT_MAX_IR_VERSION
    onnx.save(model, path)


def _fuse_spec(s: str):
    parts = s.split(",")
    if len(parts) not in (4, 5):
        raise ValueError(f"--fuse spec wants norm,scale,shift,out[,eps], got {s!r}")
    eps = float(parts[4]) if len(parts) == 5 else 1e-6
    return parts[0], parts[1], parts[2], parts[3], eps


def _dit_self_attn_spec(s: str):
    parts = s.split(",")
    if len(parts) not in (10, 11):
        raise ValueError(f"--fuse-dit-self-attn wants "
                         f"Q_I8,Q_SCALE,K_I8,K_SCALE,V_I8,V_SCALE,RMS_W_Q,RMS_W_K,INV_FREQ,ATTN_OUT[,NAME], got {s!r}")
    name = parts[10] if len(parts) == 11 else None
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9], name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rewrite ONNX subgraphs to dit-plugins nodes.")
    ap.add_argument("input", help="input .onnx")
    ap.add_argument("output", nargs="?", help="output .onnx (required unless --dry-run)")
    ap.add_argument("--fuse-adaln", action="append", default=[], metavar="NORM,SCALE,SHIFT,OUT[,EPS]")
    ap.add_argument("--fuse-rms-adaln", action="append", default=[], metavar="NORM,SCALE,SHIFT,OUT[,EPS]")
    ap.add_argument("--fuse-dit-self-attn", action="append", default=[],
                    metavar="Q_I8,Q_SCALE,K_I8,K_SCALE,V_I8,V_SCALE,RMS_W_Q,RMS_W_K,INV_FREQ,ATTN_OUT[,NAME]")
    ap.add_argument("--retarget", action="append", default=[], metavar="SRC:DST",
                    help="rename custom op SRC to plugin op DST (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args(argv)
    if not args.output and not args.dry_run:
        ap.error("output is required unless --dry-run")

    graph = load(args.input)
    for r in args.retarget:
        src, _, dst = r.partition(":")
        if not src or not dst:
            ap.error(f"--retarget wants SRC:DST, got {r!r}")
        if dst not in _SPECS:
            ap.error(f"unknown plugin op: {dst!r} (want one of {sorted(_SPECS)})")
        print(f"retarget {src} -> {dst}: {retarget_nodes(graph, {src: dst})}")
    for spec in args.fuse_adaln:
        n, s, sh, o, eps = _fuse_spec(spec)
        fuse_norm_affine(graph, norm=n, scale=s, shift=sh, out=o, plugin="adaln", eps=eps)
        print(f"fused adaln -> {o}")
    for spec in args.fuse_rms_adaln:
        n, s, sh, o, eps = _fuse_spec(spec)
        fuse_norm_affine(graph, norm=n, scale=s, shift=sh, out=o, plugin="rms_adaln", eps=eps)
        print(f"fused rms_adaln -> {o}")
    for spec in args.fuse_dit_self_attn:
        qi, qs, ki, ks, vi, vs, wq, wk, inv, ao, name = _dit_self_attn_spec(spec)
        fuse_dit_self_attn_block(graph, q_i8=qi, q_scale=qs, k_i8=ki,
                                 k_scale=ks, v_i8=vi, v_scale=vs,
                                 rms_w_q=wq, rms_w_k=wk, inv_freq=inv,
                                 attn_out=ao, name=name)
        print(f"fused fused_int8_rope_sage_attn -> {ao}")

    print(summarize(graph))
    errs = validate_plugin_nodes(graph)
    for e in errs:
        print(f"ERROR: {e}")
    if args.dry_run:
        return 1 if errs else 0
    save(graph, args.output)
    print(f"wrote {args.output}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
