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
(``rms_rope_split_half`` + ``Attention``/``sage_attn``/``int8_attention``,
legacy ``SageInt8Attn`` accepted for old graphs) into one node.
Same CLI surface: ``--fuse-dit-self-attn
Q_I8,Q_SCALE,K_I8,K_SCALE,V_I8,V_SCALE,RMS_W_Q,RMS_W_K,INV_FREQ,ATTN_OUT[,NAME]``.

The ``discover_*``/``apply_*``/``fuse_rope_blocks`` helpers below close the
loop for exporters (auto-discovery, inv_freq derivation, QuantizeLinear
anchor creation); the explicit-anchor APIs above remain the stable core.

Needs ``pip install trt-dit-plugins[onnx]``. Importing this module is cheap;
``onnx`` / ``onnx-graphsurgeon`` are imported lazily and only fail when used.
"""
from __future__ import annotations

import argparse
import itertools
from collections import deque
from typing import Any

from .select import BasePrecision, GemmKind, PluginOp, convrot_shape_ok, require_enum

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
    PluginOp.INT8_ATTENTION: (3, 1, frozenset()),
    PluginOp.SAGE_ATTN: (3, 1, frozenset({"fp8_pv"})),
    PluginOp.RMS_ADALN: (3, 1, frozenset({"eps", "has_weight", "residual_gate"})),
    PluginOp.LAYER_NORM_ADALN: (3, 1, frozenset({"eps", "has_weight", "residual_gate"})),
    PluginOp.APPLY_ROPE: (3, 2, frozenset()),
    PluginOp.RMS_ROPE_SPLIT_HALF: (5, 2, frozenset({"epsilon", "rot_dim"})),
    PluginOp.RMS_NORM: (2, 1, frozenset({"eps", "in_layout"})),
    PluginOp.STOCHASTIC_ROUND_FP8: (2, 1, frozenset({"alias_rng"})),
    PluginOp.BLOCK_SPARSE_SAGE2_ATTN: (4, 1, frozenset({"scale", "pvthreshd", "attention_sink"})),
    PluginOp.FUSED_INT8_ROPE_SAGE_ATTN: (9, 1, frozenset()),
    PluginOp.CONVROT_LINEAR: (4, 1, frozenset({"group_size", "has_bias", "out_dtype", "gemm_config", "epilogue", "residual_gate", "out_layout", "qkv_heads", "head_dim"})),
    PluginOp.CONVROT_LINEAR_W4A4: (4, 1, frozenset({"group_size", "has_bias", "out_dtype", "gemm_config", "epilogue", "residual_gate", "out_layout", "qkv_heads", "head_dim"})),
}

_LEGACY_ATTN_OPS = (PluginOp.SAGE_INT8_LEGACY, "Attention")
# Tier2 now emits dit-plugins ``sage_attn``/``int8_attention`` (shipped in
# the wheel); fusion discovery accepts them too so fuse-after-Tier2 keeps
# working exactly as it did for legacy ``SageInt8Attn``.
_FUSABLE_ATTN_OPS = _LEGACY_ATTN_OPS + (PluginOp.SAGE_ATTN, PluginOp.INT8_ATTENTION)


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
    return _plugin_node(graph, PluginOp.INT8_ATTENTION, [q, k, v], [_like(q)], name=name).outputs[0]


def add_sage_attn(graph, q, k, v, fp8_pv=False, name=None):
    """Plain dense SageAttention. q,k,v: [B,H,S,D] f32/f16/bf16 variables
    (GQA: Hq % Hkv == 0 allowed). fp8_pv opts into the FP8-PV dense Sage2
    path (sm89-only, validated at build). Returns o (same shape as q)."""
    fields = {"fp8_pv": int(bool(fp8_pv))} if fp8_pv else None
    return _plugin_node(graph, PluginOp.SAGE_ATTN, [q, k, v], [_like(q)], fields, name=name).outputs[0]


def _add_adaln(op, graph, x, scale, shift, eps=1e-6, name=None):
    return _plugin_node(graph, op, [x, scale, shift], [_like(x)],
                        {"eps": float(eps)}, name=name).outputs[0]


def add_rms_adaln(graph, x, scale, shift, eps=1e-6, name=None):
    """Fused RMSNorm AdaLN: rmsnorm(x) * (1 + scale) + shift."""
    return _add_adaln(PluginOp.RMS_ADALN, graph, x, scale, shift, eps, name)


def add_layer_norm_adaln(graph, x, scale, shift, eps=1e-6, name=None):
    """Fused LayerNorm AdaLN: layernorm(x) * (1 + scale) + shift."""
    return _add_adaln(PluginOp.LAYER_NORM_ADALN, graph, x, scale, shift, eps, name)


def add_apply_rope(graph, q, k, f, name=None):
    """q,k: [B,H,S,D]; f: [fb,f1,f2,D/2,2,2]. Returns (qo, ko)."""
    outs = [_like(q), _like(k)]
    _plugin_node(graph, PluginOp.APPLY_ROPE, [q, k, f], outs, name=name)
    return outs[0], outs[1]


def add_rms_norm(graph, x, weight, *, eps=1e-6, layout="bhsd", name=None):
    """Single-tensor RMSNorm; layout='bshd' also transposes to BHSD."""
    gs = _gs()
    shape = None
    if x.shape is not None:
        b, s, h, d = x.shape
        shape = [b, h, s, d] if layout == "bshd" else list(x.shape)
    out = gs.Variable(name=f"dit_out{next(_uid)}", dtype=x.dtype, shape=shape)
    _plugin_node(graph, PluginOp.RMS_NORM, [x, weight], [out],
                 {"eps": float(eps), "in_layout": int(layout == "bshd")},
                 name=name)
    return out


def add_rms_rope_split_half(graph, q, k, f, qs, ks, epsilon=1e-6, rot_dim=0, name=None):
    """qs,ks: [D] scales. Returns (qo, ko). rot_dim=0 means full D."""
    outs = [_like(q), _like(k)]
    _plugin_node(graph, PluginOp.RMS_ROPE_SPLIT_HALF, [q, k, f, qs, ks], outs,
                 {"epsilon": float(epsilon), "rot_dim": int(rot_dim)}, name=name)
    return outs[0], outs[1]


def add_stochastic_round_fp8(graph, x, r, alias_rng=0, name=None):
    """x: any float; r: INT32 same numel. Output dtype resolves to FP8 at TRT build."""
    return _plugin_node(graph, PluginOp.STOCHASTIC_ROUND_FP8, [x, r], [_like(x)],
                        {"alias_rng": int(alias_rng)}, name=name).outputs[0]


def add_block_sparse_sage2_attn(graph, q, k, v, m, scale=0.0, pvthreshd=50.0,
                                attention_sink=0, name=None):
    """m: INT32 [B,H,S//128,S//64]. scale=0 selects 1/sqrt(D)."""
    return _plugin_node(graph, PluginOp.BLOCK_SPARSE_SAGE2_ATTN, [q, k, v, m], [_like(q)],
                        {"scale": float(scale), "pvthreshd": float(pvthreshd),
                         "attention_sink": int(attention_sink)}, name=name).outputs[0]


def add_convrot_linear(graph, x, weight, weight_scale, bias, *, w4a4=False,
                       has_bias=True, out_dtype=-1, gemm_config=-1,
                       epilogue="none", residual=None, gate=None,
                       out_layout="bsc", qkv_heads=0, head_dim=0, name=None):
    """ConvRot linear (W8A8, or W4A4 when ``w4a4=True``).

    x: float [..., K] (pre-rotation activation); weight: INT8 [N, K] for W8A8,
    INT8 [N, K/2] packed signed INT4 for W4A4; weight_scale: FP32 [N];
    bias: [N] in the output dtype (zeros with ``has_bias=False``). The weight
    must already carry the ConvRot rotation. Returns y [..., N].

    ``out_dtype``: -1 = input dtype, 0 = BF16, 1 = FP16, 2 = FP32 (plan 4.2).
    The declared output dtype is the plugin's real output; ``bias`` must
    already be in that dtype.
    ``gemm_config``: -1 = automatic route; >= 0 pins a W4A4 GEMM tile config.
    ``epilogue``: "none" | "gelu" | "silu"; ``residual``/``gate`` append the
    fused residual+gate inputs (W8A8 only).
    """
    if isinstance(epilogue, str):
        ep = {"none": 0, "gelu": 1, "silu": 2}.get(epilogue)
    else:
        ep = int(epilogue)
    if ep not in (0, 1, 2):
        raise ValueError("epilogue must be 'none', 'gelu' or 'silu'")
    if (ep or residual is not None) and w4a4:
        raise ValueError("W4A4 does not support epilogue/residual/gate")
    if residual is not None and gate is None:
        raise ValueError("residual requires gate")
    if out_layout not in ("bsc", "bhsd"):
        raise ValueError("out_layout must be 'bsc' or 'bhsd'")
    if w4a4 and (out_layout != "bsc" or qkv_heads or head_dim):
        raise ValueError("W4A4 does not support out_layout/qkv_heads/head_dim")
    if out_layout == "bhsd" and (ep or residual is not None):
        raise ValueError("bhsd output does not support epilogue/residual")
    gs = _gs()
    op = PluginOp.CONVROT_LINEAR_W4A4 if w4a4 else PluginOp.CONVROT_LINEAR
    n = weight.shape[0] if weight.shape is not None else None
    out_shape = None
    if x.shape is not None:
        out_shape = [*x.shape[:-1], n if n is not None else -1]
    code = int(out_dtype)
    if code not in (-1, 0, 1, 2):
        raise ValueError(f"out_dtype must be -1/0/1/2, got {out_dtype!r}")
    dtype = {0: "bfloat16", 1: "float16", 2: "float32"}.get(code, x.dtype)
    outs = []
    if out_layout == "bhsd":
        if x.shape is None or len(x.shape) != 3:
            raise ValueError("bhsd output requires x [B,S,K]")
        D = head_dim or 128
        H = qkv_heads or (n // D if n is not None else None)
        if n is None or H is None or D <= 0 or H <= 0 or H * D * (3 if qkv_heads else 1) != n:
            raise ValueError("weight N must be H*D or 3*H*D")
        for nm in (["q", "k", "v"] if qkv_heads else ["y"]):
            outs.append(gs.Variable(name=f"dit_out{next(_uid)}", dtype=dtype,
                                    shape=[x.shape[0], H, x.shape[1], D]))
    else:
        outs.append(gs.Variable(name=f"dit_out{next(_uid)}", dtype=dtype, shape=out_shape))
    ins = [x, weight, weight_scale, bias]
    if residual is not None:
        ins.extend([residual, gate])
    _plugin_node(graph, op, ins, outs,
                 {"group_size": 256, "has_bias": int(bool(has_bias)),
                  "out_dtype": int(out_dtype), "gemm_config": int(gemm_config),
                  "epilogue": ep, "residual_gate": int(residual is not None),
                  "out_layout": 1 if out_layout == "bhsd" else 0,
                  "qkv_heads": int(qkv_heads), "head_dim": int(head_dim)},
                 name=name)
    return tuple(outs) if qkv_heads else outs[0]


def import_comfy_convrot(graph, *, gemm_config=-1):
    """Rewrite ``comfy_kitchen::int8_linear`` (convrot) nodes to dit-plugins.

    The Reforge exporter emits the custom op with inputs
    ``(x, weight, weight_scale, bias, ...)`` and attributes
    ``output_dtype_code`` / ``convrot`` / ``convrot_groupsize``. Only the plain
    ConvRot form is representable by this plugin (no fused input activation,
    residual or bias-less layer), and shapes must satisfy the plugin contract;
    anything else is left untouched. Returns ``(converted, skipped)`` node names
    so the caller can keep the BF16 path for the skipped ones (fail-open).
    """
    converted: list[str] = []
    skipped: list[str] = []
    for node in graph.nodes:
        if node.domain != "comfy_kitchen" or node.op != "int8_linear":
            continue
        name = node.name or node.op
        attr = node.attrs
        if not int(attr.get("convrot", 0)) or int(attr.get("convrot_groupsize", 256)) != 256:
            skipped.append(name)
            continue
        if attr.get("input_act") not in (None, "", "none") or attr.get("residual"):
            skipped.append(name)
            continue
        if int(attr.get("transposed", 0)):
            skipped.append(name)  # _qdata is [K,N]; v1 only takes [N,K]
            continue
        if (len(node.inputs) != 4 or len(node.outputs) != 1
                or node.inputs[3] is None or not node.inputs[3].name):
            # The plugin takes exactly (x, w, ws, b) and one output; comfy's
            # optional residual_scale / extra outputs are not representable.
            skipped.append(name)
            continue
        out_code = attr.get("output_dtype_code")
        if out_code not in (0, 1, 2):
            skipped.append(name)
            continue
        # comfy codes (0=FP32, 1=FP16, 2=BF16) -> plugin field codes (0=BF16,
        # 1=FP16, 2=FP32, plan 4.2).
        out_code = {0: 2, 1: 1, 2: 0}[int(out_code)]
        weight = node.inputs[1]
        n = _shape_of(weight, 0)
        k = _shape_of(weight, 1)
        if convrot_shape_ok(k, n, w4a4=False) is not None:
            skipped.append(name)
            continue
        node.op = PluginOp.CONVROT_LINEAR
        node.domain = PLUGIN_DOMAIN
        node.attrs = {"plugin_version": PLUGIN_VERSION, "plugin_namespace": PLUGIN_NAMESPACE,
                      "group_size": 256, "has_bias": 1, "out_dtype": int(out_code),
                      "gemm_config": int(gemm_config)}
        converted.append(name)
    return converted, skipped


def add_dit_self_fused_attn(graph, q_i8, q_scale, k_i8, k_scale, v_i8,
                            v_scale, rms_w_q, rms_w_k, inv_freq, name=None,
                            out_dtype=None):
    """DiT-self int8 RoPE+Sage fused (D=128).

    q_i8/k_i8/v_i8: INT8 [B,H,S,128]; q_scale/k_scale/v_scale: FP32 scalar;
    rms_w_q/rms_w_k: [128]; inv_freq: FP32 [64]. Returns BF16 [B,H,S,128].

    ``out_dtype`` overrides the output tensor dtype (pass the graph's BF16
    dtype object); default keeps the q_i8 dtype, which mistypes the output
    for ORT/shape-inference (TRT itself takes BF16 from the plugin).
    """
    return _plugin_node(graph, PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
                        [q_i8, q_scale, k_i8, k_scale, v_i8, v_scale, rms_w_q,
                         rms_w_k, inv_freq],
                        [_like(q_i8, dtype=out_dtype)], name=name).outputs[0]


def fuse_dit_self_attn_block(graph, *, q_i8, q_scale, k_i8, k_scale, v_i8,
                             v_scale, rms_w_q, rms_w_k, inv_freq, attn_out,
                             name=None):
    """Third surgery: collapse a post-rope-surgery self-attn block into one
    ``fused_int8_rope_sage_attn`` node (DiT-self, D=128).

    Expected pre-shape (rope+sage surgery done; the nine INT8 anchors
    created by the caller, e.g. ``QuantizeLinear`` Q nodes with calibrated
    per-tensor scales — a plain int8 export has NO int8 tensors here, so
    anchors never "already exist")::

        rms_rope_split_half(qt, kt, freqs, qsc, ksc) -> (rq, rk)
        Attention/sage_attn/int8_attention(rq, rk, v) -> attn_out
        (legacy SageInt8Attn accepted too)

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
    if attn.op not in _FUSABLE_ATTN_OPS:
        raise ValueError(f"{attn_out}: producer must be SageInt8Attn/Attention/sage_attn/int8_attention, "
                          f"got {attn.op!r}")
    if len(attn.inputs) != 3:
        raise ValueError(f"{attn_out}: attention node needs exactly 3 inputs "
                         f"(mask unsupported), got {len(attn.inputs)}")
    rq, rk = attn.inputs[0], attn.inputs[1]
    if not rq.inputs or not rk.inputs or rq.inputs[0] is not rk.inputs[0]:
        raise ValueError(f"{attn_out}: Q/K must share one rms_rope_split_half node")
    rope = rq.inputs[0]
    if rope.op != PluginOp.RMS_ROPE_SPLIT_HALF:
        raise ValueError(f"{attn_out}: Q/K producer must be rms_rope_split_half, "
                         f"got {rope.op!r}")
    # out_var itself becomes the plugin output: the tensor name (and any
    # graph.outputs membership / downstream consumers) is preserved, so
    # I/O bindings by name keep working. Detach the old producer; the
    # caller prunes afterwards (per-fuse cleanup would delete other
    # blocks' not-yet-consumed anchors, breaking multi-block fusion).
    node = _plugin_node(graph, PluginOp.FUSED_INT8_ROPE_SAGE_ATTN, anchor_vars,
                        [out_var], name=name or f"fused_{attn_out}")
    if node not in out_var.inputs:
        out_var.inputs.append(node)
    if attn in out_var.inputs:
        out_var.inputs.remove(attn)
    if out_var in attn.outputs:
        attn.outputs.remove(out_var)
    return node


# --- auto discovery + anchor creation (DiT-self fused) ---
#
# The explicit APIs above take caller-provided anchors. The helpers below
# close the loop for exporters: find rope+attention pairs, derive inv_freq
# from baked freqs, create QuantizeLinear anchors from calibrated scales,
# then delegate the fusion itself to fuse_dit_self_attn_block.

_ROPE_OP = PluginOp.RMS_ROPE_SPLIT_HALF

# Same-input result cache for graph queries. Keyed by structural
# fingerprint (not object identity): equal graphs share results, and any
# mutation that a query can observe (ops, domains, names, connectivity,
# shapes) changes the key, so stale hits are impossible by construction.
# Bounded FIFO; results are plain data, copied per call so callers can
# mutate them freely.
_QUERY_CACHE: dict[tuple, Any] = {}
_QUERY_CACHE_MAX = 32


def _graph_fingerprint(graph) -> tuple:
    """Hashable snapshot of everything the cached queries read."""
    nodes = tuple(
        (n.op, getattr(n, "domain", ""), n.name,
         tuple(i.name for i in n.inputs), tuple(o.name for o in n.outputs))
        for n in graph.nodes
    )
    shapes = tuple(
        (name, None if t.shape is None else tuple(
            d if isinstance(d, (int, str)) else repr(d) for d in t.shape))
        for name, t in sorted(graph.tensors().items())
    )
    return nodes, shapes


def _cached_query(graph, kind: str, compute):
    key = (kind, _graph_fingerprint(graph))
    hit = _QUERY_CACHE.get(key)
    if hit is not None:
        return [dict(b) for b in hit]
    res = compute()
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[key] = res
    return [dict(b) for b in res]


def _dims(t):
    """Static shape as tuple (non-int dims -> None); None if unknown."""
    if t is None or t.shape is None:
        return None
    return tuple(d if isinstance(d, int) else None for d in t.shape)


def _single_prod(t):
    """Sole producer node of a tensor, else None (graph input / fan-in)."""
    ps = t.inputs
    return ps[0] if len(ps) == 1 else None


def _walk_up(edge, *, stop=None, skip=()):
    """Yield producer nodes BFS from a tensor; stop at tensor name, skip node ops."""
    seen: set[str] = set()
    dq: deque = deque([edge])
    while dq:
        t = dq.popleft()
        if t.name in seen or t.name == stop:
            continue
        seen.add(t.name)
        for p in t.inputs:
            yield p
            if p.op not in skip:
                dq.extend(p.inputs)


def _checked_baked_freqs(fq):
    """Validate ``[1,1,S,r,2,2]`` baked freqs layout; return as array."""
    import numpy as np
    fq = np.asarray(fq)
    if fq.ndim != 6 or fq.shape[0] != 1 or fq.shape[1] != 1 or fq.shape[4:] != (2, 2):
        raise ValueError(f"baked freqs must be [1,1,S,r,2,2], got {tuple(fq.shape)}")
    return fq


def _dtype_name(d) -> str | None:
    """'float32' for both the string and numpy-dtype spellings."""
    if d is None:
        return None
    return d if isinstance(d, str) else getattr(d, "__name__", str(d))


def _check_freqs_tensor(t) -> None:
    """Static half of ``_checked_baked_freqs`` for a graph tensor."""
    if t.shape is not None:
        if len(t.shape) != 6:
            raise ValueError(f"freqs tensor {t.name!r} must be [1,1,S,r,2,2], "
                             f"got {tuple(t.shape)}")
        head = list(t.shape[:2]) + list(t.shape[4:])
        if any(isinstance(d, int) and d != want
               for d, want in zip(head, (1, 1, 2, 2))):
            raise ValueError(f"freqs tensor {t.name!r} must be [1,1,S,r,2,2], "
                             f"got {tuple(t.shape)}")
    dn = _dtype_name(t.dtype)
    if dn is not None and dn not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"freqs tensor {t.name!r} dtype {dn}, "
                         f"want float32/float16/bfloat16")


def discover_dit_self_blocks(graph) -> list[dict[str, Any]]:
    """Find rope+attention pairs; classify fused eligibility per block.

    A block is eligible when Q/K come from the SAME ``rms_rope_split_half``
    node (this rejects cross-attn) and no statically-known shape violates
    the plugin contract (D=128, rank-4 Q/K/V, ``[1,1,S,64,2,2]`` freqs,
    ``[128]`` norms). Unknown (dynamic/profile) dims never reject:
    TRT resolves them at build. Returns one dict per candidate with
    ``eligible`` + ``reason``; source names (``q``/``k``/``freqs``/
    ``rms_w_q``/``rms_w_k``/``v``/``attn_out``) are always populated for
    well-formed pairs. Key names follow :func:`fuse_dit_self_attn_block`
    wherever they coincide (``rms_w_q``/``rms_w_k``); ``q``/``k``/``v``
    are the pre-quant float edges that quantize into ``q_i8``/``k_i8``/
    ``v_i8``.

    Same input -> cached result (structural fingerprint; e.g. the second
    discovery inside :func:`apply_dit_self_fused` is a cache hit).
    """
    return _cached_query(graph, "discover",
                         lambda: _discover_dit_self_blocks_uncached(graph))


def _discover_dit_self_blocks_uncached(graph) -> list[dict[str, Any]]:
    tensors = graph.tensors()
    blocks: list[dict[str, Any]] = []
    for attn in graph.nodes:
        if attn.op not in _FUSABLE_ATTN_OPS:
            continue
        if len(attn.inputs) < 3 or not attn.outputs:
            blocks.append({"attn": attn.name, "attn_op": attn.op,
                           "attn_out": attn.outputs[0].name if attn.outputs else None,
                           "eligible": False, "reason": "malformed-attn"})
            continue
        base: dict[str, Any] = {"attn": attn.name, "attn_op": attn.op,
                                "attn_out": attn.outputs[0].name}
        nq = _single_prod(attn.inputs[0])
        nk = _single_prod(attn.inputs[1])
        if (nq is None or nk is None or nq is not nk
                or nq.op != _ROPE_OP or getattr(nq, "domain", "") != PLUGIN_DOMAIN
                or len(nq.inputs) != 5 or len(nq.outputs) != 2):
            blocks.append({**base, "eligible": False, "reason": "cross-or-unpaired-qk"})
            continue
        base.update({"rope": nq.name, "q": nq.inputs[0].name,
                     "k": nq.inputs[1].name, "freqs": nq.inputs[2].name,
                     "rms_w_q": nq.inputs[3].name, "rms_w_k": nq.inputs[4].name,
                     "v": attn.inputs[2].name})
        base.update(_check_dit_self_block(base, tensors))
        blocks.append(base)
    return blocks


def _check_dit_self_block(b: dict[str, Any], tensors: dict[str, Any]) -> dict[str, Any]:
    """Static contract check; ``{"eligible": bool, "reason": str}``."""
    for key in ("q", "k", "v"):
        dims = _dims(tensors.get(b[key]))
        if dims is not None:
            if len(dims) != 4:
                return {"eligible": False, "reason": f"{key}-rank-{len(dims)}"}
            if dims[3] is not None and dims[3] != 128:
                return {"eligible": False, "reason": f"D-{dims[3]}"}
    dims = _dims(tensors.get(b["freqs"]))
    if dims is not None and (len(dims) != 6 or dims[3] != 64):
        return {"eligible": False, "reason": f"bad-freqs-{dims}"}
    for key in ("rms_w_q", "rms_w_k"):
        dims = _dims(tensors.get(b[key]))
        if dims is not None and tuple(dims) != (128,):
            return {"eligible": False, "reason": f"bad-norm-{key}"}
    return {"eligible": True, "reason": "ok"}


def inv_freq_from_baked_freqs(fq) -> Any:
    """Derive the fused kernel's ``inv_freq[64]`` from baked rope freqs.

    ``fq`` is the ``[1,1,S,r,2,2]`` ``[[cos,-sin],[sin,cos]]`` baked capture
    (full D=128 split-half => ``r == 64``). Angles at position 1 equal
    ``inv_freq`` itself, so ``inv_freq = atan2(sin, cos)`` on the ``pos=1``
    row recovers it exactly. Returns FP32 ``[64]``.
    """
    import numpy as np
    fq = _checked_baked_freqs(np.asarray(fq, dtype=np.float64))
    if fq.shape[3] != 64:
        raise ValueError(f"fused path needs full-D split-half r=64, got {fq.shape[3]}")
    if fq.shape[2] < 2:
        raise ValueError(f"baked freqs need S>=2 for the pos=1 row, got {fq.shape[2]}")
    return np.arctan2(fq[0, 0, 1, :, 1, 0], fq[0, 0, 1, :, 0, 0]).astype(np.float32)


def _block_scales_ok(b: dict[str, Any], scales: dict[str, float]) -> bool:
    need = (b.get("q"), b.get("k"), b.get("v"))
    return all(k in scales and scales[k] > 0 for k in need)


def _scalar_scale(t: Any) -> float | None:
    """Static positive per-tensor scale from a GS constant, else None."""
    import numpy as np

    gs = _gs()
    if not isinstance(t, gs.Constant) or t.values is None:
        return None
    v = np.asarray(t.values).reshape(-1)
    if v.size != 1 or not np.isfinite(v[0]) or v[0] <= 0:
        return None
    return float(v[0])


def _qdq_parts(graph, edge: str):
    """Return (Quantize/Dequantize node, int8 tensor, scale tensor) or None."""
    import numpy as np

    gs = _gs()
    t = graph.tensors().get(edge)
    prod = t.inputs[0] if t is not None and len(t.inputs) == 1 else None
    if prod is None:
        return None
    if prod.op == "QuantizeLinear":
        q, i8 = prod, prod.outputs[0]
    elif prod.op == "DequantizeLinear":
        src = prod.inputs[0] if prod.inputs else None
        if src is not None and src.inputs and src.inputs[0].op == "QuantizeLinear":
            q, i8 = src.inputs[0], src
        else:
            q, i8 = prod, src
    else:
        return None
    if q is None or i8 is None or len(q.inputs) < 2:
        return None
    dt = i8.dtype
    if dt not in (np.int8, "int8", "INT8"):
        if q.attrs.get("output_dtype") != 3:
            return None
    if len(q.inputs) > 2:
        zp = q.inputs[2]
        if not isinstance(zp, gs.Constant) or zp.values is None:
            return None
        if np.any(np.asarray(zp.values) != 0):
            return None
    return q, i8, q.inputs[1]


def qdq_anchor_for(graph, edge: str) -> tuple[str, str] | None:
    """Return existing (INT8 tensor, scalar scale) at ``edge``, else None."""
    parts = _qdq_parts(graph, edge)
    if parts is None:
        return None
    _, i8, scale_t = parts
    if _scalar_scale(scale_t) is None:
        return None
    return i8.name, scale_t.name


def qdq_scale_for(graph, edge: str) -> float | None:
    """Return the per-tensor symmetric INT8 scale already present at ``edge``.

    Accepts the Q output (INT8), the DQ output (float), or a quantized graph
    input followed by DQ. Returns None for dynamic / per-channel / uint8 /
    non-zero-zero-point scales, so callers fall back to explicit calibration.
    """
    parts = _qdq_parts(graph, edge)
    return None if parts is None else _scalar_scale(parts[2])


def collect_dit_self_scales(graph, scales: dict[str, float] | None = None
                            ) -> dict[str, float]:
    """Merge explicit scales with Q/DQ scales found at eligible block edges."""
    out = dict(scales or {})
    for b in discover_dit_self_blocks(graph):
        if not b.get("eligible"):
            continue
        for edge in (b["q"], b["k"], b["v"]):
            if edge not in out:
                s = qdq_scale_for(graph, edge)
                if s is not None:
                    out[edge] = s
    return out


def quantize_dit_self_anchors(graph, scales: dict[str, float] | None = None, *,
                              inv_freq=None) -> dict[str, Any]:
    """Insert ``QuantizeLinear`` anchors for fused blocks from calibrated scales.

    ``scales`` maps ONNX tensor name (``q``/``k``/``v`` as reported by
    :func:`discover_dit_self_blocks`) to the per-tensor FP32 INT8 scale.
    Missing entries are filled from an existing Q/DQ at that edge
    (:func:`collect_dit_self_scales`); per-channel / dynamic scales are not
    guessed.
    Per anchored block this appends three ``QuantizeLinear`` nodes (INT8,
    symmetric, no zero-point) plus FP32 scalar scale constants, and picks
    one shared ``[64]`` ``inv_freq`` for all blocks (they share one capture;
    a differing block would silently reuse the first vector).

    ``inv_freq`` selects that vector: ``None`` derives it from the first
    anchored block's baked freqs constant (static graphs); a tensor name
    uses that graph tensor (dynamic freqs); a 64-element array creates a
    constant from it.

    The shared inv constant is born orphan (invisible to ``tensors()``), so
    a transient carrier ``Identity`` is appended to make it discoverable;
    the caller's post-fusion cleanup prunes the carrier while the fused
    node keeps the constant. Do not run a standalone cleanup between
    this function and the fusion.

    Anchor names follow :func:`fuse_dit_self_attn_block` kwargs verbatim.
    Not idempotent: a second run without an intervening fusion would reuse
    the ``{attn_out}_fused_*`` names and create duplicates. In the normal
    flow (:func:`apply_dit_self_fused`, or fuse right after) this cannot
    happen — fused blocks are no longer discovered as candidates.
    Returns ``{"applied": n, "skipped": m, "anchors": {attn_out: kwargs},
    "inv_freq": name | None}``.
    """
    import numpy as np
    from onnx import TensorProto
    gs = _gs()
    scales = collect_dit_self_scales(graph, scales)
    tensors = graph.tensors()
    inv_name: str | None = None
    inv_values = None
    if inv_freq is not None:
        if isinstance(inv_freq, str):
            if inv_freq not in tensors:
                raise KeyError(f"inv_freq tensor {inv_freq!r} not in graph")
            inv_name = inv_freq
        else:
            inv_values = np.ascontiguousarray(np.asarray(inv_freq, dtype=np.float32))
            if inv_values.shape != (64,):
                raise ValueError(f"inv_freq must be [64], got {inv_values.shape}")
    skipped = 0
    anchors: dict[str, dict[str, str]] = {}
    for b in discover_dit_self_blocks(graph):
        if not b["eligible"] or not _block_scales_ok(b, scales):
            skipped += 1
            continue
        a: dict[str, str] = {}
        for stem, edge in (("q", b["q"]), ("k", b["k"]), ("v", b["v"])):
            edge_var = tensors[edge]
            reuse = qdq_anchor_for(graph, edge)
            if reuse is not None:
                a[f"{stem}_i8"], a[f"{stem}_scale"] = reuse
                continue
            sname = f"{b['attn_out']}_fused_{stem}s"
            sconst = gs.Constant(sname, values=np.asarray(scales[edge], dtype=np.float32))
            qi = gs.Variable(f"{b['attn_out']}_fused_{stem}i", dtype=np.int8,
                             shape=list(edge_var.shape) if edge_var.shape is not None else None)
            graph.nodes.append(gs.Node("QuantizeLinear", inputs=[edge_var, sconst],
                                       outputs=[qi], attrs={"output_dtype": int(TensorProto.INT8)},
                                       name=qi.name))  # type: ignore[arg-type]
            tensors[qi.name] = qi
            a[f"{stem}_i8"], a[f"{stem}_scale"] = qi.name, sname
        if inv_name is None:
            if inv_values is None:
                fvar = tensors[b["freqs"]]
                if not isinstance(fvar, gs.Constant) or fvar.values is None:
                    raise ValueError(
                        f"baked freqs {b['freqs']!r} have no values; "
                        f"pass inv_freq= for dynamic freqs")
                inv_values = np.ascontiguousarray(
                    inv_freq_from_baked_freqs(fvar.values), dtype=np.float32)
            inv_name = "fused_shared_inv_freq"
            inv_const = gs.Constant(inv_name, values=inv_values)
            carrier = gs.Variable(inv_name + "_carrier", dtype=np.float32, shape=[64])
            graph.nodes.append(gs.Node("Identity", inputs=[inv_const],  # type: ignore[arg-type]
                                       outputs=[carrier], name=carrier.name))
        a.update({"rms_w_q": b["rms_w_q"], "rms_w_k": b["rms_w_k"],
                  "inv_freq": inv_name})
        anchors[b["attn_out"]] = a
    return {"applied": len(anchors), "skipped": skipped, "anchors": anchors,
            "inv_freq": inv_name}


def apply_dit_self_fused(graph, scales: dict[str, float] | None = None
                         ) -> dict[str, Any]:
    """Discover + quantize + fuse every eligible DiT-self block.

    ``scales`` maps ONNX tensor name to per-tensor FP32 INT8 scale (see
    :func:`quantize_dit_self_anchors`). Missing entries are filled from an
    existing Q/DQ at that edge (:func:`collect_dit_self_scales`); pass
    explicit calibration values when the graph has no Q/DQ there.
    Returns ``{"applied": n, "skipped": m, "by_reason": {reason: count}}``
    with ``missing-scales`` counted separately from structural rejects.
    Raises nothing on zero matches (check ``applied``); per-block fusion
    errors propagate from :func:`fuse_dit_self_attn_block`.
    """
    blocks = discover_dit_self_blocks(graph)
    scales = collect_dit_self_scales(graph, scales)
    by_reason: dict[str, int] = {}
    for b in blocks:
        reason = b["reason"] if b["eligible"] and _block_scales_ok(b, scales) \
            else ("missing-scales" if b["eligible"] else b["reason"])
        by_reason[reason] = by_reason.get(reason, 0) + 1
    q = quantize_dit_self_anchors(graph, scales)
    for attn_out, a in q["anchors"].items():
        fuse_dit_self_attn_block(graph, attn_out=attn_out, **a)
    applied = len(q["anchors"])
    if applied:
        graph.cleanup().toposort()
    return {"applied": applied, "skipped": len(blocks) - applied,
            "by_reason": by_reason}


# --- rope/sage traversal surgeries ---
#
# Unlike the constructors above, these find eligible subgraphs themselves:
# rope blocks via upstream RMSNormalization + Mul (rope math) anchors, and
# plain Attention nodes for the attention-plugin rewrite. Cross-attn
# (norm feeding attention directly, no rope math) is skipped, never fused.

def _gs_norm_anchor(edge):
    """First RMSNormalization producer upstream of a tensor (the q/k norm)."""
    return next((p for p in _walk_up(edge) if p.op == "RMSNormalization"), None)


def _gs_has_rope_math(edge, norm_out: str) -> bool:
    """True if a Mul sits between edge and norm_out (rope math present)."""
    return any(p.op == "Mul"
               for p in _walk_up(edge, stop=norm_out, skip=("RMSNormalization",)))


def _transpose_bhsd(graph, edge, out_name: str):
    """[B,S,H,d] -> [B,H,S,d] view for rope plugin inputs."""
    gs = _gs()
    # Shape only when fully static; a None entry would poison export on
    # dynamic graphs (unknown dims stay unknown, TRT resolves at build).
    shape = None
    dims = _dims(edge)
    if dims is not None and len(dims) == 4 and all(isinstance(x, int) for x in dims):
        b, s, h, d = dims
        shape = [b, h, s, d]
    out = gs.Variable(out_name, dtype=edge.dtype, shape=shape)
    graph.nodes.append(gs.Node("Transpose", inputs=[edge], outputs=[out],
                               attrs={"perm": [0, 2, 1, 3]}, name=out_name))
    return out


def fuse_rope_blocks(graph, fq=None, epsilon: float = 1e-6, rot_dim: int = 0,
                     *, freqs: str | None = None, qk_layout: str = "bshd",
                     include_cross: bool = False) -> int:
    """Replace norm+rope regions with ``rms_rope_split_half`` nodes.

    Pass exactly one of: ``fq``, the baked freqs array in ``[1,1,S,r,2,2]``
    layout (one shared constant for all blocks; grid-dependent, so callers
    must fuse only at a fixed resolution profile), or ``freqs``, the name of
    an existing graph tensor in the same layout (dynamic freqs — TRT
    resolves the grid at build time). The plugin fuses RMSNorm itself, so it
    takes the PRE-norm tensors plus the norm scales (feeding norm outputs
    would normalize twice). Q/K arrive ``[B,S,H,d]`` by default and are
    transposed to ``[B,H,S,d]`` views around the plugin; ``qk_layout="bhsd"``
    skips those transposes. ``include_cross=True`` with ``rot_dim < 0`` fuses
    the two independent RMSNorm nodes feeding a cross-attention block, since
    the pair op requires equal Q/K sequence lengths. Returns the fused block
    count; raises on zero matches or malformed attention/norm nodes.
    """
    import numpy as np
    gs = _gs()
    if qk_layout not in ("bshd", "bhsd"):
        raise ValueError("qk_layout must be 'bshd' or 'bhsd'")
    if (fq is None) == (freqs is None):
        raise ValueError("pass exactly one of fq (baked array) or freqs "
                         "(graph tensor name)")
    ftensor = None
    if freqs is not None:
        ftensor = graph.tensors().get(freqs)
        if ftensor is None:
            raise KeyError(f"freqs tensor {freqs!r} not in graph")
        _check_freqs_tensor(ftensor)
    else:
        fq = np.ascontiguousarray(_checked_baked_freqs(fq), dtype=np.float32)
    plans: list[tuple] = []
    for attn in graph.nodes:
        if attn.op not in _FUSABLE_ATTN_OPS:
            continue
        if len(attn.inputs) < 3 or not attn.outputs:
            raise ValueError(f"rope surgery: {attn.op} {attn.name} has "
                             f"malformed inputs/outputs")
        # Idempotency: skip blocks already fed by a rope/rms_norm node so a
        # second run neither double-fuses nor collides on generated names.
        if any(getattr(p, "op", "") in (_ROPE_OP, PluginOp.RMS_NORM)
               for t in attn.inputs[:2] for p in t.inputs):
            continue
        nq = _gs_norm_anchor(attn.inputs[0])
        nk = _gs_norm_anchor(attn.inputs[1])
        if nq is None or nk is None:
            continue
        if len(nq.inputs) < 2 or len(nk.inputs) < 2:
            raise ValueError("rope surgery: RMSNormalization node missing scale input")
        q_in, k_in = nq.inputs[0], nk.inputs[0]
        has_rope = (_gs_has_rope_math(attn.inputs[0], nq.outputs[0].name) and
                    _gs_has_rope_math(attn.inputs[1], nk.outputs[0].name))
        is_cross = not has_rope
        if include_cross and rot_dim < 0 and has_rope:
            # A scalar identity Mul can fake rope math; unequal Q/K lengths
            # mean cross-attn, so take the independent-norm path.
            si = 1 if qk_layout == "bshd" else 2
            dq, dk = _dims(q_in), _dims(k_in)
            sq = dq[si] if dq is not None and len(dq) == 4 else None
            sk = dk[si] if dk is not None and len(dk) == 4 else None
            is_cross = sq is not None and sk is not None and sq != sk
        if is_cross and not (include_cross and rot_dim < 0):
            continue
        plans.append(("cross" if is_cross else "self", attn, q_in, k_in,
                      nq.inputs[1], nk.inputs[1]))
    if not plans:
        raise RuntimeError("rope surgery matched 0 blocks")
    fconst = ftensor if ftensor is not None else gs.Constant(
        "dit_rope_freqs", values=fq)
    n_done = 0
    for kind, attn, q_in, k_in, qsc, ksc in plans:
        base = attn.name or attn.outputs[0].name
        if kind == "cross":
            layout = "bshd" if qk_layout == "bshd" else "bhsd"
            qo = add_rms_norm(graph, q_in, qsc, eps=epsilon, layout=layout,
                              name=base + "_rmsnorm_q")
            ko = add_rms_norm(graph, k_in, ksc, eps=epsilon, layout=layout,
                              name=base + "_rmsnorm_k")
        else:
            if qk_layout == "bhsd":
                qt, kt = q_in, k_in
            else:
                qt = _transpose_bhsd(graph, q_in, base + "_toBHSD_q")
                kt = _transpose_bhsd(graph, k_in, base + "_toBHSD_k")
            qo, ko = add_rms_rope_split_half(graph, qt, kt, fconst, qsc, ksc,
                                             epsilon=epsilon, rot_dim=rot_dim,
                                             name=base + "_rmsrope")
        attn.inputs[0] = qo
        attn.inputs[1] = ko
        n_done += 1
    graph.cleanup().toposort()
    return n_done


def apply_sage_attention(graph, only: set[str] | None = None) -> int:
    """Replace ``Attention`` nodes with legacy ``SageInt8Attn`` in place.

    .. deprecated::
        The legacy ``sage_attn_plugin.so`` creator ships in NO wheel from
        this repo, so graphs rewritten by this helper fail TRT parsing
        unless callers dlopen an external legacy ``.so`` themselves.
        Prefer :func:`apply_attention_plugins`, which emits the shipped
        ``dit-plugins`` ``sage_attn``/``int8_attention`` ops directly.
        Kept only for downstream compat with such external setups.

    This targets the legacy ``sage_attn_plugin.so`` op (empty namespace),
    NOT the ``dit-plugins`` ``sage_attn`` op: no domain is set and all
    ``Attention`` attributes are dropped. The drop is safe because the
    legacy creator registers zero fields, hardcodes ``sm_scale=1/sqrt(D)``
    and reads head counts from Q/K/V shapes; the only semantic variant
    (causal mask) is rejected below, so default-scale non-causal
    ``Attention`` is a numerical drop-in. Guards: exactly 3 inputs,
    1 output, non-causal. ``only`` restricts conversion to named nodes.
    Returns the converted count; raises on zero.
    """
    n = 0
    for node in graph.nodes:
        if node.op != "Attention":
            continue
        if only is not None and node.name not in only:
            continue
        if len(node.inputs) != 3:
            raise ValueError(f"sage surgery: Attention {node.name} has "
                             f"{len(node.inputs)} inputs (mask unsupported)")
        if len(node.outputs) != 1:
            raise ValueError(f"sage surgery: Attention {node.name} has "
                             f"{len(node.outputs)} outputs")
        if int(node.attrs.get("is_causal", 0)) != 0:
            raise RuntimeError("sage surgery: causal mask unsupported")
        node.op = PluginOp.SAGE_INT8_LEGACY
        node.attrs.clear()
        if node.name:
            node.name += "_sage"
        n += 1
    if n == 0:
        scope = f" of {len(only)} requested" if only is not None else ""
        raise RuntimeError(f"sage surgery matched 0 Attention nodes{scope}")
    return n


def apply_plugin_attention(graph,
                           plan: dict[str, tuple[Any, dict[str, Any]]]) -> dict[str, int]:
    """Rewrite ``Attention`` nodes to shipped ``dit-plugins`` ops in place.

    ``plan`` maps node name to ``(op, fields)`` where ``op`` is
    ``sage_attn``/``int8_attention`` (the Tier2 selector verdicts) and
    ``fields`` are the verdict's plugin fields (e.g. ``{"fp8_pv": 1}``).
    Each rewritten node gets the ``dit-plugins`` domain plus
    ``plugin_version``/``plugin_namespace``; stale exporter attrs are
    stripped exactly like :func:`retarget_nodes`. Same structural guards
    as :func:`apply_sage_attention` (3 inputs, 1 output, non-causal).
    Returns ``{op: count}``; raises on zero matches.
    """
    counts: dict[str, int] = {}
    for node in graph.nodes:
        if node.op != "Attention" or node.name not in plan:
            continue
        op, fields = plan[node.name]
        if op not in (PluginOp.SAGE_ATTN, PluginOp.INT8_ATTENTION):
            raise ValueError(f"plugin surgery: {node.name} wants {op!r}, "
                             f"want sage_attn/int8_attention")
        if len(node.inputs) != 3:
            raise ValueError(f"plugin surgery: Attention {node.name} has "
                             f"{len(node.inputs)} inputs (mask unsupported)")
        if len(node.outputs) != 1:
            raise ValueError(f"plugin surgery: Attention {node.name} has "
                             f"{len(node.outputs)} outputs")
        if int(node.attrs.get("is_causal", 0)) != 0:
            raise RuntimeError("plugin surgery: causal mask unsupported")
        node.op = op
        node.domain = PLUGIN_DOMAIN
        node.attrs = {"plugin_version": PLUGIN_VERSION,
                      "plugin_namespace": PLUGIN_NAMESPACE,
                      **dict(fields or {})}
        if op in _SPECS:
            allowed = _SPECS[op][2] | {"plugin_version", "plugin_namespace"}
            for k in list(node.attrs):
                if k not in allowed:
                    del node.attrs[k]
        counts[op] = counts.get(op, 0) + 1
    if not counts:
        raise RuntimeError(f"plugin surgery matched 0 Attention nodes"
                           f" of {len(plan)} requested")
    return counts


def _infer_base(graph) -> BasePrecision:
    """First Attention Q dtype; BasePrecision.BF16 when unknown."""
    for node in graph.nodes:
        if node.op == "Attention" and node.inputs:
            s = str(getattr(node.inputs[0], "dtype", "")).lower()
            if "float16" in s:
                return BasePrecision.F16
            if "float32" in s and "bfloat" not in s:
                return BasePrecision.F32
            return BasePrecision.BF16
    return BasePrecision.BF16


def _local_arch() -> str | None:
    """'smXY' of the local GPU; None without torch/CUDA (arch gates stay shut)."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception:
        pass
    return None


def apply_attention_plugins(graph, *, base: BasePrecision | None = None,
                            scales: dict[str, float] | None = None,
                            samples: dict[str, Any] | None = None,
                            arch: str | None = None,
                            gemm: GemmKind = GemmKind.AUTO,
                            dry_run: bool = False) -> dict[str, Any]:
    """Unified attention-plugin entry: decide per site, rewrite, report.

    This owns the kernel choice (fused vs sage vs native); callers only
    switch attention plugins on/off and pass calibration data. Per site:

    - Tier1 (rope-fused DiT-self block, scales cover q/k/v): strict
      selector verdict with ``samples`` evidence, else the experimental
      no-samples path (static S/D/spec-floor gates, reported as such).
    - Tier2 (plain ``Attention``): selector verdict (``sage_attn`` /
      ``int8_attention``, incl. verdict fields such as ``fp8_pv``)
      emitted directly as shipped ``dit-plugins`` nodes. Legacy
      ``SageInt8Attn`` graphs are kept untouched but flagged by
      :func:`validate_plugin_nodes` (no creator ships in this wheel).
    - Tier3 (``dit-plugins`` nodes) and native verdicts: untouched.

    Never raises on zero matches (check ``applied``); per-site structural
    errors (mask inputs, causal) still raise. ``dry_run`` decides without
    mutating. ``base``/``gemm`` take Enum members only (plain strings raise
    ``TypeError``); ``base`` is inferred from the graph when omitted.
    Returns ``{"applied": {op: n}, "native": n, "sites": [...],
    "base": ..., "arch": ..., "dry_run": bool}``.
    """
    from . import select as _sel
    scales = collect_dit_self_scales(graph, scales)
    samples = dict(samples or {})
    if base is None:
        base = _infer_base(graph)
    else:
        base = require_enum("base", base, BasePrecision)
    gemm = require_enum("gemm", gemm, GemmKind)
    if arch is None:
        arch = _local_arch()
    sites = _sel.discover_attention_sites(
        graph, base=base, scales=scales or None,
        samples=samples or None, arch=arch, gemm=gemm)
    by_attn = {s["attn"]: s for s in sites}
    applied: dict[str, int] = {}
    out_sites: list[dict[str, Any]] = []
    fused_blocks: list[dict[str, Any]] = []
    tier2_plan: dict[str, tuple[Any, dict[str, Any]]] = {}
    tier1_seen: set[str] = set()
    # Per-site fused opt-out reasons (structural N/A + Tier1 rejects) travel
    # into Tier2 entries: the old by_reason visibility, now per site.
    no_fuse = {b["attn"]: f"fused N/A ({b['reason']})"
               for b in discover_dit_self_blocks(graph)
               if not b.get("eligible") and b.get("attn")}

    tensors = graph.tensors()
    for b in discover_dit_self_blocks(graph):
        if not b["eligible"] or not _block_scales_ok(b, scales):
            continue
        tier1_seen.add(b["attn"])
        s = by_attn.get(b["attn"])
        verdict = (s or {}).get("op")
        if verdict == PluginOp.FUSED_INT8_ROPE_SAGE_ATTN:
            fused_blocks.append(b)
            out_sites.append({"attn": b["attn"], "tier": 1,
                              "selected": verdict, "emitted": verdict,
                              "reason": (s or {}).get("reason", "")})
            continue
        dims = _dims(tensors.get(b["q"]))
        ok, why = _sel.fused_static_ok(
            S=dims[2] if dims else None, D=dims[3] if dims else None)
        if ok and gemm in (GemmKind.AUTO, GemmKind.INT8):
            fused_blocks.append(b)
            out_sites.append({"attn": b["attn"], "tier": 1,
                              "selected": PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
                              "emitted": PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
                              "reason": f"experimental no-samples: {why}"})
        else:
            # Native for the fused verdict; the node stays a Tier2
            # candidate below, which owns the report entry (the fused
            # rejection reason travels along for debuggability).
            rej = (s or {}).get("reason", "") if samples \
                else f"experimental: {why}"
            no_fuse[b["attn"]] = f"fused rejected ({rej})"
            continue

    fused_attn = {b["attn"] for b in fused_blocks}

    def _note(attn: str, reason: str) -> str:
        extra = no_fuse.get(attn)
        return f"{reason} [{extra}]" if extra else reason

    for node in graph.nodes:
        if node.op == PluginOp.SAGE_INT8_LEGACY and node.name not in fused_attn:
            out_sites.append({"attn": node.name, "tier": 2,
                              "selected": PluginOp.SAGE_INT8_LEGACY,
                              "emitted": PluginOp.SAGE_INT8_LEGACY,
                              "reason": _note(node.name,
                                              "already converted: keep")})
            continue
        if node.op != "Attention" or node.name in fused_attn:
            continue
        if node.name in tier1_seen:
            # Tier1-consumed but unfused: fresh Tier2-view select (float
            # inputs), not the Tier1 INT8 verdict.
            ins = node.inputs
            q = ins[0] if len(ins) > 0 else None
            k = ins[1] if len(ins) > 1 else None
            qd = _dims(q)
            kd = _dims(k)
            s = _sel.select_attention(
                base=base, S=qd[2] if qd else None, D=qd[3] if qd else None,
                Hq=qd[1] if qd else None, Hkv=kd[1] if kd else None,
                arch=arch,
                causal=bool(int((node.attrs or {}).get("is_causal", 0))),
                has_mask=len(ins) > 3, gemm=gemm)
        else:
            s = by_attn.get(node.name, {})
        verdict = s.get("op")
        if verdict in (PluginOp.SAGE_ATTN, PluginOp.INT8_ATTENTION):
            fields = dict(s.get("fields") or {})
            tier2_plan[node.name] = (verdict, fields)
            out_sites.append({"attn": node.name, "tier": 2,
                              "selected": verdict,
                              "emitted": verdict,
                              "reason": _note(node.name,
                                              s.get("reason", ""))})
        else:
            out_sites.append({"attn": node.name, "tier": 2,
                              "selected": verdict, "emitted": None,
                              "reason": _note(node.name,
                                              s.get("reason", "native"))})

    for s in sites:
        if s["tier"] == 3:
            out_sites.append({"attn": s["attn"], "tier": 3,
                              "selected": s["op"], "emitted": s["op"],
                              "reason": s["reason"]})

    if not dry_run:
        if fused_blocks:
            sub = {e: scales[e] for b in fused_blocks
                   for e in (b["q"], b["k"], b["v"])}
            q = quantize_dit_self_anchors(graph, sub)
            for attn_out, a in q["anchors"].items():
                fuse_dit_self_attn_block(graph, attn_out=attn_out, **a)
            applied["fused_int8_rope_sage_attn"] = len(q["anchors"])
            graph.cleanup().toposort()
        if tier2_plan:
            applied.update(apply_plugin_attention(graph, tier2_plan))
    else:
        if fused_blocks:
            applied[PluginOp.FUSED_INT8_ROPE_SAGE_ATTN] = len(fused_blocks)
        for op, _fields in tier2_plan.values():
            applied[op] = applied.get(op, 0) + 1
    native = sum(1 for s in out_sites if s["emitted"] is None
                 and not any(o["attn"] == s["attn"] and o["emitted"]
                             for o in out_sites))
    return {"applied": applied, "native": native, "sites": out_sites,
            "base": base, "arch": arch, "dry_run": dry_run}


# --- explicit fusion + retarget ---

def _last_dim(t):
    if t.shape is not None and len(t.shape) > 0:
        d = t.shape[-1]
        return d if isinstance(d, int) else None
    return None


def fuse_norm_affine(graph, *, norm: str, scale: str, shift: str, out: str,
                     plugin: str | PluginOp | None = None,
                     eps: float = 1e-6, weight: str | None = None,
                     residual: str | None = None, gate: str | None = None):
    """Replace norm*(1+scale)+shift (or residual+gate*that) with one node."""
    tensors = graph.tensors()
    for n in (norm, scale, shift, out):
        if n not in tensors:
            raise KeyError(f"tensor not in graph: {n!r}")
    if residual is not None and (gate is None or residual not in tensors or gate not in tensors):
        raise ValueError("residual and gate must both name graph tensors")
    if gate is not None and residual is None:
        raise ValueError("gate requires residual")
    if weight is not None and weight not in tensors:
        raise KeyError(f"tensor not in graph: {weight!r}")
    if plugin is None:
        producer = next((p for p in tensors[norm].inputs
                         if p.op == "LayerNormalization"), None)
        plugin = PluginOp.LAYER_NORM_ADALN if producer is not None else PluginOp.RMS_ADALN
    plugin = PluginOp(plugin)
    if plugin not in (PluginOp.RMS_ADALN, PluginOp.LAYER_NORM_ADALN):
        raise TypeError(f"unsupported AdaLN plugin: {plugin!r}")

    tn, ts, tsh, to = (tensors[n] for n in (norm, scale, shift, out))
    ds = {_last_dim(t) for t in (tn, ts, tsh) if _last_dim(t) is not None}
    if len(ds) > 1:
        raise ValueError(f"last-dim mismatch in fuse anchors: {norm, scale, shift}")
    ins = [tn, ts, tsh]
    fields = {"eps": float(eps), "has_weight": int(weight is not None),
              "residual_gate": int(residual is not None)}
    if weight is not None:
        ins.append(tensors[weight])
    if residual is not None:
        ins.extend([tensors[residual], tensors[gate]])
    node = _plugin_node(graph, plugin, ins, [to], fields,
                        name=f"{plugin}_{out}")
    for prod in list(to.inputs):
        if prod is not node and to in prod.outputs:
            prod.outputs.remove(to)
        if prod is not node and prod in to.inputs:
            to.inputs.remove(prod)
    graph.cleanup().toposort()
    return node


def retarget_nodes(graph, mapping: dict[str, PluginOp]) -> int:
    """Rename custom ops to plugin ops in place: ``{"MyAttn": PluginOp.INT8_ATTENTION}``.

    Sets domain + plugin_version/namespace; attrs outside the target spec
    are stripped (stale exporter attrs would trip ``unknown fields`` and
    the TRT parser). Returns the number of nodes touched.
    """
    for src, dst in mapping.items():
        if not isinstance(dst, PluginOp):
            raise TypeError(
                f"retarget target must be PluginOp, got {dst!r} for {src!r}")
    n = 0
    for node in graph.nodes:
        if node.op in mapping and node.domain != PLUGIN_DOMAIN:
            target = mapping[node.op]
            node.op = target
            node.domain = PLUGIN_DOMAIN
            node.attrs.setdefault("plugin_version", PLUGIN_VERSION)
            node.attrs.setdefault("plugin_namespace", PLUGIN_NAMESPACE)
            if target in _SPECS:
                allowed = _SPECS[target][2] | {"plugin_version", "plugin_namespace"}
                for k in list(node.attrs):
                    if k not in allowed:
                        del node.attrs[k]
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
        if node.op == PluginOp.SAGE_INT8_LEGACY:
            # Fail fast: no wheel from this repo registers this creator, so
            # the TRT parser would reject it late with "Plugin not found".
            errs.append(f"{node.name or node.op}: legacy SageInt8Attn has no "
                        f"creator in this wheel; re-run apply_attention_plugins "
                        f"(emits shipped sage_attn/int8_attention) or retarget "
                        f"the node explicitly")
            continue
        if node.domain != PLUGIN_DOMAIN:
            continue
        spec = _SPECS.get(node.op)
        if spec is None:
            errs.append(f"{node.name or '?'}: unknown dit-plugins op {node.op!r}")
            continue
        want_in, want_out, fields = spec
        if node.op in (PluginOp.RMS_ADALN, PluginOp.LAYER_NORM_ADALN):
            want_in = 3 + int(node.attrs.get("has_weight", 0)) \
                         + 2 * int(node.attrs.get("residual_gate", 0))
        elif node.op in (PluginOp.CONVROT_LINEAR, PluginOp.CONVROT_LINEAR_W4A4):
            want_in = 4 + 2 * int(node.attrs.get("residual_gate", 0))
            want_out = 3 if int(node.attrs.get("qkv_heads", 0)) else 1
        if len(node.inputs) != want_in:
            errs.append(f"{node.name or node.op}: want {want_in} inputs, got {len(node.inputs)}")
        if len(node.outputs) != want_out:
            errs.append(f"{node.name or node.op}: want {want_out} outputs, got {len(node.outputs)}")
        extra = set(node.attrs) - {"plugin_version", "plugin_namespace"} - fields
        if extra:
            errs.append(f"{node.name or node.op}: unknown fields {sorted(extra)}")
        if node.op == PluginOp.INT8_ATTENTION and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            if d is not None and d not in (64, 128, 256):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128/256")
        if node.op == PluginOp.SAGE_ATTN and len(node.inputs) == 3:
            d = _shape_of(node.inputs[0], 3)
            if d is not None and d not in (64, 128, 256):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128/256")
            hq = _shape_of(node.inputs[0], 1)
            hkv = _shape_of(node.inputs[1], 1)
            if hq is not None and hkv is not None:
                if hkv == 0 or hq % hkv != 0:
                    errs.append(f"{node.name or node.op}: Hq={hq} must be a multiple of Hkv={hkv}")
        if node.op == PluginOp.BLOCK_SPARSE_SAGE2_ATTN and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            s = _shape_of(node.inputs[0], 2)
            if d is not None and d not in (64, 128):
                errs.append(f"{node.name or node.op}: D={d}, want 64/128")
            if s is not None and s % 128 != 0:
                errs.append(f"{node.name or node.op}: S={s}, want multiple of 128")
        if node.op == PluginOp.FUSED_INT8_ROPE_SAGE_ATTN and node.inputs:
            d = _shape_of(node.inputs[0], 3)
            s = _shape_of(node.inputs[0], 2)
            if d is not None and d != 128:
                errs.append(f"{node.name or node.op}: D={d}, want 128")
            if len(node.inputs) == 9:
                # DiT-self contract: q/k/v identical shapes, fully-static only
                # (dynamic dims/symbols never reject; TRT resolves at build).
                # Pairwise over the static subset: any two known-static
                # shapes must agree, so one dynamic input can't mask a real
                # static mismatch between the other two.
                static_shapes = []
                for t in (node.inputs[0], node.inputs[2], node.inputs[4]):
                    if t.shape is not None and all(isinstance(x, int) for x in t.shape):
                        static_shapes.append(tuple(t.shape))
                if any(a != b for a, b in itertools.combinations(static_shapes, 2)):
                    errs.append(f"{node.name or node.op}: q/k/v shapes differ: {static_shapes}")
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
        if node.op in (PluginOp.CONVROT_LINEAR, PluginOp.CONVROT_LINEAR_W4A4) and len(node.inputs) == 4:
            w4a4 = node.op == PluginOp.CONVROT_LINEAR_W4A4
            tag = node.name or node.op
            if any(t is None for t in node.inputs):
                errs.append(f"{tag}: null input")
                continue
            k = _shape_of(node.inputs[1], 1)
            n = _shape_of(node.inputs[1], 0)
            xshape = node.inputs[0].shape
            xk = _shape_of(node.inputs[0], len(xshape) - 1) if xshape else None
            if k is not None:
                k *= 2 if w4a4 else 1
                if k % 256 != 0 or k > 16384:
                    errs.append(f"{tag}: K={k}, want a multiple of 256 and <= 16384")
                if xk is not None and xk != k:
                    errs.append(f"{tag}: x K={xk} != weight K={k}")
            if n is not None and not w4a4 and n % 4 != 0:
                errs.append(f"{tag}: N={n}, want a multiple of 4")
            if node.attrs.get("group_size", 256) != 256:
                errs.append(f"{tag}: group_size must be 256")
            od = node.attrs.get("out_dtype", -1)
            if od not in (-1, 0, 1, 2):
                errs.append(f"{tag}: out_dtype={od}, want -1/0/1/2")
                od = -1
            hb = node.attrs.get("has_bias", 1)
            if hb not in (0, 1):
                errs.append(f"{tag}: has_bias={hb}, want 0/1")
            try:
                gc = int(node.attrs.get("gemm_config", -1))
            except (TypeError, ValueError):
                gc = None
            if gc is None or gc < -1:
                errs.append(f"{tag}: gemm_config={node.attrs.get('gemm_config')}, want >= -1")
            xd = _dtype_name(node.inputs[0].dtype)
            if xd not in ("float32", "float16", "bfloat16"):
                errs.append(f"{tag}: x dtype {xd}, want float32/float16/bfloat16")
            wd = _dtype_name(node.inputs[1].dtype)
            if wd != "int8":
                errs.append(f"{tag}: weight dtype {wd}, want int8")
            sd = _dtype_name(node.inputs[2].dtype)
            if sd != "float32":
                errs.append(f"{tag}: weight_scale dtype {sd}, want float32")
            want_dtype = {0: "bfloat16", 1: "float16", 2: "float32"}.get(od, xd)
            bd = _dtype_name(node.inputs[3].dtype)
            if bd != want_dtype:
                errs.append(f"{tag}: bias dtype {bd}, want {want_dtype}")
            for idx, want in ((2, (n,)), (3, (n,))):
                t = node.inputs[idx]
                if (n is not None and t.shape is not None and
                        all(isinstance(x, int) for x in t.shape) and
                        tuple(t.shape) != want):
                    errs.append(f"{tag}: input {idx} shape "
                                f"{tuple(t.shape)}, want {want}")
            ol = node.attrs.get("out_layout", 0)
            qh = node.attrs.get("qkv_heads", 0)
            hd = node.attrs.get("head_dim", 0)
            if ol not in (0, 1) or not isinstance(qh, int) or qh < 0 or not isinstance(hd, int) or hd < 0:
                errs.append(f"{tag}: bad out_layout/qkv_heads/head_dim")
            elif ol == 1:
                if node.attrs.get("epilogue", 0) or node.attrs.get("residual_gate", 0):
                    errs.append(f"{tag}: bhsd output does not support epilogue/residual")
                if xshape is None or len(xshape) != 3:
                    errs.append(f"{tag}: bhsd output requires x [B,S,K]")
                elif n is not None:
                    D = hd or 128
                    C = n // 3 if qh else n
                    H = qh or (C // D if D else 0)
                    if D <= 0 or H <= 0 or C % D or H * D != C:
                        errs.append(f"{tag}: weight N={n} is not H*D or 3*H*D")
                    want_out = 3 if qh else 1
                    if len(node.outputs) != want_out:
                        errs.append(f"{tag}: want {want_out} outputs, got {len(node.outputs)}")
                    for o in node.outputs:
                        if o.shape is not None and all(isinstance(v, int) for v in o.shape):
                            if tuple(o.shape) != (xshape[0], H, xshape[1], D):
                                errs.append(f"{tag}: output shape {tuple(o.shape)} "
                                            f"!= {(xshape[0], H, xshape[1], D)}")
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
    import os
    import warnings

    import onnx
    model = _gs().export_onnx(graph)
    if model.ir_version > TRT_MAX_IR_VERSION:
        warnings.warn(
            f"lowering ir_version {model.ir_version} -> {TRT_MAX_IR_VERSION} "
            f"for TRT parser compatibility")
        model.ir_version = TRT_MAX_IR_VERSION
    try:
        onnx.save(model, path)
    except Exception:
        # DiT models routinely exceed the 2GB protobuf limit; retry with
        # external data (location is basename-relative, never absolute).
        warnings.warn(f"standard save failed; retrying with external data: {path}")
        onnx.save(model, path, save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(path) + ".data")


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
    for spec in args.fuse_rms_adaln:
        n, s, sh, o, eps = _fuse_spec(spec)
        fuse_norm_affine(graph, norm=n, scale=s, shift=sh, out=o, plugin=PluginOp.RMS_ADALN, eps=eps)
        print(f"fused rms_adaln -> {o}")
    for spec in args.fuse_dit_self_attn:
        qi, qs, ki, ks, vi, vs, wq, wk, inv, ao, name = _dit_self_attn_spec(spec)
        fuse_dit_self_attn_block(graph, q_i8=qi, q_scale=qs, k_i8=ki,
                                 k_scale=ks, v_i8=vi, v_scale=vs,
                                 rms_w_q=wq, rms_w_k=wk, inv_freq=inv,
                                 attn_out=ao, name=name)
        print(f"fused fused_int8_rope_sage_attn -> {ao}")
    if args.fuse_dit_self_attn:
        graph.cleanup().toposort()

    print(summarize(graph))
    errs = validate_plugin_nodes(graph)
    for e in errs:
        print(f"ERROR: {e}")
    # Never persist an invalid graph; dry-run only reports.
    if errs:
        return 1
    if args.dry_run:
        return 0
    save(graph, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
