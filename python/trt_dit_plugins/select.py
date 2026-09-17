# SPDX-License-Identifier: Apache-2.0
"""Sequence-length driven plugin selection (pure functions).

Two-axis input model: ``base`` (:class:`BasePrecision`, always required)
is the compute dtype; ``quant`` (``None``/:class:`QuantKind`) is the
quantization riding on top of it with an external scale. Quantized kinds
always carry a scale, floats never do -- derived, not caller-chosen.
Public APIs take Enum members only (plain strings raise :class:`TypeError`);
the ``*_coerce`` helpers remain for string boundaries (CLI).

Quality gating is per loss class (``QUALITY_CLASS``): fusing loses
accuracy by design, so one absolute bar across classes would either
meaninglessly pass everything or systematically kill the fused path.
``min_cos`` therefore applies to ``quant``-class static values and to
measured simulation results; the ``fused`` class is guarded by the spec
floor (``FUSED_SPEC_FLOOR``, a broken-detector, not a quality bar) plus
the mandatory upcast simulation below.

``simulate_upcast_compat`` quantizes a calibration sample and upcasts it
back to ``base``, measuring what the quantized path preserves on YOUR
data. External-quant consumers (the fused path) require its result via
``compat`` -- no simulation evidence, no quantized selection. NOTE: the
E4M3 model here is a numpy approximation of HW FP8; treat margins
accordingly and cross-check on GPU when possible (see test).

``op=None`` means plain TRT (no plugin): the native path is a
first-class candidate. ``trt_flags`` names TRT enum members; plugin
paths need no precision flags (strongly-typed exact I/O, cf.
``_engine.py``), native paths do. Create networks with
``1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)`` and apply
builder flags with :func:`apply_builder_flags`.

Selector-vs-tactic boundary (the one-line rule): a choice that changes
anything on the ONNX/TRT graph belongs here; how one plugin layer is
implemented faster belongs to ``IPluginV3`` tactics. Concretely the
selector owns causal/mask, dtype/quant, D, GQA validity, rope fusion,
S valid ranges, accuracy/compat, native-vs-plugin, fused-vs-unfused and
sparse-vs-dense. Kernel variants, tile/warp/CTA configs, S- or
GQA-specific kernels and CUDA-vs-cuBLAS implementations belong to
tactics -- provided they are accuracy-equivalent. TRT tactics optimize
latency only and cannot see accuracy, so accuracy-differing variants
stay separate plugins under the quality gates above; never merge them
into tactics. ``bench`` below is therefore routing *between* plugins at
one site shape, never a substitute for in-plugin tactic benchmarking.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

NATIVE_OP = None  # selector returns op=None for the plain-TRT path.


class BasePrecision(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"


class QuantKind(str, Enum):
    INT8 = "int8"
    FP8 = "fp8"


class GemmKind(str, Enum):
    AUTO = "auto"
    INT8 = "int8"
    FP16 = "fp16"
    FP8 = "fp8"

    __str__ = str.__str__


class NormKind(str, Enum):
    LAYER = "layer"
    RMS = "rms"

    __str__ = str.__str__


class RopeStyle(str, Enum):
    SPLIT_HALF = "split-half"
    INTERLEAVED = "interleaved"

    __str__ = str.__str__


class PluginOp(str, Enum):
    """Every plugin op, managed in one place (str-compatible: == "name")."""

    INT8_ATTENTION = "int8_attention"
    SAGE_ATTN = "sage_attn"
    ADALN = "adaln"
    RMS_ADALN = "rms_adaln"
    APPLY_ROPE = "apply_rope"
    RMS_ROPE_SPLIT_HALF = "rms_rope_split_half"
    STOCHASTIC_ROUND_FP8 = "stochastic_round_fp8"
    BLOCK_SPARSE_SAGE2_ATTN = "block_sparse_sage2_attn"
    FUSED_INT8_ROPE_SAGE_ATTN = "fused_int8_rope_sage_attn"
    SAGE_INT8_LEGACY = "SageInt8Attn"

    __str__ = str.__str__


_BASE_ALIASES = {
    "f32": "f32", "float32": "f32", "fp32": "f32",
    "f16": "f16", "float16": "f16", "fp16": "f16", "half": "f16",
    "bf16": "bf16", "bfloat16": "bf16",
}
_QUANT_ALIASES = {
    "int8": "int8", "i8": "int8",
    "fp8": "fp8", "float8": "fp8", "e4m3": "fp8", "float8_e4m3fn": "fp8",
}

# Measured cosine points: {op: {S: cos}}. Single lab/GPU snapshots, NOT
# universal truths. Gate lookup uses the exact-S point when present, else
# the conservative min over known points (see _table_cos).
KNOWN_COS: dict[str, dict[int, float]] = {
    PluginOp.FUSED_INT8_ROPE_SAGE_ATTN: {2048: 0.9977, 4096: 0.99855},
    PluginOp.INT8_ATTENTION: {4096: 0.99990},
    PluginOp.SAGE_ATTN: {4096: 0.99927},
    "sage_attn+fp8_pv": {4096: 0.99925},
    PluginOp.BLOCK_SPARSE_SAGE2_ATTN: {4096: 0.99953},
    PluginOp.ADALN: {}, PluginOp.RMS_ADALN: {}, PluginOp.APPLY_ROPE: {},
    PluginOp.RMS_ROPE_SPLIT_HALF: {},
}

QUALITY_CLASS: dict[str, str] = {
    PluginOp.ADALN: "exact", PluginOp.RMS_ADALN: "exact",
    PluginOp.APPLY_ROPE: "exact", PluginOp.RMS_ROPE_SPLIT_HALF: "exact",
    PluginOp.INT8_ATTENTION: "quant", PluginOp.SAGE_ATTN: "quant",
    PluginOp.BLOCK_SPARSE_SAGE2_ATTN: "quant",
    PluginOp.FUSED_INT8_ROPE_SAGE_ATTN: "fused",
}
# Spec sec 7 floor (vs the separated path): below this the kernel is
# broken, not lossy. A broken-detector, not a quality bar -- note the
# reference differs from the table's (vs-eager) values by construction.
FUSED_SPEC_FLOOR = 0.997

_FLAG_ORDER = ("FP16", "BF16", "INT8", "FP8")
_BASE_FLAG = {BasePrecision.F32: (), BasePrecision.F16: ("FP16",),
              BasePrecision.BF16: ("BF16",)}
_QUANT_FLAG = {QuantKind.INT8: ("INT8",), QuantKind.FP8: ("FP8",)}
_GEMM_ALIASES = {
    "auto": "auto",
    "int8": "int8", "i8": "int8",
    "fp16": "fp16", "f16": "fp16",
    "fp8": "fp8", "float8": "fp8", "e4m3": "fp8",
}
_NORM_ALIASES = {
    "layer": "layer", "layernorm": "layer", "ln": "layer",
    "rms": "rms", "rmsnorm": "rms",
}
_ROPE_ALIASES = {
    "split-half": "split-half", "split_half": "split-half", "split": "split-half",
    "interleaved": "interleaved",
}
_GEMM_FLAG = {GemmKind.INT8: ("INT8",), GemmKind.FP16: ("FP16",),
              GemmKind.FP8: ("FP8",)}


def require_enum(name: str, v: Any, cls: type) -> Any:
    """Public APIs take Enum members only; plain strings raise TypeError."""
    if isinstance(v, cls):
        return v
    raise TypeError(f"{name} must be {cls.__name__} (plain strings rejected), "
                    f"got {v!r}")

# Static preference when no bench table (or incomplete keys) is given.
_STATIC_RANK = (PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
                PluginOp.BLOCK_SPARSE_SAGE2_ATTN,
                PluginOp.INT8_ATTENTION, PluginOp.SAGE_ATTN)

_DISCOVERY_OPS = ("Attention", PluginOp.SAGE_INT8_LEGACY)
_PLUGIN_ATTN_OPS = (PluginOp.SAGE_ATTN, PluginOp.INT8_ATTENTION,
                    PluginOp.FUSED_INT8_ROPE_SAGE_ATTN,
                    PluginOp.BLOCK_SPARSE_SAGE2_ATTN)
_FUSED_MAX_S = 9216  # validated range per fused spec sec 2.1 (validator gates >1024).

# Observed kernel failures. Entries must stay excluded by the gates below;
# test_known_bad_regressions enforces it via each probe. Add an entry (with
# a probe) whenever a crash/wrong-result condition is found -- never relax
# a covering gate without removing its entry first. Defense in depth for
# the S<=1024 entry also lives in the launcher (assert), the wrapper
# (configurePlugin/onShapeChange reject), ops.py (pre-launch ValueError),
# and surgeon.validate_plugin_nodes.
KNOWN_BAD: tuple[dict[str, Any], ...] = (
    {"op": PluginOp.FUSED_INT8_ROPE_SAGE_ATTN, "cond": "S<=1024",
     "failure": "L<=1024 path: cos 0.80 at S=512, illegal memory access at S=1024",
     "probe": {"base": BasePrecision.BF16, "quant": QuantKind.INT8, "S": 1024, "D": 128,
               "Hq": 16, "Hkv": 16, "arch": "sm89", "rope_fusable": True}},
)


def _coerce_base(v: Any) -> BasePrecision:
    if isinstance(v, BasePrecision):
        return v
    try:
        return BasePrecision(_BASE_ALIASES[str(v).lower()])
    except KeyError:
        raise ValueError(f"unknown base precision {v!r} (want f32/f16/bf16)") from None


def _coerce_quant(v: Any) -> QuantKind:
    if isinstance(v, QuantKind):
        return v
    try:
        return QuantKind(_QUANT_ALIASES[str(v).lower()])
    except KeyError:
        raise ValueError(f"unknown quant {v!r} (want None/int8/fp8)") from None


def _coerce_gemm(v: Any) -> GemmKind:
    if isinstance(v, GemmKind):
        return v
    try:
        return GemmKind(_GEMM_ALIASES[str(v).lower()])
    except KeyError:
        raise ValueError(f"unknown gemm {v!r} (want auto/int8/fp16/fp8)") from None


def _coerce_norm(v: Any) -> NormKind:
    if isinstance(v, NormKind):
        return v
    try:
        return NormKind(_NORM_ALIASES[str(v).lower()])
    except KeyError:
        raise ValueError(f"unknown norm {v!r} (want layer/rms)") from None


def _coerce_style(v: Any) -> RopeStyle:
    if isinstance(v, RopeStyle):
        return v
    try:
        return RopeStyle(_ROPE_ALIASES[str(v).lower()])
    except KeyError:
        raise ValueError(f"unknown rope style {v!r} (want split-half/interleaved)") from None


def require_opt(name: str, v: Any, cls: type) -> Any:
    """require_enum that also accepts None (for Optional enum params)."""
    if v is None:
        return None
    return require_enum(name, v, cls)


def _parse_arch(arch: Any) -> int | None:
    """'sm89'/89 -> 89; unknown/None -> None (arch-gated paths excluded)."""
    if arch is None:
        return None
    s = str(arch).lower().removeprefix("sm")
    return int(s) if s.isdigit() else None


def _table_cos(op: str, S: int | None) -> float:
    """Exact-S point when measured, else the conservative min. Exact ops: 1.0."""
    pts = KNOWN_COS[op]
    if not pts:
        return 1.0
    if S in pts:
        return pts[S]
    return min(pts.values())


def fused_static_ok(*, S: int | None, D: int | None) -> tuple[bool, str]:
    """Static fused gates without samples (S/D/spec-floor only).

    Mirrors the sample-independent half of :func:`select_attention`'s
    fused branch. The strict path additionally requires upcast-simulation
    evidence (``compat``); use this only to *report* the experimental
    no-samples path, never as a strict verdict.
    """
    if S is None:
        return False, "fused: unknown S"
    if S <= 1024:
        return False, f"fused: S={S} known-bad (<=1024)"
    if S > _FUSED_MAX_S:
        return False, f"fused: S={S} out of validated range (>{_FUSED_MAX_S})"
    if D is not None and D != 128:
        return False, f"fused: D={D}, want 128"
    if _table_cos(PluginOp.FUSED_INT8_ROPE_SAGE_ATTN, S) < FUSED_SPEC_FLOOR:
        return False, f"fused: below spec floor {FUSED_SPEC_FLOOR}"
    return True, "static gates pass (compat unverified: experimental)"


def _trt_flags(base: BasePrecision, gemm: str,
               quant: QuantKind | None) -> dict[str, list[str]]:
    """Advisory TRT flags. Plugins need none; native needs base+quant+gemm."""
    want = set(_BASE_FLAG[base])
    if quant is not None:
        want |= set(_QUANT_FLAG[quant])
    want |= set(_GEMM_FLAG.get(gemm, ()))
    return {"network": ["STRONGLY_TYPED"],
            "builder": [f for f in _FLAG_ORDER if f in want]}


def _plugin_flags() -> dict[str, list[str]]:
    return {"network": ["STRONGLY_TYPED"], "builder": []}


def apply_builder_flags(config, flags: dict[str, Any]) -> int:
    """Set ``trt.BuilderFlag`` members named in ``flags["builder"]``.

    Needs ``tensorrt`` installed. The network itself must have been created
    with ``STRONGLY_TYPED`` (see module docstring). Returns flags set.
    """
    import tensorrt as trt

    n = 0
    for name in flags.get("builder", ()):
        config.set_flag(getattr(trt.BuilderFlag, name))
        n += 1
    return n


def _e4m3_nearest(v):
    """Round to FP8 E4M3 code values (numpy approximation of HW FP8).

    Normal step 2^(e2-3), subnormal step 2^-9, max 448, NaN propagates,
    Inf saturates. Round-half-even via np.round. Treat as approximate:
    cross-check against GPU cast when it matters (see test).
    """
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    nan = np.isnan(v)
    ax = np.minimum(np.abs(v), 448.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2 = np.floor(np.log2(np.maximum(ax, 2.0 ** -9)))
    step = np.where(e2 >= -6, 2.0 ** (np.clip(e2, -6, 8) - 3), 2.0 ** -9)
    q = np.minimum(np.round(ax / step) * step, 448.0)
    out = np.sign(v) * q
    out[nan] = np.nan
    out[(v == np.inf)] = 448.0
    out[(v == -np.inf)] = -448.0
    return out


def _to_base(arr, base: BasePrecision):
    """Cast to base precision and back to float64 for metric computation.

    BF16 has no numpy dtype here: round the float32 mantissa to 7 bits
    (round-to-nearest-even), which is value-equivalent.
    """
    import numpy as np

    if base is BasePrecision.F32:
        return np.asarray(arr, dtype=np.float32).astype(np.float64)
    if base is BasePrecision.F16:
        return np.asarray(arr, dtype=np.float16).astype(np.float64)
    f = np.asarray(arr, dtype=np.float32)
    u = f.view(np.uint32)
    rnd = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) \
        & ~np.uint32(0xFFFF)
    return rnd.view(np.float32).astype(np.float64)


def simulate_upcast_compat(sample, scale, *, base: BasePrecision,
                           quant: QuantKind,
                           min_cos: float | None = None) -> dict[str, Any]:
    """Quantize a calibration sample, upcast to ``base``, measure fidelity.

    ``sample``: float array (a slice is fine); ``scale``: per-tensor scalar
    (arrays with !=1 element are rejected -- per-token/channel scales are
    out of contract). Reports ``{"pass", "cos", "max_abs_err",
    "saturation", "reason"}`` where ``saturation`` is the fraction past
    the representable range and ``pass`` is all-finite plus the ``min_cos``
    bar when given. Intended input to ``select_attention(compat=...)``.
    """
    import numpy as np

    base = require_enum("base", base, BasePrecision)
    quant = require_enum("quant", quant, QuantKind)
    x = np.asarray(sample, dtype=np.float64)
    try:
        s = float(np.asarray(scale, dtype=np.float64).reshape(-1)[0])
        if np.asarray(scale).size != 1:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("scale must be a per-tensor scalar") from None
    if not (s > 0):
        raise ValueError(f"scale must be positive, got {s}")
    if quant is QuantKind.INT8:
        q = np.clip(np.round(x / s), -128, 127)
        sat = float(np.mean(np.abs(x / s) > 127))
    else:
        q = _e4m3_nearest(x / s)
        sat = float(np.mean(np.abs(x / s) > 448))
    up = _to_base(q * s, base)
    ref = _to_base(x, base)
    err = float(np.max(np.abs(up - ref))) if up.size else 0.0
    finite = bool(np.all(np.isfinite(up)))
    nref = float(np.dot(ref.ravel(), ref.ravel()))
    nup = float(np.dot(up.ravel(), up.ravel()))
    if not finite:
        cos = float("nan")
    elif nref == 0.0:
        cos = 1.0 if nup == 0.0 else 0.0
    else:
        cos = float(np.dot(ref.ravel(), up.ravel()) / np.sqrt(nref * nup))
    ok = finite and (min_cos is None or (cos == cos and cos >= min_cos))
    detail = f"cos={cos:.6f} max_abs_err={err:.3g} saturation={sat:.4f}"
    if not finite:
        reason = f"non-finite upcast; {detail}"
    elif min_cos is not None and not (cos == cos and cos >= min_cos):
        reason = f"cos below min_cos {min_cos}; {detail}"
    else:
        reason = f"upcast preserves base-{base.value}; {detail}"
    return {"pass": bool(ok), "cos": cos, "max_abs_err": err,
            "saturation": sat, "reason": reason}


def _check_bench_keys(bench: dict | None) -> None:
    """Bench keys must be strict 6-tuples; legacy 4-tuples fail fast.

    A 4-tuple entry would silently never match and fall back to static
    rank, so malformed keys are rejected up front with the bad keys shown.
    """
    if not bench:
        return
    bad = [k for k in bench if not isinstance(k, tuple) or len(k) != 6]
    if bad:
        raise ValueError("bench keys must be (op, S, D, Hq, Hkv, arch), got "
                         f"{bad!r}")


def _bench_ms(bench: dict | None, op: str, S: int | None, D: int | None,
              Hq: int | None, Hkv: int | None, arch: int | None) -> float | None:
    if not bench or S is None or D is None or Hq is None or Hkv is None \
            or arch is None:
        return None
    v = bench.get((op, S, D, Hq, Hkv, arch))
    return float(v) if v is not None else None


def select_attention(*, base: BasePrecision, quant: QuantKind | None = None,
                     S: int | None = None,
                     D: int | None = None, Hq: int | None = None,
                     Hkv: int | None = None, arch: Any = None,
                     causal: bool = False, has_mask: bool = False,
                     sparse_mask: bool = False, rope_fusable: bool = False,
                     gemm: GemmKind = GemmKind.AUTO,
                     min_cos: float | None = None,
                     compat: dict | None = None,
                     bench: dict | None = None) -> dict[str, Any]:
    """Pick the attention implementation for one site.

    ``S``/``D``/``H`` accept None (dynamic): S-gated paths are then
    excluded, other unknown dims never reject (build-time resolve, same
    philosophy as ``surgeon.validate_plugin_nodes``). ``gemm`` pins the
    quantized-GEMM vehicle: ``GemmKind.INT8`` (INT8-QK family),
    ``GemmKind.FP8`` (FP8 vehicle = ``sage_attn`` ``fp8_pv`` tactic, else
    native+FP8), ``GemmKind.FP16`` (no quantized GEMM -> native).
    ``GemmKind.AUTO`` never picks ``fp8_pv``
    (measured slower on dense shapes; opt-in only). ``min_cos`` gates
    ``quant``-class static values and the ``compat`` simulation result;
    the ``fused`` class is lossy by design and answers to the spec floor
    plus ``compat`` instead (see module docstring). The native path
    (accuracy unverified) always survives. ``bench`` maps
    ``(op, S, D, Hq, Hkv, arch) -> ms``: head counts are part of the key
    because GQA changes K/V traffic (``Hkv*S*D``) and hence latency at the
    same ``(S, D)``. Measure your model's points (warmup100/iter500, same
    conditions as spec sec 7; prefill-like assumed, re-measure for
    decode-style use). Used only to route *between* plugins when every
    qualified candidate has an exact entry, else the static rank applies;
    never a substitute for in-plugin tactic benchmarking (see module
    docstring).
    """
    base = require_enum("base", base, BasePrecision)
    quant = require_opt("quant", quant, QuantKind)
    gemm = require_enum("gemm", gemm, GemmKind)
    arch_n = _parse_arch(arch)
    _check_bench_keys(bench)

    def native(reason: str) -> dict[str, Any]:
        return {"op": NATIVE_OP, "fields": {}, "reason": reason,
                "cos": None, "trt_flags": _trt_flags(base, gemm, quant)}

    notes: list[str] = []
    if causal or (has_mask and not sparse_mask):
        return native("causal-or-dense-mask: no plugin path covers it; plain TRT")
    if gemm is GemmKind.FP16:
        return native("GEMM pinned to fp16: plugin set is INT8/FP8-GEMM only; plain TRT")

    def quant_cos_ok(op: str) -> bool:
        if min_cos is None:
            return True
        c = _table_cos(op, S)
        if c < min_cos:
            notes.append(f"{op}: table cos {c} (S-matched or conservative min)"
                         f" < min_cos {min_cos}")
            return False
        return True

    gqa_ok = Hkv in (None, 0) or Hq is None or (Hkv != 0 and Hq % Hkv == 0)
    square_h = Hq is None or Hkv is None or Hq == Hkv

    qualified: list[str] = []
    fields: dict[str, dict[str, Any]] = {}

    # Fused INT8 RoPE+Sage: lossy by design; min_cos does not apply to its
    # static value. Guarded by compat evidence + the spec broken-floor.
    if gemm in (GemmKind.AUTO, GemmKind.INT8):
        if not rope_fusable:
            notes.append("fused: no rope/INT8 path ready")
        elif quant is not QuantKind.INT8:
            notes.append("fused: needs INT8+scale inputs")
        elif D is not None and D != 128:
            notes.append(f"fused: D={D}, want 128")
        elif S is None:
            notes.append("fused: unknown S (needs S>1024; dynamic profiles"
                         " must be proven >1024 before fusing)")
        elif S <= 1024:
            notes.append(f"fused: S={S} known-bad (wrong results/crash"
                         " observed on the L<=1024 path)")
        elif S > _FUSED_MAX_S:
            notes.append(f"fused: S={S} out of validated range (>{_FUSED_MAX_S})")
        elif compat is None or not isinstance(compat, dict) \
                or not compat.get("pass", False):
            notes.append("fused: quant compat simulation required"
                         " -- run simulate_upcast_compat")
        elif min_cos is not None and not (
                isinstance(compat.get("cos"), float)
                and compat["cos"] == compat["cos"]
                and compat["cos"] >= min_cos):
            notes.append(f"fused: sim cos {compat.get('cos')} < min_cos {min_cos}")
        elif _table_cos(PluginOp.FUSED_INT8_ROPE_SAGE_ATTN, S) < FUSED_SPEC_FLOOR:
            notes.append("fused: below spec floor"
                         f" {FUSED_SPEC_FLOOR} (broken, not lossy)")
        else:
            qualified.append(PluginOp.FUSED_INT8_ROPE_SAGE_ATTN)
            fields[PluginOp.FUSED_INT8_ROPE_SAGE_ATTN] = {}

    # Block-sparse (needs a block-form mask, sm89-only, float inputs).
    if sparse_mask and quant is None and gemm in (GemmKind.AUTO, GemmKind.INT8):
        if arch_n != 89:
            notes.append(f"block-sparse: needs sm89, got {arch!r}")
        elif D is not None and D not in (64, 128):
            notes.append(f"block-sparse: D={D}, want 64/128")
        elif base not in (BasePrecision.F16, BasePrecision.BF16):
            notes.append(f"block-sparse: needs f16/bf16 base, got {base.value}")
        elif S is None or S % 128 != 0:
            notes.append(f"block-sparse: S={S}, want multiple of 128")
        elif quant_cos_ok(PluginOp.BLOCK_SPARSE_SAGE2_ATTN):
            qualified.append(PluginOp.BLOCK_SPARSE_SAGE2_ATTN)
            fields[PluginOp.BLOCK_SPARSE_SAGE2_ATTN] = {}
    elif sparse_mask and quant is not None:
        notes.append("block-sparse: needs float inputs")

    # FP8 vehicle: explicit opt-in only, float inputs only. Deliberately a
    # plugin *field*, not a builder tactic: the PV-FP8 path differs in
    # accuracy (0.99925 vs 0.99927), and tactic selection is latency-only,
    # so an automatic tactic could silently trade accuracy. The selector
    # keeps this choice explicit and accuracy-visible (see module docstring).
    if gemm is GemmKind.FP8 and quant is None:
        if arch_n != 89:
            notes.append(f"fp8_pv: needs sm89, got {arch!r}")
        elif D is not None and D not in (64, 128):
            notes.append(f"fp8_pv: D={D}, want 64/128")
        elif base not in (BasePrecision.F16, BasePrecision.BF16):
            notes.append(f"fp8_pv: needs f16/bf16 inputs, got {base.value}")
        elif S is None or S % 128 != 0:
            notes.append(f"fp8_pv: S={S}, want square multiple of 128")
        elif not gqa_ok:
            notes.append("fp8_pv: GQA mismatch")
        elif quant_cos_ok("sage_attn+fp8_pv"):
            qualified.append(PluginOp.SAGE_ATTN)
            fields[PluginOp.SAGE_ATTN] = {"fp8_pv": 1}
    elif gemm is GemmKind.FP8 and quant is not None:
        notes.append("fp8_pv: needs float inputs")

    # Dense INT8 / portable paths (float inputs only).
    if gemm in (GemmKind.AUTO, GemmKind.INT8) and quant is None:
        if D is not None and D not in (64, 128, 256):
            notes.append(f"dense: D={D}, want 64/128/256")
        else:
            if square_h and quant_cos_ok(PluginOp.INT8_ATTENTION):
                qualified.append(PluginOp.INT8_ATTENTION)
                fields[PluginOp.INT8_ATTENTION] = {}
            elif not square_h:
                notes.append("int8_attention: GQA needs Hq==Hkv")
            if gqa_ok and quant_cos_ok(PluginOp.SAGE_ATTN):
                qualified.append(PluginOp.SAGE_ATTN)
                fields[PluginOp.SAGE_ATTN] = {}
            elif not gqa_ok:
                notes.append("sage_attn: Hq must be a multiple of Hkv")
    elif quant is QuantKind.INT8:
        notes.append("INT8+scale: only the fused path consumes it"
                     " (verdict above); else plain TRT")
    elif quant is QuantKind.FP8:
        notes.append("FP8+scale: no plugin consumes it")

    if not qualified:
        tail = ("min_cos gate emptied the list; " if min_cos is not None else "") \
            + ("plain TRT" if not notes else "; ".join(notes) + "; plain TRT")
        return native(tail)

    timed = {op: _bench_ms(bench, op, S, D, Hq, Hkv, arch_n)
             for op in qualified}
    if qualified and all(v is not None for v in timed.values()):
        # ponytail: exact-key bench only; incomplete tables keep static rank.
        op = min(timed, key=lambda o: timed[o])  # type: ignore[index]
        why = f"bench {timed[op]:g}ms"
    else:
        op = min(qualified, key=_STATIC_RANK.index)
        why = "static rank"
    key = "sage_attn+fp8_pv" if fields[op].get("fp8_pv") else op
    return {"op": op, "fields": fields[op],
            "reason": f"{why}; {op} fits (class {QUALITY_CLASS[op]})",
            "cos": _table_cos(key, S), "trt_flags": _plugin_flags()}


def select_norm(*, base: BasePrecision, quant: QuantKind | None = None,
                norm: NormKind,
                shape: tuple | None = None, eps: float = 1e-6,
                min_cos: float | None = None) -> dict[str, Any]:
    """``NormKind.LAYER`` -> ``adaln`` / ``NormKind.RMS`` -> ``rms_adaln`` (or native)."""
    base = require_enum("base", base, BasePrecision)
    quant = require_opt("quant", quant, QuantKind)
    norm = require_enum("norm", norm, NormKind)
    op = PluginOp.ADALN if norm is NormKind.LAYER else PluginOp.RMS_ADALN

    def native(reason: str) -> dict[str, Any]:
        return {"op": NATIVE_OP, "fields": {}, "reason": reason,
                "cos": None, "trt_flags": _trt_flags(base, GemmKind.AUTO, quant)}

    if quant is not None:
        return native(f"{op}: no quantized-norm plugin; plain TRT")
    if shape is not None and len(shape) != 2:
        return native(f"{op}: needs flat [N,D], got rank-{len(shape)}; plain TRT")
    if min_cos is not None and 1.0 < min_cos:
        return native(f"{op}: exact cos 1.0 < min_cos {min_cos}; plain TRT")
    return {"op": op, "fields": {"eps": float(eps)}, "reason": f"{op} fits",
            "cos": 1.0, "trt_flags": _plugin_flags()}


def select_rope(*, base: BasePrecision, quant: QuantKind | None = None,
                style: RopeStyle = RopeStyle.SPLIT_HALF,
                fuse_norm: bool = False, D: int | None = None,
                has_scales: bool = False, epsilon: float = 1e-6,
                rot_dim: int = 0) -> dict[str, Any]:
    """RoPE variant selection (or native). Styles don't substitute: a style
    mismatch returns native, never the other plugin."""
    base = require_enum("base", base, BasePrecision)
    quant = require_opt("quant", quant, QuantKind)
    style = require_enum("style", style, RopeStyle)

    def native(reason: str) -> dict[str, Any]:
        return {"op": NATIVE_OP, "fields": {}, "reason": reason,
                "cos": None, "trt_flags": _trt_flags(base, GemmKind.AUTO, quant)}

    if quant is not None:
        return native(f"rope: no quantized-rope plugin; plain TRT")
    if base not in (BasePrecision.F16, BasePrecision.BF16):
        return native(f"rope: q/k need f16/bf16 base, got {base.value}; plain TRT")
    if style is RopeStyle.INTERLEAVED and not fuse_norm:
        if D is not None and D % 2:
            return native(f"apply_rope: D={D} not even; plain TRT")
        return {"op": PluginOp.APPLY_ROPE, "fields": {},
                "reason": "apply_rope fits", "cos": 1.0,
                "trt_flags": _plugin_flags()}
    if style is RopeStyle.SPLIT_HALF and fuse_norm:
        if not has_scales:
            return native("rms_rope_split_half: needs norm scales; plain TRT")
        if D is not None and D % 32:
            return native(f"rms_rope_split_half: D={D} not a multiple of 32; plain TRT")
        return {"op": PluginOp.RMS_ROPE_SPLIT_HALF,
                "fields": {"epsilon": float(epsilon), "rot_dim": int(rot_dim)},
                "reason": "rms_rope_split_half fits", "cos": 1.0,
                "trt_flags": _plugin_flags()}
    return native(f"rope: no plugin for style={style!r} fuse_norm={fuse_norm}; plain TRT")


def _dim(t, i: int) -> int | None:
    if t is None or t.shape is None or i >= len(t.shape):
        return None
    d = t.shape[i]
    return d if isinstance(d, int) else None


def _block_compat(b: dict[str, Any], scales: dict[str, float],
                  samples: dict[str, Any], base: BasePrecision,
                  min_cos: float | None) -> dict[str, Any] | None:
    """Run the upcast simulation over the q/k/v edges that have samples.

    Returns None when no edge has a sample (no evidence). Otherwise the
    block passes only if every simulated edge passes; cos is the min.
    """
    import numpy as np

    have = [(e, samples[e]) for e in (b.get("q"), b.get("k"), b.get("v"))
            if e in samples]
    if not have:
        return None
    results = [simulate_upcast_compat(s, scales[e], base=base,
                                      quant=QuantKind.INT8, min_cos=min_cos)
               for e, s in have]
    cos = [r["cos"] for r in results if r["cos"] == r["cos"]]
    ok = all(r["pass"] for r in results)
    return {"pass": bool(ok),
            "cos": float(np.min(cos)) if cos else None,
            "reason": "; ".join(r["reason"] for r in results)}


def discover_attention_sites(graph, *, base: BasePrecision,
                             quant: QuantKind | None = None,
                             scales: dict[str, float] | None = None,
                             samples: dict[str, Any] | None = None,
                             arch: Any = None, gemm: GemmKind = GemmKind.AUTO,
                             min_cos: float | None = None,
                             bench: dict | None = None) -> list[dict[str, Any]]:
    """Enumerate every attention site and run the pure selector per site.

    Tier1: ``rms_rope_split_half``+attention pairs with scales AND samples
    ready -> simulated fused candidacy (pairing delegated to ``surgeon``).
    Tier2: remaining legacy ``Attention``/``SageInt8Attn`` nodes ->
    shape/attr features. Tier3: already-pluginized ``dit-plugins`` nodes
    -> keep, report only.
    """
    from . import surgeon as _s

    base = require_enum("base", base, BasePrecision)
    quant = require_opt("quant", quant, QuantKind)
    gemm = require_enum("gemm", gemm, GemmKind)
    tensors = graph.tensors()
    sites: list[dict[str, Any]] = []
    consumed: set[str] = set()

    for b in _s.discover_dit_self_blocks(graph):
        if not b["eligible"] or not b.get("attn_out"):
            continue
        need = (b.get("q"), b.get("k"), b.get("v"))
        if not (scales and all(k in scales and scales[k] > 0 for k in need)):
            continue  # scales missing -> Tier2 view of the same node.
        consumed.add(b["attn"])
        q = tensors.get(b["q"])
        compat = _block_compat(b, scales, samples or {}, base, min_cos)
        dec = select_attention(base=base, quant=QuantKind.INT8,
                               S=_dim(q, 2), D=_dim(q, 3), Hq=_dim(q, 1),
                               Hkv=_dim(q, 1), arch=arch, rope_fusable=True,
                               gemm=gemm, min_cos=min_cos, compat=compat,
                               bench=bench)
        sites.append({"attn": b["attn"], "attn_op": b["attn_op"],
                      "tier": 1, **dec})

    for n in graph.nodes:
        if getattr(n, "domain", "") == _s.PLUGIN_DOMAIN \
                and n.op in _PLUGIN_ATTN_OPS:
            s = _dim(n.inputs[0], 2) if n.inputs else None
            key = "sage_attn+fp8_pv" if (n.attrs or {}).get("fp8_pv") else n.op
            sites.append({"attn": n.name, "attn_op": n.op, "tier": 3,
                          "op": n.op, "fields": {},
                          "reason": f"already {n.op}: keep",
                          "cos": _table_cos(key, s) if key in KNOWN_COS else None,
                          "trt_flags": _plugin_flags()})
            continue
        if n.op not in _DISCOVERY_OPS or n.name in consumed:
            continue
        if getattr(n, "domain", "") == _s.PLUGIN_DOMAIN:
            continue
        ins = n.inputs
        q = ins[0] if len(ins) > 0 else None
        k = ins[1] if len(ins) > 1 else None
        causal = bool(int(n.attrs.get("is_causal", 0))) if n.attrs else False
        dec = select_attention(base=base, quant=quant,
                               S=_dim(q, 2), D=_dim(q, 3), Hq=_dim(q, 1),
                               Hkv=_dim(k, 1), arch=arch, causal=causal,
                               has_mask=len(ins) > 3, sparse_mask=False,
                               rope_fusable=False, gemm=gemm,
                               min_cos=min_cos, bench=bench)
        sites.append({"attn": n.name, "attn_op": n.op, "tier": 2, **dec})
    return sites


def main(argv=None) -> int:
    """Dry-run reporter: discover + select over an ONNX file, print only."""
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Suggest TRT plugins per attention site (dry-run, read-only).")
    ap.add_argument("input", help="input .onnx")
    ap.add_argument("--base", required=True, help="f32/f16/bf16")
    ap.add_argument("--quant", default=None, help="int8/fp8 (default: none)")
    ap.add_argument("--arch", default=None, help="e.g. sm89 (default: unknown)")
    ap.add_argument("--gemm", default="auto")
    ap.add_argument("--min-cos", type=float, default=None)
    ap.add_argument("--scales-json", default=None,
                    help="path to {tensor_name: scale} JSON")
    ap.add_argument("--samples-npz", default=None,
                    help="path to .npz with calibration arrays keyed by tensor name")
    args = ap.parse_args(argv)
    try:
        base = _coerce_base(args.base)
        quant = _coerce_quant(args.quant) if args.quant is not None else None
        gemm = _coerce_gemm(args.gemm)
    except ValueError as e:
        ap.error(str(e))
    scales = None
    if args.scales_json:
        with open(args.scales_json) as f:
            scales = {k: float(v) for k, v in json.load(f).items()}
    samples = None
    if args.samples_npz:
        import numpy as np

        samples = dict(np.load(args.samples_npz))
    from . import surgeon as _s

    graph = _s.load(args.input)
    for site in discover_attention_sites(
            graph, base=base, quant=quant, scales=scales, samples=samples,
            arch=args.arch, gemm=gemm, min_cos=args.min_cos):
        print(f"tier{site['tier']} {site['attn']} ({site['attn_op']})"
              f" -> {site['op']} {site['fields']} cos={site['cos']}"
              f" flags={site['trt_flags']['builder']} | {site['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
