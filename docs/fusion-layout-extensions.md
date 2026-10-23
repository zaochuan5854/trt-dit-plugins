# Fusion / layout extensions (0.8.1)

Contract for the ops added for the Anima 2.9B ConvRot request
(`trt-dit-plugins-fusion-request.md`). The Reforge engine comparison is the
external handoff gate; this repo verifies the op-level contracts only.

## Dynamic RMS+RoPE

- `rot_dim=0` = rotate all `D` (unchanged); `rot_dim=-1` = RMS-only
  (cross-attention).
- Dynamic `B` and `S` are accepted: the wrapper takes them from the input
  shape and `getOutputDimensions` mirrors Q/K.
- `D` must be a multiple of 32 (128 in Anima). `freqs` keeps the
  `[fb,f1,S,r,2,2]` layout (fp32/f16/bf16); each of its first three dims may
  be 1 or match Q/K, and `r` must equal `rot_dim/2` (mismatch is rejected in
  shape resolution/enqueue).
- Q/K are physical `[B,H,S,D]` (D contiguous). Use
  `fuse_rope_blocks(..., qk_layout="bhsd")` to skip the BSHD transposes.

## Cross-attention RMSNorm

- Use `rms_norm(x, weight, layout="bshd")` twice (or
  `fuse_rope_blocks(..., rot_dim=-1, include_cross=True)` on a graph).
- Q and K may have different `S`; the pair op requires equal Q/K shapes.
- `layout="bshd"` reads `[B,S,H,D]` and writes `[B,H,S,D]`;
  `layout="bhsd"` is a pass-through layout.

## AdaLN

- Ops: `rms_adaln`, `layer_norm_adaln`.
- Inputs: `x, scale, shift[, weight][, residual, gate]`.
- 3D `x` `[B,S,C]` is accepted; `scale`/`shift`/`gate` may be `[B,C]` or
  `[B,1,C]` and are broadcast over `S`.
- `has_weight` and `residual_gate` are plugin fields; the wrappers expose
  `weight=` / `residual=` / `gate=`. `out = residual + gate * y`.
- `scale`/`shift`/`weight`/`gate` share one modulation dtype (fp32 or bf16);
  `x`/`out`/`residual` share another.

## ConvRot

- `epilogue`: `0` none, `1` exact GELU (erf), `2` SiLU. W8A8 applies it in
  the dequant epilogue (no extra pass); W4A4 (0.8.1) applies it as a
  post-pass after the GEMM.
- `residual_gate`: `0` off, `1` on -> `out = residual + gate * y`. The
  residual keeps any rank with `N` columns and x's row count; its shape
  becomes the output, so a rank-5 `[1,1,H,W,C]` stream fuses without a
  reshape. The gate broadcasts over rows (`[.., 1, N]`) or repeats per batch
  (`[B, N]` / `[B, 1, N]`, sliced per batch). `M<3` falls back to gemv + a
  post-pass.
- Identity epilogue (`epilogue=0` with no residual/gate) does not launch the
  post-pass: the GEMM already stored the final value (`0.8.0`; the
  redundant `[M,N]` read+write cost 1.18 ms/layer at 8192x2048, M=16384).
- `out_layout="bsc"` (default) or `"bhsd"`; `qkv_heads=H` with
  `head_dim=D` returns three `[B,H,S,D]` tensors (q, k, v).
- W4A4 supports `epilogue`, `residual`/`gate`, `out_layout`, and QKV fields
  since 0.8.1. Its activation/residual/gate is a post-pass; `out_layout`
  runs a flat GEMM into a temporary followed by one BSC -> BHSD/QKV pass.
  `out_layout="bhsd"` still requires no epilogue/residual (use BSC for
  those, then reshape/transpose in the graph or a separate layout op).
- All new fields default to their legacy values, so existing serialized
  engines keep working (`PLUGIN_VERSION` stays `"1"`).

## Attention output layout

- `int8_attention`, `sage_attn`, `fused_int8_rope_sage_attn`, and
  `block_sparse_sage2_attn` accept `out_layout="bhsd"` (default) or
  `"bsc"` (`out_layout=1`).
- `"bsc"` returns `[B,S,C]` (`C=H*D`) directly from the attention store;
  no `Transpose` node is needed after attention. `"bhsd"` keeps the legacy
  `[B,H,S,D]` shape.
- `D` remains the contiguous dimension in both layouts. The store address
  is `b*S*C + s*C + h*D + d` for BSC vs
  `b*H*S*D + h*S*D + s*D + d` for BHSD, so vectorized stores are unchanged.
- This is a runtime-stride epilogue (the kernel already takes batch/seq/head
  output strides), not a second kernel or a layout pass.
- `fp8_pv=1` and block-sparse also honor `out_layout="bsc"` through the
  Sparge launcher.
- Op-level RTX 4070 Ti check (B=1,H=16,S=4096,D=128): `int8_attention`
  BHSD/BSC = 1.68/1.69 ms; `sage_attn` = 1.69/1.69 ms (run-to-run noise).
- Layer-level `_gemm_mha_v2` gate: the native Anima 2.9B engine
  (1024x1024, context 512, `IProfiler`) has 80 Attention layers: 40
  self-attention at ~1.82 ms plus 40 cross-attention at ~0.235 ms,
  total **82–85 ms** (ceiling 87 ms). A single-plugin `sage_attn`
  engine with a preallocated BSC output measures **0.918 ms** at
  S=4096 (self) and **0.163 ms** at Sq=4096/Sk=512 (cross), so 80
  layers = **43.2 ms**; replacing self-attention only is **46.1 ms**.
  These are layer-level proxies; the official full-engine gate still
  needs the Reforge graph with the plugin and the trailing Transpose
  removed.

## Measured op timings

RTX 4070 Ti, TensorRT 11.3 container. Op-level calls through the Python
wrapper (includes dispatch and output allocation), 20 iterations after 5
warmups. These are sanity numbers, not engine-layer times; the Anima E2E
target is measured in Reforge.

| Op | Shape | ms |
|---|---|---:|
| `rms_rope_split_half` | B=1,H=16,S=16384,D=128 | 1.04 |
| `layer_norm_adaln` | B=1,S=16384,C=2048 | 0.61 |
| `layer_norm_adaln` + residual/gate | same | 0.77 |
| `rms_adaln` | same | 0.62 |
| `rms_adaln` + residual/gate | same | 0.77 |
| `convrot_linear` epilogue none | M=16384,N=2048,K=2048 | 2.66 |
| `convrot_linear` GELU | same | 2.40 |
| `convrot_linear` QKV BHSD | M=16384,N=6144,K=2048 | 2.78 |
| minimal block (QKV+rope+attn+out) | B=1,S=4096,C=2048 | 3.99 |

The GELU row is within run-to-run noise of `none` because the epilogue is
memory-bound; it is not evidence of a regression.

## Verification

CPU:

```bash
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
PYTHONPATH=python python -m pytest python/tests/test_convrot.py -q
```

GPU (sm89, built libs):

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build python -m pytest python/tests/test_gpu.py -q
PYTHONPATH=python TRT_DIT_LIBDIR=build python -m pytest python/tests/test_fusion_e2e.py -q
```

`test_fusion_e2e.py` pins dynamic `B`/`S` (1x256, 1x4096, 2x4096), the
identity-rotation RMS parity, attention composition, determinism, and a
5-node graph (AdaLN -> QKV -> rope -> attention -> out) with no layout glue.
`test_gpu.py::test_attention_out_layout` checks that the BSC store is
bit-identical to the BHSD store followed by transpose+reshape for
`int8_attention`, `sage_attn`, `fp8_pv`, and block-sparse attention.

`fused_int8_rope_sage_attn`'s K-mean prepass is deterministic since 0.8.1:
one block per `(b,h)`, fixed 16-stripe fold order, no `atomicAdd` (the old
form could differ by one bf16 ULP between runs at B>=2/S>=512). BSC and BHSD
share the same reduction.

## Deferred

None for the 0.8.1 layout contract.

## 0.8.1 fixes

Plugin-side fixes from the ConvRot INT8 Anima integration review, no graph
contract change (`PLUGIN_VERSION` stays `"1"`, existing engines keep
loading):

- `libck_kernels.so` is linked with `-Bsymbolic` and compiled with
  `-fno-gnu-unique` (option `TRT_DIT_SYMBOLIC_BINDING`, default ON). CUTLASS's
  `STB_GNU_UNIQUE` thread_local statics are no longer unified with
  comfy-kitchen's copies, so running both in one process stays bit-exact.
  Verify with `readelf -sW libck_kernels.so | grep UNIQUE` (expect 0).
- The smooth-K K-mean prepass is deterministic: one block per `(b,h)`, 16
  row stripes, fixed fold order, no `atomicAdd`. Repeated fused-attention
  calls on identical input are bit-exact.
- W4A4 gains `epilogue`, `residual`/`gate`, `out_layout`/QKV heads. The
  activation/residual/gate runs as a post-pass; BHSD/QKV runs a flat GEMM
  into a temporary plus one layout pass. W8A8 keeps its fused epilogue.
- Tests cover W4A4 epilogue/residual parity, W4A4 BHSD/QKV (bit-exact against
  the BSC output), and fused-attention repeat determinism.

## 0.8.0 fixes

Plugin-side defects found by the Anima 2.9B engine run, no graph contract
change (`PLUGIN_VERSION` stays `"1"`, existing engines keep loading):

- ConvRot accepts rank >= 4 `x` (`M` = product of the leading dims) and an
  any-rank residual, so the MLP `[1,1,H,W,C]` stream no longer needs the
  exporter-side reshape.
- The identity epilogue post-pass is skipped (perf regression vs 0.5.1).
- The residual+gate post-pass reads/writes 128-bit vectors and hoists the
  gate broadcast to the vector index (70.7 -> 49.1 us per 4096x2048 bf16
  site; equal to the scalar kernel at bandwidth-saturated sizes such as
  8192x2048, M=16384, where the scalar form stays as the tail /
  `cols % VEC != 0` fallback).
- `surgeon.validate_plugin_nodes` resolves BF16's raw ONNX enum (`16`), which
  used to flag every BF16 convrot node after reloading a model.
- The residual+gate contract lives in one place per layer:
  `select.convrot_residual_gate_error` (Python/ONNX) mirrors
  `ConvRotLinear::checkResidualGate` (C++), and the surgeon declares the node
  output with the residual's shape.

## Release gate

- Op-level tests: pass on the RTX 4070 Ti (sm89).
- External Anima engine: `cos` at or above the existing gate, layer count
  < 1000, E2E <= 220 ms, no `Transpose` in the rope/QKV region. Until that
  is measured, 0.8.1 is "op-level verified, engine gate pending".
