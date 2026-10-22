# Fusion / layout extensions (0.7.0)

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

- `epilogue`: `0` none, `1` exact GELU (erf), `2` SiLU. Applied in the
  dequant epilogue, no extra pass.
- `residual_gate`: `0` off, `1` on -> `out = residual + gate * y`. `B` is
  looped per batch in the fused path; `M<3` falls back to gemv + a post-pass.
- `out_layout="bsc"` (default) or `"bhsd"`; `qkv_heads=H` with
  `head_dim=D` returns three `[B,H,S,D]` tensors (q, k, v).
- W4A4 rejects `epilogue`/`residual`/`out_layout`/QKV fields in 0.6.0.
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

Known pre-existing limit: `fused_int8_rope_sage_attn` uses float
`atomicAdd` in its K-mean prepass, so repeated runs can differ by one bf16
ULP at B>=2/S>=512. This is identical in BHSD and BSC and is not caused by
the layout epilogue; its BSC/BHSD cosine is >0.999999. The other three
attention ops are bit-stable.

## Deferred

None for the 0.7.0 layout contract. W4A4 still rejects
`epilogue`/`residual`/`out_layout`/QKV fields (0.6.0 behavior).

## Release gate

- Op-level tests: pass on the RTX 4070 Ti (sm89).
- External Anima engine: `cos` at or above the existing gate, layer count
  < 1000, E2E <= 220 ms, no `Transpose` in the rope/QKV region. Until that
  is measured, 0.7.0 is "op-level verified, engine gate pending".
