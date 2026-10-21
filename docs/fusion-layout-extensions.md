# Fusion / layout extensions (0.6.0)

Contract for the ops added for the Anima 2.9B ConvRot request
(`trt-dit-plugins-fusion-request.md`). The Reforge engine comparison is the
external handoff gate; this repo verifies the op-level contracts only.

## Dynamic RMS+RoPE

- `rot_dim=0` = rotate all `D` (unchanged); `rot_dim=-1` = RMS-only
  (cross-attention).
- Dynamic `B` and `S` are accepted: the wrapper takes them from the input
  shape and `getOutputDimensions` mirrors Q/K.
- `D` is fixed at 128 for the fused path. `freqs` keeps the `[1,1,S,r,2,2]`
  layout (fp32/f16/bf16); its `S` must equal the Q/K `S` when `rot_dim > 0`
  (mismatch is rejected in shape resolution/enqueue).
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

## Deferred

Attention epilogue (`[B,H,S,D]` -> `[B,S,C]` store inside the attention
plugin, request 4.4) is not implemented. The engine can keep one
`Transpose` after attention; the rope/QKV region has none.

## Release gate

- Op-level tests: pass on the RTX 4070 Ti (sm89).
- External Anima engine: `cos` at or above the existing gate, layer count
  < 1000, E2E <= 220 ms, no `Transpose` in the rope/QKV region. Until that
  is measured, 0.6.0 is "op-level verified, engine gate pending".
