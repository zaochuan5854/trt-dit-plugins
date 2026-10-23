<h1 align="center">TensorRT-DiT-Plugins</h1>

<p align="center">Existing DiT kernels, pluginized as TensorRT IPluginV3.</p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a></p>

## Highlights

- `pip install` only: no system CUDA or TensorRT. Wheels bundle the two plugin
  libraries; `torch`, `tensorrt-cu13-libs` and `nvidia-cuda-runtime` come
  from pip.
- 12 plugins under the `dit-plugins` namespace: INT8 attention, plain dense
  SageAttention, fused INT8 RoPE+SageAttention DiT-self, RMSNorm/LayerNorm
  AdaLN, single-tensor RMSNorm, RoPE variants, FP8 stochastic rounding,
  block-sparse SageAttention2 (sm89), ConvRot W8A8/W4A4 linear.
- 0.7.0: `int8_attention`, `sage_attn`, `fused_int8_rope_sage_attn`, and
  `block_sparse_sage2_attn` can write attention output directly as
  `[B,S,C]` with `out_layout="bsc"`, removing the trailing Transpose.
- 0.8.0: ConvRot `residual`/`gate` takes any rank (the residual's
  shape becomes the output, so a rank-5 `[1,1,H,W,C]` stream fuses without a
  reshape), the identity epilogue no longer runs a redundant `[M,N]`
  read+write pass (1.18 ms/layer at 8192x2048, M=16384), the residual+gate
  pass uses 128-bit vector accesses (70.7 → 49.1 µs per 4096x2048 bf16 site),
  and `validate_plugin_nodes` resolves BF16's raw ONNX enum (no false
  positives on reloaded graphs).
- 0.8.1: ConvRot W4A4 supports `epilogue`, `residual`/`gate`, `out_layout`,
  and QKV heads (activation/residual runs as a post-pass after the GEMM);
  the smooth-K K-mean reduction is deterministic (no `atomicAdd`), and
  `libck_kernels.so` is built with `-Bsymbolic` + `-fno-gnu-unique` so it no
  longer shares CUTLASS's thread_local state with comfy-kitchen.
- Kernels from ComfyKitchen and SpargeAttn (see NOTICE and Kernel sources);
  handwritten TRT glue. Not affiliated with Comfy Org; ComfyKitchen
  compatibility is not a goal.

## Architecture

```text
┌──────────────┐  pip install   ┌───────────────────────┐
│    torch     │◀──────────────▶│    trt_dit_plugins    │
│ tensorrt-cu13│  extra-index   │  ┌─────────────────┐  │
│ -libs, cudart│  pypi.nvidia   │  │ libck_kernels   │  │
└──────────────┘   .com         │  │ (CUDA kernels)  │  │
                                │  ├─────────────────┤  │
                                │  │libtrt_dit_plugins│  │
                                │  │ (IPluginV3 glue)│  │
                                │  └────────┬────────┘  │
                                └───────────┼───────────┘
                                            │ registers
                                            ▼
                                ┌───────────────────────┐
                                │ TRT plugin registry   │
                                │     dit-plugins::*     │──▶ engines
                                └───────────────────────┘
```

## Usage Example

```python
import torch, trt_dit_plugins as tdp

q = torch.randn(1, 16, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
out = tdp.int8_attention(q, k, v)          # engines are built once, then cached
out = tdp.sage_attn(q, k, v, out_layout="bsc")  # [B,S,C], no trailing Transpose

out = tdp.rms_adaln(x, scale, shift)       # eps=1e-6 by default
qo, ko = tdp.apply_rope(q, k, freqs)
out = tdp.stochastic_rounding_fp8(x, rng)  # rng: int32, same numel
# fused RoPE+Sage from a bf16 linear output (no QuantizeLinear needed)
out = tdp.fused_rope_sage_attn(q, k, v, rms_w_q, rms_w_k, inv_freq)
```

## Quick Install

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.8.1-cu13/trt_dit_plugins-0.8.1-py3-none-linux_x86_64.whl   # Linux
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.8.1-cu13/trt_dit_plugins-0.8.1-py3-none-win_amd64.whl      # Windows
```

## Quick Start

```bash
python -c "import trt_dit_plugins as t; t.ensure_loaded(); print(t.__version__, sorted(t.PLUGINS))"
```

`PLUGINS` is filled from the TensorRT registry by `ensure_loaded()`; before
that call it is empty by design.

### Migrating from 0.4.x

The old 2D `adaln` op (`PluginOp.ADALN`, `surgeon.add_adaln`,
`--fuse-adaln`) was removed in 0.5.0 and stays removed. Use `rms_adaln` or,
since 0.6.0, `layer_norm_adaln` (`fuse_norm_affine(..., plugin=...)`) with 3D
input and optional `weight` / `residual` / `gate`.

## Plugins

| TRT name (`dit-plugins::*`) | Op | I/O dtypes | Status |
|---|---|---|---|
| `int8_attention` | INT8 Q/K/V attention (SageAttention), D ∈ {64,128,256}; `out_layout="bsc"` since 0.7.0 | in f32/f16/bf16 → out f16/bf16 (f32→bf16), or `[B,S,C]` | cos=0.99990, 0.910 ms/layer @ S=4096,H=16,D=128 (RTX 4070 Ti) |
| `sage_attn` | Plain dense SageAttention, GQA ok, D ∈ {64,128,256} (new in 0.4.0); `out_layout="bsc"` since 0.7.0 | in f32/f16/bf16 (FP8 inputs rejected) → out f16/bf16 (f32→bf16), or `[B,S,C]` | sm89+D∈{64,128}+S%128==0: FP8-PV Sage2 tactic (cos=0.99925, opt-in via `fp8_pv=1`); else portable FP16-PV (cos=0.99927) |
| `rms_adaln` | Fused RMSNorm AdaLN; 2D/3D x, broadcast modulation, optional weight / residual+gate | f32/f16/bf16 in+out (uniform); modulation one dtype | cos=1.0 |
| `layer_norm_adaln` | Fused LayerNorm AdaLN (new in 0.6.0), same I/O and residual+gate as `rms_adaln` | f32/f16/bf16 in+out (uniform); modulation one dtype | cos=1.0 |
| `rms_norm` | Single-tensor RMSNorm; `layout="bshd"` also emits the BHSD view (new in 0.6.0) | x f16/bf16 → same; weight f32/f16/bf16 | cos=1.0 |
| `apply_rope` | Strided (copy-free) RoPE, D even | f16/bf16 in+out; freqs f32/f16/bf16 | cos=1.0 |
| `rms_rope_split_half` | Fused RMSNorm + split-half/partial RoPE, D multiple of 32; dynamic B/S, `rot_dim=-1` = RMS-only (new in 0.6.0) | f16/bf16 in+out; freqs/scales f32/f16/bf16 | cos=1.0 |
| `stochastic_round_fp8` | Stochastic rounding to FP8 E4M3 | in f32/f16/bf16, rng INT32 → out FP8 | exact match |
| `block_sparse_sage2_attn` | Block-sparse SageAttention2, sm89 only (kernel: SpargeAttn); `out_layout="bsc"` since 0.7.0 | q/k/v f16/bf16 [B,H,S,D], mask INT32 [B,H,S//128,S//64] → `[B,S,C]` with BSC | cos=0.99953 |
| `fused_int8_rope_sage_attn` | Fused RoPE+SageAttention DiT-self, D=128 (new in 0.4.0); `out_layout="bsc"` since 0.7.0 | in INT8 [B,H,S,128] + FP32 scales, or f32/f16/bf16 [B,H,S,128] (scales ignored; `fused_rope_sage_attn`) + norms → out BF16, or `[B,S,C]` | cos=0.99855 vs fp32 eager (matched rotation), 1.578 ms/layer @ S=4096,H=16 (1.29x vs separation; 1.12x @ S=9216, RTX 4070 Ti) |
| `convrot_linear` | ConvRot W8A8 dynamic-quant linear (INT8 weight, online Hadamard rotation); 0.6.0 adds GELU/SiLU + residual+gate epilogue and BHSD/QKV output; 0.8.0 accepts any-rank residual/gate | x f16/bf16/f32, weight INT8 [N,K], scale FP32 [N], bias [N] → f16/bf16/f32 or `[B,H,S,D]` ×3 | bit-exact vs comfy-kitchen (max_abs=0); epilogue/layout covered by GPU tests |
| `convrot_linear_w4a4` | ConvRot W4A4 linear (packed signed INT4 weight [N,K/2], INT4 activation); 0.8.1 adds epilogue, residual+gate, BHSD/QKV output (post-pass) | x f16/bf16/f32, weight INT8 packed [N,K/2], scale FP32 [N], bias [N] → f16/bf16/f32 or `[B,H,S,D]` ×3 | bit-exact vs comfy-kitchen (max_abs=0); post-pass/layout covered by GPU tests |

All plugin I/O is `kLINEAR` only. Q/K/V/scale inputs to one plugin must share
one dtype. `UINT8` plugin I/O is unsupported by TRT 11.

Fused constraints: q/k/v identical INT8 `[B,H,S,128]`;
per-tensor FP32 scales (single element); RMSNorm weights FP32/BF16 `[128]`
(matched dtype); `inv_freq` FP32 `[64]`; DiT-self only (no mask/causal).
S≤1024 runs the same BLKK=128 quant layout with `cta_k=128` (slower than the
S>1024 dispatch but correct): cos 0.9968 / 0.9984 vs fp32 eager at S=512 /
1024 (RTX 4070 Ti).

## Kernel sources

| Plugin | Kernel source |
|---|---|
| `int8_attention` | [comfy-kitchen `backends/cuda/sage_attention/`](https://github.com/Comfy-Org/comfy-kitchen/tree/main/comfy_kitchen/backends/cuda/sage_attention) (origin: [SageAttention](https://github.com/thu-ml/SageAttention)) |
| `rms_adaln` / `layer_norm_adaln` | [comfy-kitchen `backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/adaln.cu) |
| `rms_norm` | Own `src/wrapper/rms_norm_wrapper.cpp` |
| `apply_rope` / `rms_rope_split_half` | [comfy-kitchen `backends/cuda/ops/rms_rope.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/rms_rope.cu) |
| `stochastic_round_fp8` | [comfy-kitchen `backends/cuda/ops/per_tensor_quantize.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/per_tensor_quantize.cu) |
| `block_sparse_sage2_attn` | [SpargeAttn `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh) (+ [`csrc/fused/fused.cu`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/fused/fused.cu)) |
| `sage_attn` | Vendored `sage_attention` kernels above (+ sparge sm89 kernels for the `fp8_pv` path) |
| `fused_int8_rope_sage_attn` | Own `src/kernel/fused_qk_rope_requant.cu` on the vendored `sage_attention` helpers |
| `convrot_linear` / `convrot_linear_w4a4` | Vendored [comfy-kitchen `backends/cuda/ops/convrot_w4a4.cu` + `int8_linear.cu`](third_party/comfy/convrot) (0.2.39, patched; see `third_party/comfy/VENDOR.md`) on CUTLASS INT8/INT4 GEMM (submodule `third_party/cutlass`) |

## Requirements

- NVIDIA GPU `sm80`–`sm120` (sm89 verified, RTX 4070 Ti), driver R560+, Python ≥ 3.10.
- glibc ≥ 2.39 for the published Linux wheels (built on Ubuntu 24.04).
- No system CUDA or TensorRT for pip installs.

## GPU support

| Arch | Status |
|---|---|
| sm89 (RTX 40 series) | Verified — all plugins green |
| sm80 / sm86 / sm90 / sm100 / sm120 | Builds ship native SASS but are **unverified on hardware** |

We only have sm89 hardware locally. If you hit issues on other architectures,
PRs (with GPU model + repro) are actively welcome.

## Use in your own engine build

```python
import trt_dit_plugins as tdp
tdp.ensure_loaded()  # registers dit-plugins plugins into the TRT registry
import tensorrt as trt  # needs the full package: pip install tensorrt-cu13==11.3.0.99
creator = trt.get_plugin_registry().get_creator("int8_attention", "1", "dit-plugins")
plug = creator.create_plugin("attn0", fc, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([q, k, v], [], plug)
```

Build engines **strongly typed** (`kSTRONGLY_TYPED`); weakly-typed builds may
default outputs to FP32 with a silent cast.

## ConvRot quantized linear (W8A8 / W4A4)

Weights are rotated offline with the regular Hadamard (256-wide groups) and
quantized symmetric per output channel; the plugin rotates and quantizes the
activation on the fly. Rotation cancels, so the result approximates
`x @ W.T` — with 8-bit activations/weights for `convrot_linear`, and 4-bit
both for `convrot_linear_w4a4` (activation quantization is still the
plugin's own; weights are packed signed INT4, low nibble = even column).

```python
import trt_dit_plugins as tdp
from trt_dit_plugins import autotune

# w: rotated + quantized offline (torch), scale per row [N]
y = tdp.convrot_linear(x, w_i8, w_scale, bias)            # [M,K] @ [N,K].T
cfg = autotune.autotune_convrot_w4a4(M, N, K, "bfloat16")  # build-time, cached
y = tdp.convrot_linear_w4a4(x, w_i4_packed, w_scale, bias, gemm_config=cfg)
# one engine across token counts (M dynamic; K is fixed)
y = tdp.convrot_linear(x, w_i8, w_scale, bias,
                       profile={"x": ((1, K), (M, K), (M_max, K))})
# 0.6.0: fused epilogue / residual+gate, or QKV in head-major layout;
# 0.8.0: the residual keeps its shape (e.g. [1,1,H,W,C]) and gate broadcasts
y = tdp.convrot_linear(x, w_i8, w_scale, bias, epilogue="gelu",
                       residual=res, gate=mod)
q, k, v = tdp.convrot_linear(x, w_qkv, w_qkv_scale, bias_qkv,
                             out_layout="bhsd", qkv_heads=16, head_dim=128)
```

Constraints (rejected at build time): `K % 256 == 0`, `K <= 16384`,
`N % 4 == 0` for W8A8, bias `[N]` in the output dtype, dtypes f16/bf16/f32.
W4A4 picks a fixed CUTLASS tile from `gemm_config`; on `sm80/86/89` only the
tiles that fit the 99 KiB shared-memory budget exist, so measure with
`autotune` rather than hand-picking (a bad index falls back, it never
corrupts the result). On non-sm8x GPUs W4A4 degrades to W4A8 (INT8
activation) inside the same plugin. W4A4 also accepts `epilogue`,
`residual`/`gate`, `out_layout="bhsd"`, and QKV heads since 0.8.1; unlike
W8A8, its activation/residual/gate is a post-pass, so use W8A8 where that
pass is on the hot path.

`autotune_convrot_w4a4` covers the CUTLASS shapes (`N % 32 == 0`) and
returns `-1` otherwise; pass that to `gemm_config` and the legacy fallback
runs. Autotune results are cached in
`~/.cache/trt_dit_plugins/convrot_w4a4.json` keyed by
`(M, N, K, out_dtype, arch)`; override with `TRT_DIT_AUTOTUNE_CACHE`.
`enqueue` never synchronizes or measures.

## ONNX surgery

Full-graph builds via ONNX: rewrite subgraphs to `dit-plugins` nodes, then
build with the Python builder API (`tdp.ensure_loaded()` + `add_plugin_v3`)
or `trtexec --onnx model.trt.onnx --plugins libtrt_dit_plugins.so`
(`trtexec` ships with the TRT apt/deb packages, not the pip wheels).

```bash
pip install --extra-index-url https://pypi.nvidia.com "trt-dit-plugins[onnx]"
python -m trt_dit_plugins.surgeon model.onnx model.trt.onnx \
  --fuse-rms-adaln ln_out,scale,shift,blk_out \
  --fuse-dit-self-attn q_i8,q_scale,k_i8,k_scale,v_i8,v_scale,rms_w_q,rms_w_k,inv_freq,attn_out \
  --retarget MyCustomAttn:int8_attention
```

Python API: `add_int8_attention` / `add_sage_attn` /
`add_rms_adaln` / `add_apply_rope` / `add_rms_rope_split_half` /
`add_stochastic_round_fp8` / `add_block_sparse_sage2_attn` /
`add_dit_self_fused_attn` / `add_convrot_linear` / `add_rms_norm` /
`add_layer_norm_adaln`, plus explicit-anchor `fuse_norm_affine`,
`fuse_rope_blocks` (`qk_layout="bhsd"` skips transposes; `rot_dim=-1` +
`include_cross=True` inserts the two cross-attn RMSNorms),
`fuse_dit_self_attn_block`, `retarget_nodes`, `import_comfy_convrot` (rewrites
exporter-emitted `comfy_kitchen::int8_linear` ConvRot nodes, fail-open), and GPU-free
`validate_plugin_nodes` (`python/trt_dit_plugins/surgeon.py`; no auto
pattern-matching — exporter decompositions differ too much to guess safely,
so fusion takes tensor names).

RPN (FireQ-style RoPE-preserving normalization): `compute_rpn_scales` +
`fold_rpn_into_norms` in `python/trt_dit_plugins/rpn.py` fold per-pair
scales into the Q/K RMSNorm weights (attention scores invariant, no kernel
change). Calibrate on real K activations; see the module docstring.

## Build from source

Developers only (modifying kernels/wrappers). Needs Linux + CUDA 13 toolkit
+ TRT 11; see `CONTRIBUTING.md` and `.github/workflows/`.

## Troubleshooting

- `creator not found`: reinstall with `--extra-index-url https://pypi.nvidia.com`
  (missing `tensorrt-cu13-libs` / `nvidia-cuda-runtime`).
- `cos ≈ 0` after wiring: check the engine output dtype — an FP32 default with
  an implicit cast looks like correct-but-shifted output.
- Windows: the VC++ Redistributable is already required by the
  `torch`/`tensorrt-cu13-libs` wheels.

## License

Apache-2.0 (`LICENSE`). Kernel attributions in `NOTICE`.
