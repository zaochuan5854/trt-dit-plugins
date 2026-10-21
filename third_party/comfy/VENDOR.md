# Vendored comfy-kitchen sources

Source: https://github.com/Comfy-Org/comfy-kitchen
License: Apache-2.0 (`LICENSE` in this directory)

Two upstream snapshots coexist because they were vendored at different
times. Both are byte-identical to upstream except for the patches listed
below; the v0.2.33 snapshot carries a one-line vendoring notice on top of
each file, and every `convrot/` change is marked `DIT-PLUGINS-PATCH` in
place.

| directory | upstream | pin |
|---|---|---|
| `common/`, `ops/`, `sage_attention/` | `backends/cuda/{utils,dtype_dispatch,float_utils}.cuh`, `backends/cuda/ops/`, `backends/cuda/sage_attention/` | **v0.2.33** `e9ea99cf2f0af1d0c49c04690d4153a91c2b8668` |
| `convrot/` | `backends/cuda/ops/` + `backends/cuda/{utils,dtype_dispatch,float_utils}.cuh`, `input_act_codes.h` | **v0.2.39** `e40e76d9c544b3f74c3b9ea923f7bc3d45a96372` |

`common/` holds the v0.2.33 headers shared by `ops/` and `sage_attention/`;
upstream keeps one copy per directory, and the copies are byte-identical.

CUTLASS: `third_party/cutlass`, pinned submodule commit
`d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`.

## File map

| directory | files |
|---|---|
| `common/` | `utils.cuh`, `dtype_dispatch.cuh`, `float_utils.cuh` (v0.2.33) |
| `ops/` | `adaln.cu`, `rms_rope.cu`, `rope_device.cuh`, `per_tensor_quantize.cu` (v0.2.33) |
| `sage_attention/` | `quant_qk_int8.cu`, `quant_v_int8.cu`, `sage_attn_launcher.cu` + supporting `.cuh` (v0.2.33) |
| `convrot/` | `int8_linear.cu`, `cutlass_gemm_int8.cu`, `convrot_w4a4.cu`, `w4a8_gemm.cu` + supporting `.cuh`/`.h` (v0.2.39) |

`sol_layout.cuh` was dropped: no translation unit includes it.

## Patch list

Every change is marked `DIT-PLUGINS-PATCH` in place. Patches exist only to
fit a TensorRT plugin: no cuBLAS dependency, no autotune inside `enqueue`,
bias and all output dtypes on the fixed-config entry point.

`ops/adaln.cu`
- Added optional affine weight, residual+gate, and a separate modulation
  dtype for the TRT wrapper (`DIT-PLUGINS-PATCH`). Null/false arguments keep
  the upstream path unchanged.

`convrot/cutlass_gemm_int8.cu`
- `launch_cutlass_int8_dequant_config` takes `bias` and supports FP32 / FP16 /
  BF16 outputs (upstream was FP16/BF16 without bias).

`convrot/convrot_w4a4.cu`
- cuBLAS fallback removed; the chunked W4A8 GEMM uses the strided CUTLASS
  INT8 GEMM for every shape (upstream gated it to M >= 1024 and called cuBLAS
  otherwise).
- Runtime autotune (`std::mutex`, `thread_local last_*`, warmup + timing)
  removed from the non-config int4 dispatch; a deterministic tile pick is
  used instead. The same pick backs `gemm_config = -1`: the plan's 1/5/6/3
  tiles when `GemmKernel::SharedStorageSize` fits the device's opt-in
  shared-memory budget (Hopper/Blackwell, 227 KiB), otherwise the 0/5 tiles
  that fit sm80/86/89 (99-163 KiB). No launch is attempted to choose.
- Added `launch_cutlass_int4_dequant_config` and `convrot_int4_autotune`
  (build-time only, called from Python via `_lib.get_autotune`). The tuner
  skips a config whose launch fails (too much shared memory) instead of
  timing a rejected launch.

No kernel math or quantization code was changed, so results match upstream
for the shapes upstream routes to the same kernel.
