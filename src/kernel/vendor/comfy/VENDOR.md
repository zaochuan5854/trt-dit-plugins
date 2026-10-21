# Vendored ConvRot kernels (comfy-kitchen)

Source: https://github.com/Comfy-Org/comfy-kitchen
Tag: `v0.2.39` (commit `e40e76d9c544b3f74c3b9ea923f7bc3d45a96372`)
License: Apache-2.0 (`LICENSE` in this directory)

CUTLASS: `third_party/cutlass`, pinned submodule commit
`d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`.

## Files

| file | state |
|---|---|
| `utils.cuh`, `dtype_dispatch.cuh`, `float_utils.cuh`, `input_act_codes.h` | unmodified |
| `ops/int8_linear.cu`, `ops/w4a8_gemm.cu`, `ops/svdquant_utils.cuh`, `ops/cutlass_gemm_common.cuh` | unmodified |
| `ops/cutlass_gemm_int8.cu` | patched |
| `ops/convrot_w4a4.cu` | patched |

Every change is marked `DIT-PLUGINS-PATCH` in place. Patches exist only to
fit a TensorRT plugin: no cuBLAS dependency, no autotune inside `enqueue`,
bias and all output dtypes on the fixed-config entry point.

## Patch list

`ops/cutlass_gemm_int8.cu`
- `launch_cutlass_int8_dequant_config` takes `bias` and supports FP32 / FP16 /
  BF16 outputs (upstream was FP16/BF16 without bias).

`ops/convrot_w4a4.cu`
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
