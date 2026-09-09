# SPDX-License-Identifier: Apache-2.0
# TensorRT-DiT-Plugins

TensorRT `IPluginV3` wrappers around [ComfyKitchen](https://github.com/Comfy-Org/comfy-kitchen)
CUDA kernels, with comfy-kitchen-compatible op names under the `comfy_kitchen`
plugin namespace. Target: NVIDIA Ada Lovelace (`sm89`), TensorRT 10.16, CUDA 13.2.

GPU architecture support (`CMAKE_CUDA_ARCHITECTURES`,
default `80-real 86-real 89-real 100-virtual 120-virtual`):

| Arch | GPUs | Status |
|---|---|---|
| sm89 | RTX 40 series (e.g. 4070 Ti) | Verified (all E2E green) |
| sm80/sm86 | A100 / RTX 30 series | Builds, **unverified on hardware** |
| sm100/sm120 | Blackwell (B100/200, RTX 50) | PTX (driver JIT), **unverified on hardware** |

Unverified targets compile but have never run here — reports welcome.

Version pins (strict — other combinations are untested): TensorRT
`10.16.1.11-1+cuda13.2`, CUDA toolkit 13.2.x, `CMAKE_CUDA_ARCHITECTURES=89`
(sm89 only; other architectures are not built).

The kernels are vendored unmodified (plus a one-line provenance notice) under
`src/kernel/`; the handwritten TRT glue lives in `src/wrapper/`. See `NOTICE`
for rights and attributions.

## Requirements

- NVIDIA GPU with `sm89` (e.g. RTX 4070 Ti, verified 2026-09) + driver ≥ 580
- Linux with `podman` (rootless, `nvidia-container-toolkit` CDI) and ~30 GB free
- Python path needs `torch` + `tensorrt==10.16.*` only for `python/` frontend tests

## Plugins

| TRT name (`comfy_kitchen::*`) | Op | I/O dtypes | Status |
|---|---|---|---|
| `int8_attention` | INT8 Q/K/V attention (SageAttention), D ∈ {64,128,256} | in f32/f16/bf16 → out f16/bf16 (f32→bf16) | cos=0.99990, 0.910 ms/layer @ S=4096,H=16,D=128 (RTX 4070 Ti) |
| `adaln` / `rms_adaln` | Fused LayerNorm/RMSNorm AdaLN | f32/f16/bf16 in+out (uniform) | cos=1.0 |
| `apply_rope` | Strided (copy-free) RoPE, D even | f16/bf16 in+out; freqs f32/f16/bf16 | cos=1.0 |
| `rms_rope_split_half` | Fused RMSNorm + split-half/partial RoPE, D multiple of 32 | f16/bf16 in+out; freqs/scales f32/f16/bf16 | cos=1.0 |
| `stochastic_round_fp8` | Stochastic rounding to FP8 E4M3 | in f32/f16/bf16, rng INT32 → out FP8 | exact match |

All plugin I/O is `kLINEAR` only. Q/K/V/scale inputs to one plugin must share
one dtype. `UINT8` plugin I/O is unsupported by TRT 10.16.

Not yet: `FusedGeGLU/SwiGLU` (no upstream kernel exists in comfy-kitchen).

Supported dtypes: inputs FP32/FP16/BF16 (attention outputs FP16/BF16;
RoPE I/O FP16/BF16; see `plan.md`). Dynamic sequence lengths via one engine
(profiled min/opt/max). No `torch` dependency in the core libraries.

## Quickstart (Linux, sm89)

```bash
# 1. Build the image (uses cached CUDA devel base, pins TRT 10.16.1.11)
podman build --rm -t trt-plugins:sm89 .

# 2. Build the libraries (all cores + ccache)
podman run --rm --security-opt=label=disable \
  -v $PWD:/workspace -v /tmp/ck-ccache:/tmp/ccache \
  localhost/trt-plugins:sm89 bash -c \
  "cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
   cmake --build /workspace/build --parallel \$(nproc)"
# -> build/libck_kernels.so  (vendored kernels)
# -> build/libtrt_dit_plugins.so  (wrappers, ~90 KB)

# 3. Python frontend (needs torch + tensorrt>=10.16 in the env)
PYTHONPATH=python TRT_DIT_LIBDIR=build python3 python/tests/test_smoke.py
```

```python
import torch, trt_dit_plugins as tdp

q = torch.randn(1, 16, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
out = tdp.int8_attention(q, k, v)          # engines are built once, then cached

out = tdp.rms_adaln(x, scale, shift)       # eps=1e-6 by default
qo, ko = tdp.apply_rope(q, k, freqs)
out = tdp.stochastic_rounding_fp8(x, rng)  # rng: int32, same numel
```

How it works: each call builds one single-plugin engine (strongly typed) on
first use, caches it by (op, shapes, fields), then enqueues on torch's current
stream. No extra synchronisation, no copies beyond what the kernels need.

Installable wheel layout lives in `python/` (`pyproject.toml`, `pack_libs.sh`
copies the two `.so` files into the package). Windows wheels (MSVC + CUDA +
TRT) are future work; the Python loader already handles DLL directories.

## Layout

```
src/wrapper/      handwritten IPluginV3 glue (one file per plugin)
src/kernel/       vendored ComfyKitchen CUDA sources (see NOTICE)
test/             C++ engine-level tests (cos/bench, one per plugin)
python/           pure-Python package (no compiled extension needed)
Dockerfile        dev image (TRT 10.16.1.11 pinned, apt-cached)
Dockerfile.val    validation image (+ torch + TRT python bindings)
```

## Implementation notes

- Engines should be built **strongly typed** (`kSTRONGLY_TYPED`); weakly-typed
  builds may default the output to FP32 and silently insert a cast.
- `REGISTER_TENSORRT_PLUGIN` registers under namespace `""`; the wrappers
  register manually under `comfy_kitchen` so serialization round-trips.
- TRT 10.16 has no UINT8 plugin I/O: byte buffers (e.g. RNG state) travel as INT32.
- `stochastic_round_fp8` supports zero-copy via `getAliasedInput` behind the
  `alias_rng` field (engine must enable `kALIASED_PLUGIN_IO_10_03`).

## Troubleshooting

- `cos ≈ 0` after wiring a plugin: check the engine output dtype
  (`getTensorDataType`) — an FP32 default with an implicit cast looks exactly
  like correct-but-shifted output.
- `UInt8 datatype formats are not supported` at build: use INT32 for byte buffers.
- `Aliased I/O ... but PreviewFeature not enabled`: set `alias_rng=0` (default)
  or enable `kALIASED_PLUGIN_IO_10_03` on the builder config.
- `creator not found` in Python: the `.so` wasn't loaded — check
  `TRT_DIT_LIBDIR` (defaults to `build/`) and load `libck_kernels` first.

## License

Apache-2.0 (`LICENSE`). ComfyKitchen-derived naming, interfaces and vendored
sources are attributed in `NOTICE` and per-file SPDX headers.

Issues and PRs are welcome; see `CONTRIBUTING.md`, and `plan.md` §8 for the
remaining roadmap (GeGLU blocker, ONNX replacement).
