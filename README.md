<h1 align="center">TensorRT-DiT-Plugins</h1>

<p align="center">Existing DiT kernels, pluginized as TensorRT IPluginV3.</p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a></p>

## Highlights

- `pip install` only: no system CUDA or TensorRT. Wheels bundle the two plugin
  libraries; `torch`, `tensorrt-cu12-libs` and `nvidia-cuda-runtime-cu12` come
  from pip.
- 7 plugins under the `dit-plugins` namespace: INT8 attention, AdaLN,
  RoPE variants, FP8 stochastic rounding, block-sparse SageAttention2 (sm89).
- Kernels from ComfyKitchen and SpargeAttn (see NOTICE and Kernel sources);
  handwritten TRT glue. Not affiliated with Comfy Org; ComfyKitchen
  compatibility is not a goal.

## Architecture

```text
┌──────────────┐  pip install   ┌───────────────────────┐
│    torch     │◀──────────────▶│    trt_dit_plugins    │
│ tensorrt-cu12│  extra-index   │  ┌─────────────────┐  │
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

out = tdp.rms_adaln(x, scale, shift)       # eps=1e-6 by default
qo, ko = tdp.apply_rope(q, k, freqs)
out = tdp.stochastic_rounding_fp8(x, rng)  # rng: int32, same numel
```

## Quick Install

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.1.0-cu12/trt_dit_plugins-0.1.0-py3-none-linux_x86_64.whl   # Linux
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.1.0-cu12/trt_dit_plugins-0.1.0-py3-none-win_amd64.whl      # Windows
```

## Quick Start

```bash
python -c "import trt_dit_plugins as t; print(t.__version__, sorted(t.PLUGINS))"
```

## Plugins

| TRT name (`dit-plugins::*`) | Op | I/O dtypes | Status |
|---|---|---|---|
| `int8_attention` | INT8 Q/K/V attention (SageAttention), D ∈ {64,128,256} | in f32/f16/bf16 → out f16/bf16 (f32→bf16) | cos=0.99990, 0.910 ms/layer @ S=4096,H=16,D=128 (RTX 4070 Ti) |
| `adaln` / `rms_adaln` | Fused LayerNorm/RMSNorm AdaLN | f32/f16/bf16 in+out (uniform) | cos=1.0 |
| `apply_rope` | Strided (copy-free) RoPE, D even | f16/bf16 in+out; freqs f32/f16/bf16 | cos=1.0 |
| `rms_rope_split_half` | Fused RMSNorm + split-half/partial RoPE, D multiple of 32 | f16/bf16 in+out; freqs/scales f32/f16/bf16 | cos=1.0 |
| `stochastic_round_fp8` | Stochastic rounding to FP8 E4M3 | in f32/f16/bf16, rng INT32 → out FP8 | exact match |
| `block_sparse_sage2_attn` | Block-sparse SageAttention2, sm89 only (kernel: SpargeAttn) | q/k/v f16/bf16 [B,H,S,D], mask INT32 [B,H,S//128,S//64] | cos=0.99953 (master only, not in v0.1.0-cu12 wheels) |

All plugin I/O is `kLINEAR` only. Q/K/V/scale inputs to one plugin must share
one dtype. `UINT8` plugin I/O is unsupported by TRT 10.16.

## Kernel sources

| Plugin | Kernel source |
|---|---|
| `int8_attention` | [comfy-kitchen `backends/cuda/sage_attention/`](https://github.com/Comfy-Org/comfy-kitchen/tree/main/comfy_kitchen/backends/cuda/sage_attention) (origin: [SageAttention](https://github.com/thu-ml/SageAttention)) |
| `adaln` / `rms_adaln` | [comfy-kitchen `backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/adaln.cu) |
| `apply_rope` / `rms_rope_split_half` | [comfy-kitchen `backends/cuda/ops/rms_rope.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/rms_rope.cu) |
| `stochastic_round_fp8` | [comfy-kitchen `backends/cuda/ops/per_tensor_quantize.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/per_tensor_quantize.cu) |
| `block_sparse_sage2_attn` | [SpargeAttn `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh) (+ [`csrc/fused/fused.cu`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/fused/fused.cu)) |

## Requirements

- NVIDIA GPU `sm80`–`sm90` (sm89 verified, RTX 4070 Ti), driver R560+, Python ≥ 3.10.
- No system CUDA or TensorRT for pip installs.

## GPU support

| Arch | Status |
|---|---|
| sm89 (RTX 40 series) | Verified — all plugins green |
| sm80 / sm86 / sm90 | Builds ship SASS but are **unverified on hardware** |

We only have sm89 hardware locally. If you hit issues on other architectures,
PRs (with GPU model + repro) are actively welcome.

## Use in your own engine build

```python
import trt_dit_plugins as tdp
tdp.ensure_loaded()  # registers dit-plugins plugins into the TRT registry
import tensorrt as trt  # needs the full package: pip install tensorrt-cu12==10.16.*
creator = trt.get_plugin_registry().get_creator("int8_attention", "1", "dit-plugins")
plug = creator.create_plugin("attn0", fc, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([q, k, v], [], plug)
```

Build engines **strongly typed** (`kSTRONGLY_TYPED`); weakly-typed builds may
default outputs to FP32 with a silent cast.

## Build from source

Developers only (modifying kernels/wrappers). Needs Linux + CUDA 12.6 toolkit
+ TRT 10.16; see `CONTRIBUTING.md` and `.github/workflows/`.

## Troubleshooting

- `creator not found`: reinstall with `--extra-index-url https://pypi.nvidia.com`
  (missing `tensorrt-cu12-libs` / `nvidia-cuda-runtime-cu12`).
- `cos ≈ 0` after wiring: check the engine output dtype — an FP32 default with
  an implicit cast looks like correct-but-shifted output.
- Windows: the VC++ Redistributable is already required by the
  `torch`/`tensorrt-cu12-libs` wheels.

## License

Apache-2.0 (`LICENSE`). Kernel attributions in `NOTICE`.
