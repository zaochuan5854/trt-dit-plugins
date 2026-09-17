<h1 align="center">TensorRT-DiT-Plugins</h1>

<p align="center">现有 DiT kernel 的 TensorRT IPluginV3 封装。</p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a></p>

## Highlights

- 仅需 `pip install`：不需要系统级 CUDA 或 TensorRT。wheel 自带两个插件库；
  `torch`、`tensorrt-cu13-libs`、`nvidia-cuda-runtime` 均来自 pip。
- `dit-plugins` 命名空间下共 9 个插件：INT8 attention、纯 dense SageAttention、
  融合 INT8 RoPE+SageAttention DiT-self、AdaLN、RoPE 变体、FP8 随机舍入、
  block-sparse SageAttention2（仅 sm89）。
- Kernel 来自 ComfyKitchen 与 SpargeAttn（见 NOTICE 与 Kernel sources）；
  TRT 胶水为手写。与 Comfy Org 无关；不以 ComfyKitchen 兼容为目标。

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
out = tdp.int8_attention(q, k, v)          # engine 首次构建后缓存复用

out = tdp.rms_adaln(x, scale, shift)       # eps 默认为 1e-6
qo, ko = tdp.apply_rope(q, k, freqs)
out = tdp.stochastic_rounding_fp8(x, rng)  # rng: int32，元素数与 x 相同
```

## Quick Install

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.4.0-cu13/trt_dit_plugins-0.4.0-py3-none-linux_x86_64.whl   # Linux
pip install --extra-index-url https://pypi.nvidia.com \
  https://github.com/zaochuan5854/trt-dit-plugins/releases/download/v0.4.0-cu13/trt_dit_plugins-0.4.0-py3-none-win_amd64.whl      # Windows
```

## Quick Start

```bash
python -c "import trt_dit_plugins as t; print(t.__version__, sorted(t.PLUGINS))"
```

## Plugins

| TRT 名称 (`dit-plugins::*`) | Op | I/O dtypes | Status |
|---|---|---|---|
| `int8_attention` | INT8 Q/K/V attention (SageAttention)，D ∈ {64,128,256} | in f32/f16/bf16 → out f16/bf16 (f32→bf16) | cos=0.99990, 0.910 ms/layer @ S=4096,H=16,D=128 (RTX 4070 Ti) |
| `sage_attn` | 纯 dense SageAttention，支持 GQA，D ∈ {64,128,256}（0.4.0 新增） | in f32/f16/bf16（拒绝 FP8 输入） → out f16/bf16 (f32→bf16) | sm89+D∈{64,128}+S%128==0：FP8-PV Sage2 tactic（cos=0.99925，经 `fp8_pv=1` opt-in）；其余为 portable FP16-PV（cos=0.99927） |
| `adaln` / `rms_adaln` | 融合 LayerNorm/RMSNorm AdaLN | f32/f16/bf16 in+out（统一） | cos=1.0 |
| `apply_rope` | Strided（零拷贝）RoPE，D 为偶数 | f16/bf16 in+out；freqs f32/f16/bf16 | cos=1.0 |
| `rms_rope_split_half` | 融合 RMSNorm + split-half/partial RoPE，D 为 32 的倍数 | f16/bf16 in+out；freqs/scales f32/f16/bf16 | cos=1.0 |
| `stochastic_round_fp8` | 随机舍入到 FP8 E4M3 | in f32/f16/bf16, rng INT32 → out FP8 | exact match |
| `block_sparse_sage2_attn` | Block-sparse SageAttention2，仅 sm89（kernel：SpargeAttn） | q/k/v f16/bf16 [B,H,S,D], mask INT32 [B,H,S//128,S//64] | cos=0.99953 |
| `fused_int8_rope_sage_attn` | 融合 INT8 RoPE+SageAttention DiT-self，D=128，S>1024（0.4.0 新增） | in INT8 [B,H,S,128] + FP32 scales/norms → out BF16 | cos=0.99855（相对 fp32 eager，旋转条件一致），1.578 ms/layer @ S=4096,H=16（相对分离路径 1.29x；S=9216 时 1.12x，RTX 4070 Ti） |

所有插件 I/O 均为 `kLINEAR`。同一插件的 Q/K/V/scale 输入 dtype 必须一致。
TRT 11 不支持 `UINT8` 插件 I/O。

Fused 约束：q/k/v 为相同形状的 INT8 `[B,H,S,128]`（S>1024）；per-tensor
FP32 scales（单元素）；RMSNorm 权重 FP32/BF16 `[128]`（dtype 一致）；
`inv_freq` FP32 `[64]`；仅 DiT-self（无 mask/causal）。S≤1024 在构建时
拒绝（L≤1024 路径结果错误）。

## Kernel sources

| Plugin | Kernel source |
|---|---|
| `int8_attention` | [comfy-kitchen `backends/cuda/sage_attention/`](https://github.com/Comfy-Org/comfy-kitchen/tree/main/comfy_kitchen/backends/cuda/sage_attention)（源头：[SageAttention](https://github.com/thu-ml/SageAttention)） |
| `adaln` / `rms_adaln` | [comfy-kitchen `backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/adaln.cu) |
| `apply_rope` / `rms_rope_split_half` | [comfy-kitchen `backends/cuda/ops/rms_rope.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/rms_rope.cu) |
| `stochastic_round_fp8` | [comfy-kitchen `backends/cuda/ops/per_tensor_quantize.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/per_tensor_quantize.cu) |
| `block_sparse_sage2_attn` | [SpargeAttn `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh)（+ [`csrc/fused/fused.cu`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/fused/fused.cu)） |
| `sage_attn` | 上述 vendored `sage_attention` kernels（`fp8_pv` 路径另加 sparge sm89 kernels） |
| `fused_int8_rope_sage_attn` | 自研 `src/kernel/own/fused_qk_rope_requant.cu` + vendored `sage_attention` helpers |

## Requirements

- NVIDIA GPU `sm80`–`sm90`（sm89 已验证，RTX 4070 Ti），驱动 R560+，Python ≥ 3.10。
- pip 安装不需要系统级 CUDA 或 TensorRT。

## GPU support

| Arch | Status |
|---|---|
| sm89（RTX 40 系列） | Verified — 全部插件通过 |
| sm80 / sm86 / sm90 | 带有 SASS 但**未经硬件验证** |

本地只有 sm89 硬件。其他架构如有问题，欢迎积极提 PR（请注明 GPU 型号与复现方式）。

## Use in your own engine build

```python
import trt_dit_plugins as tdp
tdp.ensure_loaded()  # 将 dit-plugins 插件注册到 TRT registry
import tensorrt as trt  # 需要完整包：pip install tensorrt-cu13==11.3.0.99
creator = trt.get_plugin_registry().get_creator("int8_attention", "1", "dit-plugins")
plug = creator.create_plugin("attn0", fc, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([q, k, v], [], plug)
```

构建 engine 请使用 **strongly typed**（`kSTRONGLY_TYPED`）；弱类型构建可能把
输出默认为 FP32 并悄悄插入 cast。

## ONNX surgery

经 ONNX 的全图构建：把子图改写为 `dit-plugins` 节点，再用 Python builder
API（`tdp.ensure_loaded()`＋`add_plugin_v3`）或
`trtexec --onnx model.trt.onnx --plugins libtrt_dit_plugins.so` 构建
（`trtexec` 随 TRT apt/deb 包发布，pip wheel 不含）。

```bash
pip install --extra-index-url https://pypi.nvidia.com "trt-dit-plugins[onnx]"
python -m trt_dit_plugins.surgeon model.onnx model.trt.onnx \
  --fuse-adaln ln_out,scale,shift,blk_out \
  --fuse-dit-self-attn q_i8,q_scale,k_i8,k_scale,v_i8,v_scale,rms_w_q,rms_w_k,inv_freq,attn_out \
  --retarget MyCustomAttn:int8_attention
```

Python API：`add_int8_attention` / `add_sage_attn` / `add_adaln` /
`add_rms_adaln` / `add_apply_rope` / `add_rms_rope_split_half` /
`add_stochastic_round_fp8` / `add_block_sparse_sage2_attn` /
`add_dit_self_fused_attn`，另有显式锚点 `fuse_norm_affine`、
`fuse_dit_self_attn_block`、`retarget_nodes`，以及无需 GPU 的
`validate_plugin_nodes`（`python/trt_dit_plugins/surgeon.py`；无自动模式
匹配 —— 各 exporter 的分解差异太大无法安全猜测，因此 fusion 采用张量名）。

RPN（FireQ 式 RoPE-preserving normalization）：
`python/trt_dit_plugins/rpn.py` 中的 `compute_rpn_scales` +
`fold_rpn_into_norms` 把每对 scale folding 进 Q/K RMSNorm 权重
（attention score 不变，无需改 kernel）。请在真实 K activation 上校准，
详见模块 docstring。

## Build from source

仅开发者（修改 kernel/wrapper）。需要 Linux + CUDA 13 toolkit + TRT 11；
见 `CONTRIBUTING.md` 与 `.github/workflows/`。

## Troubleshooting

- `creator not found`：带 `--extra-index-url https://pypi.nvidia.com` 重装
  （缺少 `tensorrt-cu13-libs` / `nvidia-cuda-runtime`）。
- 接线后 `cos ≈ 0`：检查 engine 输出 dtype — FP32 默认加隐式 cast 看起来
  就像正确但整体偏移的输出。
- Windows：VC++ Redistributable 已是 `torch`/`tensorrt-cu13-libs` wheel 的
  前提条件，无需额外步骤。

## License

Apache-2.0（`LICENSE`）。Kernel 出处见 `NOTICE`。
