<h1 align="center">TensorRT-DiT-Plugins</h1>

<p align="center">既存DiTカーネルのTensorRT IPluginV3化。</p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a></p>

## Highlights

- `pip install`のみ：システムCUDA・TensorRT不要。wheelが2本のプラグイン
  ライブラリを同梱し、`torch`・`tensorrt-cu12-libs`・`nvidia-cuda-runtime-cu12`
  はpipで解決。
- `dit-plugins`名前空間に7 plugin：INT8 attention、AdaLN、RoPE変種、
  FP8確率丸め、block-sparse SageAttention2（sm89のみ）。
- KernelはComfyKitchenおよびSpargeAttn由来（NOTICE・Kernel sources参照）。
  TRT glueは手書き。Comfy Orgとは無関係。ComfyKitchen互換は目標ではない。

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
out = tdp.int8_attention(q, k, v)          # engineは初回構築後はキャッシュ再利用

out = tdp.rms_adaln(x, scale, shift)       # eps既定 1e-6
qo, ko = tdp.apply_rope(q, k, freqs)
out = tdp.stochastic_rounding_fp8(x, rng)  # rng: int32、xと同要素数
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

| TRT名 (`dit-plugins::*`) | Op | I/O dtypes | Status |
|---|---|---|---|
| `int8_attention` | INT8 Q/K/V attention (SageAttention)、D ∈ {64,128,256} | in f32/f16/bf16 → out f16/bf16 (f32→bf16) | cos=0.99990, 0.910 ms/layer @ S=4096,H=16,D=128 (RTX 4070 Ti) |
| `adaln` / `rms_adaln` | 融合 LayerNorm/RMSNorm AdaLN | f32/f16/bf16 in+out（統一） | cos=1.0 |
| `apply_rope` | Strided（コピーフリー）RoPE、D偶数 | f16/bf16 in+out；freqs f32/f16/bf16 | cos=1.0 |
| `rms_rope_split_half` | 融合 RMSNorm＋split-half/partial RoPE、Dは32の倍数 | f16/bf16 in+out；freqs/scales f32/f16/bf16 | cos=1.0 |
| `stochastic_round_fp8` | FP8 E4M3への確率丸め | in f32/f16/bf16, rng INT32 → out FP8 | exact match |
| `block_sparse_sage2_attn` | Block-sparse SageAttention2、sm89のみ（kernel：SpargeAttn） | q/k/v f16/bf16 [B,H,S,D], mask INT32 [B,H,S//128,S//64] | cos=0.99953（masterのみ。v0.1.0-cu12 wheel未収録） |

全plugin I/Oは`kLINEAR`のみ。同一pluginへのQ/K/V/scale入力はdtype統一必須。
TRT 10.16は`UINT8`プラグインI/O非対応。

## Kernel sources

| Plugin | Kernel source |
|---|---|
| `int8_attention` | [comfy-kitchen `backends/cuda/sage_attention/`](https://github.com/Comfy-Org/comfy-kitchen/tree/main/comfy_kitchen/backends/cuda/sage_attention)（源流：[SageAttention](https://github.com/thu-ml/SageAttention)） |
| `adaln` / `rms_adaln` | [comfy-kitchen `backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/adaln.cu) |
| `apply_rope` / `rms_rope_split_half` | [comfy-kitchen `backends/cuda/ops/rms_rope.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/rms_rope.cu) |
| `stochastic_round_fp8` | [comfy-kitchen `backends/cuda/ops/per_tensor_quantize.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/per_tensor_quantize.cu) |
| `block_sparse_sage2_attn` | [SpargeAttn `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh)（＋[`csrc/fused/fused.cu`](https://github.com/thu-ml/SpargeAttn/blob/main/csrc/fused/fused.cu)） |

## Requirements

- NVIDIA GPU `sm80`–`sm90`（sm89で検証、RTX 4070 Ti）、ドライバR560+、Python ≥ 3.10。
- pip installにシステムCUDA・TensorRT不要。

## GPU support

| Arch | Status |
|---|---|
| sm89（RTX 40 series） | Verified — 全plugin green |
| sm80 / sm86 / sm90 | SASS同梱だが**実機未検証** |

手元はsm89のみです。他archでの不具合はPR大歓迎（GPU型番＋再現手順付きで）。

## Use in your own engine build

```python
import trt_dit_plugins as tdp
tdp.ensure_loaded()  # dit-plugins pluginをTRT registryに登録
import tensorrt as trt  # フルパッケージが必要：pip install tensorrt-cu12==10.16.*
creator = trt.get_plugin_registry().get_creator("int8_attention", "1", "dit-plugins")
plug = creator.create_plugin("attn0", fc, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([q, k, v], [], plug)
```

engineは**strongly typed**（`kSTRONGLY_TYPED`）で構築すること。弱型では出力が
FP32既定＋暗黙castになる場合がある。

## Build from source

開発者向け（kernel/wrapper改変時）。Linux＋CUDA 12.6 toolkit＋TRT 10.16が必要。
`CONTRIBUTING.md`と`.github/workflows/`参照。

## Troubleshooting

- `creator not found`：`--extra-index-url https://pypi.nvidia.com`付きで再install
  （`tensorrt-cu12-libs`／`nvidia-cuda-runtime-cu12`不足）。
- 結線後`cos ≈ 0`：engine出力dtypeを確認。FP32既定＋暗黙castは正しいがずれた
  出力に見える。
- Windows：VC++ Redistributableは`torch`／`tensorrt-cu12-libs` wheelの前提条件
  のため追加手順不要。

## License

Apache-2.0（`LICENSE`）。Kernel出所は`NOTICE`。
