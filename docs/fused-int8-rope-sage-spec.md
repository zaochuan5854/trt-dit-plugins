# 仕様書: DiT-self専用 int8入力 RoPE+SageAttn fused (Anima / Reforge-New int8エンジン用)

## 1. 目的

Anima DiT-self attentionに対し、int8量子化TRTエンジン(`ComfyUI-TensorRT-Reforge-Dev-New` の
SmoothQuant W8A8 static系)上で動作する RoPE+SageAttn 融合プラグインを新設する。
QKV入力は **INT8+scale** であり、bf16入力パスは既存 `int8_attention` が担当するため本仕様の対象外。

## 2. スコープ

### 2.1 対象

* Anima DiT 28層の **self-attnのみ**。Q/K/V同長 (square)、H=16、D=128固定。
* S (=潜在画像のH×W積) は `1025〜9216` (代表点 S=4096・S=9216、S≦1024はビルド時拒否)。B=1固定。
* 適用先: `ComfyUI-TensorRT-Reforge-Dev-New` の int8量子化TRTエンジン
  (`src/trt_reforge/onnx_ops/int8.py` の per-channel weight + per-tensor act Q/DQ系)。

### 2.2 対象外(明示)

cross-attn、LLMAdapter (D64)、causal/mask、sparse/window、FP8、動画T>1、
5120ch/H40品、bf16 QKVパス (既存pluginが担当)。

## 3. 前提・現状

* Reforge-New int8は SmoothQuant W8A8 static
  (`src/trt_reforge/onnx_ops/int8.py:1-22`)。
* 既存surgeryは2段分離: `sage.py:37` Attention→SageInt8Attn、
  `dit_plugins.py:112-193` norm+rope→`rms_rope_split_half` (PRE-norm入力+baked freqs)。
* 既存 `int8_attention` pluginは bf16/fp16/fp32入力→内部quant→attn
  (`src/wrapper/int8_attention_wrapper.cpp:5,110-142,194-206`)。
  本新設はこの前段を置き換え、`src/kernel/sage_attention/` 本体
  (`quant_qk/quant_v/sage_attn_launcher`)は再利用する。
* Anima DiT-self確定値: `D_model 2048/H16/D128/MHA`、QK per-head RMSNorm eps1e-6、
  3D split-half行列RoPE fp32、非causal・maskなし。

## 4. 新plugin定義

* 名前案: `fused_int8_rope_sage_attn`、`domain=dit-plugins`、IPluginV3
  (OneCore/OneBuild/OneRuntime)。
* 入力 (全kLINEAR):

| # | 名 | dtype/shape | 意味 |
|---|---|---|---|
| 0 | `q_int8` | INT8 `[B,H,S,D]` | 前段int8 GEMM出力 (量子化済Q) |
| 1 | `q_scale_in` | FP32 scalar or `[B,H,S]` | 前段act scale (SmoothQuant per-tensor既定。per-tokenも受容) |
| 2 | `k_int8` | INT8 `[B,H,S,D]` | 同K |
| 3 | `k_scale_in` | FP32 同上 | 同K scale |
| 4 | `v_int8` | INT8 `[B,H,S,D]` | 同V (norm/ropeなし) |
| 5 | `v_scale_in` | FP32 同上 | 同V scale |
| 6 | `q_norm_scale` | FP32/BF16 `[D]` | q RMSNorm重み |
| 7 | `k_norm_scale` | FP32/BF16 `[D]` | k RMSNorm重み |
| 8 | `freqs` | FP32 baked initializer `[1,1,Smax,r,2,2]` | rope回転行列。grid依存のためresolution profile毎にbake |

* 出力: `o` BF16 `[B,H,S,D]` (既存規約 INT8in→BF16outに拡張)。
* 属性: `epsilon=1e-6`、`rot_dim=0` (full128)、`sm_scale=1/sqrt(128)` は
  onShapeChangeで固定、`D=128` assert。
* `supportsFormatCombination`: 入INT8はkINT8、scales/freqsはkFLOAT、
  norm scalesはkFLOAT/kHALF/kBF16許容、出力kBF16、formatはkLINEARのみ。
* `configurePlugin`: D==128・ndims==4以外拒否。
* workspaceは既存 `planWs` 流儀+前段バッファ (fp32 Q/Kタイル・Sage scale) を加算。
  `cta_k = (Lk>1024) ? 128 : 64` は既存踏襲 (本条件ではほぼ128)。

## 5. カーネルデータフロー

単一fp32パイプラインとし、fp16中間を挟まない (二重丸め禁止)。

```text
q/k_int8 --dequant(*scale_in, fp32)--> RMSNorm(D128,eps) --> RoPE split-half(freqs行列積,fp32)
   --> Sage per-thread INT8 quant(BLKQ128/BLKK=cta_k)
   --> qk_int_sv_i8 (PV=FP16既存) --> o(BF16)
v_int8 --dequant--> (norm/ropeなし) --> quant_v_int8(既存) --> 同attn
```

* `q_scale` へのsm_scale混入は禁止。sm_scaleは別扱いとしGEMM直後FP32で適用。
* Vは既存 `quant_v_int8` 経路そのまま。P/V量子化なし (Sage B型)。

## 6. ONNX surgery仕様 (Reforge-New側)

* 置換パターン: `RMSNorm(q)+RMSNorm(k)+RoPE-matmul群+Attention/SageInt8Attn` を1ノード化。
  cross (norm直結) は除外 (`dit_plugins.py:154-156` と同一判定)。
* 順序: `sage.py`→`dit_plugins.py` の後に第3surgeryとして適用、またはAttentionを直接
  `fused_int8_rope_sage_attn` に置換。前段Transpose整理 (`[B,S,H,d]→[B,H,S,d]`) は既存踏襲。
* freqs bakeは固定profile毎 (S=4096・S=9216の2profile先行)。
  動的H/Wはeager ropeにfallback。
* Q/DQ扱い: 前段int8 GEMMのact Q/DQはplugin境界で内側化し、外側に残存させない。

## 7. 受け入れ基準

* 数値: 代表2点 (S=4096・S=9216、B1H16D128、非causal) で
  参照=既存分離パスに対し `cos worst≥0.997` かつ Sage-B worst≥99.8%、RelL1≤0.06目安。
  0近傍rel爆発はp90/p99+散布で除外判定。
* 性能: quant+rope含む実効時間で既存分離比を測定 (TOPS換算 `O=4BHS²D` 併記、
  Graph off/on・warmup100/iter500同一条件)。
* 運用: 非対応 (D≠128・mask・causal・S>9216・Adapter) はビルド時拒否または分離パスfallback。
