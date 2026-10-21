# ConvRot W8A8 / W4A4 Linear TensorRT プラグイン 計画書（comfy-kitchen ベース）

- 最終更新: 2026-10-21
- 対象: `https://github.com/zaochuan5854/trt-dit-plugins`
- 検証 GPU: RTX 4070 Ti (sm_89) / CUDA 13.0 / Python 3.12 / PyTorch 2.9
- 参照実装: `comfy-kitchen` 0.2.39（Apache-2.0）

---

## 目次

- [0. TL;DR](#0-tldr)
- [1. 方針: 自前カーネルを書かず、最速の実装をそのまま使う](#1-方針-自前カーネルを書かず最速の実装をそのまま使う)
  - [1.1 結論](#11-結論)
  - [1.2 却下した案と実測根拠](#12-却下した案と実測根拠)
  - [1.3 なぜ「ラッパーだけ」で成立するのか](#13-なぜラッパーだけで成立するのか)
- [2. 仕様](#2-仕様)
  - [2.1 量子化フォーマットとチェックポイント](#21-量子化フォーマットとチェックポイント)
  - [2.2 回転（ConvRot）](#22-回転convrot)
  - [2.3 W8A8 の量子化セマンティクス](#23-w8a8-の量子化セマンティクス)
  - [2.4 W4A4 の量子化セマンティクス](#24-w4a4-の量子化セマンティクス)
  - [2.5 GEMM と dequant](#25-gemm-と-dequant)
  - [2.6 入出力契約](#26-入出力契約)
  - [2.7 W8A8 と W4A4 の差分](#27-w8a8-と-w4a4-の差分)
- [3. 計測結果](#3-計測結果)
  - [3.1 W8A8 / W4A4 / 代替実装の比較](#31-w8a8--w4a4--代替実装の比較)
  - [3.2 精度](#32-精度)
  - [3.3 解釈](#33-解釈)
- [4. `trt-dit-plugins` 側の要件](#4-trt-dit-plugins-側の要件)
- [5. 実装方針](#5-実装方針)
- [6. テスト計画](#6-テスト計画)
- [7. タスク分解](#7-タスク分解)
- [8. リスク](#8-リスク)
- [9. 受入チェックリスト](#9-受入チェックリスト)
- [Appendix A: vendor する API（実シグネチャ）](#appendix-a-vendor-する-api実シグネチャ)
- [Appendix B: IPluginV3 ラッパー](#appendix-b-ipluginv3-ラッパー)
- [Appendix C: Python / ONNX / CMake / テスト](#appendix-c-python--onnx--cmake--テスト)
- [Appendix D: 計測生データ](#appendix-d-計測生データ)

---

## 0. TL;DR

- **カーネルは書きません。** `comfy-kitchen`（Apache-2.0）の CUDA ソースを vendor し、`IPluginV3` の薄いラッパーから **Comfy が実際に使っているのと同じ関数を、同じ順序で**呼びます。したがって数値は定義上一致し、性能は「Comfy 以上」が出発点になります。
- 対応するのは 2 つの op:
  - `convrot_linear` — W8A8（既存 ConvRot INT8 チェックポイント）
  - `convrot_linear_w4a4` — W4A4（`convrot_w4a4` 量子化チェックポイント / オフライン変換）
- 速度差を生むのはカーネルではなく **経路選択**です。Comfy のルーティングをそのまま移植します（W8A8: M=1 の融合 GEMV / M=2 の GEMV dequant / M≥3 の融合 CUTLASS、W4A4 sm8x: legacy INT4 か CUTLASS INT4、W4A4 非 sm8x: W4A8）。CUTLASS のタイル config だけは **エンジンビルド時に一度だけ計測して固定**します（`enqueue` 内では絶対に計測しない。§5.3）。
- 我々が書くのは「配線」だけ: ONNX ノード生成、プラグインの shape/dtype 検証、workspace 計算、カーネル選択表、parity テスト、ComfyUI 側の Q/DQ バイパス。
- 実測（RTX 4070 Ti, 実 Anima レイヤ）:
  - W8A8: 0.67 / 0.76 / 0.20 ms（layer1 / layer2 / q_proj）— Comfy と同一
  - W4A4: 0.39 / 0.60 / 0.14 ms — W8A8 の **1.40–1.73 倍速**、重みバイト **1/2**
  - 自前実装（却下）: 1.25 / 1.81 / 0.35 ms、ggml 実装（却下）: 1.41 / 1.30 / 0.39 ms

---

## 1. 方針: 自前カーネルを書かず、最速の実装をそのまま使う

### 1.1 結論

| 項目 | 決定 |
|---|---|
| カーネル | comfy-kitchen の CUDA ソースを vendor（改変は config 固定化のみ） |
| 数値の正解 | vendor したコードそのもの（テストの oracle は `comfy_kitchen` パッケージ） |
| 速度 | Comfy と同じ経路 + ビルド時 autotune で Comfy 以上 |
| 我々の成果物 | `IPluginV3` ラッパー、ONNX/Python 配線、テスト、ビルド統合 |
| 自前カーネル | **採用しない**（§1.2 の実測根拠） |

「最速を自分で作れないなら最速を使う」という判断にそのまま従います。現時点で Anima の ConvRot linear において最速なのは Comfy の CUTLASS 融合カーネルであり、これを超える実装を自作する根拠（＝見込み）はありません。

### 1.2 却下した案と実測根拠

同一 GPU・同一実データ（Anima 実活性化、bias なし、30 iter、CUDA event）での実測です。

| 実装 | layer1 (4096×2048→8192) | layer2 (4096×8192→2048) | q_proj (4096×2048→2048) | 対 Comfy |
|---|---:|---:|---:|---:|
| **Comfy CUTLASS（採用）** | **0.67 ms** | **0.76 ms** | **0.20 ms** | 1.00× |
| 自前 v1（cuBLAS INT8 + 分離 dequant） | 1.25 ms | 1.81 ms | 0.35 ms | 1.7–2.4× 遅い |
| ggml 実装（`convrot-linear.cu`） | 1.41 ms | 1.30 ms | 0.39 ms | 1.75–2.10× 遅い |

遅い理由は明確です。

1. **融合していない。** Comfy は GEMM の epilogue で `acc * xs[m] * ws[n] + bias` を実行する。自前 v1 / ggml は INT32 の `acc` を HBM に書き戻して別カーネルで読む。layer1 では `acc` が 4096×8192×4 = 134 MB あり、これだけで 0.6–1.1 ms 失う。
2. **量子化カーネルが遅い。** Comfy の `convrot64` は 64 スレッド/グループ + 段階的 barrier で layer1 0.05 ms、layer2 0.25 ms。自前は 0.10–1.24 ms、ggml は 0.10–0.53 ms。
3. **チューニング済みタイルを持っていない。** Comfy は形状ごとに 14（INT8）/ 7（INT4）の CUTLASS config を持つ。

したがって「自前で最速を作る」路線は、**同じ CUTLASS カーネルを書き直す**ことに等しく、得るものがありません。数値一致のリスクだけが増えます。

### 1.3 なぜ「ラッパーだけ」で成立するのか

Comfy のカーネルは `extern "C"` のランチャーとして既に「生ポインタ + `cudaStream_t`」の形で export されています。`torch::Tensor` も GIL も不要です。

```cpp
// comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu
bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream);
```

これは `IPluginV3OneRuntime::enqueue` が要求する形そのものです。Python 側の `int8_linear()` がやっていること（M による経路分岐、scale の整形、bias の dtype 一致）を C++ で 50 行程度に写せば、あとは TRT のバッファをそのまま渡せます。

一方、Python の `comfy_kitchen.int8_linear` を「呼ぶ」ラッパーは成立しません。理由は §1.3.1 に残します。

#### 1.3.1 却下: Python op を呼ぶラッパー

- `enqueue` は GIL を持たないスレッドから呼ばれ得る。`PyGILState_Ensure` は CUDA Graph capture と両立しない。
- `int8_linear` は出力を内部で `torch.empty` する（`out=` が無い）。TRT の出力バッファへコピーが増える。
- torch の caching allocator を TRT の実行 stream に関連付ける必要があり、推論中の allocation は避けたい。
- プラグイン `.so` が libtorch に依存する。`trt-dit-plugins` の plugin ライブラリは torch 非依存で通している。

#### 1.3.2 却下: `_C.abi3.so` を dlopen して呼ぶ

シンボルは export されていますが、ヘッダが無く、nanobind 拡張を plugin からリンクする形になり、バージョン固定・SM 別カーネル選択・ABI の面で脆い。ソースを vendor すれば同じコードを自分のビルドに含められるため、バイナリ依存を選ぶ理由がありません。

---

## 2. 仕様

### 2.1 量子化フォーマットとチェックポイント

Anima の既存チェックポイントは `int8_tensorwise` + `convrot: true`。ComfyUI は W4A4 を別フォーマット `convrot_w4a4` として扱います。

| フォーマット | 重み storage | 付随パラメータ | `comfy_quant` JSON | 備考 |
|---|---|---|---|---|
| `int8_tensorwise` + `convrot` | INT8 `[N, K]` | `weight_scale` FP32 `[N]` | `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` | 既存 Anima |
| `convrot_w4a4` | INT8 `[N, K/2]`（4bit×2/byte） | `weight_scale` FP32 `[N]` | `{"format":"convrot_w4a4","convrot_groupsize":256}` | W4A4 |

- どちらも重みは **回転済み**。プラグインで再回転してはいけない。
- `group_size`（回転グループ）は 256 固定。
- W4A4 の重み量子化は **行単位スケール**（出力チャネルごとに 1 本）。グループスケールではない。
- `linear_dtype`（`int4` / `int8`）は実行時の経路指定であり、重み形式は同じ。

### 2.2 回転（ConvRot）

正規化 Hadamard、radix-4 を 4 段、各段で `0.5` を掛ける（全体で `1/16`）。グループ長 256。

```text
H4 = [[ 1,  1,  1, -1],
      [ 1,  1, -1,  1],
      [ 1, -1,  1,  1],
      [-1,  1,  1,  1]]
H  = (H4 ⊗ H4 ⊗ H4 ⊗ H4) / 16
```

- 活性化: `x_rot = x @ H`（行方向、256 要素ごと）
- 重み: `W_stored = W @ H`（チェックポイント作成時に適用済み）
- `(x H)(W H)^T = x W^T`（H は対称・直交）

### 2.3 W8A8 の量子化セマンティクス

Comfy は BF16 で量子化をエミュレートします（SASS で確認済み）。この通りでないとビット一致しません。

```text
scale   = max(rowmax(|x_rot|) * (1/127), 1e-30)
s_bf    = bf16(scale)
inv     = 1 / s_bf                     # MUFU.RCP
p       = bf16(bf16(x_rot) * inv)      # 積も BF16 で丸める
q       = clamp(rint(p), -128, 127)    # F2I.TRUNC 併用、FTZ
```

- 量子化の対象は **回転後**の値。
- ストキャスティック丸め
- sm_75（Turing）は使わない（`stochastic=false`）。
- 出力は `x_q` INT8 `[M, K]` と `x_scale` FP32 `[M]`。

### 2.4 W4A4 の量子化セマンティクス

**W8A8 とは別物**です。BF16 エミュレーションは無く、素の FP32 演算です。

```text
scale = max(min(rowmax(|x_rot|), dtype_max) * (1/7), 1e-10)   # kInt4Max = 7
q     = clamp(rint(x_rot * (1/scale)), -7, 7)                 # __float2int_rn
```

- `dtype_max` は入力型の有限最大値（`finite_max_for_dtype<T>()`）: BF16 `3.38953139e38`、FP16 `65504`、FP32 `FLT_MAX`。NaN/Inf 入力の暴走を止めるためのもので、通常は `min` が効かない。

- 範囲は `[-7, 7]`（15 レベル）。`-8` は使いません。
- パッキング: 行優先 `[N, K/2]`、**下位ニブル = 偶数 k、上位ニブル = 奇数 k**（符号付き 4bit）。
- `K % 64 == 0`（MMA の K グループ）と `K % 256 == 0`（回転グループ）が必要。
- 重みも同じ規則（`quantize_convrot_w4a4_weight` = 回転 → 行単位 INT4 量子化 → パック）。

### 2.5 GEMM と dequant

| 経路 | 演算 | 条件 |
|---|---|---|
| W8A8 GEMM | INT8 × INT8 → INT32、`mma.sync.m16n8k32.s8.s8.s32`（CUTLASS Sm80） | M ≥ 3 |
| W8A8 GEMV | 融合 GEMV（量子化を内包） | M = 1 |
| W8A8 GEMV2 | 量子化 + GEMV dequant | M = 2 |
| W4A4 GEMM（CUTLASS） | INT4 × INT4 → INT32、`mma.sync.m16n8k64.s4.s4.s32`、7 config | sm major == 8 かつ `!prefer_legacy_int4` |
| W4A4 GEMM（legacy） | INT4 × INT4 → INT32、自作タイル選択 | sm major == 8 かつ `prefer_legacy_int4` |
| W4A8 GEMV | INT4 重み × INT8 活性、GEMV | 非 sm8x かつ M ≤ 8 |
| W4A8 GEMM | INT4 重み × INT8 活性、チャンク GEMM | 非 sm8x かつ M > 8（Hopper/Blackwell 等） |

- **W4A4 の M による分岐は「sm8x かどうか」とは独立**です。sm8x では M=1 でも INT4 量子化 + INT4 カーネルを使います（W4A8 には落ちません）。
- sm8x 内の CUTLASS / legacy の選択は Comfy の `_prefer_legacy_int4_kernel(m, n, k, device)` によります:

  ```text
  m_tile    = max(32, next_pow2(m))
  base      = 1_048_576 if k <= 1024 else 786_432
  threshold = base * sm_count / 142
  workload  = m_tile * n
  legacy    = n < 32768 and (
                workload < threshold
                or (k > 1024 and workload == threshold and 128 <= m and m == m_tile and m < 512))
  ```

  低レイテンシ形状（小さい M×N）では legacy カーネルの方が速い、という実測に基づく分岐です。決定的（timing なし）ですが `sm_count` に依存します。

dequant（CUTLASS epilogue、または専用カーネル）:

```text
y[m, n] = float(acc[m, n]) * x_scale[m] * weight_scale[n] + bias[n]
```

- `acc` は INT32。整数加算は結合的なので、**タイル config を変えても `acc` はビット一致**します（＝config は速度だけの自由度）。
- 出力 dtype は BF16 / FP16 / FP32。

### 2.6 入出力契約

**ConvRot かどうかは op 名で表します。** `convrot_linear` / `convrot_linear_w4a4` に渡る
`weight` は常に ConvRot 回転済みで、graph 側に別フラグ（`convrot=1` 等）は持ちません。
重みのバイト列からは回転の有無を判別できないため、どのレイヤをこの op にするかは
export 側（§5.7 の判定表）が唯一の情報源になります。回転なしの INT8 をこの op に流すと
**エラーにならず静かに値が壊れる**ので、変換前に `params.convrot` を必ず検査します。

2 つの op を持ちます。

| op | inputs | 備考 |
|---|---|---|
| `convrot_linear` | `x`, `weight` `[N,K]` INT8, `weight_scale` `[N]` FP32, `bias` `[N]` **出力 dtype** | W8A8 |
| `convrot_linear_w4a4` | `x`, `qweight` `[N,K/2]` INT8, `weight_scale` `[N]` FP32, `bias` `[N]` **出力 dtype** | W4A4 |

- 出力は 1 本 `y` `[..., N]`。
- `K` は `qweight.shape[1] * 2` として復元する。
- `bias` の dtype は **出力と同じ**（Comfy の epilogue / GEMV が出力 dtype で読み、変換しないため）。BF16 出力なら BF16 bias を initializer として持つ。
- bias なしはゼロ tensor + `has_bias=0`（内容はカーネルに読まれない）。

### 2.7 W8A8 と W4A4 の差分

| 項目 | W8A8 | W4A4 |
|---|---|---|
| 活性量子化 | BF16 エミュレーション、`1/127`、`1e-30` | FP32、`1/7`、`1e-10` |
| 範囲 | `[-128, 127]` | `[-7, 7]` |
| 活性バッファ | `[M, K]` INT8 | `[M, K/2]` INT8 |
| 重み | `[N, K]` INT8 | `[N, K/2]` INT8 (packed) |
| MMA | m16n8k32 s8 | m16n8k64 s4 |
| 追加条件 | — | `K % 64 == 0`, sm major == 8（または W4A8 経路） |
| 精度（BF16 比） | cos ≈ 0.99997–0.99999 | cos ≈ 0.990–0.998 |
| 重みバイト | 1× | 0.5× |

---

## 3. 計測結果

### 3.1 W8A8 / W4A4 / 代替実装の比較

RTX 4070 Ti、Anima 実チェックポイント・実活性化、bias なし、30 iter、CUDA event。

| レイヤ | 形状 M×K→N | W8A8 | W4A4 | W4A4/W8A8 | 自前 v1 | ggml |
|---|---|---:|---:|---:|---:|---:|
| `mlp.layer1` | 4096×2048→8192 | 0.675 ms | **0.390 ms** | 1.73× | 1.248 ms | 1.406 ms |
| `mlp.layer2` | 4096×8192→2048 | 0.858 ms | **0.595 ms** | 1.44× | 1.814 ms | 1.303 ms |
| `attn.q_proj` | 4096×2048→2048 | 0.202 ms | **0.144 ms** | 1.40× | 0.354 ms | 0.394 ms |

重みサイズ（layer1 の例）: W8A8 16.81 MB → W4A4 8.42 MB。

W4A4 の内訳（Comfy 実装）: 量子化 + INT4 GEMM + 融合 dequant。活性量子化はパック済み INT4 を直接出力するため、W8A8 より HBM 書き込みが半分になります。

### 3.2 精度

| レイヤ | W8A8 cos (vs BF16) | W4A4 cos (vs BF16) | W4A4 rel L2 |
|---|---:|---:|---:|
| layer1 | 0.999967 | 0.990425 | 14.0% |
| layer2 | 0.999992 | 0.998268 | 5.9% |
| q_proj | 0.999989 | 0.997585 | 7.0% |

- W4A4 は **速度と引き換えに品質を落とします**。layer1 の劣化が大きい（rel L2 14%）。
- レイヤ単位の cos はモデル全体の cos ではない。E2E の許容可否は §6.3 でゲートを切る。
- W4A4 は「Anima の既定」ではなく **オプション**として提供する（既定は既存チェックポイントに合わせて W8A8）。

### 3.3 解釈

1. 速度差の支配要因は「融合の有無」であり、カーネルを書き直す余地はほぼ無い。
2. W4A4 の 1.4–1.7× は重み読み出しと MMA スループットの両方から来る。TRT 側でも同じカーネルを使う限り同等以上になる。
3. 品質面の判断材料はレイヤ単位では足りない。E2E で `cos >= 0.995` を満たすかで採否を決める（§6.3）。

---

## 4. `trt-dit-plugins` 側の要件

### 4.1 機能要件

- プラグイン名: `convrot_linear`（W8A8）/ `convrot_linear_w4a4`（W4A4）
- 名前空間 / ONNX domain: `dit-plugins`、バージョン `1`
- 1 ノードで「活性回転 + 行単位量子化 + INT8/INT4 GEMM + dequant + bias」を実行
- 重みは **入力テンソル / ONNX initializer**。プラグインのシリアライズフィールドに重みを入れない
- strongly typed network、全 I/O は `kLINEAR`
- `group_size` は 256 のみ

### 4.2 プラグイン契約

| 項目 | `convrot_linear` | `convrot_linear_w4a4` |
|---|---|---|
| inputs | `x`, `weight`, `weight_scale`, `bias` | `x`, `qweight`, `weight_scale`, `bias` |
| weight | `[N, K]` INT8 | `[N, K/2]` INT8 |
| outputs | `y` | `y` |
| fields | `group_size`, `has_bias`, `out_dtype`, `gemm_config` | 同左 |
| tactic | 経路（GEMV / GEMM）+ CUTLASS config | 同左 |

`out_dtype`: `-1`=入力と同じ（既定）, 0=BF16, 1=FP16, 2=FP32。
`gemm_config`: `-1` = 既定（W8A8 は Comfy の決定的ヒューリスティック、W4A4 はビルド時 autotune の結果）。`>=0` は固定。

### 4.3 形状・dtype・制約

| テンソル | 形状 | dtype |
|---|---|---|
| `x` | `[..., K]`（rank ≥ 2） | BF16 / FP16 / FP32 |
| `weight` | `[N, K]` or `[N, K/2]` | INT8 |
| `weight_scale` | `[N]` or `[N,1]` | FP32 |
| `bias` | `[N]` | 出力 dtype |
| `y` | `[..., N]` | `out_dtype` |

制約:

- `K % 256 == 0`（両方）
- `K <= 16384`（v1 の上限。量子化カーネルの shared memory は `(K + groups_in_flight*2*256) * 4` bytes（INT8 経路、`groups_in_flight = block_threads/64`、`K >= 12288` で `block_threads=1024`）。K=8192 で 48 KB、K=16384 で 96 KB、K=17408 で 100 KB となり Ada の opt-in 99 KB を超える）。Anima の最大は 8192 で、実測・parity 検証済みの範囲は `K <= 8192`。16384 は仕様上許容するが、テストは 8192 までを必須とする
- `N % 4 == 0`（W8A8）
- `N % 32 == 0`（W4A4 の CUTLASS 経路。`AlignB = 32`（int4 要素数）。満たさない場合は `can_implement` が失敗するので、legacy `launch_int4_linear_kernel`（`kBlockN = 256`、境界チェックあり）にフォールバックする。**エラーにしない**）
- `M = prod(x.shape[:-1]) > 0`
- `x` の最終次元 = `K`
- **sm_80 以上**（Ampere / Ada / Hopper / Blackwell）。Turing (sm_75) は対象外:
  - Comfy は sm_75 で `turing_int8.cu` の融合カーネル（`m <= 128 || n >= 2*k`）を使い、それ以外は cuBLAS に落ちる。前者は vendor せず、後者は削除するため、sm_75 では経路が無い。
  - Python 側で `torch.cuda.get_device_capability()[0] >= 8` を確認し、満たさなければプラグインを挿入せず BF16 パスを残す（fail-open）。

### 4.4 数値要件

- W8A8: `comfy_kitchen.int8_linear(x, w, ws, convrot=True, convrot_groupsize=256)` と **完全一致**（`max_abs == 0`, `cos == 1.0`）。
- W4A4: `comfy_kitchen.convrot_w4a4_linear(x, qw, ws, convrot_groupsize=256)` と **完全一致**。
  - ただし CUTLASS config は速度だけの自由度なので、config が違っても `acc` はビット一致する。`max_abs == 0` を要求してよい。
- 実レイヤ 3 本（layer1 / layer2 / q_proj）+ 合成形状で検証。
- 量子化単体: `q` の完全一致、`scale` の完全一致。

数値ゲート:

```text
q_eq == 1.0, scale_exact_frac == 1.0, max_abs == 0.0, cos >= 0.999999
```

### 4.5 性能要件

ゲートは「Comfy の 1.05 倍以内」。実測 Comfy を基準に:

| レイヤ | W8A8 目標 | W4A4 目標 |
|---|---:|---:|
| layer1 | ≤ 0.71 ms | ≤ 0.41 ms |
| layer2 | ≤ 0.80 ms | ≤ 0.63 ms |
| q_proj | ≤ 0.22 ms | ≤ 0.15 ms |

- `enqueue` 内で同期・計測をしない（CUDA Graph / 初回推論のスパイク防止）。
- ベンチは TRT 経由（プラグイン込み）で測る。カーネル単体の数字は参考値。

### 4.6 パッケージ / ビルド要件

- `ck_kernels`（SHARED）に vendor ソースを追加、`trt_dit_plugins`（SHARED）にラッパーを追加
- **cuBLAS 非依存**。W8A8/W4A4 の採用経路は CUTLASS と自作 MMA のみ（`cublas_gemm_int8.cu` と `cublaslt_runtime.h` は取り込まない）
- CUTLASS は header-only を submodule で固定（comfy-kitchen v0.2.39 の submodule SHA `d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`）
- **コンパイルフラグは上流と完全に同一**にする。`--use_fast_math` は上流では**全 CUDA TU**に掛かっており（FTZ を含む）、量子化だけでなく dequant epilogue の除算・変換にも影響する。ここを変えるとビット一致が壊れる
- C++20 / CUDA 12.8 以上（上流の `CMakeLists.txt` と同じ要求）
- wheel は pure-Python + prebuilt lib（`py3-none-<plat>`）。サイズ増加分は CUTLASS のビルド成果物のみ
- Windows は `WINDOWS_EXPORT_ALL_SYMBOLS` を維持し `extern "C"` を export

### 4.7 既存規約との整合

- `select.py`: `PluginOp.CONVROT_LINEAR`, `PluginOp.CONVROT_LINEAR_W4A4` を追加、`QUALITY_CLASS` は両方 `"quant"`
- `_lib.py`: `PLUGINS` に 2 エントリ追加
- `surgeon.py`: `_SPECS` に `(4, 1, frozenset({"group_size", "has_bias", "out_dtype", "gemm_config"}))`
- `ops.py`: `convrot_linear(...)` / `convrot_linear_w4a4(...)`
- `test/test_register.cpp`: `kNames[]` に 2 名追加
- ラッパーは `fused_int8_rope_sage_wrapper.cpp` の `IPluginV3` + `AutoRegister` パターンを踏襲

### 4.8 非目標（v1 で reject）

- `group_size != 256`
- `int8_tensorwise` かつ `convrot=False` / `transposed=True`（回転なし・転置レイアウトの INT8 は v1 非対応。変換側で fail-open）
- `K % 256 != 0` / `K > 16384`
- W4A4 で `K % 64 != 0`
- `x` の rank < 2、`weight` が 2D でない
- `weight` が INT8 でない、`weight_scale` が FP32 でない、`bias` が出力 dtype でない
- 動的 N / K（weight は initializer 前提で static）
- 活性化融合（`input_act=gelu/swiglu` 等）。v1 は線形のみ。グラフ側で別ノードとして残す
- residual 融合
- ストキャスティック丸め
- sm_75（Turing）

---

## 5. 実装方針

### 5.1 vendor するファイル

`comfy-kitchen` の **`v0.2.39` タグ**（計測・parity の基準にした wheel と同一）から次を `src/kernel/vendor/comfy/` にコピーします。wheel には `_C.abi3.so` しか入っていないため、ソースは GitHub のタグから取得します（`cublaslt_runtime.h` なども wheel 内にはありますが、コンパイルに必要な `.cu` は含まれません）。

| vendor 元（`comfy_kitchen/backends/cuda/` 配下） | 用途 | 改変 |
|---|---|---|
| `ops/int8_linear.cu` | W8A8 量子化（convrot64）、M=1 融合 GEMV、M≤2 GEMV、dequant | そのまま |
| `ops/cutlass_gemm_int8.cu` | W8A8 融合 GEMM（14 config、決定的ヒューリスティック） | `*_config` に bias / FP16 / FP32 を追加（§5.3.2） |
| `ops/cutlass_gemm_common.cuh` | `TileConfig` / `ConfigList` / swizzle | そのまま |
| `ops/convrot_w4a4.cu` | W4A4 量子化、INT4 MMA、CUTLASS INT4 融合 GEMM、W4A8 GEMV/GEMM | autotune を config 固定に差し替え（§5.3.3）、cuBLAS 分岐を除去（§5.1.1） |
| `ops/w4a8_gemm.cu` | W4A8 チャンク GEMM（非 sm8x フォールバック） | そのまま |
| `ops/svdquant_utils.cuh` | `kInt4Max`、pack/unpack、`mma.sync` ヘルパ | そのまま |
| `dtype_dispatch.cuh` | dtype code → 型ディスパッチ | そのまま |
| `utils.cuh`, `float_utils.cuh`, `input_act_codes.h` | 共通ヘルパ | そのまま |
| `cublaslt_runtime.h`, `ops/cublas_gemm_int8.cu` | cuBLAS 経路 | **取り込まない** |
| `dlpack_bindings.cpp`, `int8_decode_bindings.cpp` | nanobind バインディング | **取り込まない**（自前の C shim に置換） |
| `ops/turing_int8.cu`, `ops/turing_int4.cu` | sm_75 用の融合カーネル | **取り込まない**（v1 は sm_80 以上のみ、§4.3） |
| `flash_compat/`, `sage_attention/`, その他の `ops/*.cu` | 無関係 | **取り込まない** |

ディレクトリ構成は次のとおり（include の解決のため、上流と同じ相対配置を保つ）:

```text
src/kernel/vendor/comfy/
  utils.cuh  dtype_dispatch.cuh  float_utils.cuh  input_act_codes.h
  ops/
    int8_linear.cu  cutlass_gemm_int8.cu  cutlass_gemm_common.cuh
    convrot_w4a4.cu  w4a8_gemm.cu  svdquant_utils.cuh
  VENDOR.md  LICENSE  NOTICE
```

#### 5.1.1 cuBLAS を落とす

- `convrot_w4a4.cu` は冒頭で `extern "C" void launch_cublas_gemm_int8_kernel(...)` を **前方宣言**し、W4A8 チャンク GEMM の `allow_sm80_cutlass == false` 分岐から呼びます。宣言だけなので、そのままではリンク時に未定義参照になります。
- パッチ内容は 2 点だけ:
  1. 前方宣言（`extern "C" void launch_cublas_gemm_int8_kernel(...)`）を削除
  2. `if (!allow_sm80_cutlass) { ... cublas ... }` の分岐を `return;`（no-op）に置換
- 関数シグネチャ（`cublas_workspace` / `cublas_workspace_size` 引数）は **変えない**。shim は `nullptr, 0` を渡す。シグネチャを変えると上流差分が大きくなり、追従コストが上がるため。
- 前提条件は shim 側で保証する: `allow_sm80_cutlass = true`、`K % 16 == 0`（ConvRot は `K % 256 == 0` なので常に成立）。したがって no-op 分岐は到達しない。
- これで plugin ライブラリの依存は `cudart` + CUTLASS（header-only）だけになります。
- **v0.2.39 では cuBLAS 経路は実際には到達しません**。Python 側の `_prefer_cublas_int8_fallback()` は常に `False` を返すため、ルーティングは変わりません（削除は安全）。念のため `_prefer_cublas_int8_fallback` の戻り値が将来変わっていないかを、vendor 更新時に確認します。

#### 5.1.2 上流追従

- 固定する版: `comfy-kitchen` **v0.2.39**（commit `e40e76d9c544b3f74c3b9ea923f7bc3d45a96372`、Apache-2.0）。この版を `VENDOR.md` に記録する。
- 上流の修正を取り込む手順（`git diff v0.2.39..<new> -- comfy_kitchen/backends/cuda/` の当て方、改変箇所の一覧）も同ファイルに書く。
- 改変は必ず `// DIT-PLUGINS-PATCH: <理由>` でマークし、`grep -rn "DIT-PLUGINS-PATCH"` で全量を確認できるようにする。
- CUTLASS は comfy-kitchen の `third_party/cutlass` と同じタグに固定する（`cutlass_gemm_int8.cu` が EVT / `default_gemm_universal_with_visitor.h` を使うため、版の差に敏感）。

### 5.2 我々が書くコード

```text
src/
  kernel/
    vendor/comfy/            # 上表（改変は config 固定化と cuBLAS 除去のみ）
    convrot_shim.cu          # comfy ランチャーへの薄い C ラッパー + autotune ヘルパ
    convrot_shim.h
  wrapper/
    convrot_linear_wrapper.cpp   # IPluginV3（W8A8 / W4A4 をテンプレートで共通化）
python/
  trt_dit_plugins/
    select.py                # PluginOp に 2 種追加
    _lib.py                  # PLUGINS に 2 種追加
    surgeon.py               # _SPECS / add_convrot_linear / add_convrot_linear_w4a4 / validate
    ops.py                   # convrot_linear(...) / convrot_linear_w4a4(...)
    autotune.py              # ビルド時に gemm_config を決める
test/
  test_register.cpp          # kNames[]
python/tests/
  test_convrot.py            # parity / negative / perf
docs/
  convrot-linear-trt-plugin-plan.md
```

`convrot_shim.cu` は「comfy の関数を、TRT のバッファから呼ぶための最小の糊」だけを持ちます。カーネルは 1 行も書きません。

### 5.3 経路選択と config 固定（最重要）

#### 5.3.1 M による経路分岐（Comfy と同じ）

`enqueue` の先頭で `M` を見て経路を決めます。分岐だけで、計測はしません。

W8A8:

```text
M == 1                              -> launch_int8_gemv_convrot_fused_kernel   (量子化込みの 1 カーネル)
M == 2                              -> quantize_int8_rowwise_convrot64 + launch_int8_gemv_dequant_kernel
M >= 3                              -> quantize_int8_rowwise_convrot64 + launch_cutlass_int8_dequant[_config]
```

W4A4（`linear_dtype="int4"`、sm8x）:

```text
sm major == 8
  quantize_int4_rowwise_convrot64
  if prefer_legacy_int4(M, N, K, sm_count)  -> launch_int4_linear_kernel
  else                                      -> launch_cutlass_int4_dequant[_config]
```

W4A4（非 sm8x = Hopper / Blackwell）:

```text
sm major != 8
  quantize_int8_rowwise_convrot64            # 活性は INT8（W4A8）
  if M <= 8   -> launch_int4_weight_int8_act_gemv_dequant_kernel
  else        -> launch_int4_weight_int8_act_gemm_dequant_chunked_kernel
```

- `sm major == 8` は Comfy の `_cuda_device_supports_native_int4_mma` と同じ判定（Ada/Ampere は INT4 MMA、Hopper 以降は W4A8）。
- **W4A4 は M で W4A8 に落ちません。** sm8x では M=1 でも INT4 量子化 + INT4 カーネルです（Comfy の実装に一致させる）。
- `prefer_legacy_int4` は §2.5 の式。`gemm_config >= 0` が指定されている場合は legacy 判定をスキップして CUTLASS を強制してよい（`-1` のときだけ Comfy と同じ判定）。
- `K % 64 != 0` のときは W4A8 に落とす（v1 では `K % 256 == 0` を要求しているので実質発生しない）。
- W4A8 経路は sm8x 以外でのみ使うため、「M ≤ 8 で GEMV」は Hopper/Blackwell の話です。誤って sm8x で W4A8 を選ぶと、数値は一致するが遅くなります。

#### 5.3.2 W8A8 の config

- Comfy の `choose_fused_int8_config(M, N, K)` は **決定的なヒューリスティック**（timing を使わない）。そのまま使う。ただし `device_l2_bytes()` と SM 数（`wave_guard`）に依存するため、**同一 GPU 上でのみ再現**する。CUDA Graph / timing cache のキーに GPU 名を含める。
- `gemm_config == -1` のときは `launch_cutlass_int8_dequant`（既定ラッパー）をそのまま呼ぶ。これは内部で `launch_fused_int8_heuristic` を通し、選んだ config が `can_implement` で落ちた場合に `{2,12,0,13,1,6,8,7,3,4,5}`（`K%16==0`）／`{9,10,11}`（低アラインメント）の順でフォールバックする。このフォールバック列を自前で書き直さない。
- `gemm_config >= 0` のときだけ `launch_cutlass_int8_dequant_config` を呼ぶ。
- `launch_cutlass_int8_dequant_config` は現状 BF16・bias なし限定（`dispatch_fused_config` が bias に `nullptr` を渡す）なので、bias / FP16 / FP32 を扱えるよう拡張する（`dispatch_fused_config` に bias 引数を足すだけ）。

#### 5.3.3 W4A4 の config（runtime autotune を排除）

Comfy の `dispatch_fused_int4` は **初回呼び出しで 7 候補を実測**します（各 config を 8 回 + 32 回実行し `cudaStreamSynchronize`）。これを `enqueue` 内で走らせると:

- 初回推論が数十 ms 固まる
- CUDA Graph capture 中なら capture が壊れる
- TRT の timing cache と二重管理になる

そこで **ビルド時に一度だけ計測して `gemm_config` フィールドに固定**します。

1. プラグインの `convrot_int4_autotune(M, N, K, out_dtype, stream, workspace)` を用意。中身は Comfy の autotune ループと同じ（7 config をダミーバッファで計測し最小の index を返す）。
2. `python/trt_dit_plugins/autotune.py` がエンジンビルド前に、対象レイヤの `(M, N, K)` についてこの関数を呼び、結果を ONNX ノードの `gemm_config` に書き込む。
3. `enqueue` は `gemm_config` をそのまま使う。`-1` のときは保守的な既定（下表）にフォールバックし、**計測はしない**。

既定 config（autotune 未実施時のフォールバック）:

| 条件 | config | タイル |
|---|---:|---|
| `M >= 256` | 1 | 128×256×256 |
| `64 <= M < 256` | 5 | 128×128×256 |
| `M < 64` | 6 | 64×256×256 |
| `N >= 4096 && M >= 1024` | 3 | 256×256×256 |

- どの config でも `acc` はビット一致するので、これは純粋に速度の選択です。
- autotune 結果は `(M, N, K, out_dtype, arch)` をキーに JSON キャッシュし、リポジトリの `python/trt_dit_plugins/configs/` に置いて再利用する。

**vendor 時に残すべき上流の挙動**（パッチで消してはいけないもの）:

1. `dispatch_fused_int4` の先頭にある `M == 4608 && N == 3072 && K == 15360 → runners[0]` の固定分岐。これは特定モデル向けの実測済みショートカットなので、autotune を外しても残す（`convrot_int4_autotune` もこの形状では 0 を返す）。
2. `dispatch_fused_int4_no_bias` 側にも同じ 7 候補がある（bias なしの専用 epilogue）。`launch_cutlass_int4_dequant_config` は `has_bias` で両者を切り替える。
3. 上流の cache / `thread_local last_*` は **削除する**。グローバル可変状態を持ち越すと、複数エンジン・複数 stream で「直前の形状の最速 config」が使われ、再現性が失われる。config は呼び出しごとの引数で渡す。

#### 5.3.4 TRT tactic を使う案（v2）

TRT 11 の `IPluginV3OneRuntime` には tactic 機構があるため、config を tactic として公開し TRT のビルダーに選ばせる実装も可能です。ただし動的 M では 1 エンジン 1 tactic になり、M=1 と M=4096 で最適が異なる問題を解決できません。v1 は「M 依存分岐 + ビルド時 autotune」で固定し、tactic 化は v2 の課題とします。

### 5.4 enqueue のフロー

W8A8（M ≥ 3）:

```text
[x が BF16 でない] cast -> BF16
quantize_int8_rowwise_convrot64(x, x_q, x_scale, M, K, 256, in_dtype, false, kActNone, 0, nullptr, 0.f, stream)
launch_cutlass_int8_dequant[_config](x_q, W, x_scale, w_scale, bias?, y, M, N, K, out_dtype, [config,] stream)
```

W4A4（sm8x）:

```text
[x が BF16 でない] cast -> BF16
quantize_int4_rowwise_convrot64(x, x_q4, x_scale, M, K, 256, in_dtype, false, 0, stream)
if legacy: launch_int4_linear_kernel(x_q4, W4, x_scale, w_scale, bias?, y, M, N, K, has_bias, out_dtype, bias_dtype, stream)
else:      launch_cutlass_int4_dequant[_config](x_q4, W4, x_scale, w_scale, bias?, y, M, N, K, out_dtype, [config,] stream)
```

W4A4（非 sm8x）:

```text
[x が BF16 でない] cast -> BF16
quantize_int8_rowwise_convrot64(x, x_q8, x_scale, M, K, 256, in_dtype, false, kActNone, 0, nullptr, 0.f, stream)
if M <= 8: launch_int4_weight_int8_act_gemv_dequant_kernel(x_q8, W4, x_scale, w_scale, bias?, y, M, N, K, N, has_bias, out_dtype, bias_dtype, stream)
else:      launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(x_q8, W4, x_scale, w_scale, bias?, y, w_chunk, acc, nullptr, 0, M, N, K, N, chunk_cols, true, has_bias, out_dtype, bias_dtype, stream)
```

- `act_code = kActNone`、`act_weight = nullptr`、`act_eps = 0`、`seed = 0`。
- residual 系の引数はすべて `nullptr`。
- `weight_scale_size` は `N`（`[N]` へ reshape 済みであること）。
- 量子化カーネルは `bool stochastic` を取る。常に `false`。
- `launch_int4_weight_int8_act_gemm_dequant_chunked_kernel` の `chunk_cols` は Comfy の `_int4_int8_weight_chunk_cols` と同じ規則で決める（§5.5）。

### 5.5 workspace

| 経路 | 必要なもの | サイズ |
|---|---|---|
| W8A8 M=1 | なし（融合 GEMV） | 0 |
| W8A8 M=2 | `x_q`, `x_scale` | `M*K` + `M*4` |
| W8A8 M≥3 | `x_q`, `x_scale` | `M*K` + `M*4` |
| W4A4 sm8x（legacy / CUTLASS） | `x_q4`, `x_scale` | `M*K/2` + `M*4` |
| W4A4 非 sm8x, M≤8（W4A8 GEMV） | `x_q8`, `x_scale` | `M*K` + `M*4` |
| W4A4 非 sm8x, M>8（W4A8 chunked GEMM） | `x_q8`, `x_scale`, `weight_chunk`(INT8 展開), `acc` | 下記 |
| x が非 BF16 | BF16 コピー | `M*K*2` |

- W4A8 チャンク GEMM は「重みを INT8 に展開したチャンク」と `acc` を要求します。Comfy の規則（`_int4_int8_weight_chunk_cols`）は `chunk_cols = N`（`N <= 2560`）／`min(N, 4096)`（それ以外）。L2 が 16 MB 未満の GPU では `chunk_cols = N`。必要サイズは `chunk_cols * K`（weight、INT8）と `M * chunk_cols * 4`（acc）。
- `getWorkspaceSize` は **最大 profile** で計算する。`M` は `x` の先頭次元の積。
- 全バッファは 256 byte アラインで切り出す。

### 5.6 ラッパーの共通化

W8A8 と W4A4 は「入力の解釈」と「経路表」だけが違います。`template <int WeightBits>` で 1 クラスにまとめ、`WeightBits` で `K` の復元（`K` vs `K*2`）と経路表を切り替えます（付録 B）。プラグイン名・creator は 2 つ登録します。

### 5.7 Python / ONNX surgery

1. レイヤごとに `quant_format` / `params.convrot` / `params.convrot_groupsize` /
   `params.transposed` を読み、下表で op を決める。**「INT8 だから ConvRot」とは判定しない。**

   | `quant_format` | `convrot` | `transposed` | 結果 |
   |---|---|---|---|
   | `convrot_w4a4` | （暗黙に True） | False | `convrot_linear_w4a4` |
   | `int8_tensorwise` | True | False | `convrot_linear` |
   | `int8_tensorwise` | True | True | **fail-open**（`_qdata` が `[K,N]`。v1 非対応） |
   | `int8_tensorwise` | False / 無し | — | **fail-open**（回転していない。Comfy の `int8_linear(convrot=False)` 相当は v1 非対応） |
   | それ以外（fp8 / nvfp4 等） | — | — | 対象外 |

   実測: Anima の ConvRot INT8 は **640/640 が `int8_tensorwise` + `convrot:true` + groupsize 256**
   なので、このチェックポイントでは取り違えは起きない。ただし Comfy の
   `int8_tensorwise` は `convrot` 無しでも正規（`comfy/ops.py` は
   `getattr(params, "convrot", False)`）なので、他チェックポイントでは両者が混在し得る。
   重みのバイト列だけから回転の有無は判別できないため、**この判定は export 側の責務**であり、
   プラグインは「`convrot_linear*` に渡る重みは回転済み」という契約だけを持つ（§2.6）。
2. `weight` / `weight_scale` / `bias` を initializer として取り込み、`dit-plugins::convrot_linear[_w4a4]` ノードを 1 個生成。
3. 既存の Q/DQ（`QuantizeLinear` / `DequantizeLinear`）は削除し、プラグイン出力をそのまま次段へ。
4. `gemm_config` は `autotune.py` の結果を属性として埋める。
5. 対応しない形状（`K > 16384` 等）は **変換を中止して BF16 パスを残す**（fail-open）。中途半端に一部だけ量子化しない。

### 5.8 重みの変換（BF16 → W4A4）

Anima に公式の W4A4 チェックポイントは無いため、オフライン変換スクリプトを用意します。

```python
import torch, comfy_kitchen as ck

def to_w4a4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # w: [N, K] BF16 (回転前)。K % 256 == 0
    q, s = ck.quantize_convrot_w4a4_weight(w.contiguous(), 256)
    return q, s.reshape(-1)          # q: [N, K/2] int8, s: [N] fp32
```

- 変換は **必ず BF16 重みに対して行う**（INT8 重みから作ると二重量子化になる）。
- 生成物は `weight`（int8 `[N,K/2]`）+ `weight_scale`（fp32 `[N]`）+ `comfy_quant` JSON。
- 変換後は元の BF16 重みを残さない（サイズ削減の意味が無くなる）。

### 5.9 ビルド

上流 `comfy_kitchen/backends/cuda/CMakeLists.txt`（v0.2.39）のフラグを**そのまま**持ってきます。ここを「きれいに」すると数値が変わります。

```cmake
# src/kernel/CMakeLists.txt（抜粋）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(COMFY_VENDOR
    vendor/comfy/ops/int8_linear.cu
    vendor/comfy/ops/cutlass_gemm_int8.cu
    vendor/comfy/ops/convrot_w4a4.cu
    vendor/comfy/ops/w4a8_gemm.cu
    convrot_shim.cu)

# 上流と同一。落とすと __half / __nv_bfloat16 の演算子が無効化されてコンパイルが通らない。
set(CUDA_NVCC_FLAGS
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__
    -U__CUDA_NO_BFLOAT162_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math          # ← 全 TU。量子化だけに絞らない
    --compress-mode=size
    -O3)

add_library(ck_kernels SHARED ${COMFY_VENDOR})
target_compile_options(ck_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(ck_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/comfy
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/comfy/ops
    ${CUTLASS_DIR}/include)
target_compile_definitions(ck_kernels PRIVATE COMFY_HAVE_CUTLASS)

# comfy は wheel 内で完結させるため cudart_static を使うが、
# プラグインは TensorRT ホストプロセス（shared cudart）に dlopen されるので
# **shared cudart** をリンクする。static を混ぜるとランタイムが二重化する。
target_link_libraries(ck_kernels PRIVATE CUDA::cudart)
```

- CUTLASS は `third_party/cutlass` に submodule 固定（`d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`）。
- `cutlass_gemm_int8.cu` / `convrot_w4a4.cu` は CUTLASS 3.x の EVT（`default_gemm_universal_with_visitor.h`）を使うため C++20。
- `COMFY_HAVE_CUTLASS` が無い場合のフォールバックは実装しない（ビルドエラーにする）。速度要件を満たせないため。
- `--compress-mode=size` はファットバイナリ削減のため。命令列には影響しない（`-O3` のまま）。
- 我々の `convrot_shim.cu` は fast-math を必要としないが、TU 単位で分けず同一フラグで問題ない（カーネルを含まないため）。

### 5.9.1 ビルド時間

CUTLASS の EVT は 1 TU あたりのインスタンス化が重く、`convrot_w4a4.cu`（INT4 7 config × 3 dtype + W4A8）が最も重い TU になります。

- `-DCMAKE_CUDA_ARCHITECTURES` を配布対象（`89;120` など）に限定する。既定の `all` は避ける。
- ccache / sccache を有効化（上流も `COMFY_CUDA_COMPILER_LAUNCHER` で対応している）。
- CI では `convrot_w4a4.cu` を最初にビルドして失敗を早期検出する。

### 5.10 ライセンス

- comfy-kitchen は **Apache-2.0**。vendor する場合:
  - `LICENSE`（Apache-2.0 全文）と `NOTICE` を wheel に同梱
  - 改変したファイルに「変更した旨」を明記（Apache-2.0 §4(b)）
  - `VENDOR.md` に上流リポジトリ・コミット・改変一覧
- CUTLASS は BSD-3-Clause。`third_party/cutlass/LICENSE.txt` を同梱。
- wheel の `METADATA` に `License: Apache-2.0` を維持し、`NOTICE` を `dist-info` に入れる。

### 5.11 ComfyUI-TensorRT-Reforge への適用

- `src/comfyui_tensorrt_reforge/onnx_ops/qdq_int8.py` の Q/DQ 挿入を、ConvRot レイヤでは `convrot_linear` ノードに置き換える。
- `comfy_kitchen.int8_linear` / `convrot_w4a4_linear` の ONNX symbolic を登録し、export 時の `DispatchError` を解消する。
- W4A4 は既定 off。`--convrot-w4a4` フラグ（または設定ファイル）で有効化し、E2E cos が閾値を下回る場合は自動で W8A8 に戻す。

---

## 6. テスト計画

### 6.1 カーネル単体（shim）

| テスト | 内容 | 合否 |
|---|---|---|
| W8A8 quant | 実レイヤ 3 本 + 合成形状で `q`, `scale` を Comfy と比較 | `q_eq == 1.0`, `scale_exact == 1.0` |
| W4A4 quant | 同上（`[M, K/2]` のパック済みバイトを比較） | 完全一致 |
| W8A8 GEMM | `acc` を `torch._int_mm` と比較 | 完全一致 |
| W4A4 GEMM | 各 config で `acc` を比較 | config 間で一致 |
| 経路分岐（W8A8） | M=1/2/3 で融合 GEMV / GEMV dequant / CUTLASS が選ばれること | ログで確認 |
| 経路分岐（W4A4 sm8x） | M=1/8/9/4096 で legacy と CUTLASS の選択が Comfy の `_prefer_legacy_int4_kernel` と一致すること | ログで確認 |
| 経路分岐（W4A4 非 sm8x） | M=8/9 で W4A8 GEMV / chunked GEMM が切り替わること | ログで確認（Hopper 実機または強制フラグ） |

### 6.2 full linear parity

```text
W8A8: shim(x, W, ws, bias) == comfy_kitchen.int8_linear(x, W, ws, bias, convrot=True)
W4A4: shim(x, W4, ws, bias) == comfy_kitchen.convrot_w4a4_linear(x, W4, ws, bias)
```

- `max_abs == 0`, `cos == 1.0`。
- 入力 dtype: BF16 / FP16 / FP32 の 3 種（非 BF16 は BF16 丸め後の比較）。
- out_dtype: 0/1/2。
- bias あり / なし。

### 6.3 プラグイン E2E

- TRT エンジンをビルドし、同一入力で PyTorch 経路と比較。
- ゲート:
  - W8A8: `cos >= 0.99999`（実測は 1.0 相当）
  - W4A4: `cos >= 0.995`（レイヤ単位）。E2E 画像比較は別途。
- Anima の 1 ステップ推論で、プラグイン経路と Comfy 経路の出力差を測る。
- CUDA Graph capture を有効にした状態で 2 回目以降が正しく動くこと。

### 6.4 ネガティブテスト

- 変換側: `int8_tensorwise` かつ `convrot=False` のレイヤが `convrot_linear` にならず、
  BF16 のまま残る（**静かに壊れた INT8 グラフを出さない**）
- 変換側: `params.transposed=True` のレイヤも同様に fail-open する
- `group_size != 256` → configure で失敗
- `K % 256 != 0` / `K > 16384` → 失敗
- W4A4 で `weight.shape[1] * 2 != x.shape[-1]` → 失敗
- `N % 32 != 0`（W4A4）→ INT4 MMA 経路へフォールバックするか、失敗
- `out_dtype` が -1/0/1/2 以外 → 失敗
- `gemm_config` が範囲外 → 既定にフォールバック（クラッシュしない）
- workspace 不足 → 失敗（`getWorkspaceSize` と一致していること）

### 6.4.1 テスト用 ONNX（実 Anima 重み）

`ComfyUI-TensorRT-Reforge` 側で生成した ONNX 6 本を `CONVROT_ONNX_DIR` に置き、
`pytest python/tests/test_convrot_onnx.py` で TRT のビルド・実行まで検証します。

| ファイル | 判定表の行 | ゲート |
|---|---|---|
| `convrot_layer1/2/q_proj.onnx`, `convrot_all.onnx` | `int8_tensorwise` / convrot=true / transposed=false | comfy-kitchen とビット一致（M=1,2,3,8,9,64,128） |
| `failopen_nonconvrot_q_proj.onnx` | convrot=false | プラグイン 0 ノード、BF16 dense に cos >= 0.999999 |
| `failopen_transposed_q_proj.onnx` | transposed=true | 同上 |

`fixtures.npz` のキーは `<tensor_name>_m<M>`。プロファイルは min 1 / opt 128 / max 4096。

### 6.5 性能テスト

- `enqueue` 内に同期が無いこと（`cudaStreamSynchronize` / `cudaEventSynchronize` を shim に置かない。autotune は別関数）。
- 1 回目と 2 回目のレイテンシ差が 10% 以内（autotune が混入していない確認）。
- §4.5 の目標値。

### 6.6 コマンド

```bash
# 1. build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2. registration
./build/test/test_register

# 3. python tests
pytest python/tests/test_convrot.py -v

# 4. parity（実チェックポイント）
CONVROT_CKPT=/opt/ComfyUI/models/diffusion_models/Anima-2.9B-preview-v1_int8_convrot.safetensors \
pytest python/tests/test_convrot.py -k real -v

# 5. autotune（gemm_config の確定）
python -m trt_dit_plugins.autotune --shapes shapes.json --out configs/
```

---

## 7. タスク分解

### 7.1 マイルストーン

| # | 内容 | 完了条件 |
|---|---|---|
| M0 | vendor ビルド | comfy の 4 ファイル + CUTLASS で `ck_kernels` がビルドでき、shim 経由で W8A8 が 1 レイヤ通る |
| M1 | W8A8 プラグイン | `convrot_linear` が TRT で動き、Comfy と `max_abs == 0` |
| M2 | W4A4 プラグイン | `convrot_linear_w4a4` が sm8x で動き、Comfy と一致。W4A8 フォールバックも通る |
| M3 | config 固定 | `autotune.py` で `gemm_config` を確定。`enqueue` に同期が無いことをテストで担保 |
| M4 | E2E 統合 | ComfyUI-TensorRT-Reforge で Anima を TRT 実行、W8A8 / W4A4 の cos と速度を記録 |

### 7.2 作業一覧

1. vendor コピー + `VENDOR.md` + ライセンス同梱
2. `cublas_gemm_int8` 依存の除去（W4A8 フォールバックの分岐変更）
3. `launch_cutlass_int8_dequant_config` の bias/FP16/FP32 対応
4. `launch_cutlass_int4_dequant_config` の追加（7 config を index で呼ぶ）
5. `convrot_shim.cu`: 量子化 / GEMM / GEMV / autotune の C ラッパー
6. `convrot_linear_wrapper.cpp`: `template <int WeightBits>` + 2 creator
7. `python/`: `select.py`, `_lib.py`, `surgeon.py`, `ops.py`, `autotune.py`
8. `test/test_register.cpp`, `python/tests/test_convrot.py`
9. ONNX symbolic（`comfy_kitchen::int8_linear`, `comfy_kitchen::convrot_w4a4_linear`）
10. TensorRT-Reforge 側の Q/DQ バイパス + `--convrot-w4a4`
11. ドキュメント（README、この計画書の更新）

---

## 8. リスク

### 8.1 数値

| リスク | 影響 | 対策 |
|---|---|---|
| vendor 元の変更で数値が変わる | parity が壊れる | 上流コミットを固定。更新時は必ず parity テスト |
| `out_dtype` の不一致（ONNX 型 vs プラグイン） | 出力型エラー | `surgeon` で `out_dtype` と `out_dtype_code` を同時に決める |
| FP16/FP32 入力の丸め | 期待と違う | v1 は「BF16 に丸めてから量子化」を仕様として明記 |

### 8.2 性能

| リスク | 影響 | 対策 |
|---|---|---|
| `enqueue` 内 autotune | 初回数十 ms、CUDA Graph 破壊 | config 固定（§5.3.3）。テストで同期の不在を確認 |
| M が動的で経路が合わない | 遅い経路に落ちる | M 依存分岐は enqueue で行う（build 時に固定しない） |
| 非 sm8x で W4A8 に落ちる | W4A4 より遅い | 仕様として許容。README に明記 |
| CUTLASS のビルド時間増 | CI が遅くなる | ccache、`-DCMAKE_CUDA_ARCHITECTURES` の限定 |

### 8.3 統合

| リスク | 影響 | 対策 |
|---|---|---|
| CUTLASS submodule の追加 | リポジトリが重くなる | shallow clone + タグ固定。wheel にはビルド成果物のみ |
| Windows ビルド | export 漏れ | `WINDOWS_EXPORT_ALL_SYMBOLS` 維持 |
| ComfyUI 側の量子化形式変更 | 検出失敗 | `comfy_quant` JSON を読み、未知フォーマットは fail-open |
| W4A4 の品質劣化 | 画質低下 | 既定 off。E2E cos ゲート + 自動フォールバック |

### 8.4 未決事項

1. `gemm_config` の autotune をどこで実行するか（エンジンビルド前の Python か、TRT の tactic か）。
2. W4A4 の既定化可否（E2E の画質評価待ち）。
3. Anima 以外（Wan / Minimax H3 など）の形状で `K > 16384` が出るか。出るなら Hopper の 227 KB opt-in を前提にした上限緩和か、K 分割を検討。
4. 上流 comfy-kitchen への還元（config 固定 API を PR するか、vendor のままか）。

---

## 9. 受入チェックリスト

- [ ] `ck_kernels` が CUTLASS 込みでビルドできる（Linux / Windows）
- [ ] `convrot_linear` が TRT で動く
- [ ] `convrot_linear_w4a4` が sm8x で動く
- [ ] 非 sm8x で W4A8 経路にフォールバックする
- [ ] W8A8 が Comfy と `max_abs == 0`
- [ ] W4A4 が Comfy と `max_abs == 0`
- [ ] M=1/2/3/8/9 の経路分岐が期待通り
- [ ] `enqueue` に同期呼び出しが無い
- [ ] `getWorkspaceSize` が全経路を網羅
- [ ] `gemm_config` の autotune 結果が再現可能（キャッシュ）
- [ ] ネガティブテストが全て失敗側に倒れる
- [ ] 性能が §4.5 の目標内
- [ ] `NOTICE` / `LICENSE` / `VENDOR.md` が wheel に入る
- [ ] 未知の量子化フォーマットで fail-open する
- [ ] README に W4A4 の品質注意と有効化方法を記載

---

## Appendix A: vendor する API（実シグネチャ）

すべて `extern "C"`、`torch` 非依存。`in_dtype` / `out_dtype` は comfy の dtype code（0=FP32, 1=FP16, 2=BF16）。

### A.1 W8A8 量子化

`ops/int8_linear.cu`

```cpp
void launch_quantize_int8_rowwise_convrot64_kernel(
    const void* input, void* output, void* scales,
    int64_t num_rows, int64_t num_cols, int group_size,
    int input_dtype_code, bool stochastic, int act_code, uint64_t seed,
    const void* act_weight, float act_eps, cudaStream_t stream);
```

- `output`: INT8 `[M, K]`、`scales`: FP32 `[M]`
- `group_size == 256`、`K % 256 == 0`
- `stochastic=false`, `act_code=0`(kActNone), `act_weight=nullptr`, `act_eps=0`

### A.2 W8A8 GEMM（融合 dequant）

`ops/cutlass_gemm_int8.cu`

```cpp
bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream);

int cutlass_int8_selected_config(int64_t M, int64_t N, int64_t K);
```

- `A`: INT8 `[M,K]`、`B`: INT8 `[N,K]`（転置済み重み）、`xs`: FP32 `[M]`、`ws`: FP32 `[N]`
- `bias` は `D` と同じ dtype。nullptr 可（epilogue が 0 を broadcast）
- 戻り値 `false` は「この形状は CUTLASS で扱えない」。上位で失敗扱いにする

**パッチ（追加）**: `launch_cutlass_int8_dequant_config` を bias / FP16 / FP32 対応に拡張する。

```cpp
// DIT-PLUGINS-PATCH: bias と全 out dtype を * _config でも扱えるようにする
bool launch_cutlass_int8_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, int config,
    cudaStream_t stream);
```

### A.3 W8A8 GEMV（M=1 / M=2）

```cpp
// M=1: 量子化を内包した 1 カーネル
void launch_int8_gemv_convrot_fused_kernel(
    const void* input, int input_dtype_code, int act_code, const void* act_weight,
    float act_eps, const void* weight, const void* weight_scales, const void* bias,
    bool has_bias, int bias_dtype_code, void* output, int output_dtype_code,
    int64_t N, int64_t K, int64_t weight_scale_size,
    const void* residual, const void* residual_scale, cudaStream_t stream);

// M=2: 量子化済みの x_q を渡す
void launch_int8_gemv_dequant_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t num_rows, int64_t num_cols, int64_t K,
    int64_t weight_scale_size, bool has_bias, int output_dtype_code, int bias_dtype_code,
    const void* residual, const void* residual_scale, cudaStream_t stream);
```

- M=1 融合 GEMV は `weight` が 16 byte アラインされていること、および `smem = (K + 512) * 4 + K` 程度の余裕が必要。`weight.data_ptr() % 16 != 0` の場合は A.1 + A.2 にフォールバックする。

### A.4 W4A4 量子化

`ops/convrot_w4a4.cu`

```cpp
void launch_quantize_int4_rowwise_convrot64_kernel(
    const void* input, void* output, void* scales,
    int64_t M, int64_t K, int group_size,
    int input_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream);
```

- `output`: INT8 `[M, K/2]`（パック済み INT4）、`scales`: FP32 `[M]`
- `group_size ∈ {16, 64, 256}`、`K % group_size == 0`

### A.5 W4A4 GEMM（融合 dequant）

```cpp
bool launch_cutlass_int4_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream);

// sm8x の低レイテンシ経路（CUTLASS を使わない）。CUTLASS が false を返したときの受け皿。
void launch_int4_linear_kernel(
    const void* act, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t M, int64_t N, int64_t K,
    bool has_bias, int output_dtype_code, int bias_dtype_code, cudaStream_t stream);
```

- `launch_cutlass_int4_dequant` は `AlignA = AlignB = 32`（int4 要素）を要求し、満たさないと `false` を返します。`false` のときは `launch_int4_linear_kernel` に落とす（Comfy の `int4_linear` と同じ）。
- `launch_int4_linear_kernel` は `K % 64 != 0` で **何もせず return** します（`void` なのでエラーが返らない）。呼び出し前に shim で `K % 64 == 0` を検証すること。

**パッチ（追加）**: autotune を排除するため config 指定版を足す。

```cpp
// DIT-PLUGINS-PATCH: runtime autotune の代わりに config を固定で選ぶ
bool launch_cutlass_int4_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, int config,
    cudaStream_t stream);

// config 一覧（comfy の runners[] と同順）
// 0: 128x256x128 w64x64x128 s3    1: 128x256x256 w64x64x256 s3
// 2: 128x512x256 w64x128x256 s3   3: 256x256x256 w64x128x256 s3
// 4: 256x128x256 w64x64x256 s3    5: 128x128x256 w64x64x256 s3
// 6:  64x256x256 w64x64x256 s3
```

パッチ内容は `dispatch_fused_int4` / `dispatch_fused_int4_no_bias` の `runners[]` を index で呼ぶだけです（`cutlass_gemm_int8.cu` の `dispatch_fused_config` と同じ形）。`dispatch_fused_int4_no_bias` は bias 用 visitor を持たない別インスタンスなので、`has_bias` で使い分けます（bias に 0 を足すより速い）。

### A.6 W4A8（フォールバック）

```cpp
// 非 sm8x かつ M <= 8
void launch_int4_weight_int8_act_gemv_dequant_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t num_rows, int64_t num_cols, int64_t K,
    int64_t weight_scale_size, bool has_bias, int output_dtype_code, int bias_dtype_code,
    cudaStream_t stream);

// 非 sm8x かつ M > 8
void launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output,
    void* weight_workspace, void* acc_workspace,
    void* cublas_workspace, int64_t cublas_workspace_size,
    int64_t num_rows, int64_t num_cols, int64_t K, int64_t weight_scale_size,
    int64_t chunk_cols, bool allow_sm80_cutlass,
    bool has_bias, int output_dtype_code, int bias_dtype_code, cudaStream_t stream);
```

- `allow_sm80_cutlass = true`（`K % 256 == 0` なので常に CUTLASS INT8 経路が使える）
- パッチで `cublas_workspace` / `cublas_workspace_size` を削除し、cuBLAS フォールバックをエラーにする

### A.7 shim（我々が追加する C API）

```cpp
// src/kernel/convrot_shim.h
int convrot_in_dtype_code(int trt_dtype);    // 0=bf16, 1=fp16, 2=fp32
int convrot_out_dtype_code(int trt_dtype);
int convrot_arch_major(int device);          // 8 = INT4 MMA 可

size_t convrot_workspace_bytes(int weight_bits, int64_t M, int64_t N, int64_t K, int in_dtype);

// 量子化（act_bits: 8 → convrot64 INT8、4 → convrot64 INT4 パック）
int convrot_quant(const void* x, int8_t* q, float* scales, int64_t M, int64_t K,
                  int in_dtype, int act_bits, cudaStream_t stream);

// 融合 GEMM（量子化済み x_q を渡す）
// weight_bits と act_bits は独立。非 sm8x の W4A4 は (weight=4, act=8)。
int convrot_gemm(void* y, const int8_t* w, const float* ws, const void* bias,
                 const int8_t* xq, const float* xs, int64_t M, int64_t N, int64_t K,
                 int weight_bits, int act_bits, int out_dtype, int bias_dtype, int config,
                 void* ws_scratch, size_t ws_scratch_bytes, cudaStream_t stream);

// M=1 の融合 GEMV（W8A8 のみ）
int convrot_gemv_m1(void* y, const void* x, const int8_t* w, const float* ws, const void* bias,
                    int64_t N, int64_t K, int in_dtype, int out_dtype, int bias_dtype,
                    cudaStream_t stream);

// ビルド時 autotune（enqueue からは呼ばない）
int convrot_int4_autotune(int64_t M, int64_t N, int64_t K, int out_dtype,
                          void* scratch, size_t scratch_bytes, cudaStream_t stream);
```

**例外の扱い**: comfy のランチャーは引数不整合で `std::runtime_error` を投げます（例: `int8 fused convrot GEMV requires a 16-byte aligned weight`、`K % 256 != 0`）。`enqueue` は `noexcept` なので、shim の各関数は `try { ... } catch (...) { return -1; }` で **必ずエラーコードに変換**します。TRT 側は `-1` を受けて `INFER_FAILED` 相当として扱います。`__CUDA_ARCH__` 非対応の sm で `-1` を返すのも同じ経路です。

---

## Appendix B: IPluginV3 ラッパー

`src/wrapper/convrot_linear_wrapper.cpp`。W8A8 と W4A4 を `WeightBits` で共通化した完全版です。
TRT 11 の `IPluginV3` + `IPluginV3OneCore` + `IPluginV3OneBuild` + `IPluginV3OneRuntime` を実装します。

```cpp
// SPDX-License-Identifier: Apache-2.0
// ConvRot W8A8 / W4A4 linear plugin (kernels vendored from comfy-kitchen).

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "convrot_shim.h"

namespace {

constexpr int kX = 0, kW = 1, kWS = 2, kB = 3, kNbIn = 4;
constexpr size_t align256(size_t v) { return (v + 255u) & ~size_t(255u); }

int64_t numel(nvinfer1::Dims const& d) {
    int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= std::max<int64_t>(1, d.d[i]);
    return n;
}

int64_t leadingNumel(nvinfer1::Dims const& d) {
    int64_t n = 1;
    for (int i = 0; i + 1 < d.nbDims; ++i) n *= std::max<int64_t>(1, d.d[i]);
    return n;
}

bool isFloat(nvinfer1::DataType t) {
    return t == nvinfer1::DataType::kBF16 || t == nvinfer1::DataType::kHALF ||
           t == nvinfer1::DataType::kFLOAT;
}

nvinfer1::DataType outType(nvinfer1::DataType x, int32_t code) {
    switch (code) {
        case 1: return nvinfer1::DataType::kHALF;
        case 2: return nvinfer1::DataType::kFLOAT;
        case 0: return nvinfer1::DataType::kBF16;
        default: return x;  // -1 = same as input (Comfy's `out_dtype or x.dtype`)
    }
}

// ---------------------------------------------------------------- plugin ---

template <int WeightBits>
class ConvRotLinear final : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
    static_assert(WeightBits == 8 || WeightBits == 4, "WeightBits must be 8 or 4");

public:
    static constexpr char const* name() {
        return WeightBits == 8 ? "convrot_linear" : "convrot_linear_w4a4";
    }

    ConvRotLinear(int32_t groupSize, int32_t hasBias, int32_t outDtype, int32_t gemmConfig)
        : groupSize_(groupSize), hasBias_(hasBias), outDtype_(outDtype), gemmConfig_(gemmConfig) {
        refreshFields();
    }

    ConvRotLinear(ConvRotLinear const& o)
        : groupSize_(o.groupSize_), hasBias_(o.hasBias_), outDtype_(o.outDtype_),
          gemmConfig_(o.gemmConfig_), M_(o.M_), K_(o.K_), N_(o.N_), timingID_(o.timingID_) {
        refreshFields();
    }

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override {
        if (type == nvinfer1::PluginCapabilityType::kCORE)
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        try { return new ConvRotLinear(*this); } catch (...) { return nullptr; }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return name(); }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return "dit-plugins"; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
                               nvinfer1::DataType const* in, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != kNbIn || !out || !in) return -1;
        out[0] = outType(in[kX], outDtype_);
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
                            nvinfer1::DimsExprs const*, int32_t,
                            nvinfer1::DimsExprs* out, int32_t nbOut,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        if (in[kX].nbDims < 2 || in[kW].nbDims != 2) return -1;
        out[0] = in[kX];
        out[0].d[out[0].nbDims - 1] = in[kW].d[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos,
                                   nvinfer1::DynamicPluginTensorDesc const* io,
                                   int32_t nbIn, int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !io || pos < 0 || pos > kNbIn) return false;
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR) return false;
        switch (pos) {
            case kX: return isFloat(d.type);
            case kW: return d.type == nvinfer1::DataType::kINT8;
            case kWS: return d.type == nvinfer1::DataType::kFLOAT;
            case kB: return d.type == outType(io[kX].desc.type, outDtype_);
            case kNbIn: return d.type == outType(io[kX].desc.type, outDtype_);
            default: return false;
        }
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
                            nvinfer1::DynamicPluginTensorDesc const* out,
                            int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        if (!checkDesc(in[kX].desc, in[kW].desc, in[kWS].desc, in[kB].desc, out[0].desc)) return -1;
        K_ = in[kW].desc.dims.d[1] * (WeightBits == 4 ? 2 : 1);
        N_ = in[kW].desc.dims.d[0];
        try { updateTimingID(); } catch (...) { return -1; }
        return 0;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
                            nvinfer1::DynamicPluginTensorDesc const*,
                            int32_t nbOut) const noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || in[kW].max.nbDims != 2) return 0;
        int64_t N = in[kW].max.d[0];
        int64_t K = in[kW].max.d[1] * (WeightBits == 4 ? 2 : 1);
        int64_t M = leadingNumel(in[kX].max);
        if (N <= 0 || K <= 0 || M <= 0) return 0;
        return convrot_workspace_bytes(WeightBits, M, N, K, dtypeCodeOf(in[kX].desc.type));
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
                          nvinfer1::PluginTensorDesc const* out, int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        if (!checkDesc(in[kX], in[kW], in[kWS], in[kB], out[0])) return -1;
        M_ = leadingNumel(in[kX].dims);
        K_ = in[kW].dims.d[1] * (WeightBits == 4 ? 2 : 1);
        N_ = in[kW].dims.d[0];
        if (M_ <= 0) return -1;
        try { updateTimingID(); } catch (...) { return -1; }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inDesc, nvinfer1::PluginTensorDesc const*,
                    void const* const* in, void* const* out, void* workspace,
                    cudaStream_t stream) noexcept override {
        try {
            if (!inDesc || !in || !out || M_ <= 0 || K_ <= 0 || N_ <= 0) return -1;

            const int inCode = dtypeCodeOf(inDesc[kX].type);
            const int outCode = outDtype_ < 0 ? inCode : outDtype_;
            const int biasCode = outCode;

            const int arch = convrot_arch_major(0);
            const bool int4Native = (WeightBits == 4) && (arch == 8);

            // M == 1 の融合 GEMV（W8A8 のみ）
            if (WeightBits == 8 && M_ == 1) {
                return convrot_gemv_m1(out[0], in[kX], static_cast<int8_t const*>(in[kW]),
                                       static_cast<float const*>(in[kWS]),
                                       hasBias_ ? in[kB] : nullptr, N_, K_, inCode, outCode,
                                       biasCode, stream) == 0 ? 0 : -1;
            }

            // workspace を切る
            const size_t quantElems = (int4Native && M_ > 8) ? (size_t)M_ * (size_t)K_ / 2
                                                             : (size_t)M_ * (size_t)K_;
            char* p = static_cast<char*>(workspace);
            auto* xq = reinterpret_cast<int8_t*>(p);
            p += align256(quantElems);
            auto* xs = reinterpret_cast<float*>(p);
            p += align256((size_t)M_ * sizeof(float));
            void* scratch = p;

            const int actBits = int4Native ? 4 : 8;
            if (convrot_quant(in[kX], xq, xs, M_, K_, inCode, actBits, stream) != 0)
                return -1;

            return convrot_gemm(out[0], static_cast<int8_t const*>(in[kW]),
                                static_cast<float const*>(in[kWS]), hasBias_ ? in[kB] : nullptr,
                                xq, xs, M_, N_, K_, WeightBits, actBits, outCode, biasCode,
                                gemmConfig_, scratch, 0, stream) == 0 ? 0 : -1;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        refreshFields();
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return timingID_.c_str(); }

private:
    static int dtypeCodeOf(nvinfer1::DataType t) {
        if (t == nvinfer1::DataType::kFLOAT) return 2;
        if (t == nvinfer1::DataType::kHALF) return 1;
        return 0;  // BF16
    }

    bool checkDesc(nvinfer1::PluginTensorDesc const& x, nvinfer1::PluginTensorDesc const& w,
                   nvinfer1::PluginTensorDesc const& ws, nvinfer1::PluginTensorDesc const& b,
                   nvinfer1::PluginTensorDesc const& y) const {
        if (x.dims.nbDims < 2 || w.dims.nbDims != 2) return false;
        if (!isFloat(x.type) || w.type != nvinfer1::DataType::kINT8) return false;
        if (ws.type != nvinfer1::DataType::kFLOAT) return false;
        if (b.type != outType(x.type, outDtype_)) return false;   // bias is read in the output dtype
        if (groupSize_ != 256 || hasBias_ < 0 || hasBias_ > 1) return false;
        if (outDtype_ < -1 || outDtype_ > 2) return false;

        const int64_t N = w.dims.d[0];
        const int64_t K = w.dims.d[1] * (WeightBits == 4 ? 2 : 1);
        if (N <= 0 || K <= 0) return false;
        if (K % 256 != 0 || K > 16384) return false;
        if (WeightBits == 8 && N % 4 != 0) return false;
        if (WeightBits == 4 && K % 64 != 0) return false;
        if (x.dims.d[x.dims.nbDims - 1] != K) return false;

        for (int i = 0; i + 1 < x.dims.nbDims; ++i)
            if (x.dims.d[i] <= 0) return false;

        if (numel(ws.dims) != N || numel(b.dims) != N) return false;
        if (y.type != outType(x.type, outDtype_)) return false;
        return true;
    }

    void refreshFields() {
        fields_[0] = nvinfer1::PluginField{"group_size", &groupSize_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[1] = nvinfer1::PluginField{"has_bias", &hasBias_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[2] = nvinfer1::PluginField{"out_dtype", &outDtype_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[3] = nvinfer1::PluginField{"gemm_config", &gemmConfig_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 4;
        fc_.fields = fields_;
    }

    void updateTimingID() {
        timingID_ = std::string(name()) + "/1/k" + std::to_string(K_) + "/n" +
                    std::to_string(N_) + "/g" + std::to_string(groupSize_) + "/b" +
                    std::to_string(hasBias_) + "/o" + std::to_string(outDtype_) + "/c" +
                    std::to_string(gemmConfig_);
    }

    int32_t groupSize_ = 256;
    int32_t hasBias_ = 1;
    int32_t outDtype_ = -1;
    int32_t gemmConfig_ = -1;
    int64_t M_ = 0, K_ = 0, N_ = 0;
    std::string timingID_;
    nvinfer1::PluginField fields_[4]{};
    nvinfer1::PluginFieldCollection fc_{};
};

// --------------------------------------------------------------- creator ---

int32_t readInt(nvinfer1::PluginFieldCollection const* fc, char const* key, int32_t fallback) {
    if (!fc || !fc->fields) return fallback;
    for (int i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (f.name && std::strcmp(f.name, key) == 0 &&
            f.type == nvinfer1::PluginFieldType::kINT32 && f.length >= 1 && f.data)
            return static_cast<int32_t const*>(f.data)[0];
    }
    return fallback;
}

template <int WeightBits>
class Creator final : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase) noexcept override {
        try {
            return new ConvRotLinear<WeightBits>(
                readInt(fc, "group_size", 256), readInt(fc, "has_bias", 1),
                readInt(fc, "out_dtype", -1), readInt(fc, "gemm_config", -1));
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static int32_t g = 256, b = 1, o = -1, c = -1;
        static nvinfer1::PluginField fields[4] = {
            {"group_size", &g, nvinfer1::PluginFieldType::kINT32, 1},
            {"has_bias", &b, nvinfer1::PluginFieldType::kINT32, 1},
            {"out_dtype", &o, nvinfer1::PluginFieldType::kINT32, 1},
            {"gemm_config", &c, nvinfer1::PluginFieldType::kINT32, 1}};
        static nvinfer1::PluginFieldCollection fc{4, fields};
        return &fc;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return ConvRotLinear<WeightBits>::name();
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return "dit-plugins"; }
};

struct AutoRegister {
    AutoRegister() {
        static Creator<8> c8;
        static Creator<4> c4;
        getPluginRegistry()->registerCreator(c8, "dit-plugins");
        getPluginRegistry()->registerCreator(c4, "dit-plugins");
    }
};
static AutoRegister gAutoRegister;

}  // namespace
```

### B.1 実装上の注意

- `checkDesc` は `configurePlugin` と `onShapeChange` の両方で呼ぶ。片方だけだと動的 shape で検証が抜ける。
- `configurePlugin` の先頭で `convrot_arch_major(0) < 8` なら `-1` を返す（Turing では CUTLASS INT8 が使えないため、ビルド時に落とす。実行時に落ちるより早い）。
- `M == 1` の融合 GEMV は W8A8 のみ（W4A4 の M=1 は INT4 経路）。W4A8 になるのは **非 sm8x のときだけ**で、そのとき `actBits = 8`、`weight_bits` は 4 のまま。**重みビットと活性ビットは独立**に扱うこと（1 つの `bits` にまとめると、Hopper で INT4 量子化を呼んでしまう）。

  | ケース | `weight_bits` | `actBits` | カーネル |
  |---|---:|---:|---|
  | W8A8 | 8 | 8 | 融合 GEMV / GEMV dequant / CUTLASS INT8 |
  | W8A4（存在しない） | 8 | 4 | エラー |
  | W4A4（sm8x） | 4 | 4 | legacy INT4 MMA / CUTLASS INT4 |
  | W4A8（非 sm8x のみ） | 4 | 8 | W4A8 GEMV（M≤8） / W4A8 chunked GEMM（M>8） |
- `gemmConfig_` は範囲外でもクラッシュさせず既定にフォールバックする（`convrot_gemm` 内で clamp）。
- `getWorkspaceSize` は `max` dims、`onShapeChange` は実 dims。混同するとバッファオーバーランする。
- `weight` が 16 byte アラインでない場合、M=1 融合 GEMV は使えない。`convrot_gemv_m1` 内で判定して A.1 + A.2 に落とす。

---

## Appendix C: Python / ONNX / CMake / テスト

### C.1 `select.py`

```python
class PluginOp(str, Enum):
    ...
    CONVROT_LINEAR = "convrot_linear"
    CONVROT_LINEAR_W4A4 = "convrot_linear_w4a4"

    __str__ = str.__str__


QUALITY_CLASS: dict[str, str] = {
    ...
    PluginOp.CONVROT_LINEAR: "quant",
    PluginOp.CONVROT_LINEAR_W4A4: "quant",
}

# 実測後に埋める（レイヤ単位の cos）
KNOWN_COS: dict[str, dict[int, float]] = {
    ...
    PluginOp.CONVROT_LINEAR: {17: 0.99999},
    PluginOp.CONVROT_LINEAR_W4A4: {17: 0.995},
}
```

### C.2 `_lib.py`

```python
PLUGINS = {
    ...
    PluginOp.CONVROT_LINEAR: "convrot_linear",
    PluginOp.CONVROT_LINEAR_W4A4: "convrot_linear_w4a4",
}
```

cuBLAS を使わないため `_LIB_NAMES` の追加は不要（`cudart` のみ）。

### C.3 `surgeon.py`

`add_convrot_linear` は W8A8 / W4A4 を `weight_bits` で切り替える 1 関数にします（コピペで 2 本作らない）。

```python
def _convrot_op(weight_bits: int):
    return (
        PluginOp.CONVROT_LINEAR if weight_bits == 8
        else PluginOp.CONVROT_LINEAR_W4A4
    )


def _like_convrot(ref, n, name=None, dtype=None):
    gs = _gs()
    shape = list(ref.shape) if ref.shape is not None else None
    if shape is not None and n is not None:
        shape[-1] = n
    return gs.Variable(
        name or f"dit_out{next(_uid)}",
        dtype=dtype if dtype is not None else ref.dtype,
        shape=shape,
    )


def add_convrot_linear(
    graph, x, weight, weight_scale, bias=None,
    *, weight_bits=8, group_size=256, has_bias=None,
    out_dtype=None, out_dtype_code=-1, gemm_config=-1, name=None,
):
    """ConvRot linear (W8A8 when weight_bits=8, W4A4 when weight_bits=4).

    weight: [N, K] INT8 (bits=8) / [N, K/2] INT8 packed (bits=4), already rotated
    """
    import numpy as np

    if weight_bits not in (4, 8):
        raise ValueError("convrot_linear: weight_bits must be 4 or 8")

    n = int(weight.shape[0]) if getattr(weight, "shape", None) is not None else None

    if bias is None:
        if n is None:
            raise ValueError("bias=None requires a static weight.shape[0]")
        # bias は出力 dtype。out_dtype 未指定なら入力 dtype。
        bias_dtype = out_dtype if out_dtype is not None else x.dtype
        bias = _gs().Constant(f"convrot_bias{next(_uid)}",
                              np.zeros((n,), dtype=bias_dtype))
        if has_bias is None:
            has_bias = False
    elif has_bias is None:
        has_bias = True

    fields = {
        "group_size": int(group_size),
        "has_bias": int(bool(has_bias)),
        "out_dtype": int(out_dtype_code),
        "gemm_config": int(gemm_config),
    }
    out = _like_convrot(x, n, dtype=out_dtype)
    return _plugin_node(
        graph, _convrot_op(weight_bits),
        [x, weight, weight_scale, bias], [out], fields, name,
    ).outputs[0]
```

`_SPECS`:

```python
_CONVROT_FIELDS = frozenset({"group_size", "has_bias", "out_dtype", "gemm_config"})

_SPECS: dict[str, tuple[int, int, frozenset[str]]] = {
    ...
    PluginOp.CONVROT_LINEAR: (4, 1, _CONVROT_FIELDS),
    PluginOp.CONVROT_LINEAR_W4A4: (4, 1, _CONVROT_FIELDS),
}
```

検証:

```python
def _validate_convrot(node, weight_bits: int):
    if len(node.inputs) != 4 or len(node.outputs) != 1:
        raise ValueError("convrot_linear: expected 4 inputs / 1 output")

    x, w, ws, b = node.inputs

    if getattr(w, "dtype", None) != np.int8:
        raise ValueError("convrot_linear: weight must be INT8")
    if len(w.shape) != 2:
        raise ValueError("convrot_linear: weight must be 2D")

    n = int(w.shape[0])
    k = int(w.shape[1]) * (2 if weight_bits == 4 else 1)

    if k % 256 != 0:
        raise ValueError("convrot_linear: K must be divisible by 256")
    if k > 16384:
        raise ValueError("convrot_linear: K > 16384 is not supported in v1")
    if weight_bits == 8 and n % 4 != 0:
        raise ValueError("convrot_linear: N must be divisible by 4")
    if weight_bits == 4 and k % 64 != 0:
        raise ValueError("convrot_linear: W4A4 requires K divisible by 64")

    if getattr(ws, "dtype", None) != np.float32:
        raise ValueError("convrot_linear: weight_scale must be FP32")
    if ws.shape is not None and int(np.prod(ws.shape)) != n:
        raise ValueError("convrot_linear: weight_scale numel must be N")

    if getattr(b, "dtype", None) not in (np.float16, np.bfloat16, np.float32):
        raise ValueError("convrot_linear: bias must be FP16/BF16/FP32 (output dtype)")
    if b.shape is not None and int(np.prod(b.shape)) != n:
        raise ValueError("convrot_linear: bias numel must be N")

    if x.shape is not None and int(x.shape[-1]) != k:
        raise ValueError("convrot_linear: x last dim must be K")

    attrs = node.attrs
    if int(attrs.get("group_size", 256)) != 256:
        raise ValueError("convrot_linear: group_size must be 256")
    if int(attrs.get("has_bias", 1)) not in (0, 1):
        raise ValueError("convrot_linear: has_bias must be 0/1")
    if int(attrs.get("out_dtype", -1)) not in (-1, 0, 1, 2):
        raise ValueError("convrot_linear: out_dtype must be -1/0/1/2")
    if int(attrs.get("gemm_config", -1)) < -1:
        raise ValueError("convrot_linear: gemm_config must be >= -1")
```

### C.4 `ops.py`

```python
def convrot_linear(x, weight, weight_scale, bias=None, *,
                   group_size=256, out_dtype=None, gemm_config=-1):
    """ConvRot W8A8 linear (test / standalone path)."""
    return _convrot_linear(x, weight, weight_scale, bias,
                           weight_bits=8, group_size=group_size,
                           out_dtype=out_dtype, gemm_config=gemm_config)


def convrot_linear_w4a4(x, qweight, weight_scale, bias=None, *,
                        group_size=256, out_dtype=None, gemm_config=-1):
    """ConvRot W4A4 linear (test / standalone path).

    qweight: [N, K/2] INT8, packed signed INT4 (low nibble = even k)
    """
    return _convrot_linear(x, qweight, weight_scale, bias,
                           weight_bits=4, group_size=group_size,
                           out_dtype=out_dtype, gemm_config=gemm_config)


def _convrot_linear(x, weight, weight_scale, bias, *, weight_bits,
                    group_size, out_dtype, gemm_config):
    _require_cuda(x, weight, weight_scale)
    if x.dim() < 2:
        raise ValueError("convrot_linear: x must be at least 2D")
    if weight.dim() != 2 or weight.dtype != torch.int8:
        raise ValueError("convrot_linear: weight must be 2D INT8")
    if weight_scale.dtype != torch.float32:
        raise ValueError("convrot_linear: weight_scale must be FP32")

    n = weight.shape[0]
    k = weight.shape[1] * (2 if weight_bits == 4 else 1)
    if x.shape[-1] != k:
        raise ValueError(f"convrot_linear: K mismatch ({x.shape[-1]} vs {k})")
    if weight_scale.numel() != n:
        raise ValueError("convrot_linear: weight_scale numel must be N")

    if out_dtype is None:
        out_dtype = x.dtype          # Comfy と同じ既定（入力 dtype）
    out_code = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[out_dtype]

    # bias は出力 dtype。Comfy も `bias.to(out_dtype)` してからカーネルに渡す。
    if bias is None:
        bias = torch.zeros(n, dtype=out_dtype, device=x.device)
        has_bias = 0
    else:
        _require_cuda(bias)
        if bias.numel() != n:
            raise ValueError("convrot_linear: bias must be [N]")
        if bias.dtype != out_dtype:
            bias = bias.to(out_dtype)
        has_bias = 1

    out_shape = tuple(x.shape[:-1]) + (n,)
    op = (PluginOp.CONVROT_LINEAR if weight_bits == 8
          else PluginOp.CONVROT_LINEAR_W4A4)
    outs = E.run_plugin(
        op,
        {"x": x.contiguous(), "w": weight.contiguous(),
         "ws": weight_scale.contiguous().reshape(-1), "b": bias.contiguous()},
        [("o", out_dtype, out_shape)],
        {"group_size": int(group_size), "has_bias": has_bias,
         "out_dtype": out_code, "gemm_config": int(gemm_config)},
    )
    return outs["o"]
```

### C.5 ONNX symbolic

```python
def _convrot_symbolic(weight_bits: int):
    def symbolic(g, x, weight, weight_scale,
                 bias=None, out_dtype=None, convrot=False,
                 convrot_groupsize=256, *args, **kwargs):
        if not convrot:
            raise RuntimeError("convrot symbolic requires convrot=True")
        if bias is None:
            bias = torch.zeros(weight.shape[0], dtype=torch.float32,
                               device=weight.device)
            has_bias = 0
        else:
            has_bias = 1
        op = ("dit-plugins::convrot_linear" if weight_bits == 8
              else "dit-plugins::convrot_linear_w4a4")
        return g.op(op, x, weight, weight_scale, bias,
                    group_size_i=int(convrot_groupsize),
                    has_bias_i=has_bias,
                    out_dtype_i=-1,          # 入力 dtype と同じ
                    gemm_config_i=-1)
    return symbolic


torch.onnx.register_custom_op_symbolic(
    "comfy_kitchen::int8_linear", _convrot_symbolic(8), opset_version=17)
torch.onnx.register_custom_op_symbolic(
    "comfy_kitchen::convrot_w4a4_linear", _convrot_symbolic(4), opset_version=17)
```

- 実際の schema（`bias` の位置、`convrot` が positional か）を export 前に確認する。
- `weight` / `weight_scale` は量子化されたまま渡す。dequant しない。
- `gemm_config` は autotune 後にノード属性を書き換える（`autotune.py`）。

### C.6 `test/test_register.cpp`

```cpp
const char* kNames[] = {
    ...
    "convrot_linear",
    "convrot_linear_w4a4",
};
```

### C.7 CMake

§5.9 のとおり。`ck_kernels` に vendor 4 ファイル + `convrot_shim.cu`、`COMFY_HAVE_CUTLASS=1`、CUTLASS include、`cudart` のみリンク。

### C.8 `python/tests/test_convrot.py`

```python
import os

import pytest
import torch

import comfy_kitchen as ck

pytest.importorskip("trt_dit_plugins")
from trt_dit_plugins import ops

CKPT_I8 = os.environ.get("CONVROT_CKPT")  # Anima INT8 convrot
CKPT_BF16 = os.environ.get("CONVROT_CKPT_BF16")

LAYERS = ["blocks.0.mlp.layer1", "blocks.0.mlp.layer2", "blocks.0.self_attn.q_proj"]


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_w8a8_real(layer, dtype):
    if not CKPT_I8:
        pytest.skip("CONVROT_CKPT not set")
    from safetensors import safe_open

    x = torch.randn(4096, 2048 if layer != LAYERS[1] else 8192,
                    device="cuda", dtype=torch.bfloat16).to(dtype)
    with safe_open(CKPT_I8, framework="pt", device="cuda") as f:
        w = f.get_tensor(f"net.{layer}.weight")
        ws = f.get_tensor(f"net.{layer}.weight_scale")

    ref = ck.int8_linear(x, w, ws, convrot=True, convrot_groupsize=256)
    got = ops.convrot_linear(x, w, ws)
    assert _cos(got, ref) >= 0.999999
    assert float((got.float() - ref.float()).abs().max()) == 0.0


@pytest.mark.parametrize("layer", LAYERS)
def test_w4a4_real(layer):
    if not CKPT_BF16:
        pytest.skip("CONVROT_CKPT_BF16 not set")
    from safetensors import safe_open

    with safe_open(CKPT_BF16, framework="pt", device="cuda") as f:
        w = f.get_tensor(f"net.{layer}.weight")
    q, ws = ck.quantize_convrot_w4a4_weight(w, 256)
    ws = ws.reshape(-1)

    x = torch.randn(4096, w.shape[1], device="cuda", dtype=torch.bfloat16)
    ref = ck.convrot_w4a4_linear(x, q, ws, convrot_groupsize=256)
    got = ops.convrot_linear_w4a4(x, q, ws)
    assert float((got.float() - ref.float()).abs().max()) == 0.0


@pytest.mark.parametrize("m", [1, 2, 3, 8, 9, 64, 4096])
def test_route_all_m(m):
    w = torch.randint(-8, 8, (2048, 2048), device="cuda", dtype=torch.int8)
    ws = torch.rand(2048, device="cuda") * 0.01
    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    got = ops.convrot_linear(x, w, ws)
    ref = ck.int8_linear(x, w, ws, convrot=True, convrot_groupsize=256)
    assert float((got.float() - ref.float()).abs().max()) == 0.0


@pytest.mark.parametrize("bad", [
    dict(k=2048, n=2048, group_size=128),
    dict(k=1000, n=2048, group_size=256),
    dict(k=17408, n=2048, group_size=256),
])
def test_negative(bad):
    with pytest.raises(Exception):
        ops.convrot_linear(
            torch.randn(4, bad["k"], device="cuda", dtype=torch.bfloat16),
            torch.zeros(bad["n"], bad["k"], device="cuda", dtype=torch.int8),
            torch.ones(bad["n"], device="cuda"),
            group_size=bad["group_size"],
        )
```

### C.9 負のテスト（プラグイン単体）

C++ 側で `configurePlugin` / `onShapeChange` が -1 を返すことを確認します。

- `K = 1280`（`% 256 != 0`）
- `K = 17408`
- W4A4 で `weight.shape[1] * 2 != x.shape[-1]`
- `out_dtype = 3`
- `weight_scale` の要素数 ≠ N

### C.10 `autotune.py`

```python
"""エンジンビルド前に CUTLASS config を確定する（enqueue では計測しない）。"""
import json
from pathlib import Path

import torch

import trt_dit_plugins._lib as _lib

CACHE = Path(__file__).parent / "configs" / "convrot_int4.json"


def autotune_int4(shapes, out_dtype=0):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for m, n, k in shapes:
        key = f"{m}x{n}x{k}"
        if key in cache:
            continue
        scratch = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        cfg = _lib.get_autotune()(m, n, k, out_dtype, scratch)
        cache[key] = int(cfg)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache
```

運用:

1. `ops.py` / `autotune.py` が `convrot_int4_autotune` を呼び、`(M,N,K)` ごとの最速 config を得る。
2. 結果を ONNX ノードの `gemm_config` に書き込んでからエンジンをビルドする。
3. 形状が変わったらキャッシュを引き、無ければ **既定 config**（§5.3.3 の表）を使う。実行時に計測しない。

---

## Appendix D: 計測生データ

環境: RTX 4070 Ti (12 GB), CUDA 13.0, PyTorch 2.9, `comfy-kitchen` 0.2.39。
入力: Anima 実活性化（`convrot_acts.pt`）、bias なし、30 iter、CUDA event。

### D.1 W8A8 / 自前実装 / ggml

```json
{
  "blocks.0.mlp.layer1": {
    "M": 4096, "K": 2048, "N": 8192,
    "comfy_ms": 0.668, "self_v1_ms": 1.248, "ggml_ms": 1.406,
    "self_v1_parts": {"quant": 0.156, "gemm": 0.613, "dequant": 0.479},
    "ggml_parts": {"quant": 0.099, "gemm": 0.633, "dequant": 0.661}
  },
  "blocks.0.mlp.layer2": {
    "M": 4096, "K": 8192, "N": 2048,
    "comfy_ms": 0.746, "self_v1_ms": 1.814, "ggml_ms": 1.303,
    "self_v1_parts": {"quant": 1.236, "gemm": 0.536, "dequant": 0.043},
    "ggml_parts": {"quant": 0.526, "gemm": 0.519, "dequant": 0.145}
  },
  "blocks.0.self_attn.q_proj": {
    "M": 4096, "K": 2048, "N": 2048,
    "comfy_ms": 0.187, "self_v1_ms": 0.354, "ggml_ms": 0.394,
    "self_v1_parts": {"quant": 0.151, "gemm": 0.159, "dequant": 0.043},
    "ggml_parts": {"quant": 0.093, "gemm": 0.143, "dequant": 0.169}
  }
}
```

- `comfy_ms` は同一セッション内の実測。実行間のばらつきは ±10% 程度（layer1 で 0.594–0.675）。
- 自前 v1 / ggml の精度は W8A8 と `max_abs == 0`（自前）／cos 0.999985–0.999996（ggml、BF16 エミュレーション無しのため不一致）。

### D.2 W4A4

```json
{
  "blocks.0.mlp.layer1": {
    "M": 4096, "K": 2048, "N": 8192,
    "w4a4_ms": 0.390, "w8a8_ms": 0.675, "speedup": 1.73,
    "cos_vs_bf16": 0.990425, "rel_l2_vs_bf16": 0.1398,
    "bytes_w4a4": 8421376, "bytes_w8a8": 16809984
  },
  "blocks.0.mlp.layer2": {
    "M": 4096, "K": 8192, "N": 2048,
    "w4a4_ms": 0.595, "w8a8_ms": 0.858, "speedup": 1.44,
    "cos_vs_bf16": 0.998268, "rel_l2_vs_bf16": 0.0588,
    "bytes_w4a4": 8396800, "bytes_w8a8": 16785408
  },
  "blocks.0.self_attn.q_proj": {
    "M": 4096, "K": 2048, "N": 2048,
    "w4a4_ms": 0.144, "w8a8_ms": 0.202, "speedup": 1.40,
    "cos_vs_bf16": 0.997585, "rel_l2_vs_bf16": 0.0696,
    "bytes_w4a4": 2105344, "bytes_w8a8": 4202496
  }
}
```

### D.3 再現手順

```bash
# W8A8 / ggml 比較（PoC 環境）
/opt/venv/bin/python /tmp/opencode/convrot_kernel_poc/bench_all.py
/opt/venv/bin/python /tmp/opencode/convrot_kernel_poc/bench_ggml.py

# W4A4
/opt/venv/bin/python /tmp/opencode/convrot_kernel_poc/bench_w4a4.py
```

生データ: `/tmp/opencode/convrot_kernel_poc/bench_all.json`, `bench_ggml.json`, `bench_w4a4.json`。

---

## 変更履歴

| 日付 | 内容 |
|---|---|
| 2026-10-21 | 初版（自前カーネル前提） |
| 2026-10-21 | 方針転換: comfy-kitchen vendor ベースへ。W4A4 を追加。自前カーネルを却下（§1.2 の実測による） |
| 2026-10-21 | 上流 `v0.2.39` のソースと突き合わせて検証。修正点: (1) W4A4 のルーティングは M 依存ではなく sm8x 判定 + `_prefer_legacy_int4_kernel` だった（§2.5/§5.3.1）、(2) `--use_fast_math` は上流では全 CUDA TU に掛かる（§4.6/§5.9）、(3) C++20 で CUTLASS submodule は `d4b4b49`、(4) `K` 上限は shared memory から 16384（`_CONVROT_FUSED_MAX_K` と一致）、(5) bias は出力 dtype、(6) `_prefer_cublas_int8_fallback` は常に false なので cuBLAS 削除は無害、(7) sm_75 は対象外 |
