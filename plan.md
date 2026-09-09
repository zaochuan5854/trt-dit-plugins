# TensorRT-DiT-Plugins 計画書

> 権利表記: 本プロジェクトはApache-2.0。ComfyKitchen (Copyright (c) 2025 Comfy Org)
> 由来のIF名・カーネルを利用し、詳細はNOTICE・各ファイルのSPDXヘッダを参照。
> `comfy_kitchen`名前空間は互換目的であり、Comfy Orgとの提携関係を示すものではない。

## 1. 目的・スコープ
- 対象: `sm89` (Ada Lovelace) / TensorRT 10.x `IPluginV3` / C++20 / CUDA 12.x
- 主題: DiT推論のボトルネックをプラグインとして切り出し・融合する。対象外: ComfyUIノード、TeaCache、配布自動化は本計画に含めない。
- 共通原則: プラグイン境界はすべてBF16統一、内部はSRAM完結、Workspaceは256Bアライン一括確保。
- 基本精度: **BF16** (`supportsFormatCombination`は`kLINEAR+kBF16`のみ受理、出力もBF16)。
- 拡張dtype (カーネル対応範囲・検証済み):
  - `int8_attention`: in FP32/FP16/BF16 → outはFP32→BF16・他は同dtype (comfy準拠)
  - `adaln`/`rms_adaln`: in/out FP32/FP16/BF16
  - `apply_rope`/`rms_rope_split_half`: in/out FP16/BF16 (freqs/scaleはFP32可)
  - `stochastic_round_fp8`: in FP32/FP16/BF16 → out FP8(E4M3)。rng入力はINT32 (TRT10はUINT8入出力非対応のため下位8bit使用)

## 2. プラグイン一覧 (登録名はcomfy-kitchen側を踏襲)
| # | 計画名 | TRT登録名 (`namespace::name`) | comfy-kitchen対応物 | 状態 |
|---|---|---|---|---|
| P1 | `SageAttention2Plugin` | `comfy_kitchen::int8_attention` | `sage_attention.py:int8_attention` (+`_masked`) | ✅ E2E済 (cos=0.99990@S256, 0.99987@S4096; 0.908ms→分離後0.910ms@S4096/H16/D128) |
| P2 | `FusedAdaLNPlugin` | `comfy_kitchen::adaln` | `dlpack:adaln` (`ops/adaln.cu`) | ✅ E2E済 (cos=1.0) |
| P3 | `FusedRMSAdaLNPlugin` | `comfy_kitchen::rms_adaln` | `dlpack:rms_adaln` | ✅ E2E済 (cos=1.0) |
| P4 | `StridedRoPEPlugin` | `comfy_kitchen::apply_rope` | `dlpack:apply_rope` (`ops/rms_rope.cu`) | ✅ E2E済 (cos=1.0) |
| P4b | — | `comfy_kitchen::rms_rope_split_half` | `torch:rms_rope_split_half` (同・RMSNorm融合+split-half/partial rotary) | ✅ E2E済 (cos=1.0 全回転・部分rot_dim共) |
| P5 | `FusedGeGLU/SwiGLU` | (保留: comfy-kitchen側に未存在) | — | ⛔ ブロッカー: upstream不在のため未着手 |
| P6 | `FP8StochasticRounding` | `comfy_kitchen::stochastic_round_fp8` | `dlpack:stochastic_round_fp8` | ✅ E2E済 (bad=0/16384両モード) |

## 3. P1: SageAttention2Plugin (最優先)
ベース: `comfy-kitchen/backends/cuda/sage_attention/`

### 3.1 4-Launch構成 (`enqueue()`内)
1. `smooth_k`: Kのチャネル平均・外れ値平滑化
2. `per_warp_int8_cuda`: Q,Kを動的INT8量子化
3. `per_channel_fp8`: Vを動的FP8(E4M3)量子化
4. `qk_int8_sv_f8_attn`: INT8/FP8 MMA + SRAM内FP32 Softmax → BF16出力

### 3.2 IPluginV3契約
- Input[0/1/2]: Q,K,V (`kBF16`, `[B,H,S,d]`)
- Output[0]: Context (`kBF16`, `[B,S,H*d]`)
- 実装: `OneBuild + OneRuntime`、ステートレス、動的S対応 (`getOutputShapes`)
- `getWorkspaceSize()`: INT8 Q/K + FP8 V + スケール分を合算して返却

### 3.3 受け入れ基準
- 精度: BF16 SDPA比 `cos > 0.999`
- 性能: 1層 `1.99ms → 0.92ms` 相当 (S=4096, RTX4090)
- ビルド: Anima 2B ONNXへ置換挿入しTRTビルド成功

## 4. P2-P4: Phase 2仕様要点
- P2/P3: ベース `comfy-kitchen/adaln`, `rms_adaln`。4-5回のVRAM往復→Warp Shuffleで1撃化。入出力BF16。
- P4: ベース `comfy-kitchen/apply_rope`。QKVスライス後の非連続テンソルにstridedアクセスで直接回転。`contiguous`消滅をNsightで確認。
- P5/P6はP1-P4確定後に着手。P5はCUTLASS Epilogue融合、P6は丸めのみのElementwise。

## 5. データフロー (BF16境界)
```text
FusedAdaLN/RMSAdaLN(BF16) → QKV Linear(TRT FP8 GEMM) → StridedRoPE(インプレース)
→ SageAttention2Plugin(内 INT8/FP8, 出 BF16) → Out Linear(TRT) → Residual Add(TRT)
→ FusedGeGLU/SwiGLU → FFN Linear(TRT) → 次ブロック
```

## 6. ロードマップ
- Phase 1 (MVP): カーネル切り出し → 空`IPluginV3`ラッパーでDLLコンパイル → Workspace実装 → 単体ハーネスで精度・性能検証 → ONNX置換
- Phase 2: P2/P3 → P4の順に実装、Launch数・VRAM往復削減をプロファイリング
- Phase 3: P5/P6、FFN込みの通し検証

## 7. リスクと対策
| リスク | 対策 |
|---|---|
| `smooth_k`のPyTorch依存 | 生ポインタC++関数として切り出し、Torchヘッダ排除 |
| Workspaceアライン破壊 | `align_to_256`徹底、中間オフセット単体テスト |
| Myelin融合阻害 | 境界BF16・標準レイアウト維持、Shuffle/Transpose局所化 |

## 8. 次のアクション
1. [済] 最小`CMakeLists.txt` + 空`int8_attention`(BF16専用)でコンパイル・登録確認 (`test/test_register.cpp`)
2. [済] `sage_attention/`の3カーネルを組み込み`enqueue`配線 → cos=0.99990、0.908ms@S4096/H16
3. [済] P2/P3/P4/P6配線+E2E (全てcos=1.0 / bad=0)
4. P5: upstreamにGeGLU/SwiGLUカーネル追加後に着手 (現時点で存在しないため保留)
5. Anima 2BのONNXグラフへ置換挿入しTRTビルド (Phase 1残件)

## 8b. Pythonパッケージ (`python/`, wheel化)
- 構成: pure Python (`_lib`: .so発見+ロード / `_engine`: 単一plugin engineキャッシュ+実行 /
  `ops`: torch-friendly 6関数) + 同梱`.so` (`pack_libs.sh`で`build/*.so`→`lib/`)。
  コンパイル拡張なし。`pyproject.toml` (setuptools) 済み。
- 使い方: `pip install trt-dit-plugins` (将来) / 今は`PYTHONPATH=python` +
  `TRT_DIT_LIBDIR=build`。GPU実行には別途 `torch` + `tensorrt==10.16.*` が要る。
- 検証済み (`python/tests/test_smoke.py`, RTX 4070 Ti): int8 cos=0.99990、
  adaln/rms_adaln cos=1.0、rope identity、stochastic ok。
- 落とし穴2件: ①入力dictをsortするとQ/Kが入れ替わる (挿入順を保持すること)。
  ②`PluginField`のdataはBuffer要求 (list不可、`struct.pack`使用)。
- Windows: `os.add_dll_directory`対応済み。MSVC+CUDA+TRTでのwheel buildはCI課題として残件。

## 9. 実装メモ (落とし穴・最適化)
- wrapper側最適化 (実装済):
  - 動的shape: int8_attentionはprofile(min/opt/max)で1 engineがS=64/256/1024全対応
    (cos=0.99992/0.99990/0.99989)。式ベースgetOutputShapes+max基準workspaceのため対応済み。
  - fp8 in-place: `IPluginV3OneBuildV2::getAliasedInput`でout←rngエイリアス宣言
    (+`alias_rng`フィールド、既定OFF)。enqueueはポインタ一致判定で両対応。
    有効化にはengine側で`setPreviewFeature(kALIASED_PLUGIN_IO_10_03, true)`が必要。
    なおエイリアス宣言するとpreview無効のengineはビルド失敗するため、既定OFFは必須。
  - TimingCacheID: 全プラグインで返却 (engine再build高速化。field持ちはeps/rot込み)。
- 落とし穴:- engineはstrongly-typed (`kSTRONGLY_TYPED`) で作ること。weakly-typed既定では出力がFP32扱いになり
  プラグイン(BF16)出力との間に暗黙castが入ってcosが壊れる (実測: cos=0.001)。
- `REGISTER_TENSORRT_PLUGIN`マクロはnamespace `""` で登録する。`comfy_kitchen`で引くには手動登録。
- TRT 10.16はUINT8プラグイン入出力非対応。rng等のbyte列はINT32で渡す (下位8bit使用)。
- `tensorrt` pip wheelに10.16なし・NGC index到達不可のため、TRT C++はapt (`libnvinfer*`=10.16.1.11-1+cuda13.2
  ピン、`-dev`除外+headersピン) で調達。Docker baseはキャッシュ済みcuda devel (pull不要)。
