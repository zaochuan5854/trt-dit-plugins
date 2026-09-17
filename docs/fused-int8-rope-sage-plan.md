# 計画書: DiT-self専用 int8入力 RoPE+SageAttn fused

対象: Anima DiT-selfのみ / S≦9216 (Sは潜在画像のH×W積) / Adapter不要 /
QKVはint8+scale入力 / 適用先は Reforge-New int8エンジン。
詳細仕様は `fused-int8-rope-sage-spec.md` を正とする。
実測数値は README の Plugins 表を正とし、本書には残さない。
計測条件は仕様書§7 (warmup100/iter500同一条件、代表2点 B1H16D128非causal)。

## S0. ベースライン固定

* 既存 `rms_rope_split_half` + `int8_attention` (分離パス) で
  S=4096・S=9216 (B1H16D128、非causal) の参照ログ・engine・入出力npyを取得。
* surgery適用数 (28層self) の確認。
* ゲート: ビルド成功 + 既存test green。以降の比較母点。

## S1. 前段fused kernel単体

* `dequant→RMSNorm→RoPE→per-thread quant` をtorch単体op化し、eager参照と一致試験。
* scale契約 (per-tensor in → per-thread out) の単体検証。V pathは既存のまま。
* ゲート: 単体一致試験 pass (閾値は仕様書§7を借用)。

## S2. plugin配線 + surgery

* `fused_int8_rope_sage_attn` (dit-plugins) 実装 + Reforge-New側 第3surgery追加。
* 非対応条件はビルド時拒否し、分離パスにfallback (surgery側の運用)。
* ゲート: 代表2点で仕様書§7の数値ゲート通過。

## S3. profile / tactic確定 + 速度ゲート

* S=4096/S=9216の2profileでfreqs bake・timing cache分離・`CTA_K=128` 固定可否を確定。
* 代表2点で実効時間 (quant+rope含む) をS0と比較 (TOPS換算 `O=4BHS²D` 併記、
  Graph off/on・warmup100/iter500同一条件)。
* ゲート: `t_fused実効 < t_S0実効` を両代表点で満たすこと。
  ビルド時間の爆発がないこと。

## 採用済み設計判断 (理由のみ)

* V単一パスrequant: int8→int8でper-tensor入力scaleをblock scaleにfold
  (`scale_out=sc_block*s_in`、量子化値側はs_inが相殺し既存と一致)。
  bf16中間を経由する二重変換を除去。
* fast-mathは自作rope TU (`src/kernel/own/`) のみに限定。vendored TUは対象外。
* Hadamard正規化のscale集約: `convrot128` 末尾の定数倍を格納scale側に集約し、
  量子化divisor側で対応する除算に (対称INT8では丸め一致)。
* K half-staging: half-0をSMEM fp16、half-1をレジスタ保持し、PassBの再計算を
  除去。scale系はFP32のままのため精度に影響なし。
* requantレジスタキャッシュ: pass-1値をint8パックのままレジスタ保持し、
  pass-2のHBM再読を除去。形状が大きい場合は再読カーネルにfallback。
* 逐次single-stream: cross-callなevent共有によるoverlapは世代hazardのため
  不採用。健全化にはgraph captureが必要。
* K-mean事前pass (smooth-K) は維持。乱数データではmean≈0で自明に通るため、
  撤廃の可否は実Anima較正データ待ち。
* 出力量子化の上限は127 (標準対称範囲)。下流に第二の整数丸め段はないため
  protective rangeは不要。
* RoPE sincosはon-the-flyのまま。テーブル事前構築・漸化式・warp協調はいずれも
  ロード/shfl/レジスタ代償がSFU削減を上回る (SFU非律速)。

## S2 surgery完成

* `add_dit_self_fused_attn` (9入力constructor) + `fuse_dit_self_attn_block`
  (第3surgery。rope済みグラフの `rms_rope_split_half` +
  `SageInt8Attn`/`Attention` 対を9入力fusedノード化、Q/K共有検証でcross-attn拒否)
  + 検証深化 (q/k/v同一形状・scale単一要素・norm[128]/inv[64]) +
  CLI `--fuse-dit-self-attn`。
* Reforge側の自動anchor探索はReforge側に残置 (当モジュールはexplicit-anchor主義)。

## 素のsage_attnプラグイン

* `sage_attn` (domain `dit-plugins`): Q/K/V f32/f16/bf16入力のdense
  SageAttention純粋ラッパー。FP8入力は全SMで拒否 (PV-FP8はtacticとして内包)。
* tactic: 既定はQK-INT8+PV-FP16 portable path (sm80/86/89/90/100/120動作)。
  `fp8_pv=1` でQK-INT8+PV-FP8 dense SageAttention2 path (sm89のみ、
  D∈{64,128}・S%128==0・f16/bf16入力・square形状。dense形状では低速のためopt-in)。
* sm90+向けFP8-MMA tacticはカーネル未vendoringのため将来課題。

## RPN (FireQ式RoPE保存正規化)

* `python/trt_dit_plugins/rpn.py`: `compute_rpn_scales` + `fold_rpn_into_norms`
  (D=128 split-half専用、score不変)。カーネル変更なしでfold済み重みが透過する。
  K-mean除去は実データ検証まで保留。

## S4. 任意 (完全1カーネル化)

* S2で帯域律速残存が実測された場合のみ着手し、新tacticとして追加。
* ゲート: S2-fusedに対する有意な改善 + 数値ゲート再通過。
  改善なき場合はS4を捨てS2を最終形にする。

## 既知の制限 (対応済み: ビルド時拒否)

* `fused_int8_rope_sage_attn` の L≦1024 (`cta_k=64`) パスは破損していた
  (S=512でcos 0.80、S=1024で不正メモリアクセス。変更前のlibでも同一の
  先行不具合) ため、全層でビルド時拒否に変更:
  wrapper (`configurePlugin`/`getWorkspaceSize`/`onShapeChange`)、
  カーネルlauncher、Python事前検査 (`ops.py`)、surgeon validator
  (`validate_plugin_nodes`) がいずれも S>1024 を要求。実用上は S>1024 のみ対応。
  (ゲート点 S=4096/9216 は全て通過。S=2048は通過するがcos 0.9977と余裕なし。)
