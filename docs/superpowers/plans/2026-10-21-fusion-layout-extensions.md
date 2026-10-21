# trt-dit-plugins Fusion / Layout Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dynamic-shape RMS+RoPE, fused LayerNorm/RMSNorm AdaLN with residual/gate, and ConvRot output epilogues/layouts in `trt_dit_plugins` without changing the TensorRT-Reforge repository.

**Architecture:** Extend the existing TensorRT plugin wrappers first, then expose the same contracts through the Python frontend and `surgeon.py`. Keep the current fused INT8 attention path. Add the ConvRot BHSD/QKV store as a custom CUTLASS epilogue so no transpose layer is emitted; if that store misses the performance gate, fall back to a single layout kernel rather than restoring the 240 transpose nodes.

**Tech Stack:** C++17, CUDA 13, TensorRT 11.3 IPluginV3, CUTLASS (vendored), PyTorch 2.10, ONNX GraphSurgeon, pytest, CMake/Ninja.

**Spec:** `/opt/ComfyUI/custom_nodes/ComfyUI-TensorRT-Reforge/trt-dit-plugins-fusion-request.md` (source request). Existing accuracy gates: `docs/fused-int8-rope-sage-spec.md`.

## Scope Check

This specification covers three related but independently testable subsystems:

1. **4.1 RMS+RoPE / RMSNorm** — dynamic shapes and cross-attention RMS.
2. **4.2 AdaLN** — 3D input, LayerNorm, broadcast modulation, residual/gate.
3. **4.3 ConvRot** — GELU/SILU epilogue, residual/gate, BHSD/QKV store.

They share the same graph-surgery layer but can ship independently. This plan implements **4.1–4.3**.

**4.4 attention replacement is deliberately deferred to a separate plan.** The request says it is optional and only allowed if within ±5% of the current `_gemm_mha_v2` path (≤87 ms). The current fused INT8 attention is already at native parity; replacing it would add numerical risk without being required for the first three items. The acceptance gate below measures whether 4.1–4.3 already reach the target.

### Spec gaps this plan resolves

- **Cross-attention Q/K lengths differ.** A paired `rms_rope_split_half(q, k, ...)` requires `q.shape == k.shape`; cross-attention has `S_q != S_k`. The request's `rms_norm_pair(q,k,wq,wk,eps)` cannot be one pair kernel for this model. Use a single-tensor `rms_norm(x, weight, layout=...)` twice. Keep `rot_dim=-1` on the pair op only for same-shape RMS-only callers.
- **`rot_dim=-1` needs no real frequency table.** The vendored kernel treats `rot_dim=0` as RMS-only and never reads `freqs`. The pair op keeps its 5 inputs and accepts a shared dummy `[1,1,1,1,2,2]` fp32 tensor when `rot_dim=-1`; the single-tensor `rms_norm` has no `freqs` input at all.
- **BHSD is not expressible as a CUTLASS `StoreD` stride.** For QKV, `(m,n)` maps to `(b,s,qkv,h,d)`; the address uses `m/S` and `n/C`, so it needs a custom store visitor. This is the highest-risk task and has an explicit fallback.
- **`gate` is per-batch, not per-N.** For B > 1, the residual epilogue's per-N `rscale` cannot represent `gate[B,N]`. Use one GEMM launch per batch for `residual_gate`; B is 1–2 for this model. Do not materialize `[B,S,N]` gate.
- **LayerNorm affine weight is optional.** The vendored kernel currently has no weight argument. Add a documented `DIT-PLUGINS-PATCH`; pass `None` for the model's all-ones weight.

### Explicitly out of scope

- Modifying `ComfyUI-TensorRT-Reforge` (caller wiring and the final Anima engine are a handoff).
- Replacing `_gemm_mha_v2` with a new attention plugin (4.4; separate plan).
- W4A4 epilogue/layout support. W4A4 must reject the new fields with a clear error; Anima's target is W8A8.
- New custom CUDA code for the attention path.

---

## Global Constraints

- Plugin namespace stays `dit-plugins`; all plugin versions stay `"1"` so existing serialized engines still load.
- Inputs/outputs are BF16 (FP16 accepted); internal accumulators are FP32.
- Dynamic `B` and `S`; `C=2048`, `H=16`, `D=128` are fixed for Anima.
- `D % 8 == 0`, `C % 8 == 0`; 16-byte alignment assumed.
- Determinism: same input + same engine → bit-identical output.
- PluginField values are engine-cache keys: keep the documented domains fixed.
- No `Transpose` nodes in the fused graph; layout changes happen in plugin stores.
- Q/K are physically `[B,H,S,D]`; QKV and the rest of the block are token-major.
- Build with 12 threads: `cmake --build build --parallel 12` and `MAX_JOBS=12` for Python extension builds.
- `src/` is project-owned. Every `third_party/comfy` change must carry `DIT-PLUGINS-PATCH` and a matching `third_party/comfy/VENDOR.md` entry.
- Each task ends with a commit; do not combine unrelated tasks.
- Every numeric change needs before/after `cos` and milliseconds.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/wrapper/rms_rope_wrapper.cpp` | Dynamic `rms_rope_split_half`; `rot_dim=-1` RMS-only pair |
| `src/wrapper/rms_norm_wrapper.cpp` | New single-tensor `rms_norm` for cross-attention, BSHD→BHSD |
| `src/wrapper/adaln_wrapper.cpp` | RMS/LayerNorm AdaLN, 3D input, optional weight/residual/gate |
| `third_party/comfy/ops/adaln.cu` | `DIT-PLUGINS-PATCH`: weight/residual/gate and separate modulation dtype |
| `src/wrapper/convrot_linear_wrapper.cpp` | Epilogue/residual fields; BHSD/QKV outputs |
| `src/kernel/convrot/convrot_shim.{h,cu}` | Route table glue for new epilogues/layouts |
| `src/kernel/convrot/epilogue.cu` | Small M<3 / fallback activation+residual kernel |
| `third_party/comfy/convrot/cutlass_gemm_int8.cu` | `DIT-PLUGINS-PATCH`: GELU/SILU EVT and QKV store visitor |
| `python/trt_dit_plugins/select.py` | New `PluginOp` values and `select_norm` behavior |
| `python/trt_dit_plugins/ops.py` | Torch-facing signatures and validation |
| `python/trt_dit_plugins/surgeon.py` | Layout-aware rope, cross-attn RMS, AdaLN, ConvRot fields |
| `python/tests/test_gpu.py` | GPU correctness for dynamic/layout paths |
| `python/tests/test_surgeon.py` | CPU ONNX surgery tests |
| `python/tests/test_convrot.py` | ConvRot epilogue/layout tests |
| `test/test_rms_rope.cpp`, `test/test_adaln.cpp` | Existing C++ E2E harnesses extended |
| `test/test_register.cpp` | Register new plugin names |
| `docs/fusion-layout-extensions.md` | Public contract and measured results |
| `README.md`, `README.ja.md`, `README.zh-CN.md` | Feature table / supported ops |

---

### Task 1: Dynamic `rms_rope_split_half` and `rot_dim=-1`

**Files:**
- Modify: `src/wrapper/rms_rope_wrapper.cpp`
- Modify: `python/trt_dit_plugins/ops.py`
- Test: `python/tests/test_gpu.py`
- Test: `test/test_rms_rope.cpp`

**Interfaces:**
- Consumes: existing `launch_rms_rope_kernel(...)`; `rot_dim=0` already means RMS-only in the kernel.
- Produces:
  - `rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6, rot_dim=0, profile=None)`
  - `rot_dim=-1` = RMS-only; `freqs` is still a 6-D dummy for this pair op.
  - Dynamic `B`, `S`, or both accepted by `configurePlugin`.

- [ ] **Step 1: Write the failing dynamic GPU test**

Append to `python/tests/test_gpu.py`:

```python
def _rms_rope_ref(q, k, freqs, qs, ks, rot):
    """Reference for split-half RMS+RoPE; rot=0 means RMS-only."""
    def norm(x, w):
        xf = x.float()
        rr = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return xf * rr * w.float()

    nq, nk = norm(q, qs), norm(k, ks)
    if rot == 0:
        return nq.to(q.dtype), nk.to(k.dtype)

    half = rot // 2
    q0, q1 = nq[..., :half], nq[..., half:rot]
    k0, k1 = nk[..., :half], nk[..., half:rot]
    c, s = freqs[..., 0, 0].float(), freqs[..., 0, 2].float()
    qo = torch.cat([q0 * c - q1 * s, q0 * s + q1 * c,
                    nq[..., rot:]], dim=-1)
    ko = torch.cat([k0 * c - k1 * s, k0 * s + k1 * c,
                    nk[..., rot:]], dim=-1)
    return qo.to(q.dtype), ko.to(k.dtype)


def test_rms_rope_dynamic_b_s():
    B, H, D = 2, 2, 128
    for S_ in (256, 1024, 4096):
        q = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
        fq = torch.randn(1, 1, S_, D // 2, 2, 2, device="cuda")
        fq[..., 0, 0] = torch.cos(fq[..., 0, 0])
        fq[..., 0, 1] = -torch.sin(fq[..., 0, 0])
        fq[..., 0, 2] = torch.sin(fq[..., 0, 0])
        fq[..., 0, 3] = torch.cos(fq[..., 0, 0])
        qs = torch.rand(D, device="cuda")
        ks = torch.rand(D, device="cuda")
        profile = {
            "q": ((1, H, 256, D), (1, H, 1024, D), (B, H, 4096, D)),
            "k": ((1, H, 256, D), (1, H, 1024, D), (B, H, 4096, D)),
            "f": ((1, 1, 256, D // 2, 2, 2), (1, 1, 1024, D // 2, 2, 2),
                  (1, 1, 4096, D // 2, 2, 2)),
        }
        qo, ko = tdp.rms_rope_split_half(
            q, k, fq, qs, ks, epsilon=1e-6, profile=profile)
        rq, rk = _rms_rope_ref(q, k, fq, qs, ks, D)
        assert _cos(qo, rq) > 0.9999
        assert _cos(ko, rk) > 0.9999


def test_rms_rope_rms_only_rot_minus_one():
    B, H, S_, D = 1, 2, 512, 128
    q = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S_, D, device="cuda", dtype=torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, 2, 2, device="cuda")
    qs = torch.rand(D, device="cuda")
    ks = torch.rand(D, device="cuda")
    qo, ko = tdp.rms_rope_split_half(
        q, k, dummy, qs, ks, rot_dim=-1)
    rq, rk = _rms_rope_ref(q, k, dummy, qs, ks, 0)
    assert _cos(qo, rq) > 0.9999
    assert _cos(ko, rk) > 0.9999
```

- [ ] **Step 2: Run it and verify the failure**

Run:

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py::test_rms_rope_dynamic_b_s \
  python/tests/test_gpu.py::test_rms_rope_rms_only_rot_minus_one -q
```

Expected: `test_rms_rope_dynamic_b_s` FAILS while building the engine with `configurePlugin ... return code: -1`. The `rot_dim=-1` test FAILS or rotates with the wrong semantics.

- [ ] **Step 3: Allow dynamic dims in `valid`**

In `src/wrapper/rms_rope_wrapper.cpp`, replace `valid` with:

```cpp
bool valid(nvinfer1::Dims const& q, nvinfer1::Dims const& k, nvinfer1::Dims const& f,
    nvinfer1::Dims const& qs, nvinfer1::Dims const& ks) const {
    if (q.nbDims != 4 || k.nbDims != 4 || f.nbDims != 6)
        return false;
    for (int i = 0; i < 4; ++i)
        if (q.d[i] != k.d[i])
            return false;
    int64_t D = q.d[3];
    if (D < 32 || D % 32 != 0)
        return false;
    // -1 is the build-time dynamic marker; 0 is never a valid extent.
    for (int i = 0; i < 3; ++i)
        if (q.d[i] < -1 || q.d[i] == 0)
            return false;

    // rot_dim < 0 is the explicit RMS-only spelling; rot_dim == 0 stays
    // "all dimensions" for backward compatibility.
    int64_t rot = rotDim_ < 0 ? 0 : (rotDim_ > 0 ? rotDim_ : D);
    if (rot % 2 != 0 || rot > D)
        return false;
    if (rot > 0 && (f.d[3] != rot / 2 || f.d[4] != 2 || f.d[5] != 2))
        return false;
    if (rot == 0 && f.d[4] != 2)
        return false;

    int64_t qd[3]{q.d[0], q.d[1], q.d[2]}, fd[3]{f.d[0], f.d[1], f.d[2]};
    for (int i = 0; i < 3; ++i)
        if (fd[i] != 1 && fd[i] != qd[i])
            return false;
    if (qs.nbDims != 1 || ks.nbDims != 1 || qs.d[0] != D || ks.d[0] != D)
        return false;
    return true;
}
```

- [ ] **Step 4: Resolve `rot=-1` and reject mismatched freqs at runtime**

In `onShapeChange`, after the `valid` check and dimension captures, add:

```cpp
if (B_ <= 0 || H_ <= 0 || S_ <= 0)
    return -1;
int64_t rot = rotDim_ < 0 ? 0 : (rotDim_ > 0 ? rotDim_ : D_);
if (rot > 0) {
    if (f2_ != S_ || (f1_ != 1 && f1_ != H_) || (fb_ != 1 && fb_ != B_))
        return -1;
}
```

In `enqueue`, replace:

```cpp
int64_t rot = rotDim_ > 0 ? rotDim_ : D_;
```

with:

```cpp
int64_t rot = rotDim_ < 0 ? 0 : (rotDim_ > 0 ? rotDim_ : D_);
if (rot > 0 && f2_ != S_)
    return -1;
```

The kernel's norm-only tail (`d = rot_dim + lane; d < head_dim`) already covers all dimensions when `rot == 0`, so RMS-only is a normal kernel path. `fst` becomes all zeros and the kernel never reads `freqs`.

- [ ] **Step 5: Include `rot=-1` in the timing ID**

Update `stampTiming()` / the `timing_` string so `-1` is distinct from `0`:

```cpp
std::snprintf(timing_, sizeof(timing_),
    "rms_rope_split_half/1/eps=%g/rot=%d", (double)eps_, (int)rotDim_);
```

- [ ] **Step 6: Add the Python profile argument**

In `python/trt_dit_plugins/ops.py`, change:

```python
def rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6, rot_dim=0):
    """RMSNorm (full D) + split-half RoPE (rot_dim prefix, 0 = all)."""
```

to:

```python
def rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon=1e-6,
                        rot_dim=0, profile=None):
    """RMSNorm (full D) + split-half RoPE (rot_dim prefix, 0 = all).

    ``rot_dim=-1`` skips rotation entirely (RMS-only); ``freqs`` is still
    required by this pair op and may be a 1x1x1x1x2x2 dummy. Use
    ``rms_norm`` for cross-attention where q and k have different S.
    """
```

and pass `profile=profile` to `E.run_plugin`.

- [ ] **Step 7: Re-run the GPU tests**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py::test_rms_rope_dynamic_b_s \
  python/tests/test_gpu.py::test_rms_rope_rms_only_rot_minus_one -q
```

Expected: PASS, `_cos > 0.9999` for both q and k.

- [ ] **Step 8: Extend the C++ static test for `-1`**

In `test/test_rms_rope.cpp`, add a `rot=-1` case that expects the same output as `rot=0` with a dummy frequency buffer. The existing `ref` already treats `d < rot` as the only rotated range, so `rot=0` is the RMS-only reference.

- [ ] **Step 9: Commit**

```bash
git add src/wrapper/rms_rope_wrapper.cpp python/trt_dit_plugins/ops.py \
  python/tests/test_gpu.py test/test_rms_rope.cpp
git commit -m "feat(rms_rope): accept dynamic B/S and rot_dim=-1"
```

---

### Task 2: Single-tensor `rms_norm` for cross-attention

**Files:**
- Create: `src/wrapper/rms_norm_wrapper.cpp`
- Modify: `CMakeLists.txt`
- Modify: `test/test_register.cpp`
- Modify: `python/trt_dit_plugins/select.py`
- Modify: `python/trt_dit_plugins/ops.py`
- Modify: `python/trt_dit_plugins/surgeon.py`
- Test: `python/tests/test_gpu.py`
- Test: `python/tests/test_surgeon.py`

**Interfaces:**
- Consumes: `launch_rms_rope_kernel(...)` with `has_k=false`, `rot_dim=0`.
- Produces:
  - Plugin `dit-plugins::rms_norm`, version `"1"`, one output.
  - `rms_norm(x, weight, eps=1e-6, layout="bhsd", profile=None)`.
  - `layout="bshd"`: input `[B,S,H,D]`, output `[B,H,S,D]`.
  - `layout="bhsd"`: input/output `[B,H,S,D]`.

- [ ] **Step 1: Write the failing cross-attention GPU test**

Append to `python/tests/test_gpu.py`:

```python
def test_rms_norm_cross_attention_different_s():
    B, H, Sq, Sk, D = 1, 2, 1024, 256, 128
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    wq = torch.rand(D, device="cuda")
    wk = torch.rand(D, device="cuda")

    def ref(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
                * w.float()).to(x.dtype).permute(0, 2, 1, 3).contiguous()

    qo = tdp.rms_norm(q, wq, layout="bshd")
    ko = tdp.rms_norm(k, wk, layout="bshd")
    assert qo.shape == (B, H, Sq, D)
    assert ko.shape == (B, H, Sk, D)
    assert _cos(qo, ref(q, wq)) > 0.9999
    assert _cos(ko, ref(k, wk)) > 0.9999
```

- [ ] **Step 2: Run it and verify failure**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py::test_rms_norm_cross_attention_different_s -q
```

Expected: FAIL with `AttributeError: module 'trt_dit_plugins' has no attribute 'rms_norm'`.

- [ ] **Step 3: Create the wrapper**

Create `src/wrapper/rms_norm_wrapper.cpp` using the same registration pattern as `rms_rope_wrapper.cpp`. The key class members:

```cpp
constexpr char const* kName = "rms_norm";
constexpr char const* kNamespace = "dit-plugins";

class RmsNorm final : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
public:
    RmsNorm(float eps = 1e-6f, int32_t inLayout = 0)
        : eps_(eps), inLayout_(inLayout) { stampTiming(); }

    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return "1";
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return kNamespace;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
        const nvinfer1::DataType* in, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != 2 || !out || !in)
            return -1;
        if (!isFloat(in[0]) || !isFloat(in[1]))
            return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out,
        int32_t nbOut, nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 2 || nbOut != 1 || in[0].nbDims != 4)
            return -1;
        out[0] = in[0];
        if (inLayout_ == 1) {
            // [B,S,H,D] -> [B,H,S,D]
            out[0].d[1] = in[0].d[2];
            out[0].d[2] = in[0].d[1];
        }
        return 0;
    }

    bool supportsFormatCombination(int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* io, int32_t nbIn,
        int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos < nbIn)
            return d.type == nvinfer1::DataType::kFLOAT ||
                   d.type == nvinfer1::DataType::kHALF ||
                   d.type == nvinfer1::DataType::kBF16;
        return d.type == io[0].desc.type;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbIn, nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return nbIn == 2 && validDims(in[0].desc.dims, in[1].desc.dims) ? 0 : -1;
    }

    static bool validDims(nvinfer1::Dims const& x, nvinfer1::Dims const& w) {
        if (x.nbDims != 4 || w.nbDims != 1)
            return false;
        int64_t D = x.d[3];
        if (D < 32 || D % 32 != 0 || w.d[0] != D)
            return false;
        for (int i = 0; i < 4; ++i)
            if (x.d[i] < -1 || x.d[i] == 0)
                return false;
        return true;
    }
};
```

`supportsFormatCombination` accepts `kFLOAT/kHALF/kBF16` for both inputs and requires the output to match input 0. `configurePlugin` and `onShapeChange` accept dynamic `B/H/S` but require `D > 0`, `D % 32 == 0`, and `weight` length `D`.

Serialize the two fields (`eps` float, `in_layout` int32) in
`getFieldsToSerialize`, parse them in the creator with `0` default for
`in_layout`, and include `in_layout` in `getTimingCacheID`.

- [ ] **Step 4: Implement the launch strides**

In `onShapeChange`:

```cpp
if (nbIn != 2 || in[0].dims.nbDims != 4 || in[1].dims.nbDims != 1)
    return -1;
if (in[1].dims.d[0] != in[0].dims.d[3])
    return -1;

if (inLayout_ == 0) {
    B_ = in[0].dims.d[0];
    H_ = in[0].dims.d[1];
    S_ = in[0].dims.d[2];
} else {
    B_ = in[0].dims.d[0];
    S_ = in[0].dims.d[1];
    H_ = in[0].dims.d[2];
}
D_ = in[0].dims.d[3];
if (B_ <= 0 || H_ <= 0 || S_ <= 0 || D_ <= 0 || D_ % 32 != 0)
    return -1;
icode_ = dtypeCode(in[0].type);
scode_ = dtypeCode(in[1].type);
return 0;
```

In `enqueue`:

```cpp
const int64_t q_s0 = inLayout_ == 0
    ? (int64_t)H_ * S_ * D_ : (int64_t)S_ * H_ * D_;
const int64_t q_s1 = inLayout_ == 0 ? (int64_t)S_ * D_ : D_;
const int64_t q_s2 = inLayout_ == 0 ? D_ : (int64_t)H_ * D_;
const int64_t q_s3 = 1;
const int64_t qo_s0 = (int64_t)H_ * S_ * D_;
const int64_t qo_s1 = (int64_t)S_ * D_;
const int64_t qo_s2 = D_;
const int64_t qo_s3 = 1;

// No k, no rotation: the norm-only tail covers every dimension.
launch_rms_rope_kernel(in[0], nullptr, nullptr, in[1], nullptr,
    out[0], nullptr, B_, H_, S_, D_, 0, 1, 1, 1,
    q_s0, q_s1, q_s2, q_s3, 0, 0, 0, 0,
    qo_s0, qo_s1, qo_s2, qo_s3, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, eps_, icode_, 0, scode_,
    false, true, stream);
```

The `false` argument is `has_k`; the kernel's `if constexpr (HasK)` blocks are not instantiated. `freqs` is never dereferenced when `rot_dim=0`.

- [ ] **Step 5: Register and compile**

Add `src/wrapper/rms_norm_wrapper.cpp` to the `trt_dit_plugins` source list in `CMakeLists.txt`. Add the expected creator name to `test/test_register.cpp`:

```cpp
{"rms_norm", "dit-plugins", "1"},
```

Build:

```bash
cmake --build build --parallel 12
./build/regcheck ./build/libtrt_dit_plugins.so
```

- [ ] **Step 6: Add Python op and enum**

In `python/trt_dit_plugins/select.py`:

```python
RMS_NORM = "rms_norm"
```

In `surgeon._SPECS`, add:

```python
PluginOp.RMS_NORM: (2, 1, frozenset({"eps", "in_layout"})),
```

Add `"rms_norm"` to both `__all__` and the lazy `__getattr__` tuple in
`python/trt_dit_plugins/__init__.py`.

In `python/trt_dit_plugins/ops.py`:

```python
def rms_norm(x, weight, eps=1e-6, layout="bhsd", profile=None):
    """RMSNorm with optional BSHD input and BHSD output.

    layout="bshd": x [B,S,H,D] -> out [B,H,S,D] (fuses the q/k transpose)
    layout="bhsd": x [B,H,S,D] -> out [B,H,S,D]
    weight: [D] in FP32/FP16/BF16 (kernel reads it as the per-dim scale).
    """
    _require_cuda(x, weight)
    if x.dim() != 4 or weight.dim() != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("rms_norm expects x [B,S,H,D] / [B,H,S,D] and weight [D]")
    if layout not in ("bhsd", "bshd"):
        raise ValueError("layout must be 'bhsd' or 'bshd'")
    if layout == "bshd":
        b, s, h, d = x.shape
        oshape = (b, h, s, d)
    else:
        oshape = tuple(x.shape)
    outs = E.run_plugin(
        PluginOp.RMS_NORM,
        {"x": x, "w": weight},
        [("y", x.dtype, oshape)],
        {"eps": float(eps), "in_layout": int(layout == "bshd")},
        profile=profile,
    )
    return outs["y"]
```

- [ ] **Step 7: Add the surgery helper**

In `python/trt_dit_plugins/surgeon.py`, add:

```python
def add_rms_norm(graph, x, weight, *, eps=1e-6, layout="bhsd", name=None):
    """Single-tensor RMSNorm; layout='bshd' also transposes to BHSD."""
    gs = _gs()
    shape = None
    if x.shape is not None:
        b, s, h, d = x.shape
        shape = [b, h, s, d] if layout == "bshd" else list(x.shape)
    out = gs.Variable(name=f"dit_out{next(_uid)}", dtype=x.dtype, shape=shape)
    _plugin_node(graph, PluginOp.RMS_NORM, [x, weight], [out],
                 {"eps": float(eps), "in_layout": int(layout == "bshd")},
                 name=name)
    return out
```

- [ ] **Step 8: Re-run the GPU test**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py::test_rms_norm_cross_attention_different_s -q
```

Expected: PASS for both `Sq=1024`, `Sk=256`.

- [ ] **Step 9: Add a surgery test for no transpose**

In `python/tests/test_surgeon.py`:

```python
def test_add_rms_norm_bshd_has_no_transpose():
    gs = pytest.importorskip("onnx_graphsurgeon")
    x = gs.Variable("x", dtype="bfloat16", shape=[1, 64, 2, 128])
    w = gs.Variable("w", dtype="float32", shape=[128])
    g = gs.Graph(nodes=[], inputs=[x, w], outputs=[])
    y = S.add_rms_norm(g, x, w, layout="bshd")
    g.outputs = [y]
    assert y.shape == [1, 2, 64, 128]
    assert not [n for n in g.nodes if n.op == "Transpose"]
    assert S.validate_plugin_nodes(g) == []
```

- [ ] **Step 10: Commit**

```bash
git add src/wrapper/rms_norm_wrapper.cpp CMakeLists.txt test/test_register.cpp \
  python/trt_dit_plugins/select.py python/trt_dit_plugins/ops.py \
  python/trt_dit_plugins/surgeon.py python/tests/test_gpu.py \
  python/tests/test_surgeon.py
git commit -m "feat(rms_norm): add single-tensor cross-attention RMSNorm"
```

---

### Task 3: LayerNorm/RMSNorm AdaLN with 3D input, broadcast, weight, residual/gate

**Files:**
- Modify: `third_party/comfy/ops/adaln.cu`
- Modify: `third_party/comfy/VENDOR.md`
- Modify: `src/wrapper/adaln_wrapper.cpp`
- Modify: `python/trt_dit_plugins/select.py`
- Modify: `python/trt_dit_plugins/ops.py`
- Modify: `python/trt_dit_plugins/surgeon.py`
- Test: `python/tests/test_gpu.py`
- Test: `test/test_adaln.cpp`

**Interfaces:**
- Consumes: existing `launch_adaln_kernel`; extend it.
- Produces:
  - `rms_adaln(x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None)`
  - `layer_norm_adaln(...)` with the same arguments.
  - Inputs: `x, scale, shift[, weight][, residual, gate]`; count `3 + has_weight + 2*residual_gate`.
  - `x`: `[N,D]` or `[B,S,D]`; scale/shift: `[R,D]` or `[B,1,D]`; weight `[D]`; residual same shape as x; gate `[B,D]` or `[B,1,D]`.
  - `out = norm(x) * (1+scale) + shift`; with weight, `norm(x) * weight`; then `out = residual + gate*out` when `residual_gate=1`.
  - Modulation tensors (`scale/shift/weight/gate`) must share one dtype; `x/out/residual` share another. This is the supported mixed-dtype rule.

- [ ] **Step 1: Write the failing 3D/LayerNorm/residual test**

Append to `python/tests/test_gpu.py`:

```python
def test_layer_norm_adaln_3d_residual_gate():
    B, S_, D = 2, 16, 256
    x = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    sh = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    w = torch.randn(D, device="cuda", dtype=torch.float32)
    res = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)

    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    y = (xf - mean) * torch.rsqrt(var + 1e-6) * w.float()
    y = y * (1 + sc.float()) + sh.float()
    ref = (res.float() + gate.float() * y).to(torch.bfloat16)

    out = tdp.layer_norm_adaln(x, sc, sh, weight=w, residual=res, gate=gate)
    assert out.shape == x.shape
    assert _cos(out, ref) > 0.999
```

- [ ] **Step 2: Run it and verify failure**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py::test_layer_norm_adaln_3d_residual_gate -q
```

Expected: FAIL with `AttributeError: ... layer_norm_adaln`.

- [ ] **Step 3: Extend the vendored kernel with a documented patch**

At the top of the changed block in `third_party/comfy/ops/adaln.cu`, add:

```cpp
// DIT-PLUGINS-PATCH: optional affine weight, residual+gate, and independent
// modulation dtype for the TRT wrapper. Upstream callers that pass null/false
// take the original path bit-for-bit.
```

Change the kernel signature to:

```cpp
template <typename T, typename TMod, bool kSubtractMean>
__global__ void adaln_kernel(const T* __restrict__ x,
                             const TMod* __restrict__ scale,
                             const TMod* __restrict__ shift,
                             const TMod* __restrict__ weight,
                             const T* __restrict__ residual,
                             const TMod* __restrict__ gate,
                             T* __restrict__ out,
                             int64_t N, int64_t D,
                             int64_t scale_group, int64_t shift_group,
                             float eps, bool has_weight, bool residual_gate,
                             int weight_stride, int gate_stride)
```

Inside the per-row loop, apply the affine weight to the normalized value
**before** the AdaLN modulation:

```cpp
float nv = normed;
if (has_weight)
    nv *= to_float(weight[d * weight_stride]);
float y = nv * (1.0f + to_float(scale[scale_row * D + d]))
              + to_float(shift[shift_row * D + d]);
if (residual_gate)
    y = to_float(residual[row * D + d])
        + to_float(gate[scale_row * gate_stride + d]) * y;
out[row * D + d] = from_float<T>(y);
```

Apply the same block to both the existing vectorized (`float4`/`Elts`)
path and the scalar tail in `third_party/comfy/ops/adaln.cu`. Use the
file-local `to_float` / `from_float` helpers already defined there. Keep the
existing `kSubtractMean` statistics branch unchanged.

- [ ] **Step 4: Update the launcher signature**

Change the exported launcher to:

```cpp
void launch_adaln_kernel(const void* x, const void* scale, const void* shift,
    const void* weight, const void* residual, const void* gate, void* out,
    int64_t N, int64_t D, int64_t scale_group, int64_t shift_group,
    float eps, int dtype_code, int mod_dtype_code, int weight_dtype_code,
    bool subtract_mean, bool has_weight, bool residual_gate,
    cudaStream_t stream);
```

Dispatch `T` from `dtype_code` and `TMod` from `mod_dtype_code`; use separate branches for `has_weight/residual_gate` so the no-feature path stays a single template instantiation. `weight` is always `[D]` with stride 1; keep `weight_stride=1`. `gate` is `[B,D]`; `gate_stride=1` after flattening.

- [ ] **Step 5: Update `VENDOR.md`**

Add under “Patch list”:

```markdown
`ops/adaln.cu`
- Added optional affine weight, residual+gate, and a separate modulation
  dtype for the TRT wrapper (`DIT-PLUGINS-PATCH`). Null/false arguments keep
  the upstream path unchanged.
```

- [ ] **Step 6: Extend the wrapper**

In `src/wrapper/adaln_wrapper.cpp`:

```cpp
template <> struct Traits<true> {
    static constexpr char const* kName = "layer_norm_adaln";
};
```

Extend the constructor and creator to parse the new fields:

```cpp
explicit AdaLN(float eps = 1e-6f, int32_t hasWeight = 0, int32_t residualGate = 0)
    : eps_(eps), hasWeight_(hasWeight), residualGate_(residualGate) {
    stampTiming();
}
```

In `AdaLNCreator::createPlugin`, read `has_weight` and `residual_gate` with the
same `readInt` pattern used by the ConvRot creator, defaulting to `0`.
`getFieldsToSerialize` returns three fields: `eps`, `has_weight`,
`residual_gate`. `stampTiming` includes both integers.

Add fields to the class: `int32_t residualGate_ = 0; int32_t hasWeight_ = 0;`. `getNbOutputs` stays 1. Replace the fixed 3-input checks with:

```cpp
static bool inputCountOk(int32_t nbIn, int32_t hasWeight, int32_t residualGate) {
    return nbIn == 3 + hasWeight + 2 * residualGate;
}
```

`getOutputDataTypes`: accept `nbIn` in `{3,4,5,6}`; all inputs float; `out[0] = in[0]`.
`getOutputShapes`: `out[0] = in[0]` for rank 2 or 3.
`onShapeChange`:

```cpp
if (!inputCountOk(nbIn, hasWeight_, residualGate_))
    return -1;
if (in[0].dims.nbDims == 3) {
    B_ = in[0].dims.d[0];
    S_ = in[0].dims.d[1];
    N_ = B_ * S_;
    D_ = in[0].dims.d[2];
} else if (in[0].dims.nbDims == 2) {
    N_ = in[0].dims.d[0];
    D_ = in[0].dims.d[1];
    B_ = 1;
    S_ = N_;
} else {
    return -1;
}
Rs_ = in[1].dims.d[0];
Rt_ = in[2].dims.d[0];
int idx = 3;
if (hasWeight_) {
    if (in[idx].dims.nbDims != 1 || in[idx].dims.d[0] != D_)
        return -1;
    wcode_ = dtypeCode(in[idx++].type);
}
if (residualGate_) {
    // residual [B,S,D] (or [N,D]) and gate [B,D] / [B,1,D]
    if (in[idx].dims.nbDims != in[0].dims.nbDims)
        return -1;
    rescode_ = dtypeCode(in[idx++].type);
    if (in[idx].dims.nbDims < 2)
        return -1;
    const int last = in[idx].dims.nbDims - 1;
    const int gateRows = in[idx].dims.d[0];
    if (in[idx].dims.d[last] != D_)
        return -1;
    // The kernel uses scale_row for gate; require the same batch grouping
    // (or the fully broadcast single row).
    if (gateRows != Rs_ && !(gateRows == 1 && Rs_ == 1))
        return -1;
    gatecode_ = dtypeCode(in[idx].type);
}
if (N_ <= 0 || D_ <= 0 || N_ % Rs_ != 0 || N_ % Rt_ != 0)
    return -1;
if (in[1].type != in[2].type)
    return -1;
code_ = dtypeCode(in[0].type);
modcode_ = dtypeCode(in[1].type);
```

For 3D `scale/shift` `[B,1,D]`, flatten the row count as `B`:
`Rs_ = in[1].dims.nbDims == 3 ? in[1].dims.d[0] : in[1].dims.d[0]` (both give the number of vectors; the last dim must be `D`). Validate `in[1].dims.d[in[1].dims.nbDims - 1] == D_`.

In `enqueue`, call:

```cpp
const void* weight = hasWeight_ ? in[3] : nullptr;
const int resIdx = 3 + hasWeight_;
const void* residual = residualGate_ ? in[resIdx] : nullptr;
const void* gate = residualGate_ ? in[resIdx + 1] : nullptr;
launch_adaln_kernel(in[0], in[1], in[2], weight, residual, gate, out[0],
    N_, D_, N_ / Rs_, N_ / Rt_, eps_, code_, modcode_,
    hasWeight_ ? wcode_ : modcode_, kSubtractMean,
    hasWeight_ != 0, residualGate_ != 0, stream);
```

Add `Traits<true>` to the `AutoRegister` constructor:

```cpp
getPluginRegistry()->registerCreator(*gCreator<false>(), kNamespace);
getPluginRegistry()->registerCreator(*gCreator<true>(), kNamespace);
```

Add `{"layer_norm_adaln", "dit-plugins", "1"}` to `test/test_register.cpp`
so `regcheck` covers the new creator.

- [ ] **Step 7: Add Python ops**

In `python/trt_dit_plugins/ops.py`, replace `_adaln` with:

```python
def _adaln(name, x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
           profile=None):
    _require_cuda(x, scale, shift)
    if x.dim() not in (2, 3) or scale.dim() not in (2, 3) or shift.dim() not in (2, 3):
        raise ValueError(f"{name} expects x [N,D]/[B,S,D] and scale/shift [R,D]/[B,1,D]")
    d = x.shape[-1]
    if scale.shape[-1] != d or shift.shape[-1] != d:
        raise ValueError(f"{name} last-dim mismatch")
    if scale.dtype != shift.dtype:
        raise ValueError(f"{name} scale/shift dtype mismatch")
    mod_dtype = scale.dtype
    tensors = {"x": x, "scale": scale, "shift": shift}
    fields = {"eps": float(eps), "has_weight": int(weight is not None),
              "residual_gate": int(residual is not None)}
    if weight is not None:
        if weight.dim() != 1 or weight.numel() != d or weight.dtype != mod_dtype:
            raise ValueError("weight must be [D] in the modulation dtype")
        tensors["weight"] = weight
    if residual is not None:
        if gate is None or residual.shape != x.shape or residual.dtype != x.dtype:
            raise ValueError("residual needs gate and the same shape/dtype as x")
        if gate.dim() not in (2, 3) or gate.shape[-1] != d or gate.dtype != mod_dtype:
            raise ValueError("gate must be [B,D]/[B,1,D] in the modulation dtype")
        tensors["residual"] = residual
        tensors["gate"] = gate
    outs = E.run_plugin(
        name, tensors, [("o", x.dtype, tuple(x.shape))], fields, profile=profile)
    return outs["o"]


def rms_adaln(x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
              profile=None):
    """rmsnorm(x) * (1 + scale) + shift, optional weight / residual+gate."""
    return _adaln(PluginOp.RMS_ADALN, x, scale, shift, eps, weight, residual, gate,
                  profile)


def layer_norm_adaln(x, scale, shift, eps=1e-6, weight=None, residual=None, gate=None,
                     profile=None):
    """layernorm(x) * (1 + scale) + shift, optional weight / residual+gate."""
    return _adaln(PluginOp.LAYER_NORM_ADALN, x, scale, shift, eps, weight, residual,
                  gate, profile)
```

- [ ] **Step 8: Add the enum and selection**

In `select.py`:

```python
LAYER_NORM_ADALN = "layer_norm_adaln"
```

Add `"layer_norm_adaln"` to both `__all__` and the lazy `__getattr__` tuple in
`python/trt_dit_plugins/__init__.py`.

In `surgeon._SPECS`, register both AdaLN ops with the new fields:

```python
PluginOp.RMS_ADALN: (3, 1, frozenset({"eps", "has_weight", "residual_gate"})),
PluginOp.LAYER_NORM_ADALN: (3, 1, frozenset({"eps", "has_weight", "residual_gate"})),
```

In `validate_plugin_nodes`, replace the generic AdaLN input-count check with:

```python
if node.op in (PluginOp.RMS_ADALN, PluginOp.LAYER_NORM_ADALN):
    want_in = 3 + int(node.attrs.get("has_weight", 0)) \
                 + 2 * int(node.attrs.get("residual_gate", 0))
    if len(node.inputs) != want_in:
        errs.append(f"{node.name or node.op}: want {want_in} inputs, got {len(node.inputs)}")
```

Change `select_norm` so `NormKind.LAYER` returns `PluginOp.LAYER_NORM_ADALN` when the input is 3D / native LayerNorm is available; keep the existing note only when the caller asks for a native path. The plan's implementation is:

```python
if norm is NormKind.LAYER:
    return PluginOp.LAYER_NORM_ADALN
```

- [ ] **Step 9: Extend `fuse_norm_affine`**

Replace the signature and body with:

```python
def fuse_norm_affine(graph, *, norm: str, scale: str, shift: str, out: str,
                     plugin: str | PluginOp | None = None,
                     eps: float = 1e-6, weight: str | None = None,
                     residual: str | None = None, gate: str | None = None):
    """Replace norm*(1+scale)+shift (or residual+gate*that) with one node."""
    tensors = graph.tensors()
    for n in (norm, scale, shift, out):
        if n not in tensors:
            raise KeyError(f"tensor not in graph: {n!r}")
    if residual is not None and (gate is None or residual not in tensors or gate not in tensors):
        raise ValueError("residual and gate must both name graph tensors")
    if weight is not None and weight not in tensors:
        raise KeyError(f"tensor not in graph: {weight!r}")
    if plugin is None:
        producer = next((p for p in tensors[norm].inputs if p.op == "LayerNormalization"), None)
        plugin = PluginOp.LAYER_NORM_ADALN if producer is not None else PluginOp.RMS_ADALN
    plugin = PluginOp(plugin)
    if plugin not in (PluginOp.RMS_ADALN, PluginOp.LAYER_NORM_ADALN):
        raise TypeError(f"unsupported AdaLN plugin: {plugin!r}")

    tn, ts, tsh, to = (tensors[n] for n in (norm, scale, shift, out))
    ds = {_last_dim(t) for t in (tn, ts, tsh) if _last_dim(t) is not None}
    if len(ds) > 1:
        raise ValueError(f"last-dim mismatch in fuse anchors: {norm, scale, shift}")
    ins = [tn, ts, tsh]
    fields = {"eps": float(eps), "has_weight": int(weight is not None),
              "residual_gate": int(residual is not None)}
    if weight is not None:
        ins.append(tensors[weight])
    if residual is not None:
        ins.extend([tensors[residual], tensors[gate]])
    node = _plugin_node(graph, plugin, ins, [to], fields,
                        name=f"{plugin}_{out}")
    for prod in list(to.inputs):
        if prod is not node and to in prod.outputs:
            prod.outputs.remove(to)
        if prod is not node and prod in to.inputs:
            to.inputs.remove(prod)
    graph.cleanup().toposort()
    return node
```

- [ ] **Step 10: Add surgery and GPU tests**

In `python/tests/test_surgeon.py`, add:

```python
def test_fuse_layer_norm_adaln_3d_residual_gate():
    B, S_, D = 2, 16, 128
    x = gs.Variable("x", dtype=F32, shape=[B, S_, D])
    sc = gs.Variable("sc", dtype=F32, shape=[B, 1, D])
    sh = gs.Variable("sh", dtype=F32, shape=[B, 1, D])
    gate = gs.Variable("gate", dtype=F32, shape=[B, 1, D])
    res = gs.Variable("res", dtype=F32, shape=[B, S_, D])
    w = gs.Variable("w", dtype=F32, shape=[D])
    n = gs.Variable("n", dtype=F32, shape=[B, S_, D])
    y1 = gs.Variable("y1", dtype=F32, shape=[B, S_, D])
    y2 = gs.Variable("y2", dtype=F32, shape=[B, S_, D])
    y3 = gs.Variable("y3", dtype=F32, shape=[B, S_, D])
    y4 = gs.Variable("y4", dtype=F32, shape=[B, S_, D])
    out = gs.Variable("out", dtype=F32, shape=[B, S_, D])
    one = gs.Constant("one", values=np.ones((1,), dtype=F32))
    g = gs.Graph(nodes=[
        gs.Node("LayerNormalization", inputs=[x], outputs=[n],
                attrs={"epsilon": 1e-6}),
        gs.Node("Mul", inputs=[n, one], outputs=[y1]),
        gs.Node("Add", inputs=[y1, sc], outputs=[y2]),
        gs.Node("Add", inputs=[y2, sh], outputs=[y3]),
        gs.Node("Mul", inputs=[gate, y3], outputs=[y4]),
        gs.Node("Add", inputs=[res, y4], outputs=[out]),
    ], inputs=[x, sc, sh, gate, res, w], outputs=[out])
    g.opset = 23

    node = S.fuse_norm_affine(
        g, norm="n", scale="sc", shift="sh", out="out", weight="w",
        residual="res", gate="gate", plugin=PluginOp.LAYER_NORM_ADALN)
    assert node.op == PluginOp.LAYER_NORM_ADALN
    assert len(node.inputs) == 6
    assert node.attrs["has_weight"] == 1
    assert node.attrs["residual_gate"] == 1
    assert S.validate_plugin_nodes(g) == []
```

The graph uses `Mul(n, one)` as the identity-affine step; the surgery
replaces the whole chain, so the exact order of the old `Mul/Add` nodes is
only a shape anchor. The separate `w` tensor exercises the plugin's
`has_weight=1` path; do not attach it to the LayerNormalization node or the
weight would be applied twice.

In `python/tests/test_gpu.py`, add a mixed-dtype case:

```python
def test_rms_adaln_mixed_mod_dtype():
    B, S_, D = 2, 8, 128
    x = torch.randn(B, S_, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    sh = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    out = tdp.rms_adaln(x, sc, sh)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
           * (1 + sc.float()) + sh.float()).to(torch.bfloat16)
    assert _cos(out, ref) > 0.999
```

- [ ] **Step 11: Run tests and commit**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_gpu.py -k "adaln" -q
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
cmake --build build --parallel 12
git add third_party/comfy/ops/adaln.cu third_party/comfy/VENDOR.md \
  src/wrapper/adaln_wrapper.cpp python/trt_dit_plugins/select.py \
  python/trt_dit_plugins/ops.py python/trt_dit_plugins/surgeon.py \
  python/tests/test_gpu.py python/tests/test_surgeon.py test/test_adaln.cpp
git commit -m "feat(adaln): support 3D LayerNorm/AdaLN with residual+gate"
```

---

### Task 4: ConvRot GELU/SILU epilogue and residual+gate

**Files:**
- Modify: `src/wrapper/convrot_linear_wrapper.cpp`
- Modify: `src/kernel/convrot/convrot_shim.{h,cu}`
- Create: `src/kernel/convrot/epilogue.cu`
- Modify: `third_party/comfy/convrot/int8_linear.cu` (only if the M<3 path needs the activation; otherwise use `epilogue.cu`)
- Modify: `third_party/comfy/convrot/cutlass_gemm_int8.cu`
- Modify: `third_party/comfy/VENDOR.md`
- Modify: `python/trt_dit_plugins/ops.py`
- Modify: `python/trt_dit_plugins/surgeon.py`
- Test: `python/tests/test_convrot.py`

**Interfaces:**
- Consumes: existing `convrot_quant`, `convrot_gemm`, `convrot_gemv_m1`.
- Produces:
  - Fields `epilogue`: `0=NONE`, `1=GELU` (exact erf), `2=SILU`.
  - Field `residual_gate`: `0/1`.
  - Inputs when `residual_gate=1`: `x, weight, weight_scale, bias, residual, gate`.
  - `residual`: `[B,S,N]`; `gate`: `[B,N]` or `[B,1,N]`.
  - `out = residual + gate * epilogue(dequant(gemm(x)))`.
  - W8A8 only for these fields; W4A4 rejects at `configurePlugin`.

- [ ] **Step 1: Write the failing GELU/residual tests**

Append to `python/tests/test_convrot.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convrot_gelu_and_residual_gate():
    torch = pytest.importorskip("torch")
    tdp = pytest.importorskip("trt_dit_plugins")
    tdp.ensure_loaded()
    K, N, S_ = 2048, 512, 64
    x = torch.randn(1, S_, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(N, device="cuda") * 0.01
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(1, S_, N, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)

    # The reference uses the same rowwise int8 quant contract as the plugin:
    # scale = absmax / 127, q = round(x / scale), dequant = q * scale.
    xf = x.float().reshape(-1, K)
    scale = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    xq = (xf / scale).round().clamp(-127, 127) * scale
    y = (xq @ w.float().T) * ws.float() + b.float()
    y = torch.nn.functional.gelu(y)
    ref = (res.float() + gate.float() * y).to(torch.bfloat16)

    out = tdp.convrot_linear(
        x, w, ws, b, epilogue="gelu", residual=res, gate=gate)
    assert out.shape == (1, S_, N)
    assert (out.float() - ref.float()).abs().mean() < 0.02
```

Add a second test for `epilogue="silu"` without residual, and a W4A4 rejection test:

```python
def test_convrot_w4a4_rejects_epilogue():
    torch = pytest.importorskip("torch")
    K, N, S_ = 2048, 512, 8
    x = torch.randn(1, S_, K, device="cuda", dtype=torch.bfloat16)
    qw = torch.randint(-8, 7, (N, K // 2), device="cuda", dtype=torch.int8)
    ws = torch.rand(N, device="cuda") * 0.01
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="W4A4"):
        tdp.convrot_linear_w4a4(x, qw, ws, b, epilogue="gelu")
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_convrot.py -k "gelu or silu or rejects" -q
```

Expected: FAIL with `TypeError: convrot_linear() got an unexpected keyword argument 'epilogue'`.

- [ ] **Step 3: Add the exact GELU/SILU device helpers**

Create `src/kernel/convrot/epilogue.cu` with a small kernel used by the M<3 fallback:

```cpp
// SPDX-License-Identifier: Apache-2.0
// DIT-PLUGINS: elementwise epilogue for M<3 routes and as a fallback.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__global__ void convrot_epilogue_kernel(const T* __restrict__ y,
                                        const T* __restrict__ residual,
                                        const T* __restrict__ gate,
                                        T* __restrict__ out,
                                        int64_t n, int64_t stride_gate,
                                        int epilogue, bool residual_gate) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float v = static_cast<float>(y[i]);
    if (epilogue == 1) v = gelu_erf(v);
    if (epilogue == 2) v = silu(v);
    if (residual_gate) {
        const float r = static_cast<float>(residual[i]);
        const float g = static_cast<float>(gate[i % stride_gate]);
        v = r + g * v;
    }
    out[i] = static_cast<T>(v);
}
```

For the actual wrapper, the fallback is only for `M < 3`; keep the GEMV
output in the plugin's output buffer and run the epilogue in place with
`y == out`. Each thread reads and writes its own element, so in-place is safe.
The `convrot_epilogue` C entry point switches on the output dtype and passes
the corresponding `T*`.

Add to `convrot_shim.h`:

```cpp
int convrot_epilogue(void* y, const void* residual, const void* gate,
                     int64_t rows, int64_t cols, int64_t gate_stride,
                     int epilogue, int residual_gate, int dtype,
                     cudaStream_t stream);
```

and implement the launch in `epilogue.cu` with a `dtype` switch for BF16/FP16/FP32.
Add `src/kernel/convrot/epilogue.cu` to the `trt_dit_plugins` source list in
`CMakeLists.txt`.

- [ ] **Step 4: Add CUTLASS GELU/SILU visitors**

In `third_party/comfy/convrot/cutlass_gemm_int8.cu`, add above the config table:

```cpp
// DIT-PLUGINS-PATCH: output epilogues for the TRT wrapper.
struct GeluErf {
    CUTLASS_HOST_DEVICE float operator()(float x) const {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
};

struct Silu {
    CUTLASS_HOST_DEVICE float operator()(float x) const {
        return x / (1.0f + expf(-x));
    }
};
```

For the no-residual path, extend `FusedInt8Gemm` with an `Epilogue` template parameter and insert:

```cpp
using Act = cutlass::epilogue::threadblock::VisitorCompute<
    Epilogue, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;
using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<
    StoreD, Act, EVT1>;
```

For the residual path, insert `Act` before `Resid`:

```cpp
using EVT = cutlass::epilogue::threadblock::Sm80EVT<
    StoreD, Resid, Act, EVT2>;
```

Instantiate `Epilogue = cutlass::epilogue::threadblock::Identity` for the existing path so old kernels stay bit-identical. Add launch entry points:

```cpp
int launch_cutlass_int8_dequant_act(..., int epilogue, ...);
int launch_cutlass_int8_dequant_residual_act(..., int epilogue, ...);
```

`epilogue` 0/1/2 selects identity/GeluErf/Silu.

- [ ] **Step 5: Route the shim**

Declare the new entry point in `convrot_shim.h` with the same signature shown
below. In `convrot_shim.cu`, add:

```cpp
int convrot_gemm_ex(void* y, const void* x, const int8_t* w, const float* ws,
                    const void* bias, const int8_t* xq, const float* xs,
                    int64_t M, int64_t N, int64_t K, int weight_bits, int act_bits,
                    int in_dtype, int out_dtype, int bias_dtype, int config,
                    int epilogue, const void* residual, const void* gate,
                    void* ws_scratch, size_t ws_scratch_bytes, cudaStream_t stream) {
    if (weight_bits != 8)
        return -1;
    if (residual != nullptr && gate == nullptr)
        return -1;

    if (M >= 3) {
        if (residual != nullptr)
            return launch_cutlass_int8_dequant_residual_act(
                y, w, ws, bias, xq, xs, residual, gate, M, N, K, out_dtype,
                bias_dtype, config, epilogue, stream);
        return launch_cutlass_int8_dequant_act(
            y, w, ws, bias, xq, xs, M, N, K, out_dtype, bias_dtype, config,
            epilogue, stream);
    }

    // M=1 uses the fused GEMV (quantization inside); M=2 uses the existing
    // rowwise-quant + GEMV route. Both get the tiny in-place epilogue.
    if (M == 1) {
        int rc = convrot_gemv_m1(y, x, w, ws, bias, N, K, in_dtype, out_dtype,
                                 bias_dtype, stream);
        if (rc < 0)
            return -1;
        if (rc == 1 &&
            convrot_gemm(y, w, ws, bias, xq, xs, M, N, K, 8, 8, out_dtype,
                         bias_dtype, config, ws_scratch, ws_scratch_bytes,
                         stream) != 0)
            return -1;
    } else {
        if (convrot_gemm(y, w, ws, bias, xq, xs, M, N, K, 8, 8, out_dtype,
                         bias_dtype, config, ws_scratch, ws_scratch_bytes,
                         stream) != 0)
            return -1;
    }
    return convrot_epilogue(y, residual, gate, M, N, N, epilogue,
                            residual != nullptr, out_dtype, stream);
}
```

Keep the existing `convrot_gemm` unchanged for callers that do not use the new
epilogue. The wrapper passes the original `x` and `inCode` to
`convrot_gemm_ex` so the M=1 path can use the fused GEMV.

- [ ] **Step 6: Extend the wrapper**

In `surgeon._SPECS`, add `epilogue` and `residual_gate` to both ConvRot
entries, and teach `validate_plugin_nodes` the variable input count:

```python
if node.op in (PluginOp.CONVROT_LINEAR, PluginOp.CONVROT_LINEAR_W4A4):
    want_in = 4 + 2 * int(node.attrs.get("residual_gate", 0))
    if len(node.inputs) != want_in:
        errs.append(f"{node.name or node.op}: want {want_in} inputs, got {len(node.inputs)}")
```

Add fields:

```cpp
int32_t epilogue_ = 0;      // 0 none, 1 GELU, 2 SILU
int32_t residualGate_ = 0;  // 0/1
```

Grow `fields_` to 6, add both entries to `refreshFields`, and parse them with
`readInt` in `createPlugin` (defaults `0`). Append `/e<epilogue>/rg<residual>`
to `updateTimingID`. Input indices become `kRes = 4, kGate = 5`; `nbIn` is
`4 + 2*residualGate_`.

`getNbOutputs` stays 1. `getOutputDataTypes` accepts either count. `getOutputShapes` is unchanged. `getWorkspaceSize` uses the same `M` (full batch). `checkDesc` adds:

```cpp
if (epilogue_ < 0 || epilogue_ > 2) return false;
if (residualGate_ < 0 || residualGate_ > 1) return false;
if (residualGate_ && WeightBits != 8) return false;
if (epilogue_ != 0 && WeightBits != 8) return false;
if (residualGate_) {
    if (x.dims.nbDims != 3) return false;            // B,S,N
    if (residual.dims.nbDims != 3) return false;
    if (residual.dims.d[0] != x.dims.d[0] ||
        residual.dims.d[1] != x.dims.d[1] ||
        residual.dims.d[2] != w.dims.d[0]) return false;
    if (gate.dims.nbDims != 2 && gate.dims.nbDims != 3) return false;
    if (x.type != residual.type || x.type != gate.type) return false;
    if (outType(x.type, outDtype_) != x.type) return false;
    if (gate.dims.nbDims == 3 &&
        (gate.dims.d[0] != x.dims.d[0] || gate.dims.d[1] != 1 ||
         gate.dims.d[2] != w.dims.d[0])) return false;
}
```

The residual epilogue reads `rscale` and `resid` in the **output dtype**
(see `launch_cutlass_int8_dequant_residual`), so `residual_gate` requires
`residual`, `gate`, and the output to share the input dtype. If Anima needs
BF16 gate with an FP32 output, that is a separate conversion, not this task.

`onShapeChange` stores `B_ = x.dims.d[0]`, `S_ = x.dims.d[1]` for 3D, and `gateStride_ = N_` (gate is `[B,N]` after flattening `[B,1,N]`). `M_ = B_ * S_`.

Keep the existing `convrot_quant` call over the full `M` before these
branches. `xq/xs` are then valid for the M=2/GEMM routes; the M=1 fused GEMV
ignores them and uses the original `x`, but the M=1 fallback still uses them.

In `enqueue`, for no residual:

```cpp
return convrot_gemm_ex(out[0], in[kX], w, ws, bias, xq, xs, M_, N_, K_,
    WeightBits, actBits, inCode, outCode, biasCode, gemmConfig_, epilogue_,
    nullptr, nullptr, scratch, scratchBytes - quantBytes - scaleBytes,
    stream) == 0 ? 0 : -1;
```

For residual+gate, loop per batch:

```cpp
const size_t inElemSize = inDesc[kX].type == nvinfer1::DataType::kFLOAT ? 4 : 2;
const size_t gateElemSize = inDesc[kGate].type == nvinfer1::DataType::kFLOAT ? 4 : 2;
const size_t outElemSize = inElemSize;
const int64_t perBatchX = (int64_t)S_ * K_;
const int64_t perBatchY = (int64_t)S_ * N_;
for (int64_t b = 0; b < B_; ++b) {
    const void* resb = static_cast<const char*>(in[kRes]) + b * perBatchY * inElemSize;
    const void* gateb = static_cast<const char*>(in[kGate]) + b * N_ * gateElemSize;
    void* outb = static_cast<char*>(out[0]) + b * perBatchY * outElemSize;
    const int8_t* xqb = xq + b * perBatchX;
    const float* xsb = xs + b * S_;
    const void* xb = static_cast<const char*>(in[kX]) + b * perBatchX * inElemSize;
    if (convrot_gemm_ex(outb, xb, w, ws, bias, xqb, xsb, S_, N_, K_,
            WeightBits, actBits, inCode, outCode, biasCode, gemmConfig_, epilogue_,
            resb, gateb, scratch, scratchBytes - quantBytes - scaleBytes,
            stream) != 0)
        return -1;
}
return 0;
```

`convrot_quant` still runs once over `M_ = B_*S_`; only the GEMM is per-batch. This is the documented B≤2 tradeoff.

- [ ] **Step 7: Add Python fields and validation**

In `ops.py`, extend `_convrot_linear`:

```python
_EPILOGUES = {"none": 0, "gelu": 1, "silu": 2}

def _convrot_linear(..., epilogue="none", residual=None, gate=None):
    if epilogue not in _EPILOGUES:
        raise ValueError("epilogue must be 'none', 'gelu', or 'silu'")
    if residual is not None and gate is None:
        raise ValueError("residual requires gate")
    if residual is None and gate is not None:
        raise ValueError("gate requires residual")
    if residual is not None:
        if residual.dtype != x.dtype or gate.dtype != x.dtype:
            raise ValueError("residual and gate must match the input dtype")
        if out_dtype is not None and out_dtype != x.dtype:
            raise ValueError("residual_gate requires out_dtype=None/input dtype")
    if w4a4 and (epilogue != "none" or residual is not None):
        raise ValueError("W4A4 does not support epilogue/residual yet")
    tensors = {"x": x, "w": weight, "ws": weight_scale, "b": bias}
    if residual is not None:
        tensors["residual"] = residual
        tensors["gate"] = gate
    fields = {..., "epilogue": _EPILOGUES[epilogue],
              "residual_gate": int(residual is not None)}
```

Update the public signatures:

```python
def convrot_linear(x, weight, weight_scale, bias=None, *, out_dtype=None,
                   gemm_config=-1, epilogue="none", residual=None, gate=None,
                   profile=None):
```

Update `convrot_linear_w4a4` with the same keyword arguments so the Python
validation can reject them explicitly:

```python
def convrot_linear_w4a4(x, qweight, wscales, bias=None, *, out_dtype=None,
                        gemm_config=-1, epilogue="none", residual=None,
                        gate=None, profile=None):
```

and pass the new arguments through.

- [ ] **Step 8: Extend `add_convrot_linear`**

Add `epilogue="none"`, `residual=None`, `gate=None` to `surgeon.add_convrot_linear`. Append residual/gate to `_plugin_node` inputs when present and add `epilogue`/`residual_gate` fields. Reject W4A4 combinations in the helper before emitting the node.

- [ ] **Step 9: Run the tests and commit**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_convrot.py -k "gelu or silu or residual or rejects" -q
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
cmake --build build --parallel 12
git add src/wrapper/convrot_linear_wrapper.cpp src/kernel/convrot/convrot_shim.h \
  src/kernel/convrot/convrot_shim.cu src/kernel/convrot/epilogue.cu \
  third_party/comfy/convrot/cutlass_gemm_int8.cu third_party/comfy/VENDOR.md \
  python/trt_dit_plugins/ops.py python/trt_dit_plugins/surgeon.py \
  python/tests/test_convrot.py
git commit -m "feat(convrot): add GELU/SILU epilogue and residual gate"
```

---

### Task 5: ConvRot BHSD/QKV output store

**Files:**
- Modify: `src/wrapper/convrot_linear_wrapper.cpp`
- Modify: `src/kernel/convrot/convrot_shim.{h,cu}`
- Modify: `third_party/comfy/convrot/cutlass_gemm_int8.cu`
- Modify: `third_party/comfy/VENDOR.md`
- Modify: `python/trt_dit_plugins/ops.py`
- Modify: `python/trt_dit_plugins/surgeon.py`
- Test: `python/tests/test_convrot.py`
- Test: `python/tests/test_surgeon.py`

**Interfaces:**
- Consumes: Task 4's `convrot_gemm_ex` route.
- Produces:
  - Fields `out_layout`: `0=bsc` (default), `1=bhsd`; `qkv_heads`: `0` or `H`; `head_dim`: `0` or `D`.
  - `out_layout=1`, `qkv_heads=0`: one output `[B,H,S,D]`, `N=H*D`.
  - `out_layout=1`, `qkv_heads=H`: three outputs `[B,H,S,D]` (q, k, v), `N=3*H*D`.
  - `convrot_linear(..., out_layout="bsc", qkv_heads=0, head_dim=0)` returns a tensor or a 3-tuple.
  - Reject `epilogue != none` or `residual_gate` with `out_layout=bhsd` in this version.

- [ ] **Step 1: Write the failing QKV layout test**

Append to `python/tests/test_convrot.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convrot_qkv_bhsd_store():
    torch = pytest.importorskip("torch")
    tdp = pytest.importorskip("trt_dit_plugins")
    tdp.ensure_loaded()
    B, S_, H, D = 2, 8, 2, 128
    K, C = 2048, H * D
    x = torch.randn(B, S_, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 127, (3 * C, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(3 * C, device="cuda") * 0.01
    b = torch.randn(3 * C, device="cuda", dtype=torch.bfloat16)

    q, k, v = tdp.convrot_linear(
        x, w, ws, b, out_layout="bhsd", qkv_heads=H, head_dim=D)
    assert q.shape == k.shape == v.shape == (B, H, S_, D)

    # Reference uses the same rowwise int8 quant contract:
    # scale = absmax / 127, q = round(x / scale), dequant = q * scale.
    xf = x.float().reshape(-1, K)
    scale = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    xq = (xf / scale).round().clamp(-127, 127) * scale
    y = (xq @ w.float().T) * ws.float() + b.float()
    y = y.reshape(B, S_, 3, H, D)
    for got, idx in ((q, 0), (k, 1), (v, 2)):
        ref = y[:, :, idx].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        assert (got.float() - ref.float()).abs().mean() < 0.02
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_convrot.py::test_convrot_qkv_bhsd_store -q
```

Expected: FAIL with `TypeError: ... unexpected keyword argument 'out_layout'`.

- [ ] **Step 3: Add the custom QKV store visitor**

In `third_party/comfy/convrot/cutlass_gemm_int8.cu`, add a visitor modeled on `VisitorAuxStore` but with three output pointers and coordinate decomposition:

```cpp
// DIT-PLUGINS-PATCH: BHSD/QKV store. Writes each GEMM element once; no
// transpose kernel and no intermediate layout.
template <class ThreadMap, class Element, class VecType>
struct VisitorQkvStore {
    struct Arguments {
        Element* q;
        Element* k;
        Element* v;
        int64_t S;
        int64_t C;   // N per q/k/v (H*D)
        int H;
        int D;
    };

    struct Params {
        Element* q;
        Element* k;
        Element* v;
        int64_t S;
        int64_t C;
        int H;
        int D;
    };

    template <class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const&, Arguments const& args,
                                          void*) {
        return {args.q, args.k, args.v, args.S, args.C, args.H, args.D};
    }

    static bool can_implement(Arguments const& args) {
        return args.S > 0 && args.C > 0 && args.H > 0 && args.D > 0 &&
               args.C == int64_t(args.H) * args.D;
    }

    static int constexpr vec_bits =
        ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;
    static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

    template <class GTensor, class RTensor, class CTensor, class ProblemShape>
    struct Callbacks : EmptyCallbacks {
        CUTLASS_DEVICE Callbacks(GTensor&& tC_g, RTensor&& tC_r,
                                 CTensor&& tC_c, ProblemShape problem_shape,
                                 Params const* params_ptr)
            : tC_gAux(cute::forward<GTensor>(tC_g)),
              tC_rAux(cute::forward<RTensor>(tC_r)),
              tC_cAux(cute::forward<CTensor>(tC_c)),
              problem_shape(problem_shape), params_ptr(params_ptr) {}

        GTensor tC_gAux;
        RTensor tC_rAux;
        CTensor tC_cAux;
        ProblemShape problem_shape;
        Params const* params_ptr;

        CUTLASS_DEVICE void end_step(int step_idx) {
            auto src_v = filter(tC_rAux);
            auto coord_v = filter(tC_cAux(_, _, _, step_idx));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(src_v); ++i) {
                auto coord = coord_v(i);
                int m = int(get<0>(coord));
                int n = int(get<1>(coord));
                bool guard = elem_less(coord, problem_shape);
                int b = m / int(params_ptr->S);
                int s = m % int(params_ptr->S);
                int qkv = n / int(params_ptr->C);
                int c = n % int(params_ptr->C);
                int h = c / params_ptr->D;
                int d = c % params_ptr->D;
                Element* dst = qkv == 0 ? params_ptr->q
                              : (qkv == 1 ? params_ptr->k : params_ptr->v);
                if (dst == nullptr)
                    continue;
                void* addr = dst +
                    ((int64_t(b) * params_ptr->H + h) * params_ptr->S + s) *
                        params_ptr->D + d;
                cutlass::arch::global_store<VecType, sizeof(VecType)>(
                    src_v(i), addr, guard);
            }
        }
    };
};
```

`VecType` is a 128-bit vector; `D=128` and `C=H*D` are multiples of the BF16 vector length (8), so a vector never crosses a head or QKV boundary. Add to `can_implement`:

```cpp
return args.S > 0 && args.C > 0 && args.H > 0 && args.D > 0 &&
       args.C == int64_t(args.H) * args.D &&
       args.D % VecLength == 0 && args.C % VecLength == 0;
```

This is a single-pass vector store; there is no scalar fallback and no second kernel on the M≥3 path.

Add `FusedInt8GemmQkv` as a copy of `FusedInt8Gemm` with `EVTD = Sm80EVT<VisitorQkvStore, EVT1>` and a launcher:

```cpp
int launch_cutlass_int8_qkv(const int8_t* A, const int8_t* B,
    const float* xs, const float* ws, const void* bias,
    void* q, void* k, void* v, int64_t M, int64_t N, int64_t K,
    int64_t S, int H, int D, int out_dtype, int bias_dtype,
    int config, cudaStream_t stream);
```

For `qkv_heads=0`, pass `q=out`, `k=nullptr`, `v=nullptr`, `C=N`, and make the visitor write only qkv 0.

- [ ] **Step 4: Add the M<3 layout fallback**

For `M < 3`, use a project-owned `src/kernel/convrot/layout.cu` kernel:

```cpp
template <typename T>
__global__ void qkv_bhsd_kernel(const T* __restrict__ src, T* __restrict__ q,
                                T* __restrict__ k, T* __restrict__ v,
                                int64_t B, int64_t S, int64_t H, int64_t D,
                                int qkv_count) {
    int64_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * S * qkv_count * H * D;
    if (flat >= total) return;
    int64_t i = flat;
    int64_t d = i % D; i /= D;
    int64_t h = i % H; i /= H;
    int64_t qkv = 0, s = 0, b = 0;
    if (qkv_count == 3) {
        qkv = i % 3; i /= 3;
        s = i % S; i /= S;
        b = i;
    } else {
        s = i % S; i /= S;
        b = i;
    }
    T* dst = qkv_count == 3
        ? (qkv == 0 ? q : (qkv == 1 ? k : v)) : q;
    dst[((b * H + h) * S + s) * D + d] = src[flat];
}
```

The source order is `[B,S,3,H,D]` for QKV (or `[B,S,H,D]` for a single
tensor), so the index decomposition above is `d, h, qkv, s, b`; do not reuse
the earlier `s`-before-`qkv` order.

Add `src/kernel/convrot/layout.cu` to `CMakeLists.txt`. Use it only when
`M < 3`; for the target `M=16384`, the CUTLASS store is used.

- [ ] **Step 5: Extend the wrapper outputs**

In `surgeon._SPECS`, add `out_layout`, `qkv_heads`, and `head_dim` to both
ConvRot entries. In `validate_plugin_nodes`, the expected output count follows
the field:

```python
if node.op in (PluginOp.CONVROT_LINEAR, PluginOp.CONVROT_LINEAR_W4A4):
    want_out = 3 if int(node.attrs.get("qkv_heads", 0)) else 1
    if len(node.outputs) != want_out:
        errs.append(f"{node.name or node.op}: want {want_out} outputs, got {len(node.outputs)}")
```

Add fields:

```cpp
int32_t outLayout_ = 0;   // 0 bsc, 1 bhsd
int32_t qkvHeads_ = 0;    // 0 or H
int32_t headDim_ = 0;     // 0 or D
```

Grow `fields_` to 9, add the entries to `refreshFields`, and parse them with
`readInt` in `createPlugin` (defaults `0`). Append
`/l<out_layout>/h<qkv_heads>/d<head_dim>` to `updateTimingID`. `getNbOutputs`:

```cpp
return qkvHeads_ > 0 ? 3 : 1;
```

`getOutputDataTypes`: fill `nbOut` entries with `outType(in[kX], outDtype_)`; accept `nbOut == (qkvHeads_ > 0 ? 3 : 1)`.
`getOutputShapes`:

```cpp
if (outLayout_ == 0)
    return existing bsc shape;
if (in[kX].nbDims != 3)
    return -1;
// S may stay -1 here; the custom store receives the runtime S as a kernel
// argument. H and D are fixed fields, so only N must be known.
int64_t N = in[kW].d[0];
int64_t D = headDim_ > 0 ? headDim_ : 128;  // Anima default
int64_t C = qkvHeads_ > 0 ? N / 3 : N;
int64_t H = qkvHeads_ > 0 ? qkvHeads_ : C / D;
if (H <= 0 || D <= 0 || C % D != 0 || H * D != C)
    return -1;
for (int i = 0; i < nbOut; ++i) {
    out[i].nbDims = 4;
    out[i].d[0] = in[kX].d[0];
    out[i].d[1] = H;
    out[i].d[2] = in[kX].d[1];
    out[i].d[3] = D;
}
```

The custom store receives `S` as a runtime kernel argument; the build-time `DimsExprs` may keep `S=-1` and TRT substitutes the profile at execution. The only fixed requirements are `H` and `D`, which come from the plugin fields (or from `N` when `head_dim=0`).

`onShapeChange` stores `S_`, `B_`, `N_`, `H_`, `D_`; validates `N_ == (qkvHeads_ > 0 ? 3 : 1) * H_ * D_`; rejects `epilogue_ != 0 || residualGate_` with `out_layout_ == 1`.

`enqueue` calls `convrot_gemm_ex` with an output pointer array:

```cpp
// Declared in convrot_shim.h:
// int convrot_gemm_layout(void* const* y, const void* x, const int8_t* w,
//     const float* ws, const void* bias, const int8_t* xq, const float* xs,
//     int64_t M, int64_t N, int64_t K, int64_t S, int H, int D,
//     int out_layout, void* scratch, size_t scratch_bytes, cudaStream_t);
void* outs[3]{out[0], qkvHeads_ > 0 ? out[1] : nullptr,
              qkvHeads_ > 0 ? out[2] : nullptr};
return convrot_gemm_layout(outs, in[kX], w, ws, bias, xq, xs, M_, N_, K_,
    S_, (int)H_, (int)D_, outLayout_, scratch,
    scratchBytes - quantBytes - scaleBytes, stream) == 0 ? 0 : -1;
```

- [ ] **Step 6: Add Python layout arguments**

In `ops.py`, extend `convrot_linear`:

```python
def convrot_linear(x, weight, weight_scale, bias=None, *, out_dtype=None,
                   gemm_config=-1, epilogue="none", residual=None, gate=None,
                   out_layout="bsc", qkv_heads=0, head_dim=0, profile=None):
```

Validation:

```python
if out_layout not in ("bsc", "bhsd"):
    raise ValueError("out_layout must be 'bsc' or 'bhsd'")
if w4a4 and (out_layout != "bsc" or qkv_heads):
    raise ValueError("W4A4 does not support out_layout/qkv_heads yet")
if out_layout == "bhsd":
    if x.dim() != 3:
        raise ValueError("bhsd output requires x [B,S,K]")
    if epilogue != "none" or residual is not None:
        raise ValueError("bhsd output does not support epilogue/residual yet")
    D = head_dim or 128
    if qkv_heads and weight.shape[0] != 3 * qkv_heads * D:
        raise ValueError("qkv weight N must be 3*H*D")
```

Output specs:

```python
if out_layout == "bhsd":
    B, S_, _ = x.shape
    D = head_dim or 128
    H = qkv_heads or (weight.shape[0] // D)
    shapes = [(B, H, S_, D)] * (3 if qkv_heads else 1)
    names = ["q", "k", "v"] if qkv_heads else ["y"]
else:
    shapes = [tuple(out_shape)]
    names = ["y"]
```

Return `tuple(outs[n] for n in names)` if `qkv_heads` else `outs["y"]`.

- [ ] **Step 7: Extend `add_convrot_linear`**

Add `out_layout="bsc"`, `qkv_heads=0`, `head_dim=0`. When `qkv_heads > 0`, emit three output variables with shapes `[B,H,S,D]` and return a tuple. Add fields `out_layout`, `qkv_heads`, `head_dim`. Reject `epilogue/residual` combinations in the helper.

- [ ] **Step 8: Run tests and commit**

```bash
PYTHONPATH=python TRT_DIT_LIBDIR=build \
  python -m pytest python/tests/test_convrot.py -k "qkv or bhsd" -q
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
cmake --build build --parallel 12
git add src/wrapper/convrot_linear_wrapper.cpp src/kernel/convrot/convrot_shim.h \
  src/kernel/convrot/convrot_shim.cu src/kernel/convrot/layout.cu \
  third_party/comfy/convrot/cutlass_gemm_int8.cu third_party/comfy/VENDOR.md \
  python/trt_dit_plugins/ops.py python/trt_dit_plugins/surgeon.py \
  python/tests/test_convrot.py python/tests/test_surgeon.py
git commit -m "feat(convrot): add BHSD/QKV output store"
```

**Fallback if Task 5 misses the gate:** keep `out_layout="bsc"` and add a single `qkv_to_bhsd` plugin using `src/kernel/convrot/layout.cu`. That is still one layer per block, not 240 transpose layers. Measure before committing to the custom visitor. The fallback is allowed by the request's performance target if the final engine stays ≤220 ms.

---

### Task 6: Graph surgery wiring and end-to-end shape checks

**Files:**
- Modify: `python/trt_dit_plugins/surgeon.py`
- Modify: `python/trt_dit_plugins/select.py`
- Test: `python/tests/test_surgeon.py`
- Test: `python/tests/test_gpu.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `fuse_rope_blocks(..., qk_layout="bshd"|"bhsd", include_cross=False)`.
  - Cross-attention uses `add_rms_norm` inside that pass; no separate helper is
    required.
  - `fuse_norm_affine(..., plugin=None, weight=None, residual=None, gate=None)`.
  - `add_convrot_linear(..., epilogue, residual, gate, out_layout, qkv_heads, head_dim)`.

- [ ] **Step 1: Add `qk_layout` to `fuse_rope_blocks`**

Change the signature:

```python
def fuse_rope_blocks(graph, fq=None, epsilon: float = 1e-6, rot_dim: int = 0,
                     *, freqs: str | None = None, qk_layout: str = "bshd",
                     include_cross: bool = False) -> int:
```

Validate `qk_layout in ("bshd", "bhsd")`. In the plan loop:

```python
if qk_layout == "bhsd":
    qt, kt = q_in, k_in
else:
    qt = _transpose_bhsd(graph, q_in, base + "_toBHSD_q")
    kt = _transpose_bhsd(graph, k_in, base + "_toBHSD_k")
```

For `include_cross=True`, when `_gs_has_rope_math` is false and `rot_dim < 0`, insert two `rms_norm` nodes:

```python
qo = add_rms_norm(graph, q_in, qsc, layout="bshd",
                  name=base + "_rmsnorm_q")
ko = add_rms_norm(graph, k_in, ksc, layout="bshd",
                  name=base + "_rmsnorm_k")
attn.inputs[0], attn.inputs[1] = qo, ko
```

`q_in`/`k_in` for cross-attn are the pre-norm BSHD tensors; `add_rms_norm(layout="bshd")` reads BSHD and writes BHSD. For self-attn with `qk_layout="bhsd"`, `q_in`/`k_in` are already BHSD and the pair op is used with `rot_dim` (or `-1`).

Extend the plan tuple to carry a `"self"`/`"cross"` tag, and update the
idempotency check to skip attention nodes already fed by either
`rms_rope_split_half` or `rms_norm`:

```python
already = any(getattr(p, "op", "") in (_ROPE_OP, PluginOp.RMS_NORM)
              for t in attn.inputs[:2] for p in t.inputs)
if already:
    continue
```

For the cross branch, append `("cross", attn, q_in, k_in, qsc, ksc)` and
handle it separately from the self-attn plan that uses the pair op.

- [ ] **Step 2: Add a no-transpose surgery test**

Add this helper and test to `python/tests/test_surgeon.py`:

```python
def _pre_rope_graph_bhsd():
    B, H, S_, D = 1, 2, 256, 128
    q = gs.Variable("q", dtype=F32, shape=[B, H, S_, D])
    k = gs.Variable("k", dtype=F32, shape=[B, H, S_, D])
    v = gs.Variable("v", dtype=F32, shape=[B, H, S_, D])
    qs = gs.Variable("qs", dtype=F32, shape=[D])
    ks = gs.Variable("ks", dtype=F32, shape=[D])
    c1 = gs.Constant("c1", values=np.ones((1,), dtype=F32))
    nq = gs.Variable("nq", dtype=F32, shape=[B, H, S_, D])
    nk = gs.Variable("nk", dtype=F32, shape=[B, H, S_, D])
    mq = gs.Variable("mq", dtype=F32, shape=[B, H, S_, D])
    mk = gs.Variable("mk", dtype=F32, shape=[B, H, S_, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, H, S_, D])
    nodes = [
        gs.Node("RMSNormalization", inputs=[q, qs], outputs=[nq]),
        gs.Node("RMSNormalization", inputs=[k, ks], outputs=[nk]),
        gs.Node("Mul", inputs=[nq, c1], outputs=[mq]),
        gs.Node("Mul", inputs=[nk, c1], outputs=[mk]),
        gs.Node("Attention", inputs=[mq, mk, v], outputs=[ao]),
    ]
    g = gs.Graph(nodes=nodes, inputs=[q, k, v, qs, ks], outputs=[ao])
    g.opset = 23
    return g


def test_fuse_rope_blocks_bhsd_skips_transposes():
    rng = np.random.default_rng(3)
    g = _pre_rope_graph_bhsd()
    fq = _baked_freqs(rng.random(64).astype(F32) * 0.1 + 0.01, 256)
    assert S.fuse_rope_blocks(g, fq=fq, qk_layout="bhsd") == 1
    assert not [n for n in g.nodes if n.op == "Transpose"]
    assert S.validate_plugin_nodes(g) == []
```

- [ ] **Step 3: Add a cross-attention surgery test**

Add this helper and test to `python/tests/test_surgeon.py`:

```python
def _cross_norm_graph():
    B, H, Sq, Sk, D = 1, 2, 64, 16, 128
    q = gs.Variable("q", dtype=F32, shape=[B, Sq, H, D])
    k = gs.Variable("k", dtype=F32, shape=[B, Sk, H, D])
    v = gs.Variable("v", dtype=F32, shape=[B, Sk, H, D])
    qs = gs.Variable("qs", dtype=F32, shape=[D])
    ks = gs.Variable("ks", dtype=F32, shape=[D])
    c1 = gs.Constant("c1", values=np.ones((1,), dtype=F32))
    nq = gs.Variable("nq", dtype=F32, shape=[B, Sq, H, D])
    nk = gs.Variable("nk", dtype=F32, shape=[B, Sk, H, D])
    mq = gs.Variable("mq", dtype=F32, shape=[B, Sq, H, D])
    mk = gs.Variable("mk", dtype=F32, shape=[B, Sk, H, D])
    ao = gs.Variable("ao", dtype=F32, shape=[B, H, Sq, D])
    nodes = [
        gs.Node("RMSNormalization", inputs=[q, qs], outputs=[nq]),
        gs.Node("RMSNormalization", inputs=[k, ks], outputs=[nk]),
        gs.Node("Mul", inputs=[nq, c1], outputs=[mq]),
        gs.Node("Mul", inputs=[nk, c1], outputs=[mk]),
        gs.Node("Attention", inputs=[mq, mk, v], outputs=[ao]),
    ]
    g = gs.Graph(nodes=nodes, inputs=[q, k, v, qs, ks], outputs=[ao])
    g.opset = 23
    return g


def test_fuse_rope_blocks_cross_attn_different_s():
    g = _cross_norm_graph()
    dummy = np.zeros((1, 1, 1, 1, 2, 2), F32)
    assert S.fuse_rope_blocks(
        g, fq=dummy, rot_dim=-1, include_cross=True) == 1
    rms = [n for n in g.nodes if n.op == "rms_norm"]
    attn = next(n for n in g.nodes if n.op == "Attention")
    assert len(rms) == 2
    assert attn.inputs[0].shape == [1, 2, 64, 128]
    assert attn.inputs[1].shape == [1, 2, 16, 128]
    assert not [n for n in g.nodes if n.op == "Transpose"]
```

The cross-attention branch must not call the pair op; it inserts two independent `rms_norm` nodes because `Sq != Sk`.

- [ ] **Step 4: Add ConvRot surgery assertions**

In `python/tests/test_convrot.py`, assert the new fields round-trip:

```python
g, y = _convrot_node()
node = g.nodes[0]
assert node.attrs["epilogue"] == 0
assert node.attrs["residual_gate"] == 0
assert node.attrs["out_layout"] == 0
assert node.attrs["qkv_heads"] == 0
assert node.attrs["head_dim"] == 0
```

and a qkv graph returns three outputs with `[B,H,S,D]` shapes.

- [ ] **Step 5: Run all CPU tests**

```bash
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
PYTHONPATH=python python -m pytest python/tests/test_convrot.py -q
```

Expected: PASS, including the no-transpose and cross-attn cases.

- [ ] **Step 6: Commit**

```bash
git add python/trt_dit_plugins/surgeon.py python/trt_dit_plugins/select.py \
  python/tests/test_surgeon.py python/tests/test_convrot.py
git commit -m "feat(surgeon): wire layout-aware rope, RMSNorm, and ConvRot fields"
```

---

### Task 7: Minimal fused block benchmark, docs, and release

**Files:**
- Create: `python/tests/test_fusion_e2e.py`
- Create: `docs/fusion-layout-extensions.md`
- Modify: `README.md`, `README.ja.md`, `README.zh-CN.md`
- Modify: `python/trt_dit_plugins/__init__.py`
- Modify: `CONTRIBUTING.md` if new test commands are needed

**Interfaces:**
- Consumes: all prior tasks.
- Produces: measured evidence and release `0.6.0` (only after the gates pass).

- [ ] **Step 1: Add a minimal dynamic fused block test**

Create `python/tests/test_fusion_e2e.py` with a composition test (this
exercises the same op contracts without the Anima checkpoint):

```python
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

tdp = pytest.importorskip("trt_dit_plugins")
tdp.ensure_loaded()

C, H, D, K = 2048, 16, 128, 2048
PROFILES = [(1, 256), (1, 4096), (2, 4096)]


@pytest.mark.parametrize("B,S", PROFILES)
def test_fused_block_ops_dynamic(B, S):
    x = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)
    wq = torch.randint(-127, 127, (3 * C, K), device="cuda", dtype=torch.int8)
    wq_s = torch.rand(3 * C, device="cuda") * 0.01
    bq = torch.randn(3 * C, device="cuda", dtype=torch.bfloat16)

    q, k, v = tdp.convrot_linear(
        x, wq, wq_s, bq, out_layout="bhsd", qkv_heads=H, head_dim=D,
        profile={"x": ((1, 256, K), (1, 4096, K), (2, 4096, K))})

    fq = torch.zeros(1, 1, S, D // 2, 2, 2, device="cuda")
    fq[..., 0, 0] = 1.0
    fq[..., 0, 3] = 1.0
    qs = torch.ones(D, device="cuda")
    qo, ko = tdp.rms_rope_split_half(
        q, k, fq, qs, qs,
        profile={"q": ((1, H, 256, D), (1, H, 4096, D), (2, H, 4096, D)),
                 "k": ((1, H, 256, D), (1, H, 4096, D), (2, H, 4096, D)),
                 "f": ((1, 1, 256, D // 2, 2, 2),
                       (1, 1, 4096, D // 2, 2, 2),
                       (1, 1, 4096, D // 2, 2, 2))})
    attn = tdp.int8_attention(qo, ko, v)
    assert attn.shape == (B, H, S, D)

    wo = torch.randint(-127, 127, (C, C), device="cuda", dtype=torch.int8)
    wo_s = torch.rand(C, device="cuda") * 0.01
    bo = torch.randn(C, device="cuda", dtype=torch.bfloat16)
    y = tdp.convrot_linear(attn.permute(0, 2, 1, 3).contiguous(),
                           wo, wo_s, bo,
                           profile={"x": ((1, 256, C), (1, 4096, C), (2, 4096, C))})
    assert y.shape == (B, S, C)
```

Add a second test that builds the corresponding ONNX with `surgeon` and
asserts:
- `S.validate_plugin_nodes(g) == []`;
- no `Transpose` nodes in the rope/QKV region;
- the graph contains one ConvRot QKV node, one rope node, one attention node,
  one ConvRot out node, and one AdaLN node (not the decomposed Myelin chain).

Do not compare against the Anima checkpoint in this repo; that comparison is
the Reforge handoff gate.

- [ ] **Step 2: Measure op timings**

Add a small timing helper in the test (or a `bench/` script) using `torch.cuda.Event`:

```python
def _time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
```

Record before/after for:
- `rms_rope_split_half` at `B=1,H=16,S=16384,D=128`;
- `rms_adaln` / `layer_norm_adaln` at `B=1,S=16384,C=2048` with and without residual/gate;
- `convrot_linear` at `M=16384,N=2048,K=2048` with `epilogue=none/gelu`, and qkv `N=6144`;
- full minimal block.

Put the measured table in `docs/fusion-layout-extensions.md`. Do not claim the 220 ms Anima gate until the Reforge engine is measured.

- [ ] **Step 3: Write the contract doc**

Create `docs/fusion-layout-extensions.md` with:

```markdown
# Fusion / layout extensions (0.6.0)

## Dynamic RMS+RoPE
- `rot_dim=-1` = RMS-only; `rot_dim=0` = all dims (unchanged).
- `freqs` S must equal Q/K S when rotating; `freqs` H may be 1 or H;
  `freqs` B may be 1 or B.
- Dynamic B/S are accepted; D=128 is fixed.

## Cross-attention RMSNorm
- Use `rms_norm(x, weight, layout="bshd")` twice; Q and K may have different S.
- The pair op requires equal Q/K shapes.

## AdaLN
- Ops: `rms_adaln`, `layer_norm_adaln`.
- Inputs: `x, scale, shift[, weight][, residual, gate]`.
- 3D x and `[B,1,D]` modulation are accepted; `out = residual + gate*y`.
- Modulation tensors share one dtype; x/out/residual share another.

## ConvRot
- `epilogue`: 0 none, 1 exact GELU (erf), 2 SiLU.
- `residual_gate`: `out = residual + gate*y`; B is looped per batch.
- `out_layout="bhsd"`, `qkv_heads=H`: three `[B,H,S,D]` outputs.
- W4A4 rejects epilogue/layout fields in 0.6.0.
```

- [ ] **Step 4: Update READMEs**

Add the new ops/fields to the feature table in all three READMEs. Keep the existing wording for the fused INT8 attention path unchanged. Update the version in `python/trt_dit_plugins/__init__.py` to `0.6.0`.

- [ ] **Step 5: Run the full verification**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 12
./build/regcheck ./build/libtrt_dit_plugins.so
PYTHONPATH=python python -m pytest python/tests/test_surgeon.py -q
PYTHONPATH=python python -m pytest python/tests/test_convrot.py -q
PYTHONPATH=python TRT_DIT_LIBDIR=build python -m pytest python/tests/test_gpu.py -q
PYTHONPATH=python TRT_DIT_LIBDIR=build python -m pytest python/tests/test_fusion_e2e.py -q
```

Expected: all pass on sm89. If a GPU test is skipped for missing hardware, record which gate was skipped; do not claim the release gate.

- [ ] **Step 6: Release only after the Anima handoff gate**

The Reforge engine remains unchanged. Hand the plugin repo to the Reforge owner with:
- the built wheel,
- `docs/fusion-layout-extensions.md`,
- the op-level timings,
- the minimal block test output.

Acceptance for the external Anima engine:
- `cos` ≥ the existing gate (`0.9966` at S=2048, `0.99865` at S=4096 on the prior real-data runs; use the stricter documented gate for the release profile),
- layer count < 1000,
- E2E ≤ 220 ms,
- attention path unchanged,
- no `Transpose` nodes in the rope/QKV region.

If the external engine is not measured, release the plugin as `0.6.0` but mark the E2E claim as “op-level verified, engine gate pending.”

- [ ] **Step 7: Commit and tag**

```bash
git add docs/fusion-layout-extensions.md README.md README.ja.md README.zh-CN.md \
  python/trt_dit_plugins/__init__.py python/tests/test_fusion_e2e.py
git commit -m "docs: document fusion/layout extensions for 0.6.0"
git tag -a v0.6.0-cu13 -m "trt-dit-plugins 0.6.0 (CUDA 13)"
git push origin master --tags
```

Do not publish the release until the wheel build and the GitHub release workflow succeed.

---

## Performance Targets and Interpretation

| Path | Current | Target | Measurement |
|---|---:|---:|---|
| RMS+RoPE + transposes | 39.3 ms | ≤ 12 ms | 40 blocks, `S=16384,H=16,D=128` |
| LayerNorm+AdaLN glue | 30.2 ms | ≤ 15 ms | 3 AdaLN ops/block, 40 blocks; the norm read/write remains |
| GELU decomposition | 10.9 ms | ≤ 2 ms | 40 blocks, exact erf, fused epilogue |
| QKV/attention transposes | ~29.6 ms | ≤ 10 ms | QKV BHSD + (if 4.4) attention output BHSD |
| Full Anima E2E | 289 ms | ≤ 220 ms | Reforge engine, same profile |

The request's “30.2 → 7 ms” for AdaLN is optimistic: the norm itself must read and write the 64 MB activation. The measurable saving is the extra modulation/replacement/reshape passes. Treat 15 ms as the realistic first gate and improve only with measured evidence.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `configurePlugin` still rejects dynamic dims because another check assumes positive extents | Add the dynamic GPU test first; search all `> 0` checks in the wrapper; test B-only, S-only, and both |
| `freqs` S mismatch is silently accepted and produces wrong rotation | Reject in `onShapeChange` and `enqueue` when `rot > 0`; test a mismatched profile |
| Cross-attention pair shape check fails | Use `rms_norm` per tensor; test `Sq != Sk` |
| AdaLN mixed dtypes silently cast to the wrong type | Restrict modulation to one dtype; validate in `onShapeChange`; test FP32 modulation + BF16 x |
| AdaLN 3D scale `[B,1,D]` row grouping wrong | Normalize to `Rs=B` and `scale_group=N/Rs`; test B=2 |
| CUTLASS QKV store scalar writes are slow | Measure; if the store dominates, vectorize along D (D=128, BF16 vector 8) before adding a fallback kernel |
| QKV store with B>1 or dynamic S indexes wrong | Test B=2 and profiles; the store takes runtime `S`; add a CPU reference for `(b,s,qkv,h,d)` |
| Per-batch residual loop changes numerics for B=2 | Quantize once over full M; only GEMM batches; compare per-batch against torch |
| Exact GELU `erff` is slower than the decomposed path | Measure; keep exact GELU because the model's pattern uses `Erf`; do not switch to tanh without a cos gate |
| W4A4 accidentally accepts new fields and falls into an unsupported template | Reject in `configurePlugin` and Python; add a test |
| Third-party patch drifts from upstream | Keep patches small, mark every block, update `VENDOR.md`, run the existing convrot/adaln tests |
| New fields break old serialized engines | Keep plugin version `"1"`; all new fields default to 0/legacy; add a deserialization smoke test if an old engine fixture is available |

---

## Self-Review

**Spec coverage:**

- 4.1 dynamic B/S: Task 1.
- 4.1 `rot_dim=-1` / RMS-only: Task 1; cross-attention: Task 2.
- 4.1 q/k physical B and no transpose: Task 5 + Task 6.
- 4.2 3D input, broadcast, LayerNorm, weight, residual/gate, mixed dtype: Task 3.
- 4.3 GELU/SILU epilogue, residual/gate, BHSD/QKV, field domains: Tasks 4–5.
- 4.4 attention replacement: explicitly deferred; acceptance table records whether it is needed.
- Determinism, plugin version, field cache, build threads, third-party rules: Global Constraints.
- Release/docs: Task 7.

**Placeholder scan:** Every task has concrete test code, expected failure, implementation snippets, run commands, and a commit. The only intentionally deferred item is 4.4, called out in Scope Check and Task 7.

**Type consistency:** `rms_norm(x, weight, eps, layout, profile)` is used by `add_rms_norm` and the cross-attention surgery. `convrot_linear(..., epilogue, residual, gate, out_layout, qkv_heads, head_dim)` is used by both `ops.py` and `surgeon.py`. `fuse_norm_affine` uses the same `has_weight`/`residual_gate` fields as the plugin. The QKV store uses `S,C,H,D` consistently in the visitor and wrapper.
