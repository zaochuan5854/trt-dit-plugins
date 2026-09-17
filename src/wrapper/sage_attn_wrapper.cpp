// SPDX-License-Identifier: Apache-2.0
// TRT IPluginV3 for plain dense SageAttention ("sageattn").
// Registration: name="sage_attn", namespace="dit-plugins".
//
// Pure wrapper (no conversion shims): only dtypes natively consumed per SM
// are accepted, anything needing an upcast is rejected at build time.
//   Inputs Q,K,V: [B,H,S,D] contiguous LINEAR, uniform dtype FP32/FP16/BF16
//   (FP8 inputs rejected everywhere: no kernel consumes them natively).
//   GQA allowed (Hq % Hkv == 0). D in {64,128,256}. Non-causal.
// Output: same shape; BF16 for FP32 input, else input dtype.
// Tactic (plugin field fp8_pv, NOT auto):
//   0 (default) = QK-INT8 + PV-FP16 portable path (quant_qk/quant_v/
//       sage_attn_launcher, mma.sync based, runs sm80/86/89/90/100/120;
//       mirrors upstream sm80/86).
//   1 (fp8_pv=1) = QK-INT8 + PV-FP8 dense SageAttention2 via the sparge sm89
//       kernels with an all-ones mask (mirrors upstream sm89 Sage2++).
//       Requires sm89, D in {64,128}, S % 128 == 0, FP16/BF16 inputs,
//       Lq == Lk; otherwise the build fails. Measured slower than tactic 0
//       for dense shapes (sparge LUT/preprocess overhead); opt-in only.
// Workspace is sized exactly per tactic (over-reserving measurably slows
// enqueue, so no max()-of-both).
// FP8-MMA dense tactics for sm90+ need kernels not vendored here;
// tactic 0 remains the portable fallback there.
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

extern "C" {
void launch_quant_qk_per_thread_int8(const void* q, void* q_int8, void* q_scale,
    const void* k, void* k_int8, void* k_scale, int B, int H_q, int Lq, int H_kv,
    int Lk, int C, int BLKQ, int WARPQ, int BLKK, int WARPK, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b,
    int64_t k_stride_h, int64_t k_stride_n, int input_dtype_code,
    void* anchor_indices, cudaStream_t stream);
void launch_quant_v_int8_kernel(const void* v, void* out, void* scale, int B,
    int H, int N, int D, int padded_N, int64_t sb, int64_t sh, int64_t sn,
    int input_dtype_code, cudaStream_t stream);
void launch_sage_attn_kernel(const void* q, const void* k, const void* v,
    void* o, const void* q_scale, const void* k_scale, const void* v_scale,
    const void* mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int cta_k,
    int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int head_dim, int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k, int stride_bz_v,
    int stride_h_v, int stride_d_v, int stride_bz_o, int stride_seq_o,
    int stride_h_o, float sm_scale, int output_dtype_code, cudaStream_t stream);
void launch_sparge_preprocess(const void* q, const void* k, const void* mask, const void* v,
    int8_t* q_i8, float* q_s, int8_t* k_i8, float* k_s, float* km, int32_t* lut, int32_t* vbn,
    void* vT, void* v_fp8, float* v_s, void* v_fp16_tmp, int B, int Hq, int Hkv, int S, int D,
    int is_bf16, void* stream);
void launch_block_sparse_sage2_sm89(int8_t* q_i8, int8_t* k_i8, void* v_fp8, void* o,
    int32_t* lut, int32_t* vbn, float* pv_thr, float* q_s, float* k_s, float* v_s, int B, int Hq,
    int Hkv, int S, int D, int is_bf16, float sm_scale, void* stream);
}

namespace {
constexpr char const* kName = "sage_attn";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "dit-plugins";
constexpr int kMaxHeads = 256;

int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}
// FP32 input -> BF16 output, else same dtype as input.
nvinfer1::DataType outTypeFor(nvinfer1::DataType in) {
    return in == nvinfer1::DataType::kFLOAT ? nvinfer1::DataType::kBF16 : in;
}

int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

// Tactic 0 workspace (same layout as int8_attention).
struct WsPlan0 {
    int cta_k, padded_Lk;
    int64_t off_qi, off_ki, off_vi, off_qs, off_ks, off_vs, off_anchor, total;
};

WsPlan0 planWs0(int B, int Hq, int Lq, int Hkv, int Lk, int D) {
    WsPlan0 p{0, 0, 0, 0, 0, 0, 0, 0, 0};
    p.cta_k = (D >= 128 && Lk > 1024) ? 128 : 64;
    p.padded_Lk = (Lk + p.cta_k - 1) / p.cta_k * p.cta_k;
    int64_t off = 0;
    p.off_qi = off;
    off += align16((int64_t)B * Hq * Lq * D);
    p.off_ki = off;
    off += align16((int64_t)B * Hkv * Lk * D);
    p.off_vi = off;
    off += align16((int64_t)B * Hkv * D * p.padded_Lk);
    p.off_qs = off;
    off += align16((int64_t)B * Hq * ((Lq + 127) / 128) * (D == 256 ? 64 : 32) * 4);
    p.off_ks = off;
    off += align16((int64_t)B * Hkv * ((Lk + p.cta_k - 1) / p.cta_k) * 4 * 4);
    p.off_vs = off;
    off += align16((int64_t)B * Hkv * D * 4);
    p.off_anchor = off;
    off += align16((int64_t)B * Hkv * 4);
    p.total = off;
    return p;
}

// Tactic 1 workspace (same layout as block_sparse_sage2_attn) + all-ones mask.
struct WsPlan1 {
    int64_t off_qi, off_qs, off_ki, off_ks, off_km, off_lut, off_vbn, off_vT, off_vfp8, off_vs,
        off_vtmp, off_pv, off_mask, total;
};

WsPlan1 planWs1(int B, int Hq, int Hkv, int S, int D) {
    WsPlan1 p{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int NQB = (S + 127) / 128, NKB = (S + 63) / 64, P = (S + 127) / 128 * 128;
    int64_t off = 0, sd = (int64_t)B * S * D;
    p.off_qi = off;
    off += align16((int64_t)Hq * sd);
    p.off_qs = off;
    off += align16((int64_t)Hq * B * NQB * 4);
    p.off_ki = off;
    off += align16((int64_t)Hkv * sd);
    p.off_ks = off;
    off += align16((int64_t)Hkv * B * NKB * 4);
    p.off_km = off;
    off += align16((int64_t)Hkv * B * D * 4);
    p.off_lut = off;
    off += align16((int64_t)Hq * B * NQB * NKB * 4);
    p.off_vbn = off;
    off += align16((int64_t)Hq * B * NQB * 4);
    p.off_vT = off;
    off += align16((int64_t)Hkv * B * D * P * 2);
    p.off_vfp8 = off;
    off += align16((int64_t)Hkv * B * D * P);
    p.off_vs = off;
    off += align16((int64_t)Hkv * B * D * 4);
    p.off_vtmp = off;
    off += align16((int64_t)Hkv * sd * 2);
    p.off_pv = off;
    off += align16((int64_t)Hq * 4);
    p.off_mask = off;
    off += align16((int64_t)B * Hq * NQB * NKB * 4);
    p.total = off;
    return p;
}

class SageAttn : public nvinfer1::IPluginV3,
                 public nvinfer1::IPluginV3OneCore,
                 public nvinfer1::IPluginV3OneBuild,
                 public nvinfer1::IPluginV3OneRuntime {
    friend class SageAttnCreator;

public:
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
        try {
            return new SageAttn(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
        const nvinfer1::DataType* in, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != 3)
            return -1;
        for (int i = 1; i < 3; ++i)
            if (in[i] != in[0])
                return -1;
        if (in[0] != nvinfer1::DataType::kFLOAT && in[0] != nvinfer1::DataType::kHALF &&
            in[0] != nvinfer1::DataType::kBF16)
            return -1;
        out[0] = outTypeFor(in[0]);
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out,
        int32_t nbOut, nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 3 || nbOut != 1)
            return -1;
        out[0] = in[0]; // [B,Hq,Lq,D] passthrough
        return 0;
    }

    // Inputs: FP32/FP16/BF16 only (FP8 rejected: no kernel consumes it
    // natively, and this plugin ships no upcast). kLINEAR only.
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos < nbIn)
            return d.type == nvinfer1::DataType::kFLOAT || d.type == nvinfer1::DataType::kHALF ||
                   d.type == nvinfer1::DataType::kBF16;
        return d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 3)
            return -1;
        for (int i = 0; i < 3; ++i)
            if (in[i].desc.dims.nbDims != 4)
                return -1;
        int d = in[0].max.d[3];
        return (d == 64 || d == 128 || d == 256) ? 0 : -1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        if (nbIn != 3)
            return 0;
        int B = in[0].max.d[0], Hq = in[0].max.d[1], Lq = in[0].max.d[2], D = in[0].max.d[3];
        int Hkv = in[1].max.d[1], Lk = in[1].max.d[2];
        if (D != 64 && D != 128 && D != 256)
            return 0;
        if (fp8pv_) {
            // Tactic 1 workspace, exact size (no over-reserve: oversized
            // workspace measurably slows enqueue, ~0.6ms here).
            if ((D != 64 && D != 128) || Lq != Lk || Lk < 128 || Lk % 128 != 0)
                return 0;
            return (size_t)planWs1(B, Hq, Hkv, Lk, D).total;
        }
        return (size_t)planWs0(B, Hq, Lq, Hkv, Lk, D).total;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 3 || in[0].dims.nbDims != 4)
            return -1;
        B_ = in[0].dims.d[0];
        Hq_ = in[0].dims.d[1];
        Lq_ = in[0].dims.d[2];
        D_ = in[0].dims.d[3];
        Hkv_ = in[1].dims.d[1];
        Lk_ = in[1].dims.d[2];
        if (D_ != 64 && D_ != 128 && D_ != 256)
            return -1;
        if (Hq_ % Hkv_ != 0 || Hq_ > kMaxHeads)
            return -1;
        if (in[1].type != in[0].type || in[2].type != in[0].type)
            return -1;
        if (in[0].type != nvinfer1::DataType::kFLOAT &&
            in[0].type != nvinfer1::DataType::kHALF &&
            in[0].type != nvinfer1::DataType::kBF16)
            return -1;
        inCode_ = dtypeCode(in[0].type);
        outCode_ = dtypeCode(outTypeFor(in[0].type));
        smScale_ = 1.0f / std::sqrt((float)D_);
        // Per-SM tactic: fp8-PV dense Sage2 needs sm89 SASS plus
        // D in {64,128}, S % 128 == 0, FP16/BF16 inputs, square shapes.
        // Opt-in only via the fp8_pv field: measurements show the sparge
        // dense path is slower than tactic0 for dense shapes, so auto
        // selection stays on the portable path.
        tactic_ = 0;
        if (fp8pv_) {
            int dev = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&dev) != cudaSuccess ||
                cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
                return -1;
            if (prop.major != 8 || prop.minor != 9 || (D_ != 64 && D_ != 128) ||
                Lq_ != Lk_ || Lk_ < 128 || Lk_ % 128 != 0 ||
                in[0].type == nvinfer1::DataType::kFLOAT)
                return -1;
            tactic_ = 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void* ws, cudaStream_t stream) noexcept override {
        try {
            char* w = static_cast<char*>(ws);
            if (tactic_ == 1) {
                WsPlan1 p = planWs1(B_, Hq_, Hkv_, Lk_, D_);
                float pvhost[kMaxHeads];
                for (int i = 0; i < Hq_; ++i) pvhost[i] = 50.0f;
                if (cudaMemcpyAsync(w + p.off_pv, pvhost, (size_t)Hq_ * 4, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess)
                    return -1;
                // All-ones dense mask: LUT builder tests truthiness (if (m[i])),
                // so byte 0x01 (int32 0x01010101) counts as active.
                int NQB = (Lk_ + 127) / 128, NKB = (Lk_ + 63) / 64;
                if (cudaMemsetAsync(w + p.off_mask, 0x01,
                        (size_t)B_ * Hq_ * NQB * NKB * 4, stream) != cudaSuccess)
                    return -1;
                launch_sparge_preprocess(in[0], in[1], w + p.off_mask, in[2],
                    reinterpret_cast<int8_t*>(w + p.off_qi), reinterpret_cast<float*>(w + p.off_qs),
                    reinterpret_cast<int8_t*>(w + p.off_ki), reinterpret_cast<float*>(w + p.off_ks),
                    reinterpret_cast<float*>(w + p.off_km), reinterpret_cast<int32_t*>(w + p.off_lut),
                    reinterpret_cast<int32_t*>(w + p.off_vbn), w + p.off_vT, w + p.off_vfp8,
                    reinterpret_cast<float*>(w + p.off_vs), w + p.off_vtmp, B_, Hq_, Hkv_, Lk_, D_,
                    inCode_ == 2 ? 1 : 0, stream);
                launch_block_sparse_sage2_sm89(reinterpret_cast<int8_t*>(w + p.off_qi),
                    reinterpret_cast<int8_t*>(w + p.off_ki), w + p.off_vfp8, out[0],
                    reinterpret_cast<int32_t*>(w + p.off_lut), reinterpret_cast<int32_t*>(w + p.off_vbn),
                    reinterpret_cast<float*>(w + p.off_pv), reinterpret_cast<float*>(w + p.off_qs),
                    reinterpret_cast<float*>(w + p.off_ks), reinterpret_cast<float*>(w + p.off_vs),
                    B_, Hq_, Hkv_, Lk_, D_, inCode_ == 2 ? 1 : 0, smScale_, stream);
                return 0;
            }
            WsPlan0 p = planWs0(B_, Hq_, Lq_, Hkv_, Lk_, D_);
            int64_t sb_q = (int64_t)Hq_ * Lq_ * D_, sh_q = (int64_t)Lq_ * D_;
            int64_t sb_k = (int64_t)Hkv_ * Lk_ * D_, sh_k = (int64_t)Lk_ * D_;
            int64_t sb_v = (int64_t)Hkv_ * Lk_ * D_, sh_v = (int64_t)Lk_ * D_;
            launch_quant_qk_per_thread_int8(in[0], w + p.off_qi, w + p.off_qs, in[1],
                w + p.off_ki, w + p.off_ks, B_, Hq_, Lq_, Hkv_, Lk_, D_, 128,
                (D_ == 256) ? 16 : 32, p.cta_k, p.cta_k, sb_q, sh_q, D_, sb_k, sh_k,
                D_, inCode_, w + p.off_anchor, stream);
            launch_quant_v_int8_kernel(in[2], w + p.off_vi, w + p.off_vs, B_, Hkv_,
                Lk_, D_, p.padded_Lk, sb_v, sh_v, D_, inCode_, stream);
            int64_t v_bz = (int64_t)Hkv_ * D_ * p.padded_Lk, v_h = (int64_t)D_ * p.padded_Lk;
            launch_sage_attn_kernel(w + p.off_qi, w + p.off_ki, w + p.off_vi, out[0],
                w + p.off_qs, w + p.off_ks, w + p.off_vs, nullptr, 0, 0, 0, 0, -1,
                p.cta_k, B_, Lq_, Lk_, Hq_, Hkv_, D_, (int)(sb_q), D_, (int)sh_q,
                (int)(sb_k), D_, (int)(sh_k), (int)v_bz, (int)v_h, p.padded_Lk,
                (int)(sb_q), D_, (int)sh_q, smScale_, outCode_, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_[0] = {"fp8_pv", &fp8pv_, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 1;
        fc_.fields = fields_;
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return "sage_attn/1"; }

    int tactic() const noexcept { return tactic_; }

private:
    static constexpr int kMaxHeads = 256;
    int B_ = 0, Hq_ = 0, Hkv_ = 0, Lq_ = 0, Lk_ = 0, D_ = 0;
    int inCode_ = 0, outCode_ = 0, tactic_ = 0;
    int32_t fp8pv_ = 0;
    float smScale_ = 0;
    nvinfer1::PluginField fields_[1]{};
    nvinfer1::PluginFieldCollection fc_{};
};

class SageAttnCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase) noexcept override {
        try {
            auto* p = new SageAttn();
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i) {
                    if (std::strcmp(fc->fields[i].name, "fp8_pv") == 0 &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32)
                        p->fp8pv_ = *static_cast<int32_t const*>(fc->fields[i].data);
                }
            return p;
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        names_[0] = {"fp8_pv", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 1;
        fc_.fields = names_;
        return &fc_;
    }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    nvinfer1::PluginField names_[1]{};
    nvinfer1::PluginFieldCollection fc_{};
};

namespace {
SageAttnCreator* gCreator() {
    static SageAttnCreator* c = new SageAttnCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
} // namespace
