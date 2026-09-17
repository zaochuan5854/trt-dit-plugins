// SPDX-License-Identifier: Apache-2.0
// TRT IPluginV3 for int8-input RoPE+SageAttn fused DiT-self attention.
// Registration: name="fused_int8_rope_sage_attn", namespace="dit-plugins".
//
// Scope (see docs/fused-int8-rope-sage-spec.md):
// - DiT-self only (Q/K/V same [B,H,S,D]), Hq==Hkv, D==128, L>1024, non-causal.
// - Q/K/V arrive as INT8 + per-tensor FP32 input scales (int8 engine); the
//   plugin fuses dequant -> RMSNorm -> split-half RoPE (inv_freq[64]) ->
//   Sage per-thread INT8 quant, then runs the existing Sage attention body.
// - Output BF16 [B,H,S,D]. kLINEAR only.
//
// Fallback policy is surgery-level (build-time): unsupported shapes are
// rejected in configurePlugin/onShapeChange so exporters keep the eager/
// separation path. There is no runtime second tactic.

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>

extern "C" {
void launch_fused_qk_rope_int8(const void *q_i8, const void *q_s_in,
    void *q_int8, void *q_scale, const void *k_i8, const void *k_s_in,
    void *k_int8, void *k_scale, const void *rms_w_q, const void *rms_w_k,
    int norm_dtype_code, const void *inv_freq, int B, int H, int L, int C,
    void *ws_mean, cudaStream_t stream);
void launch_requant_v_int8(const void *v_i8, const void *v_s_in, void *out,
    void *scale, int B, int H, int N, int D, int padded_N, int64_t sb,
    int64_t sh, int64_t sn, cudaStream_t stream);
void launch_sage_attn_kernel(const void *q, const void *k, const void *v,
    void *o, const void *q_scale, const void *k_scale, const void *v_scale,
    const void *mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int cta_k,
    int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int head_dim, int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k, int stride_bz_v,
    int stride_h_v, int stride_d_v, int stride_bz_o, int stride_seq_o,
    int stride_h_o, float sm_scale, int output_dtype_code, cudaStream_t stream);
}

namespace {
constexpr char const *kName = "fused_int8_rope_sage_attn";
constexpr char const *kVersion = "1";
constexpr char const *kNamespace = "dit-plugins";

// Input positions.
constexpr int kQ = 0, kQsIn = 1, kK = 2, kKsIn = 3, kV = 4, kVsIn = 5,
              kQn = 6, kKn = 7, kInv = 8, kNbIn = 9;

int normCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2; // kBF16
}

int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

struct WsPlan {
    int cta_k, padded_L;
    int64_t off_qi, off_ki, off_qs, off_ks, off_mean, off_vi, off_vs, total;
};

WsPlan planWs(int B, int H, int L, int D) {
    WsPlan p{0, 0, 0, 0, 0, 0, 0, 0, 0};
    p.cta_k = (D >= 128 && L > 1024) ? 128 : 64;
    p.padded_L = (L + p.cta_k - 1) / p.cta_k * p.cta_k;
    int64_t off = 0;
    p.off_qi = off;
    off += align16((int64_t)B * H * L * D);
    p.off_ki = off;
    off += align16((int64_t)B * H * L * D);
    p.off_qs = off;
    off += align16((int64_t)B * H * ((L + 127) / 128) * 32 * 4);
    p.off_ks = off;
    off += align16((int64_t)B * H * ((L + p.cta_k - 1) / p.cta_k) * 4 * 4);
    p.off_mean = off;
    off += align16((int64_t)B * H * D * 4);
    p.off_vi = off;
    off += align16((int64_t)B * H * D * p.padded_L);
    p.off_vs = off;
    off += align16((int64_t)B * H * D * 4);
    p.total = off;
    return p;
}

class FusedInt8RopeSageAttn : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
public:
    nvinfer1::IPluginCapability *getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override {
        if (type == nvinfer1::PluginCapabilityType::kCORE)
            return static_cast<nvinfer1::IPluginV3OneCore *>(this);
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
            return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
            return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
        return nullptr;
    }
    nvinfer1::IPluginV3 *clone() noexcept override {
        try {
            return new FusedInt8RopeSageAttn(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const *getPluginName() const noexcept override {
        return kName;
    }
    nvinfer1::AsciiChar const *getPluginVersion() const noexcept override {
        return kVersion;
    }
    nvinfer1::AsciiChar const *getPluginNamespace() const noexcept override {
        return kNamespace;
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType *out, int32_t nbOut,
        const nvinfer1::DataType *in, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != kNbIn)
            return -1;
        out[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const *in, int32_t nbIn,
        nvinfer1::DimsExprs const *, int32_t, nvinfer1::DimsExprs *out,
        int32_t nbOut, nvinfer1::IExprBuilder &) noexcept override {
        if (nbIn != kNbIn || nbOut != 1)
            return -1;
        out[0] = in[kQ]; // [B,H,S,D] passthrough
        return 0;
    }

    // q/k/v INT8 [B,H,S,128]; scales FP32 scalar [1]; norms FP32/BF16 [128];
    // inv_freq FP32 [64]; output BF16. kLINEAR only.
    bool supportsFormatCombination(int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const *io, int32_t nbIn,
        int32_t) noexcept override {
        if (nbIn != kNbIn)
            return false;
        auto const &d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        switch (pos) {
        case kQ:
        case kK:
        case kV:
            return d.type == nvinfer1::DataType::kINT8;
        case kQsIn:
        case kKsIn:
        case kVsIn:
        case kInv:
            return d.type == nvinfer1::DataType::kFLOAT;
        case kQn:
        case kKn:
            return d.type == nvinfer1::DataType::kFLOAT ||
                   d.type == nvinfer1::DataType::kBF16;
        default:
            return d.type == nvinfer1::DataType::kBF16;
        }
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in,
        int32_t nbIn, nvinfer1::DynamicPluginTensorDesc const *,
        int32_t) noexcept override {
        if (nbIn != kNbIn)
            return -1;
        for (int i : {kQ, kK, kV})
            if (in[i].desc.dims.nbDims != 4)
                return -1;
        if (in[kQ].max.d[3] != 128)
            return -1; // D=128 only
        if (in[kQ].max.d[2] <= 1024)
            return -1; // ROTATION=128 path (L>1024); the L<=1024 path is
        // broken (wrong results/crash), so reject at build time.
        return 0;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const *in,
        int32_t nbIn, nvinfer1::DynamicPluginTensorDesc const *,
        int32_t) const noexcept override {
        if (nbIn != kNbIn)
            return 0;
        int B = in[kQ].max.d[0], H = in[kQ].max.d[1], L = in[kQ].max.d[2],
            D = in[kQ].max.d[3];
        if (D != 128 || L <= 1024)
            return 0;
        return (size_t)planWs(B, H, L, D).total;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const *in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const *, int32_t) noexcept override {
        if (nbIn != kNbIn)
            return -1;
        for (int i : {kQ, kK, kV}) {
            if (in[i].dims.nbDims != 4)
                return -1;
            if (in[i].type != nvinfer1::DataType::kINT8)
                return -1;
        }
        B_ = in[kQ].dims.d[0];
        H_ = in[kQ].dims.d[1];
        L_ = in[kQ].dims.d[2];
        D_ = in[kQ].dims.d[3];
        if (D_ != 128 || L_ <= 1024)
            return -1;
        for (int i : {kK, kV})
            for (int d = 0; d < 4; ++d)
                if (in[i].dims.d[d] != in[kQ].dims.d[d])
                    return -1; // DiT-self: identical shapes
        for (int i : {kQsIn, kKsIn, kVsIn}) {
            int n = 1;
            for (int d = 0; d < in[i].dims.nbDims; ++d)
                n *= in[i].dims.d[d];
            if (n != 1 || in[i].type != nvinfer1::DataType::kFLOAT)
                return -1; // per-tensor scalar FP32
        }
        for (int i : {kQn, kKn}) {
            if (in[i].dims.nbDims != 1 || in[i].dims.d[0] != 128)
                return -1;
            if (in[i].type != in[kQn].type)
                return -1;
        }
        if (in[kInv].dims.nbDims != 1 || in[kInv].dims.d[0] != 64 ||
            in[kInv].type != nvinfer1::DataType::kFLOAT)
            return -1;
        normCode_ = normCode(in[kQn].type);
        smScale_ = 1.0f / std::sqrt((float)D_);
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const *,
        nvinfer1::PluginTensorDesc const *, void const *const *in,
        void *const *out, void *ws, cudaStream_t stream) noexcept override {
        try {
            WsPlan p = planWs(B_, H_, L_, D_);
            char *w = static_cast<char *>(ws);
            // Sequential single-stream: sharing events across sequential
            // enqueues is unsound (a lagging main stream can observe a later
            // call's record, then read workspace already overwritten by that
            // call), and per-call events cannot be safely destroyed (must
            // outlive the wait's execution). Sound overlap needs full graph
            // capture.
            launch_fused_qk_rope_int8(in[kQ], in[kQsIn], w + p.off_qi,
                w + p.off_qs, in[kK], in[kKsIn], w + p.off_ki, w + p.off_ks,
                in[kQn], in[kKn], normCode_, in[kInv], B_, H_, L_, D_,
                w + p.off_mean, stream);
            const int64_t sb_v = (int64_t)H_ * L_ * D_, sh_v = (int64_t)L_ * D_;
            // Single-pass int8->int8 requant (s_in folded into block scale).
            launch_requant_v_int8(in[kV], in[kVsIn], w + p.off_vi,
                w + p.off_vs, B_, H_, L_, D_, p.padded_L, sb_v, sh_v, D_,
                stream);
            const int64_t sb_q = (int64_t)H_ * L_ * D_, sh_q = (int64_t)L_ * D_;
            const int64_t v_bz = (int64_t)H_ * D_ * p.padded_L,
                          v_h = (int64_t)D_ * p.padded_L;
            launch_sage_attn_kernel(w + p.off_qi, w + p.off_ki, w + p.off_vi,
                out[0], w + p.off_qs, w + p.off_ks, w + p.off_vs, nullptr, 0,
                0, 0, 0, -1, p.cta_k, B_, L_, L_, H_, H_, D_, (int)sb_q, D_,
                (int)sh_q, (int)sb_q, D_, (int)sh_q, (int)v_bz, (int)v_h,
                p.padded_L, (int)sb_q, D_, (int)sh_q, smScale_, 2, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3 *attachToContext(
        nvinfer1::IPluginResourceContext *) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override {
        fc_.nbFields = 0;
        fc_.fields = nullptr;
        return &fc_;
    }

    char const *getTimingCacheID() noexcept override {
        return "fused_int8_rope_sage_attn/1";
    }

private:
    int B_ = 0, H_ = 0, L_ = 0, D_ = 0, normCode_ = 0;
    float smScale_ = 0;
    nvinfer1::PluginFieldCollection fc_{};
};

class FusedCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3 *createPlugin(nvinfer1::AsciiChar const *,
        nvinfer1::PluginFieldCollection const *,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            return new FusedInt8RopeSageAttn();
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
        fc_.nbFields = 0;
        fc_.fields = nullptr;
        return &fc_;
    }
    nvinfer1::AsciiChar const *getPluginName() const noexcept override {
        return kName;
    }
    nvinfer1::AsciiChar const *getPluginVersion() const noexcept override {
        return kVersion;
    }
    nvinfer1::AsciiChar const *getPluginNamespace() const noexcept override {
        return kNamespace;
    }

private:
    nvinfer1::PluginFieldCollection fc_{};
};

namespace {
FusedCreator *gCreator() {
    static FusedCreator *c = new FusedCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
} // namespace
