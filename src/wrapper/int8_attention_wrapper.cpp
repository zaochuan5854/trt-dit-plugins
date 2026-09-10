// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's int8_attention (comfy_kitchen::int8_attention).
// Registration: name="int8_attention", namespace="dit-plugins".
// Base precision BF16. Inputs Q,K,V: [B,H,S,D] contiguous LINEAR BF16. Output: same shape.
// Kernels: third_party/sage_attention (quant_qk/quant_v/sage_attn_launcher).
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>

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
}

namespace {
constexpr char const* kName = "int8_attention";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "dit-plugins";
int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}
// comfy-kitchen convention: FP32 input -> BF16 output, else same dtype as input
nvinfer1::DataType outTypeFor(nvinfer1::DataType in) {
    return in == nvinfer1::DataType::kFLOAT ? nvinfer1::DataType::kBF16 : in;
}

int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

struct WsPlan {
    int cta_k, padded_Lk;
    int64_t off_qi, off_ki, off_vi, off_qs, off_ks, off_vs, off_anchor, total;
};

WsPlan planWs(int B, int Hq, int Lq, int Hkv, int Lk, int D) {
    WsPlan p{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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

class Int8Attention : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
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
            return new Int8Attention(*this);
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
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 3 || nbOut != 1)
            return -1;
        out[0] = in[0]; // [B,H,S,D] passthrough
        return 0;
    }

    // Inputs: FP32/FP16/BF16, outputs: FP16/BF16 (kLINEAR only)
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
        return (size_t)planWs(B, Hq, Lq, Hkv, Lk, D).total;
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
        if (in[1].type != in[0].type || in[2].type != in[0].type)
            return -1;
        inCode_ = dtypeCode(in[0].type);
        outCode_ = dtypeCode(outTypeFor(in[0].type));
        smScale_ = 1.0f / std::sqrt((float)D_);
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void* ws, cudaStream_t stream) noexcept override {
        try {
            WsPlan p = planWs(B_, Hq_, Lq_, Hkv_, Lk_, D_);
            char* w = static_cast<char*>(ws);
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
                (int)(sb_q), D_, (int)(sh_q), smScale_, outCode_, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fc_.nbFields = 0;
        fc_.fields = nullptr;
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return "int8_attention/1"; }

private:
    int B_ = 0, Hq_ = 0, Lq_ = 0, Hkv_ = 0, Lk_ = 0, D_ = 0, inCode_ = 2, outCode_ = 2;
    float smScale_ = 0;
    nvinfer1::PluginFieldCollection fc_{};
};

class Int8AttentionCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const*, nvinfer1::TensorRTPhase) noexcept override {
        try {
            return new Int8Attention();
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        fc_.nbFields = 0;
        fc_.fields = nullptr;
        return &fc_;
    }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    nvinfer1::PluginFieldCollection fc_{};
};

// REGISTER_TENSORRT_PLUGIN registers under namespace "", so register manually
// under dit-plugins (matches engine serialize/deserialize name resolution)
namespace {
Int8AttentionCreator* gCreator() {
    static Int8AttentionCreator* c = new Int8AttentionCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
} // namespace
