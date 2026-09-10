// SPDX-License-Identifier: Apache-2.0
// TRT IPluginV3 for SpargeAttn block-sparse SageAttention2 (sm89).
// Kernel origin: SpargeAttn (thu-ml, Apache-2.0), NOT ComfyKitchen; see NOTICE.
// Registered under the existing comfy_kitchen namespace for release unity.
// Registration: comfy_kitchen::block_sparse_sage2_attn.
// Inputs q,k,v: [B,H,S,D] contiguous LINEAR FP16/BF16 (uniform dtype);
// mask: INT32 [B,H,QB,KB] 0/1 (static shape, dynamic values; all-ones = dense).
// Output o: same shape/dtype as q. sm89 only. D in {64,128}, S % 128 == 0.
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

extern "C" {
void launch_sparge_preprocess(const void* q, const void* k, const void* mask, const void* v,
    int8_t* q_i8, float* q_s, int8_t* k_i8, float* k_s, float* km, int32_t* lut, int32_t* vbn,
    void* vT, void* v_fp8, float* v_s, void* v_fp16_tmp, int B, int Hq, int Hkv, int S, int D,
    int is_bf16, void* stream);
void launch_block_sparse_sage2_sm89(int8_t* q_i8, int8_t* k_i8, void* v_fp8, void* o,
    int32_t* lut, int32_t* vbn, float* pv_thr, float* q_s, float* k_s, float* v_s, int B, int Hq,
    int Hkv, int S, int D, int is_bf16, float sm_scale, void* stream);
}

namespace {
constexpr char const* kName = "block_sparse_sage2_attn";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "dit-plugins";

int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

struct WsPlan {
    int64_t off_qi, off_qs, off_ki, off_ks, off_km, off_lut, off_vbn, off_vT, off_vfp8, off_vs,
        off_vtmp, off_pv, total;
    int maxHq;
};

WsPlan planWs(int B, int Hq, int Hkv, int S, int D) {
    WsPlan p{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int NQB = (S + 127) / 128, NKB = (S + 63) / 64, P = (S + 127) / 128 * 128;
    p.maxHq = Hq;
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
    p.total = off;
    return p;
}

class BlockSparseSage2Attn : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
public:
    BlockSparseSage2Attn() = default;
    BlockSparseSage2Attn(float scale, float pvthreshd, int32_t sink)
        : scale_(scale), pvthreshd_(pvthreshd), sink_(sink) {}

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
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
            return new BlockSparseSage2Attn(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut, const nvinfer1::DataType* in,
        int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != 4)
            return -1;
        if (in[0] != nvinfer1::DataType::kHALF && in[0] != nvinfer1::DataType::kBF16)
            return -1;
        for (int i = 1; i < 3; ++i)
            if (in[i] != in[0])
                return -1;
        if (in[3] != nvinfer1::DataType::kINT32)
            return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 4 || nbOut != 1)
            return -1;
        out[0] = in[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        if (nbIn != 4)
            return false;
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos < 3)
            return d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
        if (pos == 3)
            return d.type == nvinfer1::DataType::kINT32;
        return d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 4)
            return -1;
        for (int i = 0; i < 3; ++i)
            if (in[i].desc.dims.nbDims != 4)
                return -1;
        if (in[3].desc.dims.nbDims != 4)
            return -1;
        int d = in[0].max.d[3];
        return (d == 64 || d == 128) ? 0 : -1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        if (nbIn != 4)
            return 0;
        int B = in[0].max.d[0], Hq = in[0].max.d[1], S = in[0].max.d[2], D = in[0].max.d[3];
        int Hkv = in[1].max.d[1];
        if ((D != 64 && D != 128) || S < 128 || S % 128 != 0 || Hq % Hkv != 0)
            return 0;
        return (size_t)planWs(B, Hq, Hkv, S, D).total;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 4 || in[0].dims.nbDims != 4 || in[3].dims.nbDims != 4)
            return -1;
        B_ = in[0].dims.d[0];
        Hq_ = in[0].dims.d[1];
        S_ = in[0].dims.d[2];
        D_ = in[0].dims.d[3];
        Hkv_ = in[1].dims.d[1];
        if ((D_ != 64 && D_ != 128) || S_ < 128 || S_ % 128 != 0 || Hq_ % Hkv_ != 0)
            return -1;
        if (in[1].type != in[0].type || in[2].type != in[0].type)
            return -1;
        if (in[3].dims.d[0] != B_ || in[3].dims.d[1] != Hq_ || in[3].dims.d[2] != S_ / 128 ||
            in[3].dims.d[3] != S_ / 64)
            return -1;
        isBf16_ = in[0].type == nvinfer1::DataType::kBF16;
        if (Hq_ > kMaxHeads)
            return -1;
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void* ws, cudaStream_t stream) noexcept override {
        try {
            WsPlan p = planWs(B_, Hq_, Hkv_, S_, D_);
            char* w = static_cast<char*>(ws);
            float sm = scale_ > 0 ? scale_ : 1.0f / std::sqrt((float)D_);
            float pvhost[kMaxHeads];
            for (int i = 0; i < Hq_; ++i) pvhost[i] = pvthreshd_;
            if (cudaMemcpyAsync(w + p.off_pv, pvhost, (size_t)Hq_ * 4, cudaMemcpyHostToDevice,
                    stream) != cudaSuccess)
                return -1;
            launch_sparge_preprocess(in[0], in[1], in[3], in[2],
                reinterpret_cast<int8_t*>(w + p.off_qi), reinterpret_cast<float*>(w + p.off_qs),
                reinterpret_cast<int8_t*>(w + p.off_ki), reinterpret_cast<float*>(w + p.off_ks),
                reinterpret_cast<float*>(w + p.off_km), reinterpret_cast<int32_t*>(w + p.off_lut),
                reinterpret_cast<int32_t*>(w + p.off_vbn), w + p.off_vT, w + p.off_vfp8,
                reinterpret_cast<float*>(w + p.off_vs), w + p.off_vtmp, B_, Hq_, Hkv_, S_, D_,
                isBf16_ ? 1 : 0, stream);
            launch_block_sparse_sage2_sm89(reinterpret_cast<int8_t*>(w + p.off_qi),
                reinterpret_cast<int8_t*>(w + p.off_ki), w + p.off_vfp8, out[0],
                reinterpret_cast<int32_t*>(w + p.off_lut), reinterpret_cast<int32_t*>(w + p.off_vbn),
                reinterpret_cast<float*>(w + p.off_pv), reinterpret_cast<float*>(w + p.off_qs),
                reinterpret_cast<float*>(w + p.off_ks), reinterpret_cast<float*>(w + p.off_vs),
                B_, Hq_, Hkv_, S_, D_, isBf16_ ? 1 : 0, sm, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_[0] = {"scale", &scale_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_[1] = {"pvthreshd", &pvthreshd_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_[2] = {"attention_sink", &sink_, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 3;
        fc_.fields = fields_;
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return "block_sparse_sage2_attn/1"; }

private:
    static constexpr int kMaxHeads = 256;
    float scale_ = 0, pvthreshd_ = 50;
    int32_t sink_ = 0;
    int B_ = 0, Hq_ = 0, Hkv_ = 0, S_ = 0, D_ = 0;
    bool isBf16_ = false;
    nvinfer1::PluginField fields_[3]{};
    nvinfer1::PluginFieldCollection fc_{};
};

class BlockSparseSage2AttnCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase) noexcept override {
        try {
            float scale = 0, pvthreshd = 50;
            int32_t sink = 0;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i) {
                    if (std::strcmp(fc->fields[i].name, "scale") == 0 && fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32)
                        scale = *static_cast<float const*>(fc->fields[i].data);
                    if (std::strcmp(fc->fields[i].name, "pvthreshd") == 0 && fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32)
                        pvthreshd = *static_cast<float const*>(fc->fields[i].data);
                    if (std::strcmp(fc->fields[i].name, "attention_sink") == 0 && fc->fields[i].type == nvinfer1::PluginFieldType::kINT32)
                        sink = *static_cast<int32_t const*>(fc->fields[i].data);
                }
            return new BlockSparseSage2Attn(scale, pvthreshd, sink);
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        names_[0] = {"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        names_[1] = {"pvthreshd", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        names_[2] = {"attention_sink", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 3;
        fc_.fields = names_;
        return &fc_;
    }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    nvinfer1::PluginField names_[3]{};
    nvinfer1::PluginFieldCollection fc_{};
};

namespace {
BlockSparseSage2AttnCreator* gCreator() {
    static BlockSparseSage2AttnCreator* c = new BlockSparseSage2AttnCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
} // namespace
