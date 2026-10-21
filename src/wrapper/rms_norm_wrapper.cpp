// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for a single-tensor RMSNorm.
// Registration: dit-plugins::rms_norm. Base precision BF16/FP16.
// Contract: x[B,H,S,D] or x[B,S,H,D], weight[D] (FP32/FP16/BF16) ->
//   y[B,H,S,D]. BSHD input is the cross-attention layout produced by the
//   exporter; it is transposed by the kernel stores, not a Transpose node.
// Fields: eps(float, default 1e-6), in_layout(int32, default 0=BHSD,
// 1=BSHD). B/S/H may be dynamic. The norm kernel covers every dimension
// (rot_dim=0, no k, no freqs).
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
void launch_rms_rope_kernel(const void* q, const void* k, const void* freqs,
    const void* q_scale, const void* k_scale, void* q_out, void* k_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2, int64_t q_s0,
    int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0, int64_t k_s1,
    int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
    int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
    int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon,
    int input_dtype_code, int freqs_dtype_code, int scale_dtype_code, bool has_k,
    bool split_half, cudaStream_t stream);
}

namespace {
constexpr char const* kName = "rms_norm";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "dit-plugins";

int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}

bool isHalfOrBf16(nvinfer1::DataType t) {
    return t == nvinfer1::DataType::kHALF || t == nvinfer1::DataType::kBF16;
}

bool isFloat(nvinfer1::DataType t) {
    return t == nvinfer1::DataType::kFLOAT || isHalfOrBf16(t);
}

class RmsNorm final : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
public:
    RmsNorm(float eps = 1e-6f, int32_t inLayout = 0) : eps_(eps), inLayout_(inLayout) {
        rebuildFields();
        std::snprintf(timing_, sizeof(timing_), "rms_norm/1/eps=%g/layout=%d",
            (double)eps_, (int)inLayout_);
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
        try {
            return new RmsNorm(*this);
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
        if (nbOut != 1 || nbIn != 2 || !out || !in)
            return -1;
        if (!isHalfOrBf16(in[0]) || !isFloat(in[1]))
            return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
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

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos == 0)
            return isHalfOrBf16(d.type);
        if (pos == 1)
            return isFloat(d.type);
        return d.type == io[0].desc.type;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return nbIn == 2 && validDims(in[0].desc.dims, in[1].desc.dims) ? 0 : -1;
    }

    static bool validDims(nvinfer1::Dims const& x, nvinfer1::Dims const& w) {
        if (x.nbDims != 4 || w.nbDims != 1)
            return false;
        int64_t D = x.d[3];
        if (D < 32 || D % 32 != 0 || w.d[0] != D)
            return false;
        for (int i = 0; i < 3; ++i)
            if (x.d[i] < -1 || x.d[i] == 0)
                return false;
        return true;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 2 || !validDims(in[0].dims, in[1].dims))
            return -1;
        if (!isHalfOrBf16(in[0].type) || !isFloat(in[1].type))
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
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            const int64_t q_s0 = inLayout_ == 0
                ? (int64_t)H_ * S_ * D_ : (int64_t)S_ * H_ * D_;
            const int64_t q_s1 = inLayout_ == 0 ? (int64_t)S_ * D_ : D_;
            const int64_t q_s2 = inLayout_ == 0 ? D_ : (int64_t)H_ * D_;
            const int64_t qo_s0 = (int64_t)H_ * S_ * D_;
            const int64_t qo_s1 = (int64_t)S_ * D_;
            // No k, no rotation: the norm-only tail covers every dimension.
            launch_rms_rope_kernel(in[0], nullptr, nullptr, in[1], nullptr,
                out[0], nullptr, B_, H_, S_, D_, 0, 1, 1, 1,
                q_s0, q_s1, q_s2, 1, 0, 0, 0, 0,
                qo_s0, qo_s1, D_, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 1, eps_, icode_, 0, scode_,
                false, true, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        rebuildFields();
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return timing_; }

private:
    void rebuildFields() {
        fields_[0] = nvinfer1::PluginField{"eps", &eps_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_[1] = nvinfer1::PluginField{"in_layout", &inLayout_,
            nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 2;
        fc_.fields = fields_;
    }

    float eps_;
    int32_t inLayout_;
    char timing_[48]{};
    int64_t B_ = 0, H_ = 0, S_ = 0, D_ = 0;
    int icode_ = 2, scode_ = 0;
    nvinfer1::PluginField fields_[2];
    nvinfer1::PluginFieldCollection fc_{};
};

class RmsNormCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            float eps = 1e-6f;
            int32_t layout = 0;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i) {
                    if (!std::strcmp(fc->fields[i].name, "eps") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32 &&
                        fc->fields[i].length >= 1)
                        eps = static_cast<float const*>(fc->fields[i].data)[0];
                    if (!std::strcmp(fc->fields[i].name, "in_layout") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32 &&
                        fc->fields[i].length >= 1)
                        layout = static_cast<int32_t const*>(fc->fields[i].data)[0];
                }
            return new RmsNorm(eps, layout);
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    static float defaultEps_;
    static int32_t defaultLayout_;
    static nvinfer1::PluginField fieldDescs_[2];
    static nvinfer1::PluginFieldCollection fc_;
};

float RmsNormCreator::defaultEps_ = 1e-6f;
int32_t RmsNormCreator::defaultLayout_ = 0;
nvinfer1::PluginField RmsNormCreator::fieldDescs_[2] = {
    nvinfer1::PluginField{"eps", &RmsNormCreator::defaultEps_,
        nvinfer1::PluginFieldType::kFLOAT32, 1},
    nvinfer1::PluginField{"in_layout", &RmsNormCreator::defaultLayout_,
        nvinfer1::PluginFieldType::kINT32, 1}};
nvinfer1::PluginFieldCollection RmsNormCreator::fc_{
    2, RmsNormCreator::fieldDescs_};

RmsNormCreator* gCreator() {
    static RmsNormCreator* c = new RmsNormCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
