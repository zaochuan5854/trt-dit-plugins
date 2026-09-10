// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's rms_rope_split_half.
// Registration: comfy_kitchen::rms_rope_split_half. Base precision BF16.
// Contract: q,k[B,H,S,D] (D multiple of 32), freqs[fb,f1,f2,rot/2,2,2],
//   q_scale,k_scale[D] (FP32/FP16/BF16) -> qo, ko (same shape).
// Fields: epsilon(float, default 1e-6), rot_dim(int32, default 0=all).
// Norm spans full head_dim; only the rot_dim prefix is rotated (split-half layout).
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
constexpr char const* kName = "rms_rope_split_half";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "dit-plugins";

int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}

class RmsRopeSplitHalf : public nvinfer1::IPluginV3,
                         public nvinfer1::IPluginV3OneCore,
                         public nvinfer1::IPluginV3OneBuild,
                         public nvinfer1::IPluginV3OneRuntime {
public:
    RmsRopeSplitHalf(float eps = 1e-6f, int32_t rotDim = 0) : eps_(eps), rotDim_(rotDim) {
        rebuildFields();
        std::snprintf(timing_, sizeof(timing_), "rms_rope_split_half/1/eps=%g/rot=%d",
            (double)eps_, (int)rotDim_);
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
            return new RmsRopeSplitHalf(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t getNbOutputs() const noexcept override { return 2; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
        const nvinfer1::DataType* in, int32_t nbIn) const noexcept override {
        if (nbOut != 2 || nbIn != 5)
            return -1;
        for (int i = 0; i < 2; ++i)
            if (in[i] != nvinfer1::DataType::kHALF && in[i] != nvinfer1::DataType::kBF16)
                return -1;
        if (in[0] != in[1])
            return -1;
        out[0] = in[0];
        out[1] = in[1];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 5 || nbOut != 2)
            return -1;
        out[0] = in[0];
        out[1] = in[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos < 2) // q, k: FP16/BF16
            return d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
        if (pos < 5) // freqs, q_scale, k_scale
            return d.type == nvinfer1::DataType::kFLOAT ||
                   d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
        return d.type == io[pos - nbIn].desc.type;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return nbIn == 5 && valid(in[0].desc.dims, in[1].desc.dims, in[2].desc.dims,
                                  in[3].desc.dims, in[4].desc.dims)
                   ? 0
                   : -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 5 || !valid(in[0].dims, in[1].dims, in[2].dims, in[3].dims, in[4].dims))
            return -1;
        B_ = in[0].dims.d[0];
        H_ = in[0].dims.d[1];
        S_ = in[0].dims.d[2];
        D_ = in[0].dims.d[3];
        fb_ = in[2].dims.d[0];
        f1_ = in[2].dims.d[1];
        f2_ = in[2].dims.d[2];
        if (in[1].type != in[0].type)
            return -1;
        icode_ = dtypeCode(in[0].type);
        fcode_ = dtypeCode(in[2].type);
        scode_ = dtypeCode(in[3].type);
        return (scode_ == dtypeCode(in[4].type)) ? 0 : -1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            int64_t rot = rotDim_ > 0 ? rotDim_ : D_;
            int64_t st[4]{(int64_t)H_ * S_ * D_, (int64_t)S_ * D_, D_, 1};
            int64_t f3 = rot / 2;
            int64_t fst[6]{};
            fst[5] = 1;
            fst[4] = 2;
            fst[3] = 4;
            fst[2] = 4 * f3;
            fst[1] = 4 * f3 * f2_;
            fst[0] = 4 * f3 * f2_ * f1_;
            launch_rms_rope_kernel(in[0], in[1], in[2], in[3], in[4], out[0], out[1],
                B_, H_, S_, D_, rot, fb_, f1_, f2_, st[0], st[1], st[2], st[3], st[0],
                st[1], st[2], st[3], st[0], st[1], st[2], st[3], st[0], st[1], st[2],
                st[3], fst[0], fst[1], fst[2], fst[3], fst[4], fst[5], 1, 1, eps_, icode_,
                fcode_, scode_, true, true, stream);
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
        fields_[0] = nvinfer1::PluginField{"epsilon", &eps_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_[1] = nvinfer1::PluginField{"rot_dim", &rotDim_, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 2;
        fc_.fields = fields_;
    }
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
        int64_t rot = rotDim_ > 0 ? rotDim_ : D;
        if (rot % 2 != 0 || rot > D)
            return false;
        if (f.d[3] != rot / 2 || f.d[4] != 2 || f.d[5] != 2)
            return false;
        int64_t qd[3]{q.d[0], q.d[1], q.d[2]}, fd[3]{f.d[0], f.d[1], f.d[2]};
        for (int i = 0; i < 3; ++i)
            if (fd[i] != 1 && fd[i] != qd[i])
                return false;
        if (qs.nbDims != 1 || ks.nbDims != 1 || qs.d[0] != D || ks.d[0] != D)
            return false;
        return q.d[0] > 0 && q.d[1] > 0 && q.d[2] > 0;
    }

    float eps_;
    int32_t rotDim_;
    char timing_[64]{};
    int64_t B_ = 0, H_ = 0, S_ = 0, D_ = 0, fb_ = 1, f1_ = 1, f2_ = 1;
    int icode_ = 2, fcode_ = 0, scode_ = 0;
    nvinfer1::PluginField fields_[2];
    nvinfer1::PluginFieldCollection fc_{};
};

class RmsRopeSplitHalfCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            float eps = 1e-6f;
            int32_t rot = 0;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i) {
                    if (!std::strcmp(fc->fields[i].name, "epsilon") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32 &&
                        fc->fields[i].length >= 1)
                        eps = static_cast<float const*>(fc->fields[i].data)[0];
                    if (!std::strcmp(fc->fields[i].name, "rot_dim") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32 &&
                        fc->fields[i].length >= 1)
                        rot = static_cast<int32_t const*>(fc->fields[i].data)[0];
                }
            return new RmsRopeSplitHalf(eps, rot);
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
    static int32_t defaultRot_;
    static nvinfer1::PluginField fieldDescs_[2];
    static nvinfer1::PluginFieldCollection fc_;
};

float RmsRopeSplitHalfCreator::defaultEps_ = 1e-6f;
int32_t RmsRopeSplitHalfCreator::defaultRot_ = 0;
nvinfer1::PluginField RmsRopeSplitHalfCreator::fieldDescs_[2] = {
    nvinfer1::PluginField{"epsilon", &RmsRopeSplitHalfCreator::defaultEps_,
        nvinfer1::PluginFieldType::kFLOAT32, 1},
    nvinfer1::PluginField{"rot_dim", &RmsRopeSplitHalfCreator::defaultRot_,
        nvinfer1::PluginFieldType::kINT32, 1}};
nvinfer1::PluginFieldCollection RmsRopeSplitHalfCreator::fc_{
    2, RmsRopeSplitHalfCreator::fieldDescs_};

RmsRopeSplitHalfCreator* gCreator() {
    static RmsRopeSplitHalfCreator* c = new RmsRopeSplitHalfCreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
