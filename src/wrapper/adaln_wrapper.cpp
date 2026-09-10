// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's adaln / rms_adaln (torch.ops.comfy_kitchen.*).
// Registration: comfy_kitchen::adaln, comfy_kitchen::rms_adaln. Base precision BF16.
// Contract: x[N,D], scale[Rs,D], shift[Rt,D] (N%Rs==0, N%Rt==0; materialize
// non-groupable broadcasts upstream).
// Fields: eps(float, default 1e-6). No workspace, single launch.
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
void launch_adaln_kernel(const void* x, const void* scale, const void* shift,
    void* out, int64_t N, int64_t D, int64_t scale_group, int64_t shift_group,
    float eps, int dtype_code, bool subtract_mean, cudaStream_t stream);
}

namespace {
constexpr char const* kNamespace = "dit-plugins";
constexpr int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}

template <bool kSubtractMean> struct Traits;
template <> struct Traits<true> {
    static constexpr char const* kName = "adaln";
};
template <> struct Traits<false> {
    static constexpr char const* kName = "rms_adaln";
};

template <bool kSubtractMean>
class AdaLN : public nvinfer1::IPluginV3,
              public nvinfer1::IPluginV3OneCore,
              public nvinfer1::IPluginV3OneBuild,
              public nvinfer1::IPluginV3OneRuntime {
public:
    explicit AdaLN(float eps = 1e-6f) : eps_(eps) { stampTiming(); }

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
            return new AdaLN(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Traits<kSubtractMean>::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
        const nvinfer1::DataType* in, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != 3)
            return -1;
        for (int i = 0; i < 3; ++i)
            if (in[i] != nvinfer1::DataType::kFLOAT && in[i] != nvinfer1::DataType::kHALF &&
                in[i] != nvinfer1::DataType::kBF16)
                return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 3 || nbOut != 1)
            return -1;
        out[0] = in[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos < nbIn)
            return d.type == nvinfer1::DataType::kFLOAT || d.type == nvinfer1::DataType::kHALF ||
                   d.type == nvinfer1::DataType::kBF16;
        return d.type == io[0].desc.type;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return checkDims(in, nbIn) ? 0 : -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 3 || !checkDims(in, nbIn))
            return -1;
        N_ = in[0].dims.d[0];
        D_ = in[0].dims.d[1];
        Rs_ = in[1].dims.d[0];
        Rt_ = in[2].dims.d[0];
        if (in[1].type != in[0].type || in[2].type != in[0].type)
            return -1;
        code_ = dtypeCode(in[0].type);
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            launch_adaln_kernel(in[0], in[1], in[2], out[0], N_, D_, N_ / Rs_, N_ / Rt_,
                eps_, code_, kSubtractMean, stream);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_[0] = nvinfer1::PluginField{"eps", &eps_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fc_.nbFields = 1;
        fc_.fields = fields_;
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return timing_; }

private:
    void stampTiming() {
        std::snprintf(timing_, sizeof(timing_), "%s/1/eps=%g",
            Traits<kSubtractMean>::kName, (double)eps_);
    }
    static bool checkDims(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn) {
        if (nbIn != 3)
            return false;
        for (int i = 0; i < 3; ++i)
            if (in[i].desc.dims.nbDims != 2)
                return false;
        return check2D(in[0].desc.dims, in[1].desc.dims, in[2].desc.dims);
    }
    static bool checkDims(nvinfer1::PluginTensorDesc const* in, int32_t nbIn) {
        if (nbIn != 3)
            return false;
        for (int i = 0; i < 3; ++i)
            if (in[i].dims.nbDims != 2)
                return false;
        return check2D(in[0].dims, in[1].dims, in[2].dims);
    }
    template <typename D> static bool check2D(D const& x, D const& sc, D const& sh) {
        int64_t N = x.d[0], Dm = x.d[1];
        if (Dm != sc.d[1] || Dm != sh.d[1])
            return false;
        int64_t Rs = sc.d[0], Rt = sh.d[0];
        return N > 0 && Dm > 0 && Rs > 0 && Rt > 0 && N % Rs == 0 && N % Rt == 0;
    }

    float eps_;
    int64_t N_ = 0, D_ = 0, Rs_ = 1, Rt_ = 1;
    int code_ = 2;
    char timing_[64]{};
    nvinfer1::PluginField fields_[1];
    nvinfer1::PluginFieldCollection fc_{};
};

template <bool kSubtractMean> class AdaLNCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            float eps = 1e-6f;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i)
                    if (!std::strcmp(fc->fields[i].name, "eps") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32 &&
                        fc->fields[i].length >= 1)
                        eps = static_cast<float const*>(fc->fields[i].data)[0];
            return new AdaLN<kSubtractMean>(eps);
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Traits<kSubtractMean>::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    // Field descriptors (TRT keeps references only, so static lifetime required)
    static float defaultEps_;
    static nvinfer1::PluginField fieldDescs_[1];
    static nvinfer1::PluginFieldCollection fc_;
};

template <bool B> float AdaLNCreator<B>::defaultEps_ = 1e-6f;
template <bool B>
nvinfer1::PluginField AdaLNCreator<B>::fieldDescs_[1] = {
    nvinfer1::PluginField{"eps", &AdaLNCreator<B>::defaultEps_, nvinfer1::PluginFieldType::kFLOAT32, 1}};
template <bool B> nvinfer1::PluginFieldCollection AdaLNCreator<B>::fc_{1, AdaLNCreator<B>::fieldDescs_};

// Avoid the macro's default "" namespace; register manually under comfy_kitchen
template <bool B> AdaLNCreator<B>* gCreator() {
    static AdaLNCreator<B>* c = new AdaLNCreator<B>();
    return c;
}
struct AutoRegister {
    AutoRegister() {
        getPluginRegistry()->registerCreator(*gCreator<true>(), kNamespace);
        getPluginRegistry()->registerCreator(*gCreator<false>(), kNamespace);
    }
};
static AutoRegister gAutoRegister;
} // namespace
