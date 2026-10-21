// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's rms_adaln / layer_norm_adaln.
// Registration: dit-plugins::{rms_adaln,layer_norm_adaln}. Base precision BF16.
// Contract: x[N,D] or [B,S,D]; scale/shift [R,D] or [B,1,D] (N%R==0);
//   optional weight[D]; optional residual (same shape as x) + gate [B,D] or
//   [B,1,D] -> o (same shape/dtype as x).
// Fields: eps(float, default 1e-6), has_weight(int32), residual_gate(int32).
// Inputs: x, scale, shift[, weight][, residual, gate].
// out = norm(x) * weight * (1+scale) + shift; with residual_gate:
//   out = residual + gate * out. scale/shift/weight/gate share one dtype;
//   x/residual/out share another.
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
void launch_adaln_kernel(const void* x, const void* scale, const void* shift,
    const void* weight, const void* residual, const void* gate, void* out,
    int64_t N, int64_t D, int64_t scale_group, int64_t shift_group, float eps,
    int dtype_code, int mod_dtype_code, int weight_dtype_code, bool subtract_mean,
    bool has_weight, bool residual_gate, cudaStream_t stream);
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
bool isFloat(nvinfer1::DataType t) {
    return t == nvinfer1::DataType::kFLOAT || t == nvinfer1::DataType::kHALF
           || t == nvinfer1::DataType::kBF16;
}

template <bool kSubtractMean> struct Traits;
template <> struct Traits<false> {
    static constexpr char const* kName = "rms_adaln";
};
template <> struct Traits<true> {
    static constexpr char const* kName = "layer_norm_adaln";
};

template <bool kSubtractMean>
class AdaLN : public nvinfer1::IPluginV3,
              public nvinfer1::IPluginV3OneCore,
              public nvinfer1::IPluginV3OneBuild,
              public nvinfer1::IPluginV3OneRuntime {
public:
    explicit AdaLN(float eps = 1e-6f, int32_t hasWeight = 0, int32_t residualGate = 0)
        : eps_(eps), hasWeight_(hasWeight), residualGate_(residualGate) {
        stampTiming();
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
        if (nbOut != 1 || !inputCountOk(nbIn) || !out || !in)
            return -1;
        for (int i = 0; i < nbIn; ++i)
            if (!isFloat(in[i]))
                return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn < 3 || nbOut != 1 || !inputCountOk(nbIn)
            || (in[0].nbDims != 2 && in[0].nbDims != 3))
            return -1;
        out[0] = in[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t nbIn, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos >= nbIn)
            return d.type == io[0].desc.type;
        if (pos == 0)
            return isFloat(d.type);
        if (pos == 1 || pos == 2)
            return isFloat(d.type);
        if (hasWeight_ && pos == 3)
            return d.type == io[1].desc.type;
        if (residualGate_) {
            const int ri = 3 + hasWeight_;
            if (pos == ri)
                return d.type == io[0].desc.type;
            if (pos == ri + 1)
                return d.type == io[1].desc.type;
        }
        return false;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn < 3 || nbIn > 6)
            return -1;
        nvinfer1::Dims d[6];
        for (int i = 0; i < nbIn; ++i)
            d[i] = in[i].desc.dims;
        return checkDims(d, nbIn) ? 0 : -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn < 3 || nbIn > 6 || !inputCountOk(nbIn))
            return -1;
        nvinfer1::Dims d[6];
        for (int i = 0; i < nbIn; ++i)
            d[i] = in[i].dims;
        if (!checkDims(d, nbIn))
            return -1;

        const int rank = in[0].dims.nbDims;
        D_ = in[0].dims.d[rank - 1];
        if (rank == 3) {
            B_ = in[0].dims.d[0];
            S_ = in[0].dims.d[1];
            N_ = B_ * S_;
        } else {
            N_ = in[0].dims.d[0];
            B_ = 1;
            S_ = N_;
        }
        Rs_ = in[1].dims.d[0];
        Rt_ = in[2].dims.d[0];
        if (N_ <= 0 || D_ <= 0 || Rs_ <= 0 || Rt_ <= 0
            || N_ % Rs_ != 0 || N_ % Rt_ != 0)
            return -1;

        int idx = 3;
        if (hasWeight_) {
            if (in[idx].dims.nbDims != 1 || in[idx].dims.d[0] != D_)
                return -1;
            if (in[idx].type != in[1].type)
                return -1;
            idx++;
        }
        if (residualGate_) {
            if (in[idx].dims.nbDims != rank)
                return -1;
            for (int i = 0; i < rank; ++i)
                if (in[idx].dims.d[i] != in[0].dims.d[i])
                    return -1;
            if (in[idx].type != in[0].type)
                return -1;
            idx++;
            const int gr = in[idx].dims.nbDims;
            if (gr != 2 && gr != 3)
                return -1;
            if (in[idx].dims.d[gr - 1] != D_)
                return -1;
            if (gr == 3 && in[idx].dims.d[1] != 1)
                return -1;
            if (in[idx].dims.d[0] != Rs_)
                return -1;
            if (in[idx].type != in[1].type)
                return -1;
        }

        if (in[1].type != in[2].type)
            return -1;
        code_ = dtypeCode(in[0].type);
        modcode_ = dtypeCode(in[1].type);
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            const void* weight = hasWeight_ ? in[3] : nullptr;
            const int ri = 3 + hasWeight_;
            const void* residual = residualGate_ ? in[ri] : nullptr;
            const void* gate = residualGate_ ? in[ri + 1] : nullptr;
            launch_adaln_kernel(in[0], in[1], in[2], weight, residual, gate, out[0],
                N_, D_, N_ / Rs_, N_ / Rt_, eps_, code_, modcode_,
                modcode_, kSubtractMean, hasWeight_ != 0, residualGate_ != 0, stream);
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
        fields_[1] = nvinfer1::PluginField{"has_weight", &hasWeight_,
            nvinfer1::PluginFieldType::kINT32, 1};
        fields_[2] = nvinfer1::PluginField{"residual_gate", &residualGate_,
            nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 3;
        fc_.fields = fields_;
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return timing_; }

private:
    void stampTiming() {
        std::snprintf(timing_, sizeof(timing_), "%s/1/eps=%g/hw=%d/rg=%d",
            Traits<kSubtractMean>::kName, (double)eps_, (int)hasWeight_, (int)residualGate_);
    }
    bool inputCountOk(int32_t nbIn) const {
        return nbIn == 3 + hasWeight_ + 2 * residualGate_;
    }
    bool checkDims(nvinfer1::Dims const* d, int32_t nbIn) const {
        if (!inputCountOk(nbIn))
            return false;
        const int rank = d[0].nbDims;
        if (rank != 2 && rank != 3)
            return false;
        const int64_t D = d[0].d[rank - 1];
        if (D < 1)
            return false;
        for (int i = 0; i < rank; ++i)
            if (d[0].d[i] < -1 || d[0].d[i] == 0)
                return false;

        for (int j = 1; j <= 2; ++j) {
            const int r = d[j].nbDims;
            if (r != 2 && r != 3)
                return false;
            if (d[j].d[r - 1] != D)
                return false;
            if (r == 3 && d[j].d[1] != 1)
                return false;
            if (d[j].d[0] < -1 || d[j].d[0] == 0)
                return false;
        }

        int idx = 3;
        if (hasWeight_) {
            if (d[idx].nbDims != 1 || d[idx].d[0] != D)
                return false;
            idx++;
        }
        if (residualGate_) {
            if (d[idx].nbDims != rank)
                return false;
            for (int i = 0; i < rank; ++i)
                if (d[idx].d[i] != d[0].d[i] && d[idx].d[i] != -1 && d[0].d[i] != -1)
                    return false;
            idx++;
            const int gr = d[idx].nbDims;
            if (gr != 2 && gr != 3)
                return false;
            if (d[idx].d[gr - 1] != D)
                return false;
            if (gr == 3 && d[idx].d[1] != 1)
                return false;
            if (d[idx].d[0] < -1 || d[idx].d[0] == 0)
                return false;
            const int64_t gateRows = d[idx].d[0];
            const int64_t scRows = d[1].d[0];
            if (gateRows != -1 && scRows != -1 && gateRows != scRows)
                return false;
        }
        return true;
    }

    float eps_;
    int32_t hasWeight_;
    int32_t residualGate_;
    int64_t N_ = 0, D_ = 0, B_ = 0, S_ = 0, Rs_ = 1, Rt_ = 1;
    int code_ = 2, modcode_ = 2;
    char timing_[80]{};
    nvinfer1::PluginField fields_[3];
    nvinfer1::PluginFieldCollection fc_{};
};

template <bool kSubtractMean> class AdaLNCreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            float eps = 1e-6f;
            int32_t hasWeight = 0, residualGate = 0;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i) {
                    if (!std::strcmp(fc->fields[i].name, "eps") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kFLOAT32 &&
                        fc->fields[i].length >= 1)
                        eps = static_cast<float const*>(fc->fields[i].data)[0];
                    if (!std::strcmp(fc->fields[i].name, "has_weight") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32 &&
                        fc->fields[i].length >= 1)
                        hasWeight = static_cast<int32_t const*>(fc->fields[i].data)[0];
                    if (!std::strcmp(fc->fields[i].name, "residual_gate") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32 &&
                        fc->fields[i].length >= 1)
                        residualGate = static_cast<int32_t const*>(fc->fields[i].data)[0];
                }
            return new AdaLN<kSubtractMean>(eps, hasWeight, residualGate);
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
    static int32_t defaultHasWeight_;
    static int32_t defaultResidualGate_;
    static nvinfer1::PluginField fieldDescs_[3];
    static nvinfer1::PluginFieldCollection fc_;
};

template <bool B> float AdaLNCreator<B>::defaultEps_ = 1e-6f;
template <bool B> int32_t AdaLNCreator<B>::defaultHasWeight_ = 0;
template <bool B> int32_t AdaLNCreator<B>::defaultResidualGate_ = 0;
template <bool B>
nvinfer1::PluginField AdaLNCreator<B>::fieldDescs_[3] = {
    nvinfer1::PluginField{"eps", &AdaLNCreator<B>::defaultEps_,
        nvinfer1::PluginFieldType::kFLOAT32, 1},
    nvinfer1::PluginField{"has_weight", &AdaLNCreator<B>::defaultHasWeight_,
        nvinfer1::PluginFieldType::kINT32, 1},
    nvinfer1::PluginField{"residual_gate", &AdaLNCreator<B>::defaultResidualGate_,
        nvinfer1::PluginFieldType::kINT32, 1}};
template <bool B> nvinfer1::PluginFieldCollection AdaLNCreator<B>::fc_{
    3, AdaLNCreator<B>::fieldDescs_};

template <bool B> AdaLNCreator<B>* gCreator() {
    static AdaLNCreator<B>* c = new AdaLNCreator<B>();
    return c;
}
struct AutoRegister {
    AutoRegister() {
        getPluginRegistry()->registerCreator(*gCreator<false>(), kNamespace);
        getPluginRegistry()->registerCreator(*gCreator<true>(), kNamespace);
    }
};
static AutoRegister gAutoRegister;
} // namespace
