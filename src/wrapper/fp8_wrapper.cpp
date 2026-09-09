// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's stochastic_rounding_fp8.
// Registration: comfy_kitchen::stochastic_round_fp8.
// Contract: x[N] BF16, rng[N] INT32 (low 8 bits used; TRT10 lacks UINT8 I/O),
//   -> out[N] FP8(E4M3/kFP8).
// The kernel uses a combined rng/output buffer, so enqueue copies rng->out first.
// (Preserves input immutability; costs a single N-byte memcpy.)
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
void launch_stochastic_round_fp8_kernel(void* rng_and_output, const void* input,
    int64_t numel, int rng_dtype_code, int input_dtype_code, int output_dtype_code,
    cudaStream_t stream);
}

namespace {
constexpr char const* kName = "stochastic_round_fp8";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "comfy_kitchen";

int64_t numelOf(nvinfer1::Dims const& d) {
    int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i)
        n *= d.d[i];
    return n;
}

int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2;
}

class StochasticRoundFP8 : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuildV2,
                           public nvinfer1::IPluginV3OneRuntime {
public:
    explicit StochasticRoundFP8(int32_t aliasRng = 0) : alias_(aliasRng) {
        rebuildFields();
    }
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override {
        if (type == nvinfer1::PluginCapabilityType::kCORE)
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
            return static_cast<nvinfer1::IPluginV3OneBuildV2*>(this);
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    nvinfer1::IPluginV3* clone() noexcept override {
        try {
            return new StochasticRoundFP8(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
        const nvinfer1::DataType*, int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != 2)
            return -1;
        out[0] = nvinfer1::DataType::kFP8;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
        nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out, int32_t nbOut,
        nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != 2 || nbOut != 1)
            return -1;
        out[0] = in[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t, int32_t) noexcept override {
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos == 0)
            return d.type == nvinfer1::DataType::kFLOAT || d.type == nvinfer1::DataType::kHALF ||
                   d.type == nvinfer1::DataType::kBF16;
        if (pos == 1)
            return d.type == nvinfer1::DataType::kINT32; // rng byte in low 8 bits
        return d.type == nvinfer1::DataType::kFP8;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 2)
            return -1;
        return numelOf(in[0].desc.dims) == numelOf(in[1].desc.dims) ? 0 : -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 2)
            return -1;
        N_ = numelOf(in[0].dims);
        icode_ = dtypeCode(in[0].type);
        return (N_ > 0 && N_ == numelOf(in[1].dims)) ? 0 : -1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            // Skip the copy when aliasing holds (TRT passes out==rng).
            // Otherwise stage it ourselves.
            if (out[0] != in[1])
                cudaMemcpyAsync(out[0], in[1], (size_t)N_, cudaMemcpyDeviceToDevice, stream);
            launch_stochastic_round_fp8_kernel(
                out[0], in[0], N_, 3, icode_, 5, stream);
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

    int32_t getAliasedInput(int32_t outputIndex) noexcept override {
        // Effective only with alias_rng=1 and PreviewFeature::kALIASED_PLUGIN_IO_10_03.
        // Otherwise plain I/O plus the self-staged copy (both covered by the pointer check in enqueue).
        return (alias_ && outputIndex == 0) ? 1 : -1;
    }

    char const* getTimingCacheID() noexcept override { return "stochastic_round_fp8/1"; }

private:
    void rebuildFields() {
        fields_[0] = nvinfer1::PluginField{"alias_rng", &alias_, nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 1;
        fc_.fields = fields_;
    }

private:
    int32_t alias_ = 0;
    int64_t N_ = 0;
    int icode_ = 2;
    nvinfer1::PluginField fields_[1];
    nvinfer1::PluginFieldCollection fc_{};
};

class StochasticRoundFP8Creator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            int32_t alias = 0;
            if (fc)
                for (int i = 0; i < fc->nbFields; ++i)
                    if (!std::strcmp(fc->fields[i].name, "alias_rng") &&
                        fc->fields[i].type == nvinfer1::PluginFieldType::kINT32 &&
                        fc->fields[i].length >= 1)
                        alias = static_cast<int32_t const*>(fc->fields[i].data)[0];
            return new StochasticRoundFP8(alias);
        } catch (...) {
            return nullptr;
        }
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return kNamespace; }

private:
    static int32_t defaultAlias_;
    static nvinfer1::PluginField fieldDescs_[1];
    static nvinfer1::PluginFieldCollection fc_;
};

int32_t StochasticRoundFP8Creator::defaultAlias_ = 0;
nvinfer1::PluginField StochasticRoundFP8Creator::fieldDescs_[1] = {
    nvinfer1::PluginField{"alias_rng", &StochasticRoundFP8Creator::defaultAlias_,
        nvinfer1::PluginFieldType::kINT32, 1}};
nvinfer1::PluginFieldCollection StochasticRoundFP8Creator::fc_{
    1, StochasticRoundFP8Creator::fieldDescs_};

StochasticRoundFP8Creator* gCreator() {
    static StochasticRoundFP8Creator* c = new StochasticRoundFP8Creator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
