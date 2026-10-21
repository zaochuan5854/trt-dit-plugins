// SPDX-License-Identifier: Apache-2.0
// ConvRot W8A8 / W4A4 linear plugin (kernels vendored from comfy-kitchen).
//
// op names: "convrot_linear" (weight INT8 [N,K]) and "convrot_linear_w4a4"
// (weight INT8 [N,K/2], packed signed INT4). Both carry the activation
// rotation + rowwise quantization internally, so the graph passes the
// pre-rotation activation and the already-rotated quantized weight.
//
// Inputs: x [...,K] f32/f16/bf16, weight INT8 [N,K] (or [N,K/2]), weight_scale
// FP32 [N], bias [N] in the OUTPUT dtype. Output: y [...,N].
//
// Fields: group_size=256, has_bias 0/1, out_dtype -1=input else 0=FP32,
// 1=FP16, 2=BF16 (comfy-kitchen's codes), gemm_config -1=auto or a tile index.
//
// Fallback policy is build-time: unsupported shapes return -1 from
// configurePlugin/onShapeChange so the exporter keeps the float path.
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "rt.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "convrot_shim.h"

namespace {

constexpr int kX = 0, kW = 1, kWS = 2, kB = 3, kNbIn = 4;
constexpr size_t align256(size_t v) { return (v + 255u) & ~size_t(255u); }

int64_t numel(nvinfer1::Dims const& d) {
    int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= std::max<int64_t>(1, d.d[i]);
    return n;
}

int64_t leadingNumel(nvinfer1::Dims const& d) {
    int64_t n = 1;
    for (int i = 0; i + 1 < d.nbDims; ++i) n *= std::max<int64_t>(1, d.d[i]);
    return n;
}

bool isFloat(nvinfer1::DataType t) {
    return t == nvinfer1::DataType::kBF16 || t == nvinfer1::DataType::kHALF ||
           t == nvinfer1::DataType::kFLOAT;
}

// Process-static: build/shape callbacks must not query the driver per call
// (plan improvement request R1). Mixed-SM hosts pick the build device.
int currentArchMajor() {
    static const int major = [] {
        int device = 0;
        return cudaGetDevice(&device) == cudaSuccess ? convrot_arch_major(device) : 0;
    }();
    return major;
}

nvinfer1::DataType outType(nvinfer1::DataType x, int32_t code) {
    switch (code) {
        case 1: return nvinfer1::DataType::kHALF;
        case 2: return nvinfer1::DataType::kFLOAT;
        case 0: return nvinfer1::DataType::kBF16;
        default: return x;  // -1 = same as input
    }
}

// comfy dtype codes: 0 = FP32, 1 = FP16, 2 = BF16.
int dtypeCodeOf(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT) return 0;
    if (t == nvinfer1::DataType::kHALF) return 1;
    return 2;
}

template <int WeightBits>
class ConvRotLinear final : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
    static_assert(WeightBits == 8 || WeightBits == 4, "WeightBits must be 8 or 4");

public:
    static constexpr char const* name() {
        return WeightBits == 8 ? "convrot_linear" : "convrot_linear_w4a4";
    }

    ConvRotLinear(int32_t groupSize, int32_t hasBias, int32_t outDtype, int32_t gemmConfig)
        : groupSize_(groupSize), hasBias_(hasBias), outDtype_(outDtype), gemmConfig_(gemmConfig) {
        refreshFields();
    }

    ConvRotLinear(ConvRotLinear const& o)
        : groupSize_(o.groupSize_), hasBias_(o.hasBias_), outDtype_(o.outDtype_),
          gemmConfig_(o.gemmConfig_), M_(o.M_), K_(o.K_), N_(o.N_) {
        refreshFields();
        updateTimingID();
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
            return new ConvRotLinear(*this);
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return name(); }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return "dit-plugins";
    }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
                               nvinfer1::DataType const* in,
                               int32_t nbIn) const noexcept override {
        if (nbOut != 1 || nbIn != kNbIn || !out || !in) return -1;
        out[0] = outType(in[kX], outDtype_);
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
                            nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out,
                            int32_t nbOut, nvinfer1::IExprBuilder&) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        if (in[kX].nbDims < 2 || in[kW].nbDims != 2) return -1;
        out[0] = in[kX];
        out[0].d[out[0].nbDims - 1] = in[kW].d[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
                                   int32_t nbIn, int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !io || pos < 0 || pos > kNbIn) return false;
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR) return false;
        switch (pos) {
            case kX: return isFloat(d.type);
            case kW: return d.type == nvinfer1::DataType::kINT8;
            case kWS: return d.type == nvinfer1::DataType::kFLOAT;
            case kB: return d.type == outType(io[kX].desc.type, outDtype_);
            case kNbIn: return d.type == outType(io[kX].desc.type, outDtype_);
            default: return false;
        }
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
                            nvinfer1::DynamicPluginTensorDesc const* out,
                            int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        // Turing (sm_75) has no supported route in this build (plan 4.3).
        if (currentArchMajor() < 8) {
            DIT_REJECT(name(), "configure", "sm major %d < 8", currentArchMajor());
            return -1;
        }
        if (!checkDesc(in[kX].desc, in[kW].desc, in[kWS].desc, in[kB].desc, out[0].desc)) {
            DIT_REJECT(name(), "configure", "unsupported dtype/shape (Wbits=%d out_dtype=%d)",
                       WeightBits, outDtype_);
            return -1;
        }
        K_ = in[kW].desc.dims.d[1] * (WeightBits == 4 ? 2 : 1);
        N_ = in[kW].desc.dims.d[0];
        try {
            updateTimingID();
        } catch (...) {
            return -1;
        }
        return 0;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
                            nvinfer1::DynamicPluginTensorDesc const*,
                            int32_t nbOut) const noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || in[kW].max.nbDims != 2) return 0;
        int64_t N = in[kW].max.d[0];
        int64_t K = in[kW].max.d[1] * (WeightBits == 4 ? 2 : 1);
        int64_t M = leadingNumel(in[kX].max);
        if (N <= 0 || K <= 0 || M <= 0) return 0;
        return convrot_workspace_bytes(WeightBits, M, N, K, dtypeCodeOf(in[kX].desc.type));
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
                          nvinfer1::PluginTensorDesc const* out,
                          int32_t nbOut) noexcept override {
        if (nbIn != kNbIn || nbOut != 1 || !in || !out) return -1;
        if (!checkDesc(in[kX], in[kW], in[kWS], in[kB], out[0])) {
            DIT_REJECT(name(), "onShapeChange", "unsupported dtype/shape");
            return -1;
        }
        M_ = leadingNumel(in[kX].dims);
        K_ = in[kW].dims.d[1] * (WeightBits == 4 ? 2 : 1);
        N_ = in[kW].dims.d[0];
        if (M_ <= 0) {
            DIT_REJECT(name(), "onShapeChange", "M=%lld", (long long)M_);
            return -1;
        }
        try {
            updateTimingID();
        } catch (...) {
            return -1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inDesc, nvinfer1::PluginTensorDesc const*,
                    void const* const* in, void* const* out, void* workspace,
                    cudaStream_t stream) noexcept override {
        try {
            if (!inDesc || !in || !out || M_ <= 0 || K_ <= 0 || N_ <= 0) return -1;

            // Field codes are plan 4.2 (0=BF16, 1=FP16, 2=FP32); the shim
            // takes comfy's codes (0=FP32, 1=FP16, 2=BF16).
            const int inCode = dtypeCodeOf(inDesc[kX].type);
            const int outCode = dtypeCodeOf(outType(inDesc[kX].type, outDtype_));
            const int biasCode = outCode;
            void const* bias = hasBias_ ? in[kB] : nullptr;

            // sm8x uses native INT4 MMA; other archs run the W4A8 fallback.
            const bool int4Native = WeightBits == 4 && currentArchMajor() == 8;

            // M == 1 fused GEMV (W8A8 only). Returns 1 when inapplicable.
            if (WeightBits == 8 && M_ == 1) {
                const int rc = convrot_gemv_m1(
                    out[0], in[kX], static_cast<int8_t const*>(in[kW]),
                    static_cast<float const*>(in[kWS]), bias, N_, K_, inCode, outCode, biasCode,
                    stream);
                if (rc == 0) return 0;
                if (rc < 0) return -1;
            }

            const int actBits = int4Native ? 4 : 8;
            const size_t quantBytes =
                align256(static_cast<size_t>(M_) *
                         static_cast<size_t>(int4Native ? K_ / 2 : K_));
            const size_t scaleBytes = align256(static_cast<size_t>(M_) * sizeof(float));
            const size_t scratchBytes = convrot_workspace_bytes(WeightBits, M_, N_, K_, inCode);
            if (!workspace || scratchBytes < quantBytes + scaleBytes) return -1;
            char* base = static_cast<char*>(workspace);
            auto* xq = reinterpret_cast<int8_t*>(base);
            auto* xs = reinterpret_cast<float*>(base + quantBytes);
            void* scratch = base + quantBytes + scaleBytes;

            if (convrot_quant(in[kX], xq, xs, M_, K_, inCode, actBits, stream) != 0) return -1;

            return convrot_gemm(out[0], static_cast<int8_t const*>(in[kW]),
                                static_cast<float const*>(in[kWS]), bias, xq, xs, M_, N_, K_,
                                WeightBits, actBits, outCode, biasCode, gemmConfig_, scratch,
                                scratchBytes - quantBytes - scaleBytes,
                                stream) == 0
                       ? 0
                       : -1;
        } catch (...) {
            return -1;
        }
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        refreshFields();
        return &fc_;
    }

    char const* getTimingCacheID() noexcept override { return timingID_.c_str(); }

private:
    bool checkDesc(nvinfer1::PluginTensorDesc const& x, nvinfer1::PluginTensorDesc const& w,
                   nvinfer1::PluginTensorDesc const& ws, nvinfer1::PluginTensorDesc const& b,
                   nvinfer1::PluginTensorDesc const& y) const {
        if (x.dims.nbDims < 2 || w.dims.nbDims != 2) return false;
        if (!isFloat(x.type) || w.type != nvinfer1::DataType::kINT8) return false;
        if (ws.type != nvinfer1::DataType::kFLOAT) return false;
        if (b.type != outType(x.type, outDtype_)) return false;  // bias is read in out dtype
        if (groupSize_ != 256 || hasBias_ < 0 || hasBias_ > 1) return false;
        if (outDtype_ < -1 || outDtype_ > 2) return false;
        if (gemmConfig_ < -1) return false;
        // W8A8 has 14 tiles, W4A4 seven; a pinned index beyond that is a caller bug.
        if (gemmConfig_ >= (WeightBits == 4 ? 7 : 14)) return false;

        const int64_t N = w.dims.d[0];
        const int64_t K = w.dims.d[1] * (WeightBits == 4 ? 2 : 1);
        if (N <= 0 || K <= 0) return false;
        if (K % 256 != 0 || K > 16384) return false;
        if (WeightBits == 8 && N % 4 != 0) return false;
        if (WeightBits == 4 && K % 64 != 0) return false;
        if (x.dims.d[x.dims.nbDims - 1] != K) return false;

        // -1 marks a dynamic dim at build time (resolved in onShapeChange).
        for (int i = 0; i + 1 < x.dims.nbDims; ++i)
            if (x.dims.d[i] == 0 || x.dims.d[i] < -1) return false;

        if (numel(ws.dims) != N || numel(b.dims) != N) return false;
        if (y.type != outType(x.type, outDtype_)) return false;
        return true;
    }

    void refreshFields() {
        fields_[0] = nvinfer1::PluginField{"group_size", &groupSize_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[1] = nvinfer1::PluginField{"has_bias", &hasBias_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[2] = nvinfer1::PluginField{"out_dtype", &outDtype_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[3] = nvinfer1::PluginField{"gemm_config", &gemmConfig_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 4;
        fc_.fields = fields_;
    }

    void updateTimingID() {
        timingID_ = std::string(name()) + "/1/k" + std::to_string(K_) + "/n" +
                    std::to_string(N_) + "/g" + std::to_string(groupSize_) + "/b" +
                    std::to_string(hasBias_) + "/o" + std::to_string(outDtype_) + "/c" +
                    std::to_string(gemmConfig_);
    }

    int32_t groupSize_ = 256;
    int32_t hasBias_ = 1;
    int32_t outDtype_ = -1;
    int32_t gemmConfig_ = -1;
    int64_t M_ = 0, K_ = 0, N_ = 0;
    std::string timingID_;
    nvinfer1::PluginField fields_[4]{};
    nvinfer1::PluginFieldCollection fc_{};
};

int32_t readInt(nvinfer1::PluginFieldCollection const* fc, char const* key, int32_t fallback) {
    if (!fc || !fc->fields) return fallback;
    for (int i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (f.name && std::strcmp(f.name, key) == 0 &&
            f.type == nvinfer1::PluginFieldType::kINT32 && f.length >= 1 && f.data)
            return static_cast<int32_t const*>(f.data)[0];
    }
    return fallback;
}

template <int WeightBits>
class Creator final : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase) noexcept override {
        try {
            return new ConvRotLinear<WeightBits>(
                readInt(fc, "group_size", 256), readInt(fc, "has_bias", 1),
                readInt(fc, "out_dtype", -1), readInt(fc, "gemm_config", -1));
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static int32_t g = 256, b = 1, o = -1, c = -1;
        static nvinfer1::PluginField fields[4] = {
            {"group_size", &g, nvinfer1::PluginFieldType::kINT32, 1},
            {"has_bias", &b, nvinfer1::PluginFieldType::kINT32, 1},
            {"out_dtype", &o, nvinfer1::PluginFieldType::kINT32, 1},
            {"gemm_config", &c, nvinfer1::PluginFieldType::kINT32, 1}};
        static nvinfer1::PluginFieldCollection fc{4, fields};
        return &fc;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return ConvRotLinear<WeightBits>::name();
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return "dit-plugins";
    }
};

struct AutoRegister {
    AutoRegister() {
        static Creator<8> c8;
        static Creator<4> c4;
        getPluginRegistry()->registerCreator(c8, "dit-plugins");
        getPluginRegistry()->registerCreator(c4, "dit-plugins");
    }
};
static AutoRegister gAutoRegister;

}  // namespace
