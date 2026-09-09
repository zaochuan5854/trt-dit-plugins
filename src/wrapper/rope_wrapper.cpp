// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// TRT IPluginV3 for comfy-kitchen's apply_rope (torch.ops.comfy_kitchen.apply_rope).
// Registration: comfy_kitchen::apply_rope. Base precision BF16, interleaved (split_half=false).
// Contract: q[B,H,S,D], k[B,H,S,D] (D even), freqs[fb,f1,f2,D/2,2,2] (FP32/FP16/BF16,
//   first 3 dims broadcastable) -> qo, ko (same shape). Strided inputs OK (dense strides derived).
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>

extern "C" {
void launch_apply_rope_kernel(const void* q, const void* k, const void* freqs,
    void* q_out, void* k_out, int64_t batch, int64_t dim1, int64_t dim2,
    int64_t head_dim, int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0,
    int64_t qo_s1, int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1,
    int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2,
    int64_t f_s3, int64_t f_s4, int64_t f_s5, int input_dtype_code,
    int freqs_dtype_code, bool has_k, bool split_half, cudaStream_t stream);
}

namespace {
constexpr char const* kName = "apply_rope";
constexpr char const* kVersion = "1";
constexpr char const* kNamespace = "comfy_kitchen";

int dtypeCode(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT)
        return 0;
    if (t == nvinfer1::DataType::kHALF)
        return 1;
    return 2; // kBF16
}

class ApplyRoPE : public nvinfer1::IPluginV3,
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
            return new ApplyRoPE(*this);
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
        if (nbOut != 2 || nbIn != 3)
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
        if (nbIn != 3 || nbOut != 2)
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
        if (pos == 2) // freqs
            return d.type == nvinfer1::DataType::kFLOAT ||
                   d.type == nvinfer1::DataType::kHALF || d.type == nvinfer1::DataType::kBF16;
        return d.type == io[pos - nbIn].desc.type; // out matches corresponding input
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return nbIn == 3 && valid(in[0].desc.dims, in[1].desc.dims, in[2].desc.dims) ? 0 : -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn != 3 || !valid(in[0].dims, in[1].dims, in[2].dims))
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
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
        void const* const* in, void* const* out, void*, cudaStream_t stream) noexcept override {
        try {
            int64_t qst[4]{(int64_t)H_ * S_ * D_, (int64_t)S_ * D_, D_, 1};
            int64_t f3 = D_ / 2;
            // dense contiguous freqs strides: s5=1,s4=2,s3=4,s2=4*f3,s1=4*f3*f2,s0=4*f3*f2*f1
            int64_t fst[6]{};
            fst[5] = 1;
            fst[4] = 2;
            fst[3] = 4;
            fst[2] = 4 * f3;
            fst[1] = 4 * f3 * f2_;
            fst[0] = 4 * f3 * f2_ * f1_;
            launch_apply_rope_kernel(in[0], in[1], in[2], out[0], out[1], B_, H_, S_, D_,
                fb_, f1_, f2_, qst[0], qst[1], qst[2], qst[3], qst[0], qst[1], qst[2],
                qst[3], qst[0], qst[1], qst[2], qst[3], qst[0], qst[1], qst[2],
                qst[3], fst[0], fst[1], fst[2], fst[3], fst[4], fst[5], icode_, fcode_,
                true, false, stream);
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

    char const* getTimingCacheID() noexcept override { return "apply_rope/1"; }

private:
    template <typename DimsT>
    static bool valid(DimsT const& q, DimsT const& k, DimsT const& f) {
        if (q.nbDims != 4 || k.nbDims != 4 || f.nbDims != 6)
            return false;
        for (int i = 0; i < 4; ++i)
            if (q.d[i] != k.d[i])
                return false;
        int64_t D = q.d[3];
        if (D <= 0 || D % 2 != 0)
            return false;
        if (f.d[3] != D / 2 || f.d[4] != 2 || f.d[5] != 2)
            return false;
        int64_t qd[3]{q.d[0], q.d[1], q.d[2]}, fd[3]{f.d[0], f.d[1], f.d[2]};
        for (int i = 0; i < 3; ++i)
            if (fd[i] != 1 && fd[i] != qd[i])
                return false;
        return q.d[0] > 0 && q.d[1] > 0 && q.d[2] > 0;
    }

    int64_t B_ = 0, H_ = 0, S_ = 0, D_ = 0, fb_ = 1, f1_ = 1, f2_ = 1;
    int icode_ = 2, fcode_ = 0;
    nvinfer1::PluginFieldCollection fc_{};
};

class ApplyRoPECreator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const*,
        nvinfer1::TensorRTPhase) noexcept override {
        try {
            return new ApplyRoPE();
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

ApplyRoPECreator* gCreator() {
    static ApplyRoPECreator* c = new ApplyRoPECreator();
    return c;
}
struct AutoRegister {
    AutoRegister() { getPluginRegistry()->registerCreator(*gCreator(), kNamespace); }
};
static AutoRegister gAutoRegister;
} // namespace
