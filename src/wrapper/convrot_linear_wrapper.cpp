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
// 1=FP16, 2=BF16 (comfy-kitchen's codes), gemm_config -1=auto or a tile index,
// epilogue 0=none/1=erf GELU/2=SiLU, residual_gate 0/1 (W8A8 only).
// With residual_gate the inputs are x,w,ws,b,residual,gate and
// y = residual + gate * epilogue(gemm(x)). All three use the output dtype.
// The residual keeps x's natural rank or any higher rank whose leading dims
// hold the same row count ([.., N]), and its shape becomes the output; the
// gate broadcasts over rows ([.., 1, N]) or repeats per batch ([B,N]/[B,1,N]).
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

constexpr int kX = 0, kW = 1, kWS = 2, kB = 3, kRes = 4, kGate = 5;
constexpr int kMaxIn = 6;
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

    ConvRotLinear(int32_t groupSize, int32_t hasBias, int32_t outDtype, int32_t gemmConfig,
                  int32_t epilogue = 0, int32_t residualGate = 0, int32_t outLayout = 0,
                  int32_t qkvHeads = 0, int32_t headDim = 0)
        : groupSize_(groupSize), hasBias_(hasBias), outDtype_(outDtype), gemmConfig_(gemmConfig),
          epilogue_(epilogue), residualGate_(residualGate), outLayout_(outLayout),
          qkvHeads_(qkvHeads), headDim_(headDim) {
        refreshFields();
    }

    ConvRotLinear(ConvRotLinear const& o)
        : groupSize_(o.groupSize_), hasBias_(o.hasBias_), outDtype_(o.outDtype_),
          gemmConfig_(o.gemmConfig_), epilogue_(o.epilogue_), residualGate_(o.residualGate_),
          outLayout_(o.outLayout_), qkvHeads_(o.qkvHeads_), headDim_(o.headDim_),
          M_(o.M_), K_(o.K_), N_(o.N_), B_(o.B_), S_(o.S_), H_(o.H_), D_(o.D_) {
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
    int32_t getNbOutputs() const noexcept override { return qkvHeads_ > 0 ? 3 : 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t nbOut,
                               nvinfer1::DataType const* in,
                               int32_t nbIn) const noexcept override {
        if (nbOut != outCount() || nbIn != expectedIn() || !out || !in) return -1;
        for (int i = 0; i < nbOut; ++i) out[i] = outType(in[kX], outDtype_);
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* in, int32_t nbIn,
                            nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* out,
                            int32_t nbOut, nvinfer1::IExprBuilder& b) noexcept override {
        if (nbIn != expectedIn() || nbOut != outCount() || !in || !out) return -1;
        if (in[kX].nbDims < 2 || in[kW].nbDims != 2) return -1;
        if (outLayout_ == 0) {
            // residual+gate takes its output shape from the residual, so x may
            // keep its natural rank (2/3) while the result lands in a
            // higher-rank residual stream without a reshape copy.
            nvinfer1::DimsExprs const& src = residualGate_ ? in[kRes] : in[kX];
            if (src.nbDims < 2) return -1;
            for (int i = 0; i < nbOut; ++i) {
                out[i] = src;
                out[i].d[out[i].nbDims - 1] = in[kW].d[0];
            }
            return 0;
        }
        if (in[kX].nbDims != 3) return -1;
        const int64_t D = headDim_ > 0 ? headDim_ : 128;
        int64_t H = qkvHeads_;
        if (qkvHeads_ == 0) {
            if (!in[kW].d[0]->isConstant()) return -1;
            const int64_t N = in[kW].d[0]->getConstantValue();
            H = N / D;
            if (H * D != N) return -1;
        }
        if (H <= 0 || D <= 0) return -1;
        auto* hExpr = b.constant(H);
        auto* dExpr = b.constant(D);
        for (int i = 0; i < nbOut; ++i) {
            out[i].nbDims = 4;
            out[i].d[0] = in[kX].d[0];
            out[i].d[1] = hExpr;
            out[i].d[2] = in[kX].d[1];
            out[i].d[3] = dExpr;
        }
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
                                   int32_t nbIn, int32_t nbOut) noexcept override {
        if (nbIn != expectedIn() || nbOut != outCount() || !io || pos < 0 ||
            pos >= nbIn + nbOut)
            return false;
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR) return false;
        const auto yType = outType(io[kX].desc.type, outDtype_);
        if (pos >= nbIn) return d.type == yType;
        if (pos == kX) return isFloat(d.type);
        if (pos == kW) return d.type == nvinfer1::DataType::kINT8;
        if (pos == kWS) return d.type == nvinfer1::DataType::kFLOAT;
        if (pos == kB) return d.type == yType;
        if (residualGate_ && pos == kRes) return d.type == yType;
        if (residualGate_ && pos == kGate) return d.type == yType;
        return false;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbIn,
                            nvinfer1::DynamicPluginTensorDesc const* out,
                            int32_t nbOut) noexcept override {
        if (nbIn != expectedIn() || nbOut != outCount() || !in || !out) return -1;
        // Turing (sm_75) has no supported route in this build (plan 4.3).
        if (currentArchMajor() < 8) {
            DIT_REJECT(name(), "configure", "sm major %d < 8", currentArchMajor());
            return -1;
        }
        nvinfer1::PluginTensorDesc desc[kMaxIn];
        for (int i = 0; i < nbIn; ++i) desc[i] = in[i].desc;
        if (!checkDesc(desc, nbIn, out[0].desc)) {
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
        if (nbIn != expectedIn() || nbOut != outCount() || !in || in[kW].max.nbDims != 2)
            return 0;
        int64_t N = in[kW].max.d[0];
        int64_t K = in[kW].max.d[1] * (WeightBits == 4 ? 2 : 1);
        int64_t M = leadingNumel(in[kX].max);
        if (N <= 0 || K <= 0 || M <= 0) return 0;
        return convrot_workspace_bytes(WeightBits, M, N, K, dtypeCodeOf(in[kX].desc.type));
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbIn,
                          nvinfer1::PluginTensorDesc const* out,
                          int32_t nbOut) noexcept override {
        if (nbIn != expectedIn() || nbOut != outCount() || !in || !out) return -1;
        if (!checkDesc(in, nbIn, out[0])) {
            DIT_REJECT(name(), "onShapeChange", "unsupported dtype/shape");
            return -1;
        }
        const int rank = in[kX].dims.nbDims;
        if (rank == 3) {
            B_ = in[kX].dims.d[0];
            S_ = in[kX].dims.d[1];
            M_ = B_ * S_;
        } else if (rank > 3) {
            // [..., K]: collapse every leading dim into M. Treating this as the
            // 2D case (M = d[0]) silently computed a single row.
            B_ = 1;
            M_ = leadingNumel(in[kX].dims);
            S_ = M_;
        } else {
            B_ = 1;
            S_ = in[kX].dims.d[0];
            M_ = S_;
        }
        K_ = in[kW].dims.d[1] * (WeightBits == 4 ? 2 : 1);
        N_ = in[kW].dims.d[0];
        if (outLayout_ == 1) {
            D_ = headDim_ > 0 ? headDim_ : 128;
            const int64_t C = qkvHeads_ > 0 ? N_ / 3 : N_;
            H_ = qkvHeads_ > 0 ? qkvHeads_ : C / D_;
            if (H_ <= 0 || D_ <= 0 || C % D_ != 0 || H_ * D_ != C) {
                DIT_REJECT(name(), "onShapeChange", "bad H/D for N=%lld", (long long)N_);
                return -1;
            }
        } else {
            H_ = 0;
            D_ = 0;
        }
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
            void const* residual = residualGate_ ? in[kRes] : nullptr;
            void const* gate = residualGate_ ? in[kGate] : nullptr;

            // sm8x uses native INT4 MMA; other archs run the W4A8 fallback.
            const bool int4Native = WeightBits == 4 && currentArchMajor() == 8;

            // M == 1 fused GEMV (W8A8 only). Returns 1 when inapplicable.
            if (WeightBits == 8 && M_ == 1 && outLayout_ == 0) {
                const int rc = convrot_gemv_m1(
                    out[0], in[kX], static_cast<int8_t const*>(in[kW]),
                    static_cast<float const*>(in[kWS]), bias, N_, K_, inCode, outCode, biasCode,
                    epilogue_, residual, gate, stream);
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

            if (outLayout_ == 1) {
                void* outs[3]{out[0], qkvHeads_ > 0 ? out[1] : nullptr,
                              qkvHeads_ > 0 ? out[2] : nullptr};
                return convrot_gemm_layout(
                           outs, static_cast<int8_t const*>(in[kW]),
                           static_cast<float const*>(in[kWS]), bias, xq, xs, M_, N_, K_, S_,
                           static_cast<int>(H_), static_cast<int>(D_), outLayout_, gemmConfig_,
                           outCode, biasCode, stream) == 0
                       ? 0
                       : -1;
            }

            const auto run = [&](void* y, int8_t const* q, float const* scales,
                                 int64_t rows, const void* r, const void* g) {
                return convrot_gemm(y, static_cast<int8_t const*>(in[kW]),
                                    static_cast<float const*>(in[kWS]), bias, q, scales, rows,
                                    N_, K_, WeightBits, actBits, outCode, biasCode, gemmConfig_,
                                    epilogue_, r, g, scratch,
                                    scratchBytes - quantBytes - scaleBytes, stream);
            };
            if (residualGate_ && B_ > 1) {
                const size_t elem = outCode == 0 ? sizeof(float) : sizeof(uint16_t);
                auto* y = static_cast<char*>(out[0]);
                auto const* r = static_cast<char const*>(residual);
                auto const* g = static_cast<char const*>(gate);
                // A gate with leading numel 1 broadcasts over rows and is
                // shared by every batch; only [B,N] / [B,1,N] advance.
                const bool gatePerBatch = leadingNumel(inDesc[kGate].dims) != 1;
                for (int64_t batch = 0; batch < B_; ++batch) {
                    const void* gb = gatePerBatch ? g + batch * N_ * elem : gate;
                    if (run(y + batch * S_ * N_ * elem, xq + batch * S_ * K_,
                            xs + batch * S_, S_, r + batch * S_ * N_ * elem,
                            gb) != 0)
                        return -1;
                }
                return 0;
            }
            return run(out[0], xq, xs, residualGate_ ? S_ : M_, residual, gate) == 0 ? 0 : -1;
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
    bool checkDesc(nvinfer1::PluginTensorDesc const* in, int nbIn,
                   nvinfer1::PluginTensorDesc const& y) const {
        if (nbIn != expectedIn()) return false;
        auto const& x = in[kX];
        auto const& w = in[kW];
        auto const& ws = in[kWS];
        auto const& b = in[kB];
        if (x.dims.nbDims < 2 || w.dims.nbDims != 2) return false;
        if (!isFloat(x.type) || w.type != nvinfer1::DataType::kINT8) return false;
        if (ws.type != nvinfer1::DataType::kFLOAT) return false;
        const auto yType = outType(x.type, outDtype_);
        if (b.type != yType || y.type != yType) return false;
        if (groupSize_ != 256 || hasBias_ < 0 || hasBias_ > 1) return false;
        if (outDtype_ < -1 || outDtype_ > 2) return false;
        if (gemmConfig_ < -1) return false;
        if (epilogue_ < 0 || epilogue_ > 2 || residualGate_ < 0 || residualGate_ > 1)
            return false;
        // W8A8 has 14 tiles, W4A4 seven; a pinned index beyond that is a caller bug.
        if (gemmConfig_ >= (WeightBits == 4 ? 7 : 14)) return false;
        if (WeightBits == 4 && (epilogue_ != 0 || residualGate_ != 0)) return false;
        if (outLayout_ < 0 || outLayout_ > 1 || qkvHeads_ < 0 || headDim_ < 0) return false;
        if (WeightBits == 4 && (outLayout_ != 0 || qkvHeads_ != 0)) return false;
        const int64_t N = w.dims.d[0];
        const int64_t K = w.dims.d[1] * (WeightBits == 4 ? 2 : 1);
        if (outLayout_ == 1) {
            if (epilogue_ != 0 || residualGate_ != 0 || x.dims.nbDims != 3) return false;
            const int64_t D = headDim_ > 0 ? headDim_ : 128;
            const int64_t C = qkvHeads_ > 0 ? N / 3 : N;
            const int64_t H = qkvHeads_ > 0 ? qkvHeads_ : C / D;
            if (H <= 0 || D <= 0 || C % D != 0 || H * D != C) return false;
            if (D % 8 != 0) return false;
            if (qkvHeads_ > 0 && N % 3 != 0) return false;
        }

        if (N <= 0 || K <= 0) return false;
        if (K % 256 != 0 || K > 16384) return false;
        if (WeightBits == 8 && N % 4 != 0) return false;
        if (WeightBits == 4 && K % 64 != 0) return false;
        if (x.dims.d[x.dims.nbDims - 1] != K) return false;

        // -1 marks a dynamic dim at build time (resolved in onShapeChange).
        for (int i = 0; i + 1 < x.dims.nbDims; ++i)
            if (x.dims.d[i] == 0 || x.dims.d[i] < -1) return false;

        if (numel(ws.dims) != N || numel(b.dims) != N) return false;
        if (residualGate_) {
            if (WeightBits != 8) return false;
            if (!checkResidualGate(x, in[kRes], in[kGate], N, yType)) return false;
        }
        return true;
    }

    // residual_gate contract; mirrors select.convrot_residual_gate_error in the
    // Python frontend. The residual may keep any rank: the GEMM writes a flat
    // (M, N) buffer, so the contract is residual numel == M*N with last dim N
    // (the output shape is taken from it). The gate broadcasts over rows
    // ([.., 1, N], any rank) or repeats per batch ([B, N] / [B, 1, N], which
    // enqueue slices per batch).
    static bool checkResidualGate(nvinfer1::PluginTensorDesc const& x,
                                  nvinfer1::PluginTensorDesc const& r,
                                  nvinfer1::PluginTensorDesc const& g,
                                  int64_t n, nvinfer1::DataType yType) {
        if (r.type != yType || g.type != yType) return false;
        if (r.dims.nbDims < 2 || r.dims.d[r.dims.nbDims - 1] != n) return false;
        bool dynamic = false;
        for (int i = 0; i + 1 < x.dims.nbDims; ++i)
            if (x.dims.d[i] < 0) dynamic = true;
        for (int i = 0; i + 1 < r.dims.nbDims; ++i)
            if (r.dims.d[i] < 0) dynamic = true;
        if (!dynamic && leadingNumel(r.dims) != leadingNumel(x.dims)) return false;
        // -1 (dynamic at build time) makes leadingNumel report 1, which is the
        // broadcast case; onShapeChange re-checks with resolved dims.
        if (g.dims.nbDims < 2 || g.dims.d[g.dims.nbDims - 1] != n) return false;
        if (leadingNumel(g.dims) != 1) {
            if (g.dims.nbDims > 3) return false;
            if (g.dims.nbDims == 3 && g.dims.d[1] != 1) return false;
            const int64_t rows = g.dims.d[0];
            if (rows == 0 || rows < -1) return false;
            if (rows != -1 && x.dims.d[0] != -1 && rows != x.dims.d[0]) return false;
        }
        return true;
    }

    int32_t expectedIn() const { return 4 + 2 * residualGate_; }
    int32_t outCount() const { return qkvHeads_ > 0 ? 3 : 1; }

    void refreshFields() {
        fields_[0] = nvinfer1::PluginField{"group_size", &groupSize_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[1] = nvinfer1::PluginField{"has_bias", &hasBias_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[2] = nvinfer1::PluginField{"out_dtype", &outDtype_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[3] = nvinfer1::PluginField{"gemm_config", &gemmConfig_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[4] = nvinfer1::PluginField{"epilogue", &epilogue_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[5] = nvinfer1::PluginField{"residual_gate", &residualGate_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[6] = nvinfer1::PluginField{"out_layout", &outLayout_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[7] = nvinfer1::PluginField{"qkv_heads", &qkvHeads_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fields_[8] = nvinfer1::PluginField{"head_dim", &headDim_,
                                           nvinfer1::PluginFieldType::kINT32, 1};
        fc_.nbFields = 9;
        fc_.fields = fields_;
    }

    void updateTimingID() {
        timingID_ = std::string(name()) + "/1/k" + std::to_string(K_) + "/n" +
                    std::to_string(N_) + "/g" + std::to_string(groupSize_) + "/b" +
                    std::to_string(hasBias_) + "/o" + std::to_string(outDtype_) + "/c" +
                    std::to_string(gemmConfig_) + "/e" + std::to_string(epilogue_) + "/rg" +
                    std::to_string(residualGate_) + "/l" + std::to_string(outLayout_) + "/h" +
                    std::to_string(qkvHeads_) + "/d" + std::to_string(headDim_);
    }

    int32_t groupSize_ = 256;
    int32_t hasBias_ = 1;
    int32_t outDtype_ = -1;
    int32_t gemmConfig_ = -1;
    int32_t epilogue_ = 0;
    int32_t residualGate_ = 0;
    int32_t outLayout_ = 0;
    int32_t qkvHeads_ = 0;
    int32_t headDim_ = 0;
    int64_t M_ = 0, K_ = 0, N_ = 0, B_ = 1, S_ = 0, H_ = 0, D_ = 0;
    std::string timingID_;
    nvinfer1::PluginField fields_[9]{};
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
                readInt(fc, "out_dtype", -1), readInt(fc, "gemm_config", -1),
                readInt(fc, "epilogue", 0), readInt(fc, "residual_gate", 0),
                readInt(fc, "out_layout", 0), readInt(fc, "qkv_heads", 0),
                readInt(fc, "head_dim", 0));
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static int32_t g = 256, b = 1, o = -1, c = -1, e = 0, rg = 0;
        static int32_t ol = 0, qh = 0, hd = 0;
        static nvinfer1::PluginField fields[9] = {
            {"group_size", &g, nvinfer1::PluginFieldType::kINT32, 1},
            {"has_bias", &b, nvinfer1::PluginFieldType::kINT32, 1},
            {"out_dtype", &o, nvinfer1::PluginFieldType::kINT32, 1},
            {"gemm_config", &c, nvinfer1::PluginFieldType::kINT32, 1},
            {"epilogue", &e, nvinfer1::PluginFieldType::kINT32, 1},
            {"residual_gate", &rg, nvinfer1::PluginFieldType::kINT32, 1},
            {"out_layout", &ol, nvinfer1::PluginFieldType::kINT32, 1},
            {"qkv_heads", &qh, nvinfer1::PluginFieldType::kINT32, 1},
            {"head_dim", &hd, nvinfer1::PluginFieldType::kINT32, 1}};
        static nvinfer1::PluginFieldCollection fc{9, fields};
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
