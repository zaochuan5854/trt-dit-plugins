// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// Dynamic-shape check: one engine (profile min64/opt256/max1024) runs 3 shapes.
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Logger : nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity s, const char* m) noexcept override {
        if ((int)s <= (int)nvinfer1::ILogger::Severity::kWARNING)
            std::fprintf(stderr, "[trt] %s\n", m);
    }
} gLogger;

unsigned rst = 20240;
float frand() {
    auto uni = []() {
        rst = rst * 1664525u + 1013904223u;
        return ((rst >> 8) & 0xffffff) / (float)0x1000000;
    };
    float u1 = uni() + 1e-7f, u2 = uni();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

double cpuCos(__nv_bfloat16* hq, __nv_bfloat16* hk, __nv_bfloat16* hv,
    __nv_bfloat16* ho, int H, int S, int D) {
    double dot = 0, no = 0, nr = 0;
    std::vector<float> sc(S);
    const float s = 1.0f / std::sqrt((float)D);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < S; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < S; ++j) {
                float a = 0;
                for (int d = 0; d < D; ++d)
                    a += __bfloat162float(hq[(h * S + i) * D + d]) *
                         __bfloat162float(hk[(h * S + j) * D + d]);
                sc[j] = a * s;
                if (sc[j] > mx)
                    mx = sc[j];
            }
            float sum = 0;
            for (int j = 0; j < S; ++j) {
                sc[j] = std::exp(sc[j] - mx);
                sum += sc[j];
            }
            for (int d = 0; d < D; ++d) {
                float r = 0;
                for (int j = 0; j < S; ++j)
                    r += sc[j] / sum * __bfloat162float(hv[(h * S + j) * D + d]);
                float o = __bfloat162float(ho[(h * S + i) * D + d]);
                dot += o * r;
                no += o * o;
                nr += r * r;
            }
        }
    return dot / std::sqrt(no * nr);
}

int runShape(nvinfer1::IExecutionContext* ctx, int H, int S, int D) {
    const int N = H * S * D;
    std::vector<__nv_bfloat16> hq(N), hk(N), hv(N), ho(N);
    for (int i = 0; i < N; ++i) {
        hq[i] = __float2bfloat16(frand());
        hk[i] = __float2bfloat16(frand());
        hv[i] = __float2bfloat16(frand());
    }
    __nv_bfloat16 *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, N * 2);
    cudaMalloc(&dk, N * 2);
    cudaMalloc(&dv, N * 2);
    cudaMalloc(&dout, N * 2);
    cudaMemcpy(dq, hq.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), N * 2, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->setInputShape("q", nvinfer1::Dims4{1, H, S, D});
    ctx->setInputShape("k", nvinfer1::Dims4{1, H, S, D});
    ctx->setInputShape("v", nvinfer1::Dims4{1, H, S, D});
    ctx->setTensorAddress("q", dq);
    ctx->setTensorAddress("k", dk);
    ctx->setTensorAddress("v", dv);
    ctx->setTensorAddress("o", dout);
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed S=%d\n", S);
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(ho.data(), dout, N * 2, cudaMemcpyDeviceToHost);
    double cos = cpuCos(hq.data(), hk.data(), hv.data(), ho.data(), H, S, D);
    std::fprintf(stderr, "dyn S=%d cos=%.5f\n", S, cos);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
    cudaStreamDestroy(stream);
    return !(cos >= 0.999) ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    const char* so = nullptr;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    const int H = 8, D = 128;
    auto* cre = getPluginRegistry()->getCreator("int8_attention", "1", "dit-plugins");
    if (!cre)
        return 1;
    nvinfer1::PluginFieldCollection fc{0, nullptr};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "dyn", &fc, nvinfer1::TensorRTPhase::kBUILD);
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto* q = net->addInput("q", nvinfer1::DataType::kBF16, nvinfer1::Dims4{1, H, -1, D});
    auto* k = net->addInput("k", nvinfer1::DataType::kBF16, nvinfer1::Dims4{1, H, -1, D});
    auto* v = net->addInput("v", nvinfer1::DataType::kBF16, nvinfer1::Dims4{1, H, -1, D});
    nvinfer1::ITensor* ins[3]{q, k, v};
    auto* layer = net->addPluginV3(ins, 3, nullptr, 0, *plug);
    layer->getOutput(0)->setName("o");
    net->markOutput(*layer->getOutput(0));
    auto* prof = b->createOptimizationProfile();
    prof->setDimensions("q", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, H, 64, D});
    prof->setDimensions("q", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, H, 256, D});
    prof->setDimensions("q", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, H, 1024, D});
    prof->setDimensions("k", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, H, 64, D});
    prof->setDimensions("k", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, H, 256, D});
    prof->setDimensions("k", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, H, 1024, D});
    prof->setDimensions("v", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, H, 64, D});
    prof->setDimensions("v", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, H, 256, D});
    prof->setDimensions("v", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, H, 1024, D});
    auto* cfg = b->createBuilderConfig();
    cfg->addOptimizationProfile(prof);
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    auto* ser = b->buildSerializedNetwork(*net, *cfg);
    if (!ser) {
        std::fprintf(stderr, "build engine failed\n");
        return 1;
    }
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
    auto* eng = rt->deserializeCudaEngine(ser->data(), ser->size());
    auto* ctx = eng->createExecutionContext();
    int rc = 0;
    rc |= runShape(ctx, H, 64, D);
    rc |= runShape(ctx, H, 256, D);
    rc |= runShape(ctx, H, 1024, D);
    if (!rc)
        std::fprintf(stderr, "DYN_DONE\n");
    return rc;
}
