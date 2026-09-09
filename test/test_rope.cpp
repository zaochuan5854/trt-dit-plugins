// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// apply_rope E2E: engine build -> run -> CPU-reference cos. q,k[B,H,S,D],
// freqs[1,H,S,D/2,2,2] FP32. y=M*x (2x2).
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

unsigned rst = 4242;
float frand() {
    rst = rst * 1664525u + 1013904223u;
    return ((rst >> 8) & 0xffffff) / (float)0x800000 - 1.0f;
}

int run(int S, int D, int dt) {
    const int B = 1, H = 4, P = D / 2;
    const int N = B * H * S * D, NF = B * H * S * P * 4;
    // dt: 1=FP16, 2=BF16 (shared by q/k inputs and outputs). freqs stay FP32.
    auto* cre = getPluginRegistry()->getCreator("apply_rope", "1", "comfy_kitchen");
    if (!cre) {
        std::fprintf(stderr, "creator not found\n");
        return 1;
    }
    nvinfer1::PluginFieldCollection fc{0, nullptr};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "t", &fc, nvinfer1::TensorRTPhase::kBUILD);
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto dtype = dt == 1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kBF16;
    const int esz = 2;
    auto toDev = [&](float f, int i, void* p) {
        if (dt == 1)
            static_cast<__half*>(p)[i] = __float2half(f);
        else
            static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(f);
    };
    auto toHost = [&](const void* p, int i) {
        return dt == 1 ? __half2float(static_cast<const __half*>(p)[i])
                       : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    };
    auto* q = net->addInput("q", dtype, nvinfer1::Dims4{B, H, S, D});
    auto* k = net->addInput("k", dtype, nvinfer1::Dims4{B, H, S, D});
    int64_t fd[6]{1, H, S, P, 2, 2};
    nvinfer1::Dims fdim{6, {1, H, S, P, 2, 2}};
    (void)fd;
    auto* f = net->addInput("f", nvinfer1::DataType::kFLOAT, fdim);
    nvinfer1::ITensor* ins[3]{q, k, f};
    auto* layer = net->addPluginV3(ins, 3, nullptr, 0, *plug);
    layer->getOutput(0)->setName("qo");
    layer->getOutput(1)->setName("ko");
    net->markOutput(*layer->getOutput(0));
    net->markOutput(*layer->getOutput(1));
    auto* cfg = b->createBuilderConfig();
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    auto* ser = b->buildSerializedNetwork(*net, *cfg);
    if (!ser) {
        std::fprintf(stderr, "build engine failed\n");
        return 1;
    }
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
    auto* eng = rt->deserializeCudaEngine(ser->data(), ser->size());
    auto* ctx = eng->createExecutionContext();

    std::vector<float> fqf(N), fkf(N), fqo(N), fko(N);
    std::vector<float> hf(NF);
    std::vector<uint8_t> hq(N * esz), hk(N * esz), hqo(N * esz), hko(N * esz);
    for (int i = 0; i < N; ++i) {
        fqf[i] = frand();
        fkf[i] = frand();
    }
    for (int i = 0; i < N; ++i) {
        toDev(fqf[i], i, hq.data());
        toDev(fkf[i], i, hk.data());
    }
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
            for (int p = 0; p < P; ++p) {
                float th = 0.01f * s * (p + 1);
                float c = std::cos(th), si = std::sin(th);
                int o = ((h * S + s) * P + p) * 4;
                hf[o + 0] = c;
                hf[o + 1] = -si;
                hf[o + 2] = si;
                hf[o + 3] = c;
            }
    void *dq, *dk, *dqo, *dko;
    float* df;
    cudaMalloc(&dq, N * esz);
    cudaMalloc(&dk, N * esz);
    cudaMalloc(&df, NF * 4);
    cudaMalloc(&dqo, N * esz);
    cudaMalloc(&dko, N * esz);
    cudaMemcpy(dq, hq.data(), N * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), N * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf.data(), NF * 4, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->setTensorAddress("q", dq);
    ctx->setTensorAddress("k", dk);
    ctx->setTensorAddress("f", df);
    ctx->setTensorAddress("qo", dqo);
    ctx->setTensorAddress("ko", dko);
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(hqo.data(), dqo, N * esz, cudaMemcpyDeviceToHost);
    cudaMemcpy(hko.data(), dko, N * esz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        fqo[i] = toHost(hqo.data(), i);
        fko[i] = toHost(hko.data(), i);
    }

    double dot = 0, no = 0, nr = 0;
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
            for (int p = 0; p < P; ++p) {
                int fo = ((h * S + s) * P + p) * 4;
                float f00 = hf[fo], f01 = hf[fo + 1], f10 = hf[fo + 2], f11 = hf[fo + 3];
                for (int t = 0; t < 2; ++t) {
                    float x0 = fqf[((h * S + s) * D) + 2 * p + t];
                    float r0 = (t == 0) ? f00 * x0 + f01 * fqf[((h * S + s) * D) + 2 * p + 1]
                                        : f10 * fqf[((h * S + s) * D) + 2 * p] + f11 * x0;
                    float o = fqo[((h * S + s) * D) + 2 * p + t];
                    dot += o * r0;
                    no += o * o;
                    nr += r0 * r0;
                    float y0 = fkf[((h * S + s) * D) + 2 * p + t];
                    float r1 = (t == 0) ? f00 * y0 + f01 * fkf[((h * S + s) * D) + 2 * p + 1]
                                        : f10 * fkf[((h * S + s) * D) + 2 * p] + f11 * y0;
                    float o1 = fko[((h * S + s) * D) + 2 * p + t];
                    dot += o1 * r1;
                    no += o1 * o1;
                    nr += r1 * r1;
                }
            }
    double cos = dot / std::sqrt(no * nr);
    std::fprintf(stderr, "apply_rope S=%d D=%d dt=%d cos=%.5f\n", S, D, dt, cos);
    if (!(cos >= 0.999)) { // also rejects NaN
        std::fprintf(stderr, "COS FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "ROPE_DONE\n");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const char* so = nullptr;
    int pos[3] = {128, 64, 2}, np = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
        else if (np < 3)
            pos[np++] = std::atoi(argv[i]);
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    return run(pos[0], pos[1], pos[2]);
}
