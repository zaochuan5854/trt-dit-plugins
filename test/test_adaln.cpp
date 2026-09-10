// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// adaln/rms_adaln E2E: engine build -> run -> CPU-reference cos. x[N,D].
// Usage: ./adaln_test <rms:0|1> <N> <D> <dt> <so>
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

unsigned rst = 999;
float frand() {
    auto uni = []() {
        rst = rst * 1664525u + 1013904223u;
        return ((rst >> 8) & 0xffffff) / (float)0x1000000;
    };
    float u1 = uni() + 1e-7f, u2 = uni();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

int run(bool rms, int N, int D, float eps, int dt) {
    const int M = N * D;
    // dt: 0=FP32, 1=FP16, 2=BF16 (shared by inputs and outputs)
    auto dtype = dt == 0 ? nvinfer1::DataType::kFLOAT
               : dt == 1 ? nvinfer1::DataType::kHALF
                         : nvinfer1::DataType::kBF16;
    const int esz = dt == 0 ? 4 : 2;
    auto toDev = [&](float f, int i, void* p) {
        if (dt == 0)
            static_cast<float*>(p)[i] = f;
        else if (dt == 1)
            static_cast<__half*>(p)[i] = __float2half(f);
        else
            static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(f);
    };
    auto toHost = [&](const void* p, int i) {
        return dt == 0 ? static_cast<const float*>(p)[i]
             : dt == 1 ? __half2float(static_cast<const __half*>(p)[i])
                       : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    };
    auto* cre = getPluginRegistry()->getCreator(
        rms ? "rms_adaln" : "adaln", "1", "dit-plugins");
    if (!cre) {
        std::fprintf(stderr, "creator not found\n");
        return 1;
    }
    nvinfer1::PluginField f{"eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1};
    nvinfer1::PluginFieldCollection fc{1, &f};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "t", &fc, nvinfer1::TensorRTPhase::kBUILD);
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto* x = net->addInput("x", dtype, nvinfer1::Dims2{N, D});
    auto* sc = net->addInput("sc", dtype, nvinfer1::Dims2{N, D});
    auto* sh = net->addInput("sh", dtype, nvinfer1::Dims2{N, D});
    nvinfer1::ITensor* ins[3]{x, sc, sh};
    auto* layer = net->addPluginV3(ins, 3, nullptr, 0, *plug);
    layer->getOutput(0)->setName("o");
    net->markOutput(*layer->getOutput(0));
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

    std::vector<float> fxf(M), fsf(M), fhf(M), fof(M);
    for (int i = 0; i < M; ++i) {
        fxf[i] = frand() * 2;
        fsf[i] = frand() * 0.5f;
        fhf[i] = frand() * 0.5f;
    }
    std::vector<uint8_t> hx(M * esz), hs(M * esz), hh(M * esz), ho(M * esz);
    for (int i = 0; i < M; ++i) {
        toDev(fxf[i], i, hx.data());
        toDev(fsf[i], i, hs.data());
        toDev(fhf[i], i, hh.data());
    }
    void *dx, *ds, *dh, *dout;
    cudaMalloc(&dx, M * esz);
    cudaMalloc(&ds, M * esz);
    cudaMalloc(&dh, M * esz);
    cudaMalloc(&dout, M * esz);
    cudaMemcpy(dx, hx.data(), M * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), M * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, hh.data(), M * esz, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->setTensorAddress("x", dx);
    ctx->setTensorAddress("sc", ds);
    ctx->setTensorAddress("sh", dh);
    ctx->setTensorAddress("o", dout);
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(ho.data(), dout, M * esz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i)
        fof[i] = toHost(ho.data(), i);

    double dot = 0, no = 0, nr = 0;
    for (int n = 0; n < N; ++n) {
        double mean = 0, ms = 0;
        for (int d = 0; d < D; ++d) {
            float v = fxf[n * D + d];
            mean += v;
            ms += v * v;
        }
        mean /= D;
        ms /= D;
        double var = rms ? ms : ms - mean * mean;
        double rstd = 1.0 / std::sqrt(var + eps);
        for (int d = 0; d < D; ++d) {
            float v = fxf[n * D + d];
            float s = fsf[n * D + d];
            float h = fhf[n * D + d];
            float r = (float)((rms ? v : v - mean) * rstd * (1.0 + s) + h);
            float o = fof[n * D + d];
            dot += o * r;
            no += o * o;
            nr += r * r;
        }
    }
    double cos = dot / std::sqrt(no * nr);
    std::fprintf(stderr, "%s N=%d D=%d dt=%d cos=%.5f\n", rms ? "rms_adaln" : "adaln", N, D, dt,
        cos);
    if (!(cos >= 0.999)) { // also rejects NaN
        std::fprintf(stderr, "COS FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "ADALN_DONE\n");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const char* so = nullptr;
    int pos[4] = {0, 256, 128, 2}, np = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
        else if (np < 4)
            pos[np++] = std::atoi(argv[i]);
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    return run(pos[0] != 0, pos[1], pos[2], 1e-6f, pos[3]);
}
