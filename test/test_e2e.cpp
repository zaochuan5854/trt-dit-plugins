// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// E2E: engine build -> run -> CPU-reference cos -> bench. BF16 [B,H,S,D].
// Usage: ./e2e <S> (default 256). Small S for cos check, S=4096 for bench.
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

// Deterministic pseudo-random ~N(0,1) (Box-Muller). A negatively-biased uniform
// saturates softmax and amplifies quantization error, so use a normal distribution
float frand(unsigned& st) {
    auto uni = [&]() {
        st = st * 1664525u + 1013904223u;
        return ((st >> 8) & 0xffffff) / (float)0x1000000;
    };
    float u1 = uni() + 1e-7f, u2 = uni();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

int run(int S, int H, int dt) {
    const int B = 1, D = 128;
    const int N = B * H * S * D;
    // dt: 0=FP32, 1=FP16, 2=BF16 (inputs). Outputs follow comfy convention:
    // FP32->BF16, otherwise same dtype.
    auto inType = dt == 0 ? nvinfer1::DataType::kFLOAT
                : dt == 1 ? nvinfer1::DataType::kHALF
                          : nvinfer1::DataType::kBF16;
    auto outType = dt == 0 ? nvinfer1::DataType::kBF16 : inType;
    const int esz = dt == 0 ? 4 : 2, osz = 2; // outputs always 16-bit (FP16/BF16)
    auto* reg = getPluginRegistry();
    // .so is expected preloaded (LD_PRELOAD equivalent; caller dlopens)
    auto* cre = reg->getCreator("int8_attention", "1", "dit-plugins");
    if (!cre) {
        std::fprintf(stderr, "creator not found\n");
        return 1;
    }
    nvinfer1::PluginFieldCollection fc{0, nullptr};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "e2e", &fc, nvinfer1::TensorRTPhase::kBUILD);
    if (!plug) {
        std::fprintf(stderr, "createPlugin failed\n");
        return 1;
    }
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto* q = net->addInput("q", inType, nvinfer1::Dims4{B, H, S, D});
    auto* k = net->addInput("k", inType, nvinfer1::Dims4{B, H, S, D});
    auto* v = net->addInput("v", inType, nvinfer1::Dims4{B, H, S, D});
    nvinfer1::ITensor* ins[3]{q, k, v};
    auto* layer = net->addPluginV3(ins, 3, nullptr, 0, *plug);
    layer->getOutput(0)->setName("o");
    net->markOutput(*layer->getOutput(0));
    std::fprintf(stderr, "out type=%d (2=HALF 3=FLOAT 4=BF16? check)\n",
        (int)layer->getOutput(0)->getType());
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

    auto toDev = [&](float f, int i, void* p) {
        if (dt == 0)
            static_cast<float*>(p)[i] = f;
        else if (dt == 1)
            static_cast<__half*>(p)[i] = __float2half(f);
        else
            static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(f);
    };
    auto toHost = [&](const void* p, int i) {
        return outType == nvinfer1::DataType::kHALF
                   ? __half2float(static_cast<const __half*>(p)[i])
                   : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    };
    std::vector<float> fqf(N), fkf(N), hvf(N), hof(N);
    unsigned st = 12345;
    for (int i = 0; i < N; ++i) {
        fqf[i] = frand(st);
        fkf[i] = frand(st);
        hvf[i] = frand(st);
    }
    std::vector<uint8_t> hq(N * esz), hk(N * esz), hv(N * esz), ho(N * osz);
    for (int i = 0; i < N; ++i) {
        toDev(fqf[i], i, hq.data());
        toDev(fkf[i], i, hk.data());
        toDev(hvf[i], i, hv.data());
    }
    void *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, N * esz);
    cudaMalloc(&dk, N * esz);
    cudaMalloc(&dv, N * esz);
    cudaMalloc(&dout, N * osz);
    cudaMemcpy(dq, hq.data(), N * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), N * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), N * esz, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemsetAsync(dout, 0xFF, N * osz, stream);
    std::fprintf(stderr, "io tensors:");
    for (int i = 0; i < eng->getNbIOTensors(); ++i)
        std::fprintf(stderr, " [%s]", eng->getIOTensorName(i));
    std::fprintf(stderr, "\n");
    {
        auto d = eng->getTensorShape("o");
        std::fprintf(stderr, "eng o shape=[%d %d %d %d] dtype=%d\n", d.d[0], d.d[1], d.d[2], d.d[3],
            (int)eng->getTensorDataType("o"));
    }
    bool ok = true;
    ok &= ctx->setInputShape("q", nvinfer1::Dims4{B, H, S, D});
    ok &= ctx->setInputShape("k", nvinfer1::Dims4{B, H, S, D});
    ok &= ctx->setInputShape("v", nvinfer1::Dims4{B, H, S, D});
    ok &= ctx->setTensorAddress("q", dq);
    ok &= ctx->setTensorAddress("k", dk);
    ok &= ctx->setTensorAddress("v", dv);
    ok &= ctx->setTensorAddress("o", dout);
    std::fprintf(stderr, "setAddr ok=%d\n", (int)ok);
    std::fprintf(stderr, "dout=%p getAddr(o)=%p\n", (void*)dout, ctx->getTensorAddress("o"));
    std::fprintf(stderr, "dq=%p dk=%p dv=%p getAddr(qkv)=%p %p %p\n", (void*)dq, (void*)dk,
        (void*)dv, ctx->getTensorAddress("q"), ctx->getTensorAddress("k"),
        ctx->getTensorAddress("v"));
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(ho.data(), dout, N * osz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        hof[i] = toHost(ho.data(), i);

    // CPU reference (fp32 naive) + cos
    double dot = 0, no = 0, nr = 0;
    std::vector<float> sc(S);
    const float s = 1.0f / std::sqrt((float)D);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < S; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < S; ++j) {
                float a = 0;
                for (int d = 0; d < D; ++d)
                    a += fqf[(h * S + i) * D + d] * fkf[(h * S + j) * D + d];
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
                    r += sc[j] / sum * hvf[(h * S + j) * D + d];
                float o = hof[(h * S + i) * D + d];
                dot += o * r;
                no += o * o;
                nr += r * r;
            }
        }
    }
    double cos = dot / std::sqrt(no * nr);
    std::fprintf(stderr, "S=%d dt=%d cos=%.5f\n", S, dt, cos);

    // bench
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int iters = 30;
    for (int i = 0; i < 5; ++i)
        ctx->enqueueV3(stream);
    cudaEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i)
        ctx->enqueueV3(stream);
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::fprintf(stderr, "plugin=%.3fms/iter (S=%d,H=%d,D=%d)\n", ms / iters, S, H, D);
    if (S <= 512 && !(cos >= 0.99)) { // also rejects NaN
        std::fprintf(stderr, "COS FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "E2E_DONE\n");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    // dlopen first to run static registration. Usage: ./e2e S H dt /path/to.so
    const char* so = nullptr;
    int pos[3] = {256, 8, 2}, np = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
        else if (np < 3)
            pos[np++] = std::atoi(argv[i]);
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    return run(pos[0], pos[1], pos[2]);
}
