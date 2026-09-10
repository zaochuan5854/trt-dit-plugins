// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// rms_rope_split_half E2E: RMSNorm (full D) + split-half rotation (rot prefix).
// q,k[B,H,S,D] BF16, freqs[1,H,S,rot/2,2,2] FP32, scale[D] FP32。
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

unsigned rst = 5150;
float frand() {
    rst = rst * 1664525u + 1013904223u;
    return ((rst >> 8) & 0xffffff) / (float)0x800000 - 1.0f;
}

int run(int S, int D, int rotArg, float eps, int dt) {
    const int B = 1, H = 4, P = D / 2;
    const int rot = rotArg > 0 ? rotArg : D;
    const int N = B * H * S * D, NF = B * H * S * (rot / 2) * 4;
    auto* cre = getPluginRegistry()->getCreator(
        "rms_rope_split_half", "1", "dit-plugins");
    if (!cre) {
        std::fprintf(stderr, "creator not found\n");
        return 1;
    }
    nvinfer1::PluginField f0{"epsilon", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1};
    nvinfer1::PluginField f1{"rot_dim", &rotArg, nvinfer1::PluginFieldType::kINT32, 1};
    nvinfer1::PluginField fs[2]{f0, f1};
    nvinfer1::PluginFieldCollection fc{2, fs};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "t", &fc, nvinfer1::TensorRTPhase::kBUILD);
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto dtype = dt == 1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kBF16;
    auto* q = net->addInput("q", dtype, nvinfer1::Dims4{B, H, S, D});
    auto* k = net->addInput("k", dtype, nvinfer1::Dims4{B, H, S, D});
    nvinfer1::Dims fdim{6, {1, H, S, rot / 2, 2, 2}};
    auto* f = net->addInput("f", nvinfer1::DataType::kFLOAT, fdim);
    auto* qs = net->addInput("qs", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {D}});
    auto* ks = net->addInput("ks", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {D}});
    nvinfer1::ITensor* ins[5]{q, k, f, qs, ks};
    auto* layer = net->addPluginV3(ins, 5, nullptr, 0, *plug);
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
    std::vector<float> fqf(N), fkf(N), fqo(N), fko(N);
    std::vector<uint8_t> hq(N * 2), hk(N * 2), hqo(N * 2), hko(N * 2);
    std::vector<float> hf(NF), hqs(D), hks(D);
    for (int i = 0; i < N; ++i) {
        fqf[i] = frand() * 2;
        fkf[i] = frand() * 2;
        toDev(fqf[i], i, hq.data());
        toDev(fkf[i], i, hk.data());
    }
    for (int i = 0; i < D; ++i) {
        hqs[i] = 1.0f + frand() * 0.2f;
        hks[i] = 1.0f + frand() * 0.2f;
    }
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
            for (int p = 0; p < rot / 2; ++p) {
                float th = 0.02f * s * (p + 1);
                float c = std::cos(th), si = std::sin(th);
                int o = ((h * S + s) * (rot / 2) + p) * 4;
                hf[o + 0] = c;
                hf[o + 1] = -si;
                hf[o + 2] = si;
                hf[o + 3] = c;
            }
    __nv_bfloat16 *dq, *dk, *dqo, *dko;
    float *df, *dqs, *dks;
    cudaMalloc(&dq, N * 2);
    cudaMalloc(&dk, N * 2);
    cudaMalloc(&df, NF * 4);
    cudaMalloc(&dqs, D * 4);
    cudaMalloc(&dks, D * 4);
    cudaMalloc(&dqo, N * 2);
    cudaMalloc(&dko, N * 2);
    cudaMemcpy(dq, hq.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf.data(), NF * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dqs, hqs.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dks, hks.data(), D * 4, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->setTensorAddress("q", dq);
    ctx->setTensorAddress("k", dk);
    ctx->setTensorAddress("f", df);
    ctx->setTensorAddress("qs", dqs);
    ctx->setTensorAddress("ks", dks);
    ctx->setTensorAddress("qo", dqo);
    ctx->setTensorAddress("ko", dko);
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(hqo.data(), dqo, N * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hko.data(), dko, N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        fqo[i] = toHost(hqo.data(), i);
        fko[i] = toHost(hko.data(), i);
    }

    auto ref = [&](float* x, float* sc, float* o) {
        double dot = 0, no = 0, nr = 0;
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s) {
                double ms = 0;
                for (int d = 0; d < D; ++d) {
                    float v = x[((h * S + s) * D) + d];
                    ms += v * v;
                }
                double rr = 1.0 / std::sqrt(ms / D + eps);
                for (int d = 0; d < D; ++d) {
                    float n = x[((h * S + s) * D) + d] * (float)rr * sc[d];
                    float r;
                    if (d < rot) {
                        int p = (d < rot / 2) ? d : d - rot / 2;
                        int fo = ((h * S + s) * (rot / 2) + p) * 4;
                        float other = x[((h * S + s) * D) + d + (d < rot / 2 ? rot / 2 : -rot / 2)] *
                                      (float)rr * sc[d + (d < rot / 2 ? rot / 2 : -rot / 2)];
                        r = (d < rot / 2) ? hf[fo] * n + hf[fo + 1] * other
                                          : hf[fo + 2] * other + hf[fo + 3] * n;
                    } else {
                        r = n;
                    }
                    float ov = o[((h * S + s) * D) + d];
                    dot += ov * r;
                    no += ov * ov;
                    nr += r * r;
                }
            }
        return dot / std::sqrt(no * nr);
    };
    double cq = ref(fqf.data(), hqs.data(), fqo.data());
    double ck = ref(fkf.data(), hks.data(), fko.data());
    std::fprintf(stderr, "rms_rope_split_half S=%d D=%d rot=%d dt=%d cosQ=%.5f cosK=%.5f\n", S, D,
        rot, dt, cq, ck);
    if (!(cq >= 0.999) || !(ck >= 0.999)) { // also rejects NaN
        std::fprintf(stderr, "COS FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "RMSROPE_DONE\n");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const char* so = nullptr;
    int pos[4] = {64, 64, 0, 2}, np = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
        else if (np < 4)
            pos[np++] = std::atoi(argv[i]);
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    float eps = 1e-6f;
    int rc = run(pos[0], pos[1], pos[2], eps, pos[3]);
    return rc;
}
