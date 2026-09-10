// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// stochastic_round_fp8 E2E: rng=0 pins floor, rng=FF pins ceil.
// Enumerates all 256 E4M3 values to derive down/up directly.
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
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

unsigned rst = 31337;
float frand() {
    rst = rst * 1664525u + 1013904223u;
    return ((rst >> 8) & 0xffffff) / (float)0x800000 - 1.0f;
}

// E4M3 representable-value table (excludes NaN codes 0x7F/0xFF)
float e4m3vals[256];
bool e4m3ok[256];
void buildTable() {
    for (int b = 0; b < 256; ++b) {
        int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
        e4m3ok[b] = !((b & 0x7F) == 0x7F);
        float v;
        if (!e4m3ok[b])
            v = 0;
        else if (e == 0)
            v = std::ldexp(m / 8.0f, -6);
        else
            v = std::ldexp(1.0f + m / 8.0f, e - 7);
        e4m3vals[b] = s ? -v : v;
    }
}
uint8_t e4m3down(float x) { // max{v<=x}
    uint8_t best = 0;
    float bv = -1e30f;
    for (int b = 0; b < 256; ++b) {
        if (!e4m3ok[b])
            continue;
        if (e4m3vals[b] <= x && e4m3vals[b] >= bv) {
            bv = e4m3vals[b];
            best = (uint8_t)b;
        }
    }
    return best;
}
uint8_t e4m3up(float x) { // min{v>=x}
    uint8_t best = 0;
    float bv = 1e30f;
    for (int b = 0; b < 256; ++b) {
        if (!e4m3ok[b])
            continue;
        if (e4m3vals[b] >= x && e4m3vals[b] <= bv) {
            bv = e4m3vals[b];
            best = (uint8_t)b;
        }
    }
    return best;
}
// Reference input mirrors the kernel: values after BF16->FP16 rounding
float h2f(__nv_bfloat16 v) {
    __half h = __float2half(__bfloat162float(v));
    return __half2float(h);
}
// Expected value with kernel-identical semantics: floor over the abs mantissa
// (+rng carry). rng=0 -> trunc(magnitude); rng=FF -> +1ULP when frac>=1/256.
uint8_t expectE4M3(float t, int rng) {
    float s = t < 0 ? -1.0f : 1.0f;
    float a = std::fabs(t);
    if (a == 0)
        return 0;
    int e2;
    float m = std::frexp(a, &e2); // a = m*2^e2, m in [0.5,1)
    int expf = e2 + 6;            // E4M3 exp field (bias7, m*2 in [1,2))
    float mantFull;
    if (expf <= 0) {
        expf = 0;
        mantFull = a * 512.0f; // subnormal
    } else {
        mantFull = (m * 2.0f - 1.0f) * 8.0f;
    }
    float mt = std::floor(mantFull), fr = mantFull - mt;
    int mant = (int)mt + ((rng == 0xFF && fr >= 1.0f / 256.0f) ? 1 : 0);
    int ex = expf;
    if (mant >= 8) {
        mant = 0;
        ex += 1;
    }
    if (ex > 15)
        return s < 0 ? 0xBE : 0x3E; // 448 clamp
    return (uint8_t)((s < 0 ? 0x80 : 0) | (ex << 3) | mant);
}

int runOnce(int rngFill, int N, bool useAlias, int dt) {
    int32_t aliasField = useAlias ? 1 : 0;
    auto* cre = getPluginRegistry()->getCreator(
        "stochastic_round_fp8", "1", "dit-plugins");
    if (!cre) {
        std::fprintf(stderr, "creator not found\n");
        return 1;
    }
    nvinfer1::PluginField f{"alias_rng", &aliasField, nvinfer1::PluginFieldType::kINT32, 1};
    nvinfer1::PluginFieldCollection fc{useAlias ? 1 : 0, useAlias ? &f : nullptr};
    nvinfer1::IPluginV3* plug = static_cast<nvinfer1::IPluginCreatorV3One*>(cre)->createPlugin(
        "t", &fc, nvinfer1::TensorRTPhase::kBUILD);
    nvinfer1::IBuilder* b = nvinfer1::createInferBuilder(gLogger);
    auto* net = b->createNetworkV2(
        1U << (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    // dt: 0=FP32, 1=FP16, 2=BF16 (x input). rng stays INT32, output stays FP8.
    auto xtype = dt == 0 ? nvinfer1::DataType::kFLOAT
               : dt == 1 ? nvinfer1::DataType::kHALF
                         : nvinfer1::DataType::kBF16;
    const int esz = dt == 0 ? 4 : 2;
    auto* x = net->addInput("x", xtype, nvinfer1::Dims2{N, 64});
    auto* r = net->addInput("r", nvinfer1::DataType::kINT32, nvinfer1::Dims2{N, 64});
    nvinfer1::ITensor* ins[2]{x, r};
    auto* layer = net->addPluginV3(ins, 2, nullptr, 0, *plug);
    layer->getOutput(0)->setName("o");
    net->markOutput(*layer->getOutput(0));
    auto* cfg = b->createBuilderConfig();
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if (useAlias)
        cfg->setPreviewFeature(
            nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03, true);
    auto* ser = b->buildSerializedNetwork(*net, *cfg);
    if (!ser) {
        std::fprintf(stderr, "build engine failed\n");
        return 1;
    }
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
    auto* eng = rt->deserializeCudaEngine(ser->data(), ser->size());
    auto* ctx = eng->createExecutionContext();

    const int M = N * 64;
    std::vector<float> fxf(M);
    std::vector<uint8_t> hx(M * esz);
    std::vector<int32_t> hr(M, rngFill ? -1 : 0); // 0xFF/0x00 per byte
    std::vector<uint8_t> ho(M);
    for (int i = 0; i < M; ++i)
        fxf[i] = frand() * 4;
    for (int i = 0; i < M; ++i) {
        if (dt == 0)
            reinterpret_cast<float*>(hx.data())[i] = fxf[i];
        else if (dt == 1)
            reinterpret_cast<__half*>(hx.data())[i] = __float2half(fxf[i]);
        else
            reinterpret_cast<__nv_bfloat16*>(hx.data())[i] = __float2bfloat16(fxf[i]);
    }
    void* dx;
    int32_t* dr;
    uint8_t* dout;
    cudaMalloc(&dx, M * esz);
    cudaMalloc(&dr, M * 4);
    cudaMalloc(&dout, M);
    cudaMemcpy(dx, hx.data(), M * esz, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr.data(), M * 4, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ctx->setTensorAddress("x", dx);
    ctx->setTensorAddress("r", dr);
    ctx->setTensorAddress("o", dout);
    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "enqueue failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(ho.data(), dout, M, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < M; ++i) {
        float raw = dt == 0 ? reinterpret_cast<float*>(hx.data())[i]
                  : dt == 1 ? __half2float(reinterpret_cast<__half*>(hx.data())[i])
                            : __bfloat162float(reinterpret_cast<__nv_bfloat16*>(hx.data())[i]);
        float v = __half2float(__float2half(raw)); // FP16 rounding like the kernel
        float vc = std::fmax(-448.0f, std::fmin(448.0f, v));
        uint8_t want = expectE4M3(vc, rngFill);
        if (ho[i] != want && ++bad < 5)
            std::fprintf(stderr, "mismatch@%d: in=%.4f got=0x%02x want=0x%02x\n", i, v, ho[i], want);
    }
    std::fprintf(stderr, "stochastic_round_fp8 rng=0x%02x bad=%d/%d\n", rngFill, bad, M);
    return bad ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    const char* so = nullptr;
    int pos[2] = {0, 2}, np = 0; // alias, dt
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] == '/')
            so = argv[i];
        else if (np < 2)
            pos[np++] = std::atoi(argv[i]);
    bool useAlias = pos[0] != 0;
    int dt = pos[1];
    if (so)
        dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    buildTable();
    int rc = 0;
    rc |= runOnce(0x00, 256, useAlias, dt); // always trunc
    rc |= runOnce(0xFF, 256, useAlias, dt); // +1ULP when frac>=1/256
    if (!rc)
        std::fprintf(stderr, "FP8_DONE alias=%d dt=%d\n", (int)useAlias, dt);
    return rc;
}
