// SPDX-License-Identifier: Apache-2.0
// Shared host-side helpers for the IPluginV3 wrappers.
//
// Build/shape callbacks stay validation-only: the SM probe initialises once
// per process, workspaces are planned in onShapeChange, and the only
// per-enqueue CUDA calls are kernels and capture-safe copies.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace dit {

// Process-static device capability. Called from build-time callbacks only.
inline int smVersion() {
    static const int sm = [] {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaGetDeviceProperties(&p, dev) != cudaSuccess)
            return 0;
        return p.major * 10 + p.minor;
    }();
    return sm;
}

// Attention output store strides. D is contiguous in both layouts, so only
// the (batch, seq, head) strides differ; 0 = BHSD, 1 = BSC.
struct AttnOutStrides {
    int64_t b, s, h;
};

inline AttnOutStrides attnOutStrides(int32_t outLayout, int64_t B, int64_t H,
    int64_t S, int64_t D) {
    if (outLayout == 1)
        return {S * H * D, H * D, D};
    return {H * S * D, D, S * D};
}

// Pinned host staging for small constant uploads (pv thresholds). Pinned
// H2D copies are CUDA-graph capture safe; pageable ones are not. Allocated
// once, on the first plugin construction (engine build), never in capture.
inline float* pinnedFloats() {
    static float* p = [] {
        void* q = nullptr;
        return cudaHostAlloc(&q, 256 * sizeof(float), cudaHostAllocDefault) == cudaSuccess
                   ? static_cast<float*>(q)
                   : nullptr;
    }();
    return p;
}

inline bool verbose() {
    static const bool v = std::getenv("TRT_DIT_PLUGINS_VERBOSE") != nullptr;
    return v;
}

template <typename... A>
inline void reject(char const* op, char const* where, char const* fmt, A... a) {
    if (!verbose())
        return;
    std::fprintf(stderr, "[trt_dit_plugins] %s %s: ", op, where);
    std::fprintf(stderr, fmt, a...);
    std::fputc('\n', stderr);
}

} // namespace dit

#define DIT_REJECT(op, where, ...) ::dit::reject(op, where, __VA_ARGS__)
