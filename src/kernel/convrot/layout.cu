// SPDX-License-Identifier: Apache-2.0
// BSC -> BHSD/QKV layout fallback for the ConvRot M<3 route and for the
// rare case where the fused CUTLASS QKV store cannot implement a shape.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename T>
__global__ void qkv_bhsd_kernel(const T* __restrict__ src, T* __restrict__ q,
                                T* __restrict__ k, T* __restrict__ v, int64_t total,
                                int64_t B, int64_t S, int64_t H, int64_t D,
                                int qkv_count) {
    for (int64_t flat = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; flat < total;
         flat += int64_t(gridDim.x) * blockDim.x) {
        int64_t i = flat;
        const int64_t d = i % D;
        i /= D;
        const int64_t h = i % H;
        i /= H;
        int64_t qkv = 0, s = 0, b = 0;
        if (qkv_count == 3) {
            qkv = i % 3;
            i /= 3;
        }
        s = i % S;
        i /= S;
        b = i;
        T* dst = qkv_count == 3 ? (qkv == 0 ? q : (qkv == 1 ? k : v)) : q;
        if (dst != nullptr) dst[((b * H + h) * S + s) * D + d] = src[flat];
    }
}

}  // namespace

extern "C" int convrot_layout_bhsd(const void* src, void* q, void* k, void* v, int64_t B,
                                   int64_t S, int64_t H, int64_t D, int qkv_count,
                                   int dtype_code, cudaStream_t stream) {
    if (!src || !q || B <= 0 || S <= 0 || H <= 0 || D <= 0 ||
        (qkv_count != 1 && qkv_count != 3))
        return -1;
    const int64_t total = B * S * qkv_count * H * D;
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    switch (dtype_code) {
        case 0:
            qkv_bhsd_kernel<float><<<unsigned(blocks), threads, 0, stream>>>(
                static_cast<const float*>(src), static_cast<float*>(q), static_cast<float*>(k),
                static_cast<float*>(v), total, B, S, H, D, qkv_count);
            break;
        case 1:
            qkv_bhsd_kernel<__half><<<unsigned(blocks), threads, 0, stream>>>(
                static_cast<const __half*>(src), static_cast<__half*>(q), static_cast<__half*>(k),
                static_cast<__half*>(v), total, B, S, H, D, qkv_count);
            break;
        case 2:
            qkv_bhsd_kernel<__nv_bfloat16><<<unsigned(blocks), threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(src), static_cast<__nv_bfloat16*>(q),
                static_cast<__nv_bfloat16*>(k), static_cast<__nv_bfloat16*>(v), total, B, S, H, D,
                qkv_count);
            break;
        default:
            return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
