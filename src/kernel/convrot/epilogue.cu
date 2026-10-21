// SPDX-License-Identifier: Apache-2.0
// Post-GEMM activation/residual/gate for the M<3 ConvRot routes.
// The M>=3 W8A8 route fuses the same math into the CUTLASS EVT.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float act(float v, int epilogue) {
    if (epilogue == 1)
        return 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    if (epilogue == 2)
        return v / (1.0f + expf(-v));
    return v;
}

template <typename T, bool kResidual, bool kGate>
__global__ void epilogue_kernel(T* y, const T* residual, const T* gate, int64_t n,
                                int64_t cols, int epilogue) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x) {
        float v = to_float(y[i]);
        v = act(v, epilogue);
        if (kGate)
            v *= to_float(gate[i % cols]);
        if (kResidual)
            v += to_float(residual[i]);
        y[i] = from_float<T>(v);
    }
}

template <typename T>
void launch_typed(void* y, const void* residual, const void* gate, int64_t n, int64_t cols,
                  int epilogue, cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    const T* r = static_cast<const T*>(residual);
    const T* g = static_cast<const T*>(gate);
    if (r && g) {
        epilogue_kernel<T, true, true><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            static_cast<T*>(y), r, g, n, cols, epilogue);
    } else if (r) {
        epilogue_kernel<T, true, false><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            static_cast<T*>(y), r, nullptr, n, cols, epilogue);
    } else if (g) {
        epilogue_kernel<T, false, true><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            static_cast<T*>(y), nullptr, g, n, cols, epilogue);
    } else {
        epilogue_kernel<T, false, false><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            static_cast<T*>(y), nullptr, nullptr, n, cols, epilogue);
    }
}

}  // namespace

extern "C" int convrot_apply_epilogue(void* y, const void* residual, const void* gate,
                                      int64_t rows, int64_t cols, int epilogue,
                                      int dtype_code, cudaStream_t stream) {
    if (!y || rows <= 0 || cols <= 0)
        return 0;
    const int64_t n = rows * cols;
    switch (dtype_code) {
        case 0: launch_typed<float>(y, residual, gate, n, cols, epilogue, stream); break;
        case 1: launch_typed<__half>(y, residual, gate, n, cols, epilogue, stream); break;
        case 2: launch_typed<__nv_bfloat16>(y, residual, gate, n, cols, epilogue, stream); break;
        default: return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
