// SPDX-License-Identifier: Apache-2.0
// Post-GEMM activation/residual/gate for the M<3 ConvRot routes.
// The M>=3 W8A8 route fuses the same math into the CUTLASS EVT.
//
// residual_gate on the M>=3 route runs this pass on the GEMM output, so its
// cost is pure traffic plus the per-element `gate[i % cols]` (one 64-bit
// division each) on the scalar form. A 4096x2048 bf16 site took 70.7 us there
// versus 49.1 us with 16-byte vectors (0.8.0, RTX 4070 Ti), which the
// model's 120 sites turn into ~2.6 ms. The vector form hoists the gate
// broadcast to the vector index (valid because ``cols % VEC == 0`` makes
// ``(j*VEC) % cols`` a multiple of VEC) and falls back to the scalar kernel
// for tails, unaligned buffers, and shapes where ``cols % VEC != 0``; at
// bandwidth-saturated sizes (8192x2048, M=16384) the two are equal. Math
// order (act -> *gate -> +residual, float32) is unchanged.
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

// Number of T elements packed into a 128-bit vector access.
template <typename T> struct VecWidth { static constexpr int value = 16 / sizeof(T); };

__host__ __device__ __forceinline__ bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename T>
struct alignas(16) Vec {
    static constexpr int W = VecWidth<T>::value;
    T elts[W];
};

template <typename T, bool kResidual, bool kGate>
__global__ void epilogue_vec_kernel(T* y, const T* residual, const T* gate,
                                    int64_t nvec, int64_t gate_nvec, int epilogue) {
    constexpr int VEC = VecWidth<T>::value;
    Vec<T>* yv = reinterpret_cast<Vec<T>*>(y);
    const Vec<T>* rv = reinterpret_cast<const Vec<T>*>(residual);
    const Vec<T>* gv = reinterpret_cast<const Vec<T>*>(gate);
    for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < nvec;
         j += int64_t(gridDim.x) * blockDim.x) {
        Vec<T> acc = yv[j];
        Vec<T> res{};
        Vec<T> gat{};
        if (kResidual) res = rv[j];
        if (kGate) gat = gv[j % gate_nvec];
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
            float v = to_float(acc.elts[e]);
            v = act(v, epilogue);
            if (kGate) v *= to_float(gat.elts[e]);
            if (kResidual) v += to_float(res.elts[e]);
            acc.elts[e] = from_float<T>(v);
        }
        yv[j] = acc;
    }
}

template <typename T>
void launch_typed(void* y, const void* residual, const void* gate, int64_t n, int64_t cols,
                  int epilogue, cudaStream_t stream) {
    constexpr int VEC = VecWidth<T>::value;
    const int threads = 256;
    // Vector path: whole row is a multiple of VEC elements so a flat index maps
    // to a contiguous gate vector (see file header), and every buffer it
    // touches is 16-byte aligned (TRT and torch allocate that way; a caller
    // binding an offset view keeps the scalar path).
    const bool aligned = aligned16(y)
                         && (residual == nullptr || aligned16(residual))
                         && (gate == nullptr || aligned16(gate));
    const bool vectorize = aligned && (cols % VEC) == 0 && n > 0;
    if (vectorize) {
        const int64_t nvec = n / VEC;
        const int64_t gate_nvec = cols / VEC;
        int64_t blocks = (nvec + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        const T* r = static_cast<const T*>(residual);
        const T* g = static_cast<const T*>(gate);
        if (r && g) {
            epilogue_vec_kernel<T, true, true><<<static_cast<unsigned>(blocks), threads, 0,
                stream>>>(static_cast<T*>(y), r, g, nvec, gate_nvec, epilogue);
        } else if (r) {
            epilogue_vec_kernel<T, true, false><<<static_cast<unsigned>(blocks), threads, 0,
                stream>>>(static_cast<T*>(y), r, nullptr, nvec, gate_nvec, epilogue);
        } else if (g) {
            epilogue_vec_kernel<T, false, true><<<static_cast<unsigned>(blocks), threads, 0,
                stream>>>(static_cast<T*>(y), nullptr, g, nvec, gate_nvec, epilogue);
        } else {
            epilogue_vec_kernel<T, false, false><<<static_cast<unsigned>(blocks), threads, 0,
                stream>>>(static_cast<T*>(y), nullptr, nullptr, nvec, gate_nvec, epilogue);
        }
        return;
    }
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
