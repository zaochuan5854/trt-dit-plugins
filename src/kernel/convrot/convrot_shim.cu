// SPDX-License-Identifier: Apache-2.0
// ConvRot W8A8 / W4A4 glue: maps the plugin's route table onto the vendored
// comfy-kitchen launchers (plan 5.3). No kernel code, no synchronization.
//
// Route table (same as comfy-kitchen v0.2.39, plan 5.3.1):
//   W8A8  M == 1        fused GEMV (quantization inside)
//         M == 2        rowwise quant + GEMV dequant
//         M >= 3        rowwise quant + fused CUTLASS INT8 GEMM
//   W4A4  sm8x          rowwise INT4 quant + (legacy INT4 MMA | CUTLASS INT4)
//         non-sm8x      rowwise INT8 quant + W4A8 GEMV (M<=8) | chunked GEMM
#include "convrot_shim.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

extern "C" {
void launch_quantize_int8_rowwise_convrot64_kernel(
    const void* input, void* output, void* scales, int64_t num_rows, int64_t num_cols,
    int group_size, int input_dtype_code, bool stochastic, int act_code, uint64_t seed,
    const void* act_weight, float act_eps, cudaStream_t stream);
void launch_quantize_int4_rowwise_convrot64_kernel(
    const void* input, void* output, void* scales, int64_t M, int64_t K, int group_size,
    int input_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream);
void launch_int8_gemv_convrot_fused_kernel(
    const void* input, int input_dtype_code, int act_code, const void* act_weight,
    float act_eps, const void* weight, const void* weight_scales, const void* bias,
    bool has_bias, int bias_dtype_code, void* output, int output_dtype_code, int64_t N,
    int64_t K, int64_t weight_scale_size, const void* residual, const void* residual_scale,
    cudaStream_t stream);
void launch_int8_gemv_dequant_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t num_rows, int64_t num_cols, int64_t K,
    int64_t weight_scale_size, bool has_bias, int output_dtype_code, int bias_dtype_code,
    const void* residual, const void* residual_scale, cudaStream_t stream);
bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream);
bool launch_cutlass_int8_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int config, cudaStream_t stream);
bool launch_cutlass_int8_dequant_act(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int epilogue, cudaStream_t stream);
bool launch_cutlass_int8_dequant_act_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int epilogue, int config,
    cudaStream_t stream);
bool launch_cutlass_int8_dequant_act_residual_gate(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    const void* gate, const void* residual, void* D, int64_t M, int64_t N, int64_t K,
    int out_dtype_code, int epilogue, int config, cudaStream_t stream);
bool launch_cutlass_int8_qkv(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* q, void* k, void* v, int64_t M, int64_t N, int64_t K, int64_t S, int H, int D,
    int out_dtype_code, int config, cudaStream_t stream);
int convrot_layout_bhsd(const void* src, void* q, void* k, void* v, int64_t B, int64_t S,
                        int64_t H, int64_t D, int qkv_count, int dtype_code,
                        cudaStream_t stream);
bool launch_cutlass_int4_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream);
bool launch_cutlass_int4_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int config, cudaStream_t stream);
void launch_int4_linear_kernel(
    const void* act, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t M, int64_t N, int64_t K, bool has_bias,
    int output_dtype_code, int bias_dtype_code, cudaStream_t stream);
void launch_int4_weight_int8_act_gemv_dequant_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, int64_t num_rows, int64_t num_cols, int64_t K,
    int64_t weight_scale_size, bool has_bias, int output_dtype_code, int bias_dtype_code,
    cudaStream_t stream);
void launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
    const void* input, const void* weight, const void* x_scales, const void* weight_scales,
    const void* bias, void* output, void* weight_workspace, void* acc_workspace,
    void* cublas_workspace, int64_t cublas_workspace_size, int64_t num_rows, int64_t num_cols,
    int64_t K, int64_t weight_scale_size, int64_t chunk_cols, bool allow_sm80_cutlass,
    bool has_bias, int output_dtype_code, int bias_dtype_code, cudaStream_t stream);
}

namespace {

constexpr int kConvRotGroup = 256;
constexpr int kFusedGemvThreads = 512;
constexpr int kInt4Group = 64;  // mma.sync m16n8k64 -> K % 64 == 0
constexpr size_t align256(size_t v) { return (v + 255u) & ~size_t(255u); }

int currentDevice() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

int archMajor(int device) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return 0;
    return major;
}

int smCount(int device) {
    int n = 0;
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return 0;
    return n;
}

bool nextPow2(int64_t v, int64_t& out) {
    out = 32;
    while (out < v) out <<= 1;
    return true;
}

// comfy's _prefer_legacy_int4_kernel: low-latency shapes beat CUTLASS.
bool preferLegacyInt4(int64_t M, int64_t N, int64_t K, int sms) {
    if (sms <= 0) return false;
    int64_t mTile = 0;
    nextPow2(M, mTile);
    const int64_t base = K <= 1024 ? 1048576 : 786432;
    const int64_t threshold = base * sms / 142;
    const int64_t workload = mTile * N;
    return N < 32768 &&
           (workload < threshold ||
            (K > 1024 && workload == threshold && 128 <= M && M == mTile && M < 512));
}

// comfy's _w4a8_weight_chunk_cols.
int64_t chunkCols(int64_t N, int device) {
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) == cudaSuccess && l2 > 0 &&
        l2 < (16 << 20))
        return N;
    return N <= 2560 ? N : std::min<int64_t>(N, 4096);
}

size_t maxOptinSharedMemory(int device) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
        return 0;
    return static_cast<size_t>(v);
}

// comfy's M=1 gate: aligned weight rows plus the extra INT8 row in smem.
bool fusedGemvM1Fits(int64_t K, const int8_t* w) {
    if (K % kConvRotGroup != 0) return false;
    if ((reinterpret_cast<uintptr_t>(w) & 15) != 0) return false;
    const size_t smem = (static_cast<size_t>(K) +
                         (kFusedGemvThreads / 64) * 2 * kConvRotGroup) * sizeof(float) +
                        static_cast<size_t>(K);
    return smem < maxOptinSharedMemory(currentDevice());
}

int tri(bool ok) { return ok ? 0 : -1; }

}  // namespace

extern "C" {

int convrot_arch_major(int device) {
    try {
        return archMajor(device);
    } catch (...) {
        return 0;
    }
}

size_t convrot_workspace_bytes(int weight_bits, int64_t M, int64_t N, int64_t K,
                               int in_dtype) {
    (void)in_dtype;  // quantizers consume every float dtype natively
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    const bool int4Native = weight_bits == 4 && archMajor(currentDevice()) == 8;
    size_t total = align256(static_cast<size_t>(M) *
                            (int4Native ? static_cast<size_t>(K / 2) : static_cast<size_t>(K)));
    total += align256(static_cast<size_t>(M) * sizeof(float));
    if (weight_bits == 4 && !int4Native && M > 8) {
        const int64_t cols = chunkCols(N, currentDevice());
        total += align256(static_cast<size_t>(cols) * static_cast<size_t>(K));
        total += align256(static_cast<size_t>(M) * static_cast<size_t>(cols) * sizeof(int32_t));
    }
    return total;
}

int convrot_quant(const void* x, int8_t* q, float* scales, int64_t M, int64_t K,
                  int in_dtype, int act_bits, cudaStream_t stream) {
    try {
        if (act_bits == 4) {
            launch_quantize_int4_rowwise_convrot64_kernel(x, q, scales, M, K, kConvRotGroup,
                                                          in_dtype, false, 0, stream);
        } else {
            launch_quantize_int8_rowwise_convrot64_kernel(x, q, scales, M, K, kConvRotGroup,
                                                          in_dtype, false, 0, 0, nullptr,
                                                          0.f, stream);
        }
        return tri(cudaGetLastError() == cudaSuccess);
    } catch (...) {
        return -1;
    }
}

int convrot_gemm(void* y, const int8_t* w, const float* ws, const void* bias,
                 const int8_t* xq, const float* xs, int64_t M, int64_t N, int64_t K,
                 int weight_bits, int act_bits, int out_dtype, int bias_dtype, int config,
                 int epilogue, const void* residual, const void* gate,
                 void* ws_scratch, size_t ws_scratch_bytes, cudaStream_t stream) {
    try {
        if (M <= 0 || N <= 0 || K <= 0) return 0;
        const bool hasBias = bias != nullptr;
        const int64_t wss = N;  // weight_scale is [N]

        if (weight_bits == 8 && act_bits == 8) {
            if (M == 2) {
                launch_int8_gemv_dequant_kernel(xq, w, xs, ws, bias, y, M, N, K, wss, hasBias,
                                                out_dtype, bias_dtype, nullptr, nullptr, stream);
                if (cudaGetLastError() != cudaSuccess) return -1;
                if (epilogue == 0 && residual == nullptr && gate == nullptr) return 0;
                return convrot_apply_epilogue(y, residual, gate, M, N, epilogue, out_dtype,
                                              stream);
            }
            if ((residual != nullptr) != (gate != nullptr)) return -1;
            // shortcut: M<3 stays on gemv + post-pass; the fused EVT needs a tile.
            if (M >= 3 && residual != nullptr && epilogue != 0) {
                bool ok = launch_cutlass_int8_dequant_act_residual_gate(
                    xq, w, xs, ws, bias, gate, residual, y, M, N, K, out_dtype, epilogue,
                    config, stream);
                if (!ok && config >= 0) {
                    (void)cudaGetLastError();
                    ok = launch_cutlass_int8_dequant_act_residual_gate(
                        xq, w, xs, ws, bias, gate, residual, y, M, N, K, out_dtype, epilogue,
                        -1, stream);
                }
                if (ok) return 0;
                (void)cudaGetLastError();
            }
            if (epilogue != 0 && residual == nullptr) {
                bool ok = config >= 0
                    ? launch_cutlass_int8_dequant_act_config(xq, w, xs, ws, bias, y, M, N, K,
                                                             out_dtype, epilogue, config, stream)
                    : launch_cutlass_int8_dequant_act(xq, w, xs, ws, bias, y, M, N, K,
                                                      out_dtype, epilogue, stream);
                if (!ok && config >= 0) {
                    (void)cudaGetLastError();
                    ok = launch_cutlass_int8_dequant_act(xq, w, xs, ws, bias, y, M, N, K,
                                                         out_dtype, epilogue, stream);
                }
                if (ok) return 0;
                (void)cudaGetLastError();
            }
            const bool ok = config >= 0
                ? launch_cutlass_int8_dequant_config(xq, w, xs, ws, bias, y, M, N, K, out_dtype,
                                                     config, stream)
                : launch_cutlass_int8_dequant(xq, w, xs, ws, bias, y, M, N, K, out_dtype, stream);
            if (!ok) (void)cudaGetLastError();  // failed launch leaves a sticky error
            if (!ok) return -1;
            // Identity epilogue (no act, no residual, no gate): the CUTLASS EVT
            // already stored the final result, so the in-place pass would be a
            // pure read+write of [M,N] — it dominated large-M layers (0.7.0
            // regression vs 0.5.1, which fused it).
            if (epilogue == 0 && residual == nullptr && gate == nullptr) return 0;
            return convrot_apply_epilogue(y, residual, gate, M, N, epilogue, out_dtype, stream);
        }

        if (weight_bits == 4 && act_bits == 4) {
            if (K % kInt4Group != 0) return -1;  // legacy INT4 silently no-ops otherwise
            const auto legacy = [&]() -> int {
                launch_int4_linear_kernel(xq, w, xs, ws, bias, y, M, N, K, hasBias, out_dtype,
                                          bias_dtype, stream);
                return tri(cudaGetLastError() == cudaSuccess);
            };
            // A tile whose shared storage exceeds the device budget fails at launch
            // and leaves cudaErrorInvalidValue, so clear it before any fallback.
            const auto cutlass = [&](int cfg) -> bool {
                const bool ok = cfg >= 0
                    ? launch_cutlass_int4_dequant_config(xq, w, xs, ws, bias, y, M, N, K,
                                                         out_dtype, cfg, stream)
                    : launch_cutlass_int4_dequant(xq, w, xs, ws, bias, y, M, N, K, out_dtype,
                                                  stream);
                if (!ok) (void)cudaGetLastError();
                return ok;
            };
            if (config >= 0) {
                if (cutlass(config)) return 0;
                if (config != 0 && cutlass(0)) return 0;
                return legacy();
            }
            if (preferLegacyInt4(M, N, K, smCount(currentDevice()))) return legacy();
            if (cutlass(-1)) return 0;
            if (cutlass(0)) return 0;  // auto tile too large for this device's smem
            return legacy();  // AlignB == 32 unmet, e.g. N % 32 != 0
        }

        if (weight_bits == 4 && act_bits == 8) {
            if (M <= 8) {
                launch_int4_weight_int8_act_gemv_dequant_kernel(xq, w, xs, ws, bias, y, M, N, K,
                                                                wss, hasBias, out_dtype,
                                                                bias_dtype, stream);
                return tri(cudaGetLastError() == cudaSuccess);
            }
            const int64_t cols = chunkCols(N, currentDevice());
            char* p = static_cast<char*>(ws_scratch);
            char* weightChunk = p;
            p += align256(static_cast<size_t>(cols) * static_cast<size_t>(K));
            char* acc = p;
            p += align256(static_cast<size_t>(M) * static_cast<size_t>(cols) * sizeof(int32_t));
            if (static_cast<size_t>(p - static_cast<char*>(ws_scratch)) > ws_scratch_bytes)
                return -1;
            launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
                xq, w, xs, ws, bias, y, weightChunk, acc, nullptr, 0, M, N, K, wss, cols, true,
                hasBias, out_dtype, bias_dtype, stream);
            return tri(cudaGetLastError() == cudaSuccess);
        }

        return -1;  // W8A4 does not exist
    } catch (...) {
        return -1;
    }
}

int convrot_gemv_m1(void* y, const void* x, const int8_t* w, const float* ws, const void* bias,
                    int64_t N, int64_t K, int in_dtype, int out_dtype, int bias_dtype,
                    int epilogue, const void* residual, const void* gate,
                    cudaStream_t stream) {
    try {
        if (!fusedGemvM1Fits(K, w)) return 1;
        launch_int8_gemv_convrot_fused_kernel(x, in_dtype, 0, nullptr, 0.f, w, ws, bias,
                                              bias != nullptr, bias_dtype, y, out_dtype, N, K, N,
                                              nullptr, nullptr, stream);
        if (cudaGetLastError() != cudaSuccess) return -1;
        if (epilogue == 0 && residual == nullptr && gate == nullptr) return 0;
        return convrot_apply_epilogue(y, residual, gate, 1, N, epilogue, out_dtype, stream);
    } catch (...) {
        return -1;
    }
}

int convrot_gemm_layout(void* const* y, const int8_t* w, const float* ws, const void* bias,
                        const int8_t* xq, const float* xs, int64_t M, int64_t N, int64_t K,
                        int64_t S, int H, int D, int out_layout, int config, int out_dtype,
                        int bias_dtype, cudaStream_t stream) {
    if (out_layout != 1 || !y || !y[0] || H <= 0 || D <= 0 || S <= 0) return -1;
    const int qkv = (y[1] != nullptr || y[2] != nullptr) ? 3 : 1;
    if (qkv == 3 && (!y[1] || !y[2])) return -1;
    const int64_t C = int64_t(H) * D;
    if (N != int64_t(qkv) * C) return -1;

    if (M >= 3) {
        bool ok = launch_cutlass_int8_qkv(xq, w, xs, ws, bias, y[0], y[1], y[2], M, N, K, S, H,
                                          D, out_dtype, config, stream);
        if (!ok && config >= 0) {
            (void)cudaGetLastError();
            ok = launch_cutlass_int8_qkv(xq, w, xs, ws, bias, y[0], y[1], y[2], M, N, K, S, H,
                                         D, out_dtype, -1, stream);
        }
        if (ok) return 0;
        (void)cudaGetLastError();
    }

    // Fallback: one BSC GEMM into a temporary, then a single layout kernel.
    const size_t elem = out_dtype == 0 ? sizeof(float) : sizeof(uint16_t);
    const size_t bytes = size_t(M) * size_t(N) * elem;
    void* tmp = nullptr;
    if (cudaMallocAsync(&tmp, bytes, stream) != cudaSuccess) return -1;
    bool ok = config >= 0
        ? launch_cutlass_int8_dequant_config(xq, w, xs, ws, bias, tmp, M, N, K, out_dtype,
                                             config, stream)
        : launch_cutlass_int8_dequant(xq, w, xs, ws, bias, tmp, M, N, K, out_dtype, stream);
    if (!ok && config >= 0) {
        (void)cudaGetLastError();
        ok = launch_cutlass_int8_dequant(xq, w, xs, ws, bias, tmp, M, N, K, out_dtype, stream);
    }
    int rc = -1;
    if (ok) {
        rc = convrot_layout_bhsd(tmp, y[0], y[1], y[2], M / S, S, H, D, qkv, out_dtype, stream);
    } else {
        (void)cudaGetLastError();
    }
    if (cudaFreeAsync(tmp, stream) != cudaSuccess) rc = -1;
    return rc;
}

}  // extern "C"

// Defined in the vendored convrot_w4a4.cu (DIT-PLUGINS-PATCH) next to the
// runner tables it measures.
