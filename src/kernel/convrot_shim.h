// SPDX-License-Identifier: Apache-2.0
// Thin C glue between the TensorRT plugin and the vendored comfy-kitchen
// ConvRot launchers (src/kernel/vendor/comfy). No kernels live here.
//
// dtype codes are comfy's: 0 = FP32, 1 = FP16, 2 = BF16.
// Errors: 0 = ok, 1 = "not applicable, caller falls back", -1 = failure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

// 8 = native INT4 MMA (Ampere/Ada); anything else uses the W4A8 fallback.
int convrot_arch_major(int device);

// Workspace for the worst-case profile: activation quant scratch plus, on
// non-sm8x W4A4, the W4A8 weight chunk and int32 accumulator.
size_t convrot_workspace_bytes(int weight_bits, int64_t M, int64_t N, int64_t K, int in_dtype);

// act_bits: 8 -> rowwise INT8 quant, 4 -> rowwise packed INT4 quant.
int convrot_quant(const void* x, int8_t* q, float* scales, int64_t M, int64_t K,
                  int in_dtype, int act_bits, cudaStream_t stream);

// Fused GEMM over an already-quantized activation.
// weight_bits and act_bits are independent: non-sm8x W4A4 is (4, 8).
int convrot_gemm(void* y, const int8_t* w, const float* ws, const void* bias,
                 const int8_t* xq, const float* xs, int64_t M, int64_t N, int64_t K,
                 int weight_bits, int act_bits, int out_dtype, int bias_dtype, int config,
                 void* ws_scratch, size_t ws_scratch_bytes, cudaStream_t stream);

// M == 1 fused GEMV (W8A8 only). Returns 1 when this route cannot be used.
int convrot_gemv_m1(void* y, const void* x, const int8_t* w, const float* ws, const void* bias,
                    int64_t N, int64_t K, int in_dtype, int out_dtype, int bias_dtype,
                    cudaStream_t stream);

// Build-time only (never from enqueue). Returns the best config index or -1.
int convrot_int4_autotune(int64_t M, int64_t N, int64_t K, int out_dtype,
                          void* scratch, size_t scratch_bytes, cudaStream_t stream);
}
