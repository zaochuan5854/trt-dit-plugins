// SPDX-License-Identifier: Apache-2.0
// Host declarations for the torch-free SpargeAttn sm89 path.
#pragma once
#include <cstddef>
#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

// Full pre-pipeline: K-mean, Q/K INT8 quant, mask->LUT, V transpose+FP8 quant.
// All I/O contiguous BHSD (fp16/bf16); mask INT32 [B,H,QB,KB].
// Workspace (elements): q_i8 B*H*S*D i8; q_s B*H*NQB f32; k_i8; k_s B*Hkv*NKB;
// km B*Hkv*D f32; lut B*H*QB*KB i32; vbn B*H*QB i32; vT B*Hkv*D*P f16;
// v_fp8 same-shape e4m3; v_s B*Hkv*D f32; v_fp16_tmp B*Hkv*S*D f16 (bf16 V only).
extern "C" void launch_sparge_preprocess(const void* q, const void* k, const void* mask, const void* v,
                                         int8_t* q_i8, float* q_s, int8_t* k_i8, float* k_s, float* km,
                                         int32_t* lut, int32_t* vbn, void* vT, void* v_fp8, float* v_s,
                                         void* v_fp16_tmp, int B, int Hq, int Hkv, int S, int D, int is_bf16,
                                         cudaStream_t stream);

// Main block-sparse kernel (sm89 only). V pre-transposed [B,H,D,P] fp8.
extern "C" void launch_block_sparse_sage2_sm89(int8_t* q_i8, int8_t* k_i8, void* v_fp8, void* o,
                                               int32_t* lut, int32_t* vbn, float* pv_thr, float* q_s,
                                               float* k_s, float* v_s, int B, int Hq, int Hkv, int S,
                                               int D, int is_bf16, float sm_scale, cudaStream_t stream);
