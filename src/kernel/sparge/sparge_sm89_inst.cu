// SPDX-License-Identifier: Apache-2.0
// sm89-only explicit instantiations of SpargeAttn's block-sparse kernel.
// Covers exactly the block_sparse_sage2_attn_cuda path: CTA 128x64, qkg=1,
// pv_mode=1, causal=false, fuse_v_scale=true, no pv-count; hd{64,128} x out{fp16,bf16}.
// Upstream generates 192 combos via autogen.py; we instantiate 4.
#include "csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"

#include <atomic>

extern "C" void ck_prime_once(cudaStream_t stream);

template void SpargeAttentionSM89Dispatched<128, 64, 32, 64, 64, 1, float, true, false, 1, half, false, true, false>(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, half* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale
);
template void SpargeAttentionSM89Dispatched<128, 64, 32, 64, 64, 1, float, true, false, 1, __nv_bfloat16, false, true, false>(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, __nv_bfloat16* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale
);
template void SpargeAttentionSM89Dispatched<128, 64, 32, 64, 128, 1, float, true, false, 1, half, false, true, false>(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, half* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale
);
template void SpargeAttentionSM89Dispatched<128, 64, 32, 64, 128, 1, float, true, false, 1, __nv_bfloat16, false, true, false>(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, __nv_bfloat16* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale
);

// Torch-free launcher mirroring the sm89 branch of block_sparse_sage2_attn_cuda.
// All tensors contiguous BHSD (V pre-transposed to [B,H,D,P] fp8); mask [B,H,QB,KB]
// int32 (nullptr = dense, lut filled with identity in that case is the caller's job;
// here nullptr skips LUT and the kernel still needs valid lut -> caller passes ones-mask).
//
// NOTE: upstream's SpargeAttentionSM89Dispatched launches with <<<>>> (legacy
// default stream). Replicated here with cudaLaunchKernel so TRT's enqueue stream
// is honored.
namespace {
constexpr uint32_t sparge_div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <uint32_t HD, typename DTypeOut>
void sparge_launch_sm89(int8_t* q_i8, int8_t* k_i8, __nv_fp8_e4m3* v_fp8, DTypeOut* o, int32_t* lut,
                        int32_t* vbn, float* pv_thr, float* q_s, float* k_s, float* v_s, int B, int Hq,
                        int Hkv, int S, float sm_scale, uint32_t sb_q, uint32_t ss_q, uint32_t sh_q,
                        uint32_t sb_k, uint32_t ss_k, uint32_t sh_k, uint32_t sb_v, uint32_t sh_v,
                        uint32_t sd_v, uint32_t sb_o, uint32_t ss_o, uint32_t sh_o,
                        cudaStream_t stream) {
    auto kernel_func = qk_int_sv_f8_block_sparse_attn_kernel<
        128, 64, 32, 64, HD, DataType::kInt8, QuantGranularity::kPerBlock, QuantGranularity::kPerBlock,
        float, true, false, PVThresholdMode::kPerBlock /*pv_mode=1*/, DTypeOut, ComputeUnit::kCudaCore,
        MaskMode::kNone, true, false>;
    size_t smem_max =
        std::max((size_t)128 * HD + (size_t)64 * HD + (size_t)64 * HD, (size_t)128 * HD * 2);
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
    dim3 grid(sparge_div_ceil(S, 128), Hq, B);
    dim3 block(32, 4);
    int qo_div_hkv = Hq / Hkv;
    // NOTE: chevron launch (not cudaLaunchKernel): the explicit-stream
    // cudaLaunchKernel path segfaults in the driver when torch ran kernels
    // earlier in the process; chevrons are immune (as in all other plugins).
    kernel_func<<<grid, block, smem_max, stream>>>(
        q_i8, k_i8, reinterpret_cast<int8_t*>(v_fp8) /*legacy*/, o, nullptr /*PV_Count*/, lut, vbn,
        pv_thr, q_s, k_s, v_s, (uint32_t)S, (uint32_t)S, (uint32_t)qo_div_hkv, sb_q, ss_q, sh_q,
        sb_k, ss_k, sh_k, sb_v, sh_v, sd_v, sb_o, ss_o, sh_o, sm_scale);
}
}  // namespace

extern "C" void launch_block_sparse_sage2_sm89(
    int8_t* q_i8, int8_t* k_i8, void* v_fp8, void* o, int32_t* lut, int32_t* vbn, float* pv_thr,
    float* q_s, float* k_s, float* v_s, int B, int Hq, int Hkv, int S, int D, int is_bf16, float sm_scale,
    cudaStream_t stream) {
    // Lazy per-module driver workaround (see prime.cu), same as preprocess.
    static std::atomic<bool> primed{false};
    if (!primed.exchange(true)) ck_prime_once(stream);
    const int P = (S + 127) / 128 * 128;
    // Strides in elements for contiguous BHSD (q/k/o) and BHDP (v).
    // NOTE: seq stride is D (torch layout-1 stride(2)), not S*D.
    const uint32_t sb_q = (uint32_t)Hq * S * D, ss_q = (uint32_t)D, sh_q = (uint32_t)S * D;
    const uint32_t sb_k = (uint32_t)Hkv * S * D, ss_k = ss_q, sh_k = sh_q;
    const uint32_t sb_v = (uint32_t)Hkv * D * P, sh_v = (uint32_t)D * P, sd_v = (uint32_t)P;
    const uint32_t sb_o = sb_q, ss_o = ss_q, sh_o = sh_q;
    if (D == 64 && !is_bf16)
        sparge_launch_sm89<64, half>(q_i8, k_i8, (__nv_fp8_e4m3*)v_fp8, (half*)o, lut, vbn, pv_thr,
                                     q_s, k_s, v_s, B, Hq, Hkv, S, sm_scale, sb_q, ss_q, sh_q, sb_k,
                                     ss_k, sh_k, sb_v, sh_v, sd_v, sb_o, ss_o, sh_o, stream);
    else if (D == 64)
        sparge_launch_sm89<64, __nv_bfloat16>(q_i8, k_i8, (__nv_fp8_e4m3*)v_fp8, (__nv_bfloat16*)o,
                                              lut, vbn, pv_thr, q_s, k_s, v_s, B, Hq, Hkv, S,
                                              sm_scale, sb_q, ss_q, sh_q, sb_k, ss_k, sh_k, sb_v,
                                              sh_v, sd_v, sb_o, ss_o, sh_o, stream);
    else if (!is_bf16)
        sparge_launch_sm89<128, half>(q_i8, k_i8, (__nv_fp8_e4m3*)v_fp8, (half*)o, lut, vbn, pv_thr,
                                      q_s, k_s, v_s, B, Hq, Hkv, S, sm_scale, sb_q, ss_q, sh_q, sb_k,
                                      ss_k, sh_k, sb_v, sh_v, sd_v, sb_o, ss_o, sh_o, stream);
    else
        sparge_launch_sm89<128, __nv_bfloat16>(q_i8, k_i8, (__nv_fp8_e4m3*)v_fp8, (__nv_bfloat16*)o,
                                               lut, vbn, pv_thr, q_s, k_s, v_s, B, Hq, Hkv, S,
                                               sm_scale, sb_q, ss_q, sh_q, sb_k, ss_k, sh_k, sb_v,
                                               sh_v, sd_v, sb_o, ss_o, sh_o, stream);
}
