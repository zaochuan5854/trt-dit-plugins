// SPDX-License-Identifier: Apache-2.0
// Torch-free SpargeAttn (ae5b629) pre-step kernels for sm89 block-sparse path.
// Mirrors spas_sage_attn/utils.py (qk_quantize, block_map_to_lut) and
// csrc/fused/fused.cu (TransposePadPermuteKernel, MeanScaleKernel) bit-faithfully;
// host launchers take raw pointers + stream (no torch/ATen).
#include <cstdint>

#include <atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "csrc/cp_async.cuh"
#include "csrc/numeric_conversion.cuh"
#include "csrc/reduction_utils.cuh"

// ---- K row-mean (km), fp32 out (torch .mean() equivalent, fp32 accumulation) ----
__global__ void sparge_kmean_kernel(const void* __restrict__ k, float* __restrict__ km,
                                    int S, int Hkv, int D, int stride_b, int stride_h,
                                    int is_bf16) {
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const char* base = (const char*)k + (size_t)b * stride_b + (size_t)h * stride_h;
    double acc = 0.0;
    for (int s = 0; s < S; ++s) {
        const void* p = base + ((size_t)s * D + d) * 2;
        float v = is_bf16 ? __bfloat162float(*(const __nv_bfloat16*)p)
                          : __half2float(*(const half*)p);
        acc += v;
    }
    km[((size_t)b * Hkv + h) * D + d] = (float)(acc / S);
}

// ---- Per-[BS,D]-block INT8 quant, exact mirror of triton qk_quantize ----
// scale = max|x|/127 + 1e-7; round-half-away + truncation; optional mean fuse.
template <typename T>
__global__ void sparge_qk_quant_kernel(const T* __restrict__ x, const float* __restrict__ xm,
                                       int8_t* __restrict__ xq, float* __restrict__ xs,
                                       int N, int D, int BS, int fuse_mean,
                                       int stride_b, int stride_h) {
    const int b = blockIdx.x, h = blockIdx.y, nb = blockIdx.z;
    const int tid = threadIdx.x, nt = blockDim.x;
    const T* xb = x + (size_t)b * stride_b + (size_t)h * stride_h + (size_t)nb * BS * D;
    const float* xmb = xm ? xm + ((size_t)b * gridDim.y + h) * D : nullptr;

    float amax = 0.0f;
    for (int i = tid; i < BS * D; i += nt) {
        int r = i / D, c = i % D;
        float v = 0.0f;
        if (nb * BS + r < N) {
            if constexpr (std::is_same<T, half>::value) v = __half2float(xb[i]);
            else v = __bfloat162float(xb[i]);
            if (fuse_mean) v -= xmb[c];
        }
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float sdata[256];
    sdata[tid] = amax;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    const float scale = sdata[0] / 127.0f + 1e-7f;
    if (tid == 0) xs[((size_t)b * gridDim.y + h) * ((N + BS - 1) / BS) + nb] = scale;
    __syncthreads();
    // NOTE: use division to match triton exactly (x / scale then shift+trunc).
    for (int i = tid; i < BS * D; i += nt) {
        int r = i / D, c = i % D;
        if (nb * BS + r >= N) continue;
        float v;
        if constexpr (std::is_same<T, half>::value) v = __half2float(xb[i]);
        else v = __bfloat162float(xb[i]);
        if (fuse_mean) v -= xmb[c];
        v = v / scale;
        v += 0.5f * (v >= 0.0f ? 1.0f : -1.0f);
        int q = (int)v;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        xq[(size_t)b * stride_b + (size_t)h * stride_h + (size_t)nb * BS * D + i] = (int8_t)q;
    }
}

// ---- mask -> delta-encoded lut + valid counts, mirror of triton_block_map_to_lut_kernel ----
__global__ void sparge_lut_kernel(const int32_t* __restrict__ map, int32_t* __restrict__ lut,
                                  int32_t* __restrict__ vbn, int QB, int KB) {
    const int b = blockIdx.x, h = blockIdx.y, q = blockIdx.z;
    const int32_t* m = map + ((size_t)b * gridDim.y + h) * QB * KB + (size_t)q * KB;
    int32_t* l = lut + ((size_t)b * gridDim.y + h) * QB * KB + (size_t)q * KB;
    int valid = 0, prev = 0;
    for (int i = 0; i < KB; ++i) {
        if (m[i]) {
            l[valid++] = i - prev;
            prev = i;
        }
    }
    vbn[((size_t)b * gridDim.y + h) * QB + q] = valid;
}

// ---- bf16 -> fp16 cast (V path needs fp16 upstream) ----
__global__ void sparge_cast_bf16_fp16_kernel(const __nv_bfloat16* __restrict__ src,
                                             half* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(__bfloat162float(src[i]));
}

// ---- Verbatim TransposePadPermuteKernel from csrc/fused/fused.cu ----
template <typename T>
__device__ __forceinline__ float sparge_convert_to_float(T val) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    if constexpr (std::is_same<T, half>::value) return __half2float(val);
    else return __bfloat162float(val);
}

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero, typename T>
__global__ void SpargeTransposePadPermuteKernel(
    T* __restrict__ input, T* __restrict__ output, const uint32_t num_tokens,
    const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
    const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    constexpr uint32_t pack_size = 8;
    uint32_t num_threads_per_token = head_dim / pack_size;
    uint32_t num_threads_per_cta = CTA_SIZE / pack_size;
    uint32_t bx = blockIdx.x;
    uint32_t head_id = blockIdx.y;
    uint32_t batch_id = blockIdx.z;
    uint32_t thread_id = threadIdx.x;
    uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;
    T* input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input +
                        thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
    T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output +
                         bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size +
                         thread_id / num_threads_per_cta * stride_d_output;
    __shared__ T shared_load[CTA_SIZE][head_dim];
    __shared__ T shared_store[head_dim][CTA_SIZE];
    uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
    uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
    uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod / 8) * 2 +
                             ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);
    constexpr cp_async::SharedMemFillMode fill_mode =
        pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(
        shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base,
        thread_base_token < num_tokens);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();
    uint32_t smem_row_base = thread_id % CTA_SIZE;
    uint32_t smem_col_base = thread_id / CTA_SIZE;
    uint32_t smem_col_stride = head_dim / 8;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        shared_store[smem_col_base + i * smem_col_stride][smem_row_base] =
            shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
    }
    __syncthreads();
    *(float4*)(output_ptr_base) =
        *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}

// ---- Verbatim MeanScaleKernel from csrc/fused/fused.cu (sub_mean=false use) ----
template <uint32_t pad_size, bool sub_mean, typename T>
__global__ void SpargeMeanScaleKernel(
    T* __restrict__ input, int8_t* __restrict__ output, float* __restrict__ mean, float* __restrict__ scale,
    const float scale_max, const uint32_t num_tokens, const uint32_t stride_bz_input,
    const uint32_t stride_d_input, const uint32_t stride_h_input, const uint32_t stride_bz_output,
    const uint32_t stride_d_output, const uint32_t stride_h_output, const uint32_t stride_bz_mean,
    const uint32_t stride_h_mean, const uint32_t stride_bz_scale, const uint32_t stride_h_scale) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    constexpr uint32_t pack_size = 8;
    uint32_t head_id = blockIdx.x;
    uint32_t batch_id = blockIdx.y;
    uint32_t d_id = blockIdx.z;
    uint32_t thread_id = threadIdx.x;
    uint32_t num_threads = blockDim.x;
    uint32_t gmem_stride = num_threads * pack_size;
    uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
    uint32_t num_iters =
        fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);
    T* input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input +
                        d_id * stride_d_input + thread_id * pack_size;
    int8_t* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output +
                              d_id * stride_d_output + thread_id * pack_size;
    T x_val[8];
    float x_val_float[8];
    uint32_t x_val_fp8[2];
    float max_val = -1000000.0f;
    float min_val = 1000000.0f;
    float sum_val = 0.0f;
    for (int i = 0; i < (int)num_iters; i++) {
        *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            float x_temp = sparge_convert_to_float(x_val[j]);
            max_val = fmaxf(max_val, x_temp);
            min_val = fminf(min_val, x_temp);
            if constexpr (sub_mean) sum_val += x_temp;
        }
    }
    __shared__ float s_amax_val;
    __shared__ float s_mean_val;
    float block_max_val = vllm::blockReduceMax(max_val);
    float block_min_val = vllm::blockReduceMin(min_val);
    float block_sum_val = 0.0f;
    if constexpr (sub_mean) block_sum_val = vllm::blockReduceSum(sum_val);
    if (thread_id == 0) {
        s_mean_val = block_sum_val / fp8_padded_num_tokens;
        if constexpr (sub_mean) {
            s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
            mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
        } else {
            s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
        }
        scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
    }
    __syncthreads();
    float mean_val = s_mean_val;
    float recp_scale = scale_max / s_amax_val;
    uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
    num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);
    for (int i = 0; i < (int)num_iters; i++) {
        *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            x_val_float[j] = sparge_convert_to_float(x_val[j]);
            if constexpr (sub_mean) x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
            else x_val_float[j] *= recp_scale;
        }
        floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
        floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);
        *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
    }
}

// ---- Host entry: full pre-pipeline mirroring block_sparse_sage2_attn_cuda (sm89) ----
extern "C" void ck_prime_once(cudaStream_t stream);
extern "C" void launch_sparge_preprocess(const void* q, const void* k, const void* mask, const void* v,
                                         int8_t* q_i8, float* q_s, int8_t* k_i8, float* k_s, float* km,
                                         int32_t* lut, int32_t* vbn, void* vT, void* v_fp8, float* v_s,
                                         void* v_fp16_tmp, int B, int Hq, int Hkv, int S, int D, int is_bf16,
                                         cudaStream_t stream) {
    // Lazy per-module driver workaround (see prime.cu): first launch of this
    // TU's fatbin after host GPU activity segfaults in cuLaunchKernel.
    static std::atomic<bool> primed{false};
    if (!primed.exchange(true)) ck_prime_once(stream);
    const int NQB = (S + 127) / 128, P = (S + 127) / 128 * 128;
    // K mean (fp32)
    {
        dim3 grid(B, Hkv);
        sparge_kmean_kernel<<<grid, (unsigned)D, 0, stream>>>(
            k, km, S, Hkv, D, (int)(Hkv * S * D * 2), (int)(S * D * 2), is_bf16);
    }
    const int qbs = 128, kbs = 64;
    // Q quant (no mean), K quant (fused mean)
    for (int t = 0; t < 2; ++t) {
        const bool is_k = t == 1;
        const void* x = is_k ? k : q;
        int8_t* xq = is_k ? k_i8 : q_i8;
        float* xs = is_k ? k_s : q_s;
        int BS = is_k ? kbs : qbs;
        int NB = (S + BS - 1) / BS;
        int H = is_k ? Hkv : Hq;
        dim3 grid(B, H, NB);
        if (is_bf16)
            sparge_qk_quant_kernel<__nv_bfloat16>
                <<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, is_k ? km : nullptr, xq, xs, S, D,
                                           BS, is_k ? 1 : 0, H * S * D, S * D);
        else
            sparge_qk_quant_kernel<half><<<grid, 256, 0, stream>>>(
                (const half*)x, is_k ? km : nullptr, xq, xs, S, D, BS, is_k ? 1 : 0, H * S * D, S * D);
    }
    // mask -> lut + valid counts (lut zeroed first, mirroring torch.zeros)
    cudaMemsetAsync(lut, 0, (size_t)B * Hq * NQB * ((S + 63) / 64) * 4, stream);
    {
        dim3 grid(B, Hq, NQB);
        sparge_lut_kernel<<<grid, 32, 0, stream>>>((const int32_t*)mask, lut, vbn, NQB, (S + 63) / 64);
    }
    // V -> fp16 (cast if bf16) -> transposed [B,H,D,P] -> fp8 + per-d scales
    const half* v16 = (const half*)v;
    if (is_bf16) {
        size_t n = (size_t)B * Hkv * S * D;
        sparge_cast_bf16_fp16_kernel<<<dim3((unsigned)((n + 255) / 256)), 256, 0, stream>>>(
            (const __nv_bfloat16*)v, (half*)v_fp16_tmp, n);
        v16 = (const half*)v_fp16_tmp;
    }
    // transpose_pad_permute (layout=1 BHSD): out [B,H,D,P]
    {
        dim3 grid(P / 64, Hkv, B);
        if (D == 64)
            SpargeTransposePadPermuteKernel<64, 64, true, half>
                <<<grid, 64 * 8, 0, stream>>>(const_cast<half*>(v16), (half*)vT, S, Hkv * S * D, D,
                                              S * D, Hkv * D * P, P, D * P);
        else
            SpargeTransposePadPermuteKernel<128, 64, true, half>
                <<<grid, 64 * 16, 0, stream>>>(const_cast<half*>(v16), (half*)vT, S, Hkv * S * D, D,
                                               S * D, Hkv * D * P, P, D * P);
    }
    // scale_fuse_quant (scale_max=2.25 per core.py, layout=1)
    {
        dim3 grid(Hkv, B, (unsigned)D);
        SpargeMeanScaleKernel<128, false, half>
            <<<grid, 256, 0, stream>>>((half*)vT, (int8_t*)v_fp8, nullptr, v_s, 2.25f, S, Hkv * D * P, P,
                                       D * P, Hkv * D * P, P, D * P, 0, 0, Hkv * D, D);
    }
}
