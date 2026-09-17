// SPDX-License-Identifier: Apache-2.0
// Own implementation (NOT vendored): lives in src/kernel/own/.
// History: first written as src/kernel/sage_attention/quant_qk_int8_rope.cu,
// moved here to separate own code from vendored SageAttention sources.
// Uses vendored helpers (dtype_dispatch.cuh, float_utils.cuh) via -I.
// Fused pre-stage for DiT-self int8 RoPE+SageAttn (Anima, D=128 only).
//
// Fuses, per Q/K row: INT8 dequant (per-tensor input scale) -> per-head RMSNorm
// (eps=1e-6) -> split-half RoPE from on-the-fly sincos(pos*inv_freq[64],
// [[cos,-sin],[sin,cos]] convention, same as the baked 2x2 capture)
// mean subtraction, Sage smooth-K style, softmax-invariant] -> signed H128
// Hadamard (convrot128, same as quant_qk_per_thread_int8 ROTATION=128 path) ->
// per-thread INT8 quant with Sage-identical scale layout.
//
// Output layout contract is IDENTICAL to launch_quant_qk_per_thread_int8 with
// BLKQ=128/WARPQ=32/BLKK=128/WARPK=128 (the Lk>1024 dispatch of the existing
// quantizer): q_sc_per_h = q_oblk*8, k_sc_per_h = k_oblk*4, so the existing
// launch_sage_attn_kernel consumes our outputs unchanged.
//
// Deliberate simplifications:
// - No anchor search: K uses per-head mean subtraction (smooth-K). The existing
//   detect_k_anchor path is skipped; accuracy is covered by the cos gate.
// - Norm is computed in fp32 WITHOUT the bf16 intermediate rounding that
//   rms_rope.cu applies (strictly less lossy; cos gate absorbs the delta).
// - D=128, contiguous [B,H,L,D] INT8 I/O, L>1024 (asserted, else fallback;
//   the L<=1024 path is broken, so every layer rejects it at build time).
// - Input scales are per-tensor scalars (SmoothQuant static convention).

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

using comfy::quant_int8_rcp;
using comfy::store4_i8;
using comfy::warp_reduce_fmax;

namespace {

// ---- output clamp knob ----
//
// kQuantMax bounds the stored int8 magnitude. 127.0f = standard symmetric
// range (cvt.sat still clips -128). A smaller protective range only pays off
// with a SECOND integer rounding stage downstream; this chain feeds Sage MMA
// int32 accumulators directly (fp32 rescale), so 127 stands. Revisit only if
// the consumer chain gains another integer rounding stage.
constexpr float kQuantMax = 127.0f;

__forceinline__ __device__ int8_t quant_int8_lim(float v, float inv_sc) {
  // Positive-side fminf dropped: cvt.sat.s8 saturates to +127 by itself.
  // Negative side keeps fmaxf(-kQuantMax). Inputs are bounded by
  // construction (|v*inv| <= 127, scale measured over the same data), so no
  // float->int overflow path exists.
  // cvt.pack.sat.s8.s32 (F2IP) deliberately NOT used: its [c,d,b,a]-style
  // lane interleave would force fragile store reorder for unmeasurable gain
  // (store4_i8 already emits a single int32 store).
  float q = fmaxf(v * inv_sc, -kQuantMax);
  int32_t qi;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(qi) : "f"(q));
  return static_cast<int8_t>(qi);
}

// ---- shared device helpers (mirrors quant_qk_int8.cu, int8-input variant) --

__forceinline__ __device__ void convrot4_f(float *values) {
  const float x0 = values[0];
  const float x1 = values[1];
  const float x2 = values[2];
  const float x3 = values[3];
  const float a0 = x0 + x1;
  const float a1 = x0 - x1;
  const float a2 = x2 + x3;
  const float a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

// Folding the convrot4 *0.5 into the scale was measured with zero gain
// (ALU micro-ops below noise); the explicit *0.5 loop above stands.

__forceinline__ __device__ void apply_sign128_f(float *values,
                                                 const int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs =
      lane < 8    ? signs_0
      : lane < 16 ? signs_1
      : lane < 24 ? signs_2
                  : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] =
        __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

__forceinline__ __device__ void convrot128_f(float *values) {
  apply_sign128_f(values, threadIdx.x & 31);
  convrot4_f(values);
  const int lane = threadIdx.x & 31;
  // Sign-flip-via-XOR FADD form stays unwritten: the extra XOR lengthens
  // the dependent chain while the compiler's predicated/select form is
  // already optimal.
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, values[c], bit);
      values[c] = (lane & bit) ? other - values[c] : values[c] + other;
    }
  }
  // The 1/sqrt(32) normalization is NOT applied here. For symmetric INT8,
  // round(V*C/((mx*C)/127)) == round(V/(mx/127)), so the constant folds once
  // into the written block scale (see kHadamardNorm at the scale sites).
  // Saves 1 FMUL/element/pass, bit-near-identical output.
}

// Normalization implicitly skipped in convrot128_f (== 1/sqrt(32)).
// Multiplied once into the stored block scale instead of per element;
// quant divisor is C/sc (see scale sites).
constexpr float kHadamardNorm = 0.1767766952966369f;

__forceinline__ __device__ float warp_reduce_fsum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, offset);
  return __shfl_sync(0xffffffffu, x, 0);
}

// Load 4 packed int8 as float (no scale applied here).
__forceinline__ __device__ void load4_i8(const int8_t *ptr, float *out) {
  const int32_t packed = __ldg(reinterpret_cast<const int32_t *>(ptr));
  out[0] = static_cast<float>(static_cast<int8_t>(packed & 0xff));
  out[1] = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xff));
  out[2] = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xff));
  out[3] = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xff));
}

template <typename WN>
__forceinline__ __device__ float wn_to_float(const WN *w, int idx) {
  return static_cast<float>(__ldg(&w[idx]));
}

// ---- K mean pre-pass (smooth-K): mean[ch] over L rows of dequantized K ----
//
// Latency trap this avoids: one block per (b,h) with a scalar byte loop over
// L rows under-fills the GPU and scales with S only. Instead L is split
// into 128-row chunks processed by many blocks, each vectorized, combined
// with one atomicAdd per (chunk, channel); the single-block/no-atomic form
// loses memory-level parallelism.

constexpr int KMEAN_CHUNK_ROWS = 128; // Larger chunks underfill the SMs at
// small S (fewer blocks); 128 keeps enough blocks in flight.

__global__ __launch_bounds__(128) void k_mean_kernel(
    const int8_t *__restrict__ k_in, float *__restrict__ mean_out,
    const float *__restrict__ k_s_in, int L, int C, int H, int64_t stride_b,
    int64_t stride_h) {
  const int chunk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  // 4 warps cover 128 channels, 4 ch per lane.
  const int ch = wid * 32 + (lane << 2);
  const float k_s = __ldg(k_s_in);
  const int n0 = chunk * KMEAN_CHUNK_ROWS;
  const int n1 = n0 + KMEAN_CHUNK_ROWS < L ? n0 + KMEAN_CHUNK_ROWS : L;
  const int8_t *base = k_in + (int64_t)b * stride_b + (int64_t)h * stride_h;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int n = n0; n < n1; n += 4) {
    // 4 rows x 4 ch via int32 vector loads (tail rows past n1 are masked).
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int nn = n + r;
      int32_t packed = 0;
      if (nn < n1) {
        packed = __ldg(reinterpret_cast<const int32_t *>(
            base + (int64_t)nn * C + ch));
      }
      acc[0] += static_cast<float>(static_cast<int8_t>(packed & 0xff));
      acc[1] += static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xff));
      acc[2] += static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xff));
      acc[3] += static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xff));
    }
  }
  const float k = k_s / static_cast<float>(L);
  float *mout = mean_out + ((int64_t)b * H + h) * C;
  atomicAdd(&mout[ch], acc[0] * k);
  atomicAdd(&mout[ch + 1], acc[1] * k);
  atomicAdd(&mout[ch + 2], acc[2] * k);
  atomicAdd(&mout[ch + 3], acc[3] * k);
}

// Fused Q kernel: thread map IDENTICAL to quant_q_kernel NR=4/WARPQ=32.
// RoPE rotation comes from on-the-fly sincos (pos * inv_freq[64]): this
// removes all freqs-table HBM traffic (baked [1,1,S,64,2,2] would be re-read
// per head) and any grid dependence beyond S itself. Rotation convention
// matches the baked 2x2 capture ([[cos,-sin],[sin,cos]]): y0=c*x0-s*x1.

template <typename WN>
__global__ __launch_bounds__(128, 4) void fused_q_kernel(
    const int8_t *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const float *__restrict__ q_s_in,
    const WN *__restrict__ norm_w,
    const float *__restrict__ inv_freq, const int L, const int C, const int H,
    const int q_sc_per_h, const int64_t stride_b, const int64_t stride_h) {
  constexpr int NR = 4, WARPQ = 32, BLKQ = 128, NSUB = BLKQ / WARPQ;
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H + h) * L * C;
  const int64_t sbh = ((int64_t)b * H + h) * q_sc_per_h;
  const float q_s = __ldg(q_s_in);
  float invf[4];
  {
    const int pair0 = (lane & 15) << 2;
#pragma unroll
    for (int c = 0; c < 4; ++c)
      invf[c] = inv_freq[pair0 + c];
  }

  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / NSUB) * BLKQ + (oblk % NSUB) * WARPQ + otld;

    float v[NR * 4];
    float rnorm[NR];
#pragma unroll
    for (int i = 0; i < NR * 4; ++i)
      v[i] = 0.f;

    // 1) dequant int8 rows -> fp32.
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      const int n = base + j * 8;
      if (n < L) {
        const int ch = lane << 2;
        load4_i8(&q_in[in_bh + (int64_t)n * C + ch], &v[j * 4]);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[j * 4 + c] *= q_s;
      }
    }

    // 2) RMSNorm over full D (fp32, no bf16 intermediate rounding).
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        s = fmaf(v[j * 4 + c], v[j * 4 + c], s);
      s = warp_reduce_fsum(s);
      rnorm[j] = rsqrtf(s / static_cast<float>(C) + 1e-6f);
    }
    {
      const int ch = lane << 2;
      float w[4];
#pragma unroll
      for (int c = 0; c < 4; ++c)
        w[c] = wn_to_float(norm_w, ch + c);
#pragma unroll
      for (int j = 0; j < NR; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[j * 4 + c] *= rnorm[j] * w[c];
      }
    }

    // 3) split-half RoPE: pairs (p, p+64); exchange quads with lane^16 so
    // every lane sees both halves of its 4 pairs, rotate via on-the-fly
    // sincos, keep own half. SFU-saving variants (angle-addition recurrence,
    // warp-cooperative sincos) cost more in extra regs/dependent FMAs/shfls
    // than the sincos they save: the SFU is not the bottleneck here.
    {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        float lo[4], hi[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          lo[c] = v[j * 4 + c];
          hi[c] = __shfl_xor_sync(0xffffffffu, lo[c], 16);
        }
        float x0[4], x1[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          x0[c] = (lane < 16) ? lo[c] : hi[c];
          x1[c] = (lane < 16) ? hi[c] : lo[c];
        }
        if (n < L) {
          const float pos = static_cast<float>(n);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float s, co;
            __sincosf(pos * invf[c], &s, &co);
            const float y0 = co * x0[c] - s * x1[c];
            const float y1 = s * x0[c] + co * x1[c];
            v[j * 4 + c] = (lane < 16) ? y0 : y1;
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c)
            v[j * 4 + c] = 0.f;
        }
      }
    }

    // 4) Signed H128 Hadamard (same as ROTATION=128 path).
#pragma unroll
    for (int j = 0; j < NR; ++j)
      convrot128_f(&v[j * 4]);

    // 5) absmax + quant + store (identical to process_q tail).
    // mx is unscaled here (see convrot128_f note). The STORED scale carries
    // kHadamardNorm (reconstruction must equal normalized values), while
    // quant consumes unscaled v with the matching unscaled divisor:
    // round(V*C/sc) with sc = mx*C/127. inv_sc = C/sc.
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < NR * 4; ++j)
      mx = fmaxf(mx, fabsf(v[j]));
    mx = warp_reduce_fmax(mx);
    const float sc = mx * kHadamardNorm / 127.f + 1e-7f;
    const float inv_sc = kHadamardNorm / sc;
    if (lane == 0)
      q_sb[sbh + oblk * 8 + otld] = sc;
    {
      const int ch = lane << 2;
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        if (n < L) {
          store4_i8(&q_out[out_bh + (int64_t)n * C + ch],
                    quant_int8_lim(v[j * 4], inv_sc),
                    quant_int8_lim(v[j * 4 + 1], inv_sc),
                    quant_int8_lim(v[j * 4 + 2], inv_sc),
                    quant_int8_lim(v[j * 4 + 3], inv_sc));
        }
      }
    }
  }
}

// ---- Fused K kernel: thread map IDENTICAL to quant_k_kernel NL=16/WARPK=128
//
// Register budget: the 32 rows are processed in two 16-row halves with full
// recompute (int8 reload is 8x cheaper than bf16 and keeps v[] at 64 floats,
// avoiding the local-memory spill a 128-float v[] incurs). One scale per
// (oblk, otld) is kept by reducing absmax across both halves first.

template <typename WN>
__device__ __forceinline__ void fused_k_half(
    const int8_t *__restrict__ k_in, const WN *__restrict__ norm_w,
    const float *__restrict__ mean, const float *__restrict__ inv_freq,
    float k_s_in, int j_base, int oblk, int otld, int L, int C,
    int64_t in_bh, float *v, float *rnorm) {
  constexpr int HH = 8; // j range per half (8 j x 2 p = 16 rows)
  const int lane = threadIdx.x & 31;
  float invf[4];
#pragma unroll
  for (int c = 0; c < 4; ++c)
    invf[c] = inv_freq[((lane & 15) << 2) + c];
  // smooth-K mean lives in the raw (dequantized) domain: subtract BEFORE
  // norm so norm/RoPE see mean-free values (matches Sage smooth-K).
  float mb[4];
  {
    const int chm = lane << 2;
#pragma unroll
    for (int c = 0; c < 4; ++c)
      mb[c] = mean[chm + c];
  }
#pragma unroll
  for (int jj = 0; jj < HH; ++jj) {
    const int j = j_base + jj;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = oblk * 128 + j * 8 + otld * 2 + p;
      const int vi = (jj * 2 + p) * 4;
      if (n < L) {
        const int ch = lane << 2;
        load4_i8(&k_in[in_bh + (int64_t)n * C + ch], &v[vi]);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[vi + c] = v[vi + c] * k_s_in - mb[c];
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[vi + c] = 0.f;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 2 * HH; ++r) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c)
      s = fmaf(v[r * 4 + c], v[r * 4 + c], s);
    s = warp_reduce_fsum(s);
    rnorm[r] = rsqrtf(s / static_cast<float>(C) + 1e-6f);
  }
  {
    const int ch = lane << 2;
    float w[4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
      w[c] = wn_to_float(norm_w, ch + c);
#pragma unroll
    for (int r = 0; r < 2 * HH; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
        v[r * 4 + c] *= rnorm[r] * w[c];
    }
  }
  {
#pragma unroll
    for (int r = 0; r < 2 * HH; ++r) {
      const int j = j_base + (r >> 1), p = r & 1;
      const int n = oblk * 128 + j * 8 + otld * 2 + p;
      float lo[4], hi[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        lo[c] = v[r * 4 + c];
        hi[c] = __shfl_xor_sync(0xffffffffu, lo[c], 16);
      }
      float x0[4], x1[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        x0[c] = (lane < 16) ? lo[c] : hi[c];
        x1[c] = (lane < 16) ? hi[c] : lo[c];
      }
      if (n < L) {
        const float pos = static_cast<float>(n);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float s, co;
          __sincosf(pos * invf[c], &s, &co);
          const float y0 = co * x0[c] - s * x1[c];
          const float y1 = s * x0[c] + co * x1[c];
          v[r * 4 + c] = (lane < 16) ? y0 : y1;
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[r * 4 + c] = 0.f;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 2 * HH; ++r)
    convrot128_f(&v[r * 4]);
}

template <typename WN>
__global__ __launch_bounds__(128, 4) void fused_k_kernel(
    const int8_t *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const float *__restrict__ k_s_in,
    const WN *__restrict__ norm_w, const float *__restrict__ k_mean,
    const float *__restrict__ inv_freq, const int L, const int C, const int H,
    const int k_sc_per_h, const int64_t stride_b, const int64_t stride_h) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H + h) * L * C;
  const int64_t sbh = ((int64_t)b * H + h) * k_sc_per_h;
  const float *mean = k_mean + ((int64_t)b * H + h) * C;
  const float k_s = __ldg(k_s_in);

  float v[2 * 8 * 4];
  float rnorm[2 * 8];
  // Half-staging: half-0 transform output (16 rows x 4 ch = 64 floats/thread)
  // spills to SMEM as fp16 while half-1 stays in v[]. 128 threads x 32 half2
  // __launch_bounds__(128,4) occupancy is kept. Stored as half2 [32][128]
  // (thread = column) to keep banks conflict-free-ish. fp16 rounding can
  // shift quant outputs by ~1ulp on rare elements (gate-verified); the
  // absmax/scale path still uses full-precision registers, so scales are
  // unaffected. Pass B then quantizes WITHOUT recomputing either half:
  // half-1 straight from v[], half-0 reloaded from SMEM (order swapped,
  // stores are independent).
  __shared__ __half2 smem_h0[32][128];
  float mx = 0.f;
  // Pass A: transform both halves; stage half-0 to SMEM, keep half-1 in v[].
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    fused_k_half<WN>(k_in, norm_w, mean, inv_freq, k_s, half * 8, oblk,
                     otld, L, C, in_bh, v, rnorm);
#pragma unroll
    for (int j = 0; j < 2 * 8 * 4; ++j)
      mx = fmaxf(mx, fabsf(v[j]));
    if (half == 0) {
#pragma unroll
      for (int j = 0; j < 32; ++j)
        smem_h0[j][threadIdx.x] =
            __float22half2_rn(make_float2(v[j * 2], v[j * 2 + 1]));
    }
  }
  mx = warp_reduce_fmax(mx);
  const float sc = mx * kHadamardNorm / 127.f + 1e-7f;
  const float inv_sc = kHadamardNorm / sc;
  if (lane == 0)
    k_sb[sbh + oblk * 4 + otld] = sc;
  // Pass B: half-1 from registers first, then half-0 reloaded from SMEM.
  {
    const int ch = lane << 2;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * 128 + (8 + jj) * 8 + otld * 2 + p;
        const int vi = (jj * 2 + p) * 4;
        if (n < L) {
          store4_i8(&k_out[out_bh + (int64_t)n * C + ch],
                    quant_int8_lim(v[vi], inv_sc),
                    quant_int8_lim(v[vi + 1], inv_sc),
                    quant_int8_lim(v[vi + 2], inv_sc),
                    quant_int8_lim(v[vi + 3], inv_sc));
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const float2 f = __half22float2(smem_h0[j][threadIdx.x]);
      v[j * 2] = f.x;
      v[j * 2 + 1] = f.y;
    }
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * 128 + (0 + jj) * 8 + otld * 2 + p;
        const int vi = (jj * 2 + p) * 4;
        if (n < L) {
          store4_i8(&k_out[out_bh + (int64_t)n * C + ch],
                    quant_int8_lim(v[vi], inv_sc),
                    quant_int8_lim(v[vi + 1], inv_sc),
                    quant_int8_lim(v[vi + 2], inv_sc),
                    quant_int8_lim(v[vi + 3], inv_sc));
        }
      }
    }
  }
}

// ---- V requant: int8 -> int8 (folds per-tensor s_in into block scale) ----
//
// Exact: v_fp = v_i8 * s_in; per-channel absmax mx_fp = s_in * mx_i8, so
// scale_out = mx_fp / 127 = s_in * (mx_i8 / 127) and the stored int8 values
// equal quantizing v_i8 with scale (mx_i8 / 127) — s_in cancels. Single pass
// over int8, no fp16/bf16 HBM roundtrip.
// Layout contract identical to quant_v_int8 (INT8 MMA 16-permutation).

constexpr int kRequantDTile = 8;

__device__ __forceinline__ int requant_inv_perm16(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) |
         (((w >> 2) & 1) << 3);
}

__device__ __forceinline__ void load_tile_i8(const int8_t *ptr, float *out) {
  const int32_t p0 = __ldg(reinterpret_cast<const int32_t *>(ptr));
  const int32_t p1 = __ldg(reinterpret_cast<const int32_t *>(ptr + 4));
  out[0] = static_cast<float>(static_cast<int8_t>(p0 & 0xff));
  out[1] = static_cast<float>(static_cast<int8_t>((p0 >> 8) & 0xff));
  out[2] = static_cast<float>(static_cast<int8_t>((p0 >> 16) & 0xff));
  out[3] = static_cast<float>(static_cast<int8_t>((p0 >> 24) & 0xff));
  out[4] = static_cast<float>(static_cast<int8_t>(p1 & 0xff));
  out[5] = static_cast<float>(static_cast<int8_t>((p1 >> 8) & 0xff));
  out[6] = static_cast<float>(static_cast<int8_t>((p1 >> 16) & 0xff));
  out[7] = static_cast<float>(static_cast<int8_t>((p1 >> 24) & 0xff));
}

template <int THREADS, int MAXR>
__global__ void requant_v_int8_cached(const int8_t *__restrict__ v,
                                      const float *__restrict__ s_in,
                                      int8_t *__restrict__ out,
                                      float *__restrict__ scale_out, int N,
                                      int padded_N, int H, int D, int64_t sb,
                                      int64_t sh, int64_t sn) {
  // Register-cached variant: pass-1 values stay in registers, so pass 2
  // quantizes WITHOUT re-reading V from HBM (saves one full N*D read). Since V is int8, rows are kept as packed int32 pairs
  // (2 regs/row, not 8 float regs/row), so even 36 rows fit easily; the
  // absmax runs on integers and converts once. Caller guarantees
  // ceil((N-tid)/THREADS) <= MAXR for every tid. Bit-identical to the
  // re-reading kernel below (same ints, same order, same stores).
  const int d_tiles = D / kRequantDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kRequantDTile;
  const int8_t *base = v + b * sb + h * sh + d0;
  constexpr int WARPS = THREADS / 32;

  int mxi[kRequantDTile];
#pragma unroll
  for (int i = 0; i < kRequantDTile; ++i)
    mxi[i] = 0;
  // Constant-bound loops (r < MAXR) so kept[][] stays in registers;
  // the dispatch guarantees no thread needs more than MAXR rows.
  int32_t kept[MAXR][2];
#pragma unroll
  for (int r = 0; r < MAXR; ++r) {
    const int n = threadIdx.x + r * THREADS;
    if (n < N) {
      const int32_t p0 =
          __ldg(reinterpret_cast<const int32_t *>(base + (int64_t)n * sn));
      const int32_t p1 = __ldg(
          reinterpret_cast<const int32_t *>(base + (int64_t)n * sn + 4));
      kept[r][0] = p0;
      kept[r][1] = p1;
#pragma unroll
      for (int k = 0; k < kRequantDTile; ++k) {
        const int32_t pk = (k < 4) ? p0 : p1;
        const int sh = (k & 3) * 8;
        // Extract byte k with sign extension: shift it to the top, then
        // arithmetic-shift back by 24 (NOT 24-sh: that keeps 16/24 bits).
        const int bv = (pk << (24 - sh)) >> 24;
        const int av = bv < 0 ? -bv : bv;
        if (av > mxi[k])
          mxi[k] = av;
      }
    }
  }
  float mx[kRequantDTile];
#pragma unroll
  for (int i = 0; i < kRequantDTile; ++i)
    mx[i] = static_cast<float>(mxi[i]);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int di = 0; di < kRequantDTile; ++di)
    mx[di] = warp_reduce_fmax(mx[di]);

  __shared__ float warp_mx[kRequantDTile][WARPS];
  __shared__ float inv_sc_sh[kRequantDTile];
  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di)
      warp_mx[di][warp] = mx[di];
  }
  __syncthreads();

  const float v_s = __ldg(s_in);
  if (threadIdx.x < kRequantDTile) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
      val = fmaxf(val, warp_mx[threadIdx.x][w]);
    float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
    scale_out[(b * H + h) * D + d0 + threadIdx.x] = sc * v_s;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
  }
  __syncthreads();

  __syncthreads();

  // inv_sc reloaded from SMEM per use (no reg array): the di loop is
  // unrolled, so each load is a constant-index broadcast.
  const int64_t out_row = static_cast<int64_t>((b * H + h) * D + d0);
#pragma unroll
  for (int r = 0; r < MAXR; ++r) {
    const int src = threadIdx.x + r * THREADS;
    if (src >= N)
      continue;
    const int w = src & 15;
    const int dst = (src & ~15) | requant_inv_perm16(w);
    // Unpack straight from the retained packs (no tmp[] array: saves regs).
    const int32_t p0 = kept[r][0], p1 = kept[r][1];
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di) {
      const int32_t pk = (di < 4) ? p0 : p1;
      const int sh = (di & 3) * 8;
      const float vf =
          static_cast<float>(static_cast<int8_t>((pk >> sh) & 0xff));
      out[(out_row + di) * padded_N + dst] =
          quant_int8_lim(vf, inv_sc_sh[di]);
    }
  }
  for (int src = N + threadIdx.x; src < padded_N; src += THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | requant_inv_perm16(w);
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di)
      out[(out_row + di) * padded_N + dst] = 0;
  }
}

template <int THREADS>
__global__ void requant_v_int8_kernel(const int8_t *__restrict__ v,
                                      const float *__restrict__ s_in,
                                      int8_t *__restrict__ out,
                                      float *__restrict__ scale_out, int N,
                                      int padded_N, int H, int D, int64_t sb,
                                      int64_t sh, int64_t sn) {
  const int d_tiles = D / kRequantDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kRequantDTile;
  const int8_t *base = v + b * sb + h * sh + d0;
  constexpr int WARPS = THREADS / 32;

  float mx[kRequantDTile];
#pragma unroll
  for (int i = 0; i < kRequantDTile; ++i)
    mx[i] = 0.f;
  for (int n = threadIdx.x; n < N; n += THREADS) {
    float tmp[kRequantDTile];
    load_tile_i8(base + (int64_t)n * sn, tmp);
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di)
      mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int di = 0; di < kRequantDTile; ++di)
    mx[di] = warp_reduce_fmax(mx[di]);

  __shared__ float warp_mx[kRequantDTile][WARPS];
  __shared__ float inv_sc_sh[kRequantDTile];
  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di)
      warp_mx[di][warp] = mx[di];
  }
  __syncthreads();

  const float v_s = __ldg(s_in);
  if (threadIdx.x < kRequantDTile) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
      val = fmaxf(val, warp_mx[threadIdx.x][w]);
    float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
    scale_out[(b * H + h) * D + d0 + threadIdx.x] = sc * v_s;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
  }
  __syncthreads();

  float inv_sc[kRequantDTile];
#pragma unroll
  for (int di = 0; di < kRequantDTile; ++di)
    inv_sc[di] = inv_sc_sh[di];

  const int64_t out_row = static_cast<int64_t>((b * H + h) * D + d0);
  // Scalar path stands: a cooperative SMEM transpose costs extra
  // __syncthreads/SMEM traffic while L2 already combines the scattered
  // byte stores.
  for (int src = N - 1 - threadIdx.x; src >= 0; src -= THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | requant_inv_perm16(w);
    float tmp[kRequantDTile];
    load_tile_i8(base + (int64_t)src * sn, tmp);
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di) {
      out[(out_row + di) * padded_N + dst] =
          quant_int8_lim(tmp[di], inv_sc[di]);
    }
  }
  for (int src = N + threadIdx.x; src < padded_N; src += THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | requant_inv_perm16(w);
#pragma unroll
    for (int di = 0; di < kRequantDTile; ++di)
      out[(out_row + di) * padded_N + dst] = 0;
  }
}

} // namespace

extern "C" {

void launch_fused_qk_rope_int8(
    const void *q_i8, const void *q_s_in, void *q_int8, void *q_scale,
    const void *k_i8, const void *k_s_in, void *k_int8, void *k_scale,
    const void *rms_w_q, const void *rms_w_k, int norm_dtype_code,
    const void *inv_freq, int B, int H, int L, int C, void *ws_mean,
    cudaStream_t stream) {
  if (C != 128)
    throw std::runtime_error(
        "fused_qk_rope_int8: head_dim 128 only");
  if (L <= 1024)
    throw std::runtime_error(
        "fused_qk_rope_int8: requires L>1024 (ROTATION=128 path)");
  if (!ws_mean)
    throw std::runtime_error("fused_qk_rope_int8: mean workspace required");
  if (reinterpret_cast<uintptr_t>(q_i8) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(k_i8) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(q_int8) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(k_int8) % 4 != 0)
    throw std::runtime_error("fused_qk_rope_int8: int8 pointers need 4B align");

  const int64_t stride_b = (int64_t)H * L * C;
  const int64_t stride_h = (int64_t)L * C;
  const int q_oblk = (L + 128 - 1) / 128 * 4;
  const int k_oblk = (L + 128 - 1) / 128 * 1;
  const int q_sc_per_h = q_oblk * 8;
  const int k_sc_per_h = k_oblk * 4;

  // K mean pre-pass over dequantized K (smooth-K). Chunked + atomicAdd.
  // Stream overlap (k_mean+requant on a side stream, join via events) stays
  // unwritten: sharing events across sequential enqueues is unsound (a
  // lagging main stream can observe a later call's record, then read
  // workspace already overwritten by that call), and per-call events cannot
  // be safely destroyed (must outlive the wait's execution). Sound overlap
  // needs full graph capture; sequential stands.
  {
    dim3 g((L + KMEAN_CHUNK_ROWS - 1) / KMEAN_CHUNK_ROWS, H, B);
    cudaError_t e = cudaMemsetAsync(ws_mean, 0, (size_t)B * H * C * 4, stream);
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("k_mean memset failed: ") +
                               cudaGetErrorString(e));
    k_mean_kernel<<<g, 128, 0, stream>>>((const int8_t *)k_i8,
                                         (float *)ws_mean,
                                         (const float *)k_s_in, L, C, H,
                                         stride_b, stride_h);
    e = cudaGetLastError();
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("k_mean kernel launch failed: ") +
                               cudaGetErrorString(e));
  }

  auto launch_qk = [&](auto dummy_wn) {
    using WN = decltype(dummy_wn);
    dim3 gq(q_oblk, H, B);
    fused_q_kernel<WN><<<gq, 128, 0, stream>>>(
        (const int8_t *)q_i8, (int8_t *)q_int8, (float *)q_scale,
        (const float *)q_s_in, (const WN *)rms_w_q, (const float *)inv_freq,
        L, C, H, q_sc_per_h, stride_b, stride_h);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("fused_q kernel launch failed: ") +
                               cudaGetErrorString(e));
    dim3 gk(k_oblk, H, B);
    fused_k_kernel<WN><<<gk, 128, 0, stream>>>(
        (const int8_t *)k_i8, (int8_t *)k_int8, (float *)k_scale,
        (const float *)k_s_in, (const WN *)rms_w_k, (const float *)ws_mean,
        (const float *)inv_freq, L, C, H, k_sc_per_h, stride_b, stride_h);
    e = cudaGetLastError();
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("fused_k kernel launch failed: ") +
                               cudaGetErrorString(e));
  };

  if (norm_dtype_code == 0)
    launch_qk(float{});
  else if (norm_dtype_code == 1)
    launch_qk(half{});
  else
    launch_qk(nv_bfloat16{});
}

void launch_requant_v_int8(const void *v_i8, const void *v_s_in, void *out,
                           void *scale, int B, int H, int N, int D,
                           int padded_N, int64_t sb, int64_t sh, int64_t sn,
                           cudaStream_t stream) {
  if (D <= 0 || D % kRequantDTile != 0)
    throw std::runtime_error("requant_v_int8: head_dim must be multiple of 8");
  if (reinterpret_cast<uintptr_t>(v_i8) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(out) % 4 != 0)
    throw std::runtime_error("requant_v_int8: int8 pointers need 4B align");
  const int blocks = B * H * (D / kRequantDTile);
  // Register-cached dispatch: rows-per-thread capped so the kept[][]
  // buffer fits the template; larger shapes use the re-read kernel.
  const int T = (N <= 256) ? 128 : 512;
  const int rows = (N + T - 1) / T;
  auto launch_cached = [&](auto maxr) {
    constexpr int MAXR = decltype(maxr)::value;
    if (T == 128) {
      requant_v_int8_cached<128, MAXR><<<blocks, 128, 0, stream>>>(
          static_cast<const int8_t *>(v_i8), static_cast<const float *>(v_s_in),
          static_cast<int8_t *>(out), static_cast<float *>(scale), N, padded_N,
          H, D, sb, sh, sn);
    } else {
      requant_v_int8_cached<512, MAXR><<<blocks, 512, 0, stream>>>(
          static_cast<const int8_t *>(v_i8), static_cast<const float *>(v_s_in),
          static_cast<int8_t *>(out), static_cast<float *>(scale), N, padded_N,
          H, D, sb, sh, sn);
    }
  };
  if (rows <= 2) {
    launch_cached(std::integral_constant<int, 2>{});
  } else if (rows <= 8) {
    launch_cached(std::integral_constant<int, 8>{});
  } else if (rows <= 20) {
    // int8-packed retention (2 regs/row): larger MAXR exceeds the register
    // budget at 512 threads; 20 fits and covers N<=10240 (incl. S=9216).
    // Larger shapes keep the re-read kernel.
    launch_cached(std::integral_constant<int, 20>{});
  } else if (N <= 256) {
    requant_v_int8_kernel<128><<<blocks, 128, 0, stream>>>(
        static_cast<const int8_t *>(v_i8), static_cast<const float *>(v_s_in),
        static_cast<int8_t *>(out), static_cast<float *>(scale), N, padded_N,
        H, D, sb, sh, sn);
  } else {
    requant_v_int8_kernel<512><<<blocks, 512, 0, stream>>>(
        static_cast<const int8_t *>(v_i8), static_cast<const float *>(v_s_in),
        static_cast<int8_t *>(out), static_cast<float *>(scale), N, padded_N,
        H, D, sb, sh, sn);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("requant_v kernel launch failed: ") +
                             cudaGetErrorString(error));
}

} // extern "C"
