/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 * bias (and the residual rscale) are read in the OUTPUT dtype and converted
 * to float in-register, so callers never cast them.
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and selected with a shape heuristic
 * fitted from sustained Ada and Blackwell benchmarks.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass_gemm_common.cuh"

namespace {
using namespace cute;
using comfy_cutlass::device_sm_count;
using comfy_cutlass::device_l2_bytes;
// Bands of 32 N tiles: a 154 MB weight at 32k rows reads 0.9 GB from DRAM instead of 26 GB.
// An L2 too small to hold a band narrows them, see stream_k_band_tiles.
using ThreadblockSwizzleBandedStreamK = comfy_cutlass::ThreadblockSwizzleLeanStreamKT<32>;

template <typename ThreadMap, bool Scalar>
struct WeightScaleBroadcast;

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

// DIT-PLUGINS-PATCH: BHSD/QKV store. Writes each GEMM element once, with no
// transpose kernel and no intermediate BSC tensor. q/k/v are separate because
// the three views are not contiguous in the BHSD layout.
template <class ThreadMap, class Element>
struct VisitorQkvStore {
    struct Arguments {
        Element* q = nullptr;
        Element* k = nullptr;
        Element* v = nullptr;
        int64_t S = 0;
        int64_t C = 0;  // N per q/k/v (H*D)
        int H = 0;
        int D = 0;
        int64_t N = 0;
    };
    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args,
                                                    void*) {
        return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
    struct SharedStorage {};

    static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;
    static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

    CUTLASS_HOST_DEVICE VisitorQkvStore() {}
    CUTLASS_HOST_DEVICE VisitorQkvStore(Params const& params, SharedStorage const&)
        : params_ptr(&params) {}
    Params const* params_ptr;

    template <class GTensor, class RTensor, class CTensor, class ProblemShape>
    struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
        CUTLASS_DEVICE
        Callbacks(GTensor&& tC_g, RTensor&& tC_r, CTensor&& tC_c, ProblemShape problem_shape,
                  Params const* params_ptr)
            : tC_gAux(cute::forward<GTensor>(tC_g)),
              tC_rAux(cute::forward<RTensor>(tC_r)),
              tC_cAux(cute::forward<CTensor>(tC_c)),
              problem_shape(problem_shape),
              params_ptr(params_ptr) {}

        GTensor tC_gAux;
        RTensor tC_rAux;
        CTensor tC_cAux;
        ProblemShape problem_shape;
        Params const* params_ptr;

        CUTLASS_DEVICE void begin_step(int step_idx) { clear(tC_rAux); }

        template <class ElementAccumulator, class ElementInput, int FragmentSize>
        CUTLASS_DEVICE auto visit(int, int, int, int frg_idx,
                                  cutlass::Array<ElementAccumulator, FragmentSize> const&,
                                  cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
            using ConvertInput =
                cutlass::NumericArrayConverter<Element, ElementInput, FragmentSize,
                                               cutlass::FloatRoundStyle::round_to_nearest>;
            ConvertInput convert_input{};
            Tensor tC_rAux_frg =
                recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
            tC_rAux_frg(frg_idx) = convert_input(frg_input);
            return frg_input;
        }

        CUTLASS_DEVICE void end_step(int step_idx) {
            auto src_v = filter(tC_rAux);
            auto coord_v = filter(tC_cAux(_, _, _, step_idx));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(src_v); ++i) {
                auto coord = coord_v(i);
                const int m = int(get<0>(coord));
                const int n = int(get<1>(coord));
                const bool guard = elem_less(coord, problem_shape);
                const int b = m / int(params_ptr->S);
                const int s = m % int(params_ptr->S);
                const int qkv = n / int(params_ptr->C);
                const int c = n % int(params_ptr->C);
                const int h = c / params_ptr->D;
                const int d = c % params_ptr->D;
                Element* dst = qkv == 0 ? params_ptr->q
                              : (qkv == 1 ? params_ptr->k : params_ptr->v);
                if (dst == nullptr) continue;
                void* addr = dst +
                    ((int64_t(b) * params_ptr->H + h) * params_ptr->S + s) *
                        params_ptr->D + d;
                cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), addr, guard);
            }
        }
    };

    template <class ProblemShape>
    CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset,
                                      int thread_idx, ProblemShape problem_shape) {
        Tensor mAux = make_tensor(
            make_gmem_ptr(params_ptr->q), problem_shape,
            make_stride(int64_t(params_ptr->N), _1{}, int64_t(get<0>(problem_shape)) *
                                                       params_ptr->N));
        Tensor tC_gAux = recast<VecType>(
            group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
        Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));
        Tensor cAux = make_identity_tensor(mAux.shape());
        Tensor tC_cAux = outer_partition(
            group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
            Shape<Int<VecLength>>{}, (_0{}));
        return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
            cute::move(tC_gAux), cute::move(tC_rAux), cute::move(tC_cAux), problem_shape,
            params_ptr);
    }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
// bias is read in ElementOutput (nullptr broadcasts 0).
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const ElementOutput* bias, ElementOutput* D, int M, int N, int K,
                            int output_stride, cudaStream_t stream) {
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 weight_scale_args, {} },
              {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};

// FusedInt8Gemm with the BHSD/QKV store.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmQkv {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreQkv = VisitorQkvStore<ThreadMap, ElementOutput>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreQkv, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementOutput* bias, ElementOutput* q, ElementOutput* k,
                    ElementOutput* v, int M, int N, int K, int S, int H, int D,
                    cudaStream_t stream) {
        const auto weight_scale_args =
            WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                weight_scale_args, {} },
              {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
            {q, k, v, int64_t(S), int64_t(H) * D, H, D, int64_t(N)}};
        return comfy_cutlass::launch_universal<Gemm>(A, B, cb, M, N, K, stream);
    }
};

// DIT-PLUGINS-PATCH: exact erf GELU / SiLU fused into the dequant epilogue.
// `cutlass::epilogue::thread::GELU` is the erf form (not the tanh approx).
template <class T>
struct SiluFn;

template <>
struct SiluFn<float> {
    CUTLASS_HOST_DEVICE float operator()(float const& x) const {
        return x / (1.f + expf(-x));
    }
};

template <class T, int N>
struct SiluFn<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        cutlass::Array<T, N> y;
        SiluFn<T> op;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) y[i] = op(x[i]);
        return y;
    }
};

// FusedInt8Gemm with a unary activation between bias and the store.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK,
          int NumStages, template <class> class ActOp, typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmAct {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using Act  = cutlass::epilogue::threadblock::VisitorCompute<ActOp, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT3 = cutlass::epilogue::threadblock::Sm80EVT<Act, EVT2>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT3>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementOutput* bias, ElementOutput* D, int M, int N, int K,
                    int output_stride, cudaStream_t stream) {
        const auto weight_scale_args =
            WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                  weight_scale_args, {} },
                {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
              {} },
            {D, {output_stride, _1{}, int64_t(M) * output_stride}} };
        return comfy_cutlass::launch_universal<Gemm>(A, B, cb, M, N, K, stream);
    }
};

// DIT-PLUGINS-PATCH: fused residual + gate + activation epilogue:
//   D = residual + gate[n] * act(acc * xs * ws + bias)
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK,
          int NumStages, template <class> class ActOp, typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmActResidualGate {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Gate   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Resid  = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, ElementOutput, cute::Stride<int64_t, _1, int64_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using Act  = cutlass::epilogue::threadblock::VisitorCompute<ActOp, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT3 = cutlass::epilogue::threadblock::Sm80EVT<Act, EVT2>;
    using Mul4 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT4 = cutlass::epilogue::threadblock::Sm80EVT<Mul4, EVT3, Gate>;
    using Add5 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT5 = cutlass::epilogue::threadblock::Sm80EVT<Add5, EVT4, Resid>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT5>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementOutput* bias, const ElementOutput* gate,
                    const ElementOutput* resid, ElementOutput* D, int M, int N, int K,
                    cudaStream_t stream) {
        const auto weight_scale_args =
            WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { { { { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                      weight_scale_args, {} },
                    {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
                  {} },
                {const_cast<ElementOutput*>(gate), ElementOutput(0), {_0{}, _1{}, N}}, {} },
              {const_cast<ElementOutput*>(resid), ElementOutput(0), {int64_t(N), _1{}, int64_t(M) * N}}, {} },
            {D, {int64_t(N), _1{}, int64_t(M) * N}} };
        return comfy_cutlass::launch_universal<Gemm>(A, B, cb, M, N, K, stream);
    }
};

// FusedInt8Gemm with the pre-norm block's addcmul in the epilogue:
//   D = residual + rscale[n] * (acc * xs * ws + bias); rscale and residual in the output dtype.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmResidual {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using RScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Resid  = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, ElementOutput, cute::Stride<int64_t, _1, int64_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    // bias add rounds to ElementOutput like the plain kernel; only the addcmul stays fp32
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using Mul3 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT3 = cutlass::epilogue::threadblock::Sm80EVT<Mul3, EVT2, RScale>;
    using Add4 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT4 = cutlass::epilogue::threadblock::Sm80EVT<Add4, EVT3, Resid>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT4>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementOutput* bias, const ElementOutput* rscale, const ElementOutput* resid,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                     weight_scale_args, {} },
                  {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
                {const_cast<ElementOutput*>(rscale), ElementOutput(0), {_0{}, _1{}, N}}, {} },
              {const_cast<ElementOutput*>(resid), ElementOutput(0), {int64_t(N), _1{}, int64_t(M) * N}}, {} },
            {D, {int64_t(N), _1{}, int64_t(M) * N}} };
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};

int select_fused_int8_config(int m, int n, int k) {
    if (k % 16 != 0) return 9;

    const int64_t mn = int64_t(m) * n;
    if (n <= 24832) {
        if (mn <= 1477632) {
            if (mn <= 259072) return k <= 7296 ? 6 : 12;
            return int64_t(n) * k <= 16252928 ? 2 : 12;
        }
        if (mn <= 4193792) {
            return int64_t(n) * k <= 5275648 ? 1 : 12;
        }
        return int64_t(m) * k <= int64_t(n) * 5675 ? 0 : 13;
    }
    if (int64_t(n) * k <= int64_t(m) * 11096) return 0;
    return mn <= 108003328 ? 0 : 13;
}

// The tree ignores wave quantization: at decoder-tile M (~2k rows) its 128x256
// pick can leave a 2.1-wave grid where 128x128 wins despite ~8% lower per-tile
// throughput. Between those two, take the smaller wave-rounding x tile-cost.
int wave_guard(int m, int n, int selected) {
    if (selected != 0 && selected != 1) return selected;
    const int sms = device_sm_count();
    auto cost = [&](int64_t tile_n, double per_tile) {
        const double waves = double(((m + 127) / 128) * ((n + tile_n - 1) / tile_n)) / sms;
        return std::ceil(waves) / waves * per_tile;
    };
    return cost(256, 1.0) <= cost(128, 1.08) ? 0 : 1;
}

int choose_fused_int8_config(int m, int n, int k) {
    const int selected = wave_guard(m, n, select_fused_int8_config(m, n, k));
    if (selected != 0 && selected != 1) return selected;
    // The identity order re-reads the activation per weight column once it outgrows L2;
    // banded stream-K does not, but only bands when M has at least as many tiles as N.
    // Measured crossovers: 66-88 MB at 96 MB L2, 42-63 MB at 36 MB.
    const int64_t reroute_bytes = std::max<int64_t>(int64_t(device_l2_bytes()) * 3 / 4, int64_t(48) << 20);
    const bool bands = (m + 127) / 128 >= (n + 255) / 256;
    return (bands && int64_t(m) * k > reroute_bytes) ? 13 : selected;
}

template <typename Launch>
bool launch_fused_int8_heuristic(int m, int n, int k, Launch launch) {
    const int selected = choose_fused_int8_config(m, n, k);
    if (launch(selected)) return true;

    static constexpr int aligned_fallbacks[] = {2, 12, 0, 13, 1, 6, 8, 7, 3, 4, 5};
    static constexpr int low_alignment_fallbacks[] = {9, 10, 11};
    if (k % 16 == 0) {
        for (int config : aligned_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    } else {
        for (int config : low_alignment_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    }
    return false;
}

// The tile table; select_fused_int8_config and the fallback orders index it, and
// every epilogue variant's runner table is instantiated from it.
using comfy_cutlass::TileConfig;
using comfy_cutlass::ConfigList;
using FusedInt8Configs = ConfigList<
    TileConfig<128, 256,  64, 64, 64,  64, 3>,                                       // 0
    TileConfig<128, 128,  64, 64, 64,  64, 4>,                                       // 1
    TileConfig< 64, 128,  64, 32, 64,  64, 4>,                                       // 2
    TileConfig< 64, 256,  64, 32, 64,  64, 3>,                                       // 3
    TileConfig< 32, 256,  64, 32, 64,  64, 4>,                                       // 4
    TileConfig< 32, 128,  64, 32, 64,  64, 4>,                                       // 5
    TileConfig< 16, 128,  64, 16, 64,  64, 4>,                                       // 6
    TileConfig< 64, 128, 128, 32, 64, 128, 3>,                                       // 7
    TileConfig<128,  64, 128, 64, 32, 128, 3>,                                       // 8
    TileConfig< 64, 128,  64, 32, 64,  64, 4, 8>,                                    // 9  (K % 8 alignment)
    TileConfig< 32, 128,  64, 32, 64,  64, 4, 8>,                                    // 10
    TileConfig< 16, 128,  64, 16, 64,  64, 4, 8>,                                    // 11
    TileConfig<128, 128,  64, 64, 64,  64, 4, 16, ThreadblockSwizzleBandedStreamK>,  // 12 (stream-K)
    TileConfig<128, 256,  64, 64, 64,  64, 3, 16, ThreadblockSwizzleBandedStreamK>>; // 13
constexpr int kFusedConfigCount = FusedInt8Configs::size;

template <typename OutT, typename C>
using PlainGemm = FusedInt8Gemm<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK, C::NumStages,
                                cutlass::arch::Sm80, false, C::AlignmentAB,
                                typename C::ThreadblockSwizzle>;
template <typename OutT, typename C>
using ResidualGemm = FusedInt8GemmResidual<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK,
                                           C::NumStages, cutlass::arch::Sm80, false,
                                           C::AlignmentAB, typename C::ThreadblockSwizzle>;
template <typename OutT, typename C>
using QkvGemm = FusedInt8GemmQkv<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK,
                                 C::NumStages, cutlass::arch::Sm80, false,
                                 C::AlignmentAB, typename C::ThreadblockSwizzle>;
template <typename OutT, typename C, template <class> class ActOp>
using ActGemm = FusedInt8GemmAct<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK,
                                 C::NumStages, ActOp, cutlass::arch::Sm80, false,
                                 C::AlignmentAB, typename C::ThreadblockSwizzle>;
template <typename OutT, typename C, template <class> class ActOp>
using ActResidualGemm = FusedInt8GemmActResidualGate<
    OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK, C::NumStages, ActOp,
    cutlass::arch::Sm80, false, C::AlignmentAB, typename C::ThreadblockSwizzle>;

// One run_strided table (stride == N is the plain case) serves every entry
// point; bias may be nullptr (the RowBroadcast visitor broadcasts 0).
template <typename OutT>
using FusedFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                         const OutT*, OutT*, int, int, int, int, cudaStream_t);
template <typename OutT>
using FusedResidualFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                                 const OutT*, const OutT*, const OutT*, OutT*, int, int, int,
                                 cudaStream_t);
template <typename OutT>
using FusedQkvFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                            const OutT*, OutT*, OutT*, OutT*, int, int, int, int, int, int,
                            cudaStream_t);
template <typename OutT>
using FusedActFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                            const OutT*, OutT*, int, int, int, int, cudaStream_t);
template <typename OutT>
using FusedActResidualFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                                    const OutT*, const OutT*, const OutT*, OutT*, int, int,
                                    int, cudaStream_t);

template <typename OutT, typename... Cs>
const FusedFn<OutT>* make_fused_runners(ConfigList<Cs...>) {
    static const FusedFn<OutT> runners[sizeof...(Cs)] = {&PlainGemm<OutT, Cs>::run_strided...};
    return runners;
}

template <typename OutT, typename... Cs>
const FusedResidualFn<OutT>* make_fused_residual_runners(ConfigList<Cs...>) {
    static const FusedResidualFn<OutT> runners[sizeof...(Cs)] = {&ResidualGemm<OutT, Cs>::run...};
    return runners;
}

template <typename OutT, template <class> class ActOp, typename... Cs>
const FusedActResidualFn<OutT>* make_fused_act_residual_runners(ConfigList<Cs...>) {
    static const FusedActResidualFn<OutT> runners[sizeof...(Cs)] = {
        &ActResidualGemm<OutT, Cs, ActOp>::run...};
    return runners;
}

template <typename OutT>
bool dispatch_fused_act_residual(const int8_t* A, const int8_t* B, const float* xs,
                                 const float* ws, const OutT* bias, const OutT* gate,
                                 const OutT* resid, OutT* D, int M, int N, int K, int epilogue,
                                 int config, cudaStream_t stream) {
    const FusedActResidualFn<OutT>* runners =
        epilogue == 1
            ? make_fused_act_residual_runners<OutT, cutlass::epilogue::thread::GELU>(
                  FusedInt8Configs{})
            : make_fused_act_residual_runners<OutT, SiluFn>(FusedInt8Configs{});
    if (config >= 0) {
        if (config >= kFusedConfigCount) return false;
        return runners[config](A, B, xs, ws, bias, gate, resid, D, M, N, K, stream);
    }
    return launch_fused_int8_heuristic(M, N, K, [&](int c) {
        return runners[c](A, B, xs, ws, bias, gate, resid, D, M, N, K, stream);
    });
}

template <typename OutT, template <class> class ActOp, typename... Cs>
const FusedActFn<OutT>* make_fused_act_runners(ConfigList<Cs...>) {
    static const FusedActFn<OutT> runners[sizeof...(Cs)] = {&ActGemm<OutT, Cs, ActOp>::run...};
    return runners;
}

template <typename OutT>
bool dispatch_fused_act(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                        const OutT* bias, OutT* D, int M, int N, int K, int output_stride,
                        int epilogue, int config, cudaStream_t stream) {
    const FusedActFn<OutT>* runners =
        epilogue == 1 ? make_fused_act_runners<OutT, cutlass::epilogue::thread::GELU>(
                            FusedInt8Configs{})
                      : make_fused_act_runners<OutT, SiluFn>(FusedInt8Configs{});
    if (config >= 0) {
        if (config >= kFusedConfigCount) return false;
        return runners[config](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
    }
    return launch_fused_int8_heuristic(M, N, K, [&](int c) {
        return runners[c](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
    });
}

template <typename OutT, typename... Cs>
const FusedQkvFn<OutT>* make_fused_qkv_runners(ConfigList<Cs...>) {
    static const FusedQkvFn<OutT> runners[sizeof...(Cs)] = {&QkvGemm<OutT, Cs>::run...};
    return runners;
}

template <typename OutT>
bool dispatch_fused_qkv(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                        const OutT* bias, OutT* q, OutT* k, OutT* v, int M, int N, int K,
                        int S, int H, int D, int config, cudaStream_t stream) {
    const FusedQkvFn<OutT>* runners = make_fused_qkv_runners<OutT>(FusedInt8Configs{});
    if (config >= 0) {
        if (config >= kFusedConfigCount) return false;
        return runners[config](A, B, xs, ws, bias, q, k, v, M, N, K, S, H, D, stream);
    }
    return launch_fused_int8_heuristic(M, N, K, [&](int c) {
        return runners[c](A, B, xs, ws, bias, q, k, v, M, N, K, S, H, D, stream);
    });
}

template <typename OutT>
const FusedFn<OutT>* fused_runners() {
    return make_fused_runners<OutT>(FusedInt8Configs{});
}

template <typename OutT>
bool dispatch_fused_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const OutT* bias, OutT* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
    const FusedFn<OutT>* runners = fused_runners<OutT>();
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
    });
}

template <typename OutT>
bool dispatch_fused(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const OutT* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    return dispatch_fused_strided<OutT>(A, B, xs, ws, bias, D, M, N, K, N, stream);
}

template <typename OutT>
// DIT-PLUGINS-PATCH: bias and every output dtype, matching the heuristic entry.
bool dispatch_fused_config(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                           const OutT* bias, OutT* D, int M, int N, int K, int config,
                           cudaStream_t stream) {
    if (config < 0 || config >= kFusedConfigCount) return false;
    return fused_runners<OutT>()[config](A, B, xs, ws, bias, D, M, N, K, N, stream);
}

template <typename OutT>
bool dispatch_fused_residual(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                             const OutT* bias, const OutT* rscale, const OutT* resid,
                             OutT* D, int M, int N, int K, cudaStream_t stream) {
    const FusedResidualFn<OutT>* runners = make_fused_residual_runners<OutT>(FusedInt8Configs{});
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, bias, rscale, resid, D, M, N, K, stream);
    });
}

}  // namespace

extern "C" {
int cutlass_int8_selected_config(int64_t M, int64_t N, int64_t K) {
    return choose_fused_int8_config(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

bool launch_cutlass_int8_dequant_residual(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    const void* rscale, const void* resid, void* D, int64_t M, int64_t N, int64_t K,
    int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0) return true;
    // Declining K == 0 keeps the caller's eager residual+bias fallback correct.
    if (K == 0) return false;
    // bias may be nullptr: the RowBroadcast visitor broadcasts null_default(0).
    if (rscale == nullptr || resid == nullptr) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_residual<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<const float*>(rscale), static_cast<const float*>(resid), static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused_residual<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<const cutlass::half_t*>(rscale), static_cast<const cutlass::half_t*>(resid), static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused_residual<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<const cutlass::bfloat16_t*>(rscale), static_cast<const cutlass::bfloat16_t*>(resid), static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused<float>(a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused<cutlass::half_t>(a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused<cutlass::bfloat16_t>(a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (output_stride < N) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_strided<float>(a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), M, N, K, output_stride, stream);
        case 1: return dispatch_fused_strided<cutlass::half_t>(a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
        case 2: return dispatch_fused_strided<cutlass::bfloat16_t>(a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
        default: return false;
    }
}

// DIT-PLUGINS-PATCH: bias + FP32/FP16/BF16 outputs on the fixed-config path.
bool launch_cutlass_int8_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int config,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_config<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), M, N, K, config, stream);
        case 1: return dispatch_fused_config<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<cutlass::half_t*>(D), M, N, K, config, stream);
        case 2: return dispatch_fused_config<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<cutlass::bfloat16_t*>(D), M, N, K, config, stream);
        default: return false;
    }
}
// DIT-PLUGINS-PATCH: fused exact GELU/SILU + residual/gate epilogue.
bool launch_cutlass_int8_dequant_act_residual_gate(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    const void* gate, const void* residual, void* D, int64_t M, int64_t N, int64_t K,
    int out_dtype_code, int epilogue, int config, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if ((epilogue != 1 && epilogue != 2) || gate == nullptr || residual == nullptr)
        return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_act_residual<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<const float*>(gate),
            static_cast<const float*>(residual), static_cast<float*>(D), int(M), int(N),
            int(K), epilogue, config, stream);
        case 1: return dispatch_fused_act_residual<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias),
            static_cast<const cutlass::half_t*>(gate),
            static_cast<const cutlass::half_t*>(residual), static_cast<cutlass::half_t*>(D),
            int(M), int(N), int(K), epilogue, config, stream);
        case 2: return dispatch_fused_act_residual<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias),
            static_cast<const cutlass::bfloat16_t*>(gate),
            static_cast<const cutlass::bfloat16_t*>(residual),
            static_cast<cutlass::bfloat16_t*>(D), int(M), int(N), int(K), epilogue, config,
            stream);
        default: return false;
    }
}

// DIT-PLUGINS-PATCH: fused exact GELU/SILU epilogue (epilogue 1/2).
bool launch_cutlass_int8_dequant_act_config(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int epilogue, int config,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (epilogue != 1 && epilogue != 2) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_act<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), int(M),
            int(N), int(K), int(N), epilogue, config, stream);
        case 1: return dispatch_fused_act<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias),
            static_cast<cutlass::half_t*>(D), int(M), int(N), int(K), int(N), epilogue,
            config, stream);
        case 2: return dispatch_fused_act<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias),
            static_cast<cutlass::bfloat16_t*>(D), int(M), int(N), int(K), int(N), epilogue,
            config, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_act(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int epilogue, cudaStream_t stream)
{
    return launch_cutlass_int8_dequant_act_config(A, B, xs, ws, bias, D, M, N, K,
                                                  out_dtype_code, epilogue, -1, stream);
}

// DIT-PLUGINS-PATCH: BHSD/QKV store. k/v may be null (single-tensor layout).
bool launch_cutlass_int8_qkv(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* q, void* k, void* v, int64_t M, int64_t N, int64_t K, int64_t S, int H, int D,
    int out_dtype_code, int config, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (q == nullptr || S <= 0 || H <= 0 || D <= 0) return false;
    const int64_t C = int64_t(H) * D;
    if (N != C && (k == nullptr || v == nullptr || N != 3 * C)) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_qkv<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(q),
            static_cast<float*>(k), static_cast<float*>(v), M, N, K, S, H, D, config, stream);
        case 1: return dispatch_fused_qkv<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias),
            static_cast<cutlass::half_t*>(q), static_cast<cutlass::half_t*>(k),
            static_cast<cutlass::half_t*>(v), M, N, K, S, H, D, config, stream);
        case 2: return dispatch_fused_qkv<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias),
            static_cast<cutlass::bfloat16_t*>(q), static_cast<cutlass::bfloat16_t*>(k),
            static_cast<cutlass::bfloat16_t*>(v), M, N, K, S, H, D, config, stream);
        default: return false;
    }
}
}  // extern "C"

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to cuBLAS + separate dequant.

extern "C" bool launch_cutlass_int8_dequant(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_strided(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_config(
    const void*, const void*, const void*, const void*, const void*, void*,
    int64_t, int64_t, int64_t, int, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_residual(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, int64_t, int64_t, int64_t, int,
    cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_qkv(
    const void*, const void*, const void*, const void*, const void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int, int, int, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_act(
    const void*, const void*, const void*, const void*, const void*, void*, int64_t, int64_t,
    int64_t, int, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_act_residual_gate(
    const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int, int, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_act_config(
    const void*, const void*, const void*, const void*, const void*, void*, int64_t, int64_t,
    int64_t, int, int, int, cudaStream_t) {
    return false;
}

extern "C" int cutlass_int8_selected_config(int64_t, int64_t, int64_t) { return -1; }

#endif
