# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
cmake_minimum_required(VERSION 3.24)
# Multi-arch default must be chosen before project() enables CUDA: afterwards
# CMake has already cached the compiler default (sm_75 for CUDA 13) and the
# default list below would never apply. An explicit
# -DCMAKE_CUDA_ARCHITECTURES=... still wins; nvidia/cuda images export
# CUDAARCHS, which is ignored on purpose so the build stays multi-arch.
# sm80/86 (Ampere) + sm89 (Ada, verified) + sm90 (Hopper) + sm100/120
# (Blackwell) as SASS; sm90 PTX for driver JIT on newer targets.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real 100-real 120-real 90-virtual)
endif()
project(trt_dit_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
# CUDA 13's CCCL headers hard-error (C1189) under MSVC's traditional
# preprocessor; the conforming one is required for host compilation.
# NOTE: bare /Zc:preprocessor must NOT reach nvcc: nvcc counts a
# slash-prefixed token as a second input file and dies with
# "A single input file is required ... when an outputfile is specified".
# Route it through -Xcompiler= for the CUDA language so only cl sees it
# (cudafe++ is EDG-based and unaffected either way).
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
# sm80/86 (Ampere) + sm89 (Ada, verified) + sm90 (Hopper) + sm100/120
# (Blackwell) as SASS; sm90 PTX (driver JIT).

find_package(CUDAToolkit REQUIRED)
# TensorRT location: $TENSORRT_ROOT/{include,lib} if set (Windows CI),
# otherwise system paths (Linux apt).
if(DEFINED ENV{TENSORRT_ROOT})
    set(TRT_INCLUDE_DIR "$ENV{TENSORRT_ROOT}/include")
    set(TRT_LIB_DIR "$ENV{TENSORRT_ROOT}/lib")
else()
    set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
    set(TRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
endif()
find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR})
if(NOT NVINFER_LIB)
    # No -dev package: fall back to the versioned runtime lib
    # (links by SONAME, resolves via ldconfig at runtime).
    file(GLOB _nvinfer_so "${TRT_LIB_DIR}/libnvinfer.so.*")
    list(SORT _nvinfer_so)
    list(LENGTH _nvinfer_so _nvinfer_n)
    if(_nvinfer_n GREATER 0)
        math(EXPR _nvinfer_last "${_nvinfer_n} - 1")
        list(GET _nvinfer_so ${_nvinfer_last} NVINFER_LIB)
        message(STATUS "Using versioned libnvinfer: ${NVINFER_LIB}")
    endif()
endif()
if(NOT NVINFER_LIB)
    message(FATAL_ERROR "libnvinfer not found in ${TRT_LIB_DIR}")
endif()
find_program(CCACHE_BIN ccache)
if (CCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_BIN})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_BIN})
endif()

# CUTLASS is a pinned submodule (plan 4.2): header-only, no build step.
set(CUTLASS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include")
if(NOT EXISTS "${CUTLASS_INCLUDE}/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS headers not found. Run: git submodule update --init --recursive")
endif()

# Split kernel lib (vendored, unmodified) from wrapper lib.
# Runtime overhead is 3 PLT-routed launch calls only (ns-scale, unmeasurable).
set(CONVROT_SOURCES
    src/kernel/convrot/convrot_shim.cu
    third_party/comfy/convrot/int8_linear.cu
    third_party/comfy/convrot/cutlass_gemm_int8.cu
    third_party/comfy/convrot/convrot_w4a4.cu
    third_party/comfy/convrot/w4a8_gemm.cu)
add_library(ck_kernels SHARED
    third_party/comfy/sage_attention/quant_qk_int8.cu
    src/kernel/fused_qk_rope_requant.cu
    third_party/comfy/sage_attention/quant_v_int8.cu
    third_party/comfy/sage_attention/sage_attn_launcher.cu
    third_party/comfy/ops/adaln.cu
    third_party/comfy/ops/rms_rope.cu
    third_party/comfy/ops/per_tensor_quantize.cu
    src/kernel/sparge/sparge_pre.cu
    src/kernel/sparge/sparge_sm89_inst.cu
    src/kernel/prime.cu
    ${CONVROT_SOURCES})
# ConvRot (comfy-kitchen v0.2.39): the vendor TUs keep upstream's nvcc flags
# verbatim (CMakeLists.txt of comfy_kitchen/backends/cuda) so numerics match
# the oracle.
set_source_files_properties(${CONVROT_SOURCES} PROPERTIES
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/convrot;${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/convrot;${CUTLASS_INCLUDE}"
    COMPILE_DEFINITIONS "COMFY_HAVE_CUTLASS=1"
    COMPILE_OPTIONS
    "-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_BFLOAT16_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;-U__CUDA_NO_BFLOAT162_OPERATORS__;-U__CUDA_NO_BFLOAT162_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;--compress-mode=size;-O3")
# Origin isolation: same-basename headers exist in the comfy v0.2.33 tree, the
# comfy v0.2.39 convrot tree and sparge (mma/math/cp_async/...). Each TU sees
# ONLY its own origin dir, so a cross-origin bare #include fails loudly
# instead of silently mixing. (Quote-includes resolve file-locally first; -I
# is a backstop.)
set_source_files_properties(
    third_party/comfy/sage_attention/quant_qk_int8.cu
    third_party/comfy/sage_attention/quant_v_int8.cu
    third_party/comfy/sage_attention/sage_attn_launcher.cu
    PROPERTIES INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/common;${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/sage_attention")
# Own implementation (not vendored): sees its own dir plus the vendored
# v0.2.33 common helpers it builds on.
set_source_files_properties(src/kernel/fused_qk_rope_requant.cu
    PROPERTIES INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel;${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/common")
# v0.2.33 ops (adaln / rms_rope / fp8) share one upstream ops dir + common headers.
set_source_files_properties(
    third_party/comfy/ops/adaln.cu third_party/comfy/ops/rms_rope.cu
    third_party/comfy/ops/per_tensor_quantize.cu
    PROPERTIES INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/common;${CMAKE_CURRENT_SOURCE_DIR}/third_party/comfy/ops")
set_source_files_properties(src/kernel/sparge/sparge_pre.cu src/kernel/sparge/sparge_sm89_inst.cu
    PROPERTIES INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/sparge;${CMAKE_CURRENT_SOURCE_DIR}/third_party/sparge")
# SpargeAttn sm89 kernel (wgmma): compile this TU for sm89 only; the plugin
# refuses other architectures at build time (supportsTargetArchitecture).
set_source_files_properties(src/kernel/sparge/sparge_sm89_inst.cu PROPERTIES CUDA_ARCHITECTURES 89-real)
# fast-math ONLY for our own fused rope TU (vendored TUs untouched):
# the numerics gate passes with it, so it stays scoped here.
set_source_files_properties(src/kernel/fused_qk_rope_requant.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")
target_link_libraries(ck_kernels PRIVATE CUDA::cudart)
set_target_properties(ck_kernels PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
if(WIN32)
    # extern "C" launchers lack dllexport; auto-export so the import
    # lib is generated for trt_dit_plugins to link against.
    set_target_properties(ck_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(trt_dit_plugins SHARED
    src/wrapper/int8_attention_wrapper.cpp
    src/wrapper/sage_attn_wrapper.cpp
    src/wrapper/adaln_wrapper.cpp
    src/wrapper/rope_wrapper.cpp
    src/wrapper/fp8_wrapper.cpp
    src/wrapper/fused_int8_rope_sage_wrapper.cpp
    src/wrapper/rms_rope_wrapper.cpp
    src/wrapper/sparge_wrapper.cpp
    src/wrapper/convrot_linear_wrapper.cpp)
target_include_directories(trt_dit_plugins PRIVATE ${TRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/convrot
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/sparge)
target_link_libraries(trt_dit_plugins PRIVATE ck_kernels CUDA::cudart ${NVINFER_LIB})
set_target_properties(trt_dit_plugins PROPERTIES BUILD_RPATH_USE_ORIGIN ON)

option(BUILD_REGCHECK "Build registration check tool (no GPU needed)" ON)
if(BUILD_REGCHECK)
    add_executable(regcheck test/test_register.cpp)
    target_include_directories(regcheck PRIVATE ${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(regcheck PRIVATE ${NVINFER_LIB})
endif()
