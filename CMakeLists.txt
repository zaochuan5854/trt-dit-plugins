# SPDX-License-Identifier: Apache-2.0
# Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
cmake_minimum_required(VERSION 3.24)
project(trt_dit_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
# sm80/86 (Ampere) + sm89 (Ada, verified) + sm90 (Hopper) as SASS; sm90
# PTX (driver JIT). Override with -DCMAKE_CUDA_ARCHITECTURES=89 for Ada-only.
# nvidia/cuda images export CUDAARCHS (e.g. 89), which CMake copies into
# CMAKE_CUDA_ARCHITECTURES and would silently single-arch the build.
# Drop env-initialized values so the default below applies; an explicit
# -D on the command line still wins.
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{CUDAARCHS}
   AND CMAKE_CUDA_ARCHITECTURES STREQUAL "$ENV{CUDAARCHS}")
    unset(CMAKE_CUDA_ARCHITECTURES)
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real 90-virtual)
endif()

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

# Split kernel lib (vendored, unmodified) from wrapper lib.
# Runtime overhead is 3 PLT-routed launch calls only (ns-scale, unmeasurable).
add_library(ck_kernels SHARED
    src/kernel/sage_attention/quant_qk_int8.cu
    src/kernel/sage_attention/quant_v_int8.cu
    src/kernel/sage_attention/sage_attn_launcher.cu
    src/kernel/adaln/adaln.cu
    src/kernel/rope/rms_rope.cu
    src/kernel/fp8/per_tensor_quantize.cu
    src/kernel/sparge/sparge_pre.cu
    src/kernel/sparge/sparge_sm89_inst.cu
    src/kernel/prime.cu)
target_include_directories(ck_kernels PRIVATE
    src/kernel/sage_attention src/kernel/adaln src/kernel/rope src/kernel/fp8
    src/kernel/sparge)
# SpargeAttn sm89 kernel (wgmma): compile this TU for sm89 only; the plugin
# refuses other architectures at build time (supportsTargetArchitecture).
set_source_files_properties(src/kernel/sparge/sparge_sm89_inst.cu PROPERTIES CUDA_ARCHITECTURES 89-real)
target_link_libraries(ck_kernels PRIVATE CUDA::cudart)
set_target_properties(ck_kernels PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
if(WIN32)
    # extern "C" launchers lack dllexport; auto-export so the import
    # lib is generated for trt_dit_plugins to link against.
    set_target_properties(ck_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(trt_dit_plugins SHARED
    src/wrapper/int8_attention_wrapper.cpp
    src/wrapper/adaln_wrapper.cpp
    src/wrapper/rope_wrapper.cpp
    src/wrapper/fp8_wrapper.cpp
    src/wrapper/rms_rope_wrapper.cpp)
target_include_directories(trt_dit_plugins PRIVATE ${TRT_INCLUDE_DIR})
target_link_libraries(trt_dit_plugins PRIVATE ck_kernels CUDA::cudart ${NVINFER_LIB})
set_target_properties(trt_dit_plugins PROPERTIES BUILD_RPATH_USE_ORIGIN ON)

option(BUILD_REGCHECK "Build registration check tool (no GPU needed)" ON)
if(BUILD_REGCHECK)
    add_executable(regcheck test/test_register.cpp)
    target_include_directories(regcheck PRIVATE ${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(regcheck PRIVATE ${NVINFER_LIB})
endif()
