# SPDX-License-Identifier: Apache-2.0
# sm89 / Linux / TRT11.3 cu13 — cached cuda devel base, no pull needed
FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CC=gcc CXX=g++ \
    CCACHE_DIR=/tmp/ccache \
    TRT_VER=11.3.0.99 \
    TRT_HEADERS_TAG=v11.2 \
    TENSORRT_ROOT=/tmp/trt

# No cuda13.0 TRT debs exist: link against the pip libs, fetch headers
# from GitHub (v11.2 headers + 11.3 runtime).
# Persist wheels across rebuilds via cache mount.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl cmake ninja-build ccache g++ python3 python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --break-system-packages -q \
    "tensorrt-cu13-libs==${TRT_VER}" \
    --extra-index-url https://pypi.nvidia.com \
 && TRT_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/tensorrt_libs \
 && test -f "$TRT_LIB/libnvinfer.so.11" \
 && mkdir -p /tmp/trt/include /tmp/trt/lib \
 && for h in NvInfer.h NvInferImpl.h \
    NvInferLegacyDims.h NvInferPlugin.h NvInferPluginBase.h \
    NvInferPluginUtils.h NvInferRuntime.h NvInferRuntimeBase.h \
    NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do \
      curl -fsSL "https://raw.githubusercontent.com/NVIDIA/TensorRT/${TRT_HEADERS_TAG}/include/$h" \
        -o "/tmp/trt/include/$h"; \
    done \
 && ln -sf "$TRT_LIB/libnvinfer.so.11" /tmp/trt/lib/libnvinfer.so \
 && echo "$TRT_LIB" > /etc/ld.so.conf.d/trt-pip.conf && ldconfig

WORKDIR /workspace

# C++ version assert (pip-only TRT, so verify in C++)
RUN printf '#include <NvInferVersion.h>\n#include <cassert>\n#include <cstdio>\nint main(){assert(NV_TENSORRT_MAJOR==11&&NV_TENSORRT_MINOR==3);std::printf("TRT %%d.%%d.%%d.%%d\\n",NV_TENSORRT_MAJOR,NV_TENSORRT_MINOR,NV_TENSORRT_PATCH,NV_TENSORRT_BUILD);}\n' > /tmp/vercheck.cpp \
 && g++ -o /tmp/vercheck /tmp/vercheck.cpp -I/tmp/trt/include && /tmp/vercheck \
 && nvcc --version && ls /tmp/trt/include/NvInfer.h && rm -f /tmp/vercheck*
