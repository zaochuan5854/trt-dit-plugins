# SPDX-License-Identifier: Apache-2.0
# sm89 / Linux / TRT10.16 — cached cuda devel base, no pull needed
FROM docker.io/nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CC=gcc CXX=g++ \
    CCACHE_DIR=/tmp/ccache \
    TRT_VER=10.16.1.11-1+cuda13.2

# Skip -dev (static libs + samples exceed 3GB). Shared libs + headers suffice.
# Persist debs across rebuilds via cache mount. Never clean (would wipe the cache).
RUN --mount=type=cache,target=/var/cache/apt/archives \
    apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build ccache g++ python3 python3-pip \
    libnvinfer10=${TRT_VER} libnvinfer-plugin10=${TRT_VER} \
    libnvinfer-headers-dev=${TRT_VER} libnvinfer-headers-plugin-dev=${TRT_VER} \
    libnvinfer-safe-headers-dev=${TRT_VER} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Restore the libnvinfer.so symlink missing without -dev (for shared-lib linking)
RUN ln -sf libnvinfer.so.10 /usr/lib/x86_64-linux-gnu/libnvinfer.so \
 && ln -sf libnvinfer_plugin.so.10 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so \
 && ldconfig
# C++ version assert (no 10.16 wheel on PyPI and no NGC index reach, so verify in C++)
RUN printf '#include <NvInferVersion.h>\n#include <cassert>\n#include <cstdio>\nint main(){assert(NV_TENSORRT_MAJOR==10&&NV_TENSORRT_MINOR==16);std::printf("TRT %%d.%%d.%%d.%%d\\n",NV_TENSORRT_MAJOR,NV_TENSORRT_MINOR,NV_TENSORRT_PATCH,NV_TENSORRT_BUILD);}\n' > /tmp/vercheck.cpp \
 && g++ -o /tmp/vercheck /tmp/vercheck.cpp -I/usr/include/x86_64-linux-gnu && /tmp/vercheck \
 && nvcc --version && ls /usr/include/x86_64-linux-gnu/NvInfer.h && rm -f /tmp/vercheck*
