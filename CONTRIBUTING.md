# SPDX-License-Identifier: Apache-2.0
# Contributing to TensorRT-DiT-Plugins

## Setup

```bash
podman build --rm -t trt-plugins:sm89 .
podman run --rm --security-opt=label=disable \
  -v $PWD:/workspace -v /tmp/ck-ccache:/tmp/ccache \
  localhost/trt-plugins:sm89 bash -c \
  "cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
   cmake --build /workspace/build --parallel \$(nproc)"
```

## Tests (all must pass, on sm89 hardware)
```bash
# C++ engine-level tests (see test/ for the full matrix)
./build/regcheck ./build/libtrt_dit_plugins.so
./build/e2e 256 8 2 ./build/libtrt_dit_plugins.so
# Python frontend (needs torch + tensorrt in the env)
PYTHONPATH=python TRT_DIT_LIBDIR=build python3 python/tests/test_smoke.py
```

## CI (`.github/workflows/`)

- `linux.yml`: full compile + regcheck + Python import in CUDA container (no GPU
  on hosted runners; GPU E2E stays self-hosted).
- `windows.yml`: MSVC + CUDA 12.6 (Jimver/cuda-toolkit) + TRT via the shared
  `setup-trt-windows` action (pip DLLs + pinned headers + generated import lib).

## Rules

- `src/kernel/` is vendored: content changes go upstream, only the vendoring
  notice plus documented compat patches may live here.
- Kernel origins are isolated per-TU in CMake (`src/kernel/<origin>/` only):
  comfy and sparge trees share header basenames (`mma`/`math`/`cp_async`/…).
  Never add a cross-origin `-I`; a bare cross-origin `#include` must fail,
  not silently mix.
- New plugins follow `src/wrapper/*_wrapper.cpp`: BF16-first dtype rules,
  `comfy_kitchen` namespace, `getTimingCacheID`, and an E2E test in `test/`.
- Every numeric change needs a measured before/after (`cos` + ms).
- Keep `README.md` in sync with behavior changes.
