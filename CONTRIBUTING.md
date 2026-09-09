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
- `windows.yml`: MSVC + CUDA 13.2 + TRT zip (set the `TENSORRT_WINDOWS_URL`
  repository variable — NVIDIA's site is login-walled) + regcheck + import.

## Rules

- `src/kernel/` is vendored from ComfyKitchen: content changes go upstream,
  only the one-line vendoring notice may be added here.
- New plugins follow `src/wrapper/*_wrapper.cpp`: BF16-first dtype rules,
  `comfy_kitchen` namespace, `getTimingCacheID`, and an E2E test in `test/`.
- Every numeric change needs a measured before/after (`cos` + ms).
- Keep `plan.md` and `README.md` in sync with behavior changes.
