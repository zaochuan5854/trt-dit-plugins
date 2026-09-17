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
# C++ registration check (covers all 9 plugins, no GPU needed)
./build/regcheck ./build/libtrt_dit_plugins.so
# ONNX surgery tests (CPU-only, needs pip install "trt-dit-plugins[onnx]")
PYTHONPATH=python python3 -m pytest python/tests/test_surgeon.py -q
# Python frontend (needs torch + tensorrt in the env)
PYTHONPATH=python TRT_DIT_LIBDIR=build python3 python/tests/test_smoke.py
# RPN invariant test (needs torch, no GPU/TRT)
PYTHONPATH=python python3 -m pytest python/tests/test_rpn.py -q
```

## CI (`.github/workflows/`)

- `linux.yml`: full compile + regcheck + Python import + surgery pytest in
  CUDA container (no GPU on hosted runners; GPU E2E stays self-hosted).
- `windows.yml`: MSVC + CUDA 13 (Jimver/cuda-toolkit) + TRT via the shared
  `setup-trt-windows` action (pip DLLs + pinned headers + generated import lib),
  plus the same surgery pytest.
- `release.yml`: on GitHub release publication, builds per-platform wheels
  and uploads them as release assets (decoupled from tag push).

## Rules

- `src/kernel/` is vendored: content changes go upstream, only the vendoring
  notice plus documented compat patches may live here.
- Kernel origins are isolated per-TU in CMake (`src/kernel/<origin>/` only):
  comfy and sparge trees share header basenames (`mma`/`math`/`cp_async`/…).
  Never add a cross-origin `-I`; a bare cross-origin `#include` must fail,
  not silently mix.
- New plugins follow `src/wrapper/*_wrapper.cpp`: BF16-first dtype rules,
  `dit-plugins` namespace, `getTimingCacheID`, a `regcheck` entry in
  `test/test_register.cpp`, and measured verification (`cos` + ms;
  torch-level scripts in `/tmp` are ok when no C++ harness exists).
- Every numeric change needs a measured before/after (`cos` + ms).
- Keep `README.md` (plus the ja/zh translations) in sync with behavior changes.

## Version pins (release-relevant)

- TRT is pinned exact everywhere (`tensorrt-cu13-libs==11.3.0.99`,
  v11.2 GitHub headers, `TRT_VER` in workflows): patch updates need a
  three-point sync (this repo, the comfy image, CI). A mismatch fails loud
  at pip resolution, never silently.
- ONNX is lower-bound only (`onnx>=1.20`, `onnx-graphsurgeon>=0.3`):
  `surgeon.save()` caps `ir_version` for TRT readability, and
  `test_save_caps_ir_version` guards the cap.
- CUDA toolchain is 13.x (`_lib.py` probes `libcudart.so.13` /
  `libnvinfer.so.11` only). Build with the oldest supported toolchain:
  newer builds raise the runtime floor.
