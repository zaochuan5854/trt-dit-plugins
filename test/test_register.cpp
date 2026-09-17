// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// Registration check: dlopen the plugin lib and fetch every dit-plugins
// creator. Explicit return codes (no assert: NDEBUG in Release builds would
// compile them out and the check would pass vacuously).
#include <NvInferRuntime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstdio>

namespace {
const char* kNames[] = {
    "int8_attention",
    "sage_attn",
    "adaln",
    "rms_adaln",
    "apply_rope",
    "rms_rope_split_half",
    "stochastic_round_fp8",
    "block_sparse_sage2_attn",
    "fused_int8_rope_sage_attn",
};
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: regcheck <plugin-lib>\n");
        return 2;
    }
    // loadLibrary returns null when already registered, so confirm
    // directly via dlopen (runs static registration) + getCreator
#ifdef _WIN32
    void* handle = static_cast<void*>(LoadLibraryA(argv[1]));
    if (!handle) {
        std::fprintf(stderr, "FAIL load %s winerr=%lu\n", argv[1],
                     static_cast<unsigned long>(GetLastError()));
        return 1;
    }
#else
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::fprintf(stderr, "FAIL load %s: %s\n", argv[1], dlerror());
        return 1;
    }
    (void)handle;
#endif
    auto* reg = getPluginRegistry();
    if (!reg) {
        std::fprintf(stderr, "FAIL getPluginRegistry\n");
        return 1;
    }
    int rc = 0;
    for (const char* name : kNames) {
        auto* creator = reg->getCreator(name, "1", "dit-plugins");
        if (!creator) {
            std::fprintf(stderr, "FAIL creator: %s 1 dit-plugins\n", name);
            rc = 1;
            continue;
        }
        std::fprintf(stderr, "OK creator: %s 1 dit-plugins\n", name);
    }
    return rc;
}
