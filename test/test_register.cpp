// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// Registration check: loadLibrary the .so and fetch comfy_kitchen::int8_attention.
#include <NvInferRuntime.h>
#include <dlfcn.h>
#include <cassert>
#include <cstdio>

int main(int argc, char** argv) {
    assert(argc > 1);
    // loadLibrary returns null when already registered, so confirm
    // directly via dlopen (runs static registration) + getCreator
    assert(dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL));
    auto* reg = getPluginRegistry();
    auto* c = reg->getCreator("int8_attention", "1", "comfy_kitchen");
    assert(c);
    std::fprintf(stderr, "OK creator: int8_attention 1 comfy_kitchen\n");
    auto* a = reg->getCreator("adaln", "1", "comfy_kitchen");
    assert(a);
    std::fprintf(stderr, "OK creator: adaln 1 comfy_kitchen\n");
    auto* r = reg->getCreator("rms_adaln", "1", "comfy_kitchen");
    assert(r);
    std::fprintf(stderr, "OK creator: rms_adaln 1 comfy_kitchen\n");
    auto* ro = reg->getCreator("apply_rope", "1", "comfy_kitchen");
    assert(ro);
    std::fprintf(stderr, "OK creator: apply_rope 1 comfy_kitchen\n");
    auto* fp = reg->getCreator("stochastic_round_fp8", "1", "comfy_kitchen");
    assert(fp);
    std::fprintf(stderr, "OK creator: stochastic_round_fp8 1 comfy_kitchen\n");
    auto* rr = reg->getCreator("rms_rope_split_half", "1", "comfy_kitchen");
    assert(rr);
    std::fprintf(stderr, "OK creator: rms_rope_split_half 1 comfy_kitchen\n");
    return 0;
}
