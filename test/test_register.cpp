// SPDX-License-Identifier: Apache-2.0
// Interface derived from ComfyKitchen (Copyright (c) 2025 Comfy Org, Apache-2.0).
// Registration check: loadLibrary the .so and fetch dit-plugins::int8_attention.
#include <NvInferRuntime.h>
#ifdef _WIN32
#include <windows.h>
static void* dlopen(const char* p, int) { return (void*)LoadLibraryA(p); }
#else
#include <dlfcn.h>
#endif
#include <cassert>
#include <cstdio>

int main(int argc, char** argv) {
    assert(argc > 1);
    // loadLibrary returns null when already registered, so confirm
    // directly via dlopen (runs static registration) + getCreator
#ifdef _WIN32
    assert(dlopen(argv[1], 0));
#else
    assert(dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL));
#endif
    auto* reg = getPluginRegistry();
    auto* c = reg->getCreator("int8_attention", "1", "dit-plugins");
    assert(c);
    std::fprintf(stderr, "OK creator: int8_attention 1 dit-plugins\n");
    auto* a = reg->getCreator("adaln", "1", "dit-plugins");
    assert(a);
    std::fprintf(stderr, "OK creator: adaln 1 dit-plugins\n");
    auto* r = reg->getCreator("rms_adaln", "1", "dit-plugins");
    assert(r);
    std::fprintf(stderr, "OK creator: rms_adaln 1 dit-plugins\n");
    auto* ro = reg->getCreator("apply_rope", "1", "dit-plugins");
    assert(ro);
    std::fprintf(stderr, "OK creator: apply_rope 1 dit-plugins\n");
    auto* fp = reg->getCreator("stochastic_round_fp8", "1", "dit-plugins");
    assert(fp);
    std::fprintf(stderr, "OK creator: stochastic_round_fp8 1 dit-plugins\n");
    auto* rr = reg->getCreator("rms_rope_split_half", "1", "dit-plugins");
    assert(rr);
    std::fprintf(stderr, "OK creator: rms_rope_split_half 1 dit-plugins\n");
    auto* bs = reg->getCreator("block_sparse_sage2_attn", "1", "dit-plugins");
    assert(bs);
    std::fprintf(stderr, "OK creator: block_sparse_sage2_attn 1 dit-plugins\n");
    return 0;
}
