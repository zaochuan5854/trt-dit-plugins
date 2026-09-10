// SPDX-License-Identifier: Apache-2.0
// First-launch primer (see below). A 1-thread no-op kernel.
#include <atomic>

#include <cuda_runtime.h>

__global__ void ck_prime_kernel() {}

// WORKAROUND (driver): the first cudaLaunchKernel from this library's fatbins
// into an already-GPU-busy host process (e.g. torch ran kernels) segfaults
// inside cuLaunchKernel (verified: priming first fixes all later launches;
// without it, even a trivial kmean launch dies). The prime itself always
// survives. Call lazily (see below) — NOT at import/dlopen, because the prime
// must be the first launch *after* host GPU activity, and import time cannot
// guarantee that. The kernel touches no memory, so no sync is needed.
#include <atomic>
extern "C" void ck_prime_once(cudaStream_t stream) {
    ck_prime_kernel<<<1, 1, 0, stream>>>();
}
