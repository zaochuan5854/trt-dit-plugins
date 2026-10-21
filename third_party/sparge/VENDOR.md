# Vendored SpargeAttn sources

Source: https://github.com/thu-ml/SpargeAttn
Pin: `ae5b629` ("reduce repo size")
License: Apache-2.0 (see `NOTICE`)

Extracted minimum (no whole-repo vendoring, no pybind/torch):

```
csrc/{cp_async,mma,permuted_smem,math,reduction_utils,numeric_conversion}.cuh
csrc/qattn/{qk_int_sv_f8_cuda_sm89.cuh,attn_utils.cuh,decl.cuh}
```

These are unmodified copies. Project-owned build glue lives outside this
tree, under `src/kernel/sparge/`:

| file | role |
|---|---|
| `sparge_pre.cu` | torch-free pre-steps (mirrors upstream `utils.py` / `core.py`) |
| `sparge_sm89_inst.cu` | 4 explicit instantiations (upstream `autogen.py` makes 192) |
| `sparge_launcher.h` | host declarations shared by the kernel TUs and the TRT wrapper |

SpargeAttn descends from SageAttention; see `NOTICE` for the full chain.
