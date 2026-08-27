# xpu-ltx-kernels

Custom XPU (Intel GPU) kernels for LTX inference, built with SYCL/ESIMD via
`icpx`. This is the XPU counterpart to `ltx-kernels` (CUDA): same role, native
Intel kernels, validated on 20x Intel Arc Pro B60 (Xe2/Battlemage).

## What's here

| Module | Kernel | Notes |
|--------|--------|-------|
| `xpu_ltx_kernels.ops` | `rms_norm_rope`, `rms_norm_split_rope` (fused RMS-norm + RoPE), `fp6_pack`, `fp6_unpack` | Plain SYCL ports of `ltx-kernels/csrc/ops`, drop-in for the blockwise Q8 attention path |
| `xpu_ltx_kernels.gemm` | `fp8_dequantize_blockwise` (ESIMD), `BlockwiseFP8Linear` | FP8 blockwise weight storage (2x memory) + oneDNN BF16 GEMM; Xe2 has **no FP8 tensor cores** (XMX = INT2/4/8, FP16, BF16, TF32) |
| `xpu_ltx_kernels.vae` | `na3d` | 3D neighborhood attention matching `natten.na3d` semantics for the DiffVAE decoder |
| `xpu_ltx_kernels.all2all` | `All2All` | Head redistribution for sequence-parallel multi-GPU, via `torch.distributed` |

## Build

Excluded from the uv workspace (needs the SYCL toolchain). Install into a
torch-XPU env whose runtime matches the compiler:

```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=level_zero:gpu
pip install -e packages/xpu-ltx-kernels --no-build-isolation
```

Requirements and the hard-won toolchain recipe (icpx `-x c++` + the mandatory
`-fsycl` device-link step, the `lmccl_xpu` env / torch 2.13.0+xpu) are
documented in [`docs/capability-report.md`](docs/capability-report.md). The
oneAPI 2026.1 ESIMD headers have N=1 scalar-math compile bugs; plain SYCL and
the pip 2026.0.0 headers avoid them.

## Test

```bash
source /opt/intel/oneapi/setvars.sh && SYCL_DEVICE_FILTER=level_zero:gpu \
  python -m pytest packages/xpu-ltx-kernels/src/xpu_ltx_kernels/tests -v
```

## Integration with ltx-core

`ltx_core.quantization.blockwise` resolves kernels per-device: CUDA uses
`ltx-kernels`, XPU uses this package. `--quantization blockwise` in the
pipelines selects it; the Q8 attention path's fused `rms_norm_rope` and the
fp8 blockwise weight storage come from here on XPU.