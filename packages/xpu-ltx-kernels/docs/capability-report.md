# XPU capability report (Arc Pro B60, oneAPI 2026.1)

Probed on this machine: 20x Intel Arc Pro B60 (Xe2/Battlemage, 160 XMX each),
~24 GB GDDR6, oneAPI 2026.1, torch 2.13.0+xpu (lmccl_xpu env, runtime
intel-sycl-rt 2026.0.0).

## Toolchain (the hard-won recipe)

- Build extensions with **icpx directly**, NOT torch's `cpp_extension` SYCL
  path. The working recipe (see `setup.py`):
  1. `icpx -fsycl -fsycl-targets=spir64_gen,spir64 -sycl-std=2020 -x c++ -c src.sycl -o src.o`
     — compile against **the active Python env's SYCL headers**
     (`<sys.prefix>/include`), which must match the runtime torch's XPU wheel
     bundles. Use `-x c++` (icpx does not auto-detect `.sycl`; without it the
     file is treated as a linker input and compile silently no-ops).
  2. `icpx -shared -fsycl -o mod.so src.o -ltorch_python -ltorch -ltorch_cpu
     -lc10 -ltorch_xpu -lc10_xpu -lsycl ...` — the `-fsycl` **device-link** is
     mandatory: it packages the device image. Linking with plain `g++` (no
     `-fsycl`) produces a `.so` that segfaults at kernel submission.
- Source files must be `.sycl` and keep `#include <ATen/xpu/XPUContext.h>` +
  `c10::xpu::getCurrentXPUStream()` (returns XPUStream, convertible to
  `sycl::queue&`).
- `sycl::ext::oneapi::bfloat16` is the SYCL type for torch bf16 — **not**
  `sycl::half` (that is fp16; reading bf16 through it corrupts values).
- Environment before any build/run:
  `source /opt/intel/oneapi/setvars.sh` and `SYCL_DEVICE_FILTER=level_zero:gpu`.
- oneAPI 2026.1 ESIMD headers have N=1 scalar-math bugs (`fmod`/`sin_emu`/
  `atan2` wrappers break on `reinterpret_cast`/`__builtin_convertvector` of
  scalar `simd<T,1>`). Plain SYCL and the env's 2026.0.0 headers compile fine;
  avoid instantiating those scalar wrappers (they trigger when a scalar math
  function is called from ESIMD device code).
- torch's own SYCL build path (`cpp_extension.load`) is not usable here: it
  forces `-fsycl-host-compiler=c++` + its own header order, and the 2025.3.2
  runtime (torch 2.12.0+xpu repo env) conflicts with the only installed
  compiler (2026.1). Use the `lmccl_xpu` env (torch 2.13.0+xpu, runtime
  2026.0.0) for kernel work.

## GEMM

- **bf16 GEMM is at ~96 TFLOPS (≈96% of XMX peak)** on realistic 22B shapes
  (M=1024-16384, N/K=5120-13824), via oneDNN. A hand-written ESIMD DPAS GEMM
  will not beat it; do not build a raw bf16 GEMM.
- **No FP8 tensor cores** on Xe2 (XMX = INT2/4/8, FP16, BF16, TF32). FP8 is
  storage-only: `torch._scaled_mm` is `NotImplementedError` on XPU, and fp8
  `@` fp8 falls back to nothing. Use fp8 for weight storage (2x memory saving)
  with bf16 DPAS compute.
- **INT8 XMX is ~2x bf16** (~200 TFLOPS) — the only path to >bf16 compute on
  B60. A blockwise INT8 W8A8 GEMM (int8 weights+activations, per-block fp32
  scales, fp32 accumulate via DPAS) is the B60-specific optimization this fork
  adds on top of ltx-kernels' FP8 (Blackwell) kernels.

## Memory / allocator

- Torch's XPU caching allocator holds ~29 GB after the text encoder; the
  `xpu_activation_budget_bytes()` / `empty_device_cache()` XPU handling in
  `ltx_core/devices.py` releases it between pipeline stages.

## Fused ops (xpu-ltx-kernels/ops)

- `rms_norm_rope`, `rms_norm_split_rope`, `fp6_pack`, `fp6_unpack`: plain-SYCL
  ports of the ltx-kernels CUDA kernels; validated against torch references
  (see `tests/test_ops.py`). One workgroup per row; barrier-tree reduction for
  the RMS sum (subgroup `reduce_over_group` and local-memory atomics were both
  unreliable on this runtime); cross-half partner exchange in split rope goes
  through local memory, not warp shuffle.
- Grid gotchas: use one workgroup per row (`global = rows * wg`), never
  `global_id` as the row index; never early-return a work-item before a
  subgroup/group reduction (clamp the row and mask stores instead).