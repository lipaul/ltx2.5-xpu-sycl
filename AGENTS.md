# AGENTS.md

LTX-2: a PyTorch audio-video diffusion model. This working tree is a vendored snapshot of
[`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2), pushed to a personal fork
(`lipaul/ltx-2-xpu`, `git remote -v`). It is a `uv` workspace monorepo of four packages.
The XPU support is the fork's divergence from upstream: the `ltx-core` torch pins, the XPU branches in
`devices.py` and `attention.py`, and `run_pipeline.sh`. Preserve them when rebasing upstream changes.

## Per-package instruction files (read these first)

These already encode hard-won, repo-specific guidance. Read the relevant one before editing:

- `packages/ltx-trainer/AGENTS.md` — training package: config conventions, `configs/` doc sync rule,
  frame/resolution constraints (`frames % T == 1`), split-vs-unified checkpoint layouts, testing standards.
- `packages/ltx-pipelines/CLAUDE.md` — inference pipelines: pipeline selection table, sigma schedules,
  LoRA conventions, `utils/blocks.py` memory model, DFR invariants, generated-keyframes rules.
- `packages/ltx-kernels/README.md` and `packages/ltx-core/README.md` — CUDA extension layout and model components.
- `packages/xpu-ltx-kernels/README.md` + `docs/capability-report.md` — the fork's XPU kernel
  package (SYCL/ESIMD): the icpx build recipe (compile with `-x c++`, link with `icpx -shared -fsycl`
  — skipping the device-link segfaults at kernel submission), the uv-venv build/run requirements, and
  the B60 hardware findings (no FP8 tensor cores, bf16 GEMM already at ~95 TFLOPS via oneDNN).
- Any training / LoRA / fine-tuning request: the `train-model` skill
  (`.claude/skills/train-model/SKILL.md`) is the orchestrator. The trainer is CUDA-only and excluded from
  the XPU workspace, so do not expect a plain `uv sync` to install it.

Keep changes to each package consistent with its own instruction file.

## Package layout

- `packages/ltx-core/` — model definitions (transformer, VAEs, vocoder, Gemma text encoder), loading, quantization. Depended on by both other packages.
- `packages/ltx-pipelines/` — ready-made inference pipelines (`ltx_pipelines.*`, run as `python -m ltx_pipelines.<module>`).
- `packages/ltx-trainer/` — LoRA / full fine-tuning (`scripts/train.py`, `configs/*.yaml`).
- `packages/ltx-kernels/` — optional CUDA/C++ extensions (all2all, blockwise FP8/FP6 GEMM, NVFP4, CuTe DSL VAE kernels).
- `packages/xpu-ltx-kernels/` — the fork's XPU kernel package (SYCL/ESIMD, icpx): fused
  `rms_norm_rope` / `rms_norm_split_rope`, FP6 pack, blockwise-FP8 storage + bf16 GEMM, VAE
  neighborhood attention, all2all. Excluded from the uv workspace (needs icpx + a matching SYCL
  runtime); install opt-in into the uv venv (`uv pip install -e ... --no-build-isolation`).
- `run_pipeline.sh` — the fork's inference launcher: runs the split-layout distilled pipeline on XPU with
  pre-wired model paths (default model root `/home/acm/paul/models/ltx-2.5`, override `LTX_MODEL_ROOT`;
  `PIPELINE=distilled` selects the module). Extra args forward to `python -m ltx_pipelines.distilled`; it does
  **not** add `--offload`, which the 22B model still needs on a 32 GB XPU.

## Toolchain (uv + torch XPU)

This fork runs the inference pipeline on **Intel XPU** (Linux). The environment is resolved entirely via `uv`
against the official PyTorch **XPU** index — no CUDA.

- Use `uv` everywhere: `uv sync`, `uv run`. Do not use bare `pip` or a system python.
- torch/torchaudio are pinned to the **XPU** index in `packages/ltx-core/pyproject.toml`:
  `torch==2.13.0+xpu`, `torchvision==0.28.0+xpu`, `torchaudio==2.11.0+xpu`, plus `triton-xpu==3.7.2`
  (declared as a *direct* dep — uv only applies `[tool.uv.sources]` index mappings to direct deps, so the
  transitive `triton-xpu` would otherwise resolve from PyPI at a conflicting version). The torch 2.13.0+xpu
  wheel bundles `intel-sycl-rt` 2026.0.0 (libsycl.so.9), which matches the installed oneAPI 2026.1
  compiler — that pairing is required to build/run the SYCL extension.
- **`transformers` is capped at `<5.15`** (in `ltx-core` deps). 5.15.0+ breaks the Gemma 4 text encoder build.
- `ltx-core` bounds `requires-python = ">=3.10,<3.14"`: `triton-xpu==3.7.2` has no free-threaded 3.14
  (cp314t) Windows wheel, so an unbounded range makes the universal lock unresolvable.
- **`ltx-trainer`, `ltx-kernels`, and `xpu-ltx-kernels` are excluded from the uv workspace**
  (`[tool.uv.workspace].exclude`). The trainer pins CUDA-only `torchcodec`; the kernels are CUDA
  extensions; xpu-ltx-kernels needs the icpx/SYCL toolchain. A plain `uv sync` installs only
  `ltx-core` + `ltx-pipelines`, the pipeline stack.
- **Custom XPU kernels build with icpx in the uv venv** (`uv pip install -e
  packages/xpu-ltx-kernels --python .venv/bin/python --no-build-isolation`). Do NOT use torch's
  `cpp_extension` SYCL path (it forces `-fsycl-host-compiler`); use icpx directly per
  `packages/xpu-ltx-kernels/docs/capability-report.md`. Before any XPU build/run:
  `source /opt/intel/oneapi/setvars.sh` and `SYCL_DEVICE_FILTER=level_zero:gpu`.
- Fresh clone is shallow; `git fetch --unshallow origin` before pushing a branch that depends on full history.

## Code quality gates

Ruff is configured at the root (`pyproject.toml` `[tool.ruff]`): line length **120**, strict ruleset
(ANN, B, PL, SIM, ...), `known-first-party = ["ltx_core", "ltx_pipelines", "ltx_trainer"]`. Run:

```bash
uv run ruff check .
uv run ruff format .
```

There is **no `.pre-commit-config.yaml`** in this tree — `pre-commit run` is not a working gate; ruff is the gate.
`packages/xpu-ltx-kernels/**` has per-file-ignores for tensor-shape uppercase naming (`B`, `S`, `H`, `D`) and
test annotations, mirroring the CuTe DSL treatment.

## Runtime gotchas

- Model checkpoints (`.safetensors`) and media are gitignored — never commit them. Downloads and pipeline
  runs need weights fetched via `hf download`/`hf auth login` (gated repo, Read token).
- Checkpoint paths are local only (URLs unsupported). Two layouts exist — **unified** (one `.safetensors` +
  Gemma dir) and **split** (one file per component); they are not interchangeable and the code detects the
  layout from checkpoint metadata, not a flag.
- Audio-video work needs an Intel GPU with XPU support; the pipeline selects the device in
  `packages/ltx-core/src/ltx_core/devices.py` (XPU is probed via `torch.xpu.is_available()`).
- LTX 2.5 requires the LTX-fine-tuned Gemma 4 text encoder; Google's vanilla Gemma 4 is not a substitute.
- **XPU allocator cache is never freed by the upstream memory helpers.** `cleanup_accelerator_memory` /
  `empty_device_cache` / `synchronize_device` in `devices.py` were CUDA/MPS-only, so the ~29 GB XPU
  caching-allocator cache from the text encoder stayed reserved and the DiffVAE tiling check reported
  `usable_bytes=0` (`Cannot fit a DiffVAE decode tile`). `xpu_activation_budget_bytes()` was added (mirrors
  `cuda_activation_budget_bytes`) and the sync/empty helpers now handle XPU via `torch.xpu.empty_cache()`.
- **XPU SDPA falls to the quadratic math backend by default.** torch's XPU runtime-disables the memory-efficient
  kernel, and the old XPU branch in `_sdpa_full_priority()` (`attention.py`) put `EFFICIENT_ATTENTION` first,
  so video attention materialized the full QK^T score matrix and OOM'd (~72 GB at 1536x1024/121f).
  `FLASH_ATTENTION` IS supported on XPU for bf16 — the fix prioritizes `FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH`
  on XPU, cutting attention memory to ~0.17 GB. The warning "Memory Efficient ... falling back to math" will
  still appear for float32 autocast-disabled paths; bf16 flash is what matters.
- **The 22B distilled transformer does not fit on a 32 GB XPU at once (44 GB bf16 weights).** Run pipelines
  with `--offload cpu` (needs ~36 GB RAM for pinned weights; disk is the lowest-memory option) so weights are
  streamed layer-by-layer. Activations then fit under flash attention.
- `torchvision` is required (transitively by transformers' Gemma 4 image processor) but was missing from the
  lockfile; pinned on the XPU index as `torchvision==0.28.0+xpu` (the newest XPU wheel for torch 2.13.0+xpu).
- Sourcing oneAPI (`/opt/intel/oneapi/setvars.sh`) is required before device tools (`xpu-smi`) see the GPU;
  `sycl-ls` is the reliable probe for the Level-Zero devices.
- Custom kernel gotchas (see `packages/xpu-ltx-kernels/docs/capability-report.md`): oneAPI 2026.1 ESIMD
  headers have N=1 scalar-math compile bugs; torch bf16 maps to `sycl::ext::oneapi::bfloat16` (not
  `sycl::half`, which is fp16); subgroup `reduce_over_group` and local-memory atomics are unreliable on
  this runtime — use barrier-tree reductions; one workgroup per row (never `global_id` as the row index);
  no early-returns before a group reduction.

## Tests

No `test_*.py` files are present in this vendored tree, so `pytest` is not runnable here. If you add code,
follow the testing standards in `packages/ltx-trainer/AGENTS.md` (flat `test_*` functions, test public
interfaces only, behavioral tests over config-only tests).
