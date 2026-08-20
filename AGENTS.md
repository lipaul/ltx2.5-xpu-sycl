# AGENTS.md

LTX-2: a PyTorch audio-video diffusion model. This working tree is a vendored snapshot of
[`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2), pushed to a personal fork
(`lipaul/ltx-2-xpu`, `git remote -v`). It is a `uv` workspace monorepo of four packages.

## Per-package instruction files (read these first)

These already encode hard-won, repo-specific guidance. Read the relevant one before editing:

- `packages/ltx-trainer/AGENTS.md` — training package: config conventions, `configs/` doc sync rule,
  frame/resolution constraints (`frames % T == 1`), split-vs-unified checkpoint layouts, testing standards.
- `packages/ltx-pipelines/CLAUDE.md` — inference pipelines: pipeline selection table, sigma schedules,
  LoRA conventions, `utils/blocks.py` memory model, DFR invariants, generated-keyframes rules.
- `packages/ltx-kernels/README.md` and `packages/ltx-core/README.md` — CUDA extension layout and model components.

Keep changes to each package consistent with its own instruction file.

## Package layout

- `packages/ltx-core/` — model definitions (transformer, VAEs, vocoder, Gemma text encoder), loading, quantization. Depended on by both other packages.
- `packages/ltx-pipelines/` — ready-made inference pipelines (`ltx_pipelines.*`, run as `python -m ltx_pipelines.<module>`).
- `packages/ltx-trainer/` — LoRA / full fine-tuning (`scripts/train.py`, `configs/*.yaml`).
- `packages/ltx-kernels/` — optional CUDA/C++ extensions (all2all, blockwise FP8/FP6 GEMM, NVFP4, CuTe DSL VAE kernels).

## Toolchain (uv + torch XPU)

This fork runs the inference pipeline on **Intel XPU** (Linux). The environment is resolved entirely via `uv`
against the official PyTorch **XPU** index — no CUDA.

- Use `uv` everywhere: `uv sync`, `uv run`. Do not use bare `pip` or a system python.
- torch/torchaudio are pinned to the **XPU** index in `packages/ltx-core/pyproject.toml`:
  `torch==2.12.0+xpu`, `torchaudio==2.11.0+xpu`, plus `triton-xpu==3.7.1` (declared as a *direct* dep — uv only
  applies `[tool.uv.sources]` index mappings to direct deps, so the transitive `triton-xpu` would otherwise
  resolve from PyPI at a conflicting version).
- **`transformers` is capped at `<5.15`** (in `ltx-core` deps). 5.15.0+ breaks the Gemma 4 text encoder build.
- `ltx-core` bounds `requires-python = ">=3.10,<3.14"`: `triton-xpu==3.7.1` has no free-threaded 3.14
  (cp314t) Windows wheel, so an unbounded range makes the universal lock unresolvable.
- **`ltx-trainer` and `ltx-kernels` are excluded from the uv workspace** (`[tool.uv.workspace].exclude`). The
  trainer pins CUDA-only `torchcodec`; the kernels are CUDA extensions. A plain `uv sync` installs only
  `ltx-core` + `ltx-pipelines`, the pipeline stack.
- Fresh clone is shallow; `git fetch --unshallow origin` before pushing a branch that depends on full history.

## Code quality gates

Ruff is configured at the root (`pyproject.toml` `[tool.ruff]`): line length **120**, strict ruleset
(ANN, B, PL, SIM, ...), `known-first-party = ["ltx_core", "ltx_pipelines", "ltx_trainer"]`. Run:

```bash
uv run ruff check .
uv run ruff format .
```

There is **no `.pre-commit-config.yaml`** in this tree — `pre-commit run` is not a working gate; ruff is the gate.

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
  lockfile; pinned on the XPU index as `torchvision==0.27.0+xpu` (0.27.1 requires torch 2.12.1, conflicting
  with the `torch==2.12.0+xpu` pin).
- Sourcing oneAPI (`/opt/intel/oneapi/setvars.sh`) is required before device tools (`xpu-smi`) see the GPU;
  `sycl-ls` is the reliable probe for the Level-Zero devices.

## Tests

No `test_*.py` files are present in this vendored tree, so `pytest` is not runnable here. If you add code,
follow the testing standards in `packages/ltx-trainer/AGENTS.md` (flat `test_*` functions, test public
interfaces only, behavioral tests over config-only tests).
