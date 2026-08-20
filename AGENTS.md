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

## Tests

No `test_*.py` files are present in this vendored tree, so `pytest` is not runnable here. If you add code,
follow the testing standards in `packages/ltx-trainer/AGENTS.md` (flat `test_*` functions, test public
interfaces only, behavioral tests over config-only tests).
