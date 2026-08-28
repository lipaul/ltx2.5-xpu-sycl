#!/usr/bin/env bash
# uv sync for the XPU pipeline stack (torch 2.13.0+xpu pins in ltx-core).
# After this, the uv venv has torch 2.13.0+xpu + intel-sycl-rt 2026.0.0,
# matching the installed oneAPI 2026.1 compiler (required for the SYCL build).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v uv >/dev/null || { echo "ERROR: uv not found" >&2; exit 1; }
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true   # rc=3 warning is benign

echo "[uv-sync] resolving XPU workspace (torch 2.13.0+xpu)"
uv sync

.venv/bin/python - <<'EOF' || { echo "ERROR: XPU torch not installed (see packages/ltx-core/pyproject.toml pins)" >&2; exit 1; }
import importlib.metadata as md
import torch
assert torch.__version__ == "2.13.0+xpu", torch.__version__
assert md.version("intel-sycl-rt").startswith("2026"), md.version("intel-sycl-rt")
print("torch", torch.__version__, "| runtime", md.version("intel-sycl-rt"), "| xpu", torch.xpu.is_available())
EOF

echo "[uv-sync] done"