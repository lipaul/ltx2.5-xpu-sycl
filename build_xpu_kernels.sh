#!/usr/bin/env bash
# Build packages/xpu-ltx-kernels (SYCL/ESIMD, icpx) into the uv venv.
# Prereqs: uv_sync.sh already run (torch 2.13.0+xpu), oneAPI 2026.1 on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PY="${REPO_ROOT}/.venv/bin/python"

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true   # rc=3 warning is benign
command -v icpx >/dev/null || { echo "ERROR: icpx not on PATH (source /opt/intel/oneapi/setvars.sh)" >&2; exit 1; }
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"

echo "[build-xpu-kernels] icpx compile + device-link into the uv venv"
uv pip install -e packages/xpu-ltx-kernels --python "$PY" --no-build-isolation

"$PY" - <<'EOF' || { echo "ERROR: xpu_ltx_kernels not importable on XPU" >&2; exit 1; }
import xpu_ltx_kernels
assert xpu_ltx_kernels.available(), "extension built but not importable/usable"
print("xpu_ltx_kernels OK")
EOF

echo "[build-xpu-kernels] done"