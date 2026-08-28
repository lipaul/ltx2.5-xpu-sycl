#!/usr/bin/env bash
# Measure per-stage XPU compute + VRAM utilization for the two distilled-pipeline
# modes A/B'd by reproduce_uv_bench.sh, then merge into a comparison report:
#
#   baseline : bf16 weights + `--offload cpu`            (no xpu-ltx-kernels)
#   blockwise: `--quantization blockwise` (fp8 storage, no offload)
#
# Env overrides:
#   LTX_MODEL_ROOT  model root (default /home/lm/work/models/ltx-2.5)
#   FRAMES          video frames (default 9; quick smoke, ~1-2 min per mode)
#   SEED            default 42
#   HEIGHT / WIDTH  output resolution (default 512x768; stage 1 is half)
#   PROMPT          default "A red ball bouncing on a green lawn, camera static."
#   PROFILE_DENOISE 1 to also profiler the two denoising stages
#
# Requires: uv, an Intel XPU (B60), models in LTX_MODEL_ROOT.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${LTX_MODEL_ROOT:-/home/lm/work/models/ltx-2.5}"
FRAMES="${FRAMES:-9}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-768}"
PROMPT="${PROMPT:-A red ball bouncing on a green lawn, camera static.}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/bench}"
PY="${REPO_ROOT}/.venv/bin/python"
MEASURE="${REPO_ROOT}/bench/xpu_utilization.py"

log() { echo -e "\033[1;34m[$1]\033[0m $2"; }
fail() { echo -e "\033[1;31mERROR:\033[0m $1" >&2; exit 1; }

for f in \
    "$MODEL_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    "$MODEL_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "$MODEL_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
    "$MODEL_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    "$MODEL_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"; do
    [[ -f "$f" ]] || fail "missing model file: $f (set LTX_MODEL_ROOT)"
done

command -v uv >/dev/null || fail "uv not found"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
mkdir -p "$OUTDIR"

ARGS=(--model-root "$MODEL_ROOT" --num-frames "$FRAMES" --seed "$SEED")
ARGS+=(--height "$HEIGHT" --width "$WIDTH" --prompt "$PROMPT")
[[ "${PROFILE_DENOISE:-0}" == "1" ]] && ARGS+=(--profile-denoise)

log "1/3" "baseline (bf16 + --offload cpu)"
"$PY" "$MEASURE" run --mode baseline --json-out "$OUTDIR/baseline_util.json" \
    --output-path "$OUTDIR/baseline_util.mp4" "${ARGS[@]}"

log "2/3" "blockwise (fp8 blockwise, no offload)"
"$PY" "$MEASURE" run --mode blockwise --json-out "$OUTDIR/blockwise_util.json" \
    --output-path "$OUTDIR/blockwise_util.mp4" "${ARGS[@]}"

log "3/3" "merge report"
"$PY" "$MEASURE" merge "$OUTDIR/baseline_util.json" "$OUTDIR/blockwise_util.json" \
    --out "$OUTDIR/utilization_report.md"

echo
echo "report: $OUTDIR/utilization_report.md"