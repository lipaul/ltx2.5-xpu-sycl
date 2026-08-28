#!/usr/bin/env bash
# Run the LTX-2.5 distilled pipeline on XPU with `--quantization blockwise`
# (fp8 blockwise weights -> fits the 24 GB B60 without --offload; the fused
# rms_norm_rope / fp8-storage GEMM come from xpu-ltx-kernels).
#
# Usage:
#   ./run_xpu_blockwise.sh "PROMPT" OUTPUT.mp4 [extra pipeline args...]
# Env: LTX_MODEL_ROOT (default /home/lm/work/models/ltx-2.5)
# Prereqs: uv_sync.sh + build_xpu_kernels.sh have been run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${LTX_MODEL_ROOT:-/home/lm/work/models/ltx-2.5}"
PY="${REPO_ROOT}/.venv/bin/python"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"PROMPT\" OUTPUT.mp4 [extra pipeline args...]" >&2
    exit 1
fi
PROMPT="$1"; OUTPUT="$2"; shift 2

MODELS_DIR="$MODEL_ROOT/diffusion_models"
TEXT_DIR="$MODEL_ROOT/text_encoders"
VAE_DIR="$MODEL_ROOT/vae"
UPSCALE_DIR="$MODEL_ROOT/latent_upscale_models"
PATCH_DIR="$MODEL_ROOT/model_patches"

for f in \
    "$MODELS_DIR/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    "$TEXT_DIR/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "$VAE_DIR/ltx-2.5-video-vae-bf16.safetensors" \
    "$VAE_DIR/ltx-2.5-audio-vae-bf16.safetensors" \
    "$UPSCALE_DIR/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"; do
    [[ -f "$f" ]] || { echo "ERROR: missing model file: $f (set LTX_MODEL_ROOT)" >&2; exit 1; }
done

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true   # rc=3 warning is benign
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"

cd "$REPO_ROOT"
exec "$PY" -u -m ltx_pipelines.distilled \
    --transformer-path "$MODELS_DIR/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$TEXT_DIR/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$VAE_DIR/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path "$VAE_DIR/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$UPSCALE_DIR/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --duration-head-path "$PATCH_DIR/ltx-2.5-duration-head-bf16.safetensors" \
    --quantization blockwise \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT" \
    "$@"