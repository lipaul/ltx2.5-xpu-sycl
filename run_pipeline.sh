#!/usr/bin/env bash
# Launch the LTX-2.5 distilled inference pipeline on Intel XPU.
#
# Usage:
#   ./run_pipeline.sh "A cinematic aerial shot of a mountain lake at sunrise" out.mp4 [extra args...]
#
# All extra args are forwarded to `python -m ltx_pipelines.distilled`.
# Override the model root / pipeline with env vars:
#   LTX_MODEL_ROOT=/path/to/models     PIPELINE=distilled
#
# Device is auto-selected (XPU is probed via torch.xpu.is_available()).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${LTX_MODEL_ROOT:-/home/acm/paul/models/ltx-2.5}"
PIPELINE="${PIPELINE:-distilled}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"PROMPT\" OUTPUT.mp4 [extra pipeline args...]" >&2
    echo "  e.g. $0 \"A cat playing piano\" cat.mp4 --num-frames 121" >&2
    exit 1
fi

PROMPT="$1"
OUTPUT="$2"
shift 2

MODELS_DIR="$MODEL_ROOT/diffusion_models"
TEXT_DIR="$MODEL_ROOT/text_encoders"
VAE_DIR="$MODEL_ROOT/vae"
UPSCALE_DIR="$MODEL_ROOT/latent_upscale_models"
PATCH_DIR="$MODEL_ROOT/model_patches"

# Fail fast if the split-layout model files are missing.
required=(
    "$MODELS_DIR/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    "$TEXT_DIR/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    "$VAE_DIR/ltx-2.5-video-vae-bf16.safetensors"
    "$VAE_DIR/ltx-2.5-audio-vae-bf16.safetensors"
    "$UPSCALE_DIR/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing model file: $f" >&2
        exit 1
    fi
done

echo "Running $PIPELINE pipeline on XPU..."
echo "  prompt : $PROMPT"
echo "  output : $OUTPUT"

cd "$REPO_ROOT"
exec uv run python -m "ltx_pipelines.$PIPELINE" \
    --transformer-path "$MODELS_DIR/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$TEXT_DIR/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$VAE_DIR/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path "$VAE_DIR/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$UPSCALE_DIR/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --duration-head-path "$PATCH_DIR/ltx-2.5-duration-head-bf16.safetensors" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT" \
    "$@"
