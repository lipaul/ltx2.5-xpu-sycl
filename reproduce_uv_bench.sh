#!/usr/bin/env bash
# Reproduce the xpu-ltx-kernels work under uv and A/B the two inference modes.
#
#   1. `uv sync`  — XPU torch pins (torch 2.13.0+xpu, intel-sycl-rt 2026.0.0)
#   2. build `packages/xpu-ltx-kernels` into the uv venv with icpx
#   3. run the distilled pipeline in both modes, same prompt/seed/frames:
#        - baseline : bf16 weights + `--offload cpu` (the pre-blockwise way)
#        - blockwise: `--quantization blockwise` (fp8 blockwise storage, no offload)
#   4. extract per-stage denoising step times and print a comparison table
#
# Env overrides:
#   LTX_MODEL_ROOT  model root (default /home/lm/work/models/ltx-2.5)
#   FRAMES          video frames (default 33; use 9 for a quick smoke test)
#   SEED            default 42
#   OUTDIR          output dir for videos/logs (default /tmp/opencode/bench)
#   MAX_WAIT        seconds to wait per pipeline run (default 900)
#
# Requires: uv, icpx (oneAPI 2026.1), an Intel XPU (B60), models in LTX_MODEL_ROOT.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${LTX_MODEL_ROOT:-/home/lm/work/models/ltx-2.5}"
FRAMES="${FRAMES:-33}"
SEED="${SEED:-42}"
OUTDIR="${OUTDIR:-/tmp/opencode/bench}"
MAX_WAIT="${MAX_WAIT:-900}"
PROMPT="${PROMPT:-A red ball bouncing on a green lawn, camera static.}"
PY="${REPO_ROOT}/.venv/bin/python"

TRANSFORMER="${MODEL_ROOT}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENC="${MODEL_ROOT}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE="${MODEL_ROOT}/vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE="${MODEL_ROOT}/vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL="${MODEL_ROOT}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

log() { echo -e "\033[1;34m[$1]\033[0m $2"; }

fail() { echo -e "\033[1;31mERROR:\033[0m $1" >&2; exit 1; }

# --- 0. precondition checks --------------------------------------------------
for f in "$TRANSFORMER" "$TEXT_ENC" "$VIDEO_VAE" "$AUDIO_VAE" "$SPATIAL"; do
    [[ -f "$f" ]] || fail "missing model file: $f (set LTX_MODEL_ROOT)"
done
command -v uv >/dev/null || fail "uv not found"
# setvars.sh exits non-zero on a benign 32-bit-libraries warning (oneAPI 2025+);
# the env is still set up, so verify icpx instead of the exit code.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
command -v icpx >/dev/null || fail "icpx not on PATH (source /opt/intel/oneapi/setvars.sh)"
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
mkdir -p "$OUTDIR"

# --- 1. uv sync --------------------------------------------------------------
log "1/5" "uv sync (XPU torch pins, venv: ${REPO_ROOT}/.venv)"
( cd "$REPO_ROOT" && uv sync )

"$PY" - <<'EOF' || fail "uv venv does not have XPU torch (torch 2.13.0+xpu / intel-sycl-rt 2026.0.0)"
import torch, importlib.metadata as md
assert torch.__version__ == "2.13.0+xpu", torch.__version__
assert md.version("intel-sycl-rt").startswith("2026"), md.version("intel-sycl-rt")
assert torch.xpu.is_available(), "no XPU device"
print("torch", torch.__version__, "| runtime", md.version("intel-sycl-rt"), "| xpu ok")
EOF

# --- 2. build xpu-ltx-kernels -------------------------------------------------
log "2/5" "build packages/xpu-ltx-kernels into the uv venv (icpx)"
( cd "$REPO_ROOT" && uv pip install -e packages/xpu-ltx-kernels --python "$PY" --no-build-isolation )

"$PY" -c "import xpu_ltx_kernels; assert xpu_ltx_kernels.available(), 'extension not importable'; print('xpu_ltx_kernels OK')" \
    || fail "xpu_ltx_kernels did not build/import (see capability-report.md)"

# --- 3+4. run both modes ------------------------------------------------------
run_pipeline() {  # $1=mode  $2=extra-args...  (logs to $OUTDIR/$mode.log, pid to $OUTDIR/$mode.pid)
    local mode="$1"; shift
    log "3-4/5" "run $mode (frames=$FRAMES seed=$SEED)"
    rm -f "$OUTDIR/$mode.log" "$OUTDIR/$mode.pid"
    nohup env SYCL_DEVICE_FILTER="$SYCL_DEVICE_FILTER" "$PY" -u -m ltx_pipelines.distilled \
        --transformer-path "$TRANSFORMER" --text-encoder-path "$TEXT_ENC" \
        --video-vae-path "$VIDEO_VAE" --audio-vae-path "$AUDIO_VAE" \
        --spatial-upsampler-path "$SPATIAL" \
        --num-frames "$FRAMES" --seed "$SEED" \
        --output-path "$OUTDIR/$mode.mp4" --prompt "$PROMPT" \
        "$@" > "$OUTDIR/$mode.log" 2>&1 &
    echo $! > "$OUTDIR/$mode.pid"
    # Wait for completion (video saved or process exit), bounded by MAX_WAIT.
    local waited=0
    while kill -0 "$(cat "$OUTDIR/$mode.pid")" 2>/dev/null; do
        if grep -q "Video saved" "$OUTDIR/$mode.log" 2>/dev/null; then break; fi
        sleep 10; waited=$((waited + 10))
        if (( waited >= MAX_WAIT )); then fail "$mode did not finish in ${MAX_WAIT}s; tail: $(tail -2 "$OUTDIR/$mode.log")"; fi
    done
    grep -q "Video saved" "$OUTDIR/$mode.log" || fail "$mode failed (see $OUTDIR/$mode.log)"
    echo "  -> $OUTDIR/$mode.mp4 saved"
}

# extract final per-stage step time and total elapsed from a tqdm log.
# emits "sit total_s"; if the last bar is partial (k/N, k<N) the final step is
# extrapolated (tqdm's 100% line is often not flushed to a redirected log).
stage_times() {  # $1=log  $2=stage-steps (8 or 3)
    local line sit mm ss total
    line=$(grep -a -oE "[0-9]+/$2 \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]" "$1" | tail -1)
    [[ -z "$line" ]] && { echo "0 0"; return 0; }
    local steps=$(echo "$line" | sed -E 's#^([0-9]+)/.*#\1#')
    local mm=$(echo "$line" | sed -E 's/.*\[([0-9]+):[0-9]+<.*/\1/')
    local ss=$(echo "$line" | sed -E 's/.*\[[0-9]+:([0-9]+)<.*/\1/')
    sit=$(echo "$line" | sed -E 's/.*, +([0-9.]+)s\/it\].*/\1/')
    total=$((mm * 60 + ss))
    if (( steps < $2 )); then total=$(echo "$total + $sit" | bc); fi
    echo "$sit $total"
}

run_pipeline baseline --offload cpu
run_pipeline blockwise --quantization blockwise

# --- 5. compare ---------------------------------------------------------------
log "5/5" "compare"
BASE="$OUTDIR/baseline.log"; BLK="$OUTDIR/blockwise.log"
b1=($(stage_times "$BASE" 8)); w1=($(stage_times "$BLK" 8))
b2=($(stage_times "$BASE" 3)); w2=($(stage_times "$BLK" 3))

echo
printf "%-10s %-26s %-26s %s\n" "stage" "baseline bf16 --offload cpu" "blockwise fp8 (no offload)" "speedup"
printf "%-10s %-12s %-12s %-12s %-12s %s\n" "" "s/it" "total" "s/it" "total" ""
printf "%-10s %-12s %-12s %-12s %-12s %s\n" "stage 1 (${FRAMES}f)" "${b1[0]}" "${b1[1]}s" "${w1[0]}" "${w1[1]}s" "$(echo "scale=2; ${b1[1]} / ${w1[1]}" | bc)x"
printf "%-10s %-12s %-12s %-12s %-12s %s\n" "stage 2" "${b2[0]}" "${b2[1]}s" "${w2[0]}" "${w2[1]}s" "$(echo "scale=2; ${b2[1]} / ${w2[1]}" | bc)x"
bt=$(echo "${b1[1]} + ${b2[1]}" | bc); wt=$(echo "${w1[1]} + ${w2[1]}" | bc)
printf "%-10s %-12s %-12s %-12s %-12s %s\n" "denoising" "" "${bt}s" "" "${wt}s" "$(echo "scale=2; $bt / $wt" | bc)x"
echo
echo "videos: $OUTDIR/baseline.mp4  $OUTDIR/blockwise.mp4"
echo "logs:   $OUTDIR/baseline.log  $OUTDIR/blockwise.log"