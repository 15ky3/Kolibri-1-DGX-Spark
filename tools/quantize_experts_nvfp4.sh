#!/usr/bin/env bash
# quantize_experts_nvfp4.sh — build the NVFP4-experts checkpoint from the FP8 one.
# The server must be stopped (the conversion uses the GPU and ~10 GiB of memory).
# Usage: tools/quantize_experts_nvfp4.sh [OUT_DIR] [extra args, e.g. --layers 0]
set -euo pipefail
export QUANT=fp8 MODEL_ID=   # the source is always the original FP8 checkpoint
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"
OUT="${1:-$HOME/models/Kolibri-1-NVFP4-experts}"; shift || true
SNAP="$(resolve_snapshot)"
[[ -n "$SNAP" && -f "$SNAP/model.safetensors.index.json" ]] || err "FP8 checkpoint not found ($MODEL_CACHE_DIR)"
[[ -z "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]] || err "$CONTAINER_NAME is running — stop it first"
mkdir -p "$OUT"
info "source $SNAP"
info "output $OUT"
docker run --rm --gpus all --ipc host --name kolibri-nvfp4-quant \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HF_HOME:$HF_HOME:ro" -v "$OUT:/out" -v "$RECIPE_DIR/tools:/tools:ro" \
    --entrypoint python3 "${IMAGE:-vllm/vllm-openai:v0.30.0}" \
    /tools/quantize_experts_nvfp4.py --src "$SNAP" --dst /out "$@"
