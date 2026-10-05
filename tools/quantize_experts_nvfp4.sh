#!/usr/bin/env bash
# quantize_experts_nvfp4.sh — build an NVFP4-experts checkpoint from an FP8 one.
# The source follows ABLIT: 0 = Aleph-Alpha/Kolibri-1, 1 = iSkye/Kolibri-1-heretic
# (hub cache or the local copy in ~/models/<repo name>).
# The server must be stopped (the conversion uses the GPU and ~10 GiB of memory).
# Usage: [ABLIT=1] tools/quantize_experts_nvfp4.sh [OUT_DIR] [extra args, e.g. --layers 0]
set -euo pipefail
export QUANT=fp8 MODEL_ID=   # the source is always an FP8 checkpoint
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"
if [[ "$ABLIT" == 1 ]]; then DEF_OUT="$LOCAL_MODELS/Kolibri-1-heretic-NVFP4-Experts"
else DEF_OUT="$LOCAL_MODELS/Kolibri-1-NVFP4-Experts"; fi
OUT="${1:-$DEF_OUT}"; shift || true
SNAP="$(resolve_snapshot)"
[[ -z "$SNAP" && -f "$LOCAL_COPY/model.safetensors.index.json" ]] && SNAP="$LOCAL_COPY"
[[ -n "$SNAP" && -f "$SNAP/model.safetensors.index.json" ]] || err "FP8 checkpoint $MODEL_ID not found (hub cache or $LOCAL_COPY)"
[[ -z "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]] || err "$CONTAINER_NAME is running — stop it first"
mkdir -p "$OUT"
info "source $SNAP"
info "output $OUT"
docker run --rm --gpus all --ipc host --name kolibri-nvfp4-quant \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HF_HOME:$HF_HOME:ro" -v "$SNAP:$SNAP:ro" -v "$OUT:/out" -v "$RECIPE_DIR/tools:/tools:ro" \
    --entrypoint python3 "${IMAGE:-vllm/vllm-openai:v0.30.0}" \
    /tools/quantize_experts_nvfp4.py --src "$SNAP" --dst /out "$@"
