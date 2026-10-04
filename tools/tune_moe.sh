#!/usr/bin/env bash
# tune_moe.sh — re-tune the E=384 Triton MoE tile table on this box.
# The GPU must be idle (stop the server first). Writes
# files/moe_configs/tuned/<table>.json + .csv; review, then copy over the live table.
# Usage: tools/tune_moe.sh [batch sizes...]   (default: 1 2 4 8 16 24 32 48 64)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"
TABLE='E=384,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'
mkdir -p "$RECIPE_DIR/files/moe_configs/tuned"
[[ -z "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]] || err "$CONTAINER_NAME is running — stop it first"
docker run --rm --gpus all --ipc host --name kolibri-moe-tune \
    -v "$RECIPE_DIR/tools:/tools:ro" \
    -v "$RECIPE_DIR/files/moe_configs:/cfg" \
    --entrypoint python3 "${IMAGE:-vllm/vllm-openai:v0.30.0}" \
    /tools/tune_moe.py --base "/cfg/$TABLE" --out "/cfg/tuned/$TABLE" ${1:+--batch "$@"}
