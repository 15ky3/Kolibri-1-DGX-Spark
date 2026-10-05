#!/usr/bin/env bash
# download.sh — fetch the checkpoint QUANT selects into $HF_HOME/hub
# (nvfp4: iSkye/Kolibri-1-NVFP4-Experts ~46 GB; fp8: Aleph-Alpha/Kolibri-1 ~79 GB;
#  ABLIT=1: iSkye/Kolibri-1-heretic ~80 GB). A complete local copy in
#  $LOCAL_MODELS/<repo name> counts as downloaded.
# Resumable: rerun after an interruption and it continues where it stopped.
# Uses the host `hf` CLI; falls back to the one inside the vLLM image.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

REV_ARGS=(); [[ -n "${MODEL_REVISION:-}" ]] && REV_ARGS=(--revision "$MODEL_REVISION")
# first NON-EMPTY token wins: an empty $HF_HOME/token would otherwise mask the
# login in ~/.cache/huggingface (public repos work without one, just slower)
TOKEN="${HF_TOKEN:-}"
for f in "$HF_HOME/token" "$HOME/.cache/huggingface/token"; do
    [[ -z "$TOKEN" && -s "$f" ]] && TOKEN="$(cat "$f")"
done

if [[ -z "$(resolve_snapshot)" && -f "$LOCAL_COPY/model.safetensors.index.json" ]]; then
    info "$MODEL_ID is already here as a local copy ($LOCAL_COPY) — start.sh uses it, nothing to download"
    exit 0
fi
info "model  $MODEL_ID ${MODEL_REVISION:+@ $MODEL_REVISION} (QUANT=$QUANT ABLIT=$ABLIT, ~${CHECKPOINT_GB} GB)"
info "cache  $HF_HOME"
avail=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc 0-9)
(( avail >= CHECKPOINT_GB + 5 )) || warn "only ${avail} GB free on $HF_HOME (the checkpoint is ~${CHECKPOINT_GB} GB)"

if command -v hf >/dev/null; then
    HF_HOME="$HF_HOME" HF_TOKEN="$TOKEN" HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-}" \
        hf download "$MODEL_ID" "${REV_ARGS[@]}"
else
    info "no host hf CLI, using the one in ${IMAGE:-vllm/vllm-openai:v0.30.0}"
    docker run --rm -e HF_HOME=/hf -e HF_TOKEN="$TOKEN" -v "$HF_HOME:/hf" \
        --entrypoint hf "${IMAGE:-vllm/vllm-openai:v0.30.0}" download "$MODEL_ID" "${REV_ARGS[@]}"
fi

SNAP="$(resolve_snapshot)"
[[ -n "$SNAP" ]] || err "download finished but no snapshot found under $MODEL_CACHE_DIR"
info "done: $SNAP ($(du -shL "$SNAP" | cut -f1))"
