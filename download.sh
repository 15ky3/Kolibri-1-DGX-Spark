#!/usr/bin/env bash
# download.sh — fetch the Kolibri-1 checkpoint (~79 GB) into $HF_HOME/hub.
# Resumable: rerun after an interruption and it continues where it stopped.
# Uses the host `hf` CLI; falls back to the one inside the vLLM image.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

REV_ARGS=(); [[ -n "${MODEL_REVISION:-}" ]] && REV_ARGS=(--revision "$MODEL_REVISION")
TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"

info "model  $MODEL_ID ${MODEL_REVISION:+@ $MODEL_REVISION}"
info "cache  $HF_HOME"
avail=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc 0-9)
(( avail >= 85 )) || warn "only ${avail} GB free on $HF_HOME (the checkpoint is ~79 GB)"

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
