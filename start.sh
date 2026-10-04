#!/usr/bin/env bash
# start.sh — serve Aleph-Alpha/Kolibri-1 (FP8) on ONE DGX Spark with stock vLLM 0.30.0.
#
# Usage: ./start.sh [--no-launch] [--no-wait] [--force]
#   --no-launch  print the memory budget and the docker command, run nothing
#   --no-wait    return right after the container is up (do not poll /health)
#   --force      skip the free-memory and port pre-flight checks
#
# Never downloads: the checkpoint must already be in $HF_HOME (./download.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
cd "$RECIPE_DIR"

NO_LAUNCH=false; NO_WAIT=false; FORCE=false
for a in "$@"; do
    case "$a" in
        --no-launch) NO_LAUNCH=true ;;
        --no-wait)   NO_WAIT=true ;;
        --force)     FORCE=true ;;
        -h|--help)   sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           err "unknown option: $a (try --help)" ;;
    esac
done

PORT="${PORT:-8895}"; BIND="${BIND:-0.0.0.0}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kolibri-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/vllm}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-$HOME/.cache/flashinfer}"

# ---------------------------------------------------------------- pre-flight
command -v docker >/dev/null || err "docker not found"
if [[ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]] && ! $NO_LAUNCH; then
    err "$CONTAINER_NAME is already running (./stop.sh first)"
fi

# --no-launch only reports; everything else refuses to start on a bad cache
need() { if $NO_LAUNCH; then warn "$*"; else err "$*"; fi; }
# MODEL_DIR serves a local checkpoint directory (e.g. the NVFP4-experts build
# from tools/quantize_experts_nvfp4.sh) instead of the hub snapshot.
if [[ -n "${MODEL_DIR:-}" ]]; then
    [[ -d "$MODEL_DIR" ]] || err "MODEL_DIR not found: $MODEL_DIR"
    SNAP="$(cd "$MODEL_DIR" && pwd)"
else
    SNAP="$(resolve_snapshot)"
fi
[[ -n "$SNAP" ]] || { need "checkpoint not in cache: $MODEL_CACHE_DIR (run ./download.sh)"; SNAP=unknown; }
if [[ ! -f "$SNAP/model.safetensors.index.json" ]]; then
    need "incomplete snapshot (no index yet): $SNAP — download still running? rerun ./download.sh"
else
    missing=$(python3 - "$SNAP" <<'PY'
import json, os, sys
snap = sys.argv[1]
shards = set(json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"].values())
need = shards | {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}
print(" ".join(sorted(f for f in need if not os.path.exists(os.path.join(snap, f)))))
PY
)
    [[ -z "$missing" ]] || need "snapshot is missing $(wc -w <<<"$missing") files (download still running?)"
fi
if [[ -z "${MODEL_DIR:-}" ]] && compgen -G "$MODEL_CACHE_DIR/blobs/*.incomplete" >/dev/null; then
    warn "partial blobs in $MODEL_CACHE_DIR/blobs — is a download still running?"
fi
REV="$(basename "$SNAP")"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    info "image $IMAGE not present, pulling"; docker pull "$IMAGE"; }

# ---------------------------------------------------------------- memory budget
MEM_TOTAL=$(mem_gib MemTotal); MEM_AVAIL=$(mem_gib MemAvailable)
BUDGET=$(awk -v w="${WEIGHTS_GIB:-73.4}" -v k="${KV_CACHE_GIB:-8}" -v o="${OVERHEAD_GIB:-7}" \
             'BEGIN{printf "%.2f", w+k+o}')
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    GMU="$GPU_MEMORY_UTILIZATION"
    BUDGET=$(awk -v g="$GMU" -v t="$MEM_TOTAL" 'BEGIN{printf "%.2f", g*t}')
    warn "GPU_MEMORY_UTILIZATION pinned to $GMU (= $BUDGET GiB), derived budget ignored"
else
    GMU=$(awk -v b="$BUDGET" -v t="$MEM_TOTAL" 'BEGIN{printf "%.3f", b/t}')
fi
NEED=$(awk -v b="$BUDGET" -v h="${HOST_MIN_FREE_GIB:-10}" 'BEGIN{printf "%.2f", b+h}')
KV_BYTES=$(awk -v k="${KV_CACHE_GIB:-8}" 'BEGIN{printf "%d", k*1073741824}')

# KV capacity: measured 666,780 tokens in 8 GiB on vLLM 0.30 (fp8 KV, 4 sliding
# + 1 full attention group of 10 layers each) = ~83,000 tokens per GiB. vLLM
# refuses to start when one max-length request does not fit, so say it here.
KV_TOKENS=$(awk -v k="${KV_CACHE_GIB:-8}" 'BEGIN{printf "%d", k*83000}')
if (( KV_TOKENS < MAX_MODEL_LEN )); then
    need_kv=$(awk -v n="$MAX_MODEL_LEN" 'BEGIN{printf "%d", n/83000 + 1}')
    need "KV_CACHE_GIB=${KV_CACHE_GIB:-8} holds ~$KV_TOKENS tokens, less than one MAX_MODEL_LEN=$MAX_MODEL_LEN request — set KV_CACHE_GIB >= $need_kv"
fi

if [[ -n "${MODEL_DIR:-}" ]]; then info "model    $SNAP (MODEL_DIR)"; else info "model    $MODEL_ID @ ${REV:0:12}"; fi
info "image    $IMAGE"
info "memory   MemTotal $MEM_TOTAL GiB, MemAvailable $MEM_AVAIL GiB"
info "kv       ~$KV_TOKENS tokens = $(awk -v t="$KV_TOKENS" -v n="$MAX_MODEL_LEN" 'BEGIN{printf "%.2f", t/n}') x MAX_MODEL_LEN"
info "budget   weights ${WEIGHTS_GIB:-73.4} + KV ${KV_CACHE_GIB:-8} + overhead ${OVERHEAD_GIB:-7} = $BUDGET GiB -> GMU $GMU"
info "needs    $NEED GiB available (budget + HOST_MIN_FREE_GIB ${HOST_MIN_FREE_GIB:-10})"

if ! $FORCE && ! $NO_LAUNCH; then
    awk -v a="$MEM_AVAIL" -v n="$NEED" 'BEGIN{exit !(a+0 >= n+0)}' || err \
"only $MEM_AVAIL GiB available, need $NEED. Another model is probably loaded:
$(docker ps --format '   {{.Names}}  ({{.Image}})' | grep -v "^   $CONTAINER_NAME " || true)
Stop it first, or lower KV_CACHE_GIB / HOST_MIN_FREE_GIB, or --force."
    if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
        err "port $PORT is already in use"
    fi
fi

# ---------------------------------------------------------------- vllm serve args
if [[ -n "${MODEL_DIR:-}" ]]; then
    MODEL_ARG=/models/local
else
    MODEL_ARG="$MODEL_ID"
fi
ARGS=(
    "$MODEL_ARG"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$BIND" --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "$GMU"
    --kv-cache-memory-bytes "$KV_BYTES"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
    --enable-chunked-prefill
    --enable-prompt-tokens-details
    --load-format safetensors
)
[[ -z "${MODEL_DIR:-}" ]] && ARGS+=(--revision "$REV")
if (( MAX_MODEL_LEN > 262144 )); then
    # the model card's extrapolation recipe; recommended <= 262144 for quality
    ARGS+=(--hf-overrides "{\"max_position_embeddings\": $MAX_MODEL_LEN}")
fi
[[ "${ENABLE_PREFIX_CACHING:-1}" == 1 ]] && ARGS+=(--enable-prefix-caching) || ARGS+=(--no-enable-prefix-caching)
[[ "${ENFORCE_EAGER:-0}" == 1 ]] && ARGS+=(--enforce-eager)
[[ "${REASONING:-1}" == 1 ]] && ARGS+=(--reasoning-parser kolibri1)
[[ "${TOOLS:-1}" == 1 ]] && ARGS+=(--tool-call-parser kolibri1 --enable-auto-tool-choice)
[[ -n "${ATTENTION_BACKEND:-}" ]] && ARGS+=(--attention-backend "$ATTENTION_BACKEND")
[[ -n "${MOE_BACKEND:-}" ]] && ARGS+=(--moe-backend "$MOE_BACKEND")
[[ -n "${LOAD_STRATEGY:-}" ]] && ARGS+=(--safetensors-load-strategy "$LOAD_STRATEGY")
[[ -n "${LINEAR_BACKEND:-}" ]] && ARGS+=(--linear-backend "$LINEAR_BACKEND")
case "${SPEC:-off}" in
    off|"") ;;
    # prompt-lookup drafts: 2.3x on copy-heavy output (code edits, quoting),
    # but it halves free-form prose (V1 runner, no async scheduling, and every
    # rejected draft is a 5-token MoE verify) — opt-in only
    ngram) ARGS+=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${SPEC_K:-4},\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}") ;;
    *) err "SPEC must be off or ngram (got: $SPEC)" ;;
esac
[[ -n "${API_KEY:-}" ]] && ARGS+=(--api-key "$API_KEY")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

if [[ -z "${API_KEY:-}" && "$BIND" != "127.0.0.1" ]]; then
    warn "no API_KEY and bound to $BIND — anything that reaches :$PORT reaches the model"
fi

# ---------------------------------------------------------------- docker run
PY_SITE=/usr/local/lib/python3.12/dist-packages
PLUGIN="$RECIPE_DIR/files/plugin"
DOCKER=(
    docker run -d --name "$CONTAINER_NAME"
    --gpus all --network host --ipc host
    --ulimit memlock=-1 --ulimit stack=67108864
    --memory "${CONTAINER_MEM_GIB:-100}g" --memory-swap "${CONTAINER_MEM_GIB:-100}g"
    --log-opt max-size=50m --log-opt max-file=3
    -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    # Kolibri's FP8 block scales are fp32; DeepGEMM would round them to UE8M0.
    -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_DEEP_GEMM_E8M0=0
    # tuned Triton FP8 MoE tiles for GB10 at E=384 (see files/moe_configs/README)
    -e VLLM_TUNED_CONFIG_FOLDER=/root/moe_configs
    -v "$HF_HOME:/root/.cache/huggingface"
    -v "$VLLM_CACHE:/root/.cache/vllm"
    -v "$FLASHINFER_CACHE:/root/.cache/flashinfer"
    -v "$RECIPE_DIR/files/moe_configs:/root/moe_configs:ro"
    # The Aleph Alpha plugin, mounted instead of pip-installed: its wheel pins
    # vllm<0.30, and the dist-info carries the vllm.general_plugins entry point.
    -v "$PLUGIN/aleph_alpha_inference:$PY_SITE/aleph_alpha_inference:ro"
    -v "$PLUGIN/aleph_alpha_inference-1.0.0.dist-info:$PY_SITE/aleph_alpha_inference-1.0.0.dist-info:ro"
)
[[ -n "${MODEL_DIR:-}" ]] && DOCKER+=(-v "$SNAP:/models/local:ro")
# shellcheck disable=SC2206
[[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER+=($EXTRA_DOCKER_ARGS)
DOCKER+=("$IMAGE")

mkdir -p "$VLLM_CACHE" "$FLASHINFER_CACHE"
{
    echo '#!/usr/bin/env bash'
    echo "# written by start.sh $(date -Is)"
    printf '%q ' "${DOCKER[@]}" "${ARGS[@]}"; echo
} > "$RECIPE_DIR/.last_launch.sh"
chmod 600 "$RECIPE_DIR/.last_launch.sh"

if $NO_LAUNCH; then
    info "--no-launch: command written to .last_launch.sh"
    sed -n '3p' "$RECIPE_DIR/.last_launch.sh" | sed 's/ --/ \\\n    --/g; s/ -e / \\\n    -e /g; s/ -v / \\\n    -v /g'
    exit 0
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
rm -f "$LOG_DIR/stopping"
info "launching $CONTAINER_NAME on :$PORT"
"${DOCKER[@]}" "${ARGS[@]}" >/dev/null

if [[ "${MEMWATCH:-1}" == 1 ]]; then
    pkill -f "memwatch.sh $CONTAINER_NAME" 2>/dev/null || true
    nohup setsid "$RECIPE_DIR/scripts/memwatch.sh" "$CONTAINER_NAME" \
        >> "$LOG_DIR/memwatch-$CONTAINER_NAME.log" 2>&1 < /dev/null &
    info "watchdog: MemAvailable floor ${MEMWATCH_MIN_GIB:-4} GiB (logs/memwatch-$CONTAINER_NAME.log)"
fi

WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
if $NO_WAIT || [[ "$WAIT_TIMEOUT" == 0 ]]; then
    info "started; follow with: docker logs -f $CONTAINER_NAME"
    exit 0
fi

info "waiting for http://127.0.0.1:$PORT/health (up to ${WAIT_TIMEOUT}s)"
t0=$(date +%s); last=""
while :; do
    if ! docker ps -q -f "name=^${CONTAINER_NAME}$" | grep -q .; then
        docker logs --tail 60 "$CONTAINER_NAME" 2>&1 | sed 's/^/    /' >&2 || true
        err "container exited during startup (log tail above; ./stop.sh archives the full log)"
    fi
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
        info "ready after $(( $(date +%s) - t0 ))s — model '$SERVED_MODEL_NAME' on :$PORT"
        exit 0
    fi
    (( $(date +%s) - t0 > WAIT_TIMEOUT )) && err "not healthy after ${WAIT_TIMEOUT}s (container left running)"
    line="$(docker logs --tail 1 "$CONTAINER_NAME" 2>&1 | cut -c1-160)"
    [[ "$line" != "$last" ]] && { echo "    $line"; last="$line"; }
    sleep 5
done
