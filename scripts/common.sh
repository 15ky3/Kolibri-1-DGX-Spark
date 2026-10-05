# shellcheck shell=bash
# Shared helpers for start.sh / stop.sh / download.sh. Sourced, never executed.

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -t 1 ]]; then C_R=$'\e[31m'; C_Y=$'\e[33m'; C_G=$'\e[32m'; C_0=$'\e[0m'; else C_R=; C_Y=; C_G=; C_0=; fi
info() { echo "${C_G}[kolibri]${C_0} $*"; }
warn() { echo "${C_Y}[kolibri] WARN:${C_0} $*" >&2; }
err()  { echo "${C_R}[kolibri] ERROR:${C_0} $*" >&2; exit 1; }

# load_env FILE — KEY=value lines; a key already in the environment wins.
# Inline comments need whitespace before '#'. A leading ~/ expands to $HOME.
load_env() {
    local f="$1" line k v
    [[ -f "$f" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=(.*)$ ]] || continue
        k="${BASH_REMATCH[1]}"; v="${BASH_REMATCH[2]}"
        [[ -n "${!k+x}" ]] && continue
        v="$(sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//' <<<"$v")"
        v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
        [[ "$v" == "~/"* ]] && v="$HOME/${v#\~/}"
        export "$k=$v"
    done < "$f"
}

# .env first (the operator's), then .env.sample fills whatever is still unset.
load_env "$RECIPE_DIR/.env"
load_env "$RECIPE_DIR/.env.sample"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-kolibri-1}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# QUANT picks the checkpoint and what follows from it; anything set explicitly
# (environment or .env) wins over these defaults.
#   nvfp4  routed experts NVFP4, rest FP8 — 42.8 GiB on the GPU (default)
#   fp8    Aleph Alpha's original FP8 checkpoint — 73.6 GiB on the GPU
# ABLIT=1 serves the abliterated (Heretic) variant of the same QUANT.
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == 0 || "$ABLIT" == 1 ]] || err "ABLIT must be 0 or 1 (got: $ABLIT)"
QUANT="${QUANT:-nvfp4}"
case "$QUANT/$ABLIT" in
    nvfp4/0) : "${MODEL_ID:=iSkye/Kolibri-1-NVFP4-Experts}"         "${WEIGHTS_GIB:=43}"   "${KV_CACHE_GIB:=48}" "${CHECKPOINT_GB:=46}" ;;
    nvfp4/1) : "${MODEL_ID:=iSkye/Kolibri-1-heretic-NVFP4-Experts}" "${WEIGHTS_GIB:=43}"   "${KV_CACHE_GIB:=48}" "${CHECKPOINT_GB:=46}" ;;
    fp8/0)   : "${MODEL_ID:=Aleph-Alpha/Kolibri-1}"                 "${WEIGHTS_GIB:=73.4}" "${KV_CACHE_GIB:=22}" "${CHECKPOINT_GB:=79}" ;;
    fp8/1)   : "${MODEL_ID:=iSkye/Kolibri-1-heretic}"               "${WEIGHTS_GIB:=73.4}" "${KV_CACHE_GIB:=22}" "${CHECKPOINT_GB:=80}" ;;
    *)       err "QUANT must be nvfp4 or fp8 (got: $QUANT)" ;;
esac
export ABLIT QUANT MODEL_ID WEIGHTS_GIB KV_CACHE_GIB CHECKPOINT_GB
# A checkpoint downloaded with `hf download --local-dir $LOCAL_MODELS/<repo name>`
# (what the dashboard does for some models) is used when the hub cache has none.
LOCAL_MODELS="${LOCAL_MODELS:-$HOME/models}"
LOCAL_COPY="$LOCAL_MODELS/${MODEL_ID##*/}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID%%/*}--${MODEL_ID##*/}"
LOG_DIR="$RECIPE_DIR/logs"
mkdir -p "$LOG_DIR"

# resolve_snapshot — prints the snapshot dir start.sh will serve, or nothing.
resolve_snapshot() {
    if [[ -n "${MODEL_REVISION:-}" ]]; then
        [[ -d "$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION" ]] && echo "$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION"
        return 0
    fi
    local ref
    ref="$(cat "$MODEL_CACHE_DIR/refs/main" 2>/dev/null || true)"
    if [[ -n "$ref" && -d "$MODEL_CACHE_DIR/snapshots/$ref" ]]; then
        echo "$MODEL_CACHE_DIR/snapshots/$ref"; return 0
    fi
    find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true
}

mem_gib() {  # mem_gib MemTotal|MemAvailable
    awk -v k="$1:" '$1==k {printf "%.2f", $2/1048576}' /proc/meminfo
}
