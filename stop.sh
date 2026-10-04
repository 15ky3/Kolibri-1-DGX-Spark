#!/usr/bin/env bash
# stop.sh — stop the Kolibri-1 container and its watchdog.
#
# Usage: ./stop.sh [--force]
#   default  SIGTERM, up to STOP_TIMEOUT (30 s) so vLLM can unlink its shared
#            memory (the container runs --ipc host; leftovers leak into /dev/shm)
#   --force  kill right away
# The container log is archived to logs/archive/ before the container is removed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"

FORCE=false
for a in "$@"; do
    case "$a" in
        -f|--force) FORCE=true ;;
        -h|--help)  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          err "unknown option: $a (try --help)" ;;
    esac
done
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || err "STOP_TIMEOUT must be a non-negative integer"

# watchdog first, so it cannot turn a slow graceful stop into an emergency one
pkill -f "memwatch.sh $CONTAINER_NAME" 2>/dev/null && info "watchdog stopped" || true

if [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]]; then
    info "$CONTAINER_NAME was not running"
    exit 0
fi

ARCHIVE="$LOG_DIR/archive"; TS=$(date +%Y%m%dT%H%M%S)
mkdir -p "$ARCHIVE"
docker logs --tail 5000 "$CONTAINER_NAME" > "$ARCHIVE/${CONTAINER_NAME}-${TS}.log" 2>&1 || true
ls -1t "$ARCHIVE"/*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
info "log archived: logs/archive/${CONTAINER_NAME}-${TS}.log"

if ! $FORCE; then
    info "stopping $CONTAINER_NAME (SIGTERM, up to ${STOP_TIMEOUT}s)"
    docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
info "stopped $CONTAINER_NAME"
