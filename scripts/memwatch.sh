#!/usr/bin/env bash
# memwatch.sh CONTAINER — stop CONTAINER before unified memory runs out.
#
# GB10 shares one pool between CPU and GPU. When it runs dry the driver starts
# refusing allocations (NV_ERR_NO_MEMORY) and the OOM killer picks victims on
# the host side — often not the model server. Stopping the server ourselves is
# the cheaper failure. Exits on its own once the container is gone.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

C="${1:?container name}"
FLOOR="${MEMWATCH_MIN_GIB:-4}"; GRACE="${MEMWATCH_GRACE:-3}"
low=0; min=999
log() { echo "$(date '+%F %T') $*"; }
log "watching $C, floor ${FLOOR} GiB, grace ${GRACE} samples"

while sleep 2; do
    docker ps -q -f "name=^${C}$" | grep -q . || { log "container gone (min seen ${min} GiB), exiting"; exit 0; }
    avail=$(mem_gib MemAvailable)
    min=$(awk -v a="$avail" -v m="$min" 'BEGIN{print (a<m)?a:m}')
    if awk -v a="$avail" -v f="$FLOOR" 'BEGIN{exit !(a<f)}'; then
        low=$((low + 1))
        log "LOW MemAvailable ${avail} GiB (${low}/${GRACE})"
        if (( low >= GRACE )); then
            log "EMERGENCY STOP $C"
            docker logs --tail 3000 "$C" > "$LOG_DIR/${C}-$(date +%Y%m%dT%H%M%S)-memwatch-stop.log" 2>&1 || true
            printf 'memwatch\n%s\n' "$(date -Is)" > "$LOG_DIR/stopping"
            docker stop -t 20 "$C" >/dev/null 2>&1 || docker rm -f "$C" >/dev/null 2>&1
            exit 1
        fi
    else
        low=0
    fi
done
