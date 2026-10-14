#!/usr/bin/env bash
# lu-curiosity-cycle.sh -- run one curiosity SLEEP CYCLE.
#
# Lu wonders while awake and consolidates while asleep. This script is
# the sleep. It stops the public guest, runs the FILL worker against the
# gaps the guest recorded while it was up, then brings the guest back on
# the enriched brain.
#
# WHY IT IS A CYCLE AND NOT A BACKGROUND THREAD
#
#   Loading the 233 GiB v5 brain peaks around 54 GiB and settles near
#   26 GiB, on a box with 62 GiB. Serving and filling each need their own
#   full load, so they cannot overlap -- earlier attempts to co-load a
#   second reader were OOM-killed mid-load. There is a correctness reason
#   too: the guest watches the committed MANIFEST frontier and re-opens a
#   fresh reader when it advances, so a concurrent writer would trigger a
#   SECOND full load inside the guest itself.
#
#   Running the fill half inside the guest process would avoid all of
#   this and is exactly what we must not do: it would make the
#   internet-facing process a writer to the canonical brain and throw
#   away the OP-007 privilege separation.
#
# TRAINING MUST BE STOPPED. It is the third tenant of the same RAM and
# the same brain. This script refuses to run if an ingest unit is active.
#
# Usage:
#   sudo ./lu-curiosity-cycle.sh [--ticks N] [--auto-apply] [--dry-run]

set -euo pipefail

BRAIN=/var/lib/lu/brain-v5-20260615-sst
KEYFILE=/var/lib/lu/master.key
WONDER_DIR=/var/lib/lu-wonder
WORKER=/usr/local/bin/lu_curiosity_worker
GUEST_UNIT=lu-guest-http.service
TICKS=10
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ticks)      TICKS="$2"; shift 2 ;;
    --auto-apply) EXTRA+=(--auto-apply); shift ;;
    --dry-run)    EXTRA+=(--dry-run); shift ;;
    --no-research) EXTRA+=(--no-research); shift ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ---- Preflight ------------------------------------------------------
for unit in lu4d-starcoderdata-ingest.service lu4d-curriculum-ingest.service; do
  if systemctl is-active --quiet "$unit"; then
    echo "REFUSING: $unit is active. Training, serving and curiosity are three" >&2
    echo "mutually exclusive tenants of the same brain and the same RAM." >&2
    exit 1
  fi
done

[[ -x "$WORKER" ]] || { echo "missing $WORKER" >&2; exit 1; }
mkdir -p "$WONDER_DIR"
chown dialectforge:dialectforge "$WONDER_DIR"

JOURNAL="$WONDER_DIR/wonder_queue.jsonl"
if [[ ! -s "$JOURNAL" ]]; then
  log "wonder journal is empty ($JOURNAL) -- nothing to consolidate."
  log "Ask Lu some questions first; detection runs on every query."
  exit 0
fi
log "journal has $(wc -l < "$JOURNAL") record(s)"

# ---- Sleep ----------------------------------------------------------
GUEST_WAS_UP=0
if systemctl is-active --quiet "$GUEST_UNIT"; then
  GUEST_WAS_UP=1
  log "stopping $GUEST_UNIT (freeing RAM for the fill worker)"
  systemctl stop "$GUEST_UNIT"
  # Wait for the address space to actually go away, not just for systemd
  # to report the unit inactive.
  for _ in $(seq 1 60); do
    pgrep -x lu_guest_http >/dev/null || break
    sleep 1
  done
  sleep 3
fi

# Always bring the guest back, even if the worker fails or we are
# interrupted. A demo box that is down because consolidation crashed is
# worse than one that never consolidated.
restore_guest() {
  if [[ "$GUEST_WAS_UP" == "1" ]]; then
    log "restarting $GUEST_UNIT"
    systemctl start "$GUEST_UNIT" || log "WARNING: guest failed to restart"
  fi
}
trap restore_guest EXIT INT TERM

# ---- Fill -----------------------------------------------------------
log "running fill worker (ticks=$TICKS ${EXTRA[*]:-})"
sudo -u dialectforge "$WORKER" \
  --brain "$BRAIN" \
  --keyfile "$KEYFILE" \
  --wonder-queue "$WONDER_DIR" \
  --ticks "$TICKS" \
  "${EXTRA[@]:-}" || log "WARNING: worker exited non-zero"

log "cycle complete"
