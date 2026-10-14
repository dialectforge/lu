#!/usr/bin/env bash
# train_preflight.sh -- HARD preflight gate for canonical Lu v5 training.
#
# Purpose
#   Refuse to launch a training run unless the box is in a known-safe state.
#   Called by scripts/train.sh before every start/resume. Can also be run
#   standalone as a dry-run:  bash scripts/train_preflight.sh --brain <path>
#
# Contract
#   * Runs a fixed set of checks, prints PASS/FAIL/WARN for each.
#   * Exits 0 ONLY if every hard check passes.
#   * Reports ALL failures, then exits non-zero (easier to fix in one pass).
#   * Makes NO changes to the brain, the WAL, or any running process.
#     It is read-only. Draining the WAL, stopping a run, etc. are all
#     HUMAN-GATED actions the operator performs after reading the advice.
#
# Design notes (why these thresholds)
#   * The canonical brain (brain-v5-20260615-sst) is ~198 GiB and lives on a
#     RAID1 mirror (/dev/md2) with ~188 GiB free -- i.e. there is NOT room for
#     a second local copy. So the disk check is an ABSOLUTE headroom margin for
#     WAL growth + fresh L0 SSTs between saves, NOT "free >= brain size". We
#     additionally WARN when free < brain size so the compaction / local-backup
#     impossibility is always visible.
#   * Lean ingest-only boot RSS is ~15 GiB; caps are 52G/56G on a 62 GiB box.
#     We require a healthy MemAvailable so the lean boot does not instantly
#     thrash swap.
#   * A large un-drained WAL means a long, RAM-heavy replay at boot (the "WAL
#     replay boot tax"). We fail and print the human-gated drain command rather
#     than draining automatically.
set -euo pipefail

# ---- defaults (override via env or flags) ----------------------------------
BRAIN="${LU_BRAIN:-/var/lib/lu/brain-v5-20260615-sst}"
LU4D_BIN="${LU4D_BIN:-/opt/lu/lu/merge-main/target/release/lu4d}"
LOAD_MODE="${LU_LOAD_MODE:-ingest-only}"
UNIT="${LU_UNIT:-}"                             # systemd unit train.sh intends to start
DISK_MIN_FREE_GIB="${DISK_MIN_FREE_GIB:-25}"    # hard floor of free space
RAM_MIN_AVAIL_GIB="${RAM_MIN_AVAIL_GIB:-18}"    # hard floor of MemAvailable
WAL_MAX_GIB="${WAL_MAX_GIB:-2}"                 # WAL bigger than this => drain first

usage() {
  cat <<EOF
Usage: train_preflight.sh [--brain PATH] [--bin PATH] [--load-mode MODE]
                          [--unit NAME] [--disk-min GIB] [--ram-min GIB]
                          [--wal-max GIB]
Read-only gate. Exit 0 = safe to launch, non-zero = do NOT launch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --brain)     BRAIN="$2"; shift 2;;
    --bin)       LU4D_BIN="$2"; shift 2;;
    --load-mode) LOAD_MODE="$2"; shift 2;;
    --unit)      UNIT="$2"; shift 2;;
    --disk-min)  DISK_MIN_FREE_GIB="$2"; shift 2;;
    --ram-min)   RAM_MIN_AVAIL_GIB="$2"; shift 2;;
    --wal-max)   WAL_MAX_GIB="$2"; shift 2;;
    -h|--help)   usage; exit 0;;
    *) echo "preflight: unknown arg: $1" >&2; usage >&2; exit 64;;
  esac
done

FAIL=0
say()  { printf '  %-6s %s\n' "$1" "$2"; }
pass() { say "PASS" "$1"; }
warn() { say "WARN" "$1"; }
fail() { say "FAIL" "$1"; FAIL=$((FAIL+1)); }

echo "=== Lu training preflight $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo "    brain=$BRAIN  bin=$LU4D_BIN  load-mode=$LOAD_MODE  unit=${UNIT:-<none>}"
echo

# 1) Brain path exists ------------------------------------------------------
if [[ -d "$BRAIN" ]]; then
  pass "brain directory exists: $BRAIN"
else
  fail "brain directory MISSING: $BRAIN -- refusing (never create a brain implicitly)"
fi

# 2) Binary exists, executable, supports --load-mode ------------------------
if [[ -x "$LU4D_BIN" ]]; then
  pass "binary present + executable: $LU4D_BIN ($(md5sum "$LU4D_BIN" 2>/dev/null | cut -c1-12)...)"
  # grep the binary directly (NOT `strings | grep -q`): under `set -o pipefail`
  # grep -q closes the pipe on first match, killing the upstream process with
  # SIGPIPE and making the whole pipeline report failure even on a real match.
  if grep -aq -- '--load-mode' "$LU4D_BIN" 2>/dev/null \
     && grep -aq -- 'ingest-only' "$LU4D_BIN" 2>/dev/null; then
    pass "binary supports --load-mode selector (lean ingest-only build)"
  else
    fail "binary does NOT advertise --load-mode -- wrong/old binary; rebuild+deploy the lean binary"
  fi
else
  fail "binary MISSING or not executable: $LU4D_BIN"
fi

case "$LOAD_MODE" in
  ingest-only|read-only|auto) pass "requested load-mode is valid: $LOAD_MODE";;
  *) fail "load-mode '$LOAD_MODE' not allowed for training (use ingest-only)";;
esac
if [[ "$LOAD_MODE" != "ingest-only" ]]; then
  warn "load-mode is '$LOAD_MODE', not the canonical lean 'ingest-only' -- expect higher boot RAM"
fi

# 3) Serving separation: no other lu_4_mcp writer/reader on this brain ------
#    ANY existing lu_4_mcp bound to this brain means a run is already active
#    OR a serving process holds it. Training must be the sole brain-toucher.
BRAIN_ESC="$(printf '%s' "$BRAIN" | sed 's/[.[\*^$/]/\\&/g')"
# 2026-07-22 FIX (false-positive gate). The previous implementation was:
#   OTHER_PROCS="$(pgrep -af "lu_4_mcp" | grep -E -- "--brain[= ]+${BRAIN_ESC}...")"
# `pgrep -af` matches any process whose COMMAND LINE merely CONTAINS the string
# "lu_4_mcp" -- which includes THIS preflight script, invoked by train.sh as:
#   train_preflight.sh --brain <BRAIN> --bin /usr/local/bin/lu_4_mcp --load-mode ...
# Our own argv carries both "lu_4_mcp" and "--brain <BRAIN>", so the check
# matched ITSELF and refused every launch with a phantom writer (observed
# 2026-07-22 while resuming after the first verified backup: it reported the
# two train_preflight.sh PIDs as "another lu_4_mcp bound to this brain" while
# no lu_4_mcp process existed at all).
#
# We now match only processes whose EXECUTABLE NAME is exactly `lu_4_mcp`
# (pgrep -x matches the process name, not the cmdline), read each one's real
# cmdline from /proc, and skip our own pid/ppid. A genuine second writer is
# still caught, because a real writer IS a process named lu_4_mcp.
OTHER_PROCS=""
while read -r _pid; do
  [[ -n "$_pid" ]] || continue
  [[ "$_pid" != "$$" && "$_pid" != "$PPID" ]] || continue
  _cmd="$(tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null)"
  [[ -n "$_cmd" ]] || continue
  if grep -qE -- "--brain[= ]+${BRAIN_ESC}(\b|/|$)" <<<"$_cmd"; then
    OTHER_PROCS+="$_pid $_cmd"$'\n'
  fi
done < <(pgrep -x lu_4_mcp 2>/dev/null || true)
OTHER_PROCS="${OTHER_PROCS%$'\n'}"
if [[ -z "$OTHER_PROCS" ]]; then
  pass "no other lu_4_mcp process is bound to this brain (serving separation holds)"
else
  fail "another lu_4_mcp is ALREADY bound to this brain -- serving/training separation VIOLATED:"
  printf '%s\n' "$OTHER_PROCS" | sed 's/^/           /'
  echo   "           -> a training run may already be live, or Lu is being served."
  echo   "           -> do NOT start a second writer. Stop the existing one first (human-gated)."
fi

# 3b) If train.sh told us its unit, make sure it is not already active ------
if [[ -n "$UNIT" ]]; then
  if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
    fail "target unit '$UNIT' is already ACTIVE -- a run is in progress; refusing to relaunch"
  else
    pass "target unit '$UNIT' is not active"
  fi
fi

# 4) Disk headroom ----------------------------------------------------------
MOUNT_FS="$(df -P "$BRAIN" 2>/dev/null | awk 'NR==2{print $6}')"
FREE_GIB="$(df -PB1G "$BRAIN" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4+0}')"
if [[ -n "${FREE_GIB:-}" ]]; then
  if (( FREE_GIB >= DISK_MIN_FREE_GIB )); then
    pass "disk free on ${MOUNT_FS}: ${FREE_GIB} GiB (>= ${DISK_MIN_FREE_GIB} GiB floor)"
  else
    fail "disk free on ${MOUNT_FS}: ${FREE_GIB} GiB is BELOW the ${DISK_MIN_FREE_GIB} GiB floor -- free space first"
  fi
  # Informational: can a second local copy of the brain fit? (compaction/backup)
  BRAIN_GIB="$(du -sB1G "$BRAIN" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1+0}')"
  if [[ -n "${BRAIN_GIB:-}" ]] && (( FREE_GIB < BRAIN_GIB )); then
    warn "free (${FREE_GIB} GiB) < brain size (${BRAIN_GIB} GiB): NO room for a second local copy."
    warn "  -> major-compaction 'write-new' and local brain backup are NOT possible on this disk."
    warn "  -> backups must target a remote (Hetzner Storage Box). See scripts/backup_brain.sh."
  fi
else
  fail "could not determine free disk for $BRAIN"
fi

# 5) RAM headroom -----------------------------------------------------------
AVAIL_GIB="$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
if (( AVAIL_GIB >= RAM_MIN_AVAIL_GIB )); then
  pass "MemAvailable: ${AVAIL_GIB} GiB (>= ${RAM_MIN_AVAIL_GIB} GiB floor)"
else
  fail "MemAvailable: ${AVAIL_GIB} GiB is BELOW the ${RAM_MIN_AVAIL_GIB} GiB floor -- free RAM before lean boot"
fi

# 6) WAL not carrying a huge un-drained replay ------------------------------
WAL="$BRAIN/wal/current.wal"
if [[ -f "$WAL" ]]; then
  WAL_BYTES="$(stat -c%s "$WAL" 2>/dev/null || echo 0)"
  WAL_MAX_BYTES=$(( WAL_MAX_GIB * 1024 * 1024 * 1024 ))
  if (( WAL_BYTES <= WAL_MAX_BYTES )); then
    pass "WAL current.wal is $(( WAL_BYTES/1024/1024 )) MiB (<= ${WAL_MAX_GIB} GiB) -- replay tax acceptable"
  else
    fail "WAL current.wal is $(( WAL_BYTES/1024/1024 )) MiB (> ${WAL_MAX_GIB} GiB) -- big replay tax at boot."
    echo   "           -> DRAIN it first (HUMAN-GATED, do NOT automate). With the brain quiescent:"
    echo   "              call lu4d_flush against $BRAIN, confirm current.wal shrank, then re-run preflight."
  fi
else
  warn "no WAL at $WAL (fresh brain or already drained) -- nothing to replay"
fi

echo
if (( FAIL > 0 )); then
  echo "=== PREFLIGHT REFUSED: ${FAIL} hard check(s) failed. Do NOT launch. ==="
  exit 1
fi
echo "=== PREFLIGHT OK: safe to launch. ==="
exit 0
