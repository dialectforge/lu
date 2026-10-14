#!/usr/bin/env bash
# backup_brain.sh -- snapshot the CANONICAL Lu brain to a remote target, verified.
#
# ############################################################################
# ## STATUS: ACTIVE -- remote target is provisioned and proven.             ##
# ##                                                                        ##
# ## The canonical brain (~248 GiB on disk) lives on a RAID1 mirror         ##
# ## (/dev/md2). RAID1 is NOT a backup: it protects against a single disk   ##
# ## failure, not against corruption, bad compaction, fat-finger deletion,  ##
# ## or ransomware -- all of which replicate to both mirror legs.           ##
# ##                                                                        ##
# ## The dedicated box has less free space than the brain, so a second      ##
# ## LOCAL copy does not fit and the mirror cannot be broken safely. The    ##
# ## backup therefore goes to a remote Hetzner Storage Box over ssh/rsync   ##
# ## (port 23), configured in REMOTE_TARGET below.                          ##
# ##                                                                        ##
# ## FIRST VERIFIED SNAPSHOT: brain-v5-20260615-sst__20260722T024410Z       ##
# ## (2026-07-22 -- 248.50 GiB, 344 regular files, checksum-verified        ##
# ## byte-identical to source).                                             ##
# ##                                                                        ##
# ## This script NEVER mutates the brain: it only READS the brain and       ##
# ## WRITES to the remote.                                                  ##
# ############################################################################
#
# Usage:
#   scripts/backup_brain.sh                     # quiescence -> rsync -> verify -> manifest
#   scripts/backup_brain.sh --dry-run           # show what WOULD transfer; change nothing
#   scripts/backup_brain.sh --allow-live        # (discouraged) back up while ingest is live
#   scripts/backup_brain.sh --verify-only NAME  # re-verify an EXISTING snapshot; NO transfer
#   scripts/backup_brain.sh --prune             # opt-in retention prune after a good backup
#
# BRAIN-MUTATION POLICY: read-only against the brain. Destructive remote
# pruning is opt-in (--prune) and never touches the snapshot just written.
set -euo pipefail

# ===== CONFIG ==============================================================
BRAIN="${LU_BRAIN:-/var/lib/lu/brain-v5-20260615-sst}"

# Remote target. Two supported shapes:
#   (A) rsync-over-ssh to a Hetzner Storage Box (current):
#         REMOTE_TARGET="uXXXXXX@uXXXXXX.your-storagebox.de:/home/lu-brain-backups"
#         REMOTE_SSH_PORT=23        # Hetzner Storage Box uses port 23 for ssh/rsync
#   (B) a locally-mounted remote path (e.g. Storage Box via CIFS/SSHFS):
#         REMOTE_TARGET="/mnt/storagebox/lu-brain-backups"
#         REMOTE_SSH_PORT=""        # leave empty for a mounted path
REMOTE_TARGET="${LU_BACKUP_REMOTE_TARGET:-uXXXXXX@uXXXXXX.your-storagebox.de:/home/lu-brain-backups}"
REMOTE_SSH_PORT="${LU_BACKUP_REMOTE_PORT-23}"
SSH_KEY="${LU_BACKUP_SSH_KEY:-$HOME/.ssh/storagebox_ed25519}"   # dedicated backup ssh key
RETAIN="${LU_BACKUP_RETAIN:-3}"    # keep the N most recent snapshots (only with --prune)
# ===========================================================================

DRY_RUN=0
ALLOW_LIVE=0
DO_PRUNE=0
VERIFY_ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift;;
    --allow-live)  ALLOW_LIVE=1; shift;;
    --prune)       DO_PRUNE=1; shift;;
    --verify-only)
      [[ $# -ge 2 && "$2" != --* ]] || { echo "backup_brain.sh: --verify-only needs a snapshot name" >&2; exit 64; }
      VERIFY_ONLY="$2"; shift 2;;
    -h|--help)     sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "backup_brain.sh: unknown arg: $1" >&2; exit 64;;
  esac
done

die() { echo "backup_brain.sh: $*" >&2; exit 1; }

# ---- 0) refuse on placeholder target --------------------------------------
if [[ "$REMOTE_TARGET" == REPLACE_ME* || -z "$REMOTE_TARGET" ]]; then
  cat >&2 <<'EOF'
REFUSING: REMOTE_TARGET is still the placeholder.

  Before this script can run:
    1. Provision a Hetzner Storage Box (>= 1 TB; the brain is ~248 GiB and grows).
    2. Add an SSH key to the Storage Box for passwordless rsync.
    3. Set REMOTE_TARGET (and REMOTE_SSH_PORT) at the top of this script.

  Until a verified backup exists, DO NOT run any destructive brain automation
  (major-compaction reclaim, deletes). See docs/TRAINING.md.
EOF
  exit 2
fi

# ---- 1) brain sanity -------------------------------------------------------
[[ -d "$BRAIN" ]] || die "brain directory not found: $BRAIN"
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
if [[ -n "$VERIFY_ONLY" ]]; then
  SNAP_NAME="$VERIFY_ONLY"
else
  SNAP_NAME="$(basename "$BRAIN")__$STAMP"
fi

# ---- 2) quiescence: never read a mid-write brain by default ----------------
# Applies to --verify-only too: checksumming a brain that is being mutated
# under us would produce meaningless diffs.
WRITER="$(pgrep -af 'lu_4_mcp' 2>/dev/null | grep -F -- "--brain $BRAIN" || true)"
if [[ -n "$WRITER" && "$ALLOW_LIVE" -ne 1 ]]; then
  cat >&2 <<EOF
REFUSING: a lu_4_mcp process is bound to this brain (training/serving is live):
  $WRITER

  Backing up or verifying a brain mid-write can capture/compare an inconsistent
  SST+WAL set. Preferred: stop training (scripts/train.sh stop), optionally
  flush (lu4d_flush), then run this. To override at your own risk: --allow-live.
EOF
  exit 3
fi

# ---- 3) transports ---------------------------------------------------------
SSH_CMD=""
if [[ -n "$REMOTE_SSH_PORT" ]]; then
  SSH_CMD="ssh -p $REMOTE_SSH_PORT"
  [[ -n "$SSH_KEY" ]] && SSH_CMD="$SSH_CMD -i $SSH_KEY"
fi

# Transfer rsync: progress/stats are WANTED here (they go to the run log).
RSYNC=(rsync -a --human-readable --partial --inplace --info=stats2,progress2)
[[ -n "$SSH_CMD" ]] && RSYNC+=(-e "$SSH_CMD")
DEST="${REMOTE_TARGET%/}/$SNAP_NAME"

# ---- verification ----------------------------------------------------------
# 2026-07-22 FIX (false-negative gate). The verify pass must NOT reuse the
# transfer's --info=stats2,progress2 flags. progress2 writes carriage-return
# progress lines to stdout; the previous implementation captured those into
# VERIFY_OUT and filtered only with `grep -vE '^$|/$'`, which does not strip
# them. VERIFY_OUT was therefore non-empty even when the remote was perfectly
# byte-identical, producing a FALSE "VERIFY FAILED" (observed on the first real
# run: rsync itself reported 0 files / 0 bytes left to transfer while the gate
# claimed failure). Verify now runs its own quiet rsync and matches only real
# itemize codes.
#
# rsync itemize format is  YXcstpoguax  path
#   Y (update type): '<' sent to remote, '>' received, 'c' created locally,
#                    'h' hardlink, '*' message (e.g. deleting), '.' no update
#   X (file type)  : f file, d dir, L symlink, D device, S special
#
# A CONTENT or EXISTENCE difference always carries Y in [<>ch*]:
#     <f+++++++++  file missing on the remote  (would be created)
#     <fc........  checksum mismatch           (would be re-sent)
#     cd+++++++++  directory missing on remote
# An ATTRIBUTE-ONLY difference carries Y == '.' (e.g. '.f....og...' for
# owner/group, which a Storage Box legitimately cannot round-trip for a
# non-root sender). Those are REPORTED but do not fail the gate: they are not
# a corruption of the backed-up bytes.
verify_snapshot() {
  local dest="$1"
  local -a VRSYNC=(rsync -a --dry-run --checksum --itemize-changes)
  [[ -n "$SSH_CMD" ]] && VRSYNC+=(-e "$SSH_CMD")

  local out diffs attrs
  out="$("${VRSYNC[@]}" "$BRAIN/" "$dest/" 2>/dev/null || true)"

  diffs="$(printf '%s\n' "$out" | grep -E '^[<>ch*][fdLDS]' || true)"
  attrs="$(printf '%s\n' "$out" | grep -E '^\.[fdLDS]' || true)"

  if [[ -n "$attrs" ]]; then
    echo "    note: attribute-only differences (metadata, not content -- not fatal):"
    printf '%s\n' "$attrs" | head -10 | sed 's/^/      /'
  fi

  if [[ -n "$diffs" ]]; then
    echo "!!! VERIFY FAILED -- these paths differ in CONTENT or EXISTENCE:" >&2
    printf '%s\n' "$diffs" | head -50 >&2
    return 1
  fi
  return 0
}

# Count regular files on the remote snapshot (read-only; no delete semantics).
remote_file_count() {
  local dest="$1"
  local -a LRSYNC=(rsync -a --list-only)
  [[ -n "$SSH_CMD" ]] && LRSYNC+=(-e "$SSH_CMD")
  "${LRSYNC[@]}" "$dest/" 2>/dev/null | grep -cE '^-' || true
}

write_manifest() {
  local dest="$1" snap="$2" src_bytes src_files rmt_files
  src_bytes="$(du -sb "$BRAIN" 2>/dev/null | cut -f1)"
  src_files="$(find "$BRAIN" -type f | wc -l)"
  rmt_files="$(remote_file_count "$dest")"
  MANIFEST="/tmp/${snap}.manifest"
  {
    echo "snapshot=$snap"
    echo "created_utc=$STAMP"
    echo "verified_utc=$(date -u '+%Y%m%dT%H%M%SZ')"
    echo "source=$BRAIN"
    echo "source_bytes=$src_bytes"
    echo "source_file_count=$src_files"
    echo "remote_file_count=$rmt_files"
    echo "remote=$dest"
    echo "verify=PASS (rsync --checksum: no content/existence differences)"
  } > "$MANIFEST"
  echo ">>> manifest: $MANIFEST"
  sed 's/^/      /' "$MANIFEST"

  if [[ "$src_files" != "$rmt_files" ]]; then
    echo "!!! file-count mismatch: source=$src_files remote=$rmt_files" >&2
    return 1
  fi

  # park the manifest next to the snapshot on the remote for provenance
  local -a MRSYNC=(rsync -a)
  [[ -n "$SSH_CMD" ]] && MRSYNC+=(-e "$SSH_CMD")
  if "${MRSYNC[@]}" "$MANIFEST" "${REMOTE_TARGET%/}/" 2>/dev/null; then
    echo "    manifest copied to remote: ${REMOTE_TARGET%/}/$(basename "$MANIFEST")"
  else
    echo "    WARN: could not copy manifest to remote (snapshot itself is fine)" >&2
  fi
  return 0
}

# ---- retention -------------------------------------------------------------
# Previously this block only ran for MOUNTED-path targets, so RETAIN was a
# silent no-op for the Storage Box (ssh) target -- snapshots accumulated
# forever. It now works for ssh targets too, but is OPT-IN via --prune because
# it deletes remote snapshots of an irreplaceable brain. Without --prune the
# script only REPORTS what retention would remove.
#
# Storage Box shells are restricted (no rm -rf), so a remote directory is
# emptied by rsync-ing an empty local dir over it with --delete, then removed
# with sftp rmdir.
prune_remote() {
  local keep_snap="$1"
  local -a LRSYNC=(rsync -a --list-only)
  [[ -n "$SSH_CMD" ]] && LRSYNC+=(-e "$SSH_CMD")

  local base pattern all old
  base="$(basename "$BRAIN")"
  pattern="^${base}__[0-9]{8}T[0-9]{6}Z$"

  # newest-last by name (timestamps sort lexicographically)
  all="$("${LRSYNC[@]}" "${REMOTE_TARGET%/}/" 2>/dev/null \
        | awk '/^d/ {print $NF}' | grep -E "$pattern" | sort || true)"
  [[ -n "$all" ]] || { echo "    (no snapshots found to consider)"; return 0; }

  local total
  total="$(printf '%s\n' "$all" | wc -l)"
  if (( total <= RETAIN )); then
    echo "    $total snapshot(s) on remote, RETAIN=$RETAIN -- nothing to prune."
    return 0
  fi

  old="$(printf '%s\n' "$all" | head -n $((total - RETAIN)))"
  echo "    $total snapshot(s) on remote, RETAIN=$RETAIN -- candidates to prune:"
  printf '%s\n' "$old" | sed 's/^/      /'

  if (( DO_PRUNE != 1 )); then
    echo "    (retention is REPORT-ONLY; re-run with --prune to actually delete)"
    return 0
  fi

  local empty; empty="$(mktemp -d)"
  local d
  while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    [[ "$d" != "$keep_snap" ]] || { echo "    refusing to prune the snapshot just written: $d"; continue; }
    echo "    pruning remote snapshot: $d"
    local -a DRSYNC=(rsync -a --delete)
    [[ -n "$SSH_CMD" ]] && DRSYNC+=(-e "$SSH_CMD")
    "${DRSYNC[@]}" "$empty/" "${REMOTE_TARGET%/}/$d/" || { echo "    WARN: could not empty $d" >&2; continue; }
    if [[ -n "$REMOTE_SSH_PORT" ]]; then
      local host path
      host="${REMOTE_TARGET%%:*}"
      path="${REMOTE_TARGET#*:}"
      printf 'rmdir %s\n' "${path%/}/$d" \
        | sftp -P "$REMOTE_SSH_PORT" ${SSH_KEY:+-i "$SSH_KEY"} -o BatchMode=yes "$host" >/dev/null 2>&1 \
        || echo "    WARN: emptied but could not rmdir $d" >&2
    else
      rmdir "${REMOTE_TARGET%/}/$d" 2>/dev/null || true
    fi
  done <<< "$old"
  rmdir "$empty" 2>/dev/null || true
}

# ---- 4) VERIFY-ONLY path (no transfer) -------------------------------------
if [[ -n "$VERIFY_ONLY" ]]; then
  echo "=== backup_brain.sh --verify-only $SNAP_NAME ==="
  echo "    source : $BRAIN"
  echo "    dest   : $DEST"
  echo "    NOTE   : checksum comparison only -- NOTHING is transferred."
  echo
  echo ">>> Verifying (checksum re-scan)..."
  verify_snapshot "$DEST" || die "snapshot NOT verified; investigate before trusting it."
  echo "    OK -- remote snapshot byte-verified against source."
  echo
  write_manifest "$DEST" "$SNAP_NAME" || die "manifest/file-count check failed."
  echo
  echo ">>> Retention check:"
  prune_remote "$SNAP_NAME"
  echo
  echo "=== verify-only complete + VERIFIED: $DEST ==="
  exit 0
fi

# ---- 5) normal path: transfer ----------------------------------------------
echo "=== backup_brain.sh $STAMP ==="
echo "    source : $BRAIN ($(du -sh "$BRAIN" 2>/dev/null | cut -f1))"
echo "    dest   : $DEST"
echo "    port   : ${REMOTE_SSH_PORT:-<mounted path>}"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">>> DRY RUN -- listing what would transfer, making NO changes:"
  "${RSYNC[@]}" --dry-run --itemize-changes "$BRAIN/" "$DEST/" || true
  echo ">>> dry run complete."
  exit 0
fi

echo ">>> Transferring (rsync)..."
"${RSYNC[@]}" "$BRAIN/" "$DEST/"

# ---- 6) verify -------------------------------------------------------------
echo ">>> Verifying (checksum re-scan; must show nothing left to copy)..."
verify_snapshot "$DEST" || die "backup NOT verified; investigate before trusting this snapshot."
echo "    OK -- remote snapshot byte-verified against source."

# ---- 7) manifest -----------------------------------------------------------
write_manifest "$DEST" "$SNAP_NAME" || die "manifest/file-count check failed."

# ---- 8) retention ----------------------------------------------------------
echo
echo ">>> Retention check:"
prune_remote "$SNAP_NAME"

echo
echo "=== backup complete + verified: $DEST ==="
