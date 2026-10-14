#!/usr/bin/env bash
# OP-BRAIN-V5 6.6 -- versioned-dir + symlink helper for the loader cutover.
#
# Locked decision #6 (design doc §7): the lu service points at a STABLE
# path (a `brain-current` symlink); the symlink resolves to a VERSIONED
# brain dir (the v3 `brain.lbr` today, a `brain-v5-<date>/` later). A
# cutover is then a one-line symlink swap, and rollback is the reverse --
# the never-mutated v3 brain stays on disk the whole time.
#
# IMPORTANT: this script is CAPABILITY only. It does not deploy anything
# on its own and it does NOT touch /var/lib/lu unless you explicitly pass
# --base /var/lib/lu (and even then only the symlink, never the brain
# data). Production stays on v3 until the separate Phase 8 cutover.
#
# Usage:
#   lu_v5_brain_current.sh status               --base <dir>
#   lu_v5_brain_current.sh init     <target>     --base <dir>   # create brain-current -> target (only if absent)
#   lu_v5_brain_current.sh point    <target>     --base <dir>   # atomically repoint brain-current -> target
#   lu_v5_brain_current.sh rollback              --base <dir>   # repoint to the saved previous target
#   lu_v5_brain_current.sh version-name <prefix>                # print a dated versioned dir name, e.g. brain-v5-20260611
#
# Flags:
#   --base <dir>   directory that holds the brain dirs + the symlink
#                  (default: $LU_LIB_DIR, else error -- never guessed)
#   --dry-run      print what would happen; change nothing
#
# The symlink is named `brain-current`; the previous target is recorded
# in `brain-current.prev` next to it so `rollback` is deterministic.
set -euo pipefail

LINK_NAME="brain-current"
PREV_NAME="brain-current.prev"
DRY_RUN=0
BASE="${LU_LIB_DIR:-}"

die() { echo "error: $*" >&2; exit 1; }
run() { if [[ "$DRY_RUN" == 1 ]]; then echo "DRY-RUN: $*"; else eval "$*"; fi }

# --- parse flags (anywhere on the line) ---
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"
CMD="${1:-}"

version_name() {
  local prefix="${1:-brain-v5}"
  echo "${prefix}-$(date +%Y%m%d)"
}

# `version-name` needs no --base.
if [[ "$CMD" == "version-name" ]]; then
  version_name "${2:-brain-v5}"
  exit 0
fi

[[ -n "$BASE" ]] || die "--base <dir> required (or set LU_LIB_DIR). This script never guesses the brain directory."
[[ -d "$BASE" ]] || die "--base $BASE is not a directory"
LINK="$BASE/$LINK_NAME"
PREV="$BASE/$PREV_NAME"

resolve() { [[ -L "$LINK" ]] && readlink "$LINK" || true; }

case "$CMD" in
  status)
    echo "base            : $BASE"
    if [[ -L "$LINK" ]]; then
      echo "$LINK_NAME -> $(resolve)"
      if [[ -e "$LINK" ]]; then echo "target exists   : yes"; else echo "target exists   : NO (dangling symlink)"; fi
    elif [[ -e "$LINK" ]]; then
      echo "$LINK_NAME is a NON-symlink path (unexpected): $LINK"
    else
      echo "$LINK_NAME : (absent)"
    fi
    [[ -f "$PREV" ]] && echo "recorded prev   : $(cat "$PREV")" || echo "recorded prev   : (none)"
    ;;

  init)
    TARGET="${2:-}"; [[ -n "$TARGET" ]] || die "init needs a <target>"
    [[ -e "$BASE/$TARGET" || -e "$TARGET" ]] || die "target '$TARGET' does not exist under $BASE"
    [[ -e "$LINK" || -L "$LINK" ]] && die "$LINK already exists; use 'point' to repoint"
    run "ln -s '$TARGET' '$LINK'"
    echo "created $LINK_NAME -> $TARGET"
    ;;

  point)
    TARGET="${2:-}"; [[ -n "$TARGET" ]] || die "point needs a <target>"
    [[ -e "$BASE/$TARGET" || -e "$TARGET" ]] || die "target '$TARGET' does not exist under $BASE"
    CUR="$(resolve)"
    [[ -n "$CUR" ]] && run "printf '%s' '$CUR' > '$PREV'"
    # atomic swap: create temp symlink, then rename over the old one.
    TMP="$LINK.tmp.$$"
    run "ln -s '$TARGET' '$TMP'"
    run "mv -T '$TMP' '$LINK'"
    echo "repointed $LINK_NAME: ${CUR:-<none>} -> $TARGET"
    [[ -n "$CUR" ]] && echo "(previous target saved to $PREV_NAME for rollback)"
    ;;

  rollback)
    [[ -f "$PREV" ]] || die "no $PREV_NAME recorded; cannot auto-rollback"
    TARGET="$(cat "$PREV")"
    [[ -n "$TARGET" ]] || die "$PREV_NAME is empty"
    CUR="$(resolve)"
    TMP="$LINK.tmp.$$"
    run "ln -s '$TARGET' '$TMP'"
    run "mv -T '$TMP' '$LINK'"
    # swap prev to the value we just rolled away from, so a second
    # rollback returns to where we were (toggle).
    [[ -n "$CUR" ]] && run "printf '%s' '$CUR' > '$PREV'"
    echo "rolled back $LINK_NAME: ${CUR:-<none>} -> $TARGET"
    ;;

  ""|-h|--help)
    sed -n '1,40p' "$0"
    ;;
  *)
    die "unknown command '$CMD' (status|init|point|rollback|version-name)"
    ;;
esac
