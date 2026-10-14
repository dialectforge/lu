#!/usr/bin/env bash
# OP-RETRAIN-1: create a FRESH, EMPTY, well-formed v5 brain store for a clean
# retrain, via the proven v3->v5 migration route (2026-07-23, internal notes).
# Non-destructive: builds at a NEW path only; refuses if the target exists;
# never touches an existing brain or the verified backup.
#
# Usage: init_fresh_v5_brain.sh <target-v5-dir> [keyfile]
set -euo pipefail
BIN="${LU4D_BIN_DIR:-/opt/lu/lu/merge-main/target/release}"
TARGET="${1:?usage: init_fresh_v5_brain.sh <target-v5-dir> [keyfile]}"
KEY="${2:-/var/lib/lu/master.key}"
[[ -e "$TARGET" ]] && { echo "refusing: $TARGET already exists" >&2; exit 2; }
[[ -r "$KEY" ]] || { echo "keyfile not readable: $KEY" >&2; exit 2; }
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
V3="$WORK/seed_v3.lbr"
# 1) empty encrypted v3 brain (0 nodes / 0 bonds)
printf 'save\nquit\n' | "$BIN/lu4d" --brain "$V3" --keyfile "$KEY" --require-encrypted >/dev/null
# 2) v3 -> v5 node side  (MANIFEST + node_store/ + wal/nodeintern.bin)
"$BIN/lu_brain_migrate_v3_v5_node" --v3 "$V3" --v5 "$TARGET" --keyfile "$KEY" >/dev/null
# 3) v3 -> v5 bond side  (bond_store/)
"$BIN/lu_brain_migrate_v3_v5_bond" --v3 "$V3" --v5 "$TARGET" --keyfile "$KEY" >/dev/null
echo "fresh empty v5 brain ready: $TARGET"
# OP-IDENTITY: every brain MUST carry the locked, protected identity core.
# A brain cannot be considered initialised without it (required header).
echo "=== injecting protected identity core ==="
python3 /opt/lu/lu/merge-main/scripts/identity_loader.py "$TARGET" "$KEY"
"$BIN/lu4d" --brain "$TARGET" --brain-format v5 --keyfile "$KEY" <<< 'quit' 2>&1 | grep -i 'brain status'
