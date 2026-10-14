#!/usr/bin/env bash
# train.sh -- CANONICAL, one-command launcher for Lu v5 training (code/text ingest).
#
# This REPLACES the ephemeral `systemd-run` invocations (and the reconstructed
# scripts/relaunch_starcoderdata_ingest.sh) with a persistent, version-controlled
# launcher. It encodes the hard-won safe defaults from the 2026-07 sessions:
#
#   * LEAN load: the ingest driver boots lu_4_mcp with `--load-mode ingest-only`
#     (node-intern + memtable + WAL + flush only; NO resident degmap / keyword
#     index / block caches). Boot RSS ~15 GiB instead of ~55.8 GiB. This launcher
#     VERIFIES that after start and aborts if the child is not lean.
#   * Fixed cgroup caps: MemoryHigh=58G / MemoryMax=60G on the 62 GiB box (OP-RETRAIN-1: from-zero intern growth).
#   * HF token via EnvironmentFile=/etc/lu/hf_token.env.
#   * Checkpoint/resume: the driver is checkpointed + dedup-idempotent, so a
#     clean stop then `resume` continues from the last durable point safely.
#   * A HARD preflight gate (scripts/train_preflight.sh) runs before every
#     start/resume and REFUSES on any unsafe condition (another writer on the
#     brain, low disk/RAM, huge un-drained WAL, wrong binary).
#
# It is PARAMETERIZED by a source profile so it generalizes past StarCoderData.
#
# Usage:
#   scripts/train.sh start   [--source NAME] [overrides...]
#   scripts/train.sh resume  [--source NAME] [overrides...]   # alias of start; picks up checkpoint
#   scripts/train.sh stop    [--source NAME]                  # clean SIGTERM; driver checkpoints
#   scripts/train.sh status  [--source NAME]
#   scripts/train.sh monitor [--source NAME]                  # tail journal + progress
#   scripts/train.sh preflight [--source NAME]                # dry-run the gate only
#
# Overrides (any subcommand that launches):
#   --brain PATH  --keyfile PATH  --checkpoint PATH  --langs CSV
#   --per-lang-limit N  --limit N  --corpus TAG  --bin PATH
#   --mem-high SIZE  --mem-max SIZE  --load-mode MODE
#
# BRAIN-MUTATION POLICY: this launcher only ingests (append-only landing under
# --load-mode ingest-only, --no-save). It NEVER runs major compaction, reclaim,
# or any destructive step. Those remain HUMAN-GATED (see docs/TRAINING.md).
set -euo pipefail

# --------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts"
PREFLIGHT="$SCRIPTS_DIR/train_preflight.sh"

# ---- fixed, canonical defaults (verified against the live run 2026-07-21) --
DEF_BRAIN="/var/lib/lu/brain-v5-20260615-sst"
DEF_KEYFILE="/var/lib/lu/master.key"
DEF_BIN="/usr/local/bin/lu_4_mcp"
DEF_HF_ENV="/etc/lu/hf_token.env"
DEF_MEM_HIGH="58G"     # base-1024 -> 55834574848
DEF_MEM_MAX="60G"      # base-1024 -> 60129542144  (62 GiB box leaves ~6 GiB margin)
DEF_LOAD_MODE="ingest-only"
DRIVER="$SCRIPTS_DIR/stack_v2_loader.py"

# cgroup child memory environment (matches the proven lean unit)
CHILD_ENV_ARGS=(
  --setenv=LU_BLOCK_CACHE_BYTES=2G
  --setenv=LU_NODE_READER_CACHE_CAP=64
  --setenv=LU_BOND_READER_CACHE_CAP=128
  # OP-RETRAIN-1: concept-admission gate + canonicalisation-at-ingest ON for
  # the clean rebuild (reject fragments/category-lines/bare code identifiers;
  # fold case/article variants). Default is OFF in the binary; the retrain
  # explicitly opts in here so serving/other tools are unaffected.
  --setenv=LU_CANON_INGEST=1
  # OP-RETRAIN-1: flat glibc arenas keep ingest RSS bounded/flat.
  --setenv=MALLOC_ARENA_MAX=2
)

# ---- source profiles -------------------------------------------------------
# Each profile sets: SRC, LANGS, PER_LANG_LIMIT, LIMIT, CORPUS, CKPT, DESC, DRIVER
# Add new datasets here; the launcher machinery is dataset-agnostic.
load_profile() {
  case "$1" in
    starcoderdata)
      SRC="starcoderdata"
      LANGS="python,javascript,typescript,rust,go,java,c,cpp"
      PER_LANG_LIMIT="500000"
      LIMIT="0"
      CORPUS="starcoderdata"
      CKPT="/var/lib/lu/starcoderdata_ingest_checkpoint.json"
      DESC="StarCoderData code ingest (per-lang capped 500k, corpus=starcoderdata)"
      DRIVER="$SCRIPTS_DIR/stack_v2_loader.py"
      ;;
    the-stack-v2)
      SRC="hf"
      LANGS="python,javascript,typescript,rust,go,java,c,cpp"
      PER_LANG_LIMIT="500000"
      LIMIT="0"
      CORPUS="the-stack-v2"
      CKPT="/var/lib/lu/the_stack_v2_ingest_checkpoint.json"
      DESC="The Stack v2 gated code ingest (per-lang capped 500k, corpus=the-stack-v2)"
      DRIVER="$SCRIPTS_DIR/stack_v2_loader.py"
      ;;
    cat1)
      SRC="cat1"
      LANGS=""
      PER_LANG_LIMIT="0"
      LIMIT="0"
      CORPUS="cat1-retrain"
      CKPT="/var/lib/lu/cat1_retrain_checkpoint.json"
      DESC="Cat.1 foundation-first retrain: Tier0 WordNet(is-a+glosses)+GCIDE-dictionary -> Tier1 Visual Genome -> Wikipedia; Tier2 arxiv/stackexchange + coding[code_search_net,stackoverflow,mdn,python_docs,python_code,stack_v2]; grokipedia deferred/skipped"
      DRIVER="$SCRIPTS_DIR/ingest_curriculum_full.py"
      CAT1_TIERS=(--tier 1 --tier 2)
      ;;
    *)
      echo "train.sh: unknown --source '$1'." >&2
      echo "  Known profiles: starcoderdata, the-stack-v2, cat1." >&2
      echo "  For a new dataset, add a profile in load_profile() (keeps launches version-controlled)." >&2
      exit 64
      ;;
  esac
}

# ---- arg parse -------------------------------------------------------------
CMD="${1:-}"; shift || true
SOURCE="starcoderdata"
# collect overrides
OV_BRAIN=""; OV_KEYFILE=""; OV_CKPT=""; OV_LANGS=""; OV_PLL=""; OV_LIMIT=""
OV_CORPUS=""; OV_BIN=""; OV_MEM_HIGH=""; OV_MEM_MAX=""; OV_LOAD_MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)         SOURCE="$2"; shift 2;;
    --brain)          OV_BRAIN="$2"; shift 2;;
    --keyfile)        OV_KEYFILE="$2"; shift 2;;
    --checkpoint)     OV_CKPT="$2"; shift 2;;
    --langs)          OV_LANGS="$2"; shift 2;;
    --per-lang-limit) OV_PLL="$2"; shift 2;;
    --limit)          OV_LIMIT="$2"; shift 2;;
    --corpus)         OV_CORPUS="$2"; shift 2;;
    --bin)            OV_BIN="$2"; shift 2;;
    --mem-high)       OV_MEM_HIGH="$2"; shift 2;;
    --mem-max)        OV_MEM_MAX="$2"; shift 2;;
    --load-mode)      OV_LOAD_MODE="$2"; shift 2;;
    -h|--help)        sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "train.sh: unknown arg: $1" >&2; exit 64;;
  esac
done

load_profile "$SOURCE"

# resolve effective config (override > profile/default)
BRAIN="${OV_BRAIN:-$DEF_BRAIN}"
KEYFILE="${OV_KEYFILE:-$DEF_KEYFILE}"
CKPT="${OV_CKPT:-$CKPT}"
LANGS="${OV_LANGS:-$LANGS}"
PER_LANG_LIMIT="${OV_PLL:-$PER_LANG_LIMIT}"
LIMIT="${OV_LIMIT:-$LIMIT}"
CORPUS="${OV_CORPUS:-$CORPUS}"
BIN="${OV_BIN:-$DEF_BIN}"
MEM_HIGH="${OV_MEM_HIGH:-$DEF_MEM_HIGH}"
MEM_MAX="${OV_MEM_MAX:-$DEF_MEM_MAX}"
LOAD_MODE="${OV_LOAD_MODE:-$DEF_LOAD_MODE}"
UNIT="lu4d-${SOURCE}-ingest"
SYSLOG_ID="lu4d-${SOURCE}"
MCP_STDERR="/var/lib/lu/${SOURCE}_ingest_mcp.stderr"

run_preflight() {
  echo ">>> Running hard preflight gate..."
  LU_BRAIN="$BRAIN" LU4D_BIN="$BIN" LU_LOAD_MODE="$LOAD_MODE" LU_UNIT="$UNIT" \
    bash "$PREFLIGHT" --brain "$BRAIN" --bin "$BIN" --load-mode "$LOAD_MODE" --unit "$UNIT"
}

verify_lean() {
  # After launch, confirm the spawned lu_4_mcp is actually lean + no-save.
  echo ">>> Verifying the running child is lean (--load-mode $LOAD_MODE, --no-save)..."
  local tries=0 line=""
  while (( tries < 30 )); do
    line="$(pgrep -af "lu_4_mcp" 2>/dev/null | grep -F -- "--brain $BRAIN" || true)"
    if [[ -n "$line" ]]; then break; fi
    sleep 1; tries=$((tries+1))
  done
  if [[ -z "$line" ]]; then
    echo "!!! could not observe a lu_4_mcp child within 30s -- check: journalctl -u $UNIT" >&2
    return 1
  fi
  if ! grep -q -- "--load-mode $LOAD_MODE" <<<"$line"; then
    echo "!!! child is NOT running --load-mode $LOAD_MODE. Aborting to protect RAM." >&2
    echo "    offending cmdline: $line" >&2
    echo "    stopping $UNIT ..." >&2
    sudo systemctl stop "$UNIT" || true
    return 3
  fi
  if ! grep -q -- "--no-save" <<<"$line"; then
    echo "!!! child is missing --no-save (would autosave the full brain). Aborting." >&2
    sudo systemctl stop "$UNIT" || true
    return 3
  fi
  echo "    OK -- lean child confirmed."
}

do_start() {
  echo "=== train.sh ${CMD} source=${SOURCE} ==="
  echo "    unit=$UNIT  brain=$BRAIN  corpus=$CORPUS"
  echo "    load-mode=$LOAD_MODE  caps=High:$MEM_HIGH/Max:$MEM_MAX  per-lang-limit=$PER_LANG_LIMIT"
  echo "    driver=$DRIVER  bin=$BIN"
  echo

  [[ -x "$PREFLIGHT" || -f "$PREFLIGHT" ]] || { echo "missing preflight: $PREFLIGHT" >&2; exit 1; }
  [[ -f "$DRIVER" ]] || { echo "missing driver: $DRIVER" >&2; exit 1; }
  [[ -f "$DEF_HF_ENV" ]] || echo "WARN: HF env file $DEF_HF_ENV not found (gated sources will fail auth)" >&2

  run_preflight   # exits non-zero (via set -e) if the gate refuses -> we never launch

  echo
  echo ">>> Preflight passed. Launching persistent transient unit $UNIT ..."
  # OP-RETRAIN-1: driver-specific argv. The code loaders (stack_v2_loader.py)
  # and the Cat.1 locked curriculum (ingest_curriculum_full.py) accept different
  # flags, so the arg list is chosen per driver rather than hardcoded.
  if [[ "$DRIVER" == *ingest_curriculum_full.py ]]; then
    DRIVER_ARGV=(--brain "$BRAIN" --keyfile "$KEYFILE" --checkpoint "$CKPT" \
      --mcp-stderr "$MCP_STDERR" --ingest-only --seed-trust \
      --coding-sources code_search_net,stackoverflow,mdn,python_docs,python_code,stack_v2 \
      --pycode-root /usr/lib/python3.12 \
      "${CAT1_TIERS[@]}")
  else
    DRIVER_ARGV=(--source "$SRC" --seed-curriculum --brain "$BRAIN" \
      --keyfile "$KEYFILE" --langs "$LANGS" --limit "$LIMIT" \
      --per-lang-limit "$PER_LANG_LIMIT" --checkpoint "$CKPT" \
      --mcp-stderr "$MCP_STDERR")
  fi

  sudo systemd-run \
    --unit="$UNIT" \
    --description="$DESC" \
    --property=Type=simple \
    --property=User=dialectforge \
    --property=Group=lu \
    --property=WorkingDirectory="$REPO_DIR" \
    --property=UMask=0027 \
    --property=EnvironmentFile="$DEF_HF_ENV" \
    --property=Restart=on-failure \
    --property=RestartSec=15s \
    --property=MemoryHigh="$MEM_HIGH" \
    --property=MemoryMax="$MEM_MAX" \
    --property=SyslogIdentifier="$SYSLOG_ID" \
    "${CHILD_ENV_ARGS[@]}" \
    /usr/bin/python3 -u "$DRIVER" "${DRIVER_ARGV[@]}"

  echo ">>> Started. (resume is idempotent from checkpoint: $CKPT)"
  verify_lean || { echo "lean verification FAILED -- see messages above." >&2; exit 3; }
  echo
  do_status
}

do_stop() {
  echo "=== train.sh stop source=$SOURCE (unit=$UNIT) ==="
  if systemctl is-active --quiet "$UNIT"; then
    echo ">>> Sending clean stop (SIGTERM). Driver checkpoints; resume is safe."
    sudo systemctl stop "$UNIT"
    echo ">>> Stopped. Checkpoint: $CKPT"
  else
    echo "unit $UNIT is not active; nothing to stop."
  fi
}

do_status() {
  echo "=== status: $UNIT ==="
  systemctl --no-pager --full status "$UNIT" 2>/dev/null | head -12 || true
  echo "--- memory ---"
  systemctl show "$UNIT" -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax 2>/dev/null || true
  echo "--- checkpoint tail ($CKPT) ---"
  [[ -f "$CKPT" ]] && tail -c 400 "$CKPT" 2>/dev/null && echo || echo "(no checkpoint yet)"
}

do_monitor() {
  echo "=== monitor: $UNIT  (Ctrl-C to detach; does not stop the run) ==="
  sudo journalctl -u "$UNIT" -f --no-hostname
}

case "$CMD" in
  start|resume) do_start;;
  stop)         do_stop;;
  status)       do_status;;
  monitor)      do_monitor;;
  preflight)    run_preflight;;
  ""|-h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}";;
  *) echo "train.sh: unknown command '$CMD' (start|resume|stop|status|monitor|preflight)" >&2; exit 64;;
esac
