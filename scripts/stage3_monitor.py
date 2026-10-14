#!/usr/bin/env python3
"""
Stage 3 retrain monitor (OP-020: no-brain-load probe).

Reads canonical brain state WITHOUT loading the brain, by combining:
  * the curriculum runner's checkpoint JSON (per-corpus bonds_landed),
  * `systemctl is-active` / `systemctl show` for daemon liveness/PID,
  * the most recent `status_before bonds=X nodes=Y` line from the
    lu4d-curriculum-ingest journal (best-known node count -- may be
    minutes-to-hours stale, set to -1 if not present), and
  * disk + brain.lbr filesystem stats.

Appends one TSV row per invocation; sets flag files when halt-conditions
or the 100K-doc checkpoint are reached.

Pre-OP-020 versions spawned a second `lu_4_mcp --mcp --no-save` process
on each cron tick to call `lu4d_status`, which loaded the encrypted brain
fully into RAM. With the canonical brain at ~22-25 GB resident, that
second load drove the 30 GB Hetzner host past its physical RAM ceiling
and triggered host-level OOM kills against the daemon's lu_4_mcp child
every cron tick (OP-019 diagnosis; validation port confirmed). The
old `query_lu4d_status` and `sample_asymmetry` paths are gone; the
asymmetry-sample columns are kept in the TSV (placeholder values,
audit_exit=-9, audit_msg='disabled-op020') so the long-running TSV
remains column-stable for downstream parsers. Run
`scripts/audit_subject_provenance` manually when an asymmetry probe is
actually needed.

Designed to be invoked by cron every 30 minutes.

Outputs:
  /var/lib/lu/logs/stage3-monitor.tsv     -- the long-running TSV
  /var/lib/lu/logs/stage3-monitor.last    -- last-tick JSON for delta calc
  /var/lib/lu/logs/stage3-100k-reached.flag
  /var/lib/lu/logs/stage3-halt-<reason>.flag
  /var/lib/lu/logs/stage3-monitor.log     -- monitor's own stderr
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BRAIN          = "/var/lib/lu/brain.lbr"
CHECKPOINT     = "/var/lib/lu/ingest_full_checkpoint.json"
SERVICE_UNIT   = "lu4d-curriculum-ingest.service"

LOG_DIR        = Path("/var/lib/lu/logs")
TSV            = LOG_DIR / "stage3-monitor.tsv"
LAST           = LOG_DIR / "stage3-monitor.last"
MON_LOG        = LOG_DIR / "stage3-monitor.log"
FLAG_100K      = LOG_DIR / "stage3-100k-reached.flag"

# Halt-condition thresholds (mirrors brief Gate 6)
MIN_BONDS_PER_TICK     = 10        # bond growth stalls if < 10 in 30 min
ASYMMETRY_LEAK_PCT_MAX = 2.0       # >2% interrogative/leak in any sampled category
MIN_FREE_GB            = 10        # disk fill if < 10 GB

# Corpora that count toward the 100K-doc checkpoint (BR.R2 probe).
# Visual Genome last_saved_id is a triple position (many per image),
# not a document, so it is excluded.
DOC_CORPORA = {
    "wikimedia/wikipedia",
    "HuggingFaceH4/no_robots",
    "HuggingFaceH4/instruction-dataset",
}

_STATUS_BEFORE_RE = re.compile(
    r"status_before:\s*bonds=(\d+)\s+nodes=(\d+)"
)


def logmsg(s):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MON_LOG, "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {s}\n")


def raise_flag(reason, body=""):
    flag = LOG_DIR / f"stage3-halt-{reason}.flag"
    with open(flag, "w") as f:
        f.write(f"raised_at: {datetime.now(timezone.utc).isoformat()}\n")
        if body:
            f.write(body + "\n")
    logmsg(f"FLAG raised: {flag}")


def daemon_state():
    """Return (mainpid, is_active) from `systemctl is-active` + `show MainPID`.

    `systemctl is-active` returns exit 0 when active, non-zero otherwise.
    Neither call requires sudo for read-only queries.
    """
    try:
        rc_act = subprocess.run(
            ["systemctl", "is-active", SERVICE_UNIT],
            capture_output=True, text=True, timeout=10,
        )
        is_active = rc_act.returncode == 0
    except Exception as e:
        logmsg(f"systemctl is-active failed: {e!r}")
        is_active = False
    mainpid = -1
    try:
        rc_show = subprocess.run(
            ["systemctl", "show", SERVICE_UNIT, "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=10,
        )
        mainpid = int((rc_show.stdout or "").strip() or "-1")
    except Exception as e:
        logmsg(f"systemctl show MainPID failed: {e!r}")
    return mainpid, is_active


def last_known_nodes_from_journal():
    """Parse the most recent `status_before bonds=X nodes=Y` line.

    Emitted by ingest_curriculum_full.py once per (re)start of the
    lu4d-curriculum-ingest unit. Value is the brain's node_count at
    that moment -- may be minutes-to-hours stale between restarts.
    Returns (nodes, bonds_at_status_before) or (-1, -1) if no line
    found within the lookback window.
    """
    try:
        # 24h lookback is generous; the daemon has been ~99%+ uptime so
        # one status_before per restart is enough.
        rc = subprocess.run(
            ["journalctl", "-u", SERVICE_UNIT, "--since", "24 hours ago",
             "-q", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        logmsg(f"journalctl status_before scan failed: {e!r}")
        return -1, -1
    if rc.returncode != 0:
        logmsg(f"journalctl returned rc={rc.returncode}: {rc.stderr[:200]}")
        return -1, -1
    last_match = None
    for line in rc.stdout.splitlines():
        m = _STATUS_BEFORE_RE.search(line)
        if m:
            last_match = m
    if last_match is None:
        return -1, -1
    return int(last_match.group(2)), int(last_match.group(1))


def read_checkpoint():
    try:
        with open(CHECKPOINT) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def disabled_asym_stub():
    """Placeholder for the asymmetry sample (now disabled in OP-020).

    Pre-OP-020 ran target/release/examples/audit_subject_provenance, which
    loads the brain. Loading the brain in a second process is what OP-019
    diagnosed as the host-OOM trigger. Run audit_subject_provenance
    manually when an asymmetry check is needed; the daily/30-min monitor
    no longer samples it.
    """
    return {
        "n_total":          0,
        "declarative":      0,
        "interrogative":    0,
        "editorial":        0,
        "frame_noun":       0,
        "sample_seed":      0,
        "pct_interrogative": 0.0,
        "pct_editorial":    0.0,
        "pct_frame_noun":   0.0,
        "audit_exit":       -9,
        "audit_msg":        "disabled-op020",
    }


def tsv_header():
    return "\t".join([
        "# ts_utc",
        "pid_alive",
        "nodes",
        "bonds",
        "ratio_n_to_b",
        "degree_p90",
        "docs_processed",
        "d_nodes",
        "d_bonds",
        "d_docs",
        "bonds_per_min",
        "docs_per_min",
        "brain_size_bytes",
        "disk_free_gb",
        "asym_n",
        "asym_decl",
        "asym_interr",
        "asym_edit",
        "asym_frame",
        "asym_pct_interr",
        "asym_audit_exit",
        "asym_audit_msg",
        "checkpoint_corpora",
    ])


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts_iso = now.isoformat()

    pid, alive = daemon_state()

    nodes, _bonds_at_status_before = last_known_nodes_from_journal()
    # `bonds` for the TSV uses the LIVE per-corpus bonds_landed sum from
    # the checkpoint JSON (updated after every save by the daemon),
    # which is what we want for trend monitoring. The journal
    # status_before bonds is only the snapshot at daemon (re)start.
    cp = read_checkpoint()
    bonds_landed_total = 0
    docs_processed = 0
    corpus_summary = []
    for k, v in cp.items():
        if k == "updated_at":
            continue
        bonds_landed_total += int(v.get("bonds_landed", 0) or 0)
        last_id = v.get("last_saved_id", -1)
        if isinstance(last_id, int) and last_id > 0 and k in DOC_CORPORA:
            docs_processed += last_id + 1
        short = k.split("/")[-1]
        corpus_summary.append(
            f"{short}=b{v.get('bonds_landed', 0)}@id{v.get('last_saved_id', -1)}"
        )
    cp_summary = ",".join(corpus_summary) or "none"
    bonds = bonds_landed_total

    ratio = round(bonds / nodes, 3) if nodes > 0 else 0.0
    # degree_p90 is no longer probed (was only available via brain load).
    degree_p90 = -1

    try:
        brain_size = os.path.getsize(BRAIN)
    except FileNotFoundError:
        brain_size = -1

    du = shutil.disk_usage("/var/lib/lu")
    disk_free_gb = round(du.free / (1024**3), 2)

    last = {}
    if LAST.exists():
        try:
            last = json.loads(LAST.read_text())
        except Exception:
            last = {}
    d_nodes = nodes - last.get("nodes", nodes) if nodes >= 0 else 0
    d_bonds = bonds - last.get("bonds", bonds) if bonds >= 0 else 0
    d_docs  = docs_processed - last.get("docs_processed", docs_processed)
    last_ts = last.get("ts_iso")
    if last_ts:
        last_dt = datetime.fromisoformat(last_ts)
        elapsed_min = max(0.0001, (now - last_dt).total_seconds() / 60.0)
        bonds_per_min = round(d_bonds / elapsed_min, 1)
        docs_per_min  = round(d_docs / elapsed_min, 1)
    else:
        bonds_per_min = 0
        docs_per_min  = 0

    asym = disabled_asym_stub()

    # Halt conditions (raise flags; do NOT kill the process)
    if not alive:
        raise_flag("process-died",
                   f"systemctl is-active={alive} mainpid={pid} at {ts_iso}")
    if last_ts and d_bonds < MIN_BONDS_PER_TICK and d_docs < MIN_BONDS_PER_TICK:
        raise_flag("bond-stall",
                   f"d_bonds={d_bonds} d_docs={d_docs} since {last_ts}")
    if disk_free_gb < MIN_FREE_GB:
        raise_flag("disk-low", f"free={disk_free_gb} GB < {MIN_FREE_GB} GB")
    if nodes >= bonds and nodes > 0:
        raise_flag("ratio-inverted", f"nodes={nodes} >= bonds={bonds}")
    # Asymmetry gates remain in code but will never fire under OP-020
    # because the disabled stub returns pct=0.0 for all categories.
    for k in ("pct_interrogative", "pct_editorial", "pct_frame_noun"):
        v = asym.get(k, 0.0)
        if v > ASYMMETRY_LEAK_PCT_MAX:
            raise_flag(f"asymmetry-{k}", f"{k}={v}% > {ASYMMETRY_LEAK_PCT_MAX}%")

    if docs_processed >= 100_000 and not FLAG_100K.exists():
        with open(FLAG_100K, "w") as f:
            f.write(f"reached_at: {ts_iso}\ndocs_processed: {docs_processed}\n")
        logmsg(f"100K-doc checkpoint flag set: {docs_processed}")

    if not TSV.exists():
        with open(TSV, "w") as f:
            f.write(tsv_header() + "\n")

    row = "\t".join(str(x) for x in [
        ts_iso, alive, nodes, bonds, ratio, degree_p90,
        docs_processed, d_nodes, d_bonds, d_docs,
        bonds_per_min, docs_per_min, brain_size, disk_free_gb,
        asym.get("n_total", 0), asym.get("declarative", 0),
        asym.get("interrogative", 0), asym.get("editorial", 0),
        asym.get("frame_noun", 0), asym.get("pct_interrogative", 0),
        asym.get("audit_exit", -1), asym.get("audit_msg", ""),
        cp_summary,
    ])
    with open(TSV, "a") as f:
        f.write(row + "\n")

    LAST.write_text(json.dumps({
        "ts_iso": ts_iso, "nodes": nodes, "bonds": bonds,
        "docs_processed": docs_processed,
    }))

    print(row)
    logmsg(f"tick ok alive={alive} mainpid={pid} bonds={bonds} d_bonds={d_bonds} nodes={nodes}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logmsg(f"FATAL {e!r}")
        sys.exit(1)
