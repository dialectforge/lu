#!/usr/bin/env python3
"""OP-CHUNK-1 — chunked-training wrapper for lu4d-curriculum-ingest.

External wrapper that owns the chunk-loop logic. Polls the daemon's
checkpoint file; when the configured chunk-size threshold is crossed,
waits for any in-flight save to complete, then issues
`sudo -n /bin/systemctl restart lu4d-curriculum-ingest`, waits for the
new daemon to reach mcp_init completion, and logs a structured JSONL
record per chunk.

Design rationale + race conditions: /tmp/op_chunk1_phase_b.md
(also outputs/op_chunk1_phase_b.md on the dev machine).

Usage:
    python3 scripts/chunked_trainer.py --chunk-size 50000
    python3 scripts/chunked_trainer.py --chunk-size 100 --max-chunks 1   # dry-run
    python3 scripts/chunked_trainer.py --chunk-size 50000 --max-chunks 1 # phase E

Production constraints honored:
  - OLD binary stays in place (does not swap to NEW).
  - No daemon stops except restarts on chunk boundaries.
  - Cursor resume uses the daemon's existing checkpoint mechanism.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_PATH = "/var/lib/lu/ingest_full_checkpoint.json"
BRAIN_PATH      = "/var/lib/lu/brain.lbr"
BRAIN_TMP_PATH  = "/var/lib/lu/brain.lbr.tmp"
CGROUP_BASE     = "/sys/fs/cgroup/system.slice/lu4d-curriculum-ingest.service"
SERVICE_NAME    = "lu4d-curriculum-ingest"
CHUNK_LOG_PATH  = "/var/lib/lu/chunk_log.jsonl"
WRAPPER_VERSION = "OP-CHUNK-1.0"
WIKI_CORPUS     = "wikimedia/wikipedia"


# ---------- helpers ----------

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{now_utc_iso()}] {msg}", flush=True)


def run(cmd, check=True, capture=True, timeout=60):
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed (rc={r.returncode}): {cmd}\nstderr: {r.stderr}")
    return r


def read_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log(f"WARN: checkpoint read failed: {e}; treating as empty")
        return {}


def wiki_articles(cp):
    """Return current wikipedia last_saved_id (articles processed)."""
    return cp.get(WIKI_CORPUS, {}).get("last_saved_id", -1)


def total_bonds(cp):
    total = 0
    for k, v in cp.items():
        if isinstance(v, dict):
            total += v.get("bonds_landed", 0)
    return total


def brain_sha():
    """SHA256 of brain.lbr (full file). Short for log readability."""
    h = hashlib.sha256()
    with open(BRAIN_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tmp_exists():
    return os.path.exists(BRAIN_TMP_PATH)


def systemctl_show(field):
    r = run(["systemctl", "show", SERVICE_NAME, "-p", field])
    line = r.stdout.strip()
    if "=" in line:
        return line.split("=", 1)[1]
    return ""


def main_pid():
    v = systemctl_show("MainPID")
    return int(v) if v.isdigit() else 0


def active_enter_ts():
    """ActiveEnterTimestamp as epoch seconds, or 0."""
    raw = systemctl_show("ActiveEnterTimestamp")
    if not raw:
        return 0
    # Format: "Thu 2026-05-21 21:46:51 UTC"
    try:
        # strip weekday + zone
        parts = raw.split()
        if len(parts) >= 3:
            dt = datetime.strptime(parts[1] + " " + parts[2], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        pass
    return 0


def nrestarts():
    v = systemctl_show("NRestarts")
    return int(v) if v.isdigit() else -1


def is_active():
    r = subprocess.run(["systemctl", "is-active", SERVICE_NAME],
                       capture_output=True, text=True)
    return r.stdout.strip() == "active"


def journal_status_before_ts(since_epoch):
    """Return epoch seconds of latest `status_before` journal line at or after `since_epoch`.
    Returns 0 if not found."""
    since_str = datetime.fromtimestamp(since_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    r = subprocess.run(
        ["sudo", "-n", "journalctl", "-u", SERVICE_NAME,
         "--since", since_str, "--no-pager", "--output=short-iso"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return 0
    # Look for "status_before: bonds=X nodes=Y" lines.
    # Journalctl --output=short-iso format: "2026-05-22T03:35:21+0000 host service[pid]: msg"
    pat = re.compile(r"^(\S+)\s+\S+\s+\S+\s+.*status_before:")
    latest_ts = 0
    for line in r.stdout.splitlines():
        m = pat.match(line)
        if not m:
            continue
        try:
            ts_str = m.group(1)
            # short-iso uses "+0000" instead of "Z"; strptime needs %z
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S%z")
            ts = int(dt.timestamp())
            if ts > latest_ts:
                latest_ts = ts
        except Exception:
            continue
    return latest_ts


def find_lu_4_mcp_pid(parent_pid):
    """Return the PID of the lu_4_mcp child of parent_pid, or None.
    Uses `pgrep -P <parent>` and filters to the lu_4_mcp binary.
    Raises RuntimeError if multiple lu_4_mcp children exist (shouldn't)."""
    if parent_pid <= 0:
        return None
    r = subprocess.run(
        ["pgrep", "-P", str(parent_pid)],
        capture_output=True, text=True, timeout=5,
    )
    if r.returncode != 0:
        return None
    matches = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            child_pid = int(line)
        except ValueError:
            continue
        # Confirm it's lu_4_mcp via /proc/<pid>/comm
        try:
            comm = Path(f"/proc/{child_pid}/comm").read_text().strip()
        except (FileNotFoundError, PermissionError):
            continue
        if comm == "lu_4_mcp":
            matches.append(child_pid)
    if len(matches) > 1:
        raise RuntimeError(f"multiple lu_4_mcp children of PID {parent_pid}: {matches}")
    return matches[0] if matches else None


def read_cpu_ticks(pid):
    """Return utime+stime (clock ticks) for `pid`, or None if not readable.
    Reads /proc/<pid>/stat; fields 14 (utime) and 15 (stime) per `man proc`.
    The stat line uses parentheses around the comm field which may contain
    spaces, so we slice on the LAST ')' to skip past comm safely."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    rp = raw.rfind(")")
    if rp < 0:
        return None
    fields = raw[rp + 1:].split()
    # After ')' the first field is state, then ppid, pgrp, session, tty_nr,
    # tpgid, flags, minflt, cminflt, majflt, cmajflt, utime, stime, ...
    # That makes utime at index 11 (0-indexed) post-')' and stime at 12.
    if len(fields) < 13:
        return None
    try:
        return int(fields[11]) + int(fields[12])
    except (ValueError, IndexError):
        return None


def journal_line_count_since(since_epoch):
    """Count journal lines from lu4d-curriculum-ingest since `since_epoch`.
    Used by the progress-based watchdog (OP-CHUNK-1 Phase D.1): if line
    count doesn't increase for `silence_threshold_sec`, declare hang."""
    since_str = datetime.fromtimestamp(since_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    r = subprocess.run(
        ["sudo", "-n", "journalctl", "-u", SERVICE_NAME,
         "--since", since_str, "--no-pager"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.strip())


def cg_memory_current():
    try:
        with open(f"{CGROUP_BASE}/memory.current") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def cg_memory_peak():
    try:
        with open(f"{CGROUP_BASE}/memory.peak") as f:
            return int(f.read().strip())
    except Exception:
        return 0


# ---------- chunk-loop ----------

def wait_for_tmp_gone(silence_threshold_sec=300, poll=10):
    """Wait until brain.lbr.tmp does NOT exist (no save in flight).

    OP-CHUNK-1 Phase D.2.fix: progress-based watchdog. The OLD-binary
    save scales with bond count (~25-30 min for the 10.85M-bond brain),
    so a fixed deadline never holds long-term. Watch the tmp file size;
    if it stops growing for `silence_threshold_sec` (default 5 min),
    declare hang. Otherwise stay patient.

    Returns total wait duration in seconds on success (tmp file gone).
    Raises RuntimeError on stall (size unchanged for silence_threshold_sec).

    Edge cases:
      - tmp file absent at start: returns 0 immediately.
      - tmp file appears briefly then disappears: returns at the moment
        it disappears.
      - file removed between exists() and getsize(): treated as gone.
    """
    start_ts = time.time()
    last_size = -1
    last_activity_ts = time.time()
    while True:
        if not tmp_exists():
            return int(time.time() - start_ts)
        try:
            size = os.path.getsize(BRAIN_TMP_PATH)
        except FileNotFoundError:
            return int(time.time() - start_ts)
        if size != last_size:
            last_size = size
            last_activity_ts = time.time()
        silence = int(time.time() - last_activity_ts)
        if silence > silence_threshold_sec:
            raise RuntimeError(
                f"save stalled: brain.lbr.tmp size unchanged at {size} bytes "
                f"for {silence}s (threshold {silence_threshold_sec}s); "
                f"total wait {int(time.time() - start_ts)}s"
            )
        time.sleep(poll)


def wait_for_chunk_threshold(start_articles, chunk_size, poll=30, peak_samples=None):
    """Poll checkpoint until wiki articles - start >= chunk_size.
    Also tracks peak cgroup memory.current during the chunk into peak_samples list.
    Returns the checkpoint dict at the moment the threshold was crossed."""
    log(f"watching for +{chunk_size} docs since {start_articles}; poll every {poll}s")
    while True:
        cur_mem = cg_memory_current()
        if peak_samples is not None:
            peak_samples.append(cur_mem)
        cp = read_checkpoint()
        cur_articles = wiki_articles(cp)
        delta = cur_articles - start_articles
        log(f"  articles={cur_articles} (+{delta}/{chunk_size})  current={cur_mem/1024**3:.2f}GiB  tmp={'yes' if tmp_exists() else 'no'}")
        if delta >= chunk_size:
            return cp
        time.sleep(poll)


def restart_and_wait(initiated_ts, silence_threshold_sec=300, new_pid_timeout=120, poll=10):
    """Issue systemctl restart, wait for new MainPID + status_before journal line.

    OP-CHUNK-1 Phase D.3.fix: HYBRID progress-based watchdog.
    - Phase 1 (new MainPID): bounded by `new_pid_timeout` (systemd-level).
    - Phase 2 (mcp_init): NO upper bound on total wait. Two activity signals
      watched in parallel:
        * journal line count for lu4d-curriculum-ingest (catches early
          orchestrator output: 'INGEST FULL START', 'checkpoint:').
        * lu_4_mcp child CPU ticks via /proc/<pid>/stat utime+stime
          (the actual brain-load + keyword_index-rebuild work).
      `last_activity_ts` resets on EITHER signal. Hang = silence on BOTH
      for `silence_threshold_sec`.

    Why hybrid: D.1 (journal-only) fired false positives because OLD-binary
    mcp_init at 10.85M bonds is journal-silent for ~20 min between
    `checkpoint:` and `status_before:`. The Python parent is blocked on the
    MCP handshake; the Rust child is doing all the work. CPU time on the
    child is the always-present signal.

    Returns (active_enter_ts, status_before_ts) on success.
    Raises RuntimeError on hang or systemd timeout."""
    pre_pid = main_pid()
    pre_aets = active_enter_ts()
    log(f"restart initiated; pre PID={pre_pid} ActiveEnterTimestamp={pre_aets}")
    r = subprocess.run(
        ["sudo", "-n", "/bin/systemctl", "restart", SERVICE_NAME],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"systemctl restart failed: rc={r.returncode} stderr={r.stderr}")
    log("restart command returned; polling for new MainPID")

    # Phase 1: wait for new MainPID (systemd-level signal, bounded timeout).
    phase1_deadline = time.time() + new_pid_timeout
    new_aets = 0
    new_pid = 0
    while time.time() < phase1_deadline:
        new_pid = main_pid()
        new_aets = active_enter_ts()
        if (new_pid > 0 and new_pid != pre_pid and is_active()
                and new_aets > 0 and new_aets > pre_aets):
            log(f"  new MainPID={new_pid}, ActiveEnterTimestamp={new_aets}")
            break
        time.sleep(poll)
    else:
        raise RuntimeError(f"timeout waiting for new MainPID after {new_pid_timeout}s (systemd-level)")

    # Phase 2: HYBRID progress-based watchdog for mcp_init completion.
    # OP-CHUNK-1 Phase D.3.fix: journal-only was insufficient. OLD-binary
    # mcp_init at 10.85M+ bonds emits zero journal lines for ~20 min
    # between `checkpoint:` and `status_before:`. The Python orchestrator
    # is blocked on the MCP handshake; the lu_4_mcp child is the one
    # actually working (faulting in brain, walking corpus_index, building
    # the keyword HashMap). Watch BOTH signals; reset on EITHER.
    last_activity_ts = time.time()
    prev_line_count = journal_line_count_since(new_aets - 5)

    # Discover the lu_4_mcp child PID. Allow ~30s for it to spawn.
    child_pid = None
    child_discovery_deadline = time.time() + 30
    while time.time() < child_discovery_deadline:
        child_pid = find_lu_4_mcp_pid(new_pid)
        if child_pid is not None:
            break
        time.sleep(2)
    if child_pid is None:
        raise RuntimeError(
            f"could not find lu_4_mcp child of MainPID {new_pid} within 30s"
        )
    log(f"  lu_4_mcp child PID={child_pid}; baseline journal lines={prev_line_count}; silence threshold {silence_threshold_sec}s")
    prev_cpu_ticks = read_cpu_ticks(child_pid)
    if prev_cpu_ticks is None:
        raise RuntimeError(f"could not read /proc/{child_pid}/stat for cpu ticks")

    while True:
        # Check for status_before first (the success signal).
        sb = journal_status_before_ts(new_aets - 5)
        if sb > 0 and sb >= new_aets:
            mcp_init_dur = sb - new_aets
            log(f"  status_before fired (mcp_init complete in {mcp_init_dur}s; total restart {int(time.time() - initiated_ts)}s)")
            return new_aets, sb

        activity_sources = []

        # Signal A: journal line count delta.
        cur_count = journal_line_count_since(new_aets - 5)
        if cur_count > prev_line_count:
            activity_sources.append(f"journal +{cur_count - prev_line_count}")
            prev_line_count = cur_count

        # Signal B: lu_4_mcp child CPU ticks delta.
        # Verify child is still the same PID (sanity check).
        current_child = find_lu_4_mcp_pid(new_pid)
        if current_child is None:
            raise RuntimeError(
                f"lu_4_mcp child of MainPID {new_pid} disappeared mid-wait "
                f"(was PID {child_pid}); total elapsed {int(time.time() - initiated_ts)}s"
            )
        if current_child != child_pid:
            raise RuntimeError(
                f"lu_4_mcp child PID changed mid-wait: was {child_pid}, now {current_child}; "
                f"total elapsed {int(time.time() - initiated_ts)}s"
            )
        cur_cpu_ticks = read_cpu_ticks(child_pid)
        if cur_cpu_ticks is None:
            raise RuntimeError(
                f"lost cpu-ticks signal for PID {child_pid}; total elapsed {int(time.time() - initiated_ts)}s"
            )
        if cur_cpu_ticks > prev_cpu_ticks:
            activity_sources.append(f"cpu +{cur_cpu_ticks - prev_cpu_ticks}ticks")
            prev_cpu_ticks = cur_cpu_ticks

        if activity_sources:
            last_activity_ts = time.time()
            log(f"  activity: {', '.join(activity_sources)}; reset; total elapsed {int(time.time() - initiated_ts)}s")

        silence = int(time.time() - last_activity_ts)
        if silence > silence_threshold_sec:
            raise RuntimeError(
                f"mcp_init hung: no journal-or-cpu activity for {silence}s "
                f"(threshold {silence_threshold_sec}s); total elapsed "
                f"{int(time.time() - initiated_ts)}s"
            )
        log(f"  waiting for status_before; silence={silence}s/{silence_threshold_sec}s; total elapsed {int(time.time() - initiated_ts)}s")
        time.sleep(poll)


def append_chunk_log(record):
    Path(CHUNK_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    log(f"chunk_log row appended to {CHUNK_LOG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, required=True,
                    help="docs (wikipedia articles) per chunk")
    ap.add_argument("--max-chunks", type=int, default=1,
                    help="max chunks to run before halting (default 1)")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="seconds between checkpoint polls (default 30)")
    args = ap.parse_args()

    log(f"=== chunked_trainer.py START — chunk_size={args.chunk_size} max_chunks={args.max_chunks} ===")
    log(f"  wrapper_version={WRAPPER_VERSION} pid={os.getpid()}")

    binary_md5 = subprocess.check_output(["md5sum", "/usr/local/bin/lu_4_mcp"], text=True).split()[0]
    log(f"  binary md5={binary_md5}")
    log(f"  service is-active={is_active()} MainPID={main_pid()} NRestarts={nrestarts()}")

    chunk_id = 0
    while chunk_id < args.max_chunks:
        chunk_id += 1
        log(f"\n--- chunk {chunk_id} / {args.max_chunks} ---")

        # Wait for any in-flight save to finish before starting the chunk.
        log("pre-chunk: waiting for any in-flight save to complete (tmp-gone gate)")
        try:
            pre_chunk_tmp_wait_sec = wait_for_tmp_gone()
            log(f"  pre-chunk tmp-gone after {pre_chunk_tmp_wait_sec}s")
        except RuntimeError as e:
            log(f"FATAL pre-chunk wait_for_tmp_gone: {e}")
            sys.exit(2)

        cp_start = read_checkpoint()
        start_ts = now_utc_iso()
        start_ts_epoch = int(time.time())
        start_articles = wiki_articles(cp_start)
        start_bonds = total_bonds(cp_start)
        start_sha = brain_sha()
        log(f"chunk start: articles={start_articles} bonds={start_bonds} brain_sha={start_sha[:12]}")

        peak_samples = []
        cp_end = wait_for_chunk_threshold(start_articles, args.chunk_size,
                                          poll=args.poll_interval,
                                          peak_samples=peak_samples)

        log("threshold reached; waiting for any in-flight save to complete before restart")
        try:
            pre_restart_tmp_wait_sec = wait_for_tmp_gone()
            log(f"  pre-restart tmp-gone after {pre_restart_tmp_wait_sec}s")
        except RuntimeError as e:
            log(f"FATAL pre-restart wait_for_tmp_gone: {e}")
            sys.exit(3)

        end_articles = wiki_articles(cp_end)
        end_bonds = total_bonds(cp_end)
        end_sha_pre_restart = brain_sha()
        restart_initiated_ts_epoch = int(time.time())
        restart_initiated_ts = now_utc_iso()

        try:
            new_aets, sb_ts = restart_and_wait(restart_initiated_ts_epoch)
        except RuntimeError as e:
            log(f"FATAL restart/wait failed: {e}")
            sys.exit(4)

        restart_complete_ts_epoch = sb_ts
        restart_complete_ts = datetime.fromtimestamp(restart_complete_ts_epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_ts = restart_complete_ts
        end_sha = brain_sha()  # post-restart, should match pre-restart unless a save fired in between

        # Count saves during chunk: brain SHA changes recorded by polling above.
        # Conservative: just record start vs end SHAs; saves_during_chunk requires
        # per-poll SHA capture which is expensive (~3-5s per sample). Skip for now;
        # OP-CHUNK-2 can revisit if needed.
        record = {
            "chunk_id": chunk_id,
            "wrapper_pid": os.getpid(),
            "wrapper_version": WRAPPER_VERSION,
            "chunk_size_target": args.chunk_size,
            "start_ts": start_ts,
            "start_brain_sha": start_sha,
            "start_articles": start_articles,
            "start_bonds": start_bonds,
            "end_ts": end_ts,
            "end_brain_sha": end_sha,
            "end_articles": end_articles,
            "end_bonds": end_bonds,
            "docs_processed": end_articles - start_articles,
            "bonds_added": end_bonds - start_bonds,
            "chunk_wall_duration_sec": restart_initiated_ts_epoch - start_ts_epoch,
            "restart_initiated_ts": restart_initiated_ts,
            "restart_complete_ts": restart_complete_ts,
            "restart_duration_sec": restart_complete_ts_epoch - restart_initiated_ts_epoch,
            "mcp_init_duration_sec": restart_complete_ts_epoch - new_aets,
            "peak_rss_during_chunk_bytes": max(peak_samples) if peak_samples else 0,
            "peak_rss_during_chunk_GiB": round(max(peak_samples) / 1024**3, 2) if peak_samples else 0,
            "samples_count": len(peak_samples),
            "pre_chunk_tmp_wait_sec": pre_chunk_tmp_wait_sec,
            "pre_restart_tmp_wait_sec": pre_restart_tmp_wait_sec,
            "nrestarts_at_chunk_end": nrestarts(),
            "binary_md5": binary_md5,
            "pre_restart_brain_sha": end_sha_pre_restart,
        }
        append_chunk_log(record)
        log(f"chunk {chunk_id} done: docs={record['docs_processed']} bonds_added={record['bonds_added']} "
            f"chunk_wall={record['chunk_wall_duration_sec']}s restart={record['restart_duration_sec']}s "
            f"peak={record['peak_rss_during_chunk_GiB']}GiB")

    log(f"=== chunked_trainer.py DONE — completed {chunk_id} chunk(s) ===")


if __name__ == "__main__":
    main()
