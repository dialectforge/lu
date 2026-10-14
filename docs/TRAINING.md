# Lu v5 — Canonical Training Runbook

**Status:** canonical procedure as of 2026-07-21. Supersedes ad-hoc `systemd-run`
launches. The one-command entrypoint is `scripts/train.sh`; the safety gate is
`scripts/train_preflight.sh`; the backup tool is `scripts/backup_brain.sh`.

**Golden rules**

1. **Serving and training never touch the brain at the same time.** Exactly one
   `lu_4_mcp` process may be bound to the canonical brain. Training is a writer;
   serving needs the recall residency. They are mutually exclusive on one brain.
2. **Training boots LEAN** (`--load-mode ingest-only`). The serving structures
   (resident degree-map, keyword index, block caches) are deliberately absent
   during ingest and are rebuilt afterwards for serving.
3. **No destructive brain automation runs unattended.** Major compaction,
   reclaim, and deletes are HUMAN-GATED. And none of them should run at all until
   a verified backup exists (see "Backups").
4. **The launcher refuses unsafe launches.** If the preflight gate fails, fix the
   condition — do not bypass it.

---

## 0. Ground truth (the box)

| Thing | Value |
|---|---|
| Host | `lu` -> `203.0.113.10:2222`, user `dialectforge`, box `the Lu server` |
| Repo | `/opt/lu/lu/v4` |
| Canonical brain | `/var/lib/lu/brain-v5-20260615-sst` (~198 GiB) |
| Keyfile | `/var/lib/lu/master.key` |
| Binary | `/usr/local/bin/lu_4_mcp` (supports `--load-mode`; override via `LU4D_BIN`) |
| HF token | `EnvironmentFile=/etc/lu/hf_token.env` |
| Disk | `/dev/md2` RAID1 mirror, 436 GiB, ~188 GiB free |
| RAM | 62 GiB physical, 31 GiB swap |
| cgroup caps | `MemoryHigh=52G` / `MemoryMax=56G` (leaves ~6 GiB OS margin) |

The brain (~198 GiB) is **larger than the free disk (~188 GiB)** — so there is no
room for a second local copy. This single fact drives the backup strategy and the
compaction cautions below.

---

## 1. Why lean mode (`--load-mode ingest-only`)

The serving `lu_4_mcp` boot loads the full recall residency: the resident
degree-map, the keyword index, and the cache-backed adjacency/seam. On this brain
that pushed **boot RAM to ~55.8 GiB** — right up against the cap, causing stalls
and swap thrash during ingest.

`--load-mode ingest-only` (OP-INGEST-LEAN, internal notes boots only what
ingest actually needs: **node-intern + memtable + WAL + flush**. It skips the
resident degmap, keyword index, and block caches. Result: **boot RSS ~15 GiB**
(down from 55.8 GiB) and steady throughput (~790 bonds/s observed). Bonds still
land durably (WAL + flushed SSTs); only the *serving* structures are omitted.

The driver enforces this: `scripts/stack_v2_loader.py` spawns the MCP with
`ingest_only=True`, which appends `--load-mode ingest-only --no-save`. `train.sh`
additionally **verifies** the running child carries `--load-mode ingest-only` and
`--no-save`, and aborts the launch if it does not.

---

## 2. Start / resume / stop / monitor

All commands run on the box, from `/opt/lu/lu/v4`. `train.sh` is
parameterized by `--source` (default `starcoderdata`; `the-stack-v2` also
defined). Add new datasets as profiles in `load_profile()` so every launch stays
version-controlled.

**Start (or resume) a run — one command:**

```bash
scripts/train.sh start --source starcoderdata
# 'resume' is an alias; the driver is checkpointed + dedup-idempotent, so it
# safely continues from /var/lib/lu/<source>_ingest_checkpoint.json
scripts/train.sh resume --source starcoderdata
```

`start`/`resume` run the **hard preflight gate first** and only launch if it
passes. On success they create the transient systemd unit `lu4d-<source>-ingest`
with the fixed caps and lean child, then verify leanness.

**Stop cleanly (driver checkpoints; resume is safe):**

```bash
scripts/train.sh stop --source starcoderdata
```

**Monitor / status:**

```bash
scripts/train.sh status  --source starcoderdata      # unit + memory + checkpoint tail
scripts/train.sh monitor --source starcoderdata      # live journal (Ctrl-C detaches; run keeps going)
# raw equivalents:
systemctl status lu4d-starcoderdata-ingest
journalctl -u lu4d-starcoderdata-ingest -f
systemctl show lu4d-starcoderdata-ingest -p MemoryCurrent -p MemoryPeak
tail -f /var/lib/lu/starcoderdata_ingest_progress.log
```

**Dry-run the gate without launching:**

```bash
scripts/train.sh preflight --source starcoderdata
# exits non-zero while a run is already live — that is the gate working.
```

**Overrides** (any launching subcommand): `--brain --keyfile --checkpoint
--langs --per-lang-limit --limit --corpus --bin --mem-high --mem-max
--load-mode`.

---

## 3. The preflight gate (what it refuses on)

`scripts/train_preflight.sh` is read-only and refuses to let `train.sh` launch if
**any** hard check fails:

1. **Brain path missing** — never create a brain implicitly.
2. **Wrong/old binary** — binary absent, non-executable, or doesn't advertise
   `--load-mode` (i.e. not the lean binary).
3. **Serving/training separation violated** — any other `lu_4_mcp` already bound
   to this brain (a live run, or Lu being served). This is the check that will
   correctly refuse a *second* launch while a run is active.
4. **Target unit already active** — the `lu4d-<source>-ingest` unit is running.
5. **Disk below the floor** — free space `< 25 GiB` (configurable). It also
   **warns** whenever free `< brain size`, because that means neither a second
   local copy, a compaction write-new, nor a local backup can fit.
6. **RAM below the floor** — `MemAvailable < 18 GiB` (configurable), so the lean
   boot won't immediately thrash swap.
7. **WAL too big** — `wal/current.wal > 2 GiB` (configurable) means a heavy,
   RAM-hungry replay at boot. The gate **refuses and prints the human-gated drain
   step** (flush the brain while quiescent, confirm the WAL shrank, re-run). It
   does **not** drain automatically.

Tune via env/flags: `DISK_MIN_FREE_GIB`, `RAM_MIN_AVAIL_GIB`, `WAL_MAX_GIB`.

---

## 4. Post-training handoff to serving

Training omits the serving structures, so after a training run and **before Lu is
served**, rebuild them:

1. Stop training cleanly: `scripts/train.sh stop --source <source>`.
2. Ensure durability: the driver saves at intervals and at end; if you stopped
   mid-batch, note the checkpoint covers re-formable bonds.
3. Rebuild the serving residency (serving boot, NOT ingest-only):
   - **degree-map** (`resident_degmap.bin`) and the **keyword index**
     (`keyword_index.kw`) are rebuilt on a serving/`--load-mode auto` boot.
   - Keyword-index rebuild runs land under `<brain>/.kw_rebuild_runs/`.
4. Only after the rebuild completes is the brain ready to serve recall. Serving
   and training must not overlap on the brain (Golden rule #1).

---

## 5. Major compaction — HUMAN-GATED cadence

Compaction reclaims space and collapses overlapping SSTs, but it is the most
dangerous routine operation and has bitten us (major-compaction collapse +
overlapping-L1 recall bug, internal notes. **It is never automated.** The
canonical cadence is **write-new -> verify -> STOP -> confirm -> reclaim**:

1. **Write-new.** Produce the compacted output as a *new* set; never mutate the
   live SSTs in place. On this box a full write-new needs room the disk does
   not have (free `< brain size`); plan disk/target before starting.
2. **Verify.** Run recall spot-checks against the new output (e.g.
   `examples/recall_spotcheck.rs`). Confirm recall quality and that overlapping-L1
   ranges resolve correctly — this is exactly the class of bug #1296 exposed.
3. **STOP.** Halt and surface the verification result. **Do not proceed to
   reclaim automatically.**
4. **Confirm.** A human (John) explicitly approves the reclaim.
5. **Reclaim.** Only then remove the superseded inputs.

**Never auto-reclaim.** And per the backup rule below, do not run any reclaim at
all until a verified brain backup exists.

---

## 6. Backups — the top open gap

**The canonical brain has NO backup.** RAID1 (`/dev/md2`) is redundancy, not
backup: corruption, a bad compaction, or an accidental delete replicates to both
legs. The brain (~198 GiB) is larger than local free space (~188 GiB), so a second
local copy will not fit, and the mirror cannot be broken safely.

**Plan:** periodic verified snapshots of `/var/lib/lu/brain-v5-20260615-sst` to a
**Hetzner Storage Box** (>= 1 TB), over rsync/SSH. `scripts/backup_brain.sh`
implements this: quiescence check -> `rsync -a` -> checksum re-scan verification ->
manifest -> retention. It refuses to run while `REMOTE_TARGET` is the placeholder
and refuses to back up a brain that is actively being written (unless
`--allow-live`). It does **not** mutate the brain and does **not** order storage.

> **ACTION REQUIRED — JOHN:** provision a Hetzner Storage Box, add an SSH key for
> passwordless rsync, and set `REMOTE_TARGET`/`REMOTE_SSH_PORT` at the top of
> `scripts/backup_brain.sh`. Recommended first run: `scripts/backup_brain.sh
> --dry-run` (after stopping training), then a real run.
>
> **Until a verified backup exists, do NOT run any destructive brain automation**
> (compaction reclaim, deletes).

---

## 7. Troubleshooting — failure modes we actually hit

**Boot RAM ~55.8 GiB / stalls / swap thrash.** The run booted the full serving
residency instead of lean. Confirm the child cmdline shows `--load-mode
ingest-only` (`scripts/train.sh status`, or `pgrep -af lu_4_mcp`). If not, stop
and relaunch via `train.sh` (it verifies leanness and aborts otherwise).

**Ephemeral `systemd-run` unit lost on reboot / not reproducible.** The old
transient unit existed only in `/run/systemd/transient/` and vanished on stop —
resumes lost the intended caps. Fixed: `scripts/train.sh` is the persistent,
version-controlled launcher; never launch by hand-typed `systemd-run` again.

**Preflight refuses: "another lu_4_mcp is already bound to this brain."** Working
as intended — a run is live or Lu is being served. Do not start a second writer.
`scripts/train.sh stop --source <source>` (or stop serving) first.

**Long boot with a huge WAL ("WAL replay boot tax").** `wal/current.wal` grew
large and replays into RAM at boot. With the brain quiescent, flush it
(`lu4d_flush`), confirm `current.wal` shrank, then relaunch. The preflight refuses
launches while the WAL exceeds `WAL_MAX_GIB` and prints this step. (Note: a flush
writes an SST but may not immediately truncate the WAL — confirm the size actually
dropped before relaunching.)

**Disk/RAM preflight failure.** Free space `< 25 GiB` or `MemAvailable < 18 GiB`.
Free resources before launching; do not raise caps past `56G` on a 62 GiB box.

**Major-compaction collapse / overlapping-L1 recall regression.** Seen in
#1296/#1297. Never mutate SSTs in place; always write-new -> verify recall ->
human-confirm -> reclaim (section 5). Keep a backup before any reclaim (section 6).

---

## 8. Quick reference

```bash
# start / resume (one command; preflight-gated, lean-verified)
scripts/train.sh start  --source starcoderdata
scripts/train.sh resume --source starcoderdata

# stop cleanly (checkpointed; resume-safe)
scripts/train.sh stop   --source starcoderdata

# observe
scripts/train.sh status  --source starcoderdata
scripts/train.sh monitor --source starcoderdata

# gate only (no launch)
scripts/train.sh preflight --source starcoderdata

# backup (after Storage Box is provisioned + REMOTE_TARGET set)
scripts/backup_brain.sh --dry-run
scripts/backup_brain.sh
```
