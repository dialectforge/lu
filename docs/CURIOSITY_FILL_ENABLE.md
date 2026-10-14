# Curiosity Fill Loop — Enable Steps (Faculty 7)

**Status:** producer (guest side) already wired; **fill worker deferred** until the
retrain completes and serving is brought up. **Do NOT start the worker now** —
the live `lu4d-cat1-ingest` training holds the canonical brain and the v5 brain
is single-tenant (load peaks ~54 GiB on a 62 GiB box). Ref the project log
#1352 Faculty 7 / #1353.

## The two halves

Curiosity has a **producer** (detects gaps while answering, appends them to a
durable wonder journal) and a **fill worker** (a separate WRITER process that
drains the journal and fills the gaps via `idle_tick`s). They are deliberately
two processes: the public guest stays read-only; only the worker writes.

### 1. Producer — ALREADY WIRED (verify only)

The guest binary accepts `--wonder-queue <dir>` (`crates/lu_guest_http/src/main.rs:84`)
and attaches a `WonderSidecar` that appends detected gaps to
`<dir>/wonder_queue.jsonl`. The serving unit already passes it via a drop-in:

```
/etc/systemd/system/lu-guest-http.service.d/10-wonder-queue.conf
  ExecStart= ... --wonder-queue /var/lib/lu-wonder ...
  ReadWritePaths=/var/lib/lu-wonder
```

Verify after the serving binary is deployed:

```
systemctl cat lu-guest-http.service | grep -- --wonder-queue    # expect the flag
```

Without the flag the guest logs: `no --wonder-queue: gaps are detected but NOT
persisted` (main.rs:138) — i.e. the producer degrades safely to a no-op.

### 2. Fill worker — DEFERRED. Enable AFTER retrain + serving up

Unit: `/etc/systemd/system/lu-curiosity-worker.service` (`Type=oneshot`, no
listener, WRITER). Drains `/var/lib/lu-wonder` and runs 10 `idle_tick`s. It is
`static` (no `[Install]`) — run on demand, never auto-started at boot (a
boot-start would race serving for the brain).

Enable step (only when serving is up and NO training/other tenant holds the brain):

```
systemctl start lu-curiosity-worker.service        # oneshot; runs one fill pass
journalctl -u lu-curiosity-worker.service -n 50    # inspect the pass
```

Fills land as **Pending** changes awaiting `lu4d_approve_changes` — `--auto-apply`
is deliberately absent, so Lu never writes its own long-term memory unattended
without John's approval.

## BEFORE ENABLING — two stale-config safety fixes (found 2026-07-24)

1. **`Conflicts=` is stale vs the current training unit.** The worker unit
   declares `Conflicts=lu-guest-http.service lu4d-starcoderdata-ingest.service
   lu4d-curriculum-ingest.service` but the **active training unit is
   `lu4d-cat1-ingest.service`, which is NOT in that list.** systemd would
   therefore NOT block the worker from co-loading the 233 GiB brain against a
   running `lu4d-cat1-ingest` → OOM risk. Add `lu4d-cat1-ingest.service` to the
   `Conflicts=` line before ever starting the worker.

2. **`--brain` points at an old snapshot.** The unit's ExecStart uses
   `--brain /var/lib/lu/brain-v5-20260615-sst`, but the current retrain writes
   `brain-v5-20260724-sst`. Update `--brain` (and `--wonder-queue`) to match the
   brain that serving actually loads once the retrain completes, or the worker
   fills a stale brain.

## Ready-for-live-test checklist item

- [ ] serving deployed with `--wonder-queue /var/lib/lu-wonder` (drop-in present)
- [ ] worker `Conflicts=` includes `lu4d-cat1-ingest.service`
- [ ] worker `--brain` = the served post-retrain brain dir
- [ ] training + any other brain tenant STOPPED before `systemctl start lu-curiosity-worker`
