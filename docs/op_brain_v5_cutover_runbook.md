# OP-BRAIN-V5 Loader-Cutover Runbook (capability + procedure)

**Status:** Loader-cutover code (sub-phases 6.1-6.7) is shipped on `main`.
This runbook is the *documented, executable procedure* for the eventual
production cutover. **It has NOT been executed. Production remains on v3
(`/var/lib/lu/brain.lbr`).** The actual cutover is a separate, later step
(Phase 8), gated on the real 14 GB migration run + offline verify + a soak
window. Nothing here modifies the live `lu.service` or the live
`/var/lib/lu` symlink.

This milestone (6.6/6.7) delivers the *capability*:
- the production binary can boot a v5 brain dir under `--brain-format v5`
  (or manifest auto-detect) + `--require-encrypted --keyfile`, and the
  startup banner reports format + encryption at a glance;
- a stable `brain-current` symlink scheme so cutover/rollback is a
  one-line swap (`scripts/cutover/lu_v5_brain_current.sh`);
- an offline verifier (`lu_brain_verify_v3_v5`) that gates the cutover on
  recall parity + encryption-at-rest + count parity.

---

## Locked decision #6 -- versioned dirs + symlink repoint

The service's `--brain` argument points at a **stable symlink**,
`/var/lib/lu/brain-current`, NOT at a brain dir directly. The symlink
resolves to a **versioned** brain:

```
/var/lib/lu/brain-current  ->  brain.lbr              (v3, TODAY)
/var/lib/lu/brain-current  ->  brain-v5-YYYYMMDD/     (v5, AFTER cutover)
```

Cutover = repoint the symlink. Rollback = repoint it back. The v3
`brain.lbr` is **never mutated** (migration reads a copy mmap-RO), so
rollback is instant and lossless.

Helper (capability only -- does not touch `/var/lib/lu` unless you pass
`--base /var/lib/lu`):

```
scripts/cutover/lu_v5_brain_current.sh status        --base /var/lib/lu
scripts/cutover/lu_v5_brain_current.sh version-name  brain-v5      # -> brain-v5-YYYYMMDD
scripts/cutover/lu_v5_brain_current.sh init  brain.lbr        --base /var/lib/lu   # one-time: adopt today's v3
scripts/cutover/lu_v5_brain_current.sh point brain-v5-YYYYMMDD --base /var/lib/lu   # cutover
scripts/cutover/lu_v5_brain_current.sh rollback              --base /var/lib/lu   # revert to recorded prev
```

A proposed unit template lives at `scripts/cutover/lu4d-v5.service.template`
(**template, not installed**). It differs from the live v3 unit only by the
`--brain /var/lib/lu/brain-current` target and the added `--brain-format v5`.

---

## Pre-cutover checklist

- [ ] Phase 8 approved; soak plan agreed.
- [ ] Disk: room for the v5 dir (plan >= the 14 GB v3) **plus** the
      retained v3 snapshot. `/var/lib/lu` currently also holds
      `brain.lbr.kw` (~4.5 GB) + stale `brain.lbr.tmp` -- clear cruft first.
- [ ] Master key present: `ls -la /var/lib/lu/master.key`.
- [ ] Release binary built on the box: `cargo build --release`.

## Cutover sequence (mirrors design doc §7 step-5; no destructive step until verified)

1. **Snapshot** (v3 keeps serving):
   ```
   sudo systemctl stop lu4d
   sudo cp -a /var/lib/lu/brain.lbr /var/lib/lu/brain-v5-src.lbr
   sudo systemctl start lu4d
   ```

2. **Migrate** the snapshot (node tool then bond tool), encrypting output:
   ```
   V5=/var/lib/lu/$(scripts/cutover/lu_v5_brain_current.sh version-name brain-v5)
   target/release/lu_brain_migrate_v3_v5_node --v3 /var/lib/lu/brain-v5-src.lbr --v5 "$V5" --keyfile /var/lib/lu/master.key
   target/release/lu_brain_migrate_v3_v5_bond --v3 /var/lib/lu/brain-v5-src.lbr --v5 "$V5" --keyfile /var/lib/lu/master.key
   ```
   (Confirm the migration CLIs' own flag names with `--help`; the
   programmatic configs are `migration_v3_v5_{node,bond}::MigrationConfig`.)

3. **Post-migrate compaction** to emit `.bsst.reverse` (forward-only
   migration leaves the reverse projection unbuilt).

4. **Verify offline -- the GO/NO-GO gate (sub-phase 6.7):**
   ```
   target/release/lu_brain_verify_v3_v5 \
     --v3 /var/lib/lu/brain-v5-src.lbr \
     --v5 "$V5" \
     --keyfile /var/lib/lu/master.key \
     --sample 50000
   ```
   Exit 0 = GO. **DO NOT PROCEED on NO-GO.** The report breaks down
   recall parity (match rate + diverging subjects), encryption-at-rest
   (every Node/Bond SST must carry `FLAG_ENCRYPTED`; zero plaintext
   SSTs; MANIFEST `master_key_id` set), and node/bond count parity.
   Optionally also boot a NON-production instance against `"$V5"`
   (`lu_4_mcp --brain "$V5" --brain-format v5 --require-encrypted
   --keyfile ...`) and run `eval_100_strict.py` against the v3 baseline.

5. **Swap** (one line, after GO):
   ```
   sudo systemctl stop lu4d
   # first cutover only: adopt today's v3 under the symlink if not already
   sudo scripts/cutover/lu_v5_brain_current.sh init  brain.lbr            --base /var/lib/lu   # idempotent / skip if present
   sudo scripts/cutover/lu_v5_brain_current.sh point "$(basename "$V5")"  --base /var/lib/lu
   # install the v5 unit (from the template) so --brain = brain-current + --brain-format v5
   sudo systemctl daemon-reload
   sudo systemctl start lu4d
   sudo journalctl -u lu4d -f   # confirm the 6.6 banner: format=v5 | encryption=encrypted-at-rest
   ```

6. **Resume ingest** against the v5 dir (the flush + compaction workers
   persist new bonds/nodes into the v5 stores; the WAL gives crash safety).

7. **Rollback** (any gate fails, or post-swap regression):
   ```
   sudo systemctl stop lu4d
   sudo scripts/cutover/lu_v5_brain_current.sh rollback --base /var/lib/lu   # brain-current -> brain.lbr
   # revert the unit to the v3 ExecStart (or it already points at brain-current -> brain.lbr)
   sudo systemctl daemon-reload && sudo systemctl start lu4d
   ```
   The v3 `brain.lbr` was never mutated -> instant, lossless rollback.

## Decommission (ONLY after a clean v5 soak -- gates Phase 3.9 retire-v4)

- [ ] Remove `brain-v5-src.lbr` snapshot.
- [ ] Only after the soak window: Phase 3.9 deletes the v4 read/write
      path (`sst.rs` / `sst_writer.rs` / `compaction.rs`), per locked
      decision #4 (soak-first).
