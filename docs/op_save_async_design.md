# OP-SAVE-ASYNC Design - Non-blocking background saves

**Status:** Phase A design. HALT for John review before implementation.
**Branch (planned):** none yet. Off `main` HEAD `7705e5f` (post-SAVE_INTERVAL tuning).
**Scope:** lift the ~60-70s blocking `lu4d_save` off the curriculum-ingest hot path so wall-clock save tax drops from ~20% toward ~0%. Investigation + design only; no code, no commits, no deploys.
**Anti-LLM positioning:** unchanged. This is a storage-path concurrency port, not a model surface.
**LOC budget:** depends on option (see Section 12). Primary recommendation lands ~700-900 net.

---

## 0. Today vs after-SAVE-ASYNC

**Today:** `lu4d_save` (`src/mcp.rs:2760`) calls `Brain::save` (`src/brain.rs:2607`) -> `save_to_format` (`:2803`) -> `atomic_write_with` (`:1143`) -> `write_v3_compressed_encrypted_to_borrowed` (`src/brain_io_v3.rs:2939`). Every byte streamed from the brain holds the MCP loop hostage on the main thread; ingest cannot teach during the save. v5's BufWriter spans phases (1)-(7) of the writer; the CRC pass (8) and trailer (9) run unbuffered against the tmp file. SAVE_INTERVAL_BONDS=50000 means ~3000 articles between saves, and a 60-70s save against 8.3 art/s sustained throughput is ~500-580 articles of stall per save - ~20% of wall-clock at the current cadence.

**After SAVE-ASYNC:** `lu4d_save` hands the save off to a background path (fork child, snapshot thread, or chunked yield - depends on chosen option), the MCP loop returns to receiving teach calls within milliseconds, and the on-disk brain.lbr is updated atomically at the end. The harness's existing `ok=True iff committed` semantic is preserved (Section 7).

Non-save calls: zero behavior change. Parity invariant load-bearing.

---

## 1. Phase A - the save path as it stands today

### 1.1 Entry chain

- `tool_save` (`src/mcp.rs:2760`): unconditionally calls `brain.save()`, returns `{saved: true, path}` only after success. No background hook today.
- `Brain::save` (`src/brain.rs:2607`): re-dispatches to `save_to_format` with the brain's configured format (production = `V3BinaryEncrypted`).
- `Brain::save_to_format` (`src/brain.rs:2803`): runs encryption guard (`require_encrypted` + downgrade-refusal), creates parent dir, then for the V3BinaryEncrypted branch (`:2867-2939`) extracts/snapshots overlay-passthrough sources, calls `atomic_write_with` on the configured path. Post-write it runs sidecar / WAL truncation steps (Section 1.4).
- `atomic_write_with` (`src/brain.rs:1143`): opens `<path>.tmp` (RW, truncate, create), invokes the closure, calls `f.sync_all()`, renames `.tmp` -> path, then `fsync`s the parent directory. The atomic-rename semantic is what the harness's `ok=True iff committed` contract is built on.
- `write_v3_compressed_encrypted_to_borrowed` (`src/brain_io_v3.rs:2939`): the OP-022.s.v5 writer. Wraps the tmp file in a 1 MiB BufWriter for phases (1)-(7), flushes via `into_inner` before phase (8), runs CRC and trailer unbuffered against the underlying RW handle.

### 1.2 The eight phases (as labelled in `brain_io_v3.rs:2955-3175`)

| # | Phase | Reads/Writes | Notes |
|---|---|---|---|
| 1 | Header + offset-table placeholder | write only | 52 bytes via BufWriter, no graph access |
| 2 | Compressed-encrypted network section | reads `&self.network`, writes ciphertext | dominant phase. Walks every node + every bond; for prod brain this is ~17.2M bonds + 8.5M nodes streamed through zstd+ChaCha20-Poly1305. Three sub-paths: full-passthrough (mmap blocks copied verbatim), partial-passthrough (mmap base + overlay/tombstone overlay), slow re-encode. Currently in mid-training the partial-passthrough path is the steady state. |
| 3 | RKYV_ALIGN pad | write only | <16 bytes |
| 4 | scar_log | reads scar_log, rkyv -> bytes -> write | small (~MB at most) |
| 5 | Side-table sections x14 (interest_profile, meta_reflection, research_proposals, research_misc, gap_seed_frequency, pending_log, rejection_patterns, lb_session, cs_style, pack_ledger, runtime_config v14, runtime_config v15, source_overlay, cap_pack_ledger) | reads side tables, bincode/rkyv -> write | each method on `SideTableSource for Brain` (`src/brain.rs:8560`). Most are `bincode::serialize_into(&mut w, &self.field)` - direct borrow from the brain. trust_table + hub_degree are cloned into a RuntimeConfigSnapshot once (small, <1 MiB). |
| 6 | Keyframe count tally | reads `&self.network` (`for_each_bond`) | walks every bond a SECOND time just to tally keyframes for the header. ~17.2M iter. |
| 7 | Header rewrite + offset table | write only | seek-to-0 forces BufWriter flush; 52 + 17 * 16 bytes |
| 8 | CRC pass (read whole written file back) | reads tmp file from disk | walks the entire 6.2 GB on-disk payload through crc32fast at 256 KiB chunks. Runs OUTSIDE the BufWriter (raw `rw`) |
| 9 | Trailer (16 bytes: magic + crc + reserved) | write only | raw `rw`, no buffering needed |

**Reads-from-brain-memory phases:** (2), (4), (5), (6). All of these need a consistent view of brain state - any mutation between phase (2) and phase (6) breaks the header's bond/keyframe count vs the section bytes already written.

**Writes-to-file phases:** (1), (2), (3), (4), (5), (7), (8), (9). Pure file IO: (1), (3), (7), (9). Mixed: (2), (4), (5).

### 1.3 Mutation sites

From `grep -n "pub fn teach\|push_pending_" src/brain.rs`:

- `Brain::teach_with_kind` (`src/brain.rs:3474`): primary write path. Creates nodes via `network.add_node` and bonds via `network.add_bond`/`reinforce`, optionally writes negation overlay + source overlay.
- Wrappers: `teach` (`:3165`), `teach_with_negation` (`:3047`), `teach_from_text` (`:3075`), `teach_corpus` (`:3241`), `teach_corpus_with_kind` (`:3267`), `teach_with_source` (`:3304`), `teach_with_source_kind` (`:3323`). All funnel through `teach_with_kind`.
- `Brain::push_pending_with_filter` (`:1596`): writes to the pending log (in-memory + WAL via `self.pending`).
- Network-level mutations: `AssociativeNetwork::add_node`, `add_bond`, `reinforce`, `delete_node`, `delete_bond`, `tombstone_bond`. All take `&mut self`.
- Sidecar writes: WAL appends on `pending`, `lb_session`, `cs_style_store`; metric writes on `performance_monitor`; idle / curiosity / wonder state updates.

**Concurrency model today:** single-threaded mutation. MCP loop owns `&mut Brain` on the main thread (`run_mcp_server_opts` at `src/mcp.rs:27`). Stdin reader thread and auto-tick thread send messages via mpsc; main thread dispatches. Brain never crosses thread boundaries.

### 1.4 Post-save sidecar coordination

`save_to_format` runs (after the atomic rename succeeds):

- `network.compact_keyword_sidecar(new_crc)` - rewrites the .kw file to match the new brain crc.
- `pending.truncate_wal()` - clears `<brain>.pending.wal`.
- `lb_session.truncate_wal()` - clears `<brain>.lb_session.wal`.
- `cs_style_store.truncate_wal()` - clears `<brain>.style.wal`.
- Cat.3c legacy sidecar retirement (rename to `.migrated`).

These ALL run on the same thread, AFTER the brain file rename. They mutate brain state (`self.runtime_config_pending_migration = false`).

### 1.5 Existing snapshot/clone abstractions

- `NetworkSource` trait (`src/network.rs:452`): defines `node_count`, `bond_count`, `for_each_node`, `for_each_bond`. Both `NetworkSnapshot` (an owned `Vec` form) and `AssociativeNetwork` (live, archive + overlay merged) implement it. The streaming writer takes `N: NetworkSource + ?Sized` - any future snapshot abstraction can plug in here.
- `SideTableSource` trait (`src/brain_io_v3.rs:2898`): one method per side-table section. Brain impls it directly (`src/brain.rs:8560`) - methods just bincode-serialize their respective `&self.field` references.
- `compressed_archive_for_passthrough` / `compressed_archive_for_partial_passthrough` (`src/network.rs:787, 808`): return `&CompressedNetwork` from the loaded mmap archive when overlay is empty / acceptable. The passthrough WRITER doesn't decompress or re-encrypt; it copies compressed block frames verbatim.
- `network.to_snapshot` (`src/network.rs:4442`): builds a fully-materialised `NetworkSnapshot` (owned `Vec<KnowledgeNode>` + `Vec<TemporalBond>`). OP-017 deleted the save-path use; today only used by tests + the migration tool. RAM cost is ~7 GiB on prod brain.
- `to_brain_file_no_network` (`src/brain.rs:2733`): builds a `BrainFile` with an empty `NetworkSnapshot` placeholder + cloned side tables. Cloning is shallow-ish: `interest_profile.clone()`, `meta_reflection.clone()`, etc. Small.

**Key finding:** there is no existing "read-only view of brain at point in time" abstraction below the trait level. `for_each_bond` walks live state; an in-flight mutation would be observed by phase (6) but not phase (2) (or vice versa), corrupting the header tally. Today's serial save dodges this because nothing else can mutate while it runs.

---

## 2. Option survey

### Option 1: fork-based snapshot (Redis BGSAVE pattern)

**How:** `lu4d_save` calls `nix::unistd::fork()`. Parent records the child pid, returns control to the MCP loop. Child runs the existing `write_v3_compressed_encrypted_to_borrowed` against a tmp file, calls `_exit()` on success. Parent reaps the child via `waitpid(WNOHANG)` on each subsequent `lu4d_save` (or on a timer), and renames `.tmp` -> brain.lbr only after the child exited cleanly.

**Why it works:** Linux fork() + COW. The child sees a frozen copy-on-write snapshot of the parent's address space at fork time. Memory cost = pages dirtied by the parent during the save window, NOT the full RSS. On a 23 GiB brain with ~3000 articles of ingest during a 60s save window, typical bond-add page touches are on the order of low GiB.

**Code surface affected:** `src/mcp.rs:2760` (save handler split into start + status). New `src/save_async.rs` module with fork wrapper, child pid tracking, reaping, tmp file rename coordination. Small touch to `save_to_format` to factor out the rename step (so the child writes the tmp file but the parent does the rename after reap).

**Memory cost:** parent stays at baseline. Child sees baseline + COW dirty pages. Cgroup MemoryMax counts BOTH processes - fork doubles the accounting headroom requirement nominally, but COW means actual usage stays near baseline plus dirty-during-save.

**Correctness risk:**
- **Allocator state at fork.** Rust's default global allocator (jemalloc or system malloc) can have internal mutexes held by other threads at fork time. Stdin reader thread and auto-tick thread are running. If they hold a malloc lock when we fork, the child inherits a locked mutex with no thread to unlock it -> deadlock the moment the child does any allocation. **Mitigation:** pause both background threads before fork (send a Pause message, wait for ack), fork, resume. Or `prctl(PR_SET_PDEATHSIG)` + fork without pausing and accept the small deadlock-on-alloc probability (NOT acceptable - production).
- **File descriptor inheritance.** The child inherits the MCP stdio fds, the brain.lbr mmap fd, the .kw sidecar mmap fd, the .pending.wal / .lb_session.wal / .style.wal fds. All of these need to be marked `O_CLOEXEC` or explicitly closed in the child. Stdin is the dangerous one - if both parent and child read from it, JSON-RPC messages get split between processes.
- **mmap of brain.lbr.** The brain.lbr mmap that backs `compressed_archive_for_passthrough` is shared between parent and child. If the parent's save logic ever wrote to this region (it doesn't - the archive is read-only), the COW would split. As-is, safe.
- **Sync between processes.** Side-table state in the child is the COW snapshot at fork time; that's exactly what we want. WAL truncate calls (post-rename) MUST run in the parent against post-save state - the child's view is now stale.

**LOC estimate:** 200-400 LOC for the fork + pid tracking + signal handling, plus 50-100 LOC for thread-pause coordination, plus 50-100 LOC for the lu4d_save semantic change (Section 7). ~350-600 LOC total.

**Related approaches:** Redis BGSAVE. Battle-tested over 15+ years at massive deployment scale. PostgreSQL background writer uses a different model (separate process spawned at startup, communicating via shared memory), not fork-snapshot.

### Option 2: generational counters

**How:** add an atomic `current_generation: AtomicU32` on `Brain`. Every mutation site (`teach_with_kind`, `push_pending_*`, all the overlay paths) bumps the generation on the touched node/bond. `Brain::save` captures generation `G` at start; the streaming writer skips any bond/node with generation > G. Mutations continue during save against the live structures.

**Why it works:** the writer sees a logically-frozen view defined by `generation <= G`. Newer writes are invisible to this save and get picked up on the next one.

**Code surface affected:** `KnowledgeNode` + `TemporalBond` gain a generation field (or repurpose an existing one). Every `add_node`/`add_bond`/`reinforce`/overlay mutation bumps the field. `for_each_node`/`for_each_bond` callsites that participate in save take a `max_generation` arg and filter. SideTableSource methods need parallel generational filtering on their respective sub-structures (pending_log, lb_session, cs_style, etc.).

**Memory cost:** +4 bytes per node + 4 bytes per bond. On prod brain: 4 * (8.5M + 17.2M) = ~103 MiB additional persistent overhead. If repurposing an existing u32 field (e.g. unused padding), zero additional memory.

**Correctness risk:**
- Every mutation site must be audited and updated. Miss one and that path's writes are silently included in saves whose generation should have excluded them, but the keyframe-count tally in phase (6) may not match the section bytes written in phase (2) -> CRC fails -> save fails. CI catchable but tedious.
- The 14 side tables EACH need a parallel generation model. pending_log is small; lb_session and cs_style each have their own internal data; runtime_config is config + must not skew at save. Each one is a sub-port.
- Concurrency: still need a lock or atomic snapshot of `current_generation` at save start. Mutations between read of G and "save starts walking" must come up as generation > G. Doable with a single `fetch_add` at save start: `let g = current_generation.fetch_add(1, Ordering::SeqCst);` - everything after gets generation g+1, save filters to <= g.
- **Save still runs on the main thread by default.** Unless we also move the save off-thread (which then requires `Arc<Brain>` or similar), generational counters alone don't make saves non-blocking - they make a future off-thread save *safe*. So Option 2 is half a solution.

**LOC estimate:** generation field + bump-on-mutate audit = 200-400 LOC. Side-table generational filtering = 300-500 LOC. Off-thread save runner (Arc<Brain> or RwLock<Brain> across the boundary, plus the lu4d_save semantic change) = 100-200 LOC. Total ~600-1100 LOC. Heavy audit cost.

**Related approaches:** classic database MVCC. PostgreSQL, SQLite, InnoDB - all use generation/transaction-id watermarks for snapshot isolation. Pattern is sound; the cost is upfront audit work.

### Option 3: just-in-time chunked save

**How:** the writer runs to completion, but yields control to the MCP loop every ~100-500ms. Between yields the brain accepts and processes one or more teach calls (which mutate the live structure that the writer is in the middle of walking). The writer reasons about consistency at chunk boundaries.

**Why it doesn't actually work for our shape:** the v3 writer's phases (2)+(6) walk the network twice and have to agree. If teach happens between (2) and (6), the header tally diverges from the section bytes. No chunk-boundary discipline fixes that without snapshotting (which is what Options 1 and 2 are). Sub-chunking phase (2) itself runs into a worse version of the same problem: a teach during phase (2) can change which mmap blocks the passthrough should copy vs which overlay entries need encoding, mid-stream.

**Reformulation:** chunked-yield only works if we ALSO snapshot the mutation set so the writer's view is consistent across yields. That's just Option 2 with yielding bolted on, at the cost of a much harder lock-discipline audit (the writer holds reads against live structures across yield boundaries).

**Code surface affected:** ~every loop in the writer needs yield hooks; mutation sites need to coordinate with the in-flight writer; new RwLock or fine-grained lock per data structure (node table, bond table, each side table).

**Memory cost:** none additional, but contention cost on mutation throughput is unbounded.

**Correctness risk:** high. Hard to audit, hard to test, error mode is silent CRC corruption.

**LOC estimate:** 400-700 LOC for yield hooks + lock plumbing, plus the Option-2-style snapshot machinery underneath if we want it to actually work. 700-1200 LOC total once it's correct.

**Related approaches:** Cassandra compaction throttles via chunked yields; LevelDB / RocksDB throttle compactions but don't yield mid-flush. The closest analog is RocksDB's rate-limited writes, which is yield without correctness guarantees - rocks accepts the staleness because compaction can be re-run. Our save can't be "re-run" - the brain file has to be coherent.

**Verdict:** rejected. Has all the costs of Option 2 plus worse correctness ergonomics. Not pursuing.

### Option 4: differential save (only write deltas)

**How:** track touched-since-last-save state via dirty-bits or a separate delta log. Save writes only the delta to a new file (e.g. `brain.lbr.delta.N`); load applies deltas on top of base. Periodically compact base + deltas -> new base.

**Why it would help:** delta files are small. Per-save work is O(deltas), not O(brain). 50K bonds per save * (avg keyframe + node payload) is maybe 5-30 MiB; <1s of write work.

**Why it doesn't fit:**
- The v3 format is monolithic (header + sections + trailer + CRC over the whole thing). Delta save changes the on-disk format substantively.
- Load complexity goes up significantly. Multiple files to merge, CRC strategy needs rethinking, mmap of the archive is now an mmap of base-plus-deltas, partial passthrough invariants break.
- Not actually a "background save" - it's just a much cheaper blocking save. We could make the per-50K-bond cost <1s, but every Nth save still has to compact to a new base, and that compact IS the full save. So we get cheap saves most of the time and an occasional 60-70s stall on compact. (Could combine with Option 1 to background the compact, but now we're writing two solutions.)
- Combines awkwardly with v3enc structure - delta encryption nonces need careful management for replay protection.

**LOC estimate:** 800-1500 LOC for the delta format + load path + compact policy + nonce management. Plus harness/loader changes.

**Related approaches:** RocksDB WAL + memtable -> SST flush is a delta-ish model. SQLite WAL-mode. Both are large engineering efforts that took multiple releases to stabilise. Out of scope for the immediate ingest-throughput goal.

**Verdict:** not pursuing for OP-SAVE-ASYNC. Could be a follow-on after the immediate save-tax problem is solved.

### Option 5 (added by investigation): generation-stamped snapshot thread

**How:** at `lu4d_save` time, the MCP main thread does the small-cost setup work synchronously - capture an atomic generation, snapshot the small side tables (clone-cost low), snapshot the network archive Arc - then spawn a `std::thread` that owns these snapshots + an `Arc` of the mmap-backed CompressedNetwork. The thread runs the existing writer machinery against the snapshots, with the writer's `NetworkSource` impl filtered by `max_generation`. Main thread returns to the MCP loop. Done callback on the thread updates a `last_save_status` field; next `lu4d_save` blocks until any prior in-flight save completes, then begins a new one (or returns the prior status if asked via a separate status call).

**Why this might be the cleanest:** we already have most of the abstractions. `NetworkSource` is a trait; we can define a `SnapshotSource` that wraps `(Arc<CompressedNetwork>, OverlaySnapshot, max_generation)` and impls `NetworkSource`. Side tables are clone-cheap (bincode forms are bytes; the largest clone is `interest_profile` which is bounded by a config cap, a few MiB at most; trust_table is a few hundred KiB). The streaming writer takes `&N: NetworkSource` already - drop a snapshot in and it works.

**Why it's NOT just Option 2 again:**
- No need to bump generation on every mutation site. The mutation log is the OVERLAY - new nodes and new bonds land in `network.nodes` and `network.bonds` (heap overlay), not the archive. We snapshot those by clone at save start; subsequent mutations land in the live overlay but not in the snapshot. The archive (mmap) is already immutable.
- Pending/lb_session/cs_style WALs are append-only logs with internal seq numbers. We snapshot by reading the current high-water-mark at save start. Mutations after the mark go to the live WAL; the snapshot writes everything <= mark.
- For the keyframe-count tally (phase 6), we walk the SAME `NetworkSource` snapshot as phase (2). Internally consistent.

**Code surface affected:** new `OverlaySnapshot` struct that captures the heap-overlay node+bond hashmaps at save-start time via clone. `SnapshotSource` struct that impls `NetworkSource` against `(Arc<CompressedNetwork>, OverlaySnapshot)`. New `save_async.rs` thread-spawn + status tracking. lu4d_save semantic change to return when committed (block on join) OR start-then-poll (Section 7).

**Memory cost:** overlay clone at save start - sized by overlay (mutations since last archive load). For a freshly-loaded brain this is small. For a brain mid-training between two compactions, the overlay can be ~1-2 GiB. Side-table clones: small. Archive: zero (Arc'd, shared mmap). Save runs against the snapshot; ingest writes to the live overlay. Peak RSS = baseline + overlay-clone + side-table-clones + writer-buffer = baseline + ~1-2 GiB + ~tens of MiB. Better than fork's full-COW worst case in pathological scenarios; comparable in steady state.

**Correctness risk:**
- No `unsafe`, no fork ergonomics, no PID juggling. Pure Rust threading.
- Snapshot capture must be atomic w.r.t. the brain lock - just acquire-clone-release on the main thread before spawning the worker.
- The atomic-rename at the end still happens (writer writes to tmp; on thread-finish, parent thread renames). Crash mid-write leaves a .tmp that the next save sweeps (existing `atomic_write_with` already does `let _ = fs::remove_file(&tmp_path);` at start).
- WAL truncation: ONLY runs after the rename succeeds AND on the main thread (so it can mutate `&mut self.pending` etc.). Worker thread signals completion; main thread does the truncate.

**LOC estimate:** 200-300 LOC for OverlaySnapshot + SnapshotSource impl. 100-150 LOC for save-async thread runner + completion signaling. 50-100 LOC for lu4d_save semantic change. 100-200 LOC for tests + harness sim. ~450-750 LOC.

**Related approaches:** the pattern is "capture a consistent snapshot then write it on a worker thread" - InnoDB's mtr (mini-transaction) buffer copy at log flush time is similar. SQLite's `sqlite3_backup` API in incremental mode is similar shape. The pattern is well-understood.

---

## 3. Recommendation

**Primary: Option 5 (generation-stamped snapshot thread).** It reuses our existing `NetworkSource` trait, doesn't require a fork toolkit (no `nix::fork`, no allocator-state audit, no fd inheritance audit, no cgroup memory doubling concerns), keeps the save runnable while preserving the atomic-rename contract, and lands inside the LOC envelope. The overlay-clone is bounded by mutations-since-archive-load, which on the current brain is the right cost shape (small immediately after archive load, growing toward ~1-2 GiB across many training batches).

**Fallback: Option 1 (fork-based snapshot).** If Option 5's overlay-clone memory cost turns out to be unacceptable on a fully-mutated brain (overlay = full graph at the limit), we fall back to fork+COW which only pays for pages dirtied during the save window. Higher engineering cost (allocator audit, fd close-on-exec audit, signal handling) but lower memory cost in the pathological case.

---

## 4. Recommended design (Option 5)

### 4.1 Entry point and handoff

`tool_save` (`src/mcp.rs:2760`) splits into two paths depending on whether a save is already in flight:

```rust
fn tool_save(brain: &mut Brain) -> Result<Value, (i64, String)> {
    // If a prior save is still running, join it (block). The harness
    // calls lu4d_save serially per checkpoint interval, so contention
    // is rare; if a save IS still running when the next save is
    // requested it means the prior save is taking longer than the
    // ingest interval, and the harness should naturally back-pressure.
    brain.save_runner.join_if_active().map_err(...)?;
    // Spawn the new save in the background. The brain.save_runner
    // captures a consistent snapshot inline on this thread, then
    // spawns a worker that runs the writer machinery against the
    // snapshot. Returns immediately after the snapshot is captured.
    let handle = brain.start_async_save()?;
    // For wire-compat with the harness's ok=True iff committed
    // contract, we BLOCK here on the join. (Section 7 covers the
    // alternative.)
    handle.join().map_err(...)?;
    // Post-save WAL truncates + sidecar compact run on this thread
    // against the just-renamed brain.lbr.
    brain.finalize_save()?;
    Ok(wrap_content(..., json!({"saved": true, "path": ...})))
}
```

The load-bearing trick: SNAPSHOT capture happens on the main thread (cheap, ~ms), then THE I/O-HEAVY work runs on the worker. Ingest can teach against the brain while phases (2)/(6)/(8) (the long ones) run on the worker. The MCP main thread is blocked ONLY on the snapshot capture and the final WAL truncate.

### 4.2 Wire contract

The harness expects `lu4d_save` to return `ok=True` only after the save IS COMMITTED (`scripts/ingest_curriculum_full.py:987-999` - `cp[corpus]["last_saved_id"] = last_id` runs only on `ok`). Two options:

- **(a) Preserve the contract: lu4d_save blocks until commit.** From the harness's perspective nothing changes. The MCP thread blocks on `join`, but ingest is single-threaded in the harness, so the harness was going to block on the JSON-RPC reply anyway. **The MCP main thread is free to accept other teach calls from concurrent clients (if any) during the save**, but in production the harness is the sole client, so the wall-clock save tax is unchanged for the current workload.
- **(b) Change the contract: lu4d_save returns immediately with `{started: true, save_id: N}`; harness polls a new `lu4d_save_status` until it returns `{committed: true, save_id: N}` before advancing last_saved_id.**

**Recommendation: start with (a), instrument both paths, evaluate.** Path (a) is wire-compatible and ALREADY unblocks throughput on multi-client deployments. Path (b) is the right shape for our single-client harness to keep ingesting during save, but requires harness changes (the brief acknowledges this is acceptable). We can land (a) first as a no-harness-change demonstration of the architecture, measure whether the harness's per-batch loop reveals other unrelated stalls, then land (b) as a follow-on.

If John picks (b) directly, the harness change is small (~30 LOC in `_maybe_save` to poll instead of accept the immediate return), the design above gains `tool_save_status` (`src/mcp.rs`), and the load-bearing change is: `tool_save` returns after snapshot capture, save_id incremented on the brain; `tool_save_status` reads `save_runner.last_completed_save_id` and returns `committed: last_completed >= save_id`.

### 4.3 Failure handling

- **Worker thread panics mid-save.** `JoinHandle::join` returns `Err`. The .tmp file is left on disk. The next `atomic_write_with` call begins with `let _ = fs::remove_file(&tmp_path)` (already implemented), so the leftover .tmp is swept on the next save attempt. The brain.lbr from the previous save is intact - the rename never happened.
- **Worker thread crashes the whole process.** Same disk state as above; on restart, `atomic_write_with` sweeps the .tmp on the next save. Brain.lbr from the previous save is the recovery anchor.
- **Main thread panics while worker is in flight.** Worker is a `JoinHandle` - if main panics, the process exits, worker is torn down with the process. Disk state: leftover .tmp, swept on next start. Acceptable.
- **Snapshot capture fails (allocation error during clone).** Returned as `Err` from `start_async_save`; lu4d_save returns the error; harness retries on next interval. Brain state unchanged.
- **Rename fails after worker completes.** Returned as `Err`; .tmp may or may not have been removed depending on the failure mode. `atomic_write_with`'s existing cleanup logic handles it.

### 4.4 Sidecar coordination

- **keyword_index (.kw)** - `compact_keyword_sidecar(new_crc)` runs on the main thread AFTER the rename. The CRC is read off the just-renamed brain.lbr (phase 8 wrote it into the trailer). This is post-rename main-thread work, so no concurrency hazard - but the .kw rewrite IS itself an additional ~4.3 GB write on the main thread. That's its own latency hit, separate from the v3enc save. If the .kw rewrite is also slow, it's a candidate for a follow-on port (OP-SAVE-ASYNC-KW). For now, in scope: only the brain.lbr write moves off-thread; the .kw compact stays on the main thread post-rename.
- **pending.wal / lb_session.wal / style.wal** - these are append-only WALs maintained by the main thread continuously. They are TRUNCATED only after the brain.lbr rename succeeds. Worker thread writes the WAL content (snapshot at save-start) into the brain.lbr's section #9/#11/#12; main thread truncates the WAL file after the rename. WAL state between save-start and rename-complete: still appending to disk WAL (durability for crash recovery) - but everything in those new entries is ALSO going to land in the next save's snapshot section, so the truncate-after-rename is safe.
  - **One gotcha:** the worker's snapshot includes pending entries up to mark P; after save commits, we truncate the WAL down to mark P. New entries written between save-start and rename-complete are >= P+1. Truncating to P preserves them on disk. The truncate implementations should accept a high-water-mark arg; if they currently truncate-to-zero, we need to adjust to truncate-up-to. **Audit needed:** verify `pending.truncate_wal()`, `lb_session.truncate_wal()`, `cs_style_store.truncate_wal()` semantics - do they accept a watermark, or just clear?
- **lu4d_status during save:** reads `brain.network.node_count()` etc. - lookups against live state. Since save runs on a worker thread with its own snapshot, status reads on main are unaffected. SAFE.

### 4.5 Resource cost during background save

- **Memory:** baseline + overlay-clone + side-table-clones + worker BufWriter (1 MiB). On freshly-archived brain, overlay-clone is small; on heavily-mutated brain (e.g. mid-training between archives) overlay clone can be ~1-2 GiB. Worst case: overlay equals full graph (theoretical, requires never re-archiving) = ~7 GiB. In all real cases, cheaper than v5 BufWriter's peak.
- **File descriptors:** +1 fd for the worker's tmp file (it owns the handle for the save duration). Side-table snapshots are in-memory, no fd cost.
- **IO bandwidth contention with ingest write throughput:** worker writes ~6.2 GB to disk over 60-70s = ~90 MB/s. Ingest's incremental disk writes are WAL-appends only - small, sequential. On a single-disk system the two streams contend; on Hetzner's NVMe SSDs the bandwidth ceiling is high enough that contention is mild (test under load). Worth instrumenting: ingest article throughput during a background save vs steady-state.
- **CPU:** worker thread saturates 1 core (zstd compression + ChaCha20-Poly1305 encryption are the hot paths). Main thread keeps another core busy with ingest. Cgroup CPU quota should accommodate both - confirm before deploy.

### 4.6 Test plan

- **Unit:** SnapshotSource impl tests - construct a brain with known overlay, snapshot it, mutate the brain, run the writer against the snapshot, verify the writer's output matches a pre-mutation reference save byte-for-byte. (~10 tests; tests live in `src/save_async.rs` and `src/network.rs`.)
- **Concurrency stress:** a test harness that runs `teach_with_kind` in a tight loop on the main thread while `start_async_save` is in flight. Asserts: (a) no panic, (b) the saved brain CRCs OK on read, (c) the post-save loaded brain reproduces the snapshot state EXACTLY (no leaked post-snapshot mutations).
- **WAL truncate watermark test:** start a save at WAL mark P, append entries P+1..P+10, wait for save to commit, verify post-rename WAL contains P+1..P+10 (not zero, not P+11+).
- **Save panic recovery:** inject a panic in the worker writer mid-way through phase (2). Verify: .tmp left on disk, lu4d_save returns Err, next lu4d_save sweeps and succeeds, brain.lbr is the pre-panic version.
- **End-to-end:** run a ~5K article ingest with a 100K bond save interval. Compare wall-clock + post-load brain content vs a control run of the blocking save. (Run on a non-prod brain copy, e.g. brain-catchup-start-20260511. Daemon stays stopped on /var/lib/lu/brain.lbr.)

---

## 5. Open questions for John

1. **Option choice.** Confirm Option 5 (snapshot thread) over Option 1 (fork), with Option 1 as fallback. The overlay-clone memory question is the only real concern - is overlay growth bounded enough that ~1-2 GiB clone-cost is acceptable?
2. **Wire contract (Section 4.2).** Path (a) block-until-commit preserves the harness's `ok=True iff committed` semantic with zero harness change; path (b) returns immediately with poll-status semantics for true non-blocking ingest. Which first?
3. **Memory ceiling.** Cgroup MemoryMax is currently sized for the baseline + v5 BufWriter peak. Snapshot adds overlay-clone (variable, up to ~1-2 GiB realistic). Acceptable, or do we need to raise the cgroup ceiling pre-deploy?
4. **.kw sidecar compact deferral.** Per Section 4.4 the .kw rewrite stays on the main thread post-rename. With brain.lbr ~6.2 GB and .kw ~4.3 GB, the .kw write at ~90 MB/s is its own ~50s stall. Worth a follow-on port OP-SAVE-ASYNC-KW, OR move it into the worker now (~+150 LOC)?
5. **WAL truncate watermark semantics (Section 4.4 gotcha).** Do `pending.truncate_wal()` / `lb_session.truncate_wal()` / `cs_style_store.truncate_wal()` currently accept a watermark, or do they zero-truncate? If zero-truncate, we need to extend their signatures - small change but cross-cutting.
6. **"Save in progress" indicator on lu4d_status.** Add `save_in_progress: bool` + `last_save_started_ms` + `last_save_completed_ms` to `lu4d_status`? Useful for monitoring; ~20 LOC.
7. **Test brain location.** Stress tests need a non-prod brain. Use brain-catchup-start-20260511 (434M) or a copy of brain.lbr to a scratch path? The former is small enough for fast iteration but doesn't exercise the partial-passthrough path realistically.

---

## 6. What is NOT in this port

- **No .kw sidecar compact off-thread** (deferred, Q4). Stays on main thread post-rename for now.
- **No fork-based snapshot** (deferred to Option-1 fallback if Option 5 hits memory blockers).
- **No differential save** (Option 4 - deferred, separate format work).
- **No chunked-yield save** (Option 3 - rejected, doesn't solve the consistency problem).
- **No change to the v3enc on-disk format.** Bytes-on-disk identical to the current blocking save's output. Loaders untouched.
- **No removal of the synchronous save path.** `Brain::save_to_format` is still callable from CLI flows, the migration tool, and any non-MCP caller. Only `tool_save` switches to the async runner.
- **No teach-side throttling.** Ingest continues at full rate during the background save. The whole point.
- **No production deploy.** Daemon STAYS STOPPED. Training STAYS PAUSED. No new branch created in this pass.

---

## 7. LOC estimate breakdown

Net, post-SAVE-ASYNC (Option 5, contract path (a)):

- New module `src/save_async.rs`: SnapshotSource struct + OverlaySnapshot + SaveRunner (worker thread + join handle + status fields). ~250 lines.
- `src/network.rs`: NetworkSource impl for SnapshotSource; OverlaySnapshot::capture helper that clones the heap overlay node/bond hashmaps + records archive Arc handle + bond/node-tombstone snapshots. ~120 lines.
- `src/brain.rs`: `start_async_save`, `finalize_save`, `save_runner` field on Brain, SideTableSource impl for a new `SideTableSnapshot` struct (carries cloned side tables + WAL watermarks). ~100 lines.
- `src/mcp.rs`: `tool_save` rewrite, optional `tool_save_status` stub (left empty if path (a)). ~30 lines.
- WAL truncate watermark plumbing (if Q5 currently zero-truncate): `pending.rs`, `lb_session.rs`, `cs_style.rs` signatures + impls. ~80 lines.
- Unit + concurrency tests: 10 unit + 5 stress + 3 panic-recovery tests. ~300 lines.
- lu4d_status `save_in_progress` field if Q6 yes: ~20 lines.

**Net: ~880 lines** (range ~700-950 depending on Q4/Q5/Q6 answers). Inside the 700-900 LOC envelope cited at the top.

For path (b) (non-blocking wire contract): +50 LOC for `tool_save_status` + harness coordination = ~930 total. Harness change ~30 LOC in `_maybe_save`.

If Option 1 (fork) becomes the chosen path instead: ~350-600 LOC for the fork machinery, MINUS the ~250 LOC for the snapshot module, but PLUS the allocator/fd/signal audit time (call it ~5-8 engineering hours of careful review work, not LOC).

**Time to implement (Option 5, path (a)):** 2-4 engineering hours for the snapshot + thread plumbing, 2-3 hours for the WAL watermark refactor if Q5 says yes, 2-3 hours for the test surface, 1 hour for the stress harness and end-to-end test on brain-catchup. Total ~7-11 hours. Time-of-day estimate biased toward action per [Timeline Optimism].

---

## 8. Halt boundary

This document IS Phase A. Phase B (implementation) does NOT start until John responds to the open questions in Section 5 and confirms Option 5 (or signals Option 1 fallback).

No branch created in this pass. No code edits. No commits. No deploys. No daemon start. No sub-agent delegation.

`lu.service` and `lu4d-curriculum-ingest.service` STAY STOPPED. Brain `/var/lib/lu/brain.lbr` STAYS at 6.2 GB pre-OP-SAVE-ASYNC state. Production binary `/usr/local/bin/lu_4_mcp` md5 `00f0df9cdd2da5605059056cf3e0bce9` STAYS as the deployed binary.

---

## 9. Resolved decisions

*(populated after John reviews Section 5)*

| # | Decision | Knob to flip if changed |
|---|---|---|
| 1 | _pending_ | Switch to Option 1 (fork) - new module `src/save_fork.rs`, allocator audit, fd close-on-exec audit |
| 2 | _pending_ | Switch to path (b) - add `tool_save_status`, change harness `_maybe_save` to poll |
| 3 | _pending_ | Raise cgroup `MemoryMax` in `lu.service` unit file |
| 4 | _pending_ | Move .kw compact into worker - +150 LOC, separate snapshot of keyword_index |
| 5 | _pending_ | Extend `truncate_wal()` to accept a watermark - cross-cutting signature change |
| 6 | _pending_ | Add `save_in_progress` to `lu4d_status` response - small additive change |
| 7 | _pending_ | Pick test brain - copy of prod brain.lbr to a scratch path vs brain-catchup-start-20260511 |
