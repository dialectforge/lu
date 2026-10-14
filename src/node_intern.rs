//! Node identity interning — `Seed → NodeId(u64)`.
//!
//! Phase 1 of OP-BRAIN-V4 (originally u32; widened to u64 in
//! OP-BRAIN-V5 Phase 1.1 — see `docs/OP-BRAIN-V5-Phase1.0-NodeId-Audit.md`).
//! The graph's adjacency lists are the hottest per-edge structure in
//! RAM: every bond contributes two entries (one in `outgoing[from]`,
//! one in `incoming[to]`). Storing each entry as a 32-byte `Seed` is
//! a 64-byte-per-bond tax that scales linearly with the corpus.
//! Replacing those entries with an 8-byte `NodeId` is a 4× shrink on
//! the dominant Vec content (down from the pre-widening 8× — the
//! widening trades one factor of 2 for headroom past 4.29B nodes);
//! the intern table itself is ~40 bytes per *node*, not per *bond*,
//! so the cross-over is at out-degree ≥ ~1 (i.e. every bond saves
//! memory).
//!
//! ## Platform requirement
//!
//! NodeIntern targets **64-bit platforms only** (where `usize >= u64`).
//! All `Vec` indexing uses `id.0 as usize`; this is a no-op same-width
//! cast on 64-bit Linux (the only deployed target) and would silently
//! truncate on a hypothetical 32-bit target. `reverse_lookup` carries
//! an explicit `usize::try_from` guard that fails loud rather than
//! truncating; `reverse_unchecked` documents the same precondition.
//!
//! ## Design
//!
//! - `NodeId(u64)` is a newtype around `u64`. The high 32 bits are
//!   reserved (current production has ~10M nodes; the OP-BRAIN-V5
//!   ceiling justification is in the design doc §2 Q1). The newtype
//!   keeps the compiler honest: a raw `u64` cannot be confused for
//!   a NodeId at call sites.
//!
//! - `NodeIntern` holds the forward map `HashMap<Seed, NodeId>` plus
//!   the reverse `Vec<Seed>` indexed by NodeId. Insert-then-lookup is
//!   the hot pattern (load path interns every Seed encountered in the
//!   archive's bond index); reverse lookup is a single bounds-checked
//!   `Vec` index.
//!
//! - Thread-safety: `Arc<RwLock<NodeInternState>>`. Reads dominate
//!   (every adjacency reverse-lookup); writes are concentrated at load
//!   time and at `add_bond` boundaries. We avoid `dashmap` until
//!   benchmarks show contention — the daemon is largely
//!   single-writer-multi-reader. (The spec explicitly says: "only
//!   escalate to `dashmap` if Phase 0 bench shows contention.")
//!
//! - The intern table is **in-memory only**. v3 archive on-disk
//!   format is unchanged in Phase 1. On load, every Seed in the
//!   archive's bond index is interned afresh; NodeIds are NOT stable
//!   across save/load cycles by design — they are assigned in the
//!   order Seeds are first observed during load, which is a
//!   deterministic walk of the archived bond index (sorted by Seed
//!   bytes in `network::AssociativeNetwork::from_archive`).
//!
//! ## Invariants
//!
//! 1. `forward[seed]` and `reverse[id.0 as usize] == seed` are kept in
//!    sync under the same lock.
//! 2. `intern_or_get` is idempotent: interning the same Seed twice
//!    returns the same NodeId.
//! 3. `get` is a pure lookup; returns `None` for unknown Seeds.
//! 4. `reverse_lookup(id)` returns `None` for out-of-range NodeIds, so
//!    a stale NodeId from a dropped intern table never reads garbage.
//!
//! ## Not in scope (Phase 1)
//!
//! - On-disk encoding of NodeIds. The v3 archive on-disk format does
//!   NOT carry NodeIds. Phase 2/4 will introduce the SST format with
//!   the new fixed-width 24-byte CompactBond carrying brain-wide u64
//!   NodeIds (per OP-BRAIN-V5 design doc §2 Q2).
//! - Eviction / shrinking. Once a Seed is interned it stays interned
//!   for the process lifetime. Brain reloads bring a fresh intern
//!   table, so memory is reclaimed at the reload boundary.

use crate::seed::Seed;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Compact node identity assigned by the intern table.
///
/// 64 bits wide — high 32 bits reserved per OP-BRAIN-V5 design doc §2
/// Q1. The newtype around `u64` keeps the type system honest: a raw
/// `u64` cannot be silently used where a NodeId is expected, and vice
/// versa.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub u64);

// Note: the pre-widening `as_u32` method has been dropped (Phase 1.0
// audit Q1.0.A — zero external callers; keeping it post-widening
// would have been a silent-truncation hazard).

#[derive(Default)]
struct NodeInternState {
    /// `Seed → NodeId`. Indexed by the 32-byte Seed.
    forward: HashMap<Seed, NodeId>,
    /// `NodeId → Seed`. `reverse[id.0 as usize] == seed_for(id)`.
    /// Append-only: NodeIds are assigned in insertion order.
    reverse: Vec<Seed>,
}

/// Thread-safe `Seed ↔ NodeId` interning table.
///
/// Clone the handle (it's an `Arc`) to share between threads. Internal
/// mutability via `RwLock`. The lock is held briefly in both `get` and
/// `intern_or_get`; there is no scenario in Phase 1 where it is held
/// across a non-trivial computation.
#[derive(Clone, Default)]
pub struct NodeIntern {
    inner: Arc<RwLock<NodeInternState>>,
}

impl NodeIntern {
    /// Create an empty intern table.
    pub fn new() -> Self { Self::default() }

    /// Create an intern table preallocated for `cap` nodes. Reduces
    /// reallocation pressure during a large load.
    pub fn with_capacity(cap: usize) -> Self {
        let inner = NodeInternState {
            forward: HashMap::with_capacity(cap),
            reverse: Vec::with_capacity(cap),
        };
        NodeIntern { inner: Arc::new(RwLock::new(inner)) }
    }

    /// Look up a Seed without interning. Returns `None` if the Seed
    /// has never been observed.
    pub fn get(&self, seed: &Seed) -> Option<NodeId> {
        let g = self.inner.read().expect("NodeIntern read lock");
        g.forward.get(seed).copied()
    }

    /// Look up a Seed, or assign a new NodeId if it is novel.
    /// Idempotent: calling with the same Seed twice returns the same
    /// NodeId.
    ///
    /// Note: the pre-widening `debug_assert!(raw <= u32::MAX as usize, …)`
    /// guard that lived here has been removed by design — the entire
    /// point of the u32 → u64 widening (OP-BRAIN-V5 Phase 1.1) was to
    /// retire the 4.29B-node ceiling. No meaningful replacement exists
    /// at u64 (the 18-quintillion ceiling is not a real-world failure
    /// mode); the silent-wraparound safety net is gone by design. See
    /// `docs/OP-BRAIN-V5-Phase1.0-NodeId-Audit.md` SURPRISE-2.
    pub fn intern_or_get(&self, seed: Seed) -> NodeId {
        // Fast path: read lock, hit returns immediately.
        if let Some(id) = self.get(&seed) {
            return id;
        }
        // Slow path: write lock, double-check, insert.
        let mut w = self.inner.write().expect("NodeIntern write lock");
        if let Some(&id) = w.forward.get(&seed) {
            return id;
        }
        let raw = w.reverse.len();
        let id = NodeId(raw as u64);
        w.reverse.push(seed);
        w.forward.insert(seed, id);
        id
    }

    /// Same as `intern_or_get`, but additionally surfaces whether this
    /// call performed a fresh insert. `(node_id, true)` means THIS call
    /// allocated `node_id` for `seed`; `(node_id, false)` means `seed`
    /// was already interned by an earlier call (possibly on another
    /// thread).
    ///
    /// Used by OP-BRAIN-V5 Phase 1.4 wiring (`Brain::teach_with_kind_v4`
    /// and `Brain::persist_node_v4`): only the caller that observes
    /// `was_newly_inserted == true` is responsible for writing the
    /// corresponding `REC_TYPE_NODE_INTERN` WAL record. The caller MUST
    /// invoke this method while holding the WAL append lock, so that
    /// the intern-table commit and the NodeIntern WAL write are
    /// observed by all subsequent threads in the same order (Q8
    /// ordering invariant).
    ///
    /// Idempotence: under concurrent calls with the same `seed`, the
    /// `true` flag is returned to EXACTLY ONE caller; every other
    /// caller observes `false` plus the canonical NodeId.
    pub fn intern_with_new_flag(&self, seed: Seed) -> (NodeId, bool) {
        // Fast path: read lock, hit returns (id, false) immediately.
        if let Some(id) = self.get(&seed) {
            return (id, false);
        }
        // Slow path: write lock, double-check (another thread may
        // have inserted between our read-unlock and write-lock),
        // insert if still novel.
        let mut w = self.inner.write().expect("NodeIntern write lock");
        if let Some(&id) = w.forward.get(&seed) {
            return (id, false);
        }
        let raw = w.reverse.len();
        let id = NodeId(raw as u64);
        w.reverse.push(seed);
        w.forward.insert(seed, id);
        (id, true)
    }

    /// OP-BRAIN-V5 Phase 1.3 — populate this NodeIntern by interning
    /// every Seed reachable through a v3 archive's bond_index.
    /// Returns the number of Seeds newly interned by this call.
    ///
    /// Used by both the runtime cold-load path
    /// (`AssociativeNetwork::from_archive`) and the future Phase 7
    /// migration tool. Extracted from `network.rs` for shared,
    /// testable behavior — see `docs/OP-BRAIN-V5-Phase1.0-NodeId-Audit.md`
    /// Phase 1.3 entry.
    ///
    /// Determinism: for a given archive, the NodeIds assigned by this
    /// method depend on the iteration order of `archive.iter_bond_keys()`.
    /// Within a single process invocation that order is fixed (HashMap
    /// RandomState seeded once per process); across processes it varies.
    /// Callers that need cross-process-stable NodeIds must sort the
    /// adjacency at a higher layer (`network.rs::from_archive` does
    /// exactly this for the BR.C-Hyp-Methodology determinism
    /// invariant).
    pub fn intern_from_v3_archive(
        &self,
        archive: &crate::archive::NetworkArchive,
    ) -> usize {
        let before = self.len();
        for (from, to) in archive.iter_bond_keys() {
            let _ = self.intern_or_get(from);
            let _ = self.intern_or_get(to);
        }
        self.len() - before
    }

    /// Reverse lookup. Returns `None` for out-of-range NodeIds (e.g.
    /// a stale id from a dropped intern table) — never reads garbage.
    ///
    /// 64-bit-target guard: uses `usize::try_from` so the implicit
    /// `u64 → usize` cast cannot silently truncate on a hypothetical
    /// 32-bit target. On 64-bit Linux (the only deployed target) this
    /// is a no-op same-width conversion. See module header.
    pub fn reverse_lookup(&self, id: NodeId) -> Option<Seed> {
        let idx = usize::try_from(id.0).expect("NodeIntern requires 64-bit target");
        let g = self.inner.read().expect("NodeIntern read lock");
        g.reverse.get(idx).copied()
    }

    /// Reverse lookup that panics on miss. Use only on hot paths where
    /// the NodeId is guaranteed to be live (e.g. iterating an
    /// adjacency list owned by the same network as the intern table).
    /// Same 64-bit-target precondition as `reverse_lookup`.
    pub fn reverse_unchecked(&self, id: NodeId) -> Seed {
        let g = self.inner.read().expect("NodeIntern read lock");
        g.reverse[id.0 as usize]
    }

    /// Number of interned nodes.
    pub fn len(&self) -> usize {
        let g = self.inner.read().expect("NodeIntern read lock");
        g.reverse.len()
    }

    /// True iff no Seeds have been interned.
    pub fn is_empty(&self) -> bool {
        let g = self.inner.read().expect("NodeIntern read lock");
        g.reverse.is_empty()
    }

    /// Batch reverse-lookup. Returns owned Seeds for each NodeId in
    /// the iterator; out-of-range NodeIds are skipped. Holds the read
    /// lock once for the whole batch, so this is cheaper than calling
    /// `reverse_lookup` in a loop when collecting an adjacency list.
    /// Same 64-bit-target precondition as `reverse_lookup`.
    pub fn reverse_collect<I: IntoIterator<Item = NodeId>>(&self, ids: I) -> Vec<Seed> {
        let g = self.inner.read().expect("NodeIntern read lock");
        ids.into_iter()
            .filter_map(|id| g.reverse.get(id.0 as usize).copied())
            .collect()
    }

    // ----- OP-BRAIN-V5 Phase 2.1 (Q3, Q6) -----
    //
    // `save_to` / `load_from` / `insert_with_id` add the sidecar I/O
    // surface needed by the node-store substrate. The wire format
    // lives in `src/node_store/sidecar.rs`; the methods below sit on
    // NodeIntern so callers don't reach into the inner state.
    //
    // **Not yet wired**: no production save path calls `save_to`. The
    // Phase 2.3+ flush worker takes ownership of the call site. This
    // commit ships the I/O primitives only, end-to-end unit-tested.

    /// Stream every interned `(Seed, NodeId)` binding to a sidecar
    /// file at `path`, sorted by NodeId ascending.
    ///
    /// On success returns sanity stats (record count + file size +
    /// CRC32 + elapsed). The file is opened with `create + truncate`,
    /// streamed through a 64 KiB `BufWriter`, and `fsync_all`'d before
    /// close. Atomicity is the caller's job (write to `path.tmp`,
    /// then rename); this method is the raw I/O primitive.
    ///
    /// ## Honest scope: bounded-RSS
    ///
    /// To emit records in NodeId-ascending order this fn collects a
    /// `Vec<(NodeId, Seed)>` of `self.len()` entries (~40 B each) and
    /// sorts in place. At the design-doc ceiling of 100M nodes that's
    /// **~4 GB resident during save**, on top of the existing NodeIntern.
    /// The cost is bounded by `NodeIntern::len()` -- NOT amplified by
    /// the file size -- and is documented as the chosen trade-off vs.
    /// pre-sorting NodeIntern internals (which would require swapping
    /// `reverse: Vec<Seed>` for an indexed-by-NodeId Vec; that
    /// refactor lives in Phase 2.3, NOT this port).
    ///
    /// At Lu's current 9-10M node corpus the sort buffer is ~400 MB
    /// -- comfortably inside the 64 GB AX41 budget.
    pub fn save_to(
        &self,
        path: &std::path::Path,
    ) -> Result<crate::node_store::SidecarStats, crate::node_store::SidecarError> {
        self.save_to_with_key(path, None)
    }

    /// OP-BRAIN-V5 6.8: key-aware [`Self::save_to`]. When `key` is
    /// `Some`, the records region is XOR-keystream encrypted at rest
    /// (v2 sidecar header + FLAG_ENCRYPTED + IV); `None` writes the
    /// byte-identical pre-6.8 plaintext sidecar.
    pub fn save_to_with_key(
        &self,
        path: &std::path::Path,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<crate::node_store::SidecarStats, crate::node_store::SidecarError> {
        // Collect under a single read lock to keep the snapshot
        // self-consistent; release before the I/O begins so the
        // intern stays available to readers.
        let mut records: Vec<(NodeId, Seed)> = {
            let g = self.inner.read().expect("NodeIntern read lock");
            g.forward.iter().map(|(seed, id)| (*id, *seed)).collect()
        };
        records.sort_by_key(|(id, _)| *id);
        let record_count = records.len() as u64;
        crate::node_store::sidecar::write_sidecar(path, records.into_iter(), record_count, key)
    }

    /// Load a sidecar file produced by [`Self::save_to`] into a fresh
    /// `NodeIntern`. Validates the header magic, format version,
    /// record count upper bound, and trailing CRC32 over header +
    /// records. Any mismatch surfaces as a typed
    /// [`crate::node_store::SidecarError`].
    ///
    /// On CRC mismatch the returned NodeIntern is NOT returned (the
    /// error is surfaced after records have been delivered to the
    /// intern). Caller should treat CRC failure the same as
    /// sidecar-absent: fall back to the WAL-replay rebuild path.
    /// (Phase 2.2 wires that fallback in `Brain::enable_memtable`.)
    pub fn load_from(
        path: &std::path::Path,
    ) -> Result<Self, crate::node_store::SidecarError> {
        Self::load_from_with_key(path, None)
    }

    /// OP-BRAIN-V5 6.8: key-aware [`Self::load_from`]. An encrypted
    /// (v2 + FLAG_ENCRYPTED) sidecar requires `key`; supplying `None`
    /// against one fails cleanly with
    /// [`crate::node_store::SidecarError::EncryptedButNoKey`].
    pub fn load_from_with_key(
        path: &std::path::Path,
        key: Option<crate::encrypt::MasterKey>,
    ) -> Result<Self, crate::node_store::SidecarError> {
        let intern = Self::new();
        crate::node_store::sidecar::read_sidecar(path, key, |seed, node_id| {
            intern.insert_with_id(seed, node_id)
        })?;
        Ok(intern)
    }

    /// Bulk-load helper: insert a `Seed <-> NodeId` binding with a
    /// CALLER-SPECIFIED NodeId. Used by [`Self::load_from`] (sidecar
    /// restoration) and, in Phase 2.2, by the boot-replay arm to
    /// apply WAL `REC_TYPE_NODE_INTERN` records verbatim instead of
    /// reassigning by insertion order.
    ///
    /// ## Behaviour
    ///
    /// - If `seed` is not yet interned, inserts the binding. The
    ///   reverse `Vec<Seed>` is extended on demand to `node_id.0 + 1`
    ///   entries, with any gap filled by the sentinel `Seed([0; 32])`.
    ///   Out-of-order inserts are explicitly supported (audit §6.2 +
    ///   `reverse_lookup` consumers tolerate the sentinel via
    ///   `get`-style queries -- sentinels never collide with a real
    ///   BLAKE3 seed of label content; the empty-string seed is
    ///   `Seed([0; 32])` which only matches by deliberate
    ///   construction).
    /// - If `seed` is already bound to the SAME `node_id`, succeeds
    ///   idempotently (no-op).
    /// - If `seed` is already bound to a DIFFERENT `node_id`, errors
    ///   with [`crate::node_store::SidecarError::InsertWithIdCollision`].
    ///
    /// ## Why on NodeIntern's public surface
    ///
    /// The brief's audit §6.2 marked `insert_with_id` as bulk-load
    /// + boot-replay. Both call sites live outside `node_intern.rs`
    /// (`node_store/sidecar.rs::read_sidecar` and, in Phase 2.2,
    /// `Brain::enable_memtable`). Keeping the method on the public
    /// `NodeIntern` surface avoids exposing the inner `RwLock`. It is
    /// NOT marked `#[doc(hidden)]` -- the audit recommendation to do
    /// so was reconsidered against `feedback_no_hacks`: hiding a real
    /// API behind a doc attribute is a discipline signal, not an
    /// enforcement; the typed error + the doc-comment carry the
    /// contract just as clearly.
    pub fn insert_with_id(
        &self,
        seed: Seed,
        node_id: NodeId,
    ) -> Result<(), crate::node_store::SidecarError> {
        let mut w = self.inner.write().expect("NodeIntern write lock");
        if let Some(&existing) = w.forward.get(&seed) {
            if existing == node_id {
                return Ok(()); // idempotent re-insert
            }
            return Err(crate::node_store::SidecarError::InsertWithIdCollision {
                seed,
                existing,
                attempted: node_id,
            });
        }
        // Extend reverse Vec on demand. Sentinel = Seed([0; 32]).
        let needed = (node_id.0 as usize).saturating_add(1);
        if w.reverse.len() < needed {
            w.reverse.resize(needed, Seed::from_bytes([0u8; 32]));
        }
        // Overwrite the sentinel (or a re-insertion at the same slot,
        // which only happens if a previous insert at this NodeId was
        // for a different seed -- which we'd have caught above via
        // the forward map). Safe.
        w.reverse[node_id.0 as usize] = seed;
        w.forward.insert(seed, node_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn intern_or_get_idempotent() {
        let intern = NodeIntern::new();
        let a = Seed::from_label("alpha");
        let id1 = intern.intern_or_get(a);
        let id2 = intern.intern_or_get(a);
        assert_eq!(id1, id2);
        assert_eq!(intern.get(&a), Some(id1));
    }

    #[test]
    fn distinct_seeds_get_distinct_ids() {
        let intern = NodeIntern::new();
        let a = Seed::from_label("alpha");
        let b = Seed::from_label("beta");
        let id_a = intern.intern_or_get(a);
        let id_b = intern.intern_or_get(b);
        assert_ne!(id_a, id_b);
        assert_eq!(id_a.0, 0u64);
        assert_eq!(id_b.0, 1u64);
    }

    #[test]
    fn reverse_lookup_round_trip() {
        let intern = NodeIntern::new();
        let a = Seed::from_label("alpha");
        let id = intern.intern_or_get(a);
        assert_eq!(intern.reverse_lookup(id), Some(a));
        assert_eq!(intern.reverse_unchecked(id), a);
    }

    #[test]
    fn reverse_lookup_out_of_range_returns_none() {
        let intern = NodeIntern::new();
        let _ = intern.intern_or_get(Seed::from_label("alpha"));
        assert_eq!(intern.reverse_lookup(NodeId(99)), None);
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let intern = NodeIntern::new();
        let unknown = Seed::from_label("never_interned");
        assert_eq!(intern.get(&unknown), None);
    }

    #[test]
    fn concurrent_intern_same_seed_returns_same_id() {
        // Many threads interning the SAME Seed simultaneously. All
        // must observe the same NodeId; no duplicates in the reverse
        // table.
        let intern = NodeIntern::new();
        let seed = Seed::from_label("contended");
        let mut handles = Vec::new();
        for _ in 0..32 {
            let intern = intern.clone();
            handles.push(thread::spawn(move || intern.intern_or_get(seed)));
        }
        let ids: Vec<NodeId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = ids[0];
        assert!(ids.iter().all(|&id| id == first));
        assert_eq!(intern.len(), 1);
    }

    #[test]
    fn concurrent_intern_distinct_seeds_assigns_distinct_ids() {
        // Many threads each interning a DIFFERENT Seed. All NodeIds
        // must be distinct; len() == number of threads.
        let intern = NodeIntern::new();
        let n = 64;
        let mut handles = Vec::new();
        for i in 0..n {
            let intern = intern.clone();
            handles.push(thread::spawn(move || {
                intern.intern_or_get(Seed::from_label(&format!("s_{}", i)))
            }));
        }
        let ids: Vec<NodeId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n as usize);
        assert_eq!(intern.len(), n as usize);
    }

    #[test]
    fn reverse_collect_filters_out_of_range() {
        let intern = NodeIntern::new();
        let a = Seed::from_label("alpha");
        let b = Seed::from_label("beta");
        let id_a = intern.intern_or_get(a);
        let id_b = intern.intern_or_get(b);
        let bogus = NodeId(9999);
        let out = intern.reverse_collect(vec![id_a, bogus, id_b]);
        assert_eq!(out, vec![a, b]);
    }

    #[test]
    fn with_capacity_does_not_pre_intern() {
        let intern = NodeIntern::with_capacity(1024);
        assert!(intern.is_empty());
        assert_eq!(intern.len(), 0);
    }

    #[test]
    fn newtype_does_not_coerce_to_u64_implicitly() {
        // Compile-time check: NodeId is a newtype, not a type alias.
        // The field is pub so explicit access works, but no implicit
        // coercion. This test exists to document intent.
        // Post-widening (OP-BRAIN-V5 Phase 1.1) the inner type is u64;
        // the previous u32-typed witness has been retired with `as_u32`.
        let id = NodeId(42);
        let _: u64 = id.0;
    }

    /// OP-BRAIN-V5 Phase 1.4 — `intern_with_new_flag` returns `true`
    /// exactly once per Seed, `false` on every subsequent call.
    #[test]
    fn intern_with_new_flag_returns_true_only_on_first_observe() {
        let intern = NodeIntern::new();
        let s = Seed::from_label("first_seen");
        let (id1, new1) = intern.intern_with_new_flag(s);
        assert_eq!(new1, true);
        let (id2, new2) = intern.intern_with_new_flag(s);
        assert_eq!(new2, false);
        assert_eq!(id1, id2);
        // A different Seed must again return true.
        let s2 = Seed::from_label("second_seen");
        let (_, new3) = intern.intern_with_new_flag(s2);
        assert_eq!(new3, true);
        // Idempotence: `get` returns the same id.
        assert_eq!(intern.get(&s), Some(id1));
    }

    /// OP-BRAIN-V5 Phase 1.4 — under concurrent calls with the SAME
    /// Seed, `intern_with_new_flag` returns `true` to exactly one
    /// thread. All other threads observe `false` and the same NodeId.
    #[test]
    fn intern_with_new_flag_concurrent_same_seed_true_observed_once() {
        let intern = NodeIntern::new();
        let s = Seed::from_label("contended_new_flag");
        let mut handles = Vec::new();
        for _ in 0..32 {
            let intern = intern.clone();
            handles.push(thread::spawn(move || intern.intern_with_new_flag(s)));
        }
        let results: Vec<(NodeId, bool)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let true_count = results.iter().filter(|(_, b)| *b).count();
        assert_eq!(true_count, 1, "exactly one thread must observe the fresh insert");
        let first_id = results[0].0;
        assert!(results.iter().all(|(id, _)| *id == first_id));
        assert_eq!(intern.len(), 1);
    }

}
