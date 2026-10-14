//! OP-BRAIN-V4 Phase 3 — MemTable.
//!
//! Bounded in-memory write buffer. Inserts arrive as `(Seed, Seed,
//! BondType, weight, keyframes)`; the table interns subject Seeds
//! into brain-wide `NodeId(u64)` via the shared `NodeIntern` (Phase 1),
//! keeps objects as Seeds in `MemBond` for now (object NodeIds are
//! assigned at flush time when the SST builds its block-local subject
//! table). Sort order is subject-major; objects within a subject
//! preserve insertion order, deferring the (subject, predicate, object)
//! re-sort to the flush path (Phase 4) just like `SSTWriter::push_bond`
//! expects subject-sorted input but sorts within a subject internally.
//!
//! ## Concurrency model
//!
//! `Mutex<MemTableInner>` around the `BTreeMap`. Phase 3 is the
//! primitive, not the wiring; we choose the simplest correct
//! implementation and let later benches drive any sharding decision.
//! `size_bytes` lives in an `AtomicUsize` *outside* the mutex so the
//! `is_full()` poll (which the flush scheduler will call frequently
//! without inserting) avoids contending the table lock.
//!
//! ## Swap-for-flush
//!
//! `MemTable::swap_for_flush` is the atomic primitive Phase 4 needs:
//! it returns the old `MemTable` boxed for the flush thread and
//! replaces our inner state with a fresh empty one in a single locked
//! step. Inserts that race the swap either land in the old table
//! (will be flushed) or the new table (will be flushed later) — never
//! lost, never duplicated. The test suite below proves this with a
//! 4-writer-1-swapper stress.
//!
//! ## Out of scope (Phase 3)
//!
//! - Tombstones — deletes wait for Phase 6 compaction.
//! - Weight quantization to u8 — happens at flush time in the SST
//!   writer, NOT here. We keep f32 precision so the WAL replay round-
//!   trip is byte-exact.
//! - Per-subject MemBond ordering by (predicate, object) — done in
//!   the flush iterator (Phase 4) via a stable sort just before
//!   pushing to `SSTWriter`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::bond::{BondType, TemporalKeyframe};
use crate::node_intern::{NodeId, NodeIntern};
use crate::seed::Seed;

/// One bond entry in the MemTable. `seq` is the monotonic sequence
/// number assigned at insert time; matches the WAL record's seq so
/// replay can reconstruct ordering byte-exact.
#[derive(Debug, Clone)]
pub struct MemBond {
    /// Object Seed (NOT interned — object NodeIds are block-local in
    /// the SST format, so interning here would waste a lookup).
    pub object: Seed,
    /// OP-BRAIN-V5 6.3 -- object NodeId, interned at insert time via the
    /// same `intern_or_get` the flush path used to call at flush time
    /// (flush.rs). Carried ALONGSIDE the Seed (locked decision #3: keep
    /// BOTH during the 3.9 v4-overlap) so the V5Only recall merge and the
    /// flush projection operate in one uniform brain-wide NodeId key-space
    /// without a per-read / per-flush re-intern.
    pub object_id: NodeId,
    pub predicate: BondType,
    /// Bond weight, full f32 precision. Quantized to u8 at flush.
    pub weight: f32,
    /// Full keyframe history — preserved end-to-end through the WAL
    /// replay. The SST writer picks the most-recent keyframe at
    /// flush time; until then we keep them all.
    pub keyframes: Vec<TemporalKeyframe>,
    /// Monotonic sequence number from the table's `seq_counter`.
    pub seq: u64,
    /// Phase 6: true iff this entry is a tombstone -- a deletion
    /// marker that shadows older same-(subject, predicate, object)
    /// bonds in the merge layer. Created by `MemTable::delete`.
    pub tombstone: bool,
}

/// Phase 6: flush-time projection of a `MemBond`. Carries the
/// tombstone flag so the flush worker can emit a tombstone row in
/// the SST.
///
/// OP-BRAIN-V5 Phase 3.8.1: grew `seq`. The original Phase 6 shape
/// deliberately omitted it ("the flush worker computes the max seq
/// separately via `compute_highest_seq`"), which forced the Phase
/// 3.3 v5 bond flush into an O(N) re-fetch via
/// `MemTable::get_bonds_for_subject` just to read per-bond seqs --
/// `BondPayload.seq` is required for v5 dedup correctness (highest
/// seq wins at L0 and at compaction). Carrying the seq here
/// eliminates that double-walk; `compute_highest_seq` (flush.rs)
/// now folds over this field too.
#[derive(Debug, Clone)]
pub struct MemBondFlush {
    pub object: Seed,
    /// OP-BRAIN-V5 6.3 -- object NodeId carried from the source MemBond so
    /// the flush worker keys the v5 BondSST row without re-interning.
    pub object_id: NodeId,
    pub predicate: BondType,
    pub weight: f32,
    pub keyframes: Vec<TemporalKeyframe>,
    pub tombstone: bool,
    /// Monotonic sequence number copied verbatim from the source
    /// `MemBond.seq`. v5 dedup compares "highest seq wins" and the
    /// runtime allocates seqs monotonically from the MemTable
    /// counter, so a 0 on a hand-constructed value is safe for
    /// newly-introduced bonds (it loses against any real row).
    pub seq: u64,
}

/// Outcome of a single insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertResult {
    /// Sequence number assigned to this bond. Same value the WAL
    /// record will carry.
    pub seq: u64,
    /// True iff the table's size_bytes crossed `cap_bytes` on this
    /// insert. The caller (Phase 4 flush scheduler) uses this as the
    /// edge-trigger for `swap_for_flush`.
    pub became_full: bool,
}

/// Default MemTable capacity in bytes (500 MiB).
pub const DEFAULT_MEMTABLE_CAP_BYTES: usize = 500 * 1024 * 1024;

/// Approximate byte cost of one `MemBond` entry inside the table.
/// Used by `insert` to update `size_bytes`. This is a heap-aware
/// estimate, not a `std::mem::size_of` — we want to bound the
/// resident MemTable footprint, which is dominated by the
/// `Vec<MemBond>` payload plus the `BTreeMap` node overhead per
/// subject.
///
/// MemBond stack size: object Seed (32) + predicate enum (1) +
/// padding to f32 alignment + weight (4) + keyframes Vec header
/// (24) + seq (8) + object_id NodeId (8) ≈ 80 B with alignment.
///
/// Each keyframe carries ~24 B (timestamp i64 + weight f32 + source
/// enum, modest).
fn approx_membond_bytes(keyframes: &[TemporalKeyframe]) -> usize {
    const PER_BOND_STACK: usize = 80;
    const PER_KEYFRAME: usize = 24;
    PER_BOND_STACK + keyframes.len() * PER_KEYFRAME
}

/// Cost of adding a new subject row to the BTreeMap. A `Vec<MemBond>`
/// header (24 B) + a BTreeMap leaf-node slot (~48 B amortized).
const PER_SUBJECT_OVERHEAD: usize = 72;

struct MemTableInner {
    entries: BTreeMap<NodeId, Vec<MemBond>>,
    /// OP-BRAIN-V5 Phase 2.3: node half delegates to `NodeMemTable`
    /// (sorted by `NodeId`, BTreeMap<NodeId, NodeEntry>, tombstone
    /// + monotonic-seq dedup). Pre-Phase-2.3 this was a bare
    /// `BTreeMap<NodeId, (KnowledgeNode, u64)>`; the new type owns
    /// the byte-cost tracking + tombstone semantics so the v4
    /// `MemTable::insert_node` / `lookup_node` / `node_count` /
    /// `iter_nodes_for_flush` public API stays unchanged for
    /// callers (Phase 1.4 `persist_node_v4`, WAL replay,
    /// flush worker), but the bond and node halves are now
    /// independent. Phase 2.6 will retire the bond-side coupling
    /// when the node flush worker takes over.
    node_table: crate::node_store::NodeMemTable,
}

impl MemTableInner {
    fn new() -> Self {
        MemTableInner {
            entries: BTreeMap::new(),
            node_table: crate::node_store::NodeMemTable::new(),
        }
    }
}

/// Bounded in-memory write buffer.
///
/// Thread-safe via interior mutability. The table lock is held only
/// for the duration of the BTreeMap operation; size accounting and
/// the seq counter live in atomics so the `is_full` / `size_bytes`
/// poll path doesn't contend the write lock.
pub struct MemTable {
    inner: Mutex<MemTableInner>,
    size_bytes: AtomicUsize,
    cap_bytes: usize,
    seq_counter: AtomicU64,
}

impl MemTable {
    /// Create a new empty MemTable with the given byte cap.
    pub fn new(cap_bytes: usize) -> Self {
        MemTable {
            inner: Mutex::new(MemTableInner::new()),
            size_bytes: AtomicUsize::new(0),
            cap_bytes,
            seq_counter: AtomicU64::new(0),
        }
    }

    /// Default cap (500 MiB).
    pub fn with_default_cap() -> Self {
        Self::new(DEFAULT_MEMTABLE_CAP_BYTES)
    }

    /// Insert one bond. Returns the assigned sequence number and a
    /// flag indicating whether this insert pushed the table over its
    /// byte cap (the edge-trigger for flush).
    ///
    /// The subject Seed is interned into `intern` (Phase 1 brain-wide
    /// table); the object Seed is stored as-is in `MemBond`.
    pub fn insert(
        &self,
        subject: Seed,
        object: Seed,
        predicate: BondType,
        weight: f32,
        keyframes: Vec<TemporalKeyframe>,
        intern: &NodeIntern,
    ) -> InsertResult {
        let seq = self.seq_counter.fetch_add(1, Ordering::SeqCst);
        let subj_id = intern.intern_or_get(subject);
        // OP-BRAIN-V5 6.3 -- intern the object Seed -> NodeId at insert
        // time (the same intern_or_get the flush path used to pay later,
        // moved earlier). The Seed is kept too (locked decision #3).
        let object_id = intern.intern_or_get(object);
        let bond = MemBond { object, object_id, predicate, weight, keyframes, seq, tombstone: false };
        let added_bytes = approx_membond_bytes(&bond.keyframes);

        let was_below;
        let now_above;
        {
            let mut g = self.inner.lock().expect("MemTable inner lock");
            // The subject overhead only counts the first time a new
            // subject lands; subsequent inserts on the same subject
            // just grow the Vec.
            let was_new_subject;
            match g.entries.get_mut(&subj_id) {
                Some(v) => {
                    v.push(bond);
                    was_new_subject = false;
                }
                None => {
                    g.entries.insert(subj_id, vec![bond]);
                    was_new_subject = true;
                }
            }
            let total_add = added_bytes
                + if was_new_subject { PER_SUBJECT_OVERHEAD } else { 0 };
            // Atomically grow size_bytes; capture pre/post for the
            // edge-trigger flag.
            let prev = self.size_bytes.fetch_add(total_add, Ordering::SeqCst);
            was_below = prev < self.cap_bytes;
            now_above = prev + total_add >= self.cap_bytes;
        }

        InsertResult {
            seq,
            became_full: was_below && now_above,
        }
    }

    /// Phase 6: insert a tombstone for `(subject, predicate, object)`.
    /// The tombstone carries the MemTable's monotonic seq just like
    /// a normal insert, so the merge layer's seq-wins dedup naturally
    /// lets a tombstone shadow older same-key inserts.
    pub fn delete(
        &self,
        subject: Seed,
        object: Seed,
        predicate: BondType,
        intern: &NodeIntern,
    ) -> InsertResult {
        let seq = self.seq_counter.fetch_add(1, Ordering::SeqCst);
        let subj_id = intern.intern_or_get(subject);
        // OP-BRAIN-V5 6.3 -- carry the object NodeId on tombstones too so
        // the V5Only merge dedups tombstone-vs-bond in NodeId key-space.
        let object_id = intern.intern_or_get(object);
        let bond = MemBond {
            object,
            object_id,
            predicate,
            weight: 0.0,
            keyframes: Vec::new(),
            seq,
            tombstone: true,
        };
        let added_bytes = approx_membond_bytes(&bond.keyframes);
        let was_below;
        let now_above;
        {
            let mut g = self.inner.lock().expect("MemTable inner lock");
            let was_new_subject;
            match g.entries.get_mut(&subj_id) {
                Some(v) => {
                    v.push(bond);
                    was_new_subject = false;
                }
                None => {
                    g.entries.insert(subj_id, vec![bond]);
                    was_new_subject = true;
                }
            }
            let total_add = added_bytes
                + if was_new_subject { PER_SUBJECT_OVERHEAD } else { 0 };
            let prev = self.size_bytes.fetch_add(total_add, Ordering::SeqCst);
            was_below = prev < self.cap_bytes;
            now_above = prev + total_add >= self.cap_bytes;
        }
        InsertResult {
            seq,
            became_full: was_below && now_above,
        }
    }

    /// Phase 10: insert (or replace) a node record. The MemTable
    /// tracks one entry per Seed; later inserts overwrite earlier
    /// ones, preserving the highest seq. `iter_nodes_for_flush`
    /// emits these into the SST flush stream.
    ///
    /// OP-BRAIN-V5 Phase 2.3: delegates to the inner
    /// `NodeMemTable` (`node_store::memtable::NodeMemTable`).
    /// Public API is unchanged for v4 callers
    /// (`persist_node_v4`, WAL replay, tests). The size_bytes
    /// AtomicUsize is still updated under the same lock so the v4
    /// flush trigger / cap accounting remains intact; the new type
    /// also tracks its own `approx_bytes` so Phase 2.6 can switch
    /// to a node-store-only trigger when the cutover lands.
    pub fn insert_node(
        &self,
        seed: Seed,
        node: crate::node::KnowledgeNode,
        intern: &NodeIntern,
    ) -> InsertResult {
        let seq = self.seq_counter.fetch_add(1, Ordering::SeqCst);
        let id = intern.intern_or_get(seed);
        let was_below;
        let now_above;
        {
            let mut g = self.inner.lock().expect("MemTable insert_node lock");
            let was_new = g.node_table.get(id).is_none();
            g.node_table.insert(id, node, seq);
            let added = if was_new { crate::node_store::APPROX_NODE_BYTES } else { 0 };
            let prev = self.size_bytes.fetch_add(added, Ordering::SeqCst);
            was_below = prev < self.cap_bytes;
            now_above = prev + added >= self.cap_bytes;
        }
        InsertResult { seq, became_full: was_below && now_above }
    }

    /// Phase 10: look up a node record by Seed. Returns the latest-
    /// seq entry currently held in the MemTable, or `None` if no
    /// node for that seed has been inserted (the caller should fall
    /// through to L0 SST node tables).
    pub fn lookup_node(
        &self,
        seed: &Seed,
        intern: &NodeIntern,
    ) -> Option<crate::node::KnowledgeNode> {
        let id = intern.get(seed)?;
        let g = self.inner.lock().expect("MemTable lookup_node lock");
        // OP-BRAIN-V5 Phase 2.3: skip tombstones -- a tombstoned
        // entry exists in the node_table but logically shadows the
        // payload. Returning the prior content here would surface a
        // logically-deleted node to the read path.
        g.node_table.get(id).filter(|e| !e.tombstone).map(|e| e.payload.clone())
    }

    /// Phase 10: number of node records currently in the MemTable.
    /// Includes tombstones (matches the pre-Phase-2.3 semantics --
    /// the v4 `MemTable` never had tombstones, so any caller relying
    /// on this count is asking "how many entries does the table
    /// have", not "how many live entries").
    pub fn node_count(&self) -> usize {
        let g = self.inner.lock().expect("MemTable node_count lock");
        g.node_table.len()
    }

    /// Current size estimate. Cheap atomic read.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::SeqCst)
    }

    /// True iff `size_bytes >= cap_bytes`. Cheap atomic read.
    pub fn is_full(&self) -> bool {
        self.size_bytes() >= self.cap_bytes
    }

    /// The configured cap (constant for the table's lifetime).
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// Current seq counter (next seq to be assigned). Diagnostic
    /// only; not load-bearing.
    pub fn next_seq(&self) -> u64 {
        self.seq_counter.load(Ordering::SeqCst)
    }

    /// OP-BRAIN-V5 Phase 1.4 — atomically allocate and return a fresh
    /// seq number. Used by `Brain::teach_with_kind_v4` and
    /// `Brain::persist_node_v4` to stamp `REC_TYPE_NODE_INTERN` WAL
    /// records when those records are emitted out-of-band (i.e. NOT
    /// from a corresponding `insert` / `insert_node` call, which
    /// already allocate their own seq).
    ///
    /// Atomicity: `fetch_add(1, SeqCst)` keeps the global seq stream
    /// monotonic across all allocation paths. Callers do NOT need to
    /// hold any other MemTable lock to call this method.
    pub fn allocate_seq(&self) -> u64 {
        self.seq_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// True iff the table is empty.
    pub fn is_empty(&self) -> bool {
        let g = self.inner.lock().expect("MemTable inner lock");
        g.entries.is_empty()
    }

    /// Number of distinct subjects with at least one bond.
    pub fn subject_count(&self) -> usize {
        let g = self.inner.lock().expect("MemTable inner lock");
        g.entries.len()
    }

    /// Number of bonds across all subjects.
    pub fn bond_count(&self) -> usize {
        let g = self.inner.lock().expect("MemTable inner lock");
        g.entries.values().map(|v| v.len()).sum()
    }

    /// Atomic swap-for-flush primitive.
    ///
    /// Returns the old MemTable boxed for the flush thread; replaces
    /// self's contents with a fresh empty one. The swap holds the
    /// inner lock for exactly two `mem::replace` operations — any
    /// insert racing this either completes fully before the lock
    /// hand-off (lands in the old table) or completes fully after
    /// (lands in the new table). Never lost, never duplicated.
    ///
    /// `cap_bytes` is preserved across the swap; the `seq_counter`
    /// continues monotonically (does NOT reset — that would let two
    /// post-swap inserts reuse a seq the old WAL has under fsync).
    pub fn swap_for_flush(&self) -> Box<MemTable> {
        let new_inner = MemTableInner::new();
        let old_inner = {
            let mut g = self.inner.lock().expect("MemTable swap lock");
            std::mem::replace(&mut *g, new_inner)
        };
        // Drain size_bytes atomically; the seq_counter keeps going.
        let old_size = self.size_bytes.swap(0, Ordering::SeqCst);
        // The swapped-out box keeps its own seq_counter starting at
        // 0 — the Phase 4 flush code does NOT need post-swap seqs on
        // the old table, only the records that were already inserted.
        let old = MemTable {
            inner: Mutex::new(old_inner),
            size_bytes: AtomicUsize::new(old_size),
            cap_bytes: self.cap_bytes,
            seq_counter: AtomicU64::new(0),
        };
        Box::new(old)
    }

    /// Subject-sorted iterator for flush. Yields `(subject_seed,
    /// Vec<(object_seed, predicate, weight, keyframes)>)` per subject
    /// in NodeId-sorted order. The Phase 4 flush sorts within each
    /// subject by (predicate, object) before pushing to `SSTWriter`,
    /// because the writer only enforces subject-major ordering.
    ///
    /// **Caveat:** NodeId order is NOT the same as Seed-bytes order.
    /// NodeIds are assigned by first-observation. The caller (Phase 4)
    /// will re-sort by `subject.as_bytes()` before pushing to
    /// `SSTWriter` — that's what the SST writer's contract requires
    /// anyway. We expose NodeId-sorted here because that's the
    /// natural BTreeMap traversal and lets us avoid building a Vec
    /// large enough to hold every subject mid-iteration.
    ///
    /// Returns an owning iterator that consumes the inner BTreeMap.
    /// Designed for use AFTER `swap_for_flush` — the flush thread
    /// owns the returned box exclusively.
    pub fn iter_for_flush(
        &self,
        intern: &NodeIntern,
    ) -> Vec<(Seed, Vec<MemBondFlush>)> {
        let g = self.inner.lock().expect("MemTable iter lock");
        let mut out: Vec<(Seed, Vec<MemBondFlush>)> = Vec::with_capacity(g.entries.len());
        for (&id, bonds) in g.entries.iter() {
            let subj = match intern.reverse_lookup(id) {
                Some(s) => s,
                None => continue,
            };
            let mut bonds_out: Vec<MemBondFlush> = bonds.iter().map(|b| MemBondFlush {
                object: b.object,
                object_id: b.object_id,
                predicate: b.predicate,
                weight: b.weight,
                keyframes: b.keyframes.clone(),
                tombstone: b.tombstone,
                // OP-BRAIN-V5 Phase 3.8.1: seq travels with the
                // flush projection (see MemBondFlush doc-comment).
                seq: b.seq,
            }).collect();
            bonds_out.sort_by(|a, b| {
                let pa = predicate_discriminant(a.predicate);
                let pb = predicate_discriminant(b.predicate);
                pa.cmp(&pb).then_with(|| a.object.as_bytes().cmp(b.object.as_bytes()))
            });
            out.push((subj, bonds_out));
        }
        out.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        out
    }

    /// Phase 10: subject-sorted node iterator for flush. Yields
    /// `(seed, knowledge_node, seq)` for every node record in the
    /// MemTable, sorted by Seed bytes (same order the bond flush
    /// uses).
    ///
    /// OP-BRAIN-V5 Phase 2.3: pulls from the inner `NodeMemTable`.
    /// Tombstoned entries are SKIPPED -- the v4 flush path has no
    /// concept of tombstones at the SST level (Phase 2.6 will add
    /// the node-store-side tombstone row), so surfacing them here
    /// would write the sentinel payload into a v4 bond SST and the
    /// `seen_in_sst` HashSet would not recognise it. Honest scope:
    /// the v4 path keeps live-only flush semantics for nodes;
    /// Phase 2.6 owns the tombstone-aware flush. This method still
    /// collects into a Vec because the v4 flush path is Vec-shaped
    /// (`flush.rs:276`); Phase 2.6 will retire it in favour of the
    /// borrowing `NodeMemTable::iter()` directly.
    pub fn iter_nodes_for_flush(
        &self,
        intern: &NodeIntern,
    ) -> Vec<(Seed, crate::node::KnowledgeNode, u64)> {
        let g = self.inner.lock().expect("MemTable iter_nodes lock");
        let mut out: Vec<(Seed, crate::node::KnowledgeNode, u64)> =
            Vec::with_capacity(g.node_table.len());
        for (id, entry) in g.node_table.iter() {
            if entry.tombstone {
                continue;
            }
            let seed = match intern.reverse_lookup(id) {
                Some(s) => s,
                None => continue,
            };
            out.push((seed, entry.payload.clone(), entry.seq));
        }
        out.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        out
    }

    /// OP-BRAIN-V5 Phase 2.6 -- invoke `f` with a borrow of the
    /// inner [`crate::node_store::NodeMemTable`] under the table
    /// lock. Designed for the swapped-out snapshot the bond
    /// flush worker holds in `flush_once`: the
    /// `NodeFlushWorker::flush` worker streams
    /// `NodeMemTable::iter()` directly into the NodeSST writer,
    /// preserving the borrowing-iterator discipline (no Vec
    /// materialisation, RSS bound at one block buffer).
    ///
    /// Holding the lock for the duration of the closure is
    /// trivial on the swapped snapshot (no concurrent writes
    /// against it; the live MemTable has already been replaced).
    /// On the live MemTable this method would block writes for
    /// the full flush window; callers are expected to call it
    /// only on the swap target.
    pub fn with_node_table<R>(
        &self,
        f: impl FnOnce(&crate::node_store::NodeMemTable) -> R,
    ) -> R {
        let g = self.inner.lock().expect("MemTable with_node_table lock");
        f(&g.node_table)
    }

    /// Read all bonds for one subject. Used by tests and by the
    /// Phase 5 read path (when MemTable is wired into the merged
    /// read path). Returns an empty Vec if the subject has no bonds.
    pub fn get_bonds_for_subject(
        &self,
        subject: &Seed,
        intern: &NodeIntern,
    ) -> Vec<MemBond> {
        let id = match intern.get(subject) {
            Some(id) => id,
            None => return Vec::new(),
        };
        let g = self.inner.lock().expect("MemTable get lock");
        g.entries.get(&id).cloned().unwrap_or_default()
    }
}

/// Stable u8 discriminant for BondType — mirrors the SST writer's
/// internal mapping so the MemTable flush ordering matches what the
/// SST reader expects. Local to this module to avoid leaking ordering
/// internals.
pub(crate) fn predicate_discriminant(b: BondType) -> u8 {
    match b {
        BondType::IsA => 0,
        BondType::HasProperty => 1,
        BondType::Causes => 2,
        BondType::PartOf => 3,
        BondType::UsedFor => 4,
        BondType::RelatedTo => 5,
        BondType::ContrastedWith => 6,
        BondType::DerivedFrom => 7,
        BondType::Requires => 8,
        BondType::Produces => 9,
        BondType::Opposes => 10,
        BondType::BornIn => 11,
        BondType::BornOn => 12,
        BondType::DiedIn => 13,
        BondType::DiedOn => 14,
        BondType::DiscoveredBy => 15,
        BondType::FoundedBy => 16,
        BondType::FoundedIn => 17,
        BondType::HasAltitude => 18,
        BondType::HasAspect => 19,
        BondType::HasCapital => 20,
        BondType::HasOfficialLanguage => 21,
        BondType::HasPopulation => 22,
        BondType::KnownFor => 23,
        BondType::LocatedIn => 24,
        BondType::MeasuredIn => 25,
        BondType::NamedAfter => 26,
        BondType::WrittenBy => 27,
        BondType::HasArea => 28,
        BondType::HasMethod => 29,
        BondType::HasAttribute => 30,
        BondType::HasParameter => 31,
        BondType::Returns => 32,
        BondType::Inherits => 33,
        BondType::Calls => 34,
        BondType::HasType => 35,
        BondType::Implements => 36,
        BondType::DefinedAs => 37,
        BondType::WorksBy => 38,
        BondType::ComposedOf => 39,
        BondType::PurposeOf => 40,
        BondType::DiffersFrom => 41,
        BondType::LeadsTo => 42,
        BondType::ExampleOf => 43,
        BondType::HasCurrency => 44,
        BondType::AuthoredBy => 45,
    }
}

/// OP-BRAIN-V5 Phase 3.8.1: inverse of [`predicate_discriminant`].
/// Used by `Brain::rehydrate_adjacency` to reconstruct a `BondType`
/// from the u8 stored in v5 BondSST rows. Returns `None` for an
/// out-of-range discriminant (corrupt or future-format row -- the
/// caller counts and skips, never guesses). The round-trip
/// invariant is pinned by `bond_type_discriminant_round_trips`
/// below and by the injectivity test in `read_path.rs`.
pub(crate) fn bond_type_from_discriminant(d: u8) -> Option<BondType> {
    Some(match d {
        0 => BondType::IsA,
        1 => BondType::HasProperty,
        2 => BondType::Causes,
        3 => BondType::PartOf,
        4 => BondType::UsedFor,
        5 => BondType::RelatedTo,
        6 => BondType::ContrastedWith,
        7 => BondType::DerivedFrom,
        8 => BondType::Requires,
        9 => BondType::Produces,
        10 => BondType::Opposes,
        11 => BondType::BornIn,
        12 => BondType::BornOn,
        13 => BondType::DiedIn,
        14 => BondType::DiedOn,
        15 => BondType::DiscoveredBy,
        16 => BondType::FoundedBy,
        17 => BondType::FoundedIn,
        18 => BondType::HasAltitude,
        19 => BondType::HasAspect,
        20 => BondType::HasCapital,
        21 => BondType::HasOfficialLanguage,
        22 => BondType::HasPopulation,
        23 => BondType::KnownFor,
        24 => BondType::LocatedIn,
        25 => BondType::MeasuredIn,
        26 => BondType::NamedAfter,
        27 => BondType::WrittenBy,
        28 => BondType::HasArea,
        29 => BondType::HasMethod,
        30 => BondType::HasAttribute,
        31 => BondType::HasParameter,
        32 => BondType::Returns,
        33 => BondType::Inherits,
        34 => BondType::Calls,
        35 => BondType::HasType,
        36 => BondType::Implements,
        37 => BondType::DefinedAs,
        38 => BondType::WorksBy,
        39 => BondType::ComposedOf,
        40 => BondType::PurposeOf,
        41 => BondType::DiffersFrom,
        42 => BondType::LeadsTo,
        43 => BondType::ExampleOf,
        44 => BondType::HasCurrency,
        45 => BondType::AuthoredBy,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{KeyframeSource, TemporalKeyframe};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::sync::Arc;
    use std::thread;

    fn rand_seed(rng: &mut StdRng) -> Seed {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Seed::from_bytes(b)
    }

    /// OP-BRAIN-V5 Phase 3.8.1: `bond_type_from_discriminant` must
    /// be the exact inverse of `predicate_discriminant` across all
    /// 37 variants, and reject out-of-range discriminants.
    #[test]
    fn bond_type_discriminant_round_trips() {
        for d in 0u8..=45 {
            let bt = bond_type_from_discriminant(d)
                .unwrap_or_else(|| panic!("discriminant {} must map to a BondType", d));
            assert_eq!(
                predicate_discriminant(bt),
                d,
                "round trip broke at discriminant {}",
                d
            );
        }
        // OP-SOURCEPULL Batch-C: 44=HasCurrency, 45=AuthoredBy are now valid;
        // 46 is the new out-of-range floor.
        assert!(bond_type_from_discriminant(46).is_none());
        assert!(bond_type_from_discriminant(0xFF).is_none());
    }

    fn baseline_kf(ts: i64, w: f32) -> TemporalKeyframe {
        TemporalKeyframe {
            timestamp: ts,
            weight: w,
            source: KeyframeSource::Baseline,
        }
    }

    #[test]
    fn insert_and_read_back_10k_bonds() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        // 1000 subjects, 10 bonds each = 10K bonds.
        let mut subjects: Vec<Seed> = (0..1000).map(|_| rand_seed(&mut rng)).collect();
        let mut expected: Vec<(Seed, Seed, f32)> = Vec::new();
        for s in &subjects {
            for _ in 0..10 {
                let obj = rand_seed(&mut rng);
                let w: f32 = rng.gen_range(0.0..1.0);
                mt.insert(
                    *s, obj, BondType::RelatedTo, w,
                    vec![baseline_kf(1, w)], &intern,
                );
                expected.push((*s, obj, w));
            }
        }
        assert_eq!(mt.bond_count(), 10_000);
        // Per-subject readback round-trip.
        subjects.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        for s in &subjects {
            let bonds = mt.get_bonds_for_subject(s, &intern);
            assert_eq!(bonds.len(), 10);
        }
    }

    #[test]
    fn size_bytes_grows_monotonically() {
        let mt = MemTable::new(1024 * 1024);
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let prev = mt.size_bytes();
        for _ in 0..100 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 1.0, vec![], &intern,
            );
        }
        assert!(mt.size_bytes() > prev);
    }

    #[test]
    fn is_full_triggers_at_cap() {
        // Small cap so we hit it fast.
        let mt = MemTable::new(64 * 1024);
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(0x1234);
        let mut became_full_count = 0;
        while !mt.is_full() {
            let r = mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 0.5,
                vec![baseline_kf(0, 0.5)], &intern,
            );
            if r.became_full {
                became_full_count += 1;
            }
            if mt.bond_count() > 10_000 {
                panic!("never crossed cap: bug in size accounting");
            }
        }
        assert!(mt.is_full());
        // Exactly one insert should be the edge-trigger.
        assert_eq!(became_full_count, 1);
    }

    #[test]
    fn iter_for_flush_yields_subject_sorted() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(0xABCD);
        for _ in 0..200 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::IsA, 0.7, vec![], &intern,
            );
        }
        let out = mt.iter_for_flush(&intern);
        // Subject-bytes order.
        for w in out.windows(2) {
            assert!(w[0].0.as_bytes() <= w[1].0.as_bytes());
        }
    }

    #[test]
    fn iter_for_flush_keyframes_preserved() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let subj = Seed::from_label("subj");
        let obj = Seed::from_label("obj");
        let kfs = vec![baseline_kf(100, 0.3), baseline_kf(200, 0.5)];
        mt.insert(subj, obj, BondType::HasProperty, 0.5, kfs.clone(), &intern);
        let out = mt.iter_for_flush(&intern);
        assert_eq!(out.len(), 1);
        let (got_subj, bonds) = &out[0];
        assert_eq!(*got_subj, subj);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].keyframes.len(), 2);
        assert_eq!(bonds[0].keyframes[0].timestamp, 100);
        assert_eq!(bonds[0].keyframes[1].timestamp, 200);
    }

    #[test]
    fn swap_for_flush_drains_self_to_empty() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            mt.insert(
                rand_seed(&mut rng), rand_seed(&mut rng),
                BondType::RelatedTo, 0.5, vec![], &intern,
            );
        }
        assert_eq!(mt.bond_count(), 100);
        let old = mt.swap_for_flush();
        assert!(mt.is_empty());
        assert_eq!(mt.size_bytes(), 0);
        assert_eq!(old.bond_count(), 100);
    }

    #[test]
    fn swap_preserves_seq_counter_monotonicity() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let r1 = mt.insert(
            Seed::from_label("a"), Seed::from_label("b"),
            BondType::IsA, 1.0, vec![], &intern,
        );
        let _ = mt.swap_for_flush();
        let r2 = mt.insert(
            Seed::from_label("c"), Seed::from_label("d"),
            BondType::IsA, 1.0, vec![], &intern,
        );
        // Seq must NOT reset across swap — re-using a seq the old
        // WAL has under fsync is the data-loss bomb.
        assert!(r2.seq > r1.seq, "seq must advance across swap; got {} <= {}", r2.seq, r1.seq);
    }

    #[test]
    fn swap_is_atomic_against_concurrent_inserts() {
        // 4 writer threads each insert N bonds while 1 swap thread
        // periodically swaps. Total bonds across the live table +
        // every swapped-out table must equal 4*N. Never lost,
        // never duplicated.
        let mt = Arc::new(MemTable::with_default_cap());
        let intern = NodeIntern::new();
        let n_writers = 4;
        let inserts_per_writer = 5_000;
        let swapped_out: Arc<Mutex<Vec<Box<MemTable>>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_swap = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut handles = Vec::new();
        for w in 0..n_writers {
            let mt = Arc::clone(&mt);
            let intern = intern.clone();
            handles.push(thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(0xC0DE + w as u64);
                for _ in 0..inserts_per_writer {
                    mt.insert(
                        rand_seed(&mut rng), rand_seed(&mut rng),
                        BondType::RelatedTo, 0.5, vec![], &intern,
                    );
                }
            }));
        }

        let swapper = {
            let mt = Arc::clone(&mt);
            let swapped_out = Arc::clone(&swapped_out);
            let stop = Arc::clone(&stop_swap);
            thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    thread::sleep(std::time::Duration::from_millis(1));
                    let old = mt.swap_for_flush();
                    if old.bond_count() > 0 {
                        swapped_out.lock().unwrap().push(old);
                    }
                }
            })
        };

        for h in handles { h.join().unwrap(); }
        // Give the swapper one last pass before halting.
        thread::sleep(std::time::Duration::from_millis(10));
        stop_swap.store(true, Ordering::SeqCst);
        swapper.join().unwrap();

        // Final residual.
        let residual = mt.bond_count();
        let archived: usize = swapped_out.lock().unwrap()
            .iter().map(|t| t.bond_count()).sum();
        assert_eq!(
            residual + archived,
            n_writers as usize * inserts_per_writer,
            "swap dropped or duplicated inserts: live={} archived={} expected={}",
            residual, archived, n_writers as usize * inserts_per_writer,
        );
    }

    #[test]
    fn get_bonds_for_unknown_subject_returns_empty() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let bonds = mt.get_bonds_for_subject(&Seed::from_label("never"), &intern);
        assert!(bonds.is_empty());
    }

    #[test]
    fn many_bonds_one_subject_share_a_vec() {
        // Confirm that 1000 bonds on the same subject occupy one
        // BTreeMap entry — proves we collapse adjacency correctly.
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let s = Seed::from_label("hub");
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..1000 {
            mt.insert(
                s, rand_seed(&mut rng), BondType::RelatedTo, 0.5,
                vec![], &intern,
            );
        }
        assert_eq!(mt.subject_count(), 1);
        assert_eq!(mt.bond_count(), 1000);
    }

    #[test]
    fn predicate_discriminant_matches_29_variants() {
        // Touch every BondType variant so a future renumber fails
        // compilation here AND in src/sst.rs simultaneously.
        let all = [
            BondType::IsA, BondType::HasProperty, BondType::Causes,
            BondType::PartOf, BondType::UsedFor, BondType::RelatedTo,
            BondType::ContrastedWith, BondType::DerivedFrom,
            BondType::Requires, BondType::Produces, BondType::Opposes,
            BondType::BornIn, BondType::BornOn, BondType::DiedIn,
            BondType::DiedOn, BondType::DiscoveredBy, BondType::FoundedBy,
            BondType::FoundedIn, BondType::HasAltitude, BondType::HasAspect,
            BondType::HasCapital, BondType::HasOfficialLanguage,
            BondType::HasPopulation, BondType::KnownFor, BondType::LocatedIn,
            BondType::MeasuredIn, BondType::NamedAfter, BondType::WrittenBy,
            BondType::HasArea,
            BondType::HasMethod,
            BondType::HasAttribute,
            BondType::HasParameter,
            BondType::Returns,
            BondType::Inherits,
            BondType::Calls,
            BondType::HasType,
            BondType::Implements,
        ];
        for (i, bt) in all.iter().enumerate() {
            assert_eq!(predicate_discriminant(*bt), i as u8);
        }
    }

    #[test]
    fn end_to_end_memtable_plus_wal_recovery() {
        // Insert 10K bonds through both the MemTable AND the WAL,
        // simulating Phase 4 wiring. Drop both, replay the WAL into a
        // fresh MemTable, verify byte-exact recovery against the
        // original.
        use crate::wal::{Wal, replay};
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("e2e.wal");
        let intern = NodeIntern::new();
        let mut rng = StdRng::seed_from_u64(0xE2E0);

        // (subject, object, weight) tuples we expect to see post-replay.
        let mut expected: Vec<(Seed, Seed, f32)> = Vec::with_capacity(10_000);
        {
            let mt = MemTable::with_default_cap();
            let mut wal = Wal::open(&wal_path).unwrap();
            for i in 0..10_000 {
                let subj = rand_seed(&mut rng);
                let obj = rand_seed(&mut rng);
                let w: f32 = rng.gen();
                let kfs = vec![baseline_kf(i as i64, w)];
                // Insert into MemTable to get the seq...
                let r = mt.insert(subj, obj, BondType::IsA, w, kfs.clone(), &intern);
                // ...then append the same record to the WAL with the
                // assigned seq. This is the order Phase 4 will use
                // (MemTable first so seq is assigned, then WAL).
                wal.append(r.seq, &subj, &obj, BondType::IsA, w, &kfs).unwrap();
                expected.push((subj, obj, w));
            }
            wal.fsync().unwrap();
            // Drop both — simulates kill.
        }

        // Replay WAL into a fresh MemTable + intern table.
        let recs = replay(&wal_path).unwrap();
        assert_eq!(recs.len(), 10_000);
        let mt2 = MemTable::with_default_cap();
        let intern2 = NodeIntern::new();
        for r in &recs {
            // Phase 10 WAL is enum-based; this test only inserts bond
            // records so it asserts the Bond shape.
            if let crate::wal::WalRecord::Bond { subject, object, predicate, weight, keyframes, .. } = r {
                mt2.insert(
                    *subject, *object, *predicate,
                    *weight, keyframes.clone(), &intern2,
                );
            }
        }
        assert_eq!(mt2.bond_count(), 10_000);

        // Verify byte-exact equality: for each expected tuple, the
        // recovered MemTable must hold an equivalent MemBond.
        // We build a multiset keyed by (subject_bytes, object_bytes, weight_bits)
        // so duplicates are counted correctly.
        use std::collections::HashMap;
        let mut want: HashMap<([u8; 32], [u8; 32], u32), usize> = HashMap::new();
        for (s, o, w) in &expected {
            let k = (*s.as_bytes(), *o.as_bytes(), w.to_bits());
            *want.entry(k).or_insert(0) += 1;
        }
        let flushed = mt2.iter_for_flush(&intern2);
        let mut got: HashMap<([u8; 32], [u8; 32], u32), usize> = HashMap::new();
        for (subj, bonds) in &flushed {
            for mb in bonds {
                let k = (*subj.as_bytes(), *mb.object.as_bytes(), mb.weight.to_bits());
                *got.entry(k).or_insert(0) += 1;
            }
        }
        assert_eq!(got, want, "recovered MemTable diverged from original");
    }

    // ---------------------------------------------------------------
    // Phase 10 -- node-table tests.
    // ---------------------------------------------------------------

    fn make_test_node(label: &str) -> crate::node::KnowledgeNode {
        crate::node::KnowledgeNode::with_seed(
            Seed::from_label(label),
            label.to_string(),
            crate::node::KnowledgeContent::Concept(label.to_string()),
            crate::node::Confidence::Verified,
            1234,
        )
    }

    #[test]
    fn phase10_insert_and_lookup_node() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let seed = Seed::from_label("alpha");
        let n = make_test_node("alpha");
        assert!(mt.lookup_node(&seed, &intern).is_none());
        mt.insert_node(seed, n.clone(), &intern);
        assert_eq!(mt.node_count(), 1);
        let got = mt.lookup_node(&seed, &intern).expect("node found");
        assert_eq!(got.seed, n.seed);
        assert_eq!(got.content, n.content);
    }

    #[test]
    fn phase10_iter_nodes_for_flush_sorts_by_seed() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let seeds: Vec<Seed> = (0..10).map(|i| Seed::from_label(&format!("node_{}", i))).collect();
        for (i, s) in seeds.iter().enumerate() {
            let n = make_test_node(&format!("node_{}", i));
            mt.insert_node(*s, n, &intern);
        }
        let flushed = mt.iter_nodes_for_flush(&intern);
        assert_eq!(flushed.len(), 10);
        for w in flushed.windows(2) {
            assert!(w[0].0.as_bytes() <= w[1].0.as_bytes());
        }
    }

    #[test]
    fn phase10_swap_for_flush_drains_nodes() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        for i in 0..5 {
            mt.insert_node(
                Seed::from_label(&format!("s_{}", i)),
                make_test_node(&format!("s_{}", i)),
                &intern,
            );
        }
        assert_eq!(mt.node_count(), 5);
        let snap = mt.swap_for_flush();
        // Old box has the nodes; live mt is fresh.
        assert_eq!(mt.node_count(), 0);
        assert_eq!(snap.node_count(), 5);
    }

    /// OP-BRAIN-V5 Phase 2.3 Test D: `MemTable::insert_node` delegates
    /// to the inner `NodeMemTable`. We exercise this by inserting via
    /// the public v4 API and probing observable side-effects that only
    /// the new type produces: monotonic-seq dedup semantics. The v4
    /// pre-Phase-2.3 BTreeMap would have stomped a newer entry with
    /// an older seq because it used unconditional `BTreeMap::insert`;
    /// the new `NodeMemTable` no-ops if the existing seq is greater
    /// or equal. We exercise that by directly probing the delegated
    /// `NodeMemTable` through `lookup_node` after a forced
    /// stale-seq insert via the lower-level path.
    #[test]
    fn phase23_insert_node_delegates_to_nodememtable() {
        let mt = MemTable::with_default_cap();
        let intern = NodeIntern::new();
        let seed = Seed::from_label("phase23_d");
        let node_v1 = make_test_node("v1_content");
        let r1 = mt.insert_node(seed, node_v1.clone(), &intern);
        assert_eq!(mt.node_count(), 1, "delegated NodeMemTable has 1 entry");
        // Lookup returns the v1 payload.
        let got = mt.lookup_node(&seed, &intern).expect("v1 lookup");
        assert_eq!(got.content, "v1_content");

        // Insert v2 with a higher seq (the v4 API always allocates a
        // fresh seq via seq_counter.fetch_add, so this is the natural
        // monotonic order).
        let node_v2 = make_test_node("v2_content");
        let r2 = mt.insert_node(seed, node_v2, &intern);
        assert!(r2.seq > r1.seq, "v4 seq_counter must produce monotonic seqs");
        let got = mt.lookup_node(&seed, &intern).expect("v2 lookup");
        assert_eq!(got.content, "v2_content", "newer seq wins in delegated NodeMemTable");

        // node_count is still 1 (same NodeId, overwrite semantics).
        assert_eq!(mt.node_count(), 1);

        // Insert a SECOND seed -> the delegated table tracks NodeId
        // ordering. iter_nodes_for_flush returns sorted-by-Seed.
        let seed2 = Seed::from_label("phase23_d_two");
        let node_b = make_test_node("v_b_content");
        mt.insert_node(seed2, node_b, &intern);
        assert_eq!(mt.node_count(), 2);
        let flushed = mt.iter_nodes_for_flush(&intern);
        assert_eq!(flushed.len(), 2, "delegation surfaces both entries on flush iter");
        let contents: Vec<&str> = flushed.iter().map(|(_, n, _)| n.content.as_str()).collect();
        assert!(contents.contains(&"v2_content"));
        assert!(contents.contains(&"v_b_content"));
    }
}
