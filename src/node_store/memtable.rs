//! `NodeMemTable` -- primary in-memory node store, sorted by `NodeId`.
//!
//! OP-BRAIN-V5 Phase 2.3 (Q3 of the design doc, audit S10 sub-port).
//!
//! ## Why a separate type from the v4 `MemTable`
//!
//! The v4 `MemTable` was a single inner state carrying both bonds
//! (`BTreeMap<NodeId, Vec<MemBond>>`) and nodes
//! (`BTreeMap<NodeId, (KnowledgeNode, u64)>`). Phase 2.3 splits the
//! node half out so the node store can evolve independently of the
//! bond store: the bond half stays on `MemTable` (Phase 3 owns its
//! rewrite), the node half lives in `NodeMemTable` from now on.
//!
//! `MemTable::insert_node` / `lookup_node` / `node_count` /
//! `iter_nodes_for_flush` are kept as public API; they delegate to
//! a private `NodeMemTable` field on `MemTableInner` so v4 callers
//! (Phase 1.4 `persist_node_v4`, the WAL replay loop in
//! `Brain::enable_memtable`, and tests) are unchanged.
//!
//! ## Sort order
//!
//! `BTreeMap<NodeId, NodeEntry>` -- sorted by `NodeId` ascending.
//! The audit (S2.2 / S10) confirmed `NodeId` is the only structural
//! sort dimension: `KnowledgeContent.kind` is a tag, and node
//! creation is idempotent on `Seed`, which interns to a stable
//! `NodeId`. Phase 2.6 flush worker streams `iter()` in this order
//! directly into the node-store SST writer -- no re-sort.
//!
//! ## Tombstone semantics (Q3)
//!
//! `NodeEntry { tombstone: bool, seq: u64, payload: KnowledgeNode }`.
//! Higher `seq` wins on dedup. Inserts overwrite a previous record
//! only if their `seq` is strictly greater -- this matches the v4
//! `seq_counter.fetch_add` monotonicity, and means out-of-order
//! inserts (theoretical race) cannot stomp a newer record. The
//! `tombstone` method writes a sentinel entry whose `payload` is
//! preserved from any prior insert (the audit deliberately does NOT
//! discard payload on tombstone -- the read-merge path may still
//! need it for backtracking, and the byte cost is bounded by the
//! same `APPROX_NODE_BYTES` cap).
//!
//! ## `approx_bytes` tracking
//!
//! Mirrors the v4 `MemTable::size_bytes` discipline: every insert
//! adds `APPROX_NODE_BYTES` (1800 -- audit S2.1 + the constant
//! already in `memtable.rs`), every tombstone of a NEW key adds the
//! same. The cost is exposed via `approx_bytes()` for the eventual
//! Phase 2.6 flush-trigger threshold; v4 callers continue to use
//! the v4 `MemTable`'s shared `size_bytes` AtomicUsize for the live
//! flush trigger -- Phase 2.6 owns the cutover.
//!
//! ## Streaming discipline
//!
//! `iter()` returns a borrowing iterator over the inner `BTreeMap`
//! -- it does NOT materialise into a `Vec`. The Phase 2.6 flush
//! worker is required to consume it without collecting; this commit
//! preserves the property even at the v4 callsite
//! (`MemTable::iter_nodes_for_flush` still collects because the v4
//! flush path is `Vec`-shaped and Phase 2.6 will retire it; see
//! `MemTable::iter_nodes_for_flush` docstring).

use std::collections::BTreeMap;

use crate::node::KnowledgeNode;
use crate::node_intern::NodeId;

/// Approximate resident bytes per node entry. Matches the v4
/// constant -- ~1.8 KiB per node, dominated by the 384-dim
/// embedding vector (`Vec<f32>` = 1536 B + 24 B header). The audit
/// S2.1 confirmed this number.
pub const APPROX_NODE_BYTES: usize = 1800;

/// One entry in [`NodeMemTable`]. Carries the payload, a tombstone
/// flag (Q3), and a monotonic `seq` for dedup on race / replay.
///
/// `Clone` so consumers (flush worker, read path) can take owned
/// copies without holding the table lock. The payload is the
/// dominant clone cost (~1.8 KiB / embedding); callers that don't
/// need an owned copy should use `iter()` borrows.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeEntry {
    /// The `KnowledgeNode` payload itself. Held by value (not `Arc`)
    /// because the v4 callsite (`MemTable::insert_node`) already
    /// owns by value, and an `Arc` would force callers through one
    /// extra indirection without ownership gain (every consumer
    /// today wants the owned `KnowledgeNode`).
    pub payload: KnowledgeNode,
    /// `true` iff this entry is a tombstone -- a deletion marker
    /// that shadows older same-`NodeId` inserts in the merge layer.
    /// The payload is preserved across tombstoning (per the audit
    /// S2.2 "idempotent node creation" note) so read-merge can still
    /// surface the last-known content for diagnostics.
    pub tombstone: bool,
    /// Monotonic sequence number from the parent `MemTable`'s
    /// shared counter. Higher seq wins on dedup.
    pub seq: u64,
}

/// In-memory node store -- sorted by `NodeId` ascending.
///
/// Not `Sync`: callers wrap it in the parent `MemTable`'s
/// `Mutex<MemTableInner>`. The Phase 2.6 node flush worker will
/// hold its own lock on a swap-for-flush primitive; for the Phase
/// 2.3 commit the discipline is "share the v4 lock", since the
/// existing `MemTable::insert_node` callers already acquire it.
#[derive(Debug)]
pub struct NodeMemTable {
    entries: BTreeMap<NodeId, NodeEntry>,
    approx_bytes: usize,
}

impl Default for NodeMemTable {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeMemTable {
    /// Create a new empty node memtable.
    pub fn new() -> Self {
        NodeMemTable {
            entries: BTreeMap::new(),
            approx_bytes: 0,
        }
    }

    /// Insert (or replace) a node entry at `id`.
    ///
    /// - If `id` is not yet in the table: inserts the entry,
    ///   `approx_bytes += APPROX_NODE_BYTES`.
    /// - If `id` is already present with a `seq` strictly less than
    ///   the new one: overwrites the entry. `approx_bytes` is
    ///   unchanged (one entry replaces one entry -- the audit S2.1
    ///   cap is per-key, not per-write).
    /// - If `id` is already present with `seq` >= the new one:
    ///   no-op (a later record already won; dropping the older one
    ///   preserves the highest-seq-wins invariant). This case
    ///   shouldn't happen in single-writer v4 production but
    ///   discipline-wise it is the right behaviour -- a future
    ///   crash-replay path that re-inserts older records must not
    ///   stomp newer in-memory state.
    ///
    /// `tombstone` is cleared by `insert` -- live data supersedes a
    /// prior tombstone at the same `NodeId` (same write-wins-on-seq
    /// discipline).
    pub fn insert(&mut self, id: NodeId, node: KnowledgeNode, seq: u64) {
        match self.entries.get(&id) {
            Some(existing) if existing.seq >= seq => {
                // Older write loses. No-op.
            }
            Some(_) => {
                // Newer write replaces. approx_bytes unchanged.
                self.entries.insert(id, NodeEntry { payload: node, tombstone: false, seq });
            }
            None => {
                self.entries.insert(id, NodeEntry { payload: node, tombstone: false, seq });
                self.approx_bytes = self.approx_bytes.saturating_add(APPROX_NODE_BYTES);
            }
        }
    }

    /// Mark `id` as a tombstone.
    ///
    /// - If `id` is not in the table: inserts a tombstone entry
    ///   with a sentinel `KnowledgeNode` payload (constructed via
    ///   the smallest legal node). `approx_bytes += APPROX_NODE_BYTES`.
    /// - If `id` is present with `seq < new_seq`: marks tombstone
    ///   on the existing entry, preserves the prior payload, bumps
    ///   the `seq`.
    /// - If `id` is present with `seq >= new_seq`: no-op.
    pub fn tombstone(&mut self, id: NodeId, seq: u64) {
        match self.entries.get_mut(&id) {
            Some(existing) if existing.seq >= seq => {
                // Older tombstone loses.
            }
            Some(existing) => {
                existing.tombstone = true;
                existing.seq = seq;
            }
            None => {
                // No prior payload to preserve -- materialise a
                // sentinel KnowledgeNode. The payload bytes here
                // are NEVER surfaced by the read path (tombstone
                // shadows all reads at this NodeId); they exist
                // purely so the entry shape is uniform.
                let sentinel = sentinel_node();
                self.entries.insert(
                    id,
                    NodeEntry { payload: sentinel, tombstone: true, seq },
                );
                self.approx_bytes = self.approx_bytes.saturating_add(APPROX_NODE_BYTES);
            }
        }
    }

    /// Read-path probe. Returns the entry for `id` if present (live
    /// or tombstoned -- the caller inspects `tombstone` to decide
    /// whether to surface the payload or treat as absent).
    pub fn get(&self, id: NodeId) -> Option<&NodeEntry> {
        self.entries.get(&id)
    }

    /// Sorted-by-NodeId iterator for flush. Borrows -- does NOT
    /// materialise into a `Vec`. The eventual Phase 2.6 flush
    /// worker streams this directly into the node-store SST writer.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &NodeEntry)> {
        self.entries.iter().map(|(id, e)| (*id, e))
    }

    /// Approximate resident byte cost. Feeds the eventual Phase 2.6
    /// flush-trigger threshold.
    pub fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    /// Number of entries (live + tombstones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff there are zero entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Construct the smallest legal `KnowledgeNode` for use as a
/// tombstone-with-no-prior-payload sentinel. The bytes here are
/// NEVER surfaced by the read path -- tombstone shadows the lookup.
///
/// We materialise via explicit field construction (rather than a
/// `Default` impl -- `KnowledgeNode` has fields like `Seed` and
/// `KnowledgeUnit` whose `Default` semantics would be questionable)
/// so the bytes are pinned and reviewable.
fn sentinel_node() -> KnowledgeNode {
    use crate::node::{Confidence, KnowledgeContent, KnowledgeUnit};
    use crate::seed::Seed;
    KnowledgeNode {
        seed: Seed::from_bytes([0u8; 32]),
        content: String::new(),
        knowledge: KnowledgeUnit {
            content: KnowledgeContent::Concept(String::new()),
            confidence: Confidence::Verified,
            created_at: 0,
            updated_at: 0,
        },
        embedding: Vec::new(),
        parent_seeds: Vec::new(),
        typed_ratio: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeUnit};
    use crate::seed::Seed;

    fn mk_node(label: &str) -> KnowledgeNode {
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: label.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(label.to_string()),
                confidence: Confidence::Verified,
                created_at: 0,
                updated_at: 0,
            },
            embedding: vec![0.0; 384],
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    /// Test A: insert 100 nodes with NON-monotonic NodeIds; iter()
    /// returns them sorted ascending.
    #[test]
    fn nodememtable_insert_and_sorted_iter() {
        let mut t = NodeMemTable::new();
        // Reverse-and-shuffled NodeId order so the BTreeMap actually
        // has to sort.
        let mut ids: Vec<u64> = (0..100).rev().collect();
        ids.swap(0, 50);
        ids.swap(10, 90);
        ids.swap(33, 77);
        for (seq, id) in ids.iter().enumerate() {
            t.insert(NodeId(*id), mk_node(&format!("n_{}", id)), seq as u64);
        }
        assert_eq!(t.len(), 100);
        let observed: Vec<u64> = t.iter().map(|(id, _)| id.0).collect();
        let expected: Vec<u64> = (0..100u64).collect();
        assert_eq!(observed, expected, "iter() must return NodeId-ascending order");
    }

    /// Test B: tombstone semantics. Insert -> tombstone -> get()
    /// returns Some with tombstone=true. Higher seq wins.
    #[test]
    fn nodememtable_tombstone_seq_semantics() {
        let mut t = NodeMemTable::new();
        let id = NodeId(42);
        // Insert at seq=5.
        t.insert(id, mk_node("alive"), 5);
        let e = t.get(id).expect("entry exists after insert");
        assert!(!e.tombstone);
        assert_eq!(e.seq, 5);

        // Tombstone at seq=10 (newer) -> wins, payload preserved.
        t.tombstone(id, 10);
        let e = t.get(id).expect("entry exists after tombstone");
        assert!(e.tombstone, "tombstone flag must be set");
        assert_eq!(e.seq, 10);
        assert_eq!(e.payload.content, "alive", "prior payload preserved across tombstone");

        // Tombstone at seq=3 (older) -> loses, no-op.
        t.tombstone(id, 3);
        let e = t.get(id).expect("entry exists");
        assert!(e.tombstone, "older tombstone must not unset the newer one");
        assert_eq!(e.seq, 10);

        // Insert at seq=20 (newer than the tombstone) -> live wins.
        t.insert(id, mk_node("resurrected"), 20);
        let e = t.get(id).expect("entry exists");
        assert!(!e.tombstone, "newer live insert must clear tombstone");
        assert_eq!(e.seq, 20);
        assert_eq!(e.payload.content, "resurrected");

        // Tombstone of an unknown NodeId at seq=1: inserts a
        // sentinel-payload tombstone.
        t.tombstone(NodeId(999), 1);
        let e = t.get(NodeId(999)).expect("sentinel tombstone exists");
        assert!(e.tombstone);
        assert_eq!(e.seq, 1);
        // Sentinel content is empty.
        assert_eq!(e.payload.content, "");
    }

    /// Test C: approx_bytes tracking. After N inserts at distinct
    /// NodeIds, approx_bytes == N * APPROX_NODE_BYTES.
    #[test]
    fn nodememtable_approx_bytes_tracking() {
        let mut t = NodeMemTable::new();
        assert_eq!(t.approx_bytes(), 0);
        const N: usize = 500;
        for i in 0..N {
            t.insert(NodeId(i as u64), mk_node(&format!("n_{}", i)), i as u64);
        }
        assert_eq!(
            t.approx_bytes(),
            N * APPROX_NODE_BYTES,
            "approx_bytes must track N * APPROX_NODE_BYTES exactly",
        );
        // Inserting at an already-present NodeId (newer seq)
        // overwrites -- approx_bytes unchanged.
        t.insert(NodeId(0), mk_node("n_0_v2"), (N + 1) as u64);
        assert_eq!(t.approx_bytes(), N * APPROX_NODE_BYTES);
        // Tombstoning a NEW NodeId adds APPROX_NODE_BYTES.
        t.tombstone(NodeId(N as u64 + 1), (N + 2) as u64);
        assert_eq!(t.approx_bytes(), (N + 1) * APPROX_NODE_BYTES);
        // Tombstoning an EXISTING NodeId does NOT add bytes (entry
        // already accounted for).
        t.tombstone(NodeId(0), (N + 3) as u64);
        assert_eq!(t.approx_bytes(), (N + 1) * APPROX_NODE_BYTES);
    }

    /// Insert with stale seq must NOT stomp the newer entry.
    #[test]
    fn nodememtable_insert_with_stale_seq_no_stomp() {
        let mut t = NodeMemTable::new();
        let id = NodeId(7);
        t.insert(id, mk_node("new"), 100);
        t.insert(id, mk_node("old"), 50);
        let e = t.get(id).unwrap();
        assert_eq!(e.seq, 100);
        assert_eq!(e.payload.content, "new", "newer seq must win");
    }
}
