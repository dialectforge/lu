//! Storage v2 Slice 2b/2c — abstract views over network entities.
//!
//! `AssociativeNetwork` historically returned bare references into its
//! internal `HashMap<Seed, KnowledgeNode>` and `HashMap<(Seed, Seed),
//! TemporalBond>`. That works as long as nodes and bonds live on the
//! Rust heap. Slice 2c (this revision) introduces a path where
//! nodes/bonds live inside a memory-mapped, rkyv-archived blob
//! (`ArchivedKnowledgeNode` / `ArchivedTemporalBond`); at that point
//! `&KnowledgeNode` is the wrong return type — the bytes on disk are
//! a different layout and a different Rust type.
//!
//! This module defines abstract VIEW types that callers use as
//! lightweight wrappers over EITHER a heap reference OR a reference
//! into the rkyv-archived blob. The public API hands out
//! `NodeView<'_>` / `BondView<'_>` / `BondsView<'_>` so the
//! underlying storage can change without breaking callers.
//!
//! Mut handles (`NodeMutHandle` / `BondMutHandle`) only ever wrap
//! heap-resident data. The CoW promotion that turns an archived node
//! into a heap copy happens INSIDE `AssociativeNetwork::get_node_mut`
//! / `get_bond_mut`, so by the time the caller holds a `*MutHandle`,
//! the storage is heap and DerefMut works exactly like before.

use crate::archive::{
    bond_type_from_archived, confidence_from_archived, distinct_session_count_archived,
    knowledge_content_text, latest_raw_weight_archived, seed_from_archived,
    temporal_keyframe_from_archived, weight_at_archived, weight_in_scope_archived,
    BondArchive, NetworkArchive, NodeArchive,
};
use crate::bond::{
    ArchivedTemporalBond, BondType, KeyframeSource, TemporalBond, TemporalKeyframe,
};
use crate::node::{
    ArchivedKnowledgeNode, Confidence, KnowledgeNode, KnowledgeUnit,
};
use crate::scar::{ScarAwareReading, ScarLog};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use std::collections::{HashMap, HashSet};

// ---------- NodeView ----------

/// Abstract view over a node in the network. Slice 2c added the
/// `Archive` variant backing the read path on mmap'd v3 brain files.
/// Callers should reach for the typed accessor methods on this type
/// rather than reaching for `KnowledgeNode` fields directly so the
/// archive variant is automatically supported.
#[derive(Clone)]
pub struct NodeView<'a> {
    inner: NodeViewInner<'a>,
}

#[derive(Clone)]
enum NodeViewInner<'a> {
    /// Backed by a heap-resident node.
    Heap(&'a KnowledgeNode),
    /// Backed by an `archive::NodeArchive`: either a borrowed ref into
    /// an mmap'd `ArchivedKnowledgeNode` (uncompressed) or an
    /// owning `ArcArchive` that holds an Arc clone of a decompressed
    /// block (OP-016 compressed bounded-LRU).
    Archive(NodeArchive<'a>),
}

impl<'a> NodeView<'a> {
    /// Wrap a heap-resident node.
    pub fn from_heap(node: &'a KnowledgeNode) -> Self {
        NodeView { inner: NodeViewInner::Heap(node) }
    }

    /// Wrap an archive-resident node (lives inside an mmap'd v3 file).
    /// OP-016: the inner carrier is now `NodeArchive<'a>::Borrowed`,
    /// keeping API-compat with existing callers that pass a bare ref.
    pub fn from_archive(node: &'a ArchivedKnowledgeNode) -> Self {
        NodeView { inner: NodeViewInner::Archive(NodeArchive::Borrowed(node)) }
    }

    /// OP-016: wrap a `NodeArchive<'a>` (the unified handle returned
    /// by `NetworkArchive::node_by_seed`). Used by callers that get
    /// the archive handle directly and want a NodeView.
    pub fn from_node_archive(na: NodeArchive<'a>) -> Self {
        NodeView { inner: NodeViewInner::Archive(na) }
    }

    /// True if this view is backed by an archived node.
    pub fn is_archive(&self) -> bool {
        matches!(&self.inner, NodeViewInner::Archive(_))
    }

    /// The node's seed identity.
    pub fn seed(&self) -> Seed {
        match &self.inner {
            NodeViewInner::Heap(n) => n.seed,
            NodeViewInner::Archive(n) => seed_from_archived(&n.as_archived().seed),
        }
    }

    /// The node's human-readable content / label string.
    pub fn content(&self) -> &str {
        match &self.inner {
            NodeViewInner::Heap(n) => n.content.as_str(),
            NodeViewInner::Archive(n) => n.as_archived().content.as_str(),
        }
    }

    /// The structured knowledge unit. Returns `None` for archive-backed
    /// views — callers should use `confidence()`, `created_at()`,
    /// `updated_at()`, and `knowledge_text()` instead. Heap-only paths
    /// (Thinking-Layer goal nodes etc.) can still match on the inner
    /// `KnowledgeContent` via `as_heap()`.
    pub fn knowledge(&self) -> Option<&KnowledgeUnit> {
        match &self.inner {
            NodeViewInner::Heap(n) => Some(&n.knowledge),
            NodeViewInner::Archive(_) => None,
        }
    }

    /// The confidence level on this node.
    pub fn confidence(&self) -> Confidence {
        match &self.inner {
            NodeViewInner::Heap(n) => n.knowledge.confidence,
            NodeViewInner::Archive(n) => confidence_from_archived(&n.as_archived().knowledge.confidence),
        }
    }

    /// Knowledge content's inner text — works for both Heap and
    /// Archive backings without allocating.
    pub fn knowledge_text(&self) -> &str {
        match &self.inner {
            NodeViewInner::Heap(n) => n.knowledge.content.text(),
            NodeViewInner::Archive(n) => knowledge_content_text(&n.as_archived().knowledge.content),
        }
    }

    /// Q.1 — KnowledgeContent variant tag ("definition", "mechanism",
    /// ...). Allocation-free for both Heap and Archive backings.
    pub fn kind_str(&self) -> &'static str {
        match &self.inner {
            NodeViewInner::Heap(n) => n.knowledge.content.kind_str(),
            NodeViewInner::Archive(n) => {
                crate::archive::archived_knowledge_kind_str(&n.as_archived().knowledge.content)
            }
        }
    }

    /// Q.1 — true if the variant should be surfaced as a (kind) tag
    /// in narration prose. Mirrors `KnowledgeContent::is_narration_tagged`.
    pub fn is_narration_tagged(&self) -> bool {
        let k = self.kind_str();
        !matches!(k, "raw" | "concept" | "meta")
    }

    /// Created-at timestamp on the node's knowledge unit.
    pub fn created_at(&self) -> i64 {
        match &self.inner {
            NodeViewInner::Heap(n) => n.knowledge.created_at,
            NodeViewInner::Archive(n) => n.as_archived().knowledge.created_at.into(),
        }
    }

    /// Updated-at timestamp on the node's knowledge unit.
    pub fn updated_at(&self) -> i64 {
        match &self.inner {
            NodeViewInner::Heap(n) => n.knowledge.updated_at,
            NodeViewInner::Archive(n) => n.as_archived().knowledge.updated_at.into(),
        }
    }

    /// 384-dimensional embedding vector. Allocation-free for both
    /// backings — heap returns a slice into the owned `Vec<f32>`,
    /// archive returns a slice into the mmap'd bytes.
    pub fn embedding(&self) -> &[f32] {
        match &self.inner {
            NodeViewInner::Heap(n) => n.embedding.as_slice(),
            NodeViewInner::Archive(n) => n.as_archived().embedding.as_slice(),
        }
    }

    /// Optional parent seeds (Thinking Layer step nodes point at their
    /// goal node). For archive-backed views, this materializes a
    /// short-lived `Vec<Seed>` because `ArchivedSeed` and `Seed` cannot
    /// safely share a slice across the rkyv ABI boundary. The archive
    /// parent_seeds list is typically 0 or 1 elements, so the
    /// allocation is negligible.
    pub fn parent_seeds(&self) -> std::borrow::Cow<'_, [Seed]> {
        match &self.inner {
            NodeViewInner::Heap(n) => std::borrow::Cow::Borrowed(n.parent_seeds.as_slice()),
            NodeViewInner::Archive(n) => {
                let v: Vec<Seed> = n
                    .as_archived()
                    .parent_seeds
                    .iter()
                    .map(seed_from_archived)
                    .collect();
                std::borrow::Cow::Owned(v)
            }
        }
    }

    /// Cached typed-bond ratio used by the narration HubSelector.
    pub fn typed_ratio(&self) -> f32 {
        match &self.inner {
            NodeViewInner::Heap(n) => n.typed_ratio,
            NodeViewInner::Archive(n) => n.as_archived().typed_ratio.into(),
        }
    }

    /// Cosine similarity with another node's embedding. Mirrors
    /// `KnowledgeNode::cosine_similarity` but works for any
    /// combination of Heap/Archive views.
    pub fn cosine_similarity(&self, other: &NodeView<'_>) -> f32 {
        let a = self.embedding();
        let b = other.embedding();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Escape hatch — borrow the raw heap node. Returns `None` for an
    /// archive-backed view.
    pub fn as_heap(&self) -> Option<&KnowledgeNode> {
        match &self.inner {
            NodeViewInner::Heap(n) => Some(n),
            NodeViewInner::Archive(_) => None,
        }
    }

    /// Escape hatch — borrow the raw archived node. Returns `None`
    /// for heap-backed views. The returned reference's lifetime is
    /// tied to `&self` (OP-016: the underlying Arc keeps the bytes
    /// alive for the lifetime of the NodeView).
    pub fn as_archive(&self) -> Option<&ArchivedKnowledgeNode> {
        match &self.inner {
            NodeViewInner::Heap(_) => None,
            NodeViewInner::Archive(n) => Some(n.as_archived()),
        }
    }
}

// ---------- BondView ----------

/// Abstract view over a bond in the network. Same Heap/Archive split
/// as `NodeView`.
#[derive(Clone)]
pub struct BondView<'a> {
    inner: BondViewInner<'a>,
}

#[derive(Clone)]
enum BondViewInner<'a> {
    Heap(&'a TemporalBond),
    /// See `NodeViewInner::Archive` for the OP-016 carrier rationale.
    Archive(BondArchive<'a>),
}

impl<'a> BondView<'a> {
    pub fn from_heap(bond: &'a TemporalBond) -> Self {
        BondView { inner: BondViewInner::Heap(bond) }
    }

    pub fn from_archive(bond: &'a ArchivedTemporalBond) -> Self {
        BondView { inner: BondViewInner::Archive(BondArchive::Borrowed(bond)) }
    }

    /// OP-016: wrap a `BondArchive<'a>` (the unified handle returned
    /// by `NetworkArchive::bond_by_pair`).
    pub fn from_bond_archive(ba: BondArchive<'a>) -> Self {
        BondView { inner: BondViewInner::Archive(ba) }
    }

    pub fn is_archive(&self) -> bool {
        matches!(&self.inner, BondViewInner::Archive(_))
    }

    pub fn from(&self) -> Seed {
        match &self.inner {
            BondViewInner::Heap(b) => b.from,
            BondViewInner::Archive(b) => seed_from_archived(&b.as_archived().from),
        }
    }

    pub fn to(&self) -> Seed {
        match &self.inner {
            BondViewInner::Heap(b) => b.to,
            BondViewInner::Archive(b) => seed_from_archived(&b.as_archived().to),
        }
    }

    pub fn bond_type(&self) -> BondType {
        match &self.inner {
            BondViewInner::Heap(b) => b.bond_type,
            BondViewInner::Archive(b) => bond_type_from_archived(&b.as_archived().bond_type),
        }
    }

    /// Number of keyframes on this bond.
    pub fn keyframe_count(&self) -> usize {
        match &self.inner {
            BondViewInner::Heap(b) => b.keyframes.len(),
            BondViewInner::Archive(b) => b.as_archived().keyframes.len(),
        }
    }

    /// Borrow keyframes as a `&[TemporalKeyframe]` if heap-backed;
    /// otherwise `None`. Use `keyframes_iter()` for archive-aware code.
    pub fn keyframes_heap(&self) -> Option<&[TemporalKeyframe]> {
        match &self.inner {
            BondViewInner::Heap(b) => Some(b.keyframes.as_slice()),
            BondViewInner::Archive(_) => None,
        }
    }

    /// Convenience accessor that returns `&[TemporalKeyframe]` when
    /// the bond is heap-backed, or panics otherwise. Used by legacy
    /// call sites guaranteed to be heap-resident at the call site.
    /// Prefer `keyframes_iter()` for archive-aware code.
    pub fn keyframes(&self) -> &[TemporalKeyframe] {
        match &self.inner {
            BondViewInner::Heap(b) => b.keyframes.as_slice(),
            BondViewInner::Archive(_) => {
                panic!("BondView::keyframes() called on archive-backed bond — use keyframes_iter()")
            }
        }
    }

    /// Iterate keyframes uniformly — yields owned `TemporalKeyframe`
    /// per item. For archive-backed bonds each keyframe is
    /// materialized fresh; bonds typically carry 1-3 keyframes so the
    /// allocation cost is small. For heap-backed bonds each item is a
    /// clone.
    pub fn keyframes_iter(&self) -> Box<dyn Iterator<Item = TemporalKeyframe> + '_> {
        match &self.inner {
            BondViewInner::Heap(b) => Box::new(b.keyframes.iter().cloned()),
            BondViewInner::Archive(b) => {
                // Materialise an owned Vec so the iterator outlives the
                // BondView's borrow (the BondArchive may hold an Arc).
                let kfs: Vec<TemporalKeyframe> = b
                    .as_archived()
                    .keyframes
                    .iter()
                    .map(temporal_keyframe_from_archived)
                    .collect();
                Box::new(kfs.into_iter())
            }
        }
    }

    pub fn half_life_ms(&self) -> f64 {
        match &self.inner {
            BondViewInner::Heap(b) => b.half_life_ms,
            BondViewInner::Archive(b) => b.as_archived().half_life_ms.into(),
        }
    }

    /// Iterator over keyframe sources (owned). Always allocates per
    /// item; for tests / introspection only.
    pub fn keyframe_sources(&self) -> Box<dyn Iterator<Item = KeyframeSource> + '_> {
        match &self.inner {
            BondViewInner::Heap(b) => Box::new(b.keyframes.iter().map(|kf| kf.source.clone())),
            BondViewInner::Archive(b) => {
                let srcs: Vec<KeyframeSource> = b
                    .as_archived()
                    .keyframes
                    .iter()
                    .map(|kf| crate::archive::keyframe_source_from_archived(&kf.source))
                    .collect();
                Box::new(srcs.into_iter())
            }
        }
    }

    /// Scar-aware weight read at time `t`. Dispatches to the heap or
    /// archive implementation; both apply the scar overlay anchored
    /// to wall-clock NOW.
    pub fn weight_at(&self, t: i64, scar_log: &ScarLog) -> ScarAwareReading {
        match &self.inner {
            BondViewInner::Heap(b) => b.weight_at(t, scar_log),
            BondViewInner::Archive(b) => weight_at_archived(b.as_archived(), t, scar_log),
        }
    }

    /// Scar-aware weight read under a temporal scope.
    pub fn weight_in_scope(&self, scope: &TemporalScope, scar_log: &ScarLog) -> ScarAwareReading {
        match &self.inner {
            BondViewInner::Heap(b) => b.weight_in_scope(scope, scar_log),
            BondViewInner::Archive(b) => weight_in_scope_archived(b.as_archived(), scope, scar_log),
        }
    }

    /// Latest raw keyframe weight (no decay, no bond-type multiplier).
    pub fn latest_raw_weight(&self) -> f32 {
        match &self.inner {
            BondViewInner::Heap(b) => b.latest_raw_weight(),
            BondViewInner::Archive(b) => latest_raw_weight_archived(b.as_archived()),
        }
    }

    /// Number of distinct sessions that touched this bond.
    pub fn distinct_session_count(&self) -> usize {
        match &self.inner {
            BondViewInner::Heap(b) => b.distinct_session_count(),
            BondViewInner::Archive(b) => distinct_session_count_archived(b.as_archived()),
        }
    }

    /// Escape hatch — borrow the raw heap bond. `None` for Archive.
    pub fn as_heap(&self) -> Option<&TemporalBond> {
        match &self.inner {
            BondViewInner::Heap(b) => Some(b),
            BondViewInner::Archive(_) => None,
        }
    }

    /// Escape hatch — borrow the raw archived bond. Returned
    /// reference's lifetime is tied to `&self` (OP-016: Arc held).
    pub fn as_archive(&self) -> Option<&ArchivedTemporalBond> {
        match &self.inner {
            BondViewInner::Heap(_) => None,
            BondViewInner::Archive(b) => Some(b.as_archived()),
        }
    }
}

// ---------- Mut handles ----------

/// Mutable handle to a node. Always heap-backed by construction:
/// `AssociativeNetwork::get_node_mut` performs the archive→heap CoW
/// promotion BEFORE returning the handle, so by the time the caller
/// holds one, the underlying storage is guaranteed heap.
pub struct NodeMutHandle<'a> {
    inner: &'a mut KnowledgeNode,
}

impl<'a> NodeMutHandle<'a> {
    pub fn from_heap(node: &'a mut KnowledgeNode) -> Self {
        NodeMutHandle { inner: node }
    }

    /// Run `f` against the heap-resident node.
    pub fn modify<F, R>(self, f: F) -> R
    where
        F: FnOnce(&mut KnowledgeNode) -> R,
    {
        f(self.inner)
    }

    /// Borrow the underlying heap node mutably.
    pub fn as_mut(&mut self) -> &mut KnowledgeNode {
        self.inner
    }

    /// Consume the handle and return the inner mutable reference.
    pub fn into_inner(self) -> &'a mut KnowledgeNode {
        self.inner
    }

    /// Read-side projection — same accessor surface as `NodeView`.
    pub fn view(&self) -> NodeView<'_> {
        NodeView::from_heap(self.inner)
    }
}

impl<'a> std::ops::Deref for NodeMutHandle<'a> {
    type Target = KnowledgeNode;
    fn deref(&self) -> &KnowledgeNode { self.inner }
}

impl<'a> std::ops::DerefMut for NodeMutHandle<'a> {
    fn deref_mut(&mut self) -> &mut KnowledgeNode { self.inner }
}

/// Mutable handle to a bond. Same Heap-after-CoW guarantee as
/// `NodeMutHandle`.
pub struct BondMutHandle<'a> {
    inner: &'a mut TemporalBond,
}

impl<'a> BondMutHandle<'a> {
    pub fn from_heap(bond: &'a mut TemporalBond) -> Self {
        BondMutHandle { inner: bond }
    }

    pub fn modify<F, R>(self, f: F) -> R
    where
        F: FnOnce(&mut TemporalBond) -> R,
    {
        f(self.inner)
    }

    pub fn as_mut(&mut self) -> &mut TemporalBond {
        self.inner
    }

    pub fn into_inner(self) -> &'a mut TemporalBond {
        self.inner
    }

    pub fn view(&self) -> BondView<'_> {
        BondView::from_heap(self.inner)
    }

    /// Mutable borrow of the bond's keyframes vector. Slice 2c-friendly
    /// alternative to direct `bond.keyframes` field access for sites
    /// that need to filter the keyframe list (consolidation Variant /
    /// Agent prune passes).
    pub fn keyframes_mut(&mut self) -> &mut Vec<TemporalKeyframe> {
        &mut self.inner.keyframes
    }
}

impl<'a> std::ops::Deref for BondMutHandle<'a> {
    type Target = TemporalBond;
    fn deref(&self) -> &TemporalBond { self.inner }
}

impl<'a> std::ops::DerefMut for BondMutHandle<'a> {
    fn deref_mut(&mut self) -> &mut TemporalBond { self.inner }
}

// ---------- BondsView ----------

/// View over the bond-table of a network. The `Mmap` variant chains an
/// in-memory overlay (heap copies of CoW-promoted bonds + freshly
/// created bonds) with the archived bonds, filtered by the deletion
/// overlay. Iteration order across the two halves is stable for any
/// given network state but not stable across network mutations.
#[derive(Clone, Copy)]
pub struct BondsView<'a> {
    inner: BondsViewInner<'a>,
}

#[derive(Clone, Copy)]
enum BondsViewInner<'a> {
    Heap(&'a HashMap<(Seed, Seed), TemporalBond>),
    Mmap {
        overlay: &'a HashMap<(Seed, Seed), TemporalBond>,
        archive: &'a NetworkArchive,
        deletions: &'a HashSet<Seed>,
    },
}

impl<'a> BondsView<'a> {
    pub fn from_heap(bonds: &'a HashMap<(Seed, Seed), TemporalBond>) -> Self {
        BondsView { inner: BondsViewInner::Heap(bonds) }
    }

    pub fn from_mmap(
        overlay: &'a HashMap<(Seed, Seed), TemporalBond>,
        archive: &'a NetworkArchive,
        deletions: &'a HashSet<Seed>,
    ) -> Self {
        BondsView { inner: BondsViewInner::Mmap { overlay, archive, deletions } }
    }

    pub fn len(&self) -> usize {
        match self.inner {
            BondsViewInner::Heap(m) => m.len(),
            BondsViewInner::Mmap { overlay, archive, deletions } => {
                let mut count = overlay
                    .iter()
                    .filter(|(k, _)| {
                        !deletions.contains(&k.0) && !deletions.contains(&k.1)
                    })
                    .count();
                for k in archive.iter_bond_keys() {
                    if overlay.contains_key(&k) { continue; }
                    if deletions.contains(&k.0) || deletions.contains(&k.1) {
                        continue;
                    }
                    count += 1;
                }
                count
            }
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn contains_key(&self, key: &(Seed, Seed)) -> bool {
        match self.inner {
            BondsViewInner::Heap(m) => m.contains_key(key),
            BondsViewInner::Mmap { overlay, archive, deletions } => {
                if deletions.contains(&key.0) || deletions.contains(&key.1) {
                    return false;
                }
                if overlay.contains_key(key) { return true; }
                archive.has_bond(key)
            }
        }
    }

    pub fn get(&self, key: &(Seed, Seed)) -> Option<BondView<'a>> {
        match self.inner {
            BondsViewInner::Heap(m) => m.get(key).map(BondView::from_heap),
            BondsViewInner::Mmap { overlay, archive, deletions } => {
                if deletions.contains(&key.0) || deletions.contains(&key.1) {
                    return None;
                }
                if let Some(b) = overlay.get(key) {
                    return Some(BondView::from_heap(b));
                }
                archive.bond_by_pair(key).map(BondView::from_bond_archive)
            }
        }
    }

    /// Iterate `(&(from, to), BondView)` pairs.
    pub fn iter(&self) -> BondsIter<'a> {
        match self.inner {
            BondsViewInner::Heap(m) => BondsIter { inner: BondsIterInner::Heap(m.iter()) },
            BondsViewInner::Mmap { overlay, archive, deletions } => BondsIter {
                inner: BondsIterInner::Mmap(MmapBondsIter {
                    deletions,
                    overlay,
                    overlay_iter: overlay.iter(),
                    archive_iter: archive.iter_bonds(),
                    in_overlay_phase: true,
                    yielded_archive_key: None,
                }),
            },
        }
    }

    /// Iterate the bond keys.
    pub fn keys(&self) -> BondsKeys<'a> {
        BondsKeys { iter: self.iter() }
    }

    /// Iterate `BondView` values.
    pub fn values(&self) -> BondsValues<'a> {
        BondsValues { iter: self.iter() }
    }
}

// Iterators.

pub struct BondsIter<'a> { inner: BondsIterInner<'a> }
enum BondsIterInner<'a> {
    Heap(std::collections::hash_map::Iter<'a, (Seed, Seed), TemporalBond>),
    Mmap(MmapBondsIter<'a>),
}

impl<'a> Iterator for BondsIter<'a> {
    type Item = (&'a (Seed, Seed), BondView<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            BondsIterInner::Heap(it) => it.next().map(|(k, b)| (k, BondView::from_heap(b))),
            BondsIterInner::Mmap(it) => it.next(),
        }
    }
}

pub struct MmapBondsIter<'a> {
    deletions: &'a HashSet<Seed>,
    overlay: &'a HashMap<(Seed, Seed), TemporalBond>,
    overlay_iter: std::collections::hash_map::Iter<'a, (Seed, Seed), TemporalBond>,
    /// OP-016: boxed dispatching iterator over the underlying
    /// `NetworkArchive`. Yields `(Seed, Seed)` keys plus
    /// `BondArchive<'a>` handles (which carry the Arc keeping the
    /// decompressed block alive for the lifetime of the handle).
    archive_iter: Box<dyn Iterator<Item = ((Seed, Seed), BondArchive<'a>)> + 'a>,
    in_overlay_phase: bool,
    yielded_archive_key: Option<(Seed, Seed)>,
}

impl<'a> Iterator for MmapBondsIter<'a> {
    type Item = (&'a (Seed, Seed), BondView<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.in_overlay_phase {
            for (k, v) in self.overlay_iter.by_ref() {
                if self.deletions.contains(&k.0) || self.deletions.contains(&k.1) {
                    continue;
                }
                return Some((k, BondView::from_heap(v)));
            }
            self.in_overlay_phase = false;
        }
        loop {
            let (key, archived_bond) = self.archive_iter.next()?;
            if self.overlay.contains_key(&key) { continue; }
            if self.deletions.contains(&key.0) || self.deletions.contains(&key.1) {
                continue;
            }
            self.yielded_archive_key = Some(key);
            // SAFETY: the reference borrows from `self.yielded_archive_key`
            // which is owned by `self`; consumers may only hold this
            // reference until they call `next()` again, which is the
            // standard iterator contract for borrowed-from-Self
            // returns. We use a raw pointer to extend the lifetime
            // exactly to the next() call boundary.
            let key_ref: &'a (Seed, Seed) = unsafe {
                &*(self.yielded_archive_key.as_ref().unwrap() as *const (Seed, Seed))
            };
            return Some((key_ref, BondView::from_bond_archive(archived_bond)));
        }
    }
}

/// Marker referenced from the patch script — kept around so the
/// idempotency check (`if "iter_bond_views_box" not in view`) can
/// flip after a successful patch run.
#[allow(dead_code)]
const fn iter_bond_views_box() {}

pub struct BondsKeys<'a> { iter: BondsIter<'a> }
impl<'a> Iterator for BondsKeys<'a> {
    type Item = &'a (Seed, Seed);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }
}

pub struct BondsValues<'a> { iter: BondsIter<'a> }
impl<'a> Iterator for BondsValues<'a> {
    type Item = BondView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }
}
