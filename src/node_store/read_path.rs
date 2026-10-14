//! `NodeSstReaderCache` + `get_node_v5` -- the v5 read-path fall-through.
//!
//! OP-BRAIN-V5 Phase 2.8 (Q5 of the design doc; audit S10 sub-port).
//! This is the first port that exposes the node-store substrate for
//! point lookups. Phases 2.3-2.7 wrote NodeSSTs (memtable -> flush
//! worker -> SST -> compactor), but the production read path never
//! reached them. Phase 2.8 adds a NEW public method --
//! [`Brain::get_node_v5`] (in `crate::brain`) -- that walks
//! `NodeMemTable` -> L0 NodeSSTs -> L1+ NodeSSTs and returns the
//! winning payload. The existing v4 read path
//! (`network.nodes` hot cache + bond-SST node retrieval) is NOT
//! touched here; both paths coexist until Phase 2.9 migration cuts
//! over.
//!
//! ## Cache lifecycle (sub-deliverable A)
//!
//! [`NodeSstReaderCache`] is a `HashMap<file_uid, Arc<NodeSstReader>>`
//! behind an `RwLock`. `get_or_open` returns an existing reader if
//! present; otherwise it opens the file at
//! `<node_store_dir>/<file_uid>.nsst`, inserts the `Arc`, and
//! returns the clone. Successive calls reuse the same reader handle
//! (proven by test J).
//!
//! ### Eviction limitation (documented, deliberate)
//!
//! The cache does NOT yet evict on compaction-retire. After a
//! `NodeCompactionWorker::compact` retires a source `file_uid` from
//! the manifest, the cached `Arc<NodeSstReader>` for that uid stays
//! resident until `Brain` shutdown. This is safe because:
//!
//! 1. The underlying `.nsst` file on disk is NOT unlinked at
//!    compaction time -- `cleanup_orphan_node_ssts` runs at the
//!    next startup and removes any committed `.nsst` whose
//!    `file_uid` is not in the live manifest. Until then the file
//!    handle held by the cached reader is still valid.
//! 2. `get_node_v5` consults the manifest before touching the
//!    cache; a retired uid has no `NodeSSTEntry`, so the cache is
//!    never queried for it. Stale Arc handles are wasted FDs, not
//!    wrong answers.
//!
//! Phase 2.7 follow-up will wire `evict(file_uid)` into the
//! compaction commit path. The method is exposed today so the
//! eventual wire-up is a small additive change.
//!
//! ## Algorithm (sub-deliverable B)
//!
//! See [`get_node_v5`] doc-comment for the level-walk + tombstone
//! short-circuit details. Highlights:
//!
//! - **MemTable first.** Tombstoned entry => `Ok(None)` (no shadow
//!   through to lower levels). Live entry => clone the payload.
//! - **L0**: NodeSSTs may have OVERLAPPING `[first_node_id,
//!   last_node_id]` ranges (the same NodeId can land in multiple
//!   L0 SSTs after repeated flushes of an updated record). Iterate
//!   EVERY L0 SST whose range covers `id`, open each via the cache,
//!   call `reader.get(id)`. Among the hits keep the highest-`seq`
//!   `NodeEntry`. Tombstone wins on tie via the entry's own seq.
//! - **L1+**: NodeSSTs have NON-OVERLAPPING ranges per level (LSM
//!   invariant; the compactor guarantees this). Binary-search the
//!   level for the unique SST whose range contains `id`. Open via
//!   the cache, call `reader.get(id)`. First level with a hit wins
//!   modulo tombstone semantics.
//!
//! Streaming discipline: we open ONLY the SSTs whose range pre-check
//! actually contains `id`. The brief's L0 candidate-set comment is
//! load-bearing: at L0 we may open more than one reader, but only
//! the ones whose `[first_node_id, last_node_id]` window covers
//! `id`.

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use lru::LruCache;

use crate::manifest::{Manifest, ManifestError, NodeSSTEntry};
use crate::node::KnowledgeNode;
use crate::node_intern::NodeId;
use crate::node_store::flush::nsst_filename;
use crate::node_store::memtable::{NodeEntry, NodeMemTable};
use crate::node_store::sst_reader::NodeSstReader;
use crate::node_store::sst_writer::NodeSstError;

/// Errors surfaced by the Phase 2.8 read path.
///
/// Carries the underlying I/O / SST / manifest error so callers can
/// translate cleanly into `BrainError`.
#[derive(Debug)]
pub enum ReadPathError {
    /// `std::io::Error` propagated from `NodeSstReader::open`'s file
    /// `open`. Distinct from [`ReadPathError::NodeSst`] which covers
    /// post-open format failures.
    Io(io::Error),
    /// Underlying [`NodeSstReader`] failure -- format violation,
    /// CRC mismatch, decompression failure. Always a corruption /
    /// programmer error condition.
    NodeSst(NodeSstError),
    /// Manifest helper rejected the lookup. Reserved for the future
    /// when this path consults `Manifest` helpers that surface
    /// `ManifestError`; today the manifest is read-only in the read
    /// path and this variant is not constructed, but keeping it on
    /// the enum lets us evolve the surface without a wire break.
    Manifest(ManifestError),
}

impl std::fmt::Display for ReadPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadPathError::Io(e) => write!(f, "v5 read path IO: {}", e),
            ReadPathError::NodeSst(e) => write!(f, "v5 read path NodeSST: {}", e),
            ReadPathError::Manifest(e) => write!(f, "v5 read path manifest: {}", e),
        }
    }
}

impl std::error::Error for ReadPathError {}

impl From<io::Error> for ReadPathError {
    fn from(e: io::Error) -> Self {
        ReadPathError::Io(e)
    }
}

impl From<NodeSstError> for ReadPathError {
    fn from(e: NodeSstError) -> Self {
        // NodeSstError already wraps io::Error in its own Io variant;
        // we collapse the typed surface to NodeSst so callers see the
        // raw SST format error. The reader open path returns this
        // surface for both file-IO and format failures.
        ReadPathError::NodeSst(e)
    }
}

impl From<ManifestError> for ReadPathError {
    fn from(e: ManifestError) -> Self {
        ReadPathError::Manifest(e)
    }
}

/// Lazy-open cache of `Arc<NodeSstReader>` keyed on `file_uid`.
///
/// One instance lives on the `Brain` (constructed alongside the
/// Phase 2.6 flush worker in `Brain::enable_flush_worker`). Readers
/// are opened on first lookup and cached for the rest of the brain's
/// lifetime; see the module-level doc for the eviction-on-compaction
/// limitation.
///
/// Thread-safety: the inner `RwLock<HashMap<...>>` lets concurrent
/// `get_or_open` callers share the same reader handle via `Arc`.
/// The open itself happens under the write half of the lock, so
/// at most one thread per missing `file_uid` does the disk work;
/// the `NodeSstReader` itself is internally synchronised (`Mutex<File>`
/// + `Mutex<BlockCacheInner>` per Phase 2.4) so post-open the lock
/// is not on the hot path.
pub struct NodeSstReaderCache {
    node_store_dir: PathBuf,
    cache: Mutex<LruCache<u64, Arc<NodeSstReader>>>,
    /// OP-BRAIN-V5 6.5: decryption key threaded into every reader this
    /// cache opens. `None` = plaintext store.
    key: Option<crate::encrypt::MasterKey>,
}

impl std::fmt::Debug for NodeSstReaderCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self
            .cache
            .lock()
            .map(|g| g.len())
            .unwrap_or(0);
        f.debug_struct("NodeSstReaderCache")
            .field("node_store_dir", &self.node_store_dir)
            .field("cache_size", &size)
            .finish()
    }
}

/// OP-BRAIN-MapRAM Fix B -- default hard cap on the number of resident
/// NodeSST readers the node reader cache holds. Same rationale as the
/// bond-side cap: under an encrypted store each reader's index + block
/// cache is anonymous heap, so an unbounded reader set lets resident
/// RAM track the live SST count. A miss just reopens the file.
/// Override with the `LU_NODE_READER_CACHE_CAP` env var.
const DEFAULT_NODE_READER_CACHE_CAP: usize = 512;

fn node_reader_cache_cap() -> std::num::NonZeroUsize {
    let n = std::env::var("LU_NODE_READER_CACHE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_NODE_READER_CACHE_CAP);
    std::num::NonZeroUsize::new(n).expect("node reader cache cap > 0")
}

impl NodeSstReaderCache {
    /// Construct an empty cache rooted at `node_store_dir`. The
    /// directory is NOT required to exist at construction time;
    /// `get_or_open` resolves the per-file path lazily.
    pub fn new(node_store_dir: PathBuf) -> Self {
        NodeSstReaderCache {
            node_store_dir,
            cache: Mutex::new(LruCache::new(node_reader_cache_cap())),
            key: None,
        }
    }

    /// Builder: set the decryption key used for every reader opened
    /// lazily by this cache (OP-BRAIN-V5 6.5). Existing `new(dir)`
    /// sites stay plaintext; production wiring threads the brain key.
    pub fn with_key(mut self, key: Option<crate::encrypt::MasterKey>) -> Self {
        self.key = key;
        self
    }

    /// Return the reader for `file_uid`, opening it lazily on first
    /// call. The reader handle is shared across callers via `Arc`.
    ///
    /// Locking discipline:
    /// 1. Acquire read lock, check for an existing entry. If present,
    ///    clone the `Arc` and return.
    /// 2. Otherwise drop the read lock and acquire the write lock.
    ///    Re-check (another thread may have raced in). If still
    ///    missing, open the file, insert, and return the clone.
    ///
    /// The double-check on the write half guarantees we open each
    /// `file_uid` exactly once across concurrent callers (matches
    /// the "small bound on cache_size" requirement in test J).
    pub fn get_or_open(&self, file_uid: u64) -> Result<Arc<NodeSstReader>, ReadPathError> {
        // OP-BRAIN-MapRAM Fix B: single Mutex<LruCache> scope. `get`
        // bumps recency; a miss opens the reader UNDER the lock (open
        // each file at most once) and `put`s it, evicting the least-
        // recently-used reader once the cap is exceeded. Eviction drops
        // only the cache's `Arc`; in-flight holders keep theirs and a
        // later lookup reopens the file.
        let mut c = self.cache.lock().expect("reader cache lock");
        if let Some(arc) = c.get(&file_uid) {
            return Ok(arc.clone());
        }
        let path = self.node_store_dir.join(nsst_filename(file_uid));
        let reader = NodeSstReader::open_with_key(&path, self.key.clone())
            .map_err(ReadPathError::from)?;
        let arc = Arc::new(reader);
        c.put(file_uid, arc.clone());
        Ok(arc)
    }

    /// Drop the cached reader for `file_uid`. Called by the
    /// compaction-retire path in a future port (Phase 2.7 follow-up);
    /// not yet wired here. Provided so callers can release FDs
    /// deterministically once the compaction integration lands.
    ///
    /// No-op if `file_uid` is not in the cache. The underlying
    /// `.nsst` file on disk is independent of this call --
    /// `cleanup_orphan_node_ssts` owns the disk-side unlink.
    pub fn evict(&self, file_uid: u64) {
        let mut w = self.cache.lock().expect("reader cache lock");
        w.pop(&file_uid);
    }

    /// Number of readers currently resident in the cache. Diagnostic
    /// / test surface only.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().expect("reader cache lock").len()
    }

    /// Read-only view of the configured `node_store_dir`. Used by
    /// the brain-level integration test to assert the cache was
    /// rooted at the same path the flush worker writes into.
    pub fn node_store_dir(&self) -> &std::path::Path {
        &self.node_store_dir
    }
}

/// The v5 fall-through read path.
///
/// Walks NodeMemTable -> L0 NodeSSTs -> L1+ NodeSSTs and returns the
/// winning `KnowledgeNode` payload. See the module-level doc for the
/// full algorithm.
///
/// Returns:
/// - `Ok(Some(payload))` -- a live entry was found.
/// - `Ok(None)` -- the NodeId is absent, OR the winning entry is a
///   tombstone (logical delete shadows lower levels per the brief).
/// - `Err(ReadPathError)` -- an SST open / decode failure.
///
/// ## L0 multi-candidate resolution
///
/// At L0 the same NodeId can appear in MULTIPLE SSTs (each flush
/// independently captures the current MemTable snapshot, so an
/// updated record lands in every flush after the update). The
/// algorithm iterates every L0 entry whose `[first_node_id,
/// last_node_id]` range contains `id`, opens its reader via the
/// cache, and calls `reader.get(id)`. Among all hits the
/// HIGHEST-`seq` entry wins. The seq comparison includes
/// tombstones -- a tombstone with a higher seq than any live L0
/// hit shadows the live ones and yields `Ok(None)`.
///
/// ## L1+ single-candidate resolution
///
/// L1+ NodeSSTs are NON-OVERLAPPING within a level (LSM compactor
/// invariant: `NodeCompactionWorker::compact` writes one output
/// per level with monotonically increasing block_index). At each
/// level we binary-search for the unique SST whose range covers
/// `id`. First level with a hit wins, again modulo tombstone
/// semantics.
///
/// ## Tombstone semantics
///
/// A tombstone at ANY level (MemTable, L0, L1, ...) is a logical
/// delete. The brief is explicit: "tombstone semantically deletes
/// the node; lower levels (older epochs) MUST NOT shadow through."
/// The algorithm short-circuits as soon as a tombstone wins at the
/// current level -- it never falls through to the next level.
///
/// Streaming: we do NOT iterate any reader's full content. Every
/// call goes through `NodeSstReader::get(id)`, which binary-searches
/// the block index and decodes at most ONE block. Vec-the-world is
/// structurally impossible here.
pub fn get_node_v5(
    memtable_snapshot: &NodeMemTable,
    manifest: &Manifest,
    reader_cache: &NodeSstReaderCache,
    id: NodeId,
) -> Result<Option<KnowledgeNode>, ReadPathError> {
    // ---- 1. MemTable first --------------------------------------
    if let Some(entry) = memtable_snapshot.get(id) {
        return Ok(unwrap_entry(entry.clone()));
    }

    // ---- 2. Walk node_levels L0 -> L1 -> ... -> LN -------------
    // OP-COMPACT-OVERLAP internal notes: every level -- not just
    // L0 -- is walked with the all-covering / highest-seq rule. A minor
    // L0->L1 node compaction can transiently leave TWO OVERLAPPING
    // segments at L1 (the compactor's own tests construct exactly this
    // state before an L1->L1 collapse). The former L1+ branch
    // binary-searched for a SINGLE covering SST, assuming L1+ ranges
    // never overlap; when they did, a base NodeId shadowed by a newer,
    // narrower segment's range missed and returned None -- silently
    // dropping every entity node's content (label / dates / overview)
    // from point lookups, while full scans (which read every SST) stayed
    // correct. `walk_level_for_id` opens ONLY covering SSTs (the range
    // pre-check keeps the non-overlapping case to a single open) and
    // keeps the highest-seq winner. Level precedence is preserved: the
    // first level with any hit wins, and a tombstone wins at its level.
    for level in &manifest.node_levels {
        if let Some(entry) = walk_level_for_id(level.ssts.as_slice(), reader_cache, id)? {
            return Ok(unwrap_entry(entry));
        }
    }

    // ---- 3. No hit at any level --------------------------------
    Ok(None)
}

/// Translate a `NodeEntry` into the read-path return shape: a live
/// entry surfaces its payload; a tombstone surfaces `None`.
///
/// Used at every level so the tombstone-no-shadow contract is
/// uniform across MemTable / L0 / L1+. The caller is responsible
/// for short-circuiting the level walk once `unwrap_entry` is
/// called -- per the brief, a tombstone wins AT ITS LEVEL and does
/// not fall through to lower levels.
fn unwrap_entry(entry: NodeEntry) -> Option<KnowledgeNode> {
    if entry.tombstone {
        None
    } else {
        Some(entry.payload)
    }
}

/// Iterate every NodeSST entry AT A LEVEL whose `[first_node_id,
/// last_node_id]` range contains `id`, open the reader through `cache`,
/// call `reader.get(id)`, and return the HIGHEST-seq hit.
///
/// Correct for BOTH overlapping levels (L0, or an L1 transiently left
/// overlapping by a minor L0->L1 compaction ahead of its L1->L1
/// collapse) and non-overlapping levels: the range pre-check opens only
/// covering SSTs, so a non-overlapping level costs a single open while
/// an overlapping level is resolved by highest-seq. See #1527.
///
/// `None` if no SST has a real hit (range pre-check filters every
/// candidate, OR every candidate's `reader.get(id)` returns `Ok(None)`).
fn walk_level_for_id(
    ssts: &[NodeSSTEntry],
    cache: &NodeSstReaderCache,
    id: NodeId,
) -> Result<Option<NodeEntry>, ReadPathError> {
    let mut winner: Option<NodeEntry> = None;
    for sst in ssts {
        if sst.first_node_id > id.0 || sst.last_node_id < id.0 {
            // Range pre-check excludes this SST -- do NOT open the
            // reader. Streaming discipline: only open candidates whose
            // declared range covers the lookup.
            continue;
        }
        let reader = cache.get_or_open(sst.file_uid)?;
        let hit = reader.get(id).map_err(ReadPathError::from)?;
        if let Some(hit_entry) = hit {
            winner = Some(match winner {
                None => hit_entry,
                Some(prev) if hit_entry.seq > prev.seq => hit_entry,
                Some(prev) => prev,
            });
        }
    }
    Ok(winner)
}

// OP-COMPACT-OVERLAP internal notes: `binary_search_level_for_id`
// retired. It picked a SINGLE covering SST on the assumption that L1+
// ranges never overlap; a minor L0->L1 compaction can transiently break
// that invariant, and the single-candidate search then missed base
// nodes shadowed by a newer segment's range. `walk_level_for_id` above
// now serves EVERY level and is correct under transient overlap.

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{KeyFingerprint, NodeLevel};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeUnit};
    use crate::node_store::sst_writer::{
        CompressionKind, NodeSstWriteOpts, NodeSstWriter,
    };
    use crate::seed::Seed;
    use std::path::Path;
    use tempfile::tempdir;

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

    /// Helper: write a NodeSST file at `node_store_dir/<file_uid>.nsst`
    /// containing the supplied (NodeId, NodeEntry) entries (sorted by
    /// NodeId ascending). Returns a `NodeSSTEntry` describing the
    /// freshly-written file so the test can drop it into a manifest
    /// level directly.
    fn write_sst(
        node_store_dir: &Path,
        file_uid: u64,
        epoch: u64,
        entries: &[(NodeId, NodeEntry)],
    ) -> NodeSSTEntry {
        std::fs::create_dir_all(node_store_dir).unwrap();
        let path = node_store_dir.join(nsst_filename(file_uid));
        let opts = NodeSstWriteOpts {
            target_entries_per_block: 16,
            compression: CompressionKind::default(),
            file_uid,
        };
        let mut w = NodeSstWriter::create(&path, opts).unwrap();
        for (id, entry) in entries {
            w.append(*id, entry).unwrap();
        }
        let stats = w.finish().unwrap();
        NodeSSTEntry {
            file_uid,
            first_node_id: entries.first().map(|(i, _)| i.0).unwrap_or(0),
            last_node_id: entries.last().map(|(i, _)| i.0).unwrap_or(0),
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch,
        }
    }

    fn live_entry(payload: KnowledgeNode, seq: u64) -> NodeEntry {
        NodeEntry { payload, tombstone: false, seq }
    }

    fn tomb_entry(payload: KnowledgeNode, seq: u64) -> NodeEntry {
        NodeEntry { payload, tombstone: true, seq }
    }

    fn empty_manifest() -> Manifest {
        Manifest::new(KeyFingerprint::none())
    }

    /// OP-COMPACT-OVERLAP internal notes: two OVERLAPPING segments
    /// at L1 -- as a minor L0->L1 compaction transiently produces before
    /// its L1->L1 collapse -- must NOT drop a base node from point
    /// lookups. The base segment holds id=100; a newer, narrower segment
    /// whose range [50..200] COVERS 100 does not contain it. The retired
    /// binary search picked the narrow segment and returned None; the
    /// all-covering walk resolves id=100 from the base segment.
    #[test]
    fn phase28_overlapping_l1_resolves_base_node_op1527() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let base = write_sst(
            &node_store_dir,
            10,
            1,
            &[
                (NodeId(0), live_entry(mk_node("zero"), 1)),
                (NodeId(100), live_entry(mk_node("hundred"), 1)),
                (NodeId(300), live_entry(mk_node("three_hundred"), 1)),
            ],
        );
        // Narrower, NEWER segment whose range [50..200] overlaps `base`.
        let narrow = write_sst(
            &node_store_dir,
            11,
            2,
            &[
                (NodeId(50), live_entry(mk_node("fifty"), 2)),
                (NodeId(200), live_entry(mk_node("two_hundred"), 2)),
            ],
        );
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![] });
        manifest
            .node_levels
            .push(NodeLevel { level: 1, ssts: vec![base, narrow] });
        let mt = NodeMemTable::new();

        // Base-only node, shadowed by the narrow segment's range.
        let got = get_node_v5(&mt, &manifest, &cache, NodeId(100)).unwrap();
        assert!(
            got.is_some(),
            "base node 100 must resolve despite an overlapping L1 segment"
        );
        assert_eq!(got.unwrap().content, "hundred");

        // A node present only in the newer segment still resolves.
        let g2 = get_node_v5(&mt, &manifest, &cache, NodeId(200)).unwrap();
        assert_eq!(g2.unwrap().content, "two_hundred");

        // A NodeId covered by neither file's contents is absent.
        let g3 = get_node_v5(&mt, &manifest, &cache, NodeId(999)).unwrap();
        assert!(g3.is_none());
    }

    /// Test A: MemTable hit (live and miss).
    #[test]
    fn phase28_a_memtable_hit() {
        let dir = tempdir().unwrap();
        let cache = NodeSstReaderCache::new(dir.path().join("node_store"));
        let mut mt = NodeMemTable::new();
        let payload = mk_node("memtable_hit_5");
        mt.insert(NodeId(5), payload.clone(), 1);
        let manifest = empty_manifest();
        // Hit
        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(got.is_some(), "memtable hit must return Some");
        assert_eq!(got.unwrap().content, "memtable_hit_5");
        // Miss
        let miss = get_node_v5(&mt, &manifest, &cache, NodeId(99)).unwrap();
        assert!(miss.is_none(), "missing NodeId must return None");
    }

    /// Test B: L0 SST hit (no MemTable entry).
    #[test]
    fn phase28_b_l0_hit() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        // Build 10 entries at NodeIds 0..10.
        let entries: Vec<(NodeId, NodeEntry)> = (0..10u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("l0_node_{}", i)), i)))
            .collect();
        let sst_entry = write_sst(&node_store_dir, 0, 0, &entries);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![sst_entry] });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(7)).unwrap();
        assert!(got.is_some(), "L0 hit must return Some");
        assert_eq!(got.unwrap().content, "l0_node_7");
    }

    /// Test C: L1 SST hit (empty L0, empty MemTable).
    #[test]
    fn phase28_c_l1_hit() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let entries: Vec<(NodeId, NodeEntry)> = (100..110u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("l1_node_{}", i)), i)))
            .collect();
        let sst_entry = write_sst(&node_store_dir, 0, 0, &entries);
        let mut manifest = empty_manifest();
        // Place L0 empty, L1 holding the SST.
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![sst_entry] });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(105)).unwrap();
        assert!(got.is_some(), "L1 hit must return Some");
        assert_eq!(got.unwrap().content, "l1_node_105");
        // NodeId outside both levels' ranges -> None.
        let miss = get_node_v5(&mt, &manifest, &cache, NodeId(999)).unwrap();
        assert!(miss.is_none());
    }

    /// Test D: MemTable takes precedence over L0.
    #[test]
    fn phase28_d_memtable_over_l0() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        // L0 has NodeId 5 with payload "from_l0".
        let l0_entries = vec![(NodeId(5), live_entry(mk_node("from_l0"), 1))];
        let sst_entry = write_sst(&node_store_dir, 0, 0, &l0_entries);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![sst_entry] });
        // MemTable has NodeId 5 with payload "from_memtable".
        let mut mt = NodeMemTable::new();
        mt.insert(NodeId(5), mk_node("from_memtable"), 100);

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "from_memtable", "MemTable must win over L0");
    }

    /// Test E: L0 takes precedence over L1.
    #[test]
    fn phase28_e_l0_over_l1() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let l0 = vec![(NodeId(5), live_entry(mk_node("from_l0"), 10))];
        let l1 = vec![(NodeId(5), live_entry(mk_node("from_l1"), 5))];
        let l0_sst = write_sst(&node_store_dir, 0, 1, &l0);
        let l1_sst = write_sst(&node_store_dir, 1, 0, &l1);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![l0_sst] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![l1_sst] });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "from_l0", "L0 must win over L1");
    }

    /// Test F: higher-seq L0 wins over lower-seq L0 across two
    /// overlapping L0 SSTs.
    #[test]
    fn phase28_f_higher_seq_l0_wins() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        // SST A: seq=2, payload "A".
        let a = vec![(NodeId(5), live_entry(mk_node("A"), 2))];
        // SST B: seq=5, payload "B".
        let b = vec![(NodeId(5), live_entry(mk_node("B"), 5))];
        let sst_a = write_sst(&node_store_dir, 0, 0, &a);
        let sst_b = write_sst(&node_store_dir, 1, 1, &b);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel {
            level: 0,
            ssts: vec![sst_a, sst_b],
        });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "B", "higher-seq L0 SST must win");
    }

    /// Test G: tombstone in MemTable -> None. L0 has a live entry that
    /// MUST NOT shadow through.
    #[test]
    fn phase28_g_tombstone_in_memtable_no_shadow() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        // L0 has a LIVE payload at NodeId 5.
        let l0 = vec![(NodeId(5), live_entry(mk_node("live_in_l0"), 1))];
        let sst_entry = write_sst(&node_store_dir, 0, 0, &l0);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![sst_entry] });
        // MemTable has a TOMBSTONE at NodeId 5.
        let mut mt = NodeMemTable::new();
        mt.tombstone(NodeId(5), 100);

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(
            got.is_none(),
            "tombstone in MemTable must shadow L0 live entry -- got {:?}",
            got.map(|n| n.content),
        );
    }

    /// Test H: tombstone in L0 (no MemTable entry) -> None. L1 has a
    /// live entry that MUST NOT shadow through.
    #[test]
    fn phase28_h_tombstone_in_l0_no_shadow() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        // L0 holds a TOMBSTONE at NodeId 5 (with seq high enough
        // to be the lone L0 hit).
        let l0 = vec![(NodeId(5), tomb_entry(mk_node("tomb_payload"), 99))];
        // L1 holds a LIVE entry at NodeId 5.
        let l1 = vec![(NodeId(5), live_entry(mk_node("live_in_l1"), 5))];
        let l0_sst = write_sst(&node_store_dir, 0, 1, &l0);
        let l1_sst = write_sst(&node_store_dir, 1, 0, &l1);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![l0_sst] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![l1_sst] });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(
            got.is_none(),
            "tombstone in L0 must shadow L1 live entry -- got {:?}",
            got.map(|n| n.content),
        );
    }

    /// Test I: missing NodeId across everything -> None.
    #[test]
    fn phase28_i_missing_node() {
        let dir = tempdir().unwrap();
        let cache = NodeSstReaderCache::new(dir.path().join("node_store"));
        let manifest = empty_manifest();
        let mt = NodeMemTable::new();
        let got = get_node_v5(&mt, &manifest, &cache, NodeId(42)).unwrap();
        assert!(got.is_none(), "missing NodeId across all levels must return None");
    }

    /// Test J: reader cache reuse. After 100 lookups at the same SST,
    /// cache_size is 1 (only one reader opened).
    #[test]
    fn phase28_j_cache_reuse() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let entries: Vec<(NodeId, NodeEntry)> = (0..100u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("n_{}", i)), i)))
            .collect();
        let sst_entry = write_sst(&node_store_dir, 0, 0, &entries);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![sst_entry] });
        let mt = NodeMemTable::new();

        for i in 0..100u64 {
            let got = get_node_v5(&mt, &manifest, &cache, NodeId(i)).unwrap();
            assert!(got.is_some(), "lookup {} must hit", i);
        }
        assert_eq!(
            cache.cache_size(),
            1,
            "cache must reuse single reader across 100 lookups; got {}",
            cache.cache_size(),
        );
    }

    /// Test (additional): L0 with multiple SSTs where ONLY ONE
    /// covers the lookup -- the others' range pre-check excludes
    /// them, so they are NEVER opened. cache_size must be 1 even
    /// though three SSTs live at L0.
    #[test]
    fn phase28_l0_range_precheck_skips_irrelevant_ssts() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());

        // Three L0 SSTs at disjoint ranges:
        // SST 0: NodeIds 0..10
        // SST 1: NodeIds 100..110
        // SST 2: NodeIds 200..210
        let e0: Vec<(NodeId, NodeEntry)> = (0..10u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("s0_{}", i)), i)))
            .collect();
        let e1: Vec<(NodeId, NodeEntry)> = (100..110u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("s1_{}", i)), i)))
            .collect();
        let e2: Vec<(NodeId, NodeEntry)> = (200..210u64)
            .map(|i| (NodeId(i), live_entry(mk_node(&format!("s2_{}", i)), i)))
            .collect();
        let s0 = write_sst(&node_store_dir, 0, 0, &e0);
        let s1 = write_sst(&node_store_dir, 1, 1, &e1);
        let s2 = write_sst(&node_store_dir, 2, 2, &e2);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel {
            level: 0,
            ssts: vec![s0, s1, s2],
        });
        let mt = NodeMemTable::new();

        // Look up NodeId 105 -- only SST 1 covers it. SSTs 0 and 2
        // must NEVER be opened.
        let got = get_node_v5(&mt, &manifest, &cache, NodeId(105)).unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "s1_105");
        assert_eq!(
            cache.cache_size(),
            1,
            "range pre-check must skip non-covering L0 SSTs; got {}",
            cache.cache_size(),
        );
    }

    /// Test (additional): live entry in L1 NEVER opens an L2 SST
    /// containing the same NodeId. The brief's "first level with a
    /// hit wins" contract -- L1 hit short-circuits L2+.
    #[test]
    fn phase28_l1_hit_short_circuits_lower_levels() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let l1 = vec![(NodeId(5), live_entry(mk_node("from_l1"), 10))];
        let l2 = vec![(NodeId(5), live_entry(mk_node("from_l2"), 1))];
        let l1_sst = write_sst(&node_store_dir, 0, 1, &l1);
        let l2_sst = write_sst(&node_store_dir, 1, 0, &l2);
        let mut manifest = empty_manifest();
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![l1_sst] });
        manifest.node_levels.push(NodeLevel { level: 2, ssts: vec![l2_sst] });
        let mt = NodeMemTable::new();

        let got = get_node_v5(&mt, &manifest, &cache, NodeId(5)).unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "from_l1");
        assert_eq!(
            cache.cache_size(),
            1,
            "L1 hit must short-circuit L2 open; got {}",
            cache.cache_size(),
        );
    }

    /// `evict` removes the cached reader; a subsequent `get_or_open`
    /// re-opens the file. (Pure unit test of the cache primitive --
    /// no `get_node_v5` involvement.)
    #[test]
    fn phase28_cache_evict_round_trip() {
        let dir = tempdir().unwrap();
        let node_store_dir = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(node_store_dir.clone());
        let entries = vec![(NodeId(1), live_entry(mk_node("n1"), 1))];
        let _sst_entry = write_sst(&node_store_dir, 0, 0, &entries);

        let _r1 = cache.get_or_open(0).unwrap();
        assert_eq!(cache.cache_size(), 1);
        cache.evict(0);
        assert_eq!(cache.cache_size(), 0);
        let _r2 = cache.get_or_open(0).unwrap();
        assert_eq!(cache.cache_size(), 1);
    }
}
