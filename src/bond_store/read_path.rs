//! `BondSstReaderCache` + `get_outgoing_bonds_v5` / `get_incoming_bonds_v5`
//! -- the v5 bond-store read path.
//!
//! OP-BRAIN-V5 Phase 3.7 (design doc Q5; Phase 3.0 audit S6, S14).
//! Phases 3.1-3.6 wrote BondSSTs (flush worker -> SST -> compactor
//! -> paired reverse index), but nothing read them back. Phase 3.7
//! adds TWO new public methods -- [`crate::brain::Brain::get_outgoing_bonds_v5`]
//! and [`crate::brain::Brain::get_incoming_bonds_v5`] -- that walk
//! the manifest's `bond_v5_levels` and return the winning bond rows.
//! Structural template: Phase 2.8's `node_store::read_path`
//! (fa30de4).
//!
//! ## Scope (read carefully)
//!
//! - The existing v4 production bond read path (`network` traversal
//!   + `Brain.v4_sst_cache`) is NOT touched. Both paths coexist
//!   until Phase 3.9 retires the v4 read path and `v4_sst_cache`
//!   atomically, after Phase 3.10 migration populates the v5 SSTs.
//!   The Phase 3.0 audit's "retires v4_sst_cache" wording on its
//!   Phase 3.7 line is PREPARATORY -- the v5 path is BUILT here so
//!   that 3.9's retirement is not load-bearing on read coverage.
//! - The live MemTable bond half (`src/memtable.rs::MemTableInner
//!   .entries`) is NOT consulted: both free functions below are
//!   SST-only, matching the Phase 3.7 brief's signatures. The
//!   MemTable stores bond objects as Seeds (audit Q3.0.F), so
//!   surfacing un-flushed bonds through this NodeId-keyed API
//!   requires the Seed->NodeId projection that today lives only in
//!   `flush.rs::flush_once`'s v5 dual-emit block. Consequence:
//!   un-flushed bonds are INVISIBLE to `get_*_bonds_v5` until the
//!   next flush. The brain-level round-trip test encodes this
//!   expectation explicitly (pre-flush query asserts empty).
//!   Production cutover (Phase 3.10+) must either wire the MemTable
//!   half or keep a flush-before-read discipline; this is the
//!   bond-side analogue of Phase 2.8's "manifest one flush behind
//!   MemTable" consistency note -- WIDER here, because Phase 2.8's
//!   node path does consult its memtable while this path consults
//!   none.
//!
//! ## Cache namespacing (sub-deliverable A)
//!
//! `reverse_file_uid` values are allocated from the SAME
//! `Manifest::next_bond_v5_sst_uid` counter as primary uids (Phase
//! 3.6 allocates the reverse uid with a second
//! `allocate_bond_v5_sst_uid` call), so primary uid 5 and reverse
//! uid 5 can both legitimately exist. On disk they never collide
//! (`5.bsst` vs `5.bsst.reverse`) but in a bare `HashMap<u64, _>`
//! they would. Decision: **Option A -- composite key**
//! `(BondSstKind, u64)` in ONE map, NOT two maps. Rationale:
//! one lock + one map keeps `cache_size()` meaning "total resident
//! readers" (the bound the cache-reuse test measures), keeps the
//! eviction surface uniform for the future compaction-retire
//! wiring, and encodes the namespace in the type system where a
//! two-map split would duplicate structure that future code could
//! update on one side only.
//!
//! ## Eviction limitation (documented, deliberate)
//!
//! Like Phase 2.8's `NodeSstReaderCache`, this cache does NOT yet
//! evict on compaction-retire. Stale `Arc<BondSstReader>` handles
//! are wasted FDs, not wrong answers: the read path consults the
//! manifest FIRST, and a retired uid has no live `BondSSTEntry`,
//! so the cache is never queried for it. Underlying files persist
//! until the next restart's `cleanup_orphan_bond_ssts`. `evict` /
//! `evict_reverse` exist today so the eventual wire-up is a small
//! additive change.
//!
//! ## Algorithm + tombstones
//!
//! See [`get_outgoing_bonds_v5`] and [`get_incoming_bonds_v5`] for
//! the per-direction level walks. Tombstones
//! ([`BondPayload::confidence`] `== 0xFF`,
//! [`crate::bond_store::BOND_TOMBSTONE_CONFIDENCE`]) are filtered
//! at the very END of the level walk: a tombstone WINS its
//! `(predicate, counterparty)` key during the walk -- shadowing
//! live rows at lower (older) levels exactly like Phase 2.8's
//! node-side tombstones -- and is then dropped from the yield, so
//! callers never observe one.

use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use lru::LruCache;

use crate::bond_store::compaction::BOND_TOMBSTONE_CONFIDENCE;
use crate::bond_store::flush::{bsst_filename, bsst_reverse_filename};
use crate::bond_store::sst_reader::BondSstReader;
use crate::bond_store::sst_writer::{BondPayload, BondSstError};
use crate::manifest::{BondSSTEntry, Manifest, ManifestError};
use crate::node_intern::NodeId;

/// Errors surfaced by the Phase 3.7 bond read path.
///
/// Mirrors `node_store::read_path::ReadPathError` variant-for-
/// variant so `Brain`-level error translation stays uniform.
#[derive(Debug)]
pub enum BondReadPathError {
    /// `std::io::Error` propagated from file opens that fail before
    /// the BondSST format layer is reached.
    Io(io::Error),
    /// Underlying [`BondSstReader`] failure -- magic/format
    /// violation, CRC mismatch, decode failure. Always a corruption
    /// / programmer-error condition.
    BondSst(BondSstError),
    /// Manifest helper rejected a lookup. Today the manifest is
    /// read-only in this path and the variant is not constructed;
    /// keeping it on the enum lets the surface evolve without a
    /// wire break (same reasoning as Phase 2.8's enum).
    Manifest(ManifestError),
}

impl std::fmt::Display for BondReadPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BondReadPathError::Io(e) => write!(f, "v5 bond read path IO: {}", e),
            BondReadPathError::BondSst(e) => write!(f, "v5 bond read path BondSST: {}", e),
            BondReadPathError::Manifest(e) => write!(f, "v5 bond read path manifest: {}", e),
        }
    }
}

impl std::error::Error for BondReadPathError {}

impl From<io::Error> for BondReadPathError {
    fn from(e: io::Error) -> Self {
        BondReadPathError::Io(e)
    }
}

impl From<BondSstError> for BondReadPathError {
    fn from(e: BondSstError) -> Self {
        // BondSstError already wraps io::Error in its own Io
        // variant; collapse the typed surface to BondSst so callers
        // see the raw SST error (matches Phase 2.8's From choice).
        BondReadPathError::BondSst(e)
    }
}

impl From<ManifestError> for BondReadPathError {
    fn from(e: ManifestError) -> Self {
        BondReadPathError::Manifest(e)
    }
}

/// Cache-key namespace tag. `reverse_file_uid` shares the primary
/// uid counter (see module docs, "Cache namespacing"), so the kind
/// MUST participate in the cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BondSstKind {
    /// `<uid>.bsst` -- rows sorted by (subject, predicate, object).
    Primary,
    /// `<uid>.bsst.reverse` -- same wire format, rows sorted by
    /// (object, predicate, subject); the three wire fields store
    /// the SWAPPED orientation (Phase 3.6 projection).
    Reverse,
}

/// Lazy-open cache of `Arc<BondSstReader>` keyed on
/// `(BondSstKind, file_uid)`.
///
/// One instance lives on the `Brain`
/// (`Brain.bond_reader_cache`, constructed alongside the Phase 3.3
/// flush worker in `Brain::enable_flush_worker`). Readers are
/// opened on first lookup and cached for the rest of the brain's
/// lifetime; see the module-level docs for the
/// eviction-on-compaction limitation and the composite-key
/// namespacing decision.
///
/// Thread-safety: identical discipline to Phase 2.8's
/// `NodeSstReaderCache` -- `RwLock<HashMap<...>>`, read-lock fast
/// path, write-lock + re-check slow path so each file is opened at
/// most once across concurrent callers. `BondSstReader` is
/// internally synchronised (`Mutex<File>` + `Mutex` around the
/// block LRU, Phase 3.2), so post-open the cache lock is off the
/// hot path.
pub struct BondSstReaderCache {
    bond_store_dir: PathBuf,
    cache: Mutex<LruCache<(BondSstKind, u64), Arc<BondSstReader>>>,
    /// OP-BRAIN-V5 6.5: decryption key threaded into every reader this
    /// cache opens. `None` = plaintext store.
    key: Option<crate::encrypt::MasterKey>,
}

impl std::fmt::Debug for BondSstReaderCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.cache.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("BondSstReaderCache")
            .field("bond_store_dir", &self.bond_store_dir)
            .field("cache_size", &size)
            .finish()
    }
}

/// OP-BRAIN-MapRAM Fix B -- default hard cap on the number of resident
/// SST readers (primaries + reverses) the bond reader cache holds.
/// Each reader keeps a block index plus a bounded per-reader block
/// cache; under an ENCRYPTED store those pages are anonymous heap (they
/// cannot be shared with the OS page cache), so an unbounded reader set
/// makes resident RAM track the live SST count. The cap bounds it; a
/// miss just reopens the file -- the cache is an accelerator, never the
/// store of record, so correctness is independent of what it holds.
/// Override with the `LU_BOND_READER_CACHE_CAP` env var.
const DEFAULT_BOND_READER_CACHE_CAP: usize = 512;

fn bond_reader_cache_cap() -> std::num::NonZeroUsize {
    let n = std::env::var("LU_BOND_READER_CACHE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_BOND_READER_CACHE_CAP);
    std::num::NonZeroUsize::new(n).expect("bond reader cache cap > 0")
}

impl BondSstReaderCache {
    /// Construct an empty cache rooted at `bond_store_dir`. The
    /// directory is NOT required to exist at construction time;
    /// the per-file path is resolved lazily on first open.
    pub fn new(bond_store_dir: PathBuf) -> Self {
        BondSstReaderCache {
            bond_store_dir,
            cache: Mutex::new(LruCache::new(bond_reader_cache_cap())),
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

    /// Open (or return cached) primary `.bsst` reader by `file_uid`.
    pub fn get_or_open_primary(
        &self,
        file_uid: u64,
    ) -> Result<Arc<BondSstReader>, BondReadPathError> {
        self.get_or_open(BondSstKind::Primary, file_uid)
    }

    /// Open (or return cached) reverse `.bsst.reverse` reader by
    /// `reverse_file_uid`. The reverse uid space is the same
    /// `next_bond_v5_sst_uid` counter as primaries, hence the
    /// namespaced cache key (see module docs).
    pub fn get_or_open_reverse(
        &self,
        reverse_file_uid: u64,
    ) -> Result<Arc<BondSstReader>, BondReadPathError> {
        self.get_or_open(BondSstKind::Reverse, reverse_file_uid)
    }

    fn get_or_open(
        &self,
        kind: BondSstKind,
        uid: u64,
    ) -> Result<Arc<BondSstReader>, BondReadPathError> {
        let key = (kind, uid);
        // OP-BRAIN-MapRAM Fix B: single Mutex<LruCache> scope. `get`
        // bumps recency; on a miss we open the reader UNDER the lock
        // (preserving the original "open each file at most once"
        // discipline) and `put` it, which evicts the least-recently-
        // used reader once the cap is exceeded. An evicted reader is
        // just dropped -- the value is an `Arc`, so any in-flight
        // holder keeps it alive, and a later lookup reopens the file.
        let mut c = self.cache.lock().expect("bond reader cache lock");
        if let Some(arc) = c.get(&key) {
            return Ok(arc.clone());
        }
        let filename = match kind {
            BondSstKind::Primary => bsst_filename(uid),
            BondSstKind::Reverse => bsst_reverse_filename(uid),
        };
        let path = self.bond_store_dir.join(filename);
        let reader = BondSstReader::open_with_key(&path, self.key.clone())
            .map_err(BondReadPathError::from)?;
        let arc = Arc::new(reader);
        c.put(key, arc.clone());
        Ok(arc)
    }

    /// Drop the cached PRIMARY reader for `file_uid`. No-op if not
    /// resident. Future compaction-retire wiring calls this; the
    /// on-disk file is independent (`cleanup_orphan_bond_ssts`
    /// owns the unlink).
    pub fn evict(&self, file_uid: u64) {
        let mut w = self.cache.lock().expect("bond reader cache lock");
        w.pop(&(BondSstKind::Primary, file_uid));
    }

    /// Drop the cached REVERSE reader for `reverse_file_uid`.
    /// No-op if not resident.
    pub fn evict_reverse(&self, reverse_file_uid: u64) {
        let mut w = self.cache.lock().expect("bond reader cache lock");
        w.pop(&(BondSstKind::Reverse, reverse_file_uid));
    }

    /// Total readers currently resident (primaries + reverses).
    /// Diagnostic / test surface only.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().expect("bond reader cache lock").len()
    }

    /// Read-only view of the configured `bond_store_dir`. Mirrors
    /// `NodeSstReaderCache::node_store_dir` for test parity.
    pub fn bond_store_dir(&self) -> &std::path::Path {
        &self.bond_store_dir
    }
}

/// Merge one SST hit into the per-query accumulator.
///
/// Cross-level shadowing contract:
/// - The level walk runs L0 -> L1 -> ... -> LN (younger first).
/// - At L0 the same triple can appear in MULTIPLE SSTs (each flush
///   captures an independent MemTable snapshot), so an occupied key
///   is re-resolved by `seq` -- highest wins, tombstones included.
/// - At L1+ an occupied key means a YOUNGER level (or an earlier
///   SST at this level -- impossible for the same triple, since
///   L1+ subject ranges are disjoint per the compactor invariant)
///   already owns the triple; the older row is shadowed and
///   dropped. No cross-level seq comparison is needed: the LSM
///   invariant guarantees upper levels are strictly newer.
fn merge_hit(
    acc: &mut BTreeMap<(u8, u64), BondPayload>,
    level_idx: usize,
    key: (u8, u64),
    payload: BondPayload,
) {
    use std::collections::btree_map::Entry;
    match acc.entry(key) {
        Entry::Vacant(v) => {
            v.insert(payload);
        }
        Entry::Occupied(mut occ) => {
            if level_idx == 0 && payload.seq > occ.get().seq {
                occ.insert(payload);
            }
            // level >= 1 with occupied key: the resident entry came
            // from a younger level -- keep it (shadowing).
        }
    }
}

/// Binary-search a level's BondSST entries for the unique SST whose
/// `[first_subject, last_subject]` range covers `subject`. Returns
/// `None` if the level cannot contain the subject.
///
/// Pre-conditions (NOT re-verified here, same discipline as Phase
/// 2.8's node-side helper): `ssts` sorted by `first_subject`
/// ascending AND non-overlapping ranges -- the L1+ LSM invariant
/// from `BondCompactionWorker`. A violation surfaces as a missed
/// lookup, not UB; the compactor test suite owns the invariant.
fn binary_search_level_for_subject<'a>(
    ssts: &'a [BondSSTEntry],
    subject: NodeId,
) -> Option<&'a BondSSTEntry> {
    if ssts.is_empty() {
        return None;
    }
    let mut lo: i64 = 0;
    let mut hi: i64 = ssts.len() as i64 - 1;
    let mut candidate: Option<usize> = None;
    while lo <= hi {
        let mid = ((lo + hi) / 2) as usize;
        if ssts[mid].first_subject <= subject.0 {
            candidate = Some(mid);
            lo = mid as i64 + 1;
        } else {
            hi = mid as i64 - 1;
        }
    }
    let cand = candidate?;
    if ssts[cand].last_subject >= subject.0 {
        Some(&ssts[cand])
    } else {
        None
    }
}

/// Walk `bond_v5_levels` L0 -> L1 -> ... -> LN collecting every
/// LIVE outgoing bond for `subject`, in ascending
/// `(predicate, object)` order.
///
/// Level semantics:
/// - **L0**: entries may have OVERLAPPING subject ranges. Every L0
///   SST whose `[first_subject, last_subject]` covers `subject` is
///   opened (range pre-check first -- non-covering SSTs are never
///   opened) and `iter_subject(subject)` streamed; duplicate
///   triples across L0 SSTs resolve by highest `seq`.
/// - **L1+**: NON-overlapping ranges per level; binary-search the
///   single candidate SST per level. Rows whose triple is already
///   owned by a younger level are shadowed (see [`merge_hit`]).
///
/// Tombstones (`confidence == 0xFF`) win their key during the walk
/// and are filtered from the final yield -- the caller NEVER sees
/// a tombstone, and a lower-level live row for the same triple is
/// NOT resurrected.
///
/// Memory discipline: the accumulator is bounded by the subject's
/// out-degree (one entry per live `(predicate, object)`), not by
/// SST size -- `iter_subject` streams block-by-block via the Phase
/// 3.2 LRU; no Vec-the-world.
///
/// Error contract: the walk is eager; on the first error the
/// returned iterator yields exactly one `Err` and nothing else.
pub fn get_outgoing_bonds_v5<'a>(
    manifest: &'a Manifest,
    reader_cache: &'a BondSstReaderCache,
    subject: NodeId,
) -> impl Iterator<Item = Result<(NodeId, u8, NodeId, BondPayload), BondReadPathError>> + 'a {
    let items: Vec<Result<(NodeId, u8, NodeId, BondPayload), BondReadPathError>> =
        match collect_outgoing(manifest, reader_cache, subject) {
            Ok(rows) => rows.into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        };
    items.into_iter()
}

fn collect_outgoing(
    manifest: &Manifest,
    reader_cache: &BondSstReaderCache,
    subject: NodeId,
) -> Result<Vec<(NodeId, u8, NodeId, BondPayload)>, BondReadPathError> {
    // Keyed (predicate, object) -- the subject is fixed for the
    // whole query, so the pair identifies the triple.
    let mut acc: BTreeMap<(u8, u64), BondPayload> = BTreeMap::new();

    for (level_idx, level) in manifest.bond_v5_levels.iter().enumerate() {
        if level_idx == 0 {
            // L0: overlapping ranges possible -- iterate EVERY
            // covering SST. Range pre-check keeps non-covering
            // SSTs unopened (streaming discipline).
            for sst in &level.ssts {
                if sst.first_subject > subject.0 || sst.last_subject < subject.0 {
                    continue;
                }
                let reader = reader_cache.get_or_open_primary(sst.file_uid)?;
                for item in reader.iter_subject(subject) {
                    let (row_subj, predicate, object, payload) =
                        item.map_err(BondReadPathError::from)?;
                    debug_assert_eq!(row_subj, subject, "iter_subject yielded foreign subject");
                    merge_hit(&mut acc, level_idx, (predicate, object.0), payload);
                }
            }
        } else {
            // L1+: non-overlapping -- binary-search the single
            // candidate SST for this level.
            if let Some(sst) = binary_search_level_for_subject(level.ssts.as_slice(), subject) {
                let reader = reader_cache.get_or_open_primary(sst.file_uid)?;
                for item in reader.iter_subject(subject) {
                    let (row_subj, predicate, object, payload) =
                        item.map_err(BondReadPathError::from)?;
                    debug_assert_eq!(row_subj, subject, "iter_subject yielded foreign subject");
                    merge_hit(&mut acc, level_idx, (predicate, object.0), payload);
                }
            }
        }
    }

    Ok(acc
        .into_iter()
        .filter(|(_, payload)| payload.confidence != BOND_TOMBSTONE_CONFIDENCE)
        .map(|((predicate, object), payload)| (subject, predicate, NodeId(object), payload))
        .collect())
}

/// Walk `bond_v5_levels` L0 -> L1 -> ... -> LN collecting every
/// LIVE incoming bond for `object`, in ascending
/// `(predicate, subject)` order. Returned tuples preserve the
/// bond's ORIGINAL orientation: `(subject_in_bond, predicate,
/// object_in_bond == object, payload)`.
///
/// Per-SST access path:
/// - `reverse_file_uid == Some(uid)` (compaction output, Phase
///   3.6): open `<uid>.bsst.reverse` and `iter_subject(object)` --
///   the reverse file's FIRST wire field is semantically the bond's
///   OBJECT (Phase 3.6 swap), so this is an indexed lookup.
/// - `reverse_file_uid == None` (L0 flush output; also pre-3.6
///   compaction outputs promoted by the v3->v4 manifest migration):
///   fall through to a FULL `iter_all()` scan of the primary,
///   keeping rows whose object matches. **Known cost:**
///   O(L0_bonds) per reverse-less SST per query. Acceptable
///   because L0 is bounded by flush frequency and regular L0->L1+
///   compaction prunes it; documented in the Phase 3.7 brief as the
///   honest scope.
///
/// Subject-range metadata cannot pre-filter object queries, so
/// EVERY SST in every level is a candidate (unlike the outgoing
/// walk). Cross-level shadowing and tombstone filtering are
/// identical to [`get_outgoing_bonds_v5`].
///
/// Error contract: eager walk; on the first error the returned
/// iterator yields exactly one `Err` and nothing else.
pub fn get_incoming_bonds_v5<'a>(
    manifest: &'a Manifest,
    reader_cache: &'a BondSstReaderCache,
    object: NodeId,
) -> impl Iterator<Item = Result<(NodeId, u8, NodeId, BondPayload), BondReadPathError>> + 'a {
    let items: Vec<Result<(NodeId, u8, NodeId, BondPayload), BondReadPathError>> =
        match collect_incoming(manifest, reader_cache, object) {
            Ok(rows) => rows.into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        };
    items.into_iter()
}

fn collect_incoming(
    manifest: &Manifest,
    reader_cache: &BondSstReaderCache,
    object: NodeId,
) -> Result<Vec<(NodeId, u8, NodeId, BondPayload)>, BondReadPathError> {
    // Keyed (predicate, subject_in_bond) -- the object is fixed for
    // the whole query, so the pair identifies the triple.
    let mut acc: BTreeMap<(u8, u64), BondPayload> = BTreeMap::new();

    for (level_idx, level) in manifest.bond_v5_levels.iter().enumerate() {
        for sst in &level.ssts {
            match sst.reverse_file_uid {
                Some(rev_uid) => {
                    // Indexed reverse lookup. Wire fields in a
                    // reverse file are (object, predicate, subject).
                    let reader = reader_cache.get_or_open_reverse(rev_uid)?;
                    for item in reader.iter_subject(object) {
                        let (row_obj, predicate, row_subj, payload) =
                            item.map_err(BondReadPathError::from)?;
                        debug_assert_eq!(
                            row_obj, object,
                            "reverse iter_subject yielded foreign object",
                        );
                        merge_hit(&mut acc, level_idx, (predicate, row_subj.0), payload);
                    }
                }
                None => {
                    // L0 fall-through (slow path): scan the primary
                    // for rows whose object matches. O(SST entries)
                    // -- see the doc-comment cost note.
                    let reader = reader_cache.get_or_open_primary(sst.file_uid)?;
                    for item in reader.iter_all() {
                        let (row_subj, predicate, row_obj, payload) =
                            item.map_err(BondReadPathError::from)?;
                        if row_obj != object {
                            continue;
                        }
                        merge_hit(&mut acc, level_idx, (predicate, row_subj.0), payload);
                    }
                }
            }
        }
    }

    Ok(acc
        .into_iter()
        .filter(|(_, payload)| payload.confidence != BOND_TOMBSTONE_CONFIDENCE)
        .map(|((predicate, subj), payload)| (NodeId(subj), predicate, object, payload))
        .collect())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond_store::compaction::{BondCompactionWorker, CompactionPlan, CompactionResult};
    use crate::bond_store::flush::BondFlushWorker;
    use crate::bond_store::sst_writer::{BondSstWriteOpts, BondSstWriter};
    use crate::manifest::{BondV5Level, KeyFingerprint};
    use std::path::Path;
    use tempfile::tempdir;

    fn mk_payload(seq: u64, weight: u8, confidence: u8) -> BondPayload {
        BondPayload {
            weight,
            confidence,
            keyframes: Vec::new(),
            seq,
        }
    }

    /// Write a primary BondSST at `<bond_store_dir>/<file_uid>.bsst`
    /// from rows already sorted ascending by (subject, predicate,
    /// object). Returns a manifest entry (reverse_file_uid = None,
    /// the L0 flush shape).
    fn write_bsst(
        bond_store_dir: &Path,
        file_uid: u64,
        epoch: u64,
        rows: &[(NodeId, u8, NodeId, BondPayload)],
    ) -> BondSSTEntry {
        std::fs::create_dir_all(bond_store_dir).unwrap();
        let path = bond_store_dir.join(bsst_filename(file_uid));
        let opts = BondSstWriteOpts { file_uid, ..Default::default() };
        let mut w = BondSstWriter::create(&path, opts).unwrap();
        for (s, p, o, pay) in rows {
            w.append(*s, *p, *o, pay).unwrap();
        }
        let stats = w.finish().unwrap();
        BondSSTEntry {
            file_uid,
            first_subject: rows.first().map(|r| (r.0).0).unwrap_or(0),
            last_subject: rows.last().map(|r| (r.0).0).unwrap_or(0),
            entry_count: stats.entries_written,
            file_size: stats.file_size_bytes,
            epoch,
            reverse_file_uid: None,
        }
    }

    fn empty_manifest() -> Manifest {
        Manifest::new(KeyFingerprint::none())
    }

    /// Live bonds: subjects in `subjects`, predicate 0, object =
    /// subject + 1_000_000, ascending order.
    fn build_rows(
        subjects: std::ops::Range<u64>,
        base_seq: u64,
    ) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        subjects
            .enumerate()
            .map(|(i, s)| {
                (
                    NodeId(s),
                    0u8,
                    NodeId(s + 1_000_000),
                    mk_payload(base_seq + i as u64, 200, 1),
                )
            })
            .collect()
    }

    fn as_borrow_iter<'a>(
        owned: &'a [(NodeId, u8, NodeId, BondPayload)],
    ) -> impl Iterator<Item = (NodeId, u8, NodeId, &'a BondPayload)> + 'a {
        owned.iter().map(|(s, p, o, pay)| (*s, *p, *o, pay))
    }

    /// Real-rig helper: two L0 flushes (subjects 0..50 and 50..100)
    /// + one compaction to L1. The L1 output carries a REAL Phase
    /// 3.6 reverse file. Returns the compaction result.
    fn flush_two_and_compact(
        bond_store_dir: &Path,
        manifest: &mut Manifest,
    ) -> CompactionResult {
        let fw = BondFlushWorker::new(bond_store_dir.to_path_buf());
        let a_rows = build_rows(0..50, 0);
        let b_rows = build_rows(50..100, 100);
        let a = fw.flush(as_borrow_iter(&a_rows), manifest).unwrap().expect("flush a");
        manifest.add_bond_v5_sst(0, a.clone()).unwrap();
        let b = fw.flush(as_borrow_iter(&b_rows), manifest).unwrap().expect("flush b");
        manifest.add_bond_v5_sst(0, b.clone()).unwrap();
        let worker = BondCompactionWorker::new(bond_store_dir.to_path_buf());
        worker
            .compact(
                CompactionPlan {
                    source_uids: vec![a.file_uid, b.file_uid],
                    source_level: 0,
                    target_level: 1,
                },
                manifest,
                None,
            )
            .expect("compaction")
    }

    fn collect_outgoing_ok(
        manifest: &Manifest,
        cache: &BondSstReaderCache,
        subject: NodeId,
    ) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        get_outgoing_bonds_v5(manifest, cache, subject)
            .collect::<Result<Vec<_>, _>>()
            .expect("outgoing walk errored")
    }

    fn collect_incoming_ok(
        manifest: &Manifest,
        cache: &BondSstReaderCache,
        object: NodeId,
    ) -> Vec<(NodeId, u8, NodeId, BondPayload)> {
        get_incoming_bonds_v5(manifest, cache, object)
            .collect::<Result<Vec<_>, _>>()
            .expect("incoming walk errored")
    }

    /// Test A: L0 hit on get_outgoing_bonds_v5 -- 10 bonds for
    /// subject 5 (plus neighbours that must NOT leak in), yielded
    /// in monotonic (predicate, object) order.
    #[test]
    fn phase37_a_l0_outgoing_hit() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        let mut rows: Vec<(NodeId, u8, NodeId, BondPayload)> = Vec::new();
        rows.push((NodeId(4), 0, NodeId(50), mk_payload(1, 10, 1)));
        for i in 0..10u64 {
            // predicates 0 and 1, objects 100..105 under each.
            rows.push((
                NodeId(5),
                (i / 5) as u8,
                NodeId(100 + (i % 5)),
                mk_payload(2 + i, 20, 1),
            ));
        }
        rows.push((NodeId(6), 0, NodeId(50), mk_payload(99, 10, 1)));
        let entry = write_bsst(&bsd, 0, 0, &rows);
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level { level: 0, ssts: vec![entry] });

        let got = collect_outgoing_ok(&manifest, &cache, NodeId(5));
        assert_eq!(got.len(), 10, "exactly the 10 subject-5 bonds");
        for w in got.windows(2) {
            let a = (w[0].1, (w[0].2).0);
            let b = (w[1].1, (w[1].2).0);
            assert!(a < b, "yield order must be strictly ascending (predicate, object)");
        }
        for (s, _p, _o, pay) in &got {
            assert_eq!(*s, NodeId(5));
            assert_ne!(pay.confidence, BOND_TOMBSTONE_CONFIDENCE);
        }
        // Subject with no bonds anywhere -> empty.
        assert!(collect_outgoing_ok(&manifest, &cache, NodeId(99)).is_empty());
    }

    /// Test B: L1 hit when L0 is empty (post-compaction manifest --
    /// the retired L0 entries are gone, the L1 output serves).
    #[test]
    fn phase37_b_l1_hit_when_l0_empty() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        std::fs::create_dir_all(&bsd).unwrap();
        let mut manifest = empty_manifest();
        let result = flush_two_and_compact(&bsd, &mut manifest);
        assert!(manifest.bond_v5_levels[0].ssts.is_empty(), "L0 retired by compaction");
        assert_eq!(manifest.bond_v5_levels[1].ssts.len(), 1, "single L1 output");

        let cache = BondSstReaderCache::new(bsd.clone());
        let got = collect_outgoing_ok(&manifest, &cache, NodeId(7));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, NodeId(7));
        assert_eq!(got[0].1, 0u8);
        assert_eq!(got[0].2, NodeId(1_000_007));
        // Only the L1 primary was opened.
        assert_eq!(cache.cache_size(), 1);
        cache.evict(result.output_entry.file_uid);
        assert_eq!(cache.cache_size(), 0, "the opened reader was the L1 primary");
    }

    /// Test C: L0 precedence over L1 with seq dedup -- L0 carries
    /// seq=2, L1 carries seq=1 for the same triple; L0 wins.
    #[test]
    fn phase37_c_l0_precedence_over_l1() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        let l0 = vec![(NodeId(5), 0u8, NodeId(9), mk_payload(2, 111, 1))];
        let l1 = vec![(NodeId(5), 0u8, NodeId(9), mk_payload(1, 222, 1))];
        let e0 = write_bsst(&bsd, 0, 1, &l0);
        let e1 = write_bsst(&bsd, 1, 0, &l1);
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level { level: 0, ssts: vec![e0] });
        manifest.bond_v5_levels.push(BondV5Level { level: 1, ssts: vec![e1] });

        let got = collect_outgoing_ok(&manifest, &cache, NodeId(5));
        assert_eq!(got.len(), 1, "one triple, deduped across levels");
        assert_eq!(got[0].3.seq, 2, "L0 row (higher seq) must win");
        assert_eq!(got[0].3.weight, 111, "payload must be the L0 one");
    }

    /// Test C2: WITHIN-L0 seq dedup -- two overlapping L0 SSTs
    /// carry the same triple at seq 2 and seq 7; seq 7 wins
    /// regardless of SST iteration order.
    #[test]
    fn phase37_c2_within_l0_seq_dedup() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        let old = vec![(NodeId(5), 0u8, NodeId(9), mk_payload(2, 111, 1))];
        let new = vec![(NodeId(5), 0u8, NodeId(9), mk_payload(7, 222, 1))];
        let e_old = write_bsst(&bsd, 0, 0, &old);
        let e_new = write_bsst(&bsd, 1, 1, &new);
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level {
            level: 0,
            ssts: vec![e_old, e_new],
        });

        let got = collect_outgoing_ok(&manifest, &cache, NodeId(5));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].3.seq, 7, "higher-seq L0 row must win");
        assert_eq!(got[0].3.weight, 222);
    }

    /// Test D: tombstone in L0 shadows a LIVE lower-level row --
    /// the caller sees NOTHING for that triple, and an unrelated
    /// live L1 triple still surfaces.
    #[test]
    fn phase37_d_tombstone_l0_shadows_l1() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        // L0: tombstone for (5, 0, 9) at seq 5.
        let l0 = vec![(
            NodeId(5),
            0u8,
            NodeId(9),
            mk_payload(5, 100, BOND_TOMBSTONE_CONFIDENCE),
        )];
        // L1: stale live row for the SAME triple at seq 3, plus an
        // unrelated live triple (5, 1, 9) that must survive.
        let l1 = vec![
            (NodeId(5), 0u8, NodeId(9), mk_payload(3, 100, 1)),
            (NodeId(5), 1u8, NodeId(9), mk_payload(4, 50, 1)),
        ];
        let e0 = write_bsst(&bsd, 0, 1, &l0);
        let e1 = write_bsst(&bsd, 1, 0, &l1);
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level { level: 0, ssts: vec![e0] });
        manifest.bond_v5_levels.push(BondV5Level { level: 1, ssts: vec![e1] });

        let got = collect_outgoing_ok(&manifest, &cache, NodeId(5));
        assert_eq!(
            got.len(),
            1,
            "tombstoned triple suppressed; unrelated live triple survives -- got {:?}",
            got.iter().map(|(s, p, o, pay)| (s.0, *p, o.0, pay.seq)).collect::<Vec<_>>(),
        );
        assert_eq!(got[0].1, 1u8, "the surviving row is the (5,1,9) triple");
        assert_ne!(got[0].3.confidence, BOND_TOMBSTONE_CONFIDENCE);
    }

    /// Test E: get_incoming via the L1 reverse file (real Phase 3.6
    /// rig: flush x2 + compact). Original bond orientation is
    /// preserved in the yield.
    #[test]
    fn phase37_e_incoming_via_reverse_file() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        std::fs::create_dir_all(&bsd).unwrap();
        let mut manifest = empty_manifest();
        let result = flush_two_and_compact(&bsd, &mut manifest);
        let rev_uid = result
            .output_entry
            .reverse_file_uid
            .expect("Phase 3.6 compaction output carries a reverse file");

        let cache = BondSstReaderCache::new(bsd.clone());
        let got = collect_incoming_ok(&manifest, &cache, NodeId(1_000_007));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, NodeId(7), "subject_in_bond preserved");
        assert_eq!(got[0].1, 0u8);
        assert_eq!(got[0].2, NodeId(1_000_007), "object_in_bond preserved");
        // Exactly ONE reader was opened, and it was the REVERSE
        // file -- evicting the reverse uid drains the cache.
        assert_eq!(cache.cache_size(), 1);
        cache.evict_reverse(rev_uid);
        assert_eq!(cache.cache_size(), 0, "the opened reader was the reverse file");
    }

    /// Test F: get_incoming L0 fall-through (slow path) -- bonds in
    /// L0 only, no reverse files anywhere; the primary scan serves.
    /// A tombstoned incoming row is filtered.
    #[test]
    fn phase37_f_incoming_l0_fallthrough() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        let rows = vec![
            (NodeId(1), 0u8, NodeId(42), mk_payload(1, 10, 1)),
            (NodeId(2), 0u8, NodeId(42), mk_payload(2, 20, BOND_TOMBSTONE_CONFIDENCE)),
            (NodeId(3), 0u8, NodeId(7), mk_payload(3, 30, 1)),
        ];
        let entry = write_bsst(&bsd, 0, 0, &rows);
        assert!(entry.reverse_file_uid.is_none(), "L0 flush shape has no reverse");
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level { level: 0, ssts: vec![entry.clone()] });

        let got = collect_incoming_ok(&manifest, &cache, NodeId(42));
        assert_eq!(got.len(), 1, "live incoming row only; tombstone filtered");
        assert_eq!(got[0].0, NodeId(1));
        assert_eq!(got[0].2, NodeId(42));
        // The scan opened the PRIMARY (no reverse exists).
        assert_eq!(cache.cache_size(), 1);
        cache.evict(entry.file_uid);
        assert_eq!(cache.cache_size(), 0, "the opened reader was the primary");
        // Object that only appears as a subject -> empty.
        assert!(collect_incoming_ok(&manifest, &cache, NodeId(3)).is_empty());
    }

    /// Test G: cache reuse -- 100 outgoing queries against the same
    /// single SST keep cache_size at 1.
    #[test]
    fn phase37_g_cache_reuse() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let cache = BondSstReaderCache::new(bsd.clone());
        let rows = build_rows(0..100, 0);
        let entry = write_bsst(&bsd, 0, 0, &rows);
        let mut manifest = empty_manifest();
        manifest.bond_v5_levels.push(BondV5Level { level: 0, ssts: vec![entry] });

        for s in 0..100u64 {
            let got = collect_outgoing_ok(&manifest, &cache, NodeId(s));
            assert_eq!(got.len(), 1, "subject {} must hit", s);
        }
        assert_eq!(
            cache.cache_size(),
            1,
            "100 queries against one SST must reuse a single reader",
        );
    }

    /// Test H: empty brain -- no SSTs at all. Both directions
    /// return empty and open nothing.
    #[test]
    fn phase37_h_empty_manifest() {
        let dir = tempdir().unwrap();
        let cache = BondSstReaderCache::new(dir.path().join("bond_store"));
        let manifest = empty_manifest();
        assert!(collect_outgoing_ok(&manifest, &cache, NodeId(5)).is_empty());
        assert!(collect_incoming_ok(&manifest, &cache, NodeId(5)).is_empty());
        assert_eq!(cache.cache_size(), 0, "nothing to open on an empty manifest");
    }

    /// Dual-namespace cache: primary uid 7 and reverse uid 7 are
    /// DIFFERENT files and must coexist as distinct cache entries
    /// (the composite-key decision under test).
    #[test]
    fn phase37_cache_dual_namespace() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let rows = vec![(NodeId(1), 0u8, NodeId(2), mk_payload(1, 1, 1))];
        let _entry = write_bsst(&bsd, 7, 0, &rows);
        // Craft a reverse file that ALSO has uid 7.
        let rpath = bsd.join(bsst_reverse_filename(7));
        let opts = BondSstWriteOpts { file_uid: 7, ..Default::default() };
        let mut w = BondSstWriter::create(&rpath, opts).unwrap();
        w.append(NodeId(2), 0u8, NodeId(1), &mk_payload(1, 1, 1)).unwrap();
        w.finish().unwrap();

        let cache = BondSstReaderCache::new(bsd.clone());
        let p = cache.get_or_open_primary(7).unwrap();
        let r = cache.get_or_open_reverse(7).unwrap();
        assert_eq!(cache.cache_size(), 2, "uid 7 primary and uid 7 reverse coexist");
        assert_ne!(p.path(), r.path(), "distinct files behind the shared uid");
        cache.evict(7);
        assert_eq!(cache.cache_size(), 1, "primary evicted, reverse retained");
        cache.evict_reverse(7);
        assert_eq!(cache.cache_size(), 0);
    }

    /// Evict round-trip: evicted readers re-open on demand.
    #[test]
    fn phase37_cache_evict_round_trip() {
        let dir = tempdir().unwrap();
        let bsd = dir.path().join("bond_store");
        let rows = vec![(NodeId(1), 0u8, NodeId(2), mk_payload(1, 1, 1))];
        let _entry = write_bsst(&bsd, 0, 0, &rows);
        let cache = BondSstReaderCache::new(bsd.clone());
        let _r1 = cache.get_or_open_primary(0).unwrap();
        assert_eq!(cache.cache_size(), 1);
        cache.evict(0);
        assert_eq!(cache.cache_size(), 0);
        let _r2 = cache.get_or_open_primary(0).unwrap();
        assert_eq!(cache.cache_size(), 1);
    }
}
