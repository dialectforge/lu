//! OP-022.b -- global live-node scan over the v5 node store + the v5
//! keyword-index build pipeline.
//!
//! The v3/v4 keyword index is rebuilt by walking the rkyv archive
//! (`AssociativeNetwork::ensure_keyword_index_built`). A v5 brain has
//! NO archive: its nodes live in the LSM node store (NodeMemTable ->
//! L0 NodeSSTs -> L1+ NodeSSTs). This module owns the equivalent walk
//! for v5 -- a streaming k-way merge that yields each LIVE
//! `(seed, content, knowledge_text)` exactly once, applying the same
//! LSM precedence the read path and the compactor use:
//!
//!   * highest `seq` wins per `NodeId` (mirrors `NodeCompactionWorker`'s
//!     dedup rule -- "higher `seq` wins" -- and the read path's
//!     MemTable > L0 > L1+ ordering; `seq` is monotonic, so the two
//!     formulations are equivalent);
//!   * a winning entry that is a tombstone deletes the node -- its
//!     lower-level (older-`seq`) live versions MUST NOT shadow through.
//!
//! `build_v5_keyword_sidecar` drives the existing OP-022.a.2
//! `StreamingRebuilder` (external-mergesort -> mmap `.kw` sidecar) over
//! that scan using the SAME tokenization the query/ingest path uses,
//! so the result is byte-for-byte the index `keyword_search` /
//! `select_seeds_for_query` expect.
//!
//! ## Why a streaming merge (not a HashMap-of-all-nodes)
//!
//! Every source (the NodeMemTable + every NodeSST in every manifest
//! level) yields `(NodeId, NodeEntry)` in ascending `NodeId` order, so
//! one ascending k-way merge groups all versions of a `NodeId`
//! together. We hold at most one record per source plus the current
//! winner -- O(k) resident, never O(store). The heavy `KnowledgeNode`
//! payload (incl. the 384-f32 embedding) is reduced to `(seed, content,
//! knowledge_text)` and the rest dropped the instant a record is built.

use std::sync::Arc;

use crate::keyword_index::{KeywordIndexMmap, StreamingRebuilder};
use crate::manifest::Manifest;
use crate::node_intern::NodeId;
use crate::node_store::memtable::{NodeEntry, NodeMemTable};
use crate::node_store::read_path::{NodeSstReaderCache, ReadPathError};
use crate::node_store::sst_reader::NodeSstReader;
use crate::seed::Seed;

/// Tokenize `s` EXACTLY as `AssociativeNetwork::ensure_keyword_index_built`
/// (the v3/v4 builder) and the query path (`keyword_search` /
/// `select_seeds_for_query`) do: lowercase, split on whitespace, trim
/// leading/trailing non-alphanumerics, keep tokens of length > 1.
/// Mirrored here (rather than shared) so the v3/v4 hot path stays
/// byte-for-byte untouched; the behaviour is identical.
pub(crate) fn tokenize_v5_each(s: &str, mut f: impl FnMut(String)) {
    let lower = s.to_lowercase();
    for w in lower.split_whitespace() {
        let t: String = w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        if !t.is_empty() && t.len() > 1 {
            f(t);
        }
    }
}

/// One normalized record pulled from a merge source. Carries only the
/// fields the keyword build needs (seed + the two text strings) plus the
/// merge keys (`node_id`, `seq`, `rank`, `tombstone`).
struct ScanRec {
    node_id: u64,
    seq: u64,
    rank: u32,
    tombstone: bool,
    seed: Seed,
    content: String,
    ktext: String,
}

impl ScanRec {
    fn from_entry(rank: u32, id: NodeId, e: &NodeEntry) -> ScanRec {
        ScanRec {
            node_id: id.0,
            seq: e.seq,
            rank,
            tombstone: e.tombstone,
            seed: e.payload.seed,
            content: e.payload.content.clone(),
            ktext: e.payload.knowledge.content.text().to_string(),
        }
    }
}

type BoxedSource<'a> =
    std::iter::Peekable<Box<dyn Iterator<Item = Result<ScanRec, ReadPathError>> + 'a>>;

/// Streaming global live-node scan over the v5 node store.
///
/// Invokes `emit(seed, content, knowledge_text)` once per LIVE node,
/// applying LSM precedence (highest `seq` wins per `NodeId`) and
/// tombstone filtering. Returns the number of live nodes emitted.
///
/// Ties on `seq` (which the single-writer store never actually produces)
/// are broken by source `rank`: MemTable (rank 0) first, then NodeSSTs
/// in manifest level order -- a deterministic, precedence-respecting
/// fallback.
pub fn scan_live_v5_nodes<F>(
    memtable: &NodeMemTable,
    manifest: &Manifest,
    cache: &NodeSstReaderCache,
    mut emit: F,
) -> Result<u64, ReadPathError>
where
    F: FnMut(&Seed, &str, &str),
{
    // Open every NodeSST reader up front: a full scan touches every SST
    // (the per-id range pre-check used by the point-lookup read path
    // does not apply). Hold the Arcs in a Vec so each source's
    // `iter_all` borrow stays valid for the whole merge. `rank` grows
    // with precedence depth: MemTable = 0, then L0 SSTs, then L1+ SSTs,
    // in manifest order.
    let mut readers: Vec<(u32, Arc<NodeSstReader>)> = Vec::new();
    let mut rank: u32 = 1;
    for level in &manifest.node_levels {
        for sst in &level.ssts {
            let r = cache.get_or_open(sst.file_uid)?;
            readers.push((rank, r));
            rank += 1;
        }
    }

    // Build the peekable sources. MemTable first (rank 0).
    let mut sources: Vec<BoxedSource<'_>> = Vec::with_capacity(readers.len() + 1);
    let mem_iter = memtable
        .iter()
        .map(|(id, e)| -> Result<ScanRec, ReadPathError> { Ok(ScanRec::from_entry(0, id, e)) });
    let mem_boxed: Box<dyn Iterator<Item = Result<ScanRec, ReadPathError>> + '_> =
        Box::new(mem_iter);
    sources.push(mem_boxed.peekable());
    for (rk, r) in &readers {
        let rk = *rk;
        let it = r.iter_all().map(move |res| {
            res.map(|(id, e)| ScanRec::from_entry(rk, id, &e))
                .map_err(ReadPathError::from)
        });
        let boxed: Box<dyn Iterator<Item = Result<ScanRec, ReadPathError>> + '_> = Box::new(it);
        sources.push(boxed.peekable());
    }

    let mut emitted: u64 = 0;
    loop {
        // Find the minimum NodeId across all live heads. Surface any
        // source error immediately (consume it out and propagate).
        let mut min_id: Option<u64> = None;
        for i in 0..sources.len() {
            let peeked: Option<Result<u64, ()>> = match sources[i].peek() {
                Some(Ok(rec)) => Some(Ok(rec.node_id)),
                Some(Err(_)) => Some(Err(())),
                None => None,
            };
            match peeked {
                Some(Ok(id)) => min_id = Some(min_id.map_or(id, |m| m.min(id))),
                Some(Err(())) => match sources[i].next() {
                    Some(Err(e)) => return Err(e),
                    _ => unreachable!("peeked Err head"),
                },
                None => {}
            }
        }
        let min_id = match min_id {
            Some(x) => x,
            None => break, // all sources exhausted
        };

        // Drain every head equal to `min_id`; keep the highest-seq
        // winner (tie -> lower rank).
        let mut winner: Option<ScanRec> = None;
        for i in 0..sources.len() {
            loop {
                let take = matches!(sources[i].peek(), Some(Ok(r)) if r.node_id == min_id);
                if !take {
                    break;
                }
                let rec = sources[i].next().expect("peeked Some").expect("peeked Ok");
                winner = Some(match winner {
                    None => rec,
                    Some(w) => {
                        if rec.seq > w.seq || (rec.seq == w.seq && rec.rank < w.rank) {
                            rec
                        } else {
                            w
                        }
                    }
                });
            }
        }

        if let Some(w) = winner {
            if !w.tombstone {
                emit(&w.seed, &w.content, &w.ktext);
                emitted += 1;
            }
        }
    }

    Ok(emitted)
}

/// Build the v5 keyword sidecar from the node-store scan, returning the
/// freshly-written mmap base. Drives the OP-022.a.2 `StreamingRebuilder`
/// (external-mergesort -> `.kw` sidecar) over `scan_live_v5_nodes`,
/// tokenizing each live node's `content` and `knowledge` text with the
/// shared `tokenize_v5_each`.
///
/// `sidecar_path` is the destination `.kw` file; `runs_dir` holds the
/// scratch sorted runs (cleaned by the rebuilder's Drop). The
/// `brain_*` stamp values are written into the sidecar trailer so a
/// later load can validate freshness.
#[allow(clippy::too_many_arguments)]
pub fn build_v5_keyword_sidecar(
    memtable: &NodeMemTable,
    manifest: &Manifest,
    cache: &NodeSstReaderCache,
    sidecar_path: std::path::PathBuf,
    runs_dir: std::path::PathBuf,
    brain_crc32: u32,
    brain_node_count: u64,
    brain_bond_count: u64,
) -> std::io::Result<Arc<KeywordIndexMmap>> {
    // 2 million records per buffer -- matches the v3/v4 streaming
    // rebuild path (OP-022.a.2): ~64 B/record => ~128 MiB buffer peak.
    const BUFFER_LIMIT: usize = 2 * 1024 * 1024;
    let mut rb = StreamingRebuilder::new(
        sidecar_path,
        runs_dir,
        BUFFER_LIMIT,
        brain_crc32,
        brain_node_count,
        brain_bond_count,
    )?;

    let mut push_err: Option<std::io::Error> = None;
    let scan = scan_live_v5_nodes(memtable, manifest, cache, |seed, content, ktext| {
        if push_err.is_some() {
            return;
        }
        let seed = *seed;
        // OP-BREADTH Phase-1: entity nodes ("<label>#Q<digits>") are indexed
        // ONLY under the "e_<word>" token namespace + an "eq<qid>" key, so the
        // concept keyword buckets never gain an entity seed (byte-identical
        // concept-path recall). Entities skip knowledge-text (name resolution
        // only). Concept nodes index normally (content + knowledge text).
        if let Some(qid) = crate::network::entity_qid_of(content) {
            let label = crate::network::entity_label_part(content);
            tokenize_v5_each(label, |tok| {
                if push_err.is_none() {
                    if let Err(e) = rb.push(format!("e_{}", tok), seed) {
                        push_err = Some(e);
                    }
                }
            });
            if push_err.is_none() {
                if let Err(e) = rb.push(format!("eq{}", qid), seed) {
                    push_err = Some(e);
                }
            }
        } else {
            tokenize_v5_each(content, |tok| {
                if push_err.is_none() {
                    if let Err(e) = rb.push(tok, seed) {
                        push_err = Some(e);
                    }
                }
            });
            if push_err.is_some() {
                return;
            }
            tokenize_v5_each(ktext, |tok| {
                if push_err.is_none() {
                    if let Err(e) = rb.push(tok, seed) {
                        push_err = Some(e);
                    }
                }
            });
        }
    });

    // Surface a scan (SST decode / IO) error as an io::Error so the
    // caller has one error surface.
    scan.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    if let Some(e) = push_err {
        return Err(e);
    }
    rb.finish()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{KeyFingerprint, NodeLevel, NodeSSTEntry};
    use crate::network::AssociativeNetwork;
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode, KnowledgeUnit};
    use crate::node_store::sst_writer::{CompressionKind, NodeSstWriteOpts, NodeSstWriter};
    use crate::node_store::flush::nsst_filename;
    use std::collections::HashMap;
    use std::path::Path;
    use tempfile::tempdir;

    fn mk_node(label: &str, content: &str) -> KnowledgeNode {
        KnowledgeNode {
            seed: Seed::from_label(label),
            content: content.to_string(),
            knowledge: KnowledgeUnit {
                content: KnowledgeContent::Concept(content.to_string()),
                confidence: Confidence::Verified,
                created_at: 0,
                updated_at: 0,
            },
            embedding: vec![0.0; 384],
            parent_seeds: Vec::new(),
            typed_ratio: 0.0,
        }
    }

    fn live(payload: KnowledgeNode, seq: u64) -> NodeEntry {
        NodeEntry { payload, tombstone: false, seq }
    }
    fn tomb(payload: KnowledgeNode, seq: u64) -> NodeEntry {
        NodeEntry { payload, tombstone: true, seq }
    }

    fn write_sst(
        dir: &Path,
        file_uid: u64,
        epoch: u64,
        entries: &[(NodeId, NodeEntry)],
    ) -> NodeSSTEntry {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(nsst_filename(file_uid));
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

    /// Build a store with a tombstone-shadow case and a higher-seq-L0
    /// override, then assert the scan yields the correct LIVE set with
    /// the right precedence.
    #[test]
    fn scan_lsm_precedence_and_tombstones() {
        let dir = tempdir().unwrap();
        let ns = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(ns.clone());

        // L1: four live nodes (older seq=1).
        let l1 = vec![
            (NodeId(1), live(mk_node("France", "France"), 1)),
            (NodeId(2), live(mk_node("Water", "water"), 1)),
            (NodeId(3), live(mk_node("Ghost", "ghosttoken"), 1)),
            (NodeId(4), live(mk_node("Override", "oldcontent"), 1)),
        ];
        let l1_sst = write_sst(&ns, 10, 0, &l1);
        // L0: tombstone NodeId 3 (shadows L1 live) + higher-seq override
        // of NodeId 4 (newcontent must win over oldcontent).
        let l0 = vec![
            (NodeId(3), tomb(mk_node("Ghost", "ghosttoken"), 5)),
            (NodeId(4), live(mk_node("Override", "newcontent"), 5)),
        ];
        let l0_sst = write_sst(&ns, 20, 1, &l0);

        let mut manifest = Manifest::new(KeyFingerprint::none());
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![l0_sst] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![l1_sst] });

        // MemTable: a fresh node (highest seq) -- exercises the memtable
        // source.
        let mut mt = NodeMemTable::new();
        mt.insert(NodeId(5), mk_node("Mem", "memnode"), 10);

        let mut got: HashMap<Seed, String> = HashMap::new();
        let live_count = scan_live_v5_nodes(&mt, &manifest, &cache, |seed, content, _ktext| {
            got.insert(*seed, content.to_string());
        })
        .expect("scan ok");

        // Four live nodes: France, Water, Override(newcontent), Mem.
        // Ghost (NodeId 3) tombstoned at L0 -> excluded.
        assert_eq!(live_count, 4, "exactly four live nodes");
        assert_eq!(got.get(&Seed::from_label("France")).map(String::as_str), Some("France"));
        assert_eq!(got.get(&Seed::from_label("Water")).map(String::as_str), Some("water"));
        assert_eq!(got.get(&Seed::from_label("Mem")).map(String::as_str), Some("memnode"));
        // Precedence: higher-seq L0 override wins over L1.
        assert_eq!(
            got.get(&Seed::from_label("Override")).map(String::as_str),
            Some("newcontent"),
            "higher-seq L0 must win over L1",
        );
        // Tombstone correctness: Ghost must NOT be emitted.
        assert!(
            !got.contains_key(&Seed::from_label("Ghost")),
            "tombstoned node must not be emitted",
        );
    }

    /// End-to-end: a v5 node store builds a NON-EMPTY keyword index that
    /// resolves a known token to the right Seed; tombstoned/shadowed
    /// tokens resolve to nothing.
    #[test]
    fn build_v5_keyword_index_resolves_tokens() {
        let dir = tempdir().unwrap();
        let ns = dir.path().join("node_store");
        let cache = NodeSstReaderCache::new(ns.clone());

        let l1 = vec![
            (NodeId(1), live(mk_node("France", "France"), 1)),
            (NodeId(2), live(mk_node("Water", "water"), 1)),
            (NodeId(3), live(mk_node("Ghost", "ghosttoken"), 1)),
            (NodeId(4), live(mk_node("Override", "oldcontent"), 1)),
        ];
        let l1_sst = write_sst(&ns, 10, 0, &l1);
        let l0 = vec![
            (NodeId(3), tomb(mk_node("Ghost", "ghosttoken"), 5)),
            (NodeId(4), live(mk_node("Override", "newcontent"), 5)),
        ];
        let l0_sst = write_sst(&ns, 20, 1, &l0);
        let mut manifest = Manifest::new(KeyFingerprint::none());
        manifest.node_levels.push(NodeLevel { level: 0, ssts: vec![l0_sst] });
        manifest.node_levels.push(NodeLevel { level: 1, ssts: vec![l1_sst] });
        let mt = NodeMemTable::new();

        let sidecar = dir.path().join("keyword_index.kw");
        let runs = dir.path().join(".kw_rebuild_runs");
        let mmap = build_v5_keyword_sidecar(
            &mt, &manifest, &cache, sidecar.clone(), runs, 0xABCD_1234, 4, 2,
        )
        .expect("sidecar build ok");

        let mut net = AssociativeNetwork::new();
        net.install_keyword_sidecar(mmap);
        assert!(net.keyword_index_has_base(), "keyword base must be installed");

        // Known token -> right Seed.
        let france = net.keyword_search("France");
        assert!(
            france.iter().any(|(s, _)| *s == Seed::from_label("France")),
            "keyword_search(France) must resolve to the France seed; got {:?}",
            france,
        );
        // Precedence: the live override token resolves; the shadowed
        // older token does not.
        assert!(
            net.keyword_search("newcontent")
                .iter()
                .any(|(s, _)| *s == Seed::from_label("Override")),
            "newcontent (higher-seq) must resolve to Override",
        );
        assert!(
            net.keyword_search("oldcontent").is_empty(),
            "shadowed oldcontent token must NOT resolve",
        );
        // Tombstone: ghosttoken must resolve to nothing.
        assert!(
            net.keyword_search("ghosttoken").is_empty(),
            "tombstoned ghosttoken must NOT resolve",
        );
    }
}
