//! OP-007 read-only reader — the guest-side, live-serving foundation.
//!
//! This module adds a NON-WRITING way to open a v5 brain DIRECTORY so a
//! second process (the `lu_guest_http` guest binary) can serve `ask` /
//! `narrate` against the *committed on-disk frontier* of a brain that a
//! separate writer process is actively growing — WITHOUT ever touching
//! the writer's files.
//!
//! "brain-is-memory": a second process cannot share the writer's heap, so
//! it reads the committed SSTs referenced by the atomically-swapped
//! MANIFEST. It therefore trails the writer's in-memory MemTable tail by
//! up to one flush interval — the honest, bounded lag of the live view.
//!
//! ## The zero-write guarantee is STRUCTURAL, not a runtime flag
//! A read-only reader is, by construction, a v5 brain that has
//!   * NO flush worker  (`flush_worker == None`), and
//!   * NO WAL append handle (`wal == None`).
//! It physically lacks the machinery that mutates the on-disk brain
//! (this mirrors the OP-007 privsep philosophy: absence of capability,
//! not a policy check). `Brain::is_read_only_reader()` reports it, and
//! the write paths (`save_to_format`, `flush_now`, the v5 keyword-sidecar
//! build) refuse when it holds.
//!
//! What the load path deliberately does NOT do (vs. `Brain::load_v5`):
//!   * no `enable_memtable`  -> no WAL replay, no WAL append handle;
//!   * no `enable_flush_worker` -> no worker thread, no orphan-cleanup
//!     deletes, no fresh-MANIFEST write;
//!   * no `enable_bond_v5_compaction` / `enable_node_v5_compaction`
//!     -> no background compaction (which would rewrite the store);
//!   * no `enable_resident_map_persisted` -> no degmap sidecar write
//!     (the cache-backed path builds its resident map in RAM instead);
//!   * no keyword-sidecar build/write (installs the existing one
//!     read-only if present; otherwise recall is degraded, honestly).

use std::path::Path;

use crate::brain::{Brain, BrainError, BrainLayout};
use crate::encrypt::MasterKey;

impl Brain {
    /// Structural predicate: a v5 brain with neither a flush worker nor a
    /// WAL append handle is a read-only reader — it cannot write the
    /// on-disk brain. Used by the write-path guards.
    pub fn is_read_only_reader(&self) -> bool {
        self.v5_store_root.is_some() && self.flush_worker.is_none() && self.wal.is_none()
    }

    /// Read-only entry that detects the on-disk layout and opens a v5
    /// directory as a non-writing reader. Only v5 directories are
    /// supported for the OP-007 reader foundation; v3/v4 read-only
    /// serving is out of scope.
    pub fn load_readonly(path: &Path, key: Option<MasterKey>) -> Result<Self, BrainError> {
        let layout = Brain::detect_brain_layout(path)?;
        match layout {
            BrainLayout::V5Dir => Brain::load_v5_readonly(path, key),
            other => Err(BrainError::Format(format!(
                "load_readonly supports v5 directories only (got {:?}); v3/v4 \
                 read-only serving is out of scope for the OP-007 reader foundation",
                other
            ))),
        }
    }

    /// Open a v5 brain DIRECTORY as a READ-ONLY reader. Answers
    /// `ask` / `narrate` against the committed frontier (MANIFEST + SSTs)
    /// and performs ZERO writes to `brain_dir`. See the module docs for
    /// the exact list of writer machinery that is intentionally skipped.
    pub fn load_v5_readonly(
        brain_dir: &Path,
        encryption_key: Option<MasterKey>,
    ) -> Result<Self, BrainError> {
        // 1. MANIFEST presence + v5 tag. READ-ONLY: never create one.
        let manifest_path = brain_dir.join(crate::manifest::MANIFEST_FILE_NAME);
        let manifest = match crate::manifest::Manifest::load(&manifest_path)? {
            Some(m) => m,
            None => {
                return Err(BrainError::Format(format!(
                    "read-only reader: v5 brain dir {} has no readable MANIFEST",
                    brain_dir.display()
                )))
            }
        };
        if !manifest.is_v5_format() {
            return Err(BrainError::Format(format!(
                "read-only reader: MANIFEST at {} is not v5 (format_tag {})",
                brain_dir.display(),
                manifest.format_tag
            )));
        }

        // 2. Empty brain shell; stash key + v5 store root.
        let mut brain = Brain::new();
        brain.encryption_key = encryption_key.clone();
        brain.v5_store_root = Some(brain_dir.to_path_buf());

        // 3. Manifest snapshot (in-RAM; NEVER written back) + reader
        //    caches over the committed node_store/ + bond_store/ SSTs.
        //    This is the (a) manifest-install + (b) reader-cache-install
        //    of `enable_flush_worker` WITHOUT (c) the worker thread,
        //    WITHOUT the orphan-cleanup deletes, and WITHOUT the
        //    fresh-MANIFEST `save_atomic` write.
        let snapshot = std::sync::Arc::new(std::sync::RwLock::new(std::sync::Arc::new(manifest)));
        brain.manifest_snapshot = Some(snapshot.clone());

        let node_store_dir = brain_dir.join("node_store");
        let bond_store_dir = brain_dir.join("bond_store");
        brain.node_reader_cache = Some(std::sync::Arc::new(
            crate::node_store::NodeSstReaderCache::new(node_store_dir).with_key(encryption_key.clone()),
        ));
        brain.bond_reader_cache = Some(std::sync::Arc::new(
            crate::bond_store::BondSstReaderCache::new(bond_store_dir).with_key(encryption_key.clone()),
        ));

        // No MemTable (committed frontier only) and no WAL handle.
        brain
            .network
            .install_v4_read_handles(None, Some(snapshot), None);

        // OP-LABEL-1 -- give the network the NODE-store reader cache so
        // `AssociativeNetwork::node_label` can resolve human-readable
        // labels. The cache-backed seam this reader runs on returns
        // `SeamNode::Stub` (content == ""), by design: the BFS hot path
        // needs identity and degree, not text. Only the handful of nodes
        // that reach narration need text, and this handle is how they
        // get it. Pure read -- point lookups into committed NodeSSTs.
        if let Some(nc) = brain.node_reader_cache.as_ref() {
            brain.network.install_node_label_source(nc.clone());
        }

        // 3b. Adopt the writer's CANONICAL NodeIds by merging the
        //     `wal/nodeintern.bin` sidecar into the network intern.
        //     WITHOUT this the reader interns seeds on its own and
        //     disagrees with the writer on `node_intern().len()` — which
        //     is hashed into the keyword-sidecar stamp. The sidecar is
        //     then rejected as "stale" on EVERY open (even for a brain
        //     the writer indexed seconds earlier), the keyword base stays
        //     empty, and every query silently collapses to the idk gate.
        //     Pure read: the sidecar is loaded, never written.
        let ni_sidecar = brain_dir.join("wal").join("nodeintern.bin");
        if ni_sidecar.exists() {
            let loaded = crate::node_intern::NodeIntern::load_from_with_key(
                &ni_sidecar,
                encryption_key.clone(),
            )
            .map_err(|e| {
                BrainError::Format(format!(
                    "read-only reader: nodeintern sidecar load failed at {}: {}",
                    ni_sidecar.display(),
                    e
                ))
            })?;
            let merged = Brain::merge_nodeintern_sidecar(brain.network.node_intern(), &loaded)?;
            eprintln!(
                "OP-007 read-only reader: nodeintern sidecar merged ({} bindings) from {}",
                merged,
                ni_sidecar.display()
            );
        } else {
            eprintln!(
                "OP-007 read-only reader: nodeintern sidecar ABSENT at {} — NodeIds cannot match the                  writer, so the keyword sidecar will be rejected and recall is degraded",
                ni_sidecar.display()
            );
        }

        // 4. Cache-backed adjacency + seam (in-RAM only; reads SSTs, no
        //    disk write). We deliberately SKIP enable_resident_map_persisted
        //    (it writes a degmap sidecar into the brain dir); the
        //    cache-backed path builds its own resident map in RAM and
        //    sources degree from it at seam install.
        brain.enable_cache_backed_adjacency(65_536, 0.001)?;
        brain.enable_adjacency_seam()?;
        brain.set_adjacency_cache_backed(true);

        // 5. V5Only read path + directory marker (makes save() a no-op).
        brain.set_read_path_mode(crate::read_path::ReadPathMode::V5Only);
        brain.brain_dir_v4 = Some(brain_dir.to_path_buf());

        // 6. Keyword index: install the EXISTING sidecar read-only if
        //    valid, so `ensure_keyword_index_built()` early-returns and
        //    `ask`/`narrate` are pure reads. If the sidecar is absent or
        //    stale, `is_read_only_reader()` gates the lazy build so NO
        //    write ever happens (recall degraded, logged honestly).
        brain.install_or_defer_keyword_index_v5();

        // 7. Overlays (read-only sidecar reads).
        let _ = brain.load_overlays_from_dir(brain_dir);

        Ok(brain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::node::Confidence;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::tempdir;

    /// Build a tiny, real v5 brain DIRECTORY via the WRITER path
    /// (`load_v5` + teach + flush), then drop the writer. The dir is left
    /// with a v5 MANIFEST + committed node/bond SSTs, ready to open
    /// read-only.
    fn build_v5_fixture(dir: &Path) {
        std::fs::create_dir_all(dir.join("node_store")).unwrap();
        std::fs::create_dir_all(dir.join("bond_store")).unwrap();
        std::fs::create_dir_all(dir.join("wal")).unwrap();
        let fp = crate::manifest::KeyFingerprint::none();
        let mut m = crate::manifest::Manifest::new(fp);
        m.set_brain_format_v5();
        m.save_atomic(&dir.join(crate::manifest::MANIFEST_FILE_NAME))
            .unwrap();

        let mut brain = Brain::load_v5(dir, None).expect("v5 writer must boot");
        brain.teach("rust", BondType::IsA, "programming language", 0.9, Confidence::Verified);
        brain.teach("rust", BondType::RelatedTo, "memory safety", 0.8, Confidence::Verified);
        brain.teach("python", BondType::IsA, "programming language", 0.9, Confidence::Verified);
        // OP-NEARMISS-1 fixture vocabulary: "language" must be a known
        // term so "launguage" has a legitimate repair target, and
        // "capacitance" gives the adversarial probe a real anchor to
        // sit beside its nonsense.
        brain.teach("python", BondType::UsedFor, "language", 0.9, Confidence::Verified);
        brain.teach("capacitance", BondType::IsA, "electrical property", 0.9, Confidence::Verified);
        // OP-NEARMISS-1: a repair target must clear the knowledge-density
        // floor (`near_miss::MIN_TARGET_SEEDS_D1`), so "language" has to
        // be a term this brain genuinely knows -- not a hapax. Teach
        // enough language-bearing nodes to clear it, mirroring the real
        // brain where `language` carries 140142 seeds.
        for i in 0..(crate::near_miss::MIN_TARGET_SEEDS_D1 + 8) {
            brain.teach(
                &format!("language family {}", i),
                BondType::IsA,
                "language",
                0.8,
                Confidence::Verified,
            );
        }
        // The adversarial vocabulary stays deliberately SPARSE, exactly
        // as corpus noise is on the real brain.
        brain.teach("flugle", BondType::RelatedTo, "trumpet", 0.5, Confidence::Unverified);
        brain.flush_now().expect("flush_now must seal an L0 SST");
        drop(brain);
    }

    /// Boot the WRITER once so it builds + persists the v5 keyword
    /// sidecar. Seed selection is keyword-index driven, so without this
    /// a fixture query resolves no seeds and activates nothing --
    /// which would make the OP-LABEL-1 assertions vacuous.
    fn build_keyword_index_on_writer(dir: &Path) {
        let mut w = Brain::load_v5(dir, None).expect("v5 writer must boot");
        w.ensure_keyword_index_built();
        w.flush_now().ok();
        drop(w);
    }

    /// Snapshot every file under `dir` as path -> (len, mtime_nanos).
    /// Used to prove the read-only reader mutates nothing on disk.
    fn snapshot_dir(dir: &Path) -> BTreeMap<PathBuf, (u64, i128)> {
        fn walk(dir: &Path, out: &mut BTreeMap<PathBuf, (u64, i128)>) {
            let rd = match std::fs::read_dir(dir) {
                Ok(r) => r,
                Err(_) => return,
            };
            for ent in rd.flatten() {
                let p = ent.path();
                let md = match std::fs::metadata(&p) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if md.is_dir() {
                    walk(&p, out);
                } else {
                    let mtime = md
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_nanos() as i128)
                        .unwrap_or(-1);
                    out.insert(p, (md.len(), mtime));
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(dir, &mut out);
        out
    }

    /// OP-LABEL-1 regression. Before the fix, EVERY activated node came
    /// back with `content == ""` on a v5 brain, so narration rendered
    /// `" is ."` with empty label slots. The node text was on disk the
    /// whole time -- the recall path just never asked the node store for
    /// it, because `rehydrate_adjacency` inserts content-free stubs and
    /// the cache-backed seam returns `SeamNode::Stub`.
    ///
    /// This test fails loudly on that shape: it asserts the reader
    /// resolves real labels for the nodes it activates.
    #[test]
    fn readonly_reader_resolves_real_node_labels() {
        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());
        build_keyword_index_on_writer(dir.path());

        let mut ro = Brain::load_v5_readonly(dir.path(), None).expect("read-only load");
        assert!(ro.is_read_only_reader());
        assert!(
            ro.network.has_node_label_source(),
            "OP-LABEL-1: the reader must install a node-store label source"
        );

        // Direct resolution: the seed for a node the fixture taught.
        let rust = crate::seed::Seed::from_label("rust");
        assert_eq!(
            ro.network.node_label(&rust).as_deref(),
            Some("rust"),
            "OP-LABEL-1: node_label must resolve text from the node store"
        );

        // End-to-end: the recall path must carry that text into
        // ActivatedNode.content, which is what narration renders.
        let ask = ro.ask("what is rust", 3, 10);
        assert!(
            !ask.recall.activated.is_empty(),
            "fixture query must activate something"
        );
        let labelled = ask
            .recall
            .activated
            .iter()
            .filter(|a| !a.content.is_empty())
            .count();
        assert_eq!(
            labelled,
            ask.recall.activated.len(),
            "OP-LABEL-1: every activated node must carry a resolved label; \
             got {}/{} labelled. Empty labels are the `\" is .\"` bug.",
            labelled,
            ask.recall.activated.len()
        );
        assert_eq!(
            ro.network.label_store_error_count(),
            0,
            "no node-store errors expected on a healthy fixture"
        );
    }

    /// The WRITER path must resolve labels too -- the empty-label bug was
    /// reproduced writer-side as well, so fixing only the reader would
    /// leave `lu4d_ask` on the training box still rendering blanks.
    #[test]
    fn writer_resolves_real_node_labels() {
        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());
        build_keyword_index_on_writer(dir.path());

        let mut brain = Brain::load_v5(dir.path(), None).expect("v5 writer must boot");
        assert!(
            brain.network.has_node_label_source(),
            "OP-LABEL-1: the writer must install a node-store label source"
        );
        let ask = brain.ask("what is rust", 3, 10);
        assert!(!ask.recall.activated.is_empty());
        for a in &ask.recall.activated {
            assert!(
                !a.content.is_empty(),
                "OP-LABEL-1: writer activated node {:?} has an empty label",
                a.seed
            );
        }
    }

    /// OP-NEARMISS-1 -- the two halves of John's constraint, proven
    /// together on one brain through the real narrate pipeline.
    ///
    /// (a) A near-miss on a term Lu KNOWS must now answer instead of
    ///     refusing the whole question.
    /// (b) Genuine nonsense must STILL refuse. If (b) ever regresses,
    ///     the anti-hallucination floor has been traded away and the
    ///     patch must not ship.
    #[test]
    fn typo_answers_but_genuine_nonsense_still_refuses() {
        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());
        build_keyword_index_on_writer(dir.path());
        let mut ro = Brain::load_v5_readonly(dir.path(), None).expect("read-only load");

        // --- (a) the typo is repaired onto a KNOWN term, and disclosed.
        let ix = ro.network.keyword_index_ref();
        let repaired = crate::near_miss::repair_query(ix, "tell me about the python launguage");
        let (q, reps) = repaired.expect("launguage must repair onto the known term language");
        assert!(q.contains("language"), "repaired query = {:?}", q);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].from, "launguage");
        assert_eq!(reps[0].to, "language");

        let nr = ro.narrate("tell me about the python launguage", 3, 10, false);
        assert!(
            nr.answer.contains("I read \"launguage\" as \"language\""),
            "the repair must be disclosed in the answer; got {:?}",
            nr.answer
        );

        // --- (b) genuine nonsense acquires NO spelling, at any length.
        // "flurgle" is the load-bearing one: this fixture CONTAINS
        // "flugle" one edit away, exactly as the canonical brain does.
        // It must still be refused, on density grounds.
        for nonsense in ["flurgle", "zorbnak", "quibnorth", "blorptastic"] {
            assert!(
                crate::near_miss::nearest_known_term(ro.network.keyword_index_ref(), nonsense)
                    .is_none(),
                "OP-NEARMISS-1 SAFETY VIOLATION: {:?} was given a spelling. \
                 The anti-hallucination floor has been weakened.",
                nonsense
            );
        }
        // ...and the query as a whole is left byte-identical, so the
        // K-gate sees exactly what the user typed.
        assert!(
            crate::near_miss::repair_query(
                ro.network.keyword_index_ref(),
                "how do capacitance and flurgle and zorbnak relate"
            )
            .is_none(),
            "a query whose only unknown tokens are nonsense must not be rewritten"
        );
        let refusal = ro.narrate("how do capacitance and flurgle and zorbnak relate", 3, 10, false);
        assert!(
            !refusal.answer.contains("I read "),
            "no repair may be disclosed for a query that had none: {:?}",
            refusal.answer
        );
    }

    /// A query with NO unknown tokens must be passed through untouched --
    /// spelling repair is only ever allowed to fire on a token the brain
    /// does not already know.
    #[test]
    fn known_queries_are_never_rewritten() {
        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());
        build_keyword_index_on_writer(dir.path());
        let ro = Brain::load_v5_readonly(dir.path(), None).expect("read-only load");
        for q in ["what is rust", "python language", "memory safety", "capacitance"] {
            assert!(
                crate::near_miss::repair_query(ro.network.keyword_index_ref(), q).is_none(),
                "known query {:?} must not be rewritten",
                q
            );
        }
    }

    #[test]
    fn readonly_reader_performs_zero_writes_to_brain_dir() {
        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());

        // Snapshot BEFORE opening the reader.
        let before = snapshot_dir(dir.path());
        assert!(!before.is_empty(), "fixture must have written some files");

        // Open read-only and exercise the read path several times.
        let mut ro = Brain::load_v5_readonly(dir.path(), None).expect("read-only load");
        assert!(ro.is_read_only_reader(), "must report as a read-only reader");
        for q in ["what is rust", "tell me about python", "memory safety", "unknown topic xyz"] {
            // narrate is the guest surface; must not panic and must not write.
            let _ = ro.narrate(q, 3, 10, false);
        }
        // Write paths must be refused structurally.
        assert!(ro.flush_now().is_err(), "flush_now must refuse on a reader");
        assert!(
            ro.save_to_format(dir.path().join("should_not_write.lbr"), crate::brain::BrainSaveFormat::V3BinaryEncrypted)
                .is_err(),
            "save_to_format must refuse on a reader"
        );
        drop(ro);

        // Snapshot AFTER. Nothing may have been added, removed, or changed.
        let after = snapshot_dir(dir.path());
        assert_eq!(
            before, after,
            "read-only reader must not create, delete, or modify ANY file in the brain dir"
        );
        // Explicitly assert no stray sidecars/scratch dirs were created.
        assert!(!dir.path().join(".kw_rebuild_runs").exists(), "no keyword rebuild scratch");
        assert!(!dir.path().join("should_not_write.lbr").exists(), "no refused save artifact");
    }

    #[test]
    fn arcswap_swap_under_load_drops_no_requests_and_no_deadlock() {
        // Exercises the exact guest wiring: ArcSwap<Arc<Mutex<Brain>>>
        // read-only snapshots, hammered by concurrent "request" threads
        // (load -> lock -> narrate) while a swapper thread atomically
        // replaces the live snapshot with a freshly re-opened reader.
        // Asserts every request completes (none dropped) and the run
        // does not deadlock.
        use std::sync::{Mutex, RwLock};

        let dir = tempdir().unwrap();
        build_v5_fixture(dir.path());
        let brain_dir = dir.path().to_path_buf();

        let make = || Arc::new(Mutex::new(Brain::load_v5_readonly(&brain_dir, None).unwrap()));
        // std RwLock<Arc<Mutex<Brain>>>: the RwLock is held only long enough
        // to CLONE the Arc snapshot (nanoseconds), never across narrate; the
        // heavy work runs under the per-snapshot Mutex after the RwLock is
        // released. Swap replaces the Arc; in-flight requests keep their clone.
        let live: Arc<RwLock<Arc<Mutex<Brain>>>> = Arc::new(RwLock::new(make()));

        let completed = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        const READERS: usize = 8;
        const REQS_PER_READER: usize = 25;

        // Swapper thread: re-open the reader and swap it in repeatedly.
        let swapper = {
            let live = live.clone();
            let brain_dir = brain_dir.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut swaps = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let fresh = Arc::new(Mutex::new(
                        Brain::load_v5_readonly(&brain_dir, None).unwrap(),
                    ));
                    *live.write().unwrap() = fresh;
                    swaps += 1;
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                swaps
            })
        };

        // Request threads: load the live snapshot, lock, serve a query.
        let mut handles = Vec::new();
        for _ in 0..READERS {
            let live = live.clone();
            let completed = completed.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..REQS_PER_READER {
                    let guard = live.read().unwrap().clone(); // Arc<Mutex<Brain>>; refresh can't yank it
                    {
                        let mut b = guard.lock().unwrap();
                        let _ = b.narrate("what is rust", 2, 5, false);
                    }
                    completed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for h in handles {
            h.join().expect("request thread must not panic/deadlock");
        }
        stop.store(true, Ordering::Relaxed);
        let swaps = swapper.join().expect("swapper must not panic");

        assert_eq!(
            completed.load(Ordering::Relaxed),
            READERS * REQS_PER_READER,
            "every request must complete under concurrent snapshot swaps"
        );
        assert!(swaps >= 1, "swapper must have swapped at least once");
    }
}
