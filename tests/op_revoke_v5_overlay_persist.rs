//! OP-REVOKE-V5 -- round-trip integration test for the "revocable by
//! source" guarantee on v5 brains.
//!
//! Bug being fixed: `Brain::load_v5` built the brain with an EMPTY
//! `source_overlay` and never repopulated it, and `lu4d_save` is a v5
//! no-op, so the attribution `lu4d_teach_corpus` stamps lived only in
//! RAM. After any restart `revoke_bonds_by_source` (which derives its
//! match set from `source_overlay`) matched ZERO -- silently breaking
//! GDPR delete-by-corpus / source-trust revocation.
//!
//! Fix: persist the overlay to an encrypted side-table on the save
//! checkpoint (`Brain::persist_overlays`) and reload it in `load_v5`
//! (`Brain::load_overlays_from_dir`).
//!
//! This test drives the genuine on-disk round trip: every "reload" is
//! a fresh `Brain::load_v5` reading the directory back from disk (a new
//! in-memory Brain, exactly what a fresh process gets). It asserts the
//! BEFORE behaviour (no-persist reload -> 0 matches, the original bug)
//! and the AFTER behaviour (persist + reload -> non-zero matches, and
//! `revoke_bonds_by_source` removes them).

use std::path::Path;

use lu4d::brain::Brain;
use lu4d::manifest::{KeyFingerprint, Manifest, MANIFEST_FILE_NAME};
use lu4d::source::{SourceAttribution, SourceKind};
use lu4d::{BondType, Confidence};

const CORPUS: &str = "test-corpus-A";

/// Boot a v5 brain from `dir`, creating the v5 MANIFEST on first call.
fn boot_v5(dir: &Path) -> Brain {
    let mpath = dir.join(MANIFEST_FILE_NAME);
    if !mpath.exists() {
        let mut m = Manifest::new(KeyFingerprint::none());
        m.set_brain_format_v5();
        m.save_atomic(&mpath).expect("write v5 MANIFEST");
    }
    Brain::load_v5(dir, None).expect("v5 brain must boot via load_v5")
}

/// Predicate: a TrainingCorpus attribution tagged with our test corpus
/// -- the exact shape `lu4d_revoke_by_source kind=training corpus=...`
/// builds.
fn is_test_corpus(attr: &SourceAttribution) -> bool {
    matches!(
        &attr.kind,
        SourceKind::TrainingCorpus { corpus, .. } if corpus == CORPUS
    )
}

/// Count source_overlay entries attributed to our test corpus (the
/// dry-run "matched_count").
fn corpus_match_count(brain: &Brain) -> usize {
    brain
        .source_overlay
        .read()
        .unwrap()
        .entries
        .values()
        .filter(|a| is_test_corpus(a))
        .count()
}

/// Teach a handful of distinctively-tagged corpus bonds.
fn teach_corpus_bonds(brain: &mut Brain) -> usize {
    let triples = [
        ("alpha", "letter"),
        ("beta", "letter"),
        ("gamma", "letter"),
        ("delta", "letter"),
    ];
    for (s, o) in triples {
        brain.teach_corpus(
            s,
            BondType::IsA,
            o,
            0.9,
            Confidence::Verified,
            CORPUS,
            Some("run-A".to_string()),
            None,
        );
    }
    triples.len()
}

#[test]
fn revoke_by_source_survives_v5_reload() {
    let dir = tempfile::tempdir().unwrap();

    // ---- 1. Teach corpus-tagged bonds, confirm they are attributed. ----
    let taught = {
        let mut brain = boot_v5(dir.path());
        let n = teach_corpus_bonds(&mut brain);
        let live = corpus_match_count(&brain);
        assert!(
            live >= n,
            "freshly taught corpus bonds must be attributed in source_overlay \
             (got {} attributions for {} bonds)",
            live, n,
        );
        // NOTE: intentionally do NOT persist here -- this models a restart
        // WITHOUT a save (and, equivalently, the PRE-FIX behaviour where
        // a save would not have persisted the overlay anyway).
        live
    };

    // ---- 2. BEFORE: reload without having persisted -> attribution lost. ----
    {
        let brain = boot_v5(dir.path());
        let before = corpus_match_count(&brain);
        assert_eq!(
            before, 0,
            "BEFORE FIX semantics: a reload with no persisted overlay loses ALL \
             source attribution (this is exactly the bug: revoke would match 0)",
        );
        println!(
            "[before/after] taught={} ; reload-without-persist matched={} (the bug)",
            taught, before,
        );
    }

    // ---- 3. Re-teach, PERSIST (the lu4d_save checkpoint), then reload. ----
    let after_persist = {
        let mut brain = boot_v5(dir.path());
        let n = teach_corpus_bonds(&mut brain);
        brain.persist_overlays().expect("persist_overlays (lu4d_save hook)");
        let live = corpus_match_count(&brain);
        assert!(live >= n, "re-taught bonds attributed before reload");
        live
    };

    // ---- 4. AFTER: fresh reload -> attribution restored from sidecar. ----
    {
        let mut brain = boot_v5(dir.path());
        let reloaded = corpus_match_count(&brain);
        assert!(
            reloaded > 0,
            "AFTER FIX: source attribution MUST survive a fresh load_v5 \
             (matched {} after reload)",
            reloaded,
        );
        assert_eq!(
            reloaded, after_persist,
            "every persisted attribution must round-trip exactly",
        );
        println!(
            "[before/after] reload-after-persist matched={} (fixed)",
            reloaded,
        );

        // dry_run=true equivalent: the overlay predicate matches non-zero.
        let dry_run_matched = corpus_match_count(&brain);
        assert!(
            dry_run_matched > 0,
            "lu4d_revoke_by_source dry_run=true MUST match the taught bonds",
        );

        // dry_run=false equivalent: actually revoke and confirm removal.
        let (overlay_removed, _bonds_removed) =
            brain.revoke_bonds_by_source(is_test_corpus);
        assert!(
            overlay_removed > 0,
            "revoke_bonds_by_source must remove the restored attributions \
             (removed {})",
            overlay_removed,
        );
        assert_eq!(
            corpus_match_count(&brain),
            0,
            "overlay clear after revoke",
        );
        println!(
            "[before/after] revoke removed {} overlay attributions",
            overlay_removed,
        );
    }
}
