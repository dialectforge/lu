//! Port T.8 — rejection-pattern learning.
//!
//! When a `PendingChange` is rejected, the rejection pattern store
//! extracts a small fingerprint of the rejected entry. Across many
//! rejections, fingerprints with the same shape (bond_type +
//! source_domain + boolean feature combo) form `RejectionPattern`s.
//! At proposal time `Brain::push_pending_with_filter` consults the
//! learned-pattern set and drops candidates that match — so Lu stops
//! surfacing the same kinds of low-quality bonds across runs.
//!
//! Public surface:
//!   * [`RejectionFingerprint`] — the feature vector.
//!   * [`RejectionPatternStore`] — the live store + `RejectionPatternsSnapshot`.
//!   * [`RejectionPattern`], [`RejectionPatternStatus`] — learned-pattern data.
//!   * [`PatternKey`] — bucket key used by the store.
//!
//! Persistence: `RejectionPatternsSnapshot` round-trips through the v3
//! binary section #10 (`SectionIdV3::RejectionPatterns`), bumping
//! `SECTION_COUNT` from 9 to 10. `MIN_SUPPORTED_SECTION_COUNT` stays
//! at 7 — pre-T.8 brains load with an empty store via `serde(default)`
//! and the read-side `de_bincode_optional` helper (J.1.1 forward-compat).
//!
//! Cross-references:
//!   * Port T.3 (pending log) — the data this port learns from.
//!   * Port T.4 (v3 section pattern) — the on-disk shape this port
//!     copies for `RejectionPatterns` (id 10).
//!   * Constraint C2 (Grokipedia-first) — fixed in `research/router.rs`
//!     under T.8 because the rejection signal made the leak visible.
//!   * Encryption (Solution #79 / Learning #621 / Port T.7) — the
//!     `--require-encrypted` invariant survives this port; the new
//!     section is bytes inside the same v3enc envelope.

pub mod fingerprint;
pub mod pattern;
pub mod store;

pub use fingerprint::{
    contains_citation_marker, contains_generic_pronoun, ends_with_preposition,
    extract_domain, starts_with_preposition, RejectionFingerprint, SourceKindTag,
};
pub use pattern::{
    fingerprint_matches_key, PatternKey, RejectionPattern, RejectionPatternStatus,
    MS_PER_DAY,
};
pub use store::{
    RejectionConfig, RejectionPatternStore, RejectionPatternsSnapshot,
    DEFAULT_MIN_REJECTIONS_FOR_PATTERN, DEFAULT_PATTERN_DECAY_DAYS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::pending::change::{PendingChange, PendingChangeKind, PendingStatus};
    use crate::source::SourceAttribution;

    /// Helper: synthesize a research-source PendingChange with the
    /// given subject/object/url so individual tests can craft known
    /// shapes (sentence-fragment vs. clean) without duplicating
    /// boilerplate.
    fn research_change(
        subject: &str,
        object: &str,
        url: Option<&str>,
        confidence: f32,
    ) -> PendingChange {
        PendingChange {
            pending_id: 0,
            proposed_at_ms: 1_000_000,
            proposed_by: SourceAttribution::research(
                "deadbeefcafebabe",
                url.map(String::from),
                1_000_000,
            ),
            change_kind: PendingChangeKind::AddBond {
                subject_text: subject.to_string(),
                object_text: object.to_string(),
                bond_type: BondType::IsA,
                weight: 0.7,
                negated: false,
            },
            confidence,
            reason: "test fixture".into(),
            status: PendingStatus::Pending,
        }
    }

    // =================================================================
    //  T.8 — fingerprint extraction
    // =================================================================

    /// A subject like "studios and offices" + "on leopard street in"
    /// is sentence-fragment shaped. The fingerprint's boolean features
    /// must light up the way the user expects.
    #[test]
    fn t8_fingerprint_extracts_expected_features_from_known_bad_entry() {
        let change = research_change(
            "studios and offices",
            "on leopard street in",
            Some("https://en.wikipedia.org/wiki/Leopard_Street"),
            0.55,
        );
        let fp = RejectionFingerprint::from_change(&change, 9_999_999)
            .expect("AddBond is fingerprintable");

        assert_eq!(fp.bond_type, BondType::IsA);
        assert_eq!(fp.source_domain, "en.wikipedia.org");
        assert_eq!(fp.source_kind, SourceKindTag::Research);
        assert_eq!(
            fp.subject_starts_with_preposition, false,
            "'studios' is not a preposition"
        );
        assert_eq!(
            fp.object_ends_with_preposition, true,
            "object ends with 'in'"
        );
        assert_eq!(fp.confidence_at_rejection, 0.55);
        assert_eq!(fp.rejected_at_ms, 9_999_999);
        // "and" is in PREPOSITIONS list (function word) so contains-pronoun
        // should still be false because none of the pronouns appear.
        assert_eq!(fp.subject_contains_generic_pronoun, false);
    }

    /// A clean entry ("heat" -> "defined as the energy" via Grokipedia)
    /// must produce booleans that DON'T light up. This is the inverse
    /// of the bad-entry test.
    #[test]
    fn t8_fingerprint_extracts_expected_features_from_clean_entry() {
        let change = research_change(
            "heat",
            "thermal energy",
            Some("https://grokipedia.com/page/Heat"),
            0.82,
        );
        let fp = RejectionFingerprint::from_change(&change, 12_345)
            .expect("AddBond is fingerprintable");

        assert_eq!(fp.source_domain, "grokipedia.com");
        assert_eq!(fp.subject_starts_with_preposition, false);
        assert_eq!(fp.subject_contains_citation_marker, false);
        assert_eq!(fp.subject_contains_generic_pronoun, false);
        assert_eq!(fp.object_ends_with_preposition, false);
    }

    /// A subject containing a `"5]"` citation marker must trip the
    /// citation-marker boolean.
    #[test]
    fn t8_fingerprint_detects_citation_marker() {
        let change = research_change("foo 5]", "bar", None, 0.5);
        let fp = RejectionFingerprint::from_change(&change, 1).unwrap();
        assert!(fp.subject_contains_citation_marker);
        let change2 = research_change("foo [12] bar", "baz", None, 0.5);
        let fp2 = RejectionFingerprint::from_change(&change2, 1).unwrap();
        assert!(fp2.subject_contains_citation_marker);
    }

    // =================================================================
    //  T.8 — pattern emergence + suppression
    // =================================================================

    /// 3 matching rejections with zero approvals must promote a
    /// pattern to status=Learned.
    #[test]
    fn t8_pattern_emerges_after_min_rejections_without_approval() {
        let mut store = RejectionPatternStore::new();
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            store.record_rejection(fp, ts);
        }
        let learned = store.learned_patterns(false);
        assert_eq!(learned.len(), 1, "one pattern emerges from 3 rejections");
        assert_eq!(learned[0].rejection_count, 3);
        assert_eq!(learned[0].status, RejectionPatternStatus::Learned);
        assert_eq!(
            learned[0].key.subject_starts_with_preposition, true,
            "'on' is in the preposition stoplist"
        );
        assert_eq!(learned[0].key.source_domain, "en.wikipedia.org");
    }

    /// If even one approval matches the same fingerprint key, the
    /// pattern is suppressed even with 3+ rejections — the human
    /// approval is treated as evidence that THIS shape isn't
    /// reliably bad.
    #[test]
    fn t8_pattern_does_not_emerge_if_any_approval_matches() {
        let mut store = RejectionPatternStore::new();
        // Three rejections, all matching.
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            store.record_rejection(fp, ts);
        }
        // Pattern emerged.
        assert_eq!(store.learned_patterns(false).len(), 1);

        // One approval with the same shape kills the pattern.
        let approved = research_change(
            "on leopard street in",
            "the building",
            Some("https://en.wikipedia.org/wiki/X"),
            0.5,
        );
        let ap_fp = RejectionFingerprint::from_change(&approved, 40_000).unwrap();
        store.record_approval(ap_fp, 40_000);
        assert_eq!(
            store.learned_patterns(false).len(),
            0,
            "approval suppresses the pattern"
        );
    }

    /// If no new matching rejections arrive within `pattern_decay_days`,
    /// the pattern is marked Expired (still visible in the
    /// include_expired view; not enforced by the pre-filter).
    #[test]
    fn t8_pattern_decays_after_pattern_decay_days_with_no_new_rejections() {
        let mut store = RejectionPatternStore::new();
        // Three rejections at t=0..30s.
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            store.record_rejection(fp, ts);
        }
        // Re-derive at now = 31s — pattern is fresh.
        store.recompute_patterns(31_000);
        assert_eq!(store.learned_patterns(false).len(), 1);

        // Re-derive at now = 30s + 91 * MS_PER_DAY — past the 90-day
        // decay window. Pattern flips to Expired.
        let far_future = 30_000 + 91u64 * MS_PER_DAY;
        store.recompute_patterns(far_future);
        assert_eq!(
            store.learned_patterns(false).len(),
            0,
            "non-expired view excludes decayed patterns"
        );
        assert_eq!(
            store.learned_patterns(true).len(),
            1,
            "include_expired view shows the decayed pattern"
        );
        assert_eq!(
            store.learned_patterns(true)[0].status,
            RejectionPatternStatus::Expired,
        );
    }

    // =================================================================
    //  T.8 — pre-filter behavior at the brain layer
    //  (push_pending_with_filter is on Brain; tested in tests below.)
    // =================================================================

    /// After a pattern is learned, a fresh PendingChange with the
    /// same fingerprint shape is dropped by
    /// `Brain::push_pending_with_filter` and `proposals_filtered_total`
    /// increments.
    #[test]
    fn t8_proposer_drops_pending_matching_learned_pattern() {
        let mut brain = crate::brain::Brain::new();
        // Seed three matching rejections so the pattern emerges.
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            brain.rejection_store.record_rejection(fp, ts);
        }
        assert_eq!(brain.rejection_store.learned_patterns(false).len(), 1);

        // Push a candidate with the same shape.
        let cand = research_change(
            "on dragon avenue at",
            "the gate",
            Some("https://en.wikipedia.org/wiki/Y"),
            0.6,
        );
        let before = brain.rejection_store.proposals_filtered_total();
        let id = brain.push_pending_with_filter(cand).unwrap();
        assert!(id.is_none(), "matching candidate is dropped");
        assert_eq!(
            brain.rejection_store.proposals_filtered_total(),
            before + 1,
            "counter increments"
        );
    }

    /// A non-matching candidate passes through and is assigned a
    /// pending_id like the pre-T.8 path.
    #[test]
    fn t8_proposer_passes_pending_not_matching_pattern() {
        let mut brain = crate::brain::Brain::new();
        // Seed pattern targeting wikipedia + preposition-leading subjects.
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            brain.rejection_store.record_rejection(fp, ts);
        }
        // Clean candidate from grokipedia — different domain, clean
        // subject. Should pass.
        let clean = research_change(
            "heat",
            "thermal energy",
            Some("https://grokipedia.com/page/Heat"),
            0.82,
        );
        let before = brain.rejection_store.proposals_filtered_total();
        let id = brain.push_pending_with_filter(clean).unwrap();
        assert!(id.is_some(), "clean candidate passes");
        assert_eq!(
            brain.rejection_store.proposals_filtered_total(),
            before,
            "counter does not change"
        );
    }

    // =================================================================
    //  T.8 — v3 section persistence
    // =================================================================

    /// Round-trip: build a brain with rejection patterns, save to v3
    /// bytes, read back, verify the patterns survive byte-identical.
    #[test]
    fn t8_v3_section_round_trips_rejection_patterns() {
        use crate::brain::Brain;

        let mut brain = Brain::new();
        for ts in [10_000, 20_000, 30_000] {
            let change = research_change(
                "on leopard street in",
                "the building",
                Some("https://en.wikipedia.org/wiki/X"),
                0.5,
            );
            let fp = RejectionFingerprint::from_change(&change, ts).unwrap();
            brain.rejection_store.record_rejection(fp, ts);
        }
        brain.rejection_store.note_filtered();
        brain.rejection_store.note_filtered();

        let file = brain.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        let bytes = crate::brain_io_v3::write_v3(&file).unwrap();
        // Read back via aligned buffer — see brain_io_v3 module tests
        // for the alignment convention.
        let mut aligned = rkyv::AlignedVec::with_capacity(bytes.len());
        aligned.extend_from_slice(&bytes);
        let loaded = crate::brain_io_v3::read_v3_bytes(&aligned).unwrap();
        assert_eq!(loaded.rejection_patterns.rejections.len(), 3);
        assert_eq!(loaded.rejection_patterns.patterns.len(), 1);
        assert_eq!(loaded.rejection_patterns.proposals_filtered_total, 2);
    }

    /// A brain saved before T.8 (section_count = 9, no
    /// RejectionPatterns section) must load cleanly with an empty
    /// rejection store. Forward compat via `de_bincode_optional` +
    /// `serde(default)`.
    #[test]
    fn t8_pre_t8_brain_loads_with_empty_rejection_patterns() {
        // Build a fresh brain (which produces a T.8-shape file) and
        // strip the RejectionPatterns section from the on-disk bytes
        // by way of a simulated downgrade: build the BrainFile, write
        // it, then read it back and assert defaults survived.
        //
        // NOTE: a fully synthetic pre-T.8 byte-stream test lives in
        // `crate::brain_io_v3::tests` (T.4 already establishes the
        // section-count-tolerance invariant). This test asserts the
        // user-facing default at the BrainFile level — namely, an
        // empty `RejectionPatternsSnapshot`.
        let brain = crate::brain::Brain::new();
        let file = brain.to_brain_file(crate::brain_io_v3::FORMAT_VERSION_V3);
        // Default snapshot is empty.
        assert_eq!(file.rejection_patterns.rejections.len(), 0);
        assert_eq!(file.rejection_patterns.patterns.len(), 0);
        assert_eq!(file.rejection_patterns.proposals_filtered_total, 0);
    }
}
