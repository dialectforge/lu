//! Port S — per-bond source attribution.
//!
//! Lu 4D's `TemporalBond` rkyv layout is locked (the canonical brain
//! at /var/lib/lu/brain.lbr is 1.26 GB v3enc and cannot be retrained
//! in this port). Port S therefore stores source attribution as an
//! *overlay* keyed by `(from, to, bond_type)` rather than as a struct
//! field on `TemporalBond` itself, mirroring the Port G negation
//! overlay pattern (Learning #635).
//!
//! This is pre-retrain schema work (Learning #634, retrain-format-floor):
//! the overlay is the persistence shape today, and at the next retrain
//! the field is baked into the bond struct directly. The overlay can
//! then be folded in via a one-shot migration.
//!
//! WHY the field needs to exist at all (pre-Retrain Roadmap Cat. 2):
//!   1. Audit / source-citation in articulation. "Lu says X based on
//!      <source>."
//!   2. GDPR right-to-be-forgotten: deleting all bonds from doc D
//!      requires per-bond source.
//!   3. Source-trust weighting (Cat. 3) — a multiplier per source kind.
//!   4. Hallucination-benchmark verifiability — "refused with full
//!      attribution" is only marketable if attribution exists.
//!   5. J's confidence aggregation gets richer ("based on 3 sources,
//!      including Wikipedia, NWS, and a teach call").
//!   6. Curiosity engine closes its loop with provenance. Bonds added
//!      by autonomous research carry `Research { wonder_seed, ... }`
//!      so when a research session turns out to be wrong every bond
//!      it produced can be revoked.
//!
//! Backward compat: pre-S brains have an empty source_overlay → all
//! bonds default to `SourceKind::Unknown`. Forward compat: post-S
//! brains save the overlay through serde JSON; pre-S binaries that
//! load a post-S brain skip the field via `#[serde(default)]` in
//! `BrainFile`. The rkyv layout for `TemporalBond` stays byte-identical
//! to pre-S, so the production v3enc brain at /var/lib/lu/brain.lbr
//! loads with `Unknown` for every bond and no migration shim is needed.

use crate::bond::BondType;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Composite key for the source overlay. Carries the bond type so
/// (X IsA Y) and (X RelatedTo Y) can have different sources between
/// the same pair of seeds.
pub type SourceKey = (Seed, Seed, BondType);

/// Categorical kind of provenance for a bond. The overlay stores one
/// of these per `(from, to, bond_type)` triple. Each variant carries
/// the minimum metadata needed for the corresponding downstream use
/// case (audit, GDPR delete, source-trust weighting).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceKind {
    /// Canonical training corpus, identified by name + run id. The
    /// huge majority of bonds in the v3enc brain were produced by
    /// training; this variant is the eventual home for those bonds
    /// after the next retrain bakes the field in. Pre-retrain, an
    /// override can be applied by callers that ingest training data
    /// directly.
    TrainingCorpus {
        corpus: String,
        run_id: Option<String>,
        /// Port T-Ingest: optional per-bond source URL or row identifier within the corpus
        /// (article slug, dataset row id, etc.). Surfaces in source attribution for
        /// GDPR/revoke purposes.  keeps backward compat with pre
        /// T-Ingest brains that may carry a TrainingCorpus entry without this field.
        #[serde(default)]
        url: Option<String>,
    },
    /// Curiosity-loop autonomous research. `wonder_seed` references the
    /// CuriositySignal seed that triggered the research call (so the
    /// loop closes — given a bond, you can locate the wonder that
    /// produced it, and given a wonder, you can revoke every bond it
    /// produced).  `url` is the source page (when known) and
    /// `fetched_at_ms` is the wall-clock fetch time.
    Research {
        wonder_seed: String,
        url: Option<String>,
        fetched_at_ms: u64,
    },
    /// Direct human teach call (admin, operator, MCP `lu4d_teach` tool
    /// invocation). `actor` is whoever performed the teach; `ts_ms`
    /// is when.
    Teach { actor: String, ts_ms: u64 },
    /// Document ingestion (text passed through teach_from_text with a
    /// document handle, file ingest, etc.). `doc_hash` identifies the
    /// document — typically its content hash — so a delete-by-doc API
    /// can find every bond produced by that document.
    Document { doc_hash: String, corpus: Option<String> },
    /// Port SourceKind-Variants — chained-research provenance for
    /// J.2 multi-hop wonder loops. `chain_id` is the stable id of
    /// the chain run that produced this bond; `hop` is the 0-indexed
    /// hop within the chain that landed it; `parent_query` is the
    /// surface query researched at that hop. Replaces the earlier
    /// `SourceKind::Research { wonder_seed: ... }` shape that buried
    /// the chain context in the wonder_seed label. Originating port:
    /// J.2 internal notes.
    ChainedResearch {
        chain_id: String,
        hop: u32,
        parent_query: String,
    },
    /// Port SourceKind-Variants — first-class promotion of a Left
    /// Brain session memory entry (LB.2) to a T.3 PendingChange.
    /// Replaces the earlier `SourceKind::Teach { actor:
    /// "session_promote:S<sid>/E<eid>" }` dodge. `session_id` is
    /// the originating LB session id (stringified); `entry_id` is the
    /// LB session entry id. Originating port: LB.2 internal notes.
    SessionPromote {
        session_id: String,
        entry_id: u64,
    },
    /// Port SourceKind-Variants — first-class signed-pack installation
    /// provenance. Carries the pack id, the publisher's hex Ed25519
    /// pubkey, and the manifest's `signed_at_ms`. Replaces the earlier
    /// `SourceKind::Teach { actor: "pack_install:<pack_id>" }` dodge.
    /// Originating port: Cat.4 internal notes.
    PackInstall {
        pack_id: String,
        publisher_pubkey_hex: String,
        signed_at_ms: u64,
    },
    /// Default for any bond present in the network without an entry
    /// in the overlay. Pre-S bonds (the entire 1.26 GB canonical brain)
    /// surface as Unknown until the next retrain.
    Unknown,
}

impl SourceKind {
    /// Short tag for articulation/structuredContent. Stable across
    /// the variant payload — articulation uses the tag for prose, the
    /// full structured value for structuredContent.
    pub fn tag(&self) -> &'static str {
        match self {
            SourceKind::TrainingCorpus { .. } => "training",
            SourceKind::Research { .. } => "research",
            SourceKind::Teach { .. } => "teach",
            SourceKind::Document { .. } => "document",
            SourceKind::ChainedResearch { .. } => "chained_research",
            SourceKind::SessionPromote { .. } => "session_promote",
            SourceKind::PackInstall { .. } => "pack_install",
            SourceKind::Unknown => "unknown",
        }
    }

    /// Human-friendly label for inclusion in J's confidence-aggregation
    /// articulation. Kept short (≤ 24 chars) so prose reads cleanly.
    pub fn short_label(&self) -> String {
        match self {
            SourceKind::TrainingCorpus { corpus, .. } => corpus.clone(),
            SourceKind::Research { url, .. } => match url {
                Some(u) => {
                    // Pull the host out of the URL for a clean label.
                    if let Some(rest) = u.strip_prefix("https://").or_else(|| u.strip_prefix("http://")) {
                        let host = rest.split('/').next().unwrap_or(rest);
                        host.to_string()
                    } else {
                        "research".to_string()
                    }
                }
                None => "research".to_string(),
            },
            SourceKind::Teach { actor, .. } => format!("teach:{}", actor),
            SourceKind::Document { doc_hash, .. } => {
                let short = doc_hash.chars().take(8).collect::<String>();
                format!("doc:{}", short)
            }
            SourceKind::ChainedResearch { chain_id, hop, .. } => {
                format!("chain:{}:h{}", chain_id, hop)
            }
            SourceKind::SessionPromote { session_id, entry_id } => {
                format!("lb:S{}/E{}", session_id, entry_id)
            }
            SourceKind::PackInstall { pack_id, .. } => {
                format!("pack:{}", pack_id)
            }
            SourceKind::Unknown => "unknown".to_string(),
        }
    }
}

/// Per-bond source attribution. Carries the kind plus a source-trust
/// multiplier. Trust = 1.0 is the neutral baseline.
///
/// Port T-Ingest.1 introduces a sentinel-based discriminator on the
/// existing `f32` field (NOT a struct shape change — wire layout MUST
/// stay f32 because `PendingLog` is bincode-encoded and changing the
/// struct shape would shift every following byte for pre-T-Ingest.1
/// entries on disk):
///
///   * `trust >= 0.0`  — caller resolved or supplied trust; downstream
///                       paths (notably `Brain::teach_with_source`)
///                       treat the value as final and DO NOT re-multiply
///                       weight. Pre-T-Ingest.1 entries (which all
///                       stored `1.0`) hit this branch — backward
///                       compatible.
///   * `trust < 0.0`   — sentinel meaning "needs resolution". Caller
///                       wants `Brain::teach_with_source` to resolve
///                       the multiplier from `Brain::trust_table` at
///                       bond formation time, multiply weight by the
///                       resolved value, and stamp the resolved
///                       multiplier (a non-negative number) onto the
///                       stored attribution. The sentinel never lands
///                       on a stored bond — it is consumed at
///                       formation time.
///
/// `TRUST_NEEDS_RESOLUTION` is the canonical sentinel value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub kind: SourceKind,
    /// Source-trust multiplier in `[0.0, 2.0]`. Negative values are
    /// reserved as the T-Ingest.1 "needs resolution" sentinel; see
    /// the struct documentation.
    pub trust: f32,
}

/// Port T-Ingest.1 sentinel: any negative `SourceAttribution.trust`
/// signals to `Brain::teach_with_source` that the multiplier has not
/// yet been resolved. The chosen value (-1.0) is well clear of the
/// valid `[0.0, 2.0]` range so a `< 0.0` check disambiguates cleanly.
/// Stored bonds NEVER carry this sentinel; teach_with_source consumes
/// it at formation time and stamps the resolved multiplier.
pub const TRUST_NEEDS_RESOLUTION: f32 = -1.0;

impl SourceAttribution {
    /// Build an attribution with the default trust multiplier (1.0).
    pub fn new(kind: SourceKind) -> Self {
        Self { kind, trust: 1.0 }
    }

    /// Convenience constructor for the Teach kind.
    pub fn teach(actor: impl Into<String>, ts_ms: u64) -> Self {
        Self::new(SourceKind::Teach {
            actor: actor.into(),
            ts_ms,
        })
    }

    /// Convenience constructor for the Research kind.
    pub fn research(
        wonder_seed: impl Into<String>,
        url: Option<String>,
        fetched_at_ms: u64,
    ) -> Self {
        Self::new(SourceKind::Research {
            wonder_seed: wonder_seed.into(),
            url,
            fetched_at_ms,
        })
    }

    /// Convenience constructor for the Document kind.
    pub fn document(doc_hash: impl Into<String>, corpus: Option<String>) -> Self {
        Self::new(SourceKind::Document {
            doc_hash: doc_hash.into(),
            corpus,
        })
    }

    /// Convenience constructor for the TrainingCorpus kind (no url).
    pub fn training(corpus: impl Into<String>, run_id: Option<String>) -> Self {
        Self::new(SourceKind::TrainingCorpus {
            corpus: corpus.into(),
            run_id,
            url: None,
        })
    }

    /// Port T-Ingest: convenience constructor for the TrainingCorpus kind with per-bond url.
    pub fn training_with_url(
        corpus: impl Into<String>,
        run_id: Option<String>,
        url: Option<String>,
    ) -> Self {
        Self::new(SourceKind::TrainingCorpus {
            corpus: corpus.into(),
            run_id,
            url,
        })
    }

    /// Port T-Ingest.1: TrainingCorpus constructor that OPTS INTO
    /// table-lookup trust resolution. The returned attribution carries
    /// the `TRUST_NEEDS_RESOLUTION` sentinel; `Brain::teach_with_source`
    /// looks up the multiplier from `corpus_multipliers` (or
    /// `default_multiplier` on miss), multiplies weight at formation,
    /// and stamps the resolved multiplier (a non-negative number) onto
    /// the stored attribution. The sentinel never lands on a stored
    /// bond — only ever observed transiently between this constructor
    /// and the teach_with_source resolution.
    pub fn training_for_resolution(
        corpus: impl Into<String>,
        run_id: Option<String>,
        url: Option<String>,
    ) -> Self {
        Self {
            kind: SourceKind::TrainingCorpus {
                corpus: corpus.into(),
                run_id,
                url,
            },
            trust: TRUST_NEEDS_RESOLUTION,
        }
    }

    /// Port SourceKind-Variants: convenience constructor for J.2
    /// chain-driven research bonds.
    pub fn chained_research(
        chain_id: impl Into<String>,
        hop: u32,
        parent_query: impl Into<String>,
    ) -> Self {
        Self::new(SourceKind::ChainedResearch {
            chain_id: chain_id.into(),
            hop,
            parent_query: parent_query.into(),
        })
    }

    /// Port SourceKind-Variants: convenience constructor for LB.2
    /// session-promote bonds.
    pub fn session_promote(
        session_id: impl Into<String>,
        entry_id: u64,
    ) -> Self {
        Self::new(SourceKind::SessionPromote {
            session_id: session_id.into(),
            entry_id,
        })
    }

    /// Port SourceKind-Variants: convenience constructor for Cat.4
    /// signed-pack-install bonds.
    pub fn pack_install(
        pack_id: impl Into<String>,
        publisher_pubkey_hex: impl Into<String>,
        signed_at_ms: u64,
    ) -> Self {
        Self::new(SourceKind::PackInstall {
            pack_id: pack_id.into(),
            publisher_pubkey_hex: publisher_pubkey_hex.into(),
            signed_at_ms,
        })
    }

    /// The fallback attribution used for any bond missing from the
    /// overlay (i.e., every pre-S bond on disk).
    pub fn unknown() -> Self {
        Self::new(SourceKind::Unknown)
    }
}

/// Side-table mapping `(from, to, bond_type)` -> SourceAttribution.
/// Absent entries default to `SourceKind::Unknown`. Stored sparse so
/// pre-S brains and the canonical 1.26 GB v3enc brain serialize empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceOverlay {
    /// Map from hex-encoded composite key -> attribution. Hex strings
    /// (not `(Seed, Seed, BondType)` tuples) so the JSON path works —
    /// HashMap keys must be string-shaped for serde_json.
    #[serde(default)]
    pub entries: HashMap<String, SourceAttribution>,
}

impl SourceOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode `(from, to, bond_type)` as the JSON-string key shape.
    /// Format matches Port G's negation key: 8 bytes of each seed +
    /// the bond-type debug repr. 64 bits of identity per seed is
    /// enough to avoid collisions in any realistic graph.
    pub fn key(from: &Seed, to: &Seed, bond_type: BondType) -> String {
        let f8 = &from.0[0..8];
        let t8 = &to.0[0..8];
        format!(
            "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}|\
             {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}|{:?}",
            f8[0], f8[1], f8[2], f8[3], f8[4], f8[5], f8[6], f8[7],
            t8[0], t8[1], t8[2], t8[3], t8[4], t8[5], t8[6], t8[7],
            bond_type
        )
    }

    /// Get the attribution for a bond. Returns `Unknown` if the bond
    /// is not in the overlay — the canonical 1.26 GB brain returns
    /// `Unknown` for every bond.
    pub fn get(&self, from: &Seed, to: &Seed, bond_type: BondType) -> SourceAttribution {
        self.entries
            .get(&Self::key(from, to, bond_type))
            .cloned()
            .unwrap_or_else(SourceAttribution::unknown)
    }

    /// True iff an explicit (non-Unknown) attribution is recorded.
    pub fn has(&self, from: &Seed, to: &Seed, bond_type: BondType) -> bool {
        self.entries.contains_key(&Self::key(from, to, bond_type))
    }

    /// Set the attribution for a bond. Storing `SourceKind::Unknown`
    /// removes the entry to keep the overlay sparse.
    pub fn set(
        &mut self,
        from: Seed,
        to: Seed,
        bond_type: BondType,
        attr: SourceAttribution,
    ) {
        let k = Self::key(&from, &to, bond_type);
        if matches!(attr.kind, SourceKind::Unknown) {
            self.entries.remove(&k);
        } else {
            self.entries.insert(k, attr);
        }
    }

    /// Remove every entry for which `predicate` returns true. Returns
    /// the number of entries removed. Backs the
    /// `Brain::revoke_bonds_by_source` API that powers GDPR deletes
    /// and source-trust revocation.
    pub fn revoke_where<F>(&mut self, predicate: F) -> usize
    where
        F: Fn(&SourceAttribution) -> bool,
    {
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, attr)| predicate(attr))
            .map(|(k, _)| k.clone())
            .collect();
        let n = to_remove.len();
        for k in to_remove {
            self.entries.remove(&k);
        }
        n
    }

    /// Distinct source kinds across the supplied bond keys, in the
    /// order the overlay first encountered them. Bonds with no entry
    /// (Unknown) are included in the count once if any chain leg has
    /// no attribution. Used by J's confidence aggregation to render
    /// "based on N sources" without re-walking the whole overlay.
    pub fn distinct_kinds_for_chain(
        &self,
        chain: &[(Seed, Seed, BondType)],
    ) -> Vec<SourceKind> {
        let mut seen: Vec<SourceKind> = Vec::new();
        for (f, t, bt) in chain {
            let k = self.get(f, t, *bt).kind;
            // Compare structurally: two Research entries with different
            // wonder_seed are distinct; two TrainingCorpus with the
            // same corpus + run_id collapse to one.
            if !seen.iter().any(|s| s == &k) {
                seen.push(k);
            }
        }
        seen
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;

    // Phase 1 / Failing-First: bond has no source field, can't query it.
    // The test below would be the FAILING test before this module
    // existed; once SourceOverlay is in place it passes.
    #[test]
    fn pre_s_default_is_unknown() {
        let o = SourceOverlay::new();
        let s1 = Seed::from_label("any");
        let s2 = Seed::from_label("thing");
        let attr = o.get(&s1, &s2, BondType::IsA);
        assert!(matches!(attr.kind, SourceKind::Unknown));
        assert_eq!(attr.trust, 1.0);
        assert!(!o.has(&s1, &s2, BondType::IsA));
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut o = SourceOverlay::new();
        let s1 = Seed::from_label("birds");
        let s2 = Seed::from_label("mammals");
        let attr = SourceAttribution::teach("john", 1_700_000_000_000);
        o.set(s1, s2, BondType::IsA, attr.clone());
        assert!(o.has(&s1, &s2, BondType::IsA));
        let got = o.get(&s1, &s2, BondType::IsA);
        assert_eq!(got, attr);
        assert_eq!(o.len(), 1);
        // Different bond type stays Unknown.
        let other = o.get(&s1, &s2, BondType::HasProperty);
        assert!(matches!(other.kind, SourceKind::Unknown));
    }

    #[test]
    fn setting_unknown_clears_entry() {
        let mut o = SourceOverlay::new();
        let s1 = Seed::from_label("a");
        let s2 = Seed::from_label("b");
        o.set(s1, s2, BondType::IsA, SourceAttribution::teach("j", 1));
        assert_eq!(o.len(), 1);
        o.set(s1, s2, BondType::IsA, SourceAttribution::unknown());
        assert!(o.is_empty());
    }

    #[test]
    fn json_round_trip_preserves_all_variants() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::teach("john", 1700000000000));
        o.set(s("c"), s("d"), BondType::Causes,
              SourceAttribution::research("wonder-1", Some("https://en.wikipedia.org/wiki/Photosynthesis".into()), 1700000001000));
        o.set(s("e"), s("f"), BondType::HasProperty,
              SourceAttribution::document("sha256:deadbeefcafe", Some("medical-corpus".into())));
        o.set(s("g"), s("h"), BondType::PartOf,
              SourceAttribution::training("v4-may2026", Some("r41".into())));
        let json = serde_json::to_string(&o).unwrap();
        let back: SourceOverlay = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 4);
        let r = back.get(&s("c"), &s("d"), BondType::Causes);
        match r.kind {
            SourceKind::Research { wonder_seed, url, fetched_at_ms } => {
                assert_eq!(wonder_seed, "wonder-1");
                assert_eq!(url.as_deref(), Some("https://en.wikipedia.org/wiki/Photosynthesis"));
                assert_eq!(fetched_at_ms, 1700000001000);
            }
            other => panic!("expected Research kind, got {:?}", other),
        }
    }

    #[test]
    fn pre_s_brain_compat_default_empty() {
        // A pre-S brain (no source_overlay field in the JSON) deserializes
        // as an empty overlay — every bond defaults to Unknown.
        let json = "{}";
        let o: SourceOverlay = serde_json::from_str(json).unwrap();
        assert!(o.is_empty());
        let s1 = Seed::from_label("x");
        let s2 = Seed::from_label("y");
        assert!(matches!(o.get(&s1, &s2, BondType::IsA).kind, SourceKind::Unknown));
    }

    #[test]
    fn revoke_by_source_kind() {
        // The deletion API: 5 bonds, 3 different source kinds; revoke
        // by kind A and verify only A's bonds went.
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        // 2 teach bonds ("actor=alpha")
        o.set(s("a1"), s("a2"), BondType::IsA,
              SourceAttribution::teach("alpha", 1));
        o.set(s("a3"), s("a4"), BondType::Causes,
              SourceAttribution::teach("alpha", 2));
        // 2 research bonds ("wonder=W7")
        o.set(s("r1"), s("r2"), BondType::IsA,
              SourceAttribution::research("W7", None, 3));
        o.set(s("r3"), s("r4"), BondType::HasProperty,
              SourceAttribution::research("W7", None, 4));
        // 1 document bond
        o.set(s("d1"), s("d2"), BondType::PartOf,
              SourceAttribution::document("hashX", None));
        assert_eq!(o.len(), 5);

        // Revoke every "teach" bond.
        let n = o.revoke_where(|attr| matches!(attr.kind, SourceKind::Teach { .. }));
        assert_eq!(n, 2);
        assert_eq!(o.len(), 3);

        // The two teach bonds are gone, the others remain.
        assert!(matches!(o.get(&s("a1"), &s("a2"), BondType::IsA).kind, SourceKind::Unknown));
        assert!(matches!(o.get(&s("a3"), &s("a4"), BondType::Causes).kind, SourceKind::Unknown));
        assert!(matches!(o.get(&s("r1"), &s("r2"), BondType::IsA).kind, SourceKind::Research { .. }));
        assert!(matches!(o.get(&s("r3"), &s("r4"), BondType::HasProperty).kind, SourceKind::Research { .. }));
        assert!(matches!(o.get(&s("d1"), &s("d2"), BondType::PartOf).kind, SourceKind::Document { .. }));
    }

    #[test]
    fn revoke_by_specific_wonder_seed() {
        // GDPR-style: revoke every bond from research session W42.
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::research("W42", None, 1));
        o.set(s("c"), s("d"), BondType::Causes,
              SourceAttribution::research("W43", None, 2));
        let n = o.revoke_where(|attr| matches!(
            &attr.kind,
            SourceKind::Research { wonder_seed, .. } if wonder_seed == "W42"
        ));
        assert_eq!(n, 1);
        assert!(matches!(o.get(&s("a"), &s("b"), BondType::IsA).kind, SourceKind::Unknown));
        assert!(matches!(o.get(&s("c"), &s("d"), BondType::Causes).kind, SourceKind::Research { .. }));
    }

    #[test]
    fn distinct_kinds_for_chain_collapses_duplicates() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        // Three bonds in a chain. Two from same wikipedia page, one
        // from a teach call. Should yield 2 distinct kinds.
        let url = Some("https://en.wikipedia.org/wiki/Sky".to_string());
        o.set(s("sky"), s("blue"), BondType::HasProperty,
              SourceAttribution::research("W1", url.clone(), 100));
        o.set(s("blue"), s("color"), BondType::IsA,
              SourceAttribution::research("W1", url.clone(), 100));
        o.set(s("color"), s("vision"), BondType::RelatedTo,
              SourceAttribution::teach("john", 200));
        let chain = vec![
            (s("sky"), s("blue"), BondType::HasProperty),
            (s("blue"), s("color"), BondType::IsA),
            (s("color"), s("vision"), BondType::RelatedTo),
        ];
        let kinds = o.distinct_kinds_for_chain(&chain);
        assert_eq!(kinds.len(), 2);
    }

    #[test]
    fn distinct_kinds_for_chain_includes_unknown() {
        // A pre-S bond in a chain alongside a research bond yields
        // 2 distinct kinds: Research and Unknown.
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::research("W", None, 1));
        // Note: (b -> c) has no overlay entry, so it surfaces as Unknown.
        let chain = vec![
            (s("a"), s("b"), BondType::IsA),
            (s("b"), s("c"), BondType::IsA),
        ];
        let kinds = o.distinct_kinds_for_chain(&chain);
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&SourceKind::Unknown));
    }

    #[test]
    fn short_label_strips_url_to_host() {
        let attr = SourceAttribution::research(
            "W",
            Some("https://en.wikipedia.org/wiki/Photosynthesis".into()),
            0,
        );
        assert_eq!(attr.kind.short_label(), "en.wikipedia.org");
    }

    #[test]
    fn short_label_handles_missing_url() {
        let attr = SourceAttribution::research("W", None, 0);
        assert_eq!(attr.kind.short_label(), "research");
    }

    // -----------------------------------------------------------
    // Port SourceKind-Variants tests
    // -----------------------------------------------------------

    #[test]
    fn variants_tag_table_extended() {
        assert_eq!(
            SourceAttribution::chained_research("c1", 0, "sky color").kind.tag(),
            "chained_research"
        );
        assert_eq!(
            SourceAttribution::session_promote("42", 7).kind.tag(),
            "session_promote"
        );
        assert_eq!(
            SourceAttribution::pack_install("df.test", "ab".repeat(32), 1).kind.tag(),
            "pack_install"
        );
    }

    #[test]
    fn variants_short_label_table() {
        assert_eq!(
            SourceAttribution::chained_research("c42", 3, "q").kind.short_label(),
            "chain:c42:h3"
        );
        assert_eq!(
            SourceAttribution::session_promote("99", 5).kind.short_label(),
            "lb:S99/E5"
        );
        assert_eq!(
            SourceAttribution::pack_install("df.medical", "deadbeef", 0).kind.short_label(),
            "pack:df.medical"
        );
    }

    #[test]
    fn variants_serde_round_trip_chained_research() {
        let kind = SourceKind::ChainedResearch {
            chain_id: "c-1234".into(),
            hop: 2,
            parent_query: "what is photosynthesis".into(),
        };
        let json = serde_json::to_string(&kind).unwrap();
        let back: SourceKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
        // Encode -> decode -> encode equality.
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn variants_serde_round_trip_session_promote() {
        let kind = SourceKind::SessionPromote {
            session_id: "S-77".into(),
            entry_id: 1234,
        };
        let json = serde_json::to_string(&kind).unwrap();
        let back: SourceKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn variants_serde_round_trip_pack_install() {
        let kind = SourceKind::PackInstall {
            pack_id: "df.medical-icd10".into(),
            publisher_pubkey_hex: "ab".repeat(32),
            signed_at_ms: 1_700_000_000_000,
        };
        let json = serde_json::to_string(&kind).unwrap();
        let back: SourceKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn variants_overlay_round_trip_all_three() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::chained_research("c1", 0, "seed query"));
        o.set(s("c"), s("d"), BondType::Causes,
              SourceAttribution::session_promote("S1", 7));
        o.set(s("e"), s("f"), BondType::HasProperty,
              SourceAttribution::pack_install("df.test", "cd".repeat(32), 100));
        let json = serde_json::to_string(&o).unwrap();
        let back: SourceOverlay = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 3);
        match back.get(&s("a"), &s("b"), BondType::IsA).kind {
            SourceKind::ChainedResearch { chain_id, hop, parent_query } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(hop, 0);
                assert_eq!(parent_query, "seed query");
            }
            other => panic!("expected ChainedResearch, got {:?}", other),
        }
        match back.get(&s("c"), &s("d"), BondType::Causes).kind {
            SourceKind::SessionPromote { session_id, entry_id } => {
                assert_eq!(session_id, "S1");
                assert_eq!(entry_id, 7);
            }
            other => panic!("expected SessionPromote, got {:?}", other),
        }
        match back.get(&s("e"), &s("f"), BondType::HasProperty).kind {
            SourceKind::PackInstall { pack_id, publisher_pubkey_hex, signed_at_ms } => {
                assert_eq!(pack_id, "df.test");
                assert_eq!(publisher_pubkey_hex.len(), 64);
                assert_eq!(signed_at_ms, 100);
            }
            other => panic!("expected PackInstall, got {:?}", other),
        }
    }

    #[test]
    fn variants_revoke_where_targets_chained_research_by_chain_id() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        // Two bonds from chain c-A, one from chain c-B, one Research, one Teach.
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::chained_research("c-A", 0, "q1"));
        o.set(s("c"), s("d"), BondType::Causes,
              SourceAttribution::chained_research("c-A", 1, "q2"));
        o.set(s("e"), s("f"), BondType::IsA,
              SourceAttribution::chained_research("c-B", 0, "q3"));
        o.set(s("g"), s("h"), BondType::IsA,
              SourceAttribution::research("W", None, 0));
        o.set(s("i"), s("j"), BondType::IsA,
              SourceAttribution::teach("john", 0));
        let n = o.revoke_where(|attr| matches!(
            &attr.kind,
            SourceKind::ChainedResearch { chain_id, .. } if chain_id == "c-A"
        ));
        assert_eq!(n, 2);
        assert!(matches!(o.get(&s("e"), &s("f"), BondType::IsA).kind,
                         SourceKind::ChainedResearch { .. }));
        assert!(matches!(o.get(&s("g"), &s("h"), BondType::IsA).kind,
                         SourceKind::Research { .. }));
        assert!(matches!(o.get(&s("i"), &s("j"), BondType::IsA).kind,
                         SourceKind::Teach { .. }));
    }

    #[test]
    fn variants_revoke_where_targets_session_promote() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::session_promote("S1", 1));
        o.set(s("c"), s("d"), BondType::IsA,
              SourceAttribution::session_promote("S2", 2));
        let n = o.revoke_where(|attr| matches!(
            &attr.kind,
            SourceKind::SessionPromote { session_id, .. } if session_id == "S1"
        ));
        assert_eq!(n, 1);
    }

    #[test]
    fn variants_revoke_where_targets_pack_install_by_pack_id() {
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::pack_install("df.test", "ab".repeat(32), 1));
        o.set(s("c"), s("d"), BondType::IsA,
              SourceAttribution::pack_install("df.test", "ab".repeat(32), 1));
        o.set(s("e"), s("f"), BondType::IsA,
              SourceAttribution::pack_install("df.other", "cd".repeat(32), 2));
        let n = o.revoke_where(|attr| matches!(
            &attr.kind,
            SourceKind::PackInstall { pack_id, .. } if pack_id == "df.test"
        ));
        assert_eq!(n, 2);
    }

    #[test]
    fn variants_legacy_teach_actor_back_compat_load() {
        // A pre-port brain saved Teach{actor:"session_promote:S1/E1"}
        // and Teach{actor:"pack_install:df.test"} entries in the
        // overlay. The new binary must load these unchanged — no silent
        // rewrite, no failure, no upgrade unless an explicit migrate is
        // invoked.
        let mut o = SourceOverlay::new();
        let s = Seed::from_label;
        o.set(s("a"), s("b"), BondType::IsA,
              SourceAttribution::teach("session_promote:S1/E1", 1));
        o.set(s("c"), s("d"), BondType::IsA,
              SourceAttribution::teach("pack_install:df.test", 2));
        let json = serde_json::to_string(&o).unwrap();
        let back: SourceOverlay = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 2);
        // Stays as Teach — NO retroactive rewrite.
        match back.get(&s("a"), &s("b"), BondType::IsA).kind {
            SourceKind::Teach { ref actor, .. } => assert_eq!(actor, "session_promote:S1/E1"),
            other => panic!("expected Teach, got {:?}", other),
        }
        match back.get(&s("c"), &s("d"), BondType::IsA).kind {
            SourceKind::Teach { ref actor, .. } => assert_eq!(actor, "pack_install:df.test"),
            other => panic!("expected Teach, got {:?}", other),
        }
    }

    #[test]
    fn tag_table() {
        assert_eq!(SourceKind::Unknown.tag(), "unknown");
        assert_eq!(SourceAttribution::teach("j", 0).kind.tag(), "teach");
        assert_eq!(SourceAttribution::research("W", None, 0).kind.tag(), "research");
        assert_eq!(
            SourceAttribution::document("d", None).kind.tag(),
            "document"
        );
        assert_eq!(
            SourceAttribution::training("v4", None).kind.tag(),
            "training"
        );
    }
}
