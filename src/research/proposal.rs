//! ResearchProposal: the human-approval staging area between
//! autonomous fetch and brain ingest.
//!
//! Lifecycle:
//!   Pending → Approved (then Applied when teach() runs) | Rejected
//!
//! Lu can SUGGEST what to learn (by populating bond_candidates) but
//! cannot DECIDE to learn it. Approval is mandatory and routes through
//! `Brain::approve_research_proposal`, which itself calls `teach()`
//! through the scar-aware add_bond path so a scarred concept STILL
//! refuses the bond at ingest even after John approved.
//!
//! The scar fingerprint check happens at PROPOSAL TIME (not at
//! approval time): the brain refuses to put a candidate into a
//! Pending proposal if its fingerprint matches an existing scar's
//! fingerprint above `scar_fingerprint_match_threshold`. This means
//! John never sees a candidate that we'd have to retract. Dropped
//! candidates get counted in `scar_blocked_count` for transparency.

use crate::bond::BondType;
use crate::curiosity::CuriositySignal;
use crate::node::Confidence;
use crate::research::router::{ResearchEvidence, ResearchProvider};
use serde::{Deserialize, Serialize};

/// Port SourceKind-Variants — chain-run provenance attached to a
/// research proposal so `apply_proposal` can stamp every bond with
/// `SourceKind::ChainedResearch` instead of plain `Research`. Stored
/// out-of-band on `Brain::proposal_chain_context` (keyed by proposal
/// id) rather than as a field on `ResearchProposal` itself, because
/// `ResearchProposal` rides through the bincode-serialized v3 brain
/// section #5 — bincode does not honor `#[serde(default)]` for missing
/// trailing fields, so adding a field here would break canonical-brain
/// load of any pre-existing queued proposal. The side-table is
/// in-memory only; chain runs are typically resolved within a single
/// session, and a save/restart between research_one and apply_proposal
/// gracefully falls back to plain SourceKind::Research.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainContext {
    /// Stable id of the chain run that produced this proposal.
    /// Stringified u64 — matches `SourceKind::ChainedResearch.chain_id`.
    pub chain_id: String,
    /// 0-indexed hop within the chain that produced this proposal.
    pub hop: u32,
    /// Surface query researched at this hop (the wonder-queue label).
    pub parent_query: String,
}

/// One bond candidate inside a proposal. The `(from_text, to_text)` pair
/// becomes the subject/object passed to `Brain::teach` if the proposal
/// is approved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondCandidate {
    pub from_text: String,
    pub to_text: String,
    pub bond_type: BondType,
    pub confidence: Confidence,
    pub source_url: String,
    pub source_excerpt: String,
    /// Fingerprint of the source sentence; compared against active
    /// scars at proposal time.
    pub fingerprint: Vec<f32>,
}

/// Status of a research proposal in its lifecycle. The variants carry
/// timestamps and side-effect counts so the audit trail is readable
/// without cross-referencing other logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResearchProposalStatus {
    Pending,
    Approved {
        approved_ms: u64,
    },
    Rejected {
        rejected_ms: u64,
        reason: Option<String>,
    },
    Applied {
        applied_ms: u64,
        bonds_added: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchProposal {
    pub id: u64,
    /// The wonder-queue signal that triggered this research call. `None`
    /// when the proposal was created via the manual `lu4d_research`
    /// path (no underlying signal).
    pub triggering_wonder: Option<CuriositySignal>,
    pub query: String,
    pub source_urls: Vec<String>,
    /// One excerpt per piece of evidence, in router-preference order.
    /// Excerpts are clean prose, not raw HTML.
    pub raw_excerpts: Vec<String>,
    /// The provider every piece of evidence came from, parallel to
    /// `source_urls`/`raw_excerpts`.
    pub providers: Vec<ResearchProvider>,
    pub bond_candidates: Vec<BondCandidate>,
    /// Number of candidates dropped because their fingerprint matched
    /// an active scar. The candidates themselves are NOT retained
    /// (privacy / safety: don't perpetuate a scarred fingerprint).
    pub scar_blocked_count: u32,
    pub created_ms: u64,
    pub status: ResearchProposalStatus,
}

impl ResearchProposal {
    /// True if the proposal has not yet been decided. Used by the
    /// approval-list MCP tool to filter out completed proposals.
    pub fn is_pending(&self) -> bool {
        matches!(self.status, ResearchProposalStatus::Pending)
    }
}

/// Cosine similarity between two equal-length f32 vectors. Returns 0.0
/// if either vector has zero magnitude. Used by the scar-fingerprint
/// check.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Returns true if `from_text` contains a digit+"x" token (e.g. "1x",
/// "30x"). These are Grokipedia footer artifacts ("Fact-checked by
/// Grok 30 days ago Photosynthesis Ara Eve Leo Sal 1x") that bleed
/// into noun-phrase extraction.
pub(crate) fn has_digit_x_token(s: &str) -> bool {
    s.split_whitespace().any(|tok| {
        let t = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let bytes = t.as_bytes();
        if bytes.len() < 2 || bytes[bytes.len() - 1] != b'x' {
            return false;
        }
        bytes[..bytes.len() - 1].iter().all(|c| c.is_ascii_digit())
    })
}

/// Returns true if `sentence` looks like Grokipedia footer cruft.
/// Triggers on:
///   * digit+"x" tokens anywhere in the sentence;
///   * the literal phrase "fact-checked" (case-insensitive);
///   * three or more consecutive Capitalized short alphabetic tokens
///     (length 2-5) — the "Ara Eve Leo Sal" name-list pattern.
/// Source sentences preserve case, unlike subject/object, so this
/// runs against the raw sentence.
pub(crate) fn is_grokipedia_metadata_cruft(sentence: &str) -> bool {
    let trimmed = sentence.trim();
    if trimmed.is_empty() {
        return false;
    }
    if has_digit_x_token(trimmed) {
        return true;
    }
    if trimmed.to_lowercase().contains("fact-checked") {
        return true;
    }
    let toks: Vec<&str> = trimmed.split_whitespace().collect();
    let mut run = 0usize;
    for tok in &toks {
        let t = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let bytes = t.as_bytes();
        let cap_short = !bytes.is_empty()
            && bytes[0].is_ascii_uppercase()
            && bytes.iter().skip(1).all(|c| c.is_ascii_lowercase())
            && bytes.len() >= 2
            && bytes.len() <= 5;
        if cap_short {
            run += 1;
            if run >= 3 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}
/// True if `s` ends with a dangling preposition AND has fewer than 3
/// words — the partial-phrase pattern that produces nonsense bonds.
pub(crate) fn is_dangling_partial_phrase(s: &str) -> bool {
    let trimmed = s.trim().trim_end_matches(|c: char| !c.is_alphanumeric());
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() >= 3 || words.is_empty() {
        return false;
    }
    let last = words[words.len() - 1].to_lowercase();
    matches!(
        last.as_str(),
        "of" | "by" | "for" | "in" | "on" | "at" | "to" | "with" | "from" | "as"
    )
}

/// True if `sentence` looks like a definition of `query_term`:
/// "<query_term> is/are/means ...". Used to bias candidate scoring
/// toward definitional sources. `query_term` may be empty in which
/// case we just look for a generic "<word> is/are/means" prefix.
pub(crate) fn looks_like_definition(sentence: &str, query_term: &str) -> bool {
    let lower = sentence.trim().to_lowercase();
    let query_lower = query_term.trim().to_lowercase();
    let prefix_ok = if query_lower.is_empty() {
        true
    } else {
        lower.starts_with(&query_lower)
    };
    if !prefix_ok {
        return false;
    }
    let after = if query_lower.is_empty() {
        // Check the second token.
        match lower.split_whitespace().nth(1) {
            Some(w) => w,
            None => return false,
        }
    } else {
        let rest = &lower[query_lower.len()..];
        match rest.trim_start().split_whitespace().next() {
            Some(w) => w,
            None => return false,
        }
    };
    matches!(after, "is" | "are" | "means" | "refers")
}

/// Build BondCandidates from extracted triples + their evidence
/// pieces. The fingerprint is computed from the source sentence.
///
/// `scar_fingerprints` is a snapshot of every active scar's
/// fingerprint at proposal-creation time. Candidates whose
/// source-sentence fingerprint scores above `match_threshold` against
/// any scar are dropped; the count of drops is returned alongside the
/// surviving candidates.
pub fn build_candidates(
    triples: &[crate::research::extract::ExtractedTriple],
    evidence: &[ResearchEvidence],
    scar_fingerprints: &[Vec<f32>],
    match_threshold: f32,
    confidence: Confidence,
    cap: usize,
    query_term: &str,
) -> (Vec<BondCandidate>, u32) {
    let mut all: Vec<BondCandidate> = Vec::new();
    let mut blocked: u32 = 0;
    if cap == 0 {
        return (all, 0);
    }
    for trip in triples {
        // Pick the evidence whose excerpt contains the source sentence
        // (case-insensitive); fall back to the first evidence piece.
        let lower_sentence = trip.source_sentence.to_lowercase();
        let mut chosen: Option<&ResearchEvidence> = None;
        for ev in evidence {
            if ev.excerpt.to_lowercase().contains(&lower_sentence) {
                chosen = Some(ev);
                break;
            }
        }
        let ev = match chosen.or_else(|| evidence.first()) {
            Some(e) => e,
            None => continue,
        };
        let fp = crate::research::router::fingerprint_text(&trip.source_sentence);
        // Scar fingerprint check.
        let mut matched = false;
        for scar_fp in scar_fingerprints {
            if cosine_similarity(&fp, scar_fp) > match_threshold {
                matched = true;
                break;
            }
        }
        if matched {
            blocked += 1;
            continue;
        }
        // Metadata cruft filters (Grokipedia headers, dangling
        // preposition partials). Calibration fix 2026-04-25.
        if is_grokipedia_metadata_cruft(&trip.source_sentence)
            || has_digit_x_token(&trip.subject)
            || has_digit_x_token(&trip.object)
        {
            continue;
        }
        if is_dangling_partial_phrase(&trip.subject)
            || is_dangling_partial_phrase(&trip.object)
        {
            continue;
        }
        all.push(BondCandidate {
            from_text: trip.subject.clone(),
            to_text: trip.object.clone(),
            bond_type: trip.bond_type,
            confidence,
            source_url: ev.source_url.clone(),
            source_excerpt: trip.source_sentence.clone(),
            fingerprint: fp,
        });
    }

    // Query-anchoring re-rank. Score each surviving candidate by how
    // well its subject (`from_text`) matches the original query term,
    // then take the top `cap`. Without this, an article like
    // "Photosynthesis" produces SVO triples whose subjects are
    // pronoun-headed phrases ("for this oxygenic process") that form
    // an orphan subgraph disconnected from the photosynthesis node
    // Lu already knows about. Anchoring keeps the proposal connected
    // to the queried concept. Port #14 - 2026-04-25.
    let q_lower = query_term.trim().to_lowercase();
    let q_tokens: Vec<&str> = q_lower
        .split_whitespace()
        .filter(|t| t.len() >= 3)
        .collect();
    let scored: Vec<(u32, BondCandidate)> = all
        .into_iter()
        .map(|mut c| {
            let mut score: u32 = 0;
            if !q_lower.is_empty() {
                let from_lower = c.from_text.to_lowercase();
                if from_lower.contains(&q_lower) {
                    score = 10;
                    // Snap the bond subject to the bare query term so it
                    // attaches to the EXISTING query-named node instead
                    // of forming an orphan with a cruft-prefixed label
                    // (e.g. "[2] photosynthesis" or "primary inputs for
                    // photosynthesis" become just "photosynthesis").
                    if c.from_text.to_lowercase() != q_lower {
                        c.from_text = q_lower.clone();
                    }
                } else if q_tokens
                    .iter()
                    .any(|q| from_lower.split_whitespace().any(|w| w == *q))
                {
                    score = 5;
                }
            }
            (score, c)
        })
        .collect();
    let any_anchored = scored.iter().any(|(s, _)| *s > 0);
    let mut ordered: Vec<BondCandidate> = if any_anchored {
        let mut indexed: Vec<(usize, u32, BondCandidate)> = scored
            .into_iter()
            .enumerate()
            .map(|(i, (s, c))| (i, s, c))
            .collect();
        // Sort by score DESC; original index ASC breaks ties (stable).
        indexed.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        indexed.into_iter().map(|(_, _, c)| c).collect()
    } else {
        scored.into_iter().map(|(_, c)| c).collect()
    };

    // Definitional synthesis fallback: when nothing anchored to the
    // query, scan the first evidence excerpt for "<query> is/are
    // ..." and prepend a synthesized IsA bond. The lead sentence of
    // an article is almost always a definition, so a mostly-orphan
    // proposal still grows the queried concept.
    if !any_anchored {
        if let Some(synth) = synthesize_definition_bond(query_term, evidence, confidence) {
            ordered.insert(0, synth);
        }
    }

    if ordered.len() > cap {
        ordered.truncate(cap);
    }
    (ordered, blocked)
}

/// Scan `evidence` for a sentence that looks like a definition of
/// `query_term` and emit a synthesized IsA bond from it. Returns
/// `None` when no definition sentence is found, when the object would
/// be empty, or when the synthesized object is itself dangling cruft.
fn synthesize_definition_bond(
    query_term: &str,
    evidence: &[ResearchEvidence],
    confidence: Confidence,
) -> Option<BondCandidate> {
    let q_clean = query_term.trim();
    if q_clean.is_empty() {
        return None;
    }
    let q_lower = q_clean.to_lowercase();
    for ev in evidence {
        for raw_sentence in ev.excerpt.split(|c: char| matches!(c, '.' | '!' | '?')) {
            let sentence = raw_sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            if !looks_like_definition(sentence, q_clean) {
                continue;
            }
            let lower = sentence.to_lowercase();
            let after_query = match lower.strip_prefix(&q_lower) {
                Some(s) => s.trim_start(),
                None => continue,
            };
            // Strip the definitional verb + optional article.
            let stripped = [
                "is the ", "is a ", "is an ", "is ",
                "are the ", "are ", "means ", "refers to ",
            ]
            .iter()
            .find_map(|verb| after_query.strip_prefix(*verb).map(|s| s.to_string()))
            .unwrap_or_else(|| after_query.to_string());
            let object: String = stripped
                .split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join(" ");
            if object.trim().is_empty() {
                continue;
            }
            if is_dangling_partial_phrase(&object) {
                continue;
            }
            return Some(BondCandidate {
                from_text: q_lower.clone(),
                to_text: object,
                bond_type: BondType::IsA,
                confidence,
                source_url: ev.source_url.clone(),
                source_excerpt: format!("{}.", sentence),
                fingerprint: crate::research::router::fingerprint_text(sentence),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::extract::ExtractedTriple;
    use crate::research::router::{fingerprint_text, ResearchProvider};

    fn ev_only(text: &str) -> Vec<ResearchEvidence> {
        vec![ResearchEvidence {
            source_url: "https://ex.example/page".to_string(),
            excerpt: text.to_string(),
            provider: ResearchProvider::Grokipedia,
        }]
    }

    fn trip(subject: &str, verb: &str, object: &str, sentence: &str) -> ExtractedTriple {
        ExtractedTriple {
            subject: subject.to_string(),
            predicate: verb.to_string(),
            object: object.to_string(),
            bond_type: BondType::IsA,
            source_sentence: sentence.to_string(),
        }
    }

    #[test]
    fn test_cosine_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_build_candidates_no_scars_yields_all() {
        let trips = vec![trip(
            "rust",
            "is",
            "language",
            "Rust is a systems programming language.",
        )];
        let evidence = ev_only("Rust is a systems programming language.");
        let (c, blocked) =
            build_candidates(&trips, &evidence, &[], 0.85, Confidence::Inferred, 8, "rust");
        assert_eq!(c.len(), 1);
        assert_eq!(blocked, 0);
        assert!(!c[0].fingerprint.is_empty());
    }

    #[test]
    fn test_build_candidates_blocks_scar_match() {
        let sentence = "Forbidden compound is a poison.";
        let trips = vec![trip("forbidden compound", "is", "poison", sentence)];
        let evidence = ev_only(sentence);
        // Build a scar fingerprint from the SAME sentence — guaranteed match.
        let scar_fp = fingerprint_text(sentence);
        let (c, blocked) = build_candidates(
            &trips,
            &evidence,
            &[scar_fp],
            0.85,
            Confidence::Inferred,
            8,
            "",
        );
        assert!(c.is_empty(), "candidate should have been dropped");
        assert_eq!(blocked, 1);
    }

    #[test]
    fn test_build_candidates_respects_cap() {
        let trips: Vec<ExtractedTriple> = (0..10)
            .map(|i| {
                trip(
                    &format!("s{}", i),
                    "is",
                    &format!("o{}", i),
                    &format!("S{} is O{}.", i, i),
                )
            })
            .collect();
        let evidence = ev_only("S0 is O0. S1 is O1. S2 is O2.");
        let (c, _blocked) =
            build_candidates(&trips, &evidence, &[], 0.85, Confidence::Inferred, 3, "");
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn test_research_skips_grokipedia_metadata_cruft() {
        // Real Grokipedia footer pattern: "Leo Sal 1x" was being
        // captured as a noun phrase and ending up as bond candidates.
        let sentence = "Fact-checked by Grok 30 days ago Photosynthesis Ara Eve Leo Sal 1x";
        let trips = vec![
            trip(
                "leo sal 1x",
                "is",
                "photosynthesis",
                sentence,
            ),
            trip(
                "ara eve leo",
                "is",
                "photosynthesis",
                sentence,
            ),
            // A clean triple from the same evidence should still get through.
            trip(
                "photosynthesis",
                "is",
                "the process plants use to make food",
                "Photosynthesis is the process plants use to make food.",
            ),
        ];
        let evidence = vec![
            ResearchEvidence {
                source_url: "https://grokipedia.com/page".into(),
                excerpt: format!(
                    "{} Photosynthesis is the process plants use to make food.",
                    sentence
                ),
                provider: ResearchProvider::Grokipedia,
            },
        ];
        let (c, _blocked) =
            build_candidates(&trips, &evidence, &[], 0.85, Confidence::Inferred, 16, "photosynthesis");
        let froms: Vec<&str> = c.iter().map(|b| b.from_text.as_str()).collect();
        assert!(
            !froms.iter().any(|f| f.contains("1x")),
            "metadata cruft \"1x\" pattern leaked through: {:?}",
            froms,
        );
        assert!(
            !froms.iter().any(|f| f.eq_ignore_ascii_case("ara eve leo")),
            "three-cap-name cruft leaked through: {:?}",
            froms,
        );
        assert!(
            froms.iter().any(|f| f == &"photosynthesis"),
            "clean candidate should survive: {:?}",
            froms,
        );
    }

    #[test]
    fn test_metadata_cruft_helper_recognizes_grok_pattern() {
        // digit-x tokens trigger.
        assert!(is_grokipedia_metadata_cruft("Leo Sal 1x"));
        assert!(has_digit_x_token("leo sal 1x"));
        // 3+ consecutive Capitalized short tokens trigger.
        assert!(is_grokipedia_metadata_cruft("Ara Eve Leo"));
        assert!(is_grokipedia_metadata_cruft(
            "Fact-checked by Grok 30 days ago Photosynthesis Ara Eve Leo Sal 1x"
        ));
        // Real prose does not.
        assert!(!is_grokipedia_metadata_cruft(
            "Photosynthesis is the process plants use to make food."
        ));
        assert!(!is_grokipedia_metadata_cruft("Rust"));
        assert!(!has_digit_x_token("photosynthesis is a process"));
    }

    #[test]
    fn test_dangling_preposition_helper() {
        // Strictly < 3 words, ends with a preposition.
        assert!(is_dangling_partial_phrase("process of"));
        assert!(is_dangling_partial_phrase("made by"));
        // 3 words isn't a "dangling partial" in our spec.
        assert!(!is_dangling_partial_phrase("the process of"));
        assert!(!is_dangling_partial_phrase("the process of photosynthesis"));
        assert!(!is_dangling_partial_phrase("photosynthesis"));
    }

    #[test]
    fn test_looks_like_definition_helper() {
        assert!(looks_like_definition(
            "Photosynthesis is the process plants use",
            "photosynthesis"
        ));
        assert!(looks_like_definition("Rust is a language", "rust"));
        assert!(!looks_like_definition("In photosynthesis plants make food", "photosynthesis"));
    }
    #[test]
    fn test_query_anchoring_prefers_query_subject() {
        // Five trips; two reference the query subject. Anchored
        // candidates should sort to the front and the cap should
        // keep them.
        let trips = vec![
            trip("for this oxygenic process", "produces", "6co2 + 6h2o",
                 "For this oxygenic process produces 6CO2 + 6H2O."),
            trip("photosynthesis", "is", "process",
                 "Photosynthesis is the process plants use."),
            trip("plants", "use", "sunlight",
                 "Plants use sunlight to make food."),
            trip("photosynthesis", "uses", "light",
                 "Photosynthesis uses light energy."),
            trip("three-carbon sugars", "become", "glucose",
                 "Three-carbon sugars become glucose."),
        ];
        let evidence = ev_only(
            "Photosynthesis is the process plants use. Photosynthesis uses light energy. Plants use sunlight to make food. For this oxygenic process produces 6CO2 + 6H2O. Three-carbon sugars become glucose."
        );
        let (c, _blocked) = build_candidates(
            &trips, &evidence, &[], 0.85, Confidence::Inferred, 3, "photosynthesis",
        );
        assert_eq!(c.len(), 3);
        assert!(c[0].from_text.to_lowercase().contains("photosynthesis"),
            "slot 0 should anchor: {:?}",
            c.iter().map(|b| b.from_text.clone()).collect::<Vec<_>>());
        assert!(c[1].from_text.to_lowercase().contains("photosynthesis"),
            "slot 1 should anchor: {:?}",
            c.iter().map(|b| b.from_text.clone()).collect::<Vec<_>>());
    }

    #[test]
    fn test_query_anchoring_falls_back_when_no_anchor() {
        // Trips never mention "rust" - fallback path keeps the
        // original ordering and respects the cap. Synthesis also
        // runs but the evidence has no "rust is/are ..." sentence,
        // so it produces nothing.
        let trips = vec![
            trip("foo", "is", "bar", "Foo is bar."),
            trip("baz", "is", "qux", "Baz is qux."),
        ];
        let evidence = ev_only("Foo is bar. Baz is qux.");
        let (c, _blocked) = build_candidates(
            &trips, &evidence, &[], 0.85, Confidence::Inferred, 8, "rust",
        );
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].from_text, "foo");
        assert_eq!(c[1].from_text, "baz");
    }

    #[test]
    fn test_definitional_synthesis_fires_when_no_anchored() {
        // Trips don't mention photosynthesis at all, but the
        // first-evidence sentence DOES - synthesis should emit a
        // photosynthesis -> IsA bond from it.
        let trips = vec![
            trip("for this process", "produces", "sugars",
                 "For this process produces sugars."),
        ];
        let evidence = ev_only(
            "Photosynthesis is the physico-chemical process by which plants convert solar energy. For this process produces sugars."
        );
        let (c, _blocked) = build_candidates(
            &trips, &evidence, &[], 0.85, Confidence::Inferred, 8, "photosynthesis",
        );
        assert!(!c.is_empty(), "synthesis should add at least one candidate");
        assert_eq!(c[0].from_text, "photosynthesis", "synth bond should be first");
        assert!(matches!(c[0].bond_type, BondType::IsA));
        let to_lower = c[0].to_text.to_lowercase();
        assert!(
            to_lower.contains("physico-chemical") || to_lower.contains("process"),
            "synth object should come from the definition sentence: {:?}",
            c[0].to_text,
        );
    }

    #[test]
    fn test_query_anchoring_normalizes_subject_to_query() {
        // SVO extractor produces noisy subjects with prefix/suffix cruft
        // ("[2] photosynthesis", "primary inputs for photosynthesis").
        // Normalization should snap the anchored subjects to the bare
        // query term so the resulting bonds attach to the EXISTING
        // query-named node instead of forming an orphan subgraph.
        let trips = vec![
            trip("[2] photosynthesis", "is", "indispensable to life",
                 "[2] Photosynthesis is indispensable to life."),
            trip("primary inputs for photosynthesis", "are", "light",
                 "Primary inputs for photosynthesis are light."),
        ];
        let evidence = ev_only(
            "[2] Photosynthesis is indispensable to life. Primary inputs for photosynthesis are light."
        );
        let (c, _blocked) = build_candidates(
            &trips, &evidence, &[], 0.85, Confidence::Inferred, 8, "photosynthesis",
        );
        assert_eq!(c.len(), 2);
        for cand in &c {
            assert_eq!(cand.from_text, "photosynthesis",
                "anchored subject should normalize to bare query: {:?}", cand.from_text);
        }
    }

}
