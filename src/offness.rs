//! OP-OFFNESS-1 — off-ness detector + re-query loop (John's invention).
//!
//! This is the first real implementation in canonical Lu of the mechanism
//! proven in Cattle Dog internal notes: a POST-answer self-check
//! that grounds a produced answer against what Lu actually knows and, when the
//! answer is "off" (unsupported / contradicts stronger grounded evidence),
//! kicks it back to re-query, caps the loop, and falls to an honest "I don't
//! know" — with an optional independent web-research hook.
//!
//! ## Where it sits relative to the K-gate (idk_gate)
//!
//! The two are complementary nets at different points in the pipeline:
//!
//!   * **K-gate (Port K / `idk_gate`) is PRE-answer.** It refuses when RECALL
//!     is too weak to answer at all (nonsense query, no anchors). It fires
//!     *before* narration and early-returns.
//!
//!   * **Off-ness (this module) is POST-answer.** It runs only once the K-gate
//!     has PASSED and an answer has been produced. It asks a different
//!     question: "the answer exists — is it actually grounded in my knowledge,
//!     or did I confidently compose something off?" It is the second net,
//!     strictly *below* the K-gate, and it degrades into the SAME idk floor
//!     (loop-cap -> refuse) so the two share one honest-refusal contract.
//!
//! ## Inert by default
//!
//! Gated by the process env var `LU_OFFNESS_ENABLED` (see [`detector_enabled`]).
//! When unset (the default) the wired consult in `mcp.rs` is a single boolean
//! check and produces ZERO behavior change — the production binary is
//! byte-identical. There is deliberately NO brain-schema / persisted-config
//! field for this flag, so enabling or disabling it never writes the canonical
//! brain.
//!
//! ## Deferred (write-side) work — NOT done here
//!
//!   * Re-embedding the canonical brain with BGE (the KEYSTONE) — cutover step.
//!   * Re-tuning `similarity_floor` on the dense brain. The 0.70 default is
//!     Cattle Dog's sparse-sandbox operating point and is KNOWN not to transfer
//!     directly (#1229/#1232); treat it as a placeholder until re-swept.

use crate::embedding::Embedder;
use std::sync::OnceLock;

/// Process-level enable gate. Reads `LU_OFFNESS_ENABLED` once; the detector is
/// OFF unless the value is `1` or `true` (case-insensitive). No brain write,
/// no config-schema change — flipping this is a pure process concern.
pub fn detector_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LU_OFFNESS_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Process-shared embedder for the off-ness comparison. Built once; the
/// default build yields the deterministic lexical fallback (no heavy deps).
pub fn shared_embedder() -> &'static dyn Embedder {
    static E: OnceLock<Box<dyn Embedder>> = OnceLock::new();
    E.get_or_init(crate::embedding::default_embedder).as_ref()
}

/// Tunables for the off-ness detector.
#[derive(Debug, Clone)]
pub struct OffnessConfig {
    /// Master enable (in addition to the process env gate). Off by default.
    pub enabled: bool,
    /// Cosine floor: the produced answer must reach at least this similarity
    /// to its strongest grounded evidence to count as grounded. Below -> off.
    /// Cattle Dog's 0.70 operating point is the seed value and MUST be
    /// re-tuned on the dense canonical brain (deferred).
    pub similarity_floor: f32,
    /// Maximum re-query iterations before falling to idk.
    pub max_requery_loops: u32,
    /// Whether to invoke the independent web-research hook on terminal off-ness.
    pub use_web_research: bool,
}

impl Default for OffnessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_floor: 0.70,
            max_requery_loops: 1,
            use_web_research: false,
        }
    }
}

/// One grounded evidence snippet drawn from Lu's own knowledge (e.g. an
/// activated node's content label + its activation weight).
#[derive(Debug, Clone)]
pub struct GroundedEvidence {
    pub text: String,
    pub weight: f32,
}

/// Source of grounding evidence for a query. The core detector depends only
/// on this trait so it is fully testable with a mock; the `mcp.rs` wiring
/// implements it from the recall it already computed (read-only, no re-ask).
pub trait GroundingSource {
    fn ground(&self, query: &str) -> Vec<GroundedEvidence>;
}

/// Trivial grounding source over a pre-computed evidence vector. Used by the
/// wiring (which already has the recall) and by tests.
pub struct StaticGrounding(pub Vec<GroundedEvidence>);

impl GroundingSource for StaticGrounding {
    fn ground(&self, _query: &str) -> Vec<GroundedEvidence> {
        self.0.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffnessVerdict {
    /// Answer is supported by grounded evidence — accept it.
    Grounded,
    /// Answer is unsupported — the loop is still trying (transient state,
    /// surfaced for diagnostics; a terminal outcome is Grounded or Refuse).
    Off,
    /// Terminal off-ness after the re-query cap — refuse to idk.
    Refuse,
}

#[derive(Debug, Clone)]
pub struct OffnessSignal {
    pub best_similarity: f32,
    pub evidence_count: usize,
    pub verdict: OffnessVerdict,
    pub loops_used: u32,
}

#[derive(Debug, Clone)]
pub struct OffnessOutcome {
    pub signal: OffnessSignal,
    /// The accepted answer when grounded; `None` when refused.
    pub final_answer: Option<String>,
    /// Refusal prose when the outcome is `Refuse`.
    pub refusal_prose: Option<String>,
    pub requeried: bool,
    pub web_research_invoked: bool,
}

/// Core off-ness assessment: compare `answer` against the strongest grounded
/// evidence. Returns `(best_similarity, is_off)`. Fails CLOSED: with no
/// evidence the answer is treated as off (ungrounded).
pub fn assess(
    embedder: &dyn Embedder,
    answer: &str,
    evidence: &[GroundedEvidence],
    cfg: &OffnessConfig,
) -> (f32, bool) {
    if evidence.is_empty() || answer.trim().is_empty() {
        return (0.0, true);
    }
    let mut best = 0.0f32;
    for e in evidence {
        let s = embedder.similarity(answer, &e.text);
        if s > best {
            best = s;
        }
    }
    (best, best < cfg.similarity_floor)
}

/// Run the full off-ness detector + re-query loop.
///
/// * `requery(query, iteration) -> Option<new_answer>` — the kick-back hook;
///   return `None` to stop re-querying.
/// * `web_research(query) -> Option<resolved_answer>` — independent research
///   hook, consulted once on terminal off-ness when `cfg.use_web_research`.
///
/// Both hooks are closures so the caller controls side effects; the wired
/// consult passes no-ops until the re-query/research paths are enabled and
/// tuned (deferred).
pub fn run<G, R, W>(
    embedder: &dyn Embedder,
    query: &str,
    initial_answer: &str,
    grounder: &G,
    cfg: &OffnessConfig,
    mut requery: R,
    mut web_research: W,
) -> OffnessOutcome
where
    G: GroundingSource,
    R: FnMut(&str, u32) -> Option<String>,
    W: FnMut(&str) -> Option<String>,
{
    let mut answer = initial_answer.to_string();
    let evidence = grounder.ground(query);
    let (mut best, mut is_off) = assess(embedder, &answer, &evidence, cfg);
    let mut loops_used = 0u32;
    let mut requeried = false;

    while is_off && loops_used < cfg.max_requery_loops {
        loops_used += 1;
        match requery(query, loops_used) {
            Some(new_answer) => {
                requeried = true;
                answer = new_answer;
                let ev = grounder.ground(query);
                let (b, off) = assess(embedder, &answer, &ev, cfg);
                best = b;
                is_off = off;
            }
            None => break,
        }
    }

    if !is_off {
        return OffnessOutcome {
            signal: OffnessSignal {
                best_similarity: best,
                evidence_count: evidence.len(),
                verdict: OffnessVerdict::Grounded,
                loops_used,
            },
            final_answer: Some(answer),
            refusal_prose: None,
            requeried,
            web_research_invoked: false,
        };
    }

    // Terminal off-ness: optionally consult the independent research path,
    // then refuse to the idk floor if still ungrounded.
    let mut web_invoked = false;
    if cfg.use_web_research {
        web_invoked = true;
        if let Some(resolved) = web_research(query) {
            let ev = grounder.ground(query);
            let (b, off) = assess(embedder, &resolved, &ev, cfg);
            if !off {
                return OffnessOutcome {
                    signal: OffnessSignal {
                        best_similarity: b,
                        evidence_count: ev.len(),
                        verdict: OffnessVerdict::Grounded,
                        loops_used,
                    },
                    final_answer: Some(resolved),
                    refusal_prose: None,
                    requeried: true,
                    web_research_invoked: true,
                };
            }
        }
    }

    OffnessOutcome {
        signal: OffnessSignal {
            best_similarity: best,
            evidence_count: evidence.len(),
            verdict: OffnessVerdict::Refuse,
            loops_used,
        },
        final_answer: None,
        refusal_prose: Some(refusal_prose(query, best)),
        requeried,
        web_research_invoked: web_invoked,
    }
}

/// Honest refusal prose for terminal off-ness. Deliberately flat and
/// non-moralizing, matching Port K's tone.
pub fn refusal_prose(query: &str, best: f32) -> String {
    format!(
        "I drafted an answer for \"{}\", but I can't ground it against what I \
         actually know (best support {:.2}, below my off-ness threshold), so I \
         won't assert it. I don't have a confident answer to give.",
        query.trim(),
        best
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{FixedEmbedder, LexicalEmbedder};

    fn ev(text: &str, w: f32) -> GroundedEvidence {
        GroundedEvidence { text: text.to_string(), weight: w }
    }

    #[test]
    fn detector_off_by_default() {
        // No env set in the default test process -> inert.
        assert!(!detector_enabled());
    }

    #[test]
    fn grounded_answer_passes() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig { similarity_floor: 0.6, ..Default::default() };
        let evidence = vec![ev("paris is the capital of france", 0.9)];
        let (best, off) = assess(&e, "the capital of france is paris", &evidence, &cfg);
        assert!(best > 0.6, "best {best}");
        assert!(!off);
    }

    #[test]
    fn ungrounded_answer_is_off() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig { similarity_floor: 0.6, ..Default::default() };
        let evidence = vec![ev("paris is the capital of france", 0.9)];
        let (_best, off) = assess(&e, "the mitochondria is the powerhouse of the cell", &evidence, &cfg);
        assert!(off);
    }

    #[test]
    fn no_evidence_fails_closed() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig::default();
        let (best, off) = assess(&e, "anything", &[], &cfg);
        assert_eq!(best, 0.0);
        assert!(off, "no evidence must be treated as off (fail closed)");
    }

    #[test]
    fn off_answer_refuses_when_requery_declines() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig { similarity_floor: 0.6, max_requery_loops: 1, ..Default::default() };
        let grounder = StaticGrounding(vec![ev("paris is the capital of france", 0.9)]);
        let outcome = run(
            &e,
            "capital of france",
            "the powerhouse of the cell is mitochondria",
            &grounder,
            &cfg,
            |_q, _i| None, // re-query declines
            |_q| None,
        );
        assert_eq!(outcome.signal.verdict, OffnessVerdict::Refuse);
        assert!(outcome.final_answer.is_none());
        assert!(outcome.refusal_prose.unwrap().contains("off-ness threshold"));
    }

    #[test]
    fn requery_rescues_off_answer() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig { similarity_floor: 0.6, max_requery_loops: 2, ..Default::default() };
        let grounder = StaticGrounding(vec![ev("paris is the capital of france", 0.9)]);
        let outcome = run(
            &e,
            "capital of france",
            "an off first draft about cells",
            &grounder,
            &cfg,
            |_q, i| {
                // second attempt returns a grounded answer
                if i >= 1 { Some("the capital of france is paris".to_string()) } else { None }
            },
            |_q| None,
        );
        assert_eq!(outcome.signal.verdict, OffnessVerdict::Grounded);
        assert!(outcome.requeried);
        assert_eq!(outcome.final_answer.as_deref(), Some("the capital of france is paris"));
    }

    #[test]
    fn web_research_hook_resolves_terminal_offness() {
        let e = LexicalEmbedder::new();
        let cfg = OffnessConfig {
            similarity_floor: 0.6,
            max_requery_loops: 0,
            use_web_research: true,
            enabled: true,
        };
        let grounder = StaticGrounding(vec![ev("paris is the capital of france", 0.9)]);
        let outcome = run(
            &e,
            "capital of france",
            "an off draft",
            &grounder,
            &cfg,
            |_q, _i| None,
            |_q| Some("the capital of france is paris".to_string()),
        );
        assert!(outcome.web_research_invoked);
        assert_eq!(outcome.signal.verdict, OffnessVerdict::Grounded);
        assert_eq!(outcome.final_answer.as_deref(), Some("the capital of france is paris"));
    }

    #[test]
    fn confident_false_trap_is_caught_on_sample_vectors() {
        // Cattle Dog Test-2 shape (#1228): the model asserts a confident-false
        // answer; Lu holds the truth. With sample vectors the off draft aligns
        // with a distractor, not the grounded truth -> caught.
        let mut fe = FixedEmbedder::new(3);
        fe.insert("strawberry has two rs", vec![0.0, 1.0, 0.0]); // the off answer
        fe.insert("strawberry is spelled s t r a w b e r r y with three rs", vec![1.0, 0.0, 0.0]); // truth
        let cfg = OffnessConfig { similarity_floor: 0.6, max_requery_loops: 0, ..Default::default() };
        let grounder = StaticGrounding(vec![ev(
            "strawberry is spelled s t r a w b e r r y with three rs",
            0.95,
        )]);
        let outcome = run(&fe, "how many rs in strawberry", "strawberry has two rs", &grounder, &cfg, |_q, _i| None, |_q| None);
        assert_eq!(outcome.signal.verdict, OffnessVerdict::Refuse);
    }
}
