//! Gap signal bus — operational infrastructure for routing reasoning
//! gaps into the curiosity engine.
//!
//! Lu's curiosity engine (`curiosity.rs`) only fires inside the
//! `Brain::ask` recall path. The `reason()` family (Ports A–E, plus
//! H/I/L/M/F/J etc.) bypasses that path and produces its own
//! refusals/fallbacks/hedges. Without this module, the wonder queue
//! never grows from reasoning failures — exactly the failure mode
//! observed in the pre-Port-J probe (4 `lu4d_ask` calls grew the
//! queue 0 → 3072; 3 `lu4d_reason` calls left it at 0).
//!
//! This module is the operational glue that lets every reasoning port
//! that already detects a gap emit a `GapSignal` which is then
//! translated into a `CuriositySignal` and pushed onto the existing
//! `Brain.wonder_queue` and `Brain.interest_profile`. From there the
//! existing autonomous research path (`research::loop_::run_research_loop`)
//! pops them as research targets unchanged.
//!
//! Architectural constraints (per internal notes — graph processing
//! first, schema last):
//! - NO new bond types.
//! - NO new on-disk persisted schema. `GapSignal` lives only in
//!   the in-RAM emit path; once translated to `CuriositySignal` it
//!   travels via the existing serialized wonder queue.
//! - `GapKind` exists purely to let port-emit sites declare *why*
//!   they're emitting (so the test matrix and the audit log can
//!   verify which port fired which signal). It maps to existing
//!   `curiosity::SignalType` variants — no new SignalType variants.

use crate::curiosity::{CuriositySignal, SignalType};
use crate::scope::TemporalScope;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

/// What kind of reasoning gap fired this signal. Each port that emits
/// uses exactly one variant; the verify rig asserts the right port
/// fires the right variant. Adding a variant is safe — `into_signal_type`
/// must be updated in lockstep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GapKind {
    /// Port K Mode A. No tokens matched the keyword index.
    TotalIdk,
    /// Port K Mode B. Some tokens matched, some did not.
    PartialMatchLeak,
    /// Port K (subordinate to Mode B). All tokens matched the index
    /// but at least one never produced a real activation.
    UnmatchedTokenLeak,
    /// Port K.1. One anchor dominates; coverage of the other tokens
    /// is poor.
    SingleAnchorDominance,
    /// Port L graceful degradation. Enumerative intent landed but
    /// fewer than `MIN_LIST_ITEMS` siblings could be gathered.
    ThinListFallback,
    /// Port M graceful degradation. Compare intent landed but fewer
    /// than `MIN_SHARED_BOND_TYPES` shared bond types across the
    /// subjects.
    ComparisonFallback,
    /// Port I (future). Procedural intent landed but only a linear
    /// step list was assembled where a branching plan was wanted.
    /// Currently NOT detectable from the existing trace shape; here
    /// for future heuristic.
    LinearProceduralOnly,
    /// Port F.3 (`math::wordprob`). The text matched the
    /// word-problem shape but extraction couldn't commit to a
    /// confident answer (`WordProblemOutcome::Hedge`).
    WordProblemUnparsable,
    /// Port H. Trace intent was detected but the deepest chain
    /// composer assembled was below `min_trace_chain_length`.
    ShallowWhenTraceRequested,
    /// Port J. The aggregate confidence across the recall set was
    /// below threshold OR there was only one piece of evidence.
    LowConfidence,
}

impl GapKind {
    /// Map a port-side `GapKind` to an existing `curiosity::SignalType`.
    /// The mapping is stable; changing it after release would change
    /// the histogram of signals already in serialized wonder queues,
    /// so be conservative.
    ///
    /// Mapping rationale:
    /// - `KnowledgeDesert` for gaps where there's a missing concept
    ///   the curiosity engine should research from scratch.
    /// - `HubBoundary` for gaps where one part of the query was rich
    ///   but another part hit a wall (parallel to the existing inline
    ///   hub-boundary signal).
    /// - `IsolatedNode` for gaps where an anchor was found but its
    ///   neighborhood is too thin to articulate.
    /// - `DeadEnd` for gaps where the chain composer ran out of
    ///   reachable continuation.
    pub fn into_signal_type(self) -> SignalType {
        match self {
            GapKind::TotalIdk => SignalType::KnowledgeDesert,
            GapKind::PartialMatchLeak => SignalType::HubBoundary,
            GapKind::UnmatchedTokenLeak => SignalType::HubBoundary,
            GapKind::SingleAnchorDominance => SignalType::HubBoundary,
            GapKind::ThinListFallback => SignalType::IsolatedNode,
            GapKind::ComparisonFallback => SignalType::KnowledgeDesert,
            GapKind::LinearProceduralOnly => SignalType::DeadEnd,
            GapKind::WordProblemUnparsable => SignalType::KnowledgeDesert,
            GapKind::ShallowWhenTraceRequested => SignalType::DeadEnd,
            GapKind::LowConfidence => SignalType::KnowledgeDesert,
        }
    }

    /// Human-readable label, used in audit logs and the
    /// `lu4d_wonder` structured payload (so John can see WHICH
    /// port emitted any given queue item).
    pub fn label(self) -> &'static str {
        match self {
            GapKind::TotalIdk => "total_idk",
            GapKind::PartialMatchLeak => "partial_match_leak",
            GapKind::UnmatchedTokenLeak => "unmatched_token_leak",
            GapKind::SingleAnchorDominance => "single_anchor_dominance",
            GapKind::ThinListFallback => "thin_list_fallback",
            GapKind::ComparisonFallback => "comparison_fallback",
            GapKind::LinearProceduralOnly => "linear_procedural_only",
            GapKind::WordProblemUnparsable => "word_problem_unparsable",
            GapKind::ShallowWhenTraceRequested => "shallow_when_trace_requested",
            GapKind::LowConfidence => "low_confidence",
        }
    }
}

/// One gap emitted by a reasoning port. Lives only between the port
/// emit site and the wonder-queue flush — never serialized. `severity`
/// becomes `CuriositySignal.intensity` directly.
#[derive(Debug, Clone)]
pub struct GapSignal {
    /// Surface query that produced the gap. Used to derive a stable
    /// research-target seed.
    pub query: String,
    /// Which port emitted ("K", "K.1", "L", "M", "I", "F.3", "H", "J").
    pub port_origin: &'static str,
    /// What KIND of gap.
    pub gap_kind: GapKind,
    /// 0.0..=1.0. Higher = research this sooner.
    pub severity: f32,
    /// Free-form per-port context (e.g. K: max_activation; J:
    /// evidence_count). NOT serialized; surfaced only via runtime
    /// audit logs.
    pub context: serde_json::Value,
}

impl GapSignal {
    /// Stable per-query seed: SHA-256("gap:" + query) truncated to
    /// 32 bytes. Two identical queries produce the same seed so the
    /// existing wonder-queue dedup-on-(SignalType, Seed) pathway
    /// merges duplicate emits.
    pub fn target_seed(&self) -> Seed {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"gap:");
        h.update(self.query.to_lowercase().as_bytes());
        let out = h.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out[..32]);
        Seed(bytes)
    }

    /// Translate to a `CuriositySignal` ready for `wonder_queue.push`.
    pub fn to_curiosity_signal(&self, now_ms: u64) -> CuriositySignal {
        CuriositySignal {
            seed: self.target_seed(),
            signal_type: self.gap_kind.into_signal_type(),
            intensity: self.severity.clamp(0.0, 1.0),
            scope: TemporalScope::Current,
            timestamp_ms: now_ms,
        }
    }
}

/// Per-Brain emitter. Buffers `GapSignal`s emitted during a single
/// MCP call, flushed at the end of the call by `Brain::flush_gaps`.
///
/// Implemented with `RefCell` so reasoning code holding `&Brain` can
/// `.emit()` without needing `&mut Brain` plumbed through every
/// internal call. Single-threaded by design — the MCP server
/// processes one tool call at a time per Brain.
#[derive(Debug, Default)]
pub struct GapEmitter {
    pending: RefCell<Vec<GapSignal>>,
    /// Total emits this Brain has seen across all calls. Surfaced via
    /// `lu4d_wonder` and `lu4d_status` for the verify rig.
    pub total_emitted: std::cell::Cell<u64>,
    /// Per-port counter — useful for asserting in tests "K fired N times".
    pub by_port: RefCell<std::collections::HashMap<&'static str, u64>>,
}

impl GapEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit a `GapSignal`. Cheap (just buffers); the actual wonder
    /// queue mutation happens at flush time so the call site doesn't
    /// pay for the queue-push latency mid-reasoning.
    pub fn emit(&self, signal: GapSignal) {
        let port = signal.port_origin;
        self.pending.borrow_mut().push(signal);
        self.total_emitted.set(self.total_emitted.get() + 1);
        *self.by_port.borrow_mut().entry(port).or_insert(0) += 1;
    }

    /// Drain all buffered signals. Called by `Brain::flush_gaps`.
    pub fn drain(&self) -> Vec<GapSignal> {
        std::mem::take(&mut *self.pending.borrow_mut())
    }

    /// Number of currently-buffered (un-flushed) signals.
    pub fn pending_len(&self) -> usize {
        self.pending.borrow().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_kind_label_and_signal_type_round_trip() {
        // All variants have a non-empty label and map to a real
        // SignalType (any of the 4). Sanity check that no variant
        // accidentally panics or returns empty string.
        for k in [
            GapKind::TotalIdk,
            GapKind::PartialMatchLeak,
            GapKind::UnmatchedTokenLeak,
            GapKind::SingleAnchorDominance,
            GapKind::ThinListFallback,
            GapKind::ComparisonFallback,
            GapKind::LinearProceduralOnly,
            GapKind::WordProblemUnparsable,
            GapKind::ShallowWhenTraceRequested,
            GapKind::LowConfidence,
        ] {
            assert!(!k.label().is_empty());
            let _ = k.into_signal_type();
        }
    }

    #[test]
    fn target_seed_is_deterministic_and_case_insensitive() {
        let g1 = GapSignal {
            query: "What is photosynthesis".to_string(),
            port_origin: "K",
            gap_kind: GapKind::TotalIdk,
            severity: 0.9,
            context: serde_json::json!({}),
        };
        let g2 = GapSignal {
            query: "what is photosynthesis".to_string(),
            port_origin: "K",
            gap_kind: GapKind::TotalIdk,
            severity: 0.9,
            context: serde_json::json!({}),
        };
        assert_eq!(g1.target_seed(), g2.target_seed());
        let g3 = GapSignal {
            query: "what is glucose".to_string(),
            ..g1.clone()
        };
        assert_ne!(g1.target_seed(), g3.target_seed());
    }

    #[test]
    fn emitter_buffers_until_drained() {
        let e = GapEmitter::new();
        assert_eq!(e.pending_len(), 0);
        e.emit(GapSignal {
            query: "q1".to_string(),
            port_origin: "K",
            gap_kind: GapKind::TotalIdk,
            severity: 0.9,
            context: serde_json::json!({}),
        });
        e.emit(GapSignal {
            query: "q2".to_string(),
            port_origin: "L",
            gap_kind: GapKind::ThinListFallback,
            severity: 0.5,
            context: serde_json::json!({}),
        });
        assert_eq!(e.pending_len(), 2);
        assert_eq!(e.total_emitted.get(), 2);
        let drained = e.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(e.pending_len(), 0);
        // total_emitted is cumulative — drain doesn't reset it.
        assert_eq!(e.total_emitted.get(), 2);
        // by_port counters
        let by = e.by_port.borrow();
        assert_eq!(by.get("K"), Some(&1));
        assert_eq!(by.get("L"), Some(&1));
    }

    #[test]
    fn severity_clamps_into_intensity() {
        let g = GapSignal {
            query: "q".to_string(),
            port_origin: "J",
            gap_kind: GapKind::LowConfidence,
            severity: 1.5, // out of bounds — should clamp to 1.0
            context: serde_json::json!({}),
        };
        let cs = g.to_curiosity_signal(0);
        assert!((cs.intensity - 1.0).abs() < 1e-6);
    }

    /// Port J.1 — full save/reload round-trip with persisted wonder
    /// queue and gap_label_map.
    #[test]
    fn port_j1_persistence_round_trip_via_brain_save_load() {
        use crate::brain::{Brain, BrainSaveFormat};
        use crate::gap_signal::{GapKind, GapSignal};

        // Inline temp path to avoid pulling in tempfile crate just for one test.
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("lu4d-port-j1-{}.lbr", pid));
        let _ = std::fs::remove_file(&path);

        // 1) Fresh brain, emit a gap, flush, save in v3 (uncompressed/un-encrypted).
        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.save_format = BrainSaveFormat::V3Binary;
        let q = "port j1 persistence smoke query".to_string();
        let gs = GapSignal {
            query: q.clone(),
            port_origin: "J",
            gap_kind: GapKind::LowConfidence,
            severity: 0.7,
            context: serde_json::json!({}),
        };
        let target_seed = gs.target_seed();
        brain.gap_emitter.emit(gs);
        let n = brain.flush_gaps();
        assert_eq!(n, 1);
        assert_eq!(brain.wonder_queue.len(), 1);
        assert_eq!(brain.gap_label_map.get(&target_seed), Some(&q));
        brain.save_to(&path).expect("v3 save");
        drop(brain);

        // 2) Reload from disk; queue + label_map must survive.
        let reloaded = Brain::load_or_new(&path).expect("reload");
        assert_eq!(
            reloaded.wonder_queue.len(), 1,
            "wonder queue must survive save/reload",
        );
        assert_eq!(
            reloaded.gap_label_map.get(&target_seed),
            Some(&q),
            "gap_label_map must survive save/reload",
        );
        let top = reloaded.wonder_queue.peek().expect("non-empty");
        assert_eq!(top.seed, target_seed);
        assert!((top.intensity - 0.7).abs() < 1e-3);
        let _ = std::fs::remove_file(&path);
    }
}
