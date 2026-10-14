//! Port J — Confidence aggregation across evidence.
//!
//! Per the project log Learning #580: aggregate evidence across the recall set,
//! articulate confidence as "based on N sources" / "multiple corroborating
//! bonds" / "single source — low confidence" / etc. Walks existing bond
//! weights/counts/anchor density as evidence proxies. NO new bond types
//! per Learning #626 (graph processing first, schema last) — articulator-
//! only.
//!
//! This module provides the pure aggregation primitive. Call sites in
//! `mcp::tool_reason` and `mcp::tool_ask` consume it; the gap_signal
//! emission for `LowConfidence` happens at the call site, not here.
//!
//! Why a module rather than a method on `ReasoningTrace`?  The trace
//! type lives in `reasoning.rs` and serializes to a stable JSON shape;
//! we don't want to bake a confidence-band field onto it without
//! migration consideration. By keeping this as a pure function we can
//! evolve the band thresholds without touching the trace shape.

use serde::{Deserialize, Serialize};

/// Coarse confidence band derived from evidence count + raw trace
/// confidence. Surfaced in the structured payload as a string so
/// downstream consumers (PWA, narration) can switch on it without
/// re-deriving the calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceBand {
    /// trace had no chain at all — caller still owes the user prose
    /// (already produced in `no_chain_trace`); we report the band as
    /// `None` to skip extra articulation.
    None,
    /// 1 piece of evidence. Honest but thin — gap signal emitted.
    Single,
    /// 2-3 pieces of evidence below the high threshold. Gap signal
    /// emitted at lower severity than `Single` (some corroboration).
    Low,
    /// 4+ pieces of evidence OR trace.confidence >= the high
    /// threshold. Sufficient to articulate confidently.
    Medium,
    /// Many pieces of evidence (>= 6) AND trace.confidence >= the
    /// high threshold. Strong claim.
    High,
}

impl ConfidenceBand {
    pub fn label(self) -> &'static str {
        match self {
            ConfidenceBand::None => "none",
            ConfidenceBand::Single => "single",
            ConfidenceBand::Low => "low",
            ConfidenceBand::Medium => "medium",
            ConfidenceBand::High => "high",
        }
    }

    /// Articulate the band as English prose appropriate for inclusion
    /// in `final_answer` or as a separate `confidence_articulation`
    /// field. Includes the actual evidence count + trace confidence
    /// numeric so the user sees what the band stands on.
    pub fn articulate(self, evidence_count: usize, trace_confidence: f32) -> String {
        match self {
            ConfidenceBand::None => String::new(),
            ConfidenceBand::Single => format!(
                "Confidence: single source (1 bond, weight {:.2}). I'd value corroboration before treating this as settled.",
                trace_confidence
            ),
            ConfidenceBand::Low => format!(
                "Confidence: low ({} bonds, weakest weight {:.2}). Some corroboration but the chain is thin.",
                evidence_count, trace_confidence
            ),
            ConfidenceBand::Medium => format!(
                "Confidence: medium ({} bonds, weakest weight {:.2}). Multiple corroborating bonds.",
                evidence_count, trace_confidence
            ),
            ConfidenceBand::High => format!(
                "Confidence: high ({} bonds, weakest weight {:.2}). Many corroborating bonds across the recall set.",
                evidence_count, trace_confidence
            ),
        }
    }
}

/// Default thresholds, in one place so tests + tuning stay
/// synchronized.
pub mod thresholds {
    /// Trace confidence above which we may upgrade to High band.
    pub const HIGH_CONFIDENCE: f32 = 0.70;
    /// Trace confidence above which we may upgrade to Medium band.
    pub const MEDIUM_CONFIDENCE: f32 = 0.40;
    /// Bond count >= this triggers High (with HIGH_CONFIDENCE).
    pub const HIGH_EVIDENCE_COUNT: usize = 6;
    /// Bond count >= this triggers Medium baseline.
    pub const MEDIUM_EVIDENCE_COUNT: usize = 4;
    /// Bond count <= this is Low.
    pub const LOW_EVIDENCE_COUNT: usize = 3;
}

/// Aggregate evidence count + trace confidence into a band.
///
/// Logic:
/// - `no_chain` ⇒ `None`.
/// - `evidence_count == 0` (chain present but bonds_used empty —
///   shouldn't happen but be defensive) ⇒ `Single`.
/// - `evidence_count == 1` ⇒ `Single`.
/// - `evidence_count >= HIGH_EVIDENCE_COUNT && trace_confidence >=
///   HIGH_CONFIDENCE` ⇒ `High`.
/// - `evidence_count >= MEDIUM_EVIDENCE_COUNT` OR `trace_confidence
///   >= MEDIUM_CONFIDENCE` ⇒ `Medium`.
/// - Otherwise (`evidence_count <= LOW_EVIDENCE_COUNT` AND
///   `trace_confidence < MEDIUM_CONFIDENCE`) ⇒ `Low`.
pub fn aggregate(
    evidence_count: usize,
    trace_confidence: f32,
    no_chain: bool,
) -> ConfidenceBand {
    if no_chain {
        return ConfidenceBand::None;
    }
    if evidence_count <= 1 {
        return ConfidenceBand::Single;
    }
    if evidence_count >= thresholds::HIGH_EVIDENCE_COUNT
        && trace_confidence >= thresholds::HIGH_CONFIDENCE
    {
        return ConfidenceBand::High;
    }
    if evidence_count >= thresholds::MEDIUM_EVIDENCE_COUNT
        || trace_confidence >= thresholds::MEDIUM_CONFIDENCE
    {
        return ConfidenceBand::Medium;
    }
    ConfidenceBand::Low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_chain_yields_none() {
        assert_eq!(
            aggregate(0, 0.0, true),
            ConfidenceBand::None
        );
        assert_eq!(
            aggregate(5, 0.9, true),
            ConfidenceBand::None
        );
    }

    #[test]
    fn single_source_yields_single() {
        assert_eq!(aggregate(1, 0.9, false), ConfidenceBand::Single);
        assert_eq!(aggregate(0, 0.9, false), ConfidenceBand::Single);
    }

    #[test]
    fn high_evidence_and_high_confidence_yields_high() {
        assert_eq!(
            aggregate(6, 0.75, false),
            ConfidenceBand::High
        );
        assert_eq!(
            aggregate(10, 0.95, false),
            ConfidenceBand::High
        );
    }

    #[test]
    fn high_evidence_low_confidence_is_medium() {
        assert_eq!(
            aggregate(8, 0.50, false),
            ConfidenceBand::Medium
        );
    }

    #[test]
    fn medium_evidence_yields_medium() {
        assert_eq!(
            aggregate(4, 0.30, false),
            ConfidenceBand::Medium
        );
    }

    #[test]
    fn low_evidence_low_confidence_yields_low() {
        assert_eq!(
            aggregate(2, 0.20, false),
            ConfidenceBand::Low
        );
        assert_eq!(
            aggregate(3, 0.10, false),
            ConfidenceBand::Low
        );
    }

    #[test]
    fn low_evidence_medium_confidence_yields_medium() {
        // Even 2 evidence pieces upgrade to Medium when the chain
        // weight is comfortably above MEDIUM_CONFIDENCE.
        assert_eq!(
            aggregate(2, 0.55, false),
            ConfidenceBand::Medium
        );
    }

    #[test]
    fn articulation_includes_numbers() {
        let s = ConfidenceBand::Medium.articulate(5, 0.62);
        assert!(s.contains("5"));
        assert!(s.contains("0.62"));
    }

    #[test]
    fn label_round_trip_for_all_bands() {
        for b in [
            ConfidenceBand::None,
            ConfidenceBand::Single,
            ConfidenceBand::Low,
            ConfidenceBand::Medium,
            ConfidenceBand::High,
        ] {
            assert!(!b.label().is_empty());
        }
    }

    /// Port J.1 — keep Medium band stable at evidence_count=4 / trace_confidence=0.30.
    /// (call-site emit policy now ALSO fires LowConfidence when band==Medium
    /// AND trace.confidence < 0.55 — this guard test prevents accidental band drift
    /// that would change the emit set out from under tool_reason.)
    #[test]
    fn port_j1_medium_band_stays_medium_for_weak_chain_emit_target() {
        // 4 bonds, weakest weight 0.30 -> Medium (already covered by `medium_evidence_yields_medium`)
        // 4 bonds, weakest weight 0.50 -> Medium too. Both should emit a J.1 LowConfidence gap.
        assert_eq!(aggregate(4, 0.30, false), ConfidenceBand::Medium);
        assert_eq!(aggregate(4, 0.50, false), ConfidenceBand::Medium);
        // 4 bonds, weakest weight 0.60 stays Medium (above J.1's 0.55 cutoff -> NOT emitted).
        assert_eq!(aggregate(4, 0.60, false), ConfidenceBand::Medium);
    }
}
