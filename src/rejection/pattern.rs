//! Port T.8 — pattern aggregation over rejection fingerprints.
//!
//! Fingerprints (see `fingerprint.rs`) are grouped by
//! `(bond_type, source_domain, boolean-feature-tuple)`. When a group
//! accumulates `min_rejections_for_pattern` rejections without any
//! matching approval, the group is promoted to a `RejectionPattern`.
//! Patterns then drive the proposal-time pre-filter.
//!
//! Patterns also decay: if no new rejection matches the pattern in
//! `pattern_decay_days`, the pattern is treated as expired (still
//! visible via `lu4d_rejection_patterns include_expired:true`, but no
//! longer used by the pre-filter). The decay timer prevents stale
//! patterns from suppressing genuinely improved extractions forever.

use super::fingerprint::RejectionFingerprint;
use crate::bond::BondType;
use serde::{Deserialize, Serialize};

/// One day in milliseconds. Used by the decay calculation.
pub const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000;

/// The grouping key used to bucket fingerprints into patterns. The
/// key is intentionally coarse — bond_type + domain + four booleans —
/// so a handful of similar bad extractions consolidate quickly.
/// Length / confidence are NOT in the key (continuous values would
/// fragment the buckets).
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternKey {
    pub bond_type: BondType,
    pub source_domain: String,
    pub subject_starts_with_preposition: bool,
    pub subject_contains_citation_marker: bool,
    pub subject_contains_generic_pronoun: bool,
    pub object_ends_with_preposition: bool,
}

impl PatternKey {
    pub fn from_fingerprint(fp: &RejectionFingerprint) -> Self {
        PatternKey {
            bond_type: fp.bond_type,
            source_domain: fp.source_domain.clone(),
            subject_starts_with_preposition: fp.subject_starts_with_preposition,
            subject_contains_citation_marker: fp.subject_contains_citation_marker,
            subject_contains_generic_pronoun: fp.subject_contains_generic_pronoun,
            object_ends_with_preposition: fp.object_ends_with_preposition,
        }
    }
}

/// Whether a pattern is currently active (the pre-filter consults
/// it) or expired (still visible for audit but not enforced).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionPatternStatus {
    Learned,
    Expired,
}

impl RejectionPatternStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RejectionPatternStatus::Learned => "learned",
            RejectionPatternStatus::Expired => "expired",
        }
    }
}

/// A learned rejection pattern. Built by `RejectionPatternStore::recompute`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionPattern {
    /// Stable id assigned by the store at first emergence.
    pub pattern_id: u64,
    pub key: PatternKey,
    pub rejection_count: u32,
    /// Most recent rejected_at_ms across all matching fingerprints.
    pub last_rejection_at_ms: u64,
    /// Wall-clock ms when the pattern was first promoted to `Learned`.
    pub first_learned_at_ms: u64,
    pub status: RejectionPatternStatus,
}

impl RejectionPattern {
    /// `true` if this pattern's last rejection is older than the
    /// configured decay window relative to `now_ms`.
    pub fn is_decayed(&self, now_ms: u64, pattern_decay_days: u32) -> bool {
        let window_ms = (pattern_decay_days as u64).saturating_mul(MS_PER_DAY);
        now_ms.saturating_sub(self.last_rejection_at_ms) >= window_ms
    }
}

/// Does fingerprint `fp` fall inside the bucket described by `key`?
pub fn fingerprint_matches_key(fp: &RejectionFingerprint, key: &PatternKey) -> bool {
    fp.bond_type == key.bond_type
        && fp.source_domain == key.source_domain
        && fp.subject_starts_with_preposition == key.subject_starts_with_preposition
        && fp.subject_contains_citation_marker == key.subject_contains_citation_marker
        && fp.subject_contains_generic_pronoun == key.subject_contains_generic_pronoun
        && fp.object_ends_with_preposition == key.object_ends_with_preposition
}
