//! Port T.8 — rejection pattern store.
//!
//! Holds the rolling rejection / approval fingerprint buffers, the
//! learned-pattern set derived from them, and the
//! `proposals_filtered_total` counter exposed via `lu4d_pending_status`.
//!
//! The store snapshot (`RejectionPatternsSnapshot`) is what the v3
//! binary section #10 (`SectionIdV3::RejectionPatterns`) serializes.
//! `#[serde(default)]` on `BrainFile::rejection_patterns` plus
//! `de_bincode_optional` on the read path keeps pre-T.8 brains
//! loading with an empty store, satisfying the J.1.1 forward-compat
//! guarantee.

use super::fingerprint::RejectionFingerprint;
use super::pattern::{
    fingerprint_matches_key, PatternKey, RejectionPattern, RejectionPatternStatus,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default `min_rejections_for_pattern` — three matching rejections
/// without any approvals promotes a fingerprint group to a learned
/// pattern. Empirical: with 79 pending entries from the T.6 demo, a
/// threshold of 3 cleanly catches the obvious sentence-fragment
/// shapes without firing on isolated one-offs.
pub const DEFAULT_MIN_REJECTIONS_FOR_PATTERN: u32 = 3;

/// Default `pattern_decay_days` — 90 days of zero new matching
/// rejections expires a pattern. Picked to match the T.5 freshness
/// half-life floor so the rejection signal isn't shorter-lived than
/// the bond freshness signal.
pub const DEFAULT_PATTERN_DECAY_DAYS: u32 = 90;

/// Configuration knobs for the rejection-pattern store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionConfig {
    pub min_rejections_for_pattern: u32,
    pub pattern_decay_days: u32,
    /// When `true` (default), `matches_learned_pattern` causes
    /// `push_pending_with_filter` to drop the proposal silently. When
    /// `false`, the proposal is pushed but its confidence is multiplied
    /// by `filter_confidence_multiplier`. The user spec defaults to
    /// silent drop; the multiplier path is reserved for a future
    /// configurable mode.
    pub drop_on_match: bool,
    pub filter_confidence_multiplier: f32,
}

impl Default for RejectionConfig {
    fn default() -> Self {
        RejectionConfig {
            min_rejections_for_pattern: DEFAULT_MIN_REJECTIONS_FOR_PATTERN,
            pattern_decay_days: DEFAULT_PATTERN_DECAY_DAYS,
            drop_on_match: true,
            filter_confidence_multiplier: 0.3,
        }
    }
}

/// Persistent snapshot of the store. Round-trips through the v3
/// `RejectionPatterns` section (id 10).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RejectionPatternsSnapshot {
    #[serde(default)]
    pub rejections: Vec<RejectionFingerprint>,
    #[serde(default)]
    pub approvals: Vec<RejectionFingerprint>,
    #[serde(default)]
    pub patterns: Vec<RejectionPattern>,
    #[serde(default)]
    pub proposals_filtered_total: u64,
    #[serde(default = "default_next_pattern_id")]
    pub next_pattern_id: u64,
    #[serde(default)]
    pub config: RejectionConfig,
}

fn default_next_pattern_id() -> u64 {
    1
}

/// Live store: holds rejections, approvals, derived patterns, and
/// the filter counter.
#[derive(Debug, Clone)]
pub struct RejectionPatternStore {
    rejections: Vec<RejectionFingerprint>,
    approvals: Vec<RejectionFingerprint>,
    patterns: Vec<RejectionPattern>,
    proposals_filtered_total: u64,
    next_pattern_id: u64,
    config: RejectionConfig,
}

impl Default for RejectionPatternStore {
    fn default() -> Self {
        RejectionPatternStore {
            rejections: Vec::new(),
            approvals: Vec::new(),
            patterns: Vec::new(),
            proposals_filtered_total: 0,
            next_pattern_id: 1,
            config: RejectionConfig::default(),
        }
    }
}

impl RejectionPatternStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(snap: RejectionPatternsSnapshot) -> Self {
        let next_id = snap
            .next_pattern_id
            .max(snap.patterns.iter().map(|p| p.pattern_id + 1).max().unwrap_or(1));
        RejectionPatternStore {
            rejections: snap.rejections,
            approvals: snap.approvals,
            patterns: snap.patterns,
            proposals_filtered_total: snap.proposals_filtered_total,
            next_pattern_id: next_id,
            config: snap.config,
        }
    }

    pub fn snapshot(&self) -> RejectionPatternsSnapshot {
        RejectionPatternsSnapshot {
            rejections: self.rejections.clone(),
            approvals: self.approvals.clone(),
            patterns: self.patterns.clone(),
            proposals_filtered_total: self.proposals_filtered_total,
            next_pattern_id: self.next_pattern_id,
            config: self.config.clone(),
        }
    }

    pub fn config(&self) -> &RejectionConfig {
        &self.config
    }

    pub fn set_config(&mut self, cfg: RejectionConfig) {
        self.config = cfg;
    }

    pub fn proposals_filtered_total(&self) -> u64 {
        self.proposals_filtered_total
    }

    pub fn rejection_count(&self) -> usize {
        self.rejections.len()
    }

    pub fn approval_count(&self) -> usize {
        self.approvals.len()
    }

    /// Record a fingerprint of a just-rejected pending entry. Triggers
    /// `recompute_patterns` so the pattern set reflects the new data
    /// immediately. `now_ms` is used only for the pattern-emergence
    /// timestamp on first promotion; the fingerprint's own
    /// `rejected_at_ms` is what feeds the decay calculation.
    pub fn record_rejection(&mut self, fp: RejectionFingerprint, now_ms: u64) {
        self.rejections.push(fp);
        self.recompute_patterns(now_ms);
    }

    /// Record a fingerprint of an approval. Used to suppress patterns
    /// where any matching fingerprint was approved (zero-tolerance —
    /// even one approval kills the pattern).
    pub fn record_approval(&mut self, fp: RejectionFingerprint, now_ms: u64) {
        self.approvals.push(fp);
        self.recompute_patterns(now_ms);
    }

    /// Re-derive the learned-pattern set from the current rejection /
    /// approval buffers. Idempotent — call freely.
    ///
    /// Algorithm: bucket rejection fingerprints by `PatternKey`. For
    /// each bucket whose count >= `min_rejections_for_pattern`, check
    /// whether any approval fingerprint also falls in the bucket — if
    /// so, the bucket is suppressed. Otherwise promote / refresh the
    /// matching `RejectionPattern`. Buckets whose latest rejection is
    /// older than `pattern_decay_days` get marked `Expired`.
    pub fn recompute_patterns(&mut self, now_ms: u64) {
        let cfg = self.config.clone();

        // Bucket rejections by PatternKey, also tracking the latest
        // rejected_at_ms inside the bucket.
        let mut buckets: HashMap<PatternKey, (u32, u64)> = HashMap::new();
        for fp in &self.rejections {
            let key = PatternKey::from_fingerprint(fp);
            let entry = buckets.entry(key).or_insert((0, 0));
            entry.0 += 1;
            if fp.rejected_at_ms > entry.1 {
                entry.1 = fp.rejected_at_ms;
            }
        }

        // Suppress buckets that overlap any approval fingerprint.
        let mut suppressed: std::collections::HashSet<PatternKey> = std::collections::HashSet::new();
        for ap in &self.approvals {
            let key = PatternKey::from_fingerprint(ap);
            suppressed.insert(key);
        }

        // Build the new pattern list, preserving stable pattern_id
        // for keys that already exist.
        let mut existing_ids: HashMap<PatternKey, (u64, u64)> = HashMap::new(); // key -> (id, first_learned_at)
        for p in &self.patterns {
            existing_ids.insert(p.key.clone(), (p.pattern_id, p.first_learned_at_ms));
        }

        let mut new_patterns: Vec<RejectionPattern> = Vec::new();
        for (key, (count, latest_ms)) in buckets {
            if suppressed.contains(&key) {
                continue;
            }
            if count < cfg.min_rejections_for_pattern {
                continue;
            }
            let (pattern_id, first_learned) = match existing_ids.get(&key) {
                Some((id, learned_at)) => (*id, *learned_at),
                None => {
                    let id = self.next_pattern_id;
                    self.next_pattern_id += 1;
                    (id, now_ms)
                }
            };
            let status = if (now_ms.saturating_sub(latest_ms))
                >= (cfg.pattern_decay_days as u64).saturating_mul(super::pattern::MS_PER_DAY)
            {
                RejectionPatternStatus::Expired
            } else {
                RejectionPatternStatus::Learned
            };
            new_patterns.push(RejectionPattern {
                pattern_id,
                key,
                rejection_count: count,
                last_rejection_at_ms: latest_ms,
                first_learned_at_ms: first_learned,
                status,
            });
        }
        new_patterns.sort_by(|a, b| b.last_rejection_at_ms.cmp(&a.last_rejection_at_ms));
        self.patterns = new_patterns;
    }

    /// Does `fp` match any currently `Learned` (non-expired) pattern?
    /// Used by `Brain::push_pending_with_filter` as the proposal-time
    /// gate.
    pub fn matches_learned_pattern(&self, fp: &RejectionFingerprint) -> bool {
        for p in &self.patterns {
            if p.status != RejectionPatternStatus::Learned {
                continue;
            }
            if fingerprint_matches_key(fp, &p.key) {
                return true;
            }
        }
        false
    }

    /// View into the learned-pattern set. When `include_expired` is
    /// false, only `Learned`-status patterns are returned.
    pub fn learned_patterns(&self, include_expired: bool) -> Vec<&RejectionPattern> {
        self.patterns
            .iter()
            .filter(|p| include_expired || p.status == RejectionPatternStatus::Learned)
            .collect()
    }

    /// Increment the filtered-proposal counter. Called by
    /// `Brain::push_pending_with_filter` when a proposal matches a
    /// learned pattern and is dropped.
    pub fn note_filtered(&mut self) {
        self.proposals_filtered_total =
            self.proposals_filtered_total.saturating_add(1);
    }
}
