//! Port T.5 — Bond freshness / decay signal.
//!
//! Query-time multiplier on bond effective weight. Bonds halve their
//! effective weight every `decay_half_life_days` days since creation
//! (default 90 days). **Stored bond weights are never mutated** — decay
//! is presentation-only. The on-disk `TemporalBond` rkyv layout is
//! unchanged; T.5 ships freely on either side of the next retrain.
//!
//! Time source is routed through T.1's `ClockSource` trait so the
//! multiplier is testable with a mock clock and consistent with T.1's
//! authenticated-time invariant. No call site in this module reads
//! `SystemTime::now` directly.
//!
//! Companion knobs land in [`crate::brain::Brain::decay_config`]; the
//! `lu4d_configure` MCP tool surfaces them at runtime; the
//! `lu4d_freshness_status` MCP tool reports bucket counts + stalest-N.
//! See internal notes / component 168 (Curiosity Engine).

use crate::bond::{BondType, TemporalBond};
use crate::seed::Seed;
use serde::{Deserialize, Serialize};

/// 90-day default — matches the spec; Curiosity Engine's prior
/// `DEFAULT_INTEREST_HALFLIFE_MS` is a different (per-domain interest)
/// signal and is left untouched.
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: u32 = 90;
pub const MIN_HALF_LIFE_DAYS: u32 = 1;
pub const MAX_HALF_LIFE_DAYS: u32 = 3650;
pub const DAY_MS: i64 = 86_400_000;
/// Pending entries older than this count toward `stale_count`.
pub const STALE_PENDING_THRESHOLD_MS: u64 = 30 * DAY_MS as u64;

/// Runtime knobs for T.5. Default = 90-day half-life, decay on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DecayConfig {
    pub half_life_days: u32,
    pub enabled: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        DecayConfig {
            half_life_days: DEFAULT_DECAY_HALF_LIFE_DAYS,
            enabled: true,
        }
    }
}

impl DecayConfig {
    /// Clamp `half_life_days` to the valid range. Returns the actually
    /// applied value.
    pub fn set_half_life_days(&mut self, days: u32) -> u32 {
        let clamped = days.clamp(MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS);
        self.half_life_days = clamped;
        clamped
    }
}

/// Port T.5.1 — Per-query decay context stashed on `AssociativeNetwork`.
///
/// Carries the active [`DecayConfig`] together with the wall-clock NOW
/// (in milliseconds) sourced from `Brain::clock` (T.1 `ClockSource`).
/// `AssociativeNetwork::set_decay_runtime` stores `Some(runtime)` to
/// arm decay on query-path effective-weight reads, or `None` to
/// produce identical scoring to pre-T.5.1 behavior (the regression
/// guard exercised by `t51_decay_disabled_matches_pre_t51_behavior_exactly`).
///
/// This struct is `Copy` because both fields are tiny POD; refreshing
/// it before every Brain query path is essentially free.
#[derive(Debug, Clone, Copy)]
pub struct DecayRuntime {
    pub config: DecayConfig,
    pub now_ms: i64,
}

impl DecayRuntime {
    /// Build a new runtime. There is no sensible `Default` because
    /// `now_ms = 0` (epoch) is meaningful rather than "unset"; use
    /// `None` on `AssociativeNetwork::decay_runtime` for the unset case.
    pub fn new(config: DecayConfig, now_ms: i64) -> Self {
        DecayRuntime { config, now_ms }
    }
}

/// Pure: freshness multiplier for an `(age, half_life)` pair.
///
/// `freshness = 2^(-age_days / half_life_days)`
///           = exp(-ln(2) * age_days / half_life_days)
///
/// Returns 1.0 when decay is disabled, when the half-life is zero,
/// or when the bond was created in the future relative to `now_ms`
/// (clock skew is silently floored to age=0 — the T.1 anomaly
/// detector handles real backward-jump escalation).
pub fn freshness_score(created_at_ms: i64, now_ms: i64, cfg: &DecayConfig) -> f32 {
    if !cfg.enabled || cfg.half_life_days == 0 {
        return 1.0;
    }
    let age_ms = (now_ms - created_at_ms).max(0) as f64;
    if age_ms == 0.0 {
        return 1.0;
    }
    let age_days = age_ms / DAY_MS as f64;
    let h = cfg.half_life_days as f64;
    (-std::f64::consts::LN_2 * age_days / h).exp() as f32
}

/// Bond creation timestamp = oldest visible keyframe. Variant- and
/// Agent-source keyframes are not filtered here (they are also valid
/// signals of "when this bond first appeared"); recall-time scoping
/// happens elsewhere.
pub fn bond_created_at_ms(bond: &TemporalBond) -> i64 {
    bond.keyframes.iter().map(|kf| kf.timestamp).min().unwrap_or(0)
}

/// One row in the stalest-N list returned by `lu4d_freshness_status`.
#[derive(Debug, Clone, Serialize)]
pub struct StalestBondInfo {
    pub from: Seed,
    pub to: Seed,
    pub subject_text: String,
    pub object_text: String,
    pub bond_type: String,
    pub age_days: f64,
    pub stored_weight: f32,
    pub effective_weight_after_decay: f32,
}

/// Output of `Brain::freshness_status`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct FreshnessStatus {
    pub bucket_24h: u64,
    pub bucket_7d: u64,
    pub bucket_30d: u64,
    pub bucket_90d: u64,
    pub bucket_180d: u64,
    pub bucket_365d: u64,
    pub bucket_older: u64,
    pub mean_age_days: f64,
    pub median_age_days: f64,
    pub total_bonds: u64,
    pub stalest: Vec<StalestBondInfo>,
    pub decay_enabled: bool,
    pub half_life_days: u32,
}

/// Classify an age (in ms) into the 7 freshness buckets. Mutually
/// exclusive — every bond increments exactly one counter.
pub fn classify_bucket(age_ms: i64, status: &mut FreshnessStatus) {
    let day = DAY_MS;
    if age_ms < day { status.bucket_24h += 1; }
    else if age_ms < 7 * day { status.bucket_7d += 1; }
    else if age_ms < 30 * day { status.bucket_30d += 1; }
    else if age_ms < 90 * day { status.bucket_90d += 1; }
    else if age_ms < 180 * day { status.bucket_180d += 1; }
    else if age_ms < 365 * day { status.bucket_365d += 1; }
    else { status.bucket_older += 1; }
}

/// Bond-type printable tag (mirrors the MCP wire format).
pub(crate) fn bond_type_tag(bt: BondType) -> &'static str {
    match bt {
        BondType::IsA => "IsA",
        BondType::HasProperty => "HasProperty",
        BondType::Causes => "Causes",
        BondType::PartOf => "PartOf",
        BondType::UsedFor => "UsedFor",
        BondType::RelatedTo => "RelatedTo",
        BondType::ContrastedWith => "ContrastedWith",
        BondType::DerivedFrom => "DerivedFrom",
        BondType::Requires => "Requires",
        BondType::Produces => "Produces",
        BondType::Opposes => "Opposes",
        BondType::BornIn => "BornIn",
        BondType::BornOn => "BornOn",
        BondType::DiedIn => "DiedIn",
        BondType::DiedOn => "DiedOn",
        BondType::DiscoveredBy => "DiscoveredBy",
        BondType::FoundedBy => "FoundedBy",
        BondType::FoundedIn => "FoundedIn",
        BondType::HasAltitude => "HasAltitude",
        BondType::HasAspect => "HasAspect",
        BondType::HasCapital => "HasCapital",
        BondType::HasOfficialLanguage => "HasOfficialLanguage",
        BondType::HasPopulation => "HasPopulation",
        BondType::KnownFor => "KnownFor",
        BondType::LocatedIn => "LocatedIn",
        BondType::MeasuredIn => "MeasuredIn",
        BondType::NamedAfter => "NamedAfter",
        BondType::WrittenBy => "WrittenBy",
        BondType::HasArea => "HasArea",
        BondType::HasMethod => "HasMethod",
        BondType::HasAttribute => "HasAttribute",
        BondType::HasParameter => "HasParameter",
        BondType::Returns => "Returns",
        BondType::Inherits => "Inherits",
        BondType::Calls => "Calls",
        BondType::HasType => "HasType",
        BondType::Implements => "Implements",
        BondType::DefinedAs => "DefinedAs",
        BondType::WorksBy => "WorksBy",
        BondType::ComposedOf => "ComposedOf",
        BondType::PurposeOf => "PurposeOf",
        BondType::DiffersFrom => "DiffersFrom",
        BondType::LeadsTo => "LeadsTo",
        BondType::ExampleOf => "ExampleOf",
        BondType::HasCurrency => "HasCurrency",
        BondType::AuthoredBy => "AuthoredBy",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::brain::{now_ms, Brain};
    use crate::node::Confidence;
    use crate::pending::change::{PendingChange, PendingChangeKind, PendingStatus};
    use crate::source::{SourceAttribution, SourceKind};
    use crate::time::{
        AuthenticatedTime, ClockError, ClockSource, MonotonicClock, SourceId, TimeConfig,
    };
    use std::sync::Arc;

    /// T.5 — freshness_score is monotonically non-increasing in age.
    /// At one half-life, score must be ~0.5.
    #[test]
    fn t5_freshness_score_decreases_monotonically_with_age() {
        let cfg = DecayConfig::default();
        let now = 1_000_000_000_000_i64;
        let mut prev = freshness_score(now, now, &cfg);
        assert!((prev - 1.0).abs() < 1e-6);
        for d in [1u32, 5, 30, 90, 180, 365, 1000] {
            let created = now - (d as i64) * DAY_MS;
            let cur = freshness_score(created, now, &cfg);
            assert!(
                cur <= prev + 1e-6,
                "freshness must not increase: prev={} cur={} d={}",
                prev,
                cur,
                d
            );
            prev = cur;
        }
        let one_hl = freshness_score(now - cfg.half_life_days as i64 * DAY_MS, now, &cfg);
        assert!((one_hl - 0.5).abs() < 0.01, "score at 1 half-life ~ 0.5 (got {})", one_hl);
    }

    fn teach_at_age(brain: &mut Brain, s: &str, o: &str, age_ms: i64, now: i64) {
        let r = brain.teach(s, BondType::IsA, o, 0.7, Confidence::Verified);
        let key = (r.subject_seed, r.object_seed);
        let bond = brain.network.bonds_mut_raw().get_mut(&key).unwrap();
        for kf in bond.keyframes.iter_mut() {
            kf.timestamp = now - age_ms;
        }
    }

    /// T.5 — `freshness_status` reports correct bucket counts.
    #[test]
    fn t5_freshness_status_reports_correct_bucket_counts() {
        let mut brain = Brain::new();
        let now = now_ms();
        teach_at_age(&mut brain, "a", "b", 3_600_000, now);
        teach_at_age(&mut brain, "c", "d", 3 * DAY_MS, now);
        teach_at_age(&mut brain, "e", "f", 20 * DAY_MS, now);
        teach_at_age(&mut brain, "g", "h", 60 * DAY_MS, now);
        teach_at_age(&mut brain, "i", "j", 120 * DAY_MS, now);
        teach_at_age(&mut brain, "k", "l", 250 * DAY_MS, now);
        teach_at_age(&mut brain, "m", "n", 500 * DAY_MS, now);
        let st = brain.freshness_status(0);
        assert_eq!(st.bucket_24h, 1, "24h");
        assert_eq!(st.bucket_7d, 1, "7d");
        assert_eq!(st.bucket_30d, 1, "30d");
        assert_eq!(st.bucket_90d, 1, "90d");
        assert_eq!(st.bucket_180d, 1, "180d");
        assert_eq!(st.bucket_365d, 1, "365d");
        assert_eq!(st.bucket_older, 1, "older");
        assert_eq!(st.total_bonds, 7);
    }

    /// T.5 — Decay is purely query-time. Stored bond weights and
    /// keyframes are never mutated by decay reads.
    #[test]
    fn t5_decay_does_not_mutate_stored_weight() {
        let mut brain = Brain::new();
        let r = brain.teach("alpha", BondType::IsA, "beta", 0.9, Confidence::Verified);
        let key = (r.subject_seed, r.object_seed);
        let snapshot_kfs = brain.network.bonds_raw().get(&key).unwrap().keyframes.clone();
        let orig_stored = brain
            .network.bonds().get(&key).unwrap()
            .weight_at(i64::MAX, &brain.scar_log).effective_weight;
        // Backdate.
        {
            let bond = brain.network.bonds_mut_raw().get_mut(&key).unwrap();
            bond.keyframes[0].timestamp = now_ms() - 200 * DAY_MS;
        }
        for _ in 0..5 {
            let _ = brain.bond_effective_weight_with_decay(
                r.subject_seed, r.object_seed, BondType::IsA);
        }
        let final_stored = brain
            .network.bonds().get(&key).unwrap()
            .weight_at(i64::MAX, &brain.scar_log).effective_weight;
        assert!((final_stored - orig_stored).abs() < 1e-6,
            "stored weight must not change: {} -> {}", orig_stored, final_stored);
        // Keyframe weights unchanged (only the timestamp was set by us).
        let final_kfs = brain.network.bonds_raw().get(&key).unwrap().keyframes.clone();
        assert_eq!(snapshot_kfs.len(), final_kfs.len());
        for (a, b) in snapshot_kfs.iter().zip(final_kfs.iter()) {
            assert!((a.weight - b.weight).abs() < 1e-6,
                "keyframe weight must not mutate: {} vs {}", a.weight, b.weight);
        }
    }

    fn make_pending(subject: &str, age_ms: i64, now: u64) -> PendingChange {
        PendingChange {
            pending_id: 0,
            proposed_at_ms: now.saturating_sub(age_ms.max(0) as u64),
            proposed_by: SourceAttribution::new(SourceKind::Teach {
                actor: "t5".into(), ts_ms: now,
            }),
            change_kind: PendingChangeKind::AddBond {
                subject_text: subject.into(),
                object_text: "obj".into(),
                bond_type: BondType::IsA,
                weight: 0.7,
                negated: false,
            },
            confidence: 0.8,
            reason: "t5 test".into(),
            status: PendingStatus::Pending,
        }
    }

    /// T.5 — pending_changes is ordered newest-first.
    #[test]
    fn t5_pending_changes_lists_newest_first() {
        let mut brain = Brain::new();
        let now = now_ms() as u64;
        for (s, age) in [("oldest", 60 * DAY_MS), ("middle", 5 * DAY_MS), ("newest", 0)] {
            brain.pending.push(make_pending(s, age, now)).unwrap();
        }
        let resp = crate::mcp_t3::tool_pending_changes(&mut brain, serde_json::json!({})).unwrap();
        let entries = resp["structuredContent"]["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        let summaries: Vec<&str> = entries.iter()
            .filter_map(|e| e["change_summary"].as_str()).collect();
        assert!(summaries[0].contains("newest"),
            "newest must come first: {:?}", summaries);
        assert!(summaries[2].contains("oldest"),
            "oldest must come last: {:?}", summaries);
    }

    /// T.5 — `stale_count` reports Pending entries older than 30 days.
    #[test]
    fn t5_pending_stale_count_kicks_in_after_30d() {
        let mut brain = Brain::new();
        let now = now_ms() as u64;
        brain.pending.push(make_pending("a", 40 * DAY_MS, now)).unwrap();
        brain.pending.push(make_pending("b", 35 * DAY_MS, now)).unwrap();
        brain.pending.push(make_pending("c", 5 * DAY_MS, now)).unwrap();
        let resp = crate::mcp_t3::tool_pending_changes(&mut brain, serde_json::json!({})).unwrap();
        let stale = resp["structuredContent"]["stale_count"].as_u64().unwrap();
        assert_eq!(stale, 2, "two of three pending entries are >30d old");
    }

    struct MockClock { utc_ms: u64 }
    impl MockClock { fn at_ms(utc_ms: u64) -> Self { MockClock { utc_ms } } }
    impl ClockSource for MockClock {
        fn fetch(&self) -> Result<AuthenticatedTime, ClockError> {
            Ok(AuthenticatedTime::unsigned(
                self.utc_ms,
                MonotonicClock::now(),
                SourceId::system(),
                self.utc_ms.saturating_add(TimeConfig::default().refresh_interval_ms),
            ))
        }
        fn name(&self) -> &str { "mock" }
        fn is_authenticated(&self) -> bool { false }
    }

    /// T.5 — Decay routes through `ClockSource`, not `SystemTime::now`.
    #[test]
    fn t5_decay_uses_clock_source_not_system_time() {
        let mut brain = Brain::new();
        let r = brain.teach("alpha", BondType::IsA, "beta", 0.9, Confidence::Verified);
        let key = (r.subject_seed, r.object_seed);
        let real_now = now_ms();
        {
            let bond = brain.network.bonds_mut_raw().get_mut(&key).unwrap();
            bond.keyframes[0].timestamp = real_now;
        }
        // Without mock: freshness ~= 1.0.
        let f0 = brain.freshness_for_bond_pair(r.subject_seed, r.object_seed, BondType::IsA);
        assert!((f0 - 1.0).abs() < 0.05,
            "without mock, freshness ~ 1.0 (got {})", f0);
        // Inject mock clock 365d in the future.
        let mock = Arc::new(MockClock::at_ms((real_now + 365 * DAY_MS) as u64));
        brain.set_clock_source(mock).unwrap();
        let f1 = brain.freshness_for_bond_pair(r.subject_seed, r.object_seed, BondType::IsA);
        // 365 / 90 = ~4.06 half-lives → ~0.060.
        assert!(f1 < 0.10, "mock-future freshness must drop sharply: {}", f1);
        assert!(f1 > 0.04, "but stay above ~0.04: {}", f1);
    }

    /// T.5 — Disabling decay returns unmodified weights.
    #[test]
    fn t5_decay_disabled_returns_unmodified_weights() {
        let mut brain = Brain::new();
        let r = brain.teach("alpha", BondType::IsA, "beta", 0.9, Confidence::Verified);
        let key = (r.subject_seed, r.object_seed);
        {
            let bond = brain.network.bonds_mut_raw().get_mut(&key).unwrap();
            bond.keyframes[0].timestamp = now_ms() - 365 * DAY_MS;
        }
        let stored = brain
            .network.bonds().get(&key).unwrap()
            .weight_at(i64::MAX, &brain.scar_log).effective_weight;
        let with_decay = brain
            .bond_effective_weight_with_decay(r.subject_seed, r.object_seed, BondType::IsA)
            .unwrap();
        assert!(with_decay < stored,
            "decay must reduce effective: {} < {}", with_decay, stored);
        brain.decay_config.enabled = false;
        let no_decay = brain
            .bond_effective_weight_with_decay(r.subject_seed, r.object_seed, BondType::IsA)
            .unwrap();
        assert!((no_decay - stored).abs() < 1e-5,
            "decay-disabled must equal stored: {} vs {}", no_decay, stored);
    }
}
