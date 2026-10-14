//! Port CS — Conversational Style adaptation.
//!
//! Lu observes how the user talks (verbosity, formality, technicality,
//! empathy) and persists a slow-moving EMA per user_key. The vector
//! decays toward neutral (0.5) using the same freshness curve as
//! Port T.5 — if the user goes quiet for months, the profile drifts
//! back to defaults; if they talk daily, it stays sharp.
//!
//! Persistence: `ConversationalStyleSnapshot` round-trips through the
//! v3 binary section #12 (`SectionIdV3::ConversationalStyle`), bumping
//! `SECTION_COUNT` from 11 to 12. `MIN_SUPPORTED_SECTION_COUNT` stays
//! at 7 — pre-CS brains load with an empty store via `serde(default)`
//! and the read-side `de_bincode_optional` helper (J.1.1 forward-compat).
//!
//! Port CS.2 added the OUTPUT-adaptation half: `StyleHints` (this
//! module) is the actionable shape derived from a decayed `StyleVector`
//! and consumed by the articulator path in `narration.rs`. Style
//! adaptation is presentation-only — it shifts response prose but
//! does NOT change which bonds are activated or which facts are
//! retrieved (anti-hallucination invariant preserved).
//!
//! Cross-references:
//!   * Port T.5 (decay curve) — reused for the style-vector decay-on-read.
//!   * Port T.4 (v3 section pattern) — on-disk shape this port copies.
//!   * Port LB (v3 section #11) — immediate predecessor, same scaffold.
//!   * Port Cat.3c (RuntimeConfig section #14) — where the new CS.2
//!     persistence knobs live (`output_adaptation_enabled`,
//!     `max_samples_for_full_weight`).
//!   * Encryption (Solution #79 / Learning #621 / Port T.7) — invariant
//!     preserved; new bytes inside the same v3enc envelope.
//!
//! Anti-hallucination invariant: style updates do NOT mutate
//! `bond_count` or any canonical brain state. Style is presentation
//! shape, not facts.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

// Port CS.1 — Write-Ahead Log for between-saves durability.
pub mod wal;
pub use wal::{StyleWalRecord, StyleWalWriter};

/// Default profile value for every dimension when a user has no
/// observations on file (or after `lu4d_style_reset`).
pub const STYLE_DEFAULT: f32 = 0.5;

/// Below this sample count the per-update learning rate is dampened
/// linearly to avoid one-message overfitting on a fresh profile.
pub const FULL_WEIGHT_SAMPLES: u64 = 100;

/// Default EMA learning rate, post-dampening.
pub const DEFAULT_LEARNING_RATE: f32 = 0.05;
const MIN_LEARNING_RATE: f32 = 0.001;
const MAX_LEARNING_RATE: f32 = 0.5;

/// Default half-life for the decay-on-read curve.
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: u32 = 90;
const MIN_DECAY_HALF_LIFE_DAYS: u32 = 1;
const MAX_DECAY_HALF_LIFE_DAYS: u32 = 3650;

const MS_PER_DAY: u64 = 86_400_000;

// =====================================================================
// Port CS.2 — output-adaptation knobs
// =====================================================================

/// Default for `StyleConfig::output_adaptation_enabled`. CS.2 ships
/// with adaptation ON so the retrained brain gets style-adapted output
/// from day one. Flip via `lu4d_configure
/// style_output_adaptation_enabled=false` for A/B inspection or to
/// lock down to neutral output.
pub const DEFAULT_OUTPUT_ADAPTATION_ENABLED: bool = true;

/// Default for `StyleConfig::max_samples_for_full_weight`. Below this
/// many samples the hint magnitude is dampened toward neutral (0.5)
/// to avoid over-adapting on a handful of interactions. 100 is
/// deliberately the same constant as `FULL_WEIGHT_SAMPLES` so the
/// observe-side EMA dampener and the read-side hint dampener line up
/// by default — but they're decoupled so each can be retuned
/// independently.
pub const DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT: u64 = 100;
const MIN_MAX_SAMPLES_FOR_FULL_WEIGHT: u64 = 1;
const MAX_MAX_SAMPLES_FOR_FULL_WEIGHT: u64 = 1_000_000;

/// Per-message stylistic features extracted from raw input text.
///
/// All fields are clamped to [0, 1]. Sparse signals (e.g. very short
/// inputs) still produce a sample but with mid-range values; the
/// EMA smooths them out across many observations.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct StyleSample {
    pub verbosity: f32,
    pub formality: f32,
    pub technicality: f32,
    pub empathy: f32,
}

/// Persisted style profile per user_key. Fields are mid-scale
/// (0.5) on a fresh brain; observe() drives them toward observed
/// values via EMA.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StyleVector {
    pub verbosity: f32,
    pub formality: f32,
    pub technicality: f32,
    pub empathy: f32,
    pub samples_count: u64,
    pub last_refined_at_ms: u64,
}

impl Default for StyleVector {
    fn default() -> Self {
        StyleVector {
            verbosity: STYLE_DEFAULT,
            formality: STYLE_DEFAULT,
            technicality: STYLE_DEFAULT,
            empathy: STYLE_DEFAULT,
            samples_count: 0,
            last_refined_at_ms: 0,
        }
    }
}

impl StyleVector {
    /// EMA update toward `sample`. The effective learning rate is
    /// `lr * min(1, samples_count / FULL_WEIGHT_SAMPLES)` for the
    /// first FULL_WEIGHT_SAMPLES observations to avoid one-shot
    /// overfitting on a fresh profile. After that, the full lr applies.
    pub fn observe(&mut self, sample: &StyleSample, lr: f32, now_ms: u64) {
        let lr = lr.clamp(MIN_LEARNING_RATE, MAX_LEARNING_RATE);
        let dampen = if self.samples_count >= FULL_WEIGHT_SAMPLES {
            1.0
        } else {
            // Use samples_count + 1 in the numerator so the very
            // first observation isn't multiplied by 0.
            ((self.samples_count as f32 + 1.0) / FULL_WEIGHT_SAMPLES as f32).clamp(0.0, 1.0)
        };
        let eff = lr * dampen;
        self.verbosity = ema(self.verbosity, sample.verbosity, eff);
        self.formality = ema(self.formality, sample.formality, eff);
        self.technicality = ema(self.technicality, sample.technicality, eff);
        self.empathy = ema(self.empathy, sample.empathy, eff);
        self.samples_count = self.samples_count.saturating_add(1);
        self.last_refined_at_ms = now_ms;
    }

    /// Decay-on-read: lerp every dimension back toward STYLE_DEFAULT
    /// by `(1 - freshness)`, where `freshness = 2^(-age_days /
    /// half_life_days)` (T.5 curve). `samples_count` and
    /// `last_refined_at_ms` are NOT decayed — they're observation
    /// counters, not adaptive values.
    pub fn decayed(&self, now_ms: u64, half_life_days: u32) -> StyleVector {
        if self.samples_count == 0 || self.last_refined_at_ms == 0 {
            return *self;
        }
        let half_life = half_life_days.max(1) as f32;
        let age_ms = now_ms.saturating_sub(self.last_refined_at_ms) as f32;
        let age_days = age_ms / MS_PER_DAY as f32;
        // freshness = 2^(-age_days / half_life)
        let freshness = (-(age_days) / half_life * std::f32::consts::LN_2).exp();
        let freshness = freshness.clamp(0.0, 1.0);
        StyleVector {
            verbosity: lerp(STYLE_DEFAULT, self.verbosity, freshness),
            formality: lerp(STYLE_DEFAULT, self.formality, freshness),
            technicality: lerp(STYLE_DEFAULT, self.technicality, freshness),
            empathy: lerp(STYLE_DEFAULT, self.empathy, freshness),
            samples_count: self.samples_count,
            last_refined_at_ms: self.last_refined_at_ms,
        }
    }
}

fn ema(prev: f32, sample: f32, lr: f32) -> f32 {
    let v = (1.0 - lr) * prev + lr * sample;
    v.clamp(0.0, 1.0)
}

fn lerp(toward: f32, from: f32, t: f32) -> f32 {
    // t=1 → from (fully fresh, no decay), t=0 → toward (fully decayed).
    toward + (from - toward) * t
}

// =====================================================================
// Sample extraction (Layer A)
// =====================================================================

/// Heuristic sample from a raw input string. Values are deliberately
/// directional, not exact — the EMA smooths them across many
/// observations.
pub fn compute_style_sample(input: &str) -> StyleSample {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return StyleSample {
            verbosity: STYLE_DEFAULT,
            formality: STYLE_DEFAULT,
            technicality: STYLE_DEFAULT,
            empathy: STYLE_DEFAULT,
        };
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let word_count = words.len() as f32;
    StyleSample {
        verbosity: verbosity_score(word_count, trimmed.len() as f32),
        formality: formality_score(&words, trimmed),
        technicality: technicality_score(&words),
        empathy: empathy_score(&words, trimmed),
    }
}

/// Verbosity scales with word count via a soft sigmoid centered
/// around 30 words = 0.5. ≤5 words → ~0, ≥100 words → ~1.
fn verbosity_score(word_count: f32, _char_count: f32) -> f32 {
    let x = (word_count - 30.0) / 25.0;
    let v = 1.0 / (1.0 + (-x).exp());
    v.clamp(0.0, 1.0)
}

fn formality_score(words: &[&str], _full: &str) -> f32 {
    // Casual markers vs formal markers. Each hit shifts ±0.1 from
    // mid-scale, clamped to [0, 1].
    let casual = [
        "lol", "lmao", "rofl", "wat", "wut", "ty", "thx", "yeah", "yep", "nope",
        "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "u", "ur", "r",
        "omg", "idk", "imo", "tbh", "fwiw", "ngl",
    ];
    let formal = [
        "please", "thank", "thanks", "would", "could", "should", "kindly",
        "appreciate", "regards", "sincerely", "furthermore", "however",
        "therefore", "nevertheless", "accordingly", "hereby",
    ];
    let mut score = 0.5_f32;
    for w in words {
        let lw = w.to_ascii_lowercase();
        let lw = lw.trim_matches(|c: char| !c.is_alphanumeric());
        if casual.iter().any(|m| *m == lw) { score -= 0.10; }
        if formal.iter().any(|m| *m == lw) { score += 0.08; }
    }
    score.clamp(0.0, 1.0)
}

fn technicality_score(words: &[&str]) -> f32 {
    if words.is_empty() {
        return STYLE_DEFAULT;
    }
    let long: usize = words
        .iter()
        .filter(|w| {
            // Strip surrounding punctuation; count letters only.
            let len = w.chars().filter(|c| c.is_alphabetic()).count();
            len >= 8
        })
        .count();
    let ratio = long as f32 / words.len() as f32;
    // Map 0.0 → 0.2, 0.4 → 1.0 with linear scaling, clamp.
    let v = 0.2 + (ratio / 0.4) * 0.8;
    v.clamp(0.0, 1.0)
}

fn empathy_score(words: &[&str], full: &str) -> f32 {
    let warm = [
        "thanks", "thank", "please", "sorry", "appreciate", "love", "happy",
        "glad", "hope", "care", "friend", "feel", "feeling",
    ];
    let second_person = ["you", "your", "you're", "youre", "yours"];
    let mut score = 0.5_f32;
    for w in words {
        let lw = w.to_ascii_lowercase();
        let lw = lw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        if warm.iter().any(|m| *m == lw) { score += 0.06; }
        if second_person.iter().any(|m| *m == lw) { score += 0.04; }
    }
    if full.contains('!') { score += 0.03; }
    score.clamp(0.0, 1.0)
}

// =====================================================================
// Store (Layer C — in-memory + persistence shape)
// =====================================================================

/// In-memory profile store. Future-proofed for multi-user via
/// `BTreeMap<String, StyleVector>` keyed by user_key; today Lu uses
/// the literal "default" for single-user mode.
///
/// Port CS.1 added an optional `<brain>.style.wal` handle. When
/// attached (via [`Brain::attach_cs_style_wal`] on a path-bound brain
/// load), every `observe` / `set` / `reset` mutation is appended to
/// the WAL with `flush + sync_all` BEFORE the in-memory mutation
/// lands. The WAL is truncated by the brain save path after the v3
/// section #12 snapshot becomes durable. In-memory-only stores
/// (tests, ephemeral brains) leave `wal` at `None` and skip the
/// append step entirely — same trade-off as LB.1's `SessionMemoryStore`.
#[derive(Default)]
pub struct ConversationalStyleStore {
    pub profiles: BTreeMap<String, StyleVector>,
    /// Port CS.1 — optional WAL append handle. Populated by
    /// `Brain::attach_cs_style_wal`; absent for `Brain::new`
    /// in-memory brains and for tests that build a store directly.
    wal: Option<StyleWalWriter>,
}

impl Clone for ConversationalStyleStore {
    /// `StyleWalWriter` is not `Clone` (it owns a fresh file handle).
    /// Clones therefore discard the WAL handle — the cloned store is
    /// in-memory-only. Production code paths only clone snapshots,
    /// not the full store; this matches LB.1's approach (its store
    /// is not `Clone` at all).
    fn clone(&self) -> Self {
        Self {
            profiles: self.profiles.clone(),
            wal: None,
        }
    }
}

impl std::fmt::Debug for ConversationalStyleStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationalStyleStore")
            .field("profiles", &self.profiles.len())
            .field("wal_path", &self.wal.as_ref().map(|w| w.path().to_path_buf()))
            .finish()
    }
}

impl ConversationalStyleStore {
    pub fn new() -> Self { Self::default() }

    pub fn from_snapshot(snap: ConversationalStyleSnapshot) -> Self {
        ConversationalStyleStore {
            profiles: snap.profiles,
            wal: None,
        }
    }

    /// Port CS.1 — load a store from a snapshot AND replay the WAL
    /// at `wal_path` on top. Idempotent: each WAL record is skipped
    /// when the in-memory state already reflects an equally-recent
    /// or newer mutation (`existing.last_refined_at_ms >= rec.ts_ms`),
    /// so re-running replay against a state that already absorbed
    /// the WAL produces no duplicate effect.
    pub fn from_snapshot_with_wal(
        snap: ConversationalStyleSnapshot,
        wal_path: PathBuf,
    ) -> Result<Self, crate::brain::BrainError> {
        let mut store = Self::from_snapshot(snap);
        let records = wal::read_all(&wal_path)?;
        for rec in records {
            store.apply_wal_record(rec);
        }
        store.wal = Some(StyleWalWriter::open(wal_path)?);
        Ok(store)
    }

    /// Replay one WAL record into the in-memory state. Idempotent:
    /// uses the section-vs-WAL `last_refined_at_ms` ordering as the
    /// dedupe key. Records older or equal to the existing state are
    /// dropped on the assumption that the section already absorbed
    /// them (the truncate-after-save path makes this the common case).
    fn apply_wal_record(&mut self, rec: StyleWalRecord) {
        match rec {
            StyleWalRecord::Observe { user_key, sample, learning_rate, ts_ms } => {
                let entry = self.profiles
                    .entry(user_key)
                    .or_insert_with(StyleVector::default);
                if entry.last_refined_at_ms >= ts_ms && ts_ms != 0 {
                    // Section already reflects this observation.
                    return;
                }
                entry.observe(&sample, learning_rate, ts_ms);
            }
            StyleWalRecord::Set { user_key, vector, ts_ms } => {
                let already = self.profiles.get(&user_key)
                    .map(|v| v.last_refined_at_ms)
                    .unwrap_or(0);
                if already >= ts_ms && ts_ms != 0 {
                    return;
                }
                self.profiles.insert(user_key, vector);
            }
            StyleWalRecord::Reset { user_key, ts_ms } => {
                let already = self.profiles.get(&user_key)
                    .map(|v| v.last_refined_at_ms)
                    .unwrap_or(0);
                // A reset replays only when no profile exists, or the
                // existing profile predates the reset. A profile newer
                // than the reset must have been re-created by a later
                // observe/set in the WAL, which a strict in-order
                // replay will land before this Reset record anyway.
                if already > ts_ms {
                    return;
                }
                self.profiles.remove(&user_key);
            }
        }
    }

    /// Port CS.1 — attach a freshly-opened WAL writer (used by tests
    /// and `Brain::attach_cs_style_wal`).
    pub fn attach_wal(&mut self, wal: StyleWalWriter) {
        self.wal = Some(wal);
    }

    /// Port CS.1 — reset the WAL after a successful brain save.
    /// Mirrors `PendingLog::truncate_wal` and
    /// `SessionMemoryStore::truncate_wal`. Idempotent on the no-WAL
    /// case (`Brain::new` in-memory brains skip the WAL entirely).
    pub fn wal_position(&mut self) -> Result<Option<u64>, crate::brain::BrainError> {
        match self.wal.as_mut() {
            Some(w) => Ok(Some(w.position()?)),
            None => Ok(None),
        }
    }

    pub fn truncate_wal(&mut self, watermark: Option<u64>) -> Result<(), crate::brain::BrainError> {
        if let Some(w) = self.wal.as_mut() {
            w.truncate(watermark)?;
        }
        Ok(())
    }

    /// Port CS.1 — debug accessor for the on-disk WAL path. None for
    /// in-memory stores.
    pub fn wal_path(&self) -> Option<&std::path::Path> {
        self.wal.as_ref().map(|w| w.path())
    }

    pub fn snapshot(&self) -> ConversationalStyleSnapshot {
        ConversationalStyleSnapshot {
            profiles: self.profiles.clone(),
            version: 1,
        }
    }

    /// Get the raw persisted vector (no decay applied). Returns the
    /// default vector if the user has no profile yet.
    pub fn get(&self, user_key: &str) -> StyleVector {
        self.profiles.get(user_key).copied().unwrap_or_default()
    }

    /// Get the decayed-effective vector for the moment `now_ms`.
    pub fn get_decayed(&self, user_key: &str, now_ms: u64, half_life_days: u32) -> StyleVector {
        self.get(user_key).decayed(now_ms, half_life_days)
    }

    pub fn observe(&mut self, user_key: &str, sample: &StyleSample, lr: f32, now_ms: u64) {
        // Port CS.1 — append to WAL BEFORE the in-memory mutation
        // lands. A WAL append failure surfaces via stderr but does
        // NOT block the in-memory observe: style is presentation
        // and the next save's section #12 will reconcile the state
        // anyway. Same best-effort policy as
        // `SessionMemoryStore::archive_session`.
        if let Some(w) = self.wal.as_mut() {
            let rec = StyleWalRecord::Observe {
                user_key: user_key.to_string(),
                sample: *sample,
                learning_rate: lr,
                ts_ms: now_ms,
            };
            if let Err(e) = w.append(&rec) {
                eprintln!(
                    "cs_style WAL append (observe) failed: {} (in-memory observe still applied)",
                    e
                );
            }
        }
        let entry = self.profiles
            .entry(user_key.to_string())
            .or_insert_with(StyleVector::default);
        entry.observe(sample, lr, now_ms);
    }

    pub fn reset(&mut self, user_key: &str) {
        // Port CS.1 — WAL append before in-memory mutation. ts_ms=0
        // is used as the WAL clock when the caller didn't thread a
        // clock through — the idempotent-replay guard treats ts_ms=0
        // as "always apply".
        if let Some(w) = self.wal.as_mut() {
            let rec = StyleWalRecord::Reset {
                user_key: user_key.to_string(),
                ts_ms: now_ms_best_effort(),
            };
            if let Err(e) = w.append(&rec) {
                eprintln!(
                    "cs_style WAL append (reset) failed: {} (in-memory reset still applied)",
                    e
                );
            }
        }
        self.profiles.remove(user_key);
    }

    pub fn set(&mut self, user_key: &str, vector: StyleVector) {
        if let Some(w) = self.wal.as_mut() {
            let rec = StyleWalRecord::Set {
                user_key: user_key.to_string(),
                vector,
                ts_ms: vector.last_refined_at_ms.max(now_ms_best_effort()),
            };
            if let Err(e) = w.append(&rec) {
                eprintln!(
                    "cs_style WAL append (set) failed: {} (in-memory set still applied)",
                    e
                );
            }
        }
        self.profiles.insert(user_key.to_string(), vector);
    }
}

/// Wall-clock helper for the WAL clock when the caller didn't
/// provide one. Returns 0 if the system clock is before the UNIX
/// epoch (impossible in practice but keeps the helper infallible).
fn now_ms_best_effort() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// On-disk shape persisted into v3 section #12.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ConversationalStyleSnapshot {
    #[serde(default)]
    pub profiles: BTreeMap<String, StyleVector>,
    #[serde(default)]
    pub version: u32,
}

// =====================================================================
// Configuration knobs (surfaced via lu4d_configure)
// =====================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StyleConfig {
    pub learning_enabled: bool,
    pub learning_rate: f32,
    pub decay_half_life_days: u32,
    pub user_key: [u8; 16], // small inline string, padded with 0s
    /// Port CS.2: master switch for the output-adaptation layer. When
    /// false, `StyleHints::from_style_vector` collapses every
    /// dimension to neutral (0.5) regardless of the underlying
    /// vector — articulator output reverts to pre-CS.2 baseline.
    /// Default: true (ship with adaptation on).
    #[serde(default = "default_output_adaptation_enabled")]
    pub output_adaptation_enabled: bool,
    /// Port CS.2: dampening control for `StyleHints`. Below this
    /// many samples the hint magnitude is reduced toward 0.5 so a
    /// brand-new profile doesn't over-adapt on 3 observations.
    /// Default: 100 (matches the EMA-side `FULL_WEIGHT_SAMPLES` by
    /// convention, but decoupled).
    #[serde(default = "default_max_samples_for_full_weight")]
    pub max_samples_for_full_weight: u64,
}

fn default_output_adaptation_enabled() -> bool { DEFAULT_OUTPUT_ADAPTATION_ENABLED }
fn default_max_samples_for_full_weight() -> u64 { DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT }

impl Default for StyleConfig {
    fn default() -> Self {
        let mut user_key = [0u8; 16];
        let bytes = b"default";
        user_key[..bytes.len()].copy_from_slice(bytes);
        StyleConfig {
            learning_enabled: true,
            learning_rate: DEFAULT_LEARNING_RATE,
            decay_half_life_days: DEFAULT_DECAY_HALF_LIFE_DAYS,
            user_key,
            output_adaptation_enabled: DEFAULT_OUTPUT_ADAPTATION_ENABLED,
            max_samples_for_full_weight: DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT,
        }
    }
}

impl StyleConfig {
    pub fn user_key_str(&self) -> &str {
        let end = self.user_key.iter().position(|&b| b == 0).unwrap_or(self.user_key.len());
        std::str::from_utf8(&self.user_key[..end]).unwrap_or("default")
    }

    pub fn set_learning_rate(&mut self, lr: f32) -> Result<f32, String> {
        if lr < MIN_LEARNING_RATE || lr > MAX_LEARNING_RATE {
            return Err(format!(
                "learning_rate out of bounds [{}, {}]: {}",
                MIN_LEARNING_RATE, MAX_LEARNING_RATE, lr
            ));
        }
        let prev = self.learning_rate;
        self.learning_rate = lr;
        Ok(prev)
    }

    pub fn set_decay_half_life_days(&mut self, days: u32) -> Result<u32, String> {
        if days < MIN_DECAY_HALF_LIFE_DAYS || days > MAX_DECAY_HALF_LIFE_DAYS {
            return Err(format!(
                "style_decay_half_life_days out of bounds [{}, {}]: {}",
                MIN_DECAY_HALF_LIFE_DAYS, MAX_DECAY_HALF_LIFE_DAYS, days
            ));
        }
        let prev = self.decay_half_life_days;
        self.decay_half_life_days = days;
        Ok(prev)
    }

    /// Port CS.2 — bounded setter for `max_samples_for_full_weight`.
    pub fn set_max_samples_for_full_weight(&mut self, n: u64) -> Result<u64, String> {
        if n < MIN_MAX_SAMPLES_FOR_FULL_WEIGHT || n > MAX_MAX_SAMPLES_FOR_FULL_WEIGHT {
            return Err(format!(
                "style_max_samples_for_full_weight out of bounds [{}, {}]: {}",
                MIN_MAX_SAMPLES_FOR_FULL_WEIGHT, MAX_MAX_SAMPLES_FOR_FULL_WEIGHT, n
            ));
        }
        let prev = self.max_samples_for_full_weight;
        self.max_samples_for_full_weight = n;
        Ok(prev)
    }
}

pub const STYLE_LEARNING_RATE_BOUNDS: (f32, f32) = (MIN_LEARNING_RATE, MAX_LEARNING_RATE);
pub const STYLE_DECAY_HALF_LIFE_BOUNDS: (u32, u32) =
    (MIN_DECAY_HALF_LIFE_DAYS, MAX_DECAY_HALF_LIFE_DAYS);
pub const STYLE_MAX_SAMPLES_FOR_FULL_WEIGHT_BOUNDS: (u64, u64) =
    (MIN_MAX_SAMPLES_FOR_FULL_WEIGHT, MAX_MAX_SAMPLES_FOR_FULL_WEIGHT);

// =====================================================================
// Port CS.2 — StyleHints (Layer D: presentation shaping)
// =====================================================================

/// Actionable response-shaping parameters derived from a (decayed)
/// `StyleVector`. Articulators read these in `[0, 1]` space and apply
/// graduated knobs: low → terse / casual / plain / neutral; high →
/// expansive / formal / technical / warm. Each dimension is
/// independent — the articulator may apply some and leave others
/// unchanged depending on the response shape.
///
/// Semantics:
///   * 0.0 → push as far in the "low" direction as the articulator
///     gracefully can.
///   * 0.5 → no adaptation; emit baseline output.
///   * 1.0 → push as far in the "high" direction as the articulator
///     gracefully can.
///
/// Anti-hallucination invariant: hints control PRESENTATION only.
/// `StyleHints` never affects which bonds are activated, which
/// facts are retrieved, or which response shape is selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StyleHints {
    /// 0.0 = terse single-sentence; 1.0 = expansive multi-sentence with examples.
    pub max_response_length_factor: f32,
    /// 0.0 = casual / contractions / direct address; 1.0 = formal / no contractions / impersonal.
    pub formality: f32,
    /// 0.0 = layperson explanations / analogies; 1.0 = jargon-heavy / unexplained terms.
    pub technicality: f32,
    /// 0.0 = neutral / direct; 1.0 = warm / acknowledging.
    pub empathy: f32,
    /// Source samples count from the underlying `StyleVector`. When
    /// low, the articulator (or the dampening helper) should pull
    /// each dimension toward 0.5 to avoid over-adapting on a few
    /// interactions. Carried verbatim so callers can apply their
    /// own confidence-based gating beyond the built-in dampening.
    pub samples_count: u64,
}

impl Default for StyleHints {
    fn default() -> Self {
        StyleHints::neutral()
    }
}

impl StyleHints {
    /// Hints with every dimension at 0.5 — no adaptation. Used as
    /// the kill-switch fallback when
    /// `StyleConfig::output_adaptation_enabled` is false, or as
    /// the explicit "do not shape" sentinel.
    pub fn neutral() -> Self {
        StyleHints {
            max_response_length_factor: STYLE_DEFAULT,
            formality: STYLE_DEFAULT,
            technicality: STYLE_DEFAULT,
            empathy: STYLE_DEFAULT,
            samples_count: 0,
        }
    }

    /// Convert a (decayed) `StyleVector` into actionable hints.
    /// Pre-dampens the result based on `samples_count` vs
    /// `cfg.max_samples_for_full_weight` so callers don't have to
    /// remember to call `dampened` separately. If
    /// `cfg.output_adaptation_enabled` is false, returns
    /// `StyleHints::neutral()` regardless of `sv` — the kill-switch
    /// path.
    pub fn from_style_vector(sv: &StyleVector, cfg: &StyleConfig) -> Self {
        if !cfg.output_adaptation_enabled {
            return StyleHints::neutral();
        }
        let raw = StyleHints {
            // verbosity → length factor — direct mapping.
            max_response_length_factor: sv.verbosity.clamp(0.0, 1.0),
            formality: sv.formality.clamp(0.0, 1.0),
            technicality: sv.technicality.clamp(0.0, 1.0),
            empathy: sv.empathy.clamp(0.0, 1.0),
            samples_count: sv.samples_count,
        };
        raw.dampened(cfg.max_samples_for_full_weight)
    }

    /// Return a copy with each dimension pulled toward 0.5
    /// proportionally to `min(1, samples_count / max_samples)`.
    /// Examples (max_samples = 100):
    ///   * samples_count = 0   → returns neutral (every dim = 0.5).
    ///   * samples_count = 50  → midpoint between neutral and self.
    ///   * samples_count >=100 → full magnitude (no dampening).
    /// `samples_count` itself is preserved.
    pub fn dampened(&self, max_samples: u64) -> Self {
        if max_samples == 0 {
            // Defensive: never divide by zero. max_samples=0 means
            // the dampening curve doesn't apply, which we model as
            // full weight everywhere.
            return *self;
        }
        let weight = (self.samples_count as f32 / max_samples as f32).clamp(0.0, 1.0);
        StyleHints {
            max_response_length_factor: dampen_dim(self.max_response_length_factor, weight),
            formality: dampen_dim(self.formality, weight),
            technicality: dampen_dim(self.technicality, weight),
            empathy: dampen_dim(self.empathy, weight),
            samples_count: self.samples_count,
        }
    }

    /// True iff every dimension is "close enough" to 0.5 that no
    /// articulator is expected to apply a noticeable change. Used as
    /// a fast skip check at the start of `apply_style_modulations`.
    pub fn is_neutral(&self) -> bool {
        const EPS: f32 = 1e-3;
        (self.max_response_length_factor - STYLE_DEFAULT).abs() < EPS
            && (self.formality - STYLE_DEFAULT).abs() < EPS
            && (self.technicality - STYLE_DEFAULT).abs() < EPS
            && (self.empathy - STYLE_DEFAULT).abs() < EPS
    }

    /// Coarse verbosity tier used by render_core. OP-NARRATE-1 Phase C.
    pub fn verbosity_tier(&self) -> VerbosityTier {
        if self.max_response_length_factor < 0.4 {
            VerbosityTier::Terse
        } else {
            VerbosityTier::Standard
        }
    }
}

/// Two-tier verbosity classification for per-shape render arms.
/// Standard = todays verbose template. Terse = facts-preserving compact form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbosityTier { Terse, Standard, }

/// Pull `value` toward 0.5 by `(1 - weight)`. weight=1 → unchanged;
/// weight=0 → fully neutralized.
fn dampen_dim(value: f32, weight: f32) -> f32 {
    STYLE_DEFAULT + (value - STYLE_DEFAULT) * weight
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cs_style_sample_extracts_verbosity_from_long_vs_short_input() {
        let short = compute_style_sample("wat");
        let long = compute_style_sample(
            "Could you please elaborate at considerable length on the conceptual underpinnings \
             of the persistence layer, including how the decay function interacts with the \
             EMA observation path, and walk me through any edge cases you've considered for \
             multi-user keying down the line, thank you so much for your patience here.",
        );
        assert!(short.verbosity < 0.3, "short verbosity should be low: {}", short.verbosity);
        assert!(long.verbosity > 0.65, "long verbosity should be high: {}", long.verbosity);
    }

    #[test]
    fn cs_style_sample_extracts_formality_from_casual_vs_formal_input() {
        let casual = compute_style_sample("lol wat ty omg ngl idk");
        let formal = compute_style_sample(
            "Please could you kindly elaborate; I would appreciate it. Thank you sincerely.",
        );
        assert!(casual.formality < 0.4, "casual formality should be low: {}", casual.formality);
        assert!(formal.formality > 0.6, "formal formality should be high: {}", formal.formality);
    }

    #[test]
    fn cs_style_sample_extracts_technicality_from_jargon_density() {
        let lay = compute_style_sample("the cat sat on the mat");
        let tech = compute_style_sample(
            "persistent serialization synchronization heuristic deserialization implementation",
        );
        assert!(lay.technicality < 0.4, "layperson technicality should be low: {}", lay.technicality);
        assert!(tech.technicality > 0.7, "jargon technicality should be high: {}", tech.technicality);
    }

    #[test]
    fn cs_style_observe_updates_via_ema() {
        let mut v = StyleVector::default();
        let s = StyleSample { verbosity: 1.0, formality: 1.0, technicality: 1.0, empathy: 1.0 };
        let baseline = v.verbosity;
        v.observe(&s, 0.5, 1_000);
        // Even with 0.5 lr the dampener (samples_count=0 -> 1/100) keeps
        // the move tiny, but it must move toward 1.0.
        assert!(v.verbosity > baseline, "observe must move toward sample: {}", v.verbosity);
        assert!(v.verbosity < 1.0, "single observe must not jump to extreme: {}", v.verbosity);
    }

    #[test]
    fn cs_style_observe_increments_samples_count() {
        let mut v = StyleVector::default();
        let s = StyleSample::default();
        for i in 1..=5 {
            v.observe(&s, 0.05, i as u64 * 1_000);
        }
        assert_eq!(v.samples_count, 5);
        assert_eq!(v.last_refined_at_ms, 5_000);
    }

    #[test]
    fn cs_style_decays_per_t5_curve_with_aging() {
        let mut v = StyleVector::default();
        // Force the value far from 0.5 ignoring EMA dampening.
        v.verbosity = 0.9;
        v.samples_count = 200;
        v.last_refined_at_ms = 1_000_000;
        // 1 half-life later: midpoint between 0.5 and 0.9 → 0.7
        let one_hl_later = v.last_refined_at_ms + (90 * MS_PER_DAY);
        let dec = v.decayed(one_hl_later, 90);
        assert!((dec.verbosity - 0.7).abs() < 0.02,
            "after 1 half-life expect ~0.7 (midpoint of 0.5..0.9): {}", dec.verbosity);
        // 10 half-lives later → essentially 0.5.
        let ten_hl = v.last_refined_at_ms + (10 * 90 * MS_PER_DAY);
        let far = v.decayed(ten_hl, 90);
        assert!((far.verbosity - 0.5).abs() < 0.01,
            "after 10 half-lives expect ~0.5: {}", far.verbosity);
    }

    #[test]
    fn cs_v3_section_round_trips() {
        // Round-trip via bincode — the actual on-disk encoding.
        let mut store = ConversationalStyleStore::new();
        let s = StyleSample { verbosity: 0.8, formality: 0.7, technicality: 0.6, empathy: 0.55 };
        store.observe("default", &s, 0.1, 1_500_000);
        store.observe("default", &s, 0.1, 1_600_000);
        let snap = store.snapshot();
        let bytes = bincode::serialize(&snap).expect("bincode serialize");
        let back: ConversationalStyleSnapshot =
            bincode::deserialize(&bytes).expect("bincode deserialize");
        assert_eq!(snap, back);
        assert_eq!(back.profiles.len(), 1);
        assert!(back.profiles.contains_key("default"));
    }

    #[test]
    fn cs_pre_cs_brain_loads_with_default_style() {
        // Empty / absent section → default snapshot → empty store.
        let snap = ConversationalStyleSnapshot::default();
        let store = ConversationalStyleStore::from_snapshot(snap);
        assert!(store.profiles.is_empty());
        // Default-vector lookup works for any user_key.
        let v = store.get("default");
        assert_eq!(v, StyleVector::default());
    }

    #[test]
    fn cs_store_reset_clears_profile() {
        let mut store = ConversationalStyleStore::new();
        let s = StyleSample { verbosity: 0.9, formality: 0.9, technicality: 0.9, empathy: 0.9 };
        store.observe("default", &s, 0.5, 1_000);
        assert!(store.profiles.contains_key("default"));
        store.reset("default");
        assert!(!store.profiles.contains_key("default"));
        assert_eq!(store.get("default"), StyleVector::default());
    }

    #[test]
    fn cs_store_set_overrides_persisted_value() {
        let mut store = ConversationalStyleStore::new();
        let mut v = StyleVector::default();
        v.verbosity = 0.9;
        store.set("default", v);
        let got = store.get("default");
        assert!((got.verbosity - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn cs_store_get_decayed_returns_decayed_values() {
        let mut store = ConversationalStyleStore::new();
        let mut v = StyleVector::default();
        v.verbosity = 0.9;
        v.samples_count = 200;
        v.last_refined_at_ms = 1_000_000;
        store.set("default", v);
        let raw = store.get("default");
        let dec = store.get_decayed("default", 1_000_000 + (90 * MS_PER_DAY), 90);
        assert!((raw.verbosity - 0.9).abs() < 1e-6);
        assert!((dec.verbosity - 0.7).abs() < 0.02,
            "decayed should be ~0.7 (midpoint after 1 half-life): {}", dec.verbosity);
    }

    #[test]
    fn cs_decay_uses_explicit_now_ms_not_system_time() {
        // Caller-provided now_ms is the only time source the decay
        // function consults — mirroring T.5's mock-clock pattern.
        let mut v = StyleVector::default();
        v.verbosity = 0.9;
        v.samples_count = 200;
        v.last_refined_at_ms = 1;
        // now_ms == last_refined_at_ms → zero age → no decay.
        let no_decay = v.decayed(1, 90);
        assert!((no_decay.verbosity - 0.9).abs() < 1e-6);
        // now_ms = 1 + 90 days → 1 half-life decay.
        let one_hl = v.decayed(1 + 90 * MS_PER_DAY, 90);
        assert!((one_hl.verbosity - 0.7).abs() < 0.02);
    }

    #[test]
    fn cs_pre_cs_section_defaults_via_serde_default() {
        // A snapshot deserialized from an empty struct (simulating a
        // pre-CS file with no section #12) round-trips to default.
        let empty: ConversationalStyleSnapshot = bincode::deserialize(
            &bincode::serialize(&ConversationalStyleSnapshot::default()).unwrap()
        ).unwrap();
        assert_eq!(empty, ConversationalStyleSnapshot::default());
        assert!(empty.profiles.is_empty());
    }

    // =================================================================
    // Port CS.2 — StyleHints tests
    // =================================================================

    #[test]
    fn cs2_style_hints_from_default_vector_is_neutral() {
        // samples_count=0 → hints all near 0.5 regardless of vector
        // values. Even an extreme-valued vector with zero samples
        // should produce neutral hints (the dampener pulls
        // everything to 0.5).
        let cfg = StyleConfig::default();
        let mut sv = StyleVector::default();
        // Manually push every dim to 0.9 but leave samples_count at 0.
        sv.verbosity = 0.9;
        sv.formality = 0.9;
        sv.technicality = 0.9;
        sv.empathy = 0.9;
        // samples_count is still 0.
        let hints = StyleHints::from_style_vector(&sv, &cfg);
        assert!(hints.is_neutral(),
            "samples_count=0 must yield neutral hints, got {:?}", hints);
        assert_eq!(hints.samples_count, 0);
    }

    #[test]
    fn cs2_style_hints_dampening_with_low_samples() {
        // samples_count=10, max=100 → 0.1 weight → hints pulled 90%
        // of the way to 0.5. Vector verbosity=0.9 → expected hint
        // ≈ 0.5 + (0.9 - 0.5) * 0.1 = 0.54.
        let cfg = StyleConfig::default();
        let mut sv = StyleVector::default();
        sv.verbosity = 0.9;
        sv.samples_count = 10;
        let hints = StyleHints::from_style_vector(&sv, &cfg);
        let expected = 0.5 + (0.9 - 0.5) * 0.1;
        assert!((hints.max_response_length_factor - expected).abs() < 1e-3,
            "expected ~{}, got {}", expected, hints.max_response_length_factor);
        // Other dims default 0.5 → stay 0.5.
        assert!((hints.formality - 0.5).abs() < 1e-3);
        assert!((hints.technicality - 0.5).abs() < 1e-3);
        assert!((hints.empathy - 0.5).abs() < 1e-3);
    }

    #[test]
    fn cs2_style_hints_full_weight_at_max_samples() {
        // samples_count >= max_samples → no dampening; hints reflect
        // the vector values verbatim.
        let cfg = StyleConfig::default();
        let mut sv = StyleVector::default();
        sv.verbosity = 0.9;
        sv.formality = 0.2;
        sv.technicality = 0.85;
        sv.empathy = 0.1;
        sv.samples_count = 200; // > default max=100
        let hints = StyleHints::from_style_vector(&sv, &cfg);
        assert!((hints.max_response_length_factor - 0.9).abs() < 1e-3);
        assert!((hints.formality - 0.2).abs() < 1e-3);
        assert!((hints.technicality - 0.85).abs() < 1e-3);
        assert!((hints.empathy - 0.1).abs() < 1e-3);
        assert_eq!(hints.samples_count, 200);
    }

    #[test]
    fn cs2_style_hints_decay_applied_at_read_time() {
        // Hints derived from a decayed vector should reflect the
        // decay — a 1-half-life-aged extreme vector yields hints
        // halfway between extreme and neutral.
        let cfg = StyleConfig::default();
        let mut sv = StyleVector::default();
        sv.verbosity = 0.9;
        sv.samples_count = 200; // high enough to skip dampening
        sv.last_refined_at_ms = 1_000_000;
        // Decay 1 half-life → verbosity ≈ 0.7
        let decayed = sv.decayed(1_000_000 + 90 * MS_PER_DAY, 90);
        let hints = StyleHints::from_style_vector(&decayed, &cfg);
        assert!((hints.max_response_length_factor - 0.7).abs() < 0.02,
            "decay-then-hint should yield ~0.7: {}", hints.max_response_length_factor);
    }

    #[test]
    fn cs2_style_hints_disabled_returns_neutral() {
        // The kill-switch path: even with high samples_count and an
        // extreme vector, output_adaptation_enabled=false collapses
        // every dimension to 0.5.
        let mut cfg = StyleConfig::default();
        cfg.output_adaptation_enabled = false;
        let mut sv = StyleVector::default();
        sv.verbosity = 0.9;
        sv.formality = 0.9;
        sv.technicality = 0.9;
        sv.empathy = 0.9;
        sv.samples_count = 1_000;
        let hints = StyleHints::from_style_vector(&sv, &cfg);
        assert!(hints.is_neutral(),
            "kill switch must produce neutral hints, got {:?}", hints);
    }

    #[test]
    fn cs2_style_hints_dampened_helper_is_consistent_with_from_style_vector() {
        // dampened() called explicitly should match the implicit
        // dampening inside from_style_vector for the same inputs.
        let cfg = StyleConfig::default();
        let mut sv = StyleVector::default();
        sv.verbosity = 0.8;
        sv.formality = 0.3;
        sv.technicality = 0.75;
        sv.empathy = 0.6;
        sv.samples_count = 25; // 25/100 = 0.25 weight
        let from_sv = StyleHints::from_style_vector(&sv, &cfg);
        let raw = StyleHints {
            max_response_length_factor: 0.8,
            formality: 0.3,
            technicality: 0.75,
            empathy: 0.6,
            samples_count: 25,
        };
        let dampened = raw.dampened(cfg.max_samples_for_full_weight);
        assert!((from_sv.max_response_length_factor - dampened.max_response_length_factor).abs() < 1e-6);
        assert!((from_sv.formality - dampened.formality).abs() < 1e-6);
        assert!((from_sv.technicality - dampened.technicality).abs() < 1e-6);
        assert!((from_sv.empathy - dampened.empathy).abs() < 1e-6);
    }

    #[test]
    fn cs2_max_samples_for_full_weight_setter_bounds() {
        let mut cfg = StyleConfig::default();
        // Below min.
        assert!(cfg.set_max_samples_for_full_weight(0).is_err());
        // At min.
        let prev = cfg.set_max_samples_for_full_weight(1).expect("min ok");
        assert_eq!(prev, DEFAULT_MAX_SAMPLES_FOR_FULL_WEIGHT);
        assert_eq!(cfg.max_samples_for_full_weight, 1);
        // Above max.
        assert!(cfg.set_max_samples_for_full_weight(2_000_000).is_err());
        // Healthy mid.
        cfg.set_max_samples_for_full_weight(50).unwrap();
        assert_eq!(cfg.max_samples_for_full_weight, 50);
    }

    #[test]
    fn cs2_style_hints_neutral_constructor_is_neutral() {
        let n = StyleHints::neutral();
        assert!(n.is_neutral());
        assert_eq!(n.samples_count, 0);
        assert!((n.max_response_length_factor - 0.5).abs() < f32::EPSILON);
        assert!((n.formality - 0.5).abs() < f32::EPSILON);
        assert!((n.technicality - 0.5).abs() < f32::EPSILON);
        assert!((n.empathy - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn cs2_style_config_round_trips_with_new_fields() {
        // bincode round-trip preserves the new CS.2 knobs. (Pre-CS.2
        // brains never persisted StyleConfig, but the serde(default)
        // attributes mean a new binary loading any future on-disk
        // shape gets the right defaults for missing fields.)
        let mut cfg = StyleConfig::default();
        cfg.output_adaptation_enabled = false;
        cfg.max_samples_for_full_weight = 42;
        let bytes = bincode::serialize(&cfg).expect("serialize");
        let back: StyleConfig = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(back.output_adaptation_enabled, false);
        assert_eq!(back.max_samples_for_full_weight, 42);
        // Pre-existing knobs preserved.
        assert_eq!(back.learning_enabled, cfg.learning_enabled);
        assert_eq!(back.learning_rate, cfg.learning_rate);
        assert_eq!(back.decay_half_life_days, cfg.decay_half_life_days);
    }

    #[test]
    fn cs2_dampened_with_zero_max_samples_is_unchanged() {
        // Defensive: max_samples=0 must not divide-by-zero. Behavior
        // is "no dampening" — return self unchanged.
        let raw = StyleHints {
            max_response_length_factor: 0.9,
            formality: 0.1,
            technicality: 0.9,
            empathy: 0.1,
            samples_count: 5,
        };
        let d = raw.dampened(0);
        assert_eq!(d, raw);
    }

    // =================================================================
    // Port CS.1 — WAL durability tests
    //
    // These mirror the LB.1 (`lb_session/mod.rs`) and T.3 (`pending`)
    // WAL test patterns. Each test uses a `tempfile::TempDir` for
    // an isolated brain/wal path so the suite can run in parallel.
    // =================================================================

    use std::fs;
    use std::io::Write as _;
    use tempfile::TempDir;

    fn wal_path(dir: &TempDir) -> std::path::PathBuf {
        dir.path().join("brain.lbr.style.wal")
    }

    /// Build a fresh store attached to a WAL at the given path.
    fn store_with_wal(dir: &TempDir) -> ConversationalStyleStore {
        ConversationalStyleStore::from_snapshot_with_wal(
            ConversationalStyleSnapshot::default(),
            wal_path(dir),
        )
        .expect("from_snapshot_with_wal")
    }

    #[test]
    fn cs1_wal_appends_on_observe() {
        let dir = TempDir::new().unwrap();
        let mut store = store_with_wal(&dir);
        let s = StyleSample { verbosity: 0.9, formality: 0.4, technicality: 0.6, empathy: 0.7 };
        store.observe("default", &s, 0.05, 1_700_000_000_000);
        let raw = fs::read_to_string(wal_path(&dir)).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 1, "exactly one WAL line after one observe");
        // Round-trip the line back to a StyleWalRecord and verify shape.
        let rec: StyleWalRecord = serde_json::from_str(lines[0]).unwrap();
        match rec {
            StyleWalRecord::Observe { user_key, sample, learning_rate, ts_ms } => {
                assert_eq!(user_key, "default");
                assert!((sample.verbosity - 0.9).abs() < 1e-6);
                assert!((learning_rate - 0.05).abs() < 1e-6);
                assert_eq!(ts_ms, 1_700_000_000_000);
            }
            other => panic!("expected Observe record, got {:?}", other),
        }
    }

    #[test]
    fn cs1_wal_appends_on_set() {
        let dir = TempDir::new().unwrap();
        let mut store = store_with_wal(&dir);
        let mut v = StyleVector::default();
        v.verbosity = 0.85;
        v.samples_count = 50;
        v.last_refined_at_ms = 1_700_000_000_000;
        store.set("default", v);
        let raw = fs::read_to_string(wal_path(&dir)).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 1, "exactly one WAL line after one set");
        let rec: StyleWalRecord = serde_json::from_str(lines[0]).unwrap();
        match rec {
            StyleWalRecord::Set { user_key, vector, ts_ms: _ } => {
                assert_eq!(user_key, "default");
                assert!((vector.verbosity - 0.85).abs() < 1e-6);
                assert_eq!(vector.samples_count, 50);
            }
            other => panic!("expected Set record, got {:?}", other),
        }
    }

    #[test]
    fn cs1_wal_appends_on_reset() {
        let dir = TempDir::new().unwrap();
        let mut store = store_with_wal(&dir);
        // Seed a profile, then reset it. The reset should produce a
        // second WAL line.
        let s = StyleSample::default();
        store.observe("default", &s, 0.05, 1_700_000_000_000);
        store.reset("default");
        let raw = fs::read_to_string(wal_path(&dir)).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2, "observe + reset → two WAL lines");
        let rec: StyleWalRecord = serde_json::from_str(lines[1]).unwrap();
        match rec {
            StyleWalRecord::Reset { user_key, .. } => {
                assert_eq!(user_key, "default");
            }
            other => panic!("expected Reset record, got {:?}", other),
        }
        assert!(!store.profiles.contains_key("default"));
    }

    #[test]
    fn cs1_wal_truncates_on_save() {
        let dir = TempDir::new().unwrap();
        let mut store = store_with_wal(&dir);
        let s = StyleSample { verbosity: 0.7, ..StyleSample::default() };
        store.observe("default", &s, 0.05, 1_700_000_000_000);
        store.observe("default", &s, 0.05, 1_700_000_001_000);
        let pre_size = fs::metadata(wal_path(&dir)).unwrap().len();
        assert!(pre_size > 0, "WAL must have grown after observe calls");
        // Truncate (mirrors what the brain save path calls).
        store.truncate_wal(None).expect("truncate ok");
        let post_size = fs::metadata(wal_path(&dir)).unwrap().len();
        assert_eq!(post_size, 0, "WAL must be empty after truncate");
        // Subsequent observes still work and re-grow the WAL.
        store.observe("default", &s, 0.05, 1_700_000_002_000);
        let after_size = fs::metadata(wal_path(&dir)).unwrap().len();
        assert!(after_size > 0, "WAL must accept appends after truncate");
    }

    #[test]
    fn cs1_wal_replay_after_simulated_crash() {
        let dir = TempDir::new().unwrap();
        // Phase 1: create a store, observe a few times, then DROP the
        // store WITHOUT calling truncate or save. This simulates a
        // crash between the WAL append (which is durable, fsync'd)
        // and the next brain save (which would consolidate the
        // section #12 state).
        let target = StyleSample { verbosity: 0.9, formality: 0.8, technicality: 0.7, empathy: 0.6 };
        let samples_at_crash: u64;
        let verbosity_at_crash: f32;
        {
            let mut store = store_with_wal(&dir);
            store.observe("default", &target, 0.5, 1_700_000_000_000);
            store.observe("default", &target, 0.5, 1_700_000_001_000);
            store.observe("default", &target, 0.5, 1_700_000_002_000);
            let v = store.get("default");
            samples_at_crash = v.samples_count;
            verbosity_at_crash = v.verbosity;
            // store dropped here without truncate -> WAL bytes survive.
        }
        // Phase 2: rebuild the store from an EMPTY snapshot
        // (simulating a brain whose section #12 hadn't been saved yet
        // when the crash hit) plus the WAL bytes that did fsync.
        let recovered = ConversationalStyleStore::from_snapshot_with_wal(
            ConversationalStyleSnapshot::default(),
            wal_path(&dir),
        )
        .expect("from_snapshot_with_wal");
        let v = recovered.get("default");
        assert_eq!(v.samples_count, samples_at_crash,
            "WAL replay must restore samples_count: got {} expected {}",
            v.samples_count, samples_at_crash);
        assert!((v.verbosity - verbosity_at_crash).abs() < 1e-5,
            "WAL replay must restore verbosity: got {} expected {}",
            v.verbosity, verbosity_at_crash);
        assert_eq!(v.last_refined_at_ms, 1_700_000_002_000);
    }

    #[test]
    fn cs1_wal_replay_skips_torn_line() {
        let dir = TempDir::new().unwrap();
        // Append two well-formed records, then a half-written third
        // line (no trailing newline + truncated JSON). Replay must
        // recover the first two and silently skip the torn third.
        {
            let mut store = store_with_wal(&dir);
            let s = StyleSample { verbosity: 0.8, ..StyleSample::default() };
            store.observe("alpha", &s, 0.1, 1_700_000_000_000);
            store.observe("alpha", &s, 0.1, 1_700_000_001_000);
        }
        // Now hand-write a torn line directly to the WAL.
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(wal_path(&dir))
            .unwrap();
        f.write_all(b"{\"Observe\":{\"user_key\":\"alpha\",\"sample\":{\"verbosit").unwrap();
        // Note: NO closing brace, NO newline -> torn.
        drop(f);
        // Replay against an empty section.
        let recovered = ConversationalStyleStore::from_snapshot_with_wal(
            ConversationalStyleSnapshot::default(),
            wal_path(&dir),
        )
        .expect("replay survives torn line");
        let v = recovered.get("alpha");
        assert_eq!(v.samples_count, 2,
            "two well-formed records must replay (got {})", v.samples_count);
        assert_eq!(v.last_refined_at_ms, 1_700_000_001_000);
    }

    #[test]
    fn cs1_wal_replay_idempotent_with_section_15() {
        // Test name nods to Cat.3c.1's RuntimeConfig section #15
        // (which is where the runtime style knobs live), but the
        // idempotency rule actually applies to section #12
        // (ConversationalStyle profiles). The contract: when the
        // section already absorbed a WAL record (truncate-after-save
        // failed mid-flight), re-replaying the WAL produces no
        // duplicate effect.
        let dir = TempDir::new().unwrap();
        // Phase 1: write WAL records, then snapshot the store
        // (simulating "save succeeded; truncate didn't").
        let s = StyleSample { verbosity: 0.9, ..StyleSample::default() };
        let snapshot;
        {
            let mut store = store_with_wal(&dir);
            store.observe("default", &s, 0.5, 1_700_000_000_000);
            store.observe("default", &s, 0.5, 1_700_000_001_000);
            store.observe("default", &s, 0.5, 1_700_000_002_000);
            snapshot = store.snapshot();
            // Note: NO truncate_wal() -> WAL still holds all three
            // observes that the snapshot also contains.
        }
        // Phase 2: rebuild the store from the snapshot AND replay the
        // (still-populated) WAL on top. The idempotent guard must
        // suppress every WAL record because the snapshot's
        // last_refined_at_ms (1_700_000_002_000) is >= each rec.ts_ms.
        let rebuilt = ConversationalStyleStore::from_snapshot_with_wal(
            snapshot.clone(),
            wal_path(&dir),
        )
        .expect("from_snapshot_with_wal");
        let from_snap = ConversationalStyleStore::from_snapshot(snapshot);
        let v_rebuilt = rebuilt.get("default");
        let v_snap = from_snap.get("default");
        assert_eq!(v_rebuilt.samples_count, v_snap.samples_count,
            "idempotent replay must NOT inflate samples_count: got {} expected {}",
            v_rebuilt.samples_count, v_snap.samples_count);
        assert!((v_rebuilt.verbosity - v_snap.verbosity).abs() < 1e-6,
            "idempotent replay must NOT shift verbosity: got {} expected {}",
            v_rebuilt.verbosity, v_snap.verbosity);
        assert_eq!(v_rebuilt.last_refined_at_ms, v_snap.last_refined_at_ms);
    }

    #[test]
    fn cs1_pre_cs1_brain_with_no_style_wal_loads_cleanly() {
        // Pre-CS.1 brains have no `<brain>.style.wal` alongside
        // `brain.lbr`. Loading via from_snapshot_with_wal against a
        // non-existent WAL path must succeed and yield a store
        // identical to the snapshot-only path.
        let dir = TempDir::new().unwrap();
        let snap = ConversationalStyleSnapshot::default();
        // Path that does NOT exist yet.
        let wal_p = dir.path().join("brain.lbr.style.wal");
        assert!(!wal_p.exists(), "precondition: no WAL on disk");
        let store = ConversationalStyleStore::from_snapshot_with_wal(snap.clone(), wal_p.clone())
            .expect("must load cleanly with no WAL on disk");
        // The open call creates an empty WAL file (matching
        // LbWalWriter::open semantics), so the path now exists, but
        // the in-memory state is identical to the snapshot-only path.
        assert!(store.profiles.is_empty());
        let from_snap = ConversationalStyleStore::from_snapshot(snap);
        assert_eq!(store.profiles, from_snap.profiles);
        // After open, the WAL file exists but is empty.
        assert!(wal_p.exists());
        assert_eq!(fs::metadata(&wal_p).unwrap().len(), 0);
    }
}
