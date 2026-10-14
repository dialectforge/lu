//! Foundational safety primitive — scar-aware bond access.
//!
//! Lu's analog of LLM RLHF, but enforced at the bond-access level rather
//! than as a behavioral filter on top. The architecture rests on one
//! invariant: a scar's awareness is anchored to NOW, regardless of the
//! caller's `query_time`. See `ScarLog::active_scars`.
//!
//! Design notes (from John's port spec):
//!
//! - Bond histories are preserved untouched. Time-travel queries still
//!   show the bond was active at t=10 if it was; intellectual honesty
//!   about what Lu used to believe.
//! - Scars sit in a separate log keyed by `(from, to)`. When any layer
//!   reads a bond it asks: "what scars exist on this pair *as of now*?"
//!   The answer is independent of the temporal scope of the read.
//! - Action layers (spread activation, Critic vetting) compute
//!   `effective_weight = historical * (if scarred {0.0} else {1.0})`
//!   and refuse to propagate / accept anything via a scarred bond.
//! - Observation layers (`read_bond_aware`, narration) see BOTH the
//!   historical weight AND the active-scar set so Lu can articulate
//!   "I know that connection exists; I was scarred against acting on
//!   it on date X for reason Y".
//!
//! ## Recall enforces, Critic articulates
//!
//! Two layers carry scar-awareness with overlapping but distinct jobs:
//!
//! - **Recall (network::spread_activation*)** ENFORCES. The traversal
//!   uses `effective_weight` so a scarred bond does not propagate
//!   activation. The scarred edge is functionally invisible to the
//!   activation algebra. This is the defense-in-depth layer; even if
//!   no narration ever runs, the wrong answer cannot be reached.
//! - **Critic (agents::run_critic, narration)** ARTICULATES. For
//!   reasoning passes that surface candidates anyway (Synthesis walks
//!   `historical_weight` so scarred two-hop paths surface as candidates
//!   the Critic can REJECT-WITH-REASON), the scar is mentioned in the
//!   refusal so Lu can say "I see the connection; I've chosen not to
//!   act on it because [reason]". This is the explanation layer.
//!
//! Conservative defense-in-depth: both layers run. Recall makes the
//! refusal mechanically certain; Critic makes it humanly legible. The
//! same scar drives both. Removing either layer is a regression.
//!
//! ## scarred_seeds
//!
//! `ScarLog::scarred_seeds` is a forward-protection set: any seed in
//! this set refuses NEW bonds touching it. Existing bonds are scarred
//! independently via `scar_concept`; the set guards what the brain
//! might learn next (teach() / Hebbian-creates / autonomous-research
//! approval). A seed is added via `scar_seed`; when `add_bond_with_scar_check`
//! sees a from/to in the set, it refuses the bond and emits a fresh
//! `Scar` against the would-be edge.
//!
//! ## Unscar as supersede
//!
//! `unscar_bond` does NOT delete entries from `scars`. Each affected
//! `Scar` gets `superseded_by = Some(unscar_id)` set, and an
//! `UnscarRecord` is appended. `active_scars` filters out superseded
//! ones so live behavior matches the previous semantics, but the full
//! chronology is reconstructible from a single read of `scars`.
//!
//! The scar log is persisted with the brain file. Scars survive
//! restarts; an attacker re-asking the same query post-reload still
//! gets the same refusal.

use crate::seed::Seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock NOW in epoch milliseconds. The forward-search anchor.
///
/// Lives in `scar` (not `brain`) because `bond.rs` calls it during
/// every weight read, and `bond.rs` cannot depend on `brain.rs`
/// without creating a cycle. Tests that need a deterministic anchor
/// call `ScarLog::active_scars(from, to, now_ms)` directly with a
/// fixed value instead of going through the bond signature.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// One scar on a bond.
///
/// A scar records the moral / safety decision to refuse to act on the
/// `(bond_seeds.0 -> bond_seeds.1)` connection. It does NOT delete the
/// bond — the bond's history is preserved. The scar lives in a
/// parallel log and is consulted on every bond read.
///
/// `fingerprint` is reserved for future content-similarity work
/// (suggesting related bonds when a new scar is created). Currently
/// stubbed empty — see the port-spec stretch goal.
///
/// `superseded_by` is the unscar that retired this scar, if any.
/// `None` means the scar is still in force. Set by `unscar_bond` —
/// never directly mutated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct Scar {
    /// Monotonic id assigned by the owning `ScarLog`. Used for cascade
    /// revocation and audit.
    pub scar_id: u64,
    /// The directed bond this scar is on: `(from, to)`.
    pub bond_seeds: (Seed, Seed),
    /// The scar's own identity, distinct from its id, so cascade
    /// operations can reference it cryptographically.
    pub scar_seed: Seed,
    /// Reserved for content-similarity future work. Stubbed empty.
    #[serde(default)]
    pub fingerprint: Vec<f32>,
    /// Human-readable reason; surfaces in narration when the scar
    /// drives a refusal.
    pub reason: String,
    /// Wall-clock at scar creation. The forward-search anchor in
    /// `active_scars` compares this against `now_ms`.
    pub created_ms: i64,
    /// Who created this scar (admin seed, John, etc.). `None` for
    /// system-generated scars.
    pub created_by: Option<Seed>,
    /// `unscar_id` of the unscar that retired this scar, if any.
    /// `None` means the scar is still in force. See module docs
    /// ("Unscar as supersede").
    #[serde(default)]
    pub superseded_by: Option<u64>,
}

/// Audit record for a bond unscar (admin-only, rare).
///
/// Unscars don't delete the original `Scar` entries — they mark the
/// affected scars with `superseded_by = Some(unscar_id)` and record
/// this audit trail. So "this bond was scarred at date X, then
/// unscarred at date Y for reason Z" is a fully auditable chain of
/// decisions reconstructible from a single read of `ScarLog.scars`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct UnscarRecord {
    /// Monotonic id, separate sequence from `Scar::scar_id`.
    pub unscar_id: u64,
    /// The bond the unscar applied to.
    pub bond_seeds: (Seed, Seed),
    /// Ids of the scars that were superseded (for cross-reference).
    pub removed_scar_ids: Vec<u64>,
    /// Human-readable reason for the unscar.
    pub reason: String,
    /// Wall-clock at unscar.
    pub unscarred_ms: i64,
}

/// What a scar-aware bond read returns. Both fields are surfaced so
/// observation layers can articulate the moral state of the bond.
///
/// Action layers should consume `effective_weight`; observation layers
/// (narration, time-travel comparisons) want `historical_weight` plus
/// `active_scars` to explain the gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScarAwareReading {
    /// The bond's weight at `query_time` — what the historical record
    /// says, untouched by scarring. Time-travel queries see this.
    pub historical_weight: f32,
    /// `historical_weight` zeroed if any scar is currently active,
    /// otherwise pass-through. This is the value action layers
    /// (spread_activation, Critic) use.
    pub effective_weight: f32,
    /// All scars currently active on this bond (anchored to NOW, not
    /// to `query_time`). Empty if the bond is unscarred.
    pub active_scars: Vec<Scar>,
}

/// Append-only log of every scar / unscar decision the brain has made.
///
/// Stored on `Brain` and persisted with the brain file. The bond-key
/// index is rebuilt on load — see the `#[serde(skip)]` on `by_bond`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct ScarLog {
    /// All scars ever created, in creation order. Superseded scars
    /// remain in the vector; `active_scars` filters them out.
    pub scars: Vec<Scar>,
    /// All unscar audit records.
    #[serde(default)]
    pub unscars: Vec<UnscarRecord>,
    /// Monotonic id allocator for new scars.
    #[serde(default)]
    pub next_scar_id: u64,
    /// Monotonic id allocator for unscar records.
    #[serde(default)]
    pub next_unscar_id: u64,
    /// Forward-protection set: any seed here refuses NEW bonds
    /// touching it. See module docs ("scarred_seeds"). Existing bonds
    /// are NOT auto-scarred by membership; use `scar_concept` for that.
    #[serde(default)]
    pub scarred_seeds: HashSet<Seed>,
}

impl ScarLog {
    /// Empty log.
    pub fn new() -> Self {
        ScarLog::default()
    }

    /// Record a scar on `(from, to)`. Returns the persisted `Scar`.
    ///
    /// The forward-search invariant lives entirely in `active_scars`
    /// (which compares `created_ms <= now_ms`); this method just
    /// timestamps and appends.
    pub fn scar_bond(
        &mut self,
        from: Seed,
        to: Seed,
        reason: String,
        fingerprint: Vec<f32>,
        created_by: Option<Seed>,
        created_ms: i64,
    ) -> Scar {
        let scar_id = self.next_scar_id;
        self.next_scar_id = self.next_scar_id.saturating_add(1);
        let scar = Scar {
            scar_id,
            bond_seeds: (from, to),
            scar_seed: Seed::generate(),
            fingerprint,
            reason,
            created_ms,
            created_by,
            superseded_by: None,
        };
        self.scars.push(scar.clone());
        scar
    }

    /// Mark `seed` as forward-scarred: NEW bonds touching it will be
    /// refused by `Network::add_bond_with_scar_check`. Returns true
    /// if the seed was newly inserted, false if it was already in the
    /// set. Existing bonds are not affected; cascade-scar them with
    /// `scar_concept`.
    pub fn scar_seed(&mut self, seed: Seed) -> bool {
        self.scarred_seeds.insert(seed)
    }

    /// Convenience: is `seed` forward-scarred?
    pub fn is_seed_scarred(&self, seed: Seed) -> bool {
        self.scarred_seeds.contains(&seed)
    }

    /// Reverse `scar_seed`: NEW bonds touching `seed` are no longer
    /// refused. Returns true if the seed was actually removed (false
    /// if it was not in the set). Idempotent. Does NOT unscar existing
    /// bonds; cascade those with the higher-level `Brain::unscar_concept`.
    pub fn unscar_seed(&mut self, seed: Seed) -> bool {
        self.scarred_seeds.remove(&seed)
    }

    /// All scars on `(from, to)` whose `created_ms <= now_ms` AND that
    /// have NOT been superseded by an unscar. THIS IS THE
    /// FORWARD-SEARCH PRIMITIVE. The `now_ms` argument MUST come from
    /// the caller's wall-clock NOW, never from a temporal-scope
    /// query time. The whole architecture rests on this being honored.
    pub fn active_scars(&self, from: Seed, to: Seed, now_ms: i64) -> Vec<&Scar> {
        self.scars
            .iter()
            .filter(|s| {
                s.bond_seeds == (from, to)
                    && s.created_ms <= now_ms
                    && s.superseded_by.is_none()
            })
            .collect()
    }

    /// Convenience overload: forward-search anchored to wall-clock NOW.
    /// Production callers use this; tests on the primitive use the
    /// explicit-`now_ms` form to make the anchor deterministic.
    pub fn active_scars_now(&self, from: Seed, to: Seed) -> Vec<&Scar> {
        self.active_scars(from, to, now_ms())
    }

    /// Convenience: is the bond currently scarred (excluding superseded)?
    pub fn is_scarred(&self, from: Seed, to: Seed, now_ms: i64) -> bool {
        self.scars.iter().any(|s| {
            s.bond_seeds == (from, to)
                && s.created_ms <= now_ms
                && s.superseded_by.is_none()
        })
    }

    /// Cascade scar — every bond touching `seed` (incoming or
    /// outgoing) gets a fresh scar. Caller supplies the bond list
    /// because the log doesn't know graph topology. Returns one
    /// `Scar` per affected bond.
    pub fn scar_concept(
        &mut self,
        bonds_touching: &[(Seed, Seed)],
        reason: String,
        created_by: Option<Seed>,
        created_ms: i64,
    ) -> Vec<Scar> {
        bonds_touching
            .iter()
            .map(|(from, to)| {
                self.scar_bond(*from, *to, reason.clone(), Vec::new(), created_by, created_ms)
            })
            .collect()
    }

    /// Admin-only: supersede every active scar on `(from, to)` whose
    /// `created_ms <= now_ms`. Affected scars get `superseded_by`
    /// set to the new unscar id; an `UnscarRecord` is appended to
    /// `unscars` for audit. Returns the count superseded.
    pub fn unscar_bond(
        &mut self,
        from: Seed,
        to: Seed,
        reason: String,
        unscarred_ms: i64,
    ) -> u32 {
        let unscar_id = self.next_unscar_id;
        let mut removed_ids: Vec<u64> = Vec::new();
        for s in self.scars.iter_mut() {
            if s.bond_seeds == (from, to)
                && s.created_ms <= unscarred_ms
                && s.superseded_by.is_none()
            {
                s.superseded_by = Some(unscar_id);
                removed_ids.push(s.scar_id);
            }
        }
        let count = removed_ids.len() as u32;
        if count > 0 {
            self.next_unscar_id = self.next_unscar_id.saturating_add(1);
            self.unscars.push(UnscarRecord {
                unscar_id,
                bond_seeds: (from, to),
                removed_scar_ids: removed_ids,
                reason,
                unscarred_ms,
            });
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ab() -> (Seed, Seed) {
        (Seed::from_label("alpha"), Seed::from_label("beta"))
    }

    #[test]
    fn test_scar_bond_appends_and_assigns_id() {
        let (a, b) = ab();
        let mut log = ScarLog::new();
        let s1 = log.scar_bond(a, b, "test".into(), Vec::new(), None, 100);
        let s2 = log.scar_bond(a, b, "test2".into(), Vec::new(), None, 200);
        assert_eq!(s1.scar_id, 0);
        assert_eq!(s2.scar_id, 1);
        assert_eq!(log.scars.len(), 2);
        assert!(s1.superseded_by.is_none());
    }

    #[test]
    fn test_active_scars_anchored_to_now_argument() {
        let (a, b) = ab();
        let mut log = ScarLog::new();
        log.scar_bond(a, b, "danger".into(), Vec::new(), None, 20);
        assert!(log.active_scars(a, b, 10).is_empty());
        assert_eq!(log.active_scars(a, b, 25).len(), 1);
    }

    #[test]
    fn test_unscar_supersedes_not_deletes() {
        let (a, b) = ab();
        let mut log = ScarLog::new();
        log.scar_bond(a, b, "first".into(), Vec::new(), None, 10);
        log.scar_bond(a, b, "second".into(), Vec::new(), None, 20);
        let removed = log.unscar_bond(a, b, "admin override".into(), 30);
        assert_eq!(removed, 2);
        // Originals are NOT deleted; chronology preserved.
        assert_eq!(log.scars.len(), 2);
        // Both are now superseded by the unscar (id 0 — the first unscar).
        assert_eq!(log.scars[0].superseded_by, Some(0));
        assert_eq!(log.scars[1].superseded_by, Some(0));
        // Active view is now empty.
        assert!(log.active_scars(a, b, 40).is_empty());
        assert!(!log.is_scarred(a, b, 40));
        // Audit trail exists.
        assert_eq!(log.unscars.len(), 1);
        assert_eq!(log.unscars[0].removed_scar_ids, vec![0, 1]);
    }

    #[test]
    fn test_unscar_no_scars_no_audit() {
        let (a, b) = ab();
        let mut log = ScarLog::new();
        let removed = log.unscar_bond(a, b, "noop".into(), 30);
        assert_eq!(removed, 0);
        assert!(log.unscars.is_empty());
    }

    #[test]
    fn test_unscar_only_supersedes_active() {
        // Re-scarring after an unscar should produce a new scar that is
        // visible. The previously-superseded scar stays superseded.
        let (a, b) = ab();
        let mut log = ScarLog::new();
        log.scar_bond(a, b, "old".into(), Vec::new(), None, 10);
        log.unscar_bond(a, b, "override".into(), 20);
        log.scar_bond(a, b, "new reason".into(), Vec::new(), None, 30);
        assert_eq!(log.scars.len(), 2);
        assert_eq!(log.scars[0].superseded_by, Some(0));
        assert!(log.scars[1].superseded_by.is_none());
        assert_eq!(log.active_scars(a, b, 40).len(), 1);
        assert_eq!(log.active_scars(a, b, 40)[0].reason, "new reason");
    }

    #[test]
    fn test_scar_concept_creates_one_per_bond() {
        let a = Seed::from_label("a");
        let b = Seed::from_label("b");
        let c = Seed::from_label("c");
        let mut log = ScarLog::new();
        let scars = log.scar_concept(
            &[(a, b), (b, c), (c, a)],
            "concept revoked".into(),
            None,
            42,
        );
        assert_eq!(scars.len(), 3);
        assert!(log.is_scarred(a, b, 50));
        assert!(log.is_scarred(b, c, 50));
        assert!(log.is_scarred(c, a, 50));
    }

    #[test]
    fn test_scar_seed_inserts_and_idempotent() {
        let s = Seed::from_label("poison");
        let mut log = ScarLog::new();
        assert!(log.scar_seed(s));
        assert!(!log.scar_seed(s));
        assert!(log.is_seed_scarred(s));
        assert!(!log.is_seed_scarred(Seed::from_label("safe")));
    }
}
