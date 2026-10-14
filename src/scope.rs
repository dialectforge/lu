//! Temporal scoping for graph queries.
//!
//! Every spread-activation query in Lu 4D operates within a temporal scope
//! that determines which bond weights are visible and how they're evaluated.

use serde::{Deserialize, Serialize};

/// Which of the three multi-agent roles is currently driving a scope.
///
/// Defined here (rather than alongside the rest of the multi-agent
/// machinery in `agents.rs`) so `TemporalScope::Agent` can name a typed
/// role without producing a circular dependency between `bond.rs` and
/// `agents.rs`. `KeyframeSource::Agent` still carries it as a `u8` for
/// the same reason — see the `as_u8` / `from_u8` helpers below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Surfaces top candidate seeds for a query.
    Recall,
    /// Combines recalled candidates into a→c via b inference triples.
    Synthesis,
    /// Filters synthesized beliefs by confidence threshold.
    Critic,
}

impl AgentRole {
    pub fn all() -> &'static [AgentRole] {
        &[AgentRole::Recall, AgentRole::Synthesis, AgentRole::Critic]
    }

    pub fn as_u8(self) -> u8 {
        match self {
            AgentRole::Recall => 0,
            AgentRole::Synthesis => 1,
            AgentRole::Critic => 2,
        }
    }

    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(AgentRole::Recall),
            1 => Some(AgentRole::Synthesis),
            2 => Some(AgentRole::Critic),
            _ => None,
        }
    }
}

/// Defines the temporal window for a graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalScope {
    /// Use the latest weight for every bond (default behavior).
    Current,

    /// Restrict to weights within a specific session.
    Session {
        /// Unique session identifier.
        id: u64,
        /// Session start timestamp (epoch millis).
        start: i64,
    },

    /// Evaluate all bonds as they were at a specific point in time.
    Historical(i64),

    /// Evaluate bonds using weights within a time range.
    Range {
        /// Start of the range (epoch millis, inclusive).
        start: i64,
        /// End of the range (epoch millis, inclusive).
        end: i64,
    },

    /// Evaluate bonds inside a Variant Generation arena. `id` selects
    /// which variant's keyframes are visible; everything else falls
    /// through to `parent` so variant work can ride on top of an active
    /// Session, Current, or Historical view. Recursive via `Box`.
    ///
    /// Variant-tagged keyframes are NEVER visible outside this scope —
    /// they live in the bond's history but are filtered out by
    /// `TemporalBond::weight_at` so Baseline / Current / Session reads
    /// stay clean.
    Variant {
        /// Identifies this variant within the arena (Brain-issued u64).
        id: u64,
        /// The scope this variant forked from. Falls back here when no
        /// variant-tagged keyframe exists for a bond.
        parent: Box<TemporalScope>,
    },

    /// Evaluate bonds inside a single multi-agent role's view.
    /// `multi_think_id` identifies the orchestrating multi_think call;
    /// `role` tags which of the three agents (Recall/Synthesis/Critic)
    /// is driving the read. Like Variant, this falls through to
    /// `parent` when no agent-tagged keyframe matches.
    ///
    /// Agent-tagged keyframes are filtered out of every non-Agent
    /// scope (`TemporalBond::weight_at` drops them). Multi-think is
    /// OBSERVATION, not commitment — vetted beliefs do NOT auto-promote
    /// to baseline. Agent-source keyframes age out via the half-life
    /// pruner in `consolidation`.
    ///
    /// Nesting is NOT permitted: `multi_think` rejects calls whose
    /// surrounding scope is already an `Agent` scope. Future work:
    /// hierarchical / tree-of-thought agents that permit nesting.
    Agent {
        role: AgentRole,
        multi_think_id: u64,
        parent: Box<TemporalScope>,
    },
}

impl TemporalScope {
    /// Return the effective timestamp for weight evaluation.
    ///
    /// - `Current` → `i64::MAX` (latest available)
    /// - `Session` → session start time
    /// - `Historical(t)` → t
    /// - `Range { end, .. }` → end of the range
    pub fn timestamp(&self) -> i64 {
        match self {
            TemporalScope::Current => i64::MAX,
            TemporalScope::Session { start, .. } => *start,
            TemporalScope::Historical(t) => *t,
            TemporalScope::Range { end, .. } => *end,
            // Variant inherits the parent's effective timestamp — variant
            // scoping is about source-tag filtering, not about time.
            TemporalScope::Variant { parent, .. } => parent.timestamp(),
            // Agent scopes inherit time from their parent for the same
            // reason — agent scoping filters by source tag.
            TemporalScope::Agent { parent, .. } => parent.timestamp(),
        }
    }

    /// Check if a given timestamp falls within this scope.
    pub fn contains(&self, t: i64) -> bool {
        match self {
            TemporalScope::Current => true,
            TemporalScope::Session { start, .. } => t >= *start,
            TemporalScope::Historical(at) => t <= *at,
            TemporalScope::Range { start, end } => t >= *start && t <= *end,
            TemporalScope::Variant { parent, .. } => parent.contains(t),
            TemporalScope::Agent { parent, .. } => parent.contains(t),
        }
    }

    /// Check if this scope filters by session ID.
    pub fn session_id(&self) -> Option<u64> {
        match self {
            TemporalScope::Session { id, .. } => Some(*id),
            // A variant can be forked from inside an active session; surface
            // that session through the variant boundary.
            TemporalScope::Variant { parent, .. } => parent.session_id(),
            TemporalScope::Agent { parent, .. } => parent.session_id(),
            _ => None,
        }
    }

    /// Variant id, if this scope (or its parent chain) is a Variant scope.
    pub fn variant_id(&self) -> Option<u64> {
        match self {
            TemporalScope::Variant { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// `(role, multi_think_id)` if this scope is an Agent scope.
    /// `Agent::parent` is NOT recursively unwrapped — agent scopes
    /// don't nest, so the immediate variant carries the answer.
    pub fn agent_role(&self) -> Option<(AgentRole, u64)> {
        match self {
            TemporalScope::Agent { role, multi_think_id, .. } => Some((*role, *multi_think_id)),
            _ => None,
        }
    }
}

impl Default for TemporalScope {
    fn default() -> Self {
        TemporalScope::Current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_contains_everything() {
        let scope = TemporalScope::Current;
        assert!(scope.contains(0));
        assert!(scope.contains(i64::MAX));
    }

    #[test]
    fn test_range_containment() {
        let scope = TemporalScope::Range {
            start: 100,
            end: 200,
        };
        assert!(!scope.contains(99));
        assert!(scope.contains(100));
        assert!(scope.contains(150));
        assert!(scope.contains(200));
        assert!(!scope.contains(201));
    }

    #[test]
    fn test_historical_timestamp() {
        let scope = TemporalScope::Historical(12345);
        assert_eq!(scope.timestamp(), 12345);
    }

    #[test]
    fn test_agent_role_extraction() {
        let scope = TemporalScope::Agent {
            role: AgentRole::Synthesis,
            multi_think_id: 42,
            parent: Box::new(TemporalScope::Current),
        };
        let (r, id) = scope.agent_role().unwrap();
        assert_eq!(r, AgentRole::Synthesis);
        assert_eq!(id, 42);
        // Non-agent scopes return None.
        assert!(TemporalScope::Current.agent_role().is_none());
    }

    #[test]
    fn test_agent_role_u8_roundtrip() {
        for &role in AgentRole::all() {
            let b = role.as_u8();
            assert_eq!(AgentRole::from_u8(b), Some(role));
        }
        assert_eq!(AgentRole::from_u8(99), None);
    }
}
