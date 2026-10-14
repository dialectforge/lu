//! Port T.3 — Pending change data model.
//!
//! A `PendingChange` is one proposed mutation to the brain. It must
//! pass through the approve / reject / modify gate before being
//! committed. T.2's idle research path queues `PendingChangeKind::AddBond`
//! entries; future Cat. 3 ports will populate the other variants.

use crate::bond::BondType;
use crate::seed::Seed;
use crate::source::SourceAttribution;
use serde::{Deserialize, Serialize};

/// One proposed mutation to the brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChange {
    /// Monotonic id allocated by the [`crate::pending::PendingLog`].
    pub pending_id: u64,
    /// Wall-clock ms when the change was proposed (T.1 authenticated
    /// time when available, OS time as fallback).
    pub proposed_at_ms: u64,
    /// Source attribution for the proposal — `Research` for idle
    /// loop output, `Teach` for human-driven proposals, etc.
    pub proposed_by: SourceAttribution,
    /// The shape of the proposed mutation.
    pub change_kind: PendingChangeKind,
    /// Confidence in the proposal, 0.0-1.0. Mapped to a
    /// [`crate::node::Confidence`] tier at apply time.
    pub confidence: f32,
    /// Auto-generated explanation of why the change was proposed.
    pub reason: String,
    /// Pending / Approved / Rejected / Applied lifecycle state.
    pub status: PendingStatus,
}

/// The four kinds of mutations T.3 understands. Start small; expand
/// as need arises.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PendingChangeKind {
    /// Add a new bond. Existing nodes are reused; missing nodes are
    /// created via the standard [`crate::brain::Brain::teach_with_source`]
    /// path so the scar gate fires the same way as a manual teach.
    AddBond {
        subject_text: String,
        object_text: String,
        bond_type: BondType,
        weight: f32,
        negated: bool,
    },
    /// Add a standalone node (no bond yet). Confidence comes from the
    /// parent [`PendingChange`].
    AddNode {
        content: String,
        kind_tag: Option<String>,
    },
    /// Modify the weight of an existing bond. The bond must exist at
    /// apply time; if it doesn't, apply returns an error and the
    /// status stays Approved (caller can re-reject manually).
    ModifyBondWeight {
        from: Seed,
        to: Seed,
        bond_type: BondType,
        new_weight: f32,
    },
    /// Set the negation overlay flag on an existing bond.
    SetNegation {
        from: Seed,
        to: Seed,
        bond_type: BondType,
        negated: bool,
    },
}

/// Lifecycle state of a `PendingChange`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PendingStatus {
    Pending,
    Approved {
        approved_at_ms: u64,
        approved_by: String,
    },
    Rejected {
        rejected_at_ms: u64,
        rejected_by: String,
        reason: Option<String>,
    },
    /// Approved and committed to the brain. Terminal state.
    Applied {
        applied_at_ms: u64,
    },
}

impl PendingStatus {
    pub fn tag(&self) -> &'static str {
        match self {
            PendingStatus::Pending => "Pending",
            PendingStatus::Approved { .. } => "Approved",
            PendingStatus::Rejected { .. } => "Rejected",
            PendingStatus::Applied { .. } => "Applied",
        }
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, PendingStatus::Pending)
    }

    pub fn is_approved(&self) -> bool {
        matches!(self, PendingStatus::Approved { .. })
    }

    pub fn is_applied(&self) -> bool {
        matches!(self, PendingStatus::Applied { .. })
    }
}

impl PendingChangeKind {
    /// Short human-readable summary used by the
    /// `lu4d_pending_changes` MCP tool's structured payload.
    pub fn summary(&self) -> String {
        match self {
            PendingChangeKind::AddBond {
                subject_text,
                object_text,
                bond_type,
                negated,
                weight,
                ..
            } => format!(
                "AddBond({} -[{:?}{}]-> {}, w={:.2})",
                subject_text,
                bond_type,
                if *negated { " NOT" } else { "" },
                object_text,
                weight,
            ),
            PendingChangeKind::AddNode { content, kind_tag } => format!(
                "AddNode({}{})",
                content,
                kind_tag
                    .as_ref()
                    .map(|k| format!(", kind={}", k))
                    .unwrap_or_default(),
            ),
            PendingChangeKind::ModifyBondWeight { new_weight, .. } => {
                format!("ModifyBondWeight(w={:.2})", new_weight)
            }
            PendingChangeKind::SetNegation { negated, .. } => {
                format!("SetNegation(={})", negated)
            }
        }
    }

    /// Tag string for filtering in the list/status MCP tools.
    pub fn tag(&self) -> &'static str {
        match self {
            PendingChangeKind::AddBond { .. } => "AddBond",
            PendingChangeKind::AddNode { .. } => "AddNode",
            PendingChangeKind::ModifyBondWeight { .. } => "ModifyBondWeight",
            PendingChangeKind::SetNegation { .. } => "SetNegation",
        }
    }
}
