//! Port T.3 — Apply approved pending changes to the Brain.
//!
//! Each `PendingChangeKind` variant routes to the matching
//! existing brain mutation primitive:
//!   * AddBond           → `Brain::teach_with_source` (scar gate fires)
//!   * AddNode           → `network::add_node` (with deterministic seed)
//!   * ModifyBondWeight  → keyframe append at the new weight
//!   * SetNegation       → `Brain::set_bond_negated`
//!
//! On success, the log entry's status moves Approved → Applied and
//! a WAL record is appended; on failure, the status stays Approved
//! and the error bubbles up so the caller can surface it.

use crate::bond::KeyframeSource;
use crate::brain::{now_ms, Brain, BrainError};
use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};
use crate::pending::change::{PendingChangeKind, PendingStatus};

/// Apply a single approved [`crate::pending::PendingChange`] to the
/// brain. Marks the log entry Applied on success.
pub fn apply_change(brain: &mut Brain, pending_id: u64) -> Result<(), BrainError> {
    let change = brain
        .pending
        .get(pending_id)
        .ok_or_else(|| BrainError::Format(format!("unknown pending id {}", pending_id)))?
        .clone();

    if !matches!(change.status, PendingStatus::Approved { .. }) {
        return Err(BrainError::Format(format!(
            "pending {} not approved (status={})",
            pending_id,
            change.status.tag()
        )));
    }

    // Map confidence f32 -> tier. Mirrors the same banding used by
    // the research-approval path so AddBonds derived from a
    // research proposal land at the same Confidence variant they
    // would have via `approve_research_proposal`.
    let conf = if change.confidence >= 0.9 {
        Confidence::Verified
    } else if change.confidence >= 0.6 {
        Confidence::Inferred
    } else {
        Confidence::Unverified
    };

    match &change.change_kind {
        PendingChangeKind::AddBond {
            subject_text,
            object_text,
            bond_type,
            weight,
            negated,
        } => {
            let r = brain.teach_with_source(
                subject_text,
                *bond_type,
                object_text,
                *weight,
                conf,
                change.proposed_by.clone(),
            );
            if *negated {
                brain.set_bond_negated(r.subject_seed, r.object_seed, *bond_type, true);
            }
        }
        PendingChangeKind::AddNode { content, kind_tag } => {
            let knowledge_content =
                KnowledgeContent::from_kind_or_concept(kind_tag.as_deref(), content.clone());
            let node = KnowledgeNode::new(content.clone(), knowledge_content, conf, now_ms());
            brain.network.add_node(node);
        }
        PendingChangeKind::ModifyBondWeight {
            from,
            to,
            bond_type: _,
            new_weight,
        } => {
            // The bond must exist; if it doesn't, surface that as an
            // error so the caller can decide what to do (typically
            // reject manually or modify the pending entry to AddBond
            // instead).
            if brain.network.get_bond(from, to).is_none() {
                return Err(BrainError::Format(format!(
                    "ModifyBondWeight: bond not found {}..{}..",
                    &from.to_string()[..8],
                    &to.to_string()[..8],
                )));
            }
            // Append a fresh keyframe at the new weight. KeyframeSource::Manual
            // matches the user-driven nature of the approval.
            if let Some(b) = brain.network.bond_mut_raw(from, to) {
                b.add_keyframe(now_ms(), *new_weight, KeyframeSource::Manual);
            }
        }
        PendingChangeKind::SetNegation {
            from,
            to,
            bond_type,
            negated,
        } => {
            brain.set_bond_negated(*from, *to, *bond_type, *negated);
        }
    }

    let now = now_ms() as u64;
    brain.pending.mark_applied(pending_id, now)?;
    Ok(())
}
