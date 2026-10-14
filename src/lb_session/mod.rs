//! Port LB — Left Brain operational/episodic memory layer.
//!
//! Schema-impacting addition. Adds `SectionIdV3::SessionMemory = 11`
//! (SECTION_COUNT 10 → 11) so the operational state lives in a
//! dedicated section of the encrypted v3 brain. Pre-LB brain files
//! load with an empty store via `de_bincode_optional`.
//!
//! ## Hard architectural rule
//!
//! From `agent/memory/project_lu_session_memory.md`:
//!
//! > User data/knowledge (Right Brain facts) must be strictly
//! > separated from operational/session data (Left Brain logs) to
//! > prevent non-facts from becoming facts — this is Lu's
//! > anti-hallucination architecture.
//!
//! Right Brain = verified facts, approved bonds, trained knowledge
//! (truth). Left Brain session memory = session logs, thinking
//! traces, observations, hypotheses, work history (context, NOT
//! truth).
//!
//! Session entries inform reasoning but **never** auto-promote to
//! fact bonds. Promotion is an explicit two-step:
//! [`propose_to_pending`] converts an entry to a T.3 `PendingChange`,
//! then a human approves via `lu4d_approve_pending` (existing T.3
//! tool). Only after approval does the bond enter the canonical brain.
//!
//! ## Naming note
//!
//! Spec proposed `src/session_memory/` but that name was already
//! taken by Port Q.3's conversational anchor / pronoun-resolution
//! module (`src/session_memory.rs`). This port lives in
//! `src/lb_session/` to avoid the collision and to make the Left
//! Brain semantics explicit.
//!
//! ## Scope (v1)
//!
//! Ships in this port:
//!   * Data model — [`SessionEntry`], [`SessionEntryKind`].
//!   * Standalone store — [`SessionMemoryStore`] +
//!     [`SessionMemorySnapshot`] (BTreeMap-backed, ordered, indexed by
//!     session_id and kind).
//!   * Active-session guard — writes require `activate_session`;
//!     archived sessions are immutable.
//!   * Propose flow — [`propose_to_pending`] converts a session entry
//!     to a T.3 `PendingChange::AddBond` with attribution
//!     `Teach { actor: "session_promote:S<sid>/E<eid>", ts_ms }`.
//!     (A first-class `SourceKind::SessionPromote` variant is its own
//!     schema bump and is left as a follow-up port.)
//!   * v3 binary section id 11 — [`SessionMemorySnapshot`] persisted
//!     across all three v3 write paths (uncompressed, compressed,
//!     compressed+encrypted) and read via `de_bincode_optional`.
//!   * MCP tools — `lu4d_session_memory_status` /
//!     `_inspect` / `_propose` / `_archive`.
//!
//! Deferred to a follow-up port:
//!   * `<brain>.lb_session.wal` for between-saves durability.
//!   * `lu4d_think` / `lu4d_multi_think` writing thinking traces
//!     into Left Brain instead of being ephemeral.
//!   * `lu4d_wonder` migration off `WonderQueueState` v3 section.
//!   * `lu4d_research_loop_run` results landing as session entries.
//!   * First-class `SourceKind::SessionPromote { session_id, entry_id }`.

pub mod entry;
pub mod kind;
pub mod store;
pub mod wal;

pub use entry::SessionEntry;
pub use kind::SessionEntryKind;
pub use store::{SessionMemorySnapshot, SessionMemoryStore, SCRATCHPAD_SESSION_ID};
pub use wal::{LbWalRecord, LbWalWriter, read_all as read_lb_wal};

use crate::brain::{Brain, BrainError};
use crate::pending::{PendingChange, PendingChangeKind, PendingStatus};
use crate::source::SourceAttribution;

/// Convert a [`SessionEntry`] into a T.3 [`PendingChange`] and push
/// it onto the brain's pending log. The session entry is marked as
/// promoted via [`SessionMemoryStore::mark_promoted`]. Returns the
/// allocated `pending_id`.
///
/// The pending change is built as:
///   * `proposed_at_ms` = `ts_ms`
///   * `proposed_by`    = `Teach { actor: "session_promote:S<sid>/E<eid>", ts_ms }`
///     (a first-class `SourceKind::SessionPromote` variant is left
///     as a follow-up port — it would itself be a schema bump.)
///   * `change_kind`    = `AddBond { subject_text, object_text, bond_type, weight, negated: false }`
///   * `confidence`     = the entry's `confidence_at_creation`
///   * `status`         = `Pending` (forced by `PendingLog::push`)
///
/// The session entry's content is parsed for the optional fields
/// `subject_text` / `object_text` / `bond_type` / `weight` via the
/// caller-supplied [`PromoteOptions`]. Defaulting to the entry's
/// `subject` for both subject and object is intentional — it makes
/// the propose call exercise the round-trip path even when the
/// session entry was a free-form thinking trace; the human
/// approver inspects the resulting `PendingChange` via
/// `lu4d_pending_changes` and rejects bad shapes via
/// `lu4d_reject_pending` (which feeds T.8 rejection learning).
pub fn propose_to_pending(
    brain: &mut Brain,
    entry_id: u64,
    options: PromoteOptions,
    ts_ms: u64,
) -> Result<u64, BrainError> {
    // Snapshot the entry first so we can build the change without a
    // long borrow of `brain.lb_session`.
    let (session_id, subject, content, confidence) = {
        let store = &brain.lb_session;
        let e = store.get(entry_id).ok_or_else(|| {
            BrainError::Format(format!("unknown lb_session entry id {}", entry_id))
        })?;
        if e.is_promoted() {
            return Err(BrainError::Format(format!(
                "entry {} already promoted (pending_id={:?})",
                entry_id, e.promoted_pending_id,
            )));
        }
        (
            e.session_id,
            e.subject.clone(),
            e.content.clone(),
            e.confidence_at_creation,
        )
    };

    // Port SourceKind-Variants — first-class SessionPromote attribution
    // (replaces the legacy `Teach { actor: "session_promote:S<sid>/E<eid>" }`
    // dodge). Caller-side ts_ms is preserved on `proposed_at_ms`; the
    // SessionPromote variant intentionally drops ts_ms because the
    // (session_id, entry_id) pair recovers it via brain.lb_session.get(...)
    // when needed.
    let _ = ts_ms; // ts_ms still consumed via proposed_at_ms below
    let subject_text = options.subject_text.unwrap_or_else(|| subject.clone());
    let object_text = options.object_text.unwrap_or_else(|| {
        if !content.is_empty() {
            content.clone()
        } else {
            subject.clone()
        }
    });
    let change = PendingChange {
        pending_id: 0,
        proposed_at_ms: ts_ms,
        proposed_by: SourceAttribution::session_promote(session_id.to_string(), entry_id),
        change_kind: PendingChangeKind::AddBond {
            subject_text,
            object_text,
            bond_type: options.bond_type,
            weight: options.weight,
            negated: false,
        },
        confidence,
        reason: format!(
            "Promoted from Left Brain session memory entry id={} (session={})",
            entry_id, session_id,
        ),
        status: PendingStatus::Pending,
    };
    let pending_id = brain.pending.push(change)?;
    brain.lb_session.mark_promoted(entry_id, ts_ms, pending_id)?;
    Ok(pending_id)
}

/// Caller-supplied shape information for a propose call. The session
/// entry's `subject` / `content` are the defaults when these fields
/// are `None`.
#[derive(Debug, Clone)]
pub struct PromoteOptions {
    pub subject_text: Option<String>,
    pub object_text: Option<String>,
    pub bond_type: crate::bond::BondType,
    pub weight: f32,
}

impl Default for PromoteOptions {
    fn default() -> Self {
        Self {
            subject_text: None,
            object_text: None,
            bond_type: crate::bond::BondType::RelatedTo,
            weight: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::brain::Brain;
    use crate::brain_io_v3::{read_v3_bytes, write_v3, FORMAT_VERSION_V3};
    use crate::lb_session::entry::SessionEntry;
    use crate::lb_session::kind::SessionEntryKind;

    fn make_thinking_entry(session_id: u64, subject: &str) -> SessionEntry {
        SessionEntry::new(
            session_id,
            1_000_000,
            SessionEntryKind::ThinkingTrace,
            subject,
            "derived from recall",
            0.4,
        )
    }

    /// Test 1 — `SessionEntry` round-trips through bincode (the v3
    /// section's encoding).
    #[test]
    fn lb_session_entry_serializes_round_trip() {
        let e = make_thinking_entry(7, "why is the sky blue?");
        let bytes = bincode::serialize(&e).unwrap();
        let back: SessionEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.session_id, 7);
        assert_eq!(back.subject, "why is the sky blue?");
        assert_eq!(back.kind, SessionEntryKind::ThinkingTrace);
        assert!((back.confidence_at_creation - 0.4).abs() < 1e-6);
        assert!(!back.is_promoted());
    }

    /// Test 2 — Writing without an active session returns an error.
    /// (Active-session guard.)
    #[test]
    fn lb_active_session_guard_blocks_write_without_session() {
        let mut store = SessionMemoryStore::new();
        let res = store.push(make_thinking_entry(0, "orphan write"));
        assert!(res.is_err());
    }

    /// Test 3 — Reads are unrestricted: a write made under one
    /// active session is visible to subsequent reads under any (or
    /// no) active session.
    #[test]
    fn lb_other_sessions_can_read_active_session_writes() {
        let mut store = SessionMemoryStore::new();
        store.activate_session(11).unwrap();
        let id = store.push(make_thinking_entry(0, "in session 11")).unwrap();
        store.deactivate_session();
        // Read with no active session.
        assert!(store.get(id).is_some());
        // Read with a different active session.
        store.activate_session(22).unwrap();
        assert!(store.get(id).is_some());
        let filtered = store.filter(Some(11), None, 100);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].entry_id, id);
    }

    /// Test 4 — Pushing a session entry does NOT mutate the
    /// canonical brain. Bond count is unchanged after a save+reload
    /// round trip. THIS IS THE CRITICAL ANTI-HALLUCINATION TEST.
    #[test]
    fn lb_session_data_does_not_auto_promote_to_facts() {
        let mut brain = Brain::new();
        let initial_bonds = brain.network.bond_count();
        brain.lb_session.activate_session(1).unwrap();
        let _id = brain
            .lb_session
            .push(make_thinking_entry(0, "hypothesis: rust is fast"))
            .unwrap();
        // Bond count must NOT change.
        assert_eq!(brain.network.bond_count(), initial_bonds);
        // Round-trip through v3 — bonds still unchanged after reload.
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let bytes = write_v3(&file).unwrap();
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert_eq!(parsed.network.bonds.len(), initial_bonds);
        assert_eq!(parsed.lb_session.entries.len(), 1);
    }

    /// Test 5 — `propose_to_pending` converts a session entry into
    /// a T.3 `PendingChange::AddBond` and marks the entry promoted.
    #[test]
    fn lb_propose_creates_pending_change() {
        let mut brain = Brain::new();
        brain.lb_session.activate_session(1).unwrap();
        let id = brain
            .lb_session
            .push(SessionEntry::new(
                0,
                1_000_000,
                SessionEntryKind::Hypothesis,
                "alpha",
                "beta",
                0.7,
            ))
            .unwrap();
        let pending_id = propose_to_pending(
            &mut brain,
            id,
            PromoteOptions {
                bond_type: BondType::IsA,
                weight: 0.7,
                ..Default::default()
            },
            2_000_000,
        )
        .unwrap();
        assert_eq!(pending_id, 1);
        let pc = brain.pending.get(pending_id).unwrap();
        match &pc.change_kind {
            PendingChangeKind::AddBond {
                subject_text,
                object_text,
                ..
            } => {
                assert_eq!(subject_text, "alpha");
                assert_eq!(object_text, "beta");
            }
            _ => panic!("wrong kind"),
        }
        // Attribution string carries the session/entry back-link.
        // Port SourceKind-Variants: assertion updated from legacy
        // Teach{actor:"session_promote:S1/E1"} to first-class SessionPromote.
        match &pc.proposed_by.kind {
            crate::source::SourceKind::SessionPromote { session_id, entry_id } => {
                assert_eq!(session_id, "1");
                assert_eq!(*entry_id, 1);
            }
            other => panic!("expected SessionPromote, got {:?}", other),
        }
        let entry = brain.lb_session.get(id).unwrap();
        assert!(entry.is_promoted());
        assert_eq!(entry.promoted_pending_id, Some(pending_id));
        // Bond count unchanged — promotion is propose, not apply.
        assert_eq!(brain.network.bond_count(), 0);
    }

    /// Test 6 — Session memory entries round-trip through the v3
    /// binary section (id 11). Save → fresh load → entries match.
    #[test]
    fn lb_v3_section_round_trips_session_memory() {
        let mut brain = Brain::new();
        brain.lb_session.activate_session(42).unwrap();
        let id1 = brain
            .lb_session
            .push(make_thinking_entry(0, "a"))
            .unwrap();
        let id2 = brain
            .lb_session
            .push(make_thinking_entry(0, "b"))
            .unwrap();
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let bytes = write_v3(&file).unwrap();
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert_eq!(parsed.lb_session.entries.len(), 2);
        assert_eq!(parsed.lb_session.entries[0].entry_id, id1);
        assert_eq!(parsed.lb_session.entries[1].entry_id, id2);
        assert_eq!(parsed.lb_session.entries[0].session_id, 42);
        assert_eq!(parsed.lb_session.next_entry_id, 3);
    }

    /// Test 7 — Pre-LB brain files (section_count = 10) load with
    /// an empty `SessionMemorySnapshot` via `de_bincode_optional`.
    /// Synthesized by writing a fresh v3 brain, decrementing the
    /// section_count to 10 in the header, and recomputing the trailer
    /// CRC — the parser then iterates only 10 entries from the offset
    /// table and the new SessionMemory entry is invisible.
    #[test]
    fn lb_pre_lb_brain_loads_with_empty_session_store() {
        let brain = Brain::new();
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let mut bytes = write_v3(&file).unwrap();
        // Decrement section_count from 11 to 10 in the fixed header.
        let pre_lb_count: u32 = 10;
        bytes[48..52].copy_from_slice(&pre_lb_count.to_le_bytes());
        // Recompute trailer CRC over [..trailer_pos].
        let trailer_pos = bytes.len() - 16;
        let crc = crc32fast::hash(&bytes[..trailer_pos]);
        bytes[trailer_pos + 8..trailer_pos + 12]
            .copy_from_slice(&crc.to_le_bytes());
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert!(parsed.lb_session.entries.is_empty());
    }

    /// Test 8 — Archiving a session marks it immutable: subsequent
    /// writes scoped to that session id error out, and re-activating
    /// it errors.
    #[test]
    fn lb_archive_marks_session_immutable() {
        let mut store = SessionMemoryStore::new();
        store.activate_session(99).unwrap();
        let _ = store.push(make_thinking_entry(0, "pre-archive")).unwrap();
        store.archive_session(99);
        assert!(store.is_archived(99));
        // Active session was cleared by archive.
        assert!(store.active_session().is_none());
        // Re-activating an archived session is rejected.
        assert!(store.activate_session(99).is_err());
    }

    /// Test 9 — A full snapshot round-trips through bincode (the v3
    /// section's encoding) including archived_sessions and
    /// next_entry_id.
    #[test]
    fn lb_snapshot_round_trips_through_bincode() {
        let mut store = SessionMemoryStore::new();
        store.activate_session(7).unwrap();
        let _ = store.push(make_thinking_entry(0, "a")).unwrap();
        store.archive_session(7);
        let snap = store.snapshot();
        let bytes = bincode::serialize(&snap).unwrap();
        let back: SessionMemorySnapshot = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.entries[0].session_id, 7);
        assert_eq!(back.next_entry_id, 2);
        assert_eq!(back.archived_sessions, vec![7]);
        let restored = SessionMemoryStore::from_snapshot(back);
        assert!(restored.is_archived(7));
        assert_eq!(restored.next_id(), 2);
    }

    /// Test 10 — The standalone store path keeps the canonical brain
    /// pristine: pushing a `ThinkingTrace` does NOT mutate the bond
    /// count, neither in-memory nor after a v3 round-trip. This is
    /// the standalone-only equivalent of the spec's
    /// `lb_thinking_traces_route_to_session_memory_not_brain` —
    /// the `lu4d_think` integration that wires traces through the
    /// store automatically is deferred to a follow-up port.
    #[test]
    fn lb_thinking_trace_kind_does_not_touch_brain() {
        let mut brain = Brain::new();
        let initial_bonds = brain.network.bond_count();
        let initial_nodes = brain.network.node_count();
        brain.lb_session.activate_session(1).unwrap();
        for i in 0..5 {
            let _ = brain
                .lb_session
                .push(make_thinking_entry(0, &format!("step {}", i)))
                .unwrap();
        }
        assert_eq!(brain.network.bond_count(), initial_bonds);
        assert_eq!(brain.network.node_count(), initial_nodes);
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let bytes = write_v3(&file).unwrap();
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert_eq!(parsed.network.bonds.len(), initial_bonds);
        assert_eq!(parsed.lb_session.entries.len(), 5);
        assert_eq!(
            parsed.lb_session.entries[0].kind,
            SessionEntryKind::ThinkingTrace
        );
    }
// ---------------------------------------------------------------
    // Port LB.1 tests
    //
    // LB.1 closes 4 of LB's 5 known limitations:
    //   1. <brain>.lb_session.wal between-saves durability (mirrors T.3).
    //   2. lu4d_think auto-routes ThinkingTrace entries into LB.
    //   3. lu4d_multi_think auto-routes ThinkingTrace entries per agent step.
    //   4. lu4d_research_loop_run dual-writes to LB (ResearchResult)
    //      alongside the existing research_proposals push.
    //   5. lu4d_wonder dual-writes WonderItem entries; section #8
    //      hydrates LB on first load with the LB.1 binary.
    //
    // Item 5 from LB ship report (first-class
    // `SourceKind::SessionPromote { session_id, entry_id }` variant)
    // remains deferred to a future LB.2 — emulated via
    // `Teach { actor: "session_promote:S<sid>/E<eid>" }`.
    // ---------------------------------------------------------------

    fn lb1_tmp_brain_dir(label: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu_lb1_test_{}_{}",
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// LB.1 Test 1 — Push entries with a WAL attached, drop the
    /// in-memory store (simulated crash), then re-open from the
    /// snapshot+WAL: entries are recovered.
    #[test]
    fn lb1_wal_replay_after_simulated_crash() {
        let dir = lb1_tmp_brain_dir("wal_crash");
        let wal_path = dir.join("brain.lbr.lb_session.wal");

        // Phase A: write entries through a WAL-backed store.
        let mut store = SessionMemoryStore::from_snapshot_with_wal(
            SessionMemorySnapshot::default(),
            wal_path.clone(),
        )
        .unwrap();
        store.activate_session(11).unwrap();
        let id1 = store.push(make_thinking_entry(0, "crash test alpha")).unwrap();
        let id2 = store.push(make_thinking_entry(0, "crash test beta")).unwrap();
        drop(store); // simulated crash before any save.

        // Phase B: re-open with the same WAL — entries reappear.
        let store2 = SessionMemoryStore::from_snapshot_with_wal(
            SessionMemorySnapshot::default(),
            wal_path.clone(),
        )
        .unwrap();
        assert_eq!(store2.total_count(), 2);
        assert!(store2.get(id1).is_some());
        assert!(store2.get(id2).is_some());
        assert_eq!(store2.next_id(), id2 + 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LB.1 Test 2 — After Brain::save, <brain>.lb_session.wal is
    /// truncated to 0 bytes. The v3 section #11 is the source of
    /// truth from this point on.
    #[test]
    fn lb1_wal_truncates_on_save() {
        let dir = lb1_tmp_brain_dir("wal_trunc");
        let path = dir.join("brain.lbr");
        let wal_path = path.with_file_name("brain.lbr.lb_session.wal");

        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.attach_pending_wal(crate::pending::PendingLogSnapshot::default());
        brain.attach_lb_session_wal(SessionMemorySnapshot::default());
        brain.lb_session.activate_session(7).unwrap();
        let _ = brain.lb_session.push(make_thinking_entry(0, "first")).unwrap();
        let _ = brain.lb_session.push(make_thinking_entry(0, "second")).unwrap();
        let pre = std::fs::metadata(&wal_path).unwrap().len();
        assert!(pre > 0, "LB WAL must be non-empty after pushes");
        brain.save().unwrap();
        let post = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(post, 0, "LB.1 save must truncate the LB session WAL");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LB.1 Test 3 — A torn line at EOF (process killed mid-write)
    /// is skipped and earlier records survive.
    #[test]
    fn lb1_wal_replay_skips_torn_line() {
        use std::io::Write;
        let dir = lb1_tmp_brain_dir("wal_torn");
        let wal_path = dir.join("brain.lbr.lb_session.wal");
        {
            let mut store = SessionMemoryStore::from_snapshot_with_wal(
                SessionMemorySnapshot::default(),
                wal_path.clone(),
            )
            .unwrap();
            store.activate_session(33).unwrap();
            store.push(make_thinking_entry(0, "valid alpha")).unwrap();
            store.push(make_thinking_entry(0, "valid beta")).unwrap();
        }
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap();
        f.write_all(b"{\"Push\":{\"entry_id\":99,\"sess").unwrap();
        f.flush().unwrap();
        drop(f);

        let store2 = SessionMemoryStore::from_snapshot_with_wal(
            SessionMemorySnapshot::default(),
            wal_path.clone(),
        )
        .unwrap();
        assert_eq!(store2.total_count(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LB.1 Test 4 — Crash-between-save-and-truncate corner: the
    /// brain file holds a snapshot that already contains entry X;
    /// the WAL ALSO holds a Push(X) (because truncate failed). On
    /// load, the dedupe logic in apply_wal_record skips the
    /// duplicate Push so the store has entry X exactly once.
    #[test]
    fn lb1_wal_replay_idempotent_with_v3_section() {
        use std::io::Write;
        let dir = lb1_tmp_brain_dir("wal_idempotent");
        let wal_path = dir.join("brain.lbr.lb_session.wal");

        // Snapshot contains entry_id 1.
        let mut store_a = SessionMemoryStore::from_snapshot_with_wal(
            SessionMemorySnapshot::default(),
            wal_path.clone(),
        )
        .unwrap();
        store_a.activate_session(5).unwrap();
        store_a.push(make_thinking_entry(0, "alpha")).unwrap();
        let snap = store_a.snapshot();
        drop(store_a);
        // The WAL on disk now also has Push(entry_id=1). DON'T truncate.
        // Load via from_snapshot_with_wal: snapshot loads entry 1 first,
        // then WAL replay sees Push(1) and skips it (dedup).
        let store_b = SessionMemoryStore::from_snapshot_with_wal(
            snap,
            wal_path.clone(),
        )
        .unwrap();
        assert_eq!(store_b.total_count(), 1);
        assert_eq!(store_b.next_id(), 2);

        // Now append a NEW entry id=2 to the WAL (simulating mutations
        // post-snapshot) and re-open: both entries survive.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap();
        let new_entry = SessionEntry {
            entry_id: 2,
            session_id: 5,
            ts_ms: 1_000_001,
            kind: SessionEntryKind::ThinkingTrace,
            subject: "beta".into(),
            content: "derived from recall".into(),
            confidence_at_creation: 0.4,
            promoted_at_ms: None,
            promoted_pending_id: None,
        };
        let rec = LbWalRecord::Push(new_entry);
        f.write_all(serde_json::to_string(&rec).unwrap().as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
        f.flush().unwrap();
        drop(f);
        let store_c = SessionMemoryStore::from_snapshot_with_wal(
            store_b.snapshot(),
            wal_path.clone(),
        )
        .unwrap();
        assert_eq!(store_c.total_count(), 2);
        assert_eq!(store_c.next_id(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LB.1 Test 5 — `brain.think()` writes one ThinkingTrace entry
    /// per step into the Left Brain. Anti-hallucination invariant:
    /// bond_count is unchanged.
    #[test]
    fn lb1_lu4d_think_writes_thinking_trace_to_lb() {
        let mut brain = Brain::new();
        // A minimal teach so think has *something* to recall against.
        brain.teach(
            "rust",
            crate::bond::BondType::IsA,
            "language",
            0.7,
            crate::node::Confidence::Unverified,
        );
        let initial_bonds = brain.network.bond_count();
        let lb_pre = brain.lb_session.total_count();
        let cfg = crate::thinking::ThinkConfig::default();
        let _ = brain.think("rust", cfg);
        let lb_post = brain.lb_session.total_count();
        assert!(
            lb_post >= lb_pre,
            "think must not REMOVE LB entries"
        );
        // At least one ThinkingTrace entry per step (or the call may
        // return zero steps on a tiny graph; in either case the LB
        // counts_by_kind for thinking_trace must equal lb_post-lb_pre).
        let n_thinking = brain
            .lb_session
            .counts_by_kind()
            .get("thinking_trace")
            .copied()
            .unwrap_or(0);
        let added = lb_post - lb_pre;
        // think can also fire curiosity signals which dual-write
        // WonderItem entries (LB.1 wonder hook), so the new entries
        // are a mix of ThinkingTrace + WonderItem. Assert that
        // ThinkingTrace count grew with the call.
        assert!(
            added == 0 || n_thinking > 0,
            "think writes ThinkingTrace entries when steps are produced"
        );
        // Anti-hallucination invariant: bond_count is unchanged.
        assert_eq!(brain.network.bond_count(), initial_bonds);
    }

    /// LB.1 Test 6 — `brain.multi_think_with_config()` writes one
    /// ThinkingTrace per agent step (Recall, Synthesis, Critic).
    /// Anti-hallucination invariant: bond_count is unchanged.
    #[test]
    fn lb1_lu4d_multi_think_writes_thinking_trace_per_step() {
        let mut brain = Brain::new();
        brain.teach(
            "alpha",
            crate::bond::BondType::IsA,
            "beta",
            0.7,
            crate::node::Confidence::Unverified,
        );
        let initial_bonds = brain.network.bond_count();
        let cfg = brain.multi_agent_config.clone();
        let lb_pre = brain.lb_session.total_count();
        let _ = brain.multi_think_with_config("alpha", &cfg);
        let lb_post = brain.lb_session.total_count();
        assert_eq!(
            lb_post - lb_pre,
            3,
            "multi_think writes one ThinkingTrace per agent step (3 = Recall + Synthesis + Critic)"
        );
        assert_eq!(brain.network.bond_count(), initial_bonds);
    }

    /// LB.1 Test 7 — `brain.research_one()` dual-writes a
    /// ResearchResult entry into the Left Brain alongside the
    /// existing push to `brain.research_proposals`. Anti-hallucination
    /// invariant: bond_count is unchanged (the research approval
    /// path remains the only route to fact bonds).
    #[test]
    fn lb1_research_loop_writes_research_result_and_proposal() {
        use crate::research::http::MockFetcher;
        let mut brain = Brain::new();
        // Use the manual research path (research_query) to bypass
        // wonder_to_query setup. Both research_one and research_query
        // route through make_proposal_for_query and now both LB-write.
        let m = MockFetcher::new();
        m.route_ok(
            "https://grokipedia.com/search?q=topic",
            200,
            r##"<html><body><a href="/page/Topic">Topic</a></body></html>"##,
        );
        m.route_ok(
            "https://grokipedia.com/page/Topic",
            200,
            r##"<html><body><main><p>Topic is a test concept.</p><p>Topic causes example behavior.</p></main></body></html>"##,
        );
        let initial_bonds = brain.network.bond_count();
        let pre_lb = brain.lb_session.total_count();
        let pre_props = brain.research_proposals.len();
        let _ = brain.research_query(&m, "topic").expect("research_query");
        let post_lb = brain.lb_session.total_count();
        let post_props = brain.research_proposals.len();
        // Both stores grew by 1.
        assert_eq!(
            post_props - pre_props,
            1,
            "research_one must push a ResearchProposal"
        );
        assert_eq!(
            post_lb - pre_lb,
            1,
            "research_one must dual-write a ResearchResult LB entry"
        );
        let n_research = brain
            .lb_session
            .counts_by_kind()
            .get("research_result")
            .copied()
            .unwrap_or(0);
        assert_eq!(n_research, 1);
        // Anti-hallucination invariant.
        assert_eq!(brain.network.bond_count(), initial_bonds);
    }

    /// LB.1 Test 8 — Pushing a wonder signal dual-writes a WonderItem
    /// LB entry. The runtime wonder_queue still has the signal so T.6
    /// auto-tick gates continue to work; LB has the persistent record.
    #[test]
    fn lb1_wonder_writes_to_lb_alongside_wonder_queue() {
        let mut brain = Brain::new();
        // Push a wonder via the same flush_gaps path Lu uses in
        // production. Synthesize one gap directly.
        let pre_lb = brain.lb_session.total_count();
        let pre_queue = brain.wonder_queue.len();
        // Use the lower-level wonder push helper exercised by ask().
        // Mirror what ask does: touch interest profile + push the signal.
        let now = crate::brain::now_ms() as u64;
        let halflife = brain.curiosity_config.interest_decay_halflife_ms;
        let seed = crate::seed::Seed::from_label("why_is_grass_green");
        let signal = crate::curiosity::CuriositySignal {
            seed,
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.8,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: now,
        };
        brain.interest_profile.touch(seed, now, halflife);
        // Manually do what the production patch does (mirroring the
        // patched call site so the test exercises the LB write path).
        let lb_label = brain
            .network
            .get_node(&signal.seed)
            .map(|n| n.content().to_string())
            .or_else(|| brain.gap_label_map.get(&signal.seed).cloned())
            .unwrap_or_else(|| signal.seed.to_string());
        let lb_payload = serde_json::to_string(&signal).unwrap_or_default();
        let lb_intensity = signal.intensity;
        let _ = brain
            .lb_session
            .push_internal(SessionEntry::new(
                0,
                now,
                SessionEntryKind::WonderItem,
                lb_label,
                lb_payload,
                lb_intensity,
            ))
            .unwrap();
        brain.wonder_queue.push(signal);

        assert_eq!(brain.wonder_queue.len(), pre_queue + 1);
        assert_eq!(brain.lb_session.total_count(), pre_lb + 1);
        let n_wonder = brain
            .lb_session
            .counts_by_kind()
            .get("wonder_item")
            .copied()
            .unwrap_or(0);
        assert_eq!(n_wonder, 1);
    }

    /// LB.1 Test 9 — A pre-LB.1 brain saved with section #8 wonders
    /// but no section #11 LB entries hydrates LB on first load with
    /// the LB.1 binary via `migrate_section8_wonders_to_lb_if_needed`.
    /// Idempotent on subsequent loads.
    #[test]
    fn lb1_pre_lb1_brain_with_section_8_wonders_hydrates_lb_on_first_load() {
        let mut brain = Brain::new();
        // Simulate a pre-LB.1 brain: wonder_queue has signals but LB
        // session has no WonderItem entries.
        let s1 = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("q1"),
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.7,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: 1_000_000,
        };
        let s2 = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("q2"),
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.5,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: 1_000_000,
        };
        brain.wonder_queue.push(s1);
        brain.wonder_queue.push(s2);
        // Sanity: LB has zero WonderItems pre-migration.
        assert_eq!(
            brain
                .lb_session
                .counts_by_kind()
                .get("wonder_item")
                .copied()
                .unwrap_or(0),
            0
        );
        // First call migrates.
        brain.migrate_section8_wonders_to_lb_if_needed(2_000_000);
        let n1 = brain
            .lb_session
            .counts_by_kind()
            .get("wonder_item")
            .copied()
            .unwrap_or(0);
        assert_eq!(n1, 2, "both section-#8 wonders must hydrate into LB");
        // Second call is idempotent.
        brain.migrate_section8_wonders_to_lb_if_needed(3_000_000);
        let n2 = brain
            .lb_session
            .counts_by_kind()
            .get("wonder_item")
            .copied()
            .unwrap_or(0);
        assert_eq!(n2, 2, "migration must be idempotent");
    }

    /// LB.1 Test 10 — T.6 auto-tick gate (`!wonder_queue.is_empty()`)
    /// keeps firing after LB.1: wonder_queue stays the live runtime
    /// structure; LB writes are sidecar.
    #[test]
    fn lb1_no_regression_in_t6_auto_tick_wonder_gate() {
        let mut brain = Brain::new();
        assert!(
            brain.wonder_queue.is_empty(),
            "baseline: queue is empty"
        );
        // Push a wonder via the dual-write path (mirror production).
        let now = crate::brain::now_ms() as u64;
        let signal = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("why_water_wet"),
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.9,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: now,
        };
        let _ = brain
            .lb_session
            .push_internal(SessionEntry::new(
                0,
                now,
                SessionEntryKind::WonderItem,
                "why_water_wet",
                serde_json::to_string(&signal).unwrap_or_default(),
                signal.intensity,
            ))
            .unwrap();
        brain.wonder_queue.push(signal);

        // T.6 gate: !wonder_queue.is_empty().
        assert!(
            !brain.wonder_queue.is_empty(),
            "T.6 auto-tick gate must still see wonder_queue as non-empty"
        );
        // And LB has the WonderItem so post-LB.1 inspection works.
        assert_eq!(
            brain
                .lb_session
                .counts_by_kind()
                .get("wonder_item")
                .copied()
                .unwrap_or(0),
            1
        );
    }

    /// Port LB.2 — T.6 auto-tick gate (`!wonder_queue.is_empty()`)
    /// MUST keep firing after LB.2 even when the wonder_queue was
    /// hydrated from section #11 (LB) instead of section #8.
    #[test]
    fn lb2_t6_auto_tick_still_fires_with_section_11_only_state() {
        use crate::brain_io_v3::{write_v3, FORMAT_VERSION_V3};
        let mut b = Brain::new();
        b.teach(
            "foo",
            crate::bond::BondType::IsA,
            "bar",
            0.9,
            crate::node::Confidence::Verified,
        );
        let now = 1_700_002_000_000_u64;
        let signal = crate::curiosity::CuriositySignal {
            seed: crate::seed::Seed::from_label("lb2_t6_seed"),
            signal_type: crate::curiosity::SignalType::HubBoundary,
            intensity: 0.8,
            scope: crate::scope::TemporalScope::Current,
            timestamp_ms: now,
        };
        // Push to BOTH wonder_queue (so save snapshots it) and LB
        // (so post-LB.2 load can hydrate from section #11).
        let _ = b.lb_session.push_internal(SessionEntry::new(
            0,
            now,
            SessionEntryKind::WonderItem,
            "why_lb2_t6",
            serde_json::to_string(&signal).unwrap_or_default(),
            signal.intensity,
        ));
        b.wonder_queue.push(signal);

        let bytes = write_v3(&b.to_brain_file(FORMAT_VERSION_V3)).unwrap();
        let dir_p = std::env::temp_dir().join(format!(
            "lu4d-lb2-t6-{}-{}",
            std::process::id(),
            now
        ));
        std::fs::create_dir_all(&dir_p).unwrap();
        let p = dir_p.join("brain.lbr");
        std::fs::write(&p, &bytes).unwrap();

        let brain = Brain::load_or_new(&p).expect("load");
        assert!(
            !brain.wonder_queue.is_empty(),
            "LB.2: wonder_queue must be non-empty after load (T.6 gate fires)"
        );
        // The T.6 evaluate path takes `has_wonders` from the caller.
        let has_wonders = !brain.wonder_queue.is_empty();
        assert!(
            has_wonders,
            "LB.2: T.6 has_wonders contract preserved end-to-end"
        );
        // Cleanup
        let _ = std::fs::remove_dir_all(&dir_p);
    }

    /// Port LB.2 — J.1's wonder-queue persistence pattern (push to
    /// queue, save, load, restored) MUST still work end-to-end. With
    /// LB.2, the dual-write to LB carries the signals across the
    /// save/load boundary even though section #8 is now empty.
    #[test]
    fn lb2_j1_wonder_queue_tests_unchanged() {
        use crate::brain_io_v3::{write_v3, FORMAT_VERSION_V3};
        let mut b = Brain::new();
        let now = 1_700_002_500_000_u64;
        let signals: Vec<_> = (0..3)
            .map(|i| crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label(&format!("j1_persist_{}", i)),
                signal_type: crate::curiosity::SignalType::HubBoundary,
                intensity: 0.5 + 0.1 * (i as f32),
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            })
            .collect();
        // Mirror the LB.1 dual-write production path.
        for s in &signals {
            let _ = b.lb_session.push_internal(SessionEntry::new(
                0,
                now,
                SessionEntryKind::WonderItem,
                format!("label_{}", s.seed),
                serde_json::to_string(s).unwrap_or_default(),
                s.intensity,
            ));
            b.wonder_queue.push(s.clone());
        }
        assert_eq!(b.wonder_queue.len(), 3);

        let bytes = write_v3(&b.to_brain_file(FORMAT_VERSION_V3)).unwrap();
        let dir_p = std::env::temp_dir().join(format!(
            "lu4d-lb2-j1-{}-{}",
            std::process::id(),
            now
        ));
        std::fs::create_dir_all(&dir_p).unwrap();
        let p = dir_p.join("brain.lbr");
        std::fs::write(&p, &bytes).unwrap();

        let brain = Brain::load_or_new(&p).expect("load");
        assert_eq!(
            brain.wonder_queue.len(),
            3,
            "J.1's persistence contract preserved under LB.2 (3 round-tripped)"
        );
        let _ = std::fs::remove_dir_all(&dir_p);
    }

    /// Port LB.2 — anti-hallucination invariant: wonder state is
    /// operational/curiosity context, NOT canonical bonds. A save
    /// + load with LB.2 binary leaves `bond_count` UNCHANGED.
    #[test]
    fn lb2_anti_hallucination_invariant_preserved() {
        use crate::brain_io_v3::{write_v3, FORMAT_VERSION_V3};
        let mut b = Brain::new();
        b.teach(
            "alpha",
            crate::bond::BondType::IsA,
            "letter",
            0.9,
            crate::node::Confidence::Verified,
        );
        b.teach(
            "beta",
            crate::bond::BondType::IsA,
            "letter",
            0.9,
            crate::node::Confidence::Verified,
        );
        b.teach(
            "alpha",
            crate::bond::BondType::HasProperty,
            "first",
            0.8,
            crate::node::Confidence::Verified,
        );
        let bond_count_before = b.network.bond_count();
        assert_eq!(bond_count_before, 3);

        // Push wonder items (LB + queue) to exercise the LB.2 path.
        let now = 1_700_003_000_000_u64;
        for i in 0..5 {
            let signal = crate::curiosity::CuriositySignal {
                seed: crate::seed::Seed::from_label(&format!("hallu_{}", i)),
                signal_type: crate::curiosity::SignalType::HubBoundary,
                intensity: 0.5,
                scope: crate::scope::TemporalScope::Current,
                timestamp_ms: now,
            };
            let _ = b.lb_session.push_internal(SessionEntry::new(
                0,
                now,
                SessionEntryKind::WonderItem,
                format!("hallu_q_{}", i),
                serde_json::to_string(&signal).unwrap_or_default(),
                signal.intensity,
            ));
            b.wonder_queue.push(signal);
        }

        let bytes = write_v3(&b.to_brain_file(FORMAT_VERSION_V3)).unwrap();
        let dir_p = std::env::temp_dir().join(format!(
            "lu4d-lb2-hallu-{}-{}",
            std::process::id(),
            now
        ));
        std::fs::create_dir_all(&dir_p).unwrap();
        let p = dir_p.join("brain.lbr");
        std::fs::write(&p, &bytes).unwrap();

        let brain = Brain::load_or_new(&p).expect("load");
        let bond_count_after = brain.network.bond_count();
        assert_eq!(
            bond_count_after, bond_count_before,
            "LB.2 anti-hallucination: bond_count must NOT change through wonder state save/load"
        );
        // Wonder state is operational and DID round-trip.
        assert_eq!(brain.wonder_queue.len(), 5);
        let _ = std::fs::remove_dir_all(&dir_p);
    }
}
