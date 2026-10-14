//! Port T.3 — Pending changes log + approval workflow.
//!
//! Closes the auto-commit gap left by Port T.2 (idle work loop):
//! research-derived bonds, future Cat. 3 self-rewriting passes,
//! and any other autonomous mutation route their proposals
//! through a [`PendingLog`] and wait for explicit human
//! approve / reject / modify before being committed to the brain.
//!
//! Public surface:
//!   * [`PendingChange`], [`PendingChangeKind`], [`PendingStatus`]
//!     — the data model.
//!   * [`PendingLog`], [`PendingLogSnapshot`] — the in-memory log
//!     and its serialization shape.
//!   * [`apply::apply_change`] — drives the Approved → Applied
//!     transition by invoking the underlying brain mutation
//!     primitive (`teach_with_source`, `set_bond_negated`, …).
//!
//! See [`crate::idle::loop_::idle_tick`] (Port T.2) for the
//! call-site that pushes research proposals into the log instead
//! of auto-committing them.

pub mod apply;
pub mod change;
pub mod log;
pub mod wal;

pub use apply::apply_change;
pub use change::{PendingChange, PendingChangeKind, PendingStatus};
pub use log::{PendingLog, PendingLogSnapshot};
pub use wal::{read_all as read_wal, WalRecord, WalWriter};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::brain::Brain;
    
    use crate::source::{SourceAttribution, SourceKind};
    use std::path::PathBuf;

    fn tmp_wal_path(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu_t3_test_{}_{}.pending.wal",
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Best-effort cleanup of stale files from prior runs.
        let _ = std::fs::remove_file(&p);
        p
    }

    fn make_research_change(subject: &str, object: &str, conf: f32) -> PendingChange {
        PendingChange {
            pending_id: 0,
            proposed_at_ms: 1_000_000,
            proposed_by: SourceAttribution::research(
                "deadbeefcafebabe",
                Some("https://ex.example/page".into()),
                1_000_000,
            ),
            change_kind: PendingChangeKind::AddBond {
                subject_text: subject.into(),
                object_text: object.into(),
                bond_type: BondType::IsA,
                weight: 0.7,
                negated: false,
            },
            confidence: conf,
            reason: format!(
                "Research-derived from gap signal `deadbeef` (intensity 0.85)"
            ),
            status: PendingStatus::Pending,
        }
    }

    /// Test 1 — Empty log starts empty.
    #[test]
    fn t3_pending_log_starts_empty() {
        let log = PendingLog::new_in_memory();
        assert_eq!(log.pending_count(), 0);
        assert_eq!(log.applied_count(), 0);
        assert_eq!(log.next_id(), 1);
        assert!(log.last_commit_at_ms().is_none());
    }

    /// Test 2 — Push persists to memory + WAL; reopening replays.
    #[test]
    fn t3_add_pending_persists_to_wal_and_memory() {
        let p = tmp_wal_path("persist");
        let mut log = PendingLog::open(p.clone()).unwrap();
        let id = log.push(make_research_change("rust", "language", 0.7)).unwrap();
        assert_eq!(id, 1);
        assert_eq!(log.pending_count(), 1);
        drop(log);

        // Reopening from disk reads the WAL.
        let log2 = PendingLog::open(p.clone()).unwrap();
        assert_eq!(log2.pending_count(), 1);
        let entry = log2.get(id).unwrap();
        match &entry.change_kind {
            PendingChangeKind::AddBond { subject_text, .. } => {
                assert_eq!(subject_text, "rust");
            }
            _ => panic!("wrong kind"),
        }

        let _ = std::fs::remove_file(&p);
    }

    /// Test 3 — WAL replay after a simulated crash. We write three
    /// pending entries, then simulate a crash by NOT calling save
    /// (process death between operations). Reopening reconstructs
    /// state from WAL alone.
    #[test]
    fn t3_wal_replay_after_simulated_crash() {
        let p = tmp_wal_path("crash");
        let mut log = PendingLog::open(p.clone()).unwrap();
        let _ = log.push(make_research_change("a", "b", 0.6)).unwrap();
        let _ = log.push(make_research_change("c", "d", 0.6)).unwrap();
        let _ = log.push(make_research_change("e", "f", 0.6)).unwrap();
        drop(log); // simulated crash mid-session

        let log2 = PendingLog::open(p.clone()).unwrap();
        assert_eq!(log2.pending_count(), 3);
        assert_eq!(log2.next_id(), 4);

        let _ = std::fs::remove_file(&p);
    }

    /// Test 4 — A pre-T.3 brain (no PendingLog snapshot) loads with
    /// an empty log via `serde(default)` on the snapshot field.
    #[test]
    fn t3_pre_t3_brain_loads_empty_pending() {
        // The pre-T.3 brain serialized a BrainFile WITHOUT a
        // pending_log field. Synthesize that exact shape (empty
        // JSON object) and deserialize as PendingLogSnapshot.
        let json = "{}";
        let snap: PendingLogSnapshot = serde_json::from_str(json).unwrap();
        assert!(snap.entries.is_empty());
        assert_eq!(snap.next_pending_id, 1); // default_next_id()

        // Round-trip into a log.
        let p = tmp_wal_path("pre_t3");
        let log = PendingLog::open_with_snapshot(p.clone(), snap).unwrap();
        assert_eq!(log.pending_count(), 0);
        let _ = std::fs::remove_file(&p);
    }

    /// Test 5 — Round-trip a populated log through snapshot ->
    /// serialize -> deserialize -> open_with_snapshot. State is
    /// preserved.
    #[test]
    fn t3_round_trip_pending_through_brain_save_load() {
        let p = tmp_wal_path("round_trip");
        let mut log = PendingLog::open(p.clone()).unwrap();
        let id1 = log.push(make_research_change("x", "y", 0.7)).unwrap();
        let id2 = log.push(make_research_change("a", "b", 0.5)).unwrap();
        log.approve(id1, 2_000_000, "john").unwrap();
        log.mark_applied(id1, 2_000_500).unwrap();
        let snap = log.snapshot();
        // After consolidation in-brain, the WAL would be truncated;
        // simulate that here so we test the snapshot path alone.
        log.truncate_wal(None).unwrap();
        drop(log);

        let blob = serde_json::to_string(&snap).unwrap();
        let snap2: PendingLogSnapshot = serde_json::from_str(&blob).unwrap();
        assert_eq!(snap2.entries.len(), 2);
        assert_eq!(snap2.next_pending_id, 3);

        let log2 = PendingLog::open_with_snapshot(p.clone(), snap2).unwrap();
        assert_eq!(log2.total_count(), 2);
        assert_eq!(log2.applied_count(), 1);
        assert_eq!(log2.pending_count(), 1);
        assert!(log2.get(id1).unwrap().status.is_applied());
        assert!(log2.get(id2).unwrap().status.is_pending());
        let _ = std::fs::remove_file(&p);
    }

    /// Test 6 — Modify a pending entry's confidence/weight before
    /// approval, and observe the new values flow through to apply.
    #[test]
    fn t3_modify_pending_before_approve() {
        let p = tmp_wal_path("modify");
        let mut log = PendingLog::open(p.clone()).unwrap();
        let id = log.push(make_research_change("k", "v", 0.5)).unwrap();
        log.modify(id, Some(0.95), Some(0.9), Some("edited".into()))
            .unwrap();
        let entry = log.get(id).unwrap();
        assert!((entry.confidence - 0.95).abs() < 1e-6);
        assert_eq!(entry.reason, "edited");
        match &entry.change_kind {
            PendingChangeKind::AddBond { weight, .. } => {
                assert!((weight - 0.9).abs() < 1e-6);
            }
            _ => panic!("wrong kind"),
        }
        let _ = std::fs::remove_file(&p);
    }

    /// Test 7 — Approving a pending entry then applying it grows
    /// the brain's bond count by exactly 1.
    #[test]
    fn t3_approve_pending_applies_to_brain() {
        let mut brain = Brain::new();
        let initial = brain.network.bond_count();
        let id = brain
            .pending
            .push(make_research_change("alpha", "beta", 0.7))
            .unwrap();
        brain.pending.approve(id, 1_000_000, "john").unwrap();
        apply_change(&mut brain, id).unwrap();
        assert_eq!(brain.network.bond_count(), initial + 1);
        assert!(brain.pending.get(id).unwrap().status.is_applied());
    }

    /// Test 8 — Rejecting a pending entry leaves the brain alone.
    #[test]
    fn t3_reject_pending_does_not_apply() {
        let mut brain = Brain::new();
        let initial = brain.network.bond_count();
        let id = brain
            .pending
            .push(make_research_change("alpha", "beta", 0.7))
            .unwrap();
        brain
            .pending
            .reject(id, 1_000_000, "john", Some("not useful".into()))
            .unwrap();
        // apply_change refuses to act on a Rejected entry.
        let res = apply_change(&mut brain, id);
        assert!(res.is_err());
        assert_eq!(brain.network.bond_count(), initial);
    }

    /// Test 9 — Atomic batch approve: 5 entries, all approved + applied,
    /// brain bond count grows by 5. Mirrors the MCP-layer batch tool's
    /// contract ("either all match or none commit").
    #[test]
    fn t3_atomic_batch_approve() {
        let mut brain = Brain::new();
        let initial = brain.network.bond_count();
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = brain
                .pending
                .push(make_research_change(
                    &format!("s{}", i),
                    &format!("o{}", i),
                    0.7,
                ))
                .unwrap();
            ids.push(id);
        }
        // Phase 1: approve all (no apply yet — caller controls atomicity).
        for id in &ids {
            brain.pending.approve(*id, 1_000_000, "john").unwrap();
        }
        // Phase 2: apply each. If any apply fails the caller would
        // surface the count of successes via the MCP tool's response.
        for id in &ids {
            apply_change(&mut brain, *id).unwrap();
        }
        assert_eq!(brain.network.bond_count(), initial + 5);
        assert_eq!(brain.pending.applied_count(), 5);
    }

    /// Test 10 — SetNegation pending applies via the negation
    /// overlay, not by adding a new bond.
    #[test]
    fn t3_set_negation_pending_applies() {
        use crate::node::Confidence;
        let mut brain = Brain::new();
        let r = brain.teach("sky", BondType::IsA, "blue", 0.9, Confidence::Verified);
        let initial = brain.network.bond_count();
        let change = PendingChange {
            pending_id: 0,
            proposed_at_ms: 1_000_000,
            proposed_by: SourceAttribution::new(SourceKind::Teach {
                actor: "john".into(),
                ts_ms: 1_000_000,
            }),
            change_kind: PendingChangeKind::SetNegation {
                from: r.subject_seed,
                to: r.object_seed,
                bond_type: BondType::IsA,
                negated: true,
            },
            confidence: 1.0,
            reason: "sky is not blue under all conditions".into(),
            status: PendingStatus::Pending,
        };
        let id = brain.pending.push(change).unwrap();
        brain.pending.approve(id, 1_000_000, "john").unwrap();
        apply_change(&mut brain, id).unwrap();
        // No new bond — negation is an overlay flag.
        assert_eq!(brain.network.bond_count(), initial);
        assert!(brain.bond_negated(&r.subject_seed, &r.object_seed, BondType::IsA));
    }

    /// Test 11 — A WAL Push + Approve + Apply sequence replayed
    /// after a crash leaves the log consistent and the brain
    /// updated only via apply_change after replay.
    #[test]
    fn t3_wal_replay_lifecycle_states() {
        let p = tmp_wal_path("lifecycle");
        {
            let mut log = PendingLog::open(p.clone()).unwrap();
            let id = log.push(make_research_change("k", "v", 0.7)).unwrap();
            log.approve(id, 2_000_000, "john").unwrap();
            log.mark_applied(id, 2_000_500).unwrap();
        }
        // Replay.
        let log2 = PendingLog::open(p.clone()).unwrap();
        let e = log2.get(1).unwrap();
        assert!(e.status.is_applied());
        assert_eq!(log2.applied_count(), 1);
        assert_eq!(log2.last_commit_at_ms(), Some(2_000_500));
        let _ = std::fs::remove_file(&p);
    }

    /// Test 12 — A torn line at EOF (process killed mid-write) is
    /// skipped and earlier records survive.
    #[test]
    fn t3_wal_replay_skips_torn_line() {
        let p = tmp_wal_path("torn");
        {
            let mut log = PendingLog::open(p.clone()).unwrap();
            log.push(make_research_change("a", "b", 0.6)).unwrap();
            log.push(make_research_change("c", "d", 0.6)).unwrap();
        }
        // Append a partial JSON line and verify replay tolerates it.
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap();
        f.write_all(b"{\"Push\":{\"pending_id\":99,\"propos")
            .unwrap();
        f.flush().unwrap();
        drop(f);

        let log2 = PendingLog::open(p.clone()).unwrap();
        assert_eq!(log2.total_count(), 2);
        let _ = std::fs::remove_file(&p);
    }

    // ---------------------------------------------------------------
    // Port T.4 — PendingLog persistence into the v3 binary section.
    //
    // T.3 made the WAL the only persistent record of pending state
    // between brain saves. T.4 closes that gap: every Brain::save
    // now serializes the consolidated PendingLog snapshot into v3
    // section id 9 (`pending_log`) and truncates the WAL. Loads
    // hydrate from the section first and then idempotently replay
    // the WAL on top — so a crash between brain-fsync and
    // WAL-truncate is safe (replay dedupes by pending_id), and a
    // backup of brain.lbr alone now survives a restart with full
    // pending state intact.
    // ---------------------------------------------------------------

    fn tmp_brain_dir(label: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lu_t4_test_{}_{}",
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

    /// Port T.4 Test 1 — Brain::save serializes the PendingLog into
    /// the v3 binary section (id 9, name `pending_log`). Verifies the
    /// section is present in the offset table with non-zero length and
    /// that the bincoded snapshot inside it round-trips through
    /// `read_v3_bytes` to the same entries.
    #[test]
    fn t4_save_serializes_pending_into_v3_section() {
        use crate::brain::Brain;
        use crate::brain_io_v3::{read_v3_bytes, write_v3, FORMAT_VERSION_V3};
        let mut brain = Brain::new();
        let id1 = brain.pending.push(make_research_change("a", "b", 0.7)).unwrap();
        let id2 = brain.pending.push(make_research_change("c", "d", 0.6)).unwrap();
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let bytes = write_v3(&file).unwrap();
        // Header: section_count is at bytes[48..52].
        let count = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
        assert_eq!(count, 17, "Port Cat.5 brain has 17 sections");
        // Offset table: header (64 bytes) followed by 10 * 24-byte entries.
        // Find the entry with section_id == 9 (PendingLog).
        let mut found_pending = false;
        for i in 0..count as usize {
            let off = 64 + i * 24;
            let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            if sid == 9 {
                let length = u64::from_le_bytes(
                    bytes[off + 16..off + 24].try_into().unwrap(),
                );
                assert!(length > 0, "PendingLog section length must be non-zero");
                found_pending = true;
            }
        }
        assert!(found_pending, "PendingLog (id 9) must be in the offset table");
        // Round-trip: the bincoded section bytes deserialize back to a
        // snapshot with both entries and the right next_pending_id.
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert_eq!(parsed.pending_log.entries.len(), 2);
        assert_eq!(parsed.pending_log.entries[0].pending_id, id1);
        assert_eq!(parsed.pending_log.entries[1].pending_id, id2);
        assert_eq!(parsed.pending_log.next_pending_id, 3);
    }

    /// Port T.4 Test 2 — End-to-end round-trip through Brain::save
    /// and Brain::load_or_new. Pending state is reconstructed from the
    /// v3 section, NOT just from the WAL (the WAL is truncated by
    /// save).
    #[test]
    fn t4_load_reads_pending_from_v3_section() {
        use crate::brain::Brain;
        let dir = tmp_brain_dir("load");
        let path = dir.join("brain.lbr");
        let wal_path = path.with_file_name("brain.lbr.pending.wal");

        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.attach_pending_wal(PendingLogSnapshot::default());
        let id1 = brain
            .pending
            .push(make_research_change("apple", "fruit", 0.9))
            .unwrap();
        let id2 = brain
            .pending
            .push(make_research_change("dog", "mammal", 0.95))
            .unwrap();
        brain.save().unwrap();
        // T.4 invariant: WAL is truncated post-save — the v3 section
        // is the source of truth from this point on.
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);
        drop(brain);

        let loaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(loaded.pending.total_count(), 2);
        assert!(loaded.pending.get(id1).is_some());
        assert!(loaded.pending.get(id2).is_some());
        assert_eq!(loaded.pending.next_id(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Port T.4 Test 3 — After a save, new pushes go ONLY to the
    /// WAL until the next save. On reload, open_with_snapshot replays
    /// the WAL on top of the section snapshot. When the WAL contains
    /// a Push for a pending_id already in the snapshot (the
    /// crash-between-save-and-truncate corner case), replay is
    /// idempotent. When it contains genuinely new ids, they are
    /// appended.
    #[test]
    fn t4_load_v3_section_then_replays_wal_idempotently() {
        use crate::brain::Brain;
        let dir = tmp_brain_dir("idempotent");
        let path = dir.join("brain.lbr");

        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.attach_pending_wal(PendingLogSnapshot::default());
        // Phase A: 2 entries, save — these go into the v3 section.
        brain.pending.push(make_research_change("a", "b", 0.7)).unwrap();
        brain.pending.push(make_research_change("c", "d", 0.7)).unwrap();
        brain.save().unwrap();
        // Phase B: 2 more entries — these go ONLY to the WAL.
        brain.pending.push(make_research_change("e", "f", 0.7)).unwrap();
        brain.pending.push(make_research_change("g", "h", 0.7)).unwrap();
        drop(brain);

        // Phase C: reload. Section gives 1+2, WAL gives Push(3)+Push(4).
        // open_with_snapshot replays WAL idempotently — since 1 and 2
        // are not in the WAL (post-save), no dedup is needed for them;
        // 3 and 4 are appended.
        let loaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(loaded.pending.total_count(), 4);
        assert_eq!(loaded.pending.next_id(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Port T.4 Test 4 — After a successful Brain::save the
    /// `<brain>.pending.wal` file is truncated to 0 bytes. The v3
    /// section is the source of truth; the WAL only captures
    /// mutations made AFTER the save.
    #[test]
    fn t4_save_truncates_wal() {
        use crate::brain::Brain;
        let dir = tmp_brain_dir("trunc");
        let path = dir.join("brain.lbr");
        let wal_path = path.with_file_name("brain.lbr.pending.wal");

        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.attach_pending_wal(PendingLogSnapshot::default());
        brain.pending.push(make_research_change("a", "b", 0.7)).unwrap();
        brain.pending.push(make_research_change("c", "d", 0.7)).unwrap();
        let pre = std::fs::metadata(&wal_path).unwrap().len();
        assert!(pre > 0, "WAL must be non-empty after pushes");
        brain.save().unwrap();
        let post = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(post, 0, "T.4 save truncates the WAL");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Port T.4 Test 5 — A pre-T.4 v3 brain (one that lacks the
    /// PendingLog section) loads with an empty PendingLogSnapshot.
    /// Synthesize that shape by writing a T.4 brain and patching the
    /// header to claim only 8 sections (so the parser ignores the
    /// last offset-table entry, which is the PendingLog one), then
    /// recomputing the trailer CRC.
    #[test]
    fn t4_pre_t4_brain_loads_with_empty_pending() {
        use crate::brain::Brain;
        use crate::brain_io_v3::{read_v3_bytes, write_v3, FORMAT_VERSION_V3};
        let brain = Brain::new();
        let file = brain.to_brain_file(FORMAT_VERSION_V3);
        let mut bytes = write_v3(&file).unwrap();
        // Decrement section_count from 9 to 8 in the fixed header.
        let pre_t4_count: u32 = 8;
        bytes[48..52].copy_from_slice(&pre_t4_count.to_le_bytes());
        // Recompute trailer CRC over [..trailer_pos].
        let trailer_pos = bytes.len() - 16;
        let crc = crc32fast::hash(&bytes[..trailer_pos]);
        bytes[trailer_pos + 8..trailer_pos + 12]
            .copy_from_slice(&crc.to_le_bytes());
        // The parser now iterates only 8 entries from the offset table
        // — the PendingLog entry (the 9th) is invisible.
        // de_bincode_optional returns Default::default() for the missing
        // section, so pending_log comes back empty (entries: [],
        // next_pending_id: 0). Note the snapshot.next_pending_id is 0
        // here, NOT 1 — PendingLogSnapshot uses #[derive(Default)],
        // which yields zero-initialized fields; the per-field serde
        // default (= 1) only applies during JSON deserialization, not
        // when the entire section is absent. open_with_snapshot
        // saturates the in-memory next_id with .max(1), so the
        // user-visible PendingLog still starts at id=1.
        let parsed = read_v3_bytes(&bytes).unwrap();
        assert!(parsed.pending_log.entries.is_empty());
        assert_eq!(parsed.pending_log.next_pending_id, 0);
        // Verify the saturation: feeding this snapshot into a fresh
        // PendingLog yields the expected starting state.
        let wal = std::env::temp_dir().join(format!(
            "lu_t4_pre_t4_wal_{}.wal",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let log = PendingLog::open_with_snapshot(wal.clone(), parsed.pending_log).unwrap();
        assert_eq!(log.total_count(), 0);
        assert_eq!(log.next_id(), 1);
        let _ = std::fs::remove_file(&wal);
    }

    /// Port T.4 Test 6 — Crash simulation: brain save succeeded,
    /// WAL truncate "failed" (or the OS crashed between fsync-of-brain
    /// and the WAL truncate call). On the next load, the snapshot has
    /// the entries and the WAL still has duplicate Push records for
    /// the same pending_ids. PendingLog::open_with_snapshot replays
    /// the WAL idempotently — no duplicates appear in the final
    /// in-memory log.
    ///
    /// This is a logical-state crash sim, not a process-level one: a
    /// real fork-and-SIGKILL test isn't meaningfully more rigorous,
    /// since the only post-save mutation in the save path IS the WAL
    /// truncate, and the recovery property we care about is
    /// idempotent replay over a stale WAL.
    #[test]
    fn t4_atomic_save_crash_simulation() {
        use crate::brain::Brain;
        let dir = tmp_brain_dir("crash");
        let path = dir.join("brain.lbr");
        let wal_path = path.with_file_name("brain.lbr.pending.wal");

        // Phase A: build, push, save (truncates WAL).
        let mut brain = Brain::new();
        brain.path = Some(path.clone());
        brain.attach_pending_wal(PendingLogSnapshot::default());
        let mut pushed = Vec::new();
        for s in ["a", "b", "c"].iter() {
            let id = brain
                .pending
                .push(make_research_change(s, "x", 0.7))
                .unwrap();
            let change = brain.pending.get(id).unwrap().clone();
            pushed.push(change);
        }
        let pre_count = brain.pending.total_count();
        brain.save().unwrap();
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);
        // Drop the brain so the WalWriter handle releases.
        drop(brain);

        // Phase B: simulate "WAL truncate failed" by re-appending
        // Push records for entries that are already in the v3 section.
        let mut writer = WalWriter::open(wal_path.clone()).unwrap();
        for change in &pushed {
            writer.append(&WalRecord::Push(change.clone())).unwrap();
        }
        drop(writer);
        assert!(std::fs::metadata(&wal_path).unwrap().len() > 0);

        // Phase C: reload. open_with_snapshot inserts the snapshot
        // entries first, then the WAL replay sees Push records whose
        // pending_ids already exist — the apply_wal_record `Push`
        // arm has the `if !self.entries.contains_key` guard. Result:
        // no duplicates.
        let loaded = Brain::load_or_new(&path).unwrap();
        assert_eq!(
            loaded.pending.total_count(),
            pre_count,
            "no duplicates after stale WAL replay over snapshot"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
