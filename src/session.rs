//! Session isolation — ensures Hebbian learning during a session
//! doesn't contaminate the baseline knowledge graph.
//!
//! Each session has a unique ID and a buffer of keyframes written
//! during that session. Keyframes are tagged with the session ID so
//! they can be distinguished from baseline knowledge.

use crate::network::AssociativeNetwork;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};

/// A record of a keyframe written during a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionKeyframeRecord {
    /// Bond source node.
    pub from: Seed,
    /// Bond target node.
    pub to: Seed,
    /// The weight written.
    pub weight: f32,
    /// Timestamp of the keyframe.
    pub timestamp: i64,
}

/// A temporal session that isolates Hebbian learning from the baseline graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSession {
    /// Unique session identifier.
    pub id: u64,
    /// Session start time (epoch millis).
    pub start_time: i64,
    /// Session end time, if ended (epoch millis).
    pub end_time: Option<i64>,
    /// Buffer of keyframes written during this session.
    pub keyframe_buffer: Vec<SessionKeyframeRecord>,
    /// Seeds activated during this session (for Hebbian pair tracking).
    pub activated_seeds: Vec<Seed>,
    /// Whether this session has been consolidated into baseline.
    pub consolidated: bool,
}

impl TemporalSession {
    /// Create a new session with the given ID and start time.
    pub fn new(id: u64, start_time: i64) -> Self {
        TemporalSession {
            id,
            start_time,
            end_time: None,
            keyframe_buffer: Vec::new(),
            activated_seeds: Vec::new(),
            consolidated: false,
        }
    }

    /// Record that a set of seeds were activated in this session.
    /// Deduplicates against previously activated seeds.
    pub fn record_activation(&mut self, seeds: &[Seed]) {
        for seed in seeds {
            if !self.activated_seeds.contains(seed) {
                self.activated_seeds.push(*seed);
            }
        }
    }

    /// Apply Hebbian strengthening for all co-activated pairs in this session.
    ///
    /// This calls `network.hebbian_strengthen()` and records the resulting
    /// keyframes in the session buffer.
    pub fn apply_hebbian(
        &mut self,
        network: &mut AssociativeNetwork,
        timestamp: i64,
    ) -> usize {
        let seeds = self.activated_seeds.clone();
        let strengthened = network.hebbian_strengthen(&seeds, self.id, timestamp);

        // Record the keyframes that were written
        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                let a = seeds[i];
                let b = seeds[j];

                if let Some(bond) = network.get_bond(&a, &b) {
                    self.keyframe_buffer.push(SessionKeyframeRecord {
                        from: a,
                        to: b,
                        weight: bond.latest_raw_weight(),
                        timestamp,
                    });
                }
                if let Some(bond) = network.get_bond(&b, &a) {
                    self.keyframe_buffer.push(SessionKeyframeRecord {
                        from: b,
                        to: a,
                        weight: bond.latest_raw_weight(),
                        timestamp,
                    });
                }
            }
        }

        strengthened
    }

    /// End this session, recording the end time.
    pub fn end(&mut self, end_time: i64) {
        self.end_time = Some(end_time);
    }

    /// Check if this session is still active (not ended).
    pub fn is_active(&self) -> bool {
        self.end_time.is_none()
    }

    /// Return the duration of this session in milliseconds.
    /// Returns None if the session is still active.
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time.map(|end| end - self.start_time)
    }

    /// Return the number of keyframes written during this session.
    pub fn keyframe_count(&self) -> usize {
        self.keyframe_buffer.len()
    }

    /// Return the number of unique seeds activated during this session.
    pub fn activation_count(&self) -> usize {
        self.activated_seeds.len()
    }
}

/// Manages multiple sessions and tracks session history.
pub struct SessionManager {
    /// All sessions, keyed by session ID.
    sessions: std::collections::HashMap<u64, TemporalSession>,
    /// Counter for generating unique session IDs.
    next_id: u64,
    /// The currently active session, if any.
    active_session_id: Option<u64>,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        SessionManager {
            sessions: std::collections::HashMap::new(),
            next_id: 1,
            active_session_id: None,
        }
    }

    /// Start a new session. Returns the session ID.
    ///
    /// If another session is active, it will be ended first.
    pub fn start_session(&mut self, start_time: i64) -> u64 {
        // End any active session
        if let Some(active_id) = self.active_session_id {
            if let Some(session) = self.sessions.get_mut(&active_id) {
                session.end(start_time);
            }
        }

        let id = self.next_id;
        self.next_id += 1;

        let session = TemporalSession::new(id, start_time);
        self.sessions.insert(id, session);
        self.active_session_id = Some(id);

        id
    }

    /// End the currently active session.
    pub fn end_active_session(&mut self, end_time: i64) -> Option<u64> {
        if let Some(id) = self.active_session_id.take() {
            if let Some(session) = self.sessions.get_mut(&id) {
                session.end(end_time);
            }
            Some(id)
        } else {
            None
        }
    }

    /// Get the currently active session.
    pub fn active_session(&self) -> Option<&TemporalSession> {
        self.active_session_id
            .and_then(|id| self.sessions.get(&id))
    }

    /// Get a mutable reference to the currently active session.
    pub fn active_session_mut(&mut self) -> Option<&mut TemporalSession> {
        self.active_session_id
            .and_then(|id| self.sessions.get_mut(&id))
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: u64) -> Option<&TemporalSession> {
        self.sessions.get(&id)
    }

    /// Get all completed (non-active) sessions.
    pub fn completed_sessions(&self) -> Vec<&TemporalSession> {
        self.sessions
            .values()
            .filter(|s| !s.is_active())
            .collect()
    }

    /// Return the total number of sessions (active + completed).
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, TemporalBond};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};

    #[test]
    fn test_session_lifecycle() {
        let mut mgr = SessionManager::new();
        let id = mgr.start_session(1000);
        assert_eq!(id, 1);
        assert!(mgr.active_session().unwrap().is_active());

        mgr.end_active_session(2000);
        let session = mgr.get_session(id).unwrap();
        assert!(!session.is_active());
        assert_eq!(session.duration_ms(), Some(1000));
    }

    #[test]
    fn test_session_auto_end_on_new() {
        let mut mgr = SessionManager::new();
        let id1 = mgr.start_session(1000);
        let id2 = mgr.start_session(2000);

        // First session should have been auto-ended
        assert!(!mgr.get_session(id1).unwrap().is_active());
        assert!(mgr.get_session(id2).unwrap().is_active());
    }

    #[test]
    fn test_session_hebbian() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("SessionA");
        let sb = Seed::from_label("SessionB");

        net.add_node(KnowledgeNode::with_seed(
            sa, "A".into(),
            KnowledgeContent::Raw("Node A".into()),
            Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            sb, "B".into(),
            KnowledgeContent::Raw("Node B".into()),
            Confidence::Verified, 0,
        ));
        net.add_bond(TemporalBond::new(sa, sb, BondType::RelatedTo, 0.5, 0))
            .unwrap();

        let mut session = TemporalSession::new(1, 1000);
        session.record_activation(&[sa, sb]);

        let strengthened = session.apply_hebbian(&mut net, 1500);
        assert!(strengthened > 0);
        assert!(!session.keyframe_buffer.is_empty());
    }

    #[test]
    fn test_record_activation_dedup() {
        let mut session = TemporalSession::new(1, 0);
        let s = Seed::from_label("Dedup");
        session.record_activation(&[s, s, s]);
        assert_eq!(session.activation_count(), 1);
    }
}
