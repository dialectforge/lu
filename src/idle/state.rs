//! Port T.2 — Idle state tracker.
//!
//! Brain knows whether MCP queries are currently being processed, and
//! how long since the last one finished. Built on T.1's monotonic
//! clock so the duration is immune to wall-clock skew. Runtime-only
//! — not serialised in the brain file. Same overlay-pattern as G/S/T.1.
//!
//! Concurrency model: `IdleState` wraps `Arc<RwLock<IdleStateInner>>`
//! so a `QueryGuard` (RAII) can hold an Arc clone that lives
//! independently of any `Brain` borrow. That lets the MCP dispatch
//! wrap the inner tool dispatch in a guard while still passing
//! `&mut Brain` to the tool function.

use crate::time::{MonotonicClock, MonotonicTimestamp};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Inner mutable state. Behind an `Arc<RwLock<>>` so [`QueryGuard`]
/// can hold a clone independent of any [`Brain`](crate::brain::Brain) borrow.
#[derive(Debug, Clone, Default)]
pub struct IdleStateInner {
    /// MCP queries currently in flight. Idle iff `== 0`.
    pub active_queries: u32,
    /// Monotonic timestamp captured when the most recent query
    /// finished (the moment `active_queries` dropped to 0).
    /// `None` until the first query has ever ended.
    pub last_query_end: Option<MonotonicTimestamp>,
    /// Cumulative count of `lu4d_idle_tick` invocations that reached
    /// the gate (regardless of outcome).
    pub total_idle_ticks: u64,
    /// Cumulative count of `lu4d_idle_tick` invocations that actually
    /// performed a unit of idle work (e.g. processed a wonder signal
    /// or surfaced a research failure on a popped wonder).
    pub total_idle_work_completed: u64,
}

impl IdleStateInner {
    pub fn record_query_start(&mut self) {
        self.active_queries = self.active_queries.saturating_add(1);
    }

    pub fn record_query_end(&mut self) {
        if self.active_queries > 0 {
            self.active_queries -= 1;
        }
        if self.active_queries == 0 {
            self.last_query_end = Some(MonotonicClock::now());
        }
    }

    pub fn is_idle(&self) -> bool {
        self.active_queries == 0
    }

    /// Milliseconds since the last query ended. `None` until the first
    /// query ever finishes. Does NOT consult `active_queries` — the
    /// caller decides what "idle" means.
    pub fn last_query_age_ms(&self) -> Option<u64> {
        self.last_query_end.map(|t| MonotonicClock::duration_since_ms(t))
    }

    /// Milliseconds the brain has been idle. `None` if either:
    ///   - a query is currently in flight (`active_queries > 0`), or
    ///   - no query has ever ended.
    pub fn idle_duration_ms(&self) -> Option<u64> {
        if self.active_queries > 0 {
            return None;
        }
        self.last_query_age_ms()
    }
}

/// Public idle-state handle. Cheap to clone (it's just an `Arc`).
#[derive(Debug, Clone, Default)]
pub struct IdleState {
    inner: Arc<RwLock<IdleStateInner>>,
}

impl IdleState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_idle(&self) -> bool {
        self.inner.read().map(|s| s.is_idle()).unwrap_or(false)
    }

    pub fn idle_duration_ms(&self) -> Option<u64> {
        self.inner.read().ok()?.idle_duration_ms()
    }

    /// Spec-API alias for [`Self::idle_duration_ms`] returning a
    /// `std::time::Duration`. The `_ms` flavour is preferred internally
    /// since the MCP surface ultimately reports milliseconds.
    pub fn idle_duration(&self) -> Option<Duration> {
        self.idle_duration_ms().map(Duration::from_millis)
    }

    pub fn last_query_age_ms(&self) -> Option<u64> {
        self.inner.read().ok()?.last_query_age_ms()
    }

    pub fn last_query_end(&self) -> Option<MonotonicTimestamp> {
        self.inner.read().ok()?.last_query_end
    }

    pub fn active_queries(&self) -> u32 {
        self.inner.read().map(|s| s.active_queries).unwrap_or(0)
    }

    pub fn total_idle_ticks(&self) -> u64 {
        self.inner.read().map(|s| s.total_idle_ticks).unwrap_or(0)
    }

    pub fn total_idle_work_completed(&self) -> u64 {
        self.inner.read().map(|s| s.total_idle_work_completed).unwrap_or(0)
    }

    /// Record a query start. Direct API for tests; production code
    /// should use [`QueryGuard::new`] for RAII safety.
    pub fn on_query_start(&self) {
        if let Ok(mut s) = self.inner.write() {
            s.record_query_start();
        }
    }

    /// Record a query end. Direct API for tests; production code
    /// should use [`QueryGuard::new`] for RAII safety.
    pub fn on_query_end(&self) {
        if let Ok(mut s) = self.inner.write() {
            s.record_query_end();
        }
    }

    /// Increment `total_idle_ticks`, plus `total_idle_work_completed`
    /// when the tick produced work. Called by the idle loop.
    pub fn record_tick(&self, did_work: bool) {
        if let Ok(mut s) = self.inner.write() {
            s.total_idle_ticks = s.total_idle_ticks.saturating_add(1);
            if did_work {
                s.total_idle_work_completed =
                    s.total_idle_work_completed.saturating_add(1);
            }
        }
    }

    /// Internal: clone the underlying `Arc` for [`QueryGuard`].
    pub(crate) fn arc(&self) -> Arc<RwLock<IdleStateInner>> {
        self.inner.clone()
    }
}

/// RAII query guard. On construction, increments `active_queries`. On
/// drop, decrements and (if the count reaches zero) stamps
/// `last_query_end`. Holds an `Arc` clone of the inner state — it does
/// NOT borrow [`Brain`](crate::brain::Brain) — so MCP dispatch can
/// hold a guard alongside `&mut brain` without lifetime conflicts.
pub struct QueryGuard {
    inner: Arc<RwLock<IdleStateInner>>,
}

impl QueryGuard {
    pub fn new(state: &IdleState) -> Self {
        state.on_query_start();
        QueryGuard { inner: state.arc() }
    }
}

impl Drop for QueryGuard {
    fn drop(&mut self) {
        if let Ok(mut s) = self.inner.write() {
            s.record_query_end();
        }
    }
}

/// Outcome categories surfaced by [`crate::idle::loop_::idle_tick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleTickOutcome {
    /// No work attempted — wonder queue empty / below threshold /
    /// research disabled. All collapsed into one outcome since they
    /// share the same caller-visible meaning: "there was nothing
    /// idle-worthy to do".
    NoWorkNoSignals,
    /// No work attempted — brain wasn't idle (`active_queries > 0`)
    /// or the `min_idle_ms` gate wasn't satisfied.
    NoWorkNotIdle,
    /// LEGACY (T.2): a wonder signal was popped and the research
    /// path queued a `Pending` ResearchProposal. T.3 replaced this
    /// with [`Self::PendingAdded`]; the variant remains for back-
    /// compat with any caller pinned to the old outcome shape.
    ResearchCompleted,
    /// Port T.3: research succeeded and one or more bond candidates
    /// were pushed into [`crate::pending::PendingLog`] as
    /// `PendingChangeKind::AddBond` entries. The brain bond count
    /// is UNCHANGED until those entries are explicitly approved
    /// via the `lu4d_approve_pending` MCP tool.
    PendingAdded,
    /// A wonder signal was popped but the research path returned an
    /// error (router failure, no extractable text, queue full, etc.).
    /// Counts as work completed since we made an attempt.
    ResearchFailed,
    /// Port "internal-first" (2026-05-30): the wonder was filled from
    /// existing graph knowledge via local triangle closure
    /// (`internal_fill_at_seed`) BEFORE any external research call. One
    /// or more synthesis-derived bond candidates were pushed into
    /// `PendingLog`. Brain bond count unchanged until approved.
    InternalFillAdded,
}

impl IdleTickOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            IdleTickOutcome::NoWorkNoSignals => "no_work_no_signals",
            IdleTickOutcome::NoWorkNotIdle => "no_work_not_idle",
            IdleTickOutcome::ResearchCompleted => "research_completed",
            IdleTickOutcome::PendingAdded => "pending_added",
            IdleTickOutcome::ResearchFailed => "research_failed",
            IdleTickOutcome::InternalFillAdded => "internal_fill_added",
        }
    }
}

/// Result of a single [`crate::idle::loop_::idle_tick`] call.
#[derive(Debug, Clone)]
pub struct IdleTickResult {
    pub did_work: bool,
    /// Hex-encoded 8-byte prefix of the wonder seed that was popped.
    /// `None` when no signal was popped (gate failed, queue empty).
    pub wonder_seed_processed: Option<String>,
    pub outcome: IdleTickOutcome,
    /// Idle duration captured at the start of the tick. Reported in
    /// the MCP response so callers can see "how long had Lu been idle
    /// when this tick fired".
    pub idle_duration_ms: u64,
    /// Cumulative idle-tick counter AFTER this tick was recorded.
    pub total_idle_ticks: u64,
    /// Port T.3: count of pending-log entries this tick produced.
    /// Zero unless `outcome` is [`IdleTickOutcome::PendingAdded`].
    pub pending_added_count: u32,
}
