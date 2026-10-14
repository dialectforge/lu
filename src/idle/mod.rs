//! Port T.2 — Idle detection + idle work loop.
//!
//! Lu now knows whether MCP queries are currently being processed and
//! how long since the last one finished, and exposes `idle_tick`
//! (driven by the new `lu4d_idle_tick` MCP tool) that does ONE unit of
//! idle work per call. First useful idle work: drain the wonder queue
//! into pending research proposals.
//!
//! Pre-retrain awareness: T.2 is runtime-only — `Brain.idle_state` is
//! NOT serialised. Same overlay-pattern as G/S/T.1. The auto-commit
//! behaviour (idle research lands as `Pending` proposals) is a known
//! gap that T.3 closes by adding the pending-changes-with-approval
//! gate.
//!
//! Predecessors:
//!   - internal notes — Port T scoping spec.
//!   - internal notes — Port T.1 (time primitive) ship.
//!   - internal notes — curiosity-load-bearing (idle work makes the
//!     curiosity loop autonomous).
//!   - internal notes — Port J.1 wonder queue persistence (T.2
//!     consumes it).

pub mod state;
pub mod loop_;
pub mod auto_tick;

pub use state::{
    IdleState, IdleStateInner, IdleTickOutcome, IdleTickResult, QueryGuard,
};
pub use loop_::idle_tick;
pub use auto_tick::{AutoTickConfig, AutoTickDecision, AutoTickDriver, AutoTickRunState, AutoTickSkipReason, AutoTickStateInner, AutoTickStatus, MAX_AUTO_TICK_INTERVAL_MS, MAX_AUTO_TICK_PER_HOUR, MIN_AUTO_TICK_INTERVAL_MS, MIN_AUTO_TICK_PER_HOUR};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;
    use crate::curiosity::{CuriositySignal, SignalType};
    use crate::research::http::MockFetcher;
    use crate::scope::TemporalScope;
    use crate::seed::Seed;
    use std::thread;
    use std::time::Duration;

    /// T.2: a fresh Brain reports as idle with no idle duration.
    #[test]
    fn t2_idle_state_starts_idle() {
        let b = Brain::new();
        assert!(b.is_idle(), "fresh Brain must be idle");
        assert!(
            b.idle_duration().is_none(),
            "fresh Brain must have no idle duration (no prior query)"
        );
        assert_eq!(b.idle_state.active_queries(), 0);
        assert_eq!(b.idle_state.total_idle_ticks(), 0);
        assert_eq!(b.idle_state.total_idle_work_completed(), 0);
    }

    /// T.2: a query in flight flips `is_idle` to false; ending it flips
    /// back to true and stamps a duration.
    #[test]
    fn t2_query_in_flight_sets_not_idle() {
        let b = Brain::new();
        b.idle_state.on_query_start();
        assert!(!b.is_idle(), "query in flight: brain must not be idle");
        assert!(
            b.idle_duration().is_none(),
            "query in flight: idle_duration must be None"
        );
        b.idle_state.on_query_end();
        assert!(b.is_idle(), "after query end: brain must be idle");
        // Sleep a bit so the duration is observably > 0.
        thread::sleep(Duration::from_millis(5));
        let dur = b.idle_duration().expect("idle_duration must be Some after a query end");
        assert!(dur.as_millis() > 0, "idle_duration must be > 0 ms after sleep");
    }

    /// T.2: nested / concurrent queries — start two, end one, still not
    /// idle; end the second, idle.
    #[test]
    fn t2_concurrent_queries_track_correctly() {
        let b = Brain::new();
        b.idle_state.on_query_start();
        b.idle_state.on_query_start();
        assert_eq!(b.idle_state.active_queries(), 2);
        assert!(!b.is_idle());
        b.idle_state.on_query_end();
        assert_eq!(b.idle_state.active_queries(), 1);
        assert!(!b.is_idle(), "one query still in flight: not idle");
        b.idle_state.on_query_end();
        assert_eq!(b.idle_state.active_queries(), 0);
        assert!(b.is_idle(), "both queries done: idle");
        assert!(
            b.idle_state.last_query_end().is_some(),
            "last_query_end must be stamped when active reaches 0"
        );
    }

    /// T.2: idle_tick when a query is in flight returns NoWorkNotIdle.
    #[test]
    fn t2_idle_tick_does_no_work_when_not_idle() {
        let mut b = Brain::new();
        b.idle_state.on_query_start();
        let m = MockFetcher::new();
        let res = idle_tick(&mut b, &m, 0);
        assert!(!res.did_work);
        assert_eq!(res.outcome, IdleTickOutcome::NoWorkNotIdle);
        assert!(res.wonder_seed_processed.is_none());
        assert_eq!(b.idle_state.total_idle_ticks(), 1, "tick must be counted");
        assert_eq!(b.idle_state.total_idle_work_completed(), 0);
        // Cleanup — keep state sensible for any subsequent code.
        b.idle_state.on_query_end();
    }

    /// T.2: idle_tick with empty wonder queue returns NoWorkNoSignals.
    #[test]
    fn t2_idle_tick_does_no_work_when_no_signals() {
        let mut b = Brain::new();
        assert!(b.is_idle());
        let m = MockFetcher::new();
        let res = idle_tick(&mut b, &m, 0);
        assert!(!res.did_work);
        assert_eq!(res.outcome, IdleTickOutcome::NoWorkNoSignals);
        assert!(res.wonder_seed_processed.is_none());
        assert_eq!(b.idle_state.total_idle_ticks(), 1);
        assert_eq!(b.idle_state.total_idle_work_completed(), 0);
    }

    /// T.2: idle_tick with a wonder signal pops it and runs research.
    /// We push a synthetic high-intensity wonder, route Grokipedia +
    /// the result page through MockFetcher so the research path can
    /// produce a `Pending` proposal. Outcome may be ResearchCompleted
    /// (full success) or ResearchFailed (extractor produced no
    /// triples) — both prove the wonder was popped and the path ran.
    #[test]
    fn t2_idle_tick_processes_wonder_signal() {
        let mut b = Brain::new();
        let seed = Seed::from_label("rust");
        b.wonder_queue.push(CuriositySignal {
            seed,
            signal_type: SignalType::DeadEnd,
            intensity: 0.95,
            scope: TemporalScope::Current,
            timestamp_ms: 1,
        });
        let m = MockFetcher::new();
        // Stub the Grokipedia search + result page so the router can
        // produce real evidence. The "rust" -> seed mapping is ignored
        // by the router; it queries by the wonder's surface label.
        m.route_ok(
            "https://grokipedia.com/search?q=rust",
            200,
            "<html><body><a href=\"/page/Rust\">Result</a></body></html>",
        );
        m.route_ok(
            "https://grokipedia.com/page/Rust",
            200,
            "<html><body><main><p>Rust is a systems programming language.</p></main></body></html>",
        );
        let res = idle_tick(&mut b, &m, 0);
        assert!(
            matches!(
                res.outcome,
                IdleTickOutcome::PendingAdded | IdleTickOutcome::ResearchFailed
            ),
            "expected research path to run; got {:?}",
            res.outcome
        );
        assert!(res.did_work, "a popped wonder must register as work");
        assert!(
            res.wonder_seed_processed.is_some(),
            "wonder_seed_processed must be populated when a wonder pops"
        );
        assert_eq!(b.idle_state.total_idle_ticks(), 1);
        assert_eq!(b.idle_state.total_idle_work_completed(), 1);
    }

    /// T.2: min_idle_ms gate — even when idle, do nothing if the brain
    /// hasn't been idle long enough.
    #[test]
    fn t2_min_idle_ms_gate() {
        let mut b = Brain::new();
        // Simulate a query that just finished — last_query_end is now,
        // so idle_age starts very small.
        b.idle_state.on_query_start();
        b.idle_state.on_query_end();
        let m = MockFetcher::new();
        let res = idle_tick(&mut b, &m, 10_000);
        assert!(!res.did_work, "min_idle_ms gate must block work");
        assert_eq!(res.outcome, IdleTickOutcome::NoWorkNotIdle);
        assert!(res.wonder_seed_processed.is_none());
        // Counter must still tick.
        assert_eq!(b.idle_state.total_idle_ticks(), 1);
        assert_eq!(b.idle_state.total_idle_work_completed(), 0);
    }

    /// T.2: on_query_end must update the monotonic last_query_end
    /// timestamp. Two end events separated by a sleep produce two
    /// distinct timestamps.
    #[test]
    fn t2_query_end_updates_last_query_end_timestamp() {
        let b = Brain::new();
        b.idle_state.on_query_start();
        b.idle_state.on_query_end();
        let t1 = b.idle_state.last_query_end().expect("first end stamps a timestamp");
        thread::sleep(Duration::from_millis(5));
        b.idle_state.on_query_start();
        b.idle_state.on_query_end();
        let t2 = b.idle_state.last_query_end().expect("second end stamps a timestamp");
        assert!(
            t2.as_nanos() > t1.as_nanos(),
            "second on_query_end must advance the monotonic timestamp: t1={} t2={}",
            t1.as_nanos(),
            t2.as_nanos(),
        );
    }

    /// T.2: QueryGuard RAII — increments on construction, decrements on
    /// drop. Stamp `last_query_end` when the count reaches 0. Even on
    /// panic, the drop guarantees `active_queries` does not stay stuck.
    #[test]
    fn t2_query_guard_raii_balances_active_queries() {
        let b = Brain::new();
        assert_eq!(b.idle_state.active_queries(), 0);
        {
            let _g = QueryGuard::new(&b.idle_state);
            assert_eq!(b.idle_state.active_queries(), 1);
        }
        assert_eq!(b.idle_state.active_queries(), 0, "guard drop must decrement");
        assert!(
            b.idle_state.last_query_end().is_some(),
            "guard drop must stamp last_query_end when count hits 0",
        );
    }
}
