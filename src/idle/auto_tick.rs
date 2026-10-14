//! Port T.6 — Daemon driver that fires `idle_tick` automatically when the
//! brain has been quiet for a configurable threshold.
//!
//! Same overlay-pattern as T.2 (idle detection) and T.5 (decay knobs):
//! runtime-only, not serialised in the brain file. The actual MCP work
//! still happens through the T.2 `idle_tick` primitive and the T.3
//! pending-changes path -- T.6 does NOT change what `idle_tick` does or
//! introduce auto-commit behaviour. It only schedules the call.
//!
//! Concurrency model mirrors T.2: state lives behind `Arc<RwLock<>>`
//! so a background thread can read/write it without holding any
//! [`crate::brain::Brain`] reference. The actual `idle_tick` invocation
//! still happens on the main MCP thread (which holds `&mut Brain`),
//! triggered by the driver via a wake-up channel.
//!
//! Time discipline (T.6 invariant): every "is it been long enough?"
//! decision goes through the configured [`crate::time::ClockSource`].
//! `Instant::now()` and `SystemTime::now()` MUST NOT be called directly
//! from this module (`MonotonicClock::now()` is allowed only as a
//! degraded-mode fallback when the source is unavailable). The
//! `t6_auto_tick_uses_clock_source_not_system_time` test guards this.
//!
//! Predecessors:
//!   - internal notes   — Port T scoping spec.
//!   - internal notes   — Port T.1 ship (time primitive + ClockSource).
//!   - Port T.2 commit  — IdleState + idle_tick primitive.
//!   - Port T.3 commit  — Pending-changes-with-approval gate.
//!   - Port T.5 commit  — Freshness/decay knobs (config bounds pattern).

use super::IdleState;
use crate::time::{ClockSource, MonotonicClock};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

// ----------------------------------------------------------------------
// Config knobs + bounds (surfaced via `lu4d_configure`).
// ----------------------------------------------------------------------

/// Lower bound for `idle_threshold_ms` and `min_interval_ms`. 1 second is
/// already aggressive for an "idle" detector; anything smaller flirts
/// with the per-call dispatch latency.
pub const MIN_AUTO_TICK_INTERVAL_MS: u64 = 1_000;
/// Upper bound for `idle_threshold_ms` and `min_interval_ms`. 24 hours
/// is more than long enough for any reasonable idle-detection horizon.
pub const MAX_AUTO_TICK_INTERVAL_MS: u64 = 24 * 3_600_000;
/// Lower bound for `max_per_hour`. Below 1 the driver is effectively off
/// (and `enabled = false` is the right way to express that).
pub const MIN_AUTO_TICK_PER_HOUR: u32 = 1;
/// Upper bound for `max_per_hour`. 36 000 = 10 fires per second sustained,
/// already past the point where the wonder queue would be the bottleneck.
pub const MAX_AUTO_TICK_PER_HOUR: u32 = 36_000;

/// One-hour window for `ticks_in_last_hour` accounting.
const ONE_HOUR_MS: u64 = 3_600_000;

/// Tunable knobs for the auto-tick driver. All four fields are surfaced
/// as `lu4d_configure` arguments and validated against the bounds above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoTickConfig {
    /// Master enable switch. When `false`, the driver never fires
    /// regardless of the other knobs. Equivalent to `paused`, except
    /// that `enabled = false` survives across pause/resume calls
    /// (whereas `paused` is process-lifetime-only state).
    pub enabled: bool,
    /// Brain must have been quiet for at least this long for the
    /// driver to consider firing.
    pub idle_threshold_ms: u64,
    /// Minimum elapsed time between two consecutive driver-fired
    /// ticks. Prevents the driver from saturating an empty wonder
    /// queue with back-to-back fires the moment idle conditions hold.
    pub min_interval_ms: u64,
    /// Sliding-window cap on driver fires per hour. Belt-and-braces
    /// against pathological config (e.g. `min_interval_ms = 1`).
    pub max_per_hour: u32,
}

impl Default for AutoTickConfig {
    fn default() -> Self {
        AutoTickConfig {
            enabled: true,
            idle_threshold_ms: 30_000,
            min_interval_ms: 60_000,
            max_per_hour: 60,
        }
    }
}

// ----------------------------------------------------------------------
// Decision + status types.
// ----------------------------------------------------------------------

/// Reasons the driver decided NOT to fire. Surfaced in
/// `lu4d_auto_tick_status` so operators can debug a quiet driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTickSkipReason {
    /// `config.enabled == false`.
    Disabled,
    /// `pause()` was called and `resume()` has not.
    Paused,
    /// `idle_state.active_queries() != 0` -- a query is in flight.
    NotIdle,
    /// Idle, but for less than `config.idle_threshold_ms`.
    BelowIdleThreshold,
    /// `min_interval_ms` since the last driver-fired tick has not yet
    /// elapsed.
    BelowMinInterval,
    /// Sliding-window cap of `max_per_hour` already reached.
    MaxPerHour,
    /// Wonder queue is empty -- nothing for `idle_tick` to do.
    NoWonders,
}

impl AutoTickSkipReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            AutoTickSkipReason::Disabled => "disabled",
            AutoTickSkipReason::Paused => "paused",
            AutoTickSkipReason::NotIdle => "not_idle",
            AutoTickSkipReason::BelowIdleThreshold => "below_idle_threshold",
            AutoTickSkipReason::BelowMinInterval => "below_min_interval",
            AutoTickSkipReason::MaxPerHour => "max_per_hour",
            AutoTickSkipReason::NoWonders => "no_wonders",
        }
    }
}

/// Outcome of an `evaluate()` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTickDecision {
    /// All gates passed; the driver should fire `idle_tick` now.
    Fire,
    /// At least one gate blocked the fire; reason carries which.
    Skip(AutoTickSkipReason),
}

/// Coarse run-state surfaced by `lu4d_auto_tick_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTickRunState {
    /// `config.enabled == false`.
    Disabled,
    /// `pause()` was called.
    Paused,
    /// Eligible to fire (subject to gates). The "running" name in the
    /// spec is the same as "the driver loop is alive". A tick that's
    /// genuinely in flight on the main thread is observable as
    /// `idle_state.active_queries() > 0`.
    Running,
}

impl AutoTickRunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            AutoTickRunState::Disabled => "disabled",
            AutoTickRunState::Paused => "paused",
            AutoTickRunState::Running => "running",
        }
    }
}

/// Snapshot returned by `lu4d_auto_tick_status`.
#[derive(Debug, Clone)]
pub struct AutoTickStatus {
    /// Clock-source ms at the most recent driver fire, or `None` if no
    /// fire has happened in this process lifetime.
    pub last_tick_at_ms: Option<u64>,
    /// Driver fires within the last hour (sliding window).
    pub ticks_in_last_hour: u32,
    pub state: AutoTickRunState,
    /// Best-effort estimate of milliseconds until the next eligible
    /// fire, or `None` when not calculable (paused, disabled, or no
    /// wonder queue).
    pub next_tick_eta_ms: Option<u64>,
    pub config: AutoTickConfig,
    pub paused: bool,
}

// ----------------------------------------------------------------------
// Driver state + impl.
// ----------------------------------------------------------------------

/// Inner mutable state. Behind `Arc<RwLock<>>` on `AutoTickDriver`.
#[derive(Debug, Clone, Default)]
pub struct AutoTickStateInner {
    /// Manual pause. Distinct from `config.enabled` -- this is
    /// process-lifetime-only state intended for the
    /// `lu4d_auto_tick_pause` / `_resume` pair.
    pub paused: bool,
    /// Clock-source ms at the most recent successful driver fire.
    pub last_tick_at_ms: Option<u64>,
    /// Sliding-window deque of recent fire timestamps in clock ms.
    /// Trimmed lazily by `trim_window` whenever the deque is read or
    /// written.
    pub ticks_window: VecDeque<u64>,
}

/// The auto-tick driver. Cheap to clone (just `Arc`s).
///
/// This struct does NOT hold any `Brain` reference. The main MCP thread
/// holds `&mut Brain` and calls `evaluate(...)` to decide whether to
/// fire `idle_tick(brain, ...)`. The decision uses only the
/// `IdleState`'s lock-free counters and the driver's own state.
#[derive(Clone)]
pub struct AutoTickDriver {
    inner: Arc<RwLock<AutoTickStateInner>>,
    config: Arc<RwLock<AutoTickConfig>>,
    clock: Arc<dyn ClockSource>,
}

impl std::fmt::Debug for AutoTickDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoTickDriver")
            .field(
                "config",
                &self.config.read().map(|c| *c).unwrap_or_default(),
            )
            .field(
                "inner",
                &self
                    .inner
                    .read()
                    .map(|s| s.clone())
                    .unwrap_or_default(),
            )
            .finish_non_exhaustive()
    }
}

impl AutoTickDriver {
    /// Build a driver with the default config and the supplied clock
    /// source. Production code passes `brain.clock.clone()` so the
    /// driver shares the same authenticated source as the rest of the
    /// brain (T.1 invariant).
    pub fn new(clock: Arc<dyn ClockSource>) -> Self {
        AutoTickDriver {
            inner: Arc::new(RwLock::new(AutoTickStateInner::default())),
            config: Arc::new(RwLock::new(AutoTickConfig::default())),
            clock,
        }
    }

    /// Build with an explicit initial config. Useful in tests so the
    /// gates can be exercised at sub-second resolutions.
    pub fn with_config(clock: Arc<dyn ClockSource>, config: AutoTickConfig) -> Self {
        AutoTickDriver {
            inner: Arc::new(RwLock::new(AutoTickStateInner::default())),
            config: Arc::new(RwLock::new(config)),
            clock,
        }
    }

    // -- Config accessors / mutators (each one validated by `mcp_t6`
    //    before being called; this layer enforces no extra bounds). --

    pub fn config(&self) -> AutoTickConfig {
        self.config
            .read()
            .map(|c| *c)
            .unwrap_or_default()
    }

    pub fn set_enabled(&self, v: bool) {
        if let Ok(mut c) = self.config.write() {
            c.enabled = v;
        }
    }
    pub fn set_idle_threshold_ms(&self, v: u64) {
        if let Ok(mut c) = self.config.write() {
            c.idle_threshold_ms = v;
        }
    }
    pub fn set_min_interval_ms(&self, v: u64) {
        if let Ok(mut c) = self.config.write() {
            c.min_interval_ms = v;
        }
    }
    pub fn set_max_per_hour(&self, v: u32) {
        if let Ok(mut c) = self.config.write() {
            c.max_per_hour = v;
        }
    }

    // -- Pause / resume. Returns the prior state so the MCP wrapper
    //    can echo "paused: true -> false" to the operator. --

    pub fn pause(&self) -> bool {
        if let Ok(mut s) = self.inner.write() {
            let prev = s.paused;
            s.paused = true;
            prev
        } else {
            false
        }
    }

    pub fn resume(&self) -> bool {
        if let Ok(mut s) = self.inner.write() {
            let prev = s.paused;
            s.paused = false;
            prev
        } else {
            false
        }
    }

    pub fn is_paused(&self) -> bool {
        self.inner.read().map(|s| s.paused).unwrap_or(false)
    }

    pub fn last_tick_at_ms(&self) -> Option<u64> {
        self.inner.read().ok().and_then(|s| s.last_tick_at_ms)
    }

    pub fn ticks_in_last_hour(&self) -> u32 {
        let now = self.now_ms();
        let mut s = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return 0,
        };
        Self::trim_window(&mut s.ticks_window, now);
        s.ticks_window.len() as u32
    }

    /// Read the current monotonic-ms reading from the configured clock
    /// source. Falls back to the raw `MonotonicClock` only if the
    /// source itself errors (which `SystemClockSource` never does).
    pub fn now_ms(&self) -> u64 {
        match self.clock.fetch() {
            Ok(t) => (t.monotonic_at_fetch.as_nanos() / 1_000_000) as u64,
            Err(_) => (MonotonicClock::now().as_nanos() / 1_000_000) as u64,
        }
    }

    fn trim_window(window: &mut VecDeque<u64>, now_ms: u64) {
        while let Some(&front) = window.front() {
            if now_ms.saturating_sub(front) > ONE_HOUR_MS {
                window.pop_front();
            } else {
                break;
            }
        }
    }

    /// Pure decision logic. Order of gate evaluation matches the spec
    /// in `idle/auto_tick.rs` rustdoc:
    ///   1. `enabled`
    ///   2. `paused`
    ///   3. brain idle (`active_queries == 0`)
    ///   4. `idle_threshold_ms`
    ///   5. `min_interval_ms` since last fire
    ///   6. `max_per_hour` cap
    ///   7. `has_wonders`
    ///
    /// `has_wonders` is supplied by the caller (production reads
    /// `brain.wonder_queue.is_empty()`) so the driver doesn't have to
    /// hold a Brain reference.
    pub fn evaluate(&self, idle_state: &IdleState, has_wonders: bool) -> AutoTickDecision {
        let cfg = self.config();
        if !cfg.enabled {
            return AutoTickDecision::Skip(AutoTickSkipReason::Disabled);
        }
        if self.is_paused() {
            return AutoTickDecision::Skip(AutoTickSkipReason::Paused);
        }
        if idle_state.active_queries() != 0 {
            return AutoTickDecision::Skip(AutoTickSkipReason::NotIdle);
        }
        // Treat "no query has ever ended" as max-idle (u64::MAX),
        // i.e. the threshold gate trivially passes for a fresh brain.
        let idle_age = idle_state.last_query_age_ms().unwrap_or(u64::MAX);
        if idle_age < cfg.idle_threshold_ms {
            return AutoTickDecision::Skip(AutoTickSkipReason::BelowIdleThreshold);
        }
        let now = self.now_ms();
        // Min-interval gate.
        if let Some(last) = self.last_tick_at_ms() {
            let elapsed = now.saturating_sub(last);
            if elapsed < cfg.min_interval_ms {
                return AutoTickDecision::Skip(AutoTickSkipReason::BelowMinInterval);
            }
        }
        // Max-per-hour gate. Take a write lock so we can trim in place;
        // the `evaluate` path is the natural place to GC the window.
        if let Ok(mut s) = self.inner.write() {
            Self::trim_window(&mut s.ticks_window, now);
            if s.ticks_window.len() >= cfg.max_per_hour as usize {
                return AutoTickDecision::Skip(AutoTickSkipReason::MaxPerHour);
            }
        }
        if !has_wonders {
            return AutoTickDecision::Skip(AutoTickSkipReason::NoWonders);
        }
        AutoTickDecision::Fire
    }

    /// Mark a successful auto-driven tick. The MCP layer calls this
    /// AFTER `idle_tick(brain, ...)` returns -- not before -- so a
    /// failed tick does not consume a window slot.
    pub fn record_fired(&self) {
        let now = self.now_ms();
        if let Ok(mut s) = self.inner.write() {
            s.last_tick_at_ms = Some(now);
            s.ticks_window.push_back(now);
            Self::trim_window(&mut s.ticks_window, now);
        }
    }

    /// Build a snapshot for the `lu4d_auto_tick_status` MCP tool.
    /// `has_wonders` is the same input as for `evaluate` -- without it
    /// the ETA cannot distinguish "next fire in N ms" from "no fires
    /// ever, queue is empty".
    pub fn status(&self, idle_state: &IdleState, has_wonders: bool) -> AutoTickStatus {
        let cfg = self.config();
        let now = self.now_ms();
        let (last_tick_at_ms, ticks_in_last_hour, paused) = {
            let mut s = match self.inner.write() {
                Ok(g) => g,
                Err(_) => {
                    return AutoTickStatus {
                        last_tick_at_ms: None,
                        ticks_in_last_hour: 0,
                        state: AutoTickRunState::Disabled,
                        next_tick_eta_ms: None,
                        config: cfg,
                        paused: false,
                    };
                }
            };
            Self::trim_window(&mut s.ticks_window, now);
            (s.last_tick_at_ms, s.ticks_window.len() as u32, s.paused)
        };

        let state = if !cfg.enabled {
            AutoTickRunState::Disabled
        } else if paused {
            AutoTickRunState::Paused
        } else {
            AutoTickRunState::Running
        };

        // ETA: max of the two outstanding gate deltas (interval +
        // idle threshold). Reported only in `Running` state with a
        // non-empty wonder queue, since otherwise no fire is in
        // sight regardless of the math.
        let eta = if matches!(state, AutoTickRunState::Running) && has_wonders {
            let interval_eta = match last_tick_at_ms {
                Some(last) => cfg.min_interval_ms.saturating_sub(now.saturating_sub(last)),
                None => 0,
            };
            let idle_eta = match idle_state.last_query_age_ms() {
                Some(age) => cfg.idle_threshold_ms.saturating_sub(age),
                None => 0,
            };
            Some(interval_eta.max(idle_eta))
        } else {
            None
        };

        AutoTickStatus {
            last_tick_at_ms,
            ticks_in_last_hour,
            state,
            next_tick_eta_ms: eta,
            config: cfg,
            paused,
        }
    }
}

// ============================================================
//                              TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{AuthenticatedTime, ClockError, MonotonicTimestamp, SourceId};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Test ClockSource. `advance` / `set` mutate the wall-clock the
    /// driver reads; `monotonic_at_fetch` is derived deterministically
    /// from the wall-clock so `now_ms()` returns exactly that value.
    pub(super) struct TestClock {
        ms: AtomicU64,
    }

    impl TestClock {
        pub fn new(initial_ms: u64) -> Self {
            TestClock {
                ms: AtomicU64::new(initial_ms),
            }
        }
        pub fn advance(&self, delta_ms: u64) {
            self.ms.fetch_add(delta_ms, Ordering::SeqCst);
        }
        pub fn set(&self, ms: u64) {
            self.ms.store(ms, Ordering::SeqCst);
        }
    }

    impl ClockSource for TestClock {
        fn fetch(&self) -> Result<AuthenticatedTime, ClockError> {
            let now = self.ms.load(Ordering::SeqCst);
            Ok(AuthenticatedTime::unsigned(
                now,
                MonotonicTimestamp::from_nanos((now as u128) * 1_000_000),
                SourceId::system(),
                now + 60_000,
            ))
        }
        fn name(&self) -> &str {
            "test"
        }
        fn is_authenticated(&self) -> bool {
            false
        }
    }

    /// Build an `IdleState` whose `last_query_age_ms()` is greater than
    /// `min_age_ms` after returning. We can't inject a clock into
    /// `IdleState` (T.2 didn't make `MonotonicClock::now()` swappable),
    /// so this helper sleeps real time. Used only by gate tests where
    /// the threshold is sub-200ms.
    fn idle_state_aged(min_age_ms: u64) -> IdleState {
        let s = IdleState::new();
        s.on_query_start();
        s.on_query_end();
        if min_age_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(min_age_ms));
        }
        s
    }

    /// `IdleState` whose age gate is trivially satisfied (threshold 0
    /// in the surrounding driver config). No real-time sleep.
    fn idle_state_just_idle() -> IdleState {
        let s = IdleState::new();
        s.on_query_start();
        s.on_query_end();
        s
    }

    /// T.6: the threshold gate fires once `idle_threshold_ms` of real
    /// idle time has passed. Built on a 50 ms threshold + 100 ms sleep
    /// so the test stays fast and tolerant of CI scheduler jitter.
    #[test]
    fn t6_auto_tick_fires_after_idle_threshold() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock.clone() as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 50,
                min_interval_ms: 0,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_aged(100);
        match drv.evaluate(&idle, true) {
            AutoTickDecision::Fire => (),
            other => panic!("expected Fire after threshold, got {:?}", other),
        }
    }

    /// T.6: a query-in-flight pins `active_queries > 0`, so the gate
    /// short-circuits to `NotIdle` -- the driver never fires double
    /// against an in-flight tool call.
    #[test]
    fn t6_auto_tick_skips_when_query_in_flight() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::new(clock as Arc<dyn ClockSource>);
        let idle = IdleState::new();
        idle.on_query_start();
        let dec = drv.evaluate(&idle, true);
        assert_eq!(
            dec,
            AutoTickDecision::Skip(AutoTickSkipReason::NotIdle),
            "in-flight query must short-circuit to NotIdle, got {:?}",
            dec,
        );
    }

    /// T.6: after a fire, the next fire must wait at least
    /// `min_interval_ms` of clock-source time. Test clock advance
    /// proves the gate consults the configured source.
    #[test]
    fn t6_auto_tick_respects_min_interval() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock.clone() as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 0,
                min_interval_ms: 60_000,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_just_idle();
        // First fire is allowed.
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
        drv.record_fired();
        // Without advancing the clock, the next call must skip.
        assert_eq!(
            drv.evaluate(&idle, true),
            AutoTickDecision::Skip(AutoTickSkipReason::BelowMinInterval),
        );
        // Advance just past the interval; the gate clears.
        clock.advance(60_001);
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
    }

    /// T.6: the sliding-window cap blocks fires after `max_per_hour`
    /// within an hour, then clears as the window slides.
    #[test]
    fn t6_auto_tick_respects_max_per_hour_cap() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock.clone() as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 0,
                min_interval_ms: 0,
                max_per_hour: 3,
            },
        );
        let idle = idle_state_just_idle();
        for i in 0..3 {
            assert_eq!(
                drv.evaluate(&idle, true),
                AutoTickDecision::Fire,
                "fire {} of 3 must be allowed",
                i + 1,
            );
            drv.record_fired();
        }
        // Cap reached.
        assert_eq!(
            drv.evaluate(&idle, true),
            AutoTickDecision::Skip(AutoTickSkipReason::MaxPerHour),
        );
        // Slide the window forward by > 1 hour; cap clears.
        clock.advance(3_600_001);
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
    }

    /// T.6: an empty wonder queue means `idle_tick` would return
    /// `NoWorkNoSignals` -- driver short-circuits to NoWonders without
    /// burning a window slot.
    #[test]
    fn t6_auto_tick_skips_when_wonder_queue_empty() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 0,
                min_interval_ms: 0,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_just_idle();
        assert_eq!(
            drv.evaluate(&idle, false),
            AutoTickDecision::Skip(AutoTickSkipReason::NoWonders),
        );
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
    }

    /// T.6: pause / resume mutate the run-state. Returned bool is the
    /// PRIOR state, so MCP can echo `prior -> new` to the operator.
    #[test]
    fn t6_auto_tick_pause_resume_works() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 0,
                min_interval_ms: 0,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_just_idle();
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
        let prev = drv.pause();
        assert!(!prev, "prior pause state was unpaused");
        assert!(drv.is_paused());
        assert_eq!(
            drv.evaluate(&idle, true),
            AutoTickDecision::Skip(AutoTickSkipReason::Paused),
        );
        let prev = drv.resume();
        assert!(prev, "prior pause state was paused");
        assert!(!drv.is_paused());
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
    }

    /// T.6: `status()` reports the same state the gate logic uses --
    /// fires count, last-fire timestamp, run-state transitions across
    /// pause/disable.
    #[test]
    fn t6_auto_tick_status_reports_correctly() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock.clone() as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 30_000,
                min_interval_ms: 60_000,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_just_idle();
        let st = drv.status(&idle, true);
        assert_eq!(st.last_tick_at_ms, None);
        assert_eq!(st.ticks_in_last_hour, 0);
        assert_eq!(st.state, AutoTickRunState::Running);
        assert_eq!(st.config.idle_threshold_ms, 30_000);
        assert!(!st.paused);

        drv.record_fired();
        let st2 = drv.status(&idle, true);
        assert_eq!(st2.last_tick_at_ms, Some(1_000_000));
        assert_eq!(st2.ticks_in_last_hour, 1);
        assert_eq!(st2.state, AutoTickRunState::Running);

        drv.pause();
        let st3 = drv.status(&idle, true);
        assert_eq!(st3.state, AutoTickRunState::Paused);
        assert!(st3.paused);
        assert_eq!(st3.next_tick_eta_ms, None);

        drv.resume();
        drv.set_enabled(false);
        let st4 = drv.status(&idle, true);
        assert_eq!(st4.state, AutoTickRunState::Disabled);
    }

    /// T.6: time decisions flow through the configured `ClockSource`.
    /// Mutating `TestClock` (advancing OR retreating) is observable
    /// in the gate result -- proof the driver does not call
    /// `Instant::now()` / `SystemTime::now()` directly.
    #[test]
    fn t6_auto_tick_uses_clock_source_not_system_time() {
        let clock = Arc::new(TestClock::new(1_000_000));
        let drv = AutoTickDriver::with_config(
            clock.clone() as Arc<dyn ClockSource>,
            AutoTickConfig {
                enabled: true,
                idle_threshold_ms: 0,
                min_interval_ms: 60_000,
                max_per_hour: 60,
            },
        );
        let idle = idle_state_just_idle();
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);
        drv.record_fired();
        assert_eq!(drv.last_tick_at_ms(), Some(1_000_000));

        // Advance the TEST clock just past the interval. If the driver
        // were reading `Instant::now()` directly, no real time has
        // passed since `record_fired` returned -- the gate would still
        // fail. It clears here because the driver consults TestClock.
        clock.advance(60_001);
        assert_eq!(drv.evaluate(&idle, true), AutoTickDecision::Fire);

        // Retreat the test clock to BEFORE the last fire. SystemTime
        // can't go backward; only an injected source can. The driver
        // computes `now - last` with `saturating_sub` so the gate sees
        // 0 ms elapsed and returns BelowMinInterval.
        clock.set(500_000);
        assert_eq!(
            drv.evaluate(&idle, true),
            AutoTickDecision::Skip(AutoTickSkipReason::BelowMinInterval),
        );
    }
}
