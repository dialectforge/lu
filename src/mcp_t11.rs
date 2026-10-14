//! Port T.1.1 — MCP tool wiring.
//!
//! Exposes `lu4d_time_status`, which reports:
//!  - the current `time_source` mode (`"system"` | `"df_service"`)
//!  - for `df_service` mode: last successful fetch timestamp,
//!    drift on last fetch, consecutive-failure count, current trust
//!    state (`trusted` | `degraded` | `unreachable`)
//!  - for `system` mode: confirmation that the OS clock is in use,
//!    no remote dependency.

use crate::brain::Brain;
use serde_json::{json, Value};

/// Build the structured payload returned by `lu4d_time_status`. Pulled
/// out as a free fn so unit tests can drive it without going through
/// the full MCP dispatcher.
pub fn time_status_payload(brain: &mut Brain) -> Result<Value, String> {
    let cfg = brain.time_source_config.clone();
    let clock_name = brain.clock.name().to_string();
    let is_authenticated = brain.clock.is_authenticated();

    // For `system` mode the df-service-specific fields are reported
    // as null. For `df_service` mode we surface the polling state.
    let (
        trust_state,
        last_fetch_ms,
        last_drift_ms,
        consecutive_failures,
        last_anomaly,
        localhost_warning,
        successful_fetches,
        failed_fetches,
    ) = if cfg.is_df_service() {
        // Best-effort: pull the live state if the clock is the real
        // DF source. If it's anything else (e.g. user reverted to
        // system at runtime) we report Initializing.
        match brain.df_time_state_snapshot() {
            Some(s) => (
                Some(s.trust_state),
                s.last_fetch_ms,
                s.last_drift_ms,
                s.consecutive_failures,
                s.last_anomaly,
                s.localhost_warning,
                s.total_successful_fetches,
                s.total_failed_fetches,
            ),
            None => (
                Some(crate::time::TrustState::Initializing),
                None,
                None,
                0,
                None,
                false,
                0,
                0,
            ),
        }
    } else {
        (None, None, None, 0, None, false, 0, 0)
    };

    Ok(json!({
        "time_source": cfg.mode,
        "clock_name": clock_name,
        "clock_is_authenticated": is_authenticated,
        "time_service_url": cfg.url,
        "time_service_poll_interval_ms": cfg.poll_interval_ms,
        "time_service_max_drift_ms": cfg.max_drift_ms,
        "trust_state": trust_state,
        "last_fetch_ms": last_fetch_ms,
        "last_drift_ms": last_drift_ms,
        "consecutive_failures": consecutive_failures,
        "last_anomaly": last_anomaly,
        "localhost_warning": localhost_warning,
        "successful_fetches": successful_fetches,
        "failed_fetches": failed_fetches,
    }))
}

/// Wire-level MCP tool entry. Wraps `time_status_payload` in the
/// `(content, structuredContent)` envelope the rest of the MCP layer
/// uses.
pub(crate) fn tool_time_status(brain: &mut Brain) -> Result<Value, (i64, String)> {
    let payload = time_status_payload(brain).map_err(|e| (-32603, e))?;
    let mode = payload["time_source"].as_str().unwrap_or("unknown");
    let trust = payload["trust_state"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "n/a (system mode)".into());
    let warn = if payload["localhost_warning"].as_bool().unwrap_or(false) {
        " [WARNING: localhost URL]"
    } else {
        ""
    };
    let text = format!(
        "Time service:\n  mode: {}\n  clock: {}\n  trust: {}{}\n  successful_fetches: {}\n  failed_fetches: {}\n  consecutive_failures: {}",
        mode,
        payload["clock_name"].as_str().unwrap_or("?"),
        trust,
        warn,
        payload["successful_fetches"].as_u64().unwrap_or(0),
        payload["failed_fetches"].as_u64().unwrap_or(0),
        payload["consecutive_failures"].as_u64().unwrap_or(0),
    );
    Ok(crate::mcp_envelope::wrap_content(text, payload))
}
