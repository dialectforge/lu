// =====================================================================
// Port T.6 — Auto-tick driver MCP tools.
//
//   1. `lu4d_auto_tick_status` — reports last_tick_at_ms,
//        ticks_in_last_hour, run state (running / paused / disabled),
//        next_tick_eta_ms (when calculable), full config snapshot.
//   2. `lu4d_auto_tick_pause`  — sets paused=true. Returns prior state.
//   3. `lu4d_auto_tick_resume` — sets paused=false. Returns prior state.
//
// Companion to `Brain::auto_tick`. The driver itself ships in
// `crate::idle::auto_tick`. T.6 also extends the existing T.5
// `lu4d_configure` with four new auto_tick knobs (auto_tick_enabled,
// auto_tick_idle_threshold_ms, auto_tick_min_interval_ms,
// auto_tick_max_per_hour). Bounds enforcement lives in mcp_t5.rs so the
// configure surface stays single-call.
// =====================================================================

use crate::brain::Brain;
use crate::idle::AutoTickStatus;
use serde_json::{json, Value};

fn t6_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn status_to_json(st: &AutoTickStatus) -> Value {
    json!({
        "last_tick_at_ms": st.last_tick_at_ms,
        "ticks_in_last_hour": st.ticks_in_last_hour,
        "state": st.state.as_str(),
        "next_tick_eta_ms": st.next_tick_eta_ms,
        "paused": st.paused,
        "config": {
            "enabled": st.config.enabled,
            "idle_threshold_ms": st.config.idle_threshold_ms,
            "min_interval_ms": st.config.min_interval_ms,
            "max_per_hour": st.config.max_per_hour,
        },
    })
}

/// `lu4d_auto_tick_status` — read-only snapshot of the auto-tick driver.
pub(crate) fn tool_auto_tick_status(brain: &Brain) -> Result<Value, (i64, String)> {
    let has_wonders = !brain.wonder_queue.is_empty();
    let st = brain.auto_tick.status(&brain.idle_state, has_wonders);
    let text = format!(
        "auto_tick_status: state={} ticks_in_last_hour={} last_tick_at_ms={:?} \
         eta_ms={:?} (idle_threshold={} min_interval={} max_per_hour={} enabled={} paused={})",
        st.state.as_str(),
        st.ticks_in_last_hour,
        st.last_tick_at_ms,
        st.next_tick_eta_ms,
        st.config.idle_threshold_ms,
        st.config.min_interval_ms,
        st.config.max_per_hour,
        st.config.enabled,
        st.paused,
    );
    Ok(t6_wrap(text, status_to_json(&st)))
}

/// `lu4d_auto_tick_pause` — set paused=true. Returns the prior state so
/// the caller can confirm the transition without a separate status call.
pub(crate) fn tool_auto_tick_pause(brain: &Brain) -> Result<Value, (i64, String)> {
    let prior = brain.auto_tick.pause();
    let has_wonders = !brain.wonder_queue.is_empty();
    let st = brain.auto_tick.status(&brain.idle_state, has_wonders);
    let text = format!(
        "auto_tick_pause: prior_paused={} now_paused={} state={}",
        prior, st.paused, st.state.as_str(),
    );
    let mut payload = status_to_json(&st);
    payload["prior_paused"] = json!(prior);
    Ok(t6_wrap(text, payload))
}

/// `lu4d_auto_tick_resume` — set paused=false. Returns the prior state.
pub(crate) fn tool_auto_tick_resume(brain: &Brain) -> Result<Value, (i64, String)> {
    let prior = brain.auto_tick.resume();
    let has_wonders = !brain.wonder_queue.is_empty();
    let st = brain.auto_tick.status(&brain.idle_state, has_wonders);
    let text = format!(
        "auto_tick_resume: prior_paused={} now_paused={} state={}",
        prior, st.paused, st.state.as_str(),
    );
    let mut payload = status_to_json(&st);
    payload["prior_paused"] = json!(prior);
    Ok(t6_wrap(text, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: status round-trips through JSON without panicking and the
    /// shape matches the wrapper contract.
    #[test]
    fn t6_mcp_status_smoke() {
        let mut b = Brain::new();
        let r = tool_auto_tick_status(&b).expect("status ok");
        assert!(r.get("structuredContent").is_some());
        let sc = r.get("structuredContent").unwrap();
        assert!(sc.get("config").is_some());
        assert!(sc.get("ticks_in_last_hour").is_some());
        // pause -> resume cycle reflected in payload.
        let _ = tool_auto_tick_pause(&b).expect("pause ok");
        let r2 = tool_auto_tick_status(&b).unwrap();
        assert_eq!(
            r2["structuredContent"]["state"].as_str(),
            Some("paused"),
        );
        let _ = tool_auto_tick_resume(&b).expect("resume ok");
        let r3 = tool_auto_tick_status(&b).unwrap();
        assert_eq!(
            r3["structuredContent"]["state"].as_str(),
            Some("running"),
        );
        // Touch `b` to silence the unused-mut lint without changing
        // behaviour.
        let _ = &mut b;
    }
}
