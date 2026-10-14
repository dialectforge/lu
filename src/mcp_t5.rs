// =====================================================================
// Port T.5 — Freshness/decay MCP tools.
//
//   1. `lu4d_freshness_status` — bucket counts, mean/median age, top-N
//                                stalest bonds.
//   2. `lu4d_configure`        — runtime knobs (decay_half_life_days,
//                                decay_enabled). Bounds: 1..=3650 days.
//
// Companion to `Brain::decay_config`. Stored bond weights are NEVER
// mutated by these tools — decay is presentation-only.
// =====================================================================

use crate::brain::Brain;
use crate::decay::{DecayConfig, MAX_HALF_LIFE_DAYS, MIN_HALF_LIFE_DAYS};
use crate::idle::{
    AutoTickConfig, MAX_AUTO_TICK_INTERVAL_MS, MAX_AUTO_TICK_PER_HOUR,
    MIN_AUTO_TICK_INTERVAL_MS, MIN_AUTO_TICK_PER_HOUR,
};
use serde_json::{json, Value};

fn t5_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

/// `lu4d_freshness_status` — returns bucket counts + stalest-N.
pub(crate) fn tool_freshness_status(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    let st = brain.freshness_status(top_n);
    let stalest_json: Vec<Value> = st.stalest.iter().map(|s| json!({
        "subject_text": s.subject_text,
        "object_text": s.object_text,
        "bond_type": s.bond_type,
        "age_days": s.age_days,
        "stored_weight": s.stored_weight,
        "effective_weight_after_decay": s.effective_weight_after_decay,
    })).collect();
    let text = format!(
        "freshness_status: total_bonds={} mean={:.1}d median={:.1}d stalest_returned={} (decay_enabled={}, half_life_days={})",
        st.total_bonds, st.mean_age_days, st.median_age_days,
        st.stalest.len(), st.decay_enabled, st.half_life_days
    );
    Ok(t5_wrap(text, json!({
        "buckets": {
            "last_24h": st.bucket_24h,
            "last_7d":  st.bucket_7d,
            "last_30d": st.bucket_30d,
            "last_90d": st.bucket_90d,
            "last_180d": st.bucket_180d,
            "last_365d": st.bucket_365d,
            "older":    st.bucket_older,
        },
        "mean_age_days": st.mean_age_days,
        "median_age_days": st.median_age_days,
        "total_bonds": st.total_bonds,
        "stalest": stalest_json,
        "decay_enabled": st.decay_enabled,
        "half_life_days": st.half_life_days,
    })))
}

/// `lu4d_configure` — set runtime knobs. T.5 ships `decay_half_life_days`
/// (1-3650 days, default 90) and `decay_enabled` (bool, default true).
/// Pass with no args to read the current effective config.
pub(crate) fn tool_configure(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let mut changes: Vec<String> = Vec::new();
    if let Some(v) = args.get("decay_half_life_days").and_then(|v| v.as_u64()) {
        if v < MIN_HALF_LIFE_DAYS as u64 || v > MAX_HALF_LIFE_DAYS as u64 {
            return Err((-32602, format!(
                "decay_half_life_days out of bounds [{}, {}]: {}",
                MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS, v
            )));
        }
        let prev = brain.decay_config.half_life_days;
        brain.decay_config.set_half_life_days(v as u32);
        changes.push(format!(
            "decay_half_life_days {} -> {}", prev, brain.decay_config.half_life_days,
        ));
    }
    if let Some(v) = args.get("decay_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.decay_config.enabled;
        brain.decay_config.enabled = v;
        changes.push(format!("decay_enabled {} -> {}", prev, v));
    }

    // -------- Port T.6: auto_tick knobs --------
    if let Some(v) = args.get("auto_tick_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.auto_tick.config().enabled;
        brain.auto_tick.set_enabled(v);
        changes.push(format!("auto_tick_enabled {} -> {}", prev, v));
    }
    if let Some(v) = args.get("auto_tick_idle_threshold_ms").and_then(|v| v.as_u64()) {
        if v < MIN_AUTO_TICK_INTERVAL_MS || v > MAX_AUTO_TICK_INTERVAL_MS {
            return Err((-32602, format!(
                "auto_tick_idle_threshold_ms out of bounds [{}, {}]: {}",
                MIN_AUTO_TICK_INTERVAL_MS, MAX_AUTO_TICK_INTERVAL_MS, v
            )));
        }
        let prev = brain.auto_tick.config().idle_threshold_ms;
        brain.auto_tick.set_idle_threshold_ms(v);
        changes.push(format!("auto_tick_idle_threshold_ms {} -> {}", prev, v));
    }
    if let Some(v) = args.get("auto_tick_min_interval_ms").and_then(|v| v.as_u64()) {
        if v < MIN_AUTO_TICK_INTERVAL_MS || v > MAX_AUTO_TICK_INTERVAL_MS {
            return Err((-32602, format!(
                "auto_tick_min_interval_ms out of bounds [{}, {}]: {}",
                MIN_AUTO_TICK_INTERVAL_MS, MAX_AUTO_TICK_INTERVAL_MS, v
            )));
        }
        let prev = brain.auto_tick.config().min_interval_ms;
        brain.auto_tick.set_min_interval_ms(v);
        changes.push(format!("auto_tick_min_interval_ms {} -> {}", prev, v));
    }
    if let Some(v) = args.get("auto_tick_max_per_hour").and_then(|v| v.as_u64()) {
        let v32 = v as u32;
        if v32 < MIN_AUTO_TICK_PER_HOUR || v32 > MAX_AUTO_TICK_PER_HOUR {
            return Err((-32602, format!(
                "auto_tick_max_per_hour out of bounds [{}, {}]: {}",
                MIN_AUTO_TICK_PER_HOUR, MAX_AUTO_TICK_PER_HOUR, v
            )));
        }
        let prev = brain.auto_tick.config().max_per_hour;
        brain.auto_tick.set_max_per_hour(v32);
        changes.push(format!("auto_tick_max_per_hour {} -> {}", prev, v32));
    }

    // -------- Port CS: conversational-style knobs --------
    if let Some(v) = args.get("style_learning_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.style_config.learning_enabled;
        brain.style_config.learning_enabled = v;
        changes.push(format!("style_learning_enabled {} -> {}", prev, v));
    }
    if let Some(v) = args.get("style_learning_rate").and_then(|v| v.as_f64()) {
        match brain.style_config.set_learning_rate(v as f32) {
            Ok(prev) => changes.push(format!("style_learning_rate {} -> {}", prev, v)),
            Err(e) => return Err((-32602, e)),
        }
    }
    if let Some(v) = args.get("style_decay_half_life_days").and_then(|v| v.as_u64()) {
        match brain.style_config.set_decay_half_life_days(v as u32) {
            Ok(prev) => changes.push(format!("style_decay_half_life_days {} -> {}", prev, v)),
            Err(e) => return Err((-32602, e)),
        }
    }
    // -------- Port CS.2: output-adaptation knobs --------
    if let Some(v) = args.get("style_output_adaptation_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.style_config.output_adaptation_enabled;
        brain.style_config.output_adaptation_enabled = v;
        changes.push(format!("style_output_adaptation_enabled {} -> {}", prev, v));
    }
    if let Some(v) = args.get("style_max_samples_for_full_weight").and_then(|v| v.as_u64()) {
        match brain.style_config.set_max_samples_for_full_weight(v) {
            Ok(prev) => changes.push(format!("style_max_samples_for_full_weight {} -> {}", prev, v)),
            Err(e) => return Err((-32602, e)),
        }
    }

    // -------- Port T.1.1: time-source knobs --------
    let mut time_dirty = false;
    if let Some(v) = args.get("time_service_url").and_then(|v| v.as_str()) {
        let prev = brain.time_source_config.url.clone();
        brain.time_source_config.url = v.to_string();
        changes.push(format!("time_service_url {} -> {}", prev, v));
        time_dirty = true;
    }
    if let Some(v) = args.get("time_service_pubkey_hex").and_then(|v| v.as_str()) {
        if v.len() != 64 {
            return Err((-32602, format!(
                "time_service_pubkey_hex must be 64 hex chars (32 bytes), got {}", v.len()
            )));
        }
        let prev = brain.time_source_config.pubkey_hex.clone();
        brain.time_source_config.pubkey_hex = v.to_string();
        changes.push(format!("time_service_pubkey_hex {} -> {}", prev, v));
        time_dirty = true;
    }
    if let Some(v) = args.get("time_service_poll_interval_ms").and_then(|v| v.as_u64()) {
        if v < 1_000 || v > 24 * 3_600_000 {
            return Err((-32602, format!(
                "time_service_poll_interval_ms out of bounds [1000, {}]: {}",
                24 * 3_600_000_u64, v
            )));
        }
        let prev = brain.time_source_config.poll_interval_ms;
        brain.time_source_config.poll_interval_ms = v;
        changes.push(format!("time_service_poll_interval_ms {} -> {}", prev, v));
        time_dirty = true;
    }
    if let Some(v) = args.get("time_service_max_drift_ms").and_then(|v| v.as_u64()) {
        if v < 1 || v > 600_000 {
            return Err((-32602, format!(
                "time_service_max_drift_ms out of bounds [1, 600000]: {}", v
            )));
        }
        let prev = brain.time_source_config.max_drift_ms;
        brain.time_source_config.max_drift_ms = v;
        changes.push(format!("time_service_max_drift_ms {} -> {}", prev, v));
        time_dirty = true;
    }
    if let Some(v) = args.get("time_source").and_then(|v| v.as_str()) {
        if v != "system" && v != "df_service" {
            return Err((-32602, format!(
                "time_source must be 'system' or 'df_service', got: {}", v
            )));
        }
        let prev = brain.time_source_config.mode.clone();
        brain.time_source_config.mode = v.to_string();
        changes.push(format!("time_source {} -> {}", prev, v));
        time_dirty = true;
    }
    // Apply once at the end so url/pubkey/interval/drift updates land
    // BEFORE the polling thread spawns. Tearing down + re-spawning the
    // poll thread on every config tweak in df_service mode is a tiny
    // cost (~1 thread per `lu4d_configure` call) and keeps semantics
    // simple.
    if time_dirty {
        if let Err(e) = brain.apply_time_source_config() {
            eprintln!("[lu4d] T.1.1: apply_time_source_config: {}", e);
        }
    }

    // -------- Port Cat.3a: source-trust knobs --------
    let mut trust_changed = false;
    if let Some(v) = args.get("trust_default_multiplier").and_then(|v| v.as_f64()) {
        match brain.trust_table.set_default_multiplier(v as f32) {
            Ok(prev) => {
                changes.push(format!("trust_default_multiplier {:.3} -> {:.3}", prev, v));
                trust_changed = true;
            }
            Err(e) => return Err((-32602, e)),
        }
    }
    if let Some(v) = args.get("trust_learning_enabled").and_then(|v| v.as_bool()) {
        let prev = brain.trust_table.set_learning_enabled(v);
        changes.push(format!("trust_learning_enabled {} -> {}", prev, v));
        trust_changed = true;
    }
    if trust_changed {
    }

    // -------- Port Cat.3b: hub-degree caps --------
    let mut hub_changed = false;
    if let Some(v) = args.get("hub_max_in_degree").and_then(|v| v.as_u64()) {
        match brain.hub_degree_config.set_max_in_degree(v as u32) {
            Ok(prev) => {
                changes.push(format!("hub_max_in_degree {} -> {}", prev, v));
                hub_changed = true;
            }
            Err(e) => return Err((-32602, e)),
        }
    }
    if let Some(v) = args.get("hub_max_out_degree").and_then(|v| v.as_u64()) {
        match brain.hub_degree_config.set_max_out_degree(v as u32) {
            Ok(prev) => {
                changes.push(format!("hub_max_out_degree {} -> {}", prev, v));
                hub_changed = true;
            }
            Err(e) => return Err((-32602, e)),
        }
    }
    if let Some(v) = args.get("hub_write_enforcement").and_then(|v| v.as_bool()) {
        let prev = brain.hub_degree_config.set_write_enforcement(v);
        changes.push(format!("hub_write_enforcement {} -> {}", prev, v));
        hub_changed = true;
    }
    if let Some(v) = args.get("hub_read_enforcement").and_then(|v| v.as_bool()) {
        let prev = brain.hub_degree_config.set_read_enforcement(v);
        changes.push(format!("hub_read_enforcement {} -> {}", prev, v));
        hub_changed = true;
    }
    if hub_changed {
        // Sync the read-side knobs to the network so spread paths
        // immediately see the new cap.
        brain.resync_hub_read_config();
    }

    let cfg: DecayConfig = brain.decay_config;
    let text = if changes.is_empty() {
        let at = brain.auto_tick.config();
        format!(
            "configure (read-only): decay_half_life_days={} decay_enabled={}              auto_tick_enabled={} auto_tick_idle_threshold_ms={}              auto_tick_min_interval_ms={} auto_tick_max_per_hour={}",
            brain.decay_config.half_life_days,
            brain.decay_config.enabled,
            at.enabled,
            at.idle_threshold_ms,
            at.min_interval_ms,
            at.max_per_hour,
        )
    } else {
        format!("configure: {}", changes.join("; "))
    };
    let at: AutoTickConfig = brain.auto_tick.config();
    Ok(t5_wrap(text, json!({
        "decay_half_life_days": cfg.half_life_days,
        "decay_enabled": cfg.enabled,
        "auto_tick_enabled": at.enabled,
        "auto_tick_idle_threshold_ms": at.idle_threshold_ms,
        "auto_tick_min_interval_ms": at.min_interval_ms,
        "auto_tick_max_per_hour": at.max_per_hour,
        "style_learning_enabled": brain.style_config.learning_enabled,
        "style_learning_rate": brain.style_config.learning_rate,
        "style_decay_half_life_days": brain.style_config.decay_half_life_days,
        "style_output_adaptation_enabled": brain.style_config.output_adaptation_enabled,
        "style_max_samples_for_full_weight": brain.style_config.max_samples_for_full_weight,
        "trust_default_multiplier": brain.trust_table.default_multiplier,
        "trust_learning_enabled": brain.trust_table.learning_enabled,
        "hub_max_in_degree": brain.hub_degree_config.max_in_degree,
        "hub_max_out_degree": brain.hub_degree_config.max_out_degree,
        "hub_write_enforcement": brain.hub_degree_config.write_enforcement,
        "hub_read_enforcement": brain.hub_degree_config.read_enforcement,
        "applied_changes": changes,
        "bounds": {
            "min_half_life_days": MIN_HALF_LIFE_DAYS,
            "max_half_life_days": MAX_HALF_LIFE_DAYS,
            "min_auto_tick_interval_ms": MIN_AUTO_TICK_INTERVAL_MS,
            "max_auto_tick_interval_ms": MAX_AUTO_TICK_INTERVAL_MS,
            "min_auto_tick_per_hour": MIN_AUTO_TICK_PER_HOUR,
            "max_auto_tick_per_hour": MAX_AUTO_TICK_PER_HOUR,
        },
    })))
}
