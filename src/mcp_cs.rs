//! Port CS — MCP tool surface for the Conversational Style profile.
//!
//! Exposes:
//!   * `lu4d_style_status` — current persisted vector + decayed effective values.
//!   * `lu4d_style_reset`  — clear the persisted profile, back to defaults.
//!   * `lu4d_style_set`    — explicit override of one or more dimensions.
//!
//! The configure knobs (`style_learning_enabled`, `style_learning_rate`,
//! `style_decay_half_life_days`) are wired through `mcp_t5::tool_configure`
//! alongside the existing T.5 / T.6 knobs.

use serde_json::{json, Value};

use crate::brain::Brain;
use crate::style::StyleVector;

fn style_text(v: &StyleVector, label: &str) -> String {
    format!(
        "{} verbosity={:.3} formality={:.3} technicality={:.3} empathy={:.3} samples={} last_refined_at_ms={}",
        label, v.verbosity, v.formality, v.technicality, v.empathy,
        v.samples_count, v.last_refined_at_ms,
    )
}

fn style_json(v: &StyleVector) -> Value {
    json!({
        "verbosity": v.verbosity,
        "formality": v.formality,
        "technicality": v.technicality,
        "empathy": v.empathy,
        "samples_count": v.samples_count,
        "last_refined_at_ms": v.last_refined_at_ms,
    })
}

fn wrap_content(text: String, structured: Value) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": structured,
    })
}

fn now_ms_from(args: &Value) -> u64 {
    // Allow tests + demo to inject `now_ms`; otherwise fall back to
    // SystemTime. Mirrors T.5's mock-clock-via-arg pattern.
    if let Some(v) = args.get("now_ms").and_then(|v| v.as_u64()) {
        return v;
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn user_key_from(args: &Value, brain: &Brain) -> String {
    args.get("user_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| brain.style_config.user_key_str().to_string())
}

pub(crate) fn tool_style_status(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let now_ms = now_ms_from(&args);
    let half_life = brain.style_config.decay_half_life_days;
    let user_key = user_key_from(&args, brain);
    let persisted = brain.cs_style_store.get(&user_key);
    let decayed = brain.cs_style_store.get_decayed(&user_key, now_ms, half_life);
    let text = format!(
        "style_status user_key={} half_life_days={} now_ms={}\n  {}\n  {}",
        user_key, half_life, now_ms,
        style_text(&persisted, "persisted:"),
        style_text(&decayed,   "effective:"),
    );
    let hints = brain.compute_style_hints(now_ms);
    Ok(wrap_content(text, json!({
        "user_key": user_key,
        "half_life_days": half_life,
        "now_ms": now_ms,
        "persisted": style_json(&persisted),
        "effective": style_json(&decayed),
        "learning_enabled": brain.style_config.learning_enabled,
        "learning_rate": brain.style_config.learning_rate,
        "output_adaptation_enabled": brain.style_config.output_adaptation_enabled,
        "max_samples_for_full_weight": brain.style_config.max_samples_for_full_weight,
        "hints": json!({
            "max_response_length_factor": hints.max_response_length_factor,
            "formality": hints.formality,
            "technicality": hints.technicality,
            "empathy": hints.empathy,
            "samples_count": hints.samples_count,
            "is_neutral": hints.is_neutral(),
        }),
        "profile_count": brain.cs_style_store.profiles.len(),
    })))
}

pub(crate) fn tool_style_reset(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let user_key = user_key_from(&args, brain);
    let had_profile = brain.cs_style_store.profiles.contains_key(&user_key);
    brain.cs_style_store.reset(&user_key);
    let text = if had_profile {
        format!("style_reset user_key={}: cleared profile", user_key)
    } else {
        format!("style_reset user_key={}: no profile to clear (already default)", user_key)
    };
    Ok(wrap_content(text, json!({
        "user_key": user_key,
        "had_profile": had_profile,
        "current": style_json(&StyleVector::default()),
    })))
}

pub(crate) fn tool_style_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let user_key = user_key_from(&args, brain);
    let mut v = brain.cs_style_store.get(&user_key);
    let mut changes: Vec<String> = Vec::new();

    // Apply each provided dimension override individually.
    if let Some(val) = args.get("verbosity").and_then(|x| x.as_f64()) {
        if !(0.0..=1.0).contains(&val) {
            return Err((-32602, format!("verbosity out of bounds [0.0, 1.0]: {}", val)));
        }
        let prev = v.verbosity;
        v.verbosity = val as f32;
        changes.push(format!("verbosity {:.3} -> {:.3}", prev, val));
    }
    if let Some(val) = args.get("formality").and_then(|x| x.as_f64()) {
        if !(0.0..=1.0).contains(&val) {
            return Err((-32602, format!("formality out of bounds [0.0, 1.0]: {}", val)));
        }
        let prev = v.formality;
        v.formality = val as f32;
        changes.push(format!("formality {:.3} -> {:.3}", prev, val));
    }
    if let Some(val) = args.get("technicality").and_then(|x| x.as_f64()) {
        if !(0.0..=1.0).contains(&val) {
            return Err((-32602, format!("technicality out of bounds [0.0, 1.0]: {}", val)));
        }
        let prev = v.technicality;
        v.technicality = val as f32;
        changes.push(format!("technicality {:.3} -> {:.3}", prev, val));
    }
    if let Some(val) = args.get("empathy").and_then(|x| x.as_f64()) {
        if !(0.0..=1.0).contains(&val) {
            return Err((-32602, format!("empathy out of bounds [0.0, 1.0]: {}", val)));
        }
        let prev = v.empathy;
        v.empathy = val as f32;
        changes.push(format!("empathy {:.3} -> {:.3}", prev, val));
    }

    if changes.is_empty() {
        return Err((-32602,
            "style_set: no recognized dimension provided. Pass at least one of \
             verbosity / formality / technicality / empathy in [0.0, 1.0].".to_string()
        ));
    }
    // If this is the first time the profile is being touched, set
    // last_refined_at_ms so subsequent decay-on-read starts counting
    // from now.
    if v.last_refined_at_ms == 0 {
        v.last_refined_at_ms = now_ms_from(&args);
    }
    brain.cs_style_store.set(&user_key, v);
    let text = format!("style_set user_key={}: {}", user_key, changes.join("; "));
    Ok(wrap_content(text, json!({
        "user_key": user_key,
        "applied_changes": changes,
        "current": style_json(&v),
    })))
}
