//! Port Cat.3b — hub-degree-cap MCP tools.
//!
//!   1. `lu4d_hub_status`            — read-only: returns the
//!                                      HubDegreeConfig + top-N hubs
//!                                      by current degree.
//!   2. `lu4d_hub_set`               — admin-only: update one config
//!                                      field with bounds validation.
//!   3. `lu4d_hub_reset_to_defaults` — admin-only: restore defaults.
//!   4. `lu4d_hub_inspect`           — admin-only: degree info for a
//!                                      specific node (in/out counts,
//!                                      whether read-side filter is
//!                                      currently doing work, top
//!                                      neighbors by weight).
//!
//! Mutations save the JSON sidecar at `<brain>.hub_degree.json` (NOT
//! a v3 schema section). Existing bonds are NEVER mutated; the cap
//! applies to NEW bonds at formation time (write-side) and to the
//! neighbor list returned at query time (read-side). See
//! `crate::hub_degree` for the config + sidecar pattern.

use crate::brain::Brain;
use crate::hub_degree::{HubDegreeConfig, HubPriority, MAX_MAX_DEGREE, MIN_MAX_DEGREE};
use crate::seed::Seed;
use serde_json::{json, Value};

fn cat3b_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn config_to_json(c: &HubDegreeConfig) -> Value {
    json!({
        "max_in_degree": c.max_in_degree,
        "max_out_degree": c.max_out_degree,
        "write_enforcement": c.write_enforcement,
        "read_enforcement": c.read_enforcement,
        "priority": c.priority.as_str(),
        "exempt_node_kinds": c.exempt_node_kinds,
    })
}

/// `lu4d_hub_status` — read-only listing of the current HubDegreeConfig
/// plus the top-N hubs by current in_degree (default 20, max 200).
pub(crate) fn tool_hub_status(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(20)
        .min(200) as usize;

    // OP-BRAIN-V4 Phase 9: v4-aware hub_status. The v3 in-degree /
    // out-degree counters live on `brain.network`, which stays empty
    // for v4-loaded brains (bonds live in MemTable + SSTs). Until a
    // v4 hub-degree accountant exists, report an honest empty hub
    // list plus a clear note that the figures would be wrong if we
    // computed them off the v3 graph. The Phase 9 punch list tracks
    // closing this gap; the production cutover path does not depend
    // on hub_status for correctness.
    if brain.brain_dir_v4().is_some() {
        let cfg = config_to_json(&brain.hub_degree_config);
        let text = format!(
            "hub_status: max_in={} max_out={} write={} read={} priority={} top_hubs_returned=0 (v4 brain: hub-degree accountant not yet wired; see Phase 9 PortPlan)",
            brain.hub_degree_config.max_in_degree,
            brain.hub_degree_config.max_out_degree,
            brain.hub_degree_config.write_enforcement,
            brain.hub_degree_config.read_enforcement,
            brain.hub_degree_config.priority.as_str(),
        );
        return Ok(cat3b_wrap(
            text,
            json!({
                "config": cfg,
                "top_hubs": Value::Array(Vec::new()),
                "brain_format": "v4",
                "hub_degree_unwired": true,
            }),
        ));
    }

    let mut hubs: Vec<(Seed, usize, usize)> = brain
        .network
        .all_seeds()
        .into_iter()
        .map(|s| {
            let in_d = brain.network.seam_in_degree(&s);
            let out_d = brain.network.seam_out_degree(&s);
            (s, in_d, out_d)
        })
        .collect();
    hubs.sort_by(|a, b| b.1.cmp(&a.1));
    hubs.truncate(top_n);

    let top_hubs: Vec<Value> = hubs
        .iter()
        .map(|(s, in_d, out_d)| {
            let text = brain
                .network
                .get_node(s)
                .map(|n| n.content().to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            json!({
                "node_text": text,
                "in_degree": in_d,
                "out_degree": out_d,
                "exceeds_in_cap": *in_d as u32 > brain.hub_degree_config.max_in_degree,
                "exceeds_out_cap": *out_d as u32 > brain.hub_degree_config.max_out_degree,
            })
        })
        .collect();

    let cfg = config_to_json(&brain.hub_degree_config);
    let text = format!(
        "hub_status: max_in={} max_out={} write={} read={} priority={} top_hubs_returned={}",
        brain.hub_degree_config.max_in_degree,
        brain.hub_degree_config.max_out_degree,
        brain.hub_degree_config.write_enforcement,
        brain.hub_degree_config.read_enforcement,
        brain.hub_degree_config.priority.as_str(),
        top_hubs.len(),
    );
    Ok(cat3b_wrap(
        text,
        json!({
            "config": cfg,
            "top_hubs": top_hubs,
            "counters": {
                "hub_cap_rejections_total": brain.hub_cap_rejections_total,
                "hub_cap_evictions_total": brain.hub_cap_evictions_total,
                "hub_cap_filtered_at_query_total": brain.hub_cap_filtered_at_query_total(),
            },
            "bounds": {
                "min_max_degree": MIN_MAX_DEGREE,
                "max_max_degree": MAX_MAX_DEGREE,
            },
        }),
    ))
}

/// `lu4d_hub_set` — update one config field.
/// Required: `key` (string), `value` (number/bool/string depending on key).
/// Keys: max_in_degree, max_out_degree, write_enforcement,
/// read_enforcement, priority, exempt_node_kinds.
pub(crate) fn tool_hub_set(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let key = args
        .get("key")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'key' string".to_string()))?
        .to_string();
    let value = args
        .get("value")
        .ok_or((-32602, "missing required 'value' field".to_string()))?
        .clone();

    let applied: String;
    match key.as_str() {
        "max_in_degree" => {
            let v = value
                .as_u64()
                .ok_or((-32602, "max_in_degree must be a non-negative integer".to_string()))?
                as u32;
            let prev = brain
                .hub_degree_config
                .set_max_in_degree(v)
                .map_err(|e| (-32602, e))?;
            applied = format!("max_in_degree {} -> {}", prev, v);
        }
        "max_out_degree" => {
            let v = value
                .as_u64()
                .ok_or((-32602, "max_out_degree must be a non-negative integer".to_string()))?
                as u32;
            let prev = brain
                .hub_degree_config
                .set_max_out_degree(v)
                .map_err(|e| (-32602, e))?;
            applied = format!("max_out_degree {} -> {}", prev, v);
        }
        "write_enforcement" => {
            let v = value
                .as_bool()
                .ok_or((-32602, "write_enforcement must be a boolean".to_string()))?;
            let prev = brain.hub_degree_config.set_write_enforcement(v);
            applied = format!("write_enforcement {} -> {}", prev, v);
        }
        "read_enforcement" => {
            let v = value
                .as_bool()
                .ok_or((-32602, "read_enforcement must be a boolean".to_string()))?;
            let prev = brain.hub_degree_config.set_read_enforcement(v);
            applied = format!("read_enforcement {} -> {}", prev, v);
        }
        "priority" => {
            let s = value
                .as_str()
                .ok_or((-32602, "priority must be a string".to_string()))?;
            let p = HubPriority::parse(s).ok_or((
                -32602,
                format!(
                    "priority must be one of NewestFirst, OldestFirst, LowestWeightFirst (got {})",
                    s
                ),
            ))?;
            let prev = brain.hub_degree_config.set_priority(p);
            applied = format!("priority {} -> {}", prev.as_str(), p.as_str());
        }
        "exempt_node_kinds" => {
            let arr = value
                .as_array()
                .ok_or((-32602, "exempt_node_kinds must be an array of strings".to_string()))?;
            let mut kinds: Vec<String> = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or((
                    -32602,
                    "exempt_node_kinds entries must be strings".to_string(),
                ))?;
                kinds.push(s.to_string());
            }
            let prev = brain.hub_degree_config.exempt_node_kinds.clone();
            brain.hub_degree_config.exempt_node_kinds = kinds.clone();
            applied = format!("exempt_node_kinds {:?} -> {:?}", prev, kinds);
        }
        _ => {
            return Err((
                -32602,
                format!("unknown key '{}' (valid: max_in_degree, max_out_degree, write_enforcement, read_enforcement, priority, exempt_node_kinds)", key),
            ));
        }
    }

    let text = format!("hub_set: {}", applied);
    Ok(cat3b_wrap(
        text,
        json!({
            "applied": applied,
            "config": config_to_json(&brain.hub_degree_config),
        }),
    ))
}

/// `lu4d_hub_reset_to_defaults` — restore defaults. Idempotent.
pub(crate) fn tool_hub_reset_to_defaults(
    brain: &mut Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    brain.hub_degree_config.reset_to_defaults();
    let text = format!(
        "hub_reset_to_defaults: max_in={} max_out={} write={} read={} priority={}",
        brain.hub_degree_config.max_in_degree,
        brain.hub_degree_config.max_out_degree,
        brain.hub_degree_config.write_enforcement,
        brain.hub_degree_config.read_enforcement,
        brain.hub_degree_config.priority.as_str(),
    );
    Ok(cat3b_wrap(
        text,
        json!({
            "config": config_to_json(&brain.hub_degree_config),
        }),
    ))
}

/// `lu4d_hub_inspect` — degree info for a specific node by text.
/// Returns in/out counts, whether the node would be filtered at
/// query time (i.e. has more than `max_in_degree` bonds), and the
/// top-N outgoing bonds by latest weight.
pub(crate) fn tool_hub_inspect(
    brain: &mut Brain,
    args: Value,
) -> Result<Value, (i64, String)> {
    let node_text = args
        .get("node_text")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'node_text' string".to_string()))?
        .to_string();
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(200) as usize;

    let seed = Seed::from_label(&node_text);
    if brain.network.get_node(&seed).is_none() {
        return Err((
            -32602,
            format!("no node found for text '{}'", node_text),
        ));
    }
    let in_deg = brain.network.seam_in_degree(&seed);
    let out_deg = brain.network.seam_out_degree(&seed);
    let cfg = &brain.hub_degree_config;
    let exceeds_in = in_deg as u32 > cfg.max_in_degree;
    let exceeds_out = out_deg as u32 > cfg.max_out_degree;

    // Surface the top-N outgoing bonds by latest raw weight. This
    // mirrors what the read-side filter would keep when the cap is
    // applied. Reads via get_bond so archive-resident bonds are seen.
    let neighbors: Vec<Seed> = brain.network.seam_neighbors_outgoing(&seed).to_vec();
    let mut weighted: Vec<(Seed, f32)> = Vec::with_capacity(neighbors.len());
    for n in &neighbors {
        // OP-BRAIN-MapRAM P5-b2: weight via the seam (carried payload in
        // cache mode), off resident get_bond -- the last serving-callable
        // resident bond read the final audit found.
        if let Some(b) = brain.network.seam_directed_bond(&seed, n) {
            weighted.push((*n, b.view().latest_raw_weight()));
        }
    }
    weighted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    weighted.truncate(top_n);
    let top_bonds: Vec<Value> = weighted
        .iter()
        .map(|(s, w)| {
            let text = brain
                .network
                .get_node(s)
                .map(|n| n.content().to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            json!({
                "neighbor_text": text,
                "latest_weight": *w,
            })
        })
        .collect();

    let text = format!(
        "hub_inspect '{}': in_degree={} out_degree={} exceeds_in_cap={} exceeds_out_cap={}",
        node_text, in_deg, out_deg, exceeds_in, exceeds_out,
    );
    Ok(cat3b_wrap(
        text,
        json!({
            "node_text": node_text,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "exceeds_in_cap": exceeds_in,
            "exceeds_out_cap": exceeds_out,
            "max_in_degree": cfg.max_in_degree,
            "max_out_degree": cfg.max_out_degree,
            "read_enforcement": cfg.read_enforcement,
            "would_be_filtered_at_query": cfg.read_enforcement && exceeds_in,
            "top_outgoing_bonds": top_bonds,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::node::Confidence;

    fn fresh_brain() -> Brain {
        Brain::new()
    }

    #[test]
    fn cat3b_tool_hub_status_returns_default_config() {
        let mut b = fresh_brain();
        let v = tool_hub_status(&mut b, json!({})).unwrap();
        let cfg = &v["structuredContent"]["config"];
        assert_eq!(cfg["max_in_degree"].as_u64().unwrap(), 5000);
        assert_eq!(cfg["max_out_degree"].as_u64().unwrap(), 5000);
        assert_eq!(cfg["write_enforcement"].as_bool().unwrap(), true);
        assert_eq!(cfg["read_enforcement"].as_bool().unwrap(), true);
        assert_eq!(cfg["priority"].as_str().unwrap(), "LowestWeightFirst");
    }

    #[test]
    fn cat3b_tool_hub_status_returns_top_n_hubs_by_degree() {
        let mut b = fresh_brain();
        // Build a small star: "hub" with 3 outgoing bonds.
        for o in &["a", "b", "c"] {
            b.teach("hub", BondType::RelatedTo, o, 0.5, Confidence::Verified);
        }
        let v = tool_hub_status(&mut b, json!({"top_n": 5})).unwrap();
        let hubs = v["structuredContent"]["top_hubs"].as_array().unwrap();
        // hub has out_degree=3; the leaves all have in_degree=1.
        // The top entry by in_degree is the one with the highest in.
        // In this star, the leaves have in=1; hub has in=0. Sort is by in_degree desc.
        // So the top entry is one of the leaves.
        assert!(!hubs.is_empty());
    }

    #[test]
    fn cat3b_tool_hub_set_max_in_degree_rejects_out_of_bounds() {
        let mut b = fresh_brain();
        let r = tool_hub_set(
            &mut b,
            json!({"key":"max_in_degree","value":50}),
        );
        assert!(r.is_err());
        let r2 = tool_hub_set(
            &mut b,
            json!({"key":"max_in_degree","value":2_000_000u64}),
        );
        assert!(r2.is_err());
    }

    #[test]
    fn cat3b_tool_hub_set_priority_rejects_unknown_string() {
        let mut b = fresh_brain();
        let r = tool_hub_set(
            &mut b,
            json!({"key":"priority","value":"InvalidPolicy"}),
        );
        assert!(r.is_err());
    }

    #[test]
    fn cat3b_tool_hub_set_unknown_key_returns_error() {
        let mut b = fresh_brain();
        let r = tool_hub_set(
            &mut b,
            json!({"key":"some_other_key","value":42}),
        );
        assert!(r.is_err());
    }

    #[test]
    fn cat3b_tool_hub_set_then_reset_round_trip() {
        let mut b = fresh_brain();
        let _ = tool_hub_set(
            &mut b,
            json!({"key":"max_in_degree","value":1234}),
        )
        .unwrap();
        assert_eq!(b.hub_degree_config.max_in_degree, 1234);
        let _ = tool_hub_reset_to_defaults(&mut b, json!({})).unwrap();
        assert_eq!(b.hub_degree_config.max_in_degree, 5000);
    }

    #[test]
    fn cat3b_tool_hub_inspect_returns_degree_info_for_specific_node() {
        let mut b = fresh_brain();
        for o in &["alpha", "beta", "gamma"] {
            b.teach("hub", BondType::RelatedTo, o, 0.5, Confidence::Verified);
        }
        let v = tool_hub_inspect(&mut b, json!({"node_text":"hub","top_n":5})).unwrap();
        let s = &v["structuredContent"];
        assert_eq!(s["node_text"].as_str().unwrap(), "hub");
        assert_eq!(s["out_degree"].as_u64().unwrap(), 3);
        assert_eq!(s["exceeds_out_cap"].as_bool().unwrap(), false);
        let bonds = s["top_outgoing_bonds"].as_array().unwrap();
        assert_eq!(bonds.len(), 3);
    }

    #[test]
    fn cat3b_tool_hub_inspect_unknown_node_returns_error() {
        let mut b = fresh_brain();
        let r = tool_hub_inspect(&mut b, json!({"node_text":"does_not_exist"}));
        assert!(r.is_err());
    }
}
