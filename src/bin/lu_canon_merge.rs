//! OP-CANON-1 -- offline canonicalisation + merge tool (lu4d).
//!
//! Subcommands:
//!   classify   Read the read-only NSST sampler JSON on stdin
//!              (`{ "samples": [ { "node_id": u64, "content": String } ] }`)
//!              and emit ONE JSON line per node:
//!              `{ "node_id", "reject": bool, "key": String|null, "content" }`.
//!              Uses the REAL `lu4d::canonical::canonical_key`, so the
//!              validation harness measures the exact reject/normalise logic
//!              the ingest wiring and the (gated) merge/prune will use --
//!              no Python re-implementation that could drift.
//!
//!   plan       Read a graph JSON on stdin
//!              (`{ "nodes":[{"node_id":u64,"content":String}],
//!                 "bonds":[{"from":u64,"to":u64,"kind":u16,"weight":f32}] }`;
//!              `id`/`label` and `samples` are accepted as aliases) and run
//!              the REAL `lu4d::canon_merge` fold planner + bond redirect.
//!              Prints a fold/redirect summary to stderr and, if bonds are
//!              present, the rewritten bond set as JSONL to stdout.
//!
//! READ-ONLY / write-new only. Neither subcommand opens or mutates a brain
//! store; they transform stdin JSON. The real NodeStore streaming
//! (write-new) execution against the 253 GB brain is the backup-gated
//! Phase-B step and is intentionally NOT part of this tool yet.

use lu4d::canon_merge::{plan_fold, redirect_bonds, Bond};
use lu4d::canonical::{canonical_ingest_key, canonical_key};
use serde_json::Value;
use std::io::{self, BufWriter, Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()).unwrap_or("") {
        "classify" => classify(),
        "plan" => plan(),
        "foldkey" => foldkey(),
        other => {
            eprintln!(
                "lu_canon_merge: unknown subcommand {:?}; expected one of: classify, plan, foldkey",
                other
            );
            std::process::exit(2);
        }
    }
}

fn read_stdin_json() -> Value {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).expect("read stdin");
    serde_json::from_str(&buf).expect("parse json")
}

fn node_content(o: &Value) -> &str {
    o.get("content")
        .or_else(|| o.get("label"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
}

fn node_id(o: &Value) -> u64 {
    o.get("node_id")
        .or_else(|| o.get("id"))
        .and_then(|c| c.as_u64())
        .unwrap_or(0)
}

/// Stream the sampler JSON through the real `canonical_key` classifier.
fn classify() {
    let v = read_stdin_json();
    let samples = v
        .get("samples")
        .or_else(|| v.get("nodes"))
        .and_then(|s| s.as_array())
        .expect("json missing 'samples'/'nodes' array");

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let (mut total, mut rejected) = (0u64, 0u64);
    for s in samples {
        let content = node_content(s);
        let key = canonical_key(content);
        let reject = key.is_none();
        total += 1;
        if reject {
            rejected += 1;
        }
        let rec = serde_json::json!({
            "node_id": s.get("node_id").or_else(|| s.get("id")).cloned().unwrap_or(Value::Null),
            "reject": reject,
            "key": key,
            "content": content,
        });
        writeln!(out, "{}", rec).unwrap();
    }
    out.flush().unwrap();
    eprintln!(
        "classify: total={} rejected(would-kill)={} kept={} reject_rate={:.4}",
        total,
        rejected,
        total - rejected,
        if total > 0 { rejected as f64 / total as f64 } else { 0.0 }
    );
}

/// Run the real fold planner + bond redirect over a graph JSON.
fn plan() {
    let v = read_stdin_json();
    let node_arr = v
        .get("nodes")
        .or_else(|| v.get("samples"))
        .and_then(|s| s.as_array())
        .expect("json missing 'nodes'/'samples' array");
    let nodes: Vec<(u64, String)> = node_arr
        .iter()
        .map(|o| (node_id(o), node_content(o).to_string()))
        .collect();

    let bonds: Vec<Bond> = v
        .get("bonds")
        .and_then(|b| b.as_array())
        .map(|arr| {
            arr.iter()
                .map(|o| Bond {
                    from: o.get("from").and_then(|x| x.as_u64()).unwrap_or(0),
                    to: o.get("to").and_then(|x| x.as_u64()).unwrap_or(0),
                    kind: o.get("kind").and_then(|x| x.as_u64()).unwrap_or(0) as u16,
                    weight: o.get("weight").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
                })
                .collect()
        })
        .unwrap_or_default();

    let fp = plan_fold(&nodes);
    let n_nodes = nodes.len();
    let n_targets = fp.target_of_key.len();
    let n_redirect = fp.redirect.len();
    let n_orphan = fp.orphan.len();
    let survivors = n_nodes.saturating_sub(fp.retired_count());

    if !bonds.is_empty() {
        let out_bonds = redirect_bonds(&bonds, &fp);
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        for b in &out_bonds {
            writeln!(
                out,
                "{}",
                serde_json::json!({"from": b.from, "to": b.to, "kind": b.kind, "weight": b.weight})
            )
            .unwrap();
        }
        out.flush().unwrap();
        eprintln!(
            "plan: nodes_in={} survivors={} fold_targets={} redirected={} orphaned={} | bonds_in={} bonds_out={}",
            n_nodes, survivors, n_targets, n_redirect, n_orphan, bonds.len(), out_bonds.len()
        );
    } else {
        eprintln!(
            "plan: nodes_in={} survivors={} fold_targets={} redirected={} orphaned={} | (no bonds supplied)",
            n_nodes, survivors, n_targets, n_redirect, n_orphan
        );
    }
}

/// Read `node_id<TAB>content` lines on stdin (content pre-sanitised: no
/// embedded tabs/newlines) and emit `fold_key<TAB>reject<TAB>content` using
/// the REAL canonical logic: fold_key = canonical_ingest_key (the merge
/// grouping key), reject = canonical_key(..).is_none() (the prune flag).
/// Empty fold keys are emitted under the sentinel `__EMPTY__`. Read-only.
fn foldkey() {
    use std::io::BufRead;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let mut it = line.splitn(2, char::from(9));
        let _id = it.next().unwrap_or("");
        let content = it.next().unwrap_or("");
        let fk = canonical_ingest_key(content).unwrap_or_else(|| "__EMPTY__".to_string());
        let reject = if canonical_key(content).is_none() { 1 } else { 0 };
        writeln!(out, "{}\t{}\t{}", fk, reject, content).unwrap();
    }
    out.flush().unwrap();
}
