// =====================================================================
// Port File-IO --- MCP tool wrappers for file read / write / extract.
//
//   1. `lu4d_file_read`              -- read a file, optionally route
//                                       its content into the brain.
//   2. `lu4d_file_write`             -- emit a file in a chosen format.
//   3. `lu4d_file_extract_to_pending`-- structured-data convenience
//                                       (CSV / JSON only) that pushes
//                                       extracted bonds onto the T.3
//                                       PendingLog with TrainingCorpus
//                                       attribution. Anti-bypass
//                                       structural invariant: the
//                                       function body must only mutate
//                                       brain state via
//                                       `brain.pending.push`. Enforced
//                                       by the grep guard test in this
//                                       file.
//
// Articulator-side: no brain schema changes. The brain stays a graph
// of bonds; files are a transport. Provenance flows through Port S
// source attribution (file:// URI in `source_url`) so existing audit
// + revoke paths cover file-sourced content with no extra plumbing.
// =====================================================================

use crate::brain::Brain;
use crate::bond::BondType;
use crate::file_io::{self, FileKind, ReadResult};
use serde_json::{json, Value};
use std::path::PathBuf;

fn fio_wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, (i64, String)> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| (-32602, format!("missing string arg `{}`", key)))
}

fn parse_kind(args: &Value) -> Option<FileKind> {
    args.get("kind")
        .and_then(|v| v.as_str())
        .and_then(FileKind::from_str_hint)
}

/// Map an arbitrary header / leaf-key string to a canonical `BondType`.
/// Defaults to `RelatedTo` when the string does not parse; the original
/// key is preserved in the pending change's `reason` field so audits
/// retain it.
fn relation_or_related_to(s: &str) -> BondType {
    match s {
        "IsA" | "DefinedAs" => BondType::IsA,
        "HasProperty" | "WorksBy" => BondType::HasProperty,
        "Causes" | "LeadsTo" => BondType::Causes,
        "PartOf" | "ComposedOf" => BondType::PartOf,
        "UsedFor" | "IsFor" => BondType::UsedFor,
        "RelatedTo" => BondType::RelatedTo,
        "ContrastedWith" | "Unlike" => BondType::ContrastedWith,
        "DerivedFrom" | "SuchAs" => BondType::DerivedFrom,
        "Requires" => BondType::Requires,
        "Produces" => BondType::Produces,
        "Opposes" => BondType::Opposes,
        _ => BondType::RelatedTo,
    }
}

// ---------------------------------------------------------------------
// `lu4d_file_read`
//
// Reads a file, parses it according to `kind` (or extension), and either
// returns the parsed payload (Inspect), pushes the content into LB as a
// UserStatement (UserStatement), or routes structured triples through
// `lu4d_teach_corpus` semantics (TeachCorpus).
//
// route shape:
//   { Inspect: {} }
//   { UserStatement: { source_label?: string, session_id?: integer } }
//   { TeachCorpus:   { corpus: string, run_id?: string } }
// ---------------------------------------------------------------------
pub fn tool_file_read(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let path_str = arg_str(&args, "path")?.to_string();
    let path = PathBuf::from(&path_str);
    let kind_hint = parse_kind(&args);
    let cfg = brain.file_io_config.clone();

    let read = file_io::read_file(&path, kind_hint, &cfg)
        .map_err(|e| (-32603, e))?;

    let route = args.get("route");
    let route_obj = route
        .and_then(|v| v.as_object())
        .ok_or((-32602, "missing object arg `route`".to_string()))?;

    if route_obj.contains_key("Inspect") || route_obj.contains_key("inspect") {
        return Ok(inspect_response(&path, &read));
    }

    if let Some(us) = route_obj
        .get("UserStatement")
        .or_else(|| route_obj.get("user_statement"))
    {
        let source_label = us
            .get("source_label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| Some(read.source_url_hint.clone()));
        let session_id = us.get("session_id").and_then(|v| v.as_u64());
        let text = render_for_user_statement(&read);
        let max_bytes = brain.tier2_prompt_config.max_text_bytes;
        if text.len() > max_bytes {
            return Err((
                -32602,
                format!(
                    "file_io: rendered text {} exceeds tier2_prompt_max_text_bytes ({})",
                    text.len(),
                    max_bytes,
                ),
            ));
        }
        let (sid, ephemeral) = match session_id {
            Some(s) => (s, false),
            None => match brain.tier2_prompt_config.default_session_strategy {
                crate::runtime_config::Tier2PromptSessionStrategy::Ephemeral => {
                    (brain.session_start(), true)
                }
                crate::runtime_config::Tier2PromptSessionStrategy::Reuse => {
                    let s = brain
                        .lb_session
                        .active_session()
                        .unwrap_or(crate::lb_session::SCRATCHPAD_SESSION_ID);
                    (s, false)
                }
            },
        };
        let subject = source_label
            .clone()
            .unwrap_or_else(|| "file_user_statement".to_string());
        let payload = json!({
            "text": text,
            "source_label": source_label,
            "source_url": read.source_url_hint,
            "byte_len": text.len(),
            "file_kind": read.kind.as_str(),
        })
        .to_string();
        let entry = crate::lb_session::SessionEntry::new(
            sid,
            crate::brain::now_ms() as u64,
            crate::lb_session::SessionEntryKind::UserStatement,
            subject,
            payload,
            0.0_f32,
        );
        let entry_id = brain
            .lb_session
            .push_internal(entry)
            .map_err(|e| (-32603, format!("lb_session push_internal: {}", e)))?;
        let text_msg = format!(
            "lu4d_file_read[UserStatement]: file={} kind={} entry_id={} session_id={} ephemeral={}",
            path.display(), read.kind.as_str(), entry_id, sid, ephemeral,
        );
        let structured = json!({
            "route": "UserStatement",
            "file": path_str,
            "file_kind": read.kind.as_str(),
            "byte_len": read.byte_len,
            "source_url": read.source_url_hint,
            "entry_id": entry_id,
            "session_id": sid,
            "ephemeral_session": ephemeral,
            "asymmetry_invariant": "no_bond_formation",
        });
        return Ok(fio_wrap(text_msg, structured));
    }

    if let Some(tc) = route_obj
        .get("TeachCorpus")
        .or_else(|| route_obj.get("teach_corpus"))
    {
        let corpus = tc
            .get("corpus")
            .and_then(|v| v.as_str())
            .ok_or((-32602, "route.TeachCorpus.corpus missing".to_string()))?
            .to_string();
        if corpus.trim().is_empty() {
            return Err((-32602, "route.TeachCorpus.corpus must be non-empty".to_string()));
        }
        let run_id = tc
            .get("run_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let triples = read_to_triples(&read)?;
        let mut bonds_formed = 0usize;
        let weight = 0.7f32; // conservative default for file-derived facts
        let conf = crate::node::Confidence::Inferred;
        let url = read.source_url_hint.clone();
        for (subject, relation_str, object) in &triples {
            let bt = relation_or_related_to(relation_str);
            let _r = brain.teach_corpus(
                subject,
                bt,
                object,
                weight,
                conf,
                &corpus,
                run_id.clone(),
                Some(url.clone()),
            );
            bonds_formed += 1;
        }
        let text_msg = format!(
            "lu4d_file_read[TeachCorpus]: file={} kind={} bonds_formed={} corpus={} url={}",
            path.display(),
            read.kind.as_str(),
            bonds_formed,
            corpus,
            url,
        );
        let structured = json!({
            "route": "TeachCorpus",
            "file": path_str,
            "file_kind": read.kind.as_str(),
            "byte_len": read.byte_len,
            "source_url": url,
            "corpus": corpus,
            "run_id": run_id,
            "bonds_formed": bonds_formed,
        });
        return Ok(fio_wrap(text_msg, structured));
    }

    Err((
        -32602,
        "route must be one of: { Inspect: {} } | { UserStatement: {...} } | { TeachCorpus: { corpus, ... } }"
            .to_string(),
    ))
}

fn inspect_response(path: &std::path::Path, r: &ReadResult) -> Value {
    let mut s = json!({
        "route": "Inspect",
        "file": path.to_string_lossy(),
        "file_kind": r.kind.as_str(),
        "byte_len": r.byte_len,
        "source_url": r.source_url_hint,
    });
    if let Some(t) = &r.text {
        s["text"] = Value::String(t.clone());
    }
    if let Some(rows) = &r.csv_rows {
        s["csv_rows"] = serde_json::to_value(rows).unwrap_or(Value::Null);
        s["csv_row_count"] = Value::Number(serde_json::Number::from(rows.len()));
    }
    if let Some(j) = &r.json_value {
        s["json_value"] = j.clone();
    }
    let head = match (&r.text, &r.csv_rows, &r.json_value) {
        (Some(t), _, _) => format!("{}", &t.chars().take(160).collect::<String>()),
        (_, Some(rows), _) => format!("{} rows", rows.len()),
        (_, _, Some(_)) => "json".to_string(),
        _ => "".to_string(),
    };
    let text_msg = format!(
        "lu4d_file_read[Inspect]: file={} kind={} byte_len={} preview={:?}",
        path.display(), r.kind.as_str(), r.byte_len, head,
    );
    fio_wrap(text_msg, s)
}

fn render_for_user_statement(r: &ReadResult) -> String {
    if let Some(t) = &r.text { return t.clone(); }
    if let Some(rows) = &r.csv_rows {
        return rows
            .iter()
            .map(|r| r.join("\t"))
            .collect::<Vec<_>>()
            .join("\n");
    }
    if let Some(j) = &r.json_value {
        return serde_json::to_string_pretty(j).unwrap_or_else(|_| j.to_string());
    }
    String::new()
}

/// Surface the structured triples available in the read result. Only
/// CSV (header-driven) and JSON (deep-flatten) yield triples; text
/// formats are rejected with a clear error pointing at UserStatement.
fn read_to_triples(r: &ReadResult) -> Result<Vec<(String, String, String)>, (i64, String)> {
    if let Some(rows) = &r.csv_rows {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let header = rows[0].clone();
        let mut out: Vec<(String, String, String)> = Vec::new();
        for row in rows.iter().skip(1) {
            if row.is_empty() { continue; }
            let subject = row[0].clone();
            for (i, cell) in row.iter().enumerate().skip(1) {
                if i >= header.len() { break; }
                let relation = header[i].clone();
                if cell.is_empty() { continue; }
                out.push((subject.clone(), relation, cell.clone()));
            }
        }
        return Ok(out);
    }
    if let Some(j) = &r.json_value {
        let mut triples: Vec<(String, String, String)> = Vec::new();
        file_io::flatten_json(j, "", "", &mut triples);
        return Ok(triples);
    }
    Err((
        -32602,
        format!(
            "file_kind {} is text-only and has no triple structure; use route=UserStatement instead",
            r.kind.as_str(),
        ),
    ))
}

// ---------------------------------------------------------------------
// `lu4d_file_write`
//
// Writes content to a file in the chosen kind. For PDF / DOCX the
// content is interpreted as plain text (markdown not parsed); the
// formatter renders verbatim with line breaks preserved. JSON content
// may be supplied either as a Value or as a JSON-encoded string.
// ---------------------------------------------------------------------
pub fn tool_file_write(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let path_str = arg_str(&args, "path")?.to_string();
    let kind_str = arg_str(&args, "kind")?.to_string();
    let kind = FileKind::from_str_hint(&kind_str)
        .ok_or((-32602, format!("unknown kind: {:?}", kind_str)))?;
    let path = PathBuf::from(&path_str);
    let cfg = brain.file_io_config.clone();

    match kind {
        FileKind::Md | FileKind::Txt | FileKind::Code => {
            let content = arg_str(&args, "content")?.to_string();
            file_io::write_text(&path, &content, &cfg).map_err(|e| (-32603, e))?;
        }
        FileKind::Csv => {
            let rows = csv_rows_from_args(&args)?;
            file_io::write_csv(&path, &rows, &cfg).map_err(|e| (-32603, e))?;
        }
        FileKind::Json => {
            let v = json_value_from_args(&args)?;
            file_io::write_json(&path, &v, &cfg).map_err(|e| (-32603, e))?;
        }
        FileKind::Pdf => {
            let content = arg_str(&args, "content")?.to_string();
            file_io::write_pdf(&path, &content, &cfg).map_err(|e| (-32603, e))?;
        }
        FileKind::Docx => {
            let content = arg_str(&args, "content")?.to_string();
            file_io::write_docx(&path, &content, &cfg).map_err(|e| (-32603, e))?;
        }
    }

    let md = std::fs::metadata(&path).map_err(|e| (-32603, format!("file_io: stat after write: {}", e)))?;
    let bytes = md.len();
    let canon = std::fs::canonicalize(&path).unwrap_or(path.clone());
    let text_msg = format!(
        "lu4d_file_write: kind={} path={} bytes={}",
        kind.as_str(),
        canon.display(),
        bytes,
    );
    let structured = json!({
        "file": canon.to_string_lossy(),
        "file_kind": kind.as_str(),
        "bytes": bytes,
    });
    let _ = brain; // brain unused in write path; reserved for future audit hooks.
    Ok(fio_wrap(text_msg, structured))
}

fn csv_rows_from_args(args: &Value) -> Result<Vec<Vec<String>>, (i64, String)> {
    let v = args
        .get("content")
        .ok_or((-32602, "missing arg `content` (csv expects [[string]])".to_string()))?;
    let arr = v
        .as_array()
        .ok_or((-32602, "csv `content` must be array of arrays of strings".to_string()))?;
    let mut out: Vec<Vec<String>> = Vec::new();
    for row in arr {
        let row_arr = row
            .as_array()
            .ok_or((-32602, "csv row must be an array of strings".to_string()))?;
        let r: Vec<String> = row_arr
            .iter()
            .map(|c| match c {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect();
        out.push(r);
    }
    Ok(out)
}

fn json_value_from_args(args: &Value) -> Result<Value, (i64, String)> {
    let v = args
        .get("content")
        .ok_or((-32602, "missing arg `content` (json expects value or json-encoded string)".to_string()))?;
    if let Value::String(s) = v {
        return serde_json::from_str(s)
            .map_err(|e| (-32602, format!("json content string did not parse: {}", e)));
    }
    Ok(v.clone())
}

// ---------------------------------------------------------------------
// `lu4d_file_extract_to_pending`
//
// Convenience wrapper: read + extract structured triples + queue
// `PendingChangeKind::AddBond` entries onto T.3 PendingLog stamped
// with `SourceKind::TrainingCorpus`. CSV / JSON only; text formats
// must use `lu4d_file_read { route: UserStatement }`.
//
// STRUCTURAL ANTI-MUTATION INVARIANT: the only allowed bond-creation
// path inside this function is `brain.pending.push`. Direct
// `brain.teach*` / `brain.network.add_bond` calls are forbidden and
// pinned by `fileio_grep_guard_extract_no_direct_mutation` below.
// ---------------------------------------------------------------------
pub fn tool_file_extract_to_pending(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let path_str = arg_str(&args, "path")?.to_string();
    let path = PathBuf::from(&path_str);
    let corpus = arg_str(&args, "corpus")?.to_string();
    if corpus.trim().is_empty() {
        return Err((-32602, "`corpus` must be a non-empty string".to_string()));
    }
    let max_bonds = args
        .get("max_bonds_per_file")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(10_000);
    let kind_hint = parse_kind(&args);
    let cfg = brain.file_io_config.clone();
    let read = file_io::read_file(&path, kind_hint, &cfg)
        .map_err(|e| (-32603, e))?;

    let triples = read_to_triples(&read)?;
    let url = read.source_url_hint.clone();
    let now = crate::brain::now_ms() as u64;
    let mut pending_ids: Vec<u64> = Vec::new();
    let mut emitted: usize = 0;
    let weight = 0.7f32;
    let conf_score = 0.65f32; // mirrors Confidence::Inferred numeric band.

    for (subject, relation_str, object) in triples.iter() {
        if emitted >= max_bonds { break; }
        let bt = relation_or_related_to(relation_str);
        let proposed_by = crate::source::SourceAttribution::training_with_url(
            corpus.clone(),
            Some(brain.ingest_run_id.clone()),
            Some(url.clone()),
        );
        let change = crate::pending::PendingChange {
            pending_id: 0,
            proposed_at_ms: now,
            proposed_by,
            change_kind: crate::pending::PendingChangeKind::AddBond {
                subject_text: subject.clone(),
                object_text: object.clone(),
                bond_type: bt,
                weight,
                negated: false,
            },
            confidence: conf_score,
            reason: format!(
                "File-IO extract: corpus={} kind={} url={} relation_label={:?}",
                corpus, read.kind.as_str(), url, relation_str,
            ),
            status: crate::pending::PendingStatus::Pending,
        };
        let pid = brain.pending.push(change).map_err(|e| (-32603, e.to_string()))?;
        pending_ids.push(pid);
        emitted += 1;
    }

    let text_msg = format!(
        "lu4d_file_extract_to_pending: file={} kind={} corpus={} bonds_queued={} max={}",
        path.display(),
        read.kind.as_str(),
        corpus,
        emitted,
        max_bonds,
    );
    let structured = json!({
        "file": path_str,
        "file_kind": read.kind.as_str(),
        "byte_len": read.byte_len,
        "source_url": url,
        "corpus": corpus,
        "run_id": brain.ingest_run_id.clone(),
        "bonds_queued": emitted,
        "max_bonds_per_file": max_bonds,
        "pending_ids": pending_ids,
        "routed_via": "T.3 PendingLog (brain.pending.push)",
    });
    Ok(fio_wrap(text_msg, structured))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn structured(v: &Value) -> &Value { v.get("structuredContent").unwrap() }

    fn permissive_brain() -> Brain {
        let mut b = Brain::new();
        b.file_io_config = crate::file_io::FileIoConfig {
            max_size_bytes: 1024 * 1024 * 1024,
            allowed_read_paths: Vec::new(),
            allowed_write_paths: Vec::new(),
        };
        b
    }

    #[test]
    fn fileio_mcp_inspect_md_roundtrip() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("r.md");
        std::fs::write(&p, "hello md").unwrap();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "Inspect": {} },
        })).unwrap();
        let s = structured(&resp);
        assert_eq!(s["file_kind"], "md");
        assert_eq!(s["text"], "hello md");
    }

    #[test]
    fn fileio_mcp_user_statement_text() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.txt");
        std::fs::write(&p, "some prompt text").unwrap();
        let before = b.network.bond_count();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "UserStatement": {} },
        })).unwrap();
        let s = structured(&resp);
        assert_eq!(s["asymmetry_invariant"], "no_bond_formation");
        assert_eq!(b.network.bond_count(), before, "UserStatement must NEVER form bonds");
    }

    #[test]
    fn fileio_mcp_teach_corpus_csv_forms_bonds() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.csv");
        std::fs::write(&p, "name,IsA,RelatedTo\nalpha,letter,bravo\n").unwrap();
        let before = b.network.bond_count();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "TeachCorpus": { "corpus": "test/file-io-csv" } },
        })).unwrap();
        let s = structured(&resp);
        assert!(s["bonds_formed"].as_u64().unwrap() >= 1);
        assert!(b.network.bond_count() > before);
    }

    #[test]
    fn fileio_mcp_extract_to_pending_csv_queues_bonds() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.csv");
        std::fs::write(&p, "name,IsA,RelatedTo\nfoo,thing,bar\n").unwrap();
        let bond_before = b.network.bond_count();
        let pending_before = b.pending.total_count();
        let resp = tool_file_extract_to_pending(&mut b, json!({
            "path": p.to_string_lossy(),
            "corpus": "test/file-io-extract",
        })).unwrap();
        let s = structured(&resp);
        let queued = s["bonds_queued"].as_u64().unwrap();
        assert!(queued >= 1, "expected >=1 queued, got {}", queued);
        assert_eq!(b.network.bond_count(), bond_before, "extract must NOT form bonds directly");
        assert!(b.pending.total_count() > pending_before);
    }

    #[test]
    fn fileio_mcp_write_md_then_inspect() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("out.md");
        let _ = tool_file_write(&mut b, json!({
            "path": p.to_string_lossy(),
            "kind": "md",
            "content": "hello world",
        })).unwrap();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "Inspect": {} },
        })).unwrap();
        assert_eq!(structured(&resp)["text"], "hello world");
    }

    #[test]
    fn fileio_mcp_write_pdf_round_trip() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("r.pdf");
        tool_file_write(&mut b, json!({
            "path": p.to_string_lossy(),
            "kind": "pdf",
            "content": "PDF check phrase ALPHA",
        })).unwrap();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "Inspect": {} },
        })).unwrap();
        let txt = structured(&resp)["text"].as_str().unwrap_or("").to_string();
        assert!(txt.contains("PDF check phrase"), "got: {:?}", txt);
    }

    #[test]
    fn fileio_mcp_write_docx_round_trip() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("r.docx");
        tool_file_write(&mut b, json!({
            "path": p.to_string_lossy(),
            "kind": "docx",
            "content": "docx round-trip line A\ndocx line B",
        })).unwrap();
        let resp = tool_file_read(&mut b, json!({
            "path": p.to_string_lossy(),
            "route": { "Inspect": {} },
        })).unwrap();
        let txt = structured(&resp)["text"].as_str().unwrap().to_string();
        assert!(txt.contains("line A"));
        assert!(txt.contains("line B"));
    }

    #[test]
    fn fileio_mcp_extract_to_pending_text_kind_rejected() {
        let mut b = permissive_brain();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("r.md");
        std::fs::write(&p, "some narrative text").unwrap();
        let err = tool_file_extract_to_pending(&mut b, json!({
            "path": p.to_string_lossy(),
            "corpus": "test/no-text",
        })).err().expect("text formats should be refused");
        assert!(err.1.contains("text-only"), "err={}", err.1);
    }


    /// Section #15 round-trips the new file_io knobs. Defends against
    /// future serde drift in the persisted snapshot. Mirror of
    /// tingest_runtime_config_section_15_round_trip.
    #[test]
    fn fileio_runtime_config_section_15_round_trip() {
        use crate::runtime_config::RuntimeConfigSnapshot;
        let mut snap = RuntimeConfigSnapshot::default();
        snap.file_io_max_size_bytes = 12_345_678;
        snap.file_io_allowed_read_paths = vec!["/srv/uploads".to_string()];
        snap.file_io_allowed_write_paths = vec!["/srv/outputs".to_string()];
        let bytes = snap.encode_v15_section().unwrap();
        let back = RuntimeConfigSnapshot::decode_v15_section(&bytes).unwrap();
        assert_eq!(back.file_io_max_size_bytes, 12_345_678);
        assert_eq!(back.file_io_allowed_read_paths, vec!["/srv/uploads".to_string()]);
        assert_eq!(back.file_io_allowed_write_paths, vec!["/srv/outputs".to_string()]);
        // Pre-File-IO section #15 (no file_io_* fields) decodes with defaults.
        let pre_fileio = b"{}";
        let defaults = RuntimeConfigSnapshot::decode_v15_section(pre_fileio).unwrap();
        assert_eq!(defaults.file_io_max_size_bytes, crate::file_io::default_file_io_max_size_bytes());
        assert_eq!(defaults.file_io_allowed_read_paths, crate::file_io::default_file_io_allowed_read_paths());
        assert_eq!(defaults.file_io_allowed_write_paths, crate::file_io::default_file_io_allowed_write_paths());
    }
    /// Anti-bypass STRUCTURAL grep guard. Asserts the
    /// `tool_file_extract_to_pending` function body uses ONLY
    /// `brain.pending.push` for state mutation and NEVER reaches for
    /// any teach* / network::add_bond bond-mutating API. FORBIDDEN
    /// identifiers are split via `concat!()` at compile time so the
    /// test source itself doesn't self-trigger the grep (mirrors
    /// T-Ingest / T-Ingest.2 grep guards).
    #[test]
    fn fileio_grep_guard_extract_no_direct_mutation() {
        let src = std::fs::read_to_string("src/mcp_file_io.rs")
            .expect("read src/mcp_file_io.rs");
        let start = src.find("fn tool_file_extract_to_pending(")
            .expect("tool_file_extract_to_pending exists");
        let after = &src[start..];
        let end_rel = after[1..].find("\nfn ").map(|i| i + 1).unwrap_or(after.len());
        let body = &after[..end_rel];

        // Sanctioned write path -- the only state mutation allowed.
        assert!(
            body.contains("brain.pending.push"),
            "tool_file_extract_to_pending body must use brain.pending.push",
        );

        // FORBIDDEN identifiers split at compile time to keep this
        // test source from self-triggering the grep.
        const FORBIDDEN: &[&str] = &[
            concat!("brain.", "teach("),
            concat!("brain.", "teach_corpus("),
            concat!("brain.", "teach_with_source("),
            concat!("brain.", "teach_with_kind("),
            concat!("brain.", "teach_with_negation("),
            concat!("brain.", "teach_from_text("),
            concat!("network.", "add_bond("),
            concat!("add_bond_", "with_scar_check("),
            concat!(".propose_to_", "pending("),
        ];
        for f in FORBIDDEN {
            assert!(
                !body.contains(f),
                "tool_file_extract_to_pending body must NOT reference bond-mutating API '{}': all writes must route through brain.pending.push (T.3 gate)",
                f,
            );
        }
    }
}
