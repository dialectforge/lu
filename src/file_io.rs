//! Port File-IO — file ingestion and emission across common formats.
//!
//! This module is articulator-side: the brain does NOT gain a "file
//! bond type". Files are a transport. Their content flows through the
//! existing teach / user_statement ingest paths (Right Brain bonds for
//! fact-shaped content; Left Brain session entries for prompt-shaped
//! content) and out through narrate / structured emit paths.
//!
//! Provenance is carried by Port S source attribution: the source_url
//! field stamps a `file://<path>` URI into the bond / entry so audit
//! trails (lu4d_revoke_by_source) work without any new schema.
//!
//! Supported formats:
//!
//!    READ    md / markdown   — UTF-8, returned verbatim
//!            txt             — UTF-8, returned verbatim
//!            csv             — pure-Rust `csv` crate, header-as-row-0
//!            json            — serde_json::Value
//!            pdf             — `pdf-extract` crate (text only; no images)
//!            docx            — minimal hand-rolled <w:t> extractor over
//!                              `word/document.xml` from the zip archive
//!            code            — UTF-8 fall-through for source-code files
//!
//!    WRITE   md / txt        — std::fs::write
//!            csv             — `csv` writer
//!            json            — serde_json::to_writer_pretty
//!            pdf             — `printpdf` single-page text emitter
//!            docx            — minimal hand-rolled OOXML emitter
//!
//! The structural anti-mutation invariant lives in mcp.rs at the
//! `tool_file_extract_to_pending` boundary: that function MUST route
//! every ingested unit through `brain.pending.push` (T.3 PendingLog).
//! NEVER `teach*`, NEVER `network::add_bond`. Mirror of the T-Ingest /
//! T-Ingest.2 grep guards.

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Logical file kind. Inferred from extension or supplied explicitly
/// by the caller as a hint string. Keeping this an enum (rather than
/// a free string) lets the read/write paths exhaustively dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileKind {
    Md,
    Txt,
    Csv,
    Json,
    Pdf,
    Docx,
    Code,
}

impl FileKind {
    /// Infer from file extension. Returns `None` for unknown extensions;
    /// the caller may then fall back to `Code` or surface an error.
    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("md") | Some("markdown") => Some(FileKind::Md),
            Some("txt") => Some(FileKind::Txt),
            Some("csv") | Some("tsv") => Some(FileKind::Csv),
            Some("json") => Some(FileKind::Json),
            Some("pdf") => Some(FileKind::Pdf),
            Some("docx") => Some(FileKind::Docx),
            Some("rs") | Some("py") | Some("js") | Some("ts") | Some("go") |
            Some("java") | Some("c") | Some("h") | Some("cpp") | Some("hpp") |
            Some("rb") | Some("sh") | Some("toml") | Some("yaml") | Some("yml") |
            Some("xml") | Some("html") | Some("css") | Some("sql") => Some(FileKind::Code),
            _ => None,
        }
    }

    /// Parse a human hint ("md", "pdf", "DOCX", ...). Case-insensitive.
    pub fn from_str_hint(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "md" | "markdown" => Some(FileKind::Md),
            "txt" | "text" => Some(FileKind::Txt),
            "csv" | "tsv" => Some(FileKind::Csv),
            "json" => Some(FileKind::Json),
            "pdf" => Some(FileKind::Pdf),
            "docx" => Some(FileKind::Docx),
            "code" => Some(FileKind::Code),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FileKind::Md => "md",
            FileKind::Txt => "txt",
            FileKind::Csv => "csv",
            FileKind::Json => "json",
            FileKind::Pdf => "pdf",
            FileKind::Docx => "docx",
            FileKind::Code => "code",
        }
    }
}

/// Successful read result. Variants are populated by kind:
///   md/txt/code/pdf/docx → `text` is `Some`; the rest are `None`.
///   csv                  → `csv_rows` is `Some`; row 0 is the header
///                          row IFF the source had one (we always pull
///                          headers as row 0 for predictable downstream).
///   json                 → `json_value` is `Some`.
#[derive(Debug, Clone)]
pub struct ReadResult {
    pub kind: FileKind,
    pub byte_len: u64,
    pub text: Option<String>,
    pub csv_rows: Option<Vec<Vec<String>>>,
    pub json_value: Option<serde_json::Value>,
    pub source_url_hint: String,
}


/// Runtime knob bundle, mirrored from runtime_config section #15.
/// Default-initialised in Brain::new() and rehydrated on load.
#[derive(Debug, Clone, PartialEq)]
pub struct FileIoConfig {
    pub max_size_bytes: usize,
    pub allowed_read_paths: Vec<String>,
    pub allowed_write_paths: Vec<String>,
}

impl Default for FileIoConfig {
    fn default() -> Self {
        FileIoConfig {
            max_size_bytes: default_file_io_max_size_bytes(),
            allowed_read_paths: default_file_io_allowed_read_paths(),
            allowed_write_paths: default_file_io_allowed_write_paths(),
        }
    }
}

/// 50 MB. Plenty of headroom for ordinary corpora; refuses
/// pathological inputs (e.g. accidentally pointed at a 5 GB binary).
pub fn default_file_io_max_size_bytes() -> usize { 50 * 1024 * 1024 }

/// Empty by default = no path restriction on reads. Set this to a
/// list of dir prefixes to sandbox the read surface (e.g. limit reads
/// to a designated uploads dir).
pub fn default_file_io_allowed_read_paths() -> Vec<String> { Vec::new() }

/// Default writes are constrained to two well-known dirs so a
/// runaway tool call cannot scribble across the host. Override via
/// runtime_config when running outside this layout.
pub fn default_file_io_allowed_write_paths() -> Vec<String> {
    vec![
        "/var/lib/lu/outputs/".to_string(),
        "/tmp/lu/".to_string(),
    ]
}

/// Refuse to read files larger than the configured cap. Returns the
/// observed length on success.
pub fn check_size(path: &Path, max: usize) -> Result<u64, String> {
    let md = std::fs::metadata(path)
        .map_err(|e| format!("file_io: metadata({:?}): {}", path, e))?;
    let len = md.len();
    if len as usize > max {
        return Err(format!(
            "file_io: file size {} exceeds file_io_max_size_bytes ({})",
            len, max,
        ));
    }
    Ok(len)
}

/// Empty allow-list = no restriction. Otherwise the path must canonicalise
/// under one of the configured roots. Symlink-following is intentional —
/// we let `canonicalize` resolve them so a symlink trick cannot escape
/// a sandbox dir.
pub fn check_read_allowed(path: &Path, allowed: &[String]) -> Result<(), String> {
    if allowed.is_empty() { return Ok(()); }
    let canon = std::fs::canonicalize(path)
        .map_err(|e| format!("file_io: canonicalize({:?}): {}", path, e))?;
    for root in allowed {
        let root_path = Path::new(root);
        let canon_root = std::fs::canonicalize(root_path)
            .unwrap_or_else(|_| root_path.to_path_buf());
        if canon.starts_with(&canon_root) {
            return Ok(());
        }
    }
    Err(format!(
        "file_io: read path {:?} is not under any file_io_allowed_read_paths root: {:?}",
        path, allowed,
    ))
}

/// Empty allow-list = no restriction. Otherwise the (possibly
/// non-existent) target path must lie under one of the configured
/// roots. We canonicalise the deepest existing ancestor and require
/// it to live under one of the roots; this keeps the check meaningful
/// even when the file has not been created yet.
pub fn check_write_allowed(path: &Path, allowed: &[String]) -> Result<(), String> {
    if allowed.is_empty() { return Ok(()); }
    // Walk up to the first existing ancestor.
    let mut probe: PathBuf = path.to_path_buf();
    loop {
        if probe.exists() { break; }
        if !probe.pop() || probe.as_os_str().is_empty() { break; }
    }
    let canon_anchor = if probe.exists() {
        std::fs::canonicalize(&probe)
            .map_err(|e| format!("file_io: canonicalize({:?}): {}", probe, e))?
    } else {
        // No existing ancestor — fall back to a lexical comparison.
        path.to_path_buf()
    };
    let raw_path_str = path.to_string_lossy().to_string();
    for root in allowed {
        let root_path = Path::new(root);
        let canon_root = std::fs::canonicalize(root_path)
            .unwrap_or_else(|_| root_path.to_path_buf());
        if canon_anchor.starts_with(&canon_root) || canon_anchor.starts_with(root_path) {
            return Ok(());
        }
        // Lexical match: handles the "target dir does not exist yet"
        // case where canonicalize would fail.
        if raw_path_str.starts_with(root) {
            return Ok(());
        }
    }
    Err(format!(
        "file_io: write path {:?} is not under any file_io_allowed_write_paths root: {:?}",
        path, allowed,
    ))
}
// ---------------------------------------------------------------------
// READ entry points.
// ---------------------------------------------------------------------

/// Read + parse a file according to its kind. Returns a populated
/// `ReadResult`. Caller decides how to route the parsed content into
/// the brain (Inspect / UserStatement / TeachCorpus).
pub fn read_file(
    path: &Path,
    kind_hint: Option<FileKind>,
    cfg: &FileIoConfig,
) -> Result<ReadResult, String> {
    let byte_len = check_size(path, cfg.max_size_bytes)?;
    check_read_allowed(path, &cfg.allowed_read_paths)?;
    let kind = kind_hint
        .or_else(|| FileKind::from_extension(path))
        .ok_or_else(|| format!(
            "file_io: cannot infer FileKind from path {:?} (supply `kind` hint)",
            path,
        ))?;
    let bytes = std::fs::read(path)
        .map_err(|e| format!("file_io: read({:?}): {}", path, e))?;
    let source_url_hint = format!("file://{}", path.display());
    let result = match kind {
        FileKind::Md | FileKind::Txt | FileKind::Code => {
            let text = String::from_utf8(bytes)
                .map_err(|e| format!("file_io: utf8({:?}): {}", path, e))?;
            ReadResult { kind, byte_len, text: Some(text), csv_rows: None, json_value: None, source_url_hint }
        }
        FileKind::Json => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("file_io: json parse({:?}): {}", path, e))?;
            ReadResult { kind, byte_len, text: None, csv_rows: None, json_value: Some(value), source_url_hint }
        }
        FileKind::Csv => {
            let rows = parse_csv_bytes(&bytes)?;
            ReadResult { kind, byte_len, text: None, csv_rows: Some(rows), json_value: None, source_url_hint }
        }
        FileKind::Pdf => {
            let text = pdf_extract_text(&bytes)?;
            ReadResult { kind, byte_len, text: Some(text), csv_rows: None, json_value: None, source_url_hint }
        }
        FileKind::Docx => {
            let text = docx_extract_text(&bytes)?;
            ReadResult { kind, byte_len, text: Some(text), csv_rows: None, json_value: None, source_url_hint }
        }
    };
    Ok(result)
}

/// Pull every record into memory. The csv crate is permissive about
/// non-rectangular rows (`flexible(true)`), which is what we want for
/// real-world corpora. The first row is preserved as-is so callers
/// can treat it as the header row.
fn parse_csv_bytes(bytes: &[u8]) -> Result<Vec<Vec<String>>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(bytes);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for rec in rdr.records() {
        let r = rec.map_err(|e| format!("file_io: csv parse: {}", e))?;
        rows.push(r.iter().map(|s| s.to_string()).collect());
    }
    Ok(rows)
}

fn pdf_extract_text(bytes: &[u8]) -> Result<String, String> {
    pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| format!("file_io: pdf-extract: {}", e))
}

/// Pull text out of a DOCX archive. DOCX is a zip whose meaningful
/// text content lives in `word/document.xml` inside <w:t>...</w:t>
/// runs. We do the smallest scan that gives reasonable output:
/// concatenate every <w:t> body, inserting a newline at each
/// </w:p> paragraph boundary. This is intentionally NOT a full
/// OOXML reader; it is enough to round-trip the texts we emit and
/// to ingest plain narrative content from real-world .docx files.
fn docx_extract_text(bytes: &[u8]) -> Result<String, String> {
    use std::io::Read;
    let cur = std::io::Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(cur)
        .map_err(|e| format!("file_io: docx zip: {}", e))?;
    let mut doc = zip
        .by_name("word/document.xml")
        .map_err(|e| format!("file_io: docx missing word/document.xml: {}", e))?;
    let mut xml = String::new();
    doc.read_to_string(&mut xml)
        .map_err(|e| format!("file_io: docx read xml: {}", e))?;
    Ok(scan_wt_text(&xml))
}

/// Scan an OOXML body for `<w:t>...</w:t>` runs. Newline at every
/// `</w:p>`. XML entity references are decoded.
fn scan_wt_text(xml: &str) -> String {
    let mut out = String::new();
    for para in xml.split("</w:p>") {
        let mut buf = String::new();
        let mut cursor = 0usize;
        while let Some(rel) = para[cursor..].find("<w:t") {
            let s = cursor + rel;
            let body_start = match para[s..].find('>') {
                Some(o) => s + o + 1,
                None => break,
            };
            let body_end_rel = match para[body_start..].find("</w:t>") {
                Some(o) => body_start + o,
                None => break,
            };
            buf.push_str(&decode_xml(&para[body_start..body_end_rel]));
            cursor = body_end_rel + "</w:t>".len();
        }
        if !buf.is_empty() {
            if !out.is_empty() { out.push('\n'); }
            out.push_str(&buf);
        }
    }
    out
}

fn decode_xml(s: &str) -> String {
    s.replace("&amp;", "&")
     .replace("&lt;", "<")
     .replace("&gt;", ">")
     .replace("&quot;", "\"")
     .replace("&apos;", "'")
}

fn encode_xml(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
}

// ---------------------------------------------------------------------
// JSON deep-flatten: walk a serde_json::Value into (subject, relation,
// object) triples. Object/Array containers recurse; leaves emit one
// triple per leaf where:
//   subject  = parent JSON-pointer path ("/" for root leaves)
//   relation = the leaf key (or array index as string)
//   object   = string-formatted value (raw string for Value::String,
//              JSON-formatted otherwise so e.g. true/false/null/numbers
//              round-trip predictably)
// ---------------------------------------------------------------------
pub fn flatten_json(
    v: &serde_json::Value,
    parent_path: &str,
    leaf_key: &str,
    out: &mut Vec<(String, String, String)>,
) {
    use serde_json::Value;
    match v {
        Value::Object(map) => {
            let here = if leaf_key.is_empty() {
                parent_path.to_string()
            } else if parent_path == "/" || parent_path.is_empty() {
                format!("/{}", leaf_key)
            } else {
                format!("{}/{}", parent_path, leaf_key)
            };
            for (k, vv) in map {
                flatten_json(vv, &here, k, out);
            }
        }
        Value::Array(arr) => {
            let here = if leaf_key.is_empty() {
                parent_path.to_string()
            } else if parent_path == "/" || parent_path.is_empty() {
                format!("/{}", leaf_key)
            } else {
                format!("{}/{}", parent_path, leaf_key)
            };
            for (i, vv) in arr.iter().enumerate() {
                let idx = i.to_string();
                flatten_json(vv, &here, &idx, out);
            }
        }
        leaf => {
            let val = match leaf {
                Value::String(s) => s.clone(),
                _ => leaf.to_string(),
            };
            let subject = if parent_path.is_empty() { "/".to_string() } else { parent_path.to_string() };
            out.push((subject, leaf_key.to_string(), val));
        }
    }
}

// ---------------------------------------------------------------------
// WRITE entry points.
// ---------------------------------------------------------------------

/// Write `text` to `path` as a UTF-8 text file. Used for md / txt / code.
pub fn write_text(path: &Path, text: &str, cfg: &FileIoConfig) -> Result<(), String> {
    check_write_allowed(path, &cfg.allowed_write_paths)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("file_io: create_dir_all({:?}): {}", parent, e))?;
    }
    std::fs::write(path, text)
        .map_err(|e| format!("file_io: write({:?}): {}", path, e))
}

/// Write a 2-D string matrix as CSV (header is whatever the first row contains).
pub fn write_csv(path: &Path, rows: &[Vec<String>], cfg: &FileIoConfig) -> Result<(), String> {
    check_write_allowed(path, &cfg.allowed_write_paths)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("file_io: create_dir_all({:?}): {}", parent, e))?;
    }
    let f = std::fs::File::create(path)
        .map_err(|e| format!("file_io: create({:?}): {}", path, e))?;
    let mut wtr = csv::Writer::from_writer(f);
    for row in rows {
        wtr.write_record(row)
            .map_err(|e| format!("file_io: csv write_record: {}", e))?;
    }
    wtr.flush().map_err(|e| format!("file_io: csv flush: {}", e))
}

pub fn write_json(path: &Path, value: &serde_json::Value, cfg: &FileIoConfig) -> Result<(), String> {
    check_write_allowed(path, &cfg.allowed_write_paths)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("file_io: create_dir_all({:?}): {}", parent, e))?;
    }
    let f = std::fs::File::create(path)
        .map_err(|e| format!("file_io: create({:?}): {}", path, e))?;
    serde_json::to_writer_pretty(f, value)
        .map_err(|e| format!("file_io: json write: {}", e))
}

/// Emit a single-page (or multi-page on overflow) PDF whose body is
/// the supplied markdown-flavoured text. We do not interpret markdown;
/// it is rendered verbatim with naive line-wrapping. Sufficient for
/// emitting a readable report from `lu4d_narrate` output.
pub fn write_pdf(path: &Path, content: &str, cfg: &FileIoConfig) -> Result<(), String> {
    check_write_allowed(path, &cfg.allowed_write_paths)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("file_io: create_dir_all({:?}): {}", parent, e))?;
    }
    use printpdf::*;
    let title = path.file_name().and_then(|s| s.to_str()).unwrap_or("document");
    let (doc, page1, layer1) = PdfDocument::new(title, Mm(210.0), Mm(297.0), "L1");
    let font = doc.add_builtin_font(BuiltinFont::Helvetica)
        .map_err(|e| format!("file_io: pdf font: {}", e))?;
    let mut current_layer = doc.get_page(page1).get_layer(layer1);
    let line_height_mm: f32 = 5.0;
    let max_lines_per_page = 55i32;
    let left_margin = Mm(20.0);
    let top_y = Mm(280.0_f32);
    let mut line_index = 0i32;
    for raw_line in content.split('\n') {
        // Soft-wrap absurdly long lines at ~95 chars to keep within page width.
        let chunks = soft_wrap(raw_line, 95);
        for chunk in chunks {
            if line_index >= max_lines_per_page {
                let (np, nl) = doc.add_page(Mm(210.0), Mm(297.0), "L");
                current_layer = doc.get_page(np).get_layer(nl);
                line_index = 0;
            }
            let y = Mm(top_y.0 - line_height_mm * (line_index as f32));
            current_layer.use_text(chunk, 11.0, left_margin, y, &font);
            line_index += 1;
        }
    }
    let f = std::fs::File::create(path)
        .map_err(|e| format!("file_io: pdf create({:?}): {}", path, e))?;
    let mut buf = std::io::BufWriter::new(f);
    doc.save(&mut buf).map_err(|e| format!("file_io: pdf save: {}", e))?;
    Ok(())
}

fn soft_wrap(line: &str, max_chars: usize) -> Vec<String> {
    if line.chars().count() <= max_chars { return vec![line.to_string()]; }
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in line.split(' ') {
        if current.chars().count() + word.chars().count() + 1 > max_chars && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() { current.push(' '); }
        current.push_str(word);
    }
    if !current.is_empty() { out.push(current); }
    out
}

/// Emit a minimal valid DOCX archive containing `[Content_Types].xml`,
/// `_rels/.rels`, and `word/document.xml`. The body splits on '\n' and
/// emits one paragraph per line. Word and LibreOffice both open the
/// resulting file; round-trip via `docx_extract_text` returns the same
/// text (modulo XML entity decoding).
pub fn write_docx(path: &Path, content: &str, cfg: &FileIoConfig) -> Result<(), String> {
    check_write_allowed(path, &cfg.allowed_write_paths)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("file_io: create_dir_all({:?}): {}", parent, e))?;
    }
    use std::io::Write;
    use zip::write::FileOptions;
    let f = std::fs::File::create(path)
        .map_err(|e| format!("file_io: docx create({:?}): {}", path, e))?;
    let mut zip = zip::ZipWriter::new(f);
    let opts = FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // [Content_Types].xml
    zip.start_file("[Content_Types].xml", opts)
        .map_err(|e| format!("file_io: docx zip start: {}", e))?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
        .map_err(|e| format!("file_io: docx zip write: {}", e))?;

    // _rels/.rels
    zip.start_file("_rels/.rels", opts)
        .map_err(|e| format!("file_io: docx zip start: {}", e))?;
    zip.write_all(ROOT_RELS_XML.as_bytes())
        .map_err(|e| format!("file_io: docx zip write: {}", e))?;

    // word/document.xml
    let mut body = String::new();
    body.push_str(DOC_PRELUDE);
    for line in content.split('\n') {
        body.push_str("<w:p><w:r><w:t xml:space=\"preserve\">");
        body.push_str(&encode_xml(line));
        body.push_str("</w:t></w:r></w:p>");
    }
    body.push_str(DOC_EPILOGUE);
    zip.start_file("word/document.xml", opts)
        .map_err(|e| format!("file_io: docx zip start: {}", e))?;
    zip.write_all(body.as_bytes())
        .map_err(|e| format!("file_io: docx zip write: {}", e))?;

    zip.finish().map_err(|e| format!("file_io: docx zip finish: {}", e))?;
    Ok(())
}

const CONTENT_TYPES_XML: &str = concat!(
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
    "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
    "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
    "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
    "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>",
    "</Types>",
);

const ROOT_RELS_XML: &str = concat!(
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>",
    "</Relationships>",
);

const DOC_PRELUDE: &str = concat!(
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
    "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>",
);

const DOC_EPILOGUE: &str = "</w:body></w:document>";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn permissive_cfg() -> FileIoConfig {
        FileIoConfig {
            max_size_bytes: 1024 * 1024 * 1024,
            allowed_read_paths: Vec::new(),
            allowed_write_paths: Vec::new(),
        }
    }

    #[test]
    fn fileio_md_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p: PathBuf = dir.path().join("hello.md");
        let cfg = permissive_cfg();
        let content = "# Title\n\nHello, **world** & friends.\n";
        write_text(&p, content, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        assert_eq!(r.kind, FileKind::Md);
        assert_eq!(r.text.as_deref(), Some(content));
        assert!(r.csv_rows.is_none() && r.json_value.is_none());
    }

    #[test]
    fn fileio_txt_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("plain.txt");
        let cfg = permissive_cfg();
        let content = "line one\nline two\n";
        write_text(&p, content, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        assert_eq!(r.kind, FileKind::Txt);
        assert_eq!(r.text.as_deref(), Some(content));
    }

    #[test]
    fn fileio_csv_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("data.csv");
        let cfg = permissive_cfg();
        let rows = vec![
            vec!["name".into(),  "role".into(), "city".into()],
            vec!["alice".into(), "eng".into(),  "sf".into()],
            vec!["bob".into(),   "pm".into(),   "nyc, ny".into()],
        ];
        write_csv(&p, &rows, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        let got = r.csv_rows.expect("csv rows");
        assert_eq!(got.len(), rows.len());
        assert_eq!(got[0], rows[0]);
        assert_eq!(got[1], rows[1]);
        assert_eq!(got[2], rows[2]);
    }

    #[test]
    fn fileio_json_round_trip_and_flatten() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("obj.json");
        let cfg = permissive_cfg();
        let v = serde_json::json!({
            "name": "alice",
            "meta": { "age": 31, "flag": true },
            "tags": ["a", "b"]
        });
        write_json(&p, &v, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        let got = r.json_value.expect("json value");
        assert_eq!(got, v);

        let mut triples: Vec<(String, String, String)> = Vec::new();
        flatten_json(&got, "", "", &mut triples);
        // Expect one triple per leaf: name, meta/age, meta/flag, tags/0, tags/1.
        assert_eq!(triples.len(), 5, "got triples: {:?}", triples);
        let s: std::collections::HashSet<(String, String, String)> = triples.into_iter().collect();
        assert!(s.contains(&("/".to_string(), "name".to_string(), "alice".to_string())));
        assert!(s.contains(&("/meta".to_string(), "age".to_string(), "31".to_string())));
        assert!(s.contains(&("/meta".to_string(), "flag".to_string(), "true".to_string())));
        assert!(s.contains(&("/tags".to_string(), "0".to_string(), "a".to_string())));
        assert!(s.contains(&("/tags".to_string(), "1".to_string(), "b".to_string())));
    }

    #[test]
    fn fileio_docx_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hello.docx");
        let cfg = permissive_cfg();
        let body = "first paragraph\nsecond paragraph with & and <";
        write_docx(&p, body, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        assert_eq!(r.kind, FileKind::Docx);
        let got = r.text.expect("docx text");
        assert!(got.contains("first paragraph"), "got: {:?}", got);
        assert!(got.contains("second paragraph with & and <"), "got: {:?}", got);
    }

    #[test]
    fn fileio_pdf_write_then_read_text_present() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hello.pdf");
        let cfg = permissive_cfg();
        let body = "Lu4D File-IO smoke test\nLine two\nLine three";
        write_pdf(&p, body, &cfg).unwrap();
        let r = read_file(&p, None, &cfg).unwrap();
        let got = r.text.expect("pdf text");
        // pdf-extract introduces extra whitespace; check substrings.
        assert!(got.contains("Lu4D File-IO"), "got: {:?}", got);
        assert!(got.contains("Line three"), "got: {:?}", got);
    }

    #[test]
    fn fileio_size_limit_refuses_oversize() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        // Write 2 KB; cap at 1 KB.
        std::fs::write(&p, vec![b'x'; 2048]).unwrap();
        let cfg = FileIoConfig { max_size_bytes: 1024, ..permissive_cfg() };
        let err = read_file(&p, None, &cfg).unwrap_err();
        assert!(err.contains("exceeds file_io_max_size_bytes"), "err={}", err);
    }

    #[test]
    fn fileio_read_path_restriction_refuses_outside_root() {
        let outside = tempfile::tempdir().unwrap();
        let allowed_root = tempfile::tempdir().unwrap();
        let p = outside.path().join("foo.txt");
        std::fs::write(&p, b"hi").unwrap();
        let cfg = FileIoConfig {
            max_size_bytes: 1024,
            allowed_read_paths: vec![allowed_root.path().to_string_lossy().to_string()],
            allowed_write_paths: Vec::new(),
        };
        let err = read_file(&p, None, &cfg).unwrap_err();
        assert!(err.contains("not under any file_io_allowed_read_paths"), "err={}", err);
    }

    #[test]
    fn fileio_write_path_restriction_refuses_outside_root() {
        let outside = tempfile::tempdir().unwrap();
        let allowed_root = tempfile::tempdir().unwrap();
        let p = outside.path().join("foo.txt");
        let cfg = FileIoConfig {
            max_size_bytes: 1024 * 1024,
            allowed_read_paths: Vec::new(),
            allowed_write_paths: vec![allowed_root.path().to_string_lossy().to_string()],
        };
        let err = write_text(&p, "hi", &cfg).unwrap_err();
        assert!(err.contains("not under any file_io_allowed_write_paths"), "err={}", err);
    }

    #[test]
    fn fileio_extension_inference_covers_common_kinds() {
        assert_eq!(FileKind::from_extension(Path::new("x.md")),    Some(FileKind::Md));
        assert_eq!(FileKind::from_extension(Path::new("x.txt")),   Some(FileKind::Txt));
        assert_eq!(FileKind::from_extension(Path::new("x.csv")),   Some(FileKind::Csv));
        assert_eq!(FileKind::from_extension(Path::new("x.json")),  Some(FileKind::Json));
        assert_eq!(FileKind::from_extension(Path::new("x.pdf")),   Some(FileKind::Pdf));
        assert_eq!(FileKind::from_extension(Path::new("x.docx")),  Some(FileKind::Docx));
        assert_eq!(FileKind::from_extension(Path::new("x.rs")),    Some(FileKind::Code));
        assert_eq!(FileKind::from_extension(Path::new("x.py")),    Some(FileKind::Code));
        assert_eq!(FileKind::from_extension(Path::new("x.unknown")), None);
    }
}
