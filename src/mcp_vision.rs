// =====================================================================
// Port R input --- MCP tool wrappers for user-facing image ingest.
//
//   1. `lu4d_image_ingest`            -- read an image, run the
//                                        configured SceneGraphExtractor,
//                                        route extracted triplets per
//                                        the `route` arg through the
//                                        existing teach_corpus or
//                                        user_statement pipes (or just
//                                        return them via Inspect).
//   2. `lu4d_image_describe`          -- convenience: inspect-only,
//                                        returns triplets as a
//                                        human-readable description.
//   3. `lu4d_image_extractor_status`  -- diagnostic: which extractor
//                                        is configured, what timeout,
//                                        does the subprocess path
//                                        exist.
//
// Articulator-side: no brain schema changes. The image is a transport.
// Its extracted triplets flow through existing brain APIs:
//
//   * route=TeachCorpus    -> brain.teach_corpus (T-Ingest path,
//                              Right Brain bonds, source attribution
//                              auto-stamps file:// URI).
//   * route=UserStatement  -> brain.lb_session.push_internal
//                              (Tier 2 prompt path, Left Brain entry).
//   * route=Inspect        -> no brain mutation; returns triplets.
//
// STRUCTURAL ANTI-MUTATION INVARIANT: tool_image_ingest does NOT call
// brain.teach* directly OR network::add_bond directly except through
// the existing teach_corpus path. Pinned by
// `vision_grep_guard_image_ingest_no_direct_mutation` below.
// =====================================================================

use crate::brain::Brain;
use crate::bond::BondType;
use crate::vision::extractor::{
    sniff_mime, ExtractError, ExtractHints, SceneGraphExtractor, Triplet,
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn vw_wrap(text: String, structured: Value) -> Value {
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

/// Map an arbitrary predicate string to a canonical `BondType`. The
/// raw predicate text is preserved in the bond's source attribution
/// (via teach_corpus' source_url / corpus payload) so audits retain
/// it; this only collapses similar relations onto the BondType axis
/// for the brain's typed-bond machinery.
fn predicate_to_bond_type(s: &str) -> BondType {
    match s.to_ascii_lowercase().as_str() {
        "is" | "is_a" | "isa" => BondType::IsA,
        "has" | "has_property" | "with" => BondType::HasProperty,
        "causes" | "leads_to" => BondType::Causes,
        "part_of" | "composed_of" | "contains" => BondType::PartOf,
        "used_for" | "is_for" => BondType::UsedFor,
        "contrasted_with" | "unlike" => BondType::ContrastedWith,
        "derived_from" | "such_as" => BondType::DerivedFrom,
        "requires" => BondType::Requires,
        "produces" => BondType::Produces,
        "opposes" => BondType::Opposes,
        // Spatial / scene-graph predicates ("on", "behind", "above",
        // "holding", "wearing", ...) do not map cleanly onto any
        // existing BondType — file them as RelatedTo. Real semantics
        // will arrive when Cat.5.1 capability packs ship a richer
        // typed-relation vocabulary.
        _ => BondType::RelatedTo,
    }
}

fn read_image_bytes(
    path: &std::path::Path,
    cfg: &VisionConfig,
) -> Result<(Vec<u8>, String), (i64, String)> {
    let canon = std::fs::canonicalize(path)
        .map_err(|e| (-32603, format!("image: canonicalize({:?}): {}", path, e)))?;
    let md = std::fs::metadata(&canon)
        .map_err(|e| (-32603, format!("image: metadata({:?}): {}", canon, e)))?;
    let len = md.len() as usize;
    if len > cfg.image_max_size_bytes {
        return Err((
            -32602,
            format!(
                "image: file size {} exceeds image_max_size_bytes ({})",
                len, cfg.image_max_size_bytes
            ),
        ));
    }
    let bytes = std::fs::read(&canon)
        .map_err(|e| (-32603, format!("image: read({:?}): {}", canon, e)))?;
    let url = format!("file://{}", canon.display());
    Ok((bytes, url))
}

fn validate_mime(
    bytes: &[u8],
    explicit: Option<&str>,
    cfg: &VisionConfig,
) -> Result<String, (i64, String)> {
    let inferred = sniff_mime(bytes);
    let chosen = match (explicit, inferred) {
        (Some(e), _) => e.to_string(),
        (None, Some(i)) => i.to_string(),
        (None, None) => {
            return Err((
                -32602,
                "image: could not infer mime type from leading bytes; pass `mime_type` explicitly"
                    .to_string(),
            ))
        }
    };
    if !cfg
        .image_supported_mime_types
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&chosen))
    {
        return Err((
            -32602,
            format!(
                "image: mime type {:?} not in image_supported_mime_types {:?}",
                chosen, cfg.image_supported_mime_types
            ),
        ));
    }
    Ok(chosen)
}

/// Runtime knob bundle, mirrored from runtime_config section #15.
/// Default-initialised in `Brain::new()` and rehydrated on load.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub extractor_kind: VisionExtractorKind,
    pub subprocess_path: Option<String>,
    pub subprocess_timeout_ms: u64,
    pub image_max_size_bytes: usize,
    pub image_supported_mime_types: Vec<String>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            extractor_kind: VisionExtractorKind::default(),
            subprocess_path: None,
            subprocess_timeout_ms: default_image_extractor_subprocess_timeout_ms(),
            image_max_size_bytes: default_image_max_size_bytes(),
            image_supported_mime_types: default_image_supported_mime_types(),
        }
    }
}

/// Selected scene-graph extractor implementation. `Stub` is the
/// default — the API works end-to-end during development without any
/// external CV. `Subprocess` opts in to an external CLI. The
/// `CapabilityPack` arm is documented but NOT IMPLEMENTED; it is the
/// long-term target once Cat.5.1 ships executable capability packs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionExtractorKind {
    Stub,
    Subprocess,
    CapabilityPack { pack_id: String },
}

impl Default for VisionExtractorKind {
    fn default() -> Self {
        VisionExtractorKind::Stub
    }
}

pub fn default_image_extractor_subprocess_timeout_ms() -> u64 {
    30_000
}
pub fn default_image_max_size_bytes() -> usize {
    25_000_000
}
pub fn default_image_supported_mime_types() -> Vec<String> {
    vec![
        "image/jpeg".to_string(),
        "image/png".to_string(),
        "image/webp".to_string(),
        "image/gif".to_string(),
    ]
}

/// Build the configured extractor as an `Arc<dyn SceneGraphExtractor>`.
/// Respects `cfg.extractor_kind`; the `CapabilityPack` arm is rejected
/// with a clear "deferred to Cat.5.1" message so callers get a useful
/// error during the interim.
pub fn build_extractor(
    cfg: &VisionConfig,
    clock: std::sync::Arc<dyn crate::time::ClockSource>,
    registry: &crate::capability_runtime::LoadedPackRegistry,
    ledger: &crate::capability_pack::CapabilityPackLedgerStore,
) -> Result<std::sync::Arc<dyn SceneGraphExtractor>, String> {
    match &cfg.extractor_kind {
        VisionExtractorKind::Stub => Ok(std::sync::Arc::new(
            crate::vision::extractor::StubExtractor::new(clock),
        )),
        VisionExtractorKind::Subprocess => {
            let path = cfg.subprocess_path.as_ref().ok_or_else(|| {
                "image_extractor_kind=Subprocess but image_extractor_subprocess_path is None"
                    .to_string()
            })?;
            Ok(std::sync::Arc::new(
                crate::vision::extractor::SubprocessExtractor::new(
                    PathBuf::from(path),
                    cfg.subprocess_timeout_ms,
                    clock,
                ),
            ))
        }
        VisionExtractorKind::CapabilityPack { pack_id } => {
            if !registry.is_enabled() {
                return Err("capability runtime is disabled".to_string());
            }
            registry
                .reverify(pack_id, ledger)
                .map_err(|e| format!("capability pack {} re-verify: {}", pack_id, e))?;
            let nsname = crate::capability_runtime::namespaced_tool_name(
                pack_id,
                crate::capability_runtime::CapabilityPackExtractor::EXTRACTOR_TOOL,
            );
            let tool = registry.get_tool(&nsname).ok_or_else(|| {
                format!("capability pack {} has no loaded `{}` tool", pack_id, crate::capability_runtime::CapabilityPackExtractor::EXTRACTOR_TOOL)
            })?;
            Ok(std::sync::Arc::new(
                crate::capability_runtime::CapabilityPackExtractor::new(pack_id.clone(), tool),
            ))
        }
    }
}

fn map_extract_err(e: ExtractError) -> (i64, String) {
    match e {
        ExtractError::UnsupportedMime(_) | ExtractError::OversizeImage { .. } => {
            (-32602, format!("image extractor: {}", e))
        }
        _ => (-32603, format!("image extractor: {}", e)),
    }
}

// ---------------------------------------------------------------------
// `lu4d_image_ingest`
// ---------------------------------------------------------------------
pub fn tool_image_ingest(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let path_str = arg_str(&args, "path")?.to_string();
    let path = PathBuf::from(&path_str);
    let cfg = brain.vision_config.clone();
    let (bytes, url) = read_image_bytes(&path, &cfg)?;
    let explicit_mime = args.get("mime_type").and_then(|v| v.as_str());
    let mime = validate_mime(&bytes, explicit_mime, &cfg)?;
    let max_triplets = args
        .get("max_triplets")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let route = args
        .get("route")
        .and_then(|v| v.as_object())
        .ok_or((-32602, "missing object arg `route`".to_string()))?;

    let extractor = build_extractor(&cfg, brain.clock.clone(), &brain.pack_registry, &brain.cap_pack_ledger)
        .map_err(|e| (-32603, format!("image extractor build: {}", e)))?;
    let hints = ExtractHints {
        mime_type: Some(mime.clone()),
        max_triplets,
    };
    let triplets = extractor.extract(&bytes, hints).map_err(map_extract_err)?;
    let extractor_name = extractor.name().to_string();

    if route.contains_key("Inspect") || route.contains_key("inspect") {
        return Ok(inspect_response(&path, &mime, &url, &triplets, &extractor_name));
    }

    if let Some(us) = route.get("UserStatement").or_else(|| route.get("user_statement")) {
        let source_label = us
            .get("source_label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| Some(url.clone()));
        let session_id = us.get("session_id").and_then(|v| v.as_u64());
        let text = render_triplets_human(&triplets);
        let max_bytes = brain.tier2_prompt_config.max_text_bytes;
        if text.len() > max_bytes {
            return Err((
                -32602,
                format!(
                    "image: rendered description {} exceeds tier2_prompt_max_text_bytes ({})",
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
            .unwrap_or_else(|| "image_user_statement".to_string());
        let payload = json!({
            "text": text,
            "source_label": source_label,
            "source_url": url,
            "mime_type": mime,
            "byte_len": bytes.len(),
            "triplet_count": triplets.len(),
            "extractor": extractor_name,
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
            "lu4d_image_ingest[UserStatement]: file={} mime={} triplets={} entry_id={} session_id={} ephemeral={}",
            path.display(),
            mime,
            triplets.len(),
            entry_id,
            sid,
            ephemeral,
        );
        let structured = json!({
            "route": "UserStatement",
            "file": path_str,
            "mime_type": mime,
            "byte_len": bytes.len(),
            "source_url": url,
            "entry_id": entry_id,
            "session_id": sid,
            "ephemeral_session": ephemeral,
            "extractor": extractor_name,
            "triplet_count": triplets.len(),
            "asymmetry_invariant": "no_bond_formation",
        });
        return Ok(vw_wrap(text_msg, structured));
    }

    if let Some(tc) = route.get("TeachCorpus").or_else(|| route.get("teach_corpus")) {
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
        let mut bonds_formed = 0usize;
        let conf = crate::node::Confidence::Inferred;
        for t in &triplets {
            let bt = predicate_to_bond_type(&t.predicate);
            // Weight is the extractor's confidence, clamped to a
            // conservative ceiling so a hallucinating extractor can't
            // dominate the graph.
            let weight = t.confidence.clamp(0.0, 0.9);
            let _r = brain.teach_corpus(
                &t.subject,
                bt,
                &t.object,
                weight,
                conf,
                &corpus,
                run_id.clone(),
                Some(url.clone()),
            );
            bonds_formed += 1;
        }
        let text_msg = format!(
            "lu4d_image_ingest[TeachCorpus]: file={} mime={} bonds_formed={} corpus={} extractor={}",
            path.display(),
            mime,
            bonds_formed,
            corpus,
            extractor_name,
        );
        let structured = json!({
            "route": "TeachCorpus",
            "file": path_str,
            "mime_type": mime,
            "byte_len": bytes.len(),
            "source_url": url,
            "corpus": corpus,
            "run_id": run_id,
            "bonds_formed": bonds_formed,
            "extractor": extractor_name,
            "triplets": triplets_to_json(&triplets),
        });
        return Ok(vw_wrap(text_msg, structured));
    }

    Err((
        -32602,
        "route must be one of: { Inspect: {} } | { UserStatement: {...} } | { TeachCorpus: { corpus, ... } }"
            .to_string(),
    ))
}

fn inspect_response(
    path: &std::path::Path,
    mime: &str,
    url: &str,
    triplets: &[Triplet],
    extractor_name: &str,
) -> Value {
    let preview = render_triplets_human(triplets);
    let s = json!({
        "route": "Inspect",
        "file": path.to_string_lossy(),
        "mime_type": mime,
        "source_url": url,
        "triplets": triplets_to_json(triplets),
        "extractor": extractor_name,
        "asymmetry_invariant": "no_bond_formation",
    });
    let text_msg = format!(
        "lu4d_image_ingest[Inspect]: file={} mime={} triplets={} preview={:?}",
        path.display(),
        mime,
        triplets.len(),
        preview.chars().take(160).collect::<String>(),
    );
    vw_wrap(text_msg, s)
}

fn render_triplets_human(triplets: &[Triplet]) -> String {
    triplets
        .iter()
        .map(|t| format!("{} {} {}", t.subject, t.predicate, t.object))
        .collect::<Vec<_>>()
        .join("; ")
}

fn triplets_to_json(triplets: &[Triplet]) -> Value {
    Value::Array(
        triplets
            .iter()
            .map(|t| {
                json!({
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                })
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------
// `lu4d_image_describe`
// ---------------------------------------------------------------------
pub fn tool_image_describe(brain: &mut Brain, args: Value) -> Result<Value, (i64, String)> {
    let path_str = arg_str(&args, "path")?.to_string();
    let path = PathBuf::from(&path_str);
    let cfg = brain.vision_config.clone();
    let (bytes, url) = read_image_bytes(&path, &cfg)?;
    let mime = validate_mime(&bytes, args.get("mime_type").and_then(|v| v.as_str()), &cfg)?;
    let extractor = build_extractor(&cfg, brain.clock.clone(), &brain.pack_registry, &brain.cap_pack_ledger)
        .map_err(|e| (-32603, format!("image extractor build: {}", e)))?;

    let dims = crate::vision::extractor::probe_image_dimensions(&bytes, &mime);
    let (w, h) = match dims {
        Some((w, h)) => (Some(w), Some(h)),
        None => (None, None),
    };
    let dim_str = match dims {
        Some((w, h)) => format!("{}x{}", w, h),
        None => "unknown-dimensions".to_string(),
    };

    // HONESTY FLOOR internal notes row-N): if the configured extractor
    // does not actually perceive image *contents* (the dev-only StubExtractor
    // ignores the bytes), do NOT dress its placeholder up as a description.
    // Return an honest capability statement: format/size/dimensions we can
    // truly read, plus an explicit statement that we cannot interpret the
    // contents and will not guess. No scene, no OCR text, no invented
    // objects, and -- critically -- no numeric answer to any diagram.
    if !extractor.perceives_contents() {
        let description = format!(
            "This is a {} image ({}), {} bytes. I can read its format and pixel \
             dimensions from the file, but I cannot interpret its visual \
             contents: no perceptual vision model is configured on this build \
             (extractor='{}'). I cannot read any text or labels in the image, \
             identify shapes, or solve a diagram from it, and I will not guess \
             at what it depicts.",
            mime,
            dim_str,
            bytes.len(),
            extractor.name(),
        );
        let text_msg = format!(
            "lu4d_image_describe[capability_statement]: file={} mime={} dims={} bytes={} extractor={} contents=uninterpreted",
            path.display(),
            mime,
            dim_str,
            bytes.len(),
            extractor.name(),
        );
        let structured = json!({
            "file": path_str,
            "mime_type": mime,
            "format": mime,
            "width": w,
            "height": h,
            "byte_len": bytes.len(),
            "source_url": url,
            "description": description,
            "can_describe_contents": false,
            "contents_status": "uninterpreted_no_perceptual_extractor",
            "triplets": [],
            "extractor": extractor.name(),
            "asymmetry_invariant": "no_bond_formation",
        });
        return Ok(vw_wrap(text_msg, structured));
    }

    // A real extractor is configured -- it genuinely perceives contents.
    let hints = ExtractHints {
        mime_type: Some(mime.clone()),
        max_triplets: args
            .get("max_triplets")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize),
    };
    let triplets = extractor.extract(&bytes, hints).map_err(map_extract_err)?;
    let description = render_triplets_human(&triplets);
    let text_msg = format!(
        "lu4d_image_describe: file={} mime={} dims={} triplets={} description={:?}",
        path.display(),
        mime,
        dim_str,
        triplets.len(),
        description.chars().take(160).collect::<String>(),
    );
    let structured = json!({
        "file": path_str,
        "mime_type": mime,
        "format": mime,
        "width": w,
        "height": h,
        "byte_len": bytes.len(),
        "source_url": url,
        "description": description,
        "can_describe_contents": true,
        "contents_status": "described",
        "triplets": triplets_to_json(&triplets),
        "extractor": extractor.name(),
        "asymmetry_invariant": "no_bond_formation",
    });
    Ok(vw_wrap(text_msg, structured))
}

// ---------------------------------------------------------------------
// `lu4d_image_extractor_status`
// ---------------------------------------------------------------------
pub fn tool_image_extractor_status(
    brain: &mut Brain,
    _args: Value,
) -> Result<Value, (i64, String)> {
    let cfg = brain.vision_config.clone();
    let kind_label = match &cfg.extractor_kind {
        VisionExtractorKind::Stub => "stub".to_string(),
        VisionExtractorKind::Subprocess => "subprocess".to_string(),
        VisionExtractorKind::CapabilityPack { pack_id } => {
            format!("capability_pack:{}", pack_id)
        }
    };
    let subprocess_path_exists = cfg.subprocess_path.as_ref().map(|p| {
        std::path::Path::new(p).exists()
    });
    let text_msg = format!(
        "lu4d_image_extractor_status: kind={} timeout_ms={} subprocess_path={:?} subprocess_path_exists={:?} max_size_bytes={} mime_types={:?}",
        kind_label,
        cfg.subprocess_timeout_ms,
        cfg.subprocess_path,
        subprocess_path_exists,
        cfg.image_max_size_bytes,
        cfg.image_supported_mime_types,
    );
    let structured = json!({
        "extractor_kind": kind_label,
        "subprocess_path": cfg.subprocess_path,
        "subprocess_path_exists": subprocess_path_exists,
        "subprocess_timeout_ms": cfg.subprocess_timeout_ms,
        "image_max_size_bytes": cfg.image_max_size_bytes,
        "image_supported_mime_types": cfg.image_supported_mime_types,
        "capability_pack_supported": false,
        "capability_pack_target_port": "Cat.5.1",
    });
    Ok(vw_wrap(text_msg, structured))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;

    fn ephemeral_brain() -> Brain {
        Brain::new()
    }

    fn write_tiny_png(dir: &std::path::Path, name: &str) -> PathBuf {
        // Minimal PNG: 8-byte signature + IHDR + IDAT + IEND for a
        // 1x1 transparent pixel. Hand-rolled so the test has no
        // image-crate dependency.
        let bytes: [u8; 67] = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR len + tag
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT len + tag
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND
            0x42, 0x60, 0x82,
        ];
        let p = dir.join(name);
        std::fs::write(&p, &bytes).unwrap();
        p
    }

    fn tempdir_for_test() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = format!(
            "lu_port_r_mcp_{}_{}",
            std::process::id(),
            crate::brain::now_ms()
        );
        p.push(nonce);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn extractor_status_reports_default_stub() {
        let mut b = ephemeral_brain();
        let resp = tool_image_extractor_status(&mut b, json!({})).unwrap();
        let s = &resp["structuredContent"];
        assert_eq!(s["extractor_kind"], "stub");
        assert_eq!(s["subprocess_path"], Value::Null);
        assert_eq!(s["capability_pack_supported"], false);
    }

    #[test]
    fn ingest_inspect_returns_triplets_without_mutation() {
        let mut b = ephemeral_brain();
        let dir = tempdir_for_test();
        let png = write_tiny_png(&dir, "a.png");
        let before = b.network.bond_count();
        let resp = tool_image_ingest(
            &mut b,
            json!({
                "path": png.to_string_lossy(),
                "route": { "Inspect": {} }
            }),
        )
        .unwrap();
        let s = &resp["structuredContent"];
        assert_eq!(s["route"], "Inspect");
        assert_eq!(s["mime_type"], "image/png");
        let arr = s["triplets"].as_array().expect("triplets array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["subject"], "image");
        assert_eq!(arr[0]["predicate"], "uploaded_at");
        assert_eq!(b.network.bond_count(), before, "Inspect must not mutate");
    }

    #[test]
    fn ingest_teach_corpus_forms_one_bond_with_stub() {
        let mut b = ephemeral_brain();
        let dir = tempdir_for_test();
        let png = write_tiny_png(&dir, "b.png");
        let before = b.network.bond_count();
        let _resp = tool_image_ingest(
            &mut b,
            json!({
                "path": png.to_string_lossy(),
                "route": { "TeachCorpus": { "corpus": "test/r-input-unit", "run_id": "u" } }
            }),
        )
        .unwrap();
        assert_eq!(
            b.network.bond_count(),
            before + 1,
            "stub extractor + TeachCorpus must form exactly 1 bond"
        );
    }

    #[test]
    fn ingest_oversize_image_rejected() {
        let mut b = ephemeral_brain();
        b.vision_config.image_max_size_bytes = 16; // smaller than tiny PNG
        let dir = tempdir_for_test();
        let png = write_tiny_png(&dir, "c.png");
        let err = tool_image_ingest(
            &mut b,
            json!({
                "path": png.to_string_lossy(),
                "route": { "Inspect": {} }
            }),
        )
        .unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("image_max_size_bytes"));
    }

    #[test]
    fn ingest_unsupported_mime_rejected() {
        let mut b = ephemeral_brain();
        let dir = tempdir_for_test();
        // Write a .bin file with no image signature.
        let p = dir.join("d.bin");
        std::fs::write(&p, b"not an image at all").unwrap();
        let err = tool_image_ingest(
            &mut b,
            json!({
                "path": p.to_string_lossy(),
                "route": { "Inspect": {} }
            }),
        )
        .unwrap_err();
        assert_eq!(err.0, -32602);
        // Either the sniff fails or the explicit mime rejection fires.
        assert!(
            err.1.contains("could not infer") || err.1.contains("not in image_supported_mime_types"),
            "unexpected error message: {}",
            err.1
        );
    }

    #[test]
    fn describe_stub_is_honest_capability_statement() {
        // HONESTY FLOOR internal notes row-N): with the default
        // StubExtractor (no real CV), image_describe must NOT present a
        // placeholder ("image uploaded_at ...") as a description. It must
        // return an honest capability statement: real format + dimensions,
        // an explicit "cannot interpret" and NO invented scene.
        let mut b = ephemeral_brain();
        let dir = tempdir_for_test();
        let png = write_tiny_png(&dir, "e.png");
        let resp = tool_image_describe(
            &mut b,
            json!({
                "path": png.to_string_lossy(),
            }),
        )
        .unwrap();
        let s = &resp["structuredContent"];
        let desc = s["description"].as_str().unwrap();
        assert!(!desc.contains("uploaded_at"), "must not present placeholder: {}", desc);
        assert!(desc.contains("cannot interpret"), "must state the limit: {}", desc);
        assert_eq!(s["can_describe_contents"], Value::Bool(false));
        assert_eq!(s["contents_status"], "uninterpreted_no_perceptual_extractor");
        assert_eq!(s["mime_type"], "image/png");
        assert_eq!(s["width"], 1);
        assert_eq!(s["height"], 1);
        assert_eq!(s["triplets"].as_array().unwrap().len(), 0, "no invented scene");
    }

    #[test]
    fn capability_pack_kind_requires_enabled_runtime() {
        // Cat.5.1: the CapabilityPack arm is now REAL. With the
        // capability runtime disabled by default (and no pack loaded),
        // it returns a clear runtime error rather than the old
        // "deferred to Cat.5.1" placeholder.
        let mut b = ephemeral_brain();
        b.vision_config.extractor_kind = VisionExtractorKind::CapabilityPack {
            pack_id: "future".to_string(),
        };
        let dir = tempdir_for_test();
        let png = write_tiny_png(&dir, "f.png");
        let err = tool_image_ingest(
            &mut b,
            json!({
                "path": png.to_string_lossy(),
                "route": { "Inspect": {} }
            }),
        )
        .unwrap_err();
        assert!(!err.1.contains("deferred to Cat.5.1"), "arm must no longer be deferred");
        assert!(err.1.contains("disabled"), "got: {}", err.1);
    }

    // ---------------------------------------------------------------
    // Structural anti-mutation invariant — grep guard.
    //
    // The body of `tool_image_ingest` may only mutate brain state
    // via the existing teach_corpus / lb_session paths. Any direct
    // call to teach* / network::add_bond / propose_to_pending would
    // violate the asymmetry rule and the C5 anti-mutation invariant.
    // ---------------------------------------------------------------
    #[test]
    fn vision_grep_guard_image_ingest_no_direct_mutation() {
        let src = include_str!("mcp_vision.rs");
        let start = src
            .find("pub fn tool_image_ingest")
            .expect("tool_image_ingest must exist");
        let body = &src[start..];
        let end = body.find("\nfn inspect_response").unwrap_or(body.len());
        let body = &body[..end];

        // ALLOWED: brain.teach_corpus is the canonical T-Ingest path.
        assert!(
            body.contains("brain.teach_corpus("),
            "tool_image_ingest must route TeachCorpus via brain.teach_corpus"
        );
        assert!(
            body.contains("lb_session"),
            "tool_image_ingest must route UserStatement via brain.lb_session"
        );

        // FORBIDDEN identifiers split at compile time so this test
        // source does not self-trigger the grep.
        const FORBIDDEN: &[&str] = &[
            concat!("brain.", "teach("),
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
                "tool_image_ingest body must NOT reference bond-mutating API '{}': all writes route through brain.teach_corpus / brain.lb_session",
                f,
            );
        }
    }
}
