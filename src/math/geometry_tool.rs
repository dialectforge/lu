//! Guest-safe JSON bridge to the exact-geometry solver (internal notes
//! #1258/#1262/#1267/#1269/#1353/#1412) + the Phase-5 IMAGE mode (#1549/#1550,
//! 2026-08-06). The verified solver in `super::geometry` (red-leaf shaded-area
//! partition + the tangent-constraint control problem + the sum-to-whole
//! `Partition` verifier and its IDK/K-gate) is wired here for STRUCTURED
//! scenes, and Phase 5 additionally wires the REAL perceiver
//! (`crate::vision::geo_perceive_p4::perceive_photo`) so a raw geometry PHOTO
//! flows pixels -> rectify -> arcs -> OCR -> scene -> verified solver -> exact
//! answer. This module does NOT reimplement any geometry or any perception; it
//! is the single wire between the guest surface and the already-proven engines.
//!
//! HONESTY FLOOR (absolute):
//!   * STRUCTURED scene: it solves a scene it is GIVEN. Under-constrained /
//!     inconsistent scenes surface the solver's OWN `VerifyOutcome::Idk` (a
//!     load-bearing tile below the confidence floor) or `::Flagged` (sub-
//!     regions do not sum to the whole). Both return NO number.
//!   * IMAGE mode: the number flows pixels->scene->solver EVERY time (via
//!     `perceive_photo`, which itself K-gates the OCR'd scale and runs
//!     `solve_and_verify`). A figure the perceiver cannot genuinely parse
//!     (can't decode / can't rectify / can't fit arcs / can't OCR the scale /
//!     low confidence) or a non-geometry image returns an HONEST capability
//!     statement (real format + dimensions + "I can't interpret/solve this
//!     image") with NO number -- NEVER a hardcoded fallback. There is NO
//!     filename/hash/answer-constant path: the bytes ARE the only input.
//!   * Read-only pure compute on the given data; forms no bonds, mutates no
//!     brain -- classified guest-safe internal notes: guest = read/compute).

use super::geometry::{solve_control, solve_red_leaf, Exact, Partition, Tile, VerifyOutcome};
use crate::vision::geo_perceive::decode_base64;
use crate::vision::geo_perceive_p4::{decode_image_rgb, perceive_photo, PerceiveP4Outcome};
use crate::vision::geo_perceive_p4::rectify;
use crate::vision::geo_labels::{label_on_raster, LabelResult};
use crate::vision::geo_constraints::{describe, extract_problem, ExtractResult};
use crate::math::algsolve::{solve as algsolve_solve, Solved as AlgSolved};
use serde_json::{json, Value};

fn wrap(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

/// True if the caller referenced a raw image (by any conventional key).
fn looks_like_raw_image(args: &Value) -> bool {
    ["image", "image_path", "path", "image_base64", "png", "jpeg", "png_base64", "jpeg_base64", "image_b64", "data"]
        .iter()
        .any(|k| args.get(k).map(|v| !v.is_null()).unwrap_or(false))
}

/// Extract genuine, decodable image bytes inlined as base64 under any of the
/// conventional keys. Requires a real PNG or JPEG magic signature so a
/// structured-scene string is never mistaken for an image. Returns None if no
/// key carries decodable image bytes (the caller then falls through to the
/// honest "no inline bytes" statement or to structured-scene handling).
fn image_bytes_from_args(args: &Value) -> Option<Vec<u8>> {
    for k in [
        "image_base64",
        "png_base64",
        "jpeg_base64",
        "image_b64",
        "data",
        "image",
        "png",
        "jpeg",
    ] {
        if let Some(s) = args.get(k).and_then(|v| v.as_str()) {
            // Tolerate a data: URL prefix ("data:image/png;base64,....").
            let payload = s.rsplit(',').next().unwrap_or(s);
            if let Some(bytes) = decode_base64(payload) {
                if bytes.len() >= 8
                    && (bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47])
                        || bytes.starts_with(&[0xFF, 0xD8, 0xFF]))
                {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

fn den_ok(e: &Exact) -> bool {
    e.rational.den != 0 && e.pi.den != 0
}

/// Sniff the container format from magic bytes (for the honest statement).
fn sniff_format(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        "PNG image"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "JPEG image"
    } else {
        "unrecognized (non-image?) data"
    }
}

/// Outcome of the P-E general chain on an image.
enum GeneralOutcome {
    /// The chain closed to an exact answer -> this MCP result carries a number.
    Solved(Value),
    /// The chain perceived a labelled scene but could not close it (P-D
    /// `Incomplete` or solver `Refuse`). Carries transparent provenance
    /// (primitives census + bound labels + the exact declining reason) so an
    /// honest refusal can show that perception WORKED and name the stage that
    /// declined. NO number.
    Incomplete(Value),
    /// The chain could not even obtain a labelled scene (or no decodable image)
    /// -> nothing to add; the caller falls through to the leaf perceiver.
    Nothing,
}

/// Transparent provenance object for a general-chain refusal (no number).
fn build_general_diag(
    census: &str,
    bindings: Vec<Value>,
    rectified: bool,
    assembled: Option<String>,
    reason: &str,
) -> Value {
    json!({
        "pipeline": "general_chain",
        "rectified": rectified,
        "primitives_census": census,
        "bindings": bindings,
        "assembled_system": assembled,
        "stage": "geo_constraints::extract_problem / algsolve::solve",
        "reason": reason,
    })
}

fn general_bindings_json(lab: &crate::vision::geo_labels::LabeledPrimitives) -> Vec<Value> {
    lab.bindings
        .iter()
        .map(|b| {
            json!({
                "label": b.label,
                "role": b.role.as_str(),
                "primitive_index": b.primitive_index,
                "primitive_kind": b.primitive_kind,
                "confidence": b.confidence,
                "evidence": b.evidence,
            })
        })
        .collect()
}

/// Run P-C labels + P-D extraction + algsolve on ONE already-decoded raster
/// (optionally rectified). Returns `Solved` only on an exact K-gated `Value`;
/// a labelled-but-unclosable scene returns `Incomplete` WITH provenance; an
/// unlabelable raster returns `Nothing`.
fn general_on_raster(raster: &crate::vision::geo_perceive::RgbImage, rectified: bool) -> GeneralOutcome {
    let lab = match label_on_raster(raster, rectified) {
        LabelResult::Labeled(l) => l,
        LabelResult::Incomplete { .. } => return GeneralOutcome::Nothing,
    };
    let census = lab.primitives.census();
    let bindings_json = general_bindings_json(&lab);

    match extract_problem(&lab) {
        ExtractResult::Problem(problem, target) => match algsolve_solve(&problem, &target) {
            AlgSolved::Value { rat, pic, text } => {
                let symbolic = text;
                let decimal = if rat.is_zero() {
                    pic.to_f64() * std::f64::consts::PI
                } else {
                    rat.to_f64()
                };
                let assembled = describe(&problem, &target);
                let prose = format!(
                    "I solved this geometry figure with the GENERAL pipeline (no per-figure \
                     template): decoded the image{}, detected primitives ({}), read and bound \
                     the labels, assembled an exact constraint system ({} points, {} \
                     constraints), and derived the queried quantity with the offline algebraic \
                     solver (K-gated for invariance across independent random placements): {} = \
                     {:.6}. This number was derived pixels -> primitives -> labels -> \
                     constraints -> solver; it is NOT looked up or hardcoded.",
                    if rectified { " (rectified the outer-square skew)" } else { "" },
                    census,
                    problem.pts.len(),
                    problem.cons.len(),
                    symbolic,
                    decimal,
                );
                GeneralOutcome::Solved(wrap(
                    prose,
                    json!({
                        "source": "geometry_solver",
                        "mode": "image",
                        "solved": true,
                        "pipeline": "general_chain",
                        "provenance": {
                            "stages": ["decode", "geo_primitives", "geo_labels", "geo_constraints::extract_problem", "algsolve::solve"],
                            "rectified": rectified,
                            "primitives_census": census,
                            "bindings": bindings_json,
                            "assembled_system": assembled,
                        },
                        "answer": {
                            "symbolic": symbolic,
                            "decimal": decimal,
                        },
                    }),
                ))
            }
            AlgSolved::Refused(r) => {
                let assembled = describe(&problem, &target);
                let reason = format!(
                    "the constraint system assembled but the K-gated solver refused ({:?}): it \
                     is not a determinate, exactly-recoverable, invariant value",
                    r
                );
                GeneralOutcome::Incomplete(build_general_diag(
                    &census,
                    bindings_json,
                    rectified,
                    Some(assembled),
                    &reason,
                ))
            }
        },
        ExtractResult::Incomplete { reason } => {
            GeneralOutcome::Incomplete(build_general_diag(&census, bindings_json, rectified, None, &reason))
        }
    }
}

/// P-E GENERAL geometry chain (guest-safe, pure compute): decode -> P-B
/// `geo_primitives` -> P-C `geo_labels` -> P-D `geo_constraints::extract_problem`
/// -> `algsolve::solve`. Attempts the chain on the RAW raster first (clean /
/// orthographic figures) and, if that does not close, on the RECTIFIED raster
/// (real phone-photo perspective skew). Both are genuine perceptions of the
/// SAME pixels; the first the K-gated solver closes to an exact `Value` wins.
/// Returns `Solved` ONLY on an exact solver `Value`; a perceived-but-unclosable
/// scene returns `Incomplete` with transparent provenance (so an honest refusal
/// can show perception worked + name the declining stage); an unperceivable
/// image returns `Nothing`. There is NO filename / hash / figure-name / answer
/// lookup: the bytes are the only input, the number is DERIVED. Read-only;
/// forms no bonds, mutates no brain -- classified guest-safe.
fn try_general_chain(bytes: &[u8]) -> GeneralOutcome {
    let img = match decode_image_rgb(bytes) {
        Some(i) => i,
        None => return GeneralOutcome::Nothing,
    };
    let mut diag: Option<Value> = None;
    // 1) raw raster
    match general_on_raster(&img, false) {
        GeneralOutcome::Solved(v) => return GeneralOutcome::Solved(v),
        GeneralOutcome::Incomplete(d) => {
            if diag.is_none() {
                diag = Some(d);
            }
        }
        GeneralOutcome::Nothing => {}
    }
    // 2) rectified raster (real-photo skew), when the outer square is present
    if let Some((rect, _skew)) = rectify(&img) {
        match general_on_raster(&rect, true) {
            GeneralOutcome::Solved(v) => return GeneralOutcome::Solved(v),
            GeneralOutcome::Incomplete(d) => {
                if diag.is_none() {
                    diag = Some(d);
                }
            }
            GeneralOutcome::Nothing => {}
        }
    }
    match diag {
        Some(d) => GeneralOutcome::Incomplete(d),
        None => GeneralOutcome::Nothing,
    }
}
/// Append general-chain context to an honest refusal message when the general
/// pipeline DID perceive a labelled scene (perception worked; the constraint
/// stage is what declined). Leaves the message untouched when there is no such
/// scene, preserving the plain honest-capability wording. Never adds a number.
fn augment_general(msg: String, general: &Option<Value>) -> String {
    match general {
        Some(g) => {
            let census = g.get("primitives_census").and_then(|v| v.as_str()).unwrap_or("");
            let reason = g.get("reason").and_then(|v| v.as_str()).unwrap_or("");
            format!(
                "{} NOTE: the general pipeline DID perceive a labelled scene from the pixels \
                 ({}), so perception worked; it is the constraint stage that declined: {}. \
                 Still no number (honesty floor).",
                msg, census, reason
            )
        }
        None => msg,
    }
}
/// PHASE 5 IMAGE MODE (pure; the bytes ARE the only input). Runs the real
/// perceiver and returns EITHER the exact verified answer + extracted-scene
/// provenance, OR an honest capability statement with NO number. Never a
/// hardcoded/looked-up value.
pub fn solve_image_bytes(bytes: &[u8]) -> Value {
    // P-E GENERAL CHAIN FIRST (pixels -> primitives -> labels -> constraints
    // -> algsolve). Returns a number ONLY on an exact algsolve Value; a
    // perceived-but-unclosable scene is remembered as transparent provenance
    // for the honest refusal; anything else falls through to the Phase-5 leaf
    // perceiver. Guest-safe pure compute; NO lookup.
    let general_diag = match try_general_chain(bytes) {
        GeneralOutcome::Solved(v) => return v,
        GeneralOutcome::Incomplete(d) => Some(d),
        GeneralOutcome::Nothing => None,
    };
    match perceive_photo(bytes) {
        PerceiveP4Outcome::Solved {
            class,
            side,
            cell,
            area,
            area_decimal,
            symbolic,
            confidence,
            weakest_tile,
            skew,
            ocr_digits,
            ocr_confidence,
            eye_measured,
            eye_r_measured,
        } => {
            let eye_note = if eye_measured {
                format!(
                    ", and MEASURED the interior eye arc from the pixels (r~={:.3} in side-units)",
                    eye_r_measured.unwrap_or(0.0)
                )
            } else {
                String::new()
            };
            let prose = format!(
                "I perceived a '{}' geometry figure directly from the image pixels \
                 (rectified a phone-photo skew of {:.3}), read the printed scale digit \
                 '{}' by OCR (confidence {:.2}){}, reconstructed the scene (side = {} \
                 units{}), and solved the shaded area exactly with the verified \
                 partition solver: {} = {:.4}. The decomposition verified (sub-regions \
                 reconstruct the whole with zero residual); weakest region '{}', overall \
                 confidence {:.2}. This number was derived pixels -> scene -> verified \
                 solver; it is NOT looked up or hardcoded.",
                class,
                skew,
                ocr_digits,
                ocr_confidence,
                eye_note,
                side,
                match cell {
                    Some(c) => format!(", 3x3 grid of {}-unit cells", c),
                    None => String::new(),
                },
                symbolic,
                area_decimal,
                weakest_tile,
                confidence,
            );
            wrap(
                prose,
                json!({
                    "source": "geometry_solver",
                    "mode": "image",
                    "solved": true,
                    "provenance": {
                        "perceiver": "geo_perceive_p4::perceive_photo",
                        "class": class,
                        "side": side,
                        "cell": cell,
                        "ocr_digits": ocr_digits,
                        "ocr_confidence": ocr_confidence,
                        "skew": skew,
                        "eye_measured": eye_measured,
                        "eye_r_measured": eye_r_measured,
                        "weakest_tile": weakest_tile,
                        "confidence": confidence,
                    },
                    "answer": {
                        "symbolic": symbolic,
                        "decimal": area_decimal,
                        "exact": serde_json::to_value(&area).unwrap_or(Value::Null),
                    },
                }),
            )
        }
        PerceiveP4Outcome::CantParse { stage, reason } => honest_image_statement(bytes, &stage, &reason, general_diag),
    }
}

/// The honest capability statement (#1544 form) for an image the perceiver
/// could not genuinely solve. Reports the REAL format/dimensions/size it can
/// truly read, the precise stage at which perception stopped, an explicit
/// "I can't interpret/solve this image", and -- critically -- NO number.
fn honest_image_statement(bytes: &[u8], stage: &str, reason: &str, general: Option<Value>) -> Value {
    let fmt = sniff_format(bytes);
    let dims = decode_image_rgb(bytes)
        .map(|im| format!("{}x{}", im.w, im.h))
        .unwrap_or_else(|| "undecodable-dimensions".to_string());
    let msg = format!(
        "This is {} ({}), {} bytes. I tried to solve it as a geometry figure by \
         perceiving the pixels, but I could not: perception stopped at the '{}' stage \
         ({}). I can't interpret or solve this image, and I will not guess a numeric \
         answer from an image I cannot genuinely parse. No number emitted (honesty floor).",
        fmt,
        dims,
        bytes.len(),
        stage,
        reason,
    );
    let msg = augment_general(msg, &general);
    wrap(
        format!(
            "lu4d_geometry_solve[image:honest_capability]: fmt={} dims={} bytes={} stage={} no_number",
            fmt,
            dims,
            bytes.len(),
            stage
        ),
        json!({
            "source": "geometry_solver",
            "mode": "image",
            "solved": false,
            "reason": "image_not_solvable",
            "perceiver_stage": stage,
            "perceiver_detail": reason,
            "format": fmt,
            "dimensions": dims,
            "byte_len": bytes.len(),
            "extracted_scene": general.is_some(),
            "general_chain": general.clone().unwrap_or(Value::Null),
            "message": msg,
        }),
    )
}

/// An image was referenced (by key/path) but no decodable inline bytes were
/// supplied. This pure tool does not read arbitrary filesystem paths; it asks
/// for the bytes inline. Honest, NO number.
fn honest_image_reference_no_bytes() -> Value {
    let msg = "You referenced an image but did not inline decodable image bytes. This \
        tool solves a geometry PHOTO by perceiving the pixels you send it (as base64 \
        under `image_base64`). It does not read arbitrary files, and it will not guess \
        a numeric answer without genuinely perceiving the image. Send the image bytes \
        as base64 in `image_base64`."
        .to_string();
    wrap(
        "lu4d_geometry_solve[image:no_inline_bytes]: reference present, no decodable image bytes; no number emitted".to_string(),
        json!({
            "source": "geometry_solver",
            "mode": "image",
            "solved": false,
            "reason": "image_referenced_but_not_inlined",
            "extracted_scene": false,
            "message": msg,
        }),
    )
}

/// The single real entry point wired into the guest surface as
/// `lu4d_geometry_solve`. Returns the wrapped MCP result (content +
/// structuredContent) or an MCP error tuple.
pub fn solve_scene_json(args: &Value) -> Result<Value, (i64, String)> {
    // PHASE 5 IMAGE MODE: if the caller inlined genuine image bytes, run the
    // real perceiver->verified-solver pipeline. The number flows pixels->scene
    // ->solver every time; an unparseable / non-geometry image returns an
    // honest capability statement with NO number (see solve_image_bytes).
    if let Some(bytes) = image_bytes_from_args(args) {
        return Ok(solve_image_bytes(&bytes));
    }
    // An image was REFERENCED but no decodable inline bytes were supplied. Do
    // not guess a number; ask for inline bytes. Honest, NO number.
    if looks_like_raw_image(args) {
        return Ok(honest_image_reference_no_bytes());
    }

    let scene = args
        .get("scene")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing required 'scene' string (one of: red_leaf, control, partition) or inline image bytes in 'image_base64'".to_string()))?;

    match scene {
        "red_leaf" => {
            let sol = solve_red_leaf();
            let solved = matches!(sol.outcome, VerifyOutcome::Solved { .. });
            let structured = serde_json::to_value(&sol).unwrap_or(Value::Null);
            Ok(wrap(
                sol.prose.clone(),
                json!({
                    "source": "geometry_solver",
                    "scene": "red_leaf",
                    "solved": solved,
                    "solution": structured,
                }),
            ))
        }
        "control" => {
            let t = args
                .get("tangent_cm")
                .and_then(|v| v.as_f64())
                .ok_or((-32602, "scene 'control' requires numeric 'tangent_cm'".to_string()))?;
            let c = solve_control(t);
            let structured = serde_json::to_value(&c).unwrap_or(Value::Null);
            Ok(wrap(
                c.prose.clone(),
                json!({
                    "source": "geometry_solver",
                    "scene": "control",
                    "solved": true,
                    "solution": structured,
                }),
            ))
        }
        "partition" => {
            let whole: Exact = args
                .get("whole")
                .ok_or((-32602, "scene 'partition' requires 'whole' (Exact a+b*pi)".to_string()))
                .and_then(|v| serde_json::from_value(v.clone()).map_err(|e| (-32602, format!("bad 'whole': {}", e))))?;
            let tiles: Vec<Tile> = args
                .get("tiles")
                .ok_or((-32602, "scene 'partition' requires 'tiles' array".to_string()))
                .and_then(|v| serde_json::from_value(v.clone()).map_err(|e| (-32602, format!("bad 'tiles': {}", e))))?;
            let target_role = args.get("target_role").and_then(|v| v.as_str()).unwrap_or("x");
            let idk_floor = args.get("idk_floor").and_then(|v| v.as_f64()).unwrap_or(0.40);
            // Guard against zero denominators (deserialize bypasses Rational::new,
            // whose constructor asserts den != 0). An invalid Exact is a bad input,
            // not a solvable scene -- refuse honestly rather than panic.
            if !den_ok(&whole) || tiles.iter().any(|t| !den_ok(&t.area)) {
                return Err((-32602, "invalid Exact: rational/pi denominator must be non-zero".to_string()));
            }
            let mut p = Partition::new(whole, target_role);
            for t in tiles {
                p.push(t);
            }
            let outcome = p.solve_and_verify(idk_floor);
            let outcome_json = serde_json::to_value(&outcome).unwrap_or(Value::Null);
            let (text, solved) = match &outcome {
                VerifyOutcome::Solved { target, target_decimal, weakest_tile, overall_confidence, .. } => (
                    format!(
                        "Shaded target '{}' = {} = {:.4} (exact a+b*pi). Decomposition verified: \
                         sum(known) + target reconstruct the whole with zero residual. Weakest tile: \
                         {} (confidence {:.2}); answer emitted above the IDK floor.",
                        target_role,
                        target.format(),
                        target_decimal,
                        weakest_tile,
                        overall_confidence,
                    ),
                    true,
                ),
                VerifyOutcome::Idk { unsure_region, unsure_confidence, recognized, total } => (
                    format!(
                        "I can't determine the shaded area: region '{}' is below the confidence floor \
                         ({:.2} < {:.2}); {}/{} regions clear it. No number emitted (K-gate / IDK).",
                        unsure_region, unsure_confidence, idk_floor, recognized, total,
                    ),
                    false,
                ),
                VerifyOutcome::Flagged { reason, residual } => (
                    format!(
                        "I can't return a number: the decomposition is inconsistent -- {} (residual {}). \
                         No number emitted (honesty gate).",
                        reason,
                        residual.format(),
                    ),
                    false,
                ),
            };
            Ok(wrap(
                text,
                json!({
                    "source": "geometry_solver",
                    "scene": "partition",
                    "solved": solved,
                    "outcome": outcome_json,
                }),
            ))
        }
        other => Err((-32602, format!("unknown scene kind '{}' (expected: red_leaf, control, partition)", other))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::geo_perceive::pngtest::encode_png_rgb;
    use serde_json::json;

    // ---- STRUCTURED SCENE (unchanged behaviour) --------------------------

    #[test]
    fn red_leaf_scene_solves_to_32pi_minus_64() {
        let out = solve_scene_json(&json!({"scene": "red_leaf"})).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(true));
        let target = &s["solution"]["outcome"]["target"];
        assert_eq!(target["rational"]["num"], json!(-64));
        assert_eq!(target["pi"]["num"], json!(32));
        let text = out["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("32*pi - 64"), "prose was: {}", text);
    }

    #[test]
    fn control_scene_solves_to_25() {
        let out = solve_scene_json(&json!({"scene": "control", "tangent_cm": 5.0})).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solution"]["area"], json!(25.0));
        assert_eq!(s["solution"]["identity_holds"], json!(true));
    }

    #[test]
    fn under_constrained_partition_idks_with_no_number() {
        // Mirrors the proven `low_confidence_tile_triggers_idk` case: a
        // load-bearing region below the floor must return Idk and NO number.
        let args = json!({
            "scene": "partition",
            "whole": {"rational": {"num": 144, "den": 1}, "pi": {"num": 0, "den": 1}},
            "target_role": "x",
            "idk_floor": 0.40,
            "tiles": [
                {"role": "ambiguous_arc", "formula": "square(8)-quarter(8)",
                 "area": {"rational": {"num": 64, "den": 1}, "pi": {"num": -16, "den": 1}}, "confidence": 0.30},
                {"role": "solid", "formula": "rect",
                 "area": {"rational": {"num": 16, "den": 1}, "pi": {"num": 0, "den": 1}}, "confidence": 0.95}
            ]
        });
        let out = solve_scene_json(&args).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(false));
        assert_eq!(s["outcome"]["kind"], json!("Idk"));
        assert_eq!(s["outcome"]["unsure_region"], json!("ambiguous_arc"));
        assert!(s["outcome"].get("target").is_none(), "Idk must carry no numeric target");
    }

    // ---- PHASE 5 IMAGE MODE ----------------------------------------------

    /// Test-only base64 encoder (independent of the crate decoder under test).
    fn b64encode(bytes: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0];
            let b1 = *chunk.get(1).unwrap_or(&0);
            let b2 = *chunk.get(2).unwrap_or(&0);
            out.push(T[(b0 >> 2) as usize] as char);
            out.push(T[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
            if chunk.len() > 1 {
                out.push(T[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(T[(b2 & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    /// The REAL red-leaf phone photo John sent. Fed as raw bytes: the number
    /// must flow pixels -> scene -> verified solver. Nothing is keyed on the
    /// filename; the bytes ARE the input.
    const REDLEAF_JPEG: &[u8] = include_bytes!("../../prototypes/perception/redleaf.jpeg");

    #[test]
    fn image_real_redleaf_solves_32pi_minus_64_with_provenance() {
        let out = solve_image_bytes(REDLEAF_JPEG);
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(true), "redleaf must solve; got {}", s);
        assert_eq!(s["mode"], json!("image"));
        assert_eq!(s["answer"]["symbolic"], json!("32*pi - 64"));
        let dec = s["answer"]["decimal"].as_f64().unwrap();
        assert!((dec - 36.5309649).abs() < 1e-4, "decimal was {}", dec);
        // Extracted-scene provenance is present and genuinely perceived.
        assert_eq!(s["provenance"]["class"], json!("grid_leaf"));
        assert_eq!(s["provenance"]["ocr_digits"], json!("4"));
        assert_eq!(s["provenance"]["eye_measured"], json!(true));
        let conf = s["provenance"]["confidence"].as_f64().unwrap();
        assert!(conf > 0.7 && conf < 0.85, "confidence was {}", conf);
        // Exact answer carries a+b*pi.
        assert_eq!(s["answer"]["exact"]["pi"]["num"], json!(32));
        assert_eq!(s["answer"]["exact"]["rational"]["num"], json!(-64));
    }

    #[test]
    fn image_base64_transport_via_solve_scene_json_solves() {
        let b64 = b64encode(REDLEAF_JPEG);
        let out = solve_scene_json(&json!({"image_base64": b64})).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(true));
        assert_eq!(s["answer"]["symbolic"], json!("32*pi - 64"));
        assert_eq!(s["provenance"]["class"], json!("grid_leaf"));
    }

    #[test]
    fn image_data_url_prefix_is_tolerated() {
        let b64 = format!("data:image/jpeg;base64,{}", b64encode(REDLEAF_JPEG));
        let out = solve_scene_json(&json!({"image": b64})).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(true));
        assert_eq!(s["answer"]["symbolic"], json!("32*pi - 64"));
    }

    #[test]
    fn image_garbage_bytes_honest_no_number() {
        // Not an image at all -> decode stage CantParse -> honest, NO number.
        let out = solve_image_bytes(b"this is a sentence about dogs, not an image at all");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(false));
        assert_eq!(s["reason"], json!("image_not_solvable"));
        assert!(s.get("answer").is_none(), "honest statement must carry NO answer");
        let msg = s["message"].as_str().unwrap();
        assert!(msg.contains("will not guess"), "msg: {}", msg);
        assert!(msg.contains("can't interpret or solve"), "msg: {}", msg);
    }

    #[test]
    fn image_blank_non_geometry_png_honest_no_number() {
        // A real, decodable PNG with NO geometry figure. Perception decodes it
        // (real dimensions reported) but finds no shaded boundary -> honest,
        // NO number.
        let (w, h) = (120usize, 120usize);
        let rgb = vec![255u8; w * h * 3]; // solid white
        let png = encode_png_rgb(w, h, &rgb);
        let out = solve_image_bytes(&png);
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(false));
        assert!(s.get("answer").is_none(), "non-geometry image must carry NO answer");
        assert_eq!(s["format"], json!("PNG image"));
        assert_eq!(s["dimensions"], json!("120x120"), "must report REAL dims it can read");
    }

    #[test]
    fn image_reference_without_bytes_is_honest_no_number() {
        // A path reference with no inline bytes: this pure tool does not read
        // arbitrary files. Honest, NO number. (Replaces the pre-Phase-5
        // blanket raw-image refusal.)
        let out = solve_scene_json(&json!({"image_path": "/tmp/diagram.png"})).expect("ok");
        let s = &out["structuredContent"];
        assert_eq!(s["solved"], json!(false));
        assert_eq!(s["reason"], json!("image_referenced_but_not_inlined"));
        assert!(s.get("answer").is_none(), "must carry NO number");
    }
// ---- P-E GENERAL CHAIN through the guest image path ------------------
    // Proves solve_image_bytes now runs the GENERAL chain (not just the 3 leaf
    // templates): a faithful render of John's tilted-square + corner-incircle
    // figure closes pixels -> primitives -> labels -> constraints -> algsolve =
    // 4*pi with provenance; redleaf still solves via the leaf path; garbage is
    // still an honest no-number. The renderer is the proven `render_john` from
    // geo_constraints tests (identical pixels shown to derive 4*pi there).
    mod pe_general {
        use crate::math::geometry_tool::solve_image_bytes;
        use crate::vision::geo_perceive::pngtest::encode_png_rgb;
        use serde_json::json;

        const BLACK: (u8, u8, u8) = (0, 0, 0);
        struct Canvas { w: usize, h: usize, rgb: Vec<u8> }
        impl Canvas {
            fn new(w: usize, h: usize) -> Self { Canvas { w, h, rgb: vec![255u8; w * h * 3] } }
            fn set(&mut self, x: i64, y: i64) {
                if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h { return; }
                let i = (y as usize * self.w + x as usize) * 3;
                self.rgb[i] = BLACK.0; self.rgb[i + 1] = BLACK.1; self.rgb[i + 2] = BLACK.2;
            }
            fn seg(&mut self, a: (f64, f64), b: (f64, f64), hw: i64) {
                let len = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
                let steps = (len.ceil() as i64).max(1);
                for s in 0..=steps {
                    let f = s as f64 / steps as f64;
                    let x = a.0 + (b.0 - a.0) * f; let y = a.1 + (b.1 - a.1) * f;
                    for dy in -hw..=hw { for dx in -hw..=hw { self.set(x.round() as i64 + dx, y.round() as i64 + dy); } }
                }
            }
            fn circle(&mut self, cx: f64, cy: f64, r: f64) {
                let steps = (2.0 * std::f64::consts::PI * r).ceil() as i64 * 2;
                for s in 0..steps {
                    let a = 2.0 * std::f64::consts::PI * (s as f64) / (steps as f64);
                    let x = cx + r * a.cos(); let y = cy + r * a.sin();
                    for dy in -1..=1 { for dx in -1..=1 { self.set(x.round() as i64 + dx, y.round() as i64 + dy); } }
                }
            }
            fn polygon(&mut self, verts: &[(f64, f64)], hw: i64) {
                for i in 0..verts.len() { self.seg(verts[i], verts[(i + 1) % verts.len()], hw); }
            }
            fn stamp(&mut self, rows: &[&str; 7], x0: i64, y0: i64, cell: i64) {
                for (ry, row) in rows.iter().enumerate() {
                    for (rx, ch) in row.chars().enumerate() {
                        if ch == '1' || ch == '#' {
                            for dy in 0..cell { for dx in 0..cell { self.set(x0 + rx as i64 * cell + dx, y0 + ry as i64 * cell + dy); } }
                        }
                    }
                }
            }
            fn text(&mut self, s: &str, x0: i64, y0: i64, cell: i64) {
                let mut x = x0;
                for ch in s.chars() { self.stamp(&glyph_rows(ch), x, y0, cell); x += 7 * cell; }
            }
            fn png(&self) -> Vec<u8> { encode_png_rgb(self.w, self.h, &self.rgb) }
        }

        const DIGIT5X7: [[&str; 7]; 10] = [
            ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
            ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
            ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
            ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
            ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
            ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
            ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
            ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
            ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
            ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
        ];
        const QUESTION_FONT: [&str; 7] = ["01110", "10001", "00010", "00100", "00100", "00100", "00100"];
        fn glyph_rows(ch: char) -> [&'static str; 7] {
            if ch == '?' { return QUESTION_FONT; }
            if let Some(d) = ch.to_digit(10) { return DIGIT5X7[d as usize]; }
            panic!("no font for {}", ch);
        }
        fn framed(n: usize, m: f64) -> (Canvas, f64, f64) {
            let mut c = Canvas::new(n, n);
            let lo = m; let hi = n as f64 - 1.0 - m;
            c.polygon(&[(lo, lo), (hi, lo), (hi, hi), (lo, hi)], 2);
            (c, lo, hi)
        }
        // Faithful John's-FullSizeRender geometry (same construction proven to
        // derive 4*pi in geo_constraints tests): outer square S=14u, inscribed
        // tilted square (touch points 8/14 and 6/14 => area 100), corner triangle
        // (legs 8,6 => area 24), incircle of that corner (r=2) as the '?'.
        fn render_john(n: usize, tilt_label: &str, tri_label: &str, tangent: bool) -> Vec<u8> {
            let m = 40.0;
            let (mut c, lo, hi) = framed(n, m);
            let sc = (hi - lo) / 14.0;
            let p_top = (lo + 8.0 * sc, lo);
            let p_right = (hi, lo + 8.0 * sc);
            let p_bot = (lo + 6.0 * sc, hi);
            let p_left = (lo, lo + 6.0 * sc);
            c.polygon(&[p_top, p_right, p_bot, p_left], 2);
            let mid = (lo + hi) / 2.0;
            if tangent {
                c.circle(lo + 2.0 * sc, lo + 2.0 * sc, 2.0 * sc);
                c.text("?", (lo + 2.0 * sc - 14.0) as i64, (lo + 2.0 * sc - 14.0) as i64, 4);
            } else {
                c.circle(mid, mid, (hi - lo) * 0.13);
                c.text("?", (mid - 8.0) as i64, (mid - 14.0) as i64, 4);
            }
            let tl_y = if tangent { mid - 14.0 } else { lo + (hi - lo) * 0.80 };
            c.text(tilt_label, (mid - 42.0) as i64, tl_y as i64, 4);
            c.text(tri_label, (lo + 5.0 * sc) as i64, (lo + 0.4 * sc) as i64, 3);
            c.png()
        }

        // MILESTONE: the faithful figure closes pixels -> 4*pi through the LIVE
        // guest image entry point, via the GENERAL chain, with provenance.
        #[test]
        fn general_chain_solves_faithful_tilted_square_4pi() {
            let png = render_john(640, "100", "24", true);
            let out = solve_image_bytes(&png);
            let s = &out["structuredContent"];
            assert_eq!(s["solved"], json!(true), "faithful tilted-square must solve; got {}", s);
            assert_eq!(s["pipeline"], json!("general_chain"), "must solve via the GENERAL chain; got {}", s);
            assert_eq!(s["answer"]["symbolic"], json!("4*pi"));
            let dec = s["answer"]["decimal"].as_f64().unwrap();
            assert!((dec - std::f64::consts::PI * 4.0).abs() < 1e-4, "decimal was {}", dec);
            let assembled = s["provenance"]["assembled_system"].as_str().unwrap();
            assert!(assembled.contains("PtLineDistEqCoord"), "provenance must show the tangency constraint");
            assert!(assembled.contains("Perp"), "provenance must show the perp constraint");
            // provenance carries the bound labels genuinely read from the pixels.
            assert!(s["provenance"]["bindings"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
        }

        // ANTI-CHEAT: identical geometry relabelled 225/54 derives 9*pi (computed,
        // not looked up) through the same guest entry point.
        #[test]
        fn general_chain_rescaled_labels_derive_9pi() {
            let png = render_john(640, "225", "54", true);
            let out = solve_image_bytes(&png);
            let s = &out["structuredContent"];
            assert_eq!(s["solved"], json!(true), "rescaled must solve; got {}", s);
            assert_eq!(s["pipeline"], json!("general_chain"));
            assert_eq!(s["answer"]["symbolic"], json!("9*pi"), "must derive 9*pi from the read labels, not the 4*pi base");
        }

        // HONESTY: a non-tangent circle (tangent to nothing) must NOT be forced;
        // the general chain returns None -> leaf path -> honest, NO number.
        #[test]
        fn general_chain_non_tangent_circle_no_number() {
            let png = render_john(640, "100", "24", false);
            let out = solve_image_bytes(&png);
            let s = &out["structuredContent"];
            assert_ne!(s["pipeline"], json!("general_chain"), "non-tangent circle must NOT be claimed by the general chain");
            // Either the leaf path also can't parse (honest, no number) -- the
            // required floor is simply that NO general-chain number is emitted.
            if s["solved"] == json!(false) {
                assert!(s.get("answer").is_none(), "an honest statement carries NO number");
            }
        }

        // redleaf still solves via the LEAF perceiver (a+b*pi leaf area is not an
        // algsolve length/disk target, so the general chain declines it).
        #[test]
        fn redleaf_still_solves_via_leaf_not_general() {
            let out = solve_image_bytes(super::REDLEAF_JPEG);
            let s = &out["structuredContent"];
            assert_eq!(s["solved"], json!(true));
            assert_eq!(s["answer"]["symbolic"], json!("32*pi - 64"));
            assert_eq!(s["provenance"]["class"], json!("grid_leaf"));
            assert_ne!(s["pipeline"], json!("general_chain"), "redleaf must solve via the LEAF path, not the general chain");
        }

        // garbage bytes: honest capability statement, NO number, general chain off.
        #[test]
        fn garbage_image_honest_no_number_general_off() {
            let out = solve_image_bytes(b"this is a sentence about dogs, not an image at all");
            let s = &out["structuredContent"];
            assert_eq!(s["solved"], json!(false));
            assert!(s.get("answer").is_none(), "honest statement must carry NO number");
            assert_ne!(s["pipeline"], json!("general_chain"));
        }
    }
}
