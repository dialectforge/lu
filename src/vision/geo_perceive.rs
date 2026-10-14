//! Real pixel-based perception for the red-leaf-in-a-square figure class.
//!
//! internal notes; Python reference prototype
//! `prototypes/perception/perceive.py` (6/6 synthetic, 4/5 on the real
//! `redleaf.jpeg`). This is a pure-Rust port of the CORE of that perceiver:
//!
//!   PNG bytes
//!     -> decode to RGB pixels (flate2 inflate + PNG unfilter; no `image` crate)
//!     -> detect the bounding square = extent of the dark "ink" pixels
//!     -> detect the shaded region = the saturated-colour pixels; take its
//!        boundary and drop the points lying on the straight square edges
//!     -> per square-corner, MEASURE the fraction of shaded-boundary points at
//!        distance ~= side and the median radius; SNAP the radius to a small
//!        rational in square-side units and VERIFY (accept only if the raw
//!        pixel measurement already agreed -- never force a snap)
//!     -> recognise the topology: two arcs at OPPOSITE corners each radius =
//!        1*side  ==>  the red-leaf (vesica); one arc  ==>  a quarter-circle
//!     -> build a verified `Partition` and solve for the shaded area EXACTLY
//!        as `a + b*pi` through the REAL solver.
//!
//! HONESTY FLOOR (absolute -- this is the whole point):
//!   * The scene is derived ONLY from the pixel bytes (plus an integer side
//!     length `S` for the numeric scale). There is NO filename, path, image
//!     hash, or answer lookup ANYWHERE on this path -- the perceiver is handed
//!     `&[u8]` and never learns a name. Change the pixels and the scene
//!     changes; change nothing and it cannot return a memorised number.
//!   * A figure it cannot parse -> `PerceiveOutcome::CantParse`, NO number.
//!   * A non-PNG / non-geometry image -> `CantParse` with an honest reason,
//!     never a fabricated fallback and never the hardcoded `32*pi - 64`.
//!   * The numeric answer flows pixels -> measured structure -> `Partition` ->
//!     `solve_and_verify` (the same sum-to-whole verifier + IDK/K-gate every
//!     other scene uses). It is computed, never selected.

use crate::math::geometry::{
    square, square_minus_quarter_circle, Arc, Exact, Partition, Rational, Scene, Tile,
    VerifyOutcome,
};
use crate::vision::extractor::{ExtractError, ExtractHints, SceneGraphExtractor, Triplet};
use serde::{Deserialize, Serialize};

// =====================================================================
// PNG decode (pure Rust: flate2 inflate + PNG unfilter). PNG-only is an
// explicit, honest limit for Phase 1 (JPEG decode is a Phase-4 item).
// =====================================================================

/// A decoded, de-filtered RGB8 image. `rgb.len() == w * h * 3`.
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

impl RgbImage {
    #[inline]
    fn at(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let i = (y * self.w + x) * 3;
        (self.rgb[i], self.rgb[i + 1], self.rgb[i + 2])
    }
}

#[inline]
fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Decode an 8-bit, non-interlaced PNG (colour types 0 grey, 2 RGB, 4
/// grey+alpha, 6 RGBA) to RGB8. Returns `None` if the bytes are not such a
/// PNG. Deliberately narrow: we accept exactly the shapes our figures use and
/// refuse everything else honestly rather than half-decode.
pub fn decode_png_rgb(bytes: &[u8]) -> Option<RgbImage> {
    const SIG: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() < 8 || bytes[0..8] != SIG {
        return None;
    }
    let mut i = 8usize;
    let mut w = 0usize;
    let mut h = 0usize;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut interlace = 0u8;
    let mut have_ihdr = false;
    let mut idat: Vec<u8> = Vec::new();
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let ctype = &bytes[i + 4..i + 8];
        let data_start = i + 8;
        let data_end = data_start.checked_add(len)?;
        if data_end + 4 > bytes.len() {
            return None;
        }
        match ctype {
            b"IHDR" => {
                if len < 13 {
                    return None;
                }
                w = u32::from_be_bytes([
                    bytes[data_start],
                    bytes[data_start + 1],
                    bytes[data_start + 2],
                    bytes[data_start + 3],
                ]) as usize;
                h = u32::from_be_bytes([
                    bytes[data_start + 4],
                    bytes[data_start + 5],
                    bytes[data_start + 6],
                    bytes[data_start + 7],
                ]) as usize;
                bit_depth = bytes[data_start + 8];
                color_type = bytes[data_start + 9];
                interlace = bytes[data_start + 12];
                have_ihdr = true;
            }
            b"IDAT" => idat.extend_from_slice(&bytes[data_start..data_end]),
            b"IEND" => break,
            _ => {}
        }
        i = data_end + 4; // skip CRC
    }
    if !have_ihdr || w == 0 || h == 0 || bit_depth != 8 || interlace != 0 {
        return None;
    }
    let channels = match color_type {
        0 => 1usize, // greyscale
        2 => 3usize, // truecolour
        4 => 2usize, // greyscale + alpha
        6 => 4usize, // truecolour + alpha
        _ => return None,
    };
    // Inflate the concatenated IDAT stream (zlib).
    use std::io::Read;
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(&idat[..])
        .read_to_end(&mut raw)
        .ok()?;
    let stride = w.checked_mul(channels)?;
    if raw.len() < (stride + 1).checked_mul(h)? {
        return None;
    }
    // Un-filter, row by row.
    let bpp = channels; // 8-bit => bytes-per-pixel == channels
    let mut cur = vec![0u8; stride];
    let mut prev = vec![0u8; stride];
    let mut rgb = Vec::with_capacity(w * h * 3);
    let mut p = 0usize;
    for _row in 0..h {
        let filter = raw[p];
        p += 1;
        cur.copy_from_slice(&raw[p..p + stride]);
        p += stride;
        match filter {
            0 => {}
            1 => {
                for x in bpp..stride {
                    cur[x] = cur[x].wrapping_add(cur[x - bpp]);
                }
            }
            2 => {
                for x in 0..stride {
                    cur[x] = cur[x].wrapping_add(prev[x]);
                }
            }
            3 => {
                for x in 0..stride {
                    let a = if x >= bpp { cur[x - bpp] as u16 } else { 0 };
                    let b = prev[x] as u16;
                    cur[x] = cur[x].wrapping_add(((a + b) / 2) as u8);
                }
            }
            4 => {
                for x in 0..stride {
                    let a = if x >= bpp { cur[x - bpp] as i32 } else { 0 };
                    let b = prev[x] as i32;
                    let c = if x >= bpp { prev[x - bpp] as i32 } else { 0 };
                    cur[x] = cur[x].wrapping_add(paeth(a, b, c) as u8);
                }
            }
            _ => return None,
        }
        // Emit RGB.
        for x in 0..w {
            let o = x * channels;
            let (r, g, b) = match channels {
                1 | 2 => (cur[o], cur[o], cur[o]),
                3 | 4 => (cur[o], cur[o + 1], cur[o + 2]),
                _ => unreachable!(),
            };
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
        std::mem::swap(&mut cur, &mut prev);
    }
    Some(RgbImage { w, h, rgb })
}

// =====================================================================
// Perception
// =====================================================================

#[inline]
fn is_ink(r: u8, g: u8, b: u8) -> bool {
    // Dark line-work: low brightness and low saturation.
    let mx = r.max(g).max(b) as i32;
    let mn = r.min(g).min(b) as i32;
    mx < 110 && (mx - mn) < 60
}

#[inline]
fn is_colored(r: u8, g: u8, b: u8) -> bool {
    // A saturated fill (colour-agnostic, matching the prototype): clearly
    // coloured, not white and not black.
    let mx = r.max(g).max(b) as i32;
    let mn = r.min(g).min(b) as i32;
    (mx - mn) > 45 && mx > 60
}

/// One measured arc hypothesis at a square corner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CornerArc {
    pub corner: String,      // "TL" | "TR" | "BL" | "BR"
    pub frac_on_arc: f64,    // fraction of shaded-boundary points at r~=side
    pub radius_ratio: f64,   // median measured radius / side (snap target 1.0)
    pub snap_dist: f64,      // |radius_ratio - 1.0|
    pub accepted: bool,      // K-gate: snap_dist < tol
}

/// The recognised figure class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FigureClass {
    RedLeaf, // two opposite-corner arcs, radius = side
    Quarter, // one corner arc, radius = side
}

/// What the perceiver extracted from the pixels (before any numeric scale).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceivedScene {
    pub class: FigureClass,
    pub side_px: f64,
    pub square_squareness: f64, // |w-h|/max(w,h); small == square
    pub arcs: Vec<CornerArc>,   // the accepted arcs
    pub boundary_points: usize,
}

/// Result of perceiving + (optionally) scaling to a numeric answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerceiveOutcome {
    /// Structure recognised AND a numeric side length supplied -> exact answer.
    Solved {
        scene: PerceivedScene,
        side: i64,
        area: Exact,
        area_decimal: f64,
        symbolic: String,
        confidence: f64,
    },
    /// Structure recognised but NO side length supplied -> honest: structure +
    /// symbolic form, NO number.
    StructureOnly {
        scene: PerceivedScene,
        symbolic_in_s: String,
    },
    /// Could not extract a supported figure. NO number, honest reason.
    CantParse { reason: String },
}

const K_TOL: f64 = 0.06; // snap-then-verify tolerance in side-units
const ARC_BAND: f64 = 0.08; // radius band (in side-units) counted "on the arc"
// Two opposite-corner arcs of radius = side intersect exactly at the OTHER two
// corners, so those spurious corners pick up ~13% of the boundary in the
// radius band; the true arc centres carry >50%. 0.25 cleanly separates them.
const ARC_MIN_FRAC: f64 = 0.25; // min boundary fraction for a corner to hold an arc

/// The core perceiver. `png_bytes` are the ONLY perceptual input; `side_hint`
/// is the integer side length for the numeric scale (the printed "4"/"8"). If
/// `side_hint` is `None`, the structure is still reported but no number is
/// emitted (honesty floor). OCR of the printed digit is a Phase-4 item.
pub fn perceive_red_leaf_png(png_bytes: &[u8], side_hint: Option<i64>) -> PerceiveOutcome {
    let img = match decode_png_rgb(png_bytes) {
        Some(im) => im,
        None => {
            return PerceiveOutcome::CantParse {
                reason: "not a decodable 8-bit non-interlaced PNG (Phase-1 supports PNG only; \
                         JPEG/skew are Phase-4)"
                    .to_string(),
            }
        }
    };
    if img.w < 24 || img.h < 24 {
        return PerceiveOutcome::CantParse {
            reason: format!("image too small to perceive ({}x{})", img.w, img.h),
        };
    }

    // --- 1. Square = extent of the dark ink pixels. ---
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut ink = 0usize;
    for y in 0..img.h {
        for x in 0..img.w {
            let (r, g, b) = img.at(x, y);
            if is_ink(r, g, b) {
                ink += 1;
                if x < x0 {
                    x0 = x;
                }
                if y < y0 {
                    y0 = y;
                }
                if x > x1 {
                    x1 = x;
                }
                if y > y1 {
                    y1 = y;
                }
            }
        }
    }
    if ink < 40 || x1 <= x0 || y1 <= y0 {
        return PerceiveOutcome::CantParse {
            reason: "no dark bounding square found".to_string(),
        };
    }
    let ww = (x1 - x0) as f64;
    let hh = (y1 - y0) as f64;
    let squareness = (ww - hh).abs() / ww.max(hh);
    if squareness > 0.20 {
        return PerceiveOutcome::CantParse {
            reason: format!(
                "bounding ink region is not square (w={}, h={}, skew={:.2}); \
                 perspective rectification is Phase-4",
                ww as i64, hh as i64, squareness
            ),
        };
    }
    let side_px = 0.5 * (ww + hh);
    if side_px < 20.0 {
        return PerceiveOutcome::CantParse {
            reason: "bounding square too small to measure arcs".to_string(),
        };
    }

    // --- 2. Shaded region + its boundary (drop the straight square edges). ---
    let edge_margin = (side_px * 0.03).max(3.0);
    let mut boundary: Vec<(f64, f64)> = Vec::new();
    let mut colored = 0usize;
    for y in 0..img.h {
        for x in 0..img.w {
            let (r, g, b) = img.at(x, y);
            if !is_colored(r, g, b) {
                continue;
            }
            colored += 1;
            // boundary = coloured pixel adjacent to a non-coloured pixel.
            let mut is_edge = false;
            let neigh = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neigh {
                if nx >= img.w || ny >= img.h {
                    is_edge = true;
                    break;
                }
                let (nr, ng, nb) = img.at(nx, ny);
                if !is_colored(nr, ng, nb) {
                    is_edge = true;
                    break;
                }
            }
            if !is_edge {
                continue;
            }
            let (fx, fy) = (x as f64, y as f64);
            // drop points on/near the straight square border
            if (fx - x0 as f64).abs() < edge_margin
                || (fx - x1 as f64).abs() < edge_margin
                || (fy - y0 as f64).abs() < edge_margin
                || (fy - y1 as f64).abs() < edge_margin
            {
                continue;
            }
            boundary.push((fx, fy));
        }
    }
    if colored < 50 {
        return PerceiveOutcome::CantParse {
            reason: "no shaded region found (blank square or non-geometry image)".to_string(),
        };
    }
    if boundary.len() < 30 {
        return PerceiveOutcome::CantParse {
            reason: "shaded region has no curved (arc) boundary to measure".to_string(),
        };
    }

    // --- 3. Per-corner arc measurement + snap-then-verify K-gate. ---
    let corners = [
        ("TL", x0 as f64, y0 as f64),
        ("TR", x1 as f64, y0 as f64),
        ("BL", x0 as f64, y1 as f64),
        ("BR", x1 as f64, y1 as f64),
    ];
    let mut arcs: Vec<CornerArc> = Vec::new();
    for (name, cx, cy) in corners {
        let mut on_arc: Vec<f64> = Vec::new();
        for &(px, py) in &boundary {
            let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() / side_px;
            if (d - 1.0).abs() < ARC_BAND {
                on_arc.push(d);
            }
        }
        let frac = on_arc.len() as f64 / boundary.len() as f64;
        if frac < ARC_MIN_FRAC {
            continue;
        }
        on_arc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = on_arc[on_arc.len() / 2];
        let snap_dist = (median - 1.0).abs();
        arcs.push(CornerArc {
            corner: name.to_string(),
            frac_on_arc: frac,
            radius_ratio: median,
            snap_dist,
            accepted: snap_dist < K_TOL,
        });
    }
    let accepted: Vec<&CornerArc> = arcs.iter().filter(|a| a.accepted).collect();

    // --- 4. Topology recognition. ---
    let opposite = |a: &str, b: &str| -> bool {
        matches!(
            (a, b),
            ("TL", "BR") | ("BR", "TL") | ("TR", "BL") | ("BL", "TR")
        )
    };
    let class = if accepted.len() == 2 && opposite(&accepted[0].corner, &accepted[1].corner) {
        FigureClass::RedLeaf
    } else if accepted.len() == 1 {
        FigureClass::Quarter
    } else {
        return PerceiveOutcome::CantParse {
            reason: format!(
                "unrecognised arc configuration: {} accepted corner-arc(s) {:?}; \
                 supported Phase-1 classes are the two-opposite-corner leaf and \
                 the single quarter-circle",
                accepted.len(),
                accepted.iter().map(|a| &a.corner).collect::<Vec<_>>()
            ),
        };
    };

    // confidence from how tightly the measured radius snapped (worst arc).
    let worst_dev = accepted
        .iter()
        .map(|a| a.snap_dist)
        .fold(0.0_f64, f64::max);
    let confidence = (1.0 - worst_dev / K_TOL).clamp(0.0, 1.0) * 0.5 + 0.45;

    let scene = PerceivedScene {
        class: class.clone(),
        side_px,
        square_squareness: squareness,
        arcs: accepted.iter().map(|a| (*a).clone()).collect(),
        boundary_points: boundary.len(),
    };

    // --- 5. Scale + solve exactly through the REAL Partition verifier. ---
    let side = match side_hint {
        Some(s) if s > 0 => s,
        _ => {
            let sym = match class {
                FigureClass::RedLeaf => "(pi/2 - 1)*S^2".to_string(),
                FigureClass::Quarter => "(pi/4)*S^2".to_string(),
            };
            return PerceiveOutcome::StructureOnly {
                scene,
                symbolic_in_s: sym,
            };
        }
    };

    let s_rat = Rational::int(side);
    let whole = square(s_rat); // S^2
    let mut part = Partition::new(
        whole,
        match class {
            FigureClass::RedLeaf => "red_leaf",
            FigureClass::Quarter => "quarter_disk",
        },
    );
    match class {
        FigureClass::RedLeaf => {
            // square = wingA + wingB + leaf, each wing = square - quarter(S).
            let wing = square_minus_quarter_circle(s_rat, s_rat);
            part.push(Tile::new(
                "wing_A",
                "square(S) - quarter_circle(S)",
                wing,
                confidence,
            ));
            part.push(Tile::new(
                "wing_B",
                "square(S) - quarter_circle(S)",
                wing,
                confidence,
            ));
        }
        FigureClass::Quarter => {
            // square = quarter_disk + (square - quarter).
            let comp = square_minus_quarter_circle(s_rat, s_rat);
            part.push(Tile::new(
                "corner_complement",
                "square(S) - quarter_circle(S)",
                comp,
                confidence,
            ));
        }
    }
    let outcome = part.solve_and_verify(0.40);
    match outcome {
        VerifyOutcome::Solved {
            target,
            target_decimal,
            ..
        } => PerceiveOutcome::Solved {
            scene,
            side,
            area: target,
            area_decimal: target_decimal,
            symbolic: target.format(),
            confidence,
        },
        // A clean detection above the 0.40 floor should always solve; if the
        // verifier IDKs/Flags we honour it and emit NO number.
        VerifyOutcome::Idk {
            unsure_confidence, ..
        } => PerceiveOutcome::CantParse {
            reason: format!(
                "perception below confidence floor (arc match {:.2}); no number emitted",
                unsure_confidence
            ),
        },
        VerifyOutcome::Flagged { reason, .. } => PerceiveOutcome::CantParse {
            reason: format!("decomposition inconsistent: {}", reason),
        },
    }
}

/// Build a `Scene` (topology + scale) from a perceived structure, for callers
/// that want the DSL record. Arcs are placed at their corners with radius =
/// side in absolute units.
pub fn scene_from_perceived(scene: &PerceivedScene, side: f64) -> Scene {
    let mut s = Scene::new(side, side, side);
    for a in &scene.arcs {
        let (cx, cy) = match a.corner.as_str() {
            "TL" => (0.0, side),
            "TR" => (side, side),
            "BL" => (0.0, 0.0),
            "BR" => (side, 0.0),
            _ => (0.0, 0.0),
        };
        s.add_arc(Arc {
            cx,
            cy,
            r: side,
            start_deg: 0.0,
            end_deg: 90.0,
        });
    }
    s
}

// =====================================================================
// SceneGraphExtractor citizen: a REAL extractor (perceives_contents == true)
// =====================================================================

/// A real pixel-perceiving extractor for the red-leaf figure class. Unlike
/// `StubExtractor`, it genuinely reads the image bytes, so `perceives_contents`
/// is the trait default `true`. It emits structure triplets; the numeric
/// geometry answer is produced by `perceive_red_leaf_png` (which needs the
/// side length `S`).
pub struct RealGeoExtractor;

impl SceneGraphExtractor for RealGeoExtractor {
    fn name(&self) -> &str {
        "geo_redleaf"
    }

    fn extract(
        &self,
        image_bytes: &[u8],
        _hints: ExtractHints,
    ) -> Result<Vec<Triplet>, ExtractError> {
        match perceive_red_leaf_png(image_bytes, None) {
            PerceiveOutcome::CantParse { reason } => Err(ExtractError::Parse(reason)),
            PerceiveOutcome::StructureOnly { scene, .. }
            | PerceiveOutcome::Solved { scene, .. } => {
                let class = match scene.class {
                    FigureClass::RedLeaf => "red_leaf_in_square",
                    FigureClass::Quarter => "quarter_circle_in_square",
                };
                let mut out = vec![Triplet {
                    subject: "figure".to_string(),
                    predicate: "is_a".to_string(),
                    object: class.to_string(),
                    confidence: 0.9,
                }];
                for a in &scene.arcs {
                    out.push(Triplet {
                        subject: format!("arc@{}", a.corner),
                        predicate: "radius_ratio_to_side".to_string(),
                        object: format!("{:.3}", a.radius_ratio),
                        confidence: a.frac_on_arc as f32,
                    });
                }
                Ok(out)
            }
        }
    }
}

// =====================================================================
// Minimal base64 decode (for the guest surface, which passes image bytes as
// base64). Standard alphabet; ignores ASCII whitespace; tolerant of missing
// padding. Kept tiny and dependency-free.
// =====================================================================

pub fn decode_base64(s: &str) -> Option<Vec<u8>> {
    #[inline]
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut acc = 0u32;
    let mut nbits = 0u32;
    let mut out = Vec::new();
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c)?;
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Some(out)
}

// =====================================================================
// Tests -- includes an in-crate PNG ENCODER so every anti-cheat input is a
// genuine, freshly-drawn image the perceiver reads back from pixels. Nothing
// is keyed on a filename; the bytes ARE the input.
// =====================================================================

#[cfg(test)]
pub mod pngtest {
    use super::*;
    use std::io::Write;

    /// Encode an RGB8 image as a PNG (filter 0, one zlib IDAT). Uses the
    /// crate's flate2 + crc32fast; independent of the decoder above.
    pub fn encode_png_rgb(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
        fn crc(bytes: &[u8]) -> u32 {
            let mut h = crc32fast::Hasher::new();
            h.update(bytes);
            h.finalize()
        }
        fn chunk(out: &mut Vec<u8>, ctype: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(ctype);
            out.extend_from_slice(data);
            let mut crcbuf = Vec::with_capacity(4 + data.len());
            crcbuf.extend_from_slice(ctype);
            crcbuf.extend_from_slice(data);
            out.extend_from_slice(&crc(&crcbuf).to_be_bytes());
        }
        let mut out = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, RGB, deflate, no filter, no interlace
        chunk(&mut out, b"IHDR", &ihdr);
        // filtered raw scanlines (filter byte 0 per row)
        let mut raw = Vec::with_capacity(h * (1 + w * 3));
        for y in 0..h {
            raw.push(0);
            raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        }
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&raw).unwrap();
        let idat = enc.finish().unwrap();
        chunk(&mut out, b"IDAT", &idat);
        chunk(&mut out, b"IEND", &[]);
        out
    }

    /// Draw a red-leaf-in-a-square: white bg, black square border, red fill =
    /// intersection of the two quarter-disks (radius = side) at OPPOSITE
    /// corners. `side_px` pixels for the square, `pad` border of whitespace.
    pub fn draw_red_leaf(side_px: usize, pad: usize) -> (usize, usize, Vec<u8>) {
        draw_figure(side_px, pad, true)
    }
    /// Draw a single quarter-circle (one corner arc, radius = side) shaded.
    pub fn draw_quarter(side_px: usize, pad: usize) -> (usize, usize, Vec<u8>) {
        draw_figure(side_px, pad, false)
    }

    fn draw_figure(side_px: usize, pad: usize, leaf: bool) -> (usize, usize, Vec<u8>) {
        let w = side_px + 2 * pad;
        let h = side_px + 2 * pad;
        let mut rgb = vec![255u8; w * h * 3]; // white
        let x0 = pad as f64;
        let y0 = pad as f64;
        let x1 = (pad + side_px) as f64;
        let y1 = (pad + side_px) as f64;
        let s = side_px as f64;
        // opposite corners TL and BR
        let (tlx, tly) = (x0, y0);
        let (brx, bry) = (x1, y1);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let inside_sq = fx >= x0 && fx <= x1 && fy >= y0 && fy <= y1;
                let mut set = |r: u8, g: u8, b: u8| {
                    let i = (y * w + x) * 3;
                    rgb[i] = r;
                    rgb[i + 1] = g;
                    rgb[i + 2] = b;
                };
                if inside_sq {
                    let dtl = ((fx - tlx).powi(2) + (fy - tly).powi(2)).sqrt();
                    let dbr = ((fx - brx).powi(2) + (fy - bry).powi(2)).sqrt();
                    let shaded = if leaf {
                        dtl <= s && dbr <= s
                    } else {
                        dtl <= s
                    };
                    if shaded {
                        set(220, 30, 30); // red
                    }
                }
                // black square border (2px)
                let on_border = inside_sq
                    && ((fx - x0).abs() < 2.0
                        || (fx - x1).abs() < 2.0
                        || (fy - y0).abs() < 2.0
                        || (fy - y1).abs() < 2.0);
                if on_border {
                    set(0, 0, 0);
                }
            }
        }
        (w, h, rgb)
    }
}

#[cfg(test)]
mod tests {
    use super::pngtest::*;
    use super::*;

    fn solve(side_px: usize, pad: usize, side: i64, leaf: bool) -> PerceiveOutcome {
        let (w, h, rgb) = if leaf {
            draw_red_leaf(side_px, pad)
        } else {
            draw_quarter(side_px, pad)
        };
        let png = encode_png_rgb(w, h, &rgb);
        perceive_red_leaf_png(&png, Some(side))
    }

    #[test]
    fn png_roundtrip_decodes() {
        let (w, h, rgb) = draw_red_leaf(200, 20);
        let png = encode_png_rgb(w, h, &rgb);
        let img = decode_png_rgb(&png).expect("decodes");
        assert_eq!(img.w, w);
        assert_eq!(img.h, h);
        // a pixel near the centre of the leaf is red
        let (r, g, b) = img.at(w / 2, h / 2);
        assert!(r > 150 && g < 100 && b < 100, "centre not red: {:?}", (r, g, b));
    }

    #[test]
    fn redleaf_side8_matches_known_32pi_minus_64() {
        // The one value the old hardcode returned -- here it must FLOW from
        // pixels (freshly drawn, encoded, decoded, perceived).
        match solve(240, 24, 8, true) {
            PerceiveOutcome::Solved { area, .. } => {
                assert_eq!(area.pi.num, 32, "pi coeff");
                assert_eq!(area.pi.den, 1);
                assert_eq!(area.rational.num, -64, "rational");
                assert_eq!(area.rational.den, 1);
            }
            other => panic!("expected Solved, got {:?}", other),
        }
    }

    #[test]
    fn redleaf_novel_side5_scales_correctly_not_32pi_minus_64() {
        // NOVEL scale the code has never seen: S=5 -> (25/2)*pi - 25.
        match solve(230, 22, 5, true) {
            PerceiveOutcome::Solved { area, area_decimal, .. } => {
                assert_eq!(area.pi.num, 25);
                assert_eq!(area.pi.den, 2);
                assert_eq!(area.rational.num, -25);
                assert_eq!(area.rational.den, 1);
                // definitely NOT the old constant 32pi-64 (~36.53)
                assert!((area_decimal - (12.5 * std::f64::consts::PI - 25.0)).abs() < 1e-9);
                assert!((area_decimal - (32.0 * std::f64::consts::PI - 64.0)).abs() > 1.0);
            }
            other => panic!("expected Solved, got {:?}", other),
        }
    }

    #[test]
    fn redleaf_novel_side7_scales_correctly() {
        match solve(210, 30, 7, true) {
            PerceiveOutcome::Solved { area, .. } => {
                assert_eq!(area.pi.num, 49);
                assert_eq!(area.pi.den, 2);
                assert_eq!(area.rational.num, -49);
            }
            other => panic!("expected Solved, got {:?}", other),
        }
    }

    #[test]
    fn structurally_different_quarter_same_side8_differs_from_leaf() {
        // Same S=8, DIFFERENT pixels (one arc, not two) -> different answer.
        // This proves the PIXELS drive the structure, not the side param.
        match solve(240, 24, 8, false) {
            PerceiveOutcome::Solved { area, scene, .. } => {
                assert_eq!(scene.class, FigureClass::Quarter);
                assert_eq!(area.pi.num, 16, "quarter S=8 is 16*pi");
                assert_eq!(area.rational.num, 0);
                // not the leaf's 32pi-64
                assert!(!(area.pi.num == 32 && area.rational.num == -64));
            }
            other => panic!("expected Solved quarter, got {:?}", other),
        }
    }

    #[test]
    fn structure_only_when_no_side_given() {
        let (w, h, rgb) = draw_red_leaf(230, 22);
        let png = encode_png_rgb(w, h, &rgb);
        match perceive_red_leaf_png(&png, None) {
            PerceiveOutcome::StructureOnly { scene, symbolic_in_s } => {
                assert_eq!(scene.class, FigureClass::RedLeaf);
                assert!(symbolic_in_s.contains("S^2"));
            }
            other => panic!("expected StructureOnly, got {:?}", other),
        }
    }

    #[test]
    fn blank_square_cant_parse_no_number() {
        // A square with NO shaded region -> honest can't-parse, no number.
        let side_px = 200usize;
        let pad = 20usize;
        let w = side_px + 2 * pad;
        let h = w;
        let mut rgb = vec![255u8; w * h * 3];
        // draw only the black border
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let x0 = pad as f64;
                let x1 = (pad + side_px) as f64;
                let on_border = fx >= x0
                    && fx <= x1
                    && fy >= x0
                    && fy <= x1
                    && ((fx - x0).abs() < 2.0
                        || (fx - x1).abs() < 2.0
                        || (fy - x0).abs() < 2.0
                        || (fy - x1).abs() < 2.0);
                if on_border {
                    let i = (y * w + x) * 3;
                    rgb[i] = 0;
                    rgb[i + 1] = 0;
                    rgb[i + 2] = 0;
                }
            }
        }
        let png = encode_png_rgb(w, h, &rgb);
        match perceive_red_leaf_png(&png, Some(8)) {
            PerceiveOutcome::CantParse { .. } => {}
            other => panic!("blank square must CantParse, got {:?}", other),
        }
    }

    #[test]
    fn garbage_bytes_cant_parse_no_number() {
        let junk = b"this is not a png, it is a sentence about dogs and tails";
        match perceive_red_leaf_png(junk, Some(8)) {
            PerceiveOutcome::CantParse { .. } => {}
            other => panic!("garbage must CantParse, got {:?}", other),
        }
    }

    #[test]
    fn real_extractor_perceives_contents_true() {
        let e = RealGeoExtractor;
        assert!(e.perceives_contents());
        assert_eq!(e.name(), "geo_redleaf");
        let (w, h, rgb) = draw_red_leaf(220, 20);
        let png = encode_png_rgb(w, h, &rgb);
        let t = e.extract(&png, ExtractHints::default()).expect("extracts");
        assert_eq!(t[0].object, "red_leaf_in_square");
    }

    #[test]
    fn base64_roundtrip() {
        let (w, h, rgb) = draw_quarter(120, 12);
        let png = encode_png_rgb(w, h, &rgb);
        // encode with the standard alphabet
        let b64 = {
            const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut o = String::new();
            for ch in png.chunks(3) {
                let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
                let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
                o.push(A[(n >> 18 & 63) as usize] as char);
                o.push(A[(n >> 12 & 63) as usize] as char);
                o.push(if ch.len() > 1 { A[(n >> 6 & 63) as usize] as char } else { '=' });
                o.push(if ch.len() > 2 { A[(n & 63) as usize] as char } else { '=' });
            }
            o
        };
        let back = decode_base64(&b64).expect("decodes");
        assert_eq!(back, png);
    }
}
