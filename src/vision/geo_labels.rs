//! Phase P-C — LABEL READING + ASSOCIATION.
//!
//! internal notes. Consumes the Phase P-B primitive detector output
//! (`crate::vision::geo_primitives`, commit 736ea79) and BINDS each read text
//! label to the specific primitive it annotates: a number CENTERED INSIDE a
//! region → that region's AREA; a number ADJACENT-AND-PARALLEL to a segment →
//! that segment's LENGTH; a number off-centre inside/near a circle → its
//! RADIUS; a "?" inside a region → the solve-for TARGET on that region; a
//! single letter at/near a vertex point → that point's NAME.
//!
//! This is the next link toward feeding the constraint solver
//! `crate::math::algsolve` (commit 0072069): the resolved (label, role,
//! primitive) bindings map mechanically onto algsolve's vocabulary —
//! Area→`Con::PolyArea` / `Target::DiskArea`, Length→`Con::Dist`,
//! Target→`Target::*`. P-D (constraint extraction) and P-E (wire to algsolve +
//! live) are still separate downstream work; this module performs NO wiring
//! and sits on no live path.
//!
//! HONESTY FLOOR (absolute). A label is bound to a primitive ONLY with genuine
//! geometric evidence — position-inside (containment margin) or adjacency
//! (perpendicular offset within a segment's span). If a label's target is
//! ambiguous (equidistant to two candidate regions/segments, or role unclear)
//! or the glyph is below OCR confidence, it is left UNBOUND and the reason is
//! recorded — NEVER guessed into a wrong binding, and no label value is emitted
//! that the OCR did not actually read. There is NO per-image / filename / hash
//! / figure-name / label-value / binding lookup anywhere on this path: reading
//! is done from pixels and association from geometry alone.
//!
//! OCR REUSE. The glyph front end is reused verbatim from P-4
//! (`geo_perceive_p4`): `find_glyphs` (connected-component dark-ink glyph
//! segmentation), `norm_bitmap` (aspect-preserving 20×20 normalisation),
//! `ref_bitmaps` (the built-in 5×7 digit font) and `NORM`. This module EXTENDS
//! that recogniser with a small 5×7 font for a "?" target marker and single
//! upper-case vertex letters (I and O are deliberately excluded — they collide
//! pixel-for-pixel with the digits 1 and 0, so admitting them would only
//! manufacture ambiguity), and adds per-token bounding boxes + positions that
//! P-4's `ocr_number` (which collapses to one integer) does not carry.
//!
//! OFFLINE library only. Wires to nothing; no guest surface, no promote, no
//! serving change.

use crate::vision::geo_perceive::RgbImage;
use crate::vision::geo_perceive_p4::{
    decode_image_rgb, find_glyphs, norm_bitmap, rectify, ref_bitmaps, Glyph, NORM,
};
use crate::vision::geo_primitives::{detect_on_raster, DetectResult, Primitive, Primitives};

// =====================================================================
// Public vocabulary
// =====================================================================

/// What a read label token spells.
#[derive(Debug, Clone, PartialEq)]
pub enum LabelKind {
    /// A (multi-digit) non-negative integer.
    Number(i64),
    /// A "?" solve-for marker.
    Question,
    /// A single upper-case vertex letter.
    Letter(char),
}

/// One OCR'd label token in normalized image coordinates (top-left origin,
/// divided by the working-raster width — the SAME frame P-B emits its
/// primitives in, so positions are directly comparable).
#[derive(Debug, Clone)]
pub struct LabelToken {
    pub text: String,
    pub kind: LabelKind,
    pub center: (f64, f64),
    /// (x0, y0, x1, y1) normalized.
    pub bbox: (f64, f64, f64, f64),
    pub confidence: f64,
}

/// The geometric role a bound label plays for its primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    /// Number centred inside a polygon/circle → that region's area.
    Area,
    /// Number adjacent-and-parallel to a segment → that segment's length.
    Length,
    /// Number off-centre inside / near a circle → its radius.
    Radius,
    /// "?" inside a region → the solve-for target on that region.
    Target,
    /// Letter at/near a vertex point → that point's name.
    VertexName,
    /// A numeric label in the figure margin (outside the rectified frame) →
    /// the whole figure's scale (bound to the outer square when present).
    Scale,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Area => "area",
            Role::Length => "length",
            Role::Radius => "radius",
            Role::Target => "target",
            Role::VertexName => "vertex_name",
            Role::Scale => "scale",
        }
    }
    /// Sketch of how this (role, primitive-kind) maps onto the algsolve
    /// constraint vocabulary. Documentation only — no wiring.
    pub fn algsolve_hint(&self, prim_kind: &str) -> &'static str {
        match (self, prim_kind) {
            (Role::Area, "polygon") => "Con::PolyArea(vertex_ids, value)",
            (Role::Area, "circle") | (Role::Area, "arc") => "Target/known DiskArea(center, r) = value",
            (Role::Length, "segment") => "Con::Dist(i, j, value)",
            (Role::Radius, "circle") | (Role::Radius, "arc") => "radius value feeding DiskArea(center, r)",
            (Role::Target, "circle") | (Role::Target, "arc") => "Target::DiskArea (solve-for)",
            (Role::Target, "polygon") => "Con::PolyArea with unknown value (solve-for)",
            (Role::VertexName, "point") => "PointDef name (labelling only, no numeric constraint)",
            (Role::Scale, _) => "global unit scale for the figure",
            _ => "(no direct algsolve mapping)",
        }
    }
}

/// A resolved label→primitive binding with its supporting evidence.
#[derive(Debug, Clone)]
pub struct Binding {
    pub token_index: usize,
    pub label: String,
    pub role: Role,
    /// Index into `Primitives.items`.
    pub primitive_index: usize,
    pub primitive_kind: &'static str,
    pub confidence: f64,
    pub evidence: String,
}

/// A label that was read but could NOT be confidently attached — first-class,
/// honest output (never force-bound).
#[derive(Debug, Clone)]
pub struct Unbound {
    pub token_index: usize,
    pub label: String,
    pub reason: String,
}

/// A numeric scale label found in the figure margin (outside the frame).
#[derive(Debug, Clone)]
pub struct ScaleReading {
    pub text: String,
    pub value: i64,
    pub center: (f64, f64),
    pub confidence: f64,
}

/// P-B primitives + the resolved bindings + the honest unbound list.
#[derive(Debug, Clone)]
pub struct LabeledPrimitives {
    pub primitives: Primitives,
    pub tokens: Vec<LabelToken>,
    pub bindings: Vec<Binding>,
    pub unbound: Vec<Unbound>,
    /// Out-of-frame scale reading (e.g. redleaf's grid-cell number), if any.
    pub scale: Option<ScaleReading>,
    pub rectified: bool,
}

impl LabeledPrimitives {
    /// A stable human-readable report (for tests / the P-C write-up).
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("primitives: {}\n", self.primitives.census()));
        s.push_str(&format!("tokens read: {}\n", self.tokens.len()));
        for (i, t) in self.tokens.iter().enumerate() {
            s.push_str(&format!(
                "  token[{}] '{}' {:?} @({:.3},{:.3}) conf={:.2}\n",
                i, t.text, t.kind, t.center.0, t.center.1, t.confidence
            ));
        }
        s.push_str(&format!("bindings: {}\n", self.bindings.len()));
        for b in &self.bindings {
            s.push_str(&format!(
                "  '{}' -> {} on {}[{}] conf={:.2} :: {} [{}]\n",
                b.label,
                b.role.as_str(),
                b.primitive_kind,
                b.primitive_index,
                b.confidence,
                b.evidence,
                b.role.algsolve_hint(b.primitive_kind),
            ));
        }
        s.push_str(&format!("unbound: {}\n", self.unbound.len()));
        for u in &self.unbound {
            s.push_str(&format!("  '{}' :: {}\n", u.label, u.reason));
        }
        if let Some(sc) = &self.scale {
            s.push_str(&format!(
                "scale (out-of-frame): '{}' = {} conf={:.2}\n",
                sc.text, sc.value, sc.confidence
            ));
        }
        s
    }

    /// Find the binding (if any) whose label spells `label`.
    pub fn binding_for<'a>(&'a self, label: &str) -> Option<&'a Binding> {
        self.bindings.iter().find(|b| b.label == label)
    }
    pub fn is_unbound(&self, label: &str) -> bool {
        self.unbound.iter().any(|u| u.label == label)
    }
}

/// Honest result of a label-and-associate attempt.
#[derive(Debug, Clone)]
pub enum LabelResult {
    Labeled(LabeledPrimitives),
    /// The underlying primitive detection was itself Incomplete — we do not
    /// invent primitives just to hang labels on.
    Incomplete { stage: String, reason: String },
}

// =====================================================================
// Tunables (association evidence margins — the honesty floor)
// =====================================================================

const TOKEN_CONF_MIN: f64 = 0.32; // below this OCR confidence a glyph is NOT emitted
const ADJ_PERP: f64 = 0.06; // max perpendicular offset (normalized) for segment adjacency
const ADJ_TIE: f64 = 0.015; // two segments this close ⇒ ambiguous, leave unbound
const AREA_TIE_FRAC: f64 = 0.02; // two nested regions with areas this close ⇒ ambiguous
const VN_RADIUS: f64 = 0.06; // max distance (normalized) for a letter to name a vertex
const VN_TIE: f64 = 0.02; // two vertices this close ⇒ ambiguous

// =====================================================================
// Entry point
// =====================================================================

/// Decode → rectify (shared with P-B so coordinate frames match) → detect
/// primitives → OCR label tokens on the same raster → associate. Takes ONLY
/// the image bytes; there is no side channel.
pub fn read_labels(bytes: &[u8]) -> LabelResult {
    let img = match decode_image_rgb(bytes) {
        Some(i) => i,
        None => {
            return LabelResult::Incomplete {
                stage: "decode".into(),
                reason: "not a decodable PNG or baseline JPEG".into(),
            }
        }
    };
    // `rect` owns the rectified raster when a frame is found; otherwise we work
    // on `img` directly. Both primitive detection and OCR run on the SAME
    // raster so coordinate frames match. No RgbImage clone (it isn't Clone).
    let rect = rectify(&img);
    let (raster, rectified): (&RgbImage, bool) = match &rect {
        Some((r, _skew)) => (r, true),
        None => (&img, false),
    };
    // Out-of-frame numeric scale label (e.g. redleaf's grid-cell number, which
    // sits in the margin outside the square). Only meaningful when a frame was
    // actually found and rectified away.
    let scale = if rectified {
        read_margin_scale(&img)
    } else {
        None
    };
    build_labeled(raster, rectified, scale)
}

/// Label + associate on an ALREADY-decoded raster (no decode / rectify). The
/// rectifier is P-B's front end (validated there); this entry point lets the
/// P-C reading + association logic be driven deterministically on a known
/// raster. `read_labels` is this plus decode + rectify + margin-scale.
pub fn label_on_raster(raster: &RgbImage, rectified: bool) -> LabelResult {
    build_labeled(raster, rectified, None)
}

fn build_labeled(raster: &RgbImage, rectified: bool, scale: Option<ScaleReading>) -> LabelResult {
    let prims = match detect_on_raster(raster, rectified) {
        DetectResult::Primitives(p) => p,
        DetectResult::Incomplete { stage, reason } => {
            return LabelResult::Incomplete { stage, reason }
        }
    };
    let tokens = read_tokens_on_raster(raster);
    let (mut bindings, unbound) = associate(&prims, &tokens);
    if let Some(sc) = &scale {
        if let Some(pi) = largest_polygon_index(&prims) {
            bindings.push(Binding {
                token_index: usize::MAX,
                label: sc.text.clone(),
                role: Role::Scale,
                primitive_index: pi,
                primitive_kind: "polygon",
                confidence: sc.confidence,
                evidence: "numeric label in figure margin → figure scale on outer square".into(),
            });
        }
    }
    LabelResult::Labeled(LabeledPrimitives {
        primitives: prims,
        tokens,
        bindings,
        unbound,
        scale,
        rectified,
    })
}

// =====================================================================
// OCR: read label tokens (digits / "?" / letters) with per-token bbox
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sym {
    Digit(u8),
    Letter(char),
    Question,
}

/// A single classified glyph in pixel coordinates.
struct GlyphClass {
    cx: f64,
    cy: f64,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    sym: Sym,
    conf: f64,
}

/// Read every confident label token on an already-decoded (optionally
/// rectified) raster. Public so tests can drive the exact same OCR the entry
/// point uses.
pub fn read_tokens_on_raster(raster: &RgbImage) -> Vec<LabelToken> {
    let sf = raster.w as f64;
    let templates = build_templates();
    let glyphs = find_glyphs(raster);
    let mut classed: Vec<GlyphClass> = Vec::new();
    for g in &glyphs {
        let nb = glyph_norm_bitmap(g);
        let (sym, conf) = classify(&nb, &templates);
        if conf < TOKEN_CONF_MIN {
            continue; // honest: below threshold ⇒ not emitted
        }
        classed.push(GlyphClass {
            cx: (g.x0 + g.x1) as f64 / 2.0,
            cy: (g.y0 + g.y1) as f64 / 2.0,
            x0: g.x0,
            y0: g.y0,
            x1: g.x1,
            y1: g.y1,
            sym,
            conf,
        });
    }
    group_tokens(&classed, sf)
}

/// Reconstruct a glyph's tight binary grid, then normalise via the REUSED P-4
/// `norm_bitmap`.
fn glyph_norm_bitmap(g: &Glyph) -> [bool; NORM * NORM] {
    let gw = g.x1 - g.x0 + 1;
    let gh = g.y1 - g.y0 + 1;
    let mut grid = vec![false; gw * gh];
    for &(x, y) in &g.mask {
        grid[(y - g.y0) * gw + (x - g.x0)] = true;
    }
    norm_bitmap(&grid, gw, gh)
}

/// Nearest-template classification over the COMBINED pool (digits + letters +
/// "?"), with the confidence taken from fit and the runner-up margin — so a
/// glyph that is nearly as close to a second symbol is honestly low-confidence.
fn classify(nb: &[bool; NORM * NORM], templates: &[(Sym, [bool; NORM * NORM])]) -> (Sym, f64) {
    let mut best = (Sym::Question, f64::MAX);
    let mut second = f64::MAX;
    for (sym, tmpl) in templates {
        let mut diff = 0usize;
        for i in 0..NORM * NORM {
            if nb[i] != tmpl[i] {
                diff += 1;
            }
        }
        let d = diff as f64 / (NORM * NORM) as f64;
        if d < best.1 {
            second = best.1;
            best = (*sym, d);
        } else if d < second {
            second = d;
        }
    }
    let fit = 1.0 - best.1;
    let sep = second - best.1;
    let conf = (0.6 * fit + 0.4 * (sep * 4.0)).clamp(0.0, 1.0);
    (best.0, conf)
}

/// Group classified glyphs into tokens: adjacent same-line digits merge into a
/// multi-digit number; letters and "?" stand alone.
fn group_tokens(classed: &[GlyphClass], sf: f64) -> Vec<LabelToken> {
    let mut tokens: Vec<LabelToken> = Vec::new();

    // ---- non-digit singletons ----
    for g in classed {
        match g.sym {
            Sym::Letter(c) => tokens.push(single_token(g, LabelKind::Letter(c), c.to_string(), sf)),
            Sym::Question => {
                tokens.push(single_token(g, LabelKind::Question, "?".into(), sf))
            }
            Sym::Digit(_) => {}
        }
    }

    // ---- digits grouped into numbers ----
    let digits: Vec<&GlyphClass> = classed.iter().filter(|g| matches!(g.sym, Sym::Digit(_))).collect();
    if !digits.is_empty() {
        // median glyph height sets the spacing scale
        let mut heights: Vec<f64> = digits.iter().map(|g| (g.y1 - g.y0 + 1) as f64).collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = heights[heights.len() / 2].max(1.0);

        let mut used = vec![false; digits.len()];
        // sort indices left→right
        let mut order: Vec<usize> = (0..digits.len()).collect();
        order.sort_by(|&a, &b| digits[a].cx.partial_cmp(&digits[b].cx).unwrap());
        for &si in &order {
            if used[si] {
                continue;
            }
            // start a run from si, greedily extend rightwards along the line
            let mut run = vec![si];
            used[si] = true;
            loop {
                let last = *run.last().unwrap();
                let lg = digits[last];
                let mut best: Option<usize> = None;
                let mut best_gap = f64::MAX;
                for &ci in &order {
                    if used[ci] {
                        continue;
                    }
                    let cg = digits[ci];
                    let same_line = (cg.cy - lg.cy).abs() < 0.6 * h;
                    let gap = cg.x0 as f64 - lg.x1 as f64;
                    // digits sit close together; allow a small kerning gap
                    if same_line && cg.cx > lg.cx && gap < 0.9 * h && gap > -0.3 * h {
                        if gap < best_gap {
                            best_gap = gap;
                            best = Some(ci);
                        }
                    }
                }
                match best {
                    Some(ci) => {
                        used[ci] = true;
                        run.push(ci);
                    }
                    None => break,
                }
            }
            // build number token from the run (already left→right by construction)
            let mut text = String::new();
            let mut conf = 1.0f64;
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &ri in &run {
                let g = digits[ri];
                if let Sym::Digit(d) = g.sym {
                    text.push((b'0' + d) as char);
                }
                conf = conf.min(g.conf);
                x0 = x0.min(g.x0);
                y0 = y0.min(g.y0);
                x1 = x1.max(g.x1);
                y1 = y1.max(g.y1);
                sx += g.cx;
                sy += g.cy;
            }
            if let Ok(val) = text.parse::<i64>() {
                let n = run.len() as f64;
                tokens.push(LabelToken {
                    text: text.clone(),
                    kind: LabelKind::Number(val),
                    center: (sx / n / sf, sy / n / sf),
                    bbox: (x0 as f64 / sf, y0 as f64 / sf, x1 as f64 / sf, y1 as f64 / sf),
                    confidence: conf,
                });
            }
        }
    }
    tokens
}

fn single_token(g: &GlyphClass, kind: LabelKind, text: String, sf: f64) -> LabelToken {
    LabelToken {
        text,
        kind,
        center: (g.cx / sf, g.cy / sf),
        bbox: (
            g.x0 as f64 / sf,
            g.y0 as f64 / sf,
            g.x1 as f64 / sf,
            g.y1 as f64 / sf,
        ),
        confidence: g.conf,
    }
}

// =====================================================================
// Reference font: reuse P-4's digit bitmaps, extend with letters + "?"
// =====================================================================

/// I and O are intentionally omitted — they are pixel-identical to 1 and 0 and
/// would only manufacture ambiguity between vertex names and digits.
const LETTER_FONT: &[(char, [&str; 7])] = &[
    ('A', ["01110", "10001", "10001", "11111", "10001", "10001", "10001"]),
    ('B', ["11110", "10001", "10001", "11110", "10001", "10001", "11110"]),
    ('C', ["01110", "10001", "10000", "10000", "10000", "10001", "01110"]),
    ('D', ["11110", "10001", "10001", "10001", "10001", "10001", "11110"]),
    ('E', ["11111", "10000", "10000", "11110", "10000", "10000", "11111"]),
    ('F', ["11111", "10000", "10000", "11110", "10000", "10000", "10000"]),
    ('G', ["01110", "10001", "10000", "10111", "10001", "10001", "01110"]),
    ('H', ["10001", "10001", "10001", "11111", "10001", "10001", "10001"]),
    ('K', ["10001", "10010", "10100", "11000", "10100", "10010", "10001"]),
    ('L', ["10000", "10000", "10000", "10000", "10000", "10000", "11111"]),
    ('M', ["10001", "11011", "10101", "10101", "10001", "10001", "10001"]),
    ('N', ["10001", "11001", "10101", "10011", "10001", "10001", "10001"]),
    ('P', ["11110", "10001", "10001", "11110", "10000", "10000", "10000"]),
    ('Q', ["01110", "10001", "10001", "10001", "10101", "10010", "01101"]),
    ('R', ["11110", "10001", "10001", "11110", "10100", "10010", "10001"]),
    ('S', ["01111", "10000", "10000", "01110", "00001", "00001", "11110"]),
    ('T', ["11111", "00100", "00100", "00100", "00100", "00100", "00100"]),
    ('U', ["10001", "10001", "10001", "10001", "10001", "10001", "01110"]),
    ('V', ["10001", "10001", "10001", "10001", "10001", "01010", "00100"]),
    ('W', ["10001", "10001", "10001", "10101", "10101", "10101", "01010"]),
    ('X', ["10001", "10001", "01010", "00100", "01010", "10001", "10001"]),
    ('Y', ["10001", "10001", "01010", "00100", "00100", "00100", "00100"]),
    ('Z', ["11111", "00001", "00010", "00100", "01000", "10000", "11111"]),
];

// A connected "?" (curl over a stem). NOTE (honesty): a printed "?" usually has
// a DETACHED dot below the stem; the reused connected-component front end
// (`find_glyphs`) drops such a tiny stray component, so on real print the
// dotless hook is what survives — this is the shakier half of P-C recognition
// and is reported as such. The connected form here is what the recogniser can
// reliably segment as one token.
const QUESTION_FONT: [&str; 7] = [
    "01110", "10001", "00010", "00100", "00100", "00100", "00100",
];

fn font_to_norm(rows: &[&str; 7]) -> [bool; NORM * NORM] {
    let mut grid = vec![false; 5 * 7];
    for (ry, row) in rows.iter().enumerate() {
        for (rx, ch) in row.chars().enumerate() {
            grid[ry * 5 + rx] = ch == '1' || ch == '#';
        }
    }
    norm_bitmap(&grid, 5, 7)
}

fn build_templates() -> Vec<(Sym, [bool; NORM * NORM])> {
    let mut out: Vec<(Sym, [bool; NORM * NORM])> = Vec::new();
    // digits — REUSED verbatim from P-4
    let digit_refs = ref_bitmaps();
    for (d, bm) in digit_refs.iter().enumerate() {
        out.push((Sym::Digit(d as u8), *bm));
    }
    for (c, rows) in LETTER_FONT {
        out.push((Sym::Letter(*c), font_to_norm(rows)));
    }
    out.push((Sym::Question, font_to_norm(&QUESTION_FONT)));
    out
}

// =====================================================================
// Out-of-frame scale (redleaf): read numeric glyphs on the ORIGINAL image and
// keep the best one sitting in the margin.
// =====================================================================

fn read_margin_scale(img: &RgbImage) -> Option<ScaleReading> {
    let sf = img.w.min(img.h) as f64;
    let templates = build_templates();
    let glyphs = find_glyphs(img);
    let mut classed: Vec<GlyphClass> = Vec::new();
    for g in &glyphs {
        let nb = glyph_norm_bitmap(g);
        let (sym, conf) = classify(&nb, &templates);
        if conf < TOKEN_CONF_MIN {
            continue;
        }
        if !matches!(sym, Sym::Digit(_)) {
            continue;
        }
        classed.push(GlyphClass {
            cx: (g.x0 + g.x1) as f64 / 2.0,
            cy: (g.y0 + g.y1) as f64 / 2.0,
            x0: g.x0,
            y0: g.y0,
            x1: g.x1,
            y1: g.y1,
            sym,
            conf,
        });
    }
    let tokens = group_tokens(&classed, sf);
    // keep numbers whose centre sits outside the central 76% box (i.e. the
    // margin), highest confidence first.
    let (w, h) = (img.w as f64, img.h as f64);
    let mut best: Option<ScaleReading> = None;
    for t in &tokens {
        if let LabelKind::Number(v) = t.kind {
            let px = t.center.0 * sf; // undo normalization to pixel space
            let py = t.center.1 * sf;
            let in_margin = px < 0.12 * w || px > 0.88 * w || py < 0.12 * h || py > 0.88 * h;
            if !in_margin {
                continue;
            }
            let better = match &best {
                None => true,
                Some(b) => t.confidence > b.confidence,
            };
            if better {
                best = Some(ScaleReading {
                    text: t.text.clone(),
                    value: v,
                    center: t.center,
                    confidence: t.confidence,
                });
            }
        }
    }
    best
}

// =====================================================================
// ASSOCIATION
// =====================================================================

enum RegionShape {
    Poly(Vec<(f64, f64)>),
    Circ((f64, f64), f64),
}

struct Region {
    idx: usize,
    kind: &'static str,
    shape: RegionShape,
}

impl Region {
    fn area(&self) -> f64 {
        match &self.shape {
            RegionShape::Poly(v) => poly_area(v),
            RegionShape::Circ(_, r) => std::f64::consts::PI * r * r,
        }
    }
    fn char_r(&self) -> f64 {
        (self.area() / std::f64::consts::PI).max(1e-9).sqrt()
    }
    fn contains(&self, p: (f64, f64)) -> bool {
        match &self.shape {
            RegionShape::Poly(v) => point_in_poly(p, v),
            RegionShape::Circ(c, r) => dist(*c, p) <= *r,
        }
    }
    /// Distance from an interior point to the nearest boundary (>=0 inside).
    fn boundary_margin(&self, p: (f64, f64)) -> f64 {
        match &self.shape {
            RegionShape::Poly(v) => {
                let mut m = f64::MAX;
                for i in 0..v.len() {
                    let a = v[i];
                    let b = v[(i + 1) % v.len()];
                    m = m.min(point_seg_dist(p, a, b));
                }
                m
            }
            RegionShape::Circ(c, r) => r - dist(*c, p),
        }
    }
}

fn regions_of(prims: &Primitives) -> Vec<Region> {
    let mut out = Vec::new();
    // pass 1: polygons + full circles
    for (i, p) in prims.items.iter().enumerate() {
        match p {
            Primitive::Polygon { verts, .. } => out.push(Region {
                idx: i,
                kind: "polygon",
                shape: RegionShape::Poly(verts.clone()),
            }),
            Primitive::Circle { center, r, .. } => out.push(Region {
                idx: i,
                kind: "circle",
                shape: RegionShape::Circ(*center, *r),
            }),
            _ => {}
        }
    }
    // pass 2: arcs — a detected arc is a circle fragment whose enclosed disk
    // (center, r) is still well defined, so a "?" / area / radius label inside
    // it binds to that disk (real photos often yield an arc where an ideal
    // render gives a full circle). BUT skip an arc that merely duplicates a
    // circle already added for the same disk, so a circle+arc pair of the same
    // shape is NOT mistaken for two nested equal-area regions.
    for (i, p) in prims.items.iter().enumerate() {
        if let Primitive::Arc { center, r, .. } = p {
            let dup = out.iter().any(|reg| match &reg.shape {
                RegionShape::Circ(c, rr) => {
                    dist(*c, *center) < 0.05 && (rr - r).abs() < 0.05
                }
                _ => false,
            });
            if !dup {
                out.push(Region {
                    idx: i,
                    kind: "arc",
                    shape: RegionShape::Circ(*center, *r),
                });
            }
        }
    }
    out
}

/// Segments as ((a,b), primitive_index).
fn segments_of(prims: &Primitives) -> Vec<(((f64, f64), (f64, f64)), usize)> {
    let mut out = Vec::new();
    for (i, p) in prims.items.iter().enumerate() {
        if let Primitive::Segment { a, b, .. } = p {
            out.push((((*a), (*b)), i));
        }
    }
    out
}

/// Points as (p, primitive_index).
fn points_of(prims: &Primitives) -> Vec<((f64, f64), usize)> {
    let mut out = Vec::new();
    for (i, pr) in prims.items.iter().enumerate() {
        if let Primitive::Point { p, .. } = pr {
            out.push((*p, i));
        }
    }
    out
}

/// The purely geometric association step: bind each token to a primitive by
/// containment / adjacency, or leave it honestly unbound. Public so tests can
/// exercise it with hand-built primitives + tokens (e.g. ambiguity cases).
pub fn associate(prims: &Primitives, tokens: &[LabelToken]) -> (Vec<Binding>, Vec<Unbound>) {
    let regions = regions_of(prims);
    let segments = segments_of(prims);
    let points = points_of(prims);
    let mut bindings = Vec::new();
    let mut unbound = Vec::new();

    for (ti, t) in tokens.iter().enumerate() {
        match &t.kind {
            LabelKind::Question => {
                associate_question(ti, t, &regions, &mut bindings, &mut unbound)
            }
            LabelKind::Letter(_) => {
                associate_letter(ti, t, &points, &mut bindings, &mut unbound)
            }
            LabelKind::Number(_) => {
                associate_number(ti, t, &regions, &segments, &mut bindings, &mut unbound)
            }
        }
    }
    (bindings, unbound)
}

/// Smallest region that contains `p`; returns (index-into-regions, area).
/// Records ambiguity when the two smallest containing regions are ~equal area.
fn smallest_containing<'a>(
    regions: &'a [Region],
    p: (f64, f64),
) -> Result<Option<&'a Region>, String> {
    let mut containing: Vec<&Region> = regions.iter().filter(|r| r.contains(p)).collect();
    if containing.is_empty() {
        return Ok(None);
    }
    containing.sort_by(|a, b| a.area().partial_cmp(&b.area()).unwrap());
    if containing.len() >= 2 {
        let a0 = containing[0].area();
        let a1 = containing[1].area();
        if a1 > 0.0 && (a1 - a0) / a1 < AREA_TIE_FRAC {
            return Err(format!(
                "ambiguous: two nested regions of ~equal area ({} and {})",
                containing[0].kind, containing[1].kind
            ));
        }
    }
    Ok(Some(containing[0]))
}

fn associate_question(
    ti: usize,
    t: &LabelToken,
    regions: &[Region],
    bindings: &mut Vec<Binding>,
    unbound: &mut Vec<Unbound>,
) {
    match smallest_containing(regions, t.center) {
        Ok(Some(r)) => bindings.push(Binding {
            token_index: ti,
            label: t.text.clone(),
            role: Role::Target,
            primitive_index: r.idx,
            primitive_kind: r.kind,
            confidence: t.confidence,
            evidence: format!(
                "'?' contained in {} (boundary margin {:.3})",
                r.kind,
                r.boundary_margin(t.center)
            ),
        }),
        Ok(None) => unbound.push(Unbound {
            token_index: ti,
            label: t.text.clone(),
            reason: "'?' not inside any detected region".into(),
        }),
        Err(reason) => unbound.push(Unbound {
            token_index: ti,
            label: t.text.clone(),
            reason,
        }),
    }
}

fn associate_letter(
    ti: usize,
    t: &LabelToken,
    points: &[((f64, f64), usize)],
    bindings: &mut Vec<Binding>,
    unbound: &mut Vec<Unbound>,
) {
    if points.is_empty() {
        unbound.push(Unbound {
            token_index: ti,
            label: t.text.clone(),
            reason: "no vertex points detected to name".into(),
        });
        return;
    }
    let mut d: Vec<(f64, usize)> = points
        .iter()
        .map(|(p, idx)| (dist(*p, t.center), *idx))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (d0, idx0) = d[0];
    if d0 > VN_RADIUS {
        unbound.push(Unbound {
            token_index: ti,
            label: t.text.clone(),
            reason: format!("letter {:.3} from nearest vertex (> {:.3})", d0, VN_RADIUS),
        });
        return;
    }
    // Ambiguity ONLY when the two nearest candidates are DISTINCT vertices that
    // happen to be equidistant — not when the detector emitted two near-
    // coincident points for the same corner (those are the same vertex).
    if d.len() >= 2 && (d[1].0 - d0) < VN_TIE {
        let p0 = points.iter().find(|(_, i)| *i == idx0).map(|(p, _)| *p).unwrap();
        let p1 = points.iter().find(|(_, i)| *i == d[1].1).map(|(p, _)| *p).unwrap();
        if dist(p0, p1) > VN_RADIUS {
            unbound.push(Unbound {
                token_index: ti,
                label: t.text.clone(),
                reason: "ambiguous: equidistant to two vertices".into(),
            });
            return;
        }
    }
    bindings.push(Binding {
        token_index: ti,
        label: t.text.clone(),
        role: Role::VertexName,
        primitive_index: idx0,
        primitive_kind: "point",
        confidence: t.confidence,
        evidence: format!("letter at vertex (distance {:.3})", d0),
    });
}

fn associate_number(
    ti: usize,
    t: &LabelToken,
    regions: &[Region],
    segments: &[(((f64, f64), (f64, f64)), usize)],
    bindings: &mut Vec<Binding>,
    unbound: &mut Vec<Unbound>,
) {
    // nearest in-span segment (perp offset within the segment's extent)
    let mut seg_hits: Vec<(f64, usize)> = Vec::new();
    for ((a, b), idx) in segments {
        if let Some(perp) = perp_in_span(t.center, *a, *b) {
            seg_hits.push((perp, *idx));
        }
    }
    seg_hits.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let nearest_seg = seg_hits.first().copied();

    // smallest containing region
    let contain = match smallest_containing(regions, t.center) {
        Ok(c) => c,
        Err(reason) => {
            unbound.push(Unbound {
                token_index: ti,
                label: t.text.clone(),
                reason,
            });
            return;
        }
    };

    if let Some(r) = contain {
        let margin = r.boundary_margin(t.center);
        // A number INSIDE a region annotates that region: its AREA (or, for a
        // circle, its RADIUS when the number sits off-centre). Length labels
        // are written OUTSIDE the region beside an edge — handled below. This
        // inside⇒area / outside⇒length split is the honest, robust rule and
        // avoids a false "edge" ambiguity for small polygons whose centroid is
        // naturally within an inradius of every side.
        //
        // circle: distinguish an off-centre radius label from a central area
        if let RegionShape::Circ(c, rr) = &r.shape {
            let dc = dist(*c, t.center);
            if dc > 0.5 * rr && dc < 1.2 * rr {
                bindings.push(Binding {
                    token_index: ti,
                    label: t.text.clone(),
                    role: Role::Radius,
                    primitive_index: r.idx,
                    primitive_kind: r.kind,
                    confidence: t.confidence,
                    evidence: format!("number off-centre inside circle (d/r={:.2})", dc / rr),
                });
                return;
            }
        }
        bindings.push(Binding {
            token_index: ti,
            label: t.text.clone(),
            role: Role::Area,
            primitive_index: r.idx,
            primitive_kind: r.kind,
            confidence: t.confidence,
            evidence: format!(
                "number centred inside {} (boundary margin {:.3}, char_r {:.3})",
                r.kind,
                margin,
                r.char_r()
            ),
        });
        return;
    }

    // not inside any region — try segment adjacency (length)
    if let Some((d0, idx0)) = nearest_seg {
        if d0 < ADJ_PERP {
            // ambiguity: a second segment about as close
            if seg_hits.len() >= 2 && (seg_hits[1].0 - d0) < ADJ_TIE {
                unbound.push(Unbound {
                    token_index: ti,
                    label: t.text.clone(),
                    reason: "ambiguous: equidistant to two segments".into(),
                });
                return;
            }
            bindings.push(Binding {
                token_index: ti,
                label: t.text.clone(),
                role: Role::Length,
                primitive_index: idx0,
                primitive_kind: "segment",
                confidence: t.confidence,
                evidence: format!("number adjacent-and-parallel to segment (perp {:.3})", d0),
            });
            return;
        }
    }

    unbound.push(Unbound {
        token_index: ti,
        label: t.text.clone(),
        reason: "number neither inside a region nor adjacent to a segment".into(),
    });
}

fn largest_polygon_index(prims: &Primitives) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, p) in prims.items.iter().enumerate() {
        if let Primitive::Polygon { verts, .. } = p {
            let a = poly_area(verts);
            if best.map(|(_, ba)| a > ba).unwrap_or(true) {
                best = Some((i, a));
            }
        }
    }
    best.map(|(i, _)| i)
}

// =====================================================================
// geometry helpers
// =====================================================================

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn poly_area(v: &[(f64, f64)]) -> f64 {
    let n = v.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (x0, y0) = v[i];
        let (x1, y1) = v[(i + 1) % n];
        s += x0 * y1 - x1 * y0;
    }
    (s / 2.0).abs()
}

fn point_in_poly(p: (f64, f64), v: &[(f64, f64)]) -> bool {
    let n = v.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = v[i];
        let (xj, yj) = v[j];
        let intersect = ((yi > p.1) != (yj > p.1))
            && (p.0 < (xj - xi) * (p.1 - yi) / (yj - yi + f64::EPSILON) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn point_seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let l2 = ab.0 * ab.0 + ab.1 * ab.1;
    if l2 < 1e-12 {
        return dist(p, a);
    }
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / l2).clamp(0.0, 1.0);
    dist(p, (a.0 + t * ab.0, a.1 + t * ab.1))
}

/// Perpendicular distance from `p` to the segment a-b, but ONLY if the foot of
/// the perpendicular falls within the segment's span (with a small margin).
/// Returns None when `p` projects beyond either endpoint (so a number off the
/// end of a segment is not mistaken for a length label).
fn perp_in_span(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> Option<f64> {
    let ab = (b.0 - a.0, b.1 - a.1);
    let l2 = ab.0 * ab.0 + ab.1 * ab.1;
    if l2 < 1e-12 {
        return None;
    }
    let u = ((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / l2;
    if !(-0.05..=1.05).contains(&u) {
        return None;
    }
    let foot = (a.0 + u * ab.0, a.1 + u * ab.1);
    Some(dist(p, foot))
}

#[cfg(test)]
mod tests;
