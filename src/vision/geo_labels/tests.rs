//! P-C label reading + association — in-crate tests.
//!
//! Coverage:
//!   * REAL-STYLE tilted-square figure (John's FullSizeRender equivalent):
//!     "100"→tilted-square area, "24"→triangle area, "?"→circle target.
//!   * NOVEL synthetic figures with values/positions the code never saw
//!     (proves general reading+binding, not lookup): "36"+"?", "12"+"5".
//!   * Letter vertex-naming.
//!   * ANTI-CHEAT / honesty: unlabeled ⇒ all-unbound; ambiguous (equidistant)
//!     ⇒ unbound-with-reason; garbled glyph ⇒ below confidence, not emitted.

use super::*;
use crate::vision::geo_perceive::pngtest::encode_png_rgb;
use crate::vision::geo_primitives::detect_primitives;

// ---------------------------------------------------------------------
// tiny software rasteriser (test-only) — black ink on white, with a glyph
// stamper so we can render digits / letters / "?" as PIXELS the OCR reads back.
// ---------------------------------------------------------------------

const BLACK: (u8, u8, u8) = (0, 0, 0);

struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        Canvas { w, h, rgb: vec![255u8; w * h * 3] }
    }
    fn set(&mut self, x: i64, y: i64, c: (u8, u8, u8)) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 3;
        self.rgb[i] = c.0;
        self.rgb[i + 1] = c.1;
        self.rgb[i + 2] = c.2;
    }
    fn seg(&mut self, a: (f64, f64), b: (f64, f64), hw: i64, c: (u8, u8, u8)) {
        let len = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        let steps = (len.ceil() as i64).max(1);
        for s in 0..=steps {
            let f = s as f64 / steps as f64;
            let x = a.0 + (b.0 - a.0) * f;
            let y = a.1 + (b.1 - a.1) * f;
            for dy in -hw..=hw {
                for dx in -hw..=hw {
                    self.set(x.round() as i64 + dx, y.round() as i64 + dy, c);
                }
            }
        }
    }
    fn circle(&mut self, cx: f64, cy: f64, r: f64, hw: f64, c: (u8, u8, u8)) {
        let steps = (2.0 * std::f64::consts::PI * r).ceil() as i64 * 2;
        for s in 0..steps {
            let a = 2.0 * std::f64::consts::PI * (s as f64) / (steps as f64);
            let x = cx + r * a.cos();
            let y = cy + r * a.sin();
            let hwi = hw.ceil() as i64;
            for dy in -hwi..=hwi {
                for dx in -hwi..=hwi {
                    self.set(x.round() as i64 + dx, y.round() as i64 + dy, c);
                }
            }
        }
    }
    fn polygon(&mut self, verts: &[(f64, f64)], hw: i64, c: (u8, u8, u8)) {
        for i in 0..verts.len() {
            self.seg(verts[i], verts[(i + 1) % verts.len()], hw, c);
        }
    }
    /// Stamp one 5×7 glyph with each cell rendered as a `cell`×`cell` ink block.
    fn stamp(&mut self, rows: &[&str; 7], x0: i64, y0: i64, cell: i64) {
        for (ry, row) in rows.iter().enumerate() {
            for (rx, ch) in row.chars().enumerate() {
                if ch == '1' || ch == '#' {
                    for dy in 0..cell {
                        for dx in 0..cell {
                            self.set(x0 + rx as i64 * cell + dx, y0 + ry as i64 * cell + dy, BLACK);
                        }
                    }
                }
            }
        }
    }
    /// Stamp a whole label string left→right; returns the approximate centre.
    fn text(&mut self, s: &str, x0: i64, y0: i64, cell: i64) -> (f64, f64) {
        let mut x = x0;
        for ch in s.chars() {
            self.stamp(&glyph_rows(ch), x, y0, cell);
            x += 7 * cell; // 5 wide + 2 gap
        }
        let width = s.chars().count() as i64 * 7 * cell - 2 * cell;
        (
            (x0 + width / 2) as f64,
            (y0 + 7 * cell / 2) as f64,
        )
    }
    fn png(&self) -> Vec<u8> {
        encode_png_rgb(self.w, self.h, &self.rgb)
    }
}

/// Standard 5×7 digit font (the same shapes P-4's recogniser is trained on —
/// this is a FONT, not a per-value lookup).
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

fn glyph_rows(ch: char) -> [&'static str; 7] {
    if ch == '?' {
        return QUESTION_FONT;
    }
    if let Some(d) = ch.to_digit(10) {
        return DIGIT5X7[d as usize];
    }
    for (c, rows) in LETTER_FONT {
        if *c == ch {
            return *rows;
        }
    }
    panic!("no test font for '{}'", ch);
}

fn framed(n: usize, m: f64) -> (Canvas, f64, f64) {
    let mut c = Canvas::new(n, n);
    let lo = m;
    let hi = n as f64 - 1.0 - m;
    c.polygon(&[(lo, lo), (hi, lo), (hi, hi), (lo, hi)], 2, BLACK);
    (c, lo, hi)
}

fn labeled(res: LabelResult) -> LabeledPrimitives {
    match res {
        LabelResult::Labeled(l) => l,
        LabelResult::Incomplete { stage, reason } => {
            panic!("expected Labeled, got Incomplete[{}]: {}", stage, reason)
        }
    }
}

// =====================================================================
// REAL-STYLE: tilted-square-in-square + circle + corner triangle, labelled
// exactly like John's FullSizeRender: "100" in the tilted square, "24" in the
// corner triangle, "?" in the interior circle.
// =====================================================================
#[test]
fn tilted_square_labels_report() {
    let n = 640usize;
    let (mut c, lo, hi) = framed(n, 30.0);
    let mid = (lo + hi) / 2.0;
    // inscribed tilted square through the 4 outer-edge midpoints
    c.polygon(&[(mid, lo), (hi, mid), (mid, hi), (lo, mid)], 2, BLACK);
    // interior circle (the "?" target)
    let rc = (hi - lo) * 0.16;
    c.circle(mid, mid, rc, 1.5, BLACK);
    // inset corner triangle (near the top-left corner, outside the tilted sq)
    let t = lo + (hi - lo) * 0.06;
    let u = lo + (hi - lo) * 0.30;
    c.polygon(&[(t, t), (u, t), (t, u)], 2, BLACK);

    // labels (as pixels)
    // "100" inside the tilted square, upper area, clear of the circle
    c.text("100", (mid - 40.0) as i64, (lo + (hi - lo) * 0.20) as i64, 4);
    // "24" inside the corner triangle
    c.text("24", (t + 8.0) as i64, (t + 10.0) as i64, 3);
    // "?" at the centre, inside the circle
    c.text("?", (mid - 8.0) as i64, (mid - 14.0) as i64, 4);

    let lab = labeled(label_on_raster(&decode(&c.png()), false));
    eprintln!("[tilted-labels]\n{}", lab.report());

    // "?" must bind as the Target on the circle
    let q = lab.binding_for("?").expect("'?' should bind");
    assert_eq!(q.role, Role::Target, "'?' role");
    assert_eq!(q.primitive_kind, "circle", "'?' target primitive");

    // "100" must bind as an Area on a polygon (the tilted square) — NOT the
    // outer frame: verify the bound polygon is smaller than the outer square.
    let b100 = lab.binding_for("100").expect("'100' should bind");
    assert_eq!(b100.role, Role::Area, "'100' role");
    assert_eq!(b100.primitive_kind, "polygon", "'100' primitive");
    if let Primitive::Polygon { verts, .. } = &lab.primitives.items[b100.primitive_index] {
        let a = poly_area(verts);
        // outer frame area in normalized units ≈ 1.0; tilted square ≈ 0.5
        assert!(a < 0.8, "'100' should bind the inner tilted square, area={:.3}", a);
    } else {
        panic!("'100' bound to a non-polygon");
    }

    // "24" must bind as an Area on the small corner triangle (3 verts)
    let b24 = lab.binding_for("24").expect("'24' should bind");
    assert_eq!(b24.role, Role::Area, "'24' role");
    if let Primitive::Polygon { verts, .. } = &lab.primitives.items[b24.primitive_index] {
        assert_eq!(verts.len(), 3, "'24' should bind a triangle");
    } else {
        panic!("'24' bound to a non-polygon");
    }
}

// =====================================================================
// NOVEL SYNTHETIC #1 — square labelled "36" + circle "?" (values/positions the
// code never saw). Proves general reading + role binding.
// =====================================================================
#[test]
fn novel_square36_circle_q() {
    // A standalone square (labelled "36") on the left and a standalone circle
    // (labelled "?") on the right — two disjoint regions, new values/positions.
    let mut c = Canvas::new(640, 380);
    c.polygon(&[(60.0, 70.0), (300.0, 70.0), (300.0, 310.0), (60.0, 310.0)], 2, BLACK);
    c.text("36", 150, 175, 4); // inside the square
    c.circle(480.0, 190.0, 110.0, 1.5, BLACK);
    c.text("?", 470, 172, 4); // inside the circle

    let lab = labeled(label_on_raster(&decode(&c.png()), false));
    eprintln!("[novel-36]\n{}", lab.report());

    let b36 = lab.binding_for("36").expect("'36' should bind");
    assert_eq!(b36.role, Role::Area);
    assert_eq!(b36.primitive_kind, "polygon");
    let q = lab.binding_for("?").expect("'?' should bind");
    assert_eq!(q.role, Role::Target);
    // the detector may report the disk as a full Circle or a large Arc; both are
    // circular regions and bind the target to the disk.
    assert!(
        q.primitive_kind == "circle" || q.primitive_kind == "arc",
        "'?' should target a circular region, got {}",
        q.primitive_kind
    );
}

// =====================================================================
// NOVEL SYNTHETIC #2 — triangle "12" (area) beside a lone segment "5"
// (length). Two DIFFERENT roles at new positions/values.
// =====================================================================
#[test]
fn novel_triangle12_segment5() {
    let n = 460usize;
    let mut c = Canvas::new(n, n);
    // triangle on the left (an area region)
    c.polygon(&[(70.0, 330.0), (250.0, 330.0), (160.0, 150.0)], 2, BLACK);
    c.text("12", 120, 250, 3); // near the triangle centroid

    // a lone vertical segment on the right, with "5" beside it (length label)
    c.seg((330.0, 90.0), (330.0, 340.0), 2, BLACK);
    c.text("5", 342, 200, 3); // ~12px to the right of the segment line

    let lab = labeled(label_on_raster(&decode(&c.png()), false));
    eprintln!("[novel-12-5]\n{}", lab.report());

    let b12 = lab.binding_for("12").expect("'12' should bind");
    assert_eq!(b12.role, Role::Area, "'12' area");
    assert_eq!(b12.primitive_kind, "polygon");

    let b5 = lab.binding_for("5").expect("'5' should bind");
    assert_eq!(b5.role, Role::Length, "'5' length");
    assert_eq!(b5.primitive_kind, "segment");
}

// =====================================================================
// Letter vertex naming — letters at triangle corners bind to the vertices.
// (Uses A, C, X: none collide pixel-wise with a digit.)
// =====================================================================
#[test]
fn letter_vertex_names() {
    let n = 460usize;
    let mut c = Canvas::new(n, n);
    let a = (200.0, 130.0);
    let b = (100.0, 330.0);
    let d = (300.0, 330.0);
    c.polygon(&[a, b, d], 2, BLACK);
    c.text("A", 190, 96, 3); // just above apex
    c.text("C", 74, 336, 3); // just below-left of B corner
    c.text("X", 300, 336, 3); // just below-right of D corner

    let lab = labeled(label_on_raster(&decode(&c.png()), false));
    eprintln!("[letters]\n{}", lab.report());

    let named = lab
        .bindings
        .iter()
        .filter(|b| b.role == Role::VertexName)
        .count();
    assert!(named >= 2, "expected ≥2 letters bound to vertices, got {}", named);
    for b in lab.bindings.iter().filter(|b| b.role == Role::VertexName) {
        assert_eq!(b.primitive_kind, "point");
    }
}

// =====================================================================
// ANTI-CHEAT: an UNLABELED figure yields NO tokens and NO bindings — the code
// never invents a label. (Primitives are still detected honestly.)
// =====================================================================
#[test]
fn unlabeled_all_unbound() {
    let n = 440usize;
    let (mut c, lo, hi) = framed(n, 22.0);
    let mid = (lo + hi) / 2.0;
    c.circle(mid, mid, (hi - lo) * 0.3, 1.5, BLACK);
    let lab = labeled(label_on_raster(&decode(&c.png()), false));
    eprintln!("[unlabeled]\n{}", lab.report());
    assert_eq!(lab.tokens.len(), 0, "no glyphs ⇒ no tokens read");
    assert!(lab.bindings.is_empty(), "no tokens ⇒ no bindings invented");
    assert!(lab.unbound.is_empty(), "no tokens ⇒ nothing to leave unbound");
    // primitives were still detected honestly (the outer frame at minimum)
    assert!(!lab.primitives.items.is_empty(), "figure should still yield primitives");
}

// =====================================================================
// ANTI-CHEAT: an AMBIGUOUS label (equidistant to two segments) is left UNBOUND
// with a reason — never force-bound. Built directly with primitives + a token.
// =====================================================================
#[test]
fn ambiguous_equidistant_unbound() {
    // two parallel vertical segments; a number centred exactly between them
    let prims = Primitives {
        items: vec![
            Primitive::Segment { a: (0.36, 0.20), b: (0.36, 0.80), confidence: 0.9, support: 100 },
            Primitive::Segment { a: (0.44, 0.20), b: (0.44, 0.80), confidence: 0.9, support: 100 },
        ],
        rectified: false,
        mean_confidence: 0.9,
        note: String::new(),
        relations: Vec::new(),
    };
    let tok = LabelToken {
        text: "7".into(),
        kind: LabelKind::Number(7),
        center: (0.40, 0.50), // equidistant (0.10) from both lines, in-span
        bbox: (0.39, 0.49, 0.41, 0.51),
        confidence: 0.9,
    };
    let (bindings, unbound) = associate(&prims, &[tok]);
    eprintln!("[ambiguous] bindings={} unbound={}", bindings.len(), unbound.len());
    assert!(bindings.is_empty(), "must NOT force-bind an equidistant label");
    assert_eq!(unbound.len(), 1);
    assert!(
        unbound[0].reason.contains("equidistant"),
        "reason should cite equidistance, got: {}",
        unbound[0].reason
    );
}

// =====================================================================
// ANTI-CHEAT: a garbled glyph classifies BELOW confidence ⇒ not emitted.
// =====================================================================
#[test]
fn garbled_glyph_below_confidence() {
    let templates = build_templates();
    // a checkerboard bitmap resembles no digit/letter/? ⇒ low fit + low margin
    let mut nb = [false; NORM * NORM];
    for y in 0..NORM {
        for x in 0..NORM {
            nb[y * NORM + x] = (x + y) % 2 == 0;
        }
    }
    let (_sym, conf) = classify(&nb, &templates);
    eprintln!("[garbled] conf={:.3}", conf);
    assert!(conf < TOKEN_CONF_MIN, "garbled glyph must fall below threshold, conf={:.3}", conf);
}

// =====================================================================
// Sanity: a clean digit rendered as pixels IS read (proves the OCR path works
// end-to-end, independent of association).
// =====================================================================
#[test]
fn reads_clean_multidigit() {
    let n = 300usize;
    let mut c = Canvas::new(n, n);
    c.text("48", 120, 130, 4);
    let toks = read_tokens_on_raster(&decode(&c.png()));
    eprintln!("[read48] {:?}", toks.iter().map(|t| (t.text.clone(), t.confidence)).collect::<Vec<_>>());
    assert!(toks.iter().any(|t| t.text == "48"), "should read '48'");
}

fn decode(bytes: &[u8]) -> RgbImage {
    crate::vision::geo_perceive_p4::decode_image_rgb(bytes).expect("decode")
}

// =====================================================================
// REAL IMAGE: redleaf.jpeg (on the box). Reports verbatim; skips if absent.
// The scale digit sits in the margin OUTSIDE the square — read as `scale`.
// =====================================================================
#[test]
fn real_redleaf_scale_report() {
    let path = "prototypes/perception/redleaf.jpeg";
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[redleaf-labels] {} not present — skipping", path);
            return;
        }
    };
    match read_labels(&bytes) {
        LabelResult::Labeled(lab) => {
            eprintln!("[redleaf-labels]\n{}", lab.report());
            // Report-only: we do NOT assert a particular scale value — the
            // honest outcome is whatever the OCR reads from the photo's pixels.
            let _ = detect_primitives(&bytes); // exercise the shared path too
        }
        LabelResult::Incomplete { stage, reason } => {
            eprintln!("[redleaf-labels] Incomplete[{}]: {}", stage, reason);
        }
    }
}
