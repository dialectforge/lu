//! P-B general primitive detector — in-crate tests: real images, novel
//! synthetic figures (proving GENERAL detection, not template lookup), and
//! anti-cheat / honesty cases.

use super::*;
use crate::vision::geo_perceive::pngtest::encode_png_rgb;
use std::fs;

// ---------------------------------------------------------------------
// tiny software rasteriser (test-only) — draws black ink figures on white.
// ---------------------------------------------------------------------

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
    fn png(&self) -> Vec<u8> {
        encode_png_rgb(self.w, self.h, &self.rgb)
    }
}

const BLACK: (u8, u8, u8) = (0, 0, 0);

/// A framed figure: outer square border on an N×N canvas, margin `m`.
fn framed(n: usize, m: f64) -> (Canvas, f64, f64) {
    let mut c = Canvas::new(n, n);
    let lo = m;
    let hi = n as f64 - 1.0 - m;
    c.polygon(&[(lo, lo), (hi, lo), (hi, hi), (lo, hi)], 2, BLACK);
    (c, lo, hi)
}

fn fmt_prims(ps: &Primitives) -> String {
    let mut s = format!("{}\n", ps.census());
    for p in &ps.items {
        match p {
            Primitive::Segment { a, b, confidence, support } => {
                s.push_str(&format!(
                    "  Segment ({:.3},{:.3})-({:.3},{:.3}) conf={:.2} support={}\n",
                    a.0, a.1, b.0, b.1, confidence, support
                ))
            }
            Primitive::Circle { center, r, confidence, support } => s.push_str(&format!(
                "  Circle c=({:.3},{:.3}) r={:.3} conf={:.2} support={}\n",
                center.0, center.1, r, confidence, support
            )),
            Primitive::Arc { center, r, start, end, confidence, support } => s.push_str(&format!(
                "  Arc c=({:.3},{:.3}) r={:.3} sweep[{:.2}..{:.2}] conf={:.2} support={}\n",
                center.0, center.1, r, start, end, confidence, support
            )),
            Primitive::Polygon { verts, confidence } => s.push_str(&format!(
                "  Polygon[{}] {:?} conf={:.2}\n",
                verts.len(),
                verts.iter().map(|v| (((v.0) * 1000.0).round() / 1000.0, ((v.1) * 1000.0).round() / 1000.0)).collect::<Vec<_>>(),
                confidence
            )),
            Primitive::Point { p, confidence } => {
                s.push_str(&format!("  Point ({:.3},{:.3}) conf={:.2}\n", p.0, p.1, confidence))
            }
            Primitive::RightAngle { at, confidence } => {
                s.push_str(&format!("  RightAngle at=({:.3},{:.3}) conf={:.2}\n", at.0, at.1, confidence))
            }
        }
    }
    s
}

fn prims(res: DetectResult) -> Primitives {
    match res {
        DetectResult::Primitives(p) => p,
        DetectResult::Incomplete { stage, reason } => panic!("expected primitives, got Incomplete[{}]: {}", stage, reason),
    }
}

// =====================================================================
// REAL IMAGE: redleaf.jpeg (on the box). Reports verbatim; skips cleanly if
// the file is not present in this checkout.
// =====================================================================
#[test]
fn real_redleaf_report() {
    let path = "prototypes/perception/redleaf.jpeg";
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[redleaf] {} not present — skipping", path);
            return;
        }
    };
    match detect_primitives(&bytes) {
        DetectResult::Primitives(p) => {
            eprintln!("[redleaf] {}", fmt_prims(&p));
            // redleaf is a filled leaf inside a square: we expect the outer
            // square edges AND at least one circle/arc from the leaf boundary.
            assert!(p.count("segment") >= 2, "redleaf: expected outer square edges");
            assert!(
                p.count("circle") + p.count("arc") >= 1,
                "redleaf: expected at least one leaf circle/arc"
            );
        }
        DetectResult::Incomplete { stage, reason } => {
            eprintln!("[redleaf] Incomplete[{}]: {}", stage, reason);
            panic!("redleaf should yield primitives");
        }
    }
}

// =====================================================================
// REAL-STYLE FIGURE: tilted-square-in-square + circle + corner triangle
// (a synthetic equivalent of John's FullSizeRender). Reports verbatim.
// =====================================================================
#[test]
fn tilted_square_figure_report() {
    let n = 420usize;
    let (mut c, lo, hi) = framed(n, 24.0);
    let mid = (lo + hi) / 2.0;
    // inscribed tilted square through the 4 midpoints of the outer square
    c.polygon(&[(mid, lo), (hi, mid), (mid, hi), (lo, mid)], 2, BLACK);
    // a circle inside
    c.circle(mid, mid, (hi - lo) * 0.22, 1.5, BLACK);
    // an inset corner triangle (3 distinct edges, not collinear with the frame)
    let t = lo + (hi - lo) * 0.10;
    let u = lo + (hi - lo) * 0.34;
    c.polygon(&[(t, t), (u, t), (t, u)], 2, BLACK);
    let png = c.png();
    let p = prims(detect_primitives(&png));
    eprintln!("[tilted] {}", fmt_prims(&p));
    // honest expectations: outer square + tilted square ⇒ several segments and
    // at least two quadrilaterals; the interior circle should be found.
    assert!(p.count("segment") >= 6, "tilted: expected many segments");
    assert!(p.count("circle") >= 1, "tilted: expected the interior circle");
    let quads = p.items.iter().filter(|q| matches!(q, Primitive::Polygon { verts, .. } if verts.len() == 4)).count();
    assert!(quads >= 1, "tilted: expected at least one quadrilateral");
}

// =====================================================================
// NOVEL SYNTHETIC (never-seen shapes/scales/rotations) — proves GENERAL
// detection, not per-figure lookup.
// =====================================================================
#[test]
fn novel_triangle_in_circle() {
    let n = 400usize;
    let (mut c, lo, hi) = framed(n, 20.0);
    let mid = (lo + hi) / 2.0;
    let rc = (hi - lo) * 0.40;
    c.circle(mid, mid, rc, 1.5, BLACK);
    let r = (hi - lo) * 0.20; // triangle strictly inside the circle
    // rotated triangle — 3 distinct edges
    let mut v = Vec::new();
    for k in 0..3 {
        let a = -std::f64::consts::FRAC_PI_2 + k as f64 * 2.0 * std::f64::consts::PI / 3.0 + 0.3;
        v.push((mid + r * a.cos(), mid + r * a.sin()));
    }
    c.polygon(&v, 2, BLACK);
    let p = prims(detect_primitives(&c.png()));
    eprintln!("[triangle_in_circle] {}", fmt_prims(&p));
    assert_eq!(p.count("circle"), 1, "expected exactly one circle");
    let tris = p.items.iter().filter(|q| matches!(q, Primitive::Polygon { verts, .. } if verts.len() == 3)).count();
    assert!(tris >= 1, "expected the inscribed triangle");
}

#[test]
fn novel_two_circles() {
    let n = 400usize;
    let (mut c, lo, hi) = framed(n, 20.0);
    let span = hi - lo;
    c.circle(lo + span * 0.28, lo + span * 0.5, span * 0.15, 1.5, BLACK);
    c.circle(lo + span * 0.74, lo + span * 0.5, span * 0.15, 1.5, BLACK);
    let p = prims(detect_primitives(&c.png()));
    eprintln!("[two_circles] {}", fmt_prims(&p));
    assert_eq!(p.count("circle"), 2, "expected exactly two circles");
}

#[test]
fn novel_pentagon() {
    let n = 440usize;
    let (mut c, lo, hi) = framed(n, 22.0);
    let mid = (lo + hi) / 2.0;
    let r = (hi - lo) * 0.40;
    let mut v = Vec::new();
    for k in 0..5 {
        let a = -std::f64::consts::FRAC_PI_2 + k as f64 * 2.0 * std::f64::consts::PI / 5.0;
        v.push((mid + r * a.cos(), mid + r * a.sin()));
    }
    c.polygon(&v, 2, BLACK);
    let p = prims(detect_primitives(&c.png()));
    eprintln!("[pentagon] {}", fmt_prims(&p));
    let pent = p.items.iter().filter(|q| matches!(q, Primitive::Polygon { verts, .. } if verts.len() == 5)).count();
    assert!(pent >= 1, "expected a pentagon polygon");
    assert_eq!(p.count("circle"), 0, "pentagon figure has no circle");
}

// =====================================================================
// ANTI-CHEAT / HONESTY
// =====================================================================
#[test]
fn honesty_blank_image_incomplete() {
    // pure white — no geometry at all
    let c = Canvas::new(300, 300);
    match detect_primitives(&c.png()) {
        DetectResult::Incomplete { stage, reason } => {
            eprintln!("[blank] Incomplete[{}]: {}", stage, reason);
        }
        DetectResult::Primitives(p) => panic!("blank image must NOT yield primitives, got: {}", p.census()),
    }
}

#[test]
fn honesty_garbage_bytes_incomplete() {
    let junk: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    match detect_primitives(&junk) {
        DetectResult::Incomplete { .. } => {}
        DetectResult::Primitives(p) => panic!("garbage bytes must NOT yield primitives, got: {}", p.census()),
    }
}

#[test]
fn honesty_degraded_image_low_or_incomplete() {
    // a valid framed circle, then heavy salt-and-pepper noise
    let n = 360usize;
    let (mut c, lo, hi) = framed(n, 18.0);
    let mid = (lo + hi) / 2.0;
    c.circle(mid, mid, (hi - lo) * 0.3, 1.5, BLACK);
    // deterministic noise
    let mut seed = 0x9e3779b97f4a7c15u64;
    for _ in 0..(n * n / 6) {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((seed >> 20) as usize) % n;
        let y = ((seed >> 40) as usize) % n;
        let v = if (seed & 1) == 0 { 0u8 } else { 255u8 };
        c.set(x as i64, y as i64, (v, v, v));
    }
    // must not panic and must not hallucinate an absurd number of primitives
    match detect_primitives(&c.png()) {
        DetectResult::Incomplete { stage, reason } => {
            eprintln!("[degraded] Incomplete[{}]: {}", stage, reason)
        }
        DetectResult::Primitives(p) => {
            eprintln!("[degraded] {}", fmt_prims(&p));
            assert!(p.mean_confidence.is_finite());
            // honesty: noise should not manufacture dozens of confident circles
            assert!(p.count("circle") <= 3, "degraded image hallucinated too many circles");
        }
    }
}

#[test]
fn algsolve_hint_maps_vocabulary() {
    let n = 400usize;
    let (mut c, lo, hi) = framed(n, 20.0);
    let mid = (lo + hi) / 2.0;
    c.circle(mid, mid, (hi - lo) * 0.3, 1.5, BLACK);
    let p = prims(detect_primitives(&c.png()));
    let hint = p.algsolve_hint();
    eprintln!("[hint]\n{}", hint);
    assert!(hint.contains("algsolve"));
}
