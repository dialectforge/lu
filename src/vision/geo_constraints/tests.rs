//! P-D constraint extraction — in-crate tests.
//!
//! Coverage:
//!   * CONTROLLED end-to-end on hand-built CLEAN labelled scenes (bypassing
//!     perception noise): the tilted-square incircle derives 4*pi, and three
//!     mutually-tangent circles in a unit circle derive (21-12*sqrt(3))*pi —
//!     proving the extractor->algsolve path itself is correct.
//!   * REAL-STYLE end-to-end FROM PIXELS (decode -> P-B -> P-C -> P-D -> solve)
//!     on the synthetic John's-FullSizeRender figure: reports the honest
//!     outcome (Incomplete naming the missing relation) — no fabricated answer.
//!   * ANTI-CHEAT: rescaled labels give a DIFFERENT derived answer (computed,
//!     not looked up); a missing tangency yields Incomplete; an inconsistent
//!     (impossible) label makes the solver refuse.

use super::*;
use crate::math::algnum::Alg;
use crate::math::algsolve::{fmt_alg, solve, Solved};
use crate::vision::geo_labels::{Binding, LabeledPrimitives, Role};
use crate::vision::geo_primitives::{Primitive, Primitives};
use num_bigint::BigInt;
use num_rational::BigRational;

// ---------------------------------------------------------------------
// hand-scene construction helpers
// ---------------------------------------------------------------------

fn bind(label: &str, role: Role, idx: usize, kind: &'static str) -> Binding {
    Binding {
        token_index: 0,
        label: label.to_string(),
        role,
        primitive_index: idx,
        primitive_kind: kind,
        confidence: 0.9,
        evidence: "hand-built clean scene".into(),
    }
}

fn labeled(items: Vec<Primitive>, bindings: Vec<Binding>) -> LabeledPrimitives {
    let mean = if items.is_empty() {
        0.0
    } else {
        items.iter().map(|p| p.confidence()).sum::<f64>() / items.len() as f64
    };
    LabeledPrimitives {
        primitives: Primitives { items, rectified: true, mean_confidence: mean, note: String::new(), relations: Vec::new() },
        tokens: Vec::new(),
        bindings,
        unbound: Vec::new(),
        scale: None,
        rectified: true,
    }
}

/// Clean tilted-square incircle scene at linear scale `s` (normalized /20).
/// Legs (8s,6s) -> tilted area 100s^2, corner-triangle area 24s^2, inradius 2s.
/// If `circle_r_units` is Some, override the incircle radius (to break tangency).
fn incircle_scene(s: f64, area_tilt: i64, area_tri: i64, circle_r_units: Option<f64>) -> LabeledPrimitives {
    let nrm = 20.0;
    let a = 8.0 * s;
    let b = 6.0 * s;
    let tilt = [
        (a / nrm, 0.0),
        (14.0 * s / nrm, 8.0 * s / nrm),
        (6.0 * s / nrm, 14.0 * s / nrm),
        (0.0, b / nrm),
    ];
    let tri = [(0.0, 0.0), (a / nrm, 0.0), (0.0, b / nrm)];
    let cr = circle_r_units.unwrap_or(2.0 * s) / nrm;
    let items = vec![
        Primitive::Polygon { verts: tilt.to_vec(), confidence: 0.9 },
        Primitive::Polygon { verts: tri.to_vec(), confidence: 0.9 },
        Primitive::Circle { center: (2.0 * s / nrm, 2.0 * s / nrm), r: cr, confidence: 0.9, support: 200 },
        Primitive::RightAngle { at: (0.0, 0.0), confidence: 0.6 },
    ];
    let bindings = vec![
        bind(&area_tilt.to_string(), Role::Area, 0, "polygon"),
        bind(&area_tri.to_string(), Role::Area, 1, "polygon"),
        bind("?", Role::Target, 2, "circle"),
    ];
    labeled(items, bindings)
}

/// Clean three-equal-circles-in-a-unit-circle scene. Outer radius label R=1.
fn three_circles_scene() -> LabeledPrimitives {
    let oc = (0.5, 0.5);
    let or = 0.30;
    let ratio = 2.0 * 3.0_f64.sqrt() - 3.0; // r_in / r_out
    let r_in = or * ratio;
    let d = or - r_in; // |O Ci|
    let mut items = vec![Primitive::Circle { center: oc, r: or, confidence: 0.9, support: 300 }];
    for k in 0..3 {
        let ang = std::f64::consts::FRAC_PI_2 + k as f64 * 2.0 * std::f64::consts::PI / 3.0;
        items.push(Primitive::Circle {
            center: (oc.0 + d * ang.cos(), oc.1 + d * ang.sin()),
            r: r_in,
            confidence: 0.9,
            support: 120,
        });
    }
    let bindings = vec![
        bind("1", Role::Radius, 0, "circle"),
        bind("?", Role::Target, 1, "circle"),
    ];
    labeled(items, bindings)
}

fn ratio_(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

// ---------------------------------------------------------------------
// CONTROLLED end-to-end #1 — tilted-square incircle -> 4*pi
// ---------------------------------------------------------------------
#[test]
fn controlled_incircle_derives_4pi() {
    let lab = incircle_scene(1.0, 100, 24, None);
    let res = extract_problem(&lab);
    let (prob, target) = match res {
        ExtractResult::Problem(p, t) => (p, t),
        ExtractResult::Incomplete { reason } => panic!("incircle should assemble, got Incomplete: {}", reason),
    };
    eprintln!("[incircle assembled]\n{}", describe(&prob, &target));
    match solve(&prob, &target) {
        Solved::Value { rat, pic, text } => {
            eprintln!("[incircle solved] {}", text);
            assert!(rat.is_zero(), "area must be pure pi, got rat={}", fmt_alg(&rat));
            assert!(pic.eq(&Alg::int(4)), "expected 4*pi, got {}", text);
        }
        Solved::Refused(r) => panic!("incircle should solve, refused {:?}", r),
    }
}

// ---------------------------------------------------------------------
// CONTROLLED end-to-end #2 — three tangent circles -> (21 - 12*sqrt(3))*pi
// ---------------------------------------------------------------------
#[test]
fn controlled_three_circles_derives_area() {
    let lab = three_circles_scene();
    let res = extract_problem(&lab);
    let (prob, target) = match res {
        ExtractResult::Problem(p, t) => (p, t),
        ExtractResult::Incomplete { reason } => panic!("three-circles should assemble, got Incomplete: {}", reason),
    };
    eprintln!("[three-circles assembled]\n{}", describe(&prob, &target));
    match solve(&prob, &target) {
        Solved::Value { rat, pic, text } => {
            eprintln!("[three-circles solved] {}", text);
            assert!(rat.is_zero(), "disk area must be pure pi, got rat={}", fmt_alg(&rat));
            // expected pic = (2*sqrt(3) - 3)^2 = 21 - 12*sqrt(3)
            let base = Alg::sqrt_rational(&ratio_(3, 1)).mul(&Alg::int(2)).sub(&Alg::int(3));
            let expected = base.mul(&base);
            assert!(pic.eq(&expected), "expected (21 - 12*sqrt(3))*pi, got {}", text);
            assert!(!pic.primes().is_empty(), "answer must carry a radical");
        }
        Solved::Refused(r) => panic!("three-circles should solve, refused {:?}", r),
    }
}

// ---------------------------------------------------------------------
// ANTI-CHEAT: rescaled labels -> DIFFERENT derived answer (computed, not
// looked up). Linear x1.5 => areas x2.25 (225/54) => 9*pi (r=3, not 2).
// ---------------------------------------------------------------------
#[test]
fn rescaled_incircle_derives_9pi_not_4pi() {
    let lab = incircle_scene(1.5, 225, 54, None);
    let (prob, target) = match extract_problem(&lab) {
        ExtractResult::Problem(p, t) => (p, t),
        ExtractResult::Incomplete { reason } => panic!("rescaled incircle should assemble: {}", reason),
    };
    match solve(&prob, &target) {
        Solved::Value { pic, text, .. } => {
            eprintln!("[rescaled solved] {}", text);
            assert!(pic.eq(&Alg::int(9)), "expected 9*pi, got {}", text);
            assert!(!pic.eq(&Alg::int(4)), "must NOT return the base-case 4*pi constant");
        }
        Solved::Refused(r) => panic!("rescaled incircle should solve, refused {:?}", r),
    }
}

// ---------------------------------------------------------------------
// HONESTY: a labelled scene missing the incircle tangency -> Incomplete
// (not a forced solve). Circle radius is wrong, so tangency is not measured.
// ---------------------------------------------------------------------
#[test]
fn missing_tangency_incomplete() {
    // true inradius is 2 units; give the circle radius 1.2 units -> not tangent.
    let lab = incircle_scene(1.0, 100, 24, Some(1.2));
    match extract_problem(&lab) {
        ExtractResult::Incomplete { reason } => {
            eprintln!("[missing-tangency] Incomplete: {}", reason);
            assert!(reason.contains("tangen"), "reason should cite tangency, got: {}", reason);
        }
        ExtractResult::Problem(..) => panic!("must NOT assemble a solvable problem without measured tangency"),
    }
}

// ---------------------------------------------------------------------
// HONESTY: an inconsistent (impossible) label -> the solver refuses. Corner-
// triangle area 999 is geometrically impossible with tilted area 100.
// ---------------------------------------------------------------------
#[test]
fn inconsistent_scene_refuses() {
    let lab = incircle_scene(1.0, 100, 999, None);
    let (prob, target) = match extract_problem(&lab) {
        ExtractResult::Problem(p, t) => (p, t),
        ExtractResult::Incomplete { reason } => {
            eprintln!("[inconsistent] Incomplete: {}", reason);
            return; // an honest Incomplete is also acceptable
        }
    };
    match solve(&prob, &target) {
        Solved::Refused(r) => eprintln!("[inconsistent] honest refuse: {:?}", r),
        Solved::Value { text, .. } => panic!("impossible scene must NOT yield a value, emitted {}", text),
    }
}

// ---------------------------------------------------------------------
// HONESTY: no target binding -> Incomplete.
// ---------------------------------------------------------------------
#[test]
fn no_target_incomplete() {
    let lab = incircle_scene(1.0, 100, 24, None);
    let mut lab2 = lab;
    lab2.bindings.retain(|b| b.role != Role::Target);
    match extract_problem(&lab2) {
        ExtractResult::Incomplete { reason } => {
            eprintln!("[no-target] Incomplete: {}", reason);
            assert!(reason.contains("target"));
        }
        ExtractResult::Problem(..) => panic!("no '?' -> must be Incomplete"),
    }
}

// ---------------------------------------------------------------------
// REAL-STYLE end-to-end FROM PIXELS: decode -> P-B -> P-C -> P-D -> solve on
// the synthetic John's-FullSizeRender figure. Honest outcome reported verbatim.
// ---------------------------------------------------------------------
mod pixels {
    use super::*;
    use crate::vision::geo_labels::{label_on_raster, LabelResult};
    use crate::vision::geo_perceive::pngtest::encode_png_rgb;
    use crate::vision::geo_perceive_p4::decode_image_rgb;

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

    #[test]
    fn end_to_end_from_pixels_reports_honest_outcome() {
        // John's-FullSizeRender-equivalent: outer square, inscribed tilted
        // square ("100"), corner triangle ("24"), interior circle ("?").
        let n = 640usize;
        let (mut c, lo, hi) = framed(n, 30.0);
        let mid = (lo + hi) / 2.0;
        c.polygon(&[(mid, lo), (hi, mid), (mid, hi), (lo, mid)], 2);
        c.circle(mid, mid, (hi - lo) * 0.16);
        let t = lo + (hi - lo) * 0.06;
        let u = lo + (hi - lo) * 0.30;
        c.polygon(&[(t, t), (u, t), (t, u)], 2);
        c.text("100", (mid - 40.0) as i64, (lo + (hi - lo) * 0.20) as i64, 4);
        c.text("24", (t + 8.0) as i64, (t + 10.0) as i64, 3);
        c.text("?", (mid - 8.0) as i64, (mid - 14.0) as i64, 4);

        let img = decode_image_rgb(&c.png()).expect("decode");
        let lab = match label_on_raster(&img, false) {
            LabelResult::Labeled(l) => l,
            LabelResult::Incomplete { stage, reason } => {
                eprintln!("[e2e] P-C Incomplete[{}]: {}", stage, reason);
                return;
            }
        };
        eprintln!("[e2e] P-C report:\n{}", lab.report());
        match extract_problem(&lab) {
            ExtractResult::Problem(prob, target) => {
                eprintln!("[e2e] P-D assembled:\n{}", describe(&prob, &target));
                match solve(&prob, &target) {
                    Solved::Value { text, .. } => eprintln!("[e2e] SOLVED (auto chain closed): {}", text),
                    Solved::Refused(r) => eprintln!("[e2e] solver refused honestly: {:?}", r),
                }
            }
            ExtractResult::Incomplete { reason } => {
                eprintln!("[e2e] P-D Incomplete (honest): {}", reason);
            }
        }
    }

    // ---- FAITHFUL John's-FullSizeRender geometry ------------------------
    // Outer square (S=14 units), inscribed tilted square (touch points at 8/14
    // and 6/14 along the sides => area 100), the corner triangle at O (legs 8,6
    // => area 24), and the incircle of that corner triangle (r=2) as the "?".
    // `tangent`=true draws the circle as the true corner incircle (tangent to
    // the two outer sides + the tilted edge); false floats it at the centre
    // (tangent to nothing) for the honesty case. Labels are parameters so the
    // SAME pixels can be labelled 100/24 or 225/54 (both consistent with the 8:6
    // tilt) -- proving the answer is derived from the read labels, not looked up.
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
            // floating inside the corner triangle but wrong size/position: tangent
            // to NOTHING (measured), yet a cleanly detected circle so the target
            // is circular and P-D can cite the missing tangency.
            c.circle(mid, mid, (hi - lo) * 0.13);
            c.text("?", (mid - 8.0) as i64, (mid - 14.0) as i64, 4);
        }
        let tl_y = if tangent { mid - 14.0 } else { lo + (hi - lo) * 0.80 };
        c.text(tilt_label, (mid - 42.0) as i64, tl_y as i64, 4);
        c.text(tri_label, (lo + 5.0 * sc) as i64, (lo + 0.4 * sc) as i64, 3);
        c.png()
    }

    // ---- MILESTONE: FULL AUTO CLOSE pixels -> answer --------------------
    #[test]
    fn full_auto_close_on_faithful_render() {
        let img = decode_image_rgb(&render_john(640, "100", "24", true)).expect("decode");
        let lab = match label_on_raster(&img, false) {
            LabelResult::Labeled(l) => l,
            LabelResult::Incomplete { stage, reason } => panic!("P-C Incomplete[{}]: {}", stage, reason),
        };
        eprintln!("[full-auto] P-C report:\n{}", lab.report());
        use crate::vision::geo_primitives::Relation;
        let tang = lab.primitives.relations.iter().filter(|r| matches!(r, Relation::CircleSegmentTangent { .. })).count();
        let ra = lab.primitives.relations.iter().filter(|r| matches!(r, Relation::RightAngleMeasured { .. })).count();
        eprintln!("[full-auto] P-B relations: circle-seg tangencies={}, measured right-angles={}", tang, ra);
        assert!(tang >= 2, "circle must be perceived tangent to >=2 sides, got {}", tang);
        assert!(ra >= 1, "a right-angle corner must be perceived, got {}", ra);
        match extract_problem(&lab) {
            ExtractResult::Problem(prob, target) => {
                let desc = describe(&prob, &target);
                eprintln!("[full-auto] P-D assembled:\n{}", desc);
                assert!(desc.contains("PtLineDistEqCoord"), "must include the tangency constraint");
                assert!(desc.contains("Perp"), "must include the perp constraint");
                match solve(&prob, &target) {
                    Solved::Value { rat, pic, text } => {
                        eprintln!("[full-auto] SOLVED (pixels->answer): {}", text);
                        assert!(rat.is_zero(), "area must be pure pi, got rat={}", fmt_alg(&rat));
                        assert!(pic.eq(&Alg::int(4)), "expected 4*pi, got {}", text);
                        assert_eq!(text, "4*pi");
                    }
                    Solved::Refused(r) => panic!("full auto should solve, refused {:?}", r),
                }
            }
            ExtractResult::Incomplete { reason } => panic!("full auto should close, Incomplete: {}", reason),
        }
    }

    // ---- ANTI-CHEAT: identical pixels, relabelled 225/54 -> 9*pi --------
    #[test]
    fn auto_close_rescaled_labels_derives_9pi() {
        let img = decode_image_rgb(&render_john(640, "225", "54", true)).expect("decode");
        let lab = match label_on_raster(&img, false) {
            LabelResult::Labeled(l) => l,
            LabelResult::Incomplete { stage, reason } => panic!("P-C Incomplete[{}]: {}", stage, reason),
        };
        match extract_problem(&lab) {
            ExtractResult::Problem(prob, target) => match solve(&prob, &target) {
                Solved::Value { pic, text, .. } => {
                    eprintln!("[rescale] SOLVED: {}", text);
                    assert!(pic.eq(&Alg::int(9)), "expected 9*pi, got {}", text);
                    assert!(!pic.eq(&Alg::int(4)), "must NOT return the base 4*pi constant");
                }
                Solved::Refused(r) => panic!("rescaled should solve, refused {:?}", r),
            },
            ExtractResult::Incomplete { reason } => panic!("rescaled should close: {}", reason),
        }
    }

    // ---- ANTI-CHEAT / HONESTY: non-tangent circle -> Incomplete --------
    #[test]
    fn non_tangent_circle_stays_incomplete() {
        let img = decode_image_rgb(&render_john(640, "100", "24", false)).expect("decode");
        let lab = match label_on_raster(&img, false) {
            LabelResult::Labeled(l) => l,
            LabelResult::Incomplete { stage, reason } => panic!("P-C Incomplete[{}]: {}", stage, reason),
        };
        eprintln!("[non-tangent] P-C report:\n{}", lab.report());
        use crate::vision::geo_primitives::Relation;
        let tang = lab.primitives.relations.iter().filter(|r| matches!(r, Relation::CircleSegmentTangent { .. })).count();
        eprintln!("[non-tangent] circle-seg tangencies detected = {}", tang);
        assert_eq!(tang, 0, "a non-tangent circle must yield NO tangency relation (measured, not assumed)");
        match extract_problem(&lab) {
            ExtractResult::Incomplete { reason } => {
                eprintln!("[non-tangent] honest Incomplete: {}", reason);
                assert!(reason.contains("tangen") || reason.contains("incidence"), "reason should cite the missing tangency: {}", reason);
            }
            ExtractResult::Problem(prob, target) => match solve(&prob, &target) {
                Solved::Refused(r) => eprintln!("[non-tangent] solver refused honestly: {:?}", r),
                Solved::Value { text, .. } => panic!("non-tangent circle must NOT yield a value, emitted {}", text),
            },
        }
    }
}
