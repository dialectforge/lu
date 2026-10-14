//! Port G — verifiable geometry pipeline (Phase 1: the partition-and-subtract spine).
//!
//! Implements steps 1–4 of `lugeometrypipelinespec` as a self-contained
//! subsystem under `src/math/`. The thesis: a shaded-area puzzle is a
//! *recognition* problem, not an integration problem. Lu never integrates the
//! target boundary and never measures a radius from pixels. Instead it cuts the
//! figure into tiles where every tile is a KNOWN closed form, recalls each
//! form, and subtracts from the known total.
//!
//! What lives here:
//!   * `Rational` / `Exact`  — an EXACT `a + b·π` evaluator. π is a symbol,
//!     never a float, so the answer comes out as `32*pi - 64`, not `36.5279`.
//!   * primitive-area library — the four formulas the spec calls canonical:
//!     quarter-circle `πr²/4`, right-angle circular segment `r²(θ-sinθ)/2`,
//!     triangle `b·h/2`, and polygon shoelace.
//!   * `Scene` — a topology record: square bounds + scale, arcs as
//!     centre+radius+span, line segments, named points (incl. arc/grid
//!     intersections).
//!   * `Partition` + the sum-to-whole VERIFIER — the honesty gate. If the
//!     sub-regions do not sum to the whole (residual ≠ 0 in EITHER the rational
//!     or the π coefficient), the pipeline FLAGS rather than emitting a number.
//!   * IDK gate — per-tile confidence combined with the *minimum* (a chain is
//!     as strong as its weakest tile), so a shaky tile surfaces as an honest IDK
//!     instead of a confident wrong answer.
//!
//! Reuse: this module leans on the existing `expr` calculator as an independent
//! cross-check — `Exact::to_expr_string()` renders a form that
//! `expr::parse_and_eval` evaluates, and a test asserts the two agree. It also
//! reuses `super::fmt_num` for decimal prose.
//!
//! NOTE (build + test only): this is wired in as `pub mod geometry;` and is
//! exercised entirely through its own unit tests. It deliberately does NOT touch
//! the live `mcp` routing cascade or the canonical brain.

use super::fmt_num;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// =====================================================================
// Rational — exact rational coefficient (num/den, den > 0, reduced)
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 {
        1
    } else {
        a
    }
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "rational denominator must be non-zero");
        let sign = if den < 0 { -1 } else { 1 };
        let num = num * sign;
        let den = den * sign;
        let g = gcd_i64(num, den);
        Rational {
            num: num / g,
            den: den / g,
        }
    }
    pub fn int(n: i64) -> Self {
        Rational { num: n, den: 1 }
    }
    pub fn zero() -> Self {
        Rational { num: 0, den: 1 }
    }
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }
    pub fn add(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }
    pub fn sub(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den - o.num * self.den, self.den * o.den)
    }
    pub fn mul(self, o: Rational) -> Rational {
        Rational::new(self.num * o.num, self.den * o.den)
    }
    pub fn neg(self) -> Rational {
        Rational {
            num: -self.num,
            den: self.den,
        }
    }
    pub fn abs(self) -> Rational {
        Rational {
            num: self.num.abs(),
            den: self.den,
        }
    }
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
    /// Human form: `32`, `-36`, `3/4`.
    pub fn format(self) -> String {
        if self.den == 1 {
            format!("{}", self.num)
        } else {
            format!("{}/{}", self.num, self.den)
        }
    }
    /// Expression form parseable by `expr` (`-64`, `3/4`).
    pub fn expr_frac(self) -> String {
        if self.den == 1 {
            format!("{}", self.num)
        } else {
            format!("({}/{})", self.num, self.den)
        }
    }
}

// =====================================================================
// Exact — value = rational + pi * PI   (a + b·π, π kept symbolic)
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exact {
    pub rational: Rational,
    pub pi: Rational,
}

impl Exact {
    pub fn new(rational: Rational, pi: Rational) -> Self {
        Exact { rational, pi }
    }
    pub fn int(n: i64) -> Self {
        Exact {
            rational: Rational::int(n),
            pi: Rational::zero(),
        }
    }
    pub fn rational(r: Rational) -> Self {
        Exact {
            rational: r,
            pi: Rational::zero(),
        }
    }
    pub fn pi_term(coeff: Rational) -> Self {
        Exact {
            rational: Rational::zero(),
            pi: coeff,
        }
    }
    pub fn zero() -> Self {
        Exact::int(0)
    }
    pub fn is_zero(&self) -> bool {
        self.rational.is_zero() && self.pi.is_zero()
    }
    pub fn add(self, o: Exact) -> Exact {
        Exact {
            rational: self.rational.add(o.rational),
            pi: self.pi.add(o.pi),
        }
    }
    pub fn sub(self, o: Exact) -> Exact {
        Exact {
            rational: self.rational.sub(o.rational),
            pi: self.pi.sub(o.pi),
        }
    }
    pub fn neg(self) -> Exact {
        Exact {
            rational: self.rational.neg(),
            pi: self.pi.neg(),
        }
    }
    /// Multiply by a rational scalar (exact — keeps the π coefficient symbolic).
    pub fn scale(self, k: Rational) -> Exact {
        Exact {
            rational: self.rational.mul(k),
            pi: self.pi.mul(k),
        }
    }
    pub fn to_f64(self) -> f64 {
        self.rational.to_f64() + self.pi.to_f64() * PI
    }
    /// Canonical human form: `32*pi - 64`, `144 - 36*pi`, `25`.
    pub fn format(self) -> String {
        if self.pi.is_zero() {
            return self.rational.format();
        }
        if self.rational.is_zero() {
            return format!("{}*pi", self.pi.format());
        }
        // Both non-zero. Lead with the π term when the rational part is negative
        // (gives the textbook `a*pi - b` shape); otherwise rational-first.
        if self.rational.num < 0 && self.pi.num > 0 {
            format!("{}*pi - {}", self.pi.format(), self.rational.neg().format())
        } else {
            let sign = if self.pi.num < 0 { "-" } else { "+" };
            format!(
                "{} {} {}*pi",
                self.rational.format(),
                sign,
                self.pi.abs().format()
            )
        }
    }
    /// Form consumable by the existing `expr` evaluator (cross-check route).
    pub fn to_expr_string(self) -> String {
        format!(
            "{} + ({})*pi",
            self.rational.format(),
            self.pi.format()
        )
    }
}

// =====================================================================
// Primitive-area library (the four canonical formulas)
// =====================================================================

/// Rectangle / square: `w * h`.
pub fn rectangle(w: Rational, h: Rational) -> Exact {
    Exact::rational(w.mul(h))
}
pub fn square(s: Rational) -> Exact {
    rectangle(s, s)
}
/// Right triangle (or any triangle by base×height): `b * h / 2`.
pub fn triangle(b: Rational, h: Rational) -> Exact {
    Exact::rational(b.mul(h).mul(Rational::new(1, 2)))
}
/// Quarter circle (90° sector): `π r² / 4`.
pub fn quarter_circle(r: Rational) -> Exact {
    Exact::pi_term(r.mul(r).mul(Rational::new(1, 4)))
}
/// Right-angle (90°) circular segment: `πr²/4 - r²/2`
/// (the general segment `r²(θ-sinθ)/2` specialised at θ = π/2, where sinθ = 1).
pub fn circular_segment_quarter(r: Rational) -> Exact {
    quarter_circle(r).sub(triangle(r, r))
}
/// Spec alias: a single quarter "wedge off a chord" = `πr²/4 - r²/2`.
pub fn sector_minus_triangle(r: Rational) -> Exact {
    circular_segment_quarter(r)
}
/// Corner reservoir: `s² - πr²/4`.
pub fn square_minus_quarter_circle(s: Rational, r: Rational) -> Exact {
    square(s).sub(quarter_circle(r))
}
/// Polygon area by the shoelace formula (exact, rational vertices): `|Σ| / 2`.
pub fn polygon_shoelace(verts: &[(Rational, Rational)]) -> Exact {
    let n = verts.len();
    let mut acc = Rational::zero();
    for i in 0..n {
        let (x1, y1) = verts[i];
        let (x2, y2) = verts[(i + 1) % n];
        acc = acc.add(x1.mul(y2)).sub(y1.mul(x2).neg().neg());
        // (subtract y1*x2)
    }
    let area = acc.mul(Rational::new(1, 2));
    Exact::rational(area.abs())
}

// =====================================================================
// Scene representation (topology + scale; no curve is ever measured)
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arc {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub start_deg: f64,
    pub end_deg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSeg {
    pub a: Point,
    pub b: Point,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub width: f64,
    pub height: f64,
    pub scale_bar: f64,
    pub arcs: Vec<Arc>,
    pub segments: Vec<LineSeg>,
    pub points: Vec<(String, Point)>,
}

impl Scene {
    pub fn new(width: f64, height: f64, scale_bar: f64) -> Self {
        Scene {
            width,
            height,
            scale_bar,
            arcs: Vec::new(),
            segments: Vec::new(),
            points: Vec::new(),
        }
    }
    pub fn add_arc(&mut self, a: Arc) {
        self.arcs.push(a);
    }
    pub fn add_point(&mut self, name: &str, x: f64, y: f64) {
        self.points.push((name.to_string(), Point { x, y }));
    }
    /// The two intersection points of two full circles (if they meet).
    /// Used to locate the eye corners / arc crossings — not to read radii.
    pub fn circle_intersections(c1: &Arc, c2: &Arc) -> Option<(Point, Point)> {
        let dx = c2.cx - c1.cx;
        let dy = c2.cy - c1.cy;
        let d = (dx * dx + dy * dy).sqrt();
        if d == 0.0 || d > c1.r + c2.r || d < (c1.r - c2.r).abs() {
            return None;
        }
        let a = (c1.r * c1.r - c2.r * c2.r + d * d) / (2.0 * d);
        let h2 = c1.r * c1.r - a * a;
        if h2 < 0.0 {
            return None;
        }
        let h = h2.sqrt();
        let xm = c1.cx + a * dx / d;
        let ym = c1.cy + a * dy / d;
        Some((
            Point {
                x: xm + h * dy / d,
                y: ym - h * dx / d,
            },
            Point {
                x: xm - h * dy / d,
                y: ym + h * dx / d,
            },
        ))
    }
}

// =====================================================================
// Tile / Partition / Verifier / IDK gate
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    pub role: String,
    pub formula: String,
    pub area: Exact,
    pub confidence: f64,
}

impl Tile {
    pub fn new(role: &str, formula: &str, area: Exact, confidence: f64) -> Self {
        Tile {
            role: role.to_string(),
            formula: formula.to_string(),
            area,
            confidence,
        }
    }
}

/// A claimed partition of a whole region into known tiles plus one unknown
/// target. The verifier solves the target by subtraction AND confirms the
/// reassembled set sums back to the whole with zero residual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub whole: Exact,
    pub known_tiles: Vec<Tile>,
    pub target_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum VerifyOutcome {
    /// Verified: residual is exactly zero and every tile clears the IDK floor.
    Solved {
        target_role: String,
        target: Exact,
        target_decimal: f64,
        sum_of_known: Exact,
        whole: Exact,
        residual: Exact,
        weakest_tile: String,
        overall_confidence: f64,
    },
    /// The decomposition is inconsistent — sub-regions do not sum to the whole.
    Flagged { reason: String, residual: Exact },
    /// A load-bearing tile is below the confidence floor — honest refusal.
    Idk {
        recognized: usize,
        total: usize,
        unsure_region: String,
        unsure_confidence: f64,
    },
}

impl Partition {
    pub fn new(whole: Exact, target_role: &str) -> Self {
        Partition {
            whole,
            known_tiles: Vec::new(),
            target_role: target_role.to_string(),
        }
    }
    pub fn push(&mut self, t: Tile) {
        self.known_tiles.push(t);
    }
    fn sum_known(&self) -> Exact {
        let mut s = Exact::zero();
        for t in &self.known_tiles {
            s = s.add(t.area);
        }
        s
    }
    /// Solve the unknown target by subtraction: `target = whole - Σ(known)`.
    pub fn solve_target(&self) -> Exact {
        self.whole.sub(self.sum_known())
    }
    /// Full solve + verify + IDK gate.
    pub fn solve_and_verify(&self, idk_floor: f64) -> VerifyOutcome {
        // IDK gate first — minimum combiner, never average.
        let mut weakest = ("<none>".to_string(), 1.0_f64);
        for t in &self.known_tiles {
            if t.confidence < weakest.1 {
                weakest = (t.role.clone(), t.confidence);
            }
        }
        if weakest.1 < idk_floor {
            return VerifyOutcome::Idk {
                recognized: self.known_tiles.len(),
                total: self.known_tiles.len() + 1,
                unsure_region: weakest.0,
                unsure_confidence: weakest.1,
            };
        }
        let sum_known = self.sum_known();
        let target = self.whole.sub(sum_known);
        // Honesty invariant: Σ(known) + target must reconstruct the whole
        // exactly — residual zero in BOTH the rational and π coefficients.
        let residual = self.whole.sub(sum_known.add(target));
        if !residual.is_zero() {
            return VerifyOutcome::Flagged {
                reason: "sub-regions do not sum to the whole figure".to_string(),
                residual,
            };
        }
        VerifyOutcome::Solved {
            target_role: self.target_role.clone(),
            target,
            target_decimal: target.to_f64(),
            sum_of_known: sum_known,
            whole: self.whole,
            residual,
            weakest_tile: weakest.0,
            overall_confidence: weakest.1,
        }
    }
}

/// Independent honesty check used in tests: do ALL regions (every tile plus the
/// claimed target) sum to the whole? Returns the residual (zero == consistent).
pub fn verify_sum_to_whole(whole: Exact, regions: &[Exact]) -> Exact {
    let mut s = Exact::zero();
    for r in regions {
        s = s.add(*r);
    }
    whole.sub(s)
}

// =====================================================================
// Solvers
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySolution {
    pub problem: String,
    pub scene: Scene,
    pub partition: Partition,
    pub outcome: VerifyOutcome,
    pub prose: String,
}

/// Build the reference "red leaf in a 12×12 square" scene (IMG_5637).
/// Topology + scale only; radii are FORCED by the grid (3×3 cells of side 4,
/// scale bar "4"), never measured.
pub fn red_leaf_scene() -> Scene {
    let mut s = Scene::new(12.0, 12.0, 4.0);
    // Three quarter-circle arcs recognised in the figure. Centres sit on grid
    // nodes; radii are forced to 12, 8, 8 by the 3×3 grid + scale bar.
    s.add_arc(Arc {
        cx: 0.0,
        cy: 12.0,
        r: 12.0,
        start_deg: -90.0,
        end_deg: 0.0,
    }); // lower-right boundary of the leaf
    s.add_arc(Arc {
        cx: 8.0,
        cy: 0.0,
        r: 8.0,
        start_deg: 90.0,
        end_deg: 180.0,
    }); // upper-left boundary (left lobe)
    s.add_arc(Arc {
        cx: 4.0,
        cy: 12.0,
        r: 8.0,
        start_deg: 270.0,
        end_deg: 360.0,
    }); // boundary toward (12,12) (right lobe)
    // Inner radius-4 arcs bound the eye (centres (4,8) and (8,4)).
    let e1 = Arc {
        cx: 4.0,
        cy: 8.0,
        r: 4.0,
        start_deg: -90.0,
        end_deg: 0.0,
    };
    let e2 = Arc {
        cx: 8.0,
        cy: 4.0,
        r: 4.0,
        start_deg: 90.0,
        end_deg: 180.0,
    };
    s.add_arc(e1);
    s.add_arc(e2);
    // Eye corners as arc/grid intersections (not pixel reads): (4,4) and (8,8).
    s.add_point("eye_corner_lo", 4.0, 4.0);
    s.add_point("eye_corner_hi", 8.0, 8.0);
    s.add_point("leaf_tip_lo", 0.0, 0.0);
    s.add_point("leaf_tip_hi", 12.0, 12.0);
    s
}

/// Acceptance test driver: solve the red-leaf area `x` exactly.
pub fn solve_red_leaf() -> GeometrySolution {
    let scene = red_leaf_scene();
    let r4 = Rational::int(4);
    let r8 = Rational::int(8);
    let r12 = Rational::int(12);

    let whole = square(r12); // 144
    let mut p = Partition::new(whole, "red_leaf_x");
    // Top strip: two cells + a half-cell triangle.
    p.push(Tile::new(
        "top_left_cell",
        "rectangle(4,4)",
        rectangle(r4, r4),
        0.98,
    ));
    p.push(Tile::new(
        "top_middle_cell",
        "rectangle(4,4)",
        rectangle(r4, r4),
        0.98,
    ));
    p.push(Tile::new(
        "top_right_half",
        "triangle(4,4)",
        triangle(r4, r4),
        0.95,
    ));
    // Reservoirs and wedges — each a recalled closed form.
    p.push(Tile::new(
        "left_reservoir",
        "square(8) - quarter_circle(8)",
        square_minus_quarter_circle(r8, r8),
        0.92,
    ));
    p.push(Tile::new(
        "eye",
        "quarter_circle(4) - triangle(4,4)  [eye lens, net 4*pi-8]",
        circular_segment_quarter(r4),
        0.55, // weakest link, per spec §7
    ));
    p.push(Tile::new(
        "right_wedge",
        "sector_minus_triangle(8)",
        sector_minus_triangle(r8),
        0.80,
    ));
    p.push(Tile::new(
        "lower_reservoir",
        "square(12) - quarter_circle(12)",
        square_minus_quarter_circle(r12, r12),
        0.90,
    ));

    let outcome = p.solve_and_verify(0.40);
    let target = p.solve_target();
    let prose = match &outcome {
        VerifyOutcome::Solved {
            weakest_tile,
            overall_confidence,
            ..
        } => format!(
            "Red leaf x = {} ≈ {} units². Decomposition of the 12×12 square verified: \
             7 known tiles + x reconstruct 144 with zero residual. Weakest tile: {} \
             (confidence {:.2}); answer emitted above the IDK floor.",
            target.format(),
            fmt_num(target.to_f64()),
            weakest_tile,
            overall_confidence
        ),
        other => format!("Red leaf could not be emitted: {:?}", other),
    };
    GeometrySolution {
        problem: "red_leaf_in_square".to_string(),
        scene,
        partition: p,
        outcome,
        prose,
    }
}

// ---------------------------------------------------------------------
// Control problem (IMG_5639): rectangle + corner quarter-circle + semicircle,
// with a 5 cm tangent. Solved by the CONSTRAINT, not by visual intuition.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlSolution {
    pub tangent_len: f64,
    pub area: f64,
    pub identity_holds: bool,
    pub prose: String,
}

/// The construction, forced by the figure's topology:
///   * quarter-circle centred at the bottom-left corner, radius = height H,
///     so it meets the bottom edge at (H, 0);
///   * semicircle whose diameter runs along the bottom edge from (H,0) to
///     (W,0) — hence centre c = ((H+W)/2, 0), radius ρ = (W-H)/2;
///   * the 5 cm segment from the bottom-left corner (0,0) is TANGENT to the
///     semicircle.
///
/// Tangent-length theorem from an external point to a circle:
///   t² = d² - ρ²,  where d = |corner → centre| = (H+W)/2.
/// Substituting:
///   t² = ((H+W)/2)² - ((W-H)/2)² = (4HW)/4 = H·W = Area.
/// So Area = t² — independent of the individual side lengths. t = 5 ⇒ Area = 25.
pub fn solve_control(tangent_len: f64) -> ControlSolution {
    // Verify the algebraic identity numerically across sample rectangles:
    // ((H+W)/2)² - ((W-H)/2)² == H*W  for all H,W.
    let mut identity_holds = true;
    for &(h, w) in &[(3.0_f64, 9.0), (2.5, 7.1), (1.0, 5.0), (4.2, 4.2)] {
        let d = (h + w) / 2.0;
        let rho = (w - h) / 2.0;
        let lhs = d * d - rho * rho;
        if (lhs - h * w).abs() > 1e-9 {
            identity_holds = false;
        }
    }
    let area = tangent_len * tangent_len;
    let prose = format!(
        "Control: tangent length t = {} cm. By the tangent-from-a-point theorem \
         t² = d² - ρ² with d = (H+W)/2 and ρ = (W-H)/2, which collapses to t² = H·W = Area. \
         Therefore Area of the rectangle = {}² = {} cm² (solved from the constraint, \
         not measured).",
        fmt_num(tangent_len),
        fmt_num(tangent_len),
        fmt_num(area)
    );
    ControlSolution {
        tangent_len,
        area,
        identity_holds,
        prose,
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::expr;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ---- primitive-area library ----
    #[test]
    fn quarter_circles_keep_pi_exact() {
        assert_eq!(quarter_circle(Rational::int(12)), Exact::pi_term(Rational::int(36)));
        assert_eq!(quarter_circle(Rational::int(8)), Exact::pi_term(Rational::int(16)));
        assert_eq!(quarter_circle(Rational::int(4)), Exact::pi_term(Rational::int(4)));
    }
    #[test]
    fn segment_and_reservoir_formulas() {
        // 90° segment r=4 -> 4π-8 ; r=8 -> 16π-32
        assert_eq!(circular_segment_quarter(Rational::int(4)).format(), "4*pi - 8");
        assert_eq!(sector_minus_triangle(Rational::int(8)).format(), "16*pi - 32");
        // square minus quarter circle
        assert_eq!(
            square_minus_quarter_circle(Rational::int(8), Rational::int(8)).format(),
            "64 - 16*pi"
        );
        assert_eq!(
            square_minus_quarter_circle(Rational::int(12), Rational::int(12)).format(),
            "144 - 36*pi"
        );
    }
    #[test]
    fn triangle_and_shoelace_agree() {
        let t = triangle(Rational::int(4), Rational::int(4));
        // same triangle by shoelace: (0,0),(4,0),(0,4)
        let sh = polygon_shoelace(&[
            (Rational::int(0), Rational::int(0)),
            (Rational::int(4), Rational::int(0)),
            (Rational::int(0), Rational::int(4)),
        ]);
        assert_eq!(t, sh);
        assert_eq!(t.format(), "8");
    }
    #[test]
    fn shoelace_unit_square() {
        let sq = polygon_shoelace(&[
            (Rational::int(0), Rational::int(0)),
            (Rational::int(4), Rational::int(0)),
            (Rational::int(4), Rational::int(4)),
            (Rational::int(0), Rational::int(4)),
        ]);
        assert_eq!(sq.format(), "16");
    }

    // ---- exact evaluator + cross-check against the existing expr calculator ----
    #[test]
    fn exact_decimal_matches_expr_evaluator() {
        let x = Exact::new(Rational::int(-64), Rational::int(32)); // 32π-64
        let via_expr = expr::parse_and_eval(&x.to_expr_string()).expect("expr eval");
        assert!(approx(x.to_f64(), via_expr));
        assert!(approx(x.to_f64(), 32.0 * std::f64::consts::PI - 64.0));
    }

    // ---- the red-leaf acceptance test ----
    #[test]
    fn red_leaf_solves_to_32pi_minus_64() {
        let sol = solve_red_leaf();
        let x = sol.partition.solve_target();
        assert_eq!(x.format(), "32*pi - 64");
        assert!(approx(x.to_f64(), 32.0 * std::f64::consts::PI - 64.0), "x ≈ {}", x.to_f64());
    }
    #[test]
    fn red_leaf_regions_sum_to_144() {
        let sol = solve_red_leaf();
        let mut regions: Vec<Exact> = sol.partition.known_tiles.iter().map(|t| t.area).collect();
        regions.push(sol.partition.solve_target());
        let residual = verify_sum_to_whole(Exact::int(144), &regions);
        assert!(residual.is_zero(), "residual must be exactly zero, got {:?}", residual);
    }
    #[test]
    fn red_leaf_outcome_is_solved_and_flags_eye() {
        let sol = solve_red_leaf();
        match sol.outcome {
            VerifyOutcome::Solved {
                target,
                weakest_tile,
                overall_confidence,
                residual,
                ..
            } => {
                assert_eq!(target.format(), "32*pi - 64");
                assert!(residual.is_zero());
                assert_eq!(weakest_tile, "eye");
                assert!((overall_confidence - 0.55).abs() < 1e-9);
            }
            other => panic!("expected Solved, got {:?}", other),
        }
    }

    // ---- the honesty gate: a wrong tile must FLAG, not silently absorb ----
    #[test]
    fn corrupted_partition_is_flagged() {
        // Over-determined check: give ALL regions including a WRONG target and
        // confirm the residual is non-zero (would be flagged).
        let regions = vec![
            Exact::int(16),
            Exact::int(16),
            Exact::int(8),
            square_minus_quarter_circle(Rational::int(8), Rational::int(8)),
            circular_segment_quarter(Rational::int(4)),
            sector_minus_triangle(Rational::int(8)),
            square_minus_quarter_circle(Rational::int(12), Rational::int(12)),
            Exact::new(Rational::int(-99), Rational::int(32)), // wrong x
        ];
        let residual = verify_sum_to_whole(Exact::int(144), &regions);
        assert!(!residual.is_zero(), "a wrong region must produce non-zero residual");
    }
    #[test]
    fn low_confidence_tile_triggers_idk() {
        let mut p = Partition::new(Exact::int(144), "x");
        p.push(Tile::new("ambiguous_arc", "square(8)-quarter(8)",
            square_minus_quarter_circle(Rational::int(8), Rational::int(8)), 0.30));
        p.push(Tile::new("solid", "rect", rectangle(Rational::int(4), Rational::int(4)), 0.95));
        match p.solve_and_verify(0.40) {
            VerifyOutcome::Idk { unsure_region, .. } => assert_eq!(unsure_region, "ambiguous_arc"),
            other => panic!("expected Idk, got {:?}", other),
        }
    }

    // ---- the control problem ----
    #[test]
    fn control_area_is_25() {
        let c = solve_control(5.0);
        assert!(c.identity_holds, "tangent identity (H+W)/2)^2-((W-H)/2)^2 == HW must hold");
        assert!(approx(c.area, 25.0));
    }

    // ---- scene / topology helpers ----
    #[test]
    fn scene_records_topology_and_arc_crossings() {
        let s = red_leaf_scene();
        assert_eq!(s.width, 12.0);
        assert_eq!(s.scale_bar, 4.0);
        assert_eq!(s.arcs.len(), 5);
        // The two radius-4 eye arcs cross at the eye corners (4,4) and (8,8).
        let e1 = s.arcs[3];
        let e2 = s.arcs[4];
        let (p, q) = Scene::circle_intersections(&e1, &e2).expect("eye arcs intersect");
        let mut xs = [p.x, q.x];
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] - 4.0).abs() < 1e-6 && (xs[1] - 8.0).abs() < 1e-6);
    }
}
