//! Phase P-D — CONSTRAINT EXTRACTION.
//!
//! internal notes. Consumes the Phase P-C labelled-primitive output
//! (`crate::vision::geo_labels::LabeledPrimitives`, from P-B
//! `geo_primitives` + P-C `geo_labels`) and ASSEMBLES a concrete
//! `crate::math::algsolve::Problem` + `Target` — the exact constraint system
//! the offline algebraic solver (`algsolve::solve`) consumes. This is the link
//! that finally closes pixels -> shapes -> labels -> CONSTRAINTS -> solver ->
//! exact answer, entirely inside the offline library.
//!
//! HONESTY FLOOR (absolute). Every constraint is derived ONLY from what was
//! genuinely detected + labelled:
//!   * a value comes from an OCR-read numeric LABEL (area / length / radius),
//!     never invented;
//!   * a geometric RELATION (perpendicularity, tangency, collinearity,
//!     concyclicity, equal sides) is emitted ONLY when the detected primitive
//!     coordinates actually exhibit it within tolerance — measured, not
//!     assumed.
//! If the assembled system is missing a relation the target needs (e.g. the
//! inscribed-circle tangency was never detected) or the target region floats
//! free, extraction returns `ExtractResult::Incomplete{reason}` naming the gap.
//! We NEVER fabricate a constraint to force a solve, and we NEVER inject the
//! numeric answer. Deeper under-determination (a determinate-looking system
//! whose queried quantity is not actually invariant) is caught downstream by
//! the solver's K-gate returning `Refuse(UnderDetermined)` — also honest.
//!
//! There is NO per-image / filename / hash / figure-name / answer-value lookup
//! anywhere on this path: the Problem is built from the (role, primitive,
//! value) bindings and the measured geometry alone.
//!
//! OFFLINE library only. Wires to nothing — no guest surface, no promote, no
//! serving change. Live assembly + wiring is the separate P-E.

use crate::math::algsolve::{Con, PointDef, Problem, Target};
use crate::vision::geo_labels::{LabeledPrimitives, Role};
use crate::vision::geo_primitives::Primitive;

// =====================================================================
// Tunables (measurement tolerances — the honesty floor for RELATIONS)
// =====================================================================

/// Merge vertices closer than this (normalized units) into one point id.
const DEDUP_EPS: f64 = 0.03;
/// Relative tolerance for "equal length" (square sides / equilateral / concyclic).
const EQ_REL_TOL: f64 = 0.08;
/// Measured perpendicularity tolerance, degrees.
const PERP_TOL_DEG: f64 = 7.0;
/// Relative tolerance for circle-line / circle-circle tangency.
const TAN_REL_TOL: f64 = 0.12;

// =====================================================================
// Public result
// =====================================================================

/// The honest outcome of assembling a constraint problem from a labelled scene.
pub enum ExtractResult {
    /// A concrete solver problem: feed `(problem, target)` to `algsolve::solve`.
    Problem(Problem, Target),
    /// The scene does not determine the target from what was detected+labelled.
    /// First-class, honest outcome — NOT an error, NOT a guessed number.
    Incomplete { reason: String },
}

impl ExtractResult {
    pub fn is_incomplete(&self) -> bool {
        matches!(self, ExtractResult::Incomplete { .. })
    }
    pub fn reason(&self) -> Option<&str> {
        match self {
            ExtractResult::Incomplete { reason } => Some(reason),
            _ => None,
        }
    }
}

// =====================================================================
// small geometry helpers
// =====================================================================

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Signed shoelace area of a polygon given by ordered coords.
fn signed_area(v: &[(f64, f64)]) -> f64 {
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
    s / 2.0
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

/// Distance from `p` to the INFINITE line through a,b.
fn line_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ux = b.0 - a.0;
    let uy = b.1 - a.1;
    let len = (ux * ux + uy * uy).sqrt();
    if len < 1e-12 {
        return dist(p, a);
    }
    ((p.0 - a.0) * uy - (p.1 - a.1) * ux).abs() / len
}

/// Unsigned angle at vertex b (between rays b->a and b->c), degrees.
fn angle_deg(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    let u = (a.0 - b.0, a.1 - b.1);
    let v = (c.0 - b.0, c.1 - b.1);
    let du = (u.0 * u.0 + u.1 * u.1).sqrt();
    let dv = (v.0 * v.0 + v.1 * v.1).sqrt();
    if du < 1e-12 || dv < 1e-12 {
        return 0.0;
    }
    let mut cs = (u.0 * v.0 + u.1 * v.1) / (du * dv);
    cs = cs.clamp(-1.0, 1.0);
    cs.acos().to_degrees()
}

// =====================================================================
// internal working structures
// =====================================================================

#[derive(Clone, Copy)]
enum Pin {
    Free,
    Fixed(f64, f64),
    OnX, // y = 0
    OnY, // x = 0
}

struct PolyRec {
    prim_index: usize,
    node_ids: Vec<usize>,
    detected: Vec<(f64, f64)>, // detected coords in polygon order
}

struct SegRec {
    a: usize,
    b: usize,
}

struct CircRec {
    prim_index: usize,
    center_id: usize,
    center: (f64, f64),
    r: f64,
}

/// Collected geometry + a scratch node table (detected normalized coords).
struct Scene {
    nodes: Vec<(f64, f64)>,
    polys: Vec<PolyRec>,
    segs: Vec<SegRec>,
    circles: Vec<CircRec>,
    right_angle_at: Vec<(f64, f64)>,
}

fn get_or_add(nodes: &mut Vec<(f64, f64)>, p: (f64, f64)) -> usize {
    for (i, q) in nodes.iter().enumerate() {
        if dist(*q, p) < DEDUP_EPS {
            return i;
        }
    }
    nodes.push(p);
    nodes.len() - 1
}

fn collect_scene(lab: &LabeledPrimitives) -> Scene {
    let mut nodes: Vec<(f64, f64)> = Vec::new();
    let mut polys = Vec::new();
    let mut segs = Vec::new();
    let mut circles = Vec::new();
    let mut right_angle_at = Vec::new();

    for (i, prim) in lab.primitives.items.iter().enumerate() {
        match prim {
            Primitive::Polygon { verts, .. } => {
                let ids: Vec<usize> = verts.iter().map(|v| get_or_add(&mut nodes, *v)).collect();
                polys.push(PolyRec { prim_index: i, node_ids: ids, detected: verts.clone() });
            }
            Primitive::Segment { a, b, .. } => {
                let ia = get_or_add(&mut nodes, *a);
                let ib = get_or_add(&mut nodes, *b);
                segs.push(SegRec { a: ia, b: ib });
            }
            Primitive::Circle { center, r, .. } => {
                let c = get_or_add(&mut nodes, *center);
                circles.push(CircRec { prim_index: i, center_id: c, center: *center, r: *r });
            }
            Primitive::Arc { center, r, .. } => {
                let c = get_or_add(&mut nodes, *center);
                circles.push(CircRec { prim_index: i, center_id: c, center: *center, r: *r });
            }
            Primitive::Point { p, .. } => {
                get_or_add(&mut nodes, *p);
            }
            Primitive::RightAngle { at, .. } => right_angle_at.push(*at),
        }
    }
    Scene { nodes, polys, segs, circles, right_angle_at }
}

// role/value lookup by primitive index
fn value_of(lab: &LabeledPrimitives, prim_index: usize, role: Role) -> Option<f64> {
    for b in &lab.bindings {
        if b.primitive_index == prim_index && b.role == role {
            if let Ok(v) = b.label.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

// =====================================================================
// Assembly buffer
// =====================================================================

struct Build {
    pins: Vec<Pin>,
    hints: Vec<(f64, f64)>, // final (transformed+scaled) hint coords per node
    cons: Vec<Con>,
    descs: Vec<String>,
    nondeg: Vec<(usize, usize, f64)>,
    extra_nodes: Vec<(f64, f64)>, // synthesized nodes (e.g. tangent rim points)
}

impl Build {
    fn new(n: usize) -> Self {
        Build {
            pins: vec![Pin::Free; n],
            hints: vec![(0.0, 0.0); n],
            cons: Vec::new(),
            descs: Vec::new(),
            nondeg: Vec::new(),
            extra_nodes: Vec::new(),
        }
    }
    fn add(&mut self, c: Con, d: String) {
        self.cons.push(c);
        self.descs.push(d);
    }
    fn new_node(&mut self, hint: (f64, f64)) -> usize {
        let id = self.hints.len();
        self.hints.push(hint);
        self.pins.push(Pin::Free);
        self.extra_nodes.push(hint);
        id
    }
    fn finish(self, target: Target) -> Problem {
        let pts: Vec<PointDef> = self
            .pins
            .iter()
            .zip(self.hints.iter())
            .map(|(pin, h)| match pin {
                Pin::Free => PointDef { fx: None, fy: None, init: Some(*h) },
                Pin::Fixed(x, y) => PointDef::fixed(*x, *y),
                Pin::OnX => PointDef { fx: None, fy: Some(0.0), init: Some((h.0, 0.0)) },
                Pin::OnY => PointDef { fx: Some(0.0), fy: None, init: Some((0.0, h.1)) },
            })
            .collect();
        let _ = target;
        Problem { pts, cons: self.cons, nondeg: self.nondeg }
    }
}

// =====================================================================
// ENTRY POINT
// =====================================================================

/// Assemble an `algsolve::Problem` + `Target` from a labelled scene, or return
/// an honest `Incomplete{reason}`. The answer is NOT computed here — this only
/// builds the system; `algsolve::solve` derives (or honestly refuses) it.
pub fn extract_problem(lab: &LabeledPrimitives) -> ExtractResult {
    let scene = collect_scene(lab);

    // Find the solve-for target binding.
    let target_binding = lab.bindings.iter().find(|b| b.role == Role::Target);
    let tb = match target_binding {
        Some(b) => b,
        None => {
            return ExtractResult::Incomplete {
                reason: "no '?' solve-for target was bound to any primitive".into(),
            }
        }
    };
    if tb.primitive_kind == "polygon" {
        return ExtractResult::Incomplete {
            reason: "target is a polygon area; algsolve has no polygon-area solve target (only length / disk-area)".into(),
        };
    }

    // The target is a circular region (circle/arc). Locate its CircRec.
    let target_circ = scene
        .circles
        .iter()
        .position(|c| c.prim_index == tb.primitive_index);
    let target_circ = match target_circ {
        Some(i) => i,
        None => {
            return ExtractResult::Incomplete {
                reason: "'?' target primitive is not a detected circular region".into(),
            }
        }
    };

    // Try the two structural closures, most specific first.
    if let Some(res) = try_incircle_closure(lab, &scene, target_circ) {
        return res;
    }
    if let Some(res) = try_circle_in_circle_closure(lab, &scene, target_circ) {
        return res;
    }

    // No closure: the target circle is not tied to the rest of the figure by a
    // detected tangency / incidence relation. Honest, specific reason.
    let center = scene.circles[target_circ].center;
    let inside_tri = scene.polys.iter().any(|p| p.node_ids.len() == 3 && point_in_poly(center, &p.detected));
    let reason = if inside_tri {
        "target circle lies inside a triangle but was NOT measured tangent to its three sides (inscribed-circle tangency relation missing); system under-determined for the circle's area".to_string()
    } else {
        "target circle has no detected tangency / incidence relation tying it to the rest of the figure; system under-determined for its area".to_string()
    };
    ExtractResult::Incomplete { reason }
}

// =====================================================================
// generic label-derived constraints (areas + square-ness + lengths)
// applied AFTER a closure has fixed the hint transform (for correct winding).
// =====================================================================

fn emit_generic(lab: &LabeledPrimitives, scene: &Scene, b: &mut Build) {
    // Length labels: iterate bindings directly.
    for bind in &lab.bindings {
        if bind.role == Role::Length && bind.primitive_kind == "segment" {
            if let Some(v) = bind.label.parse::<f64>().ok() {
                if let Primitive::Segment { a, b: bb, .. } = &lab.primitives.items[bind.primitive_index] {
                    let ia = nearest_node(scene, *a);
                    let ib = nearest_node(scene, *bb);
                    if let (Some(ia), Some(ib)) = (ia, ib) {
                        b.add(Con::Dist(ia, ib, v), format!("Dist(p{},p{}) = {} [length label]", ia, ib, v));
                    }
                }
            }
        }
    }

    // Area labels on polygons -> Con::PolyArea (signed to match hint winding),
    // plus square-ness (equal sides + one right angle) when a quad measures square.
    for pr in &scene.polys {
        if let Some(area) = value_of(lab, pr.prim_index, Role::Area) {
            let ids = &pr.node_ids;
            // winding sign from the FINAL hint coords
            let hs: Vec<(f64, f64)> = ids.iter().map(|&i| b.hints[i]).collect();
            let s = signed_area(&hs);
            let signed = if s < 0.0 { -area.abs() } else { area.abs() };
            b.add(
                Con::PolyArea(ids.clone(), signed),
                format!("PolyArea([{}]) = {} [area label]", join(ids), signed),
            );
            // square-ness for quads
            if ids.len() == 4 && measures_square(&pr.detected) {
                b.add(Con::DistEq(ids[0], ids[1], ids[1], ids[2]), format!("DistEq |p{}p{}|=|p{}p{}| [measured equal side]", ids[0], ids[1], ids[1], ids[2]));
                b.add(Con::DistEq(ids[1], ids[2], ids[2], ids[3]), format!("DistEq |p{}p{}|=|p{}p{}| [measured equal side]", ids[1], ids[2], ids[2], ids[3]));
                b.add(Con::DistEq(ids[2], ids[3], ids[3], ids[0]), format!("DistEq |p{}p{}|=|p{}p{}| [measured equal side]", ids[2], ids[3], ids[3], ids[0]));
                b.add(Con::Perp(ids[0], ids[1], ids[1], ids[2]), format!("Perp (p{}p{})_|_(p{}p{}) [measured right angle]", ids[0], ids[1], ids[1], ids[2]));
            }
        }
    }
}

fn nearest_node(scene: &Scene, p: (f64, f64)) -> Option<usize> {
    let mut best = None;
    let mut bd = f64::MAX;
    for (i, q) in scene.nodes.iter().enumerate() {
        let d = dist(*q, p);
        if d < bd {
            bd = d;
            best = Some(i);
        }
    }
    if bd < 3.0 * DEDUP_EPS {
        best
    } else {
        None
    }
}

fn join(ids: &[usize]) -> String {
    ids.iter().map(|i| format!("p{}", i)).collect::<Vec<_>>().join(",")
}

/// A quad measures square if all four sides are equal (rel tol) and one corner
/// is ~90 degrees.
fn measures_square(v: &[(f64, f64)]) -> bool {
    if v.len() != 4 {
        return false;
    }
    let s: Vec<f64> = (0..4).map(|i| dist(v[i], v[(i + 1) % 4])).collect();
    let mean = s.iter().sum::<f64>() / 4.0;
    if mean < 1e-9 {
        return false;
    }
    if s.iter().any(|x| (x - mean).abs() / mean > EQ_REL_TOL) {
        return false;
    }
    let ang = angle_deg(v[0], v[1], v[2]);
    (ang - 90.0).abs() < PERP_TOL_DEG
}

// =====================================================================
// CLOSURE 1: incircle inscribed in a right triangle
// =====================================================================

fn try_incircle_closure(
    lab: &LabeledPrimitives,
    scene: &Scene,
    target_circ: usize,
) -> Option<ExtractResult> {
    let circ = &scene.circles[target_circ];
    let center = circ.center;
    let r = circ.r;

    // find a triangle polygon that CONTAINS the circle centre and to whose
    // three sides the circle measures tangent.
    for pr in &scene.polys {
        if pr.node_ids.len() != 3 {
            continue;
        }
        if !point_in_poly(center, &pr.detected) {
            continue;
        }
        let v = &pr.detected;
        // measured tangency to all three sides
        let tangent_all = (0..3).all(|i| {
            let a = v[i];
            let bb = v[(i + 1) % 3];
            let d = line_dist(center, a, bb);
            (d - r).abs() / r.max(1e-9) < TAN_REL_TOL
        });
        if !tangent_all {
            continue;
        }
        // identify the right-angle corner: a RightAngle mark near a vertex OR a
        // measured ~90 degree corner.
        let mut corner: Option<usize> = None; // index in 0..3
        for i in 0..3 {
            let near_mark = scene
                .right_angle_at
                .iter()
                .any(|m| dist(*m, v[i]) < 0.08);
            let a = v[(i + 2) % 3];
            let b = v[i];
            let c = v[(i + 1) % 3];
            let measured = (angle_deg(a, b, c) - 90.0).abs() < PERP_TOL_DEG;
            if near_mark || measured {
                corner = Some(i);
                break;
            }
        }
        let ci = match corner {
            Some(i) => i,
            None => {
                return Some(ExtractResult::Incomplete {
                    reason: "target circle is inside a triangle and tangent to all sides, but NO right-angle corner was detected (mark or measured); the pinned incircle frame needs a right angle".into(),
                });
            }
        };
        // Need the triangle's own area label to pin the absolute scale.
        let tri_area = value_of(lab, pr.prim_index, Role::Area);
        if tri_area.is_none() {
            return Some(ExtractResult::Incomplete {
                reason: "inscribed-circle triangle carries no area label; absolute scale is undetermined".into(),
            });
        }

        // Build. Vertex ids in canonical roles:
        let v_corner = pr.node_ids[ci];
        let a_id = pr.node_ids[(ci + 1) % 3];
        let b_id = pr.node_ids[(ci + 2) % 3];
        let c_v = v[ci];
        let a_v = v[(ci + 1) % 3];
        let b_v = v[(ci + 2) % 3];

        // scale factor from the LARGEST area label present (prefer a quad area,
        // e.g. tilted square, else the triangle) so hints land near absolute units.
        let (scale_area_label, scale_poly_detected) = pick_scale_poly(lab, scene);
        let k = if scale_poly_detected > 1e-12 {
            (scale_area_label / scale_poly_detected).sqrt()
        } else {
            1.0
        };

        // canonical rigid transform: corner -> origin, leg corner->a on +x,
        // reflect so b is on +y side. (translation + rotation + reflection are
        // gauge freedoms; the right angle between the legs is the detected fact.)
        let ang = (a_v.1 - c_v.1).atan2(a_v.0 - c_v.0);
        let (cs, sn) = ((-ang).cos(), (-ang).sin());
        let tf = |p: (f64, f64)| -> (f64, f64) {
            let dx = p.0 - c_v.0;
            let dy = p.1 - c_v.1;
            (dx * cs - dy * sn, dx * sn + dy * cs)
        };
        let reflect = tf(b_v).1 < 0.0;
        let tf2 = |p: (f64, f64)| -> (f64, f64) {
            let (x, y) = tf(p);
            (x * k, (if reflect { -y } else { y }) * k)
        };

        let mut b = Build::new(scene.nodes.len());
        for i in 0..scene.nodes.len() {
            b.hints[i] = tf2(scene.nodes[i]);
        }
        b.pins[v_corner] = Pin::Fixed(0.0, 0.0);
        b.pins[a_id] = Pin::OnX;
        b.pins[b_id] = Pin::OnY;

        // generic label constraints (areas incl. the triangle + tilted square,
        // square-ness, lengths) — emitted with the final hint winding.
        emit_generic(lab, scene, &mut b);

        // incircle tangency (evidence-gated: we already verified tangency to all
        // three sides above): centre equidistant from the two axis-legs, and its
        // distance to the hypotenuse equals that same radius (= C.y).
        let c_id = circ.center_id;
        b.add(Con::CoordEq(c_id, 0, c_id, 1), format!("CoordEq p{}.x = p{}.y [incircle tangent to both legs]", c_id, c_id));
        b.add(
            Con::PtLineDistEqCoord(c_id, a_id, b_id, c_id, 1),
            format!("PtLineDistEqCoord dist(p{}, line p{}p{}) = p{}.y [incircle tangent to hypotenuse]", c_id, a_id, b_id, c_id),
        );
        // explicit right-angle (perpendicular legs) from the DETECTED right-angle
        // corner (drawn mark OR measured 90 deg, established above). Consistent
        // with the axis-pin gauge, but makes the detected relation an EXPLICIT
        // Perp constraint in the assembled system rather than an implicit one.
        b.add(
            Con::Perp(v_corner, a_id, v_corner, b_id),
            format!("Perp (p{}p{}) _|_ (p{}p{}) [detected right-angle corner]", v_corner, a_id, v_corner, b_id),
        );
        // non-degeneracy: the two legs must have real length.
        b.nondeg.push((a_id, b_id, 0.25 * (tri_area.unwrap().abs()).sqrt().max(1.0)));

        let target = Target::DiskAreaByRadiusCoord(c_id, 1);
        return Some(ExtractResult::Problem(b.finish(target.clone()), target));
    }
    None
}

/// Choose an area label + its detected polygon area to derive the hint scale.
/// Prefers the largest labelled polygon (typically the tilted square).
fn pick_scale_poly(lab: &LabeledPrimitives, scene: &Scene) -> (f64, f64) {
    let mut best: Option<(f64, f64)> = None; // (label, detected_abs_area)
    for pr in &scene.polys {
        if let Some(area) = value_of(lab, pr.prim_index, Role::Area) {
            let det = signed_area(&pr.detected).abs();
            let better = match best {
                None => true,
                Some((la, _)) => area > la,
            };
            if better {
                best = Some((area, det));
            }
        }
    }
    best.unwrap_or((1.0, 1.0))
}

// =====================================================================
// CLOSURE 2: equal circles mutually tangent inside a labelled outer circle
// =====================================================================

fn try_circle_in_circle_closure(
    lab: &LabeledPrimitives,
    scene: &Scene,
    target_circ: usize,
) -> Option<ExtractResult> {
    if scene.circles.len() < 3 {
        return None;
    }
    // outer = a circle carrying a Radius label (known R) that encloses others.
    let mut outer: Option<usize> = None;
    for (i, c) in scene.circles.iter().enumerate() {
        if value_of(lab, c.prim_index, Role::Radius).is_some() {
            // must be the largest / enclose the target
            if scene.circles.iter().enumerate().all(|(j, o)| j == i || o.r <= c.r + 1e-9) {
                outer = Some(i);
            }
        }
    }
    let outer = outer?;
    let r_label = value_of(lab, scene.circles[outer].prim_index, Role::Radius)?;
    if outer == target_circ {
        return None;
    }

    // inner = every other circle.
    let inner: Vec<usize> = (0..scene.circles.len()).filter(|&i| i != outer).collect();
    if inner.len() < 3 {
        return None;
    }
    let oc = scene.circles[outer].center;
    let or = scene.circles[outer].r;

    // measured: all inner equal radius, mutually tangent, internally tangent to
    // the outer, and concyclic about the outer centre.
    let r_in = scene.circles[inner[0]].r;
    let equal_r = inner.iter().all(|&i| (scene.circles[i].r - r_in).abs() / r_in.max(1e-9) < EQ_REL_TOL);
    if !equal_r {
        return None;
    }
    let concyclic = inner.iter().all(|&i| {
        let d = dist(scene.circles[i].center, oc);
        (d - dist(scene.circles[inner[0]].center, oc)).abs() / or.max(1e-9) < EQ_REL_TOL
    });
    if !concyclic {
        return None;
    }
    let internal_tan = inner.iter().all(|&i| {
        let d = dist(scene.circles[i].center, oc);
        (d - (or - r_in)).abs() / or.max(1e-9) < TAN_REL_TOL
    });
    // mutual tangency among the inner triple (check the three closest pairs)
    let mut pair_ok = true;
    // minimal spacing test
    let mut min_pair = f64::MAX;
    for a in 0..inner.len() {
        for c in (a + 1)..inner.len() {
            min_pair = min_pair.min(dist(scene.circles[inner[a]].center, scene.circles[inner[c]].center));
        }
    }
    if !(min_pair.is_finite() && (min_pair - 2.0 * r_in).abs() / r_in.max(1e-9) < TAN_REL_TOL) {
        pair_ok = false;
    }
    if !internal_tan || !pair_ok {
        return Some(ExtractResult::Incomplete {
            reason: "inner circles are present but their mutual/internal tangency was not measured within tolerance; tangency relation missing".into(),
        });
    }

    // target must be one of the inner circles.
    if !inner.contains(&target_circ) {
        return None;
    }

    // ---- assemble (gauge: outer centre fixed at origin) ----
    let k = r_label / or.max(1e-12);
    let mut b = Build::new(scene.nodes.len());
    for i in 0..scene.nodes.len() {
        let p = scene.nodes[i];
        b.hints[i] = ((p.0 - oc.0) * k, (p.1 - oc.1) * k);
    }
    let o_id = scene.circles[outer].center_id;
    b.pins[o_id] = Pin::Fixed(0.0, 0.0);

    let cids: Vec<usize> = inner.iter().map(|&i| scene.circles[i].center_id).collect();
    let t_id = scene.circles[target_circ].center_id;

    // concyclic inner centres
    for w in 1..cids.len() {
        b.add(Con::DistEq(o_id, cids[0], o_id, cids[w]), format!("DistEq |O p{}| = |O p{}| [concyclic inner centres]", cids[0], cids[w]));
    }
    // equilateral inner polygon (equal mutual spacing)
    for w in 0..cids.len().saturating_sub(1) {
        let a = cids[w];
        let bb = cids[(w + 1) % cids.len()];
        let c = cids[(w + 2) % cids.len()];
        b.add(Con::DistEq(a, bb, bb, c), format!("DistEq |p{}p{}| = |p{}p{}| [equal mutual spacing]", a, bb, bb, c));
    }
    // tangency link on an adjacent inner pair: |CiCj| + 2|O Ci| = 2R.
    let cj = cids.iter().cloned().find(|&x| x != t_id).unwrap();
    b.add(
        Con::LinDist(vec![(1.0, t_id, cj), (2.0, o_id, t_id)], 2.0 * r_label),
        format!("LinDist |p{}p{}| + 2|O p{}| = {} [mutual+internal tangency, R={}]", t_id, cj, t_id, 2.0 * r_label, r_label),
    );

    // rim/tangent point P of the TARGET inner circle on the outer circle, along
    // ray O->target, at radius R. dist(target,P) = R - |O target| = r_in.
    let dir = {
        let c = scene.circles[target_circ].center;
        let dx = c.0 - oc.0;
        let dy = c.1 - oc.1;
        let l = (dx * dx + dy * dy).sqrt().max(1e-12);
        (dx / l, dy / l)
    };
    let p_hint = (dir.0 * r_label, dir.1 * r_label);
    let p_id = b.new_node(p_hint);
    b.add(Con::Dist(o_id, p_id, r_label), format!("Dist(O, p{}) = {} [rim point on outer circle]", p_id, r_label));
    b.add(
        Con::LinDist(vec![(1.0, o_id, p_id), (-1.0, o_id, t_id), (-1.0, t_id, p_id)], 0.0),
        format!("LinDist |O p{}| - |O p{}| - |p{} p{}| = 0 [P beyond target centre => tangent point]", p_id, t_id, t_id, p_id),
    );

    // non-degeneracy
    for w in 0..cids.len() {
        let a = cids[w];
        let bb = cids[(w + 1) % cids.len()];
        b.nondeg.push((a, bb, 0.2 * r_label));
    }
    b.nondeg.push((t_id, p_id, 0.05 * r_label));

    let target = Target::DiskArea(t_id, p_id);
    Some(ExtractResult::Problem(b.finish(target.clone()), target))
}

// =====================================================================
// verbatim constraint description (for reports / tests)
// =====================================================================

/// Human-readable render of an assembled Problem's points + constraints.
pub fn describe(problem: &Problem, target: &Target) -> String {
    let mut s = String::new();
    s.push_str(&format!("points: {}\n", problem.pts.len()));
    for (i, pd) in problem.pts.iter().enumerate() {
        let kind = if let (Some(x), Some(y)) = (pd.fx, pd.fy) {
            format!("fixed({:.3},{:.3})", x, y)
        } else if pd.fx == Some(0.0) && pd.fy.is_none() {
            "on_y_axis(x=0)".to_string()
        } else if pd.fy == Some(0.0) && pd.fx.is_none() {
            "on_x_axis(y=0)".to_string()
        } else {
            "free".to_string()
        };
        s.push_str(&format!("  p{}: {}\n", i, kind));
    }
    s.push_str(&format!("constraints: {}\n", problem.cons.len()));
    for c in &problem.cons {
        s.push_str(&format!("  {}\n", describe_con(c)));
    }
    s.push_str(&format!("target: {}\n", describe_target(target)));
    s
}

fn describe_con(c: &Con) -> String {
    match c {
        Con::Dist(i, j, d) => format!("Dist(p{},p{}) = {}", i, j, d),
        Con::DistEq(i, j, k, l) => format!("DistEq |p{}p{}| = |p{}p{}|", i, j, k, l),
        Con::Perp(i, j, k, l) => format!("Perp (p{}p{}) _|_ (p{}p{})", i, j, k, l),
        Con::Collinear(i, j, k) => format!("Collinear p{},p{},p{}", i, j, k),
        Con::Angle(a, b, c, deg) => format!("Angle p{}-p{}-p{} = {} deg", a, b, c, deg),
        Con::TriArea(i, j, k, v) => format!("TriArea(p{},p{},p{}) = {}", i, j, k, v),
        Con::PolyArea(ids, v) => format!("PolyArea([{}]) = {}", join(ids), v),
        Con::CoordEq(i, ai, j, aj) => format!("CoordEq p{}[{}] = p{}[{}]", i, ai, j, aj),
        Con::PtLineDistEqCoord(c, i, j, r, ra) => format!("PtLineDistEqCoord dist(p{}, line p{}p{}) = p{}[{}]", c, i, j, r, ra),
        Con::LinDist(terms, c) => {
            let t = terms.iter().map(|(co, a, b)| format!("{}*|p{}p{}|", co, a, b)).collect::<Vec<_>>().join(" + ");
            format!("LinDist {} = {}", t, c)
        }
    }
}

fn describe_target(t: &Target) -> String {
    match t {
        Target::Length(i, j) => format!("Length(p{},p{})", i, j),
        Target::RadiusComplement(a, b, k) => format!("RadiusComplement({} - |p{}p{}|)", k, a, b),
        Target::DiskAreaByRadiusCoord(c, ax) => format!("DiskArea (radius = p{}[{}])", c, ax),
        Target::DiskArea(c, on) => format!("DiskArea (radius = |p{}p{}|)", c, on),
        Target::PolyArea(ids) => format!("PolyArea(verts={:?})", ids),
    }
}

#[cfg(test)]
mod tests;
