//! Phase P-B — GENERAL geometry PRIMITIVE DETECTOR.
//!
//! internal notes. Reads an arbitrary geometry-figure image into a
//! structured list of geometric PRIMITIVES — line segments, full circles,
//! circular arcs, polygons, points, right-angle marks — each carrying a
//! genuine confidence and pixel-evidence count, in normalized image
//! coordinates.
//!
//! This REPLACES the three hardcoded per-class arc-topology templates in
//! `geo_perceive_p4.rs` (`accepted_arcs` + the grid_leaf / leaf / quarter
//! class-selection chain in `perceive_photo`, which only ever recognised the
//! 3 leaf/quarter/grid figures). Here nothing is templated: every primitive
//! is measured from pixels by a general detector.
//!
//! It REUSES the class-agnostic front end of P-4 verbatim:
//!   * `decode_image_rgb`  — pure-Rust PNG + baseline-JPEG decode,
//!   * `rectify`           — outer-square homography → axis-aligned raster,
//!   * `shaded_boundary`   — colour-agnostic filled-region boundary points,
//!   * `ransac_circles` / `fit_circle` — the sequential-RANSAC circle fitter,
//!   * `px` / `p4_is_ink` / `p4_is_colored` — pixel predicates,
//!   * `snap_frac`         — small-denominator rational snap.
//! The RANSAC fitter is used GENERALLY (find every circle/arc in the figure),
//! not filtered down to the 3 corner/edge templates.
//!
//! HONESTY FLOOR (absolute): every emitted primitive carries real pixel
//! support and must clear an evidence threshold. A shape below threshold is
//! NOT emitted. If the raster is too blank / noisy / ambiguous to yield a
//! coherent primitive set, we return `DetectResult::Incomplete` — we never pad
//! the list with an invented primitive. There is NO per-image / filename /
//! hash / figure-name / answer lookup anywhere on this path: the detector sees
//! only pixels.
//!
//! OFFLINE library only. This module wires to nothing — no guest surface, no
//! promote, no serving change. The bridge image→primitives→constraints→solver
//! is later work (P-C label association, P-D constraint extraction, P-E wire to
//! `algsolve`). The output vocabulary is deliberately shaped toward
//! `crate::math::algsolve` (Point→PointDef, Segment→Dist, Circle→DiskArea,
//! Polygon→PolyArea, RightAngle→Perp) so that mapping is mechanical later.

use crate::vision::geo_perceive::RgbImage;
use crate::vision::geo_perceive_p4::{
    decode_image_rgb, ransac_circles, rectify, shaded_boundary, stroke_mask,
};

// =====================================================================
// Public output vocabulary
// =====================================================================

/// A single detected geometric primitive in normalized image coordinates
/// (top-left origin, x→right, y→down, scaled by the working raster width so
/// the canonical outer square spans roughly the unit square). Every variant
/// carries a `confidence` in [0,1] and, where meaningful, an integer
/// `support` = the number of pixels that genuinely voted for it.
#[derive(Debug, Clone)]
pub enum Primitive {
    /// A straight line segment from `a` to `b`.
    Segment { a: (f64, f64), b: (f64, f64), confidence: f64, support: usize },
    /// A full (≈360°) circle.
    Circle { center: (f64, f64), r: f64, confidence: f64, support: usize },
    /// A circular arc; `start`/`end` are radians (atan2 in raster space, i.e.
    /// y-down) bounding the covered sweep (the complement of the largest gap).
    Arc { center: (f64, f64), r: f64, start: f64, end: f64, confidence: f64, support: usize },
    /// A closed polygon (triangle / quad / pentagon / …), vertices in order.
    Polygon { verts: Vec<(f64, f64)>, confidence: f64 },
    /// A distinguished point (segment intersection or shared endpoint).
    Point { p: (f64, f64), confidence: f64 },
    /// A right-angle mark (small square glyph) located at `at`.
    RightAngle { at: (f64, f64), confidence: f64 },
}

impl Primitive {
    pub fn confidence(&self) -> f64 {
        match self {
            Primitive::Segment { confidence, .. }
            | Primitive::Circle { confidence, .. }
            | Primitive::Arc { confidence, .. }
            | Primitive::Polygon { confidence, .. }
            | Primitive::Point { confidence, .. }
            | Primitive::RightAngle { confidence, .. } => *confidence,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Primitive::Segment { .. } => "segment",
            Primitive::Circle { .. } => "circle",
            Primitive::Arc { .. } => "arc",
            Primitive::Polygon { .. } => "polygon",
            Primitive::Point { .. } => "point",
            Primitive::RightAngle { .. } => "right_angle",
        }
    }
}

/// A detected GEOMETRIC RELATION between two primitives (as opposed to a shape).
/// Every relation is MEASURED from detected primitive coordinates and carries a
/// `confidence` plus the residual `err` that earned it; it is emitted ONLY when
/// its geometric test passes within a principled tolerance. Indices reference
/// `Primitives::items`. This is the P-B half of the tangency / right-angle
/// relations P-D consumes; nothing here is assumed or looked up.
#[derive(Debug, Clone)]
pub enum Relation {
    /// Circle/arc `circle` tangent to the line of Segment `seg` (perp distance
    /// from centre to the segment line ~= radius, foot within the segment span).
    CircleSegmentTangent { circle: usize, seg: usize, err: f64, confidence: f64 },
    /// Circle/arc `a` tangent to Circle/arc `b`; `external` => |c1c2| ~= r1+r2,
    /// else internal |c1c2| ~= |r1-r2|.
    CircleCircleTangent { a: usize, b: usize, external: bool, err: f64, confidence: f64 },
    /// Measured right angle where Segments `s1`,`s2` meet at `at` (~90 deg),
    /// inferred from segment perpendicularity even without a drawn glyph mark.
    RightAngleMeasured { at: (f64, f64), s1: usize, s2: usize, err_deg: f64, confidence: f64 },
}

/// A coherent set of detected primitives.
#[derive(Debug, Clone)]
pub struct Primitives {
    pub items: Vec<Primitive>,
    /// True if the outer square was found and the raster was rectified.
    pub rectified: bool,
    /// Mean confidence over emitted primitives (0 if none).
    pub mean_confidence: f64,
    pub note: String,
    /// Measured geometric relations (tangency, right-angle) over `items`.
    pub relations: Vec<Relation>,
}

impl Primitives {
    pub fn count(&self, kind: &str) -> usize {
        self.items.iter().filter(|p| p.kind() == kind).count()
    }
    /// A stable, human-readable census of what was detected (for reports/tests).
    pub fn census(&self) -> String {
        format!(
            "segments={} circles={} arcs={} polygons={} points={} right_angles={} relations={} (rectified={}, mean_conf={:.2})",
            self.count("segment"),
            self.count("circle"),
            self.count("arc"),
            self.count("polygon"),
            self.count("point"),
            self.count("right_angle"),
            self.relations.len(),
            self.rectified,
            self.mean_confidence,
        )
    }
}

/// The honest result of a detection attempt.
#[derive(Debug, Clone)]
pub enum DetectResult {
    /// A coherent primitive set was found.
    Primitives(Primitives),
    /// The image could not be turned into a coherent primitive set. This is a
    /// first-class, honest outcome — NOT an error and NOT an empty guess.
    Incomplete { stage: String, reason: String },
}

// =====================================================================
// Tunables (evidence thresholds — the honesty floor)
// =====================================================================

const SEG_MIN_LEN_FRAC: f64 = 0.08; // shortest emitted segment, as fraction of raster width
const SEG_TOL_PX: f64 = 2.5; // how close a pixel must be to count as line support
const SEG_MIN_COVERAGE: f64 = 0.55; // fraction of the segment that must be inked to emit
const SEG_MERGE_ANG_DEG: f64 = 4.0; // collinear-merge angular tolerance
const ARC_FULL_DEG: f64 = 340.0; // ≥ this angular coverage ⇒ full circle, else arc
const ARC_MIN_DEG: f64 = 30.0; // shorter sweeps are not emitted as arcs
const VERTEX_EPS_FRAC: f64 = 0.04; // endpoint/vertex clustering radius
const RA_MIN_FRAC: f64 = 0.015; // right-angle glyph min bbox side (fraction of width)
const RA_MAX_FRAC: f64 = 0.09; // right-angle glyph max bbox side

// =====================================================================
// Entry point
// =====================================================================

/// Decode + (try to) rectify + detect every primitive. Takes ONLY the image
/// bytes; there is no side channel. Returns a coherent `Primitives` set or an
/// honest `Incomplete`.
pub fn detect_primitives(bytes: &[u8]) -> DetectResult {
    let img = match decode_image_rgb(bytes) {
        Some(i) => i,
        None => {
            return DetectResult::Incomplete {
                stage: "decode".into(),
                reason: "not a decodable PNG or baseline JPEG".into(),
            }
        }
    };
    if img.w < 40 || img.h < 40 {
        return DetectResult::Incomplete {
            stage: "decode".into(),
            reason: format!("image too small ({}x{})", img.w, img.h),
        };
    }
    // Reuse P-4's rectifier when the outer square is present; otherwise fall
    // back to the raw raster so figures without a clean frame still yield
    // primitives (honestly flagged rectified=false).
    let (raster, rectified) = match rectify(&img) {
        Some((r, _skew)) => (r, true),
        None => (img, false),
    };
    detect_on_raster(&raster, rectified)
}

/// Run all detectors on an already-decoded (optionally rectified) raster.
pub fn detect_on_raster(raster: &RgbImage, rectified: bool) -> DetectResult {
    let w = raster.w;
    let h = raster.h;
    let sf = w as f64; // uniform normalization scale (matches ransac_circles)

    // --- pixel evidence layers -------------------------------------------
    let ink = ink_mask(raster);
    let ink_count = ink.iter().filter(|b| **b).count();
    let ink_edges = edge_points(&ink, w, h);

    // Honest early-out: a blank / near-blank raster has essentially no ink and
    // no coloured fill. Do not manufacture primitives from noise.
    let colored_bnd = shaded_boundary(raster);
    if ink_count < 200 && colored_bnd.len() < 30 {
        return DetectResult::Incomplete {
            stage: "evidence".into(),
            reason: format!(
                "insufficient pixel evidence (ink={}, colored_boundary={}); blank or non-geometry image",
                ink_count,
                colored_bnd.len()
            ),
        };
    }

    // --- (1) line segments ------------------------------------------------
    let segments = detect_segments(&ink, w, h, sf);

    // --- (2) circles / arcs ----------------------------------------------
    // Curve evidence = coloured fill boundary ∪ ink edges not already lying on
    // a detected straight segment (so triangle/square sides do not pollute the
    // circle RANSAC). This generalises the P-4 circle fitter beyond the 3
    // filled arc templates to any outlined or filled circle/arc.
    // Curve evidence must EXCLUDE straight strokes: a coloured OUTLINE (now
    // visible via the general stroke mask) has a boundary that hugs the drawn
    // straight edges, so feeding it raw to the circle RANSAC invents a
    // huge-radius arc over the tilted-square sides / text. Filter BOTH the
    // coloured fill boundary AND the ink edges through the detected segments;
    // only points that lie on NO straight segment are genuine curve evidence.
    // A real curved boundary (e.g. redleaf eye arc, a drawn circle) is not on
    // any segment and is retained unchanged.
    let mut curve_pts: Vec<(f64, f64)> = Vec::new();
    for &(x, y) in &colored_bnd {
        if !on_any_segment(&segments, x / sf, y / sf, 4.0 / sf) {
            curve_pts.push((x, y));
        }
    }
    for &(x, y) in &ink_edges {
        if !on_any_segment(&segments, x / sf, y / sf, 4.0 / sf) {
            curve_pts.push((x, y));
        }
    }
    let circles_arcs = detect_circles_arcs(&curve_pts, w, sf);

    // --- (3) vertices / points -------------------------------------------
    let points = detect_points(&segments, sf);

    // --- (4) polygons -----------------------------------------------------
    let polygons = detect_polygons(&segments, sf);

    // --- (4b) corner triangles -------------------------------------------
    // A right triangle formed where a chord cuts two mutually-perpendicular
    // segments at their shared corner. Recovers regions (e.g. the outer /
    // tilted-square corner triangle) whose two legs are sub-runs of longer
    // sides and so never close a 3-cycle in the segment graph. Geometry only:
    // no labels, no target, no per-figure input.
    let corner_tris = detect_corner_triangles(&segments, &polygons, sf);

    // --- (5) right-angle marks -------------------------------------------
    let right_angles = detect_right_angles(&ink, w, h, sf, &points);

    // --- assemble ---------------------------------------------------------
    let mut items: Vec<Primitive> = Vec::new();
    items.extend(segments);
    items.extend(circles_arcs);
    items.extend(polygons);
    items.extend(corner_tris);
    items.extend(points);
    items.extend(right_angles);

    if items.is_empty() {
        return DetectResult::Incomplete {
            stage: "detect".into(),
            reason: "no primitive cleared its evidence threshold".into(),
        };
    }
    // --- (6) measured relations (tangency + right angle) on FINAL indices --
    let relations = detect_relations(&items);
    let mean = items.iter().map(|p| p.confidence()).sum::<f64>() / items.len() as f64;
    DetectResult::Primitives(Primitives {
        items,
        rectified,
        mean_confidence: mean,
        note: String::new(),
        relations,
    })
}

// =====================================================================
// Pixel evidence helpers
// =====================================================================

fn ink_mask(raster: &RgbImage) -> Vec<bool> {
    // General stroke mask: dark ink OR thin/boundary coloured strokes. Falls
    // back to exactly `p4_is_ink` when the figure has no coloured pixels, so
    // dark-ink renders are unchanged; thin coloured vector strokes now count.
    let m = stroke_mask(raster);
    m
}

/// Ink pixels that border a non-ink pixel (4-neighbourhood) — the stroke edge.
fn edge_points(ink: &[bool], w: usize, h: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !ink[y * w + x] {
                continue;
            }
            let mut border = false;
            let nb = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in nb {
                if nx >= w || ny >= h || !ink[ny * w + nx] {
                    border = true;
                    break;
                }
            }
            if border {
                pts.push((x as f64, y as f64));
            }
        }
    }
    pts
}

#[inline]
fn ink_at(ink: &[bool], w: usize, h: usize, x: i64, y: i64) -> bool {
    if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
        return false;
    }
    ink[y as usize * w + x as usize]
}

// =====================================================================
// (1) Line-segment detection: Hough accumulation → per-line run extraction →
//     coverage verification → collinear merge. Pure Rust, no external deps.
// =====================================================================

struct RawSeg {
    a: (f64, f64), // pixel space
    b: (f64, f64),
    coverage: f64,
    support: usize,
}

fn detect_segments(ink: &[bool], w: usize, h: usize, sf: f64) -> Vec<Primitive> {
    // ink pixels drive the accumulator (a stroke of width t votes ~L*t for its
    // line bin → a sharp peak).
    let mut ink_pts: Vec<(f64, f64)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if ink[y * w + x] {
                ink_pts.push((x as f64, y as f64));
            }
        }
    }
    if ink_pts.len() < 50 {
        return Vec::new();
    }

    let n_theta = 180usize; // 1° resolution
    let diag = ((w * w + h * h) as f64).sqrt();
    let rho_res = 2.0f64;
    let n_rho = (2.0 * diag / rho_res).ceil() as usize + 1;
    let mut cos_t = vec![0.0f64; n_theta];
    let mut sin_t = vec![0.0f64; n_theta];
    for t in 0..n_theta {
        let a = std::f64::consts::PI * (t as f64) / (n_theta as f64);
        cos_t[t] = a.cos();
        sin_t[t] = a.sin();
    }
    let mut acc = vec![0u32; n_theta * n_rho];
    for &(x, y) in &ink_pts {
        for t in 0..n_theta {
            let rho = x * cos_t[t] + y * sin_t[t];
            let ri = ((rho + diag) / rho_res).round();
            if ri < 0.0 {
                continue;
            }
            let ri = ri as usize;
            if ri < n_rho {
                acc[t * n_rho + ri] += 1;
            }
        }
    }
    let max_acc = *acc.iter().max().unwrap_or(&0);
    if max_acc == 0 {
        return Vec::new();
    }
    let acc_thresh = (0.10 * max_acc as f64).max(20.0) as u32;

    // local-maxima peak picking (3×3 NMS)
    let mut peaks: Vec<(usize, usize, u32)> = Vec::new();
    for t in 0..n_theta {
        for ri in 0..n_rho {
            let v = acc[t * n_rho + ri];
            if v < acc_thresh {
                continue;
            }
            let mut is_max = true;
            'nb: for dt in -1i64..=1 {
                for dr in -1i64..=1 {
                    if dt == 0 && dr == 0 {
                        continue;
                    }
                    let tt = t as i64 + dt;
                    let rr = ri as i64 + dr;
                    if tt < 0 || tt >= n_theta as i64 || rr < 0 || rr >= n_rho as i64 {
                        continue;
                    }
                    if acc[tt as usize * n_rho + rr as usize] > v {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                peaks.push((t, ri, v));
            }
        }
    }
    peaks.sort_by(|a, b| b.2.cmp(&a.2));
    peaks.truncate(80);

    let min_len = SEG_MIN_LEN_FRAC * sf;
    let max_gap = (0.03 * sf).max(8.0);
    let mut raws: Vec<RawSeg> = Vec::new();
    for (t, ri, _v) in peaks {
        let ct = cos_t[t];
        let st = sin_t[t];
        let rho = ri as f64 * rho_res - diag;
        // line direction (unit), perpendicular is (ct, st)
        let dir = (-st, ct);
        // gather ink pixels within tol of this line, projected along dir
        let mut proj: Vec<(f64, (f64, f64))> = Vec::new();
        for &(x, y) in &ink_pts {
            let d = (x * ct + y * st - rho).abs();
            if d <= SEG_TOL_PX {
                let p = x * dir.0 + y * dir.1;
                proj.push((p, (x, y)));
            }
        }
        if proj.len() < 20 {
            continue;
        }
        proj.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // split into runs by gaps along the line
        let mut run_start = 0usize;
        for i in 1..=proj.len() {
            let brk = i == proj.len() || (proj[i].0 - proj[i - 1].0) > max_gap;
            if brk {
                let a = proj[run_start].1;
                let b = proj[i - 1].1;
                let len = dist(a, b);
                if len >= min_len {
                    if let Some((cov, sup)) = verify_segment(ink, w, h, a, b) {
                        if cov >= SEG_MIN_COVERAGE {
                            raws.push(RawSeg { a, b, coverage: cov, support: sup });
                        }
                    }
                }
                run_start = i;
            }
        }
    }

    let merged = merge_segments(raws, sf);
    merged
        .into_iter()
        .map(|s| Primitive::Segment {
            a: (s.a.0 / sf, s.a.1 / sf),
            b: (s.b.0 / sf, s.b.1 / sf),
            confidence: s.coverage.min(0.99),
            support: s.support,
        })
        .collect()
}

/// Walk the candidate segment pixel-by-pixel; a step is "covered" if any ink
/// pixel lies within SEG_TOL_PX. Returns (coverage_fraction, support_count).
fn verify_segment(ink: &[bool], w: usize, h: usize, a: (f64, f64), b: (f64, f64)) -> Option<(f64, usize)> {
    let len = dist(a, b);
    if len < 1.0 {
        return None;
    }
    let steps = len.ceil() as usize;
    let tol = SEG_TOL_PX.ceil() as i64;
    let mut covered = 0usize;
    let mut support = 0usize;
    for i in 0..=steps {
        let f = i as f64 / steps as f64;
        let x = a.0 + (b.0 - a.0) * f;
        let y = a.1 + (b.1 - a.1) * f;
        let xi = x.round() as i64;
        let yi = y.round() as i64;
        let mut hit = false;
        'scan: for dy in -tol..=tol {
            for dx in -tol..=tol {
                if ink_at(ink, w, h, xi + dx, yi + dy) {
                    hit = true;
                    break 'scan;
                }
            }
        }
        if hit {
            covered += 1;
            support += 1;
        }
    }
    Some((covered as f64 / (steps as f64 + 1.0), support))
}

fn merge_segments(mut raws: Vec<RawSeg>, sf: f64) -> Vec<RawSeg> {
    // longest first so short duplicates fold into the dominant line
    raws.sort_by(|a, b| dist(b.a, b.b).partial_cmp(&dist(a.a, a.b)).unwrap());
    let ang_tol = SEG_MERGE_ANG_DEG.to_radians();
    let perp_tol = 0.02 * sf;
    let mut out: Vec<RawSeg> = Vec::new();
    for s in raws {
        let sa = seg_angle(&s);
        let mut folded = false;
        for m in out.iter_mut() {
            let ma = seg_angle(m);
            let da = ang_diff(sa, ma);
            if da > ang_tol {
                continue;
            }
            // collinear? both endpoints of s near m's infinite line
            let d1 = point_line_dist(s.a, m.a, m.b);
            let d2 = point_line_dist(s.b, m.a, m.b);
            if d1 > perp_tol || d2 > perp_tol {
                continue;
            }
            // overlapping / adjacent along the line?
            let dirv = norm_vec((m.b.0 - m.a.0, m.b.1 - m.a.1));
            let pts = [m.a, m.b, s.a, s.b];
            let projs: Vec<f64> = pts.iter().map(|p| (p.0 - m.a.0) * dirv.0 + (p.1 - m.a.1) * dirv.1).collect();
            let mlen = dist(m.a, m.b);
            let smin = projs[2].min(projs[3]);
            let smax = projs[2].max(projs[3]);
            if smax < -perp_tol || smin > mlen + perp_tol {
                continue; // disjoint on the line — keep separate
            }
            // take extreme endpoints as the merged span
            let mut idx_min = 0;
            let mut idx_max = 0;
            for (i, &pv) in projs.iter().enumerate() {
                if pv < projs[idx_min] {
                    idx_min = i;
                }
                if pv > projs[idx_max] {
                    idx_max = i;
                }
            }
            m.a = pts[idx_min];
            m.b = pts[idx_max];
            m.coverage = m.coverage.max(s.coverage);
            m.support += s.support;
            folded = true;
            break;
        }
        if !folded {
            out.push(s);
        }
    }
    out
}

fn on_any_segment(segs: &[Primitive], nx: f64, ny: f64, tol: f64) -> bool {
    for p in segs {
        if let Primitive::Segment { a, b, .. } = p {
            let d = point_line_dist((nx, ny), *a, *b);
            if d <= tol {
                // within the span too
                let dirv = norm_vec((b.0 - a.0, b.1 - a.1));
                let u = (nx - a.0) * dirv.0 + (ny - a.1) * dirv.1;
                let l = dist(*a, *b);
                if u >= -tol && u <= l + tol {
                    return true;
                }
            }
        }
    }
    false
}

// =====================================================================
// (2) Circle / arc detection — REUSES the P-4 sequential-RANSAC fitter, then
//     measures the angular sweep of each detected circle to decide full circle
//     vs partial arc, honestly.
// =====================================================================

fn detect_circles_arcs(curve_pts: &[(f64, f64)], n: usize, sf: f64) -> Vec<Primitive> {
    if curve_pts.len() < 70 {
        return Vec::new();
    }
    let circ = ransac_circles(curve_pts, n);
    let mut out = Vec::new();
    for c in &circ {
        // ransac reports side-units, bottom-left origin: recover pixel center.
        let cx_px = c.cx * sf;
        let cy_px = (sf - 1.0) - c.cy * sf;
        let r_px = c.r * sf;
        if r_px < 5.0 {
            continue;
        }
        // gather the curve points that actually lie on this circle
        let tol = 3.0f64;
        let mut angs: Vec<f64> = Vec::new();
        let mut inl: Vec<(f64, f64)> = Vec::new();
        for &(x, y) in curve_pts {
            let dr = ((x - cx_px).powi(2) + (y - cy_px).powi(2)).sqrt() - r_px;
            if dr.abs() <= tol {
                angs.push((y - cy_px).atan2(x - cx_px));
                inl.push((x, y));
            }
        }
        if inl.len() < 40 {
            continue; // not enough genuine support -> do not emit
        }
        // HONESTY: reject a "circle" whose inliers are actually a straight run of
        // points (a line masquerading as a huge-radius arc). A genuine arc has
        // real transverse spread because it curves.
        if looks_straight(&inl) {
            continue;
        }
        let (coverage_deg, start, end, occupancy) = angular_coverage(&mut angs);
        if occupancy < 0.6 {
            continue; // sparse/clustered inliers (e.g. polygon corners on a circumscribed circle) -> not a genuine arc
        }
        // top-left normalized center
        let center = (cx_px / sf, cy_px / sf);
        let r = r_px / sf;
        let support = inl.len();
        // confidence: inlier density around the (partial) sweep
        let conf = (support as f64 / (r_px * coverage_deg.to_radians()).max(1.0)).min(1.0);
        let conf = (0.4 + 0.6 * conf).min(0.99); // floor so a well-supported arc isn't under-reported
        if coverage_deg >= ARC_FULL_DEG {
            out.push(Primitive::Circle { center, r, confidence: conf, support });
        } else if coverage_deg >= ARC_MIN_DEG {
            out.push(Primitive::Arc { center, r, start, end, confidence: conf, support });
        }
    }
    out
}

/// True if the point set lies essentially on a straight line (transverse
/// spread negligible) -- i.e. NOT a genuine arc. Uses the smaller eigenvalue
/// of the 2x2 point covariance as the RMS transverse spread.
fn looks_straight(pts: &[(f64, f64)]) -> bool {
    let n = pts.len() as f64;
    if n < 3.0 {
        return true;
    }
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in pts {
        mx += x;
        my += y;
    }
    mx /= n;
    my /= n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in pts {
        let dx = x - mx;
        let dy = y - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    sxx /= n;
    sxy /= n;
    syy /= n;
    let tr = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let l_min = (tr / 2.0 - disc).max(0.0);
    l_min.sqrt() < 2.5
}

/// Given inlier angles (radians, any order), return (covered_degrees,
/// sweep_start, sweep_end) where the covered sweep is the complement of the
/// single largest angular gap.
fn angular_coverage(angs: &mut [f64]) -> (f64, f64, f64, f64) {
    angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = angs.len();
    if n < 2 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let two_pi = std::f64::consts::TAU;
    let mut max_gap = 0.0f64;
    let mut gap_at = 0usize; // gap between angs[gap_at] and angs[gap_at+1]
    for i in 0..n {
        let a = angs[i];
        let b = if i + 1 < n { angs[i + 1] } else { angs[0] + two_pi };
        let g = b - a;
        if g > max_gap {
            max_gap = g;
            gap_at = i;
        }
    }
    let covered = (two_pi - max_gap).to_degrees();
    // sweep runs from the point after the gap, around, to the point before it
    let start = if gap_at + 1 < n { angs[gap_at + 1] } else { angs[0] };
    let end = angs[gap_at];
    // dense-occupancy: fraction of the covered sweep actually filled by points
    // with only small (<6 deg) inter-point gaps. A real arc is ~continuous; a
    // fit through a few polygon corner-blobs is sparse spikes -> low occupancy.
    let gap_thresh = 6.0_f64.to_radians();
    let mut filled = 0.0f64;
    for i in 0..n {
        let a = angs[i];
        let b = if i + 1 < n { angs[i + 1] } else { angs[0] + two_pi };
        let g = b - a;
        if g < gap_thresh {
            filled += g;
        }
    }
    let sweep = (two_pi - max_gap).max(1e-6);
    let occupancy = (filled / sweep).min(1.0);
    (covered, start, end, occupancy)
}

// =====================================================================
// (3) Points — segment intersections (within spans) and shared endpoints,
//     clustered.
// =====================================================================

fn detect_points(segs: &[Primitive], sf: f64) -> Vec<Primitive> {
    let lines: Vec<((f64, f64), (f64, f64))> = segs
        .iter()
        .filter_map(|p| match p {
            Primitive::Segment { a, b, .. } => Some((*a, *b)),
            _ => None,
        })
        .collect();
    let mut raw: Vec<(f64, f64)> = Vec::new();
    // intersections
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if let Some(p) = seg_intersect(lines[i], lines[j]) {
                raw.push(p);
            }
        }
    }
    // endpoints
    for (a, b) in &lines {
        raw.push(*a);
        raw.push(*b);
    }
    let eps = VERTEX_EPS_FRAC; // normalized units
    let clusters = cluster_points(&raw, eps);
    let mut out = Vec::new();
    for (c, mult) in clusters {
        // a genuine vertex is where ≥2 endpoints/intersections coincide
        if mult < 2 {
            continue;
        }
        let conf = (0.5 + 0.1 * mult as f64).min(0.95);
        out.push(Primitive::Point { p: c, confidence: conf });
    }
    let _ = sf;
    out
}

fn seg_intersect(s1: ((f64, f64), (f64, f64)), s2: ((f64, f64), (f64, f64))) -> Option<(f64, f64)> {
    let (a, b) = s1;
    let (c, d) = s2;
    let r = (b.0 - a.0, b.1 - a.1);
    let s = (d.0 - c.0, d.1 - c.1);
    let rxs = r.0 * s.1 - r.1 * s.0;
    if rxs.abs() < 1e-9 {
        return None;
    }
    let qp = (c.0 - a.0, c.1 - a.1);
    let t = (qp.0 * s.1 - qp.1 * s.0) / rxs;
    let u = (qp.0 * r.1 - qp.1 * r.0) / rxs;
    if (-0.03..=1.03).contains(&t) && (-0.03..=1.03).contains(&u) {
        Some((a.0 + t * r.0, a.1 + t * r.1))
    } else {
        None
    }
}

fn cluster_points(pts: &[(f64, f64)], eps: f64) -> Vec<((f64, f64), usize)> {
    let mut centers: Vec<((f64, f64), usize)> = Vec::new();
    for &p in pts {
        let mut found = false;
        for c in centers.iter_mut() {
            if dist(c.0, p) < eps {
                let n = c.1 as f64;
                c.0 = ((c.0 .0 * n + p.0) / (n + 1.0), (c.0 .1 * n + p.1) / (n + 1.0));
                c.1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            centers.push((p, 1));
        }
    }
    centers
}

// =====================================================================
// (4) Polygon assembly — cluster segment endpoints into vertices, build the
//     segment graph, enumerate small chordless cycles (triangles..hexagons).
// =====================================================================

fn detect_polygons(segs: &[Primitive], sf: f64) -> Vec<Primitive> {
    let lines: Vec<((f64, f64), (f64, f64), f64)> = segs
        .iter()
        .filter_map(|p| match p {
            Primitive::Segment { a, b, confidence, .. } => Some((*a, *b, *confidence)),
            _ => None,
        })
        .collect();
    if lines.len() < 3 {
        return Vec::new();
    }
    let eps = VERTEX_EPS_FRAC;
    // build vertex list from all endpoints
    let mut verts: Vec<(f64, f64)> = Vec::new();
    fn vid(verts: &mut Vec<(f64, f64)>, p: (f64, f64), eps: f64) -> usize {
        for (i, v) in verts.iter().enumerate() {
            if dist(*v, p) < eps {
                return i;
            }
        }
        verts.push(p);
        verts.len() - 1
    }
    // edges (undirected) with coverage weight
    let mut adj: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut edge_set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    fn ensure(adj: &mut Vec<Vec<(usize, f64)>>, i: usize) {
        while adj.len() <= i {
            adj.push(Vec::new());
        }
    }
    for (a, b, conf) in &lines {
        let i = vid(&mut verts, *a, eps);
        let j = vid(&mut verts, *b, eps);
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        if edge_set.insert((lo, hi)) {
            ensure(&mut adj, i);
            ensure(&mut adj, j);
            adj[i].push((j, *conf));
            adj[j].push((i, *conf));
        }
    }
    // enumerate simple cycles of length 3..=6 by DFS from each start vertex
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    let mut polys: Vec<Primitive> = Vec::new();
    let nverts = verts.len();
    for start in 0..nverts {
        let mut stack: Vec<usize> = vec![start];
        dfs_cycles(start, start, &adj, &mut stack, &edge_set, &mut seen, &verts, &mut polys);
    }
    let _ = sf;
    // ADDITIVE: axis-aligned rectangle detector (Piece 1). Pairs H+V segments
    // into boxes directly, so squares whose sides are shared or subsumed by a
    // longer segment (e.g. resting on the ground line) still surface as clean
    // 4-gons -- the case the arrangement/DCEL path fragments. Endpoint-cycle
    // polygons above are kept (tilted quad + nested/simple figures) so the
    // 4*pi/9*pi/redleaf controls are unaffected. Deduped by rounded vertex set.
    let mut merged: Vec<Primitive> = polys;
    let mut keys: std::collections::HashSet<Vec<(i64, i64)>> = std::collections::HashSet::new();
    for p in &merged {
        if let Primitive::Polygon { verts, .. } = p {
            keys.insert(poly_key(verts));
        }
    }
    for r in detect_axis_rectangles(segs) {
        if let Primitive::Polygon { verts, .. } = &r {
            if keys.insert(poly_key(verts)) {
                merged.push(r);
            }
        }
    }
    // ADDITIVE: convex-quad detector. A rotated square/quad whose corners are
    // jittered across separate Hough segments (e.g. the apex where two strokes
    // blend into the ground line) will not close via endpoint clustering. Here
    // each quad CORNER is recovered as the line-line INTERSECTION of adjacent
    // edges -- robust to endpoint jitter -- and each of the four segments must
    // span its edge (corner near BOTH its endpoints), so only genuinely drawn
    // quads are emitted. General machinery; not axis-specific.
    for q in detect_convex_quads(segs) {
        if let Primitive::Polygon { verts, .. } = &q {
            if keys.insert(poly_key(verts)) {
                merged.push(q);
            }
        }
    }
    merged
}

/// Intersection of the infinite lines through a1-a2 and b1-b2.
fn line_intersect(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> Option<(f64, f64)> {
    let r = (a2.0 - a1.0, a2.1 - a1.1);
    let s2 = (b2.0 - b1.0, b2.1 - b1.1);
    let d = r.0 * s2.1 - r.1 * s2.0;
    if d.abs() < 1e-9 {
        return None;
    }
    let qp = (b1.0 - a1.0, b1.1 - a1.1);
    let t = (qp.0 * s2.1 - qp.1 * s2.0) / d;
    Some((a1.0 + t * r.0, a1.1 + t * r.1))
}

/// Corner shared by two segments: their lines meet at X, and X is within `ct`
/// of an endpoint of EACH segment (so each segment reaches the corner).
fn seg_corner(a: ((f64, f64), (f64, f64)), b: ((f64, f64), (f64, f64)), ct: f64) -> Option<(f64, f64)> {
    let x = line_intersect(a.0, a.1, b.0, b.1)?;
    let na = dist(x, a.0).min(dist(x, a.1));
    let nb = dist(x, b.0).min(dist(x, b.1));
    if na <= ct && nb <= ct {
        Some(x)
    } else {
        None
    }
}

/// Is the ordered vertex loop convex (all turns the same sign)?
fn is_convex(v: &[(f64, f64)]) -> bool {
    let n = v.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0i32;
    for i in 0..n {
        let a = v[i];
        let b = v[(i + 1) % n];
        let c = v[(i + 2) % n];
        let cr = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        let sg = if cr > 1e-9 { 1 } else if cr < -1e-9 { -1 } else { 0 };
        if sg != 0 {
            if sign == 0 {
                sign = sg;
            } else if sg != sign {
                return false;
            }
        }
    }
    true
}

/// Detect convex quadrilaterals: four segments that chain corner-to-corner,
/// each corner recovered as an adjacent-edge line intersection.
fn detect_convex_quads(segs: &[Primitive]) -> Vec<Primitive> {
    let ls: Vec<((f64, f64), (f64, f64))> = segs
        .iter()
        .filter_map(|p| match p {
            Primitive::Segment { a, b, .. } => {
                let len = dist(*a, *b);
                if len >= 0.15 {
                    Some((*a, *b))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();
    let n = ls.len();
    let ct = 0.12_f64;
    let mut out: Vec<Primitive> = Vec::new();
    let mut keys: std::collections::HashSet<Vec<(i64, i64)>> = std::collections::HashSet::new();
    if n < 4 {
        return out;
    }
    // all 4-subsets, all 3 distinct cyclic orderings (fix first element)
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    for order in [[i, j, k, l], [i, j, l, k], [i, k, j, l]] {
                        let mut corners: Vec<(f64, f64)> = Vec::with_capacity(4);
                        let mut ok = true;
                        for m in 0..4 {
                            let s1 = ls[order[m]];
                            let s2 = ls[order[(m + 1) % 4]];
                            match seg_corner(s1, s2, ct) {
                                Some(x) => corners.push(x),
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if !ok {
                            continue;
                        }
                        // corners must be distinct and reasonably separated
                        let mut distinct = true;
                        for p in 0..4 {
                            for q in (p + 1)..4 {
                                if dist(corners[p], corners[q]) < 0.05 {
                                    distinct = false;
                                }
                            }
                        }
                        if !distinct || !is_convex(&corners) {
                            continue;
                        }
                        if signed_area_abs(&corners) < 0.01 {
                            continue;
                        }
                        // Axis-aligned boxes are the rectangle detector's job;
                        // skip them here to avoid duplicate/competing polygons.
                        if quad_axis_aligned(&corners) {
                            continue;
                        }
                        // Require a genuine parallelogram (opposite sides ~parallel):
                        // the tilted square qualifies; incidental 4-segment
                        // combinations in triangle/other figures do not.
                        if !opposite_sides_parallel(&corners) {
                            continue;
                        }
                        let key = poly_key(&corners);
                        if keys.insert(key) {
                            out.push(Primitive::Polygon { verts: corners, confidence: 0.9 });
                        }
                    }
                }
            }
        }
    }
    out
}

/// True if every edge of the quad is near-horizontal or near-vertical.
fn quad_axis_aligned(v: &[(f64, f64)]) -> bool {
    if v.len() != 4 {
        return false;
    }
    (0..4).all(|i| {
        let a = v[i];
        let b = v[(i + 1) % 4];
        let dx = (b.0 - a.0).abs();
        let dy = (b.1 - a.1).abs();
        dy <= 0.15 * dx || dx <= 0.15 * dy
    })
}

/// True if both pairs of opposite edges are ~parallel (a parallelogram).
fn opposite_sides_parallel(v: &[(f64, f64)]) -> bool {
    if v.len() != 4 {
        return false;
    }
    let dir = |i: usize, j: usize| (v[j].0 - v[i].0, v[j].1 - v[i].1);
    let par = |a: (f64, f64), b: (f64, f64)| {
        let la = (a.0 * a.0 + a.1 * a.1).sqrt();
        let lb = (b.0 * b.0 + b.1 * b.1).sqrt();
        if la < 1e-9 || lb < 1e-9 {
            return false;
        }
        let cross = (a.0 * b.1 - a.1 * b.0).abs() / (la * lb);
        cross < 0.20
    };
    par(dir(0, 1), dir(3, 2)) && par(dir(1, 2), dir(0, 3))
}

/// Absolute shoelace area.
fn signed_area_abs(v: &[(f64, f64)]) -> f64 {
    let n = v.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += v[i].0 * v[j].1 - v[j].0 * v[i].1;
    }
    (a * 0.5).abs()
}

/// Rounded, order-independent key for a polygon's vertex set (dedup).
fn poly_key(verts: &[(f64, f64)]) -> Vec<(i64, i64)> {
    let mut k: Vec<(i64, i64)> = verts
        .iter()
        .map(|v| ((v.0 * 200.0).round() as i64, (v.1 * 200.0).round() as i64))
        .collect();
    k.sort();
    k
}

/// PIECE 1 -- axis-aligned rectangle / square detector. Classifies segments as
/// horizontal or vertical, then finds every (left,right) vertical + (top,bottom)
/// horizontal quadruple whose extents close a box: each side segment must COVER
/// the box span on its axis (it may extend beyond -- a shared or ground-subsumed
/// side -- or fall short only within tolerance). General machinery: a box is a
/// box regardless of which longer stroke carries one of its sides.
fn detect_axis_rectangles(segs: &[Primitive]) -> Vec<Primitive> {
    // (level, lo, hi, conf)
    let mut hs: Vec<(f64, f64, f64, f64)> = Vec::new(); // horizontal: y, x0, x1
    let mut vs: Vec<(f64, f64, f64, f64)> = Vec::new(); // vertical:   x, y0, y1
    for p in segs {
        if let Primitive::Segment { a, b, confidence, .. } = p {
            let dx = (b.0 - a.0).abs();
            let dy = (b.1 - a.1).abs();
            if dx < 1e-6 && dy < 1e-6 {
                continue;
            }
            if dy <= 0.15 * dx {
                // horizontal
                let y = (a.1 + b.1) * 0.5;
                let (x0, x1) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
                hs.push((y, x0, x1, *confidence));
            } else if dx <= 0.15 * dy {
                // vertical
                let x = (a.0 + b.0) * 0.5;
                let (y0, y1) = if a.1 < b.1 { (a.1, b.1) } else { (b.1, a.1) };
                vs.push((x, y0, y1, *confidence));
            }
        }
    }
    let align = 0.03_f64; // a side may fall short of a corner by up to this
    let min_side = 0.03_f64;
    let mut out: Vec<Primitive> = Vec::new();
    let mut keys: std::collections::HashSet<Vec<(i64, i64)>> = std::collections::HashSet::new();
    for i in 0..vs.len() {
        for j in 0..vs.len() {
            let (xl, ly0, ly1, cvl) = vs[i];
            let (xr, ry0, ry1, cvr) = vs[j];
            if xr - xl < min_side {
                continue;
            }
            for a in 0..hs.len() {
                for b in 0..hs.len() {
                    let (yt, tx0, tx1, cht) = hs[a];
                    let (yb, bx0, bx1, chb) = hs[b];
                    if yb - yt < min_side {
                        continue;
                    }
                    // each vertical must cover the vertical span [yt,yb]
                    if !(ly0 <= yt + align && ly1 >= yb - align) {
                        continue;
                    }
                    if !(ry0 <= yt + align && ry1 >= yb - align) {
                        continue;
                    }
                    // each horizontal must cover the horizontal span [xl,xr]
                    if !(tx0 <= xl + align && tx1 >= xr - align) {
                        continue;
                    }
                    if !(bx0 <= xl + align && bx1 >= xr - align) {
                        continue;
                    }
                    let verts = vec![(xl, yt), (xr, yt), (xr, yb), (xl, yb)];
                    let key = poly_key(&verts);
                    if !keys.insert(key) {
                        continue;
                    }
                    let conf = cvl.min(cvr).min(cht).min(chb);
                    out.push(Primitive::Polygon { verts, confidence: conf });
                }
            }
        }
    }
    // Near-duplicate suppression: the same box found via the ground line vs a
    // dedicated bottom segment (or jittered parallel strokes) yields boxes whose
    // four corners nearly coincide -- keep one representative.
    let mut filtered: Vec<Primitive> = Vec::new();
    for r in out {
        if let Primitive::Polygon { verts, .. } = &r {
            let dup = filtered.iter().any(|f| {
                if let Primitive::Polygon { verts: fv, .. } = f {
                    rects_close(verts, fv, 0.03)
                } else {
                    false
                }
            });
            if !dup {
                filtered.push(r);
            }
        }
    }
    filtered
}

/// Two axis-aligned rectangles (vertices in canonical TL,TR,BR,BL order) are
/// near-duplicates if all four corresponding corners are within .
fn rects_close(a: &[(f64, f64)], b: &[(f64, f64)], tol: f64) -> bool {
    if a.len() != 4 || b.len() != 4 {
        return false;
    }
    (0..4).all(|i| dist(a[i], b[i]) < tol)
}


#[allow(clippy::too_many_arguments)]
fn dfs_cycles(
    start: usize,
    cur: usize,
    adj: &[Vec<(usize, f64)>],
    stack: &mut Vec<usize>,
    edge_set: &std::collections::HashSet<(usize, usize)>,
    seen: &mut std::collections::HashSet<Vec<usize>>,
    verts: &[(f64, f64)],
    out: &mut Vec<Primitive>,
) {
    if stack.len() > 6 {
        return;
    }
    for &(nxt, _w) in &adj[cur] {
        if nxt == start && stack.len() >= 3 {
            // found a cycle; canonicalize and test chordless
            let cyc = stack.clone();
            if let Some(canon) = canonical_cycle(&cyc) {
                if seen.insert(canon.clone()) && is_chordless(&cyc, edge_set) {
                    let confs = cycle_conf(&cyc, adj);
                    let vv: Vec<(f64, f64)> = cyc.iter().map(|&i| verts[i]).collect();
                    out.push(Primitive::Polygon { verts: vv, confidence: confs });
                }
            }
            continue;
        }
        if nxt > start && !stack.contains(&nxt) {
            stack.push(nxt);
            dfs_cycles(start, nxt, adj, stack, edge_set, seen, verts, out);
            stack.pop();
        }
    }
}

/// Canonical form: rotate so smallest vertex id is first, orient so second
/// element is the smaller neighbour. Returns None for degenerate.
fn canonical_cycle(cyc: &[usize]) -> Option<Vec<usize>> {
    let n = cyc.len();
    if n < 3 {
        return None;
    }
    let min_pos = (0..n).min_by_key(|&i| cyc[i]).unwrap();
    let mut fwd: Vec<usize> = (0..n).map(|k| cyc[(min_pos + k) % n]).collect();
    let mut bwd: Vec<usize> = (0..n).map(|k| cyc[(min_pos + n - k) % n]).collect();
    // both start at min; pick lexicographically smaller
    if bwd < fwd {
        std::mem::swap(&mut fwd, &mut bwd);
    }
    Some(fwd)
}

/// A cycle is a genuine polygon face only if no two non-adjacent cycle
/// vertices are directly connected by an edge (no chord).
fn is_chordless(cyc: &[usize], edge_set: &std::collections::HashSet<(usize, usize)>) -> bool {
    let n = cyc.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let adjacent = j == i + 1 || (i == 0 && j == n - 1);
            if adjacent {
                continue;
            }
            let (lo, hi) = if cyc[i] < cyc[j] { (cyc[i], cyc[j]) } else { (cyc[j], cyc[i]) };
            if edge_set.contains(&(lo, hi)) {
                return false;
            }
        }
    }
    true
}

fn cycle_conf(cyc: &[usize], adj: &[Vec<(usize, f64)>]) -> f64 {
    let n = cyc.len();
    let mut minc = 1.0f64;
    for i in 0..n {
        let a = cyc[i];
        let b = cyc[(i + 1) % n];
        let mut w = 0.0;
        for &(nn, ww) in &adj[a] {
            if nn == b {
                w = ww;
                break;
            }
        }
        minc = minc.min(w);
    }
    minc
}

// =====================================================================
// (5) Right-angle marks — small square ink glyphs sitting at a vertex.
//     Conservative by design: a miss is preferable to an invented mark.
// =====================================================================

fn detect_right_angles(ink: &[bool], w: usize, h: usize, sf: f64, points: &[Primitive]) -> Vec<Primitive> {
    // connected components of ink
    let mut label = vec![0i32; w * h];
    let mut out = Vec::new();
    let min_side = (RA_MIN_FRAC * sf) as i64;
    let max_side = (RA_MAX_FRAC * sf) as i64;
    let mut next = 1i32;
    let mut stack: Vec<(i64, i64)> = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            if !ink[sy * w + sx] || label[sy * w + sx] != 0 {
                continue;
            }
            // flood fill
            let id = next;
            next += 1;
            let (mut x0, mut y0, mut x1, mut y1) = (sx as i64, sy as i64, sx as i64, sy as i64);
            let mut area = 0usize;
            stack.clear();
            stack.push((sx as i64, sy as i64));
            label[sy * w + sx] = id;
            while let Some((cx, cy)) = stack.pop() {
                area += 1;
                x0 = x0.min(cx);
                y0 = y0.min(cy);
                x1 = x1.max(cx);
                y1 = y1.max(cy);
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if ink_at(ink, w, h, nx, ny) && label[ny as usize * w + nx as usize] == 0 {
                        label[ny as usize * w + nx as usize] = id;
                        stack.push((nx, ny));
                    }
                }
            }
            let bw = x1 - x0 + 1;
            let bh = y1 - y0 + 1;
            if bw < min_side || bh < min_side || bw > max_side || bh > max_side {
                continue;
            }
            // near-square bbox
            let aspect = bw as f64 / bh as f64;
            if !(0.6..=1.6).contains(&aspect) {
                continue;
            }
            // an L / small-square glyph is mostly outline: fill ratio should be
            // modest (not a solid blob, not a speck)
            let fill = area as f64 / (bw as f64 * bh as f64);
            if !(0.12..=0.75).contains(&fill) {
                continue;
            }
            let cx = (x0 + x1) as f64 / 2.0 / sf;
            let cy = (y0 + y1) as f64 / 2.0 / sf;
            // must sit at/near a detected vertex to count as a right-angle mark
            let mut near_vertex = false;
            for p in points {
                if let Primitive::Point { p: vp, .. } = p {
                    if dist(*vp, (cx, cy)) < 0.08 {
                        near_vertex = true;
                        break;
                    }
                }
            }
            if !near_vertex {
                continue;
            }
            out.push(Primitive::RightAngle { at: (cx, cy), confidence: 0.5 });
        }
    }
    out
}

// =====================================================================
// (4b) corner-triangle reconstruction + (6) measured relations
// =====================================================================

/// Reconstruct right-triangle regions where a chord segment cuts across the
/// corner formed by two mutually-perpendicular segments that share an endpoint.
/// Recovers the outer/tilted-square corner triangles whose two legs are sub-runs
/// of longer sides (never a 3-cycle among detected segments). Geometry only.
fn detect_corner_triangles(segs: &[Primitive], existing: &[Primitive], _sf: f64) -> Vec<Primitive> {
    let lines: Vec<((f64, f64), (f64, f64), f64)> = segs
        .iter()
        .filter_map(|p| match p {
            Primitive::Segment { a, b, confidence, .. } => Some((*a, *b, *confidence)),
            _ => None,
        })
        .collect();
    let n = lines.len();
    let min_leg = 0.05_f64;
    let end_tol = 0.03_f64;
    let perp_tol = (8.0_f64).to_radians();
    let mut out: Vec<Primitive> = Vec::new();
    let mut seen: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        let (a1, b1, c1) = lines[i];
        for j in (i + 1)..n {
            let (a2, b2, c2) = lines[j];
            let ang = ang_between(a1, b1, a2, b2);
            if (ang - std::f64::consts::FRAC_PI_2).abs() > perp_tol {
                continue;
            }
            let corner = match shared_endpoint(a1, b1, a2, b2, end_tol) {
                Some(o) => o,
                None => continue,
            };
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                let (a3, b3, c3) = lines[k];
                for &(e1, e2) in &[(a3, b3), (b3, a3)] {
                    if !(point_on_seg_in_span(e1, a1, b1, end_tol)
                        && point_on_seg_in_span(e2, a2, b2, end_tol))
                    {
                        continue;
                    }
                    if dist(corner, e1) < min_leg || dist(corner, e2) < min_leg {
                        continue;
                    }
                    let area2 = ((e1.0 - corner.0) * (e2.1 - corner.1)
                        - (e1.1 - corner.1) * (e2.0 - corner.0))
                        .abs();
                    if area2 < 1e-6 {
                        continue;
                    }
                    let cen = (
                        (corner.0 + e1.0 + e2.0) / 3.0,
                        (corner.1 + e1.1 + e2.1) / 3.0,
                    );
                    if seen.iter().any(|s| dist(*s, cen) < 0.02) {
                        continue;
                    }
                    let dup_existing = existing.iter().any(|p| match p {
                        Primitive::Polygon { verts, .. } => {
                            verts.len() == 3
                                && poly_centroid(verts).map(|pc| dist(pc, cen) < 0.03).unwrap_or(false)
                        }
                        _ => false,
                    });
                    if dup_existing {
                        continue;
                    }
                    seen.push(cen);
                    let conf = (c1.min(c2).min(c3) * 0.95).min(0.97);
                    out.push(Primitive::Polygon { verts: vec![corner, e1, e2], confidence: conf });
                }
            }
        }
    }
    out
}

/// Measured relations over the FINAL item list: circle<->segment tangency,
/// circle<->circle tangency, and right angles between perpendicular segments
/// meeting at a corner. Every relation is measured from coordinates and emitted
/// only when its test passes; nothing is assumed.
fn detect_relations(items: &[Primitive]) -> Vec<Relation> {
    let mut rels = Vec::new();
    let mut circs: Vec<(usize, (f64, f64), f64)> = Vec::new();
    let mut segl: Vec<(usize, (f64, f64), (f64, f64))> = Vec::new();
    for (i, p) in items.iter().enumerate() {
        match p {
            Primitive::Circle { center, r, .. } | Primitive::Arc { center, r, .. } => {
                circs.push((i, *center, *r));
            }
            Primitive::Segment { a, b, .. } => segl.push((i, *a, *b)),
            _ => {}
        }
    }
    let tan_rel_tol = 0.14_f64;
    for &(ci, c, r) in &circs {
        if r <= 1e-9 {
            continue;
        }
        for &(si, a, b) in &segl {
            if let Some(d) = perp_dist_in_span(c, a, b) {
                let err = (d - r).abs();
                if err / r < tan_rel_tol {
                    let q = (1.0 - (err / r) / tan_rel_tol).clamp(0.0, 1.0);
                    let confidence = (0.4 + 0.6 * q).min(0.97);
                    rels.push(Relation::CircleSegmentTangent { circle: ci, seg: si, err, confidence });
                }
            }
        }
    }
    for a in 0..circs.len() {
        for b in (a + 1)..circs.len() {
            let (ai, ca, ra) = circs[a];
            let (bi, cb, rb) = circs[b];
            let d = dist(ca, cb);
            if d <= 1e-6 {
                continue;
            }
            let ext = (d - (ra + rb)).abs();
            let inte = (d - (ra - rb).abs()).abs();
            let scale = (ra + rb).max(1e-9);
            if ext / scale < tan_rel_tol {
                let q = (1.0 - (ext / scale) / tan_rel_tol).clamp(0.0, 1.0);
                let confidence = (0.4 + 0.6 * q).min(0.97);
                rels.push(Relation::CircleCircleTangent { a: ai, b: bi, external: true, err: ext, confidence });
            } else if inte / scale < tan_rel_tol {
                let q = (1.0 - (inte / scale) / tan_rel_tol).clamp(0.0, 1.0);
                let confidence = (0.4 + 0.6 * q).min(0.97);
                rels.push(Relation::CircleCircleTangent { a: ai, b: bi, external: false, err: inte, confidence });
            }
        }
    }
    let perp_tol = (7.0_f64).to_radians();
    for i in 0..segl.len() {
        for j in (i + 1)..segl.len() {
            let (i1, a1, b1) = segl[i];
            let (i2, a2, b2) = segl[j];
            let ang = ang_between(a1, b1, a2, b2);
            let errd = (ang - std::f64::consts::FRAC_PI_2).abs();
            if errd > perp_tol {
                continue;
            }
            if let Some(o) = shared_endpoint(a1, b1, a2, b2, 0.03) {
                let confidence = (0.5 + 0.5 * (1.0 - errd / perp_tol)).min(0.95);
                rels.push(Relation::RightAngleMeasured { at: o, s1: i1, s2: i2, err_deg: errd.to_degrees(), confidence });
            }
        }
    }
    rels
}

/// Undirected angle between two segments (as lines) in [0, pi/2].
fn ang_between(a1: (f64, f64), b1: (f64, f64), a2: (f64, f64), b2: (f64, f64)) -> f64 {
    let t1 = (b1.1 - a1.1).atan2(b1.0 - a1.0);
    let t2 = (b2.1 - a2.1).atan2(b2.0 - a2.0);
    ang_diff(t1, t2)
}

/// The near-common endpoint (midpoint of the closest endpoint pair) of two
/// segments, if any pair is within `tol`.
fn shared_endpoint(a1: (f64, f64), b1: (f64, f64), a2: (f64, f64), b2: (f64, f64), tol: f64) -> Option<(f64, f64)> {
    let pairs = [(a1, a2), (a1, b2), (b1, a2), (b1, b2)];
    let mut best: Option<((f64, f64), f64)> = None;
    for (p, q) in pairs {
        let d = dist(p, q);
        if d < tol && best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0), d));
        }
    }
    best.map(|(m, _)| m)
}

/// Perpendicular distance from `p` to the INFINITE line through a,b, but only
/// when the foot of the perpendicular lies within the segment span (+/- margin).
fn perp_dist_in_span(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> Option<f64> {
    let ab = (b.0 - a.0, b.1 - a.1);
    let l2 = ab.0 * ab.0 + ab.1 * ab.1;
    if l2 < 1e-12 {
        return None;
    }
    let u = ((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / l2;
    if !(-0.15..=1.15).contains(&u) {
        return None;
    }
    let d = ((p.0 - a.0) * ab.1 - (p.1 - a.1) * ab.0).abs() / l2.sqrt();
    Some(d)
}

fn point_on_seg_in_span(p: (f64, f64), a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
    match perp_dist_in_span(p, a, b) {
        Some(d) => d < tol,
        None => false,
    }
}

fn poly_centroid(v: &[(f64, f64)]) -> Option<(f64, f64)> {
    if v.is_empty() {
        return None;
    }
    let n = v.len() as f64;
    Some((v.iter().map(|p| p.0).sum::<f64>() / n, v.iter().map(|p| p.1).sum::<f64>() / n))
}

// =====================================================================
// small geometry helpers
// =====================================================================

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn norm_vec(v: (f64, f64)) -> (f64, f64) {
    let l = (v.0 * v.0 + v.1 * v.1).sqrt().max(1e-9);
    (v.0 / l, v.1 / l)
}

fn point_line_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let l2 = ab.0 * ab.0 + ab.1 * ab.1;
    if l2 < 1e-12 {
        return dist(p, a);
    }
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / l2).clamp(0.0, 1.0);
    dist(p, (a.0 + t * ab.0, a.1 + t * ab.1))
}

fn seg_angle(s: &RawSeg) -> f64 {
    (s.b.1 - s.a.1).atan2(s.b.0 - s.a.0)
}

/// Undirected angular difference in [0, π/2] (lines, not rays).
fn ang_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b).abs() % std::f64::consts::PI;
    if d > std::f64::consts::FRAC_PI_2 {
        d = std::f64::consts::PI - d;
    }
    d
}

// =====================================================================
// algsolve mapping documentation (P-E will implement the real wiring)
// =====================================================================

impl Primitives {
    /// A textual sketch of how these primitives map toward the
    /// `crate::math::algsolve` constraint vocabulary. This is documentation /
    /// a design anchor for P-C/P-D/P-E — it performs NO wiring and is not on
    /// any live path.
    pub fn algsolve_hint(&self) -> String {
        let mut s = String::new();
        s.push_str("# primitive -> algsolve mapping (P-E target)\n");
        for (i, p) in self.items.iter().enumerate() {
            match p {
                Primitive::Point { .. } => s.push_str(&format!("p{}: Point -> PointDef\n", i)),
                Primitive::Segment { .. } => s.push_str(&format!("p{}: Segment -> 2x PointDef + Con::Dist\n", i)),
                Primitive::Circle { .. } => s.push_str(&format!("p{}: Circle -> center PointDef + Target::DiskArea\n", i)),
                Primitive::Arc { .. } => s.push_str(&format!("p{}: Arc -> center PointDef + partial-sweep\n", i)),
                Primitive::Polygon { verts, .. } => s.push_str(&format!("p{}: Polygon({}) -> Con::PolyArea over vertex ids\n", i, verts.len())),
                Primitive::RightAngle { .. } => s.push_str(&format!("p{}: RightAngle -> Con::Perp\n", i)),
            }
        }
        s
    }
}

#[cfg(test)]
mod tests;
