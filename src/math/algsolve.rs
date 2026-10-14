//! Offline algebraic CONSTRAINT solver — derives a figure's unknowns from its
//! constraints, then RECOVERS the queried quantity as an exact algebraic number
//! (`algnum::Alg`, a multi-quadratic field over `BigRational`).
//!
//! This is the foundation that unblocks families the `a + b*pi` partition solver
//! (`geometry.rs`, whose coefficients are `i64` rationals) cannot represent —
//! e.g. lengths carrying `sqrt(3)`. The APPROACH is ported from the Phase-3
//! reference prototype `prototypes/geosolve/solver.py`:
//!
//!   1. place the figure numerically from several random seeds
//!      (Levenberg-Marquardt over the constraint residuals),
//!   2. read the queried length/area on each valid placement,
//!   3. RECOVER the exact closed form (rational, sqrt(rational), a+b*sqrt(d)),
//!   4. gate on HONESTY.
//!
//! HONESTY K-GATE (never fabricate, never guess):
//!   * require >= 5 independent valid placements that AGREE (invariance).
//!     Disagreement / not enough placements => `Refuse(UnderDetermined)`,
//!     NO number.
//!   * exact recovery must land a closed `Alg` form (optionally scaled by pi for
//!     disk areas); otherwise => `Refuse(NoClosedForm)`, NO number.
//!   * a value is emitted only when it is determinate AND exactly recovered AND
//!     it re-verifies (its f64 matches every valid placement's query).
//!
//! The answer is always DERIVED from the constraint system — there is no lookup
//! table, no figure-name switch, no stored answer constant.
//!
//! Reuse: the exact number type and every ring operation come from
//! `super::algnum::Alg`; this module never re-implements `Alg`. The answer is
//! carried in the same `(rat, pic)` shape (`a + b*pi`) that
//! `super::arrangement::area_exact` returns, and `verify_disk_area_exact` feeds a
//! recovered integer disk back through `arrangement` as an independent exact
//! cross-check.
//!
//! Build/test only: declared `pub mod algsolve;` in `math.rs` and exercised
//! entirely through its own unit tests. It does NOT touch the live `mcp` routing
//! cascade, the guest, or any serving surface.

use super::algnum::Alg;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Signed;

// =====================================================================
// exact-number helpers (thin wrappers over the real Alg API)
// =====================================================================

fn q(num: i64, den: i64) -> BigRational {
    BigRational::new(BigInt::from(num), BigInt::from(den))
}

/// sqrt of a non-negative rational, exact, via the real `Alg::sqrt_rational`.
fn alg_sqrt_ratio(num: i64, den: i64) -> Alg {
    Alg::sqrt_rational(&q(num, den))
}

/// Human rendering of an `Alg`: `2`, `-3`, `3/4`, `2*sqrt(3)`, `-3 + 2*sqrt(3)`.
pub fn fmt_alg(a: &Alg) -> String {
    if a.is_zero() {
        return "0".into();
    }
    let mut out = String::new();
    let mut first = true;
    for (primes, coeff) in &a.t {
        let radicand: u128 = primes.iter().map(|&p| p as u128).product::<u128>().max(1);
        let neg = coeff.is_negative();
        let mag = if neg { -coeff.clone() } else { coeff.clone() };
        let num = mag.numer().clone();
        let den = mag.denom().clone();
        let one = BigInt::from(1);
        let mag_is_one = num == one && den == one;
        // sign / joiner
        if first {
            if neg {
                out.push('-');
            }
        } else if neg {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        first = false;
        let coeff_str = if den == one {
            format!("{}", num)
        } else {
            format!("{}/{}", num, den)
        };
        if primes.is_empty() {
            out.push_str(&coeff_str);
        } else if mag_is_one {
            out.push_str(&format!("sqrt({})", radicand));
        } else {
            out.push_str(&format!("{}*sqrt({})", coeff_str, radicand));
        }
    }
    out
}

// =====================================================================
// constraint model
// =====================================================================

#[derive(Clone)]
pub struct PointDef {
    pub fx: Option<f64>,
    pub fy: Option<f64>,
    pub init: Option<(f64, f64)>,
}
impl PointDef {
    pub fn free() -> PointDef {
        PointDef { fx: None, fy: None, init: None }
    }
    pub fn fixed(x: f64, y: f64) -> PointDef {
        PointDef { fx: Some(x), fy: Some(y), init: None }
    }
    pub fn on_x_axis() -> PointDef {
        PointDef { fx: None, fy: Some(0.0), init: None }
    }
    pub fn on_y_axis() -> PointDef {
        PointDef { fx: Some(0.0), fy: None, init: None }
    }
    pub fn hint(x: f64, y: f64) -> PointDef {
        PointDef { fx: None, fy: None, init: Some((x, y)) }
    }
}

/// Each constraint contributes one residual that should be driven to 0.
#[derive(Clone)]
pub enum Con {
    /// |p_i - p_j| = d
    Dist(usize, usize, f64),
    /// |p_i - p_j| = |p_k - p_l|
    DistEq(usize, usize, usize, usize),
    /// (p_j - p_i) . (p_l - p_k) = 0
    Perp(usize, usize, usize, usize),
    /// cross(p_j - p_i, p_k - p_i) = 0  (i,j,k collinear)
    Collinear(usize, usize, usize),
    /// unsigned angle at vertex j between i and k = degrees
    Angle(usize, usize, usize, f64),
    /// signed triangle area (i,j,k) = v
    TriArea(usize, usize, usize, f64),
    /// signed shoelace area of the polygon = v
    PolyArea(Vec<usize>, f64),
    /// coordinate equality: p_i[axis_i] = p_j[axis_j]   (axis: 0=x, 1=y)
    CoordEq(usize, usize, usize, usize),
    /// distance from point c to the line through (i,j) equals coord p_r[axis_r]
    PtLineDistEqCoord(usize, usize, usize, usize, usize),
    /// sum_k coeff_k * |p_a - p_b| = c
    LinDist(Vec<(f64, usize, usize)>, f64),
}

pub struct Problem {
    pub pts: Vec<PointDef>,
    pub cons: Vec<Con>,
    /// pairs (i,j,min) requiring |p_i - p_j| >= min for a placement to be VALID
    /// (excludes degenerate branches, e.g. coincident tangent circles).
    pub nondeg: Vec<(usize, usize, f64)>,
}

/// What quantity to read off a placement, and how to interpret the recovery.
#[derive(Clone)]
pub enum Target {
    /// distance |p_i - p_j|; recovered as a pure length (pic = 0)
    Length(usize, usize),
    /// r = k - |p_a - p_b|; recovered as a pure length (pic = 0)
    RadiusComplement(usize, usize, f64),
    /// disk area: radius = coord p_c[axis]; answer = pi * r^2 (rat = 0)
    DiskAreaByRadiusCoord(usize, usize),
    /// disk area: radius = |p_c - p_on|; answer = pi * r^2 (rat = 0)
    DiskArea(usize, usize),
    /// area of the polygon given by the ordered vertex ids (shoelace, absolute
    /// value so it is invariant to winding/reflection). Recovered as a pure
    /// number (pic = 0); carries sqrt-terms exactly via the same recovery as a
    /// length. This is the solve-for target a labelled polygon '?' needs.
    PolyArea(Vec<usize>),
}

enum Mode {
    Length,
    DiskArea,
}

// =====================================================================
// result
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Refuse {
    /// fewer than K valid placements agreed, or the query varies across them
    UnderDetermined,
    /// determinate, but no exact closed form (rational / sqrt / a+b*pi) exists
    NoClosedForm,
    /// could not place the figure at all (numeric failure)
    PlacementFailed,
}

/// The K-gate result. `Value` carries the answer as `rat + pic*pi` over `Alg`.
pub enum Solved {
    Value { rat: Alg, pic: Alg, text: String },
    Refused(Refuse),
}

// =====================================================================
// numeric placement (Levenberg-Marquardt), ported from solver.py
// =====================================================================

type Coords = Vec<(f64, f64)>;

fn coord(p: &Coords, i: usize, axis: usize) -> f64 {
    if axis == 0 {
        p[i].0
    } else {
        p[i].1
    }
}

fn dist(p: &Coords, i: usize, j: usize) -> f64 {
    let dx = p[i].0 - p[j].0;
    let dy = p[i].1 - p[j].1;
    (dx * dx + dy * dy).sqrt()
}

fn angle_at(p: &Coords, a: usize, b: usize, c: usize) -> f64 {
    let ux = p[a].0 - p[b].0;
    let uy = p[a].1 - p[b].1;
    let vx = p[c].0 - p[b].0;
    let vy = p[c].1 - p[b].1;
    let cu = uy.atan2(ux);
    let cv = vy.atan2(vx);
    let mut d = (cu - cv).abs() % (2.0 * std::f64::consts::PI);
    if d > std::f64::consts::PI {
        d = 2.0 * std::f64::consts::PI - d;
    }
    d
}

fn residuals(prob: &Problem, p: &Coords) -> Vec<f64> {
    let mut r = Vec::new();
    for c in &prob.cons {
        match c {
            Con::Dist(i, j, d) => r.push(dist(p, *i, *j) - d),
            Con::DistEq(i, j, k, l) => r.push(dist(p, *i, *j) - dist(p, *k, *l)),
            Con::Perp(i, j, k, l) => {
                let ux = p[*j].0 - p[*i].0;
                let uy = p[*j].1 - p[*i].1;
                let vx = p[*l].0 - p[*k].0;
                let vy = p[*l].1 - p[*k].1;
                r.push(ux * vx + uy * vy);
            }
            Con::Collinear(i, j, k) => {
                let ux = p[*j].0 - p[*i].0;
                let uy = p[*j].1 - p[*i].1;
                let vx = p[*k].0 - p[*i].0;
                let vy = p[*k].1 - p[*i].1;
                r.push(ux * vy - uy * vx);
            }
            Con::Angle(a, b, c2, deg) => r.push(angle_at(p, *a, *b, *c2) - deg.to_radians()),
            Con::TriArea(i, j, k, v) => {
                let ux = p[*j].0 - p[*i].0;
                let uy = p[*j].1 - p[*i].1;
                let vx = p[*k].0 - p[*i].0;
                let vy = p[*k].1 - p[*i].1;
                r.push(0.5 * (ux * vy - uy * vx) - v);
            }
            Con::PolyArea(ids, v) => {
                let mut s = 0.0;
                let n = ids.len();
                for a in 0..n {
                    let i = ids[a];
                    let j = ids[(a + 1) % n];
                    s += p[i].0 * p[j].1 - p[j].0 * p[i].1;
                }
                r.push(0.5 * s - v);
            }
            Con::CoordEq(i, ai, j, aj) => r.push(coord(p, *i, *ai) - coord(p, *j, *aj)),
            Con::PtLineDistEqCoord(c, i, j, rr, ra) => {
                let ux = p[*j].0 - p[*i].0;
                let uy = p[*j].1 - p[*i].1;
                let wx = p[*c].0 - p[*i].0;
                let wy = p[*c].1 - p[*i].1;
                let len = (ux * ux + uy * uy).sqrt();
                let d = (ux * wy - uy * wx).abs() / len;
                r.push(d - coord(p, *rr, *ra));
            }
            Con::LinDist(terms, c) => {
                let mut s = 0.0;
                for (co, a, b) in terms {
                    s += co * dist(p, *a, *b);
                }
                r.push(s - c);
            }
        }
    }
    r
}

/// Gaussian elimination with partial pivoting for the small normal equations.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > best {
                best = a[row][col].abs();
                piv = row;
            }
        }
        if best < 1e-18 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in (row + 1)..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

// simple deterministic PRNG (xorshift64*) so tests are reproducible
fn rng_next(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let v = x.wrapping_mul(0x2545F4914F6CDD1D);
    (v >> 11) as f64 / (1u64 << 53) as f64 // in [0,1)
}

/// Build the free-variable index map: (point, axis) for every non-fixed coord.
fn free_vars(prob: &Problem) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for (idx, pd) in prob.pts.iter().enumerate() {
        if pd.fx.is_none() {
            v.push((idx, 0));
        }
        if pd.fy.is_none() {
            v.push((idx, 1));
        }
    }
    v
}

fn assemble(prob: &Problem, free: &[(usize, usize)], x: &[f64]) -> Coords {
    let mut p: Coords = prob
        .pts
        .iter()
        .map(|pd| (pd.fx.unwrap_or(0.0), pd.fy.unwrap_or(0.0)))
        .collect();
    for (k, &(pi, ax)) in free.iter().enumerate() {
        if ax == 0 {
            p[pi].0 = x[k];
        } else {
            p[pi].1 = x[k];
        }
    }
    p
}

fn solve_once(prob: &Problem, free: &[(usize, usize)], seed: u64) -> (Coords, f64) {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let nv = free.len();
    let mut x = vec![0.0; nv];
    for (k, &(pi, ax)) in free.iter().enumerate() {
        let pd = &prob.pts[pi];
        if let Some((ix, iy)) = pd.init {
            // hinted init, jittered per seed so distinct seeds explore distinct
            // placements (and the gauge freedom) rather than one fixed point
            let base = if ax == 0 { ix } else { iy };
            x[k] = base + (rng_next(&mut state) - 0.5) * 2.0;
        } else {
            x[k] = (rng_next(&mut state) - 0.5) * 6.0;
        }
    }
    let mut lam = 1e-3;
    for _ in 0..400 {
        let r = residuals(prob, &assemble(prob, free, &x));
        let cost: f64 = r.iter().map(|v| v * v).sum();
        if cost < 1e-26 {
            break;
        }
        // central-difference Jacobian
        let m = r.len();
        let mut jac = vec![vec![0.0; nv]; m];
        for k in 0..nv {
            let h = 1e-6;
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let rp = residuals(prob, &assemble(prob, free, &xp));
            let rm = residuals(prob, &assemble(prob, free, &xm));
            for row in 0..m {
                jac[row][k] = (rp[row] - rm[row]) / (2.0 * h);
            }
        }
        // A = J^T J + lam I ; g = J^T r
        let mut a = vec![vec![0.0; nv]; nv];
        let mut g = vec![0.0; nv];
        for col in 0..nv {
            for c2 in 0..nv {
                let mut s = 0.0;
                for row in 0..m {
                    s += jac[row][col] * jac[row][c2];
                }
                a[col][c2] = s;
            }
            a[col][col] += lam;
            let mut s = 0.0;
            for row in 0..m {
                s += jac[row][col] * r[row];
            }
            g[col] = s;
        }
        match solve_linear(a, g) {
            Some(step) => {
                let mut xn = x.clone();
                for k in 0..nv {
                    xn[k] -= step[k];
                }
                let rn = residuals(prob, &assemble(prob, free, &xn));
                let ncost: f64 = rn.iter().map(|v| v * v).sum();
                if ncost < cost {
                    x = xn;
                    lam = (lam * 0.5).max(1e-12);
                } else {
                    lam = (lam * 3.0).min(1e8);
                }
            }
            None => break,
        }
    }
    let p = assemble(prob, free, &x);
    let r = residuals(prob, &p);
    let cost: f64 = r.iter().map(|v| v * v).sum();
    (p, cost)
}

fn is_valid(prob: &Problem, p: &Coords, cost: f64) -> bool {
    if !cost.is_finite() || cost > 1e-18 {
        return false;
    }
    for v in p {
        if !v.0.is_finite() || !v.1.is_finite() {
            return false;
        }
    }
    for &(i, j, m) in &prob.nondeg {
        if dist(p, i, j) < m {
            return false;
        }
    }
    true
}

fn query(t: &Target, p: &Coords) -> f64 {
    match t {
        Target::Length(i, j) => dist(p, *i, *j),
        Target::RadiusComplement(a, b, k) => k - dist(p, *a, *b),
        Target::DiskAreaByRadiusCoord(c, ax) => coord(p, *c, *ax),
        Target::DiskArea(c, on) => dist(p, *c, *on),
        Target::PolyArea(ids) => {
            let mut a = 0.0;
            let n = ids.len();
            for k in 0..n {
                let i = ids[k];
                let j = ids[(k + 1) % n];
                a += p[i].0 * p[j].1 - p[j].0 * p[i].1;
            }
            (0.5 * a).abs()
        }
    }
}

fn mode(t: &Target) -> Mode {
    match t {
        Target::Length(..) | Target::RadiusComplement(..) => Mode::Length,
        Target::DiskAreaByRadiusCoord(..) | Target::DiskArea(..) => Mode::DiskArea,
        Target::PolyArea(..) => Mode::Length,
    }
}

// =====================================================================
// exact recovery: f64 -> Alg  (rational / sqrt(rational) / a + b*sqrt(d))
// =====================================================================

/// Best rational p/q with q <= maxden (continued-fraction convergents).
fn best_rational(x: f64, maxden: i64) -> (i64, i64) {
    let neg = x < 0.0;
    let mut v = x.abs();
    // continued fraction
    let (mut h0, mut h1) = (0i64, 1i64);
    let (mut k0, mut k1) = (1i64, 0i64);
    for _ in 0..40 {
        let a = v.floor();
        let ai = a as i64;
        let h2 = ai.checked_mul(h1).and_then(|m| m.checked_add(h0));
        let k2 = ai.checked_mul(k1).and_then(|m| m.checked_add(k0));
        let (h2, k2) = match (h2, k2) {
            (Some(h), Some(k)) => (h, k),
            _ => break,
        };
        if k2 > maxden {
            break;
        }
        h0 = h1;
        h1 = h2;
        k0 = k1;
        k1 = k2;
        let frac = v - a;
        if frac.abs() < 1e-12 {
            break;
        }
        v = 1.0 / frac;
    }
    let num = if neg { -h1 } else { h1 };
    (num, k1.max(1))
}

const SQUAREFREE: [i64; 20] = [
    2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33,
];

/// Recover an exact `Alg` from a numeric value, or `None` if no small-height
/// closed form fits. Candidates are validated against `Alg::to_f64` and require
/// bounded denominators, so noise cannot be fit.
fn recover_alg(x: f64, tol: f64) -> Option<Alg> {
    // 1. rational
    let (pn, pd) = best_rational(x, 64);
    if (pn as f64 / pd as f64 - x).abs() < tol {
        return Some(Alg::ratio(pn, pd));
    }
    // 2. k*sqrt(s): x^2 is rational
    let x2 = x * x;
    let (sn, sd) = best_rational(x2, 64);
    if sn >= 0 && (sn as f64 / sd as f64 - x2).abs() < tol {
        let s = alg_sqrt_ratio(sn, sd);
        let cand = if x >= 0.0 { s } else { s.neg() };
        if (cand.to_f64() - x).abs() < tol {
            return Some(cand);
        }
    }
    // 3. a + b*sqrt(d)
    for &d in SQUAREFREE.iter() {
        let sd = (d as f64).sqrt();
        for bden in 1..=4i64 {
            for bnum in -12..=12i64 {
                if bnum == 0 {
                    continue;
                }
                let b = bnum as f64 / bden as f64;
                let a = x - b * sd;
                let (an, ad) = best_rational(a, 32);
                if (an as f64 / ad as f64 - a).abs() < tol {
                    let surd = alg_sqrt_ratio(d, 1).mul(&Alg::ratio(bnum, bden));
                    let cand = Alg::ratio(an, ad).add(&surd);
                    if (cand.to_f64() - x).abs() < tol {
                        return Some(cand);
                    }
                }
            }
        }
    }
    None
}

// =====================================================================
// the solver / K-gate
// =====================================================================

const K_MIN: usize = 5;
const SPREAD_TOL: f64 = 1e-6;
const REC_TOL: f64 = 1e-9;

pub fn solve(prob: &Problem, target: &Target) -> Solved {
    let free = free_vars(prob);
    let mut vals: Vec<f64> = Vec::new();
    // enough seeds to clear the K-gate even with a few non-convergent starts
    for seed in 1..=80u64 {
        let (p, cost) = solve_once(prob, &free, seed);
        if is_valid(prob, &p, cost) {
            vals.push(query(target, &p));
        }
        if vals.len() >= 24 {
            break;
        }
    }
    if vals.is_empty() {
        return Solved::Refused(Refuse::PlacementFailed);
    }
    if vals.len() < K_MIN {
        // too few constraint-satisfying placements to trust invariance
        return Solved::Refused(Refuse::UnderDetermined);
    }
    let mut lo = vals[0];
    let mut hi = vals[0];
    let mut sum = 0.0;
    for &v in &vals {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v;
    }
    if hi - lo > SPREAD_TOL {
        // the query is NOT invariant across valid placements => under-determined
        return Solved::Refused(Refuse::UnderDetermined);
    }
    let mean = sum / vals.len() as f64;
    let base = match recover_alg(mean, REC_TOL) {
        Some(a) => a,
        None => return Solved::Refused(Refuse::NoClosedForm),
    };
    // re-verify: the exact value must match every valid placement
    let check = base.to_f64();
    for &v in &vals {
        let cmp = match mode(target) {
            Mode::Length => (check - v).abs(),
            Mode::DiskArea => (check - v).abs(),
        };
        if cmp > 1e-5 {
            return Solved::Refused(Refuse::NoClosedForm);
        }
    }
    match mode(target) {
        Mode::Length => {
            let text = fmt_alg(&base);
            Solved::Value { rat: base, pic: Alg::zero(), text }
        }
        Mode::DiskArea => {
            let pic = base.mul(&base); // pi * r^2
            let inner = fmt_alg(&pic);
            let text = if pic.primes().is_empty() {
                format!("{}*pi", inner)
            } else {
                format!("({})*pi", inner)
            };
            Solved::Value { rat: Alg::zero(), pic, text }
        }
    }
}

/// Independent exact cross-check for an integer-radius disk: rebuild it through
/// `arrangement` and confirm its exact `a + b*pi` area equals `pi * r^2`.
/// Returns None when the recovered centre/radius are not integers (the check
/// does not apply); Some(true/false) otherwise.
pub fn verify_disk_area_exact(cx: i64, cy: i64, r: i64, pic: &Alg) -> Option<bool> {
    use super::arrangement::{area_exact, build_checked, faces, Arc};
    let arc = Arc::cardinal(cx, cy, r, 0, 0, 0); // full circle (start==end)
    let arr = build_checked(&[], &[arc])?;
    // the bounded disk face; take the face whose area has the pi coefficient
    for (_ring, area) in faces(&arr) {
        if let Some((rat, pc)) = area_exact(&area, &arr.circ) {
            if rat.is_zero() && pc.eq(pic) {
                return Some(true);
            }
        }
    }
    Some(false)
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PIECE 2: polygon-area solve target ----------------------------
    // A determinate square of side sqrt(50) (side length is the GIVEN; the
    // AREA is DERIVED by the shoelace target + recovery, never fed as the
    // answer). Gauge freedoms (translate/rotate/reflect) leave the area
    // invariant, so the K-gate accepts it. Expect exact 50.
    #[test]
    fn polygon_area_target_derives_square_area_50() {
        let s = 50.0_f64.sqrt();
        let pts = vec![
            PointDef::fixed(0.0, 0.0),
            PointDef::hint(s, 0.0),
            PointDef::hint(s, s),
            PointDef::hint(0.0, s),
        ];
        let cons = vec![
            Con::Dist(0, 1, s),
            Con::DistEq(0, 1, 1, 2),
            Con::DistEq(1, 2, 2, 3),
            Con::DistEq(2, 3, 3, 0),
            Con::Perp(0, 1, 1, 2),
            Con::Perp(1, 2, 2, 3),
            Con::Perp(2, 3, 3, 0),
        ];
        let prob = Problem { pts, cons, nondeg: vec![(0, 2, 1.0)] };
        match solve(&prob, &Target::PolyArea(vec![0, 1, 2, 3])) {
            Solved::Value { rat, pic, text } => {
                assert!(pic.is_zero(), "area is a pure number, got pi-term in {}", text);
                assert!(rat.eq(&Alg::int(50)), "expected area 50, got {}", text);
            }
            Solved::Refused(r) => panic!("polygon-area target refused: {:?}", r),
        }
    }

    // A rectangle sqrt(8) x sqrt(6): area sqrt(48) = 4*sqrt(3). Proves the
    // target carries sqrt-terms EXACTLY (not just integers).
    #[test]
    fn polygon_area_target_derives_sqrt_area() {
        let (a, b) = (8.0_f64.sqrt(), 6.0_f64.sqrt());
        let pts = vec![
            PointDef::fixed(0.0, 0.0),
            PointDef::hint(a, 0.0),
            PointDef::hint(a, b),
            PointDef::hint(0.0, b),
        ];
        let cons = vec![
            Con::Dist(0, 1, a),
            Con::Dist(1, 2, b),
            Con::Perp(0, 1, 1, 2),
            Con::DistEq(0, 1, 2, 3),
            Con::DistEq(1, 2, 3, 0),
            Con::Perp(1, 2, 2, 3),
        ];
        let prob = Problem { pts, cons, nondeg: vec![(0, 2, 1.0)] };
        match solve(&prob, &Target::PolyArea(vec![0, 1, 2, 3])) {
            Solved::Value { rat, pic, text } => {
                assert!(pic.is_zero(), "area is a pure number, got {}", text);
                let want = 48.0_f64.sqrt();
                assert!((rat.to_f64() - want).abs() < 1e-6, "expected sqrt(48)=4*sqrt(3), got {}", text);
                // exact: must equal the recovered algebraic form of sqrt(48)
                assert!(rat.eq(&recover_alg(want, REC_TOL).unwrap()), "not exact sqrt form: {}", text);
                assert!(!rat.primes().is_empty(), "expected a sqrt-bearing area, got rational {}", text);
            }
            Solved::Refused(r) => panic!("sqrt polygon-area target refused: {:?}", r),
        }
    }

    // ---- recovery unit tests -------------------------------------------
    #[test]
    fn recover_forms() {
        assert!(recover_alg(2.0, REC_TOL).unwrap().eq(&Alg::int(2)));
        assert!(recover_alg(0.75, REC_TOL).unwrap().eq(&Alg::ratio(3, 4)));
        assert!(recover_alg(2.0_f64.sqrt(), REC_TOL).unwrap().eq(&alg_sqrt_ratio(2, 1)));
        // 2*sqrt(3) = sqrt(12)
        let two_root3 = alg_sqrt_ratio(3, 1).mul(&Alg::int(2));
        assert!(recover_alg(2.0 * 3.0_f64.sqrt(), REC_TOL).unwrap().eq(&two_root3));
        // 2*sqrt(3) - 3
        let target = alg_sqrt_ratio(3, 1).mul(&Alg::int(2)).sub(&Alg::int(3));
        assert!(recover_alg(2.0 * 3.0_f64.sqrt() - 3.0, REC_TOL).unwrap().eq(&target));
        // a genuinely transcendental value must NOT be "recovered"
        assert!(recover_alg(2.0 * 20.0_f64.to_radians().sin(), 1e-9).is_none());
    }

    // ---- figure builders (constraints only; no answer baked in) --------

    /// John's family: outer square, inscribed tilted square of area `sq_area`,
    /// corner right-triangle of area `tri_area`, circle inscribed in that
    /// corner. Derive the inscribed circle's area.
    ///
    /// Points: 0=O(origin), 1=R(bottom-right, y=0, x=S free), 2=T1(bottom),
    /// 3=T2(right), 4=T3(top), 5=T4(left), 6=C(incircle centre).
    fn tilted_square_incircle(sq_area: f64, tri_area: f64) -> (Problem, Target) {
        // scale-based STARTING HINTS only (aid LM convergence; the answer is
        // still derived by the solver + recovery, never read from these).
        let k = (sq_area / 100.0).sqrt();
        let pts = vec![
            PointDef::fixed(0.0, 0.0),               // 0 O
            PointDef { fx: None, fy: Some(0.0), init: Some((14.0 * k, 0.0)) }, // 1 R (y=0)
            PointDef { fx: None, fy: Some(0.0), init: Some((8.0 * k, 0.0)) },  // 2 T1 (y=0)
            PointDef::hint(14.0 * k, 8.0 * k),       // 3 T2 on right
            PointDef::hint(6.0 * k, 14.0 * k),       // 4 T3 on top
            PointDef { fx: Some(0.0), fy: None, init: Some((0.0, 6.0 * k)) },  // 5 T4 (x=0)
            PointDef::hint(2.0 * k, 2.0 * k),        // 6 C incircle centre
        ];
        let cons = vec![
            // outer square side S = R.x; right side x=S, top side y=S
            Con::CoordEq(3, 0, 1, 0), // T2.x = R.x
            Con::CoordEq(4, 1, 1, 0), // T3.y = R.x (top height = S)
            // T1T2T3T4 is a square: equal consecutive sides + one right angle
            Con::DistEq(2, 3, 3, 4),
            Con::DistEq(3, 4, 4, 5),
            Con::DistEq(4, 5, 5, 2),
            Con::Perp(2, 3, 3, 4),
            // tilted square area
            Con::PolyArea(vec![2, 3, 4, 5], sq_area),
            // corner right-triangle O,T1,T4 area
            Con::TriArea(0, 2, 5, tri_area),
            // incircle: centre equidistant from both axes (C.x = C.y = r) and
            // tangent to the hypotenuse T1T4
            Con::CoordEq(6, 0, 6, 1),
            Con::PtLineDistEqCoord(6, 2, 5, 6, 1),
        ];
        let prob = Problem { pts, cons, nondeg: vec![(2, 5, 1.0)] };
        (prob, Target::DiskAreaByRadiusCoord(6, 1)) // radius = C.y
    }

    /// Three equal mutually-tangent circles inside a unit circle (Descartes).
    /// Outer radius R=1 (baked into the constants). Points: 0=O(origin, fixed),
    /// 1,2,3 = inner centres. Derive the inner radius r.
    fn three_tangent_in_circle(with_link: bool) -> (Problem, Target) {
        let pts = vec![
            PointDef::fixed(0.0, 0.0),   // 0 O
            PointDef::hint(0.55, 0.0),   // 1 C1
            PointDef::hint(-0.27, 0.47), // 2 C2
            PointDef::hint(-0.27, -0.47),// 3 C3
        ];
        let mut cons = vec![
            // equal distance from centre (concyclic inner centres)
            Con::DistEq(0, 1, 0, 2),
            Con::DistEq(0, 2, 0, 3),
            // equal mutual distances (equilateral => 120 deg apart)
            Con::DistEq(1, 2, 2, 3),
            Con::DistEq(2, 3, 3, 1),
        ];
        if with_link {
            // tangency link: |C1C2| = 2r and |OC1| = 1 - r  =>  |C1C2| + 2|OC1| = 2
            cons.push(Con::LinDist(vec![(1.0, 1, 2), (2.0, 0, 1)], 2.0));
        }
        let prob = Problem {
            pts,
            cons,
            nondeg: vec![(1, 2, 0.2), (2, 3, 0.2), (3, 1, 0.2)],
        };
        // r = 1 - |O C1|
        (prob, Target::RadiusComplement(0, 1, 1.0))
    }

    /// Chord of a unit circle subtending 40 deg => length 2*sin(20 deg): a
    /// determinate figure whose answer has NO closed algebraic form.
    fn unit_chord_40deg() -> (Problem, Target) {
        let pts = vec![
            PointDef::fixed(0.0, 0.0), // O
            PointDef::hint(1.0, 0.1),  // A
            PointDef::hint(0.7, 0.7),  // B
        ];
        let cons = vec![
            Con::Dist(0, 1, 1.0),
            Con::Dist(0, 2, 1.0),
            Con::Angle(1, 0, 2, 40.0),
        ];
        let prob = Problem { pts, cons, nondeg: vec![(1, 2, 0.2)] };
        (prob, Target::Length(1, 2))
    }

    // ---- ACCEPTANCE 1 --------------------------------------------------
    #[test]
    fn acceptance1_tilted_square_incircle() {
        let (prob, target) = tilted_square_incircle(100.0, 24.0);
        match solve(&prob, &target) {
            Solved::Value { rat, pic, text } => {
                // derived from constraints: area = 4*pi (r = 2)
                assert!(rat.is_zero(), "area should be pure pi, got rat={}", fmt_alg(&rat));
                assert!(pic.eq(&Alg::int(4)), "expected 4*pi, got {}", text);
                assert_eq!(text, "4*pi");
                // independent exact cross-check via arrangement: disk of radius 2
                // at centre (2,2) has area 4*pi
                assert_ne!(verify_disk_area_exact(2, 2, 2, &pic), Some(false));
            }
            Solved::Refused(r) => panic!("ACC1 should solve, refused {:?}", r),
        }
    }

    // ---- ACCEPTANCE 2 --------------------------------------------------
    #[test]
    fn acceptance2_descartes_three_tangent() {
        let (prob, target) = three_tangent_in_circle(true);
        match solve(&prob, &target) {
            Solved::Value { rat, pic, text } => {
                // r = 2*sqrt(3) - 3  (carries sqrt(3); a+b*pi cannot hold it)
                let expected = alg_sqrt_ratio(3, 1).mul(&Alg::int(2)).sub(&Alg::int(3));
                assert!(rat.eq(&expected), "expected 2*sqrt(3)-3, got {}", text);
                assert!(pic.is_zero());
                assert!(!rat.primes().is_empty(), "answer must carry a radical");
                assert_eq!(rat.primes().iter().cloned().collect::<Vec<_>>(), vec![3]);
            }
            Solved::Refused(r) => panic!("ACC2 should solve, refused {:?}", r),
        }
    }

    // ---- ANTI-CHEAT: novel instance (rescale) --------------------------
    #[test]
    fn novel_instance_rescale() {
        // scale linear by 3/2 => areas * 9/4: 100->225, 24->54; answer 4pi->9pi
        let (prob, target) = tilted_square_incircle(225.0, 54.0);
        match solve(&prob, &target) {
            Solved::Value { pic, text, .. } => {
                assert!(pic.eq(&Alg::int(9)), "expected 9*pi, got {}", text);
                assert!(!pic.eq(&Alg::int(4)), "must not return the ACC1 constant");
            }
            Solved::Refused(r) => panic!("rescale should solve, refused {:?}", r),
        }
    }

    // ---- HONESTY: drop a constraint => under-determined => refuse ------
    #[test]
    fn drop_constraint_refuses() {
        let (prob, target) = three_tangent_in_circle(false); // no tangency link
        match solve(&prob, &target) {
            Solved::Refused(Refuse::UnderDetermined) => {}
            Solved::Refused(other) => panic!("expected UnderDetermined, got {:?}", other),
            Solved::Value { text, .. } => panic!("must refuse under-determined, emitted {}", text),
        }
    }

    // ---- HONESTY: determinate but non-closed => refuse ----------------
    #[test]
    fn non_closed_form_refuses() {
        let (prob, target) = unit_chord_40deg();
        match solve(&prob, &target) {
            Solved::Refused(Refuse::NoClosedForm) => {}
            Solved::Refused(other) => panic!("expected NoClosedForm, got {:?}", other),
            Solved::Value { text, .. } => panic!("2*sin(20) has no closed form, emitted {}", text),
        }
    }
}
