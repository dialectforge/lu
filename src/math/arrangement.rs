//! Exact planar arrangement + Green's-theorem areas — Rust port of the Phase-3
//! prototype `geom_exact.py`, on top of the exact field in `algnum`.
//!
//! Everything that decides topology or a reported number is exact: intersections
//! emit exact algebraic coordinates, the face walk orders edges by exact tangent
//! with a signed-curvature tie-break and a branch-cut winding term, and areas are
//! a + b*pi with the sector angle kept symbolic (angle atoms) and resolved via
//! Niven's theorem. A region whose closed form does not exist is flagged, never
//! approximated.

use crate::math::algnum::{isqrt_field, Alg};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub type Pt = (Alg, Alg);

fn z() -> Alg {
    Alg::zero()
}

// --- exact 2D vector helpers ---
fn vsub(a: &Pt, b: &Pt) -> Pt {
    (a.0.sub(&b.0), a.1.sub(&b.1))
}
fn vdot(a: &Pt, b: &Pt) -> Alg {
    a.0.mul(&b.0).add(&a.1.mul(&b.1))
}
fn vcross(a: &Pt, b: &Pt) -> Alg {
    a.0.mul(&b.1).sub(&a.1.mul(&b.0))
}

fn half(w: &Pt) -> i32 {
    let sy = w.1.sign();
    if sy > 0 {
        0
    } else if sy < 0 {
        1
    } else if w.0.sign() > 0 {
        0
    } else {
        1
    }
}
/// Compare two nonzero vectors by CCW angle in [0,2pi).
fn angle_cmp(u: &Pt, v: &Pt) -> Ordering {
    let (hu, hv) = (half(u), half(v));
    if hu != hv {
        return hu.cmp(&hv);
    }
    match vcross(u, v).sign() {
        x if x > 0 => Ordering::Less,
        x if x < 0 => Ordering::Greater,
        _ => Ordering::Equal,
    }
}
fn rot_conj(w: &Pt, r: &Pt) -> Pt {
    (
        w.0.mul(&r.0).add(&w.1.mul(&r.1)),
        w.1.mul(&r.0).sub(&w.0.mul(&r.1)),
    )
}

fn sqrt_exact(a: &Alg) -> Option<Alg> {
    isqrt_field(a)
}

// =====================================================================
// primitives
// =====================================================================

#[derive(Clone)]
pub struct Seg {
    pub a: Pt,
    pub b: Pt,
}
impl Seg {
    pub fn new(ax: Alg, ay: Alg, bx: Alg, by: Alg) -> Seg {
        Seg {
            a: (ax, ay),
            b: (bx, by),
        }
    }
    pub fn int(ax: i64, ay: i64, bx: i64, by: i64) -> Seg {
        Seg::new(Alg::int(ax), Alg::int(ay), Alg::int(bx), Alg::int(by))
    }
}

#[derive(Clone)]
pub struct Arc {
    pub o: Pt,
    pub rr: Alg,
    pub p0: Pt,
    pub p1: Pt,
    pub full: bool,
    pub cid: usize,
}
impl Arc {
    fn pt_deg(cx: &Alg, cy: &Alg, r: &Alg, deg: i64) -> Pt {
        let d = ((deg % 360) + 360) % 360;
        match d {
            0 => (cx.add(r), cy.clone()),
            90 => (cx.clone(), cy.add(r)),
            180 => (cx.sub(r), cy.clone()),
            270 => (cx.clone(), cy.sub(r)),
            _ => panic!("non-cardinal degree {}", d),
        }
    }
    pub fn cardinal(cx: i64, cy: i64, r: i64, start: i64, end: i64, cid: usize) -> Arc {
        let (cxa, cya, ra) = (Alg::int(cx), Alg::int(cy), Alg::int(r));
        let p0 = Arc::pt_deg(&cxa, &cya, &ra, start);
        let p1 = Arc::pt_deg(&cxa, &cya, &ra, end);
        let full = p0 == p1;
        Arc {
            o: (cxa, cya),
            rr: Alg::int(r * r),
            p0,
            p1,
            full,
            cid,
        }
    }
    /// Is P (assumed on the circle) within the CCW span [p0, p1]?
    fn span_le(&self, p: &Pt) -> bool {
        let s = vsub(&self.p0, &self.o);
        let e = vsub(&self.p1, &self.o);
        let w = vsub(p, &self.o);
        if e == s {
            return true; // full circle
        }
        let ep = rot_conj(&e, &s);
        let wp = rot_conj(&w, &s);
        angle_cmp(&wp, &ep) != Ordering::Greater
    }
}

// =====================================================================
// exact intersections
// =====================================================================

pub fn on_circle(arc: &Arc, p: &Pt) -> bool {
    let d = vsub(p, &arc.o);
    vdot(&d, &d).sub(&arc.rr).is_zero()
}

fn t_in_unit(t: &Alg) -> bool {
    t.sign() >= 0 && t.sub(&Alg::int(1)).sign() <= 0
}

pub fn inter_seg_seg(s1: &Seg, s2: &Seg) -> Vec<Pt> {
    let ((x1, y1), (x2, y2)) = (&s1.a, &s1.b);
    let ((x3, y3), (x4, y4)) = (&s2.a, &s2.b);
    let den = x1.sub(x2).mul(&y3.sub(y4)).sub(&y1.sub(y2).mul(&x3.sub(x4)));
    if den.is_zero() {
        return vec![];
    }
    let t = x1.sub(x3).mul(&y3.sub(y4)).sub(&y1.sub(y3).mul(&x3.sub(x4))).div(&den);
    let u = x1.sub(x3).mul(&y1.sub(y2)).sub(&y1.sub(y3).mul(&x1.sub(x2))).div(&den);
    if !t_in_unit(&t) || !t_in_unit(&u) {
        return vec![];
    }
    vec![(x1.add(&t.mul(&x2.sub(x1))), y1.add(&t.mul(&y2.sub(y1))))]
}

pub fn inter_arc_seg(arc: &Arc, seg: &Seg) -> Option<Vec<Pt>> {
    let ((x1, y1), (x2, y2)) = (&seg.a, &seg.b);
    let (ox, oy) = &arc.o;
    let dx = x2.sub(x1);
    let dy = y2.sub(y1);
    let fx = x1.sub(ox);
    let fy = y1.sub(oy);
    let aa = dx.mul(&dx).add(&dy.mul(&dy));
    let bb = Alg::int(2).mul(&fx.mul(&dx).add(&fy.mul(&dy)));
    let cc = fx.mul(&fx).add(&fy.mul(&fy)).sub(&arc.rr);
    if aa.is_zero() {
        return Some(vec![]);
    }
    let disc = bb.mul(&bb).sub(&Alg::int(4).mul(&aa).mul(&cc));
    let sd = disc.sign();
    if sd < 0 {
        return Some(vec![]);
    }
    let root = sqrt_exact(&disc)?;
    let two_a = Alg::int(2).mul(&aa);
    let ts = if sd == 0 {
        vec![bb.neg().sub(&root).div(&two_a)]
    } else {
        vec![
            bb.neg().sub(&root).div(&two_a),
            bb.neg().add(&root).div(&two_a),
        ]
    };
    let mut out = vec![];
    for t in ts {
        if !t_in_unit(&t) {
            continue;
        }
        let p = (x1.add(&t.mul(&dx)), y1.add(&t.mul(&dy)));
        if arc.span_le(&p) {
            out.push(p);
        }
    }
    Some(out)
}

pub fn inter_arc_arc(a1: &Arc, a2: &Arc) -> Option<Vec<Pt>> {
    let (ox1, oy1) = &a1.o;
    let (ox2, oy2) = &a2.o;
    let ax = Alg::int(2).mul(&ox2.sub(ox1));
    let ay = Alg::int(2).mul(&oy2.sub(oy1));
    let k = ox2.mul(ox2).add(&oy2.mul(oy2)).sub(&a2.rr)
        .sub(&ox1.mul(ox1).add(&oy1.mul(oy1)).sub(&a1.rr));
    if ax.is_zero() && ay.is_zero() {
        return Some(vec![]);
    }
    let mut pts: Vec<Pt> = vec![];
    if !ay.is_zero() {
        let m = ax.neg().div(&ay);
        let b = k.div(&ay);
        let (fx, fy) = (ox1.clone(), oy1.clone());
        let qa = Alg::int(1).add(&m.mul(&m));
        let qb = Alg::int(2).mul(&m.mul(&b.sub(&fy)).sub(&fx));
        let qc = fx.mul(&fx).add(&b.sub(&fy).mul(&b.sub(&fy))).sub(&a1.rr);
        let disc = qb.mul(&qb).sub(&Alg::int(4).mul(&qa).mul(&qc));
        if disc.sign() < 0 {
            return Some(vec![]);
        }
        let root = sqrt_exact(&disc)?;
        let two_qa = Alg::int(2).mul(&qa);
        let xs = if disc.is_zero() {
            vec![qb.neg().sub(&root).div(&two_qa)]
        } else {
            vec![
                qb.neg().sub(&root).div(&two_qa),
                qb.neg().add(&root).div(&two_qa),
            ]
        };
        for x in xs {
            let y = m.mul(&x).add(&b);
            pts.push((x, y));
        }
    } else {
        let x = k.div(&ax);
        let (fx, fy) = (ox1.clone(), oy1.clone());
        let qb = Alg::int(-2).mul(&fy);
        let qc = x.sub(&fx).mul(&x.sub(&fx)).add(&fy.mul(&fy)).sub(&a1.rr);
        let disc = qb.mul(&qb).sub(&Alg::int(4).mul(&qc));
        if disc.sign() < 0 {
            return Some(vec![]);
        }
        let root = sqrt_exact(&disc)?;
        let two = Alg::int(2);
        let ys = if disc.is_zero() {
            vec![qb.neg().sub(&root).div(&two)]
        } else {
            vec![qb.neg().sub(&root).div(&two), qb.neg().add(&root).div(&two)]
        };
        for y in ys {
            pts.push((x.clone(), y));
        }
    }
    let mut out = vec![];
    for p in pts {
        if a1.span_le(&p) && a2.span_le(&p) {
            out.push(p);
        }
    }
    Some(out)
}

// =====================================================================
// exact symbolic area
// =====================================================================

pub type AtomKey = (usize, Alg, Alg);

#[derive(Clone)]
pub struct Area {
    pub rat: Alg,
    pub pic: Alg,
    pub atoms: BTreeMap<AtomKey, Alg>,
}
impl Area {
    pub fn zero() -> Area {
        Area {
            rat: z(),
            pic: z(),
            atoms: BTreeMap::new(),
        }
    }
    pub fn add(&self, o: &Area) -> Area {
        let mut r = self.clone();
        r.rat = r.rat.add(&o.rat);
        r.pic = r.pic.add(&o.pic);
        for (k, v) in &o.atoms {
            let e = r.atoms.entry(k.clone()).or_insert_with(z);
            *e = e.add(v);
            if e.is_zero() {
                r.atoms.remove(k);
            }
        }
        r
    }
    fn add_atom(&mut self, key: AtomKey, coeff: Alg) {
        let e = self.atoms.entry(key.clone()).or_insert_with(z);
        *e = e.add(&coeff);
        if e.is_zero() {
            self.atoms.remove(&key);
        }
    }
}

type Circles = BTreeMap<usize, (Alg, Alg, Alg)>;

fn classify_angle(c: &Alg, s: &Alg, rr: &Alg) -> Option<Alg> {
    let ratio = c.mul(c).div(rr);
    if !ratio.primes().is_empty() {
        return None;
    }
    let (sc, ss) = (c.sign(), s.sign());
    let r = |n: i64, d: i64| Some(Alg::ratio(n, d));
    if ratio.eq(&Alg::int(1)) {
        return if sc > 0 { r(0, 1) } else { r(1, 1) };
    }
    if ratio.is_zero() {
        return if ss > 0 { r(1, 2) } else { r(3, 2) };
    }
    if ratio.eq(&Alg::ratio(1, 4)) {
        return if sc > 0 {
            if ss > 0 { r(1, 3) } else { r(5, 3) }
        } else if ss > 0 { r(2, 3) } else { r(4, 3) };
    }
    if ratio.eq(&Alg::ratio(1, 2)) {
        return if sc > 0 {
            if ss > 0 { r(1, 4) } else { r(7, 4) }
        } else if ss > 0 { r(3, 4) } else { r(5, 4) };
    }
    if ratio.eq(&Alg::ratio(3, 4)) {
        return if sc > 0 {
            if ss > 0 { r(1, 6) } else { r(11, 6) }
        } else if ss > 0 { r(5, 6) } else { r(7, 6) };
    }
    None
}

/// (rat, pic, residual_count). residual == 0 means a clean closed form.
pub fn resolve(area: &Area, circ: &Circles) -> (Alg, Alg, usize) {
    let mut pic = area.pic.clone();
    let mut residual = 0;
    for ((cid, px, py), coeff) in &area.atoms {
        let (cx, cy, rr) = &circ[cid];
        let c = px.sub(cx);
        let s = py.sub(cy);
        match classify_angle(&c, &s, rr) {
            Some(unit) => pic = pic.add(&coeff.mul(&unit)),
            None => residual += 1,
        }
    }
    (area.rat.clone(), pic, residual)
}

pub fn is_closed(area: &Area, circ: &Circles) -> bool {
    resolve(area, circ).2 == 0
}

pub fn area_f64(area: &Area, circ: &Circles) -> f64 {
    let (rat, pic, residual) = resolve(area, circ);
    if residual > 0 {
        return f64::NAN;
    }
    rat.to_f64() + pic.to_f64() * std::f64::consts::PI
}

fn rs(x: &Alg) -> String {
    // x is rational Alg
    if x.is_zero() {
        return "0".into();
    }
    let q = &x.t[&BTreeSet::new()];
    if q.denom() == &num_bigint::BigInt::from(1) {
        format!("{}", q.numer())
    } else {
        format!("{}/{}", q.numer(), q.denom())
    }
}

pub fn fmt(area: &Area, circ: &Circles) -> String {
    let (a, b, residual) = resolve(area, circ);
    if residual > 0 {
        return format!("<non-closed: {} residual atom(s)>", residual);
    }
    let arat = a.primes().is_empty();
    let brat = b.primes().is_empty();
    if arat && brat {
        if b.is_zero() {
            return rs(&a);
        }
        if a.is_zero() {
            return format!("{}*pi", rs(&b));
        }
        if a.sign() < 0 && b.sign() > 0 {
            return format!("{}*pi - {}", rs(&b), rs(&a.neg()));
        }
        let sign = if b.sign() < 0 { "-" } else { "+" };
        let babs = if b.sign() > 0 { b.clone() } else { b.neg() };
        return format!("{} {} {}*pi", rs(&a), sign, rs(&babs));
    }
    let mut parts = vec![];
    if !a.is_zero() {
        parts.push(format!("{:?}", a));
    }
    if !b.is_zero() {
        parts.push(format!("({:?})*pi", b));
    }
    parts.join(" + ")
}

// =====================================================================
// half-edge arrangement
// =====================================================================

#[derive(Clone)]
pub struct HE {
    pub u: usize,
    pub v: usize,
    pub area: Area,
    pub twin: usize,
    pub idx: usize,
    pub is_arc: bool,
    pub cid: usize,
    pub o: Pt,
    pub rr: Alg,
    pub tail: Pt,
    pub head: Pt,
    pub ccw: bool,
    pub tangent: Pt,
    pub kappa: i32,
    pub rr_tie: Alg,
}

fn sort_along_arc(arc: &Arc, pts: Vec<Pt>) -> Vec<Pt> {
    let s = vsub(&arc.p0, &arc.o);
    let mut keyed: Vec<(Pt, Pt)> = pts
        .into_iter()
        .map(|p| (rot_conj(&vsub(&p, &arc.o), &s), p))
        .collect();
    keyed.sort_by(|a, b| angle_cmp(&a.0, &b.0));
    keyed.into_iter().map(|(_, p)| p).collect()
}

pub struct Arrangement {
    pub nodes: Vec<Pt>,
    pub he: Vec<HE>,
    pub circ: Circles,
}

pub fn build(segs: &[Seg], arcs: &[Arc]) -> Arrangement {
    build_inner(segs, arcs).0
}

/// IDK-gate aware build: None if any intersection needed a radical outside the
/// field (so the engine refuses rather than silently dropping a crossing).
pub fn build_checked(segs: &[Seg], arcs: &[Arc]) -> Option<Arrangement> {
    let (arr, needs_nested) = build_inner(segs, arcs);
    if needs_nested {
        None
    } else {
        Some(arr)
    }
}

fn build_inner(segs: &[Seg], arcs: &[Arc]) -> (Arrangement, bool) {
    let mut needs_nested = false;
    let mut nodes: Vec<Pt> = vec![];
    let mut index: BTreeMap<Pt, usize> = BTreeMap::new();
    let mut circ: Circles = BTreeMap::new();
    for a in arcs {
        circ.insert(a.cid, (a.o.0.clone(), a.o.1.clone(), a.rr.clone()));
    }
    let mut nid = |p: &Pt, nodes: &mut Vec<Pt>, index: &mut BTreeMap<Pt, usize>| -> usize {
        if let Some(&i) = index.get(p) {
            i
        } else {
            let i = nodes.len();
            nodes.push(p.clone());
            index.insert(p.clone(), i);
            i
        }
    };

    let mut seg_nodes: Vec<BTreeSet<Pt>> = segs.iter().map(|_| BTreeSet::new()).collect();
    let mut arc_nodes: Vec<BTreeSet<Pt>> = arcs.iter().map(|_| BTreeSet::new()).collect();
    for (i, a) in arcs.iter().enumerate() {
        arc_nodes[i].insert(a.p0.clone());
        arc_nodes[i].insert(a.p1.clone());
    }
    for (i, s) in segs.iter().enumerate() {
        seg_nodes[i].insert(s.a.clone());
        seg_nodes[i].insert(s.b.clone());
    }
    for i in 0..arcs.len() {
        for j in (i + 1)..arcs.len() {
            match inter_arc_arc(&arcs[i], &arcs[j]) {
                Some(ps) => {
                    for p in ps {
                        arc_nodes[i].insert(p.clone());
                        arc_nodes[j].insert(p);
                    }
                }
                None => needs_nested = true,
            }
        }
    }
    for i in 0..arcs.len() {
        for j in 0..segs.len() {
            match inter_arc_seg(&arcs[i], &segs[j]) {
                Some(ps) => {
                    for p in ps {
                        arc_nodes[i].insert(p.clone());
                        seg_nodes[j].insert(p);
                    }
                }
                None => needs_nested = true,
            }
        }
    }
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            for p in inter_seg_seg(&segs[i], &segs[j]) {
                seg_nodes[i].insert(p.clone());
                seg_nodes[j].insert(p);
            }
        }
    }

    let mut he: Vec<HE> = vec![];
    let half = Alg::ratio(1, 2);

    // segment edges
    for (i, s) in segs.iter().enumerate() {
        let (ax, ay) = &s.a;
        let (bx, by) = &s.b;
        let dxs = bx.sub(ax);
        let dys = by.sub(ay);
        let param = |p: &Pt| -> Alg {
            if !dxs.is_zero() {
                p.0.sub(ax).div(&dxs)
            } else {
                p.1.sub(ay).div(&dys)
            }
        };
        let mut pts: Vec<Pt> = seg_nodes[i].iter().cloned().collect();
        pts.sort_by(|p, q| {
            let d = param(p).sub(&param(q)).sign();
            d.cmp(&0)
        });
        for k in 0..pts.len().saturating_sub(1) {
            let p = pts[k].clone();
            let q = pts[k + 1].clone();
            let uid = nid(&p, &mut nodes, &mut index);
            let vid = nid(&q, &mut nodes, &mut index);
            let tang_uv = vsub(&q, &p);
            let tang_vu = vsub(&p, &q);
            let mut a1 = Area::zero();
            a1.rat = p.0.mul(&q.1).sub(&q.0.mul(&p.1)).mul(&half);
            let mut a2 = Area::zero();
            a2.rat = q.0.mul(&p.1).sub(&p.0.mul(&q.1)).mul(&half);
            let i1 = he.len();
            let i2 = i1 + 1;
            he.push(HE {
                u: uid, v: vid, area: a1, twin: i2, idx: i1, is_arc: false, cid: 0,
                o: (z(), z()), rr: z(), tail: p.clone(), head: q.clone(), ccw: true,
                tangent: tang_uv, kappa: 0, rr_tie: z(),
            });
            he.push(HE {
                u: vid, v: uid, area: a2, twin: i1, idx: i2, is_arc: false, cid: 0,
                o: (z(), z()), rr: z(), tail: q.clone(), head: p.clone(), ccw: true,
                tangent: tang_vu, kappa: 0, rr_tie: z(),
            });
        }
    }

    // arc edges
    for (i, a) in arcs.iter().enumerate() {
        if a.full && arc_nodes[i].len() < 2 {
            let anti = (
                Alg::int(2).mul(&a.o.0).sub(&a.p0.0),
                Alg::int(2).mul(&a.o.1).sub(&a.p0.1),
            );
            arc_nodes[i].insert(anti);
        }
        let pa = sort_along_arc(a, arc_nodes[i].iter().cloned().collect());
        let m = pa.len();
        let npairs = if a.full { m } else { m.saturating_sub(1) };
        let (ox, oy) = (&a.o.0, &a.o.1);
        let half_r = a.rr.mul(&half);
        for k in 0..npairs {
            let p = pa[k].clone();
            let q = pa[(k + 1) % m].clone();
            if p == q {
                continue;
            }
            let uid = nid(&p, &mut nodes, &mut index);
            let vid = nid(&q, &mut nodes, &mut index);
            let key_p = (a.cid, p.0.clone(), p.1.clone());
            let key_q = (a.cid, q.0.clone(), q.1.clone());
            let wrap =
                angle_cmp(&vsub(&q, &a.o), &vsub(&p, &a.o)) == Ordering::Less;
            // ccw half-edge p->q
            let drift1 = ox.mul(&q.1.sub(&p.1)).add(&oy.mul(&p.0.sub(&q.0))).mul(&half);
            let mut a1 = Area::zero();
            a1.rat = drift1;
            a1.add_atom(key_q.clone(), half_r.clone());
            a1.add_atom(key_p.clone(), half_r.neg());
            if wrap {
                a1.pic = a1.pic.add(&a.rr);
            }
            let tan1 = (p.1.sub(oy).neg(), p.0.sub(ox));
            // cw twin q->p
            let drift2 = ox.mul(&p.1.sub(&q.1)).add(&oy.mul(&q.0.sub(&p.0))).mul(&half);
            let mut a2 = Area::zero();
            a2.rat = drift2;
            a2.add_atom(key_p.clone(), half_r.clone());
            a2.add_atom(key_q.clone(), half_r.neg());
            if wrap {
                a2.pic = a2.pic.sub(&a.rr);
            }
            let tan2 = (q.1.sub(oy), q.0.sub(ox).neg());
            let i1 = he.len();
            let i2 = i1 + 1;
            he.push(HE {
                u: uid, v: vid, area: a1, twin: i2, idx: i1, is_arc: true, cid: a.cid,
                o: a.o.clone(), rr: a.rr.clone(), tail: p.clone(), head: q.clone(),
                ccw: true, tangent: tan1, kappa: 1, rr_tie: a.rr.clone(),
            });
            he.push(HE {
                u: vid, v: uid, area: a2, twin: i1, idx: i2, is_arc: true, cid: a.cid,
                o: a.o.clone(), rr: a.rr.clone(), tail: q.clone(), head: p.clone(),
                ccw: false, tangent: tan2, kappa: -1, rr_tie: a.rr.clone(),
            });
        }
    }

    (Arrangement { nodes, he, circ }, needs_nested)
}

fn edge_cmp(h1: &HE, h2: &HE) -> Ordering {
    let c = angle_cmp(&h1.tangent, &h2.tangent);
    if c != Ordering::Equal {
        return c;
    }
    if h1.kappa != h2.kappa {
        return h1.kappa.cmp(&h2.kappa);
    }
    if h1.kappa == 0 {
        return Ordering::Equal;
    }
    let dr = h1.rr_tie.sub(&h2.rr_tie).sign();
    if dr == 0 {
        return Ordering::Equal;
    }
    if h1.kappa > 0 {
        if dr > 0 { Ordering::Less } else { Ordering::Greater }
    } else if dr < 0 {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

pub fn faces(arr: &Arrangement) -> Vec<(Vec<usize>, Area)> {
    let he = &arr.he;
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for h in he {
        out.entry(h.u).or_default().push(h.idx);
    }
    for v in out.values_mut() {
        v.sort_by(|&a, &b| edge_cmp(&he[a], &he[b]));
    }
    let mut posn: BTreeMap<usize, usize> = BTreeMap::new();
    for lst in out.values() {
        for (k, &idx) in lst.iter().enumerate() {
            posn.insert(idx, k);
        }
    }
    let nxt = |h: &HE| -> usize {
        let g = h.twin;
        let target = he[g].u; // == h.v
        let lst = &out[&target];
        let k = posn[&g];
        lst[(k + lst.len() - 1) % lst.len()]
    };
    let mut seen = vec![false; he.len()];
    let mut res = vec![];
    for h0 in 0..he.len() {
        if seen[h0] {
            continue;
        }
        let mut loop_ids = vec![];
        let mut h = h0;
        while !seen[h] {
            seen[h] = true;
            loop_ids.push(h);
            h = nxt(&he[h]);
        }
        let mut area = Area::zero();
        for &e in &loop_ids {
            area = area.add(&he[e].area);
        }
        res.push((loop_ids, area));
    }
    res
}

// =====================================================================
// exact point-in-face (generic rational ray)
// =====================================================================

fn ray_slopes() -> Vec<Alg> {
    vec![
        Alg::int(0), Alg::int(1), Alg::int(-1), Alg::int(2), Alg::ratio(1, 2),
        Alg::int(-2), Alg::ratio(-1, 2), Alg::int(3), Alg::ratio(1, 3),
        Alg::ratio(3, 2), Alg::ratio(-3, 2), Alg::ratio(5, 7), Alg::ratio(7, 5),
    ]
}

fn ray_generic(qx: &Alg, qy: &Alg, m: &Alg, loop_ids: &[usize], arr: &Arrangement) -> bool {
    for &hi in loop_ids {
        let h = &arr.he[hi];
        for vtx in [&h.tail, &h.head] {
            if vtx.0.sub(qx).sign() > 0
                && vtx.1.sub(qy).sub(&m.mul(&vtx.0.sub(qx))).is_zero()
            {
                return false;
            }
        }
    }
    for &hi in loop_ids {
        let h = &arr.he[hi];
        if h.is_arc {
            let (cx, cy, rr) = &arr.circ[&h.cid];
            let b0 = qy.sub(&m.mul(qx));
            let aa = Alg::int(1).add(&m.mul(m));
            let bb = Alg::int(2).mul(&m.mul(&b0.sub(cy)).sub(cx));
            let cc = cx.mul(cx).add(&b0.sub(cy).mul(&b0.sub(cy))).sub(rr);
            let disc = bb.mul(&bb).sub(&Alg::int(4).mul(&aa).mul(&cc));
            if disc.sign() == 0 {
                return false;
            }
        }
    }
    true
}

fn on_he_arc_span(h: &HE, p: &Pt) -> bool {
    let mut s = vsub(&h.tail, &h.o);
    let mut e = vsub(&h.head, &h.o);
    let w = vsub(p, &h.o);
    if !h.ccw {
        std::mem::swap(&mut s, &mut e);
    }
    if e == s {
        return true;
    }
    let ep = rot_conj(&e, &s);
    let wp = rot_conj(&w, &s);
    if angle_cmp(&wp, &ep) != Ordering::Less {
        return false;
    }
    let at_tail = wp.1.is_zero() && wp.0.sign() > 0;
    !at_tail
}

pub fn point_in_face(q: &Pt, loop_ids: &[usize], arr: &Arrangement) -> bool {
    let (qx, qy) = q;
    let mut m = None;
    for cand in ray_slopes() {
        if ray_generic(qx, qy, &cand, loop_ids, arr) {
            m = Some(cand);
            break;
        }
    }
    let m = m.expect("no generic ray slope");
    let d = (Alg::int(1), m.clone());
    let mut cnt = 0;
    for &hi in loop_ids {
        let h = &arr.he[hi];
        if !h.is_arc {
            let aa = &h.tail;
            let bb = &h.head;
            let ex = bb.0.sub(&aa.0);
            let ey = bb.1.sub(&aa.1);
            let den = d.0.mul(&ey).sub(&d.1.mul(&ex));
            if den.is_zero() {
                continue;
            }
            let rx = aa.0.sub(qx);
            let ry = aa.1.sub(qy);
            let t = rx.mul(&ey).sub(&ry.mul(&ex)).div(&den);
            let u = rx.mul(&d.1).sub(&ry.mul(&d.0)).div(&den);
            if t.sign() > 0 && u.sign() > 0 && u.sub(&Alg::int(1)).sign() < 0 {
                cnt += 1;
            }
        } else {
            let (cx, cy, rr) = &arr.circ[&h.cid];
            let b0 = qy.sub(&m.mul(qx));
            let aa = Alg::int(1).add(&m.mul(&m));
            let bb = Alg::int(2).mul(&m.mul(&b0.sub(cy)).sub(cx));
            let cc = cx.mul(cx).add(&b0.sub(cy).mul(&b0.sub(cy))).sub(rr);
            let disc = bb.mul(&bb).sub(&Alg::int(4).mul(&aa).mul(&cc));
            if disc.sign() < 0 {
                continue;
            }
            let root = match sqrt_exact(&disc) {
                Some(r) => r,
                None => continue,
            };
            let two_a = Alg::int(2).mul(&aa);
            for x in [
                bb.neg().sub(&root).div(&two_a),
                bb.neg().add(&root).div(&two_a),
            ] {
                if x.sub(qx).sign() <= 0 {
                    continue;
                }
                let p = (x.clone(), m.mul(&x).add(&b0));
                if on_he_arc_span(h, &p) {
                    cnt += 1;
                }
            }
        }
    }
    cnt % 2 == 1
}

pub fn area_exact(area: &Area, circ: &Circles) -> Option<(Alg, Alg)> {
    let (rat, pic, res) = resolve(area, circ);
    if res > 0 {
        None
    } else {
        Some((rat, pic))
    }
}

// =====================================================================
// nested / disconnected components: subtract islands (exact)
// =====================================================================

fn numeric_signed_area(loop_ids: &[usize], arr: &Arrangement) -> f64 {
    use std::f64::consts::PI;
    let mut tot = 0.0;
    for &hi in loop_ids {
        let h = &arr.he[hi];
        if !h.is_arc {
            let (ux, uy) = (h.tail.0.to_f64(), h.tail.1.to_f64());
            let (vx, vy) = (h.head.0.to_f64(), h.head.1.to_f64());
            tot += 0.5 * (ux * vy - vx * uy);
        } else {
            let (cx, cy) = (h.o.0.to_f64(), h.o.1.to_f64());
            let r = h.rr.to_f64().sqrt();
            let ap = (h.tail.1.to_f64() - cy).atan2(h.tail.0.to_f64() - cx);
            let aq = (h.head.1.to_f64() - cy).atan2(h.head.0.to_f64() - cx);
            let mut dd = aq - ap;
            if h.ccw && dd < 0.0 {
                dd += 2.0 * PI;
            }
            if !h.ccw && dd > 0.0 {
                dd -= 2.0 * PI;
            }
            let n = 200;
            let mut prev: Option<(f64, f64)> = None;
            for k in 0..=n {
                let th = ap + dd * (k as f64) / (n as f64);
                let x = cx + r * th.cos();
                let y = cy + r * th.sin();
                if let Some((px, py)) = prev {
                    tot += 0.5 * (px * y - x * py);
                }
                prev = Some((x, y));
            }
        }
    }
    tot
}

fn components(arr: &Arrangement) -> Vec<usize> {
    let n = arr.nodes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut x = x;
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for h in &arr.he {
        let ra = find(&mut parent, h.u);
        let rb = find(&mut parent, h.v);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    (0..n).map(|i| find(&mut parent, i)).collect()
}

/// Bounded faces with island-corrected exact areas.
pub fn faces_nested(arr: &Arrangement) -> Vec<(Vec<usize>, Area)> {
    let comp = components(arr);
    let mut positives: Vec<(Vec<usize>, Area, usize, f64)> = vec![];
    let mut neg_by_comp: BTreeMap<usize, Area> = BTreeMap::new();
    for (loop_ids, area) in faces(arr) {
        let c = comp[arr.he[loop_ids[0]].u];
        let sa = numeric_signed_area(&loop_ids, arr);
        if sa < 0.0 {
            neg_by_comp.insert(c, area.clone());
        }
        if sa > 0.0 {
            positives.push((loop_ids, area, c, sa));
        }
    }
    let mut comp_node: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..arr.nodes.len() {
        comp_node.entry(comp[i]).or_insert(i);
    }
    for (c, narea) in &neg_by_comp {
        let rep = arr.nodes[comp_node[c]].clone();
        let mut best: Option<usize> = None;
        let mut best_area = f64::INFINITY;
        for (idx, (loop_ids, _, pc, sa)) in positives.iter().enumerate() {
            if pc != c && point_in_face(&rep, loop_ids, arr) && sa.abs() < best_area {
                best_area = sa.abs();
                best = Some(idx);
            }
        }
        if let Some(idx) = best {
            positives[idx].1 = positives[idx].1.add(narea);
        }
    }
    positives.into_iter().map(|(l, a, _, _)| (l, a)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(s: i64) -> Vec<Seg> {
        let c = [(0, 0), (s, 0), (s, s), (0, s)];
        (0..4)
            .map(|i| {
                let a = c[i];
                let b = c[(i + 1) % 4];
                Seg::int(a.0, a.1, b.0, b.1)
            })
            .collect()
    }

    fn red_leaf() -> Vec<Arc> {
        vec![
            Arc::cardinal(0, 12, 12, -90, 0, 0),
            Arc::cardinal(8, 0, 8, 90, 180, 1),
            Arc::cardinal(4, 12, 8, 270, 360, 2),
            Arc::cardinal(4, 8, 4, -90, 0, 3),
            Arc::cardinal(8, 4, 4, 90, 180, 4),
        ]
    }

    #[test]
    fn red_leaf_exact() {
        let arr = build(&square(12), &red_leaf());
        let fs: Vec<_> = faces(&arr)
            .into_iter()
            .filter(|(_, a)| area_f64(a, &arr.circ) > 1e-6)
            .collect();
        assert_eq!(fs.len(), 4, "four interior faces");
        let mut total = Area::zero();
        for (_, a) in &fs {
            total = total.add(a);
        }
        assert_eq!(fmt(&total, &arr.circ), "144", "faces sum to 144");
        // left lobe via seed (7,3)
        let seed = (Alg::int(7), Alg::int(3));
        let lobe: Vec<_> = fs
            .iter()
            .filter(|(l, _)| point_in_face(&seed, l, &arr))
            .collect();
        assert_eq!(lobe.len(), 1);
        assert_eq!(fmt(&lobe[0].1, &arr.circ), "32*pi - 64");
    }

    fn s(n: u64) -> Alg {
        Alg::sqrt_rational(&num_rational::BigRational::from_integer(num_bigint::BigInt::from(n)))
    }
    fn box_seg(x0: i64, y0: i64, x1: i64, y1: i64) -> Vec<Seg> {
        let c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        (0..4).map(|i| Seg::int(c[i].0, c[i].1, c[(i + 1) % 4].0, c[(i + 1) % 4].1)).collect()
    }
    fn smallest(q: &Pt, fs: &[(Vec<usize>, Area)], arr: &Arrangement) -> Area {
        let mut hits: Vec<&Area> = fs
            .iter()
            .filter(|(l, _)| point_in_face(q, l, arr))
            .map(|(_, a)| a)
            .collect();
        hits.sort_by(|a, b| area_f64(a, &arr.circ).partial_cmp(&area_f64(b, &arr.circ)).unwrap());
        hits[0].clone()
    }

    #[test]
    fn non_pythagorean_segment() {
        // circle r=2 @ origin cut by x=1; right segment = 4/3*pi - sqrt3
        let mut segs = box_seg(-3, -3, 3, 3);
        segs.push(Seg::int(1, -3, 1, 3));
        let arcs = vec![Arc::cardinal(0, 0, 2, 0, 0, 0)];
        let arr = build(&segs, &arcs);
        let fs: Vec<_> = faces(&arr)
            .into_iter()
            .filter(|(_, a)| area_f64(a, &arr.circ) > 1e-6)
            .collect();
        let seg = smallest(&(Alg::ratio(3, 2), Alg::int(0)), &fs, &arr);
        let (rat, pic) = area_exact(&seg, &arr.circ).unwrap();
        assert!(rat.eq(&s(3).neg()), "rat = -sqrt3");
        assert!(pic.eq(&Alg::ratio(4, 3)), "pic = 4/3");
    }

    #[test]
    fn tangent_circles() {
        // two externally tangent unit circles + diameter line -> four pi/2 halves
        let arcs = vec![Arc::cardinal(0, 0, 1, 0, 0, 0), Arc::cardinal(2, 0, 1, 0, 0, 1)];
        let segs = vec![Seg::int(-1, 0, 3, 0)];
        let arr = build(&segs, &arcs);
        let fs: Vec<_> = faces(&arr)
            .into_iter()
            .filter(|(_, a)| area_f64(a, &arr.circ) > 1e-6)
            .collect();
        for seed in [
            (Alg::int(0), Alg::ratio(1, 2)),
            (Alg::int(0), Alg::ratio(-1, 2)),
            (Alg::int(2), Alg::ratio(1, 2)),
            (Alg::int(2), Alg::ratio(-1, 2)),
        ] {
            let a = smallest(&seed, &fs, &arr);
            let (rat, pic) = area_exact(&a, &arr.circ).unwrap();
            assert!(rat.is_zero() && pic.eq(&Alg::ratio(1, 2)), "half-disk = pi/2");
        }
    }

    #[test]
    fn nested_island() {
        // square 12x12 with a floating circle r=2 @ (6,6) -> 144-4pi and 4pi
        let arcs = vec![Arc::cardinal(6, 6, 2, 0, 0, 0)];
        let arr = build(&box_seg(0, 0, 12, 12), &arcs);
        let fs = faces_nested(&arr);
        let annulus = smallest(&(Alg::int(1), Alg::int(1)), &fs, &arr);
        let disk = smallest(&(Alg::int(6), Alg::int(6)), &fs, &arr);
        assert_eq!(fmt(&annulus, &arr.circ), "144 - 4*pi");
        assert_eq!(fmt(&disk, &arr.circ), "4*pi");
    }
}
