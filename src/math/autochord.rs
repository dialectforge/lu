//! Auto-chord derivation + the IDK gate — Rust port of `auto_chord.py`.
//!
//! Candidate chords are the figure's own latent lines (through >=2 vertices, plus
//! every centre-to-centre line), clipped to the bounding box. They are selected
//! by SHADING PURITY: add the fewest chords so no face holds both a shaded and an
//! unshaded seed; the shaded area is then the exact union of the shaded faces.
//!
//! `solve_shaded` is the K-gate: it returns an exact closed form ONLY when one
//! provably exists (every angle resolvable, no radical outside the field, shading
//! separable). Otherwise it refuses with an Idk reason instead of a number.

use crate::math::algnum::Alg;
use crate::math::arrangement::*;
use std::collections::BTreeMap;

fn vsub(a: &Pt, b: &Pt) -> Pt {
    (a.0.sub(&b.0), a.1.sub(&b.1))
}
fn vcross(a: &Pt, b: &Pt) -> Alg {
    a.0.mul(&b.1).sub(&a.1.mul(&b.0))
}

fn bbox(nodes: &[Pt]) -> (Alg, Alg, Alg, Alg) {
    let mut xmin = nodes[0].0.clone();
    let mut xmax = xmin.clone();
    let mut ymin = nodes[0].1.clone();
    let mut ymax = ymin.clone();
    for p in nodes {
        if p.0.sub(&xmin).sign() < 0 {
            xmin = p.0.clone();
        }
        if p.0.sub(&xmax).sign() > 0 {
            xmax = p.0.clone();
        }
        if p.1.sub(&ymin).sign() < 0 {
            ymin = p.1.clone();
        }
        if p.1.sub(&ymax).sign() > 0 {
            ymax = p.1.clone();
        }
    }
    (xmin, xmax, ymin, ymax)
}

fn line_key(p0: &Pt, d: &Pt) -> (Alg, Alg, Alg) {
    let a = d.1.clone();
    let b = d.0.neg();
    let c = a.mul(&p0.0).add(&b.mul(&p0.1));
    if !a.is_zero() {
        (Alg::int(1), b.div(&a), c.div(&a))
    } else {
        (Alg::int(0), Alg::int(1), c.div(&b))
    }
}

fn clip_line(p0: &Pt, d: &Pt, bb: &(Alg, Alg, Alg, Alg)) -> Option<Seg> {
    let (xmin, xmax, ymin, ymax) = bb;
    let (dx, dy) = d;
    let mut pts: Vec<Pt> = vec![];
    if !dx.is_zero() {
        for xv in [xmin, xmax] {
            let t = xv.sub(&p0.0).div(dx);
            let y = p0.1.add(&t.mul(dy));
            if y.sub(ymin).sign() >= 0 && y.sub(ymax).sign() <= 0 {
                pts.push((xv.clone(), y));
            }
        }
    }
    if !dy.is_zero() {
        for yv in [ymin, ymax] {
            let t = yv.sub(&p0.1).div(dy);
            let x = p0.0.add(&t.mul(dx));
            if x.sub(xmin).sign() >= 0 && x.sub(xmax).sign() <= 0 {
                pts.push((x, yv.clone()));
            }
        }
    }
    let mut uniq: Vec<Pt> = vec![];
    for p in pts {
        if !uniq.iter().any(|q| q == &p) {
            uniq.push(p);
        }
    }
    if uniq.len() < 2 {
        return None;
    }
    uniq.sort_by(|a, b| {
        a.0.sub(&b.0).mul(dx).add(&a.1.sub(&b.1).mul(dy)).sign().cmp(&0)
    });
    let a = &uniq[0];
    let b = &uniq[uniq.len() - 1];
    if a == b {
        return None;
    }
    Some(Seg::new(a.0.clone(), a.1.clone(), b.0.clone(), b.1.clone()))
}

pub fn candidate_chords(nodes: &[Pt], arcs: &[Arc]) -> Vec<Seg> {
    let bb = bbox(nodes);
    let mut lines: BTreeMap<(Alg, Alg, Alg), (Pt, Pt)> = BTreeMap::new();
    let n = nodes.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = vsub(&nodes[j], &nodes[i]);
            if d.0.is_zero() && d.1.is_zero() {
                continue;
            }
            lines.entry(line_key(&nodes[i], &d)).or_insert((nodes[i].clone(), d));
        }
    }
    for i in 0..arcs.len() {
        for j in (i + 1)..arcs.len() {
            let d = vsub(&arcs[j].o, &arcs[i].o);
            if d.0.is_zero() && d.1.is_zero() {
                continue;
            }
            lines.entry(line_key(&arcs[i].o, &d)).or_insert((arcs[i].o.clone(), d));
        }
    }
    let mut chords = vec![];
    let mut seen: Vec<(Pt, Pt)> = vec![];
    for (p0, d) in lines.values() {
        if let Some(seg) = clip_line(p0, d, &bb) {
            let key = if seg.a <= seg.b {
                (seg.a.clone(), seg.b.clone())
            } else {
                (seg.b.clone(), seg.a.clone())
            };
            if seen.iter().any(|k| k == &key) {
                continue;
            }
            seen.push(key);
            chords.push(seg);
        }
    }
    chords
}

fn combos(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut out = vec![];
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    rec(0, n, k, &mut Vec::new(), &mut out);
    out
}

pub struct Refined {
    pub chords: Vec<Seg>,
    pub area: Area,
    pub arr: Arrangement,
}

pub fn refine_until_pure(
    arcs: &[Arc],
    base_segs: &[Seg],
    shaded: &[Pt],
    unshaded: &[Pt],
    max_chords: usize,
) -> Option<Refined> {
    let base = build(base_segs, arcs);
    let cands = candidate_chords(&base.nodes, arcs);
    for k in 0..=max_chords {
        for combo in combos(cands.len(), k) {
            let mut segs = base_segs.to_vec();
            for &ci in &combo {
                segs.push(cands[ci].clone());
            }
            let arr = match build_checked(&segs, arcs) {
                Some(a) => a,
                None => continue,
            };
            let fs: Vec<(Vec<usize>, Area)> = faces(&arr)
                .into_iter()
                .filter(|(_, a)| area_f64(a, &arr.circ) > 1e-6)
                .collect();
            let mut pure = true;
            let mut acc: Option<Area> = None;
            let mut cs = vec![false; shaded.len()];
            let mut cu = vec![false; unshaded.len()];
            for (l, a) in &fs {
                let has_s = shaded.iter().any(|p| point_in_face(p, l, &arr));
                let has_u = unshaded.iter().any(|p| point_in_face(p, l, &arr));
                if has_s && has_u {
                    pure = false;
                    break;
                }
                if has_s {
                    acc = Some(match acc {
                        Some(x) => x.add(a),
                        None => a.clone(),
                    });
                    for (idx, p) in shaded.iter().enumerate() {
                        if point_in_face(p, l, &arr) {
                            cs[idx] = true;
                        }
                    }
                }
                if has_u {
                    for (idx, p) in unshaded.iter().enumerate() {
                        if point_in_face(p, l, &arr) {
                            cu[idx] = true;
                        }
                    }
                }
            }
            if !pure || !cs.iter().all(|&x| x) || !cu.iter().all(|&x| x) {
                continue;
            }
            let acc = match acc {
                Some(a) => a,
                None => continue,
            };
            if !is_closed(&acc, &arr.circ) {
                continue;
            }
            let chords = combo.iter().map(|&ci| cands[ci].clone()).collect();
            return Some(Refined { chords, area: acc, arr });
        }
    }
    None
}

/// The IDK / K-gate. Returns an exact closed form only when one provably exists.
pub enum Solved {
    Exact { rat: Alg, pic: Alg, text: String },
    Idk(&'static str),
}

pub fn solve_shaded(
    segs: &[Seg],
    arcs: &[Arc],
    shaded: &[Pt],
    unshaded: &[Pt],
) -> Solved {
    match refine_until_pure(arcs, segs, shaded, unshaded, 2) {
        Some(r) => match area_exact(&r.area, &r.arr.circ) {
            Some((rat, pic)) => Solved::Exact {
                text: fmt(&r.area, &r.arr.circ),
                rat,
                pic,
            },
            None => Solved::Idk("non-closed: no a+b*pi form"),
        },
        None => Solved::Idk("refuse: shading not separable into closed-form faces"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(s: i64) -> Vec<Seg> {
        let c = [(0, 0), (s, 0), (s, s), (0, s)];
        (0..4).map(|i| Seg::int(c[i].0, c[i].1, c[(i + 1) % 4].0, c[(i + 1) % 4].1)).collect()
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
    fn p(x: (i64, i64), y: (i64, i64)) -> Pt {
        (Alg::ratio(x.0, x.1), Alg::ratio(y.0, y.1))
    }

    #[test]
    fn full_leaf_autochord() {
        let shaded = vec![
            p((5, 2), (5, 1)),
            p((5, 1), (4, 1)),
            p((11, 2), (13, 2)),
            p((19, 2), (8, 1)),
        ];
        let unshaded = vec![p((10, 1), (1, 2)), p((4, 1), (15, 2)), p((13, 2), (11, 2))];
        match solve_shaded(&square(12), &red_leaf(), &shaded, &unshaded) {
            Solved::Exact { text, .. } => assert_eq!(text, "48*pi - 96"),
            Solved::Idk(r) => panic!("expected exact, got Idk: {}", r),
        }
    }

    #[test]
    fn left_lobe_no_chord() {
        let shaded = vec![p((7, 1), (3, 1))];
        let unshaded = vec![p((10, 1), (1, 2)), p((4, 1), (15, 2)), p((6, 1), (6, 1))];
        match solve_shaded(&square(12), &red_leaf(), &shaded, &unshaded) {
            Solved::Exact { text, .. } => assert_eq!(text, "32*pi - 64"),
            Solved::Idk(r) => panic!("expected exact, got Idk: {}", r),
        }
    }

    #[test]
    fn gate_refuses_non_closed() {
        // two circles, d=3/2: lens angle arccos(3/4) is not a rational multiple
        // of pi -> no a+b*pi closed form. The gate must refuse.
        // two circles r=2 at (0,0) and (3,0): crossings at (3/2, +-sqrt7/2),
        // lens angle arccos(3/4) is not a rational multiple of pi.
        let arcs = vec![Arc::cardinal(0, 0, 2, 0, 0, 0), Arc::cardinal(3, 0, 2, 0, 0, 1)];
        let shaded = vec![(Alg::ratio(3, 2), Alg::int(0))]; // lens interior
        let unshaded = vec![(Alg::int(-1), Alg::int(0)), (Alg::int(4), Alg::int(0))];
        match solve_shaded(&[], &arcs, &shaded, &unshaded) {
            Solved::Idk(_) => {}
            Solved::Exact { text, .. } => panic!("gate should have refused, got {}", text),
        }
    }
}
