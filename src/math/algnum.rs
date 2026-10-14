//! Exact multi-quadratic algebraic numbers — Rust port of the Phase-3 prototype
//! `algnum.py`. A number is a Q-linear combination over a basis of square-root
//! products, each basis element keyed by the SET of primes under the radical
//! (empty set = the rational part). Multiplication is exact because
//! sqrt(A)*sqrt(B) = (prod of shared primes) * sqrt(A symmetric-difference B).
//! The sign test is exact and float-free: it recurses on the number of radicals,
//! comparing squares. Uses num-rational BigRational so nothing overflows.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

pub type PrimeSet = BTreeSet<u64>;
type Q = BigRational;

fn q_zero() -> Q {
    BigRational::zero()
}
fn q_int(n: i64) -> Q {
    BigRational::from_integer(BigInt::from(n))
}
fn q_u64(n: u64) -> Q {
    BigRational::from_integer(BigInt::from(n))
}

fn factor(mut n: u128) -> Vec<(u64, u32)> {
    let mut f = Vec::new();
    let mut d: u128 = 2;
    while d * d <= n {
        if n % d == 0 {
            let mut e = 0u32;
            while n % d == 0 {
                n /= d;
                e += 1;
            }
            f.push((d as u64, e));
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        f.push((n as u64, 1));
    }
    f
}

/// sqrt(n) = k * sqrt(squarefree s); returns (k, primes(s)).
fn sqrt_int_parts(n: u128) -> (BigInt, PrimeSet) {
    if n == 1 {
        return (BigInt::one(), PrimeSet::new());
    }
    let mut k = BigInt::one();
    let mut primes = PrimeSet::new();
    for (p, e) in factor(n) {
        for _ in 0..(e / 2) {
            k *= BigInt::from(p);
        }
        if e % 2 == 1 {
            primes.insert(p);
        }
    }
    (k, primes)
}

// Canonical form (zeros pruned, prime-set keys) makes structural Eq/Ord/Hash
// agree with mathematical equality, so Alg can key the angle-atom maps.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alg {
    pub t: BTreeMap<PrimeSet, Q>,
}

impl Alg {
    pub fn zero() -> Alg {
        Alg { t: BTreeMap::new() }
    }
    pub fn from_q(q: Q) -> Alg {
        let mut t = BTreeMap::new();
        if !q.is_zero() {
            t.insert(PrimeSet::new(), q);
        }
        Alg { t }
    }
    pub fn int(n: i64) -> Alg {
        Alg::from_q(q_int(n))
    }
    pub fn ratio(num: i64, den: i64) -> Alg {
        Alg::from_q(BigRational::new(BigInt::from(num), BigInt::from(den)))
    }

    /// Exact sqrt of a non-negative rational.
    pub fn sqrt_rational(q: &Q) -> Alg {
        if q.is_zero() {
            return Alg::zero();
        }
        assert!(q.is_positive(), "sqrt of negative rational");
        let n_big = q.numer() * q.denom();
        let n = n_big.to_u128().expect("radicand too large to factor");
        let (k, primes) = sqrt_int_parts(n);
        let coeff = BigRational::new(k, q.denom().clone());
        let mut t = BTreeMap::new();
        t.insert(primes, coeff);
        Alg { t }
    }

    pub fn is_zero(&self) -> bool {
        self.t.is_empty()
    }

    pub fn primes(&self) -> PrimeSet {
        let mut s = PrimeSet::new();
        for k in self.t.keys() {
            for &p in k {
                s.insert(p);
            }
        }
        s
    }

    fn prune(mut t: BTreeMap<PrimeSet, Q>) -> Alg {
        t.retain(|_, v| !v.is_zero());
        Alg { t }
    }

    pub fn add(&self, o: &Alg) -> Alg {
        let mut t = self.t.clone();
        for (k, v) in &o.t {
            *t.entry(k.clone()).or_insert_with(q_zero) += v.clone();
        }
        Alg::prune(t)
    }
    pub fn neg(&self) -> Alg {
        Alg {
            t: self.t.iter().map(|(k, v)| (k.clone(), -v.clone())).collect(),
        }
    }
    pub fn sub(&self, o: &Alg) -> Alg {
        self.add(&o.neg())
    }

    pub fn mul(&self, o: &Alg) -> Alg {
        let mut t: BTreeMap<PrimeSet, Q> = BTreeMap::new();
        for (ka, ca) in &self.t {
            for (kb, cb) in &o.t {
                let common: PrimeSet = ka.intersection(kb).cloned().collect();
                let key: PrimeSet = ka.symmetric_difference(kb).cloned().collect();
                let mut coeff = ca.clone() * cb.clone();
                for &p in &common {
                    coeff *= q_u64(p);
                }
                *t.entry(key).or_insert_with(q_zero) += coeff;
            }
        }
        Alg::prune(t)
    }

    fn sigma(&self, p: u64) -> Alg {
        Alg {
            t: self
                .t
                .iter()
                .map(|(k, v)| (k.clone(), if k.contains(&p) { -v.clone() } else { v.clone() }))
                .collect(),
        }
    }

    pub fn inverse(&self) -> Alg {
        assert!(!self.is_zero(), "inverse of zero");
        let ps = self.primes();
        if ps.is_empty() {
            let q = self.t.get(&PrimeSet::new()).unwrap();
            return Alg::from_q(q.recip());
        }
        let p = *ps.iter().next().unwrap();
        let conj = self.sigma(p);
        let denom = self.mul(&conj);
        conj.mul(&denom.inverse())
    }
    pub fn div(&self, o: &Alg) -> Alg {
        self.mul(&o.inverse())
    }

    /// Split into (A, C) so that self = A + sqrt(p)*C, both in the subfield
    /// without prime p.
    fn split_p(&self, p: u64) -> (Alg, Alg) {
        let mut at: BTreeMap<PrimeSet, Q> = BTreeMap::new();
        let mut ct: BTreeMap<PrimeSet, Q> = BTreeMap::new();
        for (k, v) in &self.t {
            if k.contains(&p) {
                let mut kk = k.clone();
                kk.remove(&p);
                *ct.entry(kk).or_insert_with(q_zero) += v.clone();
            } else {
                *at.entry(k.clone()).or_insert_with(q_zero) += v.clone();
            }
        }
        (Alg::prune(at), Alg::prune(ct))
    }

    /// Exact sign in {-1, 0, 1}; float-free.
    pub fn sign(&self) -> i32 {
        if self.is_zero() {
            return 0;
        }
        let ps = self.primes();
        if ps.is_empty() {
            let q = self.t.get(&PrimeSet::new()).unwrap();
            return if q.is_positive() { 1 } else { -1 };
        }
        let p = *ps.iter().next().unwrap();
        let (a, c) = self.split_p(p);
        let sa = a.sign();
        let sc = c.sign();
        if sc == 0 {
            return sa;
        }
        if sa == 0 {
            return sc;
        }
        if sa == sc {
            return sa;
        }
        let d = a.mul(&a).sub(&Alg::from_q(q_u64(p)).mul(&c.mul(&c)));
        let sd = d.sign();
        if sd == 0 {
            0
        } else if sd > 0 {
            sa
        } else {
            sc
        }
    }

    pub fn eq(&self, o: &Alg) -> bool {
        self.sub(o).is_zero()
    }

    pub fn to_f64(&self) -> f64 {
        let mut s = 0.0;
        for (k, v) in &self.t {
            let mut base = 1.0f64;
            for &p in k {
                base *= p as f64;
            }
            let r = v.numer().to_f64().unwrap() / v.denom().to_f64().unwrap();
            s += r * base.sqrt();
        }
        s
    }
}

/// Exact sqrt within the multi-quadratic field (denesting), or None if the
/// square root is a genuine new nested radical (or the input is negative).
pub fn isqrt_field(a: &Alg) -> Option<Alg> {
    isqrt_d(a, 0)
}

fn isqrt_d(a: &Alg, depth: u32) -> Option<Alg> {
    if a.is_zero() {
        return Some(Alg::zero());
    }
    if a.sign() < 0 {
        return None;
    }
    if depth > 12 {
        return None;
    }
    let ps = a.primes();
    if ps.is_empty() {
        let q = a.t.get(&PrimeSet::new()).unwrap();
        return Some(Alg::sqrt_rational(q));
    }
    let p = *ps.iter().next().unwrap();
    let (a_, b) = a.split_p(p);
    if b.is_zero() {
        return isqrt_d(&a_, depth + 1);
    }
    let pe = Alg::from_q(q_u64(p));
    let disc = a_.mul(&a_).sub(&pe.mul(&b.mul(&b)));
    let sd = isqrt_d(&disc, depth + 1)?;
    let sqrtp = Alg::sqrt_rational(&q_u64(p));
    let half = Alg::ratio(1, 2);
    for s in [sd.clone(), sd.neg()] {
        let c2 = a_.add(&s).mul(&half);
        if c2.sign() < 0 {
            continue;
        }
        let c = match isqrt_d(&c2, depth + 1) {
            Some(c) => c,
            None => continue,
        };
        if c.is_zero() {
            continue;
        }
        let d = b.mul(&half.mul(&c.inverse()));
        let cand = c.add(&sqrtp.mul(&d));
        if cand.mul(&cand).sub(a).is_zero() {
            return Some(if cand.sign() >= 0 { cand } else { cand.neg() });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: u64) -> Alg {
        Alg::sqrt_rational(&q_u64(n))
    }

    #[test]
    fn ring_and_radicals() {
        assert!(s(2).mul(&s(2)).eq(&Alg::int(2)));
        assert!(s(2).mul(&s(3)).eq(&s(6)));
        assert!(s(2).mul(&s(8)).eq(&Alg::int(4)));
        assert!(s(12).eq(&Alg::int(2).mul(&s(3))));
        // (1+sqrt2)(sqrt2-1) = 1
        let a = Alg::int(1).add(&s(2));
        let b = s(2).sub(&Alg::int(1));
        assert!(a.mul(&b).eq(&Alg::int(1)));
        // inverse
        assert!(Alg::int(1).div(&a).eq(&b));
    }

    #[test]
    fn exact_sign() {
        // sqrt2 = 1.41421356237309504880...; bracket at 1e-16 (float cannot)
        let lo = Alg::ratio(14142135623730950, 10000000000000000);
        let hi = Alg::ratio(14142135623730951, 10000000000000000);
        assert_eq!(s(2).sub(&lo).sign(), 1);
        assert_eq!(s(2).sub(&hi).sign(), -1);
        assert_eq!(s(8).sub(&Alg::int(2).mul(&s(2))).sign(), 0);
        // (sqrt2-1)(sqrt3-1) > 0
        let v = s(6).sub(&s(2)).sub(&s(3)).add(&Alg::int(1));
        assert_eq!(v.sign(), 1);
    }

    #[test]
    fn denesting() {
        // sqrt(3+2sqrt2) = 1+sqrt2
        let a = Alg::int(3).add(&Alg::int(2).mul(&s(2)));
        assert!(isqrt_field(&a).unwrap().eq(&Alg::int(1).add(&s(2))));
        // sqrt(5+2sqrt6) = sqrt2+sqrt3
        let a = Alg::int(5).add(&Alg::int(2).mul(&s(6)));
        assert!(isqrt_field(&a).unwrap().eq(&s(2).add(&s(3))));
        // sqrt(7+4sqrt3) = 2+sqrt3
        let a = Alg::int(7).add(&Alg::int(4).mul(&s(3)));
        assert!(isqrt_field(&a).unwrap().eq(&Alg::int(2).add(&s(3))));
        // genuine new radical -> None
        assert!(isqrt_field(&Alg::int(1).add(&s(2))).is_none());
        assert!(isqrt_field(&Alg::int(-4)).is_none());
    }
}
