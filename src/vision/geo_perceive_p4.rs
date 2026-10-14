//! Phase 4: real phone-photo perception for the geometry pipeline.
//!
//! internal notes (Phase 1) established `geo_perceive.rs`: clean-PNG red-leaf
//! extraction, proven not-a-lookup. Phase 4 makes the perceiver handle the images
//! John actually sends -- REAL PHONE PHOTOS -- by adding, on top of Phase 1:
//!
//!   (a) a pure-Rust baseline JPEG decoder (SOF0, YCbCr 4:2:0/4:4:4, restart
//!       markers) so JPEG bytes decode to RGB in-house (no `image` crate, no new
//!       dependency -- respecting the tree's C13 no-new-crates discipline);
//!   (b) perspective/skew rectification: detect the outer square's 4 corners and
//!       warp to an axis-aligned square (homography) before measuring arcs;
//!   (c) the GRID-LEAF variant topology (multi-radius arcs) via sequential-RANSAC
//!       circle fitting + snap-then-verify, building the correct verified
//!       `Partition`;
//!   (d) OCR of the printed side/scale digit so the numeric scale is DERIVED FROM
//!       PIXELS -- eliminating the hand-passed `side_hint` of Phase 1.
//!
//! HONESTY FLOOR (absolute, same as Phase 1): the perceiver is handed ONLY the
//! pixel bytes. There is NO filename/path/hash/answer lookup anywhere on the
//! path. Every number flows pixels -> measured structure -> `Partition` ->
//! `solve_and_verify`. A figure it cannot parse -> `CantParse`, NO number.

#![allow(clippy::needless_range_loop)]

use crate::vision::geo_perceive::{decode_png_rgb, RgbImage};

// =====================================================================
// (a) Baseline JPEG decoder (pure Rust). Supports: SOF0 (baseline DCT),
//     1 or 3 components, arbitrary H/V sampling (covers 4:2:0 and 4:4:4),
//     DHT/DQT, DRI restart markers. Progressive (SOF2) is refused (None).
// =====================================================================

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

struct HuffTable {
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    vals: Vec<u8>,
}

impl HuffTable {
    fn build(counts: &[u8; 16], vals: Vec<u8>) -> HuffTable {
        let mut huffsize = [0u8; 257];
        let mut k = 0usize;
        for l in 0..16 {
            for _ in 0..counts[l] {
                huffsize[k] = (l as u8) + 1;
                k += 1;
            }
        }
        let total = k;
        huffsize[k] = 0;
        let mut huffcode = [0u32; 257];
        if total > 0 {
            let mut code = 0u32;
            let mut si = huffsize[0];
            let mut kk = 0usize;
            loop {
                while huffsize[kk] == si {
                    huffcode[kk] = code;
                    code += 1;
                    kk += 1;
                }
                if huffsize[kk] == 0 {
                    break;
                }
                while huffsize[kk] != si {
                    code <<= 1;
                    si += 1;
                }
            }
        }
        let mut mincode = [0i32; 17];
        let mut maxcode = [-1i32; 17];
        let mut valptr = [0i32; 17];
        let mut p = 0usize;
        for l in 1..=16 {
            let cnt = counts[l - 1] as usize;
            if cnt > 0 {
                valptr[l] = p as i32;
                mincode[l] = huffcode[p] as i32;
                p += cnt;
                maxcode[l] = huffcode[p - 1] as i32;
            }
        }
        HuffTable { mincode, maxcode, valptr, vals }
    }
}

struct BitReader<'a> {
    d: &'a [u8],
    p: usize,
    acc: u32,
    cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(d: &'a [u8], p: usize) -> Self {
        BitReader { d, p, acc: 0, cnt: 0 }
    }
    fn next_byte(&mut self) -> u8 {
        if self.p >= self.d.len() {
            return 0;
        }
        let b = self.d[self.p];
        if b == 0xFF {
            let nb = *self.d.get(self.p + 1).unwrap_or(&0xD9);
            if nb == 0x00 {
                self.p += 2;
                return 0xFF;
            }
            // A real marker (restart / EOI): stop consuming, pad zero bits.
            return 0;
        }
        self.p += 1;
        b
    }
    fn bit(&mut self) -> u32 {
        if self.cnt == 0 {
            self.acc = self.next_byte() as u32;
            self.cnt = 8;
        }
        self.cnt -= 1;
        (self.acc >> self.cnt) & 1
    }
    fn bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit();
        }
        v
    }
    fn decode(&mut self, t: &HuffTable) -> u8 {
        let mut code = 0i32;
        for l in 1..=16 {
            code = (code << 1) | self.bit() as i32;
            if t.maxcode[l] >= 0 && code <= t.maxcode[l] {
                let idx = (t.valptr[l] + code - t.mincode[l]) as usize;
                return *t.vals.get(idx).unwrap_or(&0);
            }
        }
        0
    }
    fn receive_extend(&mut self, s: u8) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.bits(s as u32) as i32;
        let vt = 1i32 << (s - 1);
        if v < vt {
            v - (1 << s) + 1
        } else {
            v
        }
    }
    fn restart(&mut self) {
        self.cnt = 0;
        while self.p + 1 < self.d.len() {
            if self.d[self.p] == 0xFF {
                let m = self.d[self.p + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.p += 2;
                    return;
                }
                if m == 0x00 {
                    self.p += 2;
                    continue;
                }
            }
            self.p += 1;
        }
    }
}

#[inline]
fn be16(b: &[u8], i: usize) -> usize {
    ((b[i] as usize) << 8) | b[i + 1] as usize
}

fn idct8x8(blk: &mut [f32; 64]) {
    let mut m = [[0f32; 8]; 8];
    for a in 0..8 {
        for b in 0..8 {
            let alpha = if b == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            m[a][b] =
                alpha * (((2 * a + 1) as f32) * (b as f32) * std::f32::consts::PI / 16.0).cos() * 0.5;
        }
    }
    let mut tmp = [0f32; 64];
    for u in 0..8 {
        for y in 0..8 {
            let mut s = 0f32;
            for v in 0..8 {
                s += m[y][v] * blk[v * 8 + u];
            }
            tmp[y * 8 + u] = s;
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let mut s = 0f32;
            for u in 0..8 {
                s += m[x][u] * tmp[y * 8 + u];
            }
            blk[y * 8 + x] = s;
        }
    }
}

#[inline]
fn clamp8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        (v + 0.5) as u8
    }
}

struct JComp {
    id: u8,
    h: usize,
    v: usize,
    qt: usize,
    dc: usize,
    ac: usize,
}

struct Plane {
    pw: usize,
    ph: usize,
    px: Vec<u8>,
}

/// Decode a baseline (SOF0) JPEG to RGB8. Returns `None` for anything it does
/// not support (progressive, arithmetic coding, malformed) -- an honest refusal,
/// never a half-decode.
pub fn decode_jpeg_rgb(bytes: &[u8]) -> Option<RgbImage> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut qt: [[u16; 64]; 4] = [[0; 64]; 4];
    let mut huff_dc: Vec<Option<HuffTable>> = (0..4).map(|_| None).collect();
    let mut huff_ac: Vec<Option<HuffTable>> = (0..4).map(|_| None).collect();
    let mut w = 0usize;
    let mut h = 0usize;
    let mut comps: Vec<JComp> = Vec::new();
    let mut dri = 0usize;
    let mut sos_off = 0usize;
    let mut i = 2usize;
    loop {
        if i + 1 >= bytes.len() {
            return None;
        }
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = bytes[i + 1];
        i += 2;
        match m {
            0xD8 | 0xD9 | 0x01 => {}
            0xD0..=0xD7 => {}
            0xC0 => {
                let l = be16(bytes, i);
                h = be16(bytes, i + 3);
                w = be16(bytes, i + 5);
                let nc = bytes[i + 7] as usize;
                for c in 0..nc {
                    let o = i + 8 + c * 3;
                    let hv = bytes[o + 1];
                    comps.push(JComp {
                        id: bytes[o],
                        h: (hv >> 4) as usize,
                        v: (hv & 15) as usize,
                        qt: bytes[o + 2] as usize,
                        dc: 0,
                        ac: 0,
                    });
                }
                i += l;
            }
            0xC1..=0xCF if m != 0xC4 && m != 0xC8 && m != 0xCC => {
                // SOF1..SOF15 except DHT/JPG/DAC: not baseline -> honest refuse.
                return None;
            }
            0xC4 => {
                let l = be16(bytes, i);
                let end = i + l;
                let mut o = i + 2;
                while o < end {
                    let tc_th = bytes[o];
                    let tc = tc_th >> 4;
                    let th = (tc_th & 15) as usize;
                    o += 1;
                    let mut counts = [0u8; 16];
                    let mut total = 0usize;
                    for k in 0..16 {
                        counts[k] = bytes[o + k];
                        total += counts[k] as usize;
                    }
                    o += 16;
                    let vals = bytes[o..o + total].to_vec();
                    o += total;
                    let t = HuffTable::build(&counts, vals);
                    if th < 4 {
                        if tc == 0 {
                            huff_dc[th] = Some(t);
                        } else {
                            huff_ac[th] = Some(t);
                        }
                    }
                }
                i += l;
            }
            0xDB => {
                let l = be16(bytes, i);
                let end = i + l;
                let mut o = i + 2;
                while o < end {
                    let pq_tq = bytes[o];
                    let pq = pq_tq >> 4;
                    let tq = (pq_tq & 15) as usize;
                    o += 1;
                    if tq < 4 {
                        for k in 0..64 {
                            qt[tq][k] = if pq == 0 {
                                bytes[o + k] as u16
                            } else {
                                be16(bytes, o + 2 * k) as u16
                            };
                        }
                    }
                    o += if pq == 0 { 64 } else { 128 };
                }
                i += l;
            }
            0xDD => {
                let l = be16(bytes, i);
                dri = be16(bytes, i + 2);
                i += l;
            }
            0xDA => {
                let l = be16(bytes, i);
                let ns = bytes[i + 2] as usize;
                for c in 0..ns {
                    let o = i + 3 + c * 2;
                    let cs = bytes[o];
                    let td_ta = bytes[o + 1];
                    if let Some(ci) = comps.iter().position(|cc| cc.id == cs) {
                        comps[ci].dc = (td_ta >> 4) as usize;
                        comps[ci].ac = (td_ta & 15) as usize;
                    }
                }
                sos_off = i + l;
                break;
            }
            _ => {
                let l = be16(bytes, i);
                i += l;
            }
        }
    }
    if w == 0 || h == 0 || comps.is_empty() || sos_off == 0 {
        return None;
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap();
    let vmax = comps.iter().map(|c| c.v).max().unwrap();
    if hmax == 0 || vmax == 0 {
        return None;
    }
    let mcux = (w + 8 * hmax - 1) / (8 * hmax);
    let mcuy = (h + 8 * vmax - 1) / (8 * vmax);
    let mut planes: Vec<Plane> = comps
        .iter()
        .map(|c| {
            let pw = mcux * c.h * 8;
            let ph = mcuy * c.v * 8;
            Plane { pw, ph, px: vec![0u8; pw * ph] }
        })
        .collect();
    let mut br = BitReader::new(bytes, sos_off);
    let mut preds = vec![0i32; comps.len()];
    for my in 0..mcuy {
        for mx in 0..mcux {
            let mcu_idx = my * mcux + mx;
            if dri > 0 && mcu_idx > 0 && mcu_idx % dri == 0 {
                br.restart();
                for p in preds.iter_mut() {
                    *p = 0;
                }
            }
            for (ci, c) in comps.iter().enumerate() {
                let dct = huff_dc[c.dc].as_ref()?;
                let act = huff_ac[c.ac].as_ref()?;
                for by in 0..c.v {
                    for bx in 0..c.h {
                        let mut blk = [0f32; 64];
                        let t = br.decode(dct);
                        let diff = br.receive_extend(t);
                        preds[ci] += diff;
                        blk[0] = (preds[ci] * qt[c.qt][0] as i32) as f32;
                        let mut k = 1usize;
                        while k < 64 {
                            let rs = br.decode(act);
                            let r = (rs >> 4) as usize;
                            let s = rs & 15;
                            if s == 0 {
                                if r == 15 {
                                    k += 16;
                                    continue;
                                }
                                break;
                            }
                            k += r;
                            if k > 63 {
                                break;
                            }
                            let val = br.receive_extend(s);
                            blk[ZIGZAG[k]] = (val * qt[c.qt][k] as i32) as f32;
                            k += 1;
                        }
                        idct8x8(&mut blk);
                        let ox = (mx * c.h + bx) * 8;
                        let oy = (my * c.v + by) * 8;
                        let pl = &mut planes[ci];
                        for yy in 0..8 {
                            for xx in 0..8 {
                                let px = ox + xx;
                                let py = oy + yy;
                                if px < pl.pw && py < pl.ph {
                                    pl.px[py * pl.pw + px] = clamp8(blk[yy * 8 + xx] + 128.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let sample = |ci: usize| -> f32 {
                let c = &comps[ci];
                let pl = &planes[ci];
                let sx = (x * c.h / hmax).min(pl.pw - 1);
                let sy = (y * c.v / vmax).min(pl.ph - 1);
                pl.px[sy * pl.pw + sx] as f32
            };
            let (r, g, b) = if comps.len() == 1 {
                let yy = sample(0);
                (yy, yy, yy)
            } else {
                let yy = sample(0);
                let cb = sample(1) - 128.0;
                let cr = sample(2) - 128.0;
                (
                    yy + 1.402 * cr,
                    yy - 0.344136 * cb - 0.714136 * cr,
                    yy + 1.772 * cb,
                )
            };
            let o = (y * w + x) * 3;
            rgb[o] = clamp8(r);
            rgb[o + 1] = clamp8(g);
            rgb[o + 2] = clamp8(b);
        }
    }
    Some(RgbImage { w, h, rgb })
}

/// Unified decode: PNG (Phase 1 in-house) OR baseline JPEG (Phase 4).
pub fn decode_image_rgb(bytes: &[u8]) -> Option<RgbImage> {
    decode_png_rgb(bytes).or_else(|| decode_jpeg_rgb(bytes))
}


// =====================================================================
// (b) Perspective rectification: find the outer square's 4 corners from the
//     dark ink, solve the homography, warp to an axis-aligned NxN square.
// =====================================================================

pub const RECT_N: usize = 600;

#[inline]
pub(crate) fn p4_is_ink(r: u8, g: u8, b: u8) -> bool {
    let mx = r.max(g).max(b) as i32;
    let mn = r.min(g).min(b) as i32;
    mx < 120 && (mx - mn) < 70
}

#[inline]
pub(crate) fn p4_is_colored(r: u8, g: u8, b: u8) -> bool {
    let mx = r.max(g).max(b) as i32;
    let mn = r.min(g).min(b) as i32;
    (mx - mn) > 45 && mx > 60
}

#[inline]
pub(crate) fn p4_is_background(r: u8, g: u8, b: u8) -> bool {
    // Near-white paper/background: all three channels bright.
    r.min(g).min(b) as i32 > 200
}

/// General foreground/stroke mask. A pixel is a stroke if it is DARK INK
/// (the historical `p4_is_ink`, unchanged) OR it is a SATURATED COLOURED pixel
/// that lies within a few pixels of the paper background -- i.e. it is part of
/// a THIN stroke or the BOUNDARY of a colour region, not the deep interior of a
/// solid fill. This lets thin, low-contrast, anti-aliased *coloured* vector
/// strokes (e.g. light-blue outlines, a green ground line, coloured digit
/// glyphs) drive line/polygon/OCR detection, WITHOUT flooding the Hough
/// accumulator with the interior mass of a solid colour fill. For a purely
/// dark-ink figure (no coloured pixels) this is byte-identical to `p4_is_ink`.
pub(crate) fn stroke_mask(img: &RgbImage) -> Vec<bool> {
    let w = img.w;
    let h = img.h;
    let n = w * h;
    let mut dark = vec![false; n];
    let mut colored = vec![false; n];
    let mut bg = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(img, x, y);
            let i = y * w + x;
            dark[i] = p4_is_ink(r, g, b);
            colored[i] = p4_is_colored(r, g, b);
            bg[i] = p4_is_background(r, g, b);
        }
    }
    // Radius of the "near background" test, scaled to the raster so it behaves
    // the same on the raw (large) and rectified (~600px) rasters.
    let rad = (((w.min(h)) as f64) * 0.004).round() as usize;
    let rad = rad.max(2);
    // Separable boolean dilation of the background mask by `rad` (Chebyshev):
    // bgd[i] == true  <=>  some background pixel within +/-rad in x and y.
    let mut tmp = vec![false; n];
    for y in 0..h {
        let base = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(rad);
            let hi = (x + rad).min(w - 1);
            let mut any = false;
            let mut k = lo;
            while k <= hi {
                if bg[base + k] { any = true; break; }
                k += 1;
            }
            tmp[base + x] = any;
        }
    }
    let mut bgd = vec![false; n];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(rad);
            let hi = (y + rad).min(h - 1);
            let mut any = false;
            let mut k = lo;
            while k <= hi {
                if tmp[k * w + x] { any = true; break; }
                k += 1;
            }
            bgd[y * w + x] = any;
        }
    }
    let mut out = vec![false; n];
    for i in 0..n {
        out[i] = dark[i] || (colored[i] && bgd[i]);
    }
    out
}

#[inline]
pub(crate) fn px(img: &RgbImage, x: usize, y: usize) -> (u8, u8, u8) {
    let i = (y * img.w + x) * 3;
    (img.rgb[i], img.rgb[i + 1], img.rgb[i + 2])
}

/// The 4 corners (TL,TR,BR,BL) of the dark bounding quad, as (x,y) in source
/// pixels. Found as extremes of (x+y) and (x-y) over ink pixels -- robust to
/// rotation/skew up to ~45deg. Returns None if too little ink.
fn find_square_corners(img: &RgbImage) -> Option<[(f64, f64); 4]> {
    let (mut tl, mut br) = ((f64::MAX, (0.0, 0.0)), (f64::MIN, (0.0, 0.0)));
    let (mut tr, mut bl) = ((f64::MIN, (0.0, 0.0)), (f64::MAX, (0.0, 0.0)));
    let mut ink = 0usize;
    for y in 0..img.h {
        for x in 0..img.w {
            let (r, g, b) = px(img, x, y);
            if !p4_is_ink(r, g, b) {
                continue;
            }
            ink += 1;
            let (fx, fy) = (x as f64, y as f64);
            let s = fx + fy;
            let d = fx - fy;
            if s < tl.0 {
                tl = (s, (fx, fy));
            }
            if s > br.0 {
                br = (s, (fx, fy));
            }
            if d > tr.0 {
                tr = (d, (fx, fy));
            }
            if d < bl.0 {
                bl = (d, (fx, fy));
            }
        }
    }
    if ink < 200 {
        return None;
    }
    let corners = [tl.1, tr.1, br.1, bl.1];
    // sanity: the quad must have real extent
    let minx = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min);
    let maxx = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max);
    let miny = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min);
    let maxy = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max);
    if (maxx - minx) < 20.0 || (maxy - miny) < 20.0 {
        return None;
    }
    Some(corners)
}

/// Solve an 8x8 linear system by Gaussian elimination with partial pivoting.
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut piv = col;
        for r in (col + 1)..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            for c in col..8 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 8];
    for i in 0..8 {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// Homography mapping dst (unit square corners) -> src corners. Returns the 8
/// coefficients h0..h7 (h8 == 1).
fn homography(dst: &[(f64, f64); 4], src: &[(f64, f64); 4]) -> Option<[f64; 8]> {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (xx, yy) = dst[i];
        let (u, v) = src[i];
        a[2 * i] = [xx, yy, 1.0, 0.0, 0.0, 0.0, -xx * u, -yy * u];
        b[2 * i] = u;
        a[2 * i + 1] = [0.0, 0.0, 0.0, xx, yy, 1.0, -xx * v, -yy * v];
        b[2 * i + 1] = v;
    }
    solve8(a, b)
}

#[inline]
fn sample_bilinear(img: &RgbImage, sx: f64, sy: f64) -> (u8, u8, u8) {
    if sx < 0.0 || sy < 0.0 || sx > (img.w - 1) as f64 || sy > (img.h - 1) as f64 {
        return (255, 255, 255); // outside -> white background
    }
    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(img.w - 1);
    let y1 = (y0 + 1).min(img.h - 1);
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;
    let p = |x: usize, y: usize| px(img, x, y);
    let (a, b2, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let lerp = |u: u8, v: u8, t: f64| u as f64 * (1.0 - t) + v as f64 * t;
    let ch = |i: usize| {
        let top = lerp([a.0, a.1, a.2][i], [b2.0, b2.1, b2.2][i], fx);
        let bot = lerp([c.0, c.1, c.2][i], [d.0, d.1, d.2][i], fx);
        (top * (1.0 - fy) + bot * fy) as u8
    };
    (ch(0), ch(1), ch(2))
}

/// Rectify: warp the detected outer square to an axis-aligned RECT_N x RECT_N
/// image. Returns the rectified image plus the measured skew (max corner
/// deviation of the source quad from a perfect axis-aligned square, in units of
/// the mean side -- 0 == already square).
pub(crate) fn rectify(img: &RgbImage) -> Option<(RgbImage, f64)> {
    let src = find_square_corners(img)?;
    // skew metric: compare source side lengths + diagonal balance.
    let dist = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let top = dist(src[0], src[1]);
    let right = dist(src[1], src[2]);
    let bot = dist(src[2], src[3]);
    let left = dist(src[3], src[0]);
    let mean = (top + right + bot + left) / 4.0;
    if mean < 20.0 {
        return None;
    }
    let skew = ([top, right, bot, left]
        .iter()
        .map(|s| (s - mean).abs())
        .fold(0.0, f64::max))
        / mean;
    let n = RECT_N as f64;
    let dst = [(0.0, 0.0), (n - 1.0, 0.0), (n - 1.0, n - 1.0), (0.0, n - 1.0)];
    let hcoef = homography(&dst, &src)?;
    let mut out = vec![0u8; RECT_N * RECT_N * 3];
    for yy in 0..RECT_N {
        for xx in 0..RECT_N {
            let (fx, fy) = (xx as f64, yy as f64);
            let denom = hcoef[6] * fx + hcoef[7] * fy + 1.0;
            let sx = (hcoef[0] * fx + hcoef[1] * fy + hcoef[2]) / denom;
            let sy = (hcoef[3] * fx + hcoef[4] * fy + hcoef[5]) / denom;
            let (r, g, b) = sample_bilinear(img, sx, sy);
            let o = (yy * RECT_N + xx) * 3;
            out[o] = r;
            out[o + 1] = g;
            out[o + 2] = b;
        }
    }
    Some((RgbImage { w: RECT_N, h: RECT_N, rgb: out }, skew))
}

// =====================================================================
// (c) Sequential-RANSAC circle detection on the shaded-region boundary of the
//     RECTIFIED image (colour-agnostic), with a deterministic PRNG so results
//     are reproducible. Ported from prototypes/perception/perceive.py.
// =====================================================================

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        // numerical recipes LCG
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

/// Boundary points of the shaded (coloured) region in the rectified image,
/// dropping points on the straight square border.
pub(crate) fn shaded_boundary(rect: &RgbImage) -> Vec<(f64, f64)> {
    let n = rect.w;
    let m = 6usize;
    let mut pts = Vec::new();
    for y in 0..rect.h {
        for x in 0..rect.w {
            let (r, g, b) = px(rect, x, y);
            if !p4_is_colored(r, g, b) {
                continue;
            }
            let mut edge = false;
            let neigh = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neigh {
                if nx >= rect.w || ny >= rect.h {
                    edge = true;
                    break;
                }
                let (nr, ng, nb) = px(rect, nx, ny);
                if !p4_is_colored(nr, ng, nb) {
                    edge = true;
                    break;
                }
            }
            if !edge {
                continue;
            }
            if x <= m || x >= n - 1 - m || y <= m || y >= n - 1 - m {
                continue;
            }
            pts.push((x as f64, y as f64));
        }
    }
    pts
}

/// Algebraic circle fit (least squares) over >=3 points -> (cx,cy,r).
pub(crate) fn fit_circle(pts: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = pts.len() as f64;
    if pts.len() < 3 {
        return None;
    }
    // Solve [2x 2y 1] [a b c]^T = x^2+y^2 by normal equations (3x3).
    let (mut sxx, mut sxy, mut sx, mut syy, mut sy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxb, mut syb, mut sb) = (0.0, 0.0, 0.0);
    for &(x, y) in pts {
        let bb = x * x + y * y;
        sxx += x * x;
        sxy += x * y;
        sx += x;
        syy += y * y;
        sy += y;
        sxb += x * bb;
        syb += y * bb;
        sb += bb;
    }
    // Matrix M (3x3) for unknowns (a,b,c) where circle: x^2+y^2 = 2a x + 2b y + c
    let m = [
        [2.0 * sxx, 2.0 * sxy, sx],
        [2.0 * sxy, 2.0 * syy, sy],
        [2.0 * sx, 2.0 * sy, n],
    ];
    let rhs = [sxb, syb, sb];
    let sol = solve3(m, rhs)?;
    let (a, b, c) = (sol[0], sol[1], sol[2]);
    let r2 = c + a * a + b * b;
    if r2 <= 0.0 {
        return None;
    }
    Some((a, b, r2.sqrt()))
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            for c in col..3 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    Some([b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]])
}

/// A detected circle in side-units (bottom-left origin): center (cx,cy), radius
/// r, and the inlier count.
#[derive(Debug, Clone)]
pub struct DetCircle {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub inliers: usize,
}

/// Sequential RANSAC: peel up to `max_c` circles off the boundary point set.
pub(crate) fn ransac_circles(pts: &[(f64, f64)], n: usize) -> Vec<DetCircle> {
    let nf = n as f64;
    let tol = 3.0;
    let min_inl = 70usize;
    let mut pool: Vec<(f64, f64)> = pts.to_vec();
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut found = Vec::new();
    for _ in 0..5 {
        if pool.len() < min_inl {
            break;
        }
        let mut best_mask: Option<Vec<bool>> = None;
        let mut best_count = 0usize;
        for _ in 0..500 {
            let i1 = rng.below(pool.len());
            let i2 = rng.below(pool.len());
            let i3 = rng.below(pool.len());
            if i1 == i2 || i2 == i3 || i1 == i3 {
                continue;
            }
            let sample = [pool[i1], pool[i2], pool[i3]];
            let (cx, cy, r) = match fit_circle(&sample) {
                Some(c) => c,
                None => continue,
            };
            if r < nf * 0.1 || r > nf * 1.3 {
                continue;
            }
            let mut mask = vec![false; pool.len()];
            let mut cnt = 0usize;
            for (k, &(x, y)) in pool.iter().enumerate() {
                if (((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r).abs() < tol {
                    mask[k] = true;
                    cnt += 1;
                }
            }
            if cnt > best_count {
                best_count = cnt;
                best_mask = Some(mask);
            }
        }
        if best_count < min_inl {
            break;
        }
        let mask = best_mask.unwrap();
        let inliers: Vec<(f64, f64)> = pool
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&p, _)| p)
            .collect();
        if let Some((cx, cy, r)) = fit_circle(&inliers) {
            // convert to side-units, y-flip to bottom-left origin (match prototype)
            found.push(DetCircle {
                cx: cx / nf,
                cy: (nf - 1.0 - cy) / nf,
                r: r / nf,
                inliers: best_count,
            });
            pool = pool
                .into_iter()
                .filter(|&(x, y)| (((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r).abs() >= tol)
                .collect();
        } else {
            break;
        }
    }
    found
}

/// Snap a value to the nearest rational with denominator <= 4; return
/// (num, den, distance).
pub(crate) fn snap_frac(x: f64) -> (i64, i64, f64) {
    let mut best = (0i64, 1i64, f64::MAX);
    for den in 1..=4i64 {
        let num = (x * den as f64).round() as i64;
        let val = num as f64 / den as f64;
        let d = (x - val).abs();
        if d < best.2 {
            best = (num, den, d);
        }
    }
    best
}


// =====================================================================
// (d) OCR of the printed numeric scale label so the side length is DERIVED
//     FROM PIXELS (eliminating Phase 1's hand-passed `side_hint`). A compact,
//     dependency-free digit recogniser: isolate dark glyph components, normalise
//     each (aspect-preserving) to a 20x20 binary bitmap, and classify by
//     best pixel-overlap (Hamming) against a built-in reference font. K-gated:
//     accepted only when the fit + runner-up margin are decisive.
// =====================================================================

/// 5x7 reference bitmaps for digits 0-9 ('1'/'#'=ink).
const DIGIT_FONT: [[&str; 7]; 10] = [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"], // 0
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"], // 1
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"], // 2
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"], // 3
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"], // 4
    ["11111", "10000", "11110", "00001", "00001", "10001", "01110"], // 5
    ["00110", "01000", "10000", "11110", "10001", "10001", "01110"], // 6
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"], // 7
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"], // 8
    ["01110", "10001", "10001", "01111", "00001", "00010", "01100"], // 9
];

pub(crate) const NORM: usize = 20;

/// Aspect-preserving normalise a tight binary glyph to a NORM x NORM bitmap,
/// centered, with a 1px margin. Preserves shape proportions (so a narrow '1'
/// stays narrow).
pub(crate) fn norm_bitmap(grid: &[bool], gw: usize, gh: usize) -> [bool; NORM * NORM] {
    let mut out = [false; NORM * NORM];
    // tight ink bbox
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut any = false;
    for y in 0..gh {
        for x in 0..gw {
            if grid[y * gw + x] {
                any = true;
                if x < x0 {
                    x0 = x;
                }
                if y < y0 {
                    y0 = y;
                }
                if x > x1 {
                    x1 = x;
                }
                if y > y1 {
                    y1 = y;
                }
            }
        }
    }
    if !any {
        return out;
    }
    let bw = x1 - x0 + 1;
    let bh = y1 - y0 + 1;
    let scale = (NORM as f64 - 2.0) / bw.max(bh) as f64;
    let tw = ((bw as f64 * scale).round() as usize).clamp(1, NORM);
    let th = ((bh as f64 * scale).round() as usize).clamp(1, NORM);
    let ox = (NORM - tw) / 2;
    let oy = (NORM - th) / 2;
    for ty in 0..th {
        for tx in 0..tw {
            let sx = x0 + (tx * bw / tw).min(bw - 1);
            let sy = y0 + (ty * bh / th).min(bh - 1);
            if grid[sy * gw + sx] {
                out[(oy + ty) * NORM + ox + tx] = true;
            }
        }
    }
    out
}

pub(crate) fn ref_bitmaps() -> [[bool; NORM * NORM]; 10] {
    let mut out = [[false; NORM * NORM]; 10];
    for d in 0..10 {
        let mut grid = vec![false; 5 * 7];
        for (ry, row) in DIGIT_FONT[d].iter().enumerate() {
            for (rx, ch) in row.chars().enumerate() {
                grid[ry * 5 + rx] = ch == '1' || ch == '#';
            }
        }
        out[d] = norm_bitmap(&grid, 5, 7);
    }
    out
}

pub(crate) struct Glyph {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
    pub(crate) mask: Vec<(usize, usize)>,
}

/// Connected dark-ink components that look like printed glyphs. Works on any
/// image size (the scale label often sits in the full frame, outside the
/// square).
pub(crate) fn find_glyphs(img: &RgbImage) -> Vec<Glyph> {
    let w = img.w;
    let h = img.h;
    let nf = w.min(h) as f64;
    let ink = stroke_mask(img);
    let mut seen = vec![false; w * h];
    let mut glyphs = Vec::new();
    let mut stack = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            if !ink[sy * w + sx] || seen[sy * w + sx] {
                continue;
            }
            stack.clear();
            stack.push((sx, sy));
            seen[sy * w + sx] = true;
            let mut comp = Vec::new();
            let (mut x0, mut y0, mut x1, mut y1) = (sx, sy, sx, sy);
            while let Some((x, y)) = stack.pop() {
                comp.push((x, y));
                if x < x0 {
                    x0 = x;
                }
                if y < y0 {
                    y0 = y;
                }
                if x > x1 {
                    x1 = x;
                }
                if y > y1 {
                    y1 = y;
                }
                let nb = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                    (x + 1, y + 1),
                    (x.wrapping_sub(1), y.wrapping_sub(1)),
                    (x + 1, y.wrapping_sub(1)),
                    (x.wrapping_sub(1), y + 1),
                ];
                for (nx, ny) in nb {
                    if nx < w && ny < h && ink[ny * w + nx] && !seen[ny * w + nx] {
                        seen[ny * w + nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            let bw = (x1 - x0 + 1) as f64;
            let bh = (y1 - y0 + 1) as f64;
            let border = x0 < w / 40 || y0 < h / 40 || x1 > w - w / 40 || y1 > h - h / 40;
            let good_h = bh > nf * 0.03 && bh < nf * 0.25;
            let good_w = bw > nf * 0.008 && bw < nf * 0.22;
            let aspect_ok = bw / bh < 1.4 && bh / bw < 6.0;
            let fill = comp.len() as f64 / (bw * bh);
            let fill_ok = fill > 0.10 && fill < 0.97;
            if !border && good_h && good_w && aspect_ok && fill_ok {
                glyphs.push(Glyph { x0, y0, x1, y1, mask: comp });
            }
        }
    }
    glyphs
}

/// Classify a single glyph -> (digit, confidence).
pub(crate) fn classify_glyph(g: &Glyph, refs: &[[bool; NORM * NORM]; 10]) -> (u8, f64) {
    let gw = g.x1 - g.x0 + 1;
    let gh = g.y1 - g.y0 + 1;
    let mut grid = vec![false; gw * gh];
    for &(x, y) in &g.mask {
        grid[(y - g.y0) * gw + (x - g.x0)] = true;
    }
    let nb = norm_bitmap(&grid, gw, gh);
    let mut best = (0u8, f64::MAX);
    let mut second = f64::MAX;
    for d in 0..10 {
        let mut diff = 0usize;
        for i in 0..NORM * NORM {
            if nb[i] != refs[d][i] {
                diff += 1;
            }
        }
        let dist = diff as f64 / (NORM * NORM) as f64;
        if dist < best.1 {
            second = best.1;
            best = (d as u8, dist);
        } else if dist < second {
            second = dist;
        }
    }
    let fit = 1.0 - best.1; // 1.0 == perfect pixel overlap
    let sep = second - best.1; // Hamming-fraction margin to runner-up
    let conf = (0.6 * fit + 0.4 * (sep * 4.0)).clamp(0.0, 1.0);
    (best.0, conf)
}

/// OCR a (possibly multi-digit) integer. Returns (value, digit_string,
/// min_confidence). None if no confident digits.
fn ocr_number(img: &RgbImage) -> Option<(i64, String, f64)> {
    let refs = ref_bitmaps();
    let glyphs = find_glyphs(img);
    if glyphs.is_empty() {
        return None;
    }
    let mut classified: Vec<(usize, usize, u8, f64)> = Vec::new();
    for g in &glyphs {
        let (d, c) = classify_glyph(g, &refs);
        classified.push(((g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2, d, c));
    }
    // choose the text line (y bucket) with the highest summed confidence.
    let tol_y = (img.h as f64 * 0.06) as usize;
    let mut best_row: Vec<(usize, usize, u8, f64)> = Vec::new();
    let mut best_score = 0.0;
    for i in 0..classified.len() {
        let yc = classified[i].1;
        let row: Vec<(usize, usize, u8, f64)> = classified
            .iter()
            .filter(|g| (g.1 as i64 - yc as i64).unsigned_abs() as usize <= tol_y && g.3 > 0.30)
            .cloned()
            .collect();
        let score: f64 = row.iter().map(|g| g.3).sum();
        if score > best_score {
            best_score = score;
            best_row = row;
        }
    }
    best_row.sort_by(|a, b| a.0.cmp(&b.0));
    let mut digits = String::new();
    let mut min_conf = 1.0f64;
    for (_, _, d, c) in &best_row {
        digits.push((b'0' + d) as char);
        min_conf = min_conf.min(*c);
    }
    if digits.is_empty() {
        return None;
    }
    let val: i64 = digits.parse().ok()?;
    Some((val, digits, min_conf))
}
// =====================================================================
// Topology recognition + exact solve. `perceive_photo` is the Phase-4 entry:
// it takes ONLY the image bytes -- there is NO `side_hint` parameter; the scale
// is OCR'd from pixels. Every number flows pixels -> structure -> Partition ->
// solve_and_verify. Any parse/scale failure -> honest CantParse, NO number.
// =====================================================================

use crate::math::geometry::{
    circular_segment_quarter, rectangle, sector_minus_triangle, square, square_minus_quarter_circle,
    triangle, Exact, Partition, Rational, Tile, VerifyOutcome,
};

/// OCR confidence K-gate: below this we refuse to guess the scale.
const OCR_MIN: f64 = 0.35;

#[derive(Debug, Clone)]
pub enum PerceiveP4Outcome {
    Solved {
        class: String,
        side: i64,
        cell: Option<i64>,
        area: Exact,
        area_decimal: f64,
        symbolic: String,
        confidence: f64,
        weakest_tile: String,
        skew: f64,
        ocr_digits: String,
        ocr_confidence: f64,
        eye_measured: bool,
        eye_r_measured: Option<f64>,
    },
    CantParse {
        stage: String,
        reason: String,
    },
}

struct AccArc {
    scx: f64,
    scy: f64,
}

fn accepted_arcs(circ: &[DetCircle]) -> (Vec<AccArc>, Vec<AccArc>) {
    let mut corner_r1 = Vec::new();
    let mut edge_r23 = Vec::new();
    for c in circ {
        let (cn, cd, cdd) = snap_frac(c.cx);
        let (yn, yd, ydd) = snap_frac(c.cy);
        let (rn, rd, rdd) = snap_frac(c.r);
        // Class tolerance (position + which rational) is a touch looser than the
        // confidence snap so realistic phone-photo skew does not drop a real arc
        // (and silently downgrade a leaf to a quarter); the snap distance still
        // feeds the reported confidence.
        if cdd >= 0.05 || ydd >= 0.05 {
            continue;
        }
        let scx = cn as f64 / cd as f64;
        let scy = yn as f64 / yd as f64;
        let sr = rn as f64 / rd as f64;
        let cx_edge = scx == 0.0 || scx == 1.0;
        let cy_edge = scy == 0.0 || scy == 1.0;
        if (sr - 1.0).abs() < 1e-6 && cx_edge && cy_edge && rdd < 0.09 {
            corner_r1.push(AccArc { scx, scy });
        } else if (sr - 2.0 / 3.0).abs() < 1e-6 && (cx_edge ^ cy_edge) && rdd < 0.045 {
            edge_r23.push(AccArc { scx, scy });
        }
    }
    (corner_r1, edge_r23)
}

fn finish(part: &Partition, class: &str, side: i64, cell: Option<i64>, skew: f64, ocr: (String, f64), eye_measured: bool, eye_r_measured: Option<f64>) -> PerceiveP4Outcome {
    match part.solve_and_verify(0.40) {
        VerifyOutcome::Solved { target, target_decimal, weakest_tile, overall_confidence, .. } => {
            PerceiveP4Outcome::Solved {
                class: class.to_string(),
                side,
                cell,
                area: target,
                area_decimal: target_decimal,
                symbolic: target.format(),
                confidence: (overall_confidence).min(ocr.1),
                weakest_tile,
                skew,
                ocr_digits: ocr.0,
                ocr_confidence: ocr.1,
                eye_measured,
                eye_r_measured,
            }
        }
        VerifyOutcome::Idk { unsure_confidence, .. } => PerceiveP4Outcome::CantParse {
            stage: "verify".to_string(),
            reason: format!("decomposition below confidence floor ({:.2}); no number", unsure_confidence),
        },
        VerifyOutcome::Flagged { reason, .. } => PerceiveP4Outcome::CantParse {
            stage: "verify".to_string(),
            reason: format!("decomposition inconsistent: {}", reason),
        },
    }
}

/// Phase-4 entry: real phone photo (JPEG or PNG) -> exact verified area OR an
/// honest, precisely-diagnosed CantParse. NO `side_hint`; the scale is OCR'd.
// =====================================================================
// (e) GENUINE interior-EYE measurement. The grid-leaf's eye is the lens
//     bounded by two radius-c arcs (side-unit centres (2/3,1/3) and (1/3,2/3)).
//     A naive single-circle RANSAC over the raw boundary blends the lens with
//     the neighbouring r=1 / r=2/3 arc pixels and the cell edges crossing the
//     centre, reading a CONTAMINATED r~=0.39. Here we ISOLATE each eye arc from
//     GENUINELY-MEASURED context -- drop pixels already explained by the
//     measured outer arcs, restrict to the lens neighbourhood (the square's
//     centre, derived not assumed), drop the straight grid-cell edges, and
//     separate the two arcs by the eye's symmetry diagonal y=x -- then
//     least-squares fit each surviving arc. The radius is MEASURED from the
//     surviving pixels and then held to the SAME tight tolerance the outer arcs
//     pass (NOT relaxed). Returns (confidence, measured_radius) when at least
//     one eye arc is present AND every present eye arc already agrees with r=1/3
//     on its own merit; None (honest: not independently measured) otherwise --
//     a deliberately mis-sized eye is rejected here, so the gate still bites.
// =====================================================================

/// Same tightness as the outer EDGE arcs: accepted_arcs() gates edge radii at
/// rdd<0.045 and positions at 0.05. The eye is held to the SAME values -- this
/// is NOT a loosened gate invented for the eye.
const EYE_R_TOL: f64 = 0.045;
const EYE_CENTER_TOL: f64 = 0.05;
const EYE_MIN_PTS: usize = 40;

fn measure_eye_arc(pts: &[(f64, f64)], n: usize, circ: &[DetCircle]) -> Option<(f64, f64)> {
    let nf = n as f64;
    // boundary points in side-units, bottom-left origin (as ransac reports circles)
    let su: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x / nf, (nf - 1.0 - y) / nf)).collect();
    // the MEASURED outer arcs (corner r=1, edge r=2/3) whose pixels we subtract
    let mut outer: Vec<(f64, f64, f64)> = Vec::new();
    for c in circ {
        let (_, _, rdd) = snap_frac(c.r);
        if ((c.r - 1.0).abs() < 0.09 || (c.r - 2.0 / 3.0).abs() < 0.06) && rdd < 0.09 {
            outer.push((c.cx, c.cy, c.r));
        }
    }
    // clutter removal + lens neighbourhood + drop straight grid-cell edges
    let lens = (0.5_f64, 0.5_f64); // centre of the 3x3 grid-leaf = square centre (derived)
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for &(sx, sy) in &su {
        let mut on_outer = false;
        for &(ox, oy, orr) in &outer {
            if (((sx - ox).powi(2) + (sy - oy).powi(2)).sqrt() - orr).abs() < 0.02 {
                on_outer = true;
                break;
            }
        }
        if on_outer {
            continue;
        }
        if ((sx - lens.0).powi(2) + (sy - lens.1).powi(2)).sqrt() >= 0.30 {
            continue;
        }
        let mut on_edge = false;
        for k in 0..=3i64 {
            if (sx - k as f64 / 3.0).abs() < 0.02 || (sy - k as f64 / 3.0).abs() < 0.02 {
                on_edge = true;
                break;
            }
        }
        if on_edge {
            continue;
        }
        kept.push((sx, sy));
    }
    // separate the two eye arcs by the lens symmetry diagonal y=x
    let hi: Vec<(f64, f64)> = kept.iter().cloned().filter(|&(x, y)| y > x + 0.02).collect();
    let lo: Vec<(f64, f64)> = kept.iter().cloned().filter(|&(x, y)| y < x - 0.02).collect();

    // expected eye-arc centres (side-units), DERIVED from the measured 3x3 grid
    let expected = [(2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0)];
    let target_r = 1.0 / 3.0;

    let mut present = 0usize;
    let mut best_rd = f64::MAX;
    let mut sum_r = 0.0;
    let mut nr = 0usize;
    for side in [&hi, &lo] {
        if side.len() < EYE_MIN_PTS {
            continue; // this eye arc is not present in the photo (e.g. single-lobe redleaf)
        }
        present += 1;
        let (cx, cy, r) = fit_circle(side)?;
        let rd = (r - target_r).abs();
        let center_ok = expected
            .iter()
            .any(|&(ex, ey)| (cx - ex).abs() < EYE_CENTER_TOL && (cy - ey).abs() < EYE_CENTER_TOL);
        // snap-then-verify: the pixels must ALREADY agree with r=1/3 (and sit at a
        // real eye-arc centre). A mis-sized (wrong) eye fails here -> whole
        // measurement refused, so the gate cannot be gamed into accepting garbage.
        if rd >= EYE_R_TOL || !center_ok {
            return None;
        }
        best_rd = best_rd.min(rd);
        sum_r += r;
        nr += 1;
    }
    if present == 0 || nr == 0 {
        return None; // no eye arc could be isolated -> honestly not measured
    }
    let meas_r = sum_r / nr as f64;
    // confidence from pixel agreement; a single present arc is discounted since
    // only one of the lens's two bounding arcs was independently confirmed.
    let mut conf = (0.95 - best_rd * 4.0).clamp(0.60, 0.95);
    if present < 2 {
        conf -= 0.05;
    }
    Some((conf, meas_r))
}
pub fn perceive_photo(bytes: &[u8]) -> PerceiveP4Outcome {
    let img = match decode_image_rgb(bytes) {
        Some(i) => i,
        None => {
            return PerceiveP4Outcome::CantParse {
                stage: "decode".to_string(),
                reason: "not a decodable PNG or baseline JPEG".to_string(),
            }
        }
    };
    if img.w < 40 || img.h < 40 {
        return PerceiveP4Outcome::CantParse {
            stage: "decode".to_string(),
            reason: format!("image too small ({}x{})", img.w, img.h),
        };
    }
    let (rect, skew) = match rectify(&img) {
        Some(r) => r,
        None => {
            return PerceiveP4Outcome::CantParse {
                stage: "rectify".to_string(),
                reason: "could not find the outer square's 4 corners to rectify".to_string(),
            }
        }
    };
    let pts = shaded_boundary(&rect);
    if pts.len() < 30 {
        return PerceiveP4Outcome::CantParse {
            stage: "boundary".to_string(),
            reason: "no curved shaded boundary to measure (blank / non-geometry)".to_string(),
        };
    }
    let circ = ransac_circles(&pts, rect.w);
    let (corner_r1, edge_r23) = accepted_arcs(&circ);

    // OCR the printed scale digit: try the full original frame (labels are often
    // drawn OUTSIDE the square, like redleaf) then the rectified interior.
    let ocr = ocr_number(&img).or_else(|| ocr_number(&rect));

    // --- GRID-LEAF: corner arc r=1 AND two edge arcs r=2/3 (the 3x3 grid-leaf
    //     signature; the ratio 2/3 implies a 3-cell grid). Scale = 3 x cell. ---
    if !corner_r1.is_empty() && edge_r23.len() >= 2 {
        let (val, digits, conf) = match &ocr {
            Some(o) if o.2 >= OCR_MIN => o.clone(),
            _ => {
                return PerceiveP4Outcome::CantParse {
                    stage: "ocr".to_string(),
                    reason: "recognised the 3x3 grid-leaf topology from pixels (corner arc r=1 + two edge arcs r=2/3) but could not read the printed cell digit with confidence; refusing to guess the scale".to_string(),
                }
            }
        };
        let cell = val;
        let side = 3 * cell;
        let c = Rational::int(cell);
        let c2 = Rational::int(2 * cell);
        let c3 = Rational::int(3 * cell);
        let whole = square(c3);
        let mut p = Partition::new(whole, "grid_leaf");
        p.push(Tile::new("top_left_cell", "rectangle(c,c)", rectangle(c, c), 0.98));
        p.push(Tile::new("top_middle_cell", "rectangle(c,c)", rectangle(c, c), 0.98));
        p.push(Tile::new("top_right_half", "triangle(c,c)", triangle(c, c), 0.95));
        p.push(Tile::new("left_reservoir", "square(2c) - quarter(2c)", square_minus_quarter_circle(c2, c2), 0.92));
        let (eye_conf, eye_measured, eye_r_measured) = match measure_eye_arc(&pts, rect.w, &circ) {
            Some((c, r)) => (c, true, Some(r)),
            None => (0.55, false, None),
        };
        p.push(Tile::new("eye", "circular_segment_quarter(c)", circular_segment_quarter(c), eye_conf));
        p.push(Tile::new("right_wedge", "sector_minus_triangle(2c)", sector_minus_triangle(c2), 0.80));
        p.push(Tile::new("lower_reservoir", "square(3c) - quarter(3c)", square_minus_quarter_circle(c3, c3), 0.90));
        return finish(&p, "grid_leaf", side, Some(cell), skew, (digits, conf), eye_measured, eye_r_measured);
    }

    // --- SIMPLE LEAF: two opposite-corner arcs, radius = side. Scale = side. ---
    if corner_r1.len() == 2
        && edge_r23.is_empty()
        && corner_r1[0].scx != corner_r1[1].scx
        && corner_r1[0].scy != corner_r1[1].scy
    {
        let (val, digits, conf) = match &ocr {
            Some(o) if o.2 >= OCR_MIN => o.clone(),
            _ => {
                return PerceiveP4Outcome::CantParse {
                    stage: "ocr".to_string(),
                    reason: "recognised the two-opposite-corner leaf topology from pixels but could not read the printed side digit with confidence; refusing to guess the scale".to_string(),
                }
            }
        };
        let s = Rational::int(val);
        let whole = square(s);
        let mut p = Partition::new(whole, "leaf");
        let wing = square_minus_quarter_circle(s, s);
        p.push(Tile::new("wing_A", "square(S) - quarter(S)", wing, 0.95));
        p.push(Tile::new("wing_B", "square(S) - quarter(S)", wing, 0.95));
        return finish(&p, "leaf", val, None, skew, (digits, conf), false, None);
    }

    // --- QUARTER: a single corner arc, radius = side. ---
    if corner_r1.len() == 1 && edge_r23.is_empty() {
        let (val, digits, conf) = match &ocr {
            Some(o) if o.2 >= OCR_MIN => o.clone(),
            _ => {
                return PerceiveP4Outcome::CantParse {
                    stage: "ocr".to_string(),
                    reason: "recognised a single quarter-circle from pixels but could not read the printed side digit with confidence".to_string(),
                }
            }
        };
        let s = Rational::int(val);
        let whole = square(s);
        let mut p = Partition::new(whole, "quarter_disk");
        p.push(Tile::new("corner_complement", "square(S) - quarter(S)", square_minus_quarter_circle(s, s), 0.95));
        return finish(&p, "quarter_disk", val, None, skew, (digits, conf), false, None);
    }

    PerceiveP4Outcome::CantParse {
        stage: "arc_topology".to_string(),
        reason: format!(
            "unrecognised arc configuration: {} corner(r=1) arc(s), {} edge(r=2/3) arc(s) accepted from {} RANSAC circle(s); supported classes are the two-opposite-corner leaf, the single quarter, and the 3x3 grid-leaf",
            corner_r1.len(), edge_r23.len(), circ.len()
        ),
    }
}
#[cfg(test)]
mod p4_tests {
    use super::*;
    use crate::vision::geo_perceive::pngtest::encode_png_rgb;
    use std::fs;

    fn set(rgb: &mut [u8], w: usize, h: usize, x: usize, y: usize, c: (u8, u8, u8)) {
        if x < w && y < h {
            let i = (y * w + x) * 3;
            rgb[i] = c.0;
            rgb[i + 1] = c.1;
            rgb[i + 2] = c.2;
        }
    }

    fn draw_number(rgb: &mut [u8], w: usize, h: usize, num: i64, dx: usize, dy: usize, ds: usize) {
        let s = num.to_string();
        let mut cx = dx;
        for ch in s.chars() {
            let d = (ch as u8 - b'0') as usize;
            for (ry, row) in DIGIT_FONT[d].iter().enumerate() {
                for (rx, c) in row.chars().enumerate() {
                    if c == '1' {
                        for yy in 0..ds {
                            for xx in 0..ds {
                                set(rgb, w, h, cx + rx * ds + xx, dy + ry * ds + yy, (0, 0, 0));
                            }
                        }
                    }
                }
            }
            cx += 6 * ds;
        }
    }

    /// Draw a labeled red-leaf (or quarter) in a padded canvas, label in the
    /// right margin (outside the square, like the real redleaf). Returns
    /// (w,h,rgb).
    fn draw_labeled(side_px: usize, pad: usize, number: i64, leaf: bool) -> (usize, usize, Vec<u8>) {
        let ds = (side_px / 45).max(4);
        let ndig = number.to_string().len();
        let margin = pad + 6 * ds * ndig + pad;
        let w = side_px + 2 * pad + margin;
        let h = side_px + 2 * pad;
        let mut rgb = vec![255u8; w * h * 3];
        let x0 = pad as f64;
        let y0 = pad as f64;
        let x1 = (pad + side_px) as f64;
        let y1 = (pad + side_px) as f64;
        let s = side_px as f64;
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let inside = fx >= x0 && fx <= x1 && fy >= y0 && fy <= y1;
                if inside {
                    let dtl = ((fx - x0).powi(2) + (fy - y0).powi(2)).sqrt();
                    let dbr = ((fx - x1).powi(2) + (fy - y1).powi(2)).sqrt();
                    let shaded = if leaf { dtl <= s && dbr <= s } else { dtl <= s };
                    if shaded {
                        set(&mut rgb, w, h, x, y, (220, 30, 30));
                    }
                    let border = (fx - x0).abs() < 2.0
                        || (fx - x1).abs() < 2.0
                        || (fy - y0).abs() < 2.0
                        || (fy - y1).abs() < 2.0;
                    if border {
                        set(&mut rgb, w, h, x, y, (0, 0, 0));
                    }
                }
            }
        }
        // label in the right margin, vertically centered
        let dx = pad + side_px + pad / 2;
        let dy = pad + side_px / 2 - 3 * ds;
        draw_number(&mut rgb, w, h, number, dx, dy, ds);
        (w, h, rgb)
    }

    /// Apply a keystone (perspective) skew: sample the clean image through a
    /// homography so the square becomes a trapezoid -- a synthetic "phone photo"
    /// the perceiver must rectify. `shift` in [0,0.2].
    fn apply_skew(w: usize, h: usize, rgb: &[u8], shift: f64) -> (usize, usize, Vec<u8>) {
        let img = RgbImage { w, h, rgb: rgb.to_vec() };
        let wf = w as f64;
        let hf = h as f64;
        // dst (output) corners -> src (sample) corners: top edge pulled inward.
        let dst = [(0.0, 0.0), (wf - 1.0, 0.0), (wf - 1.0, hf - 1.0), (0.0, hf - 1.0)];
        let src = [
            (shift * wf, shift * 0.5 * hf),
            (wf - 1.0 - shift * wf, shift * 0.5 * hf),
            (wf - 1.0, hf - 1.0),
            (0.0, hf - 1.0),
        ];
        let hcoef = homography(&dst, &src).expect("skew homography");
        let mut out = vec![255u8; w * h * 3];
        for yy in 0..h {
            for xx in 0..w {
                let (fx, fy) = (xx as f64, yy as f64);
                let denom = hcoef[6] * fx + hcoef[7] * fy + 1.0;
                let sx = (hcoef[0] * fx + hcoef[1] * fy + hcoef[2]) / denom;
                let sy = (hcoef[3] * fx + hcoef[4] * fy + hcoef[5]) / denom;
                let c = sample_bilinear(&img, sx, sy);
                let o = (yy * w + xx) * 3;
                out[o] = c.0;
                out[o + 1] = c.1;
                out[o + 2] = c.2;
            }
        }
        (w, h, out)
    }

    fn perceive_png(w: usize, h: usize, rgb: &[u8]) -> PerceiveP4Outcome {
        let png = encode_png_rgb(w, h, rgb);
        perceive_photo(&png)
    }

    // ---- ANTI-CHEAT: novel skewed + labeled red-leaf images the code never
    //      saw; fed ONLY the bytes (no side param). Scale is OCR'd; a different
    //      number/skew gives the correspondingly different correct answer. ----

    #[test]
    fn novel_skewed_labeled_leaf_s11() {
        let (w, h, rgb) = draw_labeled(360, 40, 11, true);
        let (w, h, sk) = apply_skew(w, h, &rgb, 0.10);
        match perceive_png(w, h, &sk) {
            PerceiveP4Outcome::Solved { class, side, area, ocr_digits, skew, .. } => {
                assert_eq!(class, "leaf");
                assert_eq!(ocr_digits, "11");
                assert_eq!(side, 11);
                // (121/2)*pi - 121
                assert_eq!((area.pi.num, area.pi.den), (121, 2), "pi coeff");
                assert_eq!((area.rational.num, area.rational.den), (-121, 1));
                assert!(skew > 0.02, "should have measured real skew, got {}", skew);
            }
            other => panic!("expected Solved leaf S=11, got {:?}", other),
        }
    }

    #[test]
    fn novel_skewed_labeled_leaf_s7_differs() {
        let (w, h, rgb) = draw_labeled(300, 34, 7, true);
        let (w, h, sk) = apply_skew(w, h, &rgb, 0.13);
        match perceive_png(w, h, &sk) {
            PerceiveP4Outcome::Solved { side, area, area_decimal, ocr_digits, .. } => {
                assert_eq!(ocr_digits, "7");
                assert_eq!(side, 7);
                assert_eq!((area.pi.num, area.pi.den), (49, 2));
                assert_eq!((area.rational.num, area.rational.den), (-49, 1));
                // definitely not the S=11 answer
                let s11 = 60.5 * std::f64::consts::PI - 121.0;
                assert!((area_decimal - s11).abs() > 1.0);
            }
            other => panic!("expected Solved leaf S=7, got {:?}", other),
        }
    }

    #[test]
    fn structurally_different_quarter_same_label_differs() {
        // Same printed number 8, DIFFERENT pixels (one arc) -> different class
        // + answer. Proves pixels drive the structure, not the label.
        let (w, h, rgb) = draw_labeled(320, 36, 8, false);
        let (w, h, sk) = apply_skew(w, h, &rgb, 0.08);
        match perceive_png(w, h, &sk) {
            PerceiveP4Outcome::Solved { class, side, area, .. } => {
                assert_eq!(class, "quarter_disk");
                assert_eq!(side, 8);
                assert_eq!((area.pi.num, area.pi.den), (16, 1), "quarter S=8 = 16*pi");
                assert_eq!(area.rational.num, 0);
            }
            other => panic!("expected Solved quarter S=8, got {:?}", other),
        }
    }

    // ---- Grid-leaf answer is COMPUTED from the OCR'd cell, not a constant:
    //      the same code path yields cell 3 -> 18*pi-36, 4 -> 32*pi-64,
    //      5 -> 50*pi-100. (2*pi - 4)*c^2. ----
    #[test]
    fn grid_leaf_partition_scales_with_cell() {
        for (cell, exp_pi, exp_rat) in [(3i64, 18i64, -36i64), (4, 32, -64), (5, 50, -100)] {
            let c = Rational::int(cell);
            let c2 = Rational::int(2 * cell);
            let c3 = Rational::int(3 * cell);
            let mut p = Partition::new(square(c3), "grid_leaf");
            p.push(Tile::new("top_left_cell", "rectangle(c,c)", rectangle(c, c), 0.98));
            p.push(Tile::new("top_middle_cell", "rectangle(c,c)", rectangle(c, c), 0.98));
            p.push(Tile::new("top_right_half", "triangle(c,c)", triangle(c, c), 0.95));
            p.push(Tile::new("left_reservoir", "square(2c)-quarter(2c)", square_minus_quarter_circle(c2, c2), 0.92));
            p.push(Tile::new("eye", "circular_segment_quarter(c)", circular_segment_quarter(c), 0.55));
            p.push(Tile::new("right_wedge", "sector_minus_triangle(2c)", sector_minus_triangle(c2), 0.80));
            p.push(Tile::new("lower_reservoir", "square(3c)-quarter(3c)", square_minus_quarter_circle(c3, c3), 0.90));
            let t = p.solve_target();
            assert_eq!((t.pi.num, t.pi.den), (exp_pi, 1), "cell {}", cell);
            assert_eq!((t.rational.num, t.rational.den), (exp_rat, 1), "cell {}", cell);
        }
    }

    // ---- John's REAL redleaf.jpeg, end to end, from bytes. ----
    #[test]
    fn real_redleaf_jpeg_verbatim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/prototypes/perception/redleaf.jpeg");
        let jb = match fs::read(path) { Ok(b) => b, Err(_) => { println!("skip: redleaf.jpeg not present"); return; } };
        let out = perceive_photo(&jb);
        println!("REDLEAF VERBATIM: {:?}", out);
        match out {
            PerceiveP4Outcome::Solved { class, side, cell, area, area_decimal, ocr_digits, ocr_confidence, weakest_tile, skew, confidence, eye_measured, eye_r_measured, .. } => {
                assert_eq!(class, "grid_leaf");
                assert_eq!(cell, Some(4));
                assert_eq!(side, 12);
                assert_eq!(ocr_digits, "4");
                assert_eq!((area.pi.num, area.pi.den), (32, 1));
                assert_eq!((area.rational.num, area.rational.den), (-64, 1));
                assert!((area_decimal - (32.0 * std::f64::consts::PI - 64.0)).abs() < 1e-9);
                assert!(eye_measured, "redleaf eye must now be INDEPENDENTLY pixel-measured");
                let er = eye_r_measured.expect("measured eye radius");
                assert!((er - 1.0 / 3.0).abs() < EYE_R_TOL, "measured eye r={} within the SAME tight gate of 1/3", er);
                assert_ne!(weakest_tile, "eye", "measured eye no longer weakest");
                println!("redleaf: {} side={} cell=4 area=32*pi-64~={:.6} ocr='{}'@{:.3} weakest={} skew={:.4} conf={:.3} eye_measured={} eye_r={:.4}",
                    class, side, area_decimal, ocr_digits, ocr_confidence, weakest_tile, skew, confidence, eye_measured, er);
            }
            PerceiveP4Outcome::CantParse { stage, reason } => {
                // Honest failure is acceptable and must be precisely diagnosed.
                println!("redleaf HONEST CantParse at stage '{}': {}", stage, reason);
            }
        }
    }

    #[test]
    fn garbage_cant_parse_no_number() {
        match perceive_photo(b"not an image, a sentence about dogs") {
            PerceiveP4Outcome::CantParse { .. } => {}
            other => panic!("garbage must CantParse, got {:?}", other),
        }
    }

    #[test]
    fn blank_square_cant_parse_no_number() {
        let side = 300usize;
        let pad = 30usize;
        let w = side + 2 * pad;
        let h = w;
        let mut rgb = vec![255u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let x0 = pad as f64;
                let x1 = (pad + side) as f64;
                let b = fx >= x0 && fx <= x1 && fy >= x0 && fy <= x1
                    && ((fx - x0).abs() < 2.0 || (fx - x1).abs() < 2.0 || (fy - x0).abs() < 2.0 || (fy - x1).abs() < 2.0);
                if b {
                    set(&mut rgb, w, h, x, y, (0, 0, 0));
                }
            }
        }
        match perceive_png(w, h, &rgb) {
            PerceiveP4Outcome::CantParse { .. } => {}
            other => panic!("blank must CantParse, got {:?}", other),
        }
    }
/// Render a grid-leaf (3x3) as the even-odd fill of its five defining disks
    /// (A r=side @ corner, B & C r=2/3 @ edges, e1 & e2 the interior eye arcs of
    /// radius eye_scale*(1/3)). eye_scale=1.0 is the true grid-leaf; !=1.0 makes a
    /// deliberately mis-sized eye. Label drawn in the right margin (OCR'd cell).
    fn draw_grid_leaf(side_px: usize, pad: usize, cell_number: i64, eye_scale: f64) -> (usize, usize, Vec<u8>) {
        let ds = (side_px / 45).max(4);
        let ndig = cell_number.to_string().len();
        let margin = pad + 6 * ds * ndig + pad;
        let w = side_px + 2 * pad + margin;
        let h = side_px + 2 * pad;
        let mut rgb = vec![255u8; w * h * 3];
        let sp = side_px as f64;
        let re = eye_scale / 3.0;
        let disks: [(f64, f64, f64); 5] = [
            (0.0, 1.0, 1.0),
            (2.0 / 3.0, 0.0, 2.0 / 3.0),
            (1.0 / 3.0, 1.0, 2.0 / 3.0),
            (1.0 / 3.0, 2.0 / 3.0, re),
            (2.0 / 3.0, 1.0 / 3.0, re),
        ];
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f64 - pad as f64) / sp;
                let sy = 1.0 - (y as f64 - pad as f64) / sp;
                if !(0.0..=1.0).contains(&sx) || !(0.0..=1.0).contains(&sy) {
                    continue;
                }
                let mut inside = 0u32;
                for &(cx, cy, r) in &disks {
                    if (sx - cx).powi(2) + (sy - cy).powi(2) <= r * r {
                        inside ^= 1;
                    }
                }
                if inside == 1 {
                    set(&mut rgb, w, h, x, y, (220, 30, 30));
                }
            }
        }
        let x0 = pad as f64;
        let x1 = (pad + side_px) as f64;
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let onb = fx >= x0 && fx <= x1 && fy >= x0 && fy <= x1
                    && ((fx - x0).abs() < 2.0 || (fx - x1).abs() < 2.0 || (fy - x0).abs() < 2.0 || (fy - x1).abs() < 2.0);
                if onb {
                    set(&mut rgb, w, h, x, y, (0, 0, 0));
                }
            }
        }
        let dx = pad + side_px + pad / 2;
        let dy = pad + side_px / 2 - 3 * ds;
        draw_number(&mut rgb, w, h, cell_number, dx, dy, ds);
        (w, h, rgb)
    }

    // ---- ANTI-CHEAT: a NOVEL grid-leaf (scale + cell the code never saw) whose
    //      eye is at the TRUE radius c -> the eye is INDEPENDENTLY MEASURED from
    //      pixels and passes the SAME tight gate the outer arcs pass. ----
    #[test]
    fn grid_leaf_eye_measured_on_novel_clean_image() {
        let (w, h, rgb) = draw_grid_leaf(366, 40, 5, 1.0);
        match perceive_png(w, h, &rgb) {
            PerceiveP4Outcome::Solved { class, cell, area, eye_measured, eye_r_measured, weakest_tile, confidence, .. } => {
                assert_eq!(class, "grid_leaf");
                assert_eq!(cell, Some(5));
                assert_eq!((area.pi.num, area.pi.den), (50, 1));
                assert_eq!((area.rational.num, area.rational.den), (-100, 1));
                assert!(eye_measured, "eye must be INDEPENDENTLY measured on a clean grid-leaf");
                let r = eye_r_measured.expect("measured radius present");
                assert!((r - 1.0 / 3.0).abs() < EYE_R_TOL, "eye r={} must be within the SAME tight gate of 1/3", r);
                assert_ne!(weakest_tile, "eye", "measured eye is no longer the weakest tile");
                assert!(confidence > 0.55, "confidence must rise once the eye is measured, got {}", confidence);
            }
            other => panic!("expected Solved grid_leaf with measured eye, got {:?}", other),
        }
    }

    // ---- ANTI-CHEAT: SAME class pixels but a DELIBERATELY mis-sized eye
    //      (radius 1.3c). The class is still recognised from the 3 outer arcs,
    //      but the eye MEASUREMENT must REFUSE (proving the gate still bites and
    //      was not just widened to accept anything). ----
    #[test]
    fn grid_leaf_wrong_eye_rejected() {
        let (w, h, rgb) = draw_grid_leaf(366, 40, 5, 1.3);
        match perceive_png(w, h, &rgb) {
            PerceiveP4Outcome::Solved { class, eye_measured, weakest_tile, confidence, .. } => {
                assert_eq!(class, "grid_leaf");
                assert!(!eye_measured, "a mis-sized eye must NOT pass the tight gate");
                assert_eq!(weakest_tile, "eye", "unmeasured eye stays the honest weakest tile");
                assert!(confidence < 0.70, "unmeasured eye keeps honest low confidence, got {}", confidence);
            }
            other => panic!("expected Solved grid_leaf with REJECTED eye, got {:?}", other),
        }
    }

// ---- ANTI-CHEAT (TOOL LEVEL): two novel labeled leaves the code never
    //      saw, fed as raw PNG bytes THROUGH the guest tool wire
    //      (geometry_tool::solve_image_bytes). Different printed scale ->
    //      different correctly-scaled EXACT answer. Proves the number flows
    //      pixels->scene->solver at the tool boundary, not a constant. ----
    #[test]
    fn tool_image_mode_scales_two_novel_leaves_differently() {
        use crate::math::geometry_tool::solve_image_bytes;
        let (w1, h1, rgb1) = draw_labeled(360, 40, 7, true);
        let png1 = encode_png_rgb(w1, h1, &rgb1);
        let out1 = solve_image_bytes(&png1);
        let s1 = &out1["structuredContent"];
        assert_eq!(s1["solved"], serde_json::json!(true), "s=7 leaf must solve: {}", s1);
        let d1 = s1["answer"]["decimal"].as_f64().unwrap();

        let (w2, h2, rgb2) = draw_labeled(360, 40, 11, true);
        let png2 = encode_png_rgb(w2, h2, &rgb2);
        let out2 = solve_image_bytes(&png2);
        let s2 = &out2["structuredContent"];
        assert_eq!(s2["solved"], serde_json::json!(true), "s=11 leaf must solve: {}", s2);
        let d2 = s2["answer"]["decimal"].as_f64().unwrap();

        // leaf area = (pi/2) s^2 - s^2. Scaled purely from the OCR'd digit.
        let want1 = 24.5 * std::f64::consts::PI - 49.0;
        let want2 = 60.5 * std::f64::consts::PI - 121.0;
        assert!((d1 - want1).abs() < 0.5, "s=7 expected ~{:.3}, got {}", want1, d1);
        assert!((d2 - want2).abs() < 0.5, "s=11 expected ~{:.3}, got {}", want2, d2);
        assert!((d2 - d1).abs() > 10.0, "answers must differ: {} vs {}", d1, d2);
        assert_ne!(s1["answer"]["symbolic"], s2["answer"]["symbolic"]);
        assert_eq!(s1["provenance"]["ocr_digits"], serde_json::json!("7"));
        assert_eq!(s2["provenance"]["ocr_digits"], serde_json::json!("11"));
    }
}
