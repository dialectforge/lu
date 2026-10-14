//! Port F.4 — construction calculator easter egg.
//!
//! When a query is shaped like a contractor would phrase it (feet-inches
//! tick marks, framing/lumber/concrete vocabulary), Lu switches into
//! construction-calculator idiom and answers in the same units a job
//! site would: feet-inches-fractions for lengths, board feet for lumber,
//! cubic yards for concrete (because that's what the truck delivers).
//!
//! Cascade order in `mcp::try_arithmetic_response` is
//!     F.1 single-op → F.4 construction → F.2 expression parser → recall
//! so a construction-shaped query gets first crack at construction. If
//! `looks_like_construction` returns false, or the dispatch fails to
//! match any of the nine capabilities, we fall through cleanly to F.2.
//!
//! Capabilities (9):
//!   1. feet-inches-fractions arithmetic (+ - * /)
//!   2. decimal-feet ↔ feet-inches conversion
//!   3. board feet (T × W × L ÷ 12, nominal sizing)
//!   4. wall framing — stud count at OC spacing
//!   5. sheet goods — 4×8 sheets from area
//!   6. rafter / hypotenuse — Pythagorean for roof framing
//!   7. roof pitch — N/12 ↔ slope angle in degrees
//!   8. concrete volume in cubic yards
//!   9. stair calcs — riser count + tread depth from total rise
//!
//! Subtle easter-egg flavor: every successful response tags itself with
//! `(construction calculator)` so the mode change is visible in the
//! prose; `source = "construction_primitive"` in structuredContent
//! distinguishes the route programmatically.

use super::fmt_num;
use serde::{Deserialize, Serialize};

// =====================================================================
// FeetInches — internal representation
// =====================================================================

/// A length in feet+inches, stored as total inches (f64) so arithmetic
/// is just f64 add/sub/mul.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeetInches {
    pub total_inches: f64,
}

impl FeetInches {
    pub fn from_inches(inches: f64) -> Self { Self { total_inches: inches } }
    pub fn from_feet(feet: f64) -> Self { Self { total_inches: feet * 12.0 } }
    pub fn from_feet_inches(feet: f64, inches: f64) -> Self {
        Self { total_inches: feet * 12.0 + inches }
    }
    pub fn feet_decimal(&self) -> f64 { self.total_inches / 12.0 }

    /// Format as `F'-I[ N/D]"` with fraction-of-inch reduced to lowest
    /// terms. Default granularity 1/16 inch (the closest sixteenth).
    pub fn format_fi(&self, granularity: u32) -> String {
        let neg = self.total_inches < 0.0;
        let total = self.total_inches.abs();
        let g = granularity.max(1) as f64;
        let units_total = (total * g).round() as i128;
        let g_i = granularity as i128;
        let total_inches_whole = units_total / g_i;
        let mut feet = (total_inches_whole / 12) as i64;
        let mut whole_inches = (total_inches_whole % 12) as i64;
        let mut frac_num = (units_total - total_inches_whole * g_i) as i64;
        let mut frac_den = granularity as i64;
        if frac_num > 0 {
            let d = gcd(frac_num as u64, frac_den as u64) as i64;
            frac_num /= d;
            frac_den /= d;
        }
        if frac_num == frac_den && frac_den > 0 {
            whole_inches += 1;
            frac_num = 0;
            frac_den = granularity as i64;
        }
        if whole_inches >= 12 {
            feet += whole_inches / 12;
            whole_inches %= 12;
        }
        let sign = if neg { "-" } else { "" };
        let frac_str = if frac_num > 0 {
            format!(" {}/{}", frac_num, frac_den)
        } else {
            String::new()
        };
        format!("{}{}'-{}{}\"", sign, feet, whole_inches, frac_str)
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// =====================================================================
// Parser for FI strings
// Accepts: 12'-3 1/2"  12'3.5"  12 ft 3.5 in  12'-3"
//          3-1/2"  3 1/2"  12.292'  12'  3"  1/2"
// =====================================================================

pub fn parse_fi(s: &str) -> Option<FeetInches> {
    let s = s.trim();
    if s.is_empty() { return None; }
    let normalized = normalize_units(s);
    let s = normalized.as_str();

    let (feet, rest) = if let Some(idx) = s.find('\'') {
        let feet_str = s[..idx].trim();
        let feet = parse_decimal_or_int(feet_str)?;
        let mut rest = s[idx + 1..].trim_start();
        if let Some(stripped) = rest.strip_prefix('-') {
            rest = stripped.trim_start();
        }
        (feet, rest)
    } else {
        (0.0, s)
    };

    let inches_str = rest.trim().trim_end_matches('\"').trim();
    let inches = if inches_str.is_empty() {
        0.0
    } else {
        parse_inches_with_fraction(inches_str)?
    };

    Some(FeetInches::from_feet_inches(feet, inches))
}

fn normalize_units(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut s2 = lower
        .replace(" feet", "'")
        .replace(" foot", "'")
        .replace(" ft.", "'")
        .replace(" ft", "'")
        .replace(" inches", "\"")
        .replace(" inch", "\"")
        .replace(" in.", "\"")
        .replace(" in ", "\" ");
    if let Some(pre) = s2.strip_suffix(" in") {
        s2 = format!("{}\"", pre);
    }
    s2
}

fn parse_decimal_or_int(s: &str) -> Option<f64> { s.trim().parse::<f64>().ok() }

fn parse_inches_with_fraction(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() { return None; }
    if let Some(idx) = s.find('-') {
        let whole_s = s[..idx].trim();
        let frac_s = s[idx + 1..].trim();
        if frac_s.contains('/') {
            if let (Ok(whole), Some(frac)) = (whole_s.parse::<f64>(), parse_fraction(frac_s)) {
                return Some(whole + frac);
            }
        }
    }
    if let Some(idx) = s.find(' ') {
        let whole_s = s[..idx].trim();
        let frac_s = s[idx + 1..].trim();
        if frac_s.contains('/') {
            if let (Ok(whole), Some(frac)) = (whole_s.parse::<f64>(), parse_fraction(frac_s)) {
                return Some(whole + frac);
            }
        }
    }
    if s.contains('/') { return parse_fraction(s); }
    s.parse::<f64>().ok()
}

fn parse_fraction(s: &str) -> Option<f64> {
    let s = s.trim();
    let mut parts = s.split('/');
    let n = parts.next()?.trim().parse::<f64>().ok()?;
    let d = parts.next()?.trim().parse::<f64>().ok()?;
    if parts.next().is_some() { return None; }
    if d == 0.0 { return None; }
    Some(n / d)
}

// =====================================================================
// ConstructionOutcome
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ConstructionOutcome {
    FeetInchesArith {
        op: String,
        left_inches: f64,
        right_inches_or_scalar: f64,
        result_inches: f64,
        formatted: String,
        prose: String,
    },
    FeetInchesConversion {
        decimal_feet: f64,
        formatted: String,
        prose: String,
    },
    BoardFeet {
        thickness_in: f64,
        width_in: f64,
        length_ft: f64,
        board_feet: f64,
        prose: String,
    },
    StudCount {
        wall_length_ft: f64,
        spacing_in: f64,
        count: u64,
        prose: String,
    },
    SheetGoods {
        area_sqft: f64,
        sheets: u64,
        prose: String,
    },
    Rafter {
        run_ft: f64,
        rise_ft: f64,
        rafter_ft: f64,
        formatted: String,
        prose: String,
    },
    Pitch {
        rise_in_12: f64,
        slope_deg: f64,
        prose: String,
    },
    Concrete {
        length_ft: f64,
        width_ft: f64,
        depth_in: f64,
        cubic_yards: f64,
        prose: String,
    },
    Stairs {
        total_rise_in: f64,
        risers: u64,
        riser_height_in: f64,
        treads: u64,
        tread_depth_in: f64,
        prose: String,
    },
    Error {
        route: String,
        error: String,
        prose: String,
    },
}

impl ConstructionOutcome {
    pub fn prose(&self) -> &str {
        match self {
            ConstructionOutcome::FeetInchesArith { prose, .. }
            | ConstructionOutcome::FeetInchesConversion { prose, .. }
            | ConstructionOutcome::BoardFeet { prose, .. }
            | ConstructionOutcome::StudCount { prose, .. }
            | ConstructionOutcome::SheetGoods { prose, .. }
            | ConstructionOutcome::Rafter { prose, .. }
            | ConstructionOutcome::Pitch { prose, .. }
            | ConstructionOutcome::Concrete { prose, .. }
            | ConstructionOutcome::Stairs { prose, .. }
            | ConstructionOutcome::Error { prose, .. } => prose,
        }
    }
}

const FLAG: &str = "(construction calculator)";

// =====================================================================
// Detection — "does this look construction-shaped?"
// =====================================================================

pub fn looks_like_construction(query: &str) -> bool {
    let q = query.to_lowercase();
    if q.contains('\'') || q.contains('"') { return true; }

    let padded = format!(" {} ", q);
    const WORDS: &[&str] = &[
        "stud", "studs", "rafter", "rafters", "pitch",
        "on center", "oc", "o.c.",
        "board feet", "board foot", "bf",
        "drywall", "plywood", "sheets", "sheet good", "sheet goods",
        "cubic yard", "cubic yards", "concrete", "slab",
        "stair", "stairs", "riser", "risers", "tread", "treads",
        "feet-inches", "feet inches", "feet-inches-fractions",
        "hypotenuse",
    ];
    if WORDS.iter().any(|w| padded.contains(&format!(" {} ", w))) {
        return true;
    }

    if has_digit_unit(&q) { return true; }
    if has_lumber_dim(&q) { return true; }
    false
}

fn has_digit_unit(q: &str) -> bool {
    for unit in &[" foot ", " feet ", " foot,", " feet,",
                  " foot.", " feet.", " inch ", " inches",
                  " inch.", " inches.", " inches,"] {
        let mut start = 0;
        while let Some(rel) = q[start..].find(unit) {
            let abs = start + rel;
            if abs > 0 {
                let prev = q[..abs].chars().last().unwrap_or(' ');
                if prev.is_ascii_digit() || prev == '.' { return true; }
            }
            start = abs + unit.len();
        }
    }
    false
}

fn has_lumber_dim(q: &str) -> bool {
    let bytes = q.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if !(bytes[i] as char).is_ascii_digit() { i += 1; continue; }
        let mut j = i + 1;
        while j < n && (bytes[j] as char).is_ascii_digit() { j += 1; }
        if j >= n || bytes[j] != b'x' { i = j + 1; continue; }
        let mut k = j + 1;
        while k < n && (bytes[k] as char).is_ascii_digit() { k += 1; }
        if k == j + 1 || k >= n || bytes[k] != b'x' { i = k + 1; continue; }
        let mut m = k + 1;
        while m < n && (bytes[m] as char).is_ascii_digit() { m += 1; }
        if m > k + 1 { return true; }
        i = m;
    }
    false
}

// =====================================================================
// detect_construction — entrypoint
// =====================================================================

pub fn detect_construction(query: &str) -> Option<ConstructionOutcome> {
    if !looks_like_construction(query) { return None; }
    let lower = query.trim().to_lowercase();
    let stripped = lower.trim_end_matches(|c: char| matches!(c, '.' | '?' | '!')).trim();
    let q = strip_one_prefix(stripped, &[
        "what is the value of ",
        "what is ",
        "whats ",
        "what's ",
        "compute ",
        "calculate ",
        "evaluate ",
        "how many ",
    ]);

    if let Some(o) = try_stud_count(q) { return Some(o); }
    if let Some(o) = try_pitch(q) { return Some(o); }
    if let Some(o) = try_rafter(q) { return Some(o); }
    if let Some(o) = try_concrete(q) { return Some(o); }
    if let Some(o) = try_sheet_goods(q) { return Some(o); }
    if let Some(o) = try_stairs(q) { return Some(o); }
    if let Some(o) = try_board_feet(q) { return Some(o); }
    if let Some(o) = try_fi_conversion(q) { return Some(o); }
    if let Some(o) = try_fi_arith(q) { return Some(o); }
    None
}

fn strip_one_prefix<'a>(s: &'a str, prefixes: &[&str]) -> &'a str {
    for p in prefixes {
        if let Some(rest) = s.strip_prefix(*p) { return rest.trim_start(); }
    }
    s
}

// =====================================================================
// (1) FI arithmetic — `12'-3 1/2" + 4'-7 1/4"` etc.
// =====================================================================

fn try_fi_arith(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains('\'') || q.contains('"')) { return None; }
    for (sep, op_char) in &[(" + ", '+'), (" - ", '-'), (" * ", '*'), (" / ", '/')] {
        if let Some(idx) = q.find(sep) {
            let lhs = q[..idx].trim();
            let rhs = q[idx + sep.len()..].trim();
            return Some(compute_fi_arith(lhs, rhs, *op_char));
        }
    }
    None
}

fn compute_fi_arith(lhs: &str, rhs: &str, op: char) -> ConstructionOutcome {
    let left_fi = parse_fi(lhs);
    let right_fi = parse_fi(rhs);
    let right_scalar = rhs.parse::<f64>().ok();
    let left_inches = match left_fi {
        Some(f) => f.total_inches,
        None => return error_outcome("fi_arith", &format!("could not parse '{}'", lhs)),
    };
    match op {
        '+' | '-' => {
            let r = match right_fi {
                Some(f) => f.total_inches,
                None => return error_outcome("fi_arith", &format!("could not parse '{}'", rhs)),
            };
            let result = if op == '+' { left_inches + r } else { left_inches - r };
            let out = FeetInches::from_inches(result);
            let formatted = out.format_fi(16);
            let prose = format!("{} {} {} = {} {}", lhs, op, rhs, formatted, FLAG);
            ConstructionOutcome::FeetInchesArith {
                op: op.to_string(), left_inches, right_inches_or_scalar: r,
                result_inches: result, formatted, prose,
            }
        }
        '*' => {
            let s = match right_scalar {
                Some(v) => v,
                None => return error_outcome("fi_arith",
                    &format!("multiplier must be a scalar, got '{}'", rhs)),
            };
            let result = left_inches * s;
            let out = FeetInches::from_inches(result);
            let formatted = out.format_fi(16);
            let prose = format!("{} * {} = {} {}", lhs, fmt_num(s), formatted, FLAG);
            ConstructionOutcome::FeetInchesArith {
                op: "*".into(), left_inches, right_inches_or_scalar: s,
                result_inches: result, formatted, prose,
            }
        }
        '/' => {
            let s = match right_scalar {
                Some(v) => v,
                None => return error_outcome("fi_arith",
                    &format!("divisor must be a scalar, got '{}'", rhs)),
            };
            if s == 0.0 { return error_outcome("fi_arith", "division by zero"); }
            let result = left_inches / s;
            let out = FeetInches::from_inches(result);
            let formatted = out.format_fi(16);
            let prose = format!("{} / {} = {} {}", lhs, fmt_num(s), formatted, FLAG);
            ConstructionOutcome::FeetInchesArith {
                op: "/".into(), left_inches, right_inches_or_scalar: s,
                result_inches: result, formatted, prose,
            }
        }
        _ => error_outcome("fi_arith", "unsupported operator"),
    }
}

// =====================================================================
// (2) FI ↔ decimal-feet conversion
// =====================================================================

fn try_fi_conversion(q: &str) -> Option<ConstructionOutcome> {
    for sep in &[" to feet-inches", " in feet-inches", " to feet inches",
                 " in feet inches", " to feet-inches-fractions",
                 " in feet-inches-fractions"] {
        if let Some(idx) = q.find(sep) {
            let val_str = q[..idx].trim();
            let val_str = val_str.strip_prefix("convert ").unwrap_or(val_str).trim();
            let val_str = val_str.strip_prefix("the ").unwrap_or(val_str).trim();
            let val_str = val_str.trim_end_matches("feet").trim();
            let val_str = val_str.trim_end_matches("foot").trim();
            let val_str = val_str.trim_end_matches("ft").trim();
            if let Ok(decimal_feet) = val_str.parse::<f64>() {
                let fi = FeetInches::from_feet(decimal_feet);
                let formatted = fi.format_fi(16);
                let prose = format!("{} feet = {} {}",
                    fmt_num(decimal_feet), formatted, FLAG);
                return Some(ConstructionOutcome::FeetInchesConversion {
                    decimal_feet, formatted, prose,
                });
            }
        }
    }
    for sep in &[" to decimal feet", " in decimal feet"] {
        if let Some(idx) = q.find(sep) {
            let val_str = q[..idx].trim();
            let val_str = val_str.strip_prefix("convert ").unwrap_or(val_str).trim();
            if let Some(fi) = parse_fi(val_str) {
                let decimal_feet = fi.feet_decimal();
                let prose = format!("{} = {} feet {}",
                    val_str, format_decimal3(decimal_feet), FLAG);
                return Some(ConstructionOutcome::FeetInchesConversion {
                    decimal_feet,
                    formatted: fi.format_fi(16),
                    prose,
                });
            }
        }
    }
    if let Some(rest) = q.strip_prefix("decimal feet of ") {
        if let Some(fi) = parse_fi(rest.trim()) {
            let decimal_feet = fi.feet_decimal();
            let prose = format!("{} = {} feet {}",
                rest.trim(), format_decimal3(decimal_feet), FLAG);
            return Some(ConstructionOutcome::FeetInchesConversion {
                decimal_feet, formatted: fi.format_fi(16), prose,
            });
        }
    }
    None
}

fn format_decimal3(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        fmt_num(v.round())
    } else {
        let rounded = (v * 1000.0).round() / 1000.0;
        format!("{}", rounded)
    }
}

// =====================================================================
// (3) Board feet — (T × W × L) ÷ 12 with nominal sizing
// =====================================================================

fn try_board_feet(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains("board feet") || q.contains("board foot") || q.contains(" bf ")) {
        return None;
    }
    let dims = extract_lumber_dims(q)?;
    let (t, w, l) = dims;
    if t <= 0.0 || w <= 0.0 || l <= 0.0 { return None; }
    let bf = (t * w * l) / 12.0;
    let bf_rounded = (bf * 100.0).round() / 100.0;
    let prose = format!(
        "board feet of {}x{}x{} = ({} × {} × {}) ÷ 12 = {} BF {}",
        fmt_num(t), fmt_num(w), fmt_num(l),
        fmt_num(t), fmt_num(w), fmt_num(l),
        fmt_num(bf_rounded), FLAG,
    );
    Some(ConstructionOutcome::BoardFeet {
        thickness_in: t, width_in: w, length_ft: l,
        board_feet: bf_rounded, prose,
    })
}

fn extract_lumber_dims(q: &str) -> Option<(f64, f64, f64)> {
    let bytes = q.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if !(bytes[i] as char).is_ascii_digit() { i += 1; continue; }
        let mut j = i + 1;
        while j < n && (bytes[j] as char).is_ascii_digit() { j += 1; }
        if j >= n || bytes[j] != b'x' { i = j + 1; continue; }
        let mut k = j + 1;
        while k < n && (bytes[k] as char).is_ascii_digit() { k += 1; }
        if k == j + 1 || k >= n || bytes[k] != b'x' { i = k + 1; continue; }
        let mut m = k + 1;
        while m < n && (bytes[m] as char).is_ascii_digit() { m += 1; }
        if m > k + 1 {
            let a = q[i..j].parse::<f64>().ok()?;
            let b = q[j + 1..k].parse::<f64>().ok()?;
            let c = q[k + 1..m].parse::<f64>().ok()?;
            return Some((a, b, c));
        }
        i = m;
    }
    None
}

// =====================================================================
// (4) Stud count — floor(wall_in / spacing_in) + 1
// =====================================================================

fn try_stud_count(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains("stud") && (q.contains("wall") || q.contains("oc")
        || q.contains("on center") || q.contains("o.c."))) {
        return None;
    }
    let spacing_in = extract_spacing(q)?;
    if spacing_in <= 0.0 { return None; }
    let wall_ft = extract_wall_length(q)?;
    if wall_ft <= 0.0 { return None; }
    let length_in = wall_ft * 12.0;
    let count = ((length_in / spacing_in).floor() as u64) + 1;
    let prose = format!(
        "{} ft × 12 in/ft ÷ {} in OC + 1 = {} studs {}",
        fmt_num(wall_ft), fmt_num(spacing_in), count, FLAG,
    );
    Some(ConstructionOutcome::StudCount {
        wall_length_ft: wall_ft, spacing_in, count, prose,
    })
}

fn extract_spacing(q: &str) -> Option<f64> {
    for kw in &["on center", " oc", " o.c.", " oc."] {
        if let Some(idx) = q.find(kw) {
            let before = q[..idx].trim_end();
            let before = before.trim_end_matches("inches").trim_end_matches("inch")
                .trim_end().trim_end_matches('"').trim_end();
            if let Some(n) = trailing_number(before) { return Some(n); }
        }
    }
    None
}

fn extract_wall_length(q: &str) -> Option<f64> {
    for kw in &[" foot wall", " feet wall", "' wall", " ft wall"] {
        if let Some(idx) = q.find(kw) {
            let before = q[..idx].trim_end();
            if let Some(n) = trailing_number(before) { return Some(n); }
        }
    }
    None
}

fn trailing_number(s: &str) -> Option<f64> {
    let s = s.trim_end();
    let mut end = s.len();
    let bytes = s.as_bytes();
    while end > 0 {
        let c = bytes[end - 1] as char;
        if c.is_ascii_digit() || c == '.' { end -= 1; }
        else { break; }
    }
    if end == s.len() { return None; }
    s[end..].parse::<f64>().ok()
}

// =====================================================================
// (5) Sheet goods — ceil(area_sqft / 32)
// =====================================================================

fn try_sheet_goods(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains("sheet") || q.contains("drywall") || q.contains("plywood")) {
        return None;
    }
    for kw in &[" square feet", " sqft", " sq ft", " square foot"] {
        if let Some(idx) = q.find(kw) {
            let before = q[..idx].trim_end();
            if let Some(n) = trailing_number(before) {
                return Some(emit_sheet_goods(n));
            }
        }
    }
    if let Some((a, b)) = extract_2d_dims(q) {
        let area = a * b;
        return Some(emit_sheet_goods(area));
    }
    None
}

fn emit_sheet_goods(area_sqft: f64) -> ConstructionOutcome {
    let sheets = (area_sqft / 32.0).ceil() as u64;
    let prose = format!(
        "{} sqft ÷ 32 sqft/sheet = {} sheets {}",
        format_decimal3(area_sqft), sheets, FLAG,
    );
    ConstructionOutcome::SheetGoods { area_sqft, sheets, prose }
}

fn extract_2d_dims(q: &str) -> Option<(f64, f64)> {
    if let Some(idx) = q.find(" by ") {
        let before = q[..idx].trim_end();
        let after = q[idx + 4..].trim_start();
        let a = trailing_number(before)?;
        let b = leading_number(after)?;
        return Some((a, b));
    }
    // Walk all NxM occurrences and return the LAST one. Sheet-goods
    // queries like "4x8 sheets to cover a 12x10 floor" mention the
    // sheet size first; the area we actually want is at the end.
    let bytes = q.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut last: Option<(f64, f64)> = None;
    while i < n {
        if !(bytes[i] as char).is_ascii_digit() && bytes[i] != b'.' { i += 1; continue; }
        let mut j = i;
        while j < n && ((bytes[j] as char).is_ascii_digit() || bytes[j] == b'.') { j += 1; }
        if j >= n || bytes[j] != b'x' { i = j + 1; continue; }
        let mut k = j + 1;
        while k < n && ((bytes[k] as char).is_ascii_digit() || bytes[k] == b'.') { k += 1; }
        if k > j + 1 {
            if k < n && bytes[k] == b'x' { i = k + 1; continue; }
            if let (Ok(a), Ok(b)) = (q[i..j].parse::<f64>(), q[j + 1..k].parse::<f64>()) {
                last = Some((a, b));
            }
            i = k;
            continue;
        }
        i = j + 1;
    }
    last
}

fn leading_number(s: &str) -> Option<f64> {
    let s = s.trim_start();
    let mut end = 0;
    let bytes = s.as_bytes();
    while end < bytes.len() {
        let c = bytes[end] as char;
        if c.is_ascii_digit() || c == '.' { end += 1; }
        else { break; }
    }
    if end == 0 { return None; }
    s[..end].parse::<f64>().ok()
}

// =====================================================================
// (6) Rafter — sqrt(run² + rise²)
// =====================================================================

fn try_rafter(q: &str) -> Option<ConstructionOutcome> {
    if q.contains("hypotenuse") {
        let rest = q.strip_prefix("hypotenuse of ").unwrap_or(q);
        let rest = rest.strip_prefix("the hypotenuse of ").unwrap_or(rest);
        if let Some((a, b)) = split_two_numbers(rest, " and ") {
            let h = (a * a + b * b).sqrt();
            let h_round = (h * 1000.0).round() / 1000.0;
            let h_str = if h.fract() == 0.0 { fmt_num(h) } else { format!("{}", h_round) };
            let prose = format!(
                "sqrt({}² + {}²) = sqrt({}) = {} {}",
                fmt_num(a), fmt_num(b), fmt_num(a*a + b*b), h_str, FLAG,
            );
            return Some(ConstructionOutcome::Rafter {
                run_ft: a, rise_ft: b, rafter_ft: h,
                formatted: h_str, prose,
            });
        }
    }
    if q.contains("rafter") {
        let run = extract_named_value(q, &["run"])?;
        let rise = extract_named_value(q, &["rise"])?;
        let h = (run * run + rise * rise).sqrt();
        let fi = FeetInches::from_feet(h);
        let formatted = fi.format_fi(16);
        let prose = format!(
            "sqrt({}² + {}²) = sqrt({}) = {} {}",
            fmt_num(run), fmt_num(rise), fmt_num(run*run + rise*rise),
            formatted, FLAG,
        );
        return Some(ConstructionOutcome::Rafter {
            run_ft: run, rise_ft: rise, rafter_ft: h, formatted, prose,
        });
    }
    None
}

fn split_two_numbers(s: &str, sep: &str) -> Option<(f64, f64)> {
    let idx = s.find(sep)?;
    let a = s[..idx].trim().parse::<f64>().ok()?;
    let b = s[idx + sep.len()..].trim().parse::<f64>().ok()?;
    Some((a, b))
}

fn extract_named_value(q: &str, keys: &[&str]) -> Option<f64> {
    for key in keys {
        let pat = format!(" {}", key);
        if let Some(idx) = q.find(&pat) {
            let before = q[..idx].trim_end();
            let before = before.trim_end_matches("foot").trim_end_matches("feet")
                .trim_end_matches("ft").trim_end();
            if let Some(n) = trailing_number(before) { return Some(n); }
        }
    }
    None
}

// =====================================================================
// (7) Roof pitch — N/12 ↔ slope angle
// =====================================================================

fn try_pitch(q: &str) -> Option<ConstructionOutcome> {
    let mentions_pitch = q.contains("pitch") || q.contains("slope");
    if !mentions_pitch { return None; }
    if let Some(rise) = extract_pitch_ratio(q) {
        let slope_rad = (rise / 12.0).atan();
        let slope_deg = slope_rad.to_degrees();
        let slope_round = (slope_deg * 100.0).round() / 100.0;
        let prose = format!(
            "{}/12 pitch → atan({}/12) = {}° {}",
            fmt_num(rise), fmt_num(rise), slope_round, FLAG,
        );
        return Some(ConstructionOutcome::Pitch {
            rise_in_12: rise, slope_deg: slope_round, prose,
        });
    }
    None
}

fn extract_pitch_ratio(q: &str) -> Option<f64> {
    if let Some(idx) = q.find("/12") {
        let before = q[..idx].trim_end();
        if let Some(n) = trailing_number(before) {
            if n >= 0.0 && n <= 24.0 { return Some(n); }
        }
    }
    if let Some(idx) = q.find(":12") {
        let before = q[..idx].trim_end();
        if let Some(n) = trailing_number(before) {
            if n >= 0.0 && n <= 24.0 { return Some(n); }
        }
    }
    if let Some(idx) = q.find(" in 12") {
        let before = q[..idx].trim_end();
        if let Some(n) = trailing_number(before) {
            if n >= 0.0 && n <= 24.0 { return Some(n); }
        }
    }
    None
}

// =====================================================================
// (8) Concrete — (L × W × D/12) ÷ 27 cubic yards
// =====================================================================

fn try_concrete(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains("concrete") || q.contains("cubic yard")
        || q.contains("cubic feet of concrete") || q.contains("slab")) {
        return None;
    }
    let (length_ft, width_ft, depth_in) = extract_concrete_dims(q)?;
    if length_ft <= 0.0 || width_ft <= 0.0 || depth_in <= 0.0 { return None; }
    let cubic_feet = length_ft * width_ft * (depth_in / 12.0);
    let cubic_yards = cubic_feet / 27.0;
    let cy_round = (cubic_yards * 100.0).round() / 100.0;
    let prose = format!(
        "({} × {} × {}/12) ÷ 27 = {} cubic yards {}",
        fmt_num(length_ft), fmt_num(width_ft), fmt_num(depth_in),
        cy_round, FLAG,
    );
    Some(ConstructionOutcome::Concrete {
        length_ft, width_ft, depth_in, cubic_yards: cy_round, prose,
    })
}

fn extract_concrete_dims(q: &str) -> Option<(f64, f64, f64)> {
    // First check the 3-way "by" chain: "N by M by K inches"
    if let Some(idx1) = q.find(" by ") {
        let after = &q[idx1 + 4..];
        if let Some(idx2) = after.find(" by ") {
            let third_str = &after[idx2 + 4..];
            if let (Some(a), Some(b), Some(c)) = (
                trailing_number(q[..idx1].trim_end()),
                leading_number(after[..idx2].trim_start()),
                leading_number(third_str.trim_start()),
            ) {
                return Some((a, b, c));
            }
        }
    }
    if let Some((a, b)) = extract_2d_dims(q) {
        if let Some(d) = extract_depth(q) {
            return Some((a, b, d));
        }
    }
    None
}

fn extract_depth(q: &str) -> Option<f64> {
    for kw in &[" inch thick", " inches thick", " inch deep", " inches deep",
                " inch slab", " inches slab", "\" thick", "\" deep"] {
        if let Some(idx) = q.find(kw) {
            let before = q[..idx].trim_end();
            if let Some(n) = trailing_number(before) { return Some(n); }
        }
    }
    let mut last: Option<f64> = None;
    for kw in &[" inch", " inches"] {
        let mut start = 0;
        while let Some(rel) = q[start..].find(kw) {
            let abs = start + rel;
            let before = q[..abs].trim_end();
            if let Some(n) = trailing_number(before) { last = Some(n); }
            start = abs + kw.len();
        }
    }
    last
}

// =====================================================================
// (9) Stair calcs — risers + treads from total rise
// =====================================================================

fn try_stairs(q: &str) -> Option<ConstructionOutcome> {
    if !(q.contains("stair") || q.contains("step count") || q.contains("riser")) {
        return None;
    }
    let total_rise_in = extract_total_rise(q)?;
    if total_rise_in <= 0.0 { return None; }
    let risers = ((total_rise_in / 7.5).round() as u64).max(1);
    let riser_height = total_rise_in / risers as f64;
    let riser_round = (riser_height * 100.0).round() / 100.0;
    let treads = risers - 1;
    let mut tread = 25.0 - 2.0 * riser_height;
    if !(9.0..=12.0).contains(&tread) { tread = 10.0; }
    let tread_round = (tread * 100.0).round() / 100.0;
    let prose = format!(
        "{} in ÷ ~7.5 in/step = {} risers @ {}\" each, {} treads @ {}\" {}",
        fmt_num(total_rise_in), risers, riser_round, treads, tread_round, FLAG,
    );
    Some(ConstructionOutcome::Stairs {
        total_rise_in, risers, riser_height_in: riser_round,
        treads, tread_depth_in: tread_round, prose,
    })
}

fn extract_total_rise(q: &str) -> Option<f64> {
    for (kw, in_inches) in &[
        (" inch rise", true), (" inches rise", true),
        (" foot rise", false), (" feet rise", false),
        (" foot ceiling", false), (" feet ceiling", false),
    ] {
        if let Some(idx) = q.find(*kw) {
            let before = q[..idx].trim_end();
            if let Some(n) = trailing_number(before) {
                return Some(if *in_inches { n } else { n * 12.0 });
            }
        }
    }
    for kw in &[" feet", " foot"] {
        if let Some(idx) = q.find(kw) {
            let before = q[..idx].trim_end();
            if let Some(n) = trailing_number(before) {
                return Some(n * 12.0);
            }
        }
    }
    if let Some(idx) = q.find(" rise") {
        let before = q[..idx].trim_end();
        if let Some(n) = trailing_number(before) { return Some(n); }
    }
    None
}

// =====================================================================
// helpers
// =====================================================================

fn error_outcome(route: &str, err: &str) -> ConstructionOutcome {
    ConstructionOutcome::Error {
        route: route.into(),
        error: err.into(),
        prose: format!("Cannot compute ({}): {} {}", route, err, FLAG),
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fi(s: &str) -> FeetInches { parse_fi(s).expect("parse") }

    mod fraction_parse_tests {
        use super::*;

        #[test]
        fn parse_feet_inches_dash_with_mixed_fraction() {
            let v = fi("12'-3 1/2\"");
            assert!((v.total_inches - 147.5).abs() < 1e-9);
        }

        #[test]
        fn parse_feet_inches_no_dash_with_decimal() {
            let v = fi("12'3.5\"");
            assert!((v.total_inches - 147.5).abs() < 1e-9);
        }

        #[test]
        fn parse_textual_ft_in() {
            let v = fi("12 ft 3.5 in");
            assert!((v.total_inches - 147.5).abs() < 1e-9);
        }

        #[test]
        fn parse_feet_inches_integer_only() {
            let v = fi("12'-3\"");
            assert!((v.total_inches - 147.0).abs() < 1e-9);
        }

        #[test]
        fn parse_inches_only_dash_fraction() {
            let v = fi("3-1/2\"");
            assert!((v.total_inches - 3.5).abs() < 1e-9);
        }

        #[test]
        fn parse_inches_only_space_fraction() {
            let v = fi("3 1/2\"");
            assert!((v.total_inches - 3.5).abs() < 1e-9);
        }

        #[test]
        fn parse_decimal_feet_only() {
            let v = fi("12.292'");
            assert!((v.total_inches - 147.504).abs() < 1e-3);
        }

        #[test]
        fn parse_feet_only() {
            let v = fi("12'");
            assert!((v.total_inches - 144.0).abs() < 1e-9);
        }

        #[test]
        fn parse_inches_only() {
            let v = fi("3\"");
            assert!((v.total_inches - 3.0).abs() < 1e-9);
        }

        #[test]
        fn parse_inches_only_pure_fraction() {
            let v = fi("1/2\"");
            assert!((v.total_inches - 0.5).abs() < 1e-9);
        }

        #[test]
        fn parse_zero_inches() {
            let v = fi("8'-0\"");
            assert!((v.total_inches - 96.0).abs() < 1e-9);
        }

        #[test]
        fn parse_invalid_returns_none() {
            assert!(parse_fi("").is_none());
            assert!(parse_fi("hello").is_none());
            assert!(parse_fi("12'foo").is_none());
        }

        #[test]
        fn format_reduces_fraction_to_lowest_terms() {
            let v = FeetInches::from_inches(0.5);
            assert_eq!(v.format_fi(16), "0'-0 1/2\"");
        }

        #[test]
        fn format_no_fraction_when_whole() {
            let v = FeetInches::from_inches(36.0);
            assert_eq!(v.format_fi(16), "3'-0\"");
        }

        #[test]
        fn format_carries_inches_to_feet() {
            let v = FeetInches::from_inches(12.0);
            assert_eq!(v.format_fi(16), "1'-0\"");
        }
    }

    mod feet_inch_tests {
        use super::*;

        fn arith(q: &str) -> ConstructionOutcome { detect_construction(q).expect(q) }

        #[test]
        fn add_two_fi_values() {
            let r = arith("12'-3 1/2\" + 4'-7 1/4\"");
            assert!(r.prose().contains("16'-10 3/4\""), "prose={}", r.prose());
            assert!(r.prose().contains("construction calculator"));
        }

        #[test]
        fn sub_two_fi_values() {
            let r = arith("8'-0\" - 3 1/2\"");
            assert!(r.prose().contains("7'-8 1/2\""), "prose={}", r.prose());
        }

        #[test]
        fn div_fi_by_scalar() {
            let r = arith("8' / 3");
            assert!(r.prose().contains("2'-8\""), "prose={}", r.prose());
        }

        #[test]
        fn mul_fi_by_scalar() {
            let r = arith("12'-3 1/2\" * 2");
            assert!(r.prose().contains("24'-7\""), "prose={}", r.prose());
        }

        #[test]
        fn add_inches_only_pieces() {
            let r = arith("3 1/2\" + 1/4\"");
            assert!(r.prose().contains("0'-3 3/4\""), "prose={}", r.prose());
        }

        #[test]
        fn divide_by_zero_errors() {
            let r = arith("8' / 0");
            match r {
                ConstructionOutcome::Error { .. } => {}
                other => panic!("expected Error, got {:?}", other),
            }
        }
    }

    mod conversion_tests {
        use super::*;

        #[test]
        fn decimal_feet_to_fi() {
            let r = detect_construction("12.292 feet to feet-inches").expect("detect");
            assert!(r.prose().contains("12'-3 1/2\""), "prose={}", r.prose());
        }

        #[test]
        fn fi_to_decimal_feet() {
            let r = detect_construction("12'-3 1/2\" to decimal feet").expect("detect");
            assert!(r.prose().contains("12.292"), "prose={}", r.prose());
        }

        #[test]
        fn whats_in_fif() {
            let r = detect_construction("what is 12.292 in feet-inches-fractions")
                .expect("detect");
            assert!(r.prose().contains("12'-3 1/2\""));
        }

        #[test]
        fn decimal_feet_of_phrase() {
            let r = detect_construction("decimal feet of 12'3 1/2\"").expect("detect");
            assert!(r.prose().contains("12.292"));
        }
    }

    mod board_foot_tests {
        use super::*;

        #[test]
        fn bf_2x4x8() {
            let r = detect_construction("board feet of 2x4x8").expect("detect");
            match r {
                ConstructionOutcome::BoardFeet { board_feet, .. } => {
                    assert!((board_feet - 5.33).abs() < 0.01);
                }
                _ => panic!("expected BoardFeet"),
            }
        }

        #[test]
        fn bf_1x12x10() {
            let r = detect_construction("board feet of 1x12x10").expect("detect");
            match r {
                ConstructionOutcome::BoardFeet { board_feet, .. } => {
                    assert!((board_feet - 10.0).abs() < 0.01);
                }
                _ => panic!("expected BoardFeet"),
            }
        }

        #[test]
        fn bf_phrasing_variant() {
            let r = detect_construction("how many board feet in a 2x4x8").expect("detect");
            match r {
                ConstructionOutcome::BoardFeet { board_feet, .. } => {
                    assert!((board_feet - 5.33).abs() < 0.01);
                }
                _ => panic!("expected BoardFeet"),
            }
        }

        #[test]
        fn bf_trailing_phrasing() {
            let r = detect_construction("1x12x10 board feet").expect("detect");
            match r {
                ConstructionOutcome::BoardFeet { board_feet, .. } => {
                    assert!((board_feet - 10.0).abs() < 0.01);
                }
                _ => panic!("expected BoardFeet"),
            }
        }
    }

    mod stud_count_tests {
        use super::*;

        #[test]
        fn studs_32ft_16oc() {
            let r = detect_construction("studs in a 32 foot wall at 16 OC").expect("detect");
            match r {
                ConstructionOutcome::StudCount { count, .. } => assert_eq!(count, 25),
                _ => panic!("expected StudCount"),
            }
        }

        #[test]
        fn studs_12ft_24oc() {
            let r = detect_construction("studs in a 12 foot wall at 24 OC").expect("detect");
            match r {
                ConstructionOutcome::StudCount { count, .. } => assert_eq!(count, 7),
                _ => panic!("expected StudCount"),
            }
        }

        #[test]
        fn studs_alt_phrasing() {
            let r = detect_construction("how many studs at 24 inch on center for a 12 foot wall")
                .expect("detect");
            match r {
                ConstructionOutcome::StudCount { count, .. } => assert_eq!(count, 7),
                _ => panic!("expected StudCount"),
            }
        }

        #[test]
        fn studs_tick_marks() {
            let r = detect_construction("32' wall 16 OC how many studs").expect("detect");
            match r {
                ConstructionOutcome::StudCount { count, .. } => assert_eq!(count, 25),
                _ => panic!("expected StudCount"),
            }
        }
    }

    mod sheet_goods_tests {
        use super::*;

        #[test]
        fn sheets_32_by_8_wall() {
            let r = detect_construction("sheets for a 32 by 8 wall").expect("detect");
            match r {
                ConstructionOutcome::SheetGoods { sheets, .. } => assert_eq!(sheets, 8),
                _ => panic!("expected SheetGoods"),
            }
        }

        #[test]
        fn sheets_drywall_400_sqft() {
            let r = detect_construction("drywall sheets for 400 square feet").expect("detect");
            match r {
                ConstructionOutcome::SheetGoods { sheets, .. } => assert_eq!(sheets, 13),
                _ => panic!("expected SheetGoods"),
            }
        }

        #[test]
        fn sheets_256_sqft_exact() {
            let r = detect_construction("drywall sheets for 256 square feet").expect("detect");
            match r {
                ConstructionOutcome::SheetGoods { sheets, .. } => assert_eq!(sheets, 8),
                _ => panic!("expected SheetGoods"),
            }
        }

        #[test]
        fn sheets_4x8_phrasing() {
            let r = detect_construction("4x8 sheets to cover a 12x10 floor").expect("detect");
            match r {
                ConstructionOutcome::SheetGoods { sheets, .. } => assert_eq!(sheets, 4),
                _ => panic!("expected SheetGoods"),
            }
        }
    }

    mod rafter_tests {
        use super::*;

        #[test]
        fn rafter_12_run_6_rise() {
            let r = detect_construction("rafter for 12 foot run 6 foot rise").expect("detect");
            assert!(r.prose().contains("13'-5\"") || r.prose().contains("13'-4"),
                "prose={}", r.prose());
        }

        #[test]
        fn hypotenuse_3_4() {
            let r = detect_construction("hypotenuse of 3 and 4").expect("detect");
            match r {
                ConstructionOutcome::Rafter { rafter_ft, .. } => {
                    assert!((rafter_ft - 5.0).abs() < 1e-9);
                }
                _ => panic!("expected Rafter"),
            }
        }

        #[test]
        fn hypotenuse_5_12() {
            let r = detect_construction("hypotenuse of 5 and 12").expect("detect");
            match r {
                ConstructionOutcome::Rafter { rafter_ft, .. } => {
                    assert!((rafter_ft - 13.0).abs() < 1e-9);
                }
                _ => panic!("expected Rafter"),
            }
        }
    }

    mod pitch_tests {
        use super::*;

        #[test]
        fn pitch_6_over_12() {
            let r = detect_construction("slope angle of 6/12 pitch").expect("detect");
            match r {
                ConstructionOutcome::Pitch { slope_deg, .. } => {
                    assert!((slope_deg - 26.57).abs() < 0.05, "got {}", slope_deg);
                }
                _ => panic!("expected Pitch"),
            }
        }

        #[test]
        fn pitch_4_over_12() {
            let r = detect_construction("pitch in degrees for 4/12").expect("detect");
            match r {
                ConstructionOutcome::Pitch { slope_deg, .. } => {
                    assert!((slope_deg - 18.43).abs() < 0.05, "got {}", slope_deg);
                }
                _ => panic!("expected Pitch"),
            }
        }

        #[test]
        fn pitch_colon_form() {
            let r = detect_construction("pitch angle for 6:12").expect("detect");
            match r {
                ConstructionOutcome::Pitch { slope_deg, .. } => {
                    assert!((slope_deg - 26.57).abs() < 0.05);
                }
                _ => panic!("expected Pitch"),
            }
        }

        #[test]
        fn pitch_in_form() {
            let r = detect_construction("what's the slope of a 6 in 12 roof").expect("detect");
            match r {
                ConstructionOutcome::Pitch { slope_deg, .. } => {
                    assert!((slope_deg - 26.57).abs() < 0.05);
                }
                _ => panic!("expected Pitch"),
            }
        }
    }

    mod concrete_tests {
        use super::*;

        #[test]
        fn concrete_20x10_slab_4in() {
            let r = detect_construction("concrete for a 20 by 10 slab 4 inches thick")
                .expect("detect");
            match r {
                ConstructionOutcome::Concrete { cubic_yards, .. } => {
                    assert!((cubic_yards - 2.47).abs() < 0.05, "got {}", cubic_yards);
                }
                _ => panic!("expected Concrete"),
            }
        }

        #[test]
        fn concrete_30_by_10_by_6_inches() {
            let r = detect_construction("cubic yards for 30 by 10 by 6 inches").expect("detect");
            match r {
                ConstructionOutcome::Concrete { cubic_yards, .. } => {
                    assert!((cubic_yards - 5.56).abs() < 0.05, "got {}", cubic_yards);
                }
                _ => panic!("expected Concrete"),
            }
        }

        #[test]
        fn concrete_phrasing_with_driveway() {
            let r = detect_construction("cubic yards for a 30x10 driveway 6 inch thick")
                .expect("detect");
            match r {
                ConstructionOutcome::Concrete { cubic_yards, .. } => {
                    assert!((cubic_yards - 5.56).abs() < 0.05, "got {}", cubic_yards);
                }
                _ => panic!("expected Concrete"),
            }
        }
    }

    mod stair_tests {
        use super::*;

        #[test]
        fn stairs_108_inch_rise() {
            let r = detect_construction("stairs for 108 inch rise").expect("detect");
            match r {
                ConstructionOutcome::Stairs { risers, treads, .. } => {
                    assert_eq!(risers, 14);
                    assert_eq!(treads, 13);
                }
                _ => panic!("expected Stairs"),
            }
        }

        #[test]
        fn stairs_9_foot_ceiling() {
            let r = detect_construction("stairs for 9 foot ceiling").expect("detect");
            match r {
                ConstructionOutcome::Stairs { risers, .. } => assert_eq!(risers, 14),
                _ => panic!("expected Stairs"),
            }
        }

        #[test]
        fn step_count_phrasing() {
            let r = detect_construction("step count for 108 inch rise").expect("detect");
            match r {
                ConstructionOutcome::Stairs { risers, .. } => assert_eq!(risers, 14),
                _ => panic!("expected Stairs"),
            }
        }
    }

    mod detection_tests {
        use super::*;

        #[test]
        fn looks_like_construction_positive_cases() {
            assert!(looks_like_construction("12'-3\""));
            assert!(looks_like_construction("32 foot wall 16 OC studs"));
            assert!(looks_like_construction("board feet of 2x4x8"));
            assert!(looks_like_construction("6/12 pitch in degrees"));
            assert!(looks_like_construction("concrete cubic yards"));
            assert!(looks_like_construction("hypotenuse of 3 and 4"));
            assert!(looks_like_construction("stairs for 108 inch rise"));
        }

        #[test]
        fn looks_like_construction_negative_cases() {
            assert!(!looks_like_construction("2 + 3 * 4"));
            assert!(!looks_like_construction("what is 7 plus 5"));
            assert!(!looks_like_construction("how many seconds in a day"));
            assert!(!looks_like_construction("convert 100 km to miles"));
            assert!(!looks_like_construction("mean of 1, 2, 3"));
            assert!(!looks_like_construction("sqrt(144)"));
            assert!(!looks_like_construction("what is mitochondria"));
        }

        #[test]
        fn detect_returns_none_for_non_construction() {
            assert!(detect_construction("2 + 3 * 4").is_none());
            assert!(detect_construction("what is mitochondria").is_none());
            assert!(detect_construction("how many seconds in a day").is_none());
        }

        #[test]
        fn detect_returns_none_when_shape_matches_but_no_pattern() {
            assert!(detect_construction("give me the pitch").is_none());
            assert!(detect_construction("concrete is hard").is_none());
        }

        #[test]
        fn flag_appears_in_prose() {
            let r = detect_construction("board feet of 2x4x8").expect("detect");
            assert!(r.prose().contains("(construction calculator)"));
        }
    }
}
