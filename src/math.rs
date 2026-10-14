//! Port F internal notes — math intent detector + arithmetic primitive.
//! Port F.2 (this revision) — recursive-descent expression parser,
//! scientific functions, named constants, single-step unit conversion,
//! descriptive statistics, English word-number operands, and a simple
//! linear solver.
//!
//! Lu's recall surface is associative. Pre-Port-F, basic arithmetic
//! ("what is 7 plus 5") fell through to spreading activation and
//! returned whichever node held the highest residual activation —
//! the eval baseline in #598 caught five canonical math queries
//! returning unrelated nodes (Serge Gainsbourg for `7 + 5`, etc.).
//!
//! Port F closed that gap for single-binary-op shapes. Port F.2
//! widens the surface to general-purpose calculator queries, while
//! keeping the dispatch conservative: a query is only routed through
//! math if it parses cleanly. Anything else ("what is mitochondria")
//! still falls through to recall.
//!
//! The response is flagged with `source: "arithmetic_primitive"`
//! (legacy single-op path) or `source: "expression_primitive"`
//! (any new F.2 capability) so callers can tell math fired and
//! distinguish the two routes.
//!
//! Round-2 IN scope (F.2): full expression parser with precedence,
//! ~25 scientific functions, the constants π/e/τ/inf, comma-separated
//! statistics, single-step unit conversions across time/distance/
//! weight/temperature, English word-form operands up to billions,
//! and a one-unknown linear solver.
//!
//! Round-2 explicitly OUT of scope (deferred to F.3+): polynomial /
//! quadratic / higher-degree algebra; systems of equations; calculus;
//! linear algebra; word problems with prose extraction; scientific
//! notation literals; symbolic simplification; multi-step unit
//! conversions ("3 hours 5 minutes in seconds").

use serde::{Deserialize, Serialize};

// F.2 sub-modules — each ships its own tests so cargo's test count
// scales with capability surface, not with one giant file.
pub mod construction;
pub mod algnum;
pub mod arrangement;
pub mod autochord;
pub mod expr;
pub mod solve;
pub mod stats;
pub mod units;
pub mod word_num;
pub mod wordprob;
pub mod geometry;
pub mod geometry_tool;
pub mod algsolve;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    Greater,
    Less,
    Equal,
}

/// A query parsed into a structured arithmetic operation (F.1 single-op
/// shape — preserved verbatim for backward compatibility with the 28
/// existing tests in this module).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticIntent {
    Addition(f64, f64),
    Subtraction(f64, f64),
    Multiplication(f64, f64),
    Division(f64, f64),
    Percent { percent: f64, of: f64 },
    Sqrt(f64),
    Compare { left: f64, op: CompareOp, right: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum ArithmeticValue {
    Number(f64),
    Boolean(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmeticResult {
    pub intent: ArithmeticIntent,
    pub value: ArithmeticValue,
    pub prose: String,
}

impl ArithmeticIntent {
    pub fn compute(&self) -> Result<ArithmeticResult, String> {
        let (value, prose) = match *self {
            ArithmeticIntent::Addition(a, b) => {
                let v = a + b;
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("{} + {} = {} (computed)", fmt_num(a), fmt_num(b), fmt_num(v)))
            }
            ArithmeticIntent::Subtraction(a, b) => {
                let v = a - b;
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("{} - {} = {} (computed)", fmt_num(a), fmt_num(b), fmt_num(v)))
            }
            ArithmeticIntent::Multiplication(a, b) => {
                let v = a * b;
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("{} * {} = {} (computed)", fmt_num(a), fmt_num(b), fmt_num(v)))
            }
            ArithmeticIntent::Division(a, b) => {
                if b == 0.0 {
                    return Err("division by zero".into());
                }
                let v = a / b;
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("{} / {} = {} (computed)", fmt_num(a), fmt_num(b), fmt_num(v)))
            }
            ArithmeticIntent::Percent { percent, of } => {
                let v = (percent / 100.0) * of;
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("{}% of {} = {} (computed)", fmt_num(percent), fmt_num(of), fmt_num(v)))
            }
            ArithmeticIntent::Sqrt(n) => {
                if n < 0.0 {
                    return Err("square root of negative number".into());
                }
                let v = n.sqrt();
                check_finite(v)?;
                (ArithmeticValue::Number(v), format!("sqrt({}) = {} (computed)", fmt_num(n), fmt_num(v)))
            }
            ArithmeticIntent::Compare { left, op, right } => {
                let truth = match op {
                    CompareOp::Greater => left > right,
                    CompareOp::Less => left < right,
                    CompareOp::Equal => (left - right).abs() < f64::EPSILON,
                };
                let symbol = match op {
                    CompareOp::Greater => ">",
                    CompareOp::Less => "<",
                    CompareOp::Equal => "==",
                };
                (
                    ArithmeticValue::Boolean(truth),
                    format!(
                        "{} {} {} -> {} (computed)",
                        fmt_num(left),
                        symbol,
                        fmt_num(right),
                        if truth { "yes" } else { "no" },
                    ),
                )
            }
        };
        Ok(ArithmeticResult {
            intent: self.clone(),
            value,
            prose,
        })
    }
}

fn check_finite(v: f64) -> Result<(), String> {
    if v.is_nan() || v.is_infinite() {
        Err("non-finite result".into())
    } else {
        Ok(())
    }
}

pub(crate) fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

// =====================================================================
// Detection — F.1 single-op path
// =====================================================================

pub fn detect_arithmetic(query: &str) -> Option<ArithmeticIntent> {
    let lower = query.trim().to_lowercase();
    let stripped_punct: &str = lower
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();

    // OP-MATH-DETECT: strip chain-of-thought / reasoning framings
    // ("step by step, what is X") so the downstream prefix-strip
    // can recognize the underlying arithmetic question. The CoT prefix
    // alone does not change the operands — same intent, same answer.
    let de_cot = strip_cot_framing(stripped_punct);

    let q = strip_one_prefix(de_cot, &[
        "what is the value of ",
        "what is ",
        "whats ",
        "what's ",
        "compute ",
        "calculate ",
        "tell me ",
        "tell me what ",
        "can you compute ",
        "can you calculate ",
        "can you tell me ",
        "please compute ",
        "please calculate ",
    ]);

    // sqrt — matched first so "square root of" doesn't get split by
    // the " of " check inside percent-of detection.
    if let Some(rest) = strip_one_prefix_opt(q, &[
        "the square root of ",
        "square root of ",
        "sqrt of ",
    ]) {
        if let Some(n) = parse_number(rest.trim()) {
            return Some(ArithmeticIntent::Sqrt(n));
        }
    }
    if let Some(inner) = q.strip_prefix("sqrt(").and_then(|r| r.strip_suffix(')')) {
        if let Some(n) = parse_number(inner.trim()) {
            return Some(ArithmeticIntent::Sqrt(n));
        }
    }

    // Fractional shortcuts.
    if let Some(rest) = strip_one_prefix_opt(q, &["half of "]) {
        if let Some(n) = parse_number(rest.trim()) {
            return Some(ArithmeticIntent::Division(n, 2.0));
        }
    }
    if let Some(rest) = strip_one_prefix_opt(q, &["a third of ", "one third of ", "third of "]) {
        if let Some(n) = parse_number(rest.trim()) {
            return Some(ArithmeticIntent::Division(n, 3.0));
        }
    }
    if let Some(rest) = strip_one_prefix_opt(q, &["a quarter of ", "one quarter of ", "quarter of "]) {
        if let Some(n) = parse_number(rest.trim()) {
            return Some(ArithmeticIntent::Division(n, 4.0));
        }
    }

    // Comparisons.
    let cmp_target: &str = q.strip_prefix("is ").map(str::trim_start).unwrap_or(q);
    for (kw, op) in COMPARE_KEYWORDS.iter() {
        if let Some(idx) = cmp_target.find(kw) {
            let left = &cmp_target[..idx];
            let right = &cmp_target[idx + kw.len()..];
            if let (Some(a), Some(b)) = (parse_number(left.trim()), parse_number(right.trim())) {
                return Some(ArithmeticIntent::Compare {
                    left: a,
                    op: *op,
                    right: b,
                });
            }
        }
    }

    // Percent-of.
    for kw in [" percent of ", "% of "].iter() {
        if let Some((lhs, rhs)) = split_once_kw(q, kw) {
            if let (Some(p), Some(o)) = (parse_number(lhs.trim()), parse_number(rhs.trim())) {
                return Some(ArithmeticIntent::Percent { percent: p, of: o });
            }
        }
    }

    // Word-form binary infix.
    type Builder = fn(f64, f64) -> ArithmeticIntent;
    let word_ops: &[(&str, Builder)] = &[
        (" plus ",          |a, b| ArithmeticIntent::Addition(a, b)),
        (" added to ",      |a, b| ArithmeticIntent::Addition(a, b)),
        (" minus ",         |a, b| ArithmeticIntent::Subtraction(a, b)),
        (" subtracted from ", |a, b| ArithmeticIntent::Subtraction(b, a)),
        (" times ",         |a, b| ArithmeticIntent::Multiplication(a, b)),
        (" multiplied by ", |a, b| ArithmeticIntent::Multiplication(a, b)),
        (" divided by ",    |a, b| ArithmeticIntent::Division(a, b)),
        (" over ",          |a, b| ArithmeticIntent::Division(a, b)),
    ];
    for (kw, ctor) in word_ops {
        if let Some((lhs, rhs)) = split_once_kw(q, kw) {
            if let (Some(a), Some(b)) = (parse_number(lhs.trim()), parse_number(rhs.trim())) {
                return Some(ctor(a, b));
            }
        }
    }

    // Symbol-form binary infix. Only used when both sides parse as a
    // single number — multi-operator expressions (e.g. "2 + 3 * 4")
    // intentionally fall through here and are handled by `detect_math`
    // via the F.2 expression parser.
    let symbol_ops: &[(char, Builder)] = &[
        ('+', |a, b| ArithmeticIntent::Addition(a, b)),
        ('*', |a, b| ArithmeticIntent::Multiplication(a, b)),
        ('x', |a, b| ArithmeticIntent::Multiplication(a, b)),
        ('/', |a, b| ArithmeticIntent::Division(a, b)),
        ('-', |a, b| ArithmeticIntent::Subtraction(a, b)),
    ];
    for (op, ctor) in symbol_ops {
        if let Some(intent) = try_split_around_op(q, *op, *ctor) {
            return Some(intent);
        }
    }

    None
}

static COMPARE_KEYWORDS: &[(&str, CompareOp)] = &[
    (" greater than ", CompareOp::Greater),
    (" more than ",    CompareOp::Greater),
    (" bigger than ",  CompareOp::Greater),
    (" larger than ",  CompareOp::Greater),
    (" less than ",    CompareOp::Less),
    (" smaller than ", CompareOp::Less),
    (" equal to ",     CompareOp::Equal),
];

fn strip_one_prefix<'a>(s: &'a str, prefixes: &[&str]) -> &'a str {
    for p in prefixes {
        if let Some(rest) = s.strip_prefix(*p) {
            return rest.trim_start();
        }
    }
    s
}

// OP-MATH-DETECT: strip common chain-of-thought / reasoning framings
// that wrap an arithmetic question. Examples handled:
//   "step by step, what is 47 times 13"     -> "what is 47 times 13"
//   "think step by step: 47 times 13"       -> "47 times 13"
//   "let's think step by step. 7 + 5"          -> "7 + 5"
//   "walk me through 47 times 13"           -> "47 times 13"
//
// Pure prefix scrub — never touches operands or operators. Loops once
// so a CoT prefix can sit in front of a question prefix (which is
// stripped by the caller). Trailing punctuation after the framing
// (",", ":", ".") is consumed alongside the framing word.
fn strip_cot_framing(s: &str) -> &str {
    const COT_PREFIXES: &[&str] = &[
        "step by step",
        "step-by-step",
        "think step by step",
        "think step-by-step",
        "let's think step by step",
        "lets think step by step",
        "let me think step by step",
        "let me think",
        "let's think",
        "lets think",
        "think about",
        "think through",
        "reason through",
        "reason about",
        "walk me through",
        "work out",
        "figure out",
        "work through",
        "in detail",
        "explain in detail",
        "please",
        "hey",
        "hi",
    ];
    let mut cur = s.trim_start();
    // Try up to two passes so "please, step by step, ..." also reduces.
    for _ in 0..2 {
        let mut matched = false;
        for p in COT_PREFIXES {
            if let Some(rest) = cur.strip_prefix(*p) {
                // Must be followed by a word boundary — a space or one
                // of the separator punctuation chars we expect after a
                // CoT framing. Bare "thinker" must NOT match "think".
                let next = rest.chars().next();
                let boundary = match next {
                    None => true,
                    Some(c) => matches!(c, ' ' | ',' | ':' | '.' | ';' | '-'),
                };
                if !boundary {
                    continue;
                }
                cur = rest
                    .trim_start_matches(|c: char| matches!(c, ' ' | ',' | ':' | '.' | ';' | '-'));
                matched = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }
    cur
}

fn strip_one_prefix_opt<'a>(s: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    for p in prefixes {
        if let Some(rest) = s.strip_prefix(*p) {
            return Some(rest.trim_start());
        }
    }
    None
}

fn split_once_kw<'a>(haystack: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    haystack
        .find(needle)
        .map(|i| (&haystack[..i], &haystack[i + needle.len()..]))
}

fn try_split_around_op(
    q: &str,
    op: char,
    ctor: fn(f64, f64) -> ArithmeticIntent,
) -> Option<ArithmeticIntent> {
    let mut start = 0usize;
    while let Some(rel) = q[start..].find(op) {
        let abs = start + rel;
        let lhs = q[..abs].trim();
        let rhs = q[abs + op.len_utf8()..].trim();
        if !lhs.is_empty() && !rhs.is_empty() {
            if let (Some(a), Some(b)) = (parse_number(lhs), parse_number(rhs)) {
                return Some(ctor(a, b));
            }
        }
        start = abs + op.len_utf8();
    }
    None
}

/// Parse a single numeric token. F.1 accepted plain integers and
/// decimals; F.2 additionally accepts English word-form numbers
/// ("twelve", "one hundred fifty", "three billion").
///
/// Scientific notation is still rejected — `1e3` does NOT parse, and
/// the `nondetect_scientific_notation_round1` test that pins this
/// behaviour stays green.
fn parse_number(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let s2 = s.trim_end_matches(|c: char| matches!(c, '?' | '!' | ','));
    // Decide whether the token looks numeric (digits / sign / dot only)
    // before deciding which parser to try. This keeps "1e3" from sneaking
    // through the f64 fast path AND keeps tokens like "twelve" — which
    // contain an 'e' but are word numbers — from being misclassified.
    let numeric_looking = !s2.is_empty()
        && s2.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == '+');
    if numeric_looking {
        if let Ok(v) = s2.parse::<f64>() {
            if !v.is_nan() && !v.is_infinite() {
                return Some(v);
            }
        }
        return None;
    }
    // Fall back to word-number parsing for cases like "twelve".
    word_num::parse(s)
}

// =====================================================================
// F.2 unified entry — detect_math
// =====================================================================

/// A computed answer from any F.2 capability. Carries everything mcp.rs
/// needs to render `content[0].text` and `structuredContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum MathOutcome {
    /// Result of the recursive-descent expression parser (functions,
    /// constants, multi-operator infix, parens, etc.).
    Expression {
        input: String,
        value: f64,
        prose: String,
    },
    /// A unit conversion (time / distance / weight / temperature).
    UnitConversion {
        conversion: units::UnitConversion,
        prose: String,
    },
    /// A descriptive statistic over a comma- or space-separated list.
    Stat {
        op: String,
        values: Vec<f64>,
        result: f64,
        prose: String,
    },
    /// One-unknown linear solve.
    Solve {
        equation: String,
        x: f64,
        prose: String,
    },
    /// Detection succeeded but computation failed (e.g. divide-by-zero
    /// inside a parsed expression). Still routed through the math
    /// surface so callers can distinguish a failed math attempt from a
    /// recall miss.
    Error {
        route: String,
        error: String,
        prose: String,
    },
}

impl MathOutcome {
    pub fn prose(&self) -> &str {
        match self {
            MathOutcome::Expression { prose, .. }
            | MathOutcome::UnitConversion { prose, .. }
            | MathOutcome::Stat { prose, .. }
            | MathOutcome::Solve { prose, .. }
            | MathOutcome::Error { prose, .. } => prose,
        }
    }
}

/// Try every F.2 detector in priority order and return the first hit.
/// Returns `None` if the query is not math-shaped — caller falls back
/// to recall as before.
/// Contained geometry detector for the F.2 math cascade: recognizes
/// "area of a circle [with] radius N" (and `r = N` / `r N` forms) and
/// returns `(radius, pi * r^2)`. English geometry phrasing does not parse as
/// an arithmetic expression, so without this it falls through to recall.
fn detect_circle_area(q: &str) -> Option<(f64, f64)> {
    let s = q.to_lowercase();
    if !(s.contains("area") && s.contains("circle")) {
        return None;
    }
    let norm = s.replace('=', " ");
    let toks: Vec<&str> = norm.split_whitespace().collect();
    let mut radius: Option<f64> = None;
    for (i, t) in toks.iter().enumerate() {
        if *t == "radius" || *t == "r" {
            for t2 in toks[i + 1..].iter() {
                let cleaned = t2.trim_matches(|c: char| {
                    !(c.is_ascii_digit() || c == '.' || c == '-')
                });
                if let Ok(v) = cleaned.parse::<f64>() {
                    radius = Some(v);
                    break;
                }
            }
            if radius.is_some() {
                break;
            }
        }
    }
    let r = radius?;
    if !r.is_finite() || r < 0.0 {
        return None;
    }
    Some((r, std::f64::consts::PI * r * r))
}

pub fn detect_math(query: &str) -> Option<MathOutcome> {
    let lower = query.trim().to_lowercase();
    let stripped: &str = lower
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();
    let q = strip_one_prefix(stripped, &[
        "what is the value of ",
        "what is ",
        "whats ",
        "what's ",
        "compute ",
        "calculate ",
        "evaluate ",
    ]);

    // 1) Solver — must come first because "solve" prefix changes meaning.
    if let Some(eq) = q.strip_prefix("solve ") {
        return Some(match solve::solve_linear(eq) {
            Ok(s) => MathOutcome::Solve {
                equation: s.equation.clone(),
                x: s.x,
                prose: format!("x = {} (computed)", fmt_num(s.x)),
            },
            Err(e) => MathOutcome::Error {
                route: "solve".into(),
                error: e.clone(),
                prose: format!("Cannot solve '{}': {}", eq.trim(), e),
            },
        });
    }

    // 2) Statistics — "mean of 1,2,3", "sum 4 5 6", etc.
    if let Some((op, values)) = stats::detect(q) {
        return Some(match stats::compute(op, &values) {
            Ok(r) => MathOutcome::Stat {
                op: op.name().to_string(),
                values: values.clone(),
                result: r,
                prose: format!("{} of {} = {} (computed)", op.name(), fmt_list(&values), fmt_num(r)),
            },
            Err(e) => MathOutcome::Error {
                route: "stat".into(),
                error: e.clone(),
                prose: format!("Cannot compute {}: {}", op.name(), e),
            },
        });
    }

    // 3) Unit conversion.
    if let Some((value, from, to)) = units::detect(q) {
        return Some(match units::convert(value, &from, &to) {
            Ok(c) => {
                let p = format!(
                    "{} {} = {} {} (computed)",
                    fmt_num(value),
                    from,
                    fmt_num(c.result),
                    to,
                );
                MathOutcome::UnitConversion { conversion: c, prose: p }
            }
            Err(e) => MathOutcome::Error {
                route: "unit_conversion".into(),
                error: e.clone(),
                prose: format!("Cannot convert {} {} to {}: {}", fmt_num(value), from, to, e),
            },
        });
    }

    // 3.5) Geometry: area of a circle (pi * r^2). English forms like
    //      "area of a circle with radius 3" do not parse as an arithmetic
    //      expression, so without this they fall through to recall. Part of
    //      the existing F.2 math cascade (no new component).
    if let Some((r, area)) = detect_circle_area(q) {
        return Some(MathOutcome::Expression {
            input: q.to_string(),
            value: area,
            prose: format!(
                "area of a circle with radius {} = {} (pi * r^2, computed)",
                fmt_num(r),
                fmt_num(area),
            ),
        });
    }

    // 4) Expression parser. Conservative gate: must parse cleanly,
    //    must contain at least one operation (so a bare literal or
    //    bare identifier doesn't trigger), and must evaluate to a
    //    finite (or ±∞) number.
    match expr::parse(q) {
        Ok(e) if expr::has_operation(&e) => match expr::eval(&e, None) {
            Ok(v) => Some(MathOutcome::Expression {
                input: q.to_string(),
                value: v,
                prose: format!("{} = {} (computed)", q.trim(), fmt_num(v)),
            }),
            Err(err) => Some(MathOutcome::Error {
                route: "expression".into(),
                error: err.clone(),
                prose: format!("Cannot evaluate '{}': {}", q.trim(), err),
            }),
        },
        // Parse failed OR result is a literal — fall through to recall.
        _ => None,
    }
}

fn fmt_list(xs: &[f64]) -> String {
    let parts: Vec<String> = xs.iter().copied().map(fmt_num).collect();
    parts.join(", ")
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- F.1 detection (preserved) ----------

    #[test]
    fn detect_addition_word_form() {
        assert_eq!(
            detect_arithmetic("what is 7 plus 5"),
            Some(ArithmeticIntent::Addition(7.0, 5.0))
        );
    }

    #[test]
    fn detect_addition_symbol_form() {
        assert_eq!(
            detect_arithmetic("what is 7 + 5"),
            Some(ArithmeticIntent::Addition(7.0, 5.0))
        );
        assert_eq!(
            detect_arithmetic("what is 7 + 5?"),
            Some(ArithmeticIntent::Addition(7.0, 5.0))
        );
    }

    #[test]
    fn detect_subtraction() {
        assert_eq!(
            detect_arithmetic("what is 10 minus 4"),
            Some(ArithmeticIntent::Subtraction(10.0, 4.0))
        );
        assert_eq!(
            detect_arithmetic("what is 10 - 4"),
            Some(ArithmeticIntent::Subtraction(10.0, 4.0))
        );
    }

    #[test]
    fn detect_multiplication() {
        assert_eq!(
            detect_arithmetic("what is 6 times 7"),
            Some(ArithmeticIntent::Multiplication(6.0, 7.0))
        );
        assert_eq!(
            detect_arithmetic("what is 6 * 7"),
            Some(ArithmeticIntent::Multiplication(6.0, 7.0))
        );
        assert_eq!(
            detect_arithmetic("what is 6 x 7"),
            Some(ArithmeticIntent::Multiplication(6.0, 7.0))
        );
    }

    #[test]
    fn detect_division() {
        assert_eq!(
            detect_arithmetic("what is 20 divided by 4"),
            Some(ArithmeticIntent::Division(20.0, 4.0))
        );
        assert_eq!(
            detect_arithmetic("what is 20 / 4"),
            Some(ArithmeticIntent::Division(20.0, 4.0))
        );
    }

    #[test]
    fn detect_percent_of() {
        assert_eq!(
            detect_arithmetic("what is 25 percent of 80"),
            Some(ArithmeticIntent::Percent { percent: 25.0, of: 80.0 })
        );
        assert_eq!(
            detect_arithmetic("what is 25% of 80"),
            Some(ArithmeticIntent::Percent { percent: 25.0, of: 80.0 })
        );
    }

    #[test]
    fn detect_square_root() {
        assert_eq!(
            detect_arithmetic("what is the square root of 16"),
            Some(ArithmeticIntent::Sqrt(16.0))
        );
        assert_eq!(
            detect_arithmetic("what is sqrt(81)"),
            Some(ArithmeticIntent::Sqrt(81.0))
        );
    }

    #[test]
    fn detect_comparison_greater() {
        assert_eq!(
            detect_arithmetic("is 100 greater than 50"),
            Some(ArithmeticIntent::Compare {
                left: 100.0,
                op: CompareOp::Greater,
                right: 50.0,
            })
        );
    }

    #[test]
    fn detect_comparison_less() {
        assert_eq!(
            detect_arithmetic("is 3 less than 9"),
            Some(ArithmeticIntent::Compare {
                left: 3.0,
                op: CompareOp::Less,
                right: 9.0,
            })
        );
    }

    #[test]
    fn detect_half_of() {
        assert_eq!(
            detect_arithmetic("what is half of 200"),
            Some(ArithmeticIntent::Division(200.0, 2.0))
        );
    }

    #[test]
    fn detect_decimal_operands() {
        assert_eq!(
            detect_arithmetic("what is 12.5 plus 0.5"),
            Some(ArithmeticIntent::Addition(12.5, 0.5))
        );
    }

    // ---------- non-detection ----------

    #[test]
    fn nondetect_concept_query() {
        assert_eq!(detect_arithmetic("what is mitochondria"), None);
    }

    #[test]
    fn nondetect_mixed_query_falls_through() {
        assert_eq!(detect_arithmetic("what is 7 plus mitochondria"), None);
        assert_eq!(detect_arithmetic("what is mitochondria + 5"), None);
    }

    #[test]
    fn nondetect_x_inside_word() {
        assert_eq!(detect_arithmetic("what is the boxer's weight"), None);
        assert_eq!(detect_arithmetic("what is texas"), None);
    }

    #[test]
    fn nondetect_scientific_notation_round1() {
        assert_eq!(detect_arithmetic("what is 1e3 plus 5"), None);
    }

    // ---------- OP-MATH-DETECT: CoT-framed word arithmetic ----------

    #[test]
    fn opmd_step_by_step_what_is_times() {
        // Original failure prompt from OP-ROUTE-3c canonical sanity.
        assert_eq!(
            detect_arithmetic("step by step, what is 47 times 13"),
            Some(ArithmeticIntent::Multiplication(47.0, 13.0))
        );
    }

    #[test]
    fn opmd_multiplied_by_no_framing() {
        assert_eq!(
            detect_arithmetic("47 multiplied by 13"),
            Some(ArithmeticIntent::Multiplication(47.0, 13.0))
        );
    }

    #[test]
    fn opmd_plus_bare() {
        assert_eq!(
            detect_arithmetic("47 plus 13"),
            Some(ArithmeticIntent::Addition(47.0, 13.0))
        );
    }

    #[test]
    fn opmd_minus_bare() {
        assert_eq!(
            detect_arithmetic("100 minus 23"),
            Some(ArithmeticIntent::Subtraction(100.0, 23.0))
        );
    }

    #[test]
    fn opmd_divided_by_bare() {
        assert_eq!(
            detect_arithmetic("100 divided by 4"),
            Some(ArithmeticIntent::Division(100.0, 4.0))
        );
    }

    #[test]
    fn opmd_47_times_13_evaluates_to_611() {
        let intent = detect_arithmetic("step by step, what is 47 times 13")
            .expect("should detect multiplication");
        let r = intent.compute().expect("should compute");
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 611.0),
            _ => panic!("expected Number(611)"),
        }
    }

    #[test]
    fn opmd_cot_variants() {
        for q in [
            "think step by step: 47 times 13",
            "step-by-step what is 47 times 13",
            "let's think step by step. what is 47 times 13?",
            "walk me through 47 times 13",
            "please compute 47 times 13",
            "work through 47 times 13",
            "think through what is 47 times 13",
        ] {
            assert_eq!(
                detect_arithmetic(q),
                Some(ArithmeticIntent::Multiplication(47.0, 13.0)),
                "failed on: {q}"
            );
        }
    }

    #[test]
    fn opmd_router_sets_needs_math_for_original_failure() {
        // Closes OP-ROUTE-3c gap. detect_arithmetic firing is the
        // structural precondition for QueryRouter::route -> needs_math=true.
        // We assert the precondition here; query_router.rs has its own
        // tests for the routing surface.
        assert!(
            detect_arithmetic("step by step, what is 47 times 13").is_some(),
            "detect_arithmetic must fire so router sets needs_math=true"
        );
    }

    #[test]
    fn opmd_negative_not_arithmetic_with_cot() {
        assert_eq!(detect_arithmetic("step by step, i love you"), None);
        assert_eq!(detect_arithmetic("i love you"), None);
        // CoT-stripped to "how does fire start" — still not arithmetic.
        assert_eq!(detect_arithmetic("step by step, how does fire start"), None);
    }

    #[test]
    fn opmd_word_boundary_does_not_eat_word() {
        // "thinker is fast" must NOT be stripped to "er is fast".
        // detect_arithmetic returns None either way, but exercise the
        // boundary check by inspecting strip_cot_framing directly.
        assert_eq!(strip_cot_framing("thinker is fast"), "thinker is fast");
        assert_eq!(strip_cot_framing("please, step by step, what is 7 plus 5"), "what is 7 plus 5");
    }

    // ---------- compute (F.1) ----------

    #[test]
    fn compute_addition_returns_correct_value() {
        let r = ArithmeticIntent::Addition(7.0, 5.0).compute().unwrap();
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 12.0),
            _ => panic!("expected Number"),
        }
        assert!(r.prose.contains("12"));
        assert!(r.prose.contains("computed"));
    }

    #[test]
    fn compute_subtraction() {
        let r = ArithmeticIntent::Subtraction(10.0, 4.0).compute().unwrap();
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 6.0),
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn compute_multiplication() {
        let r = ArithmeticIntent::Multiplication(6.0, 7.0).compute().unwrap();
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 42.0),
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn compute_division_by_zero_returns_error() {
        let err = ArithmeticIntent::Division(5.0, 0.0).compute();
        assert!(err.is_err());
        assert!(err.unwrap_err().to_lowercase().contains("zero"));
    }

    #[test]
    fn compute_square_root_negative_returns_error() {
        let err = ArithmeticIntent::Sqrt(-1.0).compute();
        assert!(err.is_err());
    }

    #[test]
    fn compute_square_root_positive() {
        let r = ArithmeticIntent::Sqrt(16.0).compute().unwrap();
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 4.0),
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn compute_percent() {
        let r = ArithmeticIntent::Percent { percent: 25.0, of: 80.0 }.compute().unwrap();
        match r.value {
            ArithmeticValue::Number(n) => assert_eq!(n, 20.0),
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn compute_compare_greater_true() {
        let r = ArithmeticIntent::Compare {
            left: 100.0,
            op: CompareOp::Greater,
            right: 50.0,
        }
        .compute()
        .unwrap();
        match r.value {
            ArithmeticValue::Boolean(b) => assert!(b),
            _ => panic!("expected Boolean"),
        }
        assert!(r.prose.to_lowercase().contains("yes"));
    }

    #[test]
    fn compute_compare_greater_false() {
        let r = ArithmeticIntent::Compare {
            left: 5.0,
            op: CompareOp::Greater,
            right: 50.0,
        }
        .compute()
        .unwrap();
        match r.value {
            ArithmeticValue::Boolean(b) => assert!(!b),
            _ => panic!("expected Boolean"),
        }
    }

    // ---------- F.2 word-number arithmetic via legacy detector ------

    #[test]
    fn f2_word_numbers_in_legacy_detector() {
        assert_eq!(
            detect_arithmetic("twelve plus thirteen"),
            Some(ArithmeticIntent::Addition(12.0, 13.0))
        );
        assert_eq!(
            detect_arithmetic("one hundred fifty minus fifty"),
            Some(ArithmeticIntent::Subtraction(150.0, 50.0))
        );
    }

    // ---------- F.2 unified detect_math ----------

    #[test]
    fn detect_math_expression_routes_through_parser() {
        let r = detect_math("what is 2 + 3 * 4").unwrap();
        match r {
            MathOutcome::Expression { value, .. } => assert_eq!(value, 14.0),
            _ => panic!("expected Expression outcome"),
        }
    }

    #[test]
    fn detect_math_paren_grouping() {
        let r = detect_math("(7 + 5) * (3 - 1)").unwrap();
        match r {
            MathOutcome::Expression { value, .. } => assert_eq!(value, 24.0),
            _ => panic!("expected Expression outcome"),
        }
    }

    #[test]
    fn detect_math_function_call() {
        let r = detect_math("sqrt(144)").unwrap();
        match r {
            MathOutcome::Expression { value, .. } => assert_eq!(value, 12.0),
            _ => panic!("expected Expression outcome"),
        }
    }

    #[test]
    fn detect_math_constant_pi() {
        let r = detect_math("2 * pi").unwrap();
        match r {
            MathOutcome::Expression { value, .. } => {
                assert!((value - 6.283_185_307).abs() < 1e-6)
            }
            _ => panic!("expected Expression outcome"),
        }
    }

    #[test]
    fn detect_circle_area_computes_pi_r_squared() {
        let o = detect_math("area of a circle with radius 3").expect("circle area is math");
        if let MathOutcome::Expression { value, .. } = o {
            assert!((value - std::f64::consts::PI * 9.0).abs() < 1e-9, "pi*3^2");
        } else {
            panic!("expected Expression for circle area");
        }
        assert!(detect_math("area of a circle radius 5").is_some());
        assert!(detect_math("what is the area of a circle with radius 2.5").is_some());
        // must NOT hijack the other math routes:
        assert!(matches!(detect_math("mean of 2, 4, 6"), Some(MathOutcome::Stat { .. })));
        assert!(matches!(detect_math("convert 5 km to miles"), Some(MathOutcome::UnitConversion { .. })));
        // a bare circle mention (no radius) is not math.
        assert!(detect_circle_area("what is a circle").is_none());
    }

    #[test]
    fn detect_math_unit_conversion() {
        let r = detect_math("convert 100 km to miles").unwrap();
        match r {
            MathOutcome::UnitConversion { conversion, .. } => {
                assert!((conversion.result - 62.137_119).abs() < 1e-3);
            }
            _ => panic!("expected UnitConversion outcome"),
        }
    }

    #[test]
    fn detect_math_temperature_conversion() {
        let r = detect_math("convert 32 F to C").unwrap();
        match r {
            MathOutcome::UnitConversion { conversion, .. } => {
                assert!((conversion.result).abs() < 1e-9)
            }
            _ => panic!("expected UnitConversion outcome"),
        }
    }

    #[test]
    fn detect_math_seconds_in_a_day() {
        let r = detect_math("how many seconds in a day").unwrap();
        match r {
            MathOutcome::UnitConversion { conversion, .. } => {
                assert_eq!(conversion.result, 86_400.0)
            }
            _ => panic!("expected UnitConversion outcome"),
        }
    }

    #[test]
    fn detect_math_stat_mean() {
        let r = detect_math("mean of 1, 2, 3, 4, 5").unwrap();
        match r {
            MathOutcome::Stat { result, op, .. } => {
                assert_eq!(result, 3.0);
                assert_eq!(op, "mean");
            }
            _ => panic!("expected Stat outcome"),
        }
    }

    #[test]
    fn detect_math_stat_stddev() {
        let r = detect_math("stddev of 2, 4, 4, 4, 5, 5, 7, 9").unwrap();
        match r {
            MathOutcome::Stat { result, .. } => assert!((result - 2.0).abs() < 1e-9),
            _ => panic!("expected Stat outcome"),
        }
    }

    #[test]
    fn detect_math_solve_simple() {
        let r = detect_math("solve x + 5 = 10").unwrap();
        match r {
            MathOutcome::Solve { x, .. } => assert!((x - 5.0).abs() < 1e-9),
            _ => panic!("expected Solve outcome"),
        }
    }

    #[test]
    fn detect_math_solve_two_x() {
        let r = detect_math("solve 2x = 14").unwrap();
        match r {
            MathOutcome::Solve { x, .. } => assert!((x - 7.0).abs() < 1e-9),
            _ => panic!("expected Solve outcome"),
        }
    }

    #[test]
    fn detect_math_solve_quadratic_rejects() {
        let r = detect_math("solve x^2 = 9").unwrap();
        match r {
            MathOutcome::Error { route, .. } => assert_eq!(route, "solve"),
            _ => panic!("expected Error outcome for non-linear"),
        }
    }

    #[test]
    fn detect_math_falls_through_for_concept_query() {
        // Pure recall query — math must not fire.
        assert!(detect_math("what is mitochondria").is_none());
        assert!(detect_math("who is the president of the moon").is_none());
    }

    #[test]
    fn detect_math_bare_literal_falls_through() {
        // "what is 5" parses as a literal; we deliberately do NOT
        // route bare numbers through math.
        assert!(detect_math("5").is_none());
    }

    #[test]
    fn detect_math_division_by_zero_inside_expr() {
        let r = detect_math("5 / (3 - 3)").unwrap();
        match r {
            MathOutcome::Error { route, .. } => assert_eq!(route, "expression"),
            other => panic!("expected Error outcome, got {:?}", other),
        }
    }
}
