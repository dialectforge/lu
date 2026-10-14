//! Port F.3 — word problem extraction with honest hedging.
//!
//! When a query is shaped like a textbook arithmetic word problem
//! (prose with numbers, an operation cue verb, and a question),
//! Lu reads the prose, lifts the numbers and the operation, and
//! commits to a computed answer — or, when the extraction is too
//! ambiguous to commit to, returns an honest hedge that surfaces
//! the extraction state instead of fabricating.
//!
//! Cascade order in `mcp::try_arithmetic_response` is
//!     F.1 single-op → F.4 construction → F.3 word problem → F.2 expression → recall
//! so a prose-shaped query gets first crack at F.3 before falling
//! through to F.2's expression parser (which would fail on
//! "Sarah had 5 apples. She gave 2 to Mary.").
//!
//! Round 1 IN scope: simple add/subtract, multiply/divide, percent
//! (off / tip / of), per-rate two-step (earn $X/hr × Y hrs, pages /
//! pages-per-day), "less than the sum of", and the boxes-of-eggs
//! "there are N X" pattern.
//!
//! Round 1 OUT of scope (returns honest hedge instead of fabricating):
//!   - algebra needing a system of equations beyond F.2's linear solver
//!   - multi-step problems with hidden intermediate values
//!   - trick problems (bat-and-ball, hen-and-a-half)
//!   - construction-flavored word problems (deferred to F.3.1 — direct
//!     construction is already covered by F.4)
//!   - problems requiring world knowledge (slices in a pizza,
//!     days in a year unless explicit)
//!
//! Lu's structural advantage is provenance — when extraction is shaky
//! we return `(word problem, hedged — verify before relying)` rather
//! than guess. Every answer here is tagged with `source = "wordprob_primitive"`
//! and a `confidence` flag (`confident` or `hedged`).

use serde::{Deserialize, Serialize};

use super::{fmt_num, word_num};

const FLAG_COMPUTED: &str = "(word problem, computed)";
const FLAG_HEDGED: &str = "(word problem, hedged — verify before relying)";

// =====================================================================
// Public types
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordOp {
    Add,
    Sub,
    Mul,
    Div,
    PercentOff,    // base - (pct/100) * base       — sale price
    PercentAdd,    // base + (pct/100) * base       — tip / increase
    PercentOf,     // (pct/100) * base              — straight take
    LessThanSumOf, // (b + c) - a
    Speed,         // distance / time
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordConfidence {
    Confident,
    Hedged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WordProblemOutcome {
    /// A computed answer with provenance. `confidence` distinguishes
    /// a clean extraction from a best-effort one.
    Computed {
        confidence: WordConfidence,
        numbers: Vec<f64>,
        op: WordOp,
        result: f64,
        prose: String,
    },
    /// We see a word-problem shape but the extraction is too ambiguous
    /// to commit to a single answer. Optionally carries a naive guess
    /// the user can sanity-check.
    Hedge {
        numbers: Vec<f64>,
        cues: Vec<String>,
        guess: Option<f64>,
        prose: String,
    },
}

impl WordProblemOutcome {
    pub fn prose(&self) -> &str {
        match self {
            WordProblemOutcome::Computed { prose, .. }
            | WordProblemOutcome::Hedge { prose, .. } => prose,
        }
    }

    pub fn is_hedged(&self) -> bool {
        matches!(
            self,
            WordProblemOutcome::Hedge { .. }
                | WordProblemOutcome::Computed {
                    confidence: WordConfidence::Hedged,
                    ..
                }
        )
    }
}

// =====================================================================
// Detection — "does this look word-problem-shaped?"
// =====================================================================

/// Narrative verbs that suggest subject-verb-object prose rather than
/// calculator-style imperative ("convert ...", "compute ..."). Bare
/// auxiliaries like "is" / "are" intentionally NOT included — they're
/// too common in non-prose queries ("what is mitochondria").
const PROSE_VERBS: &[&str] = &[
    "has", "had", "have", "contains", "holds", "contain",
    "gives", "gave", "giving", "give",
    "buys", "bought", "buying", "buy",
    "splits", "split", "shared", "shares", "sharing", "share",
    "earns", "earned", "earning", "earn",
    "costs", "cost",
    "sells", "sold", "selling",
    "works", "worked", "working", "work",
    "reads", "read", "reading",
    "runs", "ran", "running",
    "eats", "ate", "eaten",
    "spends", "spent", "spending", "spend",
    "loses", "lost", "losing",
    "travels", "traveled", "travelling", "travelled", "travel",
    "absent", "present", "missing",
    "receives", "received", "receiving", "receive",
    "combined", "combines",
    "purchase", "purchased", "purchases",
    "drives", "drove", "driving",
    "walks", "walked", "walking",
    "flies", "flew", "flying",
];

/// Phrases that, on their own, are strong enough signals to fire F.3
/// even without other prose markers.
const TRIGGER_PHRASES: &[&str] = &[
    "less than the sum of",
    "more than the sum of",
    "% off",
    " percent off",
    " discount of",
];

pub fn looks_like_word_problem(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return false;
    }

    // Imperative calculator phrasing — F.2 territory, not a word problem.
    // (Caught here so "solve 2x = 14" doesn't fall through to F.3 just
    // because of the `2x` algebra shape.)
    if q.starts_with("solve ")
        || q.starts_with("compute ")
        || q.starts_with("calculate ")
        || q.starts_with("evaluate ")
        || q.starts_with("convert ")
    {
        return false;
    }

    if TRIGGER_PHRASES.iter().any(|p| q.contains(p)) {
        return true;
    }

    // Algebra-shape patterns (twice/thrice/double + variable letter).
    if has_algebraic_shape(&q) {
        return true;
    }

    let cleaned = preclean(query);
    let nums = extract_numbers(query);
    if nums.is_empty() {
        return false;
    }

    let padded = format!(" {} ", cleaned);
    let has_prose_verb = PROSE_VERBS
        .iter()
        .any(|v| padded.contains(&format!(" {} ", v)));
    let multiple_sentences = query.matches(". ").count() >= 1;

    has_prose_verb || multiple_sentences
}

// =====================================================================
// Number extraction
// =====================================================================

#[derive(Debug, Clone)]
struct NumToken {
    value: f64,
    has_dollar: bool,
    has_percent: bool,
    /// Token-stream position — used so callers can pick "the first
    /// number" or "the second number" deterministically.
    #[allow(dead_code)]
    pos: usize,
}

#[derive(Debug, Clone, Copy)]
struct ParsedNum {
    value: f64,
    has_dollar: bool,
    has_percent: bool,
}

fn extract_numbers(query: &str) -> Vec<NumToken> {
    let cleaned = preclean(query);
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        if let Some(num) = parse_numeric_token(t) {
            out.push(NumToken {
                value: num.value,
                has_dollar: num.has_dollar,
                has_percent: num.has_percent,
                pos: i,
            });
            i += 1;
            continue;
        }
        if is_word_num_starter(t) {
            let start = i;
            let mut end = i + 1;
            while end < tokens.len() && is_word_num_continuer(tokens[end]) {
                end += 1;
            }
            // Trim trailing connectives like "and" so we don't fold a
            // sentence-bridging "and" into the number.
            while end > start + 1 && tokens[end - 1] == "and" {
                end -= 1;
            }
            let phrase = tokens[start..end].join(" ");
            if let Some(v) = word_num::parse(&phrase) {
                out.push(NumToken {
                    value: v,
                    has_dollar: false,
                    has_percent: false,
                    pos: start,
                });
            }
            i = end;
            continue;
        }
        i += 1;
    }
    out
}

fn parse_numeric_token(t: &str) -> Option<ParsedNum> {
    let mut s = t;
    let mut has_dollar = false;
    let mut has_percent = false;
    if let Some(rest) = s.strip_prefix('$') {
        has_dollar = true;
        s = rest;
    }
    // Strip trailing punctuation including %.
    loop {
        let last = match s.chars().last() {
            Some(c) => c,
            None => break,
        };
        match last {
            '%' => {
                has_percent = true;
                let new_len = s.len() - last.len_utf8();
                s = &s[..new_len];
            }
            '.' | ',' | '?' | '!' | ';' | ':' | '"' | ')' => {
                let new_len = s.len() - last.len_utf8();
                s = &s[..new_len];
            }
            _ => break,
        }
    }
    if s.is_empty() {
        return None;
    }
    // Strip thousands separators ("1,000" → "1000").
    let s_clean: String = s.chars().filter(|c| *c != ',').collect();
    // Reject scientific notation (matches existing math rules).
    if s_clean.bytes().any(|b| b == b'e' || b == b'E') {
        return None;
    }
    if !s_clean
        .chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == '+')
    {
        return None;
    }
    let v: f64 = s_clean.parse().ok()?;
    if v.is_nan() || v.is_infinite() {
        return None;
    }
    Some(ParsedNum {
        value: v,
        has_dollar,
        has_percent,
    })
}

fn is_word_num_starter(t: &str) -> bool {
    matches!(
        t,
        "zero"
            | "naught"
            | "nought"
            | "one"
            | "two"
            | "three"
            | "four"
            | "five"
            | "six"
            | "seven"
            | "eight"
            | "nine"
            | "ten"
            | "eleven"
            | "twelve"
            | "thirteen"
            | "fourteen"
            | "fifteen"
            | "sixteen"
            | "seventeen"
            | "eighteen"
            | "nineteen"
            | "twenty"
            | "thirty"
            | "forty"
            | "fifty"
            | "sixty"
            | "seventy"
            | "eighty"
            | "ninety"
    )
}

fn is_word_num_continuer(t: &str) -> bool {
    is_word_num_starter(t)
        || matches!(
            t,
            "hundred" | "thousand" | "million" | "billion" | "and"
        )
}

/// Lowercase `query` and replace sentence-level punctuation with
/// spaces, but keep `.` and `,` when they sit between digits (so
/// "1.5" and "1,000" survive intact). Preserves `$` and `%`
/// because those carry meaning for our extractor.
fn preclean(query: &str) -> String {
    let lower = query.trim().to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(lower.len());
    for i in 0..n {
        let c = chars[i];
        match c {
            '.' | ',' => {
                let prev_d = i > 0 && chars[i - 1].is_ascii_digit();
                let next_d = i + 1 < n && chars[i + 1].is_ascii_digit();
                if prev_d && next_d {
                    out.push(c);
                } else {
                    out.push(' ');
                }
            }
            '?' | '!' | ';' | ':' | '"' | '(' | ')' | '\'' => out.push(' '),
            other => out.push(other),
        }
    }
    out
}

// =====================================================================
// Cue / shape detection helpers
// =====================================================================

fn has_word(haystack: &str, word: &str) -> bool {
    let padded = format!(" {} ", haystack);
    padded.contains(&format!(" {} ", word))
}

fn has_any_word(haystack: &str, words: &[&str]) -> bool {
    let padded = format!(" {} ", haystack);
    words
        .iter()
        .any(|w| padded.contains(&format!(" {} ", w)))
}

fn has_phrase(haystack: &str, phrase: &str) -> bool {
    haystack.contains(phrase)
}

/// "X more than Y" relational pattern (bat-and-ball): "the bat costs
/// $1 more than the ball" or "Alice has 3 more apples than Bob".
/// Also "as many as Y" / "as much as Y" — these need algebra to
/// resolve cleanly.
fn has_relational_more_than(query: &str) -> bool {
    let q = query.to_lowercase();

    // The Round-1 supported "less than the sum of" shape is NOT
    // relational — short-circuit before any "less than" check fires.
    let is_supported_less_than_sum = q.contains("less than the sum of");

    if !is_supported_less_than_sum {
        if q.contains(" more than the ") || q.contains(" less than the ") {
            return true;
        }
    }
    if q.contains(" as much as ") || q.contains(" as many as ") {
        return true;
    }
    if q.contains(" twice as ") || q.contains(" thrice as ") || q.contains(" double as ") {
        return true;
    }

    // "X more <noun> than Y" — "more" followed by "than" with a word
    // between. Catches "3 more apples than Bob".
    if !is_supported_less_than_sum {
        let padded = format!(" {} ", q);
        if let Some(more_idx) = padded.find(" more ") {
            // " more " is 6 chars; window of "than" within ~30 chars
            // after rules out unrelated "more" / "than" pairs.
            let after_more = &padded[more_idx + 6..];
            if let Some(than_rel) = after_more.find(" than ") {
                let between = &after_more[..than_rel];
                if !between.trim().is_empty() && than_rel < 30 {
                    return true;
                }
            }
        }
    }

    false
}

/// Algebraic shape: "twice y", "if x is", "x + y", etc. Round 1
/// hedges instead of attempting a setup-of-equations solve.
fn has_algebraic_shape(query: &str) -> bool {
    let q = query.to_lowercase();
    let padded = format!(" {} ", q);
    if padded.contains(" if x ") || padded.contains(" if y ") || padded.contains(" if n ") {
        return true;
    }
    // "x + y" / "x = " / "2x"
    if q.contains("x +") || q.contains("x=") || q.contains(" x = ") || q.contains(" 2x ") {
        return true;
    }
    if (padded.contains(" twice ") || padded.contains(" thrice ") || padded.contains(" double ") || padded.contains(" half as "))
        && (padded.contains(" x ") || padded.contains(" y ") || padded.contains(" n "))
    {
        return true;
    }
    false
}

// =====================================================================
// detect_word_problem — entrypoint
// =====================================================================

pub fn detect_word_problem(query: &str) -> Option<WordProblemOutcome> {
    if !looks_like_word_problem(query) {
        return None;
    }

    let cleaned = preclean(query);
    let nums = extract_numbers(query);

    // ---- algebra / trick patterns: hedge first ----
    if has_algebraic_shape(query) {
        return Some(make_hedge_no_guess(
            &nums,
            "I see twice/thrice/double or an x/y variable, which usually needs algebra to set up.",
        ));
    }
    if has_relational_more_than(query) {
        // Bat-and-ball naive guess: smallest minus next smallest.
        let guess = if nums.len() >= 2 {
            let mut sorted: Vec<f64> = nums.iter().map(|n| n.value).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(sorted[1] - sorted[0])
        } else {
            None
        };
        let prose_guess = guess
            .map(|g| format!("a naive subtract gives {}", fmt_num(g)))
            .unwrap_or_else(|| "no clean guess available".to_string());
        return Some(make_hedge_with_optional_guess(
            &nums,
            &format!(
                "I see an 'X more than Y' relational pattern (bat-and-ball shape) which needs algebra; {}",
                prose_guess
            ),
            guess,
        ));
    }

    // ---- "less than the sum of" ----
    if cleaned.contains("less than the sum of") && nums.len() >= 3 {
        let a = nums[0].value;
        let b = nums[1].value;
        let c = nums[2].value;
        let result = (b + c) - a;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![a, b, c],
            op: WordOp::LessThanSumOf,
            result,
            prose: format!(
                "{} less than the sum of {} and {} → ({} + {}) - {} = {} {}",
                fmt_num(a),
                fmt_num(b),
                fmt_num(c),
                fmt_num(b),
                fmt_num(c),
                fmt_num(a),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- percent off / discount ----
    if (cleaned.contains("% off")
        || cleaned.contains(" percent off")
        || cleaned.contains("discount"))
        && nums.len() >= 2
    {
        let pct = nums.iter().find(|n| n.has_percent).map(|n| n.value);
        let price = nums.iter().find(|n| n.has_dollar).map(|n| n.value);
        if let (Some(p), Some(price_v)) = (pct, price) {
            let result = price_v - (p / 100.0) * price_v;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Confident,
                numbers: vec![price_v, p],
                op: WordOp::PercentOff,
                result,
                prose: format!(
                    "${} with {}% off → ${} - ({}/100 × ${}) = ${} {}",
                    fmt_num(price_v),
                    fmt_num(p),
                    fmt_num(price_v),
                    fmt_num(p),
                    fmt_num(price_v),
                    fmt_num(result),
                    FLAG_COMPUTED,
                ),
            });
        }
        // Best-effort fallback: assume larger number is the price.
        if nums.len() >= 2 {
            let a = nums[0].value;
            let b = nums[1].value;
            let (price_v, p) = if a >= b { (a, b) } else { (b, a) };
            let result = price_v - (p / 100.0) * price_v;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Hedged,
                numbers: vec![price_v, p],
                op: WordOp::PercentOff,
                result,
                prose: format!(
                    "Best guess: {}% off {} → {}. {}",
                    fmt_num(p),
                    fmt_num(price_v),
                    fmt_num(result),
                    FLAG_HEDGED,
                ),
            });
        }
    }

    // ---- tip / total with % ----
    if (cleaned.contains("tip") || cleaned.contains(" tax "))
        && (cleaned.contains("%") || has_word(&cleaned, "percent"))
        && nums.len() >= 2
    {
        let pct = nums.iter().find(|n| n.has_percent).map(|n| n.value);
        let base = nums.iter().find(|n| n.has_dollar).map(|n| n.value);
        if let (Some(p), Some(base_v)) = (pct, base) {
            let result = base_v + (p / 100.0) * base_v;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Confident,
                numbers: vec![base_v, p],
                op: WordOp::PercentAdd,
                result,
                prose: format!(
                    "${} with {}% tip → ${} + ({}/100 × ${}) = ${} {}",
                    fmt_num(base_v),
                    fmt_num(p),
                    fmt_num(base_v),
                    fmt_num(p),
                    fmt_num(base_v),
                    fmt_num(result),
                    FLAG_COMPUTED,
                ),
            });
        }
    }

    // ---- speed/rate from two numbers ("60 miles in 2 hours") ----
    if (has_word(&cleaned, "speed") || has_word(&cleaned, "rate"))
        && nums.len() >= 2
    {
        let a = nums[0].value;
        let b = nums[1].value;
        if b != 0.0 {
            let result = a / b;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Confident,
                numbers: vec![a, b],
                op: WordOp::Speed,
                result,
                prose: format!(
                    "{} ÷ {} = {} {}",
                    fmt_num(a),
                    fmt_num(b),
                    fmt_num(result),
                    FLAG_COMPUTED,
                ),
            });
        }
    }

    // ---- per-rate (multiplication or division based on question) ----
    //
    // The rate is the number nearest the "per" token in the cleaned
    // token stream. The other number is the total (for duration
    // questions: total ÷ rate) or the count (for total questions:
    // rate × count). Position-based selection keeps us correct
    // regardless of whether the rate appears before or after the
    // count: "Lisa earns $15 per hour" (rate first) vs "factory makes
    // 60 per minute. We need 180" (rate first too) vs "200 pages.
    // 25 pages per day" (rate second).
    if cleaned.contains(" per ") && nums.len() >= 2 {
        let cleaned_tokens: Vec<&str> = cleaned.split_whitespace().collect();
        let per_pos = cleaned_tokens
            .iter()
            .position(|&t| t == "per")
            .unwrap_or(0) as i64;

        let mut sorted = nums.clone();
        sorted.sort_by_key(|n| (n.pos as i64 - per_pos).abs());
        let rate = sorted[0].value;
        let other = sorted[1].value;

        let asks_duration = cleaned.contains("how long")
            || cleaned.contains("how many days")
            || cleaned.contains("how many hours")
            || cleaned.contains("how many minutes")
            || cleaned.contains("how many weeks")
            || cleaned.contains("to finish");

        if asks_duration && rate != 0.0 {
            let result = other / rate;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Confident,
                numbers: vec![other, rate],
                op: WordOp::Div,
                result,
                prose: format!(
                    "{} ÷ {} = {} {}",
                    fmt_num(other),
                    fmt_num(rate),
                    fmt_num(result),
                    FLAG_COMPUTED,
                ),
            });
        }
        // Default per-rate: multiply (commutative — order independent).
        let result = rate * other;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![rate, other],
            op: WordOp::Mul,
            result,
            prose: format!(
                "{} × {} = {} {}",
                fmt_num(rate),
                fmt_num(other),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- division cues ----
    let div_cues: &[&str] = &["split", "shared", "divided", "evenly", "average"];
    if has_any_word(&cleaned, div_cues) && nums.len() >= 2 {
        let a = nums[0].value;
        let b = nums[1].value;
        if b != 0.0 {
            let result = a / b;
            return Some(WordProblemOutcome::Computed {
                confidence: WordConfidence::Confident,
                numbers: vec![a, b],
                op: WordOp::Div,
                result,
                prose: format!(
                    "{} ÷ {} = {} {}",
                    fmt_num(a),
                    fmt_num(b),
                    fmt_num(result),
                    FLAG_COMPUTED,
                ),
            });
        }
    }

    // ---- "there are N X" + "a X has M Y" → multiplication ----
    if (has_phrase(&cleaned, "there are") || has_phrase(&cleaned, "there is"))
        && (has_word(&cleaned, "total") || has_phrase(&cleaned, "how many"))
        && nums.len() >= 2
    {
        let a = nums[0].value;
        let b = nums[1].value;
        let result = a * b;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![a, b],
            op: WordOp::Mul,
            result,
            prose: format!(
                "{} × {} = {} {}",
                fmt_num(a),
                fmt_num(b),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- multiplication cues ----
    let mul_cues: &[&str] = &["each", "every", "groups", "times"];
    if has_any_word(&cleaned, mul_cues) && nums.len() >= 2 {
        let a = nums[0].value;
        let b = nums[1].value;
        let result = a * b;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![a, b],
            op: WordOp::Mul,
            result,
            prose: format!(
                "{} × {} = {} {}",
                fmt_num(a),
                fmt_num(b),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- subtraction cues ----
    let sub_cues: &[&str] = &[
        "gave", "gives", "lost", "spent", "spends", "ate", "absent",
        "fewer", "remaining", "left", "took", "removed", "decreased",
        "loses", "missing",
    ];
    if has_any_word(&cleaned, sub_cues) && nums.len() >= 2 {
        let a = nums[0].value;
        let b = nums[1].value;
        let result = a - b;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![a, b],
            op: WordOp::Sub,
            result,
            prose: format!(
                "{} - {} = {} {}",
                fmt_num(a),
                fmt_num(b),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- addition cues ----
    let add_cues: &[&str] = &[
        "altogether", "combined", "buys", "bought", "received",
        "more", "gains", "gained", "joined", "increased",
    ];
    if (has_any_word(&cleaned, add_cues)
        || has_phrase(&cleaned, "in total")
        || has_phrase(&cleaned, "altogether"))
        && nums.len() >= 2
    {
        let a = nums[0].value;
        let b = nums[1].value;
        let result = a + b;
        return Some(WordProblemOutcome::Computed {
            confidence: WordConfidence::Confident,
            numbers: vec![a, b],
            op: WordOp::Add,
            result,
            prose: format!(
                "{} + {} = {} {}",
                fmt_num(a),
                fmt_num(b),
                fmt_num(result),
                FLAG_COMPUTED,
            ),
        });
    }

    // ---- prose-shaped but ambiguous → final hedge ----
    Some(make_hedge_no_guess(
        &nums,
        "Prose-shaped but I couldn't pin down the operation.",
    ))
}

fn make_hedge_no_guess(nums: &[NumToken], reason: &str) -> WordProblemOutcome {
    let nums_vec: Vec<f64> = nums.iter().map(|n| n.value).collect();
    let nums_str = nums_vec
        .iter()
        .map(|n| fmt_num(*n))
        .collect::<Vec<_>>()
        .join(", ");
    let prose = format!(
        "This looks like a word problem. {} Numbers I see: [{}]. I won't commit to an answer. {}",
        reason, nums_str, FLAG_HEDGED,
    );
    WordProblemOutcome::Hedge {
        numbers: nums_vec,
        cues: vec![],
        guess: None,
        prose,
    }
}

fn make_hedge_with_optional_guess(
    nums: &[NumToken],
    reason: &str,
    guess: Option<f64>,
) -> WordProblemOutcome {
    let nums_vec: Vec<f64> = nums.iter().map(|n| n.value).collect();
    let nums_str = nums_vec
        .iter()
        .map(|n| fmt_num(*n))
        .collect::<Vec<_>>()
        .join(", ");
    let prose = format!(
        "This looks like a word problem. {} Numbers I see: [{}]. Verify before relying on any guess. {}",
        reason, nums_str, FLAG_HEDGED,
    );
    WordProblemOutcome::Hedge {
        numbers: nums_vec,
        cues: vec![],
        guess,
        prose,
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn computed_value(query: &str) -> f64 {
        match detect_word_problem(query).unwrap_or_else(|| panic!("no outcome for {:?}", query)) {
            WordProblemOutcome::Computed { result, .. } => result,
            WordProblemOutcome::Hedge { prose, .. } => {
                panic!("expected Computed for {:?}, got Hedge: {}", query, prose)
            }
        }
    }

    fn computed_op(query: &str) -> WordOp {
        match detect_word_problem(query).unwrap() {
            WordProblemOutcome::Computed { op, .. } => op,
            WordProblemOutcome::Hedge { .. } => panic!("expected Computed for {:?}", query),
        }
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} to be within {} of {}",
            a,
            tol,
            b
        );
    }

    // -----------------------------------------------------------------
    // simple_addsub_tests — ~10 tests
    // -----------------------------------------------------------------
    mod simple_addsub_tests {
        use super::*;

        #[test]
        fn sarah_apples_subtract() {
            let q = "Sarah had 5 apples. She gave 2 to Mary. How many does Sarah have?";
            assert_eq!(computed_value(q), 3.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }

        #[test]
        fn bob_cookies_add() {
            let q = "Bob has 7 cookies. He buys 5 more. How many cookies does Bob have?";
            assert_eq!(computed_value(q), 12.0);
            assert_eq!(computed_op(q), WordOp::Add);
        }

        #[test]
        fn class_present_subtract() {
            let q = "A class has 30 students. 4 are absent. How many are present?";
            assert_eq!(computed_value(q), 26.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }

        #[test]
        fn pencils_add() {
            let q = "Tom had 8 pencils. He received 3 more. How many does he have?";
            assert_eq!(computed_value(q), 11.0);
            assert_eq!(computed_op(q), WordOp::Add);
        }

        #[test]
        fn ate_subtract() {
            let q = "There were 12 cookies on the plate. The kids ate 5. How many remain?";
            assert_eq!(computed_value(q), 7.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }

        #[test]
        fn spent_subtract() {
            let q = "Maya had $50. She spent 20. How much is left?";
            assert_eq!(computed_value(q), 30.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }

        #[test]
        fn altogether_add() {
            let q = "Anna has 7 marbles. Joe has 4 marbles. How many altogether?";
            assert_eq!(computed_value(q), 11.0);
            assert_eq!(computed_op(q), WordOp::Add);
        }

        #[test]
        fn lost_subtract() {
            let q = "The team had 18 footballs. They lost 6 in the rain. How many remain?";
            assert_eq!(computed_value(q), 12.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }

        #[test]
        fn word_form_numbers_add() {
            // "twelve" and "five" parsed via word_num.
            let q = "Bob has twelve cookies. He buys five more. How many does Bob have?";
            assert_eq!(computed_value(q), 17.0);
            assert_eq!(computed_op(q), WordOp::Add);
        }

        #[test]
        fn fewer_subtract() {
            let q = "There were 15 birds. 4 fewer flew away. How many are left?";
            assert_eq!(computed_value(q), 11.0);
            assert_eq!(computed_op(q), WordOp::Sub);
        }
    }

    // -----------------------------------------------------------------
    // multiply_divide_tests — ~8 tests
    // -----------------------------------------------------------------
    mod multiply_divide_tests {
        use super::*;

        #[test]
        fn boxes_of_eggs_multiply() {
            let q = "A box has 12 eggs. There are 5 boxes. How many eggs total?";
            assert_eq!(computed_value(q), 60.0);
            assert_eq!(computed_op(q), WordOp::Mul);
        }

        #[test]
        fn split_cookies_divide() {
            let q = "If 24 cookies are split among 6 children, how many does each get?";
            assert_eq!(computed_value(q), 4.0);
            assert_eq!(computed_op(q), WordOp::Div);
        }

        #[test]
        fn train_speed_divide() {
            let q = "A train travels 60 miles in 2 hours. What is the speed?";
            assert_eq!(computed_value(q), 30.0);
            assert_eq!(computed_op(q), WordOp::Speed);
        }

        #[test]
        fn shared_evenly_divide() {
            let q = "100 candies are shared evenly among 4 friends. How many each?";
            assert_eq!(computed_value(q), 25.0);
            assert_eq!(computed_op(q), WordOp::Div);
        }

        #[test]
        fn each_multiply() {
            let q = "Each table has 6 chairs. 8 tables. How many chairs total?";
            assert_eq!(computed_value(q), 48.0);
            assert_eq!(computed_op(q), WordOp::Mul);
        }

        #[test]
        fn divided_by_divide() {
            let q = "A pizza has 12 slices divided among 3 friends. How many slices each?";
            assert_eq!(computed_value(q), 4.0);
            assert_eq!(computed_op(q), WordOp::Div);
        }

        #[test]
        fn average_divide() {
            let q = "A car covered 240 miles. The trip lasted 4 hours. What is the average rate?";
            assert_eq!(computed_value(q), 60.0);
            assert_eq!(computed_op(q), WordOp::Speed);
        }

        #[test]
        fn there_are_pattern_multiply() {
            let q = "A bag contains 9 marbles. There are 7 bags. How many marbles total?";
            assert_eq!(computed_value(q), 63.0);
            assert_eq!(computed_op(q), WordOp::Mul);
        }
    }

    // -----------------------------------------------------------------
    // percent_word_tests — ~6 tests
    // -----------------------------------------------------------------
    mod percent_word_tests {
        use super::*;

        #[test]
        fn shirt_25_percent_off() {
            let q = "A shirt costs $20. It is 25% off. What is the sale price?";
            assert_close(computed_value(q), 15.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::PercentOff);
        }

        #[test]
        fn meal_18_percent_tip() {
            let q = "A meal cost $40. Tip is 18%. What is the total?";
            assert_close(computed_value(q), 47.20, 1e-9);
            assert_eq!(computed_op(q), WordOp::PercentAdd);
        }

        #[test]
        fn hat_10_percent_off() {
            let q = "A hat costs $30. It is 10% off. What is the sale price?";
            assert_close(computed_value(q), 27.0, 1e-9);
        }

        #[test]
        fn dinner_20_percent_tip() {
            let q = "Dinner cost $50. The tip is 20%. What is the total?";
            assert_close(computed_value(q), 60.0, 1e-9);
        }

        #[test]
        fn discount_word_form() {
            let q = "A book costs $25. It has a discount of 20%. What is the price after discount?";
            assert_close(computed_value(q), 20.0, 1e-9);
        }

        #[test]
        fn percent_off_with_decimals() {
            let q = "An item costs $19.99. It is 50% off. What is the sale price?";
            assert_close(computed_value(q), 9.995, 1e-6);
        }
    }

    // -----------------------------------------------------------------
    // twostep_tests — ~6 tests
    // -----------------------------------------------------------------
    mod twostep_tests {
        use super::*;

        #[test]
        fn lisa_earnings_per_hour() {
            let q = "Lisa earns $15 per hour. She works 8 hours. How much does she earn?";
            assert_close(computed_value(q), 120.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Mul);
        }

        #[test]
        fn book_pages_per_day() {
            let q = "A book has 200 pages. Tom reads 25 pages per day. How long to finish?";
            assert_close(computed_value(q), 8.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Div);
        }

        #[test]
        fn driver_per_hour_distance() {
            let q = "A driver travels 50 miles per hour. He drives 4 hours. How far?";
            // F.3 multiplies first two numbers when per-rate fires.
            // 50 × 4 = 200.
            assert_close(computed_value(q), 200.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Mul);
        }

        #[test]
        fn worker_per_day_total() {
            let q = "A worker earns $100 per day. He works 7 days. How much total?";
            assert_close(computed_value(q), 700.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Mul);
        }

        #[test]
        fn cookies_per_minute_duration() {
            let q = "A factory makes 60 cookies per minute. We need 180. How long to finish?";
            assert_close(computed_value(q), 3.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Div);
        }

        #[test]
        fn miles_per_hour_duration() {
            let q = "A trip is 240 miles. The car goes 60 miles per hour. How many hours?";
            assert_close(computed_value(q), 4.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::Div);
        }
    }

    // -----------------------------------------------------------------
    // less_than_sum_tests — ~4 tests
    // -----------------------------------------------------------------
    mod less_than_sum_tests {
        use super::*;

        #[test]
        fn five_less_than_sum_three_four() {
            let q = "What is 5 less than the sum of 3 and 4?";
            assert_close(computed_value(q), 2.0, 1e-9);
            assert_eq!(computed_op(q), WordOp::LessThanSumOf);
        }

        #[test]
        fn ten_less_than_sum_seven_eight() {
            let q = "What is 10 less than the sum of 7 and 8?";
            assert_close(computed_value(q), 5.0, 1e-9);
        }

        #[test]
        fn one_less_than_sum_two_three() {
            let q = "What is 1 less than the sum of 2 and 3?";
            assert_close(computed_value(q), 4.0, 1e-9);
        }

        #[test]
        fn zero_less_than_sum_six_seven() {
            let q = "What is 0 less than the sum of 6 and 7?";
            assert_close(computed_value(q), 13.0, 1e-9);
        }
    }

    // -----------------------------------------------------------------
    // hedge_tests — ~5 tests where Lu SHOULD hedge
    // -----------------------------------------------------------------
    mod hedge_tests {
        use super::*;

        fn assert_hedged(query: &str) {
            let outcome = detect_word_problem(query)
                .unwrap_or_else(|| panic!("expected an outcome for {:?}, got None", query));
            assert!(
                outcome.is_hedged(),
                "expected hedged for {:?}, got: {}",
                query,
                outcome.prose()
            );
        }

        #[test]
        fn bat_and_ball_hedges() {
            let q = "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?";
            assert_hedged(q);
        }

        #[test]
        fn algebra_twice_hedges() {
            let q = "If x is twice y and x + y is 30, what is x?";
            assert_hedged(q);
        }

        #[test]
        fn pizza_slices_one_number_hedges() {
            let q = "Three friends split a pizza. How many slices each?";
            assert_hedged(q);
        }

        #[test]
        fn relational_more_than_hedges() {
            let q = "Alice has 3 more apples than Bob. They have 11 apples total. How many does Alice have?";
            assert_hedged(q);
        }

        #[test]
        fn twice_as_many_hedges() {
            let q = "Sam has twice as many marbles as Joe and they have 30 total. How many does Sam have?";
            assert_hedged(q);
        }
    }

    // -----------------------------------------------------------------
    // decline_tests — ~3 tests where Lu should pass through
    // -----------------------------------------------------------------
    mod decline_tests {
        use super::*;

        #[test]
        fn plain_arithmetic_declines() {
            assert!(detect_word_problem("what is 7 plus 5").is_none());
        }

        #[test]
        fn expression_declines() {
            assert!(detect_word_problem("2 + 3 * 4").is_none());
        }

        #[test]
        fn unit_conversion_declines() {
            assert!(detect_word_problem("convert 100 km to miles").is_none());
        }

        #[test]
        fn concept_query_declines() {
            assert!(detect_word_problem("what is mitochondria").is_none());
        }
    }

    // -----------------------------------------------------------------
    // looks_like_word_problem direct unit tests
    // -----------------------------------------------------------------
    mod shape_tests {
        use super::*;

        #[test]
        fn prose_with_numbers_qualifies() {
            assert!(looks_like_word_problem(
                "Sarah had 5 apples. She gave 2 to Mary."
            ));
        }

        #[test]
        fn calculator_does_not_qualify() {
            assert!(!looks_like_word_problem("what is 7 plus 5"));
            assert!(!looks_like_word_problem("2 + 3 * 4"));
        }

        #[test]
        fn unit_conversion_does_not_qualify() {
            assert!(!looks_like_word_problem("convert 100 km to miles"));
        }

        #[test]
        fn trigger_phrase_qualifies() {
            assert!(looks_like_word_problem(
                "What is 5 less than the sum of 3 and 4?"
            ));
        }

        #[test]
        fn concept_query_does_not_qualify() {
            assert!(!looks_like_word_problem("what is mitochondria"));
        }
    }

    // -----------------------------------------------------------------
    // number extraction direct tests
    // -----------------------------------------------------------------
    mod extract_tests {
        use super::*;

        #[test]
        fn extracts_digits() {
            let nums = extract_numbers("Sarah had 5 apples and 12 pears.");
            let vals: Vec<f64> = nums.iter().map(|n| n.value).collect();
            assert_eq!(vals, vec![5.0, 12.0]);
        }

        #[test]
        fn extracts_dollar_and_percent() {
            let nums = extract_numbers("$20 with 25% off");
            assert_eq!(nums.len(), 2);
            assert!(nums[0].has_dollar);
            assert_eq!(nums[0].value, 20.0);
            assert!(nums[1].has_percent);
            assert_eq!(nums[1].value, 25.0);
        }

        #[test]
        fn extracts_decimals() {
            let nums = extract_numbers("The bat costs $1.10 and the ball $0.05.");
            let vals: Vec<f64> = nums.iter().map(|n| n.value).collect();
            assert_eq!(vals.len(), 2);
            assert_close((vals[0] - 1.10).abs(), 0.0, 1e-9);
            assert_close((vals[1] - 0.05).abs(), 0.0, 1e-9);
        }

        #[test]
        fn extracts_word_form() {
            let nums = extract_numbers("Bob has twelve cookies. He buys five more.");
            let vals: Vec<f64> = nums.iter().map(|n| n.value).collect();
            assert_eq!(vals, vec![12.0, 5.0]);
        }
    }
}
