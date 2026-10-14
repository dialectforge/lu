//! Phase 5 (OP-PLANNER): native free-form code planner, ported from the verified
//! Python prototype (scripts/code_exec/freeform_planner.py). Pipeline:
//!   NL prompt -> intent (pattern | plumbing | none)
//!            -> retrieve a grounded `Code` KU pattern OR compose from grounded C2 ops
//!            -> verify in the native X1 sandbox (run->fix)
//!            -> PATTERN_PASS / COMPOSE_PASS / HONEST_ABORT.
//! Honesty floor: unknown shapes abort; nothing is faked.
//!
//! Brain policy: READ-ONLY over the canonical brain (op-grounding + pattern KUs).
//! `record_case` stages Inferred provenance to a sidecar; it never mutates the
//! canonical brain.

use crate::brain::Brain;
use crate::bond::BondType;
use crate::code_exec::{run, X1Input, X1Limits, X1Request, X1Result};
use regex::Regex;
use std::sync::OnceLock;

// ===== 5a: op-grounding over the canonical brain's C2 handles ===============

fn is_handleish(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("pyfn") || t.starts_with("jsfn") || t.starts_with("builtins.")
        || t.starts_with("js:") || t.starts_with("pyop")
}

/// True iff the brain grounds this op: the canonical `the <op> function` node
/// exists and carries at least one non-handle semantic edge
/// (Returns/Produces/UsedFor/HasParameter). Mirrors narration's resolution so
/// grounding and Q&A agree on what "known" means.
pub fn ground_op(brain: &Brain, op: &str) -> bool {
    let cand = format!("the {} function", op);
    let raw = crate::seed::Seed::from_label(&cand);
    let seed = if brain.network.node_intern().get(&raw).is_some() {
        raw
    } else {
        crate::seed::Seed::from_label(&crate::canonical::fold_seed_label(&cand))
    };
    if brain.network.node_intern().get(&seed).is_none() {
        return false;
    }
    for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(64) {
        if !matches!(
            bt,
            BondType::Returns | BondType::Produces | BondType::UsedFor | BondType::HasParameter
        ) {
            continue;
        }
        let obj = brain.network.node_label(&nbr).unwrap_or_default();
        if obj.is_empty() || is_handleish(&obj) {
            continue;
        }
        return true;
    }
    false
}

// ===== 5b: pattern library as grounded Code KUs (retrieve + adapt) ==========

/// A taught coding pattern, carried as a grounded `Code` knowledge-unit. The
/// `code` is a known-correct skeleton; retrieval uses the prototype's exact
/// regex shape so native and prototype select identically.
pub struct CodePattern {
    pub name: &'static str,
    pub rx: &'static str,
    pub code: &'static str,
}

pub const PATTERNS: &[CodePattern] = &[
    CodePattern { name: "fizzbuzz", rx: r"fizzbuzz",
        code: "for i in range(1,21):\n    if i%15==0: print('FizzBuzz')\n    elif i%3==0: print('Fizz')\n    elif i%5==0: print('Buzz')\n    else: print(i)" },
    CodePattern { name: "palindrome", rx: r"palindrome",
        code: "def is_palindrome(s):\n    s=''.join(c.lower() for c in s if c.isalnum())\n    return s==s[::-1]\nprint(is_palindrome('racecar'), is_palindrome('hello'))" },
    CodePattern { name: "fibonacci", rx: r"fibonacci|fibonnaci",
        code: "def fib(n):\n    a,b=0,1; out=[]\n    for _ in range(n): out.append(a); a,b=b,a+b\n    return out\nprint(fib(10))" },
    CodePattern { name: "factorial", rx: r"factorial",
        code: "def factorial(n):\n    r=1\n    for i in range(2,n+1): r*=i\n    return r\nprint(factorial(5))" },
    CodePattern { name: "reverse_string", rx: r"reverse (?:a |the )?(?:string|word|phrase|text)",
        code: "def reverse(s): return s[::-1]\nprint(reverse('hello'))" },
    CodePattern { name: "is_prime", rx: r"is.*prime|prime number",
        code: "def is_prime(n):\n    if n<2: return False\n    for i in range(2,int(n**0.5)+1):\n        if n%i==0: return False\n    return True\nprint(is_prime(7), is_prime(8))" },
    CodePattern { name: "count_vowels", rx: r"count.*vowel",
        code: "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels('hello world'))" },
    CodePattern { name: "sum_range", rx: r"sum.*1 to|sum of.*integers from 1",
        code: "print(sum(range(1,101)))" },
    CodePattern { name: "hello_world", rx: r"hello,? world",
        code: "print(\"Hello, World!\")" },
];

fn pattern_regexes() -> &'static Vec<(&'static str, Regex, &'static str)> {
    static RX: OnceLock<Vec<(&'static str, Regex, &'static str)>> = OnceLock::new();
    RX.get_or_init(|| {
        PATTERNS.iter().map(|p| (p.name, Regex::new(p.rx).unwrap(), p.code)).collect()
    })
}

/// Retrieve the taught pattern whose shape matches this lowercased prompt.
pub fn match_pattern(pl: &str) -> Option<(&'static str, &'static str)> {
    for (name, rx, code) in pattern_regexes() {
        if rx.is_match(pl) {
            return Some((name, code));
        }
    }
    None
}

/// The pattern's code framed as a `Code` knowledge-unit (provenance carrier).
pub fn pattern_code_ku(code: &str) -> crate::node::KnowledgeContent {
    crate::node::KnowledgeContent::Code(code.to_string())
}

// ===== 5c: intent parser (NL -> plumbing goal/spec) =========================

#[derive(Debug, Clone, PartialEq)]
pub struct PlumbingSpec {
    pub src: String,
    pub agg: Option<String>,
}

/// Port of the prototype's `parse_plumbing`: detect a data source and an
/// aggregation from the lowercased prompt. Returns None only when neither is
/// present (caller then aborts honestly).
pub fn parse_plumbing(pl: &str) -> Option<PlumbingSpec> {
    let src = if pl.contains("json") { Some("json") }
        else if pl.contains("csv") { Some("csv") }
        else if pl.contains("file") { Some("file") }
        else if pl.contains("list") { Some("list") }
        else { None };
    let agg = if pl.contains("average") || pl.contains("mean") { Some("avg") }
        else if pl.contains("longest line") { Some("longest") }
        else if pl.contains("word frequenc") || pl.contains("word count") || pl.contains("frequenc") { Some("wordfreq") }
        else if pl.contains("duplicate") { Some("dups") }
        else if pl.contains("second largest") { Some("second_max") }
        else if pl.contains("max") || pl.contains("largest") || pl.contains("maximum") { Some("max") }
        else if pl.contains("min") || pl.contains("smallest") { Some("min") }
        else if pl.contains("sort") { Some("sort") }
        else if pl.contains("how many") || pl.contains("count") { Some("count_lines") }
        else if pl.contains("keys") { Some("json_keys") }
        else if pl.contains("sum") { Some("sum") }
        else { None };
    if src.is_none() && agg.is_none() {
        return None;
    }
    Some(PlumbingSpec {
        src: src.unwrap_or("file").to_string(),
        agg: agg.map(|s| s.to_string()),
    })
}

// ===== 5d: general output/compute-a-value synthesis (REASON -> compose) ======
//
// The first GENERAL from-scratch synthesis route (NOT a hardcoded pattern).
// It reasons: (1) read the intent -- does the request ask to output/compute a
// value? (2) recall what the op does -- is `print` grounded in the brain?
// (3) compose the minimal correct program from the grounded op + the extracted
// value; (4) [caller] run it in X1 and return the real stdout. General over the
// VALUE (string literal, number, or pure-arithmetic expression); conservative
// over intent so it never hijacks recall; gated on grounding so it declines
// rather than fabricates. Built as a seam: `build_value_output_plan` threads the
// op name, so adding more grounded ops later is additive and the grounding gate
// stays directly testable.

/// A value the composed program should emit, resolved from the request.
#[derive(Debug, Clone, PartialEq)]
pub enum OutValue {
    /// A literal string to print verbatim (e.g. `hello world`).
    Str(String),
    /// A bare number or pure-arithmetic expression to evaluate then print
    /// (e.g. `42`, `2+2`). Validated to contain only digits/operators so the
    /// sandbox evaluation is total and side-effect-free; the real value comes
    /// from actually running it.
    Expr(String),
}

fn value_verb_rx() -> &'static Regex {
    static RX: OnceLock<Regex> = OnceLock::new();
    RX.get_or_init(|| Regex::new(r"\b(prints|print|outputs|output|displays|display)\b").unwrap())
}

/// A Python string literal for `s` (double-quoted, minimally escaped).
fn py_str_literal(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// First single- or double-quoted span in `s`, if any (returns the inner text).
fn extract_quoted(s: &str) -> Option<String> {
    for q in ['"', '\''] {
        if let Some(i) = s.find(q) {
            if let Some(j) = s[i + 1..].find(q) {
                return Some(s[i + 1..i + 1 + j].to_string());
            }
        }
    }
    None
}

/// True iff `s` is a non-empty pure-arithmetic expression: only digits,
/// operators (+ - * / %), parentheses, dot and spaces, with at least one digit.
/// Keeps the sandbox evaluation total and free of side effects.
fn is_pure_arith(s: &str) -> bool {
    let mut has_digit = false;
    for c in s.chars() {
        match c {
            '0'..='9' => has_digit = true,
            '+' | '-' | '*' | '/' | '%' | '(' | ')' | '.' | ' ' => {}
            _ => return false,
        }
    }
    has_digit
}

/// True iff `s` is a short, clean literal phrase (a real thing to print), not a
/// computation description. Collection/computation nouns and relational
/// connectors disqualify it, so `hello world` passes but `the largest number in
/// a file` does not.
fn is_clean_phrase(s: &str) -> bool {
    let wc = s.split_whitespace().count();
    if wc == 0 || wc > 6 {
        return false;
    }
    for c in s.chars() {
        if !(c.is_ascii_alphanumeric() || " ,'!?.:-".contains(c)) {
            return false;
        }
    }
    const BANNED: &[&str] = &[
        "file", "csv", "json", "list", "array", "dict", "tuple", "column",
        "row", "line", "string", "number", "integer", "float", "reverse",
        "sort", "sum", "average", "mean", "count", "largest", "smallest",
        "maximum", "minimum", "length", "frequenc", "prime", "fibonacci",
        "factorial", "palindrome", "vowel",
    ];
    for w in BANNED {
        if s.contains(w) {
            return false;
        }
    }
    let padded = format!(" {} ", s);
    for conn in [" of ", " in ", " from ", " into ", " to ", " for "] {
        if padded.contains(conn) {
            return false;
        }
    }
    true
}

/// Detect an "output / compute a value" intent and extract the value. General
/// over the value; conservative over intent. Returns None when there is no
/// explicit program-output framing or no clean value can be extracted (the
/// caller then falls through to honest abort -- never a fabrication).
pub fn parse_value_output(pl: &str) -> Option<OutValue> {
    // Intent gate: a whole-word output verb (print(s)/output(s)/display(s)) in a
    // program-writing frame OR at the imperative head. Recall questions ("who
    // wrote Hamlet", "what is the python language", "what is a printer") carry no
    // such verb-in-frame and never reach the value logic.
    let m = value_verb_rx().find(pl)?;
    let vpos = m.start();
    let program_frame = pl.contains("program") || pl.contains("script")
        || pl.contains("code") || pl.starts_with("write ") || pl.starts_with("make ");
    let imperative = vpos == 0;
    if !program_frame && !imperative {
        return None;
    }
    // The phrase after the output verb.
    let mut rest = pl[m.end()..].trim();
    // Strip leading fillers that name the value without changing it.
    loop {
        let before = rest;
        for lead in [
            "the result of", "the value of", "the number", "the string",
            "the text", "the phrase", "the message", "the word", "result of",
            "value of", "out", ":",
        ] {
            if let Some(s) = rest.strip_prefix(lead) {
                rest = s.trim();
            }
        }
        if rest == before {
            break;
        }
    }
    // Strip trailing language / destination qualifiers.
    loop {
        let before = rest;
        for tail in [
            " in python", " in python3", " in python 3", " in py", " using python",
            " with python", " to the console", " to console", " to the screen",
            " on the screen", " to stdout", " to terminal", " to the terminal",
        ] {
            if let Some(s) = rest.strip_suffix(tail) {
                rest = s.trim();
            }
        }
        if rest == before {
            break;
        }
    }
    let rest = rest
        .trim()
        .trim_matches(|c| c == '.' || c == '!' || c == '?')
        .trim();
    if rest.is_empty() {
        return None;
    }
    // 1) Explicitly quoted literal -> string.
    if let Some(q) = extract_quoted(rest) {
        return Some(OutValue::Str(q));
    }
    // 2) Bare number or pure-arithmetic expression -> compute then print.
    if is_pure_arith(rest) {
        return Some(OutValue::Expr(rest.replace(' ', "")));
    }
    // 3) Short clean literal phrase -> print verbatim. (Anything carrying a data
    //    source or aggregation was already claimed by the plumbing composer.)
    if is_clean_phrase(rest) {
        return Some(OutValue::Str(rest.to_string()));
    }
    None
}

/// Compose the minimal correct program for `val`, GATED on the output `op` being
/// grounded in the brain (the "recall what the op does" step). `op` is threaded
/// (not hardcoded at the gate) so the gate is load-bearing and directly
/// testable, and so more grounded ops can extend this seam later. If the op is
/// not grounded, returns an honest-decline Plan (`code == None`) instead of
/// faking output.
pub fn build_value_output_plan(brain: &Brain, val: &OutValue, op: &str) -> Plan {
    if !ground_op(brain, op) {
        return Plan {
            path: format!("synth:{}/ungrounded", op),
            kind: "none",
            code: None,
            grounded_ops: Vec::new(),
            input_name: None,
            message: Some(format!(
                "I can't ground the '{}' operation in what I know, so I won't fake a program.",
                op
            )),
        };
    }
    let (code, tag) = match val {
        OutValue::Str(s) => (format!("{}({})", op, py_str_literal(s)), "str"),
        OutValue::Expr(e) => (format!("{}({})", op, e), "expr"),
    };
    Plan {
        path: format!("synth:{}/{}", op, tag),
        kind: "compose",
        code: Some(code),
        grounded_ops: vec![op.to_string()],
        input_name: None,
        message: None,
    }
}
// ===== 5d.2: additional GENERAL from-scratch synthesis routes ================
//
// Three more reasoned routes, same discipline as the value-output route: read
// intent -> ground the observable op (`print`) -> compose a minimal correct
// program -> [caller runs it in X1 and returns real stdout]. Each is general
// over its parameters (range bounds, arithmetic operator, language) and gated
// on grounding so it declines rather than fabricates. NO FABRICATION: the
// caller always actually runs the composed program; nothing is emitted unrun.

/// Detect a "print a range of integers" request and extract inclusive (start,
/// end). Matches "print 1 to 10", "print the numbers 1 to 10", "print numbers
/// from 1 to 10", "count from 1 to 10", "list 5 through 1". Returns None when
/// there is no output verb with two bounds joined by a range connector, so a
/// single-value "print 42" is left to the value-output route untouched.
pub fn parse_range_print(pl: &str) -> Option<(i64, i64)> {
    static RX: OnceLock<Regex> = OnceLock::new();
    let rx = RX.get_or_init(|| {
        Regex::new(r"(?:prints?|outputs?|displays?|lists?|shows?|counts?)\b[^0-9-]*(-?\d+)\s*(?:to|through|thru|until|up to|\.\.=?)\s*(-?\d+)").unwrap()
    });
    let c = rx.captures(pl)?;
    let a: i64 = c.get(1)?.as_str().parse().ok()?;
    let b: i64 = c.get(2)?.as_str().parse().ok()?;
    Some((a, b))
}

/// Compose the inclusive-range loop for (a,b), ascending or descending.
fn range_program(a: i64, b: i64) -> String {
    if a <= b {
        format!("for i in range({}, {}):\n    print(i)", a, b + 1)
    } else {
        format!("for i in range({}, {}, -1):\n    print(i)", a, b - 1)
    }
}

/// Detect a "do <binary arithmetic> on two numbers" from-scratch request and
/// return (python operator, human word). Requires the phrase "two numbers" so
/// it never hijacks a concrete arithmetic query (those go to the math
/// primitive) nor a value-output print. General over the four basic operators.
pub fn parse_binary_arith(pl: &str) -> Option<(&'static str, &'static str)> {
    if !pl.contains("two numbers") {
        return None;
    }
    let op = if pl.contains("add") || pl.contains("sum") || pl.contains("addition") {
        ("+", "sum")
    } else if pl.contains("subtract") || pl.contains("difference") || pl.contains("minus") {
        ("-", "difference")
    } else if pl.contains("multiply") || pl.contains("product") || pl.contains("times") {
        ("*", "product")
    } else if pl.contains("divide") || pl.contains("quotient") {
        ("/", "quotient")
    } else {
        return None;
    };
    Some(op)
}

/// Compose a runnable program for a binary arithmetic op on two SAMPLE inputs.
/// The values are literal so the program is total and stdout is real; the
/// comment names them as samples so nothing is implied to be user data.
fn binary_arith_program(op: &str) -> String {
    let verb = match op { "+" => "add", "-" => "subtract", "*" => "multiply", _ => "divide" };
    format!("# {} two numbers (sample inputs)\na = 7\nb = 5\nprint(a {} b)", verb, op)
}

/// Canonical name for a KNOWN programming language token, else None.
fn norm_lang(tok: &str) -> Option<&'static str> {
    Some(match tok {
        "python" | "py" | "python3" => "python",
        "rust" => "rust",
        "javascript" | "js" => "javascript",
        "typescript" | "ts" => "typescript",
        "java" => "java",
        "go" | "golang" => "go",
        "c" => "c",
        "c++" | "cpp" => "c++",
        "c#" | "csharp" => "c#",
        "ruby" => "ruby",
        "php" => "php",
        "bash" | "shell" | "sh" => "bash",
        "sql" => "sql",
        "html" => "html",
        "css" => "css",
        "kotlin" => "kotlin",
        "swift" => "swift",
        "perl" => "perl",
        "scala" => "scala",
        _ => return None,
    })
}

/// Languages Lu can actually SYNTHESISE AND RUN itself. The X1 sandbox runs
/// Python only, so that is the honest set; other known languages are recognised
/// (so the request routes here) but declined in plain English, never faked.
pub fn is_synthesizable_lang(lang: &str) -> bool {
    matches!(lang, "python")
}

/// Recognise "give me an example of <language>" / "do you have an example of
/// <language>" / "show me a <language> example" as a request to DEMONSTRATE
/// code in that language. Returns the canonical language name only when the
/// example subject is a KNOWN programming language; None otherwise, so
/// "example of a mammal" stays a recall/Q&A query and is never hijacked.
pub fn detect_language_example(pl: &str) -> Option<String> {
    if !pl.contains("example") {
        return None;
    }
    let toks: Vec<&str> = pl
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '+' || c == '#'))
        .filter(|s| !s.is_empty())
        .collect();
    for (i, t) in toks.iter().enumerate() {
        if *t == "example" || *t == "examples" {
            if i + 2 < toks.len() && matches!(toks[i + 1], "of" | "in" | "for" | "using") {
                if let Some(l) = norm_lang(toks[i + 2]) {
                    return Some(l.to_string());
                }
            }
            if i >= 1 {
                if let Some(l) = norm_lang(toks[i - 1]) {
                    return Some(l.to_string());
                }
            }
        }
    }
    None
}

/// Conservative gate answering "does this look like a from-scratch code request
/// the planner might synthesize?" Used ONLY to decide whether the serving path
/// should consult `plan_only`; the actual divert is still gated on `plan_only`
/// yielding real code, so a loose `true` here is harmless (it falls through to
/// normal recall/refuse). Never fabricates and never hijacks a recall query.
pub fn looks_like_code_request(pl: &str) -> bool {
    // A concrete programming language named anywhere in the query.
    if pl
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '+' || c == '#'))
        .any(|t| norm_lang(t).is_some())
    {
        return true;
    }
    // A code-shaped noun.
    for m in [
        " function", " a class", " a script", " code",
        "regex", " a loop", "for loop", "while loop",
    ] {
        if pl.contains(m) {
            return true;
        }
    }
    // A recognizable from-scratch synthesis shape (range/loop or two-number
    // arithmetic). match_pattern is intentionally NOT consulted here so a bare
    // recall like "who wrote hello world" never diverts to the code faculty.
    parse_range_print(pl).is_some() || parse_binary_arith(pl).is_some()
}

/// A minimal, real example program for a synthesizable language (Python only
/// for now -- the sandbox runs Python), composed from the grounded `print` op.
fn language_example_program(lang: &str) -> Option<String> {
    match lang {
        "python" => Some(
            "# A small Python example\nname = \"Python\"\nprint(\"Hello from \" + name + \"!\")".to_string(),
        ),
        _ => None,
    }
}

/// Build a compose Plan gated on `print` grounding, or an honest-decline Plan.
/// Shared by the range / arith / example routes so grounding stays load-bearing
/// and identical everywhere.
fn build_grounded_print_plan(brain: &Brain, path: String, code: String) -> Plan {
    if !ground_op(brain, "print") {
        return Plan {
            path: format!("{}/ungrounded", path),
            kind: "none",
            code: None,
            grounded_ops: Vec::new(),
            input_name: None,
            message: Some(
                "I can't ground the 'print' operation in what I know, so I won't fake a program.".to_string(),
            ),
        };
    }
    Plan {
        path,
        kind: "compose",
        code: Some(code),
        grounded_ops: vec!["print".to_string()],
        input_name: None,
        message: None,
    }
}

// ===== 5e: run -> fix loop via the native X1 sandbox ========================

/// Pure fix-selector (unit-testable without the sandbox): given the current
/// program and stderr, return an adapted program + fix label, or None.
fn pick_fix(code: &str, err: &str) -> Option<(String, &'static str)> {
    if err.contains("invalid literal for int") || (err.contains("ValueError") && code.contains("int(")) {
        let nc = code.replace("int(", "float(");
        if nc != code {
            return Some((nc, "int->float"));
        }
    }
    None
}

pub struct RunFix {
    pub ok: bool,
    pub code: String,
    pub result: X1Result,
    pub attempts: usize,
    pub fixes: Vec<String>,
}

/// Run `code` in X1; on failure, apply the prototype's bounded observe->fix
/// (int()->float() on a ValueError; give up honestly on missing-file / other
/// errors). At most 3 attempts. `check` (if any) must also pass for `ok`.
pub fn run_fix(code: &str, inputs: &[X1Input], check: Option<&dyn Fn(&str) -> bool>) -> RunFix {
    let mut code = code.to_string();
    let mut fixes: Vec<String> = Vec::new();
    let mut last = X1Result {
        exit_code: -1, stdout: String::new(), stderr: String::new(),
        killed_by: None, timed_out: false, wall_ms: 0, seccomp: None, file: None,
    };
    let mut attempts = 0usize;
    for attempt in 1..=3 {
        attempts = attempt;
        let req = X1Request {
            code: code.clone(),
            inputs: inputs.to_vec(),
            want_file: None,
            limits: X1Limits { mem_mib: 256, pids: 64, cpu_cores: 1, wall_ms: 6000 },
        };
        let res = run(&req);
        let out = res.stdout.clone();
        let err = res.stderr.clone();
        let ok = res.exit_code == 0 && check.map_or(true, |c| c(&out));
        last = res;
        if ok {
            return RunFix { ok: true, code, result: last, attempts, fixes };
        }
        if err.contains("FileNotFoundError") {
            fixes.push("missing input file".into());
            break;
        }
        match pick_fix(&code, &err) {
            Some((nc, label)) => { fixes.push(label.into()); code = nc; }
            None => break,
        }
    }
    RunFix { ok: false, code, result: last, attempts, fixes }
}

// ===== solve: dispatch (pattern -> plumbing compose -> honest abort) ========

#[derive(Debug, Clone)]
pub struct SolveTrace {
    pub prompt: String,
    pub path: String,
    pub verdict: String,
    pub grounded_ops: Vec<String>,
    pub final_code: String,
    pub stdout: String,
    pub message: Option<String>,
}

/// A plan produced WITHOUT executing code (no X1). Guest-side planning needs
/// only the brain (for op-grounding); execution is delegated to the isolated
/// code-exec broker. `code == None` means honest abort.
#[derive(Debug, Clone)]
pub struct Plan {
    pub path: String,
    pub kind: &'static str, // "pattern" | "compose" | "none"
    pub code: Option<String>,
    pub grounded_ops: Vec<String>,
    pub input_name: Option<String>, // expected input filename for a compose plan
    pub message: Option<String>,
}

/// Plan a prompt without running anything: try a taught pattern, then a
/// data-plumbing composition, else abort honestly. Pure over the brain
/// (read-only grounding); no privilege, no sandbox.
// ===== 5b.2: operand binding for taught patterns ===========================
//
// A matched pattern carries a known-correct code SKELETON with an EXAMPLE
// operand baked in (reverse('hello'), count_vowels('hello world'), fib(10),
// factorial(5)). That skeleton correctly answers a *technique* question ("how
// do I reverse a string") but returns the WRONG result the moment the user
// names a CONCRETE operand ("reverse the string hello world" must print
// "dlrow olleh", not "olleh"; "10th fibonacci" must print 55, not the list).
// This layer reads the operand the user actually named and re-binds the
// skeleton to it. Same discipline as the value-output route -- general over the
// operand, conservative over intent, HONEST on failure:
//   * operand named and cleanly parsed  -> Bound (specialised, real program),
//   * NO operand named (technique/demo)  -> Demo (run the skeleton as a labelled
//     example: it still executes and prints a real, non-faked result),
//   * operand named but NOT parseable    -> Abort (never guess/fabricate).
// Only the parameterisable patterns opt in; fizzbuzz/palindrome/is_prime/
// sum_range/hello_world keep their demonstration behaviour unchanged.
#[derive(Debug, PartialEq)]
pub enum PatternBind {
    Bound { code: String, input_name: Option<String> },
    Demo,
    Abort,
}

/// Split into ASCII alnum tokens (drops punctuation/whitespace).
fn op_tokens(s: &str) -> Vec<&str> {
    s.split(|c: char| !c.is_ascii_alphanumeric()).filter(|t| !t.is_empty()).collect()
}

/// Map an ASCII ordinal or cardinal token to its value: digits ("10", "10th",
/// "3rd", "21st"), and spelled numbers/ordinals up to twenty plus the round
/// tens. None for anything else, so callers abort rather than guess.
fn word_to_number(tok: &str) -> Option<u64> {
    let t = tok.trim().trim_end_matches(|c: char| c == '.' || c == ',');
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let rest = &t[digits.len()..];
        if rest.is_empty() || matches!(rest.to_lowercase().as_str(), "st" | "nd" | "rd" | "th") {
            return digits.parse().ok();
        }
        return None;
    }
    Some(match t.to_lowercase().as_str() {
        "zero" => 0, "one" | "first" => 1, "two" | "second" => 2,
        "three" | "third" => 3, "four" | "fourth" => 4, "five" | "fifth" => 5,
        "six" | "sixth" => 6, "seven" | "seventh" => 7, "eight" | "eighth" => 8,
        "nine" | "ninth" => 9, "ten" | "tenth" => 10, "eleven" | "eleventh" => 11,
        "twelve" | "twelfth" => 12, "thirteen" | "thirteenth" => 13,
        "fourteen" | "fourteenth" => 14, "fifteen" | "fifteenth" => 15,
        "sixteen" | "sixteenth" => 16, "seventeen" | "seventeenth" => 17,
        "eighteen" | "eighteenth" => 18, "nineteen" | "nineteenth" => 19,
        "twenty" | "twentieth" => 20, "thirty" => 30, "forty" => 40,
        "fifty" => 50, "sixty" => 60, "seventy" => 70, "eighty" => 80,
        "ninety" => 90, "hundred" => 100,
        _ => return None,
    })
}

/// Trim connective fillers/qualifiers around an extracted operand phrase.
fn strip_operand_fillers(s: &str) -> String {
    let mut r = s.trim().trim_start_matches(|c: char| c == ':' || c == '-' || c == '=').trim();
    loop {
        let before = r;
        for lead in ["the ", "a ", "following ", "this ", "that ", "of "] {
            if let Some(x) = r.strip_prefix(lead) { r = x.trim(); }
        }
        if r == before { break; }
    }
    loop {
        let before = r;
        for tail in [" in python", " in python3", " in py", " using python",
                     " to the console", " to console", " to stdout", " to the screen"] {
            if let Some(x) = r.strip_suffix(tail) { r = x.trim(); }
        }
        if r == before { break; }
    }
    r.trim_matches(|c: char| c == '.' || c == '!' || c == '?').trim().to_string()
}

/// Strip a leading structural noun ("word"/"string"/"text"/"phrase"/"letter(s)")
/// that names the operand's kind rather than the operand itself.
fn strip_word_nouns(s: &str) -> String {
    let mut r = s.trim();
    loop {
        let before = r;
        for lead in ["word ", "words ", "string ", "text ", "phrase ",
                     "letter ", "letters ", "the ", "a "] {
            if let Some(x) = r.strip_prefix(lead) { r = x.trim(); }
        }
        if r == before { break; }
    }
    r.to_string()
}

/// Bind the `reverse_string` pattern to the operand named after the structural
/// noun ("reverse the string X" / "reverse the word X").
fn bind_reverse(prompt: &str, pl: &str) -> PatternBind {
    static RX: OnceLock<Regex> = OnceLock::new();
    let rx = RX.get_or_init(|| Regex::new(r"reverse (?:a |the )?(?:string|word|phrase|text)\b").unwrap());
    let m = match rx.find(pl) { Some(m) => m, None => return PatternBind::Demo };
    let tail_src = if prompt.is_ascii() { &prompt[m.end()..] } else { &pl[m.end()..] };
    let tail = tail_src.trim();
    if tail.is_empty() { return PatternBind::Demo; }
    let operand = match extract_quoted(tail) {
        Some(q) => q,
        None => strip_operand_fillers(tail),
    };
    if operand.is_empty() { return PatternBind::Demo; }
    if is_clean_phrase(&operand.to_lowercase()) {
        PatternBind::Bound {
            code: format!("def reverse(s): return s[::-1]\nprint(reverse({}))", py_str_literal(&operand)),
            input_name: None,
        }
    } else {
        PatternBind::Abort
    }
}

/// True iff the operand phrase refers to the caller-supplied input rather than a
/// literal (so we read it from the input file instead of hardcoding a value).
fn refers_to_input(low: &str) -> bool {
    matches!(low,
        "input" | "this input" | "the input" | "that input" | "the input text"
        | "this" | "it" | "the string" | "this string" | "the text" | "this text"
        | "the following" | "the following text" | "following text" | "stdin"
        | "the given text" | "given input" | "the given input" | "the file"
        | "this file" | "the provided input" | "the provided text")
        || low.starts_with("the input") || low.starts_with("this input")
}

/// Bind the `count_vowels` pattern to the operand named after "in"/"of", or to
/// the caller-supplied input when the operand refers to it.
fn bind_count_vowels(prompt: &str, pl: &str) -> PatternBind {
    let vpos = match pl.find("vowel") { Some(p) => p, None => return PatternBind::Demo };
    let mut conn_off: Option<usize> = None;
    for c in [" in ", " of "] {
        if let Some(k) = pl[vpos..].find(c) {
            let off = vpos + k + c.len();
            conn_off = Some(conn_off.map_or(off, |cur: usize| cur.min(off)));
        }
    }
    let off = match conn_off { Some(o) => o, None => return PatternBind::Demo };
    let tail_src = if prompt.is_ascii() { &prompt[off..] } else { &pl[off..] };
    let tail = tail_src.trim();
    if tail.is_empty() { return PatternBind::Demo; }
    let low = tail.to_lowercase();
    let low_full = strip_operand_fillers(&low);
    if refers_to_input(&low) || refers_to_input(&low_full) {
        return PatternBind::Bound {
            code: "s = open('input.txt').read()\ndef count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels(s))".to_string(),
            input_name: Some("input.txt".to_string()),
        };
    }
    let operand = match extract_quoted(tail) {
        Some(q) => q,
        None => strip_word_nouns(&strip_operand_fillers(tail)),
    };
    if operand.is_empty() { return PatternBind::Demo; }
    if is_clean_phrase(&operand.to_lowercase()) {
        PatternBind::Bound {
            code: format!("def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels({}))", py_str_literal(&operand)),
            input_name: None,
        }
    } else {
        PatternBind::Abort
    }
}

/// Extract the requested Fibonacci term index N from an ordinal/selection cue.
/// Returns None when no specific term is selected (bare "fibonacci sequence"),
/// leaving the demonstration path to show a runnable example.
fn extract_fib_n(pl: &str) -> Option<u64> {
    static ORD: OnceLock<Regex> = OnceLock::new();
    if let Some(c) = ORD.get_or_init(|| Regex::new(r"\b(\d+)\s*(?:st|nd|rd|th)\b").unwrap()).captures(pl) {
        if let Ok(n) = c.get(1).unwrap().as_str().parse() { return Some(n); }
    }
    static NUMOF: OnceLock<Regex> = OnceLock::new();
    if let Some(c) = NUMOF.get_or_init(|| Regex::new(r"fib\w*\s+(?:number|term|element|index)\s+(\d+)").unwrap()).captures(pl) {
        if let Ok(n) = c.get(1).unwrap().as_str().parse() { return Some(n); }
    }
    let toks = op_tokens(pl);
    for (i, t) in toks.iter().enumerate() {
        if t.starts_with("fib") && i >= 1 {
            if let Some(n) = word_to_number(toks[i - 1]) { return Some(n); }
        }
    }
    None
}

/// Program that prints the conventional 1-indexed Nth Fibonacci number
/// (F(1)=1, F(2)=1, F(3)=2, ... F(10)=55; OEIS A000045). Returns the SCALAR
/// term, not the sequence.
fn fib_program(n: u64) -> String {
    format!("def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a\nprint(fib({}))", n)
}

fn factorial_program(n: u64) -> String {
    format!("def factorial(n):\n    r = 1\n    for i in range(2, n + 1): r *= i\n    return r\nprint(factorial({}))", n)
}

/// Bind the `factorial` pattern to N from "factorial of N" / "N factorial" /
/// "N!"; Abort when an operand is named but is not a number; Demo when no
/// operand is named at all.
fn bind_factorial(pl: &str) -> PatternBind {
    static OF: OnceLock<Regex> = OnceLock::new();
    static POST: OnceLock<Regex> = OnceLock::new();
    let of = OF.get_or_init(|| Regex::new(r"factorial\s+of\s+(?:the\s+)?(?:number\s+)?([a-z0-9]+)").unwrap());
    let post = POST.get_or_init(|| Regex::new(r"([a-z0-9]+)\s*(?:!|factorial)").unwrap());
    if let Some(c) = of.captures(pl) {
        let tok = c.get(1).unwrap().as_str();
        return match word_to_number(tok) {
            Some(n) => PatternBind::Bound { code: factorial_program(n), input_name: None },
            None => PatternBind::Abort,
        };
    }
    if let Some(c) = post.captures(pl) {
        let tok = c.get(1).unwrap().as_str();
        if let Some(n) = word_to_number(tok) {
            return PatternBind::Bound { code: factorial_program(n), input_name: None };
        }
    }
    PatternBind::Demo
}

/// Re-bind a matched taught pattern to the operand the user actually named.
fn bind_pattern(name: &str, prompt: &str, pl: &str) -> PatternBind {
    match name {
        "reverse_string" => bind_reverse(prompt, pl),
        "count_vowels" => bind_count_vowels(prompt, pl),
        "fibonacci" => match extract_fib_n(pl) {
            Some(n) => PatternBind::Bound { code: fib_program(n), input_name: None },
            None => PatternBind::Demo,
        },
        "factorial" => bind_factorial(pl),
        _ => PatternBind::Demo,
    }
}

pub fn plan_only(brain: &Brain, prompt: &str) -> Plan {
    let pl = prompt.to_lowercase();
    if let Some((name, code)) = match_pattern(&pl) {
        return match bind_pattern(name, prompt, &pl) {
            PatternBind::Bound { code: bound, input_name } => Plan {
                path: format!("pattern:{}", name), kind: "pattern",
                code: Some(bound), grounded_ops: Vec::new(),
                input_name, message: None,
            },
            PatternBind::Demo => Plan {
                path: format!("pattern:{}", name), kind: "pattern",
                code: Some(code.to_string()), grounded_ops: Vec::new(),
                input_name: None, message: None,
            },
            PatternBind::Abort => Plan {
                path: format!("pattern:{}/operand_unparsed", name), kind: "none",
                code: None, grounded_ops: Vec::new(), input_name: None,
                message: Some("You named a specific input for that operation, but I couldn't parse it cleanly, so I won't guess.".to_string()),
            },
        };
    }
    if let Some(spec) = parse_plumbing(&pl) {
        let agg = spec.agg.clone().unwrap_or_default();
        if let Some((code, ops)) = crate::synthesis::compose_plumbing(&spec.src, &agg) {
            let grounded: Vec<String> =
                ops.iter().filter(|o| ground_op(brain, o)).map(|s| s.to_string()).collect();
            let input_name = match spec.src.as_str() {
                "csv" => "input.csv", "json" => "input.json", _ => "input.txt",
            };
            return Plan {
                path: format!("compose:{}/{}", spec.src, agg), kind: "compose",
                code: Some(code), grounded_ops: grounded,
                input_name: Some(input_name.to_string()), message: None,
            };
        }
    }
    // General range/loop synthesis: "print 1 to 10" -> prints 1..10.
    if let Some((a, b)) = parse_range_print(&pl) {
        return build_grounded_print_plan(brain, format!("synth:range/{}..{}", a, b), range_program(a, b));
    }
    // General binary-arithmetic synthesis: "add two numbers".
    if let Some((op, word)) = parse_binary_arith(&pl) {
        return build_grounded_print_plan(brain, format!("synth:arith/{}", word), binary_arith_program(op));
    }
    // PART B: "example of <language>" -> demonstrate a real runnable snippet.
    if let Some(lang) = detect_language_example(&pl) {
        if is_synthesizable_lang(&lang) {
            if let Some(code) = language_example_program(&lang) {
                return build_grounded_print_plan(brain, format!("synth:example/{}", lang), code);
            }
        }
        return Plan {
            path: format!("synth:example/{}/unsupported", lang),
            kind: "none",
            code: None,
            grounded_ops: Vec::new(),
            input_name: None,
            message: Some(format!(
                "I can only write and run Python programs myself, so I can't give you a runnable {} example.",
                lang
            )),
        };
    }
// General output/compute-a-value synthesis (reason -> ground `print` ->
    // compose -> [caller runs in X1]). Runs AFTER pattern + plumbing so those
    // keep precedence; only value-shaped requests with no data source/aggregation
    // reach it. Returns a compose plan when grounded, else an honest-decline
    // plan (code == None) -- never a fabrication.
    if let Some(val) = parse_value_output(&pl) {
        return build_value_output_plan(brain, &val, "print");
    }
    Plan {
        path: "none".into(), kind: "none", code: None, grounded_ops: Vec::new(),
        input_name: None,
        message: Some("I don't have a grounded pattern or a composition for this; I won't fake a solution.".into()),
    }
}

/// Top-level solve = plan_only + X1 run->fix. `check` is an optional output
/// validator used by callers (e.g. the parity harness) to confirm correctness,
/// never to fabricate a pass. The live path plans in the guest and executes in
/// the broker; this keeps the two behaviours identical.
pub fn solve(
    brain: &Brain,
    prompt: &str,
    inputs: &[X1Input],
    check: Option<&dyn Fn(&str) -> bool>,
) -> SolveTrace {
    let plan = plan_only(brain, prompt);
    if let Some(code) = plan.code.clone() {
        let rf = run_fix(&code, inputs, check);
        let verdict = match plan.kind {
            "pattern" => if rf.ok { "PATTERN_PASS" } else { "PATTERN_FAIL" },
            _ => if rf.ok { "COMPOSE_PASS" } else { "COMPOSE_FAIL" },
        };
        return SolveTrace {
            prompt: prompt.to_string(),
            path: plan.path,
            verdict: verdict.into(),
            grounded_ops: plan.grounded_ops,
            final_code: rf.code,
            stdout: rf.result.stdout,
            message: None,
        };
    }
    SolveTrace {
        prompt: prompt.to_string(),
        path: "none".into(),
        verdict: "HONEST_ABORT".into(),
        grounded_ops: Vec::new(),
        final_code: String::new(),
        stdout: String::new(),
        message: plan.message,
    }
}
// ===== 5f: case-writeback to the staged store (Inferred + provenance) =======

/// Record a solved case as staged, Inferred-tier provenance WITHOUT mutating the
/// canonical brain: append a JSONL line to a sidecar cases store. The staged
/// tier is Inferred by construction; promotion into the canonical brain is a
/// separate, gated, offline step and is intentionally NOT performed here.
pub fn record_case(store_path: &str, tr: &SolveTrace) -> std::io::Result<bool> {
    use std::io::Write;
    if !tr.verdict.ends_with("PASS") {
        return Ok(false);
    }
    let rec = serde_json::json!({
        "prompt": tr.prompt,
        "path": tr.path,
        "verdict": tr.verdict,
        "confidence": "Inferred",
        "provenance": "op-planner/native",
        "grounded_ops": tr.grounded_ops,
        "code": tr.final_code,
        "stdout": tr.stdout,
    });
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(store_path)?;
    writeln!(f, "{}", rec)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
#[test]
    fn value_output_detects_literal_number_and_expr() {
        assert_eq!(
            parse_value_output("write a program that prints hello world in python"),
            Some(OutValue::Str("hello world".into()))
        );
        assert_eq!(
            parse_value_output("write a program that prints 2+2"),
            Some(OutValue::Expr("2+2".into()))
        );
        assert_eq!(
            parse_value_output("write a program that prints 42"),
            Some(OutValue::Expr("42".into()))
        );
        assert_eq!(
            parse_value_output("print 'hi there'"),
            Some(OutValue::Str("hi there".into()))
        );
    }

    #[test]
    fn value_output_does_not_hijack_recall() {
        // No output verb -> recall stays recall.
        assert_eq!(parse_value_output("who wrote hamlet"), None);
        assert_eq!(parse_value_output("what is the python language"), None);
        // "print" as a substring of a topic, or a non-imperative mention: not a command.
        assert_eq!(parse_value_output("what is a printer"), None);
        assert_eq!(parse_value_output("explain the print function"), None);
        // A computation description is NOT a literal to print verbatim.
        assert_eq!(
            parse_value_output("write a program that prints the largest number in a file"),
            None
        );
    }

    #[test]
    fn value_output_helpers_shape() {
        assert_eq!(py_str_literal("hello world"), "\"hello world\"");
        assert_eq!(py_str_literal("a\"b"), "\"a\\\"b\"");
        assert!(is_pure_arith("2+2"));
        assert!(is_pure_arith("42"));
        assert!(is_pure_arith("6*7"));
        assert!(!is_pure_arith("hello"));
        assert!(!is_pure_arith(""));
        assert!(is_clean_phrase("hello world"));
        assert!(!is_clean_phrase("the largest number in a file"));
        assert!(!is_clean_phrase("reverse of a string"));
    }

    #[test]
    fn range_arith_example_synthesis_shapes() {
        assert_eq!(parse_range_print("print 1 to 10"), Some((1, 10)));
        assert_eq!(parse_range_print("print the numbers 1 to 10"), Some((1, 10)));
        assert_eq!(parse_range_print("print numbers from 1 to 10"), Some((1, 10)));
        assert_eq!(parse_range_print("print 42"), None);
        assert_eq!(parse_binary_arith("add two numbers"), Some(("+", "sum")));
        assert_eq!(parse_binary_arith("write a program that adds two numbers"), Some(("+", "sum")));
        assert_eq!(parse_binary_arith("multiply two numbers"), Some(("*", "product")));
        assert_eq!(parse_binary_arith("what is 2+2"), None);
        assert_eq!(detect_language_example("give me an example of python").as_deref(), Some("python"));
        assert_eq!(detect_language_example("do you have an example of python?").as_deref(), Some("python"));
        assert_eq!(detect_language_example("show me a python example").as_deref(), Some("python"));
        assert_eq!(detect_language_example("example of rust").as_deref(), Some("rust"));
        assert_eq!(detect_language_example("give me an example of a mammal"), None);
        assert_eq!(detect_language_example("who wrote hamlet"), None);
        assert!(is_synthesizable_lang("python"));
        assert!(!is_synthesizable_lang("rust"));
    }

    #[test]
    fn pattern_match_selects_and_rejects() {
        assert_eq!(match_pattern("write fizzbuzz").map(|(n, _)| n), Some("fizzbuzz"));
        assert_eq!(match_pattern("reverse a string").map(|(n, _)| n), Some("reverse_string"));
        assert_eq!(match_pattern("sum the integers from 1 to 100").map(|(n, _)| n), Some("sum_range"));
        assert_eq!(match_pattern("check if a number is prime").map(|(n, _)| n), Some("is_prime"));
        // research-hard shapes must NOT match a pattern
        assert_eq!(match_pattern("reverse a linked list"), None);
        assert_eq!(match_pattern("shortest path with dijkstra"), None);
    }

    #[test]
    fn plumbing_parse_matches_prototype() {
        assert_eq!(parse_plumbing("read a file and print the longest line"),
            Some(PlumbingSpec { src: "file".into(), agg: Some("longest".into()) }));
        assert_eq!(parse_plumbing("read a csv and print the average of the first column"),
            Some(PlumbingSpec { src: "csv".into(), agg: Some("avg".into()) }));
        assert_eq!(parse_plumbing("print the second largest number in a file"),
            Some(PlumbingSpec { src: "file".into(), agg: Some("second_max".into()) }));
        assert_eq!(parse_plumbing("parse json and print all keys"),
            Some(PlumbingSpec { src: "json".into(), agg: Some("json_keys".into()) }));
        // linked list: src=list, no agg -> compose will decline -> abort
        assert_eq!(parse_plumbing("reverse a linked list"),
            Some(PlumbingSpec { src: "list".into(), agg: None }));
        // no src, no agg -> None
        assert_eq!(parse_plumbing("shortest path with dijkstra"), None);
    }

    #[test]
    fn compose_emits_expected_shapes() {
        let (c, ops) = crate::synthesis::compose_plumbing("file", "longest").unwrap();
        assert!(c.contains("max(lines, key=len)"));
        assert!(ops.contains(&"max"));
        let (c2, _) = crate::synthesis::compose_plumbing("csv", "avg").unwrap();
        assert!(c2.contains("csv.reader"));
        // unknown agg (linked-list case) declines
        assert!(crate::synthesis::compose_plumbing("list", "").is_none());
    }

    #[test]
    fn pick_fix_int_to_float() {
        let (nc, label) = pick_fix("x=int(a)", "ValueError: invalid literal for int() with base 10").unwrap();
        assert_eq!(label, "int->float");
        assert!(nc.contains("float(a)"));
        assert!(pick_fix("print(1)", "KeyError: 'x'").is_none());
    }

    #[test]
    fn record_case_only_persists_pass() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("planner_cases_test_{}.jsonl", std::process::id()));
        let p = path.to_str().unwrap();
        let pass = SolveTrace { prompt: "x".into(), path: "pattern:fizzbuzz".into(),
            verdict: "PATTERN_PASS".into(), grounded_ops: vec![], final_code: "print(1)".into(),
            stdout: "1".into(), message: None };
        let abort = SolveTrace { verdict: "HONEST_ABORT".into(), ..pass.clone() };
        assert_eq!(record_case(p, &pass).unwrap(), true);
        assert_eq!(record_case(p, &abort).unwrap(), false);
        let body = std::fs::read_to_string(p).unwrap();
        assert_eq!(body.lines().count(), 1);
        assert!(body.contains("\"confidence\":\"Inferred\""));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn operand_binding_rebinds_named_operands() {
        assert_eq!(
            bind_pattern("reverse_string", "reverse the string hello world", "reverse the string hello world"),
            PatternBind::Bound { code: "def reverse(s): return s[::-1]\nprint(reverse(\"hello world\"))".into(), input_name: None }
        );
        assert!(matches!(bind_pattern("reverse_string", "reverse the word banana", "reverse the word banana"), PatternBind::Bound { .. }));
        assert_eq!(bind_pattern("reverse_string", "reverse a string", "reverse a string"), PatternBind::Demo);
        assert_eq!(
            bind_pattern("count_vowels", "count the vowels in education", "count the vowels in education"),
            PatternBind::Bound { code: "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels(\"education\"))".into(), input_name: None }
        );
        assert!(matches!(bind_pattern("count_vowels", "count vowels in the word mississippi", "count vowels in the word mississippi"), PatternBind::Bound { input_name: None, .. }));
        assert!(matches!(bind_pattern("count_vowels", "count vowels in this input", "count vowels in this input"), PatternBind::Bound { input_name: Some(_), .. }));
        assert_eq!(bind_pattern("count_vowels", "count the vowels", "count the vowels"), PatternBind::Demo);
        assert_eq!(extract_fib_n("compute the 10th fibonacci number"), Some(10));
        assert_eq!(extract_fib_n("5th fibonacci"), Some(5));
        assert_eq!(extract_fib_n("the tenth fibonacci"), Some(10));
        assert_eq!(extract_fib_n("fibonacci sequence"), None);
        assert!(fib_program(10).contains("print(fib(10))") && fib_program(10).contains("return a"));
        assert!(matches!(bind_pattern("factorial", "factorial of 6", "factorial of 6"), PatternBind::Bound { .. }));
        assert_eq!(bind_pattern("factorial", "factorial of banana", "factorial of banana"), PatternBind::Abort);
        assert_eq!(bind_pattern("factorial", "write a factorial function", "write a factorial function"), PatternBind::Demo);
        assert_eq!(word_to_number("10th"), Some(10));
        assert_eq!(word_to_number("tenth"), Some(10));
        assert_eq!(word_to_number("banana"), None);
    }
}
