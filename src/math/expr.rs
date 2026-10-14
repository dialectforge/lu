//! Port F.2 — recursive-descent expression parser.
//!
//! Hand-rolled lexer + parser + AST + evaluator. No dependencies
//! beyond `std` and `serde` (already in the workspace). All numeric
//! work is done in `f64`.
//!
//! Grammar (precedence low → high):
//!
//! ```text
//! expr      := add_sub
//! add_sub   := mul_div ((+|-) mul_div)*
//! mul_div   := unary  ((*|/|%) unary)*
//! unary     := ('-' | '+') unary | pow
//! pow       := atom ('^' unary)?         // right-assoc
//! atom      := NUM
//!            | IDENT '(' (expr (',' expr)*)? ')'
//!            | IDENT                       // var x or constant
//!            | '(' expr ')'
//! ```
//!
//! Implicit multiplication is supported between juxtaposed atoms
//! whose pairing is unambiguous: `2x`, `2(x+1)`, `(x+1)y`, `(x+1)(y)`
//! all behave like their explicit `*` siblings. `IDENT(...)` is
//! ALWAYS parsed as a function call, never as multiplication.
//!
//! `**` is accepted as an alias for `^` (Python convention), making
//! `2**10` and `2^10` identical. Both are right-associative.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    PercentTok,
    Caret,
    LParen,
    RParen,
    Comma,
    Eq,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Num(f64),
    /// The variable `x` (used by the linear solver).
    Var,
    /// A bare identifier — resolved at eval time against the constants
    /// table. Unknown idents become an evaluation error.
    Const(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Func(String, Vec<Expr>),
}

/// Tokenise an expression. Returns a clean error on invalid characters.
pub fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let mut buf = String::new();
            let mut saw_dot = false;
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    buf.push(d);
                    chars.next();
                } else if d == '.' && !saw_dot {
                    saw_dot = true;
                    buf.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            // Reject scientific notation explicitly (out-of-scope per
            // F.2 spec — same rule as F.1 `parse_number`).
            if let Some(&d) = chars.peek() {
                if d == 'e' || d == 'E' {
                    return Err(format!("scientific notation not supported: '{}e...'", buf));
                }
            }
            let v: f64 = buf.parse().map_err(|_| format!("bad number '{}'", buf))?;
            tokens.push(Token::Num(v));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut buf = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_alphanumeric() || d == '_' {
                    buf.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Ident(buf));
            continue;
        }
        match c {
            '+' => { chars.next(); tokens.push(Token::Plus); }
            '-' => { chars.next(); tokens.push(Token::Minus); }
            '*' => {
                chars.next();
                if chars.peek() == Some(&'*') {
                    chars.next();
                    tokens.push(Token::Caret);
                } else {
                    tokens.push(Token::Star);
                }
            }
            '/' => { chars.next(); tokens.push(Token::Slash); }
            '%' => { chars.next(); tokens.push(Token::PercentTok); }
            '^' => { chars.next(); tokens.push(Token::Caret); }
            '(' => { chars.next(); tokens.push(Token::LParen); }
            ')' => { chars.next(); tokens.push(Token::RParen); }
            ',' => { chars.next(); tokens.push(Token::Comma); }
            '=' => { chars.next(); tokens.push(Token::Eq); }
            other => return Err(format!("unexpected character '{}'", other)),
        }
    }
    insert_implicit_mul(&mut tokens);
    Ok(tokens)
}

/// Insert `*` between juxtaposed tokens that imply multiplication:
/// `Num·Ident`, `Num·LParen`, `RParen·Ident`, `RParen·LParen`,
/// `RParen·Num`. We deliberately do NOT insert between `Ident·Num`
/// or `Ident·Ident` to keep behaviour conservative — those can come
/// from word-form input (`twelve plus thirteen`) which is handled
/// upstream by the legacy detector, not here.
///
/// `Ident·LParen` is always a function call.
fn insert_implicit_mul(tokens: &mut Vec<Token>) {
    let mut i = 0;
    while i + 1 < tokens.len() {
        let needs = matches!(
            (&tokens[i], &tokens[i + 1]),
            (Token::Num(_), Token::Ident(_))
                | (Token::Num(_), Token::LParen)
                | (Token::RParen, Token::Ident(_))
                | (Token::RParen, Token::LParen)
                | (Token::RParen, Token::Num(_))
        );
        if needs {
            tokens.insert(i + 1, Token::Star);
            i += 2;
        } else {
            i += 1;
        }
    }
}

pub struct Parser {
    pub tokens: Vec<Token>,
    pub pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn advance(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, expected: &Token) -> Result<(), String> {
        if self.peek() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!(
                "expected {:?}, got {:?}",
                expected,
                self.peek()
            ))
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_mul_div()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    let r = self.parse_mul_div()?;
                    left = Expr::Add(Box::new(left), Box::new(r));
                }
                Some(Token::Minus) => {
                    self.advance();
                    let r = self.parse_mul_div()?;
                    left = Expr::Sub(Box::new(left), Box::new(r));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    let r = self.parse_unary()?;
                    left = Expr::Mul(Box::new(left), Box::new(r));
                }
                Some(Token::Slash) => {
                    self.advance();
                    let r = self.parse_unary()?;
                    left = Expr::Div(Box::new(left), Box::new(r));
                }
                Some(Token::PercentTok) => {
                    self.advance();
                    let r = self.parse_unary()?;
                    left = Expr::Mod(Box::new(left), Box::new(r));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Token::Minus)) {
            self.advance();
            let inner = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        if matches!(self.peek(), Some(Token::Plus)) {
            self.advance();
            return self.parse_unary();
        }
        self.parse_pow()
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let base = self.parse_atom()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.advance();
            // Right-assoc: re-enter through unary so subsequent `^`
            // binds tighter on the right.
            let exp = self.parse_unary()?;
            return Ok(Expr::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Some(Token::Num(n)) => Ok(Expr::Num(n)),
            Some(Token::Ident(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Token::Comma)) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.eat(&Token::RParen)?;
                    Ok(Expr::Func(name, args))
                } else if name == "x" || name == "X" {
                    Ok(Expr::Var)
                } else {
                    Ok(Expr::Const(name))
                }
            }
            Some(Token::LParen) => {
                let e = self.parse_expr()?;
                self.eat(&Token::RParen)?;
                Ok(e)
            }
            other => Err(format!("expected atom, got {:?}", other)),
        }
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Parse one expression and require all tokens to be consumed.
    pub fn finish(mut self) -> Result<Expr, String> {
        let e = self.parse_expr()?;
        if !self.at_end() {
            return Err(format!(
                "unexpected token after expression: {:?}",
                self.tokens[self.pos]
            ));
        }
        Ok(e)
    }
}

/// Parse a complete expression. Errors on leftover tokens.
pub fn parse(s: &str) -> Result<Expr, String> {
    let toks = tokenize(s)?;
    Parser::new(toks).finish()
}

/// Evaluate an AST. `x` provides the value of the `Var` node when
/// the expression came from the linear solver — `None` for normal
/// expressions (and `Var` will then return an error).
pub fn eval(expr: &Expr, x: Option<f64>) -> Result<f64, String> {
    let v = match expr {
        Expr::Num(n) => *n,
        Expr::Var => x.ok_or_else(|| "variable x is not bound in this expression".to_string())?,
        Expr::Const(name) => match name.to_lowercase().as_str() {
            "pi" => std::f64::consts::PI,
            "e" => std::f64::consts::E,
            "tau" => std::f64::consts::TAU,
            "inf" | "infinity" => f64::INFINITY,
            other => return Err(format!("unknown identifier '{}'", other)),
        },
        Expr::Neg(a) => -eval(a, x)?,
        Expr::Add(a, b) => eval(a, x)? + eval(b, x)?,
        Expr::Sub(a, b) => eval(a, x)? - eval(b, x)?,
        Expr::Mul(a, b) => eval(a, x)? * eval(b, x)?,
        Expr::Div(a, b) => {
            let bv = eval(b, x)?;
            if bv == 0.0 {
                return Err("division by zero".into());
            }
            eval(a, x)? / bv
        }
        Expr::Mod(a, b) => {
            let bv = eval(b, x)?;
            if bv == 0.0 {
                return Err("modulo by zero".into());
            }
            eval(a, x)? % bv
        }
        Expr::Pow(a, b) => eval(a, x)?.powf(eval(b, x)?),
        Expr::Func(name, args) => {
            let vs: Result<Vec<f64>, String> =
                args.iter().map(|a| eval(a, x)).collect();
            call_function(name, &vs?)?
        }
    };
    if v.is_nan() {
        return Err("non-finite result (NaN)".into());
    }
    Ok(v)
}

/// Convenience wrapper: parse and evaluate in one step (no `x`).
pub fn parse_and_eval(s: &str) -> Result<f64, String> {
    eval(&parse(s)?, None)
}

fn call_function(name: &str, args: &[f64]) -> Result<f64, String> {
    let n = name.to_lowercase();
    match n.as_str() {
        "sqrt" => arg1(name, args).and_then(|v| {
            if v < 0.0 {
                Err(format!("sqrt of negative: {}", v))
            } else {
                Ok(v.sqrt())
            }
        }),
        "cbrt" => arg1(name, args).map(f64::cbrt),
        "abs" => arg1(name, args).map(f64::abs),
        "sign" | "signum" => arg1(name, args).map(|v| if v == 0.0 { 0.0 } else { v.signum() }),
        "sin" => arg1(name, args).map(f64::sin),
        "cos" => arg1(name, args).map(f64::cos),
        "tan" => arg1(name, args).map(f64::tan),
        "asin" => arg1(name, args).and_then(|v| {
            if !(-1.0..=1.0).contains(&v) {
                Err(format!("asin domain error: {}", v))
            } else {
                Ok(v.asin())
            }
        }),
        "acos" => arg1(name, args).and_then(|v| {
            if !(-1.0..=1.0).contains(&v) {
                Err(format!("acos domain error: {}", v))
            } else {
                Ok(v.acos())
            }
        }),
        "atan" => arg1(name, args).map(f64::atan),
        "atan2" => arg2(name, args).map(|(y, x)| y.atan2(x)),
        "log" | "log10" => arg1(name, args).and_then(|v| {
            if v <= 0.0 {
                Err(format!("log of non-positive: {}", v))
            } else {
                Ok(v.log10())
            }
        }),
        "ln" => arg1(name, args).and_then(|v| {
            if v <= 0.0 {
                Err(format!("ln of non-positive: {}", v))
            } else {
                Ok(v.ln())
            }
        }),
        "log2" => arg1(name, args).and_then(|v| {
            if v <= 0.0 {
                Err(format!("log2 of non-positive: {}", v))
            } else {
                Ok(v.log2())
            }
        }),
        "exp" => arg1(name, args).map(f64::exp),
        "floor" => arg1(name, args).map(f64::floor),
        "ceil" => arg1(name, args).map(f64::ceil),
        "round" => arg1(name, args).map(f64::round),
        "trunc" => arg1(name, args).map(f64::trunc),
        "min" => {
            if args.is_empty() {
                return Err("min needs at least 1 argument".into());
            }
            Ok(args.iter().copied().fold(f64::INFINITY, f64::min))
        }
        "max" => {
            if args.is_empty() {
                return Err("max needs at least 1 argument".into());
            }
            Ok(args.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        "gcd" => arg2(name, args).and_then(|(a, b)| {
            if a.fract() != 0.0 || b.fract() != 0.0 {
                return Err("gcd requires integer arguments".into());
            }
            Ok(gcd_i64(a as i64, b as i64) as f64)
        }),
        "lcm" => arg2(name, args).and_then(|(a, b)| {
            if a.fract() != 0.0 || b.fract() != 0.0 {
                return Err("lcm requires integer arguments".into());
            }
            Ok(lcm_i64(a as i64, b as i64) as f64)
        }),
        _ => Err(format!("unknown function '{}'", name)),
    }
}

fn arg1(name: &str, args: &[f64]) -> Result<f64, String> {
    if args.len() != 1 {
        return Err(format!("{} expects 1 argument, got {}", name, args.len()));
    }
    Ok(args[0])
}
fn arg2(name: &str, args: &[f64]) -> Result<(f64, f64), String> {
    if args.len() != 2 {
        return Err(format!("{} expects 2 arguments, got {}", name, args.len()));
    }
    Ok((args[0], args[1]))
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a.abs() / gcd_i64(a, b)) * b.abs()
    }
}

/// True if the expression has at least one operation (function call,
/// arithmetic, etc.) — used by the dispatch layer to filter out
/// queries like "5" that look like math literals but are really
/// trivial recall ("what is 5").
pub fn has_operation(e: &Expr) -> bool {
    !matches!(e, Expr::Num(_) | Expr::Const(_) | Expr::Var)
}

// =====================================================================
// Tests — lexer, parser, evaluator
// =====================================================================

#[cfg(test)]
mod expr_tests {
    use super::*;

    fn eval_str(s: &str) -> f64 {
        parse_and_eval(s).expect("eval failed")
    }

    #[test]
    fn precedence_addition_then_mult() {
        assert_eq!(eval_str("2 + 3 * 4"), 14.0);
    }

    #[test]
    fn precedence_paren_over_mult() {
        assert_eq!(eval_str("(2 + 3) * 4"), 20.0);
    }

    #[test]
    fn precedence_pow_above_mult() {
        assert_eq!(eval_str("2 ^ 3 + 1"), 9.0);
    }

    #[test]
    fn precedence_nested_parens() {
        assert_eq!(eval_str("(7 + 5) * (3 - 1)"), 24.0);
    }

    #[test]
    fn left_assoc_division() {
        assert_eq!(eval_str("10 / 2 / 5"), 1.0);
    }

    #[test]
    fn right_assoc_pow() {
        assert_eq!(eval_str("2 ^ 3 ^ 2"), 512.0);
    }

    #[test]
    fn star_star_alias() {
        assert_eq!(eval_str("2 ** 10"), 1024.0);
    }

    #[test]
    fn unary_minus() {
        assert_eq!(eval_str("-5 + 7"), 2.0);
    }

    #[test]
    fn unary_minus_pow() {
        // -2^2 -> -(2^2) = -4 (BODMAS calculator convention).
        assert_eq!(eval_str("-2 ^ 2"), -4.0);
    }

    #[test]
    fn unary_chain_minuses() {
        assert_eq!(eval_str("--5"), 5.0);
    }

    #[test]
    fn modulo_basic() {
        assert_eq!(eval_str("17 % 5"), 2.0);
    }

    #[test]
    fn decimal_addition() {
        assert!((eval_str("1.5 + 2.25") - 3.75).abs() < 1e-12);
    }

    #[test]
    fn fn_sqrt() {
        assert_eq!(eval_str("sqrt(16)"), 4.0);
        assert_eq!(eval_str("sqrt(144)"), 12.0);
    }

    #[test]
    fn fn_cbrt() {
        assert!((eval_str("cbrt(27)") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn fn_abs_sign() {
        assert_eq!(eval_str("abs(-7)"), 7.0);
        assert_eq!(eval_str("sign(-3)"), -1.0);
        assert_eq!(eval_str("sign(0)"), 0.0);
        assert_eq!(eval_str("sign(2.5)"), 1.0);
    }

    #[test]
    fn fn_trig_zero() {
        assert!((eval_str("sin(0)")).abs() < 1e-12);
        assert!((eval_str("cos(0)") - 1.0).abs() < 1e-12);
        assert!((eval_str("tan(0)")).abs() < 1e-12);
    }

    #[test]
    fn fn_atan2() {
        assert!((eval_str("atan2(0, 1)")).abs() < 1e-12);
    }

    #[test]
    fn fn_logs() {
        assert!((eval_str("log(100)") - 2.0).abs() < 1e-12);
        assert!((eval_str("log(1000)") - 3.0).abs() < 1e-12);
        assert!((eval_str("ln(e)") - 1.0).abs() < 1e-12);
        assert!((eval_str("log2(8)") - 3.0).abs() < 1e-12);
        assert!((eval_str("exp(0)") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fn_log_of_nonpositive_errors() {
        assert!(parse_and_eval("log(0)").is_err());
        assert!(parse_and_eval("ln(-1)").is_err());
    }

    #[test]
    fn fn_rounding() {
        assert_eq!(eval_str("floor(3.7)"), 3.0);
        assert_eq!(eval_str("ceil(3.2)"), 4.0);
        assert_eq!(eval_str("round(2.5)"), 3.0);
        assert_eq!(eval_str("trunc(2.9)"), 2.0);
        assert_eq!(eval_str("trunc(-2.9)"), -2.0);
    }

    #[test]
    fn fn_min_max() {
        assert_eq!(eval_str("min(3, 7)"), 3.0);
        assert_eq!(eval_str("max(3, 7)"), 7.0);
        assert_eq!(eval_str("max(3, 7, 12)"), 12.0);
    }

    #[test]
    fn fn_gcd_lcm() {
        assert_eq!(eval_str("gcd(12, 18)"), 6.0);
        assert_eq!(eval_str("lcm(4, 6)"), 12.0);
        assert_eq!(eval_str("gcd(0, 5)"), 5.0);
    }

    #[test]
    fn fn_sqrt_negative_errors() {
        assert!(parse_and_eval("sqrt(-1)").is_err());
    }

    #[test]
    fn fn_unknown_errors() {
        assert!(parse_and_eval("bogus(3)").is_err());
    }

    #[test]
    fn const_pi() {
        assert!((eval_str("2 * pi") - 6.283185307179586).abs() < 1e-12);
    }

    #[test]
    fn const_e() {
        assert!((eval_str("e^1") - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn const_tau() {
        assert!((eval_str("tau") - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn const_unknown_errors() {
        assert!(parse_and_eval("foobar").is_err());
    }

    #[test]
    fn implicit_num_paren() {
        assert_eq!(eval_str("3(2 + 4)"), 18.0);
    }

    #[test]
    fn implicit_paren_paren() {
        assert_eq!(eval_str("(2 + 3)(4 + 1)"), 25.0);
    }

    #[test]
    fn division_by_zero_errors() {
        assert!(parse_and_eval("5 / 0").is_err());
        assert!(parse_and_eval("5 / (3 - 3)").is_err());
    }

    #[test]
    fn syntax_error_unmatched_paren() {
        assert!(parse("(2 + 3").is_err());
    }

    #[test]
    fn syntax_error_trailing_op() {
        assert!(parse("2 + 3 +").is_err());
    }

    #[test]
    fn syntax_error_bad_char() {
        assert!(parse("2 & 3").is_err());
    }

    #[test]
    fn syntax_error_scientific_rejected() {
        assert!(parse("1e3").is_err());
    }

    #[test]
    fn var_unbound_errors() {
        let e = parse("x + 1").unwrap();
        assert!(eval(&e, None).is_err());
        assert_eq!(eval(&e, Some(4.0)).unwrap(), 5.0);
    }

    #[test]
    fn has_op_distinguishes_literal_from_expr() {
        assert!(!has_operation(&parse("5").unwrap()));
        assert!(!has_operation(&parse("pi").unwrap()));
        assert!(has_operation(&parse("5 + 1").unwrap()));
        assert!(has_operation(&parse("sqrt(4)").unwrap()));
    }
}
