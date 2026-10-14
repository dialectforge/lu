//! Port F.2 — simple linear solver for `<expr in x> = <expr in x>`.
//!
//! Strategy: parse left and right with the expression module, evaluate
//! `f(x) = left(x) - right(x)` at three sample points (0, 1, 2) and
//! check that the points lie on a straight line. If they do, slope =
//! `f(1) - f(0)` and intercept = `f(0)`, so `x = -intercept / slope`.
//!
//! Anything non-linear (`x^2`, `sin(x)`, etc.) fails the linearity
//! check and is rejected with "out of Round 1 scope". This is by
//! design — quadratics/algebra are deferred to F.3.

use super::expr::{eval, parse};

#[derive(Debug, Clone)]
pub struct LinearSolution {
    pub x: f64,
    pub equation: String,
}

/// Solve `<lhs> = <rhs>` for `x`. The input is everything AFTER the
/// optional "solve" keyword has already been stripped.
pub fn solve_linear(equation: &str) -> Result<LinearSolution, String> {
    let cleaned = equation
        .trim()
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();
    let mut parts = cleaned.splitn(2, '=');
    let lhs_str = parts.next().ok_or("missing left side")?.trim();
    let rhs_str = parts.next().ok_or("equation needs '='")?.trim();
    let lhs = parse(lhs_str)?;
    let rhs = parse(rhs_str)?;
    let f = |x: f64| -> Result<f64, String> {
        Ok(eval(&lhs, Some(x))? - eval(&rhs, Some(x))?)
    };
    let f0 = f(0.0)?;
    let f1 = f(1.0)?;
    let f2 = f(2.0)?;
    let slope = f1 - f0;
    let intercept = f0;
    // Linearity check: f(2) must agree with the line through f(0), f(1).
    let predicted = slope * 2.0 + intercept;
    let scale = f2.abs().max(1.0);
    if (f2 - predicted).abs() > 1e-6 * scale {
        return Err(
            "equation is not linear in x (out of Round 1 scope)".into(),
        );
    }
    if slope.abs() < 1e-12 {
        return Err(if intercept.abs() < 1e-12 {
            "degenerate equation: all x satisfy 0 = 0".into()
        } else {
            "degenerate equation: no x satisfies the constraint".into()
        });
    }
    Ok(LinearSolution {
        x: -intercept / slope,
        equation: cleaned.to_string(),
    })
}

#[cfg(test)]
mod solve_tests {
    use super::*;

    fn solve_x(s: &str) -> f64 {
        solve_linear(s).unwrap().x
    }

    // --- the spec's acceptance cases --------------------------------
    #[test]
    fn solve_x_plus_5_eq_10() {
        assert!((solve_x("x + 5 = 10") - 5.0).abs() < 1e-9);
    }
    #[test]
    fn solve_2x_eq_14() {
        assert!((solve_x("2x = 14") - 7.0).abs() < 1e-9);
    }
    #[test]
    fn solve_3x_minus_6_eq_0() {
        assert!((solve_x("3x - 6 = 0") - 2.0).abs() < 1e-9);
    }
    #[test]
    fn solve_x_over_2_eq_5() {
        assert!((solve_x("x/2 = 5") - 10.0).abs() < 1e-9);
    }

    // --- additional shapes -----------------------------------------
    #[test]
    fn solve_with_parens() {
        assert!((solve_x("2(x + 3) = 14") - 4.0).abs() < 1e-9);
    }
    #[test]
    fn solve_both_sides_have_x() {
        // 5x - 3 = 2x + 9 → 3x = 12 → x = 4.
        assert!((solve_x("5x - 3 = 2x + 9") - 4.0).abs() < 1e-9);
    }
    #[test]
    fn solve_negative_solution() {
        // x + 7 = 0 → x = -7
        assert!((solve_x("x + 7 = 0") + 7.0).abs() < 1e-9);
    }
    #[test]
    fn solve_decimal_solution() {
        assert!((solve_x("4x = 1") - 0.25).abs() < 1e-12);
    }

    // --- rejection -------------------------------------------------
    #[test]
    fn reject_quadratic() {
        // x^2 = 9 — non-linear, must be rejected cleanly.
        let r = solve_linear("x^2 = 9");
        assert!(r.is_err());
        assert!(r.unwrap_err().to_lowercase().contains("linear"));
    }
    #[test]
    fn reject_trig_of_x() {
        let r = solve_linear("sin(x) = 0");
        assert!(r.is_err());
    }
    #[test]
    fn reject_no_equals() {
        assert!(solve_linear("x + 5").is_err());
    }
    #[test]
    fn reject_degenerate_no_solution() {
        // 0 = 5 — no x satisfies, regardless of whether x appears.
        let r = solve_linear("0 = 5");
        assert!(r.is_err());
    }
    #[test]
    fn reject_degenerate_all_x() {
        let r = solve_linear("2x + 1 = 1 + 2x");
        assert!(r.is_err());
    }
}
