//! Port F.2 — descriptive statistics over comma- or space-separated lists.
//!
//! All operators take a slice of `f64` and return a single value.
//! Population (n) variance is used by default — sample (n-1) variance
//! and skew/kurtosis are deferred to F.3.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatOp {
    Mean,
    Median,
    Mode,
    Sum,
    Max,
    Min,
    Count,
    Variance,
    Stddev,
}

impl StatOp {
    pub fn name(&self) -> &'static str {
        match self {
            StatOp::Mean => "mean",
            StatOp::Median => "median",
            StatOp::Mode => "mode",
            StatOp::Sum => "sum",
            StatOp::Max => "max",
            StatOp::Min => "min",
            StatOp::Count => "count",
            StatOp::Variance => "variance",
            StatOp::Stddev => "stddev",
        }
    }
}

pub fn compute(op: StatOp, values: &[f64]) -> Result<f64, String> {
    match op {
        StatOp::Mean => mean(values),
        StatOp::Median => median(values),
        StatOp::Mode => mode(values),
        StatOp::Sum => Ok(sum(values)),
        StatOp::Max => max(values),
        StatOp::Min => min(values),
        StatOp::Count => Ok(values.len() as f64),
        StatOp::Variance => variance(values),
        StatOp::Stddev => stddev(values),
    }
}

pub fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

pub fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Ok(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Mode: returns the most-frequent value. Ties broken by first occurrence.
/// Equality compared with a small tolerance so 1.0 and 1.0000000001 are
/// treated as the same mode bucket.
pub fn mode(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for &v in values {
        let mut found = false;
        for c in counts.iter_mut() {
            if (c.0 - v).abs() < 1e-12 {
                c.1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            counts.push((v, 1));
        }
    }
    counts
        .iter()
        .max_by_key(|c| c.1)
        .map(|c| c.0)
        .ok_or_else(|| "empty list".into())
}

pub fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

pub fn max(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

pub fn min(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    Ok(values.iter().copied().fold(f64::INFINITY, f64::min))
}

pub fn variance(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("empty list".into());
    }
    let m = mean(values)?;
    Ok(values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64)
}

pub fn stddev(values: &[f64]) -> Result<f64, String> {
    Ok(variance(values)?.sqrt())
}

// =====================================================================
// Detection — "mean of 1, 2, 3" / "sum 1 2 3" / "stddev of 2,4,4,4,5"
// =====================================================================

/// If `query` looks like a stats request, return the operator and the
/// parsed numeric list. Otherwise return None.
pub fn detect(query: &str) -> Option<(StatOp, Vec<f64>)> {
    let q = query.trim().to_lowercase();
    let q = q
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();
    let table: &[(&str, StatOp)] = &[
        ("mean of ", StatOp::Mean),
        ("average of ", StatOp::Mean),
        ("average ", StatOp::Mean),
        ("mean ", StatOp::Mean),
        ("median of ", StatOp::Median),
        ("median ", StatOp::Median),
        ("mode of ", StatOp::Mode),
        ("mode ", StatOp::Mode),
        ("sum of ", StatOp::Sum),
        ("sum ", StatOp::Sum),
        ("max of ", StatOp::Max),
        ("maximum of ", StatOp::Max),
        ("maximum ", StatOp::Max),
        ("min of ", StatOp::Min),
        ("minimum of ", StatOp::Min),
        ("minimum ", StatOp::Min),
        ("count of ", StatOp::Count),
        ("count ", StatOp::Count),
        ("variance of ", StatOp::Variance),
        ("variance ", StatOp::Variance),
        ("standard deviation of ", StatOp::Stddev),
        ("standard deviation ", StatOp::Stddev),
        ("stddev of ", StatOp::Stddev),
        ("stddev ", StatOp::Stddev),
        ("stdev of ", StatOp::Stddev),
        ("stdev ", StatOp::Stddev),
    ];
    for (prefix, op) in table {
        if let Some(rest) = q.strip_prefix(*prefix) {
            if let Some(values) = parse_number_list(rest) {
                return Some((*op, values));
            }
        }
    }
    None
}

fn parse_number_list(s: &str) -> Option<Vec<f64>> {
    let cleaned = s.replace(',', " ");
    let mut out = Vec::new();
    for tok in cleaned.split_whitespace() {
        let v: f64 = tok.parse().ok()?;
        if v.is_nan() || v.is_infinite() {
            return None;
        }
        out.push(v);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod stat_tests {
    use super::*;

    #[test]
    fn mean_5() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(), 3.0);
    }
    #[test]
    fn median_odd() {
        assert_eq!(median(&[1.0, 5.0, 2.0, 8.0, 3.0]).unwrap(), 3.0);
    }
    #[test]
    fn median_even() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 2.5);
    }
    #[test]
    fn mode_basic() {
        assert_eq!(mode(&[1.0, 2.0, 2.0, 3.0]).unwrap(), 2.0);
    }
    #[test]
    fn stddev_sample_pop() {
        // Population stddev of 1..5 = sqrt(2) ~ 1.414
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((stddev(&v).unwrap() - 1.4142135).abs() < 1e-5);
    }
    #[test]
    fn stddev_textbook_example() {
        // The classic textbook list 2,4,4,4,5,5,7,9 — population stddev = 2.
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((stddev(&v).unwrap() - 2.0).abs() < 1e-9);
    }
    #[test]
    fn variance_basic() {
        assert!((variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap() - 2.0).abs() < 1e-9);
    }
    #[test]
    fn sum_min_max_count() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sum(&v), 15.0);
        assert_eq!(min(&v).unwrap(), 1.0);
        assert_eq!(max(&v).unwrap(), 5.0);
        assert_eq!(compute(StatOp::Count, &v).unwrap(), 5.0);
    }
    #[test]
    fn empty_list_errors() {
        assert!(mean(&[]).is_err());
        assert!(median(&[]).is_err());
        assert!(mode(&[]).is_err());
        assert!(variance(&[]).is_err());
    }
    #[test]
    fn detect_mean_of_comma() {
        let (op, v) = detect("mean of 1, 2, 3, 4, 5").unwrap();
        assert_eq!(op, StatOp::Mean);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn detect_sum_space() {
        let (op, v) = detect("sum 1 2 3 4 5").unwrap();
        assert_eq!(op, StatOp::Sum);
        assert_eq!(v.len(), 5);
    }
    #[test]
    fn detect_stddev_full_phrase() {
        let (op, _) = detect("standard deviation of 2, 4, 4, 4, 5, 5, 7, 9").unwrap();
        assert_eq!(op, StatOp::Stddev);
    }
    #[test]
    fn detect_no_match() {
        assert!(detect("what is mitochondria").is_none());
        assert!(detect("mean of mitochondria").is_none());
    }
}
