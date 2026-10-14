//! Port F.2 — single-step unit conversion.
//!
//! Categories: time, distance, weight, temperature.
//! For time/distance/weight we use a base unit per category and convert
//! via a scale factor; for temperature we go through Celsius (offset
//! conversion, not just scaling).
//!
//! Multi-step conversions ("3 hours 5 minutes in seconds") are out of
//! scope for F.2 — see the deferred list.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitConversion {
    pub value: f64,
    pub from: String,
    pub to: String,
    pub result: f64,
    pub category: String,
}

pub fn convert(value: f64, from_raw: &str, to_raw: &str) -> Result<UnitConversion, String> {
    let from = canonical_unit(from_raw)
        .ok_or_else(|| format!("unknown unit '{}'", from_raw))?;
    let to = canonical_unit(to_raw)
        .ok_or_else(|| format!("unknown unit '{}'", to_raw))?;
    let cat_from = unit_category(from)?;
    let cat_to = unit_category(to)?;
    if cat_from != cat_to {
        return Err(format!(
            "cannot convert {} ({}) to {} ({}): incompatible categories",
            from, cat_from, to, cat_to
        ));
    }
    let result = if cat_from == "temperature" {
        convert_temperature(value, from, to)?
    } else {
        let f = unit_factor(from)?;
        let t = unit_factor(to)?;
        value * f / t
    };
    Ok(UnitConversion {
        value,
        from: from.to_string(),
        to: to.to_string(),
        result,
        category: cat_from.to_string(),
    })
}

/// Map a free-form string ("kilometers", "km", "KM", "mi", "MILE")
/// to the internal canonical token used in the rest of the module.
fn canonical_unit(raw: &str) -> Option<&'static str> {
    let s = raw.trim().trim_end_matches('.').to_lowercase();
    let s = s.trim_start_matches('°');
    Some(match s {
        // time
        "second" | "seconds" | "sec" | "secs" | "s" => "second",
        "minute" | "minutes" | "min" | "mins" => "minute",
        "hour" | "hours" | "hr" | "hrs" | "h" => "hour",
        "day" | "days" | "d" => "day",
        "week" | "weeks" | "wk" | "wks" => "week",
        "year" | "years" | "yr" | "yrs" | "y" => "year",
        // distance
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => "mm",
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => "cm",
        "m" | "meter" | "meters" | "metre" | "metres" => "m",
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => "km",
        "in" | "inch" | "inches" => "in",
        "ft" | "foot" | "feet" => "ft",
        "yd" | "yard" | "yards" => "yd",
        "mi" | "mile" | "miles" => "mile",
        // weight
        "g" | "gram" | "grams" => "g",
        "kg" | "kilogram" | "kilograms" => "kg",
        "ton" | "tons" | "tonne" | "tonnes" | "t" => "ton",
        "oz" | "ounce" | "ounces" => "oz",
        "lb" | "lbs" | "pound" | "pounds" => "lb",
        // temperature
        "c" | "celsius" | "centigrade" => "C",
        "f" | "fahrenheit" => "F",
        "k" | "kelvin" => "K",
        _ => return None,
    })
}

fn unit_category(unit: &str) -> Result<&'static str, String> {
    match unit {
        "second" | "minute" | "hour" | "day" | "week" | "year" => Ok("time"),
        "mm" | "cm" | "m" | "km" | "in" | "ft" | "yd" | "mile" => Ok("distance"),
        "g" | "kg" | "ton" | "oz" | "lb" => Ok("weight"),
        "C" | "F" | "K" => Ok("temperature"),
        _ => Err(format!("unknown unit '{}'", unit)),
    }
}

fn unit_factor(unit: &str) -> Result<f64, String> {
    Ok(match unit {
        // time → seconds
        "second" => 1.0,
        "minute" => 60.0,
        "hour" => 3600.0,
        "day" => 86_400.0,
        "week" => 604_800.0,
        "year" => 86_400.0 * 365.25, // Julian year per spec.
        // distance → meters
        "mm" => 0.001,
        "cm" => 0.01,
        "m" => 1.0,
        "km" => 1000.0,
        "in" => 0.0254,
        "ft" => 0.3048,
        "yd" => 0.9144,
        "mile" => 1609.344,
        // weight → grams
        "g" => 1.0,
        "kg" => 1000.0,
        "ton" => 1_000_000.0, // metric ton
        "oz" => 28.349_523_125,
        "lb" => 453.592_37,
        _ => return Err(format!("no scale factor for '{}'", unit)),
    })
}

fn convert_temperature(value: f64, from: &str, to: &str) -> Result<f64, String> {
    let c = match from {
        "C" => value,
        "F" => (value - 32.0) * 5.0 / 9.0,
        "K" => value - 273.15,
        _ => return Err(format!("not a temperature unit: {}", from)),
    };
    Ok(match to {
        "C" => c,
        "F" => c * 9.0 / 5.0 + 32.0,
        "K" => c + 273.15,
        _ => return Err(format!("not a temperature unit: {}", to)),
    })
}

// =====================================================================
// Detection
// =====================================================================

/// Try to recognise a unit-conversion query and return `(value, from, to)`.
///
/// Patterns supported (Round 1):
/// * `convert N UNIT to UNIT`
/// * `convert N UNIT in UNIT`
/// * `how many UNIT in [N|a|an] UNIT`
/// * `how many UNIT are in [N|a|an] UNIT`
/// * `N UNIT to UNIT`
/// * `N UNIT in UNIT`
pub fn detect(query: &str) -> Option<(f64, String, String)> {
    let q = query.trim().to_lowercase();
    let q = q
        .trim_end_matches(|c: char| matches!(c, '.' | '?' | '!'))
        .trim();

    if let Some(rest) = q.strip_prefix("convert ") {
        return parse_n_unit_to_unit(rest);
    }
    if let Some(rest) = q.strip_prefix("how many ") {
        return parse_how_many(rest);
    }
    parse_n_unit_to_unit(q)
}

fn parse_n_unit_to_unit(s: &str) -> Option<(f64, String, String)> {
    let s = s.trim();
    let (lhs, rhs) = if let Some(idx) = s.find(" to ") {
        (&s[..idx], &s[idx + 4..])
    } else if let Some(idx) = s.find(" in ") {
        (&s[..idx], &s[idx + 4..])
    } else {
        return None;
    };
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    let (val_str, from_unit) = lhs.split_once(' ')?;
    let value: f64 = val_str.parse().ok()?;
    Some((value, from_unit.trim().to_string(), rhs.to_string()))
}

fn parse_how_many(rest: &str) -> Option<(f64, String, String)> {
    let rest = rest.trim();
    let (to_unit, after) = if let Some(idx) = rest.find(" are in ") {
        (&rest[..idx], &rest[idx + 8..])
    } else if let Some(idx) = rest.find(" in ") {
        (&rest[..idx], &rest[idx + 4..])
    } else {
        return None;
    };
    let to_unit = to_unit.trim();
    let after = after.trim();
    let (val, from_unit) = if let Some(rest) = after.strip_prefix("a ") {
        (1.0, rest.trim())
    } else if let Some(rest) = after.strip_prefix("an ") {
        (1.0, rest.trim())
    } else if let Some((num, unit)) = after.split_once(' ') {
        let v: f64 = num.parse().ok()?;
        (v, unit.trim())
    } else {
        // Bare unit, no value: "how many minutes in hour" — assume 1.
        (1.0, after)
    };
    Some((val, from_unit.to_string(), to_unit.to_string()))
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn ok(value: f64, from: &str, to: &str) -> f64 {
        convert(value, from, to).unwrap().result
    }

    // --- factor-based -----------------------------------------------
    #[test]
    fn time_seconds_in_a_day() {
        assert_eq!(ok(1.0, "day", "second"), 86_400.0);
    }
    #[test]
    fn time_minutes_in_hour() {
        assert_eq!(ok(1.0, "hour", "minute"), 60.0);
    }
    #[test]
    fn distance_km_to_miles() {
        let r = ok(100.0, "km", "mile");
        assert!((r - 62.137_119).abs() < 1e-3, "got {}", r);
    }
    #[test]
    fn distance_meters_to_feet() {
        let r = ok(1.0, "m", "ft");
        assert!((r - 3.280_840).abs() < 1e-3);
    }
    #[test]
    fn weight_lb_to_kg() {
        let r = ok(10.0, "lb", "kg");
        assert!((r - 4.535_924).abs() < 1e-3);
    }
    #[test]
    fn weight_oz_to_g() {
        let r = ok(16.0, "oz", "g");
        assert!((r - 453.592_37).abs() < 1e-3);
    }

    // --- temperature -------------------------------------------------
    #[test]
    fn temp_f_to_c_freezing() {
        assert!((ok(32.0, "F", "C")).abs() < 1e-9);
    }
    #[test]
    fn temp_c_to_f_boiling() {
        assert!((ok(100.0, "C", "F") - 212.0).abs() < 1e-9);
    }
    #[test]
    fn temp_c_to_k_zero() {
        assert!((ok(0.0, "C", "K") - 273.15).abs() < 1e-9);
    }
    #[test]
    fn temp_k_to_c() {
        assert!((ok(300.0, "K", "C") - 26.85).abs() < 1e-9);
    }

    // --- mismatches and unknowns -------------------------------------
    #[test]
    fn mismatched_categories_error() {
        assert!(convert(5.0, "kg", "hour").is_err());
    }
    #[test]
    fn unknown_unit_error() {
        assert!(convert(5.0, "smoots", "m").is_err());
    }

    // --- detection ---------------------------------------------------
    #[test]
    fn detect_convert_explicit() {
        let (v, f, t) = detect("convert 100 km to miles").unwrap();
        assert_eq!(v, 100.0);
        assert_eq!(f, "km");
        assert_eq!(t, "miles");
    }
    #[test]
    fn detect_how_many_implicit_one() {
        let (v, f, t) = detect("how many seconds in a day").unwrap();
        assert_eq!(v, 1.0);
        assert_eq!(f, "day");
        assert_eq!(t, "seconds");
    }
    #[test]
    fn detect_how_many_with_count() {
        let (v, f, t) = detect("how many seconds in 3 hours").unwrap();
        assert_eq!(v, 3.0);
        assert_eq!(f, "hours");
        assert_eq!(t, "seconds");
    }
    #[test]
    fn detect_temperature_phrase() {
        let (v, f, t) = detect("convert 32 F to C").unwrap();
        assert_eq!(v, 32.0);
        assert_eq!(f, "f"); // lowercased before normalisation
        assert_eq!(t, "c");
    }
    #[test]
    fn detect_short_phrase() {
        let (v, f, t) = detect("100 km in miles").unwrap();
        assert_eq!(v, 100.0);
        assert_eq!(f, "km");
        assert_eq!(t, "miles");
    }
    #[test]
    fn detect_no_match() {
        assert!(detect("what is mitochondria").is_none());
        assert!(detect("convert mitochondria to a cell").is_none());
    }

    // End-to-end: detection followed by conversion.
    #[test]
    fn end_to_end_km_to_miles() {
        let (v, f, t) = detect("convert 100 km to miles").unwrap();
        let res = convert(v, &f, &t).unwrap();
        assert!((res.result - 62.137_119).abs() < 1e-3);
    }
    #[test]
    fn end_to_end_seconds_per_day() {
        let (v, f, t) = detect("how many seconds in a day").unwrap();
        let res = convert(v, &f, &t).unwrap();
        assert_eq!(res.result, 86_400.0);
    }
    #[test]
    fn end_to_end_freezing_point() {
        let (v, f, t) = detect("convert 32 F to C").unwrap();
        let res = convert(v, &f, &t).unwrap();
        assert!((res.result).abs() < 1e-9);
    }
}
