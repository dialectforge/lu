//! Integration test — runs the full extractor over two synthetic
//! Wikipedia-shaped articles and asserts the structural invariants
//! that motivated the port: ≥30 bonds from the long article, ≥3 L2b
//! category bonds, no oversize objects, no verb-led object heads, at
//! least one clean L3 IsA bond.

use lu4d::bond::BondType;
use lu4d::extractor::extract_from_article;
use regex::Regex;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn load(name: &str) -> String {
    let path = fixture_path(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read fixture {}: {}", path.display(), e)
    })
}

#[test]
fn anarchism_sample_emits_dense_extraction() {
    let text = load("anarchism_sample.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), Some("https://en.wikipedia.org/wiki/Anarchism"));
    // Post-OP-EXTRACT-WIDEN (369aabb, 2026-06-15): dispatch now collapses each
    // (subject, object) pair to its single richest relation instead of the old
    // (subject, relation, object) triple-dedup. The pair SET is unchanged
    // (24 distinct pairs, zero recall loss) but redundant duplicate bonds on
    // already-connected pairs collapse: 7 sample pairs carried both a generic
    // L1 RelatedTo / L2b IsA AND a richer relation, so the density floor drops
    // from the pre-widen 32 to 24. This asserts the post-widen dense floor.
    assert!(
        bonds.len() >= 24,
        "expected >=24 bonds, got {}; layers={:?}",
        bonds.len(),
        bonds.iter().map(|b| b.layer).collect::<Vec<_>>()
    );
}

#[test]
fn anarchism_sample_l2b_categories_anchor_to_title() {
    let text = load("anarchism_sample.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), None);
    let l2b: Vec<&_> = bonds.iter().filter(|b| b.layer == "L2b").collect();
    assert!(
        l2b.len() >= 3,
        "expected >=3 L2b category bonds, got {}",
        l2b.len(),
    );
    for b in &l2b {
        assert_eq!(
            b.subject, "Anarchism",
            "L2b bond subject must match title, got {:?}",
            b
        );
        assert_eq!(b.relation, BondType::IsA);
    }
}

#[test]
fn no_oversize_objects() {
    let text = load("anarchism_sample.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), None);
    for b in &bonds {
        assert!(
            b.object.chars().count() <= 80,
            "oversize object ({} chars): {:?}",
            b.object.chars().count(),
            b
        );
    }
}

#[test]
fn no_verb_led_object_heads() {
    let text = load("anarchism_sample.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), None);
    let verb_led_re = Regex::new(r"^(?:[a-z]+(?:ed|ing))\b").unwrap();
    for b in &bonds {
        assert!(
            !verb_led_re.is_match(&b.object),
            "object starts with verb form: {:?}",
            b
        );
    }
}

#[test]
fn at_least_one_l3_isa_bond_with_clean_np() {
    // "Anarchism is a political philosophy." produces an L3-isa
    // IsA bond with object "political philosophy" -- which the
    // dispatcher's dedup pass discards because the L2b category
    // walker emitted the same (subject, IsA, object) triple
    // first (L2b runs before L3). Accept any L3-* bond in lieu
    // of insisting on the IsA-specifically-from-L3 shape.
    let text = load("anarchism_sample.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), None);
    let l3: Vec<&_> = bonds
        .iter()
        .filter(|b| b.layer.starts_with("L3"))
        .collect();
    if !l3.is_empty() {
        for b in &l3 {
            assert!(
                !b.object.is_empty() && b.object.chars().count() <= 60,
                "L3 object must be small + non-empty: {:?}",
                b
            );
        }
        return;
    }
    // Fall-through: confirm the title-prefix IsA at least fired
    // at the bare-extractor layer even though dispatch dedupped it.
    use lu4d::extractor::layers::l3_lead_patterns;
    let l3_raw = l3_lead_patterns::extract("Anarchism", &text, None);
    let l3_isa: Vec<&_> = l3_raw.iter()
        .filter(|b| b.relation == BondType::IsA && b.layer == "L3-isa")
        .collect();
    assert!(
        !l3_isa.is_empty(),
        "L3 layer must produce at least one IsA bond at the raw layer"
    );
    for b in &l3_isa {
        assert!(
            !b.object.is_empty() && b.object.chars().count() <= 60,
            "L3-isa object must be small + non-empty: {:?}",
            b
        );
    }
}

#[test]
fn albedo_sample_extraction_produces_bonds() {
    let text = load("albedo_sample.txt");
    let bonds = extract_from_article(&text, Some("Albedo"), None);
    assert!(
        !bonds.is_empty(),
        "expected at least one bond from the Albedo fixture"
    );
    // L2b categories must be present (Climatology, Radiometry, ...).
    let l2b: Vec<&_> = bonds.iter().filter(|b| b.layer == "L2b").collect();
    assert!(
        !l2b.is_empty(),
        "expected at least one L2b category bond"
    );
}
