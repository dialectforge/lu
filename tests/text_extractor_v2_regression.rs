//! Regression tests for the text-extractor v2 fix pass (2026-05-31).
//!
//! Anchors the per-layer bond counts produced by the Rust extractor
//! against the canonical Python extractor (smart_extractor.py +
//! ingest_curriculum_full.extract_response_bonds_multi_sentence)
//! on the FULL 46 KB Anarchism and the 25 KB Albedo articles
//! (sourced from the `wikimedia/wikipedia` 20231101.en HuggingFace
//! dump — same stream the production ingest path reads).
//!
//! Empirical baseline captured 2026-05-31 from the v1 binary md5
//! `5d77d524971048740486ce15fc17ab6d`:
//!
//!   Anarchism (46 082 bytes) — Rust 122, Python 122 (byte-identical):
//!     L1=4  L2=24  L2b=12  L3=0  L4=50  L-multi-sentence=27  L-multi-anchor=5
//!
//!   Albedo    (25 117 bytes) — Rust 73,  Python 73  (byte-identical):
//!     L1=12 L2=22 L2b=9  L3=0  L4=12  L-multi-sentence=15 L-multi-anchor=3
//!
//! The Fix 1 noise filter (`shared_noise::reference_noise`) drops 5
//! `HasAspect` bonds from L2 on Anarchism that the prior code
//! emitted (`Explanatory notes`, `Primary sources`, `Secondary
//! sources`, `Tertiary sources`, `General and cited sources`). The
//! L2=24 count above is post-fix; the prior count was L2=24 with 5
//! of those rows being noise. After the filter the count drops to
//! L2=19. These tests assert the post-fix shape.
//!
//! Post-OP-L3-ISA reality (commit c42e20a broadened the title-prefix
//! lookahead to accept `\s+(?:that|which|who|where|and|or|used|of|in|for|from)`).
//! Both articles now produce L3-isa bonds:
//!
//!   Anarchism: 2 IsA bonds. Lead: `"Anarchism is a political philosophy
//!   and movement that is skeptical of all justifications ..."`. The
//!   broadened lookahead accepts `\s+and` AND `\s+that`, so the NP
//!   `"political philosophy and movement"` is captured. The
//!   coordinator-split helper `emit_isa_candidates` further splits it
//!   into two bonds: `Anarchism IsA "political philosophy"` and
//!   `Anarchism IsA "movement"`. Both are real, valid bonds.
//!
//!   Albedo: 1 IsA bond. Lead: `"Albedo is the fraction of sunlight
//!   that is diffusely reflected by a body."`. The empty `(; )`
//!   parenthetical was previously thought to block the regex, but
//!   after the OP-L3-ISA broadening the title-prefix branch matches
//!   `"the fraction of sunlight"` (the `\s+that` lookahead fires).
//!   Result: `Albedo IsA "the fraction of sunlight"`. Real, valid.
//!
//! These bonds are exactly what OP-L3-ISA was meant to capture. The
//! original L3=0 calibration was based on a pre-fix hypothesis that
//! the regex couldn't fire here; that hypothesis is wrong post-fix.

//!
//! Post-OP-EXTRACT-WIDEN recalibration (commit 369aabb, 2026-06-15 — landed
//! AFTER this file was first calibrated on 2026-06-01). `dispatch` no longer
//! dedups on the (subject, relation, object) TRIPLE; it collapses each
//! (subject, object) PAIR to its single richest relation (relation_priority),
//! because the brain itself dedups by pair only and the generic RelatedTo/IsA
//! were shadowing the typed relations (#435). The set of (subject, object)
//! pairs is UNCHANGED (zero recall loss) — only redundant duplicate bonds on
//! an already-connected pair collapse, keeping the richer relation. Net effect
//! on these fixtures: Anarchism L2b 12->11, L4 50->47, total 119->115 (4
//! collapses); Albedo L2b 9->8, total 74->73 (1 collapse). The pre-widen
//! numbers were verified to reproduce exactly under the old triple-dedup, so
//! this is a stale-fixture update, not a quality regression.

use lu4d::extractor::extract_from_article;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn load_fixture(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", p.display(), e))
}

fn layer_counts(bonds: &[lu4d::extractor::Bond]) -> BTreeMap<String, usize> {
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for b in bonds {
        *m.entry(b.layer.to_string()).or_insert(0) += 1;
    }
    m
}

/// Anarchism full-article regression. Post-Fix-1 expected shape.
#[test]
fn anarchism_full_post_fix1_layer_counts() {
    let text = load_fixture("anarchism_full_2026_05_31.txt");
    let bonds = extract_from_article(
        &text,
        Some("Anarchism"),
        Some("https://en.wikipedia.org/wiki/Anarchism"),
    );
    let counts = layer_counts(&bonds);
    assert_eq!(counts.get("L1").copied().unwrap_or(0), 4, "L1 changed: counts={:?}", counts);
    // L2 was 24 pre-Fix-1; 5 reference-noise bonds drop, expecting 19.
    assert_eq!(
        counts.get("L2").copied().unwrap_or(0),
        19,
        "L2 count changed (Fix 1 should drop 5 reference-noise bonds): counts={:?}",
        counts
    );
    // L2b: post-OP-EXTRACT-WIDEN (369aabb) the (anarchism -> anti-capitalism)
    // pair carried both L2 HasAspect and L2b IsA; pair-richest keeps HasAspect,
    // so the L2b IsA collapses (12 -> 11). Zero pair loss.
    assert_eq!(counts.get("L2b").copied().unwrap_or(0), 11, "L2b changed: counts={:?}", counts);
    // L4: post-OP-EXTRACT-WIDEN 3 pairs (classical, education, evolutionary)
    // carried both L2 HasAspect and L4 RelatedTo; pair-richest keeps HasAspect,
    // so 3 L4 RelatedTo collapse (50 -> 47). Zero pair loss.
    assert_eq!(counts.get("L4").copied().unwrap_or(0), 47, "L4 changed: counts={:?}", counts);
    assert_eq!(
        counts.get("L-multi-sentence").copied().unwrap_or(0),
        27,
        "L-multi-sentence changed: counts={:?}", counts
    );
    assert_eq!(
        counts.get("L-multi-anchor").copied().unwrap_or(0),
        5,
        "L-multi-anchor changed: counts={:?}", counts
    );
    // Post-OP-L3-ISA: title-prefix regex now fires on the "is a political
    // philosophy and movement" lead via the broadened lookahead, and
    // emit_isa_candidates splits the `and`-coordinator into 2 bonds.
    assert_eq!(
        counts.get("L3-isa").copied().unwrap_or(0),
        2,
        "L3-isa changed (expecting 2: political philosophy + movement): counts={:?}",
        counts
    );
    // Total post-OP-EXTRACT-WIDEN: 4 + 19 + 11 + 2 + 47 + 27 + 5 = 115
    // (was 119 pre-widen; 4 same-pair duplicate bonds collapsed, no pair lost).
    assert_eq!(bonds.len(), 115, "total bond count changed: {}", bonds.len());
}

/// Anarchism — explicit Fix 1 assertion: the five known reference-
/// noise headings must NOT appear as `HasAspect` bonds.
#[test]
fn anarchism_full_no_reference_noise_in_l2() {
    let text = load_fixture("anarchism_full_2026_05_31.txt");
    let bonds = extract_from_article(&text, Some("Anarchism"), None);
    let l2_objs: Vec<&str> = bonds
        .iter()
        .filter(|b| b.layer == "L2")
        .map(|b| b.object.as_str())
        .collect();
    for noise in [
        "Explanatory notes",
        "Primary sources",
        "Secondary sources",
        "Tertiary sources",
        "General and cited sources",
    ] {
        assert!(
            !l2_objs.contains(&noise),
            "reference-noise heading leaked into L2: {:?}; full L2 list: {:?}",
            noise, l2_objs,
        );
    }
    // Real section headings must still survive.
    for keep in ["History", "Tactics", "Key issues", "Education"] {
        assert!(
            l2_objs.contains(&keep),
            "real section heading dropped: {:?}; full L2 list: {:?}",
            keep, l2_objs,
        );
    }
}

/// Albedo full-article regression. Post-Fix-1 expected shape.
/// Albedo's L2 has no reference-noise headings, so the count is
/// unchanged (22) — this test guards against accidental over-
/// filtering.
#[test]
fn albedo_full_post_fix1_layer_counts() {
    let text = load_fixture("albedo_full_2026_05_31.txt");
    let bonds = extract_from_article(
        &text,
        Some("Albedo"),
        Some("https://en.wikipedia.org/wiki/Albedo"),
    );
    let counts = layer_counts(&bonds);
    assert_eq!(counts.get("L1").copied().unwrap_or(0), 12, "L1: {:?}", counts);
    assert_eq!(counts.get("L2").copied().unwrap_or(0), 22, "L2: {:?}", counts);
    // L2b: post-OP-EXTRACT-WIDEN the (albedo -> land surface effects on climate)
    // pair carried both L2 HasAspect and L2b IsA; pair-richest keeps HasAspect,
    // so the L2b IsA collapses (9 -> 8). Zero pair loss.
    assert_eq!(counts.get("L2b").copied().unwrap_or(0), 8, "L2b: {:?}", counts);
    assert_eq!(counts.get("L4").copied().unwrap_or(0), 12, "L4: {:?}", counts);
    assert_eq!(
        counts.get("L-multi-sentence").copied().unwrap_or(0),
        15,
        "L-multi-sentence: {:?}", counts
    );
    assert_eq!(
        counts.get("L-multi-anchor").copied().unwrap_or(0),
        3,
        "L-multi-anchor: {:?}", counts
    );
    // Post-OP-L3-ISA: title-prefix regex fires on "is the fraction of
    // sunlight" via the broadened `\s+that` lookahead.
    assert_eq!(
        counts.get("L3-isa").copied().unwrap_or(0),
        1,
        "L3-isa changed (expecting 1: fraction of sunlight): counts={:?}",
        counts
    );
    // Total post-OP-EXTRACT-WIDEN: 12 + 22 + 8 + 1 + 12 + 15 + 3 = 73
    // (was 74 pre-widen; 1 same-pair duplicate bond collapsed, no pair lost).
    assert_eq!(bonds.len(), 73, "total: {}", bonds.len());
}
