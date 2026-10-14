//! HTML cleaning + heuristic SVO triple extraction.
//!
//! Per port-spec: "heuristic only … will be lossy and that's fine —
//! proposals go through approval anyway." The Critic at approval time
//! is the real filter; here we just want plausible candidates.
//!
//! Extraction strategy:
//! 1. Parse with `scraper`. Strip `<script>`, `<style>`, `<nav>`,
//!    `<aside>`, `<footer>`, `<form>` subtrees outright.
//! 2. Prefer `<main>` or `<article>` if present; fall back to `<body>`.
//! 3. Concatenate visible text with whitespace normalization.
//!
//! SVO: regex-free, depends on a small predicate-verb whitelist. For
//! each sentence (split on `.!?`), look for `<NP> <VERB> <NP>` where
//! the verb is in the whitelist (is/was/are/were/has/have/contains/
//! includes/causes/requires/produces/uses/affects/enables). The two
//! NPs are the subject (text before the verb) and the object (text
//! after the verb, up to the next verb or end of sentence). NPs are
//! lower-cased and trimmed of leading articles.

use crate::bond::BondType;
use scraper::{Html, Selector};

/// One sentence-derived triple. `bond_type` is best-guess from the verb;
/// the Critic / John can override at approval time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub bond_type: BondType,
    /// The full sentence the triple was extracted from. Carried so the
    /// proposal can show provenance to the human approver.
    pub source_sentence: String,
}

/// Parse `html` and return cleaned visible text. Empty string if no
/// extractable content. Never panics on malformed HTML — `scraper`
/// uses `html5ever` which is tolerant.
pub fn extract_clean_text(html: &str) -> String {
    if html.trim().is_empty() {
        return String::new();
    }
    let doc = Html::parse_document(html);
    // Selectors are constructed from string literals so `parse` cannot
    // realistically fail at runtime — unwrap is the same panic discipline
    // scraper's own examples use.
    let strip_sel = Selector::parse("script, style, nav, aside, footer, form, noscript").unwrap();
    // Walk into the preferred container if any.
    let main_sel = Selector::parse("main, article").unwrap();
    let body_sel = Selector::parse("body").unwrap();

    // To strip we'd need a mutable doc; scraper doesn't expose one.
    // Instead, when collecting text we skip nodes whose ancestors match
    // the strip selectors. Build an "is-junk" predicate by collecting
    // ids of junk nodes.
    let junk_ids: std::collections::HashSet<_> =
        doc.select(&strip_sel).map(|e| e.id()).collect();

    let collect_from = |root: scraper::ElementRef| -> String {
        let mut out = String::new();
        for descendant in root.descendants() {
            if let Some(text) = descendant.value().as_text() {
                // Skip if any ancestor is junk.
                let is_junk = descendant.ancestors().any(|a| junk_ids.contains(&a.id()));
                if !is_junk {
                    let s = text.trim();
                    if !s.is_empty() {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(s);
                    }
                }
            }
        }
        // Whitespace normalization.
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    };

    if let Some(node) = doc.select(&main_sel).next() {
        return collect_from(node);
    }
    if let Some(node) = doc.select(&body_sel).next() {
        return collect_from(node);
    }
    // Last resort: raw text of the doc root.
    let mut out = String::new();
    for n in doc.tree.nodes() {
        if let scraper::node::Node::Text(t) = n.value() {
            let s = t.trim();
            if !s.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(s);
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split on .!? — naive but sufficient for proposal candidates.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if c == '.' || c == '!' || c == '?' {
            let s = cur.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Verb whitelist + bond-type best-guess mapping. Lowercase form.
fn verb_to_bond(verb: &str) -> Option<BondType> {
    match verb {
        "is" | "are" | "was" | "were" | "be" | "being" | "been" => Some(BondType::IsA),
        "has" | "have" | "had" | "contains" | "includes" => Some(BondType::HasProperty),
        "causes" | "caused" | "cause" => Some(BondType::Causes),
        "requires" | "required" | "require" => Some(BondType::Requires),
        "produces" | "produced" | "produce" => Some(BondType::Produces),
        "uses" | "used" | "use" => Some(BondType::UsedFor),
        "opposes" | "opposed" | "oppose" => Some(BondType::Opposes),
        "derived" => Some(BondType::DerivedFrom),
        _ => None,
    }
}

/// Strip leading articles / common stopwords.
fn normalize_np(s: &str) -> String {
    let s = s.trim().to_lowercase();
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return String::new();
    }
    let drop_lead = matches!(parts[0], "the" | "a" | "an" | "this" | "that" | "these" | "those");
    let kept = if drop_lead && parts.len() > 1 {
        &parts[1..]
    } else {
        &parts[..]
    };
    // Keep at most 4 tokens to avoid run-on phrases.
    let take = kept.iter().take(4).copied().collect::<Vec<_>>().join(" ");
    take.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ')
        .trim()
        .to_string()
}

/// Extract candidate triples from a chunk of clean text.
///
/// `max_triples` caps the output. Sentences are scanned in order; once
/// the cap is hit further sentences are ignored. Returned triples may
/// have empty subject or object — the proposal layer filters those.
pub fn extract_triples(text: &str, max_triples: usize) -> Vec<ExtractedTriple> {
    let mut out: Vec<ExtractedTriple> = Vec::new();
    if max_triples == 0 {
        return out;
    }
    for sentence in split_sentences(text) {
        if out.len() >= max_triples {
            break;
        }
        let lower = sentence.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        // Find first verb hit; walk left for subject, right for object.
        for (i, tok) in tokens.iter().enumerate() {
            // Strip trailing punctuation from the token before verb-lookup.
            let clean: String = tok
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            let bond_type = match verb_to_bond(&clean) {
                Some(b) => b,
                None => continue,
            };
            if i == 0 || i + 1 >= tokens.len() {
                continue;
            }
            // Subject: the 1-4 tokens immediately before the verb.
            let s_lo = i.saturating_sub(4);
            let subj_raw = tokens[s_lo..i].join(" ");
            // Object: the next 1-4 tokens after the verb (until another
            // verb or sentence boundary).
            let mut obj_end = i + 1;
            while obj_end < tokens.len() && obj_end - i <= 4 {
                let cl: String = tokens[obj_end]
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
                if verb_to_bond(&cl).is_some() {
                    break;
                }
                obj_end += 1;
            }
            let obj_raw = tokens[i + 1..obj_end].join(" ");
            let subject = normalize_np(&subj_raw);
            let object = normalize_np(&obj_raw);
            if subject.is_empty() || object.is_empty() {
                continue;
            }
            if subject == object {
                continue;
            }
            out.push(ExtractedTriple {
                subject,
                predicate: clean,
                object,
                bond_type,
                source_sentence: sentence.clone(),
            });
            // One triple per sentence — keeps the candidate space tight.
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_clean_text_strips_scripts() {
        let html = r#"<html><head><script>var x=1;</script></head><body><main><p>Hello world.</p><script>alert(2);</script></main></body></html>"#;
        let t = extract_clean_text(html);
        assert!(t.contains("Hello world"));
        assert!(!t.contains("alert"));
        assert!(!t.contains("var x"));
    }

    #[test]
    fn test_extract_clean_text_prefers_main() {
        let html = r#"<html><body><nav>SKIP</nav><main>KEEP</main><footer>NO</footer></body></html>"#;
        let t = extract_clean_text(html);
        assert_eq!(t, "KEEP");
    }

    #[test]
    fn test_extract_clean_text_real_article_shape() {
        // Mimics a real Grokipedia / Wikipedia-style page with an
        // <article> wrapper and surrounding chrome.
        let html = r#"<html><head><title>X</title><script>1</script></head>
            <body>
              <nav>HOME ABOUT</nav>
              <article>
                <h1>Rust (programming language)</h1>
                <p>Rust is a memory-safe systems programming language.</p>
                <p>Rust uses ownership instead of garbage collection.</p>
              </article>
              <footer>(c) 2026</footer>
            </body></html>"#;
        let t = extract_clean_text(html);
        assert!(t.contains("Rust is a memory-safe systems programming language"));
        assert!(t.contains("ownership instead of garbage collection"));
        assert!(!t.contains("HOME"));
        assert!(!t.contains("2026"));
    }

    #[test]
    fn test_split_sentences() {
        let s = split_sentences("One. Two! Three? Four");
        assert_eq!(s, vec!["One.", "Two!", "Three?", "Four"]);
    }

    #[test]
    fn test_extract_triples_finds_is_a() {
        let text = "Rust is a systems programming language. Mozilla sponsored Rust.";
        let trips = extract_triples(text, 8);
        assert!(!trips.is_empty());
        // Ensure at least one IsA was caught.
        let has_isa = trips.iter().any(|t| matches!(t.bond_type, BondType::IsA));
        assert!(has_isa);
    }

    #[test]
    fn test_extract_triples_finds_causes_and_requires() {
        let text = "Friction causes heat. Combustion requires oxygen.";
        let trips = extract_triples(text, 8);
        let kinds: Vec<_> = trips.iter().map(|t| t.bond_type).collect();
        assert!(kinds.iter().any(|b| matches!(b, BondType::Causes)));
        assert!(kinds.iter().any(|b| matches!(b, BondType::Requires)));
    }

    #[test]
    fn test_extract_triples_respects_cap() {
        let text =
            "A is one. B is two. C is three. D is four. E is five. F is six. G is seven.";
        let trips = extract_triples(text, 3);
        assert_eq!(trips.len(), 3);
    }

    #[test]
    fn test_normalize_np_drops_articles() {
        assert_eq!(normalize_np("The Rust Compiler"), "rust compiler");
        assert_eq!(normalize_np("a memory-safe language."), "memory-safe language");
    }
}
