//! Sentence + paragraph splitting helpers.
//!
//! Mirrors Python `_split_sentences`, `_split_paragraphs`,
//! `_first_n_paragraphs` from `scripts/smart_extractor.py`.

use super::patterns::sentence_split_re;

/// Crude sentence splitter mirroring Python `_split_sentences`.
/// Splits on `.!?` followed by whitespace, or on newlines. The Python
/// uses fancy-regex lookbehind which is supported by `fancy_regex`,
/// so we keep the same pattern.
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut last = 0usize;
    // fancy_regex find_iter returns Result<Match>; iterate manually.
    let re = sentence_split_re();
    let mut iter = re.find_iter(text);
    while let Some(Ok(m)) = iter.next() {
        if m.start() > last {
            let chunk = &text[last..m.start()];
            let trimmed = chunk.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
        }
        last = m.end();
    }
    if last < text.len() {
        let trimmed = text[last..].trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    out
}

/// Paragraph splitter (Python `_split_paragraphs`).
pub fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

pub fn first_n_paragraphs(text: &str, n: usize) -> Vec<String> {
    split_paragraphs(text).into_iter().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_basic() {
        let s = split_sentences("Hello world. This is a test.\nNew line.");
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], "Hello world.");
        assert_eq!(s[1], "This is a test.");
        assert_eq!(s[2], "New line.");
    }

    #[test]
    fn split_paragraphs_basic() {
        let p = split_paragraphs("para 1\n\npara 2\n\n   \n\npara 3");
        assert_eq!(p, vec!["para 1", "para 2", "para 3"]);
    }

    #[test]
    fn first_n_paragraphs_caps() {
        let p = first_n_paragraphs("a\n\nb\n\nc\n\nd", 2);
        assert_eq!(p, vec!["a", "b"]);
    }
}
