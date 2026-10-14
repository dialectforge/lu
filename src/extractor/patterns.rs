//! All regex constants and stopword sets used by the extractor.
//!
//! Mirrors the corresponding Python `_RX_*` / `_*_SET` constants in
//! `scripts/smart_extractor.py` and `scripts/ingest_curriculum_full.py`.
//!
//! Patterns that use lookahead (`(?=...)`) are compiled with the
//! `fancy_regex` crate. Everything else uses `regex` (faster, no
//! lookahead support).

use fancy_regex::Regex as FRegex;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

// ===================================================================
// Heading + drop sets (smart_extractor.py)
// ===================================================================

pub fn heading_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^([A-Z][A-Za-z0-9 ,\-/&'()]{2,79})\s*$").unwrap())
}

pub fn drop_headings() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "see also", "references", "external links", "notes",
            "bibliography", "further reading", "sources", "footnotes",
            "citations", "gallery", "contents", "index",
        ]
        .into_iter()
        .collect()
    })
}

pub fn stop_objects() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "", "the", "a", "an", "of", "and", "or", "but", "to", "with",
            "in", "on", "at", "by", "for", "from", "this", "that",
            "these", "those", "it", "its", "is", "was", "are", "were",
            "be", "been", "being",
        ]
        .into_iter()
        .collect()
    })
}

pub fn stop_isa_nouns() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "name", "romanization", "transliteration", "variant", "form",
            "version", "alternate", "alternative", "abbreviation",
            "acronym", "way", "thing", "matter", "manner", "pronunciation",
            "spelling", "young", "youngest", "oldest", "first", "second",
            "third", "latter", "former", "only", "such", "least", "most",
            "more", "less", "better", "worse", "best", "worst", "earlier",
            "later", "final", "initial", "original", "main", "primary",
            "secondary", "latest", "earliest", "standard", "normal",
            "typical", "common", "rare", "strange", "odd", "even", "same",
            "other", "another", "each", "type", "kind", "species",
            "variety", "category", "set", "group", "class", "family",
            "order", "phylum", "genus", "piece",
        ]
        .into_iter()
        .collect()
    })
}

pub fn l4_single_word_stop() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            // Sentence-start function words
            "the", "a", "an", "this", "that", "these", "those", "it", "its",
            "he", "she", "his", "her", "hers", "him", "i", "we", "they",
            "them", "their", "theirs", "you", "your", "yours", "my", "mine",
            "our", "ours", "in", "on", "at", "by", "for", "from", "to",
            "with", "of", "and", "or", "but", "so", "yet", "nor", "if",
            "when", "while", "as", "because", "although", "though",
            "whereas", "whilst", "during", "before", "after", "until",
            "since", "however", "therefore", "thus", "moreover",
            "furthermore", "nevertheless", "still", "meanwhile", "many",
            "much", "several", "some", "all", "both", "each", "every",
            "other", "another", "such", "any", "none", "few", "most",
            "more", "less", "least", "best", "worst", "first", "second",
            "third", "yes", "no", "maybe", "perhaps", "often", "sometimes",
            "always", "never", "there", "here", "now", "then", "today",
            "tomorrow", "yesterday", "once", "twice", "also", "only",
            "even", "just",
            // Spelled-out numbers (sentence-start)
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "hundred", "thousand",
            "million",
            // Sentence-start hedges
            "according", "besides", "thereafter", "thereby", "thereof",
            "thereupon", "unlike", "like", "upon", "throughout", "about",
            "around", "despite", "essentially", "basically", "finally",
            "firstly", "secondly", "thirdly", "originally", "eventually",
            "recently", "currently", "previously", "subsequently",
            // Auxiliaries
            "is", "was", "are", "were", "be", "been", "being", "has",
            "have", "had", "do", "does", "did", "will", "would", "could",
            "should", "can", "may", "might", "must", "shall",
            // Months
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november",
            "december",
            // Days
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday",
            // Heading-like
            "see", "references", "notes", "sources", "bibliography",
            "contents", "introduction", "summary", "overview", "chapter",
            "section", "part", "page", "figure", "table", "image", "file",
        ]
        .into_iter()
        .collect()
    })
}

pub fn l4_generic_adjectives() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "american", "british", "french", "german", "italian", "spanish",
            "russian", "chinese", "japanese", "korean", "indian",
            "european", "asian", "african", "australian", "canadian",
            "mexican", "brazilian", "greek", "roman", "latin", "arabic",
            "persian", "jewish", "christian", "muslim", "buddhist",
            "hindu", "islamic", "orthodox", "catholic", "protestant",
            "democratic", "republican", "socialist", "communist",
            "northern", "southern", "eastern", "western", "central",
            "upper", "lower", "new", "old", "modern", "ancient",
            "medieval", "early", "late", "middle", "native",
        ]
        .into_iter()
        .collect()
    })
}

pub fn person_name_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z\.]+){1,2}$").unwrap())
}

pub fn category_hints_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:s|people|writers|musicians|artists|scientists|families|works|albums|songs|films|books|species|cuisine|history|culture|states|cities|towns|villages|countries|presidents|composers|philosophers|kings|queens|wars|battles|treaties|empires|kingdoms|republics|theorists|navies|forces|academics|staff|theory|theories|systems|methods|laws|rules|religions|languages|alphabets|equations|articles|inscriptions|saints|gods|members|games|plays|essays)\b",
        )
        .unwrap()
    })
}

// ===================================================================
// Sentence splitting (smart_extractor.py)
// ===================================================================

pub fn sentence_split_re() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| FRegex::new(r"(?<=[.!?])\s+|\n+").unwrap())
}

pub fn capital_word_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Z][A-Za-z]+").unwrap())
}

pub fn alpha_word_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Za-z]+").unwrap())
}

// ===================================================================
// L2b tightening
// ===================================================================

pub fn digits_only_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\d{1,4}$").unwrap())
}

// ===================================================================
// L3 — lead-paragraph typed patterns (lookahead-heavy → fancy_regex).
// ===================================================================

pub fn rx_located() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:located|situated|based|headquartered)\s+(?:in|at|near)\s+(the\s+)?([A-Z][a-zA-Z]{1,30}(?:\s+(?:[A-Z][a-zA-Z]{1,30}|of|the|de)){0,3})(?=[.;\n]|,\s+(?:which|who|where|that|and))",
        )
        .unwrap()
    })
}

pub fn rx_in_place() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\bin\s+the\s+(?:city|town|village|state|country|province|region|county|district|kingdom|empire|republic)\s+of\s+([A-Z][a-zA-Z]{1,30}(?:\s+(?:[A-Z][a-zA-Z]{1,30}|of|the|de)){0,3})(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_part_of() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:part|member|component)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z]{1,30}(?:\s+(?:[A-Z][a-zA-Z]{1,30}|of|the|de)){0,3})(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_founded_by() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:founded|established|created|invented|developed)\s+by\s+([A-Z][a-z\.\-]+(?:\s+(?:[A-Z][a-z\.\-]+|de|van|von|of|the|den)){0,3})(?=[.;\n,]|\s+in\s+\d)",
        )
        .unwrap()
    })
}

pub fn rx_founded_in() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b(?:founded|established|created|built|formed)\s+in\s+(\d{3,4}(?:\s*BC)?)\b").unwrap()
    })
}

pub fn rx_discovered_by() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\bdiscovered\s+by\s+([A-Z][a-z\.\-]+(?:\s+(?:[A-Z][a-z\.\-]+|de|van|von|of|the|den)){0,3})(?=[.;\n,]|\s+in\s+\d)",
        )
        .unwrap()
    })
}

pub fn rx_named_after() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\bnamed\s+(?:after|for|in honor of|in honour of)\s+([A-Z][a-z\.\-]+(?:\s+(?:[A-Z][a-z\.\-]+|de|van|von|of|the|den)){0,3})(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_measured_in() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(r"\b(?:measured|expressed|recorded)\s+in\s+([a-z][a-z\- ]{2,40}?)(?=[.;\n,])").unwrap()
    })
}

pub fn rx_born_in() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\bborn\s+(?:on\s+\w+\s+\d+,?\s+)?(?:in\s+)?([A-Z][A-Za-z\-, ]{2,80}?|\d{3,4})(?=[.;\n,]|\s+in\s+\d)",
        )
        .unwrap()
    })
}

pub fn rx_died_in() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\bdied\s+(?:on\s+\w+\s+\d+,?\s+)?(?:in\s+)?([A-Z][A-Za-z\-, ]{2,80}?|\d{3,4})(?=[.;\n,]|\s+in\s+\d)",
        )
        .unwrap()
    })
}

pub fn rx_written_by() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:written|composed|directed|produced|painted|sculpted|designed)\s+by\s+([A-Z][a-z\.\-]+(?:\s+(?:[A-Z][a-z\.\-]+|de|van|von|of|the|den)){0,3})(?=[.;\n,]|\s+in\s+\d)",
        )
        .unwrap()
    })
}

pub fn rx_composed_of() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(r"\b(?:consists|composed|made|comprised|formed)\s+of\s+([a-z][a-zA-Z \-,]{2,80}?)(?=[.;\n])").unwrap()
    })
}

pub fn rx_known_for() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(r"\b(?:known|famous|noted|celebrated)\s+for\s+([a-z][a-zA-Z \-,]{2,80}?)(?=[.;\n])").unwrap()
    })
}

pub fn rx_used_for() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(r"\b(?:used|employed)\s+(?:as|for|in)\s+([a-z][a-zA-Z \-,]{2,80}?)(?=[.;\n])").unwrap()
    })
}

pub fn rx_type_of() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(r"\b(?:a|an)\s+(?:type|kind|form|species|variety|category)\s+of\s+([a-z][a-zA-Z \-]{2,60}?)(?=[.;\n,])").unwrap()
    })
}

pub fn rx_isa_fallback() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        // Widened 2026-05-31: NP inner-token min 0 (allows "of"/"in"/"to"
        // inside NP), NP repetition cap raised to 5 (6 total tokens).
        // Lookahead alternates now include the 11 widened set requiring
        // \s+ separator so the boundary is unambiguous.
        FRegex::new(
            r"\b(?:is|was|are|were)\s+(?:an?|the)\s+([a-z][a-zA-Z\-]{2,30}(?:\s+[a-z][a-zA-Z\-]{0,30}){0,5})(?=[.;,]|\s+(?:that|which|who|where|and|or|used|of|in|for|from)\b|$)",
        )
        .unwrap()
    })
}

pub fn trailing_junk_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\s+(?:and|or|but|which|who|where|that|in|on|of|at|by|for|from|to|with|the)\s*$").unwrap()
    })
}

// ===================================================================
// L4 — body proper-noun co-mention.
// ===================================================================

pub fn l4_cap_phrase_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\b((?:[A-Z][a-z]+(?:[\-'][A-Z]?[a-z]+)?)(?:\s+(?:[A-Z][a-z]+(?:[\-'][A-Z]?[a-z]+)?|of|the|de|van|von|den|du|la)){0,3})\b",
        )
        .unwrap()
    })
}

// ===================================================================
// L5 — numerical / typed attribute patterns.
// ===================================================================

pub fn rx_body_founded_in() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b(?:founded|established|created|built|formed|incorporated)\s+in\s+(\d{3,4}(?:\s*BC)?)\b").unwrap()
    })
}

pub fn rx_population() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:population|populace)\s+of\s+(?:over\s+|about\s+|approximately\s+|around\s+)?([\d,]{2,15}(?:\.\d+)?\s*(?:million|thousand|billion)?)\b",
        )
        .unwrap()
    })
}

pub fn rx_area_km2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:area|covers|spans)\s+(?:of\s+)?(?:about\s+|approximately\s+|around\s+)?([\d,]{1,12}(?:\.\d+)?\s*(?:square\s+kilometres?|square\s+kilometers?|km2|km²|sq\s+mi))\b",
        )
        .unwrap()
    })
}

pub fn rx_altitude() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:altitude|elevation|height)\s+of\s+(?:about\s+|approximately\s+)?([\d,]{1,10}(?:\.\d+)?\s*(?:metres?|meters?|m|feet|ft))\b",
        )
        .unwrap()
    })
}

pub fn rx_lang_official() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\bofficial\s+language(?:s)?\s+(?:is|are|of\s+\w+\s+(?:is|are))\s+([A-Z][a-zA-Z\-]{2,30}(?:(?:,\s+|\s+and\s+)[A-Z][a-zA-Z\-]{2,30}){0,3})\b",
        )
        .unwrap()
    })
}

pub fn rx_capital() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\bcapital(?:\s+city)?\s+(?:is|of\s+\w+\s+is)\s+([A-Z][a-zA-Z\-]{2,30}(?:\s+[A-Z][a-zA-Z\-]{2,30}){0,2})\b",
        )
        .unwrap()
    })
}

pub fn rx_born_date() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\b(?:born|b\.)\s+(?:on\s+)?((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{3,4})",
        )
        .unwrap()
    })
}

pub fn rx_died_date() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\b(?:died|d\.)\s+(?:on\s+)?((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{3,4})",
        )
        .unwrap()
    })
}

// ===================================================================
// NP truncator (Phase-1 2026-05-30).
// ===================================================================

// ===================================================================
// OP-EXTRACT-BREADTH B2 — new typed getters (lockstep smart_extractor.py).
// ===================================================================

pub fn rx_geo_in() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:city|town|village|commune|municipality|hamlet|borough|township|region|province|state|country|county|district|department|prefecture|canton|island|archipelago|peninsula|river|lake|mountain|peak|volcano|desert|forest|sea|bay|gulf|valley|plateau|neighbourhood|neighborhood|suburb)\s+(?:in|of|near|on)\s+(the\s+)?([A-Z][a-zA-Z]{1,30}(?:\s+(?:[A-Z][a-zA-Z]{1,30}|of|the|de)){0,3})(?=[.;\n]|,\s+(?:which|who|where|that|and)|$)",
        )
        .unwrap()
    })
}

pub fn rx_subsidiary() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:subsidiary|division|branch|affiliate|unit)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z]{1,30}(?:\s+(?:[A-Z][a-zA-Z]{1,30}|of|the|de|and|&)){0,4})(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_causes() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:causes|caused|cause|leads\s+to|led\s+to|results\s+in|resulted\s+in|triggers|trigger|induces|induce)\s+(?!of\b)(?:a\s+|an\s+|the\s+)?([a-z][a-zA-Z \-]{3,50}?)(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_requires() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:requires|require|requiring|depends\s+on|depend\s+on|needs|necessitates)\s+(?!of\b)(?:a\s+|an\s+|the\s+)?([a-z][a-zA-Z \-]{3,50}?)(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_produces() -> &'static FRegex {
    static R: OnceLock<FRegex> = OnceLock::new();
    R.get_or_init(|| {
        FRegex::new(
            r"\b(?:produces|produce|generates|generate|emits|emit|yields|yield)\s+(?!of\b)(?:a\s+|an\s+|the\s+)?([a-z][a-zA-Z \-]{3,50}?)(?=[.;\n,])",
        )
        .unwrap()
    })
}

pub fn rx_population2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:population\s+(?:was|is|reached|stood\s+at|of\s+(?:about|around|approximately|over|nearly))|home\s+to)\s+([\d,]{2,15}(?:\.\d+)?\s*(?:million|thousand|billion)?)\b",
        )
        .unwrap()
    })
}

pub fn rx_population3() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b([\d,]{3,15}(?:\.\d+)?\s*(?:million|thousand|billion)?)\s+(?:inhabitants|residents)\b",
        )
        .unwrap()
    })
}

pub fn rx_area2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:total\s+area\s+of|has\s+an\s+area\s+of|area\s+of\s+about)\s+([\d,]{1,12}(?:\.\d+)?\s*(?:square\s+kilometres?|square\s+kilometers?|km2|km²|sq\s+mi|hectares?|acres?))\b",
        )
        .unwrap()
    })
}

pub fn rx_capital2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"\b(?:whose\s+capital\s+is|with\s+its\s+capital\s+(?:at|in|being))\s+([A-Z][a-zA-Z\-]{2,30}(?:\s+[A-Z][a-zA-Z\-]{2,30}){0,2})\b",
        )
        .unwrap()
    })
}

pub fn rx_founded_year2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:opened|inaugurated|chartered|constructed|completed|launched)\s+in\s+(\d{3,4}(?:\s*BC)?)\b",
        )
        .unwrap()
    })
}

pub fn rx_person_name() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z\.]+){1,2}$").unwrap())
}

pub fn rx_one_date() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{3,4})",
        )
        .unwrap()
    })
}

pub fn rx_date_range() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"((?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{3,4}))\s*(?:–|—|-|to)\s*((?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{3,4}))",
        )
        .unwrap()
    })
}

pub fn rx_numeric_ok() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)^\d[\d,\.]*\s*(?:BC|million|thousand|billion|square\s+kilometres?|square\s+kilometers?|km2|km²|sq\s+mi|hectares?|acres?|metres?|meters?|m|feet|ft)?\s*$",
        )
        .unwrap()
    })
}

pub fn np_stop_punct_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[,;:—–]").unwrap())
}

pub fn np_stop_conjunction_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\s+(?:because|although|while|that|which|who|where|when|if|as|since|though|so that|in order to)\s+",
        )
        .unwrap()
    })
}

pub fn np_predicate_verb_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:is|was|are|were|has|have|had|will|would|could|should|did|does|do|seems|appears|became|becomes)\b",
        )
        .unwrap()
    })
}

pub fn np_leading_verb_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)^(?:[a-z]+(?:ed|ing))\b").unwrap())
}

// ===================================================================
// LM multi-sentence — Pattern A/B + Path 2 helpers.
// ===================================================================

/// The predicate-verb fragment shared by Pattern A/B/Path-2. Mirrors
/// the Python `_PRED_RE`.
pub const PRED_RE_BODY: &str =
    r"(?:is|was|are|were|be|being|been|am|has|have|had|does|do|did|will|would|should|could|can|may|might|must|means|refers|denotes|represents|includes|contains|involves|consists|comprises|signifies|describes|equals|defines|marks|originates|derives|originated|derived|occurred|happens|happened|exists|existed|came|comes|became|becomes|remains|stayed|stays|stood|stands)";

const DETS_RE_BODY: &str = r"(?:[Tt]he|[Aa]n?|[Tt]his|[Tt]hat|[Tt]hese|[Tt]hose|[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Ii]ts|[Oo]ur|[Tt]heir|[Ss]ome|[Aa]ny|[Ee]ach|[Ee]very|[Aa]ll|[Bb]oth|[Nn]o)";

/// Pattern A — determiner-led NP + predicate. The single capture
/// group is the NP head; `pred` (regex name) is captured separately
/// for typed-relation routing.
pub fn pat_a_resp() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        let pat = format!(
            r"^\s*{dets}\s+([A-Za-z][\w'\-]*(?:\s+(?:[A-Za-z][\w'\-]*)){{0,4}})\s+(?P<pred>{pred})\b",
            dets = DETS_RE_BODY,
            pred = PRED_RE_BODY,
        );
        Regex::new(&pat).unwrap()
    })
}

/// Pattern B — capital-led NP + predicate.
pub fn pat_b_resp() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        let pat = format!(
            r"^\s*([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){{0,2}})\s+(?P<pred>{pred})\b",
            pred = PRED_RE_BODY,
        );
        Regex::new(&pat).unwrap()
    })
}

/// Path 2 (pronoun-led) extractor regex.
pub fn portb_path2_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        let pat = format!(
            r"(?i)^\s*(?:it|this|that|these|those|he|she|they|we|you|i)\s+(?P<pred>{pred})\s+(?P<tail>[^.!?\n]{{3,200}})",
            pred = PRED_RE_BODY,
        );
        Regex::new(&pat).unwrap()
    })
}

pub fn portb_tail_clause_split_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\s+(?:that|which|because|since|while|although|though)\s+").unwrap()
    })
}

pub fn forbidden_subj_head() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "what", "who", "when", "where", "why", "how", "which", "whose",
            "whom", "it", "he", "she", "they", "we", "you", "i", "me",
            "us", "him", "them", "this", "that", "these", "those", "there",
            "here", "however", "therefore", "moreover", "furthermore",
            "indeed", "though", "although", "because", "since", "while",
            "yes", "no", "well", "so", "yet", "still", "anyway",
            "additionally", "also", "finally", "lastly", "first", "second",
            "third", "next", "then", "now", "basically", "essentially",
            "actually", "perhaps", "maybe",
        ]
        .into_iter()
        .collect()
    })
}

pub fn frame_noun_subjects() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "paper", "study", "work", "article", "manuscript", "report",
            "thesis", "dissertation", "letter", "preprint", "draft",
            "method", "approach", "technique", "framework", "model",
            "system", "scheme", "algorithm", "procedure",
        ]
        .into_iter()
        .collect()
    })
}

pub fn portb_pronoun_lead_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)^\s*(?:it|this|that|these|those|he|she|they|we|you|i)\b").unwrap())
}

pub fn portb_interrogative_lead_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)^\s*(?:how|what|why|when|where|who|whose|whom|which)\b").unwrap())
}

pub fn portb_question_end_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\?\s*$").unwrap())
}

pub fn portb_code_token_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)(?:\bdef\s+\w|\bfunction\s+\w|\bimport\s+\w|console\.log|\bclass\s+\w|\basync\s+def|=>|</?[a-z][\w-]*>|<[a-z][\w-]*\s+[^>]*>|#include\b|#define\b|\bnpm\s+install\b|\bpip\s+install\b|\bSELECT\s+\*\s+FROM\b|\bINSERT\s+INTO\b|\bUPDATE\s+\w+\s+SET\b)",
        )
        .unwrap()
    })
}

pub fn portb_citation_marker_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\[\d+(?:[,\s\-\d]*)\]").unwrap())
}

pub fn portb_editorial_frame_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:according to|as noted|see also|as mentioned|as discussed|as described|this paper|this article|this study|this report|this work|this thesis|this manuscript|the present paper|this section|this chapter|i argue|we argue|we propose|i propose|i claim|we claim|i suggest|we suggest)\b",
        )
        .unwrap()
    })
}

pub fn portb_quote_wrapped_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"^\s*["'].+["']\s*[.!?]?\s*$"#).unwrap())
}

pub fn portb_frame_fragment_sentences() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "references", "see also", "this paper", "this article",
            "this study", "this work", "this report", "this section",
            "external links", "further reading", "notes", "footnotes",
            "bibliography", "introduction", "conclusion", "abstract",
            "summary", "background", "methods", "method", "results",
            "discussion", "acknowledgements", "acknowledgments",
        ]
        .into_iter()
        .collect()
    })
}

pub const PORTB_MIN_SENTENCE_WORDS: usize = 4;

pub fn portb_path2_metalinguistic() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        [
            "question", "answer", "case", "issue", "problem", "point",
            "topic", "subject", "fact", "statement", "example", "thing",
            "matter", "claim", "argument", "assumption", "premise",
            "conclusion", "remark", "comment", "observation", "note",
            "hypothesis", "true", "false", "correct", "incorrect",
            "wrong", "right", "obvious", "clear", "important", "necessary",
            "possible", "impossible", "likely", "unlikely", "common",
            "rare", "good", "bad", "easy", "hard", "simple", "complex",
            "complicated", "useful", "useless",
        ]
        .into_iter()
        .collect()
    })
}

pub fn portb_title_interrogative_lead_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)^\s*(?:how|what|why|when|where|who|whose|whom|which|does|do|did|is|are|was|were|can|could|should|would|will|may|might|must|shall|am)\b",
        )
        .unwrap()
    })
}

// L2b tail-section markers (line-anchored, mirrors the Python
// _strip_tail_sections markers but as `\n` strings).
pub const TAIL_SECTION_MARKERS: &[&str] = &[
    "\nSee also\n",
    "\nReferences\n",
    "\nExternal links\n",
    "\nBibliography\n",
    "\nFurther reading\n",
    "\nNotes\n",
    "\nFootnotes\n",
    "\nCitations\n",
];

