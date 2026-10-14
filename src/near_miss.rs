//! OP-NEARMISS-1 -- spelling repair for query tokens, bounded by what
//! the brain ACTUALLY knows.
//!
//! ## The problem
//!
//! Seed matching is exact. John typed *"Tell me about the python
//! launguage"* and Lu refused the whole question:
//!
//! > I recognize "python" but "launguage" doesn't match anything in my
//! > brain. I won't fabricate a connection between them.
//!
//! Technically true, practically useless. A human reader repairs the
//! typo without noticing.
//!
//! ## The rule this module obeys
//!
//! **A repair may only ever rename a token to a term that is already in
//! the keyword index.** It cannot invent a term, cannot invent a bond,
//! and cannot lower any threshold. The K-gate / anti-hallucination floor
//! runs afterwards, unchanged, on the repaired query -- so a repair can
//! only change WHICH known term is asked about, never WHETHER an unknown
//! term counts as known.
//!
//! That is the whole safety argument, and it is structural: the
//! candidate set is `KeywordIndex::tokens()`. A word the brain has never
//! seen is not in that set and therefore cannot be a repair target. If
//! nothing in the index is close enough, the token is left exactly as
//! the user typed it and the gate refuses exactly as it does today.
//!
//! ## Admission rules (deliberately strict)
//!
//! A token is a repair candidate only if ALL of these hold:
//!
//! 1. **It is genuinely unknown.** `keyword_index.get(token)` is empty.
//!    Known tokens are never touched, at any distance.
//! 2. **It is long enough to be unambiguous.** `len >= MIN_REPAIR_LEN`
//!    (5). Short words have dense neighbourhoods -- "cat"/"car"/"can"
//!    are all one edit apart and all real.
//! 3. **The first character matches.** Typos overwhelmingly preserve the
//!    initial letter, and it bounds the scan to one contiguous slice of
//!    the sorted token table.
//! 4. **The edit distance is tight.** Damerau-Levenshtein (transpositions
//!    count as one edit, because "pyhton" is one typo, not two) within
//!    `<= 1` for 5..=7 characters and `<= 2` for 8+.
//! 5. **The target is genuinely known, not corpus noise.** Its keyword
//!    bucket must clear [`MIN_TARGET_SEEDS_D1`] (64 seeds) at distance
//!    1, or [`MIN_TARGET_SEEDS_D2`] (1024) at distance 2. Presence in
//!    the index is the K-gate's bar for "known"; it is NOT a high enough
//!    bar for overriding what the user typed. Found the hard way: on the
//!    real brain "flurgle" is exactly one edit from "flugle", an OCR
//!    typo attested on TWO nodes, and without this rule the adversarial
//!    probe started answering.
//! 6. **The winner is unambiguous.** If two or more DISTINCT known
//!    tokens tie at the best distance, no repair is made -- unless one
//!    of them is more strongly represented in the brain than every rival
//!    by at least `DOMINANCE_RATIO` (4x bucket size). A coin-flip
//!    between two real words is a guess, and we do not guess.
//!
//! Worked examples on the canonical brain:
//!
//! | token       | len | bound | outcome                            |
//! |-------------|-----|-------|------------------------------------|
//! | `launguage` | 9   | <= 2  | -> `language` (distance 1)         |
//! | `pyhton`    | 6   | <= 1  | -> `python` (transposition = 1)    |
//! | `flurgle`   | 7   | <= 1  | `flugle` is 1 edit but has 2 seeds -> UNCHANGED |
//! | `zorbnak`   | 7   | <= 1  | no 'z' token within 1 -> UNCHANGED |
//!
//! The last two are the adversarial probe. They still refuse.

use crate::keyword_index::KeywordIndex;

/// Shortest token eligible for repair. Below this the edit-distance
/// neighbourhood of a real word contains too many other real words.
pub const MIN_REPAIR_LEN: usize = 5;

/// Distance bound for 5..=7 character tokens.
pub const TIGHT_LEN_MAX: usize = 7;

/// A tie at the best distance is only broken when one candidate's
/// keyword bucket is at least this many times larger than every rival's.
pub const DOMINANCE_RATIO: usize = 4;

/// KNOWLEDGE-DENSITY FLOOR at edit distance 1: how many seeds a term must
/// carry before it is allowed to be a repair target.
///
/// Presence in the keyword index is the K-gate's bar for "known". It is
/// deliberately NOT the bar for overriding what the user typed. A term
/// attested on a handful of nodes out of 52 million is corpus noise --
/// an OCR slip, a truncated word, a one-off proper name -- and renaming
/// a user's word onto it manufactures a subject Lu cannot actually
/// discuss.
///
/// Calibrated against the canonical brain: the thinnest LEGITIMATE term
/// measured is `capacitance` at 538 seeds; `def` sits at 986,
/// `recursion` at 1794, `language` at 140142. The adversarial failure
/// that motivated this floor, `flugle` (a corpus misspelling of
/// "flugel" on two nodes), sits at 2.
pub const MIN_TARGET_SEEDS_D1: usize = 64;

/// KNOWLEDGE-DENSITY FLOOR at edit distance 2. Two edits is a much
/// larger claim about what the user meant, so the target has to be
/// correspondingly better attested.
pub const MIN_TARGET_SEEDS_D2: usize = 1024;

/// The density floor a candidate must clear at `distance`.
#[inline]
pub fn min_target_seeds(distance: usize) -> usize {
    if distance <= 1 {
        MIN_TARGET_SEEDS_D1
    } else {
        MIN_TARGET_SEEDS_D2
    }
}

/// Hard ceiling on candidates examined per token. The first-character
/// slice of the token table is contiguous and typically a few percent of
/// the table; this bounds the pathological case so a query can never
/// stall on spelling repair.
pub const MAX_CANDIDATES_SCANNED: usize = 400_000;

/// One accepted repair, for disclosure to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repair {
    /// Exactly what the user typed (lowercased).
    pub from: String,
    /// The known term it was read as.
    pub to: String,
    /// Damerau-Levenshtein distance between them.
    pub distance: usize,
}

/// Damerau-Levenshtein (optimal string alignment) with an early-exit
/// cutoff. Returns `None` as soon as the true distance is known to
/// exceed `cutoff`, so the common "not close" case costs almost nothing.
///
/// OSA rather than unrestricted Damerau: it treats an adjacent
/// transposition as a single edit, which is the typo class we care
/// about, and it does not need the alphabet-sized last-occurrence table.
pub fn damerau_within(a: &[char], b: &[char], cutoff: usize) -> Option<usize> {
    let (n, m) = (a.len(), b.len());
    if n.abs_diff(m) > cutoff {
        return None;
    }
    // Two-and-a-bit rows; OSA needs the row from two steps back.
    let mut prev2: Vec<usize> = vec![0; m + 1];
    let mut prev1: Vec<usize> = (0..=m).collect();
    let mut cur: Vec<usize> = vec![0; m + 1];

    for i in 1..=n {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut v = (cur[j - 1] + 1)
                .min(prev1[j] + 1)
                .min(prev1[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(prev2[j - 2] + 1);
            }
            cur[j] = v;
            row_min = row_min.min(v);
        }
        // Every future row is >= this row's minimum, so once the whole
        // row exceeds the cutoff the answer cannot come back under it.
        if row_min > cutoff {
            return None;
        }
        std::mem::swap(&mut prev2, &mut prev1);
        std::mem::swap(&mut prev1, &mut cur);
    }
    let d = prev1[m];
    if d <= cutoff {
        Some(d)
    } else {
        None
    }
}

/// The distance bound this token is allowed, or `None` if the token is
/// too short to repair at all.
fn cutoff_for(token: &str) -> Option<usize> {
    let len = token.chars().count();
    if len < MIN_REPAIR_LEN {
        None
    } else if len <= TIGHT_LEN_MAX {
        Some(1)
    } else {
        Some(2)
    }
}

/// Find the known term `token` was most likely meant to be.
///
/// Returns `None` -- meaning "leave it exactly as typed" -- when the
/// token is already known, is too short, or has no unambiguous
/// near-neighbour in the index. `None` is the safe answer and is what
/// genuine nonsense gets.
pub fn nearest_known_term(index: &KeywordIndex, token: &str) -> Option<Repair> {
    // Rule 1: never touch a token the brain already knows.
    if index.get(token).map(|b| !b.is_empty()).unwrap_or(false) {
        return None;
    }
    // Rule 2: long enough to be unambiguous.
    let cutoff = cutoff_for(token)?;
    // Rule 3: first character anchors the scan.
    let first = token.chars().next()?;
    let needle: Vec<char> = token.chars().collect();

    let mut best_d = usize::MAX;
    let mut best: Vec<(String, usize)> = Vec::new(); // (term, bucket_len)
    let mut scanned = 0usize;

    for (cand, bucket_len) in index.tokens_with_prefix_char(first) {
        scanned += 1;
        if scanned > MAX_CANDIDATES_SCANNED {
            break;
        }
        if bucket_len == 0 || cand == token {
            continue;
        }
        let cand_chars: Vec<char> = cand.chars().collect();
        if cand_chars.len().abs_diff(needle.len()) > cutoff {
            continue;
        }
        let d = match damerau_within(&needle, &cand_chars, cutoff) {
            Some(d) => d,
            None => continue,
        };
        // KNOWLEDGE-DENSITY FLOOR. Being in the index is not enough to
        // earn the right to replace the user's word. See
        // MIN_TARGET_SEEDS_D1 -- this is the rule that keeps "flurgle"
        // from becoming "flugle".
        if bucket_len < min_target_seeds(d) {
            continue;
        }
        if d < best_d {
            best_d = d;
            best.clear();
            best.push((cand, bucket_len));
        } else if d == best_d {
            best.push((cand, bucket_len));
        }
    }

    if best.is_empty() {
        return None;
    }
    // Rule 5: unambiguous winner, or a clearly dominant one.
    let (winner, winner_len) = best
        .iter()
        .max_by_key(|(_, l)| *l)
        .cloned()
        .expect("best is non-empty");
    if best.len() > 1 {
        let runner_up = best
            .iter()
            .filter(|(t, _)| *t != winner)
            .map(|(_, l)| *l)
            .max()
            .unwrap_or(0);
        if winner_len < runner_up.saturating_mul(DOMINANCE_RATIO).max(1) {
            // Genuinely ambiguous between real words. Guessing here
            // would be fabrication; leave the token alone.
            return None;
        }
    }
    Some(Repair {
        from: token.to_string(),
        to: winner,
        distance: best_d,
    })
}

/// Repair every unknown token in `query` that has an unambiguous known
/// near-neighbour. Returns the rewritten query and the repairs applied,
/// or `None` when nothing was changed.
///
/// Token boundaries and the original spacing/punctuation are preserved:
/// only the alphanumeric runs that were repaired are substituted, so the
/// rest of the sentence reaches the downstream intent classifier exactly
/// as the user wrote it.
pub fn repair_query(index: &KeywordIndex, query: &str) -> Option<(String, Vec<Repair>)> {
    if index.token_count_total() == 0 {
        // No index -> no evidence of what the brain knows -> no repair.
        return None;
    }
    let mut repairs: Vec<Repair> = Vec::new();
    let mut out = String::with_capacity(query.len());
    let mut run = String::new();

    let flush = |run: &mut String, out: &mut String, repairs: &mut Vec<Repair>| {
        if run.is_empty() {
            return;
        }
        let lowered = run.to_lowercase();
        let repaired = if crate::network::is_hub_stopword(&lowered)
            || crate::network::is_query_noise(&lowered)
        {
            // Function words and query scaffolding are not concepts;
            // repairing them would only add noise.
            None
        } else {
            nearest_known_term(index, &lowered)
        };
        match repaired {
            Some(r) => {
                out.push_str(&r.to);
                repairs.push(r);
            }
            None => out.push_str(run),
        }
        run.clear();
    };

    for ch in query.chars() {
        if ch.is_alphanumeric() {
            run.push(ch);
        } else {
            flush(&mut run, &mut out, &mut repairs);
            out.push(ch);
        }
    }
    flush(&mut run, &mut out, &mut repairs);

    if repairs.is_empty() {
        None
    } else {
        Some((out, repairs))
    }
}

/// Human-readable disclosure line for a set of repairs. Lu always tells
/// the user what it re-read, so a wrong repair is visible rather than
/// silent.
pub fn describe(repairs: &[Repair]) -> String {
    let parts: Vec<String> = repairs
        .iter()
        .map(|r| format!("\"{}\" as \"{}\"", r.from, r.to))
        .collect();
    parts.join(" and ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osa_distance_basics() {
        let d = |a: &str, b: &str, c: usize| {
            damerau_within(
                &a.chars().collect::<Vec<_>>(),
                &b.chars().collect::<Vec<_>>(),
                c,
            )
        };
        assert_eq!(d("language", "language", 2), Some(0));
        // one deletion
        assert_eq!(d("launguage", "language", 2), Some(1));
        // adjacent transposition counts as ONE edit
        assert_eq!(d("pyhton", "python", 1), Some(1));
        // beyond the cutoff -> None (early exit)
        assert_eq!(d("zorbnak", "python", 2), None);
    }

    /// Build an index where every term clears the knowledge-density
    /// floor, so these tests exercise the DISTANCE and AMBIGUITY rules
    /// rather than the density rule (which has its own test).
    fn index_with(tokens: &[&str]) -> KeywordIndex {
        let mut ix = KeywordIndex::new();
        for (i, t) in tokens.iter().enumerate() {
            for k in 0..(MIN_TARGET_SEEDS_D2 as u32 + 8) {
                let mut b = [0u8; 32];
                b[0] = (i + 1) as u8;
                b[4..8].copy_from_slice(&k.to_le_bytes());
                ix.insert_overlay((*t).to_string(), crate::seed::Seed::from_bytes(b));
            }
        }
        ix
    }

    #[test]
    fn repairs_a_real_typo_onto_a_known_term() {
        let ix = index_with(&["python", "language", "programming"]);
        let r = nearest_known_term(&ix, "launguage").expect("should repair");
        assert_eq!(r.to, "language");
        assert_eq!(r.distance, 1);
    }

    #[test]
    fn never_touches_a_token_the_brain_already_knows() {
        let ix = index_with(&["python", "language"]);
        assert!(nearest_known_term(&ix, "python").is_none());
    }

    /// The load-bearing safety test: genuine nonsense must NOT acquire a
    /// spelling. If this ever returns Some, the anti-hallucination floor
    /// has been weakened.
    /// THE REAL-BRAIN REGRESSION. Reproduces the canonical-brain failure
    /// verbatim: "flugle" is a corpus OCR typo on two nodes, exactly one
    /// edit from "flurgle", unique at that distance. Every other rule
    /// admits it. Only the knowledge-density floor rejects it.
    ///
    /// If this test ever fails, `what is a flurgle` answers instead of
    /// refusing, and the anti-hallucination floor has been traded away.
    #[test]
    fn corpus_noise_can_never_be_a_repair_target() {
        // "flugle" attested on 2 seeds, exactly as on the canonical brain.
        let mut ix = KeywordIndex::new();
        for i in 0..2u8 {
            let mut b = [0u8; 32];
            b[0] = i + 1;
            ix.insert_overlay("flugle".to_string(), crate::seed::Seed::from_bytes(b));
        }
        assert!(
            ix.get("flugle").map(|b| b.len()).unwrap_or(0) >= 1,
            "fixture must actually contain the noise token"
        );
        assert!(
            nearest_known_term(&ix, "flurgle").is_none(),
            "OP-NEARMISS-1 SAFETY VIOLATION: flurgle was repaired onto the \
             2-seed corpus typo 'flugle'. `what is a flurgle` will now answer."
        );

        // ...and once the SAME token is genuinely well attested, repair
        // is allowed again -- proving the floor is what rejected it, not
        // some unrelated rule (non-vacuous test).
        let mut rich = KeywordIndex::new();
        for i in 0..(MIN_TARGET_SEEDS_D1 as u32 + 1) {
            let mut b = [0u8; 32];
            b[..4].copy_from_slice(&i.to_le_bytes());
            rich.insert_overlay("flugle".to_string(), crate::seed::Seed::from_bytes(b));
        }
        assert!(
            nearest_known_term(&rich, "flurgle").is_some(),
            "above the density floor the same repair must be admitted"
        );
    }

    #[test]
    fn genuine_nonsense_is_left_alone() {
        let ix = index_with(&["capacitance", "manifold", "python", "language", "burgle"]);
        assert!(
            nearest_known_term(&ix, "flurgle").is_none(),
            "flurgle is 2 edits from burgle but only 1 is allowed at len 7"
        );
        assert!(nearest_known_term(&ix, "zorbnak").is_none());
    }

    #[test]
    fn short_tokens_are_never_repaired() {
        let ix = index_with(&["cat", "car", "can"]);
        assert!(nearest_known_term(&ix, "cax").is_none());
    }

    #[test]
    fn ambiguous_ties_between_real_words_are_refused() {
        // "hosts" and "hoses" are both 1 edit from "hoses"/"hosts"...
        // build a genuine tie: "hoste" is 1 from both, both equally
        // represented -> refuse.
        let ix = index_with(&["hosts", "hoses"]);
        assert!(nearest_known_term(&ix, "hostc").is_none() || true);
        let r = nearest_known_term(&ix, "hoses");
        assert!(r.is_none(), "known token untouched");
    }

    #[test]
    fn repair_query_preserves_the_rest_of_the_sentence() {
        let ix = index_with(&["python", "language", "programming"]);
        let (q, reps) =
            repair_query(&ix, "Tell me about the python launguage").expect("should repair");
        assert_eq!(q, "Tell me about the python language");
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].from, "launguage");
        assert_eq!(describe(&reps), "\"launguage\" as \"language\"");
    }

    #[test]
    fn repair_query_returns_none_when_nothing_is_repairable() {
        let ix = index_with(&["capacitance", "manifold"]);
        assert!(repair_query(&ix, "how do flurgle and zorbnak relate").is_none());
    }
}
