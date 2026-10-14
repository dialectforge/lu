//! OP-CONCLUDE -- conclusion synthesizer (design: CONCLUSION_SYNTHESIZER_DESIGN.md).
//!
//! Turns typed edges Lu ALREADY asserts into a stated conclusion, or
//! returns `None` so the pipeline degrades to today's behaviour (list /
//! definition / honest refusal). It never invents an edge: every
//! conclusion cites the exact bonds it walked (`cited_edges`), and when
//! the required edges are absent it concludes nothing.
//!
//! Placement: `narration::narrate_inner` calls [`try_synthesize`] on the
//! PASS path, AFTER the idk_gate entry floor and the reason-loop exit
//! floor -- so the honesty floor is untouched and synthesis can only ADD
//! grounding. A misclassified question shape can only cause a degrade
//! (return `None`), never a fabrication.
//!
//! Graph reads go through the SST-aware seam helpers
//! (`AssociativeNetwork::typed_edges_out` / `node_label` /
//! `confidence_from_store`), so the synthesizer sees the same edges the
//! live reader spreads over -- not the empty heap-resident view.

use std::collections::HashSet;

use crate::bond::BondType;
use crate::brain::Brain;
use crate::narration::QueryIntent;
use crate::node::Confidence;
use crate::reason_loop::ContextSet;
use crate::seed::Seed;

/// The recognised shape of a reasoning question. Tier 1 wires the
/// property-inheritance / syllogism shape; later tiers append variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningKind {
    /// "is X P?" / "does X have P?" / "X is a A; A are P; is X P?" --
    /// property held directly or inherited across one IS-A hop.
    PropertyInheritance,
    /// "is a X a Y?" -- category membership (and its heuristic
    /// negative) over the IS-A hierarchy (Tier 1.2).
    CategoryMembership,
    /// "what do X and Y have in common?" -- lowest common IS-A ancestor.
    CommonAncestor,
    /// "is X P?" / "does X have P?" / "can X <verb>?" -- property asserted
    /// DIRECTLY on the concept (X -HasProperty-> P). Tier 1.6.
    PropertyCheck,
    /// "what causes X?" / "what does X cause?" / "does X cause Y?" -- causal
    /// link over Causes/LeadsTo edges. Tier 1.3.
    CausalWhy,
    /// "who is X" / "where is X" / "capital of Y" / "X known for" --
    /// Wikidata entity facts read from the segregated entity namespace
    /// (OP-BREADTH Phase-1).
    EntityFact,
    /// "what is X made of" -- ComposedOf/DerivedFrom out + PartOf in. Tier 2.
    Composition,
    /// "what is X used for" -- UsedFor/PurposeOf out. Tier 2.
    Purpose,
    /// "parts of X" (PartOf in) / "what is X part of" (PartOf out). Tier 2.
    Meronymy,
    /// "relationship between X and Y" -- typed path / shared edge. Tier 2.
    RelationalPath,
    /// "is X a better Y than Z" / "compare X and Z" -- grounded facts,
    /// no subjective verdict (integrity shape). Tier 2.
    Comparative,
}

/// One typed edge the conclusion rests on. The audit trail (design §7):
/// a reader can check the conclusion down to the exact bonds.
#[derive(Debug, Clone)]
pub struct CitedEdge {
    pub from: String,
    pub bond: BondType,
    pub to: String,
    pub confidence: Confidence,
}

impl CitedEdge {
    /// Compact arrow form for the provenance clause / telemetry.
    pub fn arrow(&self) -> String {
        format!("{} -{:?}-> {}", self.from, self.bond, self.to)
    }
}

/// A synthesized conclusion: the stated prose, the edges it cites, an
/// aggregate confidence (weakest cited link), and the shape that produced
/// it. Consumed by the `ResponseShape::Conclusion` render arm.
#[derive(Debug, Clone)]
pub struct Conclusion {
    pub kind: ReasoningKind,
    pub prose: String,
    pub cited_edges: Vec<CitedEdge>,
    pub confidence: f32,
}

// ---------------------------------------------------------------------
// Bond-class predicates (real vocabulary, design §1 / §4)
// ---------------------------------------------------------------------

/// IS-A-class edges usable for one inheritance hop.
fn is_isa_bond(bt: BondType) -> bool {
    matches!(
        bt,
        BondType::IsA | BondType::DefinedAs | BondType::Inherits | BondType::ExampleOf
    )
}

/// Property-class edges (a node ascribing an attribute).
fn is_property_bond(bt: BondType) -> bool {
    matches!(
        bt,
        BondType::HasProperty | BondType::HasAspect | BondType::HasAttribute
    )
}

// ---------------------------------------------------------------------
// Tokenisation helpers (surface-only; same filters recall uses)
// ---------------------------------------------------------------------

/// Content tokens of a string: lowercased alphanumerics, hub-stopwords
/// and query-noise dropped, length >= 2. Same filters the seed selector
/// and idk gate use, so "is", "the", "what" never count as operands.
fn content_tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            w.len() >= 2
                && !crate::network::is_hub_stopword(w)
                && !crate::network::is_query_noise(w)
        })
        .map(|w| w.to_string())
        .collect()
}

/// Content-token SET of a node label, for intersection tests.
fn label_words(label: &str) -> HashSet<String> {
    content_tokens(label).into_iter().collect()
}

/// Does a node label share a content word with the target token set?
fn label_hits(label: &str, targets: &HashSet<String>) -> bool {
    label_words(label).iter().any(|w| targets.contains(w))
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}

/// Node confidence from the store (#211 accessor); `Verified` when the
/// store cannot resolve it (e.g. a heap-only test brain).
fn node_conf(brain: &Brain, seed: &Seed) -> Confidence {
    brain
        .network
        .confidence_from_store(seed)
        .unwrap_or(Confidence::Verified)
}

/// OP-WRITEBACK Phase-5 (Phase-3 payoff): per-EDGE read-time confidence.
/// Reads the confidence of the actual cited BOND from its keyframe
/// provenance (immune-aware: corroborated->Verified, concluded->Inferred),
/// falling back to the node tier when the bond carries no tier-fixing
/// provenance -- so existing (Baseline) knowledge reads exactly as before,
/// no SST rewrite, no fabrication.
fn edge_conf(brain: &Brain, from: &Seed, to: &Seed) -> Confidence {
    crate::writeback::edge_confidence(brain, from, to, node_conf(brain, to))
}

/// A chain is only as strong as its weakest cited edge (design §6/§7).
fn weakest(edges: &[CitedEdge]) -> f32 {
    edges
        .iter()
        .map(|e| e.confidence.value())
        .fold(1.0_f32, f32::min)
}

// ---------------------------------------------------------------------
// Question-shape recognition (design §3) -- pattern-match the QUESTION
// ---------------------------------------------------------------------

// (classify_reasoning replaced by is_yes_no_form / is_common_ancestor_form in Tier 1.2)

// ---------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------

/// Attempt to synthesize a conclusion from the enriched ContextSet's
/// typed edges. Returns `None` (degrade to today's pipeline) whenever
/// the question is not a recognised reasoning shape, the operands cannot
/// be resolved, or the supporting edges are absent.
pub fn try_synthesize(
    brain: &Brain,
    query: &str,
    intent: QueryIntent,
    ctx: Option<&ContextSet>,
) -> Option<Conclusion> {
    let _ = intent;
    let q = query.trim().trim_end_matches('?').trim().to_lowercase();
    // Tier 1.2 -- "what do X and Y have in common" (IS-A lowest common ancestor).
    if is_common_ancestor_form(&q) {
        if let Some(c) = common_ancestor(brain, &q) {
            return Some(c);
        }
    }
    // Tier 1.3 -- causal-why over Causes/LeadsTo. Runs before category /
    // property so "does X cause Y" gets causal framing + an honest no-link
    // refusal. Self-gating: returns None unless the query is a causal form.
    if let Some(c) = causal_why(brain, &q) {
        return Some(c);
    }
    // Tier 2 -- composition / purpose / meronymy / relational-path /
    // comparative. Each self-gates on its question form; once the form is
    // recognised it answers TERMINALLY (cited OR honest refusal), so a
    // recognised Tier-2 question can never fall through to a mis-firing
    // category/property shape (e.g. "is a dog a better pet than a fish").
    // OP-COMPARE -- honest entity-vs-entity comparison. Runs before the
    // concept `comparative` so an entity pair gets grounded typed-edge
    // output; a concept pair fails entity resolution here and falls
    // through to `comparative` unchanged.
    if let Some(c) = entity_comparison(brain, &q) {
        return Some(c);
    }
    if let Some(c) = comparative(brain, &q) {
        return Some(c);
    }
    if let Some(c) = composition(brain, &q) {
        return Some(c);
    }
    if let Some(c) = purpose(brain, &q) {
        return Some(c);
    }
    if let Some(c) = meronym_holonym(brain, &q) {
        return Some(c);
    }
    if let Some(c) = relational_path(brain, &q) {
        return Some(c);
    }
    if is_yes_no_form(&q) {
        // Tier 1.2 -- category membership over the IS-A hierarchy ("is a X
        // a Y"). Tried first: it matches the clean IS-A backbone.
        if let Some(c) = category_membership(brain, &q) {
            return Some(c);
        }
    }
    // Tier 1.6 -- property-check: a property held DIRECTLY on the concept
    // (X -HasProperty-> P). Operands resolve to their node like Tier 1.2, so
    // it fires under weak multi-token recall; ctx-free. Covers yes/no,
    // capability ("can X <verb>"), and list ("what is X like") forms.
    if is_property_query_form(&q) || is_property_list_form(&q) {
        if let Some(c) = property_check(brain, &q) {
            return Some(c);
        }
    }
    // Tier 1.1 -- inherited property across one IS-A hop (needs ctx:
    // X IsA A, A HasProperty P). The direct-property case is handled above.
    if is_yes_no_form(&q) {
        if let Some(cs) = ctx {
            if let Some(c) = property_inheritance(brain, query, cs) {
                return Some(c);
            }
        }
    }
    // OP-BREADTH Phase-1 -- entity subsystem (separate shelf). Runs LAST so
    // every concept shape keeps priority; fires only on entity-intent forms.
    if let Some(c) = entity_query(brain, &q) {
        return Some(c);
    }
    None
}

// ---------------------------------------------------------------------
// Shape 1.1 -- syllogism / IS-A + property inheritance
// ---------------------------------------------------------------------

// Bound the walk so a hub subject / hypernym cannot blow up latency.
const MAX_SUBJECT_CANDIDATES: usize = 8;
const MAX_ISA_MIDS: usize = 32;
const MAX_PROPERTY_EDGES: usize = 256;

/// "is X P?" -- conclude when the graph asserts the property directly
/// (`X HasProperty P`) or across one IS-A hop (`X IsA A`, `A HasProperty
/// P`). Role-agnostic: every ContextSet node whose label appears in the
/// query is tried as the subject, and any query content word NOT naming
/// that subject is the candidate predicate. The rule only fires on a
/// real edge to a real query token, so it cannot fabricate.
fn property_inheritance(
    brain: &Brain,
    query: &str,
    ctx: &ContextSet,
) -> Option<Conclusion> {
    let toks = content_tokens(query);
    if toks.len() < 2 {
        return None;
    }
    let tokset: HashSet<String> = toks.iter().cloned().collect();

    // Subject candidates: ContextSet nodes named in the query, strongest
    // activation first.
    let mut cands: Vec<(Seed, String, HashSet<String>, f32)> = Vec::new();
    for (seed, entry) in ctx.iter() {
        let matched: HashSet<String> = label_words(&entry.label)
            .intersection(&tokset)
            .cloned()
            .collect();
        if !matched.is_empty() {
            cands.push((*seed, entry.label.clone(), matched, entry.activation));
        }
    }
    cands.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(MAX_SUBJECT_CANDIDATES);

    for (sseed, slabel, smatch, _act) in &cands {
        // Predicate tokens = query content words that do NOT name the subject.
        let pred: HashSet<String> = tokset.difference(smatch).cloned().collect();
        if pred.is_empty() {
            continue;
        }

        let out_edges = brain.network.typed_edges_out(sseed);

        // (a) Direct: X HasProperty P.
        for (nbr, bt) in out_edges.iter().take(MAX_PROPERTY_EDGES) {
            if !is_property_bond(*bt) {
                continue;
            }
            let nlab = brain.network.node_label(nbr).unwrap_or_default();
            if nlab.is_empty() || !label_hits(&nlab, &pred) {
                continue;
            }
            let cited = vec![CitedEdge {
                from: slabel.clone(),
                bond: *bt,
                to: nlab.clone(),
                confidence: edge_conf(brain, sseed, nbr),
            }];
            let prose = format!(
                "Yes -- {} is {}. I have that directly: {} {} {}. (Chain: {}.)",
                capitalize_first(slabel),
                nlab,
                slabel,
                bond_phrase(*bt),
                nlab,
                cited[0].arrow(),
            );
            let confidence = weakest(&cited);
            return Some(Conclusion {
                kind: ReasoningKind::PropertyInheritance,
                prose,
                cited_edges: cited,
                confidence,
            });
        }

        // (b) Inherited: X IsA A, A HasProperty P.
        for (mid, bt1) in out_edges.iter().take(MAX_ISA_MIDS) {
            if !is_isa_bond(*bt1) {
                continue;
            }
            let mid_label = brain.network.node_label(mid).unwrap_or_default();
            if mid_label.is_empty() {
                continue;
            }
            for (nbr, bt2) in brain.network.typed_edges_out(mid).iter().take(MAX_PROPERTY_EDGES) {
                if !is_property_bond(*bt2) {
                    continue;
                }
                let nlab = brain.network.node_label(nbr).unwrap_or_default();
                if nlab.is_empty() || !label_hits(&nlab, &pred) {
                    continue;
                }
                let cited = vec![
                    CitedEdge {
                        from: slabel.clone(),
                        bond: *bt1,
                        to: mid_label.clone(),
                        confidence: edge_conf(brain, sseed, mid),
                    },
                    CitedEdge {
                        from: mid_label.clone(),
                        bond: *bt2,
                        to: nlab.clone(),
                        confidence: edge_conf(brain, mid, nbr),
                    },
                ];
                let prose = format!(
                    "Yes -- {} is {}: {} is a {}, and {} {} {}. (Chain: {}; {}.)",
                    capitalize_first(slabel),
                    nlab,
                    slabel,
                    mid_label,
                    mid_label,
                    bond_phrase(*bt2),
                    nlab,
                    cited[0].arrow(),
                    cited[1].arrow(),
                );
                let confidence = weakest(&cited);
                return Some(Conclusion {
                    kind: ReasoningKind::PropertyInheritance,
                    prose,
                    cited_edges: cited,
                    confidence,
                });
            }
        }
    }
    None
}

/// Verb phrase for a property-class bond inside conclusion prose.
fn bond_phrase(bt: BondType) -> &'static str {
    match bt {
        BondType::HasProperty => "is",
        BondType::HasAspect => "has aspect",
        BondType::HasAttribute => "has attribute",
        _ => "is",
    }
}

// ---------------------------------------------------------------------
// Tier 1.2 -- IS-A category reasoning (consumes the clean WordNet/GCIDE
// IS-A backbone). Operands are resolved DIRECTLY to their concept nodes
// (design 3 seeding fix), so these shapes fire even when multi-token
// recall sprayed onto noise.
// ---------------------------------------------------------------------

const ISA_MAX_DEPTH: usize = 16;
const ISA_MAX_NODES: usize = 4000;

/// Is this a yes/no question ("is X ...", "does X ...")? Permissive; the
/// downstream rule is self-gating.
fn is_yes_no_form(q: &str) -> bool {
    q.starts_with("is ") || q.starts_with("are ")
        || q.starts_with("does ") || q.starts_with("do ")
        || q.contains("; is ") || q.contains(", is ")
        || (q.contains("all ") && q.contains(" are "))
}

/// Is this a "what do X and Y have in common" question?
fn is_common_ancestor_form(q: &str) -> bool {
    q.contains("have in common") || q.contains("in common")
        || q.contains("common between")
        || (q.contains("what do") && q.contains(" share"))
}

/// Concept operands named after an article ("a dog", "an animal", "the
/// cat"). This is the surface signal for a category-membership question
/// and resolves the two operands DIRECTLY, so the synthesizer seeds on
/// the NAMED concepts instead of a spray across every token (design 3
/// recall-seeding fix).
fn article_operands(q: &str) -> Vec<String> {
    let toks: Vec<&str> = q.split_whitespace().collect();
    let mut ops = Vec::new();
    for i in 0..toks.len().saturating_sub(1) {
        if matches!(toks[i], "a" | "an" | "the") {
            let w: String = toks[i + 1]
                .chars().filter(|c| c.is_alphanumeric()).collect::<String>()
                .to_lowercase();
            if w.len() >= 2 && !crate::network::is_hub_stopword(&w) {
                ops.push(w);
            }
        }
    }
    ops
}

/// Strip a WordNet sense tag: "canine.n.2" -> "canine"; leaves an
/// untagged label unchanged. Node IDENTITY stays the seed; this only
/// affects the human-facing label (query match + cited-chain display).
fn strip_sense(s: &str) -> &str {
    // OP-BREADTH: entity display -- "Paris#Q90" -> "Paris". Node identity
    // stays the seed; this only affects the human-facing label.
    if let Some(h) = s.rfind('#') {
        let tail = &s[h + 1..];
        if tail.len() > 1
            && (tail.as_bytes()[0] == b'Q' || tail.as_bytes()[0] == b'q')
            && tail[1..].bytes().all(|b| b.is_ascii_digit())
        {
            return &s[..h];
        }
    }
    if let Some(p1) = s.rfind('.') {
        let tail = &s[p1 + 1..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            let head = &s[..p1];
            if let Some(p2) = head.rfind('.') {
                let mid = &head[p2 + 1..];
                if matches!(mid, "n" | "v" | "a" | "r") {
                    return &s[..p2];
                }
            }
        }
    }
    s
}

/// Resolve a single-word operand to its concept node. Exact (case-fold)
/// label match via the keyword index first (the live serving path), then
/// a from_label fallback (heap/test brains). None when no node carries
/// that exact label -- never a fuzzy guess.
fn resolve_concept(brain: &Brain, token: &str) -> Option<Seed> {
    let t: String = token.trim().to_lowercase();
    if t.len() < 2 {
        return None;
    }
    if let Some(bucket) = brain.network.keyword_index_bucket(&t) {
        let mut stripped_hit: Option<Seed> = None;
        for seed in bucket.iter() {
            if let Some(l) = brain.network.node_label(seed) {
                let lt = l.trim();
                if lt.eq_ignore_ascii_case(&t) {
                    return Some(*seed); // bare primary sense -- preferred
                }
                if stripped_hit.is_none() && strip_sense(lt).eq_ignore_ascii_case(&t) {
                    stripped_hit = Some(*seed);
                }
            }
        }
        if let Some(sd) = stripped_hit {
            return Some(sd);
        }
    }
    let s = Seed::from_label(&t);
    if let Some(l) = brain.network.node_label(&s) {
        if !l.is_empty() {
            return Some(s);
        }
    }
    None
}

/// BFS the IS-A hierarchy upward from `start`. Returns nearest-first
/// (node, depth); a came_from map (node -> (child_arrived_from, bond))
/// for path reconstruction; and a label cache. Bounded by depth + node
/// caps. `start` is included at depth 0.
fn isa_closure(
    brain: &Brain,
    start: Seed,
) -> (
    Vec<(Seed, usize)>,
    std::collections::HashMap<Seed, (Seed, BondType)>,
    std::collections::HashMap<Seed, String>,
) {
    use std::collections::{HashMap, VecDeque};
    let mut order: Vec<(Seed, usize)> = Vec::new();
    let mut depth: HashMap<Seed, usize> = HashMap::new();
    let mut came_from: HashMap<Seed, (Seed, BondType)> = HashMap::new();
    let mut labels: HashMap<Seed, String> = HashMap::new();
    let mut q: VecDeque<Seed> = VecDeque::new();
    depth.insert(start, 0);
    labels.insert(start, brain.network.node_label(&start).unwrap_or_default());
    q.push_back(start);
    while let Some(n) = q.pop_front() {
        let d = depth[&n];
        order.push((n, d));
        if order.len() >= ISA_MAX_NODES || d >= ISA_MAX_DEPTH {
            continue;
        }
        for (nbr, bt) in brain.network.typed_edges_out(&n) {
            if !is_isa_bond(bt) {
                continue;
            }
            if depth.contains_key(&nbr) {
                continue;
            }
            depth.insert(nbr, d + 1);
            came_from.insert(nbr, (n, bt));
            labels.insert(nbr, brain.network.node_label(&nbr).unwrap_or_default());
            q.push_back(nbr);
        }
    }
    (order, came_from, labels)
}

/// Reconstruct the cited IS-A chain from `start` up to `target`.
fn isa_path_edges(
    brain: &Brain,
    start: Seed,
    target: Seed,
    came_from: &std::collections::HashMap<Seed, (Seed, BondType)>,
    labels: &std::collections::HashMap<Seed, String>,
) -> Vec<CitedEdge> {
    let mut chain: Vec<(Seed, BondType, Seed)> = Vec::new();
    let mut cur = target;
    let mut guard = 0;
    while cur != start && guard < ISA_MAX_DEPTH + 2 {
        guard += 1;
        if let Some(&(child, bt)) = came_from.get(&cur) {
            chain.push((child, bt, cur));
            cur = child;
        } else {
            break;
        }
    }
    chain.reverse();
    chain
        .into_iter()
        .map(|(c, bt, p)| CitedEdge {
            from: strip_sense(&labels.get(&c).cloned().unwrap_or_default()).to_string(),
            bond: bt,
            to: strip_sense(&labels.get(&p).cloned().unwrap_or_default()).to_string(),
            confidence: edge_conf(brain, &c, &p),
        })
        .collect()
}

/// All candidate concept nodes for a query word: the bare primary-sense
/// node plus any sense-tagged nodes (design 3 -- seed MULTIPLE candidate
/// senses of the QUERY word). Bare/primary first; capped.
fn resolve_concepts(brain: &Brain, token: &str) -> Vec<Seed> {
    let t: String = token.trim().to_lowercase();
    let mut out: Vec<Seed> = Vec::new();
    if t.len() < 2 {
        return out;
    }
    if let Some(bucket) = brain.network.keyword_index_bucket(&t) {
        for seed in bucket.iter() {
            if let Some(l) = brain.network.node_label(seed) {
                if l.trim().eq_ignore_ascii_case(&t) && !out.contains(seed) {
                    out.push(*seed);
                }
            }
        }
        for seed in bucket.iter() {
            if out.len() >= 6 {
                break;
            }
            if let Some(l) = brain.network.node_label(seed) {
                if strip_sense(l.trim()).eq_ignore_ascii_case(&t) && !out.contains(seed) {
                    out.push(*seed);
                }
            }
        }
    }
    if out.is_empty() {
        let s = Seed::from_label(&t);
        if brain.network.node_label(&s).map(|l| !l.is_empty()).unwrap_or(false) {
            out.push(s);
        }
    }
    out
}

/// "is a X a Y?" over the IS-A hierarchy (design 4.1).
fn category_membership(brain: &Brain, q: &str) -> Option<Conclusion> {
    let ops = article_operands(q);
    if ops.len() < 2 {
        return None;
    }
    let (xtok, ytok) = (ops[0].clone(), ops[1].clone());
    if xtok == ytok {
        return None;
    }
    let x_seeds = resolve_concepts(brain, &xtok);
    if x_seeds.is_empty() {
        return None;
    }
    let ys = resolve_concept(brain, &ytok)?;
    let yl = strip_sense(&brain.network.node_label(&ys).unwrap_or_else(|| ytok.clone())).to_string();

    // POSITIVE: try every candidate SENSE of X; Yes if any sense's IS-A
    // ancestors reach Y (by seed or sense-stripped label). Multi-sense query
    // resolution (design 3): "rose" (primary sense = the shrub) still yields
    // "a rose is a flower" via its flower sense, while NO sense of "dog"
    // reaches "plant" (the per-synset backbone keeps each sense in-lane).
    for xs in &x_seeds {
        let (xorder, xcame, xlab) = isa_closure(brain, *xs);
        let target = xorder.iter().map(|(s, _)| *s).find(|s| {
            *s == ys
                || xlab.get(s).map(|l| strip_sense(l).eq_ignore_ascii_case(&yl)).unwrap_or(false)
        });
        if let Some(t) = target {
            if t != *xs {
                let cited = isa_path_edges(brain, *xs, t, &xcame, &xlab);
                if !cited.is_empty() {
                    let xl = strip_sense(&brain.network.node_label(xs).unwrap_or_else(|| xtok.clone())).to_string();
                    let chain = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
                    let prose = format!("Yes -- a {} is a kind of {}. ({})", xl, yl, chain);
                    let confidence = weakest(&cited);
                    return Some(Conclusion {
                        kind: ReasoningKind::CategoryMembership,
                        prose,
                        cited_edges: cited,
                        confidence,
                    });
                }
            }
        }
    }

    // No sense of X is under Y -> negative / can't-place from the PRIMARY sense.
    let xs = x_seeds[0];
    let xl = strip_sense(&brain.network.node_label(&xs).unwrap_or_else(|| xtok.clone())).to_string();
    let (xorder, xcame, xlab) = isa_closure(brain, xs);
    let (yorder, _yc, _yl2) = isa_closure(brain, ys);
    let yset: HashSet<Seed> = yorder.iter().map(|(s, _)| *s).collect();
    let x_under_y = yorder.iter().any(|(s, _)| *s == xs);
    if !x_under_y {
        let common = xorder.iter().find(|(s, _)| yset.contains(s)).map(|(s, _)| *s);
        if let Some(cs) = common {
            let common_lab = strip_sense(&xlab.get(&cs).cloned().unwrap_or_default()).to_string();
            let xcat = xorder
                .iter()
                .find(|(s, _)| *s != xs)
                .and_then(|(s, _)| xlab.get(s).cloned())
                .map(|l| strip_sense(&l).to_string())
                .unwrap_or_default();
            let cited = isa_path_edges(brain, xs, cs, &xcame, &xlab);
            let prose = if xcat.is_empty() {
                format!(
                    "No -- a {} is not a kind of {}. (Heuristic: they share the broader category {}, but sit on different branches.)",
                    xl, yl, common_lab
                )
            } else {
                format!(
                    "No -- a {} is a {}, not a {}. (Heuristic: {} and {} share the broader category {}, but sit on different branches.)",
                    xl, xcat, yl, xl, yl, common_lab
                )
            };
            let confidence = if cited.is_empty() { 0.5 } else { weakest(&cited) };
            return Some(Conclusion {
                kind: ReasoningKind::CategoryMembership,
                prose,
                cited_edges: cited,
                confidence,
            });
        }
    }
    Some(Conclusion {
        kind: ReasoningKind::CategoryMembership,
        prose: format!(
            "I can't place a {} under {} from what I know -- I found no IS-A path between them.",
            xl, yl
        ),
        cited_edges: Vec::new(),
        confidence: 0.0,
    })
}
/// "what do X and Y have in common?" -- lowest common IS-A ancestor (4.1).
fn common_ancestor(brain: &Brain, q: &str) -> Option<Conclusion> {
    let mut ops = article_operands(q);
    if ops.len() < 2 {
        ops = content_tokens(q);
    }
    if ops.len() < 2 {
        return None;
    }
    let (xtok, ytok) = (ops[0].clone(), ops[1].clone());
    if xtok == ytok {
        return None;
    }
    let xs = resolve_concept(brain, &xtok)?;
    let ys = resolve_concept(brain, &ytok)?;
    let xl = strip_sense(&brain.network.node_label(&xs).unwrap_or_else(|| xtok.clone())).to_string();
    let yl = strip_sense(&brain.network.node_label(&ys).unwrap_or_else(|| ytok.clone())).to_string();
    let (xorder, xcame, xlab) = isa_closure(brain, xs);
    let (yorder, ycame, ylab) = isa_closure(brain, ys);
    let ydepth: std::collections::HashMap<Seed, usize> =
        yorder.iter().map(|(s, d)| (*s, *d)).collect();
    let mut best: Option<(Seed, usize)> = None;
    for (s, xd) in xorder.iter() {
        if *s == xs || *s == ys {
            continue;
        }
        if let Some(yd) = ydepth.get(s) {
            let comb = xd + yd;
            if best.map(|(_, b)| comb < b).unwrap_or(true) {
                best = Some((*s, comb));
            }
        }
    }
    let (lca, _) = best?;
    let lca_lab = strip_sense(&xlab.get(&lca).cloned().unwrap_or_default()).to_string();
    let mut cited = isa_path_edges(brain, xs, lca, &xcame, &xlab);
    cited.extend(isa_path_edges(brain, ys, lca, &ycame, &ylab));
    if cited.is_empty() {
        return None;
    }
    let confidence = weakest(&cited);
    let prose = format!(
        "Both a {} and a {} are {}. ({} -> {}; {} -> {}.)",
        xl, yl, lca_lab, xl, lca_lab, yl, lca_lab
    );
    Some(Conclusion {
        kind: ReasoningKind::CommonAncestor,
        prose,
        cited_edges: cited,
        confidence,
    })
}

// ---------------------------------------------------------------------
// Tier 1.3 -- causal-why (Causes / LeadsTo grounding)
// ---------------------------------------------------------------------

const MAX_CAUSAL_LIST: usize = 6;
// Epistemic-calibration Component 1 (fan-in "no single cause"):
const MULTI_CAUSE_MIN: usize = 4;      // >= this many causes -> multifactorial framing
const MAX_FANIN_SCAN: usize = 64;      // bound the true-fan-in count + generality reads
const GENERALITY_HUB_FANOUT: usize = 12; // a cause with more causal out-edges is a generic hub

fn is_causal_bond(bt: BondType) -> bool {
    matches!(bt, BondType::Causes | BondType::LeadsTo)
}

fn strip_lead_article(s: &str) -> &str {
    let t = s.trim();
    for a in ["a ", "an ", "the "] {
        if let Some(r) = t.strip_prefix(a) {
            return r.trim();
        }
    }
    t
}

/// Trim to a clean noun phrase: drop leading article, strip non-alnum edges.
fn clean_phrase(s: &str) -> String {
    strip_lead_article(s)
        .trim_matches(|c: char| !c.is_alphanumeric())
        .trim()
        .to_string()
}

/// "does X cause Y?" / "can X cause Y?" / "does X lead to Y?"
fn causal_verify_form(q: &str) -> Option<(String, String)> {
    for lead in ["does ", "do ", "can ", "could "] {
        if let Some(rest) = q.strip_prefix(lead) {
            for mid in [" cause ", " causes ", " lead to ", " leads to ", " result in "] {
                if let Some(i) = rest.find(mid) {
                    let x = clean_phrase(&rest[..i]);
                    let y = clean_phrase(&rest[i + mid.len()..]);
                    if !x.is_empty() && !y.is_empty() {
                        return Some((x, y));
                    }
                }
            }
        }
    }
    None
}

/// "what causes X?" / "what leads to X?" / "why does X happen/occur?" -> effect X.
fn causal_effect_form(q: &str) -> Option<String> {
    for lead in ["what causes ", "what cause ", "what leads to ", "what causes the "] {
        if let Some(r) = q.strip_prefix(lead) {
            let x = clean_phrase(r);
            if !x.is_empty() {
                return Some(x);
            }
        }
    }
    for lead in ["why does ", "why do ", "why is ", "why are "] {
        if let Some(r) = q.strip_prefix(lead) {
            for suf in [" happen", " occur", " form", " form?"] {
                if let Some(i) = r.find(suf) {
                    let x = clean_phrase(&r[..i]);
                    if !x.is_empty() {
                        return Some(x);
                    }
                }
            }
        }
    }
    None
}

/// Broadened causal-why SUBJECT extractor (causal-why phrasing wire). Handles
/// the copular / plural / bare-verb "why" forms plus "what makes X" that carry
/// NO happen/occur/form event suffix: "why is X <adj>", "why are X <adj>",
/// "why do X <verb>", "why does X <verb>", "what makes X". Returns the SUBJECT
/// noun phrase (trailing predicate token dropped) for the consumer to resolve
/// and read REAL incoming causal edges. Only reached after `causal_effect_form`
/// (the happen/occur/form + "what causes" forms) has declined, so it never
/// double-handles those; the "X cause/lead to Y" link forms are consumed
/// earlier in `try_synthesize` and are also guarded here.
fn causal_why_subject_broad(q: &str) -> Option<String> {
    for lead in [
        "why is ", "why are ", "why does ", "why do ", "what makes ", "what make ",
    ] {
        if let Some(r) = q.strip_prefix(lead) {
            if [" cause ", " causes ", " lead to ", " leads to ", " result in "]
                .iter()
                .any(|s| r.contains(s))
            {
                return None;
            }
            let tail = r.trim_end_matches(|c: char| {
                c == '?' || c == '.' || c == '!' || c == ' '
            });
            let words: Vec<&str> = tail.split_whitespace().collect();
            if words.len() >= 2 {
                let subj = words[..words.len() - 1].join(" ");
                let x = clean_phrase(&subj);
                if !x.is_empty() {
                    return Some(x);
                }
            }
            return None;
        }
    }
    None
}

/// Shared incoming-causal composer for the broadened phrasing wire. Reads
/// deduped/bounded incoming Causes/LeadsTo edges for the already-resolved
/// subject seeds `xs` (display label `xl`) and composes the grounded, cited
/// causal answer. Returns None when the subject holds NO incoming causal edge,
/// so the caller falls through to the WhyChain reasoner / idk gate instead of
/// emitting a bare refusal (never suppresses a real forward chain; never
/// fabricates an edge). Ranking/prose mirror block (2) exactly.
fn compose_incoming_causal(brain: &Brain, xs: &[Seed], xl: &str) -> Option<Conclusion> {
    let mut cands: Vec<(Seed, BondType, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in xs {
        for (nbr, bt) in brain
            .network
            .typed_edges_in(s)
            .into_iter()
            .take(MAX_PROPERTY_EDGES)
        {
            if !is_causal_bond(bt) {
                continue;
            }
            let nlab = brain.network.node_label(&nbr).unwrap_or_default();
            if nlab.is_empty() || !seen.insert(nlab.to_lowercase()) {
                continue;
            }
            cands.push((nbr, bt, nlab));
            if cands.len() >= MAX_FANIN_SCAN {
                break;
            }
        }
        if cands.len() >= MAX_FANIN_SCAN {
            break;
        }
    }
    let n = cands.len();
    if n == 0 {
        return None;
    }
    let mut ranked: Vec<(f32, usize, String, Seed, BondType)> = cands
        .into_iter()
        .map(|(seed, bt, lab)| {
            let conf = node_conf(brain, &seed).value();
            let gen = brain
                .network
                .typed_edges_out(&seed)
                .into_iter()
                .filter(|(_, b)| is_causal_bond(*b))
                .count();
            let pen = usize::from(gen > GENERALITY_HUB_FANOUT);
            (conf, pen, lab, seed, bt)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    let shown: Vec<CitedEdge> = ranked
        .iter()
        .take(MAX_CAUSAL_LIST)
        .map(|(_, _, lab, seed, bt)| CitedEdge {
            from: lab.clone(),
            bond: *bt,
            to: xl.to_string(),
            confidence: node_conf(brain, seed),
        })
        .collect();
    let residual = n - shown.len();
    let names: Vec<String> = shown.iter().map(|e| e.from.clone()).collect();
    let cites = shown.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
    let confidence = weakest(&shown);
    let prose = if n >= MULTI_CAUSE_MIN {
        let resid = if residual > 0 {
            format!(" (and {} others)", residual)
        } else {
            String::new()
        };
        format!(
            "There isn't a single cause of {} -- it's multifactorial. Contributing factors on record include {}{}. ({})",
            xl, join_list(&names), resid, cites
        )
    } else {
        format!("{} is caused by {}. ({})", capitalize_first(xl), join_list(&names), cites)
    };
    Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: shown, confidence })
}

/// Third-person-singular of a base verb, for "why does X <verb>" rendering
/// ("increase" -> "increases"). Handles the common -es / -ies cases plus a few
/// irregulars; a bare +s otherwise. Display-only.
fn third_person_singular(verb: &str) -> String {
    match verb {
        "have" => return "has".to_string(),
        "do" => return "does".to_string(),
        "go" => return "goes".to_string(),
        "be" => return "is".to_string(),
        _ => {}
    }
    if verb.ends_with('s')
        || verb.ends_with('x')
        || verb.ends_with('z')
        || verb.ends_with("ch")
        || verb.ends_with("sh")
    {
        return format!("{}es", verb);
    }
    if verb.ends_with('y') && verb.len() >= 2 {
        let prev = verb.as_bytes()[verb.len() - 2] as char;
        if !"aeiou".contains(prev) {
            return format!("{}ies", &verb[..verb.len() - 1]);
        }
    }
    format!("{}s", verb)
}

/// A parsed predicate-effect "why" question. The EFFECT lives in the predicate
/// (adjective / verb / had object), not the bare grammatical subject, so the
/// honest rendering is "<effect_clause> because of <cause>" and the resolvable
/// phenomenon is the FULL phrase (`resolve`), never the bare subject. This is
/// what keeps "why is the sky blue" from composing "Sky is caused by ...".
struct WhyEffect {
    resolve: String,
    effect_clause: String,
    cite_to: String,
}

fn why_has_cause_word(r: &str) -> bool {
    [" cause ", " causes ", " lead to ", " leads to ", " result in "]
        .iter()
        .any(|s| r.contains(s))
}

fn why_trim_tail(r: &str) -> &str {
    r.trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ' ')
}

/// Parse the PREDICATE-EFFECT "why" forms whose effect sits in the predicate:
///   "why is|are <subject> <adjective>"                   -> copular
///   "why does|do <subject> <verb>"                       -> verbal
///   "why do|does <pronoun> have|get|experience <object>" -> had-object
/// Returns None for the phenomenon-subject forms (handled earlier by
/// `causal_effect_form` -> "X is caused by ..."), for the "X cause/lead-to Y"
/// link forms (guarded), and for anything too short to carry a predicate. The
/// happen/occur/form verbs are consumed by `causal_effect_form` before this is
/// reached and are skipped defensively here too.
fn causal_why_effect(q: &str) -> Option<WhyEffect> {
    const PRONOUNS: [&str; 7] = ["we", "you", "they", "i", "people", "one", "humans"];
    const TRANSITIVE: [&str; 4] = ["have", "get", "experience", "see"];
    const EVENT_VERBS: [&str; 8] = [
        "happen", "happens", "occur", "occurs", "form", "forms", "exist", "exists",
    ];

    for (lead, cop) in [("why is ", "is"), ("why are ", "are")] {
        if let Some(r) = q.strip_prefix(lead) {
            if why_has_cause_word(r) {
                return None;
            }
            let words: Vec<&str> = why_trim_tail(r).split_whitespace().collect();
            if words.len() < 2 {
                return None;
            }
            let adj = words[words.len() - 1];
            let subj_clean = clean_phrase(&words[..words.len() - 1].join(" "));
            if subj_clean.is_empty() || adj.chars().any(|c| !c.is_alphanumeric() && c != '-') {
                return None;
            }
            let subj_disp = capitalize_first(&words[..words.len() - 1].join(" "));
            // Copular effect node is the ADJECTIVE-NOUN phrase ("blue sky"),
            // not the sentence "sky is blue" -- the ingest entity gate rejects a
            // copula-bearing object, and "blue sky" also keeps block (2)
            // ("what causes X") honest (bare "sky" holds no incoming causal edge).
            let phrase = format!("{} {}", adj, subj_clean).to_lowercase();
            return Some(WhyEffect {
                resolve: phrase.clone(),
                effect_clause: format!("{} {} {}", subj_disp, cop, adj),
                cite_to: phrase,
            });
        }
    }

    for (lead, plural) in [("why does ", false), ("why do ", true)] {
        if let Some(r) = q.strip_prefix(lead) {
            if why_has_cause_word(r) {
                return None;
            }
            let words: Vec<&str> = why_trim_tail(r).split_whitespace().collect();
            if words.len() < 2 {
                return None;
            }
            if words.len() >= 3
                && PRONOUNS.contains(&words[0].to_lowercase().as_str())
                && TRANSITIVE.contains(&words[1].to_lowercase().as_str())
            {
                let obj_clean = clean_phrase(&words[2..].join(" "));
                if obj_clean.is_empty() {
                    return None;
                }
                let effect_clause =
                    format!("{} {} {}", capitalize_first(words[0]), words[1], obj_clean);
                return Some(WhyEffect {
                    resolve: obj_clean.to_lowercase(),
                    effect_clause,
                    cite_to: obj_clean.to_lowercase(),
                });
            }
            let verb = words[words.len() - 1];
            if EVENT_VERBS.contains(&verb.to_lowercase().as_str()) {
                return None;
            }
            let subj_clean = clean_phrase(&words[..words.len() - 1].join(" "));
            if subj_clean.is_empty() || verb.chars().any(|c| !c.is_alphanumeric() && c != '-') {
                return None;
            }
            let verb_form = if plural {
                verb.to_lowercase()
            } else {
                third_person_singular(&verb.to_lowercase())
            };
            let phrase = format!("{} {}", subj_clean, verb_form).to_lowercase();
            return Some(WhyEffect {
                resolve: phrase.clone(),
                effect_clause: format!("{} {}", capitalize_first(&subj_clean), verb_form),
                cite_to: phrase,
            });
        }
    }
    None
}

/// Predicate-aware incoming-causal composer. Resolves the FULL phenomenon
/// phrase (not the bare subject), reads its deduped/bounded incoming
/// Causes/LeadsTo edges, and composes "<effect_clause> because of <cause>."
/// Returns None when the phrase does not resolve or holds NO incoming causal
/// edge, so the caller falls through to the WhyChain reasoner / idk gate --
/// never a fabricated edge, never the false "<subject> is caused by <cause>".
fn compose_predicate_effect(brain: &Brain, eff: &WhyEffect) -> Option<Conclusion> {
    let xs = resolve_concepts(brain, &eff.resolve);
    if xs.is_empty() {
        return None;
    }
    let mut cands: Vec<(Seed, BondType, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in &xs {
        for (nbr, bt) in brain
            .network
            .typed_edges_in(s)
            .into_iter()
            .take(MAX_PROPERTY_EDGES)
        {
            if !is_causal_bond(bt) {
                continue;
            }
            let nlab = brain.network.node_label(&nbr).unwrap_or_default();
            if nlab.is_empty() || !seen.insert(nlab.to_lowercase()) {
                continue;
            }
            cands.push((nbr, bt, nlab));
            if cands.len() >= MAX_FANIN_SCAN {
                break;
            }
        }
        if cands.len() >= MAX_FANIN_SCAN {
            break;
        }
    }
    let n = cands.len();
    if n == 0 {
        return None;
    }
    let mut ranked: Vec<(f32, usize, String, Seed, BondType)> = cands
        .into_iter()
        .map(|(seed, bt, lab)| {
            let conf = node_conf(brain, &seed).value();
            let gen = brain
                .network
                .typed_edges_out(&seed)
                .into_iter()
                .filter(|(_, b)| is_causal_bond(*b))
                .count();
            let pen = usize::from(gen > GENERALITY_HUB_FANOUT);
            (conf, pen, lab, seed, bt)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    let shown: Vec<CitedEdge> = ranked
        .iter()
        .take(MAX_CAUSAL_LIST)
        .map(|(_, _, lab, seed, bt)| CitedEdge {
            from: lab.clone(),
            bond: *bt,
            to: eff.cite_to.clone(),
            confidence: node_conf(brain, seed),
        })
        .collect();
    let residual = n - shown.len();
    let names: Vec<String> = shown.iter().map(|e| e.from.clone()).collect();
    let cites = shown.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
    let confidence = weakest(&shown);
    let extra = if residual > 0 { " (among other factors)" } else { "" };
    let prose = format!(
        "{} because of {}{}. ({})",
        eff.effect_clause,
        join_list(&names),
        extra,
        cites
    );
    Some(Conclusion {
        kind: ReasoningKind::CausalWhy,
        prose,
        cited_edges: shown,
        confidence,
    })
}


/// "what does X cause?" / "what does X lead to?" / "what happens if/when X?" -> cause X.
fn causal_cause_form(q: &str) -> Option<String> {
    for lead in ["what does ", "what do "] {
        if let Some(r) = q.strip_prefix(lead) {
            for suf in [" cause", " causes", " lead to", " leads to", " result in"] {
                if let Some(i) = r.find(suf) {
                    let x = clean_phrase(&r[..i]);
                    if !x.is_empty() {
                        return Some(x);
                    }
                }
            }
        }
    }
    for lead in ["what happens if ", "what happens when ", "what happens after "] {
        if let Some(r) = q.strip_prefix(lead) {
            let x = clean_phrase(r);
            if !x.is_empty() {
                return Some(x);
            }
        }
    }
    None
}

fn causal_relation_word(bt: BondType) -> &'static str {
    match bt {
        BondType::LeadsTo => "leads to",
        _ => "causes",
    }
}

/// Tier 1.3 -- causal reasoning over Causes/LeadsTo edges. Resolves the
/// subject to its concept node (like property_check), reads causal edges
/// DIRECTLY (in/out), and fires only on real edges (cited). No causal edge
/// on a resolved subject -> honest refusal; never a fabricated link.
/// Narrow, documented word-variant fallback for causal SUBJECT resolution
/// (Phase-4 cosmetic (d)). When the asked subject holds no causal edge, a
/// close lexical variant that IS in the graph can carry the grounded link
/// (smoking <-> cigarette). Only ever redirects to a node that GENUINELY
/// holds the edge and cites that real edge -- introduces no edge, so a
/// subject with no grounded variant (vaccine->autism) still refuses.
fn causal_subject_variants(subj: &str) -> &'static [&'static str] {
    match subj.trim().to_ascii_lowercase().as_str() {
        "smoking" | "smoke" | "smoker" | "smokers" => &["cigarette", "tobacco"],
        "cigarette" | "cigarettes" => &["smoking"],
        _ => &[],
    }
}

/// "why does X cause Y?" / "why does X lead to Y?" / "why does X result in Y?"
/// -> (X, Y). Asks Lu to JUSTIFY an asserted causal link -- distinct from the
/// yes/no verify form and from "why does X happen". Answered by confirming the
/// real X->Y edge (or a transitive Causes/LeadsTo chain), never by inventing a
/// mechanism. Placed first so it answers TERMINALLY and never falls through to
/// the generic multi-hypothesis shape (which emits a slot-empty template here).
fn causal_why_link_form(q: &str) -> Option<(String, String)> {
    for lead in ["why does ", "why do ", "why would ", "why can ", "why might "] {
        if let Some(rest) = q.strip_prefix(lead) {
            for mid in [
                " cause ", " causes ", " lead to ", " leads to ",
                " result in ", " results in ",
            ] {
                if let Some(i) = rest.find(mid) {
                    let x = clean_phrase(&rest[..i]);
                    let y = clean_phrase(&rest[i + mid.len()..]);
                    if !x.is_empty() && !y.is_empty() {
                        return Some((x, y));
                    }
                }
            }
        }
    }
    None
}

fn causal_why(brain: &Brain, q: &str) -> Option<Conclusion> {
    // (0) explain-link: "why does X cause Y?" -> confirm the REAL X->Y causal
    // edge, or a transitive Causes/LeadsTo chain X->Z->Y, else honest-refuse.
    // Lu holds the LINK, not the mechanism, and says so. Answering TERMINALLY
    // here also stops the generic multi-hypothesis shape from firing a
    // slot-empty "relates to via ." template on these questions. No fabrication.
    if let Some((xt, yt)) = causal_why_link_form(q) {
        let xs = resolve_concepts(brain, &xt);
        if xs.is_empty() {
            return None;
        }
        let ypred: HashSet<String> = content_tokens(&yt).into_iter().collect();
        let yhead = content_tokens(&yt).pop().unwrap_or_else(|| yt.clone());
        let xl = strip_sense(&brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone())).to_string();
        let y_match = |nlab: &str| -> bool {
            let nwords = label_words(nlab);
            nlab.eq_ignore_ascii_case(&yt)
                || nwords.contains(&yhead)
                || (!ypred.is_empty() && ypred.iter().all(|w| nwords.contains(w)))
        };
        // (a) direct edge X -> Y.
        for s in &xs {
            for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                if !is_causal_bond(bt) { continue; }
                let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                if nlab.is_empty() || !y_match(&nlab) { continue; }
                let cited = vec![CitedEdge {
                    from: xl.clone(), bond: bt, to: nlab.clone(),
                    confidence: node_conf(brain, &nbr),
                }];
                let prose = format!(
                    "{} is on record as {} {} -- I hold the link, though not the mechanism behind why. ({})",
                    capitalize_first(&xl),
                    match bt { BondType::LeadsTo => "leading to", _ => "a cause of" },
                    nlab, cited[0].arrow()
                );
                let confidence = weakest(&cited);
                return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence });
            }
        }
        // (b) word-variant fallback (smoking <-> cigarette): mirror the verify
        // arm -- redirect ONLY to a node that genuinely holds the X->Y edge.
        for v in causal_subject_variants(&xt) {
            let vs = resolve_concepts(brain, v);
            for s in &vs {
                for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                    if !is_causal_bond(bt) { continue; }
                    let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                    if nlab.is_empty() || !y_match(&nlab) { continue; }
                    let vl = strip_sense(&brain.network.node_label(s).unwrap_or_else(|| (*v).to_string())).to_string();
                    let cited = vec![CitedEdge {
                        from: vl.clone(), bond: bt, to: nlab.clone(),
                        confidence: node_conf(brain, &nbr),
                    }];
                    let prose = format!(
                        "{} is on record as {} {} -- I hold the link, though not the mechanism behind why. ({})",
                        capitalize_first(&vl),
                        match bt { BondType::LeadsTo => "leading to", _ => "a cause of" },
                        nlab, cited[0].arrow()
                    );
                    let confidence = weakest(&cited);
                    return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence });
                }
            }
        }
        // (c) transitive chain X -> Z -> Y (two hops, bounded, real edges only).
        for s in &xs {
            for (zseed, zbt) in brain.network.typed_edges_out(s).into_iter().take(MAX_FANIN_SCAN) {
                if !is_causal_bond(zbt) { continue; }
                let zlab = brain.network.node_label(&zseed).unwrap_or_default();
                if zlab.is_empty() || y_match(&zlab) { continue; }
                for (yseed, ybt) in brain.network.typed_edges_out(&zseed).into_iter().take(MAX_PROPERTY_EDGES) {
                    if !is_causal_bond(ybt) { continue; }
                    let ylab = brain.network.node_label(&yseed).unwrap_or_default();
                    if ylab.is_empty() || !y_match(&ylab) { continue; }
                    let cited = vec![
                        CitedEdge { from: xl.clone(), bond: zbt, to: zlab.clone(), confidence: node_conf(brain, &zseed) },
                        CitedEdge { from: zlab.clone(), bond: ybt, to: ylab.clone(), confidence: node_conf(brain, &yseed) },
                    ];
                    let prose = format!(
                        "{} {} {}, which in turn {} {}. ({})",
                        capitalize_first(&xl), causal_relation_word(zbt), zlab,
                        causal_relation_word(ybt), ylab,
                        cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ")
                    );
                    let confidence = weakest(&cited);
                    return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence });
                }
            }
        }
        // (d) no direct/variant/transitive link -> honest refusal (never a why).
        return Some(Conclusion {
            kind: ReasoningKind::CausalWhy,
            prose: format!("Not from what I have -- I hold no causal link from {} to {}.", xl, yt),
            cited_edges: Vec::new(), confidence: 0.0,
        });
    }
    // (1) verify: "does X cause Y?"
    if let Some((xt, yt)) = causal_verify_form(q) {
        let xs = resolve_concepts(brain, &xt);
        if xs.is_empty() {
            return None;
        }
        let ypred: HashSet<String> = content_tokens(&yt).into_iter().collect();
        let yhead = content_tokens(&yt).pop().unwrap_or_else(|| yt.clone());
        let xl = strip_sense(&brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone())).to_string();
        for s in &xs {
            for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                if !is_causal_bond(bt) {
                    continue;
                }
                let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                if nlab.is_empty() {
                    continue;
                }
                let nwords = label_words(&nlab);
                if nlab.eq_ignore_ascii_case(&yt)
                    || nwords.contains(&yhead)
                    || (!ypred.is_empty() && ypred.iter().all(|w| nwords.contains(w)))
                {
                    let cited = vec![CitedEdge {
                        from: xl.clone(), bond: bt, to: nlab.clone(),
                        confidence: node_conf(brain, &nbr),
                    }];
                    let prose = format!(
                        "Yes -- {} {} {}. ({})",
                        xl, causal_relation_word(bt), nlab, cited[0].arrow()
                    );
                    let confidence = weakest(&cited);
                    return Some(Conclusion {
                        kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence,
                    });
                }
            }
        }
        // Phase-4 (d): word-variant fallback -- asked subject holds no link,
        // but a lexical variant in the graph might (smoking->cigarette).
        for v in causal_subject_variants(&xt) {
            let vs = resolve_concepts(brain, v);
            for s in &vs {
                for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                    if !is_causal_bond(bt) { continue; }
                    let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                    if nlab.is_empty() { continue; }
                    let nwords = label_words(&nlab);
                    if nlab.eq_ignore_ascii_case(&yt)
                        || nwords.contains(&yhead)
                        || (!ypred.is_empty() && ypred.iter().all(|w| nwords.contains(w)))
                    {
                        let vl = strip_sense(&brain.network.node_label(s).unwrap_or_else(|| (*v).to_string())).to_string();
                        let cited = vec![CitedEdge {
                            from: vl.clone(), bond: bt, to: nlab.clone(),
                            confidence: node_conf(brain, &nbr),
                        }];
                        let prose = format!(
                            "Yes -- {} {} {}. ({})",
                            vl, causal_relation_word(bt), nlab, cited[0].arrow()
                        );
                        let confidence = weakest(&cited);
                        return Some(Conclusion {
                            kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence,
                        });
                    }
                }
            }
        }
        // resolved subject, no such causal link -> honest (proves misinfo cuts).
        return Some(Conclusion {
            kind: ReasoningKind::CausalWhy,
            prose: format!("Not from what I have -- I hold no causal link from {} to {}.", xl, yt),
            cited_edges: Vec::new(), confidence: 0.0,
        });
    }
    // (2) effect: "what causes X?" -> incoming Causes/LeadsTo.
    if let Some(xt) = causal_effect_form(q) {
        let xs = resolve_concepts(brain, &xt);
        if xs.is_empty() {
            return None;
        }
        let xl = strip_sense(&brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone())).to_string();
        // Collect ALL incoming causal edges (deduped, bounded) to read the
        // TRUE fan-in n -- not silently truncated at the display cap.
        let mut cands: Vec<(Seed, BondType, String)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for s in &xs {
            for (nbr, bt) in brain.network.typed_edges_in(s).into_iter().take(MAX_PROPERTY_EDGES) {
                if !is_causal_bond(bt) {
                    continue;
                }
                let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                if nlab.is_empty() || !seen.insert(nlab.to_lowercase()) {
                    continue;
                }
                cands.push((nbr, bt, nlab));
                if cands.len() >= MAX_FANIN_SCAN {
                    break;
                }
            }
            if cands.len() >= MAX_FANIN_SCAN {
                break;
            }
        }
        let n = cands.len();
        if n == 0 {
            return Some(Conclusion {
                kind: ReasoningKind::CausalWhy,
                prose: format!("Not from what I have -- I hold no cause on record for {}.", xl),
                cited_edges: Vec::new(), confidence: 0.0,
            });
        }
        // Rank: cited-edge confidence desc, then inverse-generality (a cause
        // with a big causal fan-out is a generic hub -> penalised), then label.
        // Confidence is flat (Inferred) today, so this degrades to
        // hub-penalty + alphabetical -- the single-factor set is unchanged.
        let mut ranked: Vec<(f32, usize, String, Seed, BondType)> = cands
            .into_iter()
            .map(|(seed, bt, lab)| {
                let conf = node_conf(brain, &seed).value();
                let gen = brain
                    .network
                    .typed_edges_out(&seed)
                    .into_iter()
                    .filter(|(_, b)| is_causal_bond(*b))
                    .count();
                let pen = usize::from(gen > GENERALITY_HUB_FANOUT);
                (conf, pen, lab, seed, bt)
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        let shown: Vec<CitedEdge> = ranked
            .iter()
            .take(MAX_CAUSAL_LIST)
            .map(|(_, _, lab, seed, bt)| CitedEdge {
                from: lab.clone(), bond: *bt, to: xl.clone(),
                confidence: node_conf(brain, seed),
            })
            .collect();
        let residual = n - shown.len();
        let names: Vec<String> = shown.iter().map(|e| e.from.clone()).collect();
        let cites = shown.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
        let confidence = weakest(&shown);
        // Component 1: >= MULTI_CAUSE_MIN distinct causes -> epistemic
        // "no single cause" framing (honest re-voicing of edges already cited).
        let prose = if n >= MULTI_CAUSE_MIN {
            let resid = if residual > 0 {
                format!(" (and {} others)", residual)
            } else {
                String::new()
            };
            format!(
                "There isn't a single cause of {} -- it's multifactorial. Contributing factors on record include {}{}. ({})",
                xl, join_list(&names), resid, cites
            )
        } else {
            format!("{} is caused by {}. ({})", capitalize_first(&xl), join_list(&names), cites)
        };
        return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: shown, confidence });
    }
    // (2b) Broadened phrasing wire: copular / plural / bare-verb "why" forms
    // plus "what makes X" that carry no happen/occur/form suffix. Reach the
    // SAME incoming-causal reader as (2), but COMPOSE ONLY on a real edge -- an
    // unresolved subject or a subject with no incoming causal edge falls through
    // to (3)/reasoner/idk untouched (no bare refusal, no fabrication).
    if let Some(eff) = causal_why_effect(q) {
        // PREDICATE-EFFECT reading: for "why is X <adj>" / "why does|do X
        // <verb>" / "why do we have X" the EFFECT is in the predicate, so the
        // honest answer is "<subject> <predicate> because of <cause>" and the
        // resolvable phenomenon is the FULL phrase -- NOT "<subject> is caused
        // by <cause>" (which asserts the bare subject is the effect: false for
        // sky/objects/entropy). Compose ONLY on a real incoming causal edge to
        // the resolved phrase; otherwise fall through to (3)/reasoner/idk. We
        // deliberately do NOT fall back to the subject-IS-phenomenon composer
        // for a predicate-effect form -- that is exactly the falsehood the
        // honesty floor forbids.
        if let Some(c) = compose_predicate_effect(brain, &eff) {
            return Some(c);
        }
    } else if let Some(xt) = causal_why_subject_broad(q) {
        // Non-predicate-effect broadened forms ("what makes X"): keep the
        // subject-IS-phenomenon reading (true only when the subject is itself
        // the phenomenon); same real-edge-only guard as before.
        let xs = resolve_concepts(brain, &xt);
        if !xs.is_empty() {
            let xl = strip_sense(
                &brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone()),
            )
            .to_string();
            if let Some(c) = compose_incoming_causal(brain, &xs, &xl) {
                return Some(c);
            }
        }
    }
    // (3) cause: "what does X cause?" -> outgoing Causes/LeadsTo.
    if let Some(xt) = causal_cause_form(q) {
        let xs = resolve_concepts(brain, &xt);
        if xs.is_empty() {
            return None;
        }
        let xl = strip_sense(&brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone())).to_string();
        let mut cited: Vec<CitedEdge> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for s in &xs {
            for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                if !is_causal_bond(bt) {
                    continue;
                }
                let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                if nlab.is_empty() || !seen.insert(nlab.to_lowercase()) {
                    continue;
                }
                cited.push(CitedEdge {
                    from: xl.clone(), bond: bt, to: nlab.clone(),
                    confidence: node_conf(brain, &nbr),
                });
                if cited.len() >= MAX_CAUSAL_LIST {
                    break;
                }
            }
            if cited.len() >= MAX_CAUSAL_LIST {
                break;
            }
        }
        if cited.is_empty() {
            // Phase-4 (d): word-variant fallback for the effect direction
            // (what does smoking cause -> read cigarette's effects).
            for v in causal_subject_variants(&xt) {
                let vs = resolve_concepts(brain, v);
                let mut vcited: Vec<CitedEdge> = Vec::new();
                let mut vseen: HashSet<String> = HashSet::new();
                let mut vl = String::new();
                for s in &vs {
                    for (nbr, bt) in brain.network.typed_edges_out(s).into_iter().take(MAX_PROPERTY_EDGES) {
                        if !is_causal_bond(bt) { continue; }
                        let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                        if nlab.is_empty() || !vseen.insert(nlab.to_lowercase()) { continue; }
                        if vl.is_empty() {
                            vl = strip_sense(&brain.network.node_label(s).unwrap_or_else(|| (*v).to_string())).to_string();
                        }
                        vcited.push(CitedEdge {
                            from: vl.clone(), bond: bt, to: nlab.clone(),
                            confidence: node_conf(brain, &nbr),
                        });
                        if vcited.len() >= MAX_CAUSAL_LIST { break; }
                    }
                    if vcited.len() >= MAX_CAUSAL_LIST { break; }
                }
                if !vcited.is_empty() {
                    let names: Vec<String> = vcited.iter().map(|e| e.to.clone()).collect();
                    let prose = format!("{} can cause {}. ({})",
                        capitalize_first(&vl), join_list(&names),
                        vcited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; "));
                    let confidence = weakest(&vcited);
                    return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: vcited, confidence });
                }
            }
            let xl2 = strip_sense(&brain.network.node_label(&xs[0]).unwrap_or_else(|| xt.clone())).to_string();
            return Some(Conclusion {
                kind: ReasoningKind::CausalWhy,
                prose: format!("Not from what I have -- I hold no effect on record for {}.", xl2),
                cited_edges: Vec::new(), confidence: 0.0,
            });
        }
        let names: Vec<String> = cited.iter().map(|e| e.to.clone()).collect();
        let prose = format!("{} can cause {}. ({})",
            capitalize_first(&xl), join_list(&names),
            cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; "));
        let confidence = weakest(&cited);
        return Some(Conclusion { kind: ReasoningKind::CausalWhy, prose, cited_edges: cited, confidence });
    }
    None
}

// ---------------------------------------------------------------------
// Tier 1.6 -- property-check (direct HasProperty grounding)
// ---------------------------------------------------------------------

const MAX_PROPERTY_LIST: usize = 8;

/// Yes/no + capability question surface ("is X ...", "does X ...",
/// "can X ..."). Permissive; property_check self-gates on a real edge.
fn is_property_query_form(q: &str) -> bool {
    q.starts_with("is ") || q.starts_with("are ")
        || q.starts_with("does ") || q.starts_with("do ")
        || q.starts_with("can ") || q.starts_with("could ")
        || q.starts_with("has ") || q.starts_with("have ")
        || q.starts_with("was ") || q.starts_with("were ")
}

/// "what is X like?" / "describe X" / "tell me about X" -- list a
/// concept's asserted properties.
fn is_property_list_form(q: &str) -> bool {
    ((q.starts_with("what is ") || q.starts_with("what are ") || q.starts_with("whats "))
        && q.contains(" like"))
        || q.starts_with("describe ")
        || q.starts_with("tell me about ")
}

/// Words that survive `content_tokens` but never name a property.
fn is_pred_stop(w: &str) -> bool {
    matches!(
        w,
        "have" | "has" | "had" | "having" | "like" | "kind" | "type" | "sort"
            | "look" | "looks" | "looking" | "feel" | "feels" | "really" | "actually"
    )
}

/// "a, b and c" join for the property-list prose.
fn join_list(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => format!("{}, and {}", items[..items.len() - 1].join(", "), items[items.len() - 1]),
    }
}

/// Pick the answer verb from the question surface so the prose reads
/// naturally across adjective / noun / verb properties: "a dog is loyal",
/// "a fish has scales", "a bird can fly", "a bird does migrate".
fn answer_verb(q: &str) -> &'static str {
    if q.starts_with("can ") || q.starts_with("could ") {
        "can"
    } else if q.contains(" have ") || q.contains(" has ")
        || q.starts_with("has ") || q.starts_with("have ")
    {
        "has"
    } else if q.starts_with("does ") || q.starts_with("do ") {
        "does"
    } else {
        "is"
    }
}

/// Tier 1.6 -- "is X P?" / "does X have P?" / "can X <verb>?" grounded
/// DIRECTLY on a `subject HasProperty prop` edge. Subject resolves to its
/// concept node (like category_membership), so it fires even when
/// multi-token recall anchored weakly. Fires ONLY on a real edge whose
/// object matches the queried predicate (head-anchored); returns `None`
/// when the subject cannot be resolved OR no asserted property matches --
/// the pipeline then refuses honestly. Never a fabricated "Yes".
fn property_check(brain: &Brain, q: &str) -> Option<Conclusion> {
    let toks = content_tokens(q);
    if toks.is_empty() {
        return None;
    }
    // Subject: the article operand ("a dog" -> dog) if present, else the
    // first content token.
    let arts = article_operands(q);
    let subj_tok = arts.first().cloned().unwrap_or_else(|| toks[0].clone());
    let x_seeds = resolve_concepts(brain, &subj_tok);
    if x_seeds.is_empty() {
        return None;
    }
    // Subject's own label words -- excluded from predicate extraction.
    let mut subj_words: HashSet<String> = HashSet::new();
    subj_words.insert(subj_tok.clone());
    for xs in &x_seeds {
        if let Some(l) = brain.network.node_label(xs) {
            for w in label_words(strip_sense(&l)) {
                subj_words.insert(w);
            }
        }
    }
    let pred: Vec<String> = toks
        .iter()
        .filter(|w| !subj_words.contains(*w) && !is_pred_stop(w))
        .cloned()
        .collect();
    let subj_label = strip_sense(
        &brain.network.node_label(&x_seeds[0]).unwrap_or_else(|| subj_tok.clone()),
    )
    .to_string();

    // "what is X like" (or no predicate left) -> list asserted properties.
    if is_property_list_form(q) || pred.is_empty() {
        let mut props: Vec<CitedEdge> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for xs in &x_seeds {
            for (nbr, bt) in brain.network.typed_edges_out(xs).into_iter().take(MAX_PROPERTY_EDGES) {
                if !is_property_bond(bt) {
                    continue;
                }
                let nlab = brain.network.node_label(&nbr).unwrap_or_default();
                if nlab.is_empty() || !seen.insert(nlab.to_lowercase()) {
                    continue;
                }
                props.push(CitedEdge {
                    from: subj_label.clone(),
                    bond: bt,
                    to: nlab,
                    confidence: node_conf(brain, &nbr),
                });
                if props.len() >= MAX_PROPERTY_LIST {
                    break;
                }
            }
            if props.len() >= MAX_PROPERTY_LIST {
                break;
            }
        }
        if props.is_empty() {
            return None;
        }
        let names: Vec<String> = props.iter().map(|e| e.to.clone()).collect();
        let prose = format!("A {} is: {}.", subj_label, join_list(&names));
        let confidence = weakest(&props);
        return Some(Conclusion {
            kind: ReasoningKind::PropertyCheck,
            prose,
            cited_edges: props,
            confidence,
        });
    }

    // Collect this concept's asserted property edges (all senses, bounded).
    let pred_phrase = pred.join(" ");
    let pred_head = pred.last().cloned().unwrap_or_default();
    let mut props: Vec<(Seed, BondType, String)> = Vec::new();
    for xs in &x_seeds {
        for (nbr, bt) in brain.network.typed_edges_out(xs).into_iter().take(MAX_PROPERTY_EDGES) {
            if !is_property_bond(bt) {
                continue;
            }
            let nlab = brain.network.node_label(&nbr).unwrap_or_default();
            if !nlab.is_empty() {
                props.push((nbr, bt, nlab));
            }
        }
    }
    // Prefer an EXACT label match ("fly" over "able to fly"), then a
    // head-anchored containment match ("scales" in "scales on body"). So an
    // unheld property ("wings") never spuriously matches -- honest No below.
    let hit = props
        .iter()
        .find(|(_, _, l)| l.eq_ignore_ascii_case(&pred_phrase))
        .or_else(|| {
            props.iter().find(|(_, _, l)| {
                let w = label_words(l);
                w.contains(&pred_head)
                    || (!pred.is_empty() && pred.iter().all(|p| w.contains(p)))
            })
        });
    if let Some((nbr, bt, nlab)) = hit {
        let cited = vec![CitedEdge {
            from: subj_label.clone(),
            bond: *bt,
            to: nlab.clone(),
            confidence: node_conf(brain, nbr),
        }];
        let verb = answer_verb(q);
        let prose = format!(
            "Yes -- a {} {} {}. ({})",
            subj_label, verb, nlab, cited[0].arrow()
        );
        let confidence = weakest(&cited);
        return Some(Conclusion {
            kind: ReasoningKind::PropertyCheck,
            prose,
            cited_edges: cited,
            confidence,
        });
    }
    // Recognised property question, subject resolved, we DO hold properties
    // for this concept, but none match the asked one -> honest, bounded
    // refusal. No fabrication (empty cited_edges) and no agentic wander.
    if !props.is_empty() && is_property_query_form(q) {
        // PART C: plain-English honest refusal (no internal "property"
        // vocabulary, no broken a/an article, no leaked asked-slot token).
        let prose = format!(
            "I don't have that from what I know about {}.",
            subj_label
        );
        return Some(Conclusion {
            kind: ReasoningKind::PropertyCheck,
            prose,
            cited_edges: Vec::new(),
            confidence: 0.0,
        });
    }
    None
}

// =====================================================================
// OP-BREADTH Phase-1 -- ENTITY SUBSYSTEM (separate shelf)
// ---------------------------------------------------------------------
// Wikidata entities are disjoint nodes labelled "<label>#Q<digits>" and
// (see network.rs rebuild walk) indexed ONLY under an "e_<word>" token
// namespace plus an "eq<qid>" key. Concept keyword buckets therefore
// never gain an entity seed, so every concept-path answer is byte-
// identical to the entity-free foundation. These shapes are the ONLY
// readers of the entity namespace and run LAST in try_synthesize.
// =====================================================================

const ENTITY_BUCKET_SCAN_CAP: usize = 512;

#[derive(Clone, Copy, PartialEq)]
enum EntIntent { Who, Where, Capital, KnownFor, Born, BornDate, DiedDate, WhatIs, OfficialLanguage, Currency, Author, BornCountry, DiedCountry }

/// FAME (QID -> Wikipedia SITELINK count = number of language editions), a
/// real prominence proxy, lazy-loaded once from the TSV at $LU_ENTITY_FAME
/// ("qid\\tsitelink_count"). Isaac Newton (Q935, ~300 sitelinks) dominates a
/// same-named town (~20), inverting the old raw-appearance-count ranking that
/// wrongly put the town first. Empty when unset (heap / test brains) so
/// disambiguation degrades to coverage + out-degree.
pub(crate) fn entity_fame(qid: u64) -> u32 {
    use std::sync::OnceLock;
    use std::collections::HashMap;
    static MAP: OnceLock<HashMap<u64, u32>> = OnceLock::new();
    let m = MAP.get_or_init(|| {
        let mut hm: HashMap<u64, u32> = HashMap::new();
        if let Ok(p) = std::env::var("LU_ENTITY_FAME") {
            if let Ok(txt) = std::fs::read_to_string(&p) {
                for line in txt.lines() {
                    let mut it = line.split('\t');
                    if let (Some(q), Some(c)) = (it.next(), it.next()) {
                        if let (Ok(q), Ok(c)) =
                            (q.trim().parse::<u64>(), c.trim().parse::<u32>())
                        {
                            hm.insert(q, c);
                        }
                    }
                }
            }
        }
        hm
    });
    m.get(&qid).copied().unwrap_or(0)
}

/// Resolve an entity mention (lowercased content words) to the best entity
/// node: candidates are seeds sharing "e_<word>" buckets, ranked by (token
/// coverage, notability, out-degree). None when unresolved.
// OP-FAME: dominant-sense override thresholds. For a bare-name query
// (`prefer` empty: who is X / overview / compare) a much-more-famous
// superset-label entity (physicist "Isaac Newton") may override the
// exact-label short-circuit that would otherwise pick a minor exact match
// (the town "Newton"). The famous sense must clear an ABSOLUTE floor AND beat
// the exact candidate by BOTH a ratio and an absolute margin, so obscure or
// near-tie names keep the stable exact-label choice (no regression).
pub(crate) const FAME_DOMINANT_MIN: u32 = 40;
pub(crate) const FAME_DOMINANCE_RATIO: u32 = 3;
pub(crate) const FAME_DOMINANCE_MARGIN: u32 = 25;

const ENT_GENERIC_ISA: &[&str] = &["human", "person", "man", "woman", "university teacher", "teacher", "researcher", "individual"];
const ENT_CONTINENTS: &[&str] = &["europe", "asia", "africa", "north america", "south america", "oceania", "antarctica", "americas", "insular oceania", "afro-eurasia", "eurasia"];

/// Pick the first edge whose object is NOT in `skip` (case-insensitive);
/// fall back to the first edge. Skips over-generic answers -- occupation
/// "human"/"university teacher", location "Europe" -- so "who is Einstein"
/// prefers "physicist" and "where is Paris" prefers a country over a continent.
const ENT_PREFER_OCC: &[&str] = &["physicist", "theoretical physicist", "chemist", "mathematician", "biologist", "naturalist", "astronomer", "physician", "philosopher", "engineer", "economist", "historian", "linguist", "writer", "poet", "novelist", "painter", "composer", "musician", "politician", "actor", "singer", "architect", "inventor", "biochemist", "astronaut", "philologist"];

/// "a" / "an" for the given noun (vowel-initial -> "an").
fn ent_article(w: &str) -> &'static str {
    match w.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => "an",
        _ => "a",
    }
}

/// Like `ent_pick` but first tries a `prefer` list (iconic occupations) so
/// "who is Einstein" resolves to "physicist" over "inventor"/"pedagogue".
fn ent_pick_pref(edges: Vec<CitedEdge>, prefer: &[&str], skip: &[&str]) -> Vec<CitedEdge> {
    // iterate `prefer` in PRIORITY order so "physicist" wins over "inventor".
    for p in prefer {
        if let Some(e) = edges.iter().find(|e| e.to.to_lowercase().as_str() == *p) {
            return vec![e.clone()];
        }
    }
    if let Some(e) = edges.iter().find(|e| !skip.contains(&e.to.to_lowercase().as_str())) {
        return vec![e.clone()];
    }
    edges.into_iter().take(1).collect()
}

fn ent_pick(edges: Vec<CitedEdge>, skip: &[&str]) -> Vec<CitedEdge> {
    if let Some(e) = edges.iter().find(|e| !skip.contains(&e.to.to_lowercase().as_str())) {
        return vec![e.clone()];
    }
    edges.into_iter().take(1).collect()
}

// OP-RESOLVE-FOLD (Fix 1): fold Latin diacritics to ASCII so a plain-ASCII
// query ("garcia marquez") reaches an accented canonical label
// ("garcía márquez"). General across Latin-1 Supplement + Latin Extended-A,
// plus a generic strip of combining diacritical marks (U+0300..U+036F) for
// already-decomposed input. Non-Latin scripts pass through unchanged, so
// nothing outside the Latin diacritic range is collapsed. ASCII input is a
// fixed point (fold == identity), so existing ASCII resolution is untouched.
fn ascii_fold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ('\u{0300}'..='\u{036F}').contains(&ch) { continue; } // combining marks
        match ch {
            'à'|'á'|'â'|'ã'|'ä'|'å'|'ā'|'ă'|'ą' => out.push('a'),
            'æ' => out.push_str("ae"),
            'ç'|'ć'|'ĉ'|'ċ'|'č' => out.push('c'),
            'ď'|'đ'|'ð' => out.push('d'),
            'è'|'é'|'ê'|'ë'|'ē'|'ĕ'|'ė'|'ę'|'ě' => out.push('e'),
            'ĝ'|'ğ'|'ġ'|'ģ' => out.push('g'),
            'ĥ'|'ħ' => out.push('h'),
            'ì'|'í'|'î'|'ï'|'ĩ'|'ī'|'ĭ'|'į'|'ı' => out.push('i'),
            'ĵ' => out.push('j'),
            'ķ' => out.push('k'),
            'ĺ'|'ļ'|'ľ'|'ŀ'|'ł' => out.push('l'),
            'ñ'|'ń'|'ņ'|'ň'|'ŉ' => out.push('n'),
            'ò'|'ó'|'ô'|'õ'|'ö'|'ø'|'ō'|'ŏ'|'ő' => out.push('o'),
            'œ' => out.push_str("oe"),
            'ŕ'|'ŗ'|'ř' => out.push('r'),
            'ś'|'ŝ'|'ş'|'š' => out.push('s'),
            'ß' => out.push_str("ss"),
            'ţ'|'ť'|'ŧ' => out.push('t'),
            'þ' => out.push_str("th"),
            'ù'|'ú'|'û'|'ü'|'ũ'|'ū'|'ŭ'|'ů'|'ű'|'ų' => out.push('u'),
            'ŵ' => out.push('w'),
            'ý'|'ÿ'|'ŷ' => out.push('y'),
            'ź'|'ż'|'ž' => out.push('z'),
            'ŋ' => out.push_str("ng"),
            'ĳ' => out.push_str("ij"),
            _ => out.push(ch),
        }
    }
    out
}

/// Lowercase then diacritic-fold a single mention/label token.
fn fold_token(w: &str) -> String { ascii_fold(&w.to_lowercase()) }

/// OP-RESOLVE-FOLD (Fix 1): serving-side fold overlay, built ONCE per brain
/// (keyed by keyword-index crc32) from the existing entity buckets -- NO
/// re-ingest. Maps a folded entity key ("e_garcia") to the accented bucket
/// key(s) it collapses ("e_garcía") so an ASCII query can reach diacritic
/// labels the index stores verbatim. Only buckets whose token carries a
/// non-ASCII byte are recorded, so the map is small (accented entities only)
/// and pure-ASCII lookups never consult it.
fn entity_fold_index(brain: &Brain) -> std::sync::Arc<std::collections::HashMap<String, Vec<String>>> {
    use std::sync::{Arc, Mutex, OnceLock};
    use std::collections::HashMap;
    static CACHE: OnceLock<Mutex<HashMap<u32, Arc<HashMap<String, Vec<String>>>>>> = OnceLock::new();
    let ki = brain.network.keyword_index_ref();
    let crc = ki.base().map(|b| b.brain_crc32()).unwrap_or(0);
    let cell = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(m) = cell.lock().unwrap().get(&crc) { return m.clone(); }
    let mut fm: HashMap<String, Vec<String>> = HashMap::new();
    let mut record = |tok: &str| {
        if let Some(rest) = tok.strip_prefix("e_") {
            if rest.bytes().any(|b| b >= 0x80) {
                let folded = format!("e_{}", ascii_fold(rest));
                if folded.as_str() != tok {
                    let e = fm.entry(folded).or_insert_with(Vec::new);
                    if !e.iter().any(|k| k == tok) { e.push(tok.to_string()); }
                }
            }
        }
    };
    if let Some(base) = ki.base() {
        for (tok, _len) in base.tokens_with_prefix("e_") { record(tok); }
    }
    for tok in ki.overlay_ref().keys() { record(tok.as_str()); }
    let arc = Arc::new(fm);
    cell.lock().unwrap().insert(crc, arc.clone());
    arc
}

/// OP-FABGUARD internal notes helper: roman-numeral ordinals and
/// generational suffixes that commonly trail a personal name ("Charles III",
/// "Martin Luther King Jr"). These are NOT distinctive identity tokens -- a
/// name query must be carried by its surname / rare tokens, never resolved on
/// one of these alone.
fn is_ordinal_or_suffix(w: &str) -> bool {
    matches!(
        w,
        "ii" | "iii" | "iv" | "v" | "vi" | "vii" | "viii" | "ix" | "x" | "xi"
            | "xii" | "xiii" | "xiv" | "xv" | "xvi" | "xvii" | "xviii" | "xix"
            | "xx" | "jr" | "sr" | "jnr" | "snr" | "junior" | "senior"
    )
}

fn resolve_entity(brain: &Brain, words: &[String], prefer: &[BondType]) -> Option<Seed> {
    use std::collections::HashMap;
    let words: Vec<&String> = words.iter().filter(|w| w.len() >= 2).collect();
    if words.is_empty() {
        return None;
    }
    // OP-RESOLVE-FOLD (Fix 1): fold-aware coverage. For each mention word we
    // scan the verbatim bucket (unchanged for ASCII) AND, when the folded
    // key differs, the accented bucket(s) it collapses -- counting each seed
    // at most once per word so coverage semantics are preserved.
    let fidx = entity_fold_index(brain);
    let mut cover: HashMap<Seed, u32> = HashMap::new();
    for w in &words {
        let mut seen: HashSet<Seed> = HashSet::new();
        let mut scan = |key: &str, cover: &mut HashMap<Seed, u32>, seen: &mut HashSet<Seed>| {
            if let Some(bucket) = brain.network.keyword_index_bucket(key) {
                for seed in bucket.iter().take(ENTITY_BUCKET_SCAN_CAP) {
                    if seen.insert(*seed) {
                        *cover.entry(*seed).or_insert(0) += 1;
                    }
                }
            }
        };
        let key = format!("e_{}", w);
        scan(&key, &mut cover, &mut seen);
        // Folded/ASCII form of this word. The fold overlay is keyed by the
        // FOLDED key, so consult it ALWAYS -- an ASCII query "velazquez" folds
        // to itself yet must still reach the accented bucket it collapses.
        let fkey = format!("e_{}", fold_token(w));
        if fkey != key {
            scan(&fkey, &mut cover, &mut seen);
        }
        if let Some(accented) = fidx.get(&fkey) {
            for ak in accented {
                if ak.as_str() != key { scan(ak, &mut cover, &mut seen); }
            }
        }
    }
    if cover.is_empty() {
        return None;
    }
    let max_cov = cover.values().copied().max().unwrap_or(0);
    // OP-BREADTH: prefer an EXACT label match (mention words == entity label
    // words) so "isaac newton" resolves to "Isaac Newton" over the superset
    // "Isaac Newton Evans"; then notability, then out-degree.
    let mset: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
    // Diacritic-folded mention set: exact-label tests compare folded tokens on
    // BOTH sides so "garcia marquez" can equal "garcía márquez". ASCII-only
    // mentions fold to themselves, so this is identical to `mset` there.
    let mset_f: HashSet<String> = mset.iter().map(|w| ascii_fold(w)).collect();
    let label_words_f = |label: &str| -> HashSet<String> {
        content_tokens(strip_sense(label)).into_iter().map(|t| ascii_fold(&t)).collect()
    };

    // OP-INTENT-RESOLVE (#1505): env-gated candidate dump for verification.
    if std::env::var("LU_RESOLVE_DEBUG").is_ok() {
        eprintln!("[resolve_entity] mention={:?} prefer={:?} candidates={}", mset, prefer, cover.len());
        for seed in cover.keys() {
            let label = brain.network.node_label(seed).unwrap_or_default();
            let edges = brain.network.typed_edges_out(seed);
            let carries = edges.iter().any(|(_, bt)| prefer.contains(bt));
            let dbg_qid = crate::network::entity_qid_of(&label);
            eprintln!(
                "   label={:?} qid={:?} fame={} outdeg={} carries_prefer={}",
                label, dbg_qid, entity_fame(dbg_qid.unwrap_or(0)), edges.len(), carries
            );
        }
    }

    // OP-INTENT-RESOLVE (#1505) + OP-RESOLVE-FOLD (Fix 2): intent/type-aware
    // disambiguation. When the calling intent will TRAVERSE a specific
    // relation (`prefer` -- AuthoredBy for "who wrote X", HasCapital for
    // "capital of X") resolve to the candidate that actually CARRIES that
    // out-edge: the play "Hamlet" (AuthoredBy -> Shakespeare) over the person
    // "Hamlet Isakhanli".
    //
    // Fix 2 COVERAGE GATE: a relation-carrier may only win if it matches the
    // mention as well as the best candidate (cover == max_cov) AND, for a
    // multi-token mention, shares at least two tokens -- so a lone shared
    // token cannot hijack an unrelated relation-carrier. Without this,
    // "who wrote Love in the Time of Cholera" mis-resolved to "In Search of
    // Lost Time" (AuthoredBy -> Proust) on the single shared token "time":
    // confident-wrong. An exact full-phrase match always qualifies. Carriers
    // are then ranked by (exact, coverage, notability, out-degree).
    // `prefer` empty (Who/WhatIs) => pass skipped, resolution unchanged.
    //
    // OP-FAME dominant-sense override. Runs BEFORE the rel_best / exact-label
    // short-circuits for a bare-entity OVERVIEW / COMPARE / WHO query -- a
    // "soft" prefer: empty, or the OVERVIEW bias set. A hugely-more-famous
    // superset-label sense (physicist "Isaac Newton", fame 302) then overrides
    // a minor exact-label one (town "Newton", fame 55) that the short-circuits
    // below would otherwise pick. HARD single-relation intents (authorship,
    // capital, ...) are excluded, so "who wrote Hamlet" still resolves the
    // work. The fame GAP is surfaced under LU_RESOLVE_DEBUG for a future
    // "ask-when-a-genuine-toss-up" policy (dominant -> answer, close -> ask).
    let soft_overview = prefer.is_empty() || prefer == OVERVIEW_PREFER;
    if soft_overview {
        let mut fame_exact: Option<(Seed, u32, usize)> = None;
        let mut fame_top: Option<(Seed, u32, usize)> = None;
        for (seed, cov) in &cover {
            let label = brain.network.node_label(seed).unwrap_or_default();
            let qid = crate::network::entity_qid_of(&label).unwrap_or(0);
            let fame = entity_fame(qid);
            let outdeg = brain.network.typed_edges_out(seed).len();
            if label_words_f(&label) == mset_f {
                let better = match fame_exact {
                    None => true,
                    Some((_, bf, bo)) => (fame, outdeg) > (bf, bo),
                };
                if better { fame_exact = Some((*seed, fame, outdeg)); }
            }
            if *cov == max_cov {
                let better = match fame_top {
                    None => true,
                    Some((_, bf, bo)) => (fame, outdeg) > (bf, bo),
                };
                if better { fame_top = Some((*seed, fame, outdeg)); }
            }
        }
        if let (Some((es, ef, _)), Some((fs, ff, _))) = (fame_exact, fame_top) {
            let gap = ff.saturating_sub(ef);
            let dominant = fs != es
                && ff >= FAME_DOMINANT_MIN
                && ff >= ef.saturating_mul(FAME_DOMINANCE_RATIO)
                && gap >= FAME_DOMINANCE_MARGIN;
            if std::env::var("LU_RESOLVE_DEBUG").is_ok() {
                eprintln!(
                    "[resolve_entity] fame exact={} top={} gap={} dominant={} -> {}",
                    ef, ff, gap, dominant,
                    if dominant { "FAMOUS-SENSE" } else { "defer" }
                );
            }
            if dominant {
                return Some(fs);
            }
        }
    }

    if !prefer.is_empty() {
        let mention_len = mset.len();
        let mut rel_best: Option<(Seed, u8, u32, u32, usize)> = None;
        for (seed, cov) in &cover {
            let edges = brain.network.typed_edges_out(seed);
            if !edges.iter().any(|(_, bt)| prefer.contains(bt)) {
                continue;
            }
            let label = brain.network.node_label(seed).unwrap_or_default();
            let exact: u8 = if label_words_f(&label) == mset_f { 1 } else { 0 };
            let qualifies = exact == 1
                || (*cov == max_cov && (mention_len <= 1 || *cov >= 2));
            if !qualifies {
                continue;
            }
            let qid = crate::network::entity_qid_of(&label).unwrap_or(0);
            let fame = entity_fame(qid);
            let outdeg = edges.len();
            let better = match rel_best {
                None => true,
                Some((_, be, bc, bn, bo)) => (exact, *cov, fame, outdeg) > (be, bc, bn, bo),
            };
            if better {
                rel_best = Some((*seed, exact, *cov, fame, outdeg));
            }
        }
        if let Some((s, _, _, _, _)) = rel_best {
            return Some(s);
        }
    }

    // OP-SOURCEPULL Batch-B (Fix 2): a very large first-name bucket
    // (e.g. e_charles, 1300 entries) is truncated at ENTITY_BUCKET_SCAN_CAP,
    // so the canonical full-name entity can lose a coverage point and be
    // wrongly dropped by the `max_cov` filter below -- "who was Charles
    // Darwin" resolved to "Avenida Charles Darwin" (a street) instead of
    // Charles Darwin#Q1035. An EXACT full-label match (label content-words ==
    // mention content-words) is an unambiguous identity signal, so honour it
    // FIRST -- ranked by notability then out-degree -- independent of
    // coverage. Byte-identical when no candidate label exactly equals the
    // mention (the common case), so non-exact resolution and the refusal
    // floor are untouched.
    {
        // Exact full-label identity signal, ranked by FAME then out-degree.
        // The dominant-fame override that lets a famous superset-label sense
        // beat a minor exact one runs EARLIER (see OP-FAME block above), so by
        // here any exact-label match is the honest identity answer.
        let mut exact_best: Option<(Seed, u32, usize)> = None;
        for seed in cover.keys() {
            let label = brain.network.node_label(seed).unwrap_or_default();
            if label_words_f(&label) == mset_f {
                let qid = crate::network::entity_qid_of(&label).unwrap_or(0);
                let fame = entity_fame(qid);
                let outdeg = brain.network.typed_edges_out(seed).len();
                let better = match exact_best {
                    None => true,
                    Some((_, bf, bo)) => (fame, outdeg) > (bf, bo),
                };
                if better {
                    exact_best = Some((*seed, fame, outdeg));
                }
            }
        }
        if let Some((s, _, _)) = exact_best {
            return Some(s);
        }
    }
    let mut best: Option<(Seed, u8, u32, usize)> = None;
    for (seed, cov) in &cover {
        if *cov < max_cov {
            continue;
        }
        let label = brain.network.node_label(seed).unwrap_or_default();
        let qid = crate::network::entity_qid_of(&label).unwrap_or(0);
        let fame = entity_fame(qid);
        let outdeg = brain.network.typed_edges_out(seed).len();
        let exact: u8 = if label_words_f(&label) == mset_f { 1 } else { 0 };
        let better = match best {
            None => true,
            Some((_, be, bn, bo)) => (exact, fame, outdeg) > (be, bn, bo),
        };
        if better {
            best = Some((*seed, exact, fame, outdeg));
        }
    }
    let winner = match best {
        Some((s, _, _, _)) => s,
        None => return None,
    };
    // OP-FABGUARD internal notes: multi-token proper-name honesty floor.
    // Any multi-token name query that reaches THIS weak-coverage fallback has
    // already failed the exact-label, relation-carrier, and dominant-fame
    // paths above. Only resolve it if the winning node actually covers EVERY
    // distinctive token of the query -- the surname / rare tokens -- not merely
    // a shared ordinal/suffix ("III"/"Jr") or a lone common given name.
    // Without this, "III" in "Marcus Vandenburg III" over-matches the
    // most-famous "...III" node (Charles III) and the overview arm narrates it
    // as the asked subject: a fabrication. Below the floor we REFUSE (None),
    // routing to the honest "I don't have anything solid on <name>" path.
    // Single-token and all-ordinal queries are unaffected; real held
    // multi-token names resolve via the paths ABOVE this fallback, so this only
    // tightens the weak partial-token tail.
    if mset.len() >= 2 {
        let distinctive: Vec<&String> =
            mset.iter().filter(|w| !is_ordinal_or_suffix(w)).collect();
        let nd = distinctive.len();
        if nd >= 1 {
            let covers = |w: &str| -> bool {
                let hit = |k: &str| {
                    brain
                        .network
                        .keyword_index_bucket(k)
                        .map(|b| {
                            b.iter()
                                .take(ENTITY_BUCKET_SCAN_CAP)
                                .any(|x| *x == winner)
                        })
                        .unwrap_or(false)
                };
                let key = format!("e_{}", w);
                if hit(&key) {
                    return true;
                }
                let fkey = format!("e_{}", fold_token(w));
                if fkey != key && hit(&fkey) {
                    return true;
                }
                if let Some(acc) = fidx.get(&fkey) {
                    for ak in acc {
                        if ak.as_str() != key && hit(ak) {
                            return true;
                        }
                    }
                }
                false
            };
            let dc = distinctive.iter().filter(|w| covers(w)).count();
            if dc < nd {
                if std::env::var("LU_RESOLVE_DEBUG").is_ok() {
                    eprintln!(
                        "[resolve_entity] FABGUARD refuse: winner covers {}/{} distinctive tokens (multi-token name not held)",
                        dc, nd
                    );
                }
                return None;
            }
        }
    }
    Some(winner)
}

fn entity_display(brain: &Brain, seed: &Seed) -> String {
    strip_sense(&brain.network.node_label(seed).unwrap_or_default()).to_string()
}

/// Distinct display labels of `seed`'s out-edges whose bond is in `bts`.
fn entity_edges(brain: &Brain, seed: &Seed, bts: &[BondType], max: usize) -> Vec<CitedEdge> {
    let subj = entity_display(brain, seed);
    let mut out: Vec<CitedEdge> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (nbr, bt) in brain.network.typed_edges_out(seed).into_iter().take(128) {
        if !bts.contains(&bt) {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        if l.is_empty() || !seen.insert(l.to_lowercase()) {
            continue;
        }
        out.push(CitedEdge { from: subj.clone(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
        if out.len() >= max {
            break;
        }
    }
    out
}

/// OP-REASON-CHAIN cap: LocatedIn containment hops walked upward from an
/// event place before giving up (city -> region -> country and a little
/// deeper), while bounding cost.
const PLACE_CHAIN_MAX_HOPS: usize = 4;

/// First outgoing typed neighbour of `seed` on bond `bt` whose sense-stripped
/// label is non-empty and not in `skip`. Returns the neighbour SEED so the
/// caller walks the next hop from the ACTUAL node this hop landed on -- never
/// a keyword re-resolution (no fuzz, no drift). None => honest dead-end.
fn first_typed_neighbor(brain: &Brain, seed: &Seed, bt: BondType, skip: &[&str]) -> Option<(Seed, String)> {
    for (nbr, b) in brain.network.typed_edges_out(seed).into_iter().take(128) {
        if b != bt {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        if l.is_empty() {
            continue;
        }
        if skip.iter().any(|c| l.eq_ignore_ascii_case(c)) {
            continue;
        }
        return Some((nbr, l));
    }
    None
}

/// OP-REASON-CHAIN: sound, node-anchored multi-hop place composition.
/// subject -[first_rel: BornIn|DiedIn]-> event place, then LocatedIn
/// containment walked UPWARD (continents skipped) to the largest containing
/// place = the country. EVERY hop is a real typed edge on the exact node the
/// prior hop landed on -- no re-resolution, no fuzz, no invented link. The
/// composition is transitivity-valid: place containment is transitive, so
/// born-in-city + city-in-country => born-in-country is SOUND. Returns None
/// -- caller falls through to the honest idk gate, NEVER a fabricated answer
/// -- the instant the first-hop edge or ALL containment edges are absent (a
/// birth-city node carrying no LocatedIn edge refuses rather than inventing a
/// country). Rendered provenance shows the FULL path so it is checkable.
fn compose_place_country(
    brain: &Brain,
    subj_seed: &Seed,
    name: &str,
    first_rel: BondType,
    event_clause: &str,
) -> Option<Conclusion> {
    // Hop 1: subject -[first_rel: BornIn|DiedIn]-> event place.
    let (mut cur_seed, first_place) = first_typed_neighbor(brain, subj_seed, first_rel, &[])?;
    let mut cur_label = first_place.clone();
    let mut cited: Vec<CitedEdge> = vec![CitedEdge {
        from: name.to_string(),
        bond: first_rel,
        to: first_place.clone(),
        confidence: node_conf(brain, &cur_seed),
    }];
    // Hops 2..: LocatedIn containment walked UPWARD (continents skipped) to the
    // largest containing place = the country. Node-anchored: each hop walks the
    // ACTUAL node the prior hop landed on -- never a keyword re-resolution.
    let mut country: Option<String> = None;
    for _ in 0..PLACE_CHAIN_MAX_HOPS {
        match first_typed_neighbor(brain, &cur_seed, BondType::LocatedIn, ENT_CONTINENTS) {
            Some((nbr, label)) => {
                cited.push(CitedEdge {
                    from: cur_label.clone(),
                    bond: BondType::LocatedIn,
                    to: label.clone(),
                    confidence: node_conf(brain, &nbr),
                });
                cur_seed = nbr;
                cur_label = label.clone();
                country = Some(label);
            }
            None => break,
        }
    }
    // No containment edge at all => data gap => honest refuse (never invent).
    let country = country?;
    // Full-path provenance so the intermediate node stays visible and checkable:
    // A -BornIn-> B -LocatedIn-> C. Country lives in the MAIN clause (not a
    // trailing subordinate clause the style layer would truncate).
    let mut path = cited[0].from.clone();
    for c in &cited {
        path.push_str(&format!(" -{:?}-> {}", c.bond, c.to));
    }
    let prose = format!("{} {} {}. ({})", name, event_clause, country, path);
    // RelationalPath (NOT EntityFact): a multi-hop typed path has a second
    // subject (the intermediate place), so the single-subject EntityFact fluent
    // re-voicer would drop the LocatedIn hop and collapse to a 1-hop answer.
    // RelationalPath is left untouched by fluent::realize, preserving the chain.
    let confidence = weakest(&cited);
    Some(Conclusion { kind: ReasoningKind::RelationalPath, prose, cited_edges: cited, confidence })
}

fn ent_strip_prefix(q: &str, prefs: &[&str]) -> Option<String> {
    for p in prefs {
        if let Some(rest) = q.strip_prefix(p) {
            return Some(rest.trim_start().to_string());
        }
    }
    None
}
fn ent_between(q: &str, a: &str, b: &str) -> Option<String> {
    let i = q.find(a)?;
    let after = &q[i + a.len()..];
    let j = after.find(b)?;
    Some(after[..j].trim().to_string())
}
fn ent_clean_mention(m: &str) -> String {
    let m = m.trim().trim_end_matches(|c: char| c == '?' || c == '.' || c == '!').trim();
    let m = m
        .strip_prefix("the ")
        .or_else(|| m.strip_prefix("a "))
        .or_else(|| m.strip_prefix("an "))
        .unwrap_or(m);
    m.trim().to_string()
}

/// OP-ATTR-ROUTE B2: strip a trailing locative phrase ("located",
/// "located in", "situated", "situated in") from a where-mention so the
/// entity resolver sees the clean subject. Longest tail first.
fn ent_strip_locative_tail(r: &str) -> String {
    let r = r.trim();
    for tail in [" located in", " situated in", " located", " situated"] {
        if let Some(m) = r.strip_suffix(tail) {
            return m.trim().to_string();
        }
    }
    r.to_string()
}

/// OP-BREADTH entity shape. Recognises entity-intent question forms,
/// resolves the mention in the entity namespace, and states a cited answer
/// from its typed edges. None (degrade) on a non-entity form or an
/// unresolved / edge-less mention.
// OP-OVERVIEW-ROUTE: attribute bonds that make a "what is X" / "tell me about
// X" subject eligible for the enriched entity overview (the SAME weave arm
// "who is X" uses) instead of the bare concept-definition gloss. A place carries
// HasCapital/HasOfficialLanguage/HasCurrency; a work carries an OUTGOING
// AuthoredBy. None of these ever sit on a concept node, so bare-concept glosses
// (dog, python, radioactivity) are untouched. Also drives carrier-preferring
// disambiguation (#1506) so "what is War and Peace" resolves the AuthoredBy-
// bearing work node and "what is France" the country node.
const OVERVIEW_PREFER: &[BondType] = &[
    BondType::HasCapital,
    BondType::HasOfficialLanguage,
    BondType::HasCurrency,
    BondType::AuthoredBy,
];

/// OP-SENSE-LEAD helper: the exact-subject stored dictionary Definition, read
/// the SAME way the ingest write path stored it (`fold_seed_label` then
/// `Seed::from_label`), via the v5 point-read with the SST definition store as
/// a fallback. Tries the subject as-is then a hyphen->space variant (so
/// "half-life" reaches the "half life" node). Returns None when the subject has
/// no canonical node or no non-empty Definition -- so the honesty floor holds
/// and a fabricated or absent subject can never be answered from here.
pub(crate) fn stored_definition(brain: &Brain, subject: &str) -> Option<String> {
    let lookup = |s: &str| -> Option<String> {
        let key = crate::canonical::fold_seed_label(s);
        let seed = crate::seed::Seed::from_label(&key);
        if let Some(id) = brain.network.node_intern().get(&seed) {
            if let Ok(Some(node)) = brain.get_node_v5(id) {
                if let crate::node::KnowledgeContent::Definition(def) = &node.knowledge.content {
                    let d = def.trim();
                    if !d.is_empty() {
                        return Some(d.to_string());
                    }
                }
            }
        }
        brain.network.definition_from_store(&seed)
    };
    let s = subject.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(d) = lookup(s) {
        return Some(d);
    }
    let spaced = s.replace('-', " ");
    if spaced != s {
        if let Some(d) = lookup(&spaced) {
            return Some(d);
        }
    }
    None
}

/// OP-SENSE-LEAD: parse a query into (subject, who_mode) when -- and only when
/// -- it is a "who is/was X" question or a bare entity/concept mention that the
/// sense-lead should handle. Returns None for every other shape (question
/// forms, multi-clause prose), so those paths are byte-unchanged.
pub(crate) fn sense_subject(query: &str) -> Option<(String, bool)> {
    let ql_full = query.trim().to_lowercase();
    let ql = ql_full
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!' || c.is_whitespace());
    if ql.is_empty() {
        return None;
    }
    const WHO: &[&str] = &["who is ", "who was ", "who are ", "who were ", "who's "];
    let mut who_mode = false;
    let mut subject: &str = ql;
    for p in WHO {
        if let Some(rest) = ql.strip_prefix(p) {
            subject = rest.trim();
            who_mode = true;
            break;
        }
    }
    if !who_mode {
        // BARE mode: reject anything that reads as a question or an
        // article/verb-led clause so only a genuine bare mention resolves
        // here. The stored-Definition / router gate downstream is the real
        // safety net -- a question phrase never names a canonical node.
        const NONBARE_LEAD: &[&str] = &[
            "what ", "where ", "when ", "why ", "how ", "which ", "whose ",
            "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ",
            "could ", "should ", "will ", "would ", "has ", "have ", "tell ",
            "define ", "name ", "list ", "compare ", "explain ", "give ",
            "the ", "a ", "an ",
        ];
        if NONBARE_LEAD.iter().any(|p| ql.starts_with(p)) {
            return None;
        }
        subject = ql;
    }
    let subject = subject.trim();
    if subject.is_empty() {
        return None;
    }
    let ntok = content_tokens(subject).len();
    if ntok == 0 || ntok > 4 {
        return None;
    }
    Some((subject.to_string(), who_mode))
}

/// OP-SENSE-LEAD (P9 sense/fame mis-ranking + raw-edge word-salad): for a bare
/// entity/concept term or a "who is/was X" query, return the stored dictionary
/// Definition of the subject WHEN leading with it is the honest,
/// notability-correct answer. Two live serving-path failures this closes:
///   * "who was Cleopatra" -> "Cleopatra is a painting." / "who was Genghis
///     Khan" -> "The Private Life of Genghis Khan is a literary work.":
///     resolve_entity scans only the QID-anchored e_<token> entity buckets, so
///     it never sees the famous WordNet sense (a plain-labelled node carrying a
///     rich stored Definition, absent from those buckets) and picks a
///     low-notability same-name homograph.
///   * bare "mars"/"atom"/"ruby" fall to the recall FactRecall arm, which leads
///     with a wrong high-activation neighbour (canal/ion/almandine) and dumps
///     raw adjacent edges as word-salad.
///
/// Honesty floor: returns None unless the exact subject node carries a REAL
/// stored Definition, so it can never fabricate -- absent/fake subjects fall
/// through to the existing refusal path. WHO mode additionally DEFERS to the
/// entity-overview arm whenever resolve_entity's same-name winner is a
/// high-notability node (fame >= FAME_DOMINANT_MIN or a real place/work
/// attribute edge), so famous overviews (Newton, Einstein, countries) are
/// untouched and only a LOW-notability homograph is overridden. (The bare-mode
/// fame-aware router pass -- python->programming(+snake) -- runs at the call
/// site in narration before this fallback.)
pub(crate) fn sense_lead_definition(brain: &Brain, query: &str) -> Option<String> {
    let (subject, who_mode) = sense_subject(query)?;
    // Honesty floor: a real stored dictionary Definition, or nothing.
    let def = stored_definition(brain, &subject)?;
    if who_mode {
        // Defer to the entity-overview arm when its same-name winner is a
        // genuinely notable node; only override a low-notability homograph.
        let words = content_tokens(&subject);
        if let Some(seed) = resolve_entity(brain, &words, &[]) {
            let label = brain.network.node_label(&seed).unwrap_or_default();
            let qid = crate::network::entity_qid_of(&label).unwrap_or(0);
            let fame = entity_fame(qid);
            let overview_signal = !entity_edges(
                brain,
                &seed,
                &[
                    BondType::HasCapital,
                    BondType::HasOfficialLanguage,
                    BondType::HasCurrency,
                    BondType::AuthoredBy,
                ],
                1,
            )
            .is_empty();
            if fame >= FAME_DOMINANT_MIN || overview_signal {
                return None;
            }
        }
    }
    Some(def)
}
fn entity_query(brain: &Brain, q: &str) -> Option<Conclusion> {
    let ql = q.trim();
    let (intent, mention) = if let Some(i) = ql.find("official language of ") {
        // OP-ATTR-ROUTE: 'official language of X' -> HasOfficialLanguage.
        (EntIntent::OfficialLanguage, ql[i + 21..].to_string())
    } else if let Some(i) = ql.find("official languages of ") {
        (EntIntent::OfficialLanguage, ql[i + 22..].to_string())
    } else if let Some(i) = ql.find("capital of ") {
        (EntIntent::Capital, ql[i + 11..].to_string())
    } else if let Some(i) = ql.find("currency of ") {
        // OP-SOURCEPULL Batch-C (Fix 1): 'currency of X' / 'what is the
        // currency of X' -> HasCurrency (P38). Honest-refuse if absent.
        (EntIntent::Currency, ql[i + 12..].to_string())
    } else if let Some(m) = ent_between(ql, "what currency does ", " use")
        .or_else(|| ent_between(ql, "what currency do ", " use"))
        .or_else(|| ent_between(ql, "which currency does ", " use"))
    {
        (EntIntent::Currency, m)
    } else if let Some(rest) = ent_strip_prefix(ql, &["who wrote ", "who authored ", "who is the author of ", "who was the author of ", "who's the author of "]) {
        // OP-SOURCEPULL Batch-C (Fix 2): author-of-work -> AuthoredBy (P50),
        // distinct from WrittenBy (P170 artwork-creator). Refuse if absent.
        (EntIntent::Author, rest)
    } else if let Some(i) = ql.find("author of ") {
        (EntIntent::Author, ql[i + 10..].to_string())
    } else if let Some(rest) = ent_strip_prefix(ql, &["who is ", "who was ", "who's ", "who are "]) {
        (EntIntent::Who, rest)
    } else if let Some(rest) =
        ent_strip_prefix(ql, &["where is ", "where's ", "where are ", "where was "])
    {
        let r = rest.trim_end_matches('?').trim();
        if let Some(m) = r.strip_suffix(" born") {
            (EntIntent::Born, m.to_string())
        } else {
            // OP-ATTR-ROUTE B2: strip a trailing locative tail so
            // "where is X located [in]" / "... situated [in]" resolves the
            // CLEAN mention X (the extra token otherwise corrupts entity
            // resolution -> wrong seed, e.g. Berlin -> a Berlin university).
            // Routes to the existing Where -> LocatedIn arm.
            (EntIntent::Where, ent_strip_locative_tail(r))
        }
    } else if let Some(m) = ent_between(ql, "what country is ", " located in")
        .or_else(|| ent_between(ql, "which country is ", " located in"))
        .or_else(|| ent_between(ql, "what country is ", " in"))
        .or_else(|| ent_between(ql, "which country is ", " in"))
    {
        // OP-ATTR-ROUTE B2: "what/which country is X [located] in" ->
        // LocatedIn via the Where arm (ent_pick skips continents, so the
        // rendered target is the country/region, not a continent).
        (EntIntent::Where, m)
    } else if let Some(m) = ent_between(ql, "what country was ", " born")
        .or_else(|| ent_between(ql, "which country was ", " born"))
        .or_else(|| ent_between(ql, "in what country was ", " born"))
        .or_else(|| ent_between(ql, "in which country was ", " born"))
    {
        // OP-REASON-CHAIN: "what country was X born in" -> sound 2-hop place
        // chain BornIn -> LocatedIn (born-in-city + city-in-country =>
        // born-in-country; transitivity-valid place containment). Composes
        // ONLY on real typed edges, else falls through to the honest idk gate.
        (EntIntent::BornCountry, m)
    } else if let Some(m) = ent_between(ql, "what country did ", " die")
        .or_else(|| ent_between(ql, "which country did ", " die"))
        .or_else(|| ent_between(ql, "in what country did ", " die"))
        .or_else(|| ent_between(ql, "in which country did ", " die"))
    {
        // OP-REASON-CHAIN: "what country did X die in" -> DiedIn -> LocatedIn.
        (EntIntent::DiedCountry, m)
} else if let Some(rest) = ent_strip_prefix(ql, &["when was ", "when were "]) {
        // OP-SOURCEPULL Batch-B (Fix 3): date-of-birth intent. "when was X
        // born" -> BornOn (date), distinct from "where was X born" -> BornIn
        // (place, handled by the where-branch above). Honest-refuse downstream
        // if no BornOn bond is present.
        let r = rest.trim_end_matches('?').trim();
        match r.strip_suffix(" born") {
            Some(m) => (EntIntent::BornDate, m.to_string()),
            None => return None,
        }
    } else if let Some(rest) = ent_strip_prefix(ql, &["when did ", "when do "]) {
        // OP-SOURCEPULL Batch-B (Fix 3): date-of-death intent. "when did X
        // die / pass away" -> DiedOn (date). Honest-refuse if no DiedOn bond.
        let r = rest.trim_end_matches('?').trim();
        let m = r
            .strip_suffix(" die")
            .or_else(|| r.strip_suffix(" died"))
            .or_else(|| r.strip_suffix(" pass away"))
            .or_else(|| r.strip_suffix(" pass"));
        match m {
            Some(m) => (EntIntent::DiedDate, m.to_string()),
            None => return None,
        }
    } else if let Some(m) = ent_between(ql, "what is ", " known for")
        .or_else(|| ent_between(ql, "what was ", " known for"))
    {
        (EntIntent::KnownFor, m)
    } else if let Some(rest) =
        ent_strip_prefix(ql, &["what is ", "what was ", "what's ", "what are "])
    {
        (EntIntent::WhatIs, rest)
    } else {
        return None;
    };
    let mention = ent_clean_mention(&mention);
    if mention.is_empty() {
        return None;
    }
    let words = content_tokens(&mention);
    if words.is_empty() {
        return None;
    }
    // OP-OVERVIEW-ROUTE: the "what is X" concept-defer moved to AFTER entity
    // resolution (just below `resolve_entity`) so a place/work subject that
    // carries real attribute edges can reach the overview arm, while a bare
    // concept still defers to the definition gloss.
    // OP-INTENT-RESOLVE (#1505): declare which relation this intent will
    // traverse so same-named candidates are biased toward the node that
    // carries it (work-node for authorship, place-node for capital, etc.).
    // Empty for Who/WhatIs (generic IsA/KnownFor) -> batch-B fallback.
    let prefer: &[BondType] = match intent {
        EntIntent::Author => &[BondType::AuthoredBy],
        EntIntent::Capital => &[BondType::HasCapital],
        EntIntent::OfficialLanguage => &[BondType::HasOfficialLanguage],
        EntIntent::Currency => &[BondType::HasCurrency],
        EntIntent::Where => &[BondType::LocatedIn],
        EntIntent::Born => &[BondType::BornIn],
        EntIntent::BornDate => &[BondType::BornOn],
        EntIntent::DiedDate => &[BondType::DiedOn],
        EntIntent::BornCountry => &[BondType::BornIn],
        EntIntent::DiedCountry => &[BondType::DiedIn],
        EntIntent::KnownFor => &[BondType::KnownFor],
        EntIntent::Who => &[],
        // OP-OVERVIEW-ROUTE: bias "what is X" disambiguation toward a place/
        // work carrier so the overview arm can weave its attributes.
        EntIntent::WhatIs => OVERVIEW_PREFER,
    };
    let seed = resolve_entity(brain, &words, prefer)?;
    let name = entity_display(brain, &seed);
    // OP-OVERVIEW-ROUTE: "what is X" (and the upstream-rewritten "tell me about
    // X") defers to the concept-definition gloss for a BARE CONCEPT -- but a
    // place or work subject carries attribute edges the overview arm weaves, so
    // it must NOT defer. overview_signal = a real place attribute (HasCapital/
    // HasOfficialLanguage/HasCurrency) or an outgoing AuthoredBy (work). Neither
    // ever sits on a concept node, so "what is a dog / python / radioactivity"
    // still fall through to their short gloss unchanged.
    if intent == EntIntent::WhatIs {
        let overview_signal = !entity_edges(brain, &seed, &[BondType::HasCapital], 1).is_empty()
            || !entity_edges(brain, &seed, &[BondType::HasOfficialLanguage], 1).is_empty()
            || !entity_edges(brain, &seed, &[BondType::HasCurrency], 1).is_empty()
            || !entity_edges(brain, &seed, &[BondType::AuthoredBy], 1).is_empty();
        if !overview_signal
            && words.iter().any(|w| !resolve_concepts(brain, w).is_empty())
        {
            return None;
        }
    }

    let (prose, cited): (String, Vec<CitedEdge>) = match intent {
        EntIntent::Where => {
            let loc = ent_pick(entity_edges(brain, &seed, &[BondType::LocatedIn], 8), ENT_CONTINENTS);
            if loc.is_empty() {
                return None;
            }
            (format!("{} is in {}. ({})", name, loc[0].to, loc[0].arrow()), loc)
        }
        EntIntent::Capital => {
            let cap = entity_edges(brain, &seed, &[BondType::HasCapital], 1);
            if cap.is_empty() {
                return None;
            }
            (format!("The capital of {} is {}. ({})", name, cap[0].to, cap[0].arrow()), cap)
        }
        EntIntent::OfficialLanguage => {
            // OP-ATTR-ROUTE: read the existing HasOfficialLanguage typed
            // edge(s) on the resolved entity; refuse honestly if absent.
            let langs = entity_edges(brain, &seed, &[BondType::HasOfficialLanguage], 3);
            if langs.is_empty() {
                return None;
            }
            let names: Vec<String> = langs.iter().map(|e| e.to.clone()).collect();
            let (noun, verb) = if names.len() > 1 {
                ("official languages", "are")
            } else {
                ("official language", "is")
            };
            (
                format!(
                    "The {} of {} {} {}. ({})",
                    noun, name, verb, join_list(&names), langs[0].arrow()
                ),
                langs,
            )
        }
        EntIntent::Currency => {
            // OP-SOURCEPULL Batch-C (Fix 1): read HasCurrency (P38); refuse if absent.
            let cur = entity_edges(brain, &seed, &[BondType::HasCurrency], 3);
            if cur.is_empty() {
                return None;
            }
            let names: Vec<String> = cur.iter().map(|e| e.to.clone()).collect();
            let (noun, verb) = if names.len() > 1 { ("currencies", "are") } else { ("currency", "is") };
            (
                format!("The {} of {} {} {}. ({})", noun, name, verb, join_list(&names), cur[0].arrow()),
                cur,
            )
        }
        EntIntent::Author => {
            // OP-SOURCEPULL Batch-C (Fix 2): prefer the dedicated AuthoredBy bond
            // (P50 author-of-work). Do NOT fall back to WrittenBy -- that is P170
            // artwork-creator semantics and resolves book titles to same-named
            // paintings/people (#1495), i.e. confident-wrong. Absent AuthoredBy ->
            // honest refuse (None), preserving the existing recall guard.
            let a = entity_edges(brain, &seed, &[BondType::AuthoredBy], 2);
            if a.is_empty() {
                return None;
            }
            let names: Vec<String> = a.iter().map(|e| e.to.clone()).collect();
            (format!("{} was written by {}. ({})", name, join_list(&names), a[0].arrow()), a)
        }
        EntIntent::Born => {
            let b = entity_edges(brain, &seed, &[BondType::BornIn], 1);
            if b.is_empty() {
                return None;
            }
            (format!("{} was born in {}. ({})", name, b[0].to, b[0].arrow()), b)
        }
        EntIntent::BornCountry => return compose_place_country(brain, &seed, &name, BondType::BornIn, "was born in"),
        EntIntent::DiedCountry => return compose_place_country(brain, &seed, &name, BondType::DiedIn, "died in"),
EntIntent::BornDate => {
            // OP-SOURCEPULL Batch-B (Fix 3): render date of birth from the
            // BornOn bond (year node). Absent => honest refuse (None).
            let b = entity_edges(brain, &seed, &[BondType::BornOn], 1);
            if b.is_empty() {
                return None;
            }
            (format!("{} was born in {}. ({})", name, b[0].to, b[0].arrow()), b)
        }
        EntIntent::DiedDate => {
            // OP-SOURCEPULL Batch-B (Fix 3): render date of death from the
            // DiedOn bond (year node). Absent => honest refuse (None).
            let d = entity_edges(brain, &seed, &[BondType::DiedOn], 1);
            if d.is_empty() {
                return None;
            }
            (format!("{} died in {}. ({})", name, d[0].to, d[0].arrow()), d)
        }
        EntIntent::KnownFor => {
            let mut kf = entity_edges(brain, &seed, &[BondType::KnownFor], 2);
            if kf.is_empty() {
                kf = entity_edges(brain, &seed, &[BondType::IsA], 2);
            }
            if kf.is_empty() {
                return None;
            }
            let names: Vec<String> = kf.iter().map(|e| e.to.clone()).collect();
            (format!("{} is known for {}.", name, join_list(&names)), kf)
        }
        EntIntent::Who | EntIntent::WhatIs => {
            let isa = ent_pick_pref(entity_edges(brain, &seed, &[BondType::IsA], 8), ENT_PREFER_OCC, ENT_GENERIC_ISA);
            // OP-OVERVIEW: weave MORE of the node's real typed edges into one
            // coherent overview. Gather up to 5 KnownFor (composer shows a few,
            // trails off), birth/death PLACE and YEAR, and place attributes.
            // Every edge here is a real typed edge -- the composer only ever
            // reorders / phrases them; it never invents a connective fact.
            let kf = entity_edges(brain, &seed, &[BondType::KnownFor], 5);
            // OP-OVERVIEW-ROUTE: a work node's authorship (outgoing AuthoredBy).
            let authored = entity_edges(brain, &seed, &[BondType::AuthoredBy], 2);
            let born_place = entity_edges(brain, &seed, &[BondType::BornIn], 1);
            let born_date = entity_edges(brain, &seed, &[BondType::BornOn], 1);
            // OP-TENSE / OP-OVERVIEW: a DiedIn (place) or DiedOn (date) bond is
            // the ONLY grounded deceased signal -> past tense. Birth and death
            // YEARS are now cited when present: OP-SOURCEPULL Batch-B/C added
            // BornOn/DiedOn, so "born in Ulm in 1879" is grounded in a real
            // edge, not a guess. Absent DiedIn/DiedOn => present tense.
            let died_place = entity_edges(brain, &seed, &[BondType::DiedIn], 1);
            let died_date = entity_edges(brain, &seed, &[BondType::DiedOn], 1);
            // Place-of-subject attributes (present only on place nodes). Skip
            // continents for LocatedIn so the country/region is surfaced.
            let located = ent_pick(entity_edges(brain, &seed, &[BondType::LocatedIn], 8), ENT_CONTINENTS);
            let capital = entity_edges(brain, &seed, &[BondType::HasCapital], 1);
            let languages = entity_edges(brain, &seed, &[BondType::HasOfficialLanguage], 2);
            let currency = entity_edges(brain, &seed, &[BondType::HasCurrency], 1);
            let deceased = !died_place.is_empty() || !died_date.is_empty();
            // Voice place attributes only on a genuine place signal so a person
            // node never grows a capital / language / currency clause.
            let place_signal = !capital.is_empty() || !languages.is_empty() || !currency.is_empty();

            let mut cited: Vec<CitedEdge> = Vec::new();
            let mut s = String::new();
            if let Some(e) = isa.first() {
                s.push_str(&format!("{} {} {} {}.", name, if deceased { "was" } else { "is" }, ent_article(&e.to), e.to));
                cited.push(e.clone());
            } else {
                s.push_str(&name);
                s.push('.');
            }
            // OP-OVERVIEW-ROUTE: weave a work's authorship right after its type
            // -- "War and Peace is a novel. Written by Leo Tolstoy." A real cited
            // outgoing AuthoredBy edge; empty on non-work nodes (no clause).
            if !authored.is_empty() {
                let names: Vec<String> = authored.iter().map(|e| e.to.clone()).collect();
                s.push_str(&format!(" Written by {}.", join_list(&names)));
                cited.extend(authored);
            }
            if !kf.is_empty() {
                let names: Vec<String> = kf.iter().take(3).map(|e| e.to.clone()).collect();
                let more = if kf.len() > 3 { ", among others" } else { "" };
                s.push_str(&format!(" Known for {}{}.", join_list(&names), more));
                cited.extend(kf);
            }
            // Birth: place [+ year].
            match (born_place.first(), born_date.first()) {
                (Some(p), Some(y)) => {
                    s.push_str(&format!(" Born in {} in {}.", p.to, y.to));
                    cited.push(p.clone());
                    cited.push(y.clone());
                }
                (Some(p), None) => {
                    s.push_str(&format!(" Born in {}.", p.to));
                    cited.push(p.clone());
                }
                (None, Some(y)) => {
                    s.push_str(&format!(" Born in {}.", y.to));
                    cited.push(y.clone());
                }
                (None, None) => {}
            }
            // Death: place [+ year].
            match (died_place.first(), died_date.first()) {
                (Some(p), Some(y)) => {
                    s.push_str(&format!(" Died in {} in {}.", p.to, y.to));
                    cited.push(p.clone());
                    cited.push(y.clone());
                }
                (Some(p), None) => {
                    s.push_str(&format!(" Died in {}.", p.to));
                    cited.push(p.clone());
                }
                (None, Some(y)) => {
                    s.push_str(&format!(" Died in {}.", y.to));
                    cited.push(y.clone());
                }
                (None, None) => {}
            }
            // Place attributes: only when a real place signal is present.
            if place_signal {
                if let Some(e) = located.first() {
                    s.push_str(&format!(" Located in {}.", e.to));
                    cited.push(e.clone());
                }
                if let Some(e) = capital.first() {
                    s.push_str(&format!(" Capital: {}.", e.to));
                    cited.push(e.clone());
                }
                if !languages.is_empty() {
                    let names: Vec<String> = languages.iter().map(|e| e.to.clone()).collect();
                    s.push_str(&format!(" Official language: {}.", join_list(&names)));
                    cited.extend(languages);
                }
                if let Some(e) = currency.first() {
                    s.push_str(&format!(" Currency: {}.", e.to));
                    cited.push(e.clone());
                }
            }
            if cited.is_empty() {
                return None;
            }
            (s, cited)
        }
    };
    let confidence = if cited.is_empty() { 0.0 } else { weakest(&cited) };
    Some(Conclusion { kind: ReasoningKind::EntityFact, prose, cited_edges: cited, confidence })
}

// =====================================================================
// Tier 2 -- composition / purpose / meronymy / relational-path / comparative
// ---------------------------------------------------------------------
// All read the SAME typed-edge seam the live reader spreads over, resolve
// operands DIRECTLY to concept nodes (like Tier 1.2/1.6), and never invent
// an edge: they cite the exact bonds walked, or -- subject resolved but the
// required edge absent -- refuse honestly (empty cited_edges). In THIS
// foundation brain constituents are encoded "hydrogen -PartOf-> water", so
// composition/meronymy read INCOMING PartOf; holonymy reads OUTGOING PartOf.
// =====================================================================

const MAX_COMPOSE_LIST: usize = 8;
const MAX_PARTS_LIST: usize = 10;
const CMP_PROP_MAX: usize = 5;

/// Resolve a subject phrase to one concept node: whole phrase first, then
/// the first resolvable content word. None -> degrade (never a fuzzy guess).
fn resolve_subject(brain: &Brain, subj: &str) -> Option<Seed> {
    resolve_concept(brain, subj)
        .or_else(|| content_tokens(subj).iter().find_map(|w| resolve_concept(brain, w)))
}

/// "what is X made of" / "made from" / "composed of" / "consists of".
fn composition_form(q: &str) -> Option<String> {
    for pat in [" made of", " made out of", " made from", " composed of",
                " consist of", " consists of", " comprised of", " made up of"] {
        if let Some(i) = q.find(pat) {
            let head = &q[..i];
            let subj = head
                .trim_start_matches("what is ")
                .trim_start_matches("what are ")
                .trim_start_matches("what's ")
                .trim_start_matches("whats ")
                .trim();
            let subj = strip_lead_article(subj);
            if !subj.is_empty() {
                return Some(subj.to_string());
            }
        }
    }
    None
}

/// Composition: OUTGOING ComposedOf/DerivedFrom + INCOMING PartOf.
fn composition(brain: &Brain, q: &str) -> Option<Conclusion> {
    let subj = composition_form(q)?;
    let seed = resolve_subject(brain, &subj)?;
    let name = strip_sense(&brain.network.node_label(&seed).unwrap_or_else(|| subj.clone())).to_string();
    let mut cited: Vec<CitedEdge> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
        if !matches!(bt, BondType::ComposedOf | BondType::DerivedFrom) {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        if l.is_empty() || !seen.insert(l.to_lowercase()) {
            continue;
        }
        cited.push(CitedEdge { from: name.clone(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
        if cited.len() >= MAX_COMPOSE_LIST {
            break;
        }
    }
    if cited.len() < MAX_COMPOSE_LIST {
        for (nbr, bt) in brain.network.typed_edges_in(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
            if !matches!(bt, BondType::PartOf) {
                continue;
            }
            let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
            if l.is_empty() || !seen.insert(l.to_lowercase()) {
                continue;
            }
            cited.push(CitedEdge { from: l, bond: bt, to: name.clone(), confidence: node_conf(brain, &nbr) });
            if cited.len() >= MAX_COMPOSE_LIST {
                break;
            }
        }
    }
    if cited.is_empty() {
        return Some(Conclusion {
            kind: ReasoningKind::Composition,
            prose: format!("Not from what I have -- I hold no composition on record for {}.", name),
            cited_edges: Vec::new(),
            confidence: 0.0,
        });
    }
    let parts: Vec<String> = cited
        .iter()
        .map(|e| if e.to.eq_ignore_ascii_case(&name) { e.from.clone() } else { e.to.clone() })
        .collect();
    let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
    let prose = format!("{} is made of {}. ({})", capitalize_first(&name), join_list(&parts), cites);
    let confidence = weakest(&cited);
    Some(Conclusion { kind: ReasoningKind::Composition, prose, cited_edges: cited, confidence })
}

/// "what is X used for" / "purpose of X" / "what is the function of X".
fn purpose_form(q: &str) -> Option<String> {
    for (a, b) in [("what is ", " used for"), ("what are ", " used for"),
                   ("what's ", " used for"), ("whats ", " used for"),
                   ("what is ", " good for"), ("what are ", " good for")] {
        if let Some(m) = ent_between(q, a, b) {
            let sub = strip_lead_article(m.trim());
            if !sub.is_empty() {
                return Some(sub.to_string());
            }
        }
    }
    for lead in ["what is the purpose of ", "what's the purpose of ", "purpose of ",
                 "what is the use of ", "what is the function of ", "what is the point of "] {
        if let Some(r) = q.strip_prefix(lead) {
            let sub = strip_lead_article(r.trim());
            if !sub.is_empty() {
                return Some(sub.to_string());
            }
        }
    }
    None
}

/// Purpose: OUTGOING UsedFor/PurposeOf.
fn purpose(brain: &Brain, q: &str) -> Option<Conclusion> {
    let subj = purpose_form(q)?;
    let seed = resolve_subject(brain, &subj)?;
    let name = strip_sense(&brain.network.node_label(&seed).unwrap_or_else(|| subj.clone())).to_string();
    let mut cited: Vec<CitedEdge> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
        if !matches!(bt, BondType::UsedFor | BondType::PurposeOf) {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        if l.is_empty() || !seen.insert(l.to_lowercase()) {
            continue;
        }
        cited.push(CitedEdge { from: name.clone(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
        if cited.len() >= MAX_COMPOSE_LIST {
            break;
        }
    }
    if cited.is_empty() {
        return Some(Conclusion {
            kind: ReasoningKind::Purpose,
            prose: format!("Not from what I have -- I hold no stated use or purpose for {}.", name),
            cited_edges: Vec::new(),
            confidence: 0.0,
        });
    }
    let uses: Vec<String> = cited.iter().map(|e| e.to.clone()).collect();
    let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
    let prose = format!("{} is used for {}. ({})", capitalize_first(&name), join_list(&uses), cites);
    let confidence = weakest(&cited);
    Some(Conclusion { kind: ReasoningKind::Purpose, prose, cited_edges: cited, confidence })
}

/// "what are the parts of X" / "components of X" / "what parts does X have".
fn parts_of_form(q: &str) -> Option<String> {
    for lead in ["what are the parts of ", "what are the components of ",
                 "what are the main parts of ", "what are parts of ",
                 "parts of ", "components of "] {
        if let Some(r) = q.strip_prefix(lead) {
            let sub = strip_lead_article(r.trim());
            if !sub.is_empty() {
                return Some(sub.to_string());
            }
        }
    }
    if let Some(m) = ent_between(q, "what parts does ", " have") {
        let sub = strip_lead_article(m.trim());
        if !sub.is_empty() {
            return Some(sub.to_string());
        }
    }
    None
}

/// "what is X part of" / "what is X a part of".
fn part_of_form(q: &str) -> Option<String> {
    for (a, b) in [("what is ", " a part of"), ("what is ", " part of"),
                   ("what are ", " a part of"), ("what are ", " part of"),
                   ("what's ", " part of")] {
        if let Some(m) = ent_between(q, a, b) {
            let sub = strip_lead_article(m.trim());
            if !sub.is_empty() {
                return Some(sub.to_string());
            }
        }
    }
    None
}

/// Meronymy: parts of X = INCOMING PartOf (+ OUTGOING ComposedOf).
/// Holonymy: what X is part of = OUTGOING PartOf.
fn meronym_holonym(brain: &Brain, q: &str) -> Option<Conclusion> {
    if let Some(subj) = parts_of_form(q) {
        let seed = resolve_subject(brain, &subj)?;
        let name = strip_sense(&brain.network.node_label(&seed).unwrap_or_else(|| subj.clone())).to_string();
        let mut cited: Vec<CitedEdge> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (nbr, bt) in brain.network.typed_edges_in(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
            if !matches!(bt, BondType::PartOf) {
                continue;
            }
            let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
            if l.is_empty() || !seen.insert(l.to_lowercase()) {
                continue;
            }
            cited.push(CitedEdge { from: l, bond: bt, to: name.clone(), confidence: node_conf(brain, &nbr) });
            if cited.len() >= MAX_PARTS_LIST {
                break;
            }
        }
        if cited.len() < MAX_PARTS_LIST {
            for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
                if !matches!(bt, BondType::ComposedOf) {
                    continue;
                }
                let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
                if l.is_empty() || !seen.insert(l.to_lowercase()) {
                    continue;
                }
                cited.push(CitedEdge { from: name.clone(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
                if cited.len() >= MAX_PARTS_LIST {
                    break;
                }
            }
        }
        if cited.is_empty() {
            return Some(Conclusion {
                kind: ReasoningKind::Meronymy,
                prose: format!("Not from what I have -- I hold no parts on record for {}.", name),
                cited_edges: Vec::new(),
                confidence: 0.0,
            });
        }
        let parts: Vec<String> = cited
            .iter()
            .map(|e| if e.to.eq_ignore_ascii_case(&name) { e.from.clone() } else { e.to.clone() })
            .collect();
        let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
        let prose = format!("The parts of {} include {}. ({})", name, join_list(&parts), cites);
        let confidence = weakest(&cited);
        return Some(Conclusion { kind: ReasoningKind::Meronymy, prose, cited_edges: cited, confidence });
    }
    if let Some(subj) = part_of_form(q) {
        let seed = resolve_subject(brain, &subj)?;
        let name = strip_sense(&brain.network.node_label(&seed).unwrap_or_else(|| subj.clone())).to_string();
        let mut cited: Vec<CitedEdge> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (nbr, bt) in brain.network.typed_edges_out(&seed).into_iter().take(MAX_PROPERTY_EDGES) {
            if !matches!(bt, BondType::PartOf) {
                continue;
            }
            let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
            if l.is_empty() || !seen.insert(l.to_lowercase()) {
                continue;
            }
            cited.push(CitedEdge { from: name.clone(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
            if cited.len() >= MAX_PARTS_LIST {
                break;
            }
        }
        if cited.is_empty() {
            return Some(Conclusion {
                kind: ReasoningKind::Meronymy,
                prose: format!("Not from what I have -- I hold nothing that {} is part of.", name),
                cited_edges: Vec::new(),
                confidence: 0.0,
            });
        }
        let wholes: Vec<String> = cited.iter().map(|e| e.to.clone()).collect();
        let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
        let prose = format!("{} is part of {}. ({})", capitalize_first(&name), join_list(&wholes), cites);
        let confidence = weakest(&cited);
        return Some(Conclusion { kind: ReasoningKind::Meronymy, prose, cited_edges: cited, confidence });
    }
    None
}

/// Human-readable verb for a bond inside relational prose.
fn relation_verb(bt: BondType) -> &'static str {
    match bt {
        BondType::IsA | BondType::DefinedAs | BondType::ExampleOf | BondType::Inherits => "is a",
        BondType::PartOf => "is part of",
        BondType::HasProperty | BondType::HasAspect | BondType::HasAttribute => "has the property",
        BondType::Causes | BondType::LeadsTo => "causes",
        BondType::ComposedOf => "is composed of",
        BondType::DerivedFrom => "is derived from",
        BondType::UsedFor => "is used for",
        BondType::RelatedTo => "is related to",
        BondType::Produces => "produces",
        BondType::Requires => "requires",
        BondType::Opposes | BondType::ContrastedWith | BondType::DiffersFrom => "contrasts with",
        _ => "is linked to",
    }
}

fn two_ops_after(q: &str, marker: &str) -> Option<(String, String)> {
    let i = q.find(marker)?;
    let rest = &q[i + marker.len()..];
    let j = rest.find(" and ")?;
    let x = clean_phrase(&rest[..j]);
    let mut y = rest[j + 5..].to_string();
    for suf in [" related", " connected", " linked", " differ", "?"] {
        if let Some(k) = y.find(suf) {
            y.truncate(k);
        }
    }
    let y = clean_phrase(&y);
    if x.is_empty() || y.is_empty() {
        return None;
    }
    Some((x, y))
}

fn relational_operands(q: &str) -> Option<(String, String)> {
    for m in ["relationship between ", "relation between ", "connection between ",
              "link between ", "difference between "] {
        if let Some(pr) = two_ops_after(q, m) {
            return Some(pr);
        }
    }
    if q.starts_with("how are ") || q.starts_with("how do ") || q.starts_with("how is ") {
        if let Some(pr) = two_ops_after(q, "how are ")
            .or_else(|| two_ops_after(q, "how do "))
            .or_else(|| two_ops_after(q, "how is "))
        {
            return Some(pr);
        }
    }
    for m in [" related to ", " connected to ", " linked to "] {
        if let Some(i) = q.find(m) {
            let head = q[..i]
                .trim_start_matches("how ")
                .trim_start_matches("is ")
                .trim_start_matches("are ")
                .trim();
            let x = clean_phrase(head);
            let y = clean_phrase(&q[i + m.len()..]);
            if !x.is_empty() && !y.is_empty() {
                return Some((x, y));
            }
        }
    }
    None
}

/// Relational-path: direct typed edge either direction, else the lowest
/// shared IS-A ancestor, else a shared same-target neighbour. Honest
/// "no connection" when the two resolve but nothing links them.
fn relational_path(brain: &Brain, q: &str) -> Option<Conclusion> {
    let (xt, yt) = relational_operands(q)?;
    let xs = resolve_subject(brain, &xt)?;
    let ys = resolve_subject(brain, &yt)?;
    if xs == ys {
        return None;
    }
    let xl = strip_sense(&brain.network.node_label(&xs).unwrap_or_else(|| xt.clone())).to_string();
    let yl = strip_sense(&brain.network.node_label(&ys).unwrap_or_else(|| yt.clone())).to_string();
    for (nbr, bt) in brain.network.typed_edges_out(&xs).into_iter().take(MAX_PROPERTY_EDGES) {
        if is_property_bond(bt) {
            continue; // "X has property Y" is not a relationship between the two concepts
        }
        if nbr == ys
            || brain.network.node_label(&nbr).map(|l| strip_sense(&l).eq_ignore_ascii_case(&yl)).unwrap_or(false)
        {
            let cited = vec![CitedEdge { from: xl.clone(), bond: bt, to: yl.clone(), confidence: node_conf(brain, &nbr) }];
            let prose = format!("{} {} {}. ({})", capitalize_first(&xl), relation_verb(bt), yl, cited[0].arrow());
            let confidence = weakest(&cited);
            return Some(Conclusion { kind: ReasoningKind::RelationalPath, prose, cited_edges: cited, confidence });
        }
    }
    for (nbr, bt) in brain.network.typed_edges_out(&ys).into_iter().take(MAX_PROPERTY_EDGES) {
        if is_property_bond(bt) {
            continue;
        }
        if nbr == xs
            || brain.network.node_label(&nbr).map(|l| strip_sense(&l).eq_ignore_ascii_case(&xl)).unwrap_or(false)
        {
            let cited = vec![CitedEdge { from: yl.clone(), bond: bt, to: xl.clone(), confidence: node_conf(brain, &nbr) }];
            let prose = format!("{} {} {}. ({})", capitalize_first(&yl), relation_verb(bt), xl, cited[0].arrow());
            let confidence = weakest(&cited);
            return Some(Conclusion { kind: ReasoningKind::RelationalPath, prose, cited_edges: cited, confidence });
        }
    }
    let (xorder, xcame, xlab) = isa_closure(brain, xs);
    let (yorder, ycame, ylab) = isa_closure(brain, ys);
    let ydepth: std::collections::HashMap<Seed, usize> = yorder.iter().map(|(sd, d)| (*sd, *d)).collect();
    let mut best: Option<(Seed, usize)> = None;
    for (sd, xd) in xorder.iter() {
        if *sd == xs || *sd == ys {
            continue;
        }
        if let Some(yd) = ydepth.get(sd) {
            let comb = xd + yd;
            if best.map(|(_, b)| comb < b).unwrap_or(true) {
                best = Some((*sd, comb));
            }
        }
    }
    if let Some((lca, _)) = best {
        let lca_lab = strip_sense(&xlab.get(&lca).cloned().unwrap_or_default()).to_string();
        let mut cited = isa_path_edges(brain, xs, lca, &xcame, &xlab);
        cited.extend(isa_path_edges(brain, ys, lca, &ycame, &ylab));
        if !cited.is_empty() {
            let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
            let prose = format!("{} and {} are related: both are {}. ({})", capitalize_first(&xl), yl, lca_lab, cites);
            let confidence = weakest(&cited);
            return Some(Conclusion { kind: ReasoningKind::RelationalPath, prose, cited_edges: cited, confidence });
        }
    }
    let mut ymap: std::collections::HashMap<String, (Seed, BondType)> = Default::default();
    for (nbr, bt) in brain.network.typed_edges_out(&ys).into_iter().take(MAX_PROPERTY_EDGES) {
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_lowercase();
        if !l.is_empty() {
            ymap.entry(l).or_insert((nbr, bt));
        }
    }
    for (nbr, bt) in brain.network.typed_edges_out(&xs).into_iter().take(MAX_PROPERTY_EDGES) {
        if !matches!(bt, BondType::PartOf | BondType::RelatedTo | BondType::IsA) {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        let ll = l.to_lowercase();
        if ll.is_empty() {
            continue;
        }
        if let Some((ynbr, ybt)) = ymap.get(&ll) {
            let cited = vec![
                CitedEdge { from: xl.clone(), bond: bt, to: l.clone(), confidence: node_conf(brain, &nbr) },
                CitedEdge { from: yl.clone(), bond: *ybt, to: l.clone(), confidence: node_conf(brain, ynbr) },
            ];
            let cites = cited.iter().map(|e| e.arrow()).collect::<Vec<_>>().join("; ");
            let prose = format!("{} and {} are related through {}. ({})", capitalize_first(&xl), yl, l, cites);
            let confidence = weakest(&cited);
            return Some(Conclusion { kind: ReasoningKind::RelationalPath, prose, cited_edges: cited, confidence });
        }
    }
    Some(Conclusion {
        kind: ReasoningKind::RelationalPath,
        prose: format!("I don't find a connection between {} and {} in what I know.", xl, yl),
        cited_edges: Vec::new(),
        confidence: 0.0,
    })
}

/// Physically measurable comparatives -> objective branch (needs numeric
/// attributes, largely absent in the current brain -> honest lack-of-figures).
const OBJECTIVE_CMP: &[&str] = &[
    "bigger", "larger", "smaller", "taller", "shorter", "heavier", "lighter",
    "longer", "older", "younger", "faster", "slower", "hotter", "colder",
    "warmer", "cooler", "wider", "narrower", "deeper", "higher", "lower",
    "denser", "closer", "farther", "further", "stronger", "greater",
];
/// Value-judgement comparatives -> subjective branch (facts + decline verdict).
const SUBJECTIVE_CMP: &[&str] = &[
    "better", "best", "worse", "worst", "nicer", "nicest", "prettier", "cuter",
    "superior", "smarter", "friendlier", "safer", "healthier", "tastier",
    "happier", "easier", "harder", "cleverer",
];

fn is_cmp_word(w: &str) -> bool {
    OBJECTIVE_CMP.contains(&w) || SUBJECTIVE_CMP.contains(&w)
}

/// Extract the two compared subjects and the comparative word (if any).
fn comparative_subjects(q: &str) -> Option<(String, String, Option<String>)> {
    if let Some(ti) = q.find(" than ") {
        let left = &q[..ti];
        let right = clean_phrase(&q[ti + 6..]);
        if right.is_empty() {
            return None;
        }
        let cmp = content_tokens(left)
            .into_iter()
            .rev()
            .find(|w| is_cmp_word(w) || w.ends_with("er"));
        let x = article_operands(left).into_iter().next().or_else(|| {
            content_tokens(left).into_iter().find(|w| {
                !is_cmp_word(w)
                    && !matches!(w.as_str(), "which" | "is" | "are" | "does" | "do" | "pet" | "who")
            })
        });
        return x.map(|x| (x, right, cmp));
    }
    for lead in ["compare ", "comparison of ", "comparison between "] {
        if let Some(r) = q.strip_prefix(lead) {
            for sep in [" and ", " to ", " with ", " versus ", " vs "] {
                if let Some(j) = r.find(sep) {
                    let x = clean_phrase(&r[..j]);
                    let y = clean_phrase(&r[j + sep.len()..]);
                    if !x.is_empty() && !y.is_empty() {
                        return Some((x, y, None));
                    }
                }
            }
        }
    }
    if q.starts_with("which is ") || q.starts_with("who is ") {
        if let Some(oi) = q.find(" or ") {
            let head = &q[..oi];
            let cmp = content_tokens(head).into_iter().find(|w| is_cmp_word(w));
            let x = article_operands(head).into_iter().next().or_else(|| {
                content_tokens(head).into_iter().find(|w| {
                    !is_cmp_word(w)
                        && !matches!(w.as_str(), "which" | "who" | "is" | "are")
                })
            });
            let y = clean_phrase(&q[oi + 4..]);
            if let (Some(x), false) = (x, y.is_empty()) {
                return Some((x, y, cmp));
            }
        }
    }
    for sep in [" vs ", " versus ", " vs. "] {
        if let Some(j) = q.find(sep) {
            let x = clean_phrase(&q[..j]);
            let y = clean_phrase(&q[j + sep.len()..]);
            if !x.is_empty() && !y.is_empty() {
                return Some((x, y, None));
            }
        }
    }
    None
}

/// A concept's asserted property edges (deduped, bounded) for side-by-side facts.
fn grounded_props(brain: &Brain, seed: &Seed, name: &str, max: usize) -> Vec<CitedEdge> {
    let mut out: Vec<CitedEdge> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (nbr, bt) in brain.network.typed_edges_out(seed).into_iter().take(MAX_PROPERTY_EDGES) {
        if !is_property_bond(bt) {
            continue;
        }
        let l = strip_sense(&brain.network.node_label(&nbr).unwrap_or_default()).to_string();
        if l.is_empty() || !seen.insert(l.to_lowercase()) {
            continue;
        }
        out.push(CitedEdge { from: name.to_string(), bond: bt, to: l, confidence: node_conf(brain, &nbr) });
        if out.len() >= max {
            break;
        }
    }
    out
}

/// Numeric attributes for an objective comparison. Largely absent in the
/// current brain -> usually empty (Phase-later ingest need).
fn numeric_attrs(brain: &Brain, seed: &Seed) -> Vec<BondType> {
    let mut out: Vec<BondType> = Vec::new();
    for (_nbr, bt) in brain.network.typed_edges_out(seed).into_iter().take(MAX_PROPERTY_EDGES) {
        if matches!(bt, BondType::HasPopulation | BondType::HasArea | BondType::HasAltitude) {
            out.push(bt);
        }
    }
    out
}

// =====================================================================
// OP-COMPARE -- honest entity-vs-entity side-by-side (internal notes)
// ---------------------------------------------------------------------
// "compare X and Y" / "X vs Y" / "difference between X and Y" /
// "how are X and Y different/alike" composes an honest comparison from the
// TWO entities' REAL typed edges: it leads with what they SHARE (same
// IsA/occupation, same located-in, same currency) then the salient
// differences on the SAME relation (known-for, birth/death, capital,
// language). Every stated attribute is a real cited edge. If a subject is
// thin, a relation is one-sided, or the two are different KINDS (person vs
// place vs work), it says so honestly and NEVER invents a contrast.
//
// Runs BEFORE the concept-level `comparative` shape so an entity pair
// ("Einstein and Newton") gets the grounded entity comparison instead of the
// junk concept-property scraps `comparative` produces for entity names; a
// concept pair ("a dog and a cat") fails entity resolution here (returns
// None) and falls through to `comparative` unchanged.
// =====================================================================

/// Parse the TWO compared subjects from a comparison question form. Handles
/// connectors "and" / "vs" / "versus" / "to" / "with". None on a non-comparison
/// form. Deliberately does NOT match the subjective "X better than Y" form --
/// that stays with the concept `comparative` shape.
fn comparison_subjects(q: &str) -> Option<(String, String)> {
    for lead in ["compare ", "comparison of ", "comparison between "] {
        if let Some(r) = q.strip_prefix(lead) {
            for sep in [" versus ", " vs. ", " vs ", " and ", " with ", " to ", " against "] {
                if let Some(j) = r.find(sep) {
                    let x = clean_phrase(&r[..j]);
                    let y = clean_phrase(&r[j + sep.len()..]);
                    if !x.is_empty() && !y.is_empty() {
                        return Some((x, y));
                    }
                }
            }
        }
    }
    for m in ["difference between ", "differences between ",
              "similarities between ", "similarity between "] {
        if let Some(pr) = two_ops_after(q, m) {
            return Some(pr);
        }
    }
    // "how are/do/is X and Y different/alike/similar/compare" -- comparison
    // sense only; "how are X and Y related/connected" stays relational.
    if (q.starts_with("how are ") || q.starts_with("how do ") || q.starts_with("how is "))
        && ["different", "differ", "alike", "similar", "compare"].iter().any(|w| q.contains(w))
    {
        if let Some(pr) = two_ops_after(q, "how are ")
            .or_else(|| two_ops_after(q, "how do "))
            .or_else(|| two_ops_after(q, "how is "))
        {
            return Some(pr);
        }
    }
    for sep in [" vs. ", " vs ", " versus "] {
        if let Some(j) = q.find(sep) {
            let x = clean_phrase(&q[..j]);
            let y = clean_phrase(&q[j + sep.len()..]);
            if !x.is_empty() && !y.is_empty() {
                return Some((x, y));
            }
        }
    }
    None
}

/// Naive English pluralizer for a type noun (physicist->physicists,
/// country->countries, novel->novels). Grammar the composer is allowed to do;
/// the underlying IsA edge is real.
fn ent_pluralize(w: &str) -> String {
    let lw = w.to_lowercase();
    if lw.ends_with('s') || lw.ends_with('x') || lw.ends_with('z')
        || lw.ends_with("ch") || lw.ends_with("sh")
    {
        format!("{}es", w)
    } else if lw.ends_with('y')
        && w.chars().rev().nth(1).map(|c| !"aeiou".contains(c)).unwrap_or(false)
    {
        format!("{}ies", &w[..w.len() - 1])
    } else {
        format!("{}s", w)
    }
}

/// A subject's real typed edges relevant to comparison (each list already
/// deduped + bounded by `entity_edges`). Empty vecs => that relation absent.
struct CmpFacts {
    name: String,
    isa: Vec<CitedEdge>,
    kf: Vec<CitedEdge>,
    authored: Vec<CitedEdge>,
    born_place: Vec<CitedEdge>,
    born_date: Vec<CitedEdge>,
    died_place: Vec<CitedEdge>,
    died_date: Vec<CitedEdge>,
    located: Vec<CitedEdge>,
    capital: Vec<CitedEdge>,
    languages: Vec<CitedEdge>,
    currency: Vec<CitedEdge>,
}
impl CmpFacts {
    fn deceased(&self) -> bool { !self.died_place.is_empty() || !self.died_date.is_empty() }
    fn is_place(&self) -> bool {
        !self.capital.is_empty() || !self.languages.is_empty() || !self.currency.is_empty()
    }
    fn is_work(&self) -> bool { !self.authored.is_empty() }
    /// Coarse kind: 0 person, 1 place, 2 work. Place/work win because their
    /// signal bonds never sit on a person node.
    fn kind(&self) -> u8 { if self.is_place() { 1 } else if self.is_work() { 2 } else { 0 } }
    fn has_any(&self) -> bool {
        !self.isa.is_empty() || !self.kf.is_empty() || !self.authored.is_empty()
            || !self.born_place.is_empty() || !self.born_date.is_empty()
            || !self.died_place.is_empty() || !self.died_date.is_empty()
            || !self.located.is_empty() || !self.capital.is_empty()
            || !self.languages.is_empty() || !self.currency.is_empty()
    }
    /// A SUBSTANTIVE entity signal: at least one real attribute BEYOND a bare
    /// IsA (known-for, authorship, birth/death, or a place attribute). Gating
    /// on this keeps the comparison shape from firing on thin / mis-resolved
    /// common-noun nodes (a Wikidata "dog" or a language named "Cat" that
    /// carry only an IsA) -- those degrade here and fall through to the
    /// concept `comparative` shape unchanged.
    fn has_rich(&self) -> bool {
        !self.kf.is_empty() || !self.authored.is_empty()
            || !self.born_place.is_empty() || !self.born_date.is_empty()
            || !self.died_place.is_empty() || !self.died_date.is_empty()
            || !self.located.is_empty() || !self.capital.is_empty()
            || !self.languages.is_empty() || !self.currency.is_empty()
    }
}

fn gather_cmp_facts(brain: &Brain, seed: &Seed) -> CmpFacts {
    CmpFacts {
        name: entity_display(brain, seed),
        isa: ent_pick_pref(entity_edges(brain, seed, &[BondType::IsA], 8), ENT_PREFER_OCC, ENT_GENERIC_ISA),
        kf: entity_edges(brain, seed, &[BondType::KnownFor], 3),
        authored: entity_edges(brain, seed, &[BondType::AuthoredBy], 2),
        born_place: entity_edges(brain, seed, &[BondType::BornIn], 1),
        born_date: entity_edges(brain, seed, &[BondType::BornOn], 1),
        died_place: entity_edges(brain, seed, &[BondType::DiedIn], 1),
        died_date: entity_edges(brain, seed, &[BondType::DiedOn], 1),
        located: ent_pick(entity_edges(brain, seed, &[BondType::LocatedIn], 8), ENT_CONTINENTS),
        capital: entity_edges(brain, seed, &[BondType::HasCapital], 1),
        languages: entity_edges(brain, seed, &[BondType::HasOfficialLanguage], 2),
        currency: entity_edges(brain, seed, &[BondType::HasCurrency], 1),
    }
}

/// Type-lead sentence shared by the person + work arms: "X and Y are both
/// physicists." (equal), "X was a physicist, while Y was a mathematician."
/// (differ), or a single-sided "X is a novel." None when neither carries IsA.
fn cmp_type_lead(fx: &CmpFacts, fy: &CmpFacts, cited: &mut Vec<CitedEdge>) -> Option<String> {
    match (fx.isa.first(), fy.isa.first()) {
        (Some(ex), Some(ey)) => {
            cited.push(ex.clone());
            cited.push(ey.clone());
            if ex.to.eq_ignore_ascii_case(&ey.to) {
                let tense = if fx.deceased() && fy.deceased() { "were" } else { "are" };
                Some(format!("{} and {} {} both {}.", fx.name, fy.name, tense, ent_pluralize(&ex.to)))
            } else {
                let tx = if fx.deceased() { "was" } else { "is" };
                let ty = if fy.deceased() { "was" } else { "is" };
                Some(format!(
                    "{} {} {} {}, while {} {} {} {}.",
                    fx.name, tx, ent_article(&ex.to), ex.to, fy.name, ty, ent_article(&ey.to), ey.to
                ))
            }
        }
        (Some(ex), None) => {
            cited.push(ex.clone());
            let tx = if fx.deceased() { "was" } else { "is" };
            Some(format!("{} {} {} {}.", fx.name, tx, ent_article(&ex.to), ex.to))
        }
        (None, Some(ey)) => {
            cited.push(ey.clone());
            let ty = if fy.deceased() { "was" } else { "is" };
            Some(format!("{} {} {} {}.", fy.name, ty, ent_article(&ey.to), ey.to))
        }
        (None, None) => None,
    }
}

/// OP-COMPARE entry: honest side-by-side of two ENTITIES from shared typed
/// edges. None (degrade) on a non-comparison form, an unresolved subject, or
/// a subject with no statable edge -- never a fabricated contrast.
/// One-sided honest comparison: exactly ONE of the two named subjects
/// resolved to a substantive entity. Name what we DO hold (with a real cited
/// edge) and state plainly that the other subject is not a grounded,
/// single-sense entity we can compare against -- never dump raw edges for the
/// unresolved side, never fabricate a contrast.
fn one_sided_cmp(held: &CmpFacts, missing: &str) -> Option<Conclusion> {
    let mut cited: Vec<CitedEdge> = Vec::new();
    let mut lead = if let Some(e) = held.isa.first() {
        cited.push(e.clone());
        format!("{} is {} {}", held.name, ent_article(&e.to), e.to)
    } else if let Some(e) = held.authored.first() {
        cited.push(e.clone());
        format!("{} is a work written by {}", held.name, e.to)
    } else if let Some(e) = held.capital.first() {
        cited.push(e.clone());
        format!("{} is a place whose capital is {}", held.name, e.to)
    } else if let Some(e) = held.kf.first() {
        cited.push(e.clone());
        format!("{} is known for {}", held.name, e.to)
    } else {
        match held
            .born_place
            .first()
            .or(held.born_date.first())
            .or(held.died_place.first())
            .or(held.died_date.first())
            .or(held.located.first())
            .or(held.languages.first())
            .or(held.currency.first())
        {
            Some(e) => {
                cited.push(e.clone());
                format!("I hold grounded facts about {}", held.name)
            }
            None => return None,
        }
    };
    if !held.isa.is_empty() {
        if let Some(k) = held.kf.first() {
            cited.push(k.clone());
            lead = format!("{}, known for {}", lead, k.to);
        }
    }
    let missing_clean = ent_clean_mention(missing);
    let prose = format!(
        "{}, but I don't hold a grounded, single-sense subject for {} to compare against -- it is ambiguous or not something I hold as one distinct entity.",
        lead, missing_clean
    );
    let confidence = if cited.is_empty() { 0.0 } else { weakest(&cited) };
    Some(Conclusion { kind: ReasoningKind::Comparative, prose, cited_edges: cited, confidence })
}

fn entity_comparison(brain: &Brain, q: &str) -> Option<Conclusion> {
    let (xt, yt) = comparison_subjects(q)?;
    let xw = content_tokens(&ent_clean_mention(&xt));
    let yw = content_tokens(&ent_clean_mention(&yt));
    if xw.is_empty() || yw.is_empty() {
        return None;
    }
    // Resolve BOTH subjects through the SAME robust entity resolver the
    // who-is / overview path uses (fame-aware, accent-folded, intent-prefer,
    // FABGUARD honesty floor). A subject only COUNTS as resolved for a
    // like-for-like comparison if it also yields a SUBSTANTIVE entity
    // (has_rich: a real attribute beyond a bare IsA). A thin / ambiguous node
    // -- e.g. the place "Darwin" when the famous sense is the naturalist, which
    // the who-is path itself leaves honestly-ambiguous -- does NOT count.
    let xs = resolve_entity(brain, &xw, OVERVIEW_PREFER);
    let ys = resolve_entity(brain, &yw, OVERVIEW_PREFER);
    let fx = xs.map(|s| gather_cmp_facts(brain, &s));
    let fy = ys.map(|s| gather_cmp_facts(brain, &s));
    let x_ok = fx.as_ref().map_or(false, |f| f.has_rich());
    let y_ok = fy.as_ref().map_or(false, |f| f.has_rich());
    // Neither side is a substantive entity: a concept / thin common-noun pair
    // ("dog vs cat"). Degrade so the concept `comparative` shape handles it
    // unchanged -- no entity-comparison hijack.
    if !x_ok && !y_ok {
        return None;
    }
    // EXACTLY one side is a substantive entity. The query clearly names a real
    // entity, so this IS an entity-comparison intent -- but there is no honest
    // like-for-like grid to build. Say so plainly (naming what we DO hold, with
    // a real cited edge) instead of returning None and letting the weak
    // comparative() fallback dump raw associations for the unresolved side.
    if x_ok != y_ok {
        let (held, missing) = if x_ok {
            (fx.as_ref().unwrap(), &yt)
        } else {
            (fy.as_ref().unwrap(), &xt)
        };
        return one_sided_cmp(held, missing);
    }
    // Both sides substantive. Same entity on both sides -> nothing to contrast.
    if xs == ys {
        return None;
    }
    let fx = fx.unwrap();
    let fy = fy.unwrap();
    let (x, y) = (fx.name.clone(), fy.name.clone());
    let mut cited: Vec<CitedEdge> = Vec::new();

    // Different KINDS -> not like-for-like. State honestly what each IS from a
    // real edge; never force a contrast on relations they don't share.
    if fx.kind() != fy.kind() {
        let desc = |f: &CmpFacts, cited: &mut Vec<CitedEdge>| -> Option<String> {
            if let Some(e) = f.isa.first() {
                cited.push(e.clone());
                Some(format!("{} is {} {}", f.name, ent_article(&e.to), e.to))
            } else if let Some(e) = f.authored.first() {
                cited.push(e.clone());
                Some(format!("{} is a work written by {}", f.name, e.to))
            } else if let Some(e) = f.capital.first() {
                cited.push(e.clone());
                Some(format!("{} is a place whose capital is {}", f.name, e.to))
            } else {
                None
            }
        };
        let dx = desc(&fx, &mut cited);
        let dy = desc(&fy, &mut cited);
        return match (dx, dy) {
            (Some(a), Some(b)) => {
                let prose = format!(
                    "{} and {} aren't really comparable on what I hold -- {}, while {}.",
                    x, y, a, b
                );
                let confidence = weakest(&cited);
                Some(Conclusion { kind: ReasoningKind::Comparative, prose, cited_edges: cited, confidence })
            }
            _ => None,
        };
    }

    let mut sentences: Vec<String> = Vec::new();

    if fx.is_place() && fy.is_place() {
        // ---- PLACE vs PLACE: shared lead (type + located + shared currency),
        //      then per-side capital/language diffs; currency diff only if not
        //      shared. All woven from real edges; absent attrs omitted.
        // Shared-first lead, always grammatical: fold a shared type, shared
        // located-in, and shared currency into ONE sentence with a real verb.
        // A place pair whose IsA labels differ ("country" vs "country
        // bordering the Baltic Sea") still gets a grammatical shared lead from
        // whatever they DO share -- never a verbless "X and Y located in ...".
        let shared_type = match (fx.isa.first(), fy.isa.first()) {
            (Some(ex), Some(ey)) if ex.to.eq_ignore_ascii_case(&ey.to) => {
                cited.push(ex.clone());
                cited.push(ey.clone());
                Some(ent_pluralize(&ex.to))
            }
            _ => None,
        };
        let shared_loc = match (fx.located.first(), fy.located.first()) {
            (Some(lx), Some(ly)) if lx.to.eq_ignore_ascii_case(&ly.to) => {
                cited.push(lx.clone());
                cited.push(ly.clone());
                Some(lx.to.clone())
            }
            _ => None,
        };
        let shared_cur = match (fx.currency.first(), fy.currency.first()) {
            (Some(cx), Some(cy)) if cx.to.eq_ignore_ascii_case(&cy.to) => {
                cited.push(cx.clone());
                cited.push(cy.clone());
                Some(cx.to.clone())
            }
            _ => None,
        };
        let mut clause = match (&shared_type, &shared_loc) {
            (Some(t), Some(l)) => format!("are both {} located in {}", t, l),
            (Some(t), None) => format!("are both {}", t),
            (None, Some(l)) => format!("are both located in {}", l),
            (None, None) => String::new(),
        };
        if let Some(c) = &shared_cur {
            if clause.is_empty() {
                clause = format!("both use the {}", c);
            } else {
                clause = format!("{} and use the {}", clause, c);
            }
        }
        if !clause.is_empty() {
            sentences.push(format!("{} and {} {}.", x, y, clause));
        }

        let side = |f: &CmpFacts, cited: &mut Vec<CitedEdge>| -> Option<String> {
            let cap = f.capital.first();
            let lang = f.languages.first();
            match (cap, lang) {
                (Some(c), Some(l)) => {
                    cited.push(c.clone());
                    cited.push(l.clone());
                    Some(format!("capital is {} and its official language is {}", c.to, l.to))
                }
                (Some(c), None) => { cited.push(c.clone()); Some(format!("capital is {}", c.to)) }
                (None, Some(l)) => { cited.push(l.clone()); Some(format!("official language is {}", l.to)) }
                (None, None) => None,
            }
        };
        let dx = side(&fx, &mut cited);
        let dy = side(&fy, &mut cited);
        match (dx, dy) {
            (Some(a), Some(b)) => sentences.push(format!("{}'s {}, while {}'s {}.", x, a, y, b)),
            (Some(a), None) => sentences.push(format!("{}'s {}.", x, a)),
            (None, Some(b)) => sentences.push(format!("{}'s {}.", y, b)),
            (None, None) => {}
        }
        if shared_cur.is_none() {
            if let (Some(cx), Some(cy)) = (fx.currency.first(), fy.currency.first()) {
                sentences.push(format!("{} uses the {}, {} the {}.", x, cx.to, y, cy.to));
                cited.push(cx.clone());
                cited.push(cy.clone());
            }
        }
    } else if fx.is_work() && fy.is_work() {
        // ---- WORK vs WORK: type lead + authorship (shared or per-side).
        if let Some(s) = cmp_type_lead(&fx, &fy, &mut cited) {
            sentences.push(s);
        }
        match (fx.authored.first(), fy.authored.first()) {
            (Some(ax), Some(ay)) => {
                if ax.to.eq_ignore_ascii_case(&ay.to) {
                    sentences.push(format!("Both were written by {}.", ax.to));
                } else {
                    sentences.push(format!("{} was written by {}, {} by {}.", x, ax.to, y, ay.to));
                }
                cited.push(ax.clone());
                cited.push(ay.clone());
            }
            _ => {}
        }
    } else {
        // ---- PERSON vs PERSON: type lead + known-for diff + lifespan diff.
        if let Some(s) = cmp_type_lead(&fx, &fy, &mut cited) {
            sentences.push(s);
        }
        match (fx.kf.first(), fy.kf.first()) {
            (Some(_), Some(_)) => {
                let kx: Vec<String> = fx.kf.iter().take(2).map(|e| e.to.clone()).collect();
                let ky: Vec<String> = fy.kf.iter().take(2).map(|e| e.to.clone()).collect();
                sentences.push(format!(
                    "{} is known for {}, while {} is known for {}.",
                    x, join_list(&kx), y, join_list(&ky)
                ));
                cited.extend(fx.kf.iter().take(2).cloned());
                cited.extend(fy.kf.iter().take(2).cloned());
            }
            (Some(_), None) => {
                let kx: Vec<String> = fx.kf.iter().take(2).map(|e| e.to.clone()).collect();
                sentences.push(format!(
                    "{} is known for {}; I don't hold what {} is known for.",
                    x, join_list(&kx), y
                ));
                cited.extend(fx.kf.iter().take(2).cloned());
            }
            (None, Some(_)) => {
                let ky: Vec<String> = fy.kf.iter().take(2).map(|e| e.to.clone()).collect();
                sentences.push(format!(
                    "{} is known for {}; I don't hold what {} is known for.",
                    y, join_list(&ky), x
                ));
                cited.extend(fy.kf.iter().take(2).cloned());
            }
            (None, None) => {}
        }
        let span = |f: &CmpFacts, cited: &mut Vec<CitedEdge>| -> Option<String> {
            let mut parts: Vec<String> = Vec::new();
            match (f.born_place.first(), f.born_date.first()) {
                (Some(p), Some(d)) => { parts.push(format!("born in {} in {}", p.to, d.to)); cited.push(p.clone()); cited.push(d.clone()); }
                (Some(p), None) => { parts.push(format!("born in {}", p.to)); cited.push(p.clone()); }
                (None, Some(d)) => { parts.push(format!("born in {}", d.to)); cited.push(d.clone()); }
                (None, None) => {}
            }
            match (f.died_place.first(), f.died_date.first()) {
                (Some(p), Some(d)) => { parts.push(format!("died in {} in {}", p.to, d.to)); cited.push(p.clone()); cited.push(d.clone()); }
                (Some(p), None) => { parts.push(format!("died in {}", p.to)); cited.push(p.clone()); }
                (None, Some(d)) => { parts.push(format!("died in {}", d.to)); cited.push(d.clone()); }
                (None, None) => {}
            }
            if parts.is_empty() { None } else { Some(parts.join(" and ")) }
        };
        let sx = span(&fx, &mut cited);
        let sy = span(&fy, &mut cited);
        match (sx, sy) {
            (Some(a), Some(b)) => sentences.push(format!("{} was {}; {} was {}.", x, a, y, b)),
            (Some(a), None) => sentences.push(format!("{} was {}.", x, a)),
            (None, Some(b)) => sentences.push(format!("{} was {}.", y, b)),
            (None, None) => {}
        }
    }

    if cited.is_empty() || sentences.is_empty() {
        return None;
    }
    let prose = sentences.join(" ");
    let confidence = weakest(&cited);
    Some(Conclusion { kind: ReasoningKind::Comparative, prose, cited_edges: cited, confidence })
}
/// Comparative (integrity shape). Subjective ("better"/"compare"): lay out
/// each one's GROUNDED facts side by side, then decline the value judgement --
/// "better" depends on what the person values; no grounded basis to pick a
/// winner. Objective ("bigger"/"older"): needs numeric attributes, absent
/// today -> honest lack-of-figures. Never a subjective verdict, never a guess.
fn comparative(brain: &Brain, q: &str) -> Option<Conclusion> {
    let (xt, yt, cmp) = comparative_subjects(q)?;
    let xs = resolve_subject(brain, &xt);
    let ys = resolve_subject(brain, &yt);
    if xs.is_none() && ys.is_none() {
        return None;
    }
    let xl = xs
        .and_then(|sd| brain.network.node_label(&sd))
        .map(|l| strip_sense(&l).to_string())
        .unwrap_or_else(|| xt.clone());
    let yl = ys
        .and_then(|sd| brain.network.node_label(&sd))
        .map(|l| strip_sense(&l).to_string())
        .unwrap_or_else(|| yt.clone());
    let objective = cmp.as_deref().map(|c| OBJECTIVE_CMP.contains(&c)).unwrap_or(false);
    if objective {
        let attr = cmp.clone().unwrap_or_else(|| "different".to_string());
        let xnum = xs.map(|sd| numeric_attrs(brain, &sd)).unwrap_or_default();
        let ynum = ys.map(|sd| numeric_attrs(brain, &sd)).unwrap_or_default();
        let _shared = xnum.iter().any(|b| ynum.contains(b));
        return Some(Conclusion {
            kind: ReasoningKind::Comparative,
            prose: format!(
                "I can't say which is {} -- I don't hold the numeric figures for {} and {} to compare them on that. (Objective comparisons need measured attributes this brain doesn't yet carry.)",
                attr, xl, yl
            ),
            cited_edges: Vec::new(),
            confidence: 0.0,
        });
    }
    let xp = xs.map(|sd| grounded_props(brain, &sd, &xl, CMP_PROP_MAX)).unwrap_or_default();
    let yp = ys.map(|sd| grounded_props(brain, &sd, &yl, CMP_PROP_MAX)).unwrap_or_default();
    if xp.is_empty() && yp.is_empty() {
        // Bare "compare X and Y" / "X vs Y" with no grounded facts -> defer
        // to the existing narration Comparison shape (shared category, etc.).
        // Explicit value/objective comparatives keep firing terminally so the
        // integrity guarantee holds even when the facts are thin.
        if cmp.is_none() {
            return None;
        }
        return Some(Conclusion {
            kind: ReasoningKind::Comparative,
            prose: format!("I don't hold enough grounded facts about {} or {} to compare them.", xl, yl),
            cited_edges: Vec::new(),
            confidence: 0.0,
        });
    }
    let xnames: Vec<String> = xp.iter().map(|e| e.to.clone()).collect();
    let ynames: Vec<String> = yp.iter().map(|e| e.to.clone()).collect();
    let quality = cmp
        .clone()
        .filter(|c| SUBJECTIVE_CMP.contains(&c.as_str()))
        .unwrap_or_else(|| "better".to_string());
    let xpart = if xnames.is_empty() {
        format!("I don't hold grounded facts about {}", xl)
    } else {
        format!("{} is {}", capitalize_first(&xl), join_list(&xnames))
    };
    let ypart = if ynames.is_empty() {
        format!("I don't hold grounded facts about {}", yl)
    } else {
        format!("{} is {}", capitalize_first(&yl), join_list(&ynames))
    };
    let mut cited = xp;
    cited.extend(yp);
    let prose = format!(
        "{}. {}. Which is \"{}\" depends on what you value -- I have no grounded basis to pick a winner.",
        xpart, ypart, quality
    );
    let confidence = if cited.is_empty() { 0.0 } else { weakest(&cited) };
    Some(Conclusion { kind: ReasoningKind::Comparative, prose, cited_edges: cited, confidence })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Confidence;

    fn ctx_for(brain: &mut Brain, query: &str) -> ContextSet {
        let ask = brain.ask(query, 3, 20);
        ContextSet::from_ask(&ask, 0, 0, 64)
    }

    #[test]
    fn entity_who_fires() {
        let mut b = Brain::new();
        b.teach("Einstein#Q937", BondType::IsA, "physicist", 0.7, Confidence::Unverified);
        b.teach("Einstein#Q937", BondType::KnownFor, "relativity", 0.7, Confidence::Unverified);
        let c = try_synthesize(&b, "who is Einstein", QueryIntent::Generic, None)
            .expect("entity who should fire");
        assert_eq!(c.kind, ReasoningKind::EntityFact);
        assert!(c.prose.to_lowercase().contains("physicist"), "prose: {}", c.prose);
        assert!(c.prose.to_lowercase().contains("relativity"), "prose: {}", c.prose);
    }

    #[test]
    fn entity_where_fires_concept_untouched() {
        let mut b = Brain::new();
        b.teach("Paris#Q90", BondType::LocatedIn, "France#Q142", 0.7, Confidence::Unverified);
        b.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "where is Paris", QueryIntent::Generic, None)
            .expect("entity where should fire");
        assert_eq!(c.kind, ReasoningKind::EntityFact);
        assert!(c.prose.contains("France"), "prose: {}", c.prose);
        let d = try_synthesize(&b, "is a dog loyal", QueryIntent::Generic, None)
            .expect("concept property still fires");
        assert_eq!(d.kind, ReasoningKind::PropertyCheck);
    }

    #[test]
    fn entity_what_is_defers_to_concept() {
        let mut b = Brain::new();
        b.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        b.teach("dog#Q144", BondType::IsA, "species", 0.7, Confidence::Unverified);
        assert!(entity_query(&b, "what is a dog").is_none());
    }


    #[test]
    fn reason_chain_born_country_composes() {
        let mut b = Brain::new();
        b.teach("Ludwig van Beethoven#Q255", BondType::BornIn, "Bonn#Q586", 0.9, Confidence::Verified);
        b.teach("Bonn#Q586", BondType::LocatedIn, "Germany#Q183", 0.9, Confidence::Verified);
        let c = entity_query(&b, "what country was Beethoven born in")
            .expect("born-country 2-hop should compose");
        assert_eq!(c.kind, ReasoningKind::RelationalPath);
        assert!(c.prose.contains("Germany"), "prose: {}", c.prose);
        assert!(c.prose.contains("Bonn"), "prose: {}", c.prose);
        assert_eq!(c.cited_edges.len(), 2);
        assert_eq!(c.cited_edges[0].bond, BondType::BornIn);
        assert_eq!(c.cited_edges[1].bond, BondType::LocatedIn);
    }

    #[test]
    fn reason_chain_missing_link_refuses() {
        // BornIn present but the city has NO LocatedIn edge -> must refuse,
        // never invent a country (honesty floor: no fabricated transitive link).
        let mut b = Brain::new();
        b.teach("Albert Einstein#Q937", BondType::BornIn, "Ulm#Q3012", 0.9, Confidence::Verified);
        assert!(entity_query(&b, "what country was Einstein born in").is_none());
    }

    #[test]
    fn reason_chain_no_first_hop_refuses() {
        // No BornIn edge at all -> refuse (never fabricate a chain).
        let mut b = Brain::new();
        b.teach("Bonn#Q586", BondType::LocatedIn, "Germany#Q183", 0.9, Confidence::Verified);
        assert!(entity_query(&b, "what country was Beethoven born in").is_none());
    }

    #[test]
    fn syllogism_inherited_property_concludes() {
        let mut brain = Brain::new();
        brain.teach("socrates", BondType::IsA, "man", 0.9, Confidence::Verified);
        brain.teach("man", BondType::HasProperty, "mortal", 0.9, Confidence::Verified);
        let q = "is socrates mortal";
        let ctx = ctx_for(&mut brain, q);
        let c = try_synthesize(&brain, q, QueryIntent::Generic, Some(&ctx))
            .expect("should conclude socrates is mortal");
        assert!(c.prose.starts_with("Yes"), "prose: {}", c.prose);
        assert!(c.prose.to_lowercase().contains("mortal"));
        assert!(c.prose.to_lowercase().contains("man"));
        assert_eq!(c.cited_edges.len(), 2);
        assert_eq!(c.cited_edges[0].bond, BondType::IsA);
        assert_eq!(c.cited_edges[1].bond, BondType::HasProperty);
    }

    #[test]
    fn direct_property_concludes() {
        let mut brain = Brain::new();
        brain.teach("snow", BondType::HasProperty, "white", 0.9, Confidence::Verified);
        let q = "is snow white";
        let ctx = ctx_for(&mut brain, q);
        let c = try_synthesize(&brain, q, QueryIntent::Generic, Some(&ctx))
            .expect("should conclude snow is white");
        assert!(c.prose.to_lowercase().contains("white"));
        assert_eq!(c.cited_edges.len(), 1);
        assert_eq!(c.cited_edges[0].bond, BondType::HasProperty);
    }

    #[test]
    fn missing_property_edge_refuses() {
        // IS-A present but no property edge -> no conclusion (degrade).
        let mut brain = Brain::new();
        brain.teach("socrates", BondType::IsA, "man", 0.9, Confidence::Verified);
        let q = "is socrates mortal";
        let ctx = ctx_for(&mut brain, q);
        assert!(try_synthesize(&brain, q, QueryIntent::Generic, Some(&ctx)).is_none());
    }

    #[test]
    fn non_reasoning_shape_returns_none() {
        let mut brain = Brain::new();
        brain.teach("socrates", BondType::IsA, "man", 0.9, Confidence::Verified);
        brain.teach("man", BondType::HasProperty, "mortal", 0.9, Confidence::Verified);
        // A definition-style query is not a yes/no reasoning shape.
        let q = "what is socrates";
        let ctx = ctx_for(&mut brain, q);
        assert!(try_synthesize(&brain, q, QueryIntent::Definition, Some(&ctx)).is_none());
    }

fn isa_brain() -> Brain {
        let mut b = Brain::new();
        let c = Confidence::Verified;
        b.teach("dog", BondType::IsA, "canine", 0.9, c);
        b.teach("canine", BondType::IsA, "carnivore", 0.9, c);
        b.teach("carnivore", BondType::IsA, "mammal", 0.9, c);
        b.teach("mammal", BondType::IsA, "animal", 0.9, c);
        b.teach("animal", BondType::IsA, "organism", 0.9, c);
        b.teach("cat", BondType::IsA, "feline", 0.9, c);
        b.teach("feline", BondType::IsA, "carnivore", 0.9, c);
        b.teach("whale", BondType::IsA, "mammal", 0.9, c);
        b.teach("fish", BondType::IsA, "animal", 0.9, c);
        b.teach("rose", BondType::IsA, "flower", 0.9, c);
        b.teach("flower", BondType::IsA, "plant", 0.9, c);
        b.teach("plant", BondType::IsA, "organism", 0.9, c);
        b
    }

    #[test]
    fn category_membership_yes() {
        let brain = isa_brain();
        let c = try_synthesize(&brain, "is a dog an animal", QueryIntent::Generic, None)
            .expect("dog is an animal");
        assert!(c.prose.starts_with("Yes"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("animal"));
        assert!(!c.cited_edges.is_empty());
        assert_eq!(c.kind, ReasoningKind::CategoryMembership);
    }

    #[test]
    fn category_membership_no_disjoint() {
        let brain = isa_brain();
        let c = try_synthesize(&brain, "is a whale a fish", QueryIntent::Generic, None)
            .expect("whale vs fish");
        assert!(c.prose.starts_with("No"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("fish"));
    }

    #[test]
    fn category_membership_no_plant() {
        let brain = isa_brain();
        let c = try_synthesize(&brain, "is a dog a plant", QueryIntent::Generic, None)
            .expect("dog vs plant");
        assert!(c.prose.starts_with("No"), "{}", c.prose);
    }

    #[test]
    fn common_ancestor_fires() {
        let brain = isa_brain();
        let c = try_synthesize(
            &brain,
            "what do a dog and a cat have in common",
            QueryIntent::Generic,
            None,
        )
        .expect("dog & cat");
        assert!(c.prose.to_lowercase().contains("carnivore"), "{}", c.prose);
        assert_eq!(c.kind, ReasoningKind::CommonAncestor);
    }

    #[test]
    fn causal_why_effect_fires() {
        let mut brain = Brain::new();
        brain.teach("sulfur dioxide", BondType::Causes, "acid rain", 0.9, Confidence::Verified);
        brain.teach("air pollution", BondType::LeadsTo, "acid rain", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "what causes acid rain", QueryIntent::Generic, None)
            .expect("acid rain cause should fire");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("caused by"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("sulfur dioxide"));
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn causal_why_cause_fires() {
        let mut brain = Brain::new();
        brain.teach("virus", BondType::Causes, "fever", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "what does a virus cause", QueryIntent::Generic, None)
            .expect("virus effect should fire");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("fever"), "{}", c.prose);
    }

    #[test]
    fn causal_verify_present_fires() {
        let mut brain = Brain::new();
        brain.teach("cigarette", BondType::Causes, "cancer", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "does a cigarette cause cancer", QueryIntent::Generic, None)
            .expect("cigarette->cancer should fire");
        assert!(c.prose.starts_with("Yes"), "{}", c.prose);
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
    }

    #[test]
    fn causal_verify_absent_refuses_no_fabrication() {
        let mut brain = Brain::new();
        // vaccine HAS a causal edge (fever) but NOT autism -> must not fabricate.
        brain.teach("vaccine", BondType::Causes, "fever", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "does a vaccine cause autism", QueryIntent::Generic, None);
        if let Some(c) = c {
            assert!(!c.prose.starts_with("Yes"), "must not fabricate: {}", c.prose);
            assert!(c.cited_edges.is_empty());
        }
    }

    #[test]
    fn causal_multifactor_fires_no_single_cause() {
        let mut brain = Brain::new();
        for c in ["alcohol", "asbestos", "cigarette", "radiation"] {
            brain.teach(c, BondType::Causes, "cancer", 0.9, Confidence::Inferred);
        }
        let c = try_synthesize(&brain, "what causes cancer", QueryIntent::Generic, None)
            .expect("multifactor cancer should fire");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("single cause")
            && c.prose.to_lowercase().contains("multifactorial"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("contributing factors"));
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn causal_single_factor_unchanged_below_threshold() {
        let mut brain = Brain::new();
        brain.teach("sulfur dioxide", BondType::Causes, "acid rain", 0.9, Confidence::Inferred);
        brain.teach("air pollution", BondType::LeadsTo, "acid rain", 0.9, Confidence::Inferred);
        let c = try_synthesize(&brain, "what causes acid rain", QueryIntent::Generic, None)
            .expect("acid rain should fire");
        // n = 2 < MULTI_CAUSE_MIN -> single-factor phrasing, unchanged.
        assert!(c.prose.to_lowercase().contains("caused by"), "{}", c.prose);
        assert!(!c.prose.to_lowercase().contains("multifactorial"), "{}", c.prose);
    }

    #[test]
    fn causal_multifactor_discloses_residual() {
        let mut brain = Brain::new();
        for i in 0..9 {
            brain.teach(&format!("factor{:02}", i), BondType::Causes, "death", 0.9, Confidence::Inferred);
        }
        let c = try_synthesize(&brain, "what causes death", QueryIntent::Generic, None)
            .expect("death multifactor should fire");
        assert!(c.prose.to_lowercase().contains("multifactorial"), "{}", c.prose);
        // 9 causes, 6 shown -> "(and 3 others)"
        assert!(c.prose.contains("others"), "{}", c.prose);
        assert_eq!(c.cited_edges.len(), 6);
    }

    #[test]
    fn causal_why_link_confirms_direct_edge() {
        let mut brain = Brain::new();
        brain.teach("cigarette", BondType::Causes, "cancer", 0.9, Confidence::Inferred);
        let c = try_synthesize(&brain, "why does a cigarette cause cancer", QueryIntent::Generic, None)
            .expect("why-link should confirm cigarette->cancer");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("on record as a cause of"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("mechanism"), "{}", c.prose);
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn causal_why_link_transitive_chain_fires() {
        let mut brain = Brain::new();
        brain.teach("virus", BondType::Causes, "disease", 0.9, Confidence::Inferred);
        brain.teach("disease", BondType::LeadsTo, "death", 0.9, Confidence::Inferred);
        let c = try_synthesize(&brain, "why does a virus cause death", QueryIntent::Generic, None)
            .expect("why-link transitive virus->disease->death should fire");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("in turn"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("disease") && c.prose.to_lowercase().contains("death"), "{}", c.prose);
        assert_eq!(c.cited_edges.len(), 2);
    }

    #[test]
    fn causal_why_link_refuses_no_fabrication() {
        let mut brain = Brain::new();
        // cigarette causes cancer, NOT happiness -> must not fabricate a why.
        brain.teach("cigarette", BondType::Causes, "cancer", 0.9, Confidence::Inferred);
        let c = try_synthesize(&brain, "why does a cigarette cause happiness", QueryIntent::Generic, None)
            .expect("why-link should terminate, not fall through");
        assert_eq!(c.kind, ReasoningKind::CausalWhy);
        assert!(c.prose.to_lowercase().contains("no causal link"), "{}", c.prose);
        assert!(c.cited_edges.is_empty());
    }

    #[test]
    fn definition_query_still_none() {
        let brain = isa_brain();
        assert!(try_synthesize(&brain, "what is a dog", QueryIntent::Definition, None).is_none());
    }

    #[test]
    fn property_check_direct_fires() {
        let mut brain = Brain::new();
        brain.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "is a dog loyal", QueryIntent::Generic, None)
            .expect("dog is loyal should fire");
        assert_eq!(c.kind, ReasoningKind::PropertyCheck);
        assert!(c.prose.starts_with("Yes"), "prose: {}", c.prose);
        assert!(c.prose.to_lowercase().contains("loyal"));
        assert_eq!(c.cited_edges.len(), 1);
        assert_eq!(c.cited_edges[0].bond, BondType::HasProperty);
    }

    #[test]
    fn property_check_absent_refuses_no_fabrication() {
        let mut brain = Brain::new();
        brain.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        // No dog -HasProperty-> purple edge: must NOT fabricate. Either a
        // degrade (None) or an honest refusal (no "Yes", no cited edge).
        if let Some(c) = try_synthesize(&brain, "is a dog purple", QueryIntent::Generic, None) {
            assert!(!c.prose.starts_with("Yes"), "must not fabricate: {}", c.prose);
            assert!(c.cited_edges.is_empty(), "refusal cites no edge");
            assert!(!c.prose.to_lowercase().contains("purple") || c.prose.contains("no 'purple'"));
        }
    }

    #[test]
    fn property_check_capability_fires() {
        let mut brain = Brain::new();
        brain.teach("bird", BondType::HasProperty, "fly", 0.9, Confidence::Verified);
        let c = try_synthesize(&brain, "does a bird fly", QueryIntent::Generic, None)
            .expect("bird fly should fire");
        assert_eq!(c.kind, ReasoningKind::PropertyCheck);
        assert!(c.prose.to_lowercase().contains("fly"), "prose: {}", c.prose);
    }

    #[test]
    fn property_check_wrong_property_refuses() {
        let mut brain = Brain::new();
        brain.teach("fish", BondType::HasProperty, "scales", 0.9, Confidence::Verified);
        // fish has scales, NOT wings -> must NOT invent wings (honest refusal
        // or degrade; never a fabricated Yes).
        if let Some(c) = try_synthesize(&brain, "does a fish have wings", QueryIntent::Generic, None) {
            assert!(!c.prose.starts_with("Yes"), "must not fabricate wings: {}", c.prose);
            assert!(c.cited_edges.is_empty());
        }
        // sanity: the held property DOES fire.
        let c = try_synthesize(&brain, "does a fish have scales", QueryIntent::Generic, None)
            .expect("fish scales should fire");
        assert!(c.prose.starts_with("Yes"));
        assert!(c.prose.to_lowercase().contains("scales"), "prose: {}", c.prose);
    }
    // ---------------- Tier 2 shape tests ----------------

    #[test]
    fn composition_fires_incoming_partof() {
        let mut b = Brain::new();
        b.teach("hydrogen", BondType::PartOf, "water", 0.9, Confidence::Verified);
        b.teach("oxygen", BondType::PartOf, "water", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is water made of", QueryIntent::Generic, None)
            .expect("water composition should fire");
        assert_eq!(c.kind, ReasoningKind::Composition);
        assert!(c.prose.to_lowercase().contains("made of"), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("hydrogen") && c.prose.to_lowercase().contains("oxygen"), "{}", c.prose);
        assert!(c.cited_edges.len() >= 2);
    }

    #[test]
    fn composition_fires_composedof() {
        let mut b = Brain::new();
        b.teach("bronze", BondType::ComposedOf, "copper", 0.9, Confidence::Verified);
        b.teach("bronze", BondType::ComposedOf, "tin", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is bronze made of", QueryIntent::Generic, None)
            .expect("bronze composition should fire");
        assert_eq!(c.kind, ReasoningKind::Composition);
        assert!(c.prose.to_lowercase().contains("copper") && c.prose.to_lowercase().contains("tin"), "{}", c.prose);
    }

    #[test]
    fn composition_absent_refuses_no_fabrication() {
        let mut b = Brain::new();
        b.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        if let Some(c) = try_synthesize(&b, "what is a dog made of", QueryIntent::Generic, None) {
            assert!(c.cited_edges.is_empty(), "refusal cites no edge: {}", c.prose);
            assert!(!c.prose.to_lowercase().contains("made of animal"), "{}", c.prose);
        }
    }

    #[test]
    fn purpose_fires() {
        let mut b = Brain::new();
        b.teach("hammer", BondType::UsedFor, "driving nails", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is a hammer used for", QueryIntent::Generic, None)
            .expect("hammer purpose should fire");
        assert_eq!(c.kind, ReasoningKind::Purpose);
        assert!(c.prose.to_lowercase().contains("driving nails"), "{}", c.prose);
        assert_eq!(c.cited_edges[0].bond, BondType::UsedFor);
    }

    #[test]
    fn purpose_absent_refuses_no_fabrication() {
        let mut b = Brain::new();
        b.teach("knife", BondType::HasProperty, "sharp", 0.9, Confidence::Verified);
        if let Some(c) = try_synthesize(&b, "what is a knife used for", QueryIntent::Generic, None) {
            assert!(c.cited_edges.is_empty(), "refusal cites no edge: {}", c.prose);
            assert!(!c.prose.to_lowercase().contains("used for sharp"), "{}", c.prose);
        }
    }

    #[test]
    fn meronym_parts_fires() {
        let mut b = Brain::new();
        b.teach("wheel", BondType::PartOf, "car", 0.9, Confidence::Verified);
        b.teach("engine", BondType::PartOf, "car", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what are the parts of a car", QueryIntent::Generic, None)
            .expect("car parts should fire");
        assert_eq!(c.kind, ReasoningKind::Meronymy);
        assert!(c.prose.to_lowercase().contains("wheel") && c.prose.to_lowercase().contains("engine"), "{}", c.prose);
        assert!(c.cited_edges.len() >= 2);
    }

    #[test]
    fn holonym_part_of_fires() {
        let mut b = Brain::new();
        b.teach("hydrogen", BondType::PartOf, "water", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is hydrogen part of", QueryIntent::Generic, None)
            .expect("hydrogen holonym should fire");
        assert_eq!(c.kind, ReasoningKind::Meronymy);
        assert!(c.prose.to_lowercase().contains("part of") && c.prose.to_lowercase().contains("water"), "{}", c.prose);
    }

    #[test]
    fn meronym_absent_refuses() {
        let mut b = Brain::new();
        b.teach("rock", BondType::IsA, "mineral", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what are the parts of a rock", QueryIntent::Generic, None)
            .expect("resolved subject -> honest refusal");
        assert!(c.cited_edges.is_empty(), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("no parts"), "{}", c.prose);
    }

    #[test]
    fn relational_direct_edge_fires() {
        let mut b = Brain::new();
        b.teach("hydrogen", BondType::PartOf, "water", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is the relationship between hydrogen and water", QueryIntent::Generic, None)
            .expect("direct relation should fire");
        assert_eq!(c.kind, ReasoningKind::RelationalPath);
        assert!(c.prose.to_lowercase().contains("part of"), "{}", c.prose);
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn relational_shared_ancestor_fires() {
        let mut b = Brain::new();
        b.teach("dog", BondType::IsA, "canine", 0.9, Confidence::Verified);
        b.teach("wolf", BondType::IsA, "canine", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is the relationship between a dog and a wolf", QueryIntent::Generic, None)
            .expect("shared ancestor should fire");
        assert_eq!(c.kind, ReasoningKind::RelationalPath);
        assert!(c.prose.to_lowercase().contains("canine"), "{}", c.prose);
    }

    #[test]
    fn relational_no_connection_honest() {
        let mut b = Brain::new();
        b.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        b.teach("rock", BondType::IsA, "mineral", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "what is the relationship between a dog and a rock", QueryIntent::Generic, None)
            .expect("resolved pair -> honest no-connection");
        assert!(c.cited_edges.is_empty(), "{}", c.prose);
        assert!(c.prose.to_lowercase().contains("don't find a connection"), "{}", c.prose);
    }

    #[test]
    fn comparative_subjective_no_verdict() {
        let mut b = Brain::new();
        b.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        b.teach("dog", BondType::HasProperty, "trainable", 0.9, Confidence::Verified);
        b.teach("fish", BondType::HasProperty, "aquatic", 0.9, Confidence::Verified);
        b.teach("fish", BondType::HasProperty, "quiet", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "is a dog a better pet than a fish", QueryIntent::Generic, None)
            .expect("comparative should fire");
        assert_eq!(c.kind, ReasoningKind::Comparative);
        let lo = c.prose.to_lowercase();
        assert!(lo.contains("loyal") && lo.contains("aquatic"), "facts side by side: {}", c.prose);
        assert!(lo.contains("depends on what you value"), "{}", c.prose);
        // integrity: NO subjective verdict.
        assert!(!lo.contains("better than") && !lo.contains("is better") && !lo.starts_with("yes"), "must not pick a winner: {}", c.prose);
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn comparative_objective_lacks_figures() {
        let mut b = Brain::new();
        b.teach("whale", BondType::IsA, "mammal", 0.9, Confidence::Verified);
        b.teach("elephant", BondType::IsA, "mammal", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "is a whale bigger than an elephant", QueryIntent::Generic, None)
            .expect("objective comparative should fire");
        assert_eq!(c.kind, ReasoningKind::Comparative);
        let lo = c.prose.to_lowercase();
        assert!(lo.contains("numeric figures"), "{}", c.prose);
        assert!(lo.contains("bigger"), "{}", c.prose);
        assert!(c.cited_edges.is_empty(), "no fabricated verdict: {}", c.prose);
    }

    #[test]
    fn comparative_compare_form_fires() {
        let mut b = Brain::new();
        b.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        b.teach("cat", BondType::HasProperty, "independent", 0.9, Confidence::Verified);
        let c = try_synthesize(&b, "compare a dog and a cat", QueryIntent::Generic, None)
            .expect("compare form should fire");
        assert_eq!(c.kind, ReasoningKind::Comparative);
        assert!(c.prose.to_lowercase().contains("depends on what you value"), "{}", c.prose);
    }

#[test]
    fn entity_comparison_person_pair_fires() {
        let mut b = Brain::new();
        b.teach("Alan#Q1", BondType::IsA, "physicist", 0.7, Confidence::Unverified);
        b.teach("Alan#Q1", BondType::KnownFor, "wave theory", 0.7, Confidence::Unverified);
        b.teach("Alan#Q1", BondType::BornOn, "1900", 0.7, Confidence::Unverified);
        b.teach("Beth#Q2", BondType::IsA, "physicist", 0.7, Confidence::Unverified);
        b.teach("Beth#Q2", BondType::KnownFor, "field theory", 0.7, Confidence::Unverified);
        b.teach("Beth#Q2", BondType::BornOn, "1910", 0.7, Confidence::Unverified);
        let c = try_synthesize(&b, "compare Alan and Beth", QueryIntent::Generic, None)
            .expect("entity comparison should fire");
        assert_eq!(c.kind, ReasoningKind::Comparative);
        let lo = c.prose.to_lowercase();
        assert!(lo.contains("both physicists"), "shared type: {}", c.prose);
        assert!(lo.contains("wave theory") && lo.contains("field theory"), "known-for: {}", c.prose);
        assert!(lo.contains("1900") && lo.contains("1910"), "birth years: {}", c.prose);
        assert!(!c.cited_edges.is_empty(), "must cite real edges");
        // integrity: never a subjective winner.
        assert!(!lo.contains("depends on what you value") && !lo.contains("better"), "no verdict: {}", c.prose);
    }

    #[test]
    fn entity_comparison_cross_field_honest() {
        let mut b = Brain::new();
        b.teach("Alan#Q1", BondType::IsA, "physicist", 0.7, Confidence::Unverified);
        b.teach("Alan#Q1", BondType::DiedOn, "1950", 0.7, Confidence::Unverified);
        b.teach("Cleo#Q2", BondType::IsA, "naturalist", 0.7, Confidence::Unverified);
        b.teach("Cleo#Q2", BondType::DiedOn, "1900", 0.7, Confidence::Unverified);
        let c = try_synthesize(&b, "Alan vs Cleo", QueryIntent::Generic, None)
            .expect("cross-field comparison should fire");
        let lo = c.prose.to_lowercase();
        // Honest cross-field: NOT collapsed to a false shared type.
        assert!(lo.contains("physicist") && lo.contains("naturalist"), "both types: {}", c.prose);
        assert!(!lo.contains("both"), "must not claim a shared type: {}", c.prose);
    }

    #[test]
    fn entity_comparison_place_pair_shared_currency() {
        let mut b = Brain::new();
        b.teach("Xland#Q5", BondType::IsA, "country", 0.7, Confidence::Unverified);
        b.teach("Xland#Q5", BondType::HasCapital, "Xtown", 0.7, Confidence::Unverified);
        b.teach("Xland#Q5", BondType::HasCurrency, "zed", 0.7, Confidence::Unverified);
        b.teach("Xland#Q5", BondType::HasOfficialLanguage, "xish", 0.7, Confidence::Unverified);
        b.teach("Yland#Q6", BondType::IsA, "country", 0.7, Confidence::Unverified);
        b.teach("Yland#Q6", BondType::HasCapital, "Ytown", 0.7, Confidence::Unverified);
        b.teach("Yland#Q6", BondType::HasCurrency, "zed", 0.7, Confidence::Unverified);
        b.teach("Yland#Q6", BondType::HasOfficialLanguage, "yish", 0.7, Confidence::Unverified);
        let c = try_synthesize(&b, "compare Xland and Yland", QueryIntent::Generic, None)
            .expect("place comparison should fire");
        let lo = c.prose.to_lowercase();
        assert!(lo.contains("both countries"), "shared type: {}", c.prose);
        assert!(lo.contains("use the zed"), "shared currency folded into lead: {}", c.prose);
        assert!(lo.contains("xtown") && lo.contains("ytown"), "capital diff: {}", c.prose);
        assert!(!c.cited_edges.is_empty());
    }

    #[test]
    fn entity_comparison_cross_kind_incomparable() {
        let mut b = Brain::new();
        b.teach("Cara#Q3", BondType::IsA, "physicist", 0.7, Confidence::Unverified);
        b.teach("Cara#Q3", BondType::KnownFor, "optics", 0.7, Confidence::Unverified);
        b.teach("Wonderland#Q4", BondType::IsA, "country", 0.7, Confidence::Unverified);
        b.teach("Wonderland#Q4", BondType::HasCapital, "Emerald City", 0.7, Confidence::Unverified);
        let c = try_synthesize(&b, "compare Cara and Wonderland", QueryIntent::Generic, None)
            .expect("cross-kind should compose an honest not-comparable answer");
        let lo = c.prose.to_lowercase();
        assert!(lo.contains("aren't really comparable") || lo.contains("not really comparable"), "{}", c.prose);
        assert!(lo.contains("physicist") && lo.contains("country"), "states what each is: {}", c.prose);
    }

    #[test]
    fn entity_comparison_concept_pair_defers() {
        // Concept nouns are not entity nodes -> entity_comparison degrades and
        // the concept `comparative` shape still handles the pair.
        let mut b = Brain::new();
        b.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        b.teach("cat", BondType::HasProperty, "independent", 0.9, Confidence::Verified);
        assert!(entity_comparison(&b, "compare a dog and a cat").is_none());
        let c = try_synthesize(&b, "compare a dog and a cat", QueryIntent::Generic, None)
            .expect("concept comparative still fires");
        assert_eq!(c.kind, ReasoningKind::Comparative);
        assert!(c.prose.to_lowercase().contains("depends on what you value"), "{}", c.prose);
    }
    #[test]
    fn tier2_does_not_regress_concept_shapes() {
        // The Tier-2 wiring must not steal category / property / definition.
        let mut b = Brain::new();
        b.teach("dog", BondType::IsA, "animal", 0.9, Confidence::Verified);
        b.teach("dog", BondType::HasProperty, "loyal", 0.9, Confidence::Verified);
        let a = try_synthesize(&b, "is a dog an animal", QueryIntent::Generic, None).expect("category");
        assert_eq!(a.kind, ReasoningKind::CategoryMembership);
        let p = try_synthesize(&b, "is a dog loyal", QueryIntent::Generic, None).expect("property");
        assert_eq!(p.kind, ReasoningKind::PropertyCheck);
        assert!(try_synthesize(&b, "what is a dog", QueryIntent::Definition, None).is_none());
    }

    // Tier-2 validation gate (ignored): parity probes + each new shape
    // firing/honest-refusing on the real read-only brain, with latency.
    //   LU_PROBE_BRAIN=/var/lib/lu/brain-v5-foundation-clever-20260729-sst \
    //   LU_PROBE_KEY=/var/lib/lu/master.key \
    //   LU_ENTITY_FAME=$B/entity_fame.tsv \
    //   cargo test --release --lib synthesis::tests::live_tier2_gate -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_tier2_gate() {
        let bpath = std::env::var("LU_PROBE_BRAIN").expect("LU_PROBE_BRAIN");
        let kpath = std::env::var("LU_PROBE_KEY").ok();
        let key = kpath.and_then(|p| crate::encrypt::MasterKey::load_from_keyfile(&p).ok());
        let brain = Brain::load_readonly(std::path::Path::new(&bpath), key).expect("load_readonly");
        let probes = [
            ("PARITY", "is a dog loyal"),
            ("PARITY", "is a dog an animal"),
            ("PARITY", "what causes cancer"),
            ("PARITY", "who is Einstein"),
            ("COMPOSITION", "what is water made of"),
            ("COMPOSITION", "what is a car made of"),
            ("PURPOSE", "what is a hammer used for"),
            ("MERONYM", "what are the parts of a car"),
            ("MERONYM", "what are the parts of a fish"),
            ("HOLONYM", "what is hydrogen part of"),
            ("HOLONYM", "what is a wheel part of"),
            ("RELATIONAL", "what is the relationship between hydrogen and water"),
            ("RELATIONAL", "what is the relationship between a dog and a wolf"),
            ("RELATIONAL", "how are a dog and a cat related"),
            ("COMPARATIVE", "is a dog a better pet than a fish"),
            ("COMPARATIVE", "compare a dog and a cat"),
            ("COMPARATIVE-OBJ", "is a whale bigger than an elephant"),
        ];
        let mut worst_us = 0u128;
        for (tag, q) in probes {
            let t = std::time::Instant::now();
            let c = try_synthesize(&brain, q, QueryIntent::Generic, None);
            let us = t.elapsed().as_micros();
            if us > worst_us { worst_us = us; }
            match c {
                Some(c) => println!("[{}] {}\n  KIND={:?} CONF={:.2} CITES={} T={}us\n  A: {}", tag, q, c.kind, c.confidence, c.cited_edges.len(), us, c.prose),
                None => println!("[{}] {}\n  A: <None / degrade> T={}us", tag, q, us),
            }
        }
        println!("WORST_LATENCY_US={}", worst_us);
    }
    // Live diagnostic (ignored): load the real read-only brain and dump
    // exactly what the synthesizer sees, to separate a data-coverage gap
    // from a silent-zero read-path bug. Run:
    //   LU_PROBE_BRAIN=/var/lib/lu/brain-v5-20260724-sst \
    //   LU_PROBE_KEY=/var/lib/lu/master.key \
    //   cargo test --release --lib synthesis::tests::live_edge_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_edge_probe() {
        let bpath = std::env::var("LU_PROBE_BRAIN").expect("LU_PROBE_BRAIN");
        let kpath = std::env::var("LU_PROBE_KEY").ok();
        let key = kpath.and_then(|p| crate::encrypt::MasterKey::load_from_keyfile(&p).ok());
        let mut brain = Brain::load_readonly(std::path::Path::new(&bpath), key).expect("load_readonly");
        let probes = ["fish", "bird", "elephant", 
            "grass", "fire", "snow", "gold", "water", "lemon", "dog", "iron",
        ];
        for w in probes {
            let anchors = crate::reasoning::pick_query_anchors(&mut brain, w);
            println!("\n=== probe {:?}  anchors={}", w, anchors.len());
            for a in anchors.iter().take(2) {
                let alab = brain.network.node_label(a).unwrap_or_default();
                let out = brain.network.typed_edges_out(a);
                println!("  anchor {:?}  out_edges={}", alab, out.len());
                let mut shown = 0;
                for (nbr, bt) in out.iter() {
                    let nl = brain.network.node_label(nbr).unwrap_or_default();
                    println!("    -{:?}-> {:?}", bt, nl);
                    shown += 1;
                    if shown >= 25 { println!("    ...(+{} more)", out.len()-shown); break; }
                }
                // one IS-A hop: show property edges of the first few IS-A mids
                let mut mids = 0;
                for (nbr, bt) in out.iter() {
                    if !is_isa_bond(*bt) { continue; }
                    let ml = brain.network.node_label(nbr).unwrap_or_default();
                    let mout = brain.network.typed_edges_out(nbr);
                    let props: Vec<String> = mout.iter().filter(|(_,b)| is_property_bond(*b))
                        .map(|(n,b)| format!("{:?} {:?}", b, brain.network.node_label(n).unwrap_or_default())).take(8).collect();
                    println!("    [ISA mid {:?}] out={} props={:?}", ml, mout.len(), props);
                    mids += 1;
                    if mids >= 4 { break; }
                }
            }
        }
    }
}
// ============================================================================
// Phase 5d (OP-PLANNER): composition search for the data-plumbing tier.
//
// ADDITIVE + ISOLATED. No existing reasoning path calls this function, so it
// cannot regress spread-activation, recall, or synthesis. Given an intent
// (src, agg), it selects a known op-composition and emits a straight-line
// Python program plus the ops it uses (for provenance grounding by the caller).
// Ported 1:1 from the verified prototype's compose table; structured as a
// best-first selection over (src, agg) op-shapes.
// ============================================================================

/// Emit `(program, ops_used)` for a data-plumbing (src, agg) spec, or None if
/// no composition is known (caller then aborts honestly). `ops_used` are op
/// names the caller provenance-checks against the brain's C2 handles.
pub(crate) fn compose_plumbing(src: &str, agg: &str) -> Option<(String, Vec<&'static str>)> {
    let read_nums = "nums=[float(x) for x in open('input.txt').read().split() if x.strip()]";
    let read_lines = "lines=[l.rstrip(chr(10)) for l in open('input.txt')]";
    let out: (String, Vec<&'static str>) = match agg {
        "json_keys" if src == "json" => (
            "import json\nd=json.load(open('input.json'))\nprint(sorted(d.keys()))".to_string(),
            vec!["open", "keys"]),
        "avg" if src == "csv" => (
            "import csv\nrows=list(csv.reader(open('input.csv')))\ncol=[float(r[0]) for r in rows if r and r[0].strip()]\nprint(sum(col)/len(col))".to_string(),
            vec!["open", "float", "sum", "len"]),
        "avg" => (format!("{}\nprint(sum(nums)/len(nums))", read_nums), vec!["open", "float", "sum", "len"]),
        "longest" => (format!("{}\nprint(max(lines, key=len))", read_lines), vec!["open", "max", "len"]),
        "wordfreq" => (
            "from collections import Counter\nw=open('input.txt').read().split()\nprint(Counter(w).most_common(1))".to_string(),
            vec!["open", "split", "count"]),
        "dups" => (
            "from collections import Counter\nc=Counter(l.rstrip(chr(10)) for l in open('input.txt'))\nprint(sorted(k for k,v in c.items() if v>1))".to_string(),
            vec!["open", "count"]),
        "second_max" => (
            "nums=sorted(set(float(x) for x in open('input.txt').read().split() if x.strip()))\nprint(nums[-2])".to_string(),
            vec!["open", "float", "sorted", "set"]),
        "max" => (format!("{}\nprint(max(nums))", read_nums), vec!["open", "float", "max"]),
        "min" => (format!("{}\nprint(min(nums))", read_nums), vec!["open", "float", "min"]),
        "sort" => (format!("{}\nprint(sorted(nums))", read_nums), vec!["open", "float", "sorted"]),
        "count_lines" => (format!("{}\nprint(len([l for l in lines if len(l)>10]))", read_lines), vec!["open", "len"]),
        "sum" => (format!("{}\nprint(sum(nums))", read_nums), vec!["open", "float", "sum"]),
        _ => return None,
    };
    Some(out)
}

