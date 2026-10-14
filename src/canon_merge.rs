//! OP-CANON-1 -- offline merge planning: fold surface variants onto one
//! canonical node and redirect their bonds (merge, never delete).
//!
//! PURE + deterministic. There is deliberately NO brain I/O in this
//! module: the real NodeStore streaming (write-new) execution is the
//! backup-gated Phase-B step. This module is the algorithmic core that
//! the offline `lu_canon_merge` tool drives and that the unit tests
//! exercise in full (fold correctness, bond redirect, idempotency).
//!
//! TWO keys, one source of truth (`crate::canonical`):
//!   * FOLD key   = `canonical_ingest_key` (case-fold + article-strip +
//!     whitespace-collapse). Groups `Dog`/`dog`/`the dog` -> one node.
//!     Lossless: it never rejects on content shape, so folding alone can
//!     never drop a real concept.
//!   * PRUNE flag = `canonical_key(..).is_none()` (content-shape reject).
//!     Marks a node whose label is a non-concept fragment.
//!
//! MERGE POLICY (knowledge-preserving):
//!   * Within a fold group the TARGET is the cleanest real-concept node
//!     (`canonical_key == Some`); deterministic tiebreak = the surface
//!     that already equals the fold key, else the lowest node id.
//!   * Every other node in the group REDIRECTS to the target: its bonds
//!     are rewritten onto the target and merged. No bond is dropped while
//!     a target exists -- this is what lifts bonds-per-concept.
//!   * A group with NO real-concept member (all reject fragments) has no
//!     target: those nodes are ORPHANs, their bonds drop. That is the
//!     "folds to nothing real" case Phase C trims, behind the backup gate.

use crate::canonical::{canonical_ingest_key, canonical_key};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A directed, typed, weighted bond in the abstract (I/O-free) form the
/// planner rewrites. `kind` is the bond-type discriminant; the real
/// integration maps `BondType` <-> `u16`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bond {
    pub from: u64,
    pub to: u64,
    pub kind: u16,
    pub weight: f32,
}

/// The result of planning a fold over a node set.
#[derive(Debug, Default)]
pub struct FoldPlan {
    /// fold key -> surviving (target) node id.
    pub target_of_key: HashMap<String, u64>,
    /// variant node id -> the target it folds onto (targets excluded).
    pub redirect: HashMap<u64, u64>,
    /// node ids to retire: every folded variant PLUS every node in a
    /// no-target (all-reject) group.
    pub retire: HashSet<u64>,
    /// node ids with no surviving target (their bonds must drop).
    pub orphan: HashSet<u64>,
}

impl FoldPlan {
    /// Map a node id to its surviving representative: itself if it is a
    /// target or was never folded; the target if it is a redirected
    /// variant; `None` if it is an orphan (no survivor).
    pub fn resolve(&self, id: u64) -> Option<u64> {
        if self.orphan.contains(&id) {
            return None;
        }
        Some(*self.redirect.get(&id).unwrap_or(&id))
    }

    /// Nodes removed from the graph (folded variants + orphans).
    pub fn retired_count(&self) -> usize {
        self.retire.len()
    }
}

/// Plan the fold over `(node_id, label)` pairs. Deterministic: grouping
/// is by fold key and target choice is a total order, so a given node set
/// always yields the same plan regardless of input order.
pub fn plan_fold<S: AsRef<str>>(nodes: &[(u64, S)]) -> FoldPlan {
    let mut groups: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut is_concept: HashMap<u64, bool> = HashMap::new();
    let mut label_of: HashMap<u64, String> = HashMap::new();
    let mut no_key: Vec<u64> = Vec::new();

    for (id, label) in nodes {
        let l = label.as_ref();
        match canonical_ingest_key(l) {
            Some(k) => {
                groups.entry(k).or_default().push(*id);
                is_concept.insert(*id, canonical_key(l).is_some());
                label_of.insert(*id, l.to_string());
            }
            None => no_key.push(*id),
        }
    }

    let mut plan = FoldPlan::default();
    for (key, mut ids) in groups {
        ids.sort_unstable();
        match choose_target(&ids, &is_concept, &label_of, &key) {
            Some(t) => {
                plan.target_of_key.insert(key, t);
                for &id in &ids {
                    if id != t {
                        plan.redirect.insert(id, t);
                        plan.retire.insert(id);
                    }
                }
            }
            None => {
                for &id in &ids {
                    plan.retire.insert(id);
                    plan.orphan.insert(id);
                }
            }
        }
    }
    for id in no_key {
        plan.retire.insert(id);
        plan.orphan.insert(id);
    }
    plan
}

/// Choose the surviving target for a fold group: the cleanest real
/// concept. Returns `None` when the whole group is reject fragments.
fn choose_target(
    ids: &[u64],
    is_concept: &HashMap<u64, bool>,
    label_of: &HashMap<u64, String>,
    key: &str,
) -> Option<u64> {
    let concepts: Vec<u64> = ids
        .iter()
        .copied()
        .filter(|id| *is_concept.get(id).unwrap_or(&false))
        .collect();
    if concepts.is_empty() {
        return None;
    }
    // Prefer a node whose trimmed/lowercased surface already equals the
    // fold key (the cleanest form); else the lowest id (earliest, most
    // established). `ids` is pre-sorted so `min` is deterministic.
    concepts
        .iter()
        .copied()
        .find(|id| {
            label_of
                .get(id)
                .map(|l| l.trim().to_lowercase() == key)
                .unwrap_or(false)
        })
        .or_else(|| concepts.iter().copied().min())
}

/// Rewrite bonds through a fold plan: redirect both endpoints to their
/// survivors, drop any bond touching an orphan, drop self-loops created
/// by the fold, and merge duplicate `(from,to,kind)` bonds by combining
/// weight (saturating at 1.0). Output is sorted, making the pass
/// deterministic and IDEMPOTENT (re-running on the output is a no-op).
pub fn redirect_bonds(bonds: &[Bond], plan: &FoldPlan) -> Vec<Bond> {
    let mut merged: BTreeMap<(u64, u64, u16), f32> = BTreeMap::new();
    for b in bonds {
        let (from, to) = match (plan.resolve(b.from), plan.resolve(b.to)) {
            (Some(f), Some(t)) => (f, t),
            _ => continue, // endpoint folded to nothing -> drop bond
        };
        if from == to {
            continue; // self-loop after fold -> drop
        }
        let e = merged.entry((from, to, b.kind)).or_insert(0.0);
        *e = (*e + b.weight).min(1.0);
    }
    merged
        .into_iter()
        .map(|((from, to, kind), weight)| Bond { from, to, kind, weight })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: u64, s: &str) -> (u64, String) {
        (id, s.to_string())
    }

    #[test]
    fn folds_case_and_article_variants_to_one_target() {
        let nodes = vec![n(1, "Dog"), n(2, "the dog"), n(3, "DOG")];
        let plan = plan_fold(&nodes);
        // "Dog" (id 1) is the exact-key surface -> target.
        assert_eq!(plan.target_of_key.get("dog"), Some(&1));
        assert_eq!(plan.resolve(1), Some(1));
        assert_eq!(plan.resolve(2), Some(1));
        assert_eq!(plan.resolve(3), Some(1));
        assert!(plan.retire.contains(&2) && plan.retire.contains(&3));
        assert!(!plan.retire.contains(&1));
        assert!(plan.orphan.is_empty());
    }

    #[test]
    fn target_tiebreaks_to_lowest_id_when_no_exact_surface() {
        // No surface equals the key "united states" exactly.
        let nodes = vec![n(7, "The United States"), n(4, "the United States")];
        let plan = plan_fold(&nodes);
        assert_eq!(plan.target_of_key.get("united states"), Some(&4));
        assert_eq!(plan.resolve(7), Some(4));
    }

    #[test]
    fn redirects_and_merges_variant_bonds_onto_target() {
        let nodes = vec![n(1, "Dog"), n(2, "the dog"), n(9, "Animal")];
        let plan = plan_fold(&nodes);
        let bonds = vec![
            Bond { from: 2, to: 9, kind: 0, weight: 0.4 }, // variant -> Animal
            Bond { from: 1, to: 9, kind: 0, weight: 0.5 }, // target  -> Animal (dup after fold)
            Bond { from: 9, to: 2, kind: 1, weight: 0.3 }, // Animal  -> variant
        ];
        let out = redirect_bonds(&bonds, &plan);
        // (2->9) folds onto (1->9); same (from,to,kind) merges weights.
        assert!(out.contains(&Bond { from: 1, to: 9, kind: 0, weight: 0.9f32.min(1.0) }));
        // (9->2) becomes (9->1).
        assert!(out.contains(&Bond { from: 9, to: 1, kind: 1, weight: 0.3 }));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn drops_self_loops_created_by_fold() {
        let nodes = vec![n(1, "Dog"), n(2, "the dog")];
        let plan = plan_fold(&nodes);
        let bonds = vec![Bond { from: 1, to: 2, kind: 0, weight: 0.7 }];
        let out = redirect_bonds(&bonds, &plan);
        assert!(out.is_empty()); // 1->1 after fold, dropped
    }

    #[test]
    fn all_reject_group_is_orphaned_and_its_bonds_drop() {
        // Two distinct reject fragments -> two 1-node orphan groups.
        let nodes = vec![
            n(5, "an off-leash dog park located at the township municipal"),
            n(6, "a very suitable platform to be investigated by the"),
            n(1, "Dog"),
        ];
        let plan = plan_fold(&nodes);
        assert!(plan.orphan.contains(&5) && plan.orphan.contains(&6));
        assert_eq!(plan.resolve(5), None);
        let bonds = vec![
            Bond { from: 5, to: 1, kind: 0, weight: 0.9 }, // touches orphan -> drop
            Bond { from: 1, to: 6, kind: 0, weight: 0.9 }, // touches orphan -> drop
        ];
        assert!(redirect_bonds(&bonds, &plan).is_empty());
    }

    #[test]
    fn reject_variant_with_clean_twin_is_preserved_by_redirect() {
        // A reject-shaped variant that folds to the SAME key as a clean
        // concept keeps its bonds by redirecting onto the clean twin.
        // "the Dog" -> key "dog"; "Dog" -> key "dog" (clean concept).
        let nodes = vec![n(1, "Dog"), n(2, "the Dog")];
        let plan = plan_fold(&nodes);
        assert_eq!(plan.resolve(2), Some(1));
        let bonds = vec![Bond { from: 2, to: 42, kind: 0, weight: 0.5 }];
        let out = redirect_bonds(&bonds, &plan);
        assert_eq!(out, vec![Bond { from: 1, to: 42, kind: 0, weight: 0.5 }]);
    }

    #[test]
    fn redirect_bonds_is_idempotent() {
        let nodes = vec![n(1, "Dog"), n(2, "the dog"), n(9, "Animal"), n(8, "Pet")];
        let plan = plan_fold(&nodes);
        let bonds = vec![
            Bond { from: 2, to: 9, kind: 0, weight: 0.4 },
            Bond { from: 1, to: 9, kind: 0, weight: 0.5 },
            Bond { from: 8, to: 2, kind: 2, weight: 0.6 },
            Bond { from: 2, to: 1, kind: 0, weight: 0.9 }, // -> self-loop, dropped
        ];
        let once = redirect_bonds(&bonds, &plan);
        let twice = redirect_bonds(&once, &plan);
        assert_eq!(once, twice, "redirect must be idempotent");
    }

    #[test]
    fn unknown_endpoints_pass_through_unchanged() {
        // Bonds referencing ids outside the planned node set are kept.
        let plan = plan_fold(&[n(1, "Dog"), n(2, "the dog")]);
        let bonds = vec![Bond { from: 100, to: 200, kind: 0, weight: 0.5 }];
        assert_eq!(redirect_bonds(&bonds, &plan), bonds);
    }
}
