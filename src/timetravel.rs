//! Time travel queries — run activation at a historical point and
//! diff the results against the current graph state.
//!
//! This module lets you answer questions like "what did I know about X
//! a week ago?" and "how has my understanding of X changed?"

use crate::network::{ActivatedNode, AssociativeNetwork, RecallResult};
use crate::scope::TemporalScope;
use crate::scar::ScarLog;
use crate::seed::Seed;
use serde::{Deserialize, Serialize};

/// A single difference between historical and current activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDiff {
    /// The node's seed.
    pub seed: Seed,
    /// The node's content label.
    pub content: String,
    /// Activation level in the historical query.
    pub historical_activation: f32,
    /// Activation level in the current query.
    pub current_activation: f32,
    /// The change in activation (current - historical).
    pub delta: f32,
    /// Categorization of the change.
    pub change_type: ChangeType,
}

/// How a node's activation changed between historical and current state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Node appeared in current but not historical results.
    Emerged,
    /// Node appeared in historical but not current results.
    Faded,
    /// Node's activation increased significantly.
    Strengthened,
    /// Node's activation decreased significantly.
    Weakened,
    /// Node's activation stayed roughly the same.
    Stable,
}

/// The result of a time travel query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDiff {
    /// The historical timestamp queried.
    pub historical_time: i64,
    /// All node-level differences, sorted by absolute delta (largest first).
    pub diffs: Vec<NodeDiff>,
    /// The historical recall result.
    pub historical_recall: RecallResult,
    /// The current recall result.
    pub current_recall: RecallResult,
    /// Summary statistics.
    pub summary: DiffSummary,
}

/// Summary statistics for a temporal diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Number of nodes that emerged (new in current).
    pub emerged: usize,
    /// Number of nodes that faded (gone from current).
    pub faded: usize,
    /// Number of nodes that strengthened.
    pub strengthened: usize,
    /// Number of nodes that weakened.
    pub weakened: usize,
    /// Number of stable nodes.
    pub stable: usize,
    /// Average absolute delta across all diffed nodes.
    pub avg_abs_delta: f32,
}

/// Significance threshold — changes below this are considered stable.
const SIGNIFICANCE_THRESHOLD: f32 = 0.05;

/// Run a time travel query: activate the graph at a historical point,
/// then at the current state, and produce a diff.
///
/// The `query` parameter is used for keyword search to find seed nodes.
/// If seed nodes are provided directly, the query string is ignored.
pub fn time_travel_query(
    query: &str,
    at: i64,
    network: &AssociativeNetwork,
) -> TemporalDiff {
    // Find seed nodes via keyword search
    let search_results = network.keyword_search(query);
    let seeds: Vec<Seed> = search_results
        .iter()
        .take(5) // Use top 5 keyword matches as seeds
        .map(|(seed, _)| *seed)
        .collect();

    time_travel_with_seeds(&seeds, at, network)
}

/// Run a time travel query with explicit seed nodes.
pub fn time_travel_with_seeds(
    seeds: &[Seed],
    at: i64,
    network: &AssociativeNetwork,
) -> TemporalDiff {
    let depth = 3;
    let top_k = 50;

    // Run activation at historical time
    let historical_scope = TemporalScope::Historical(at);
    let historical_recall =
        network.spread_activation(seeds, &historical_scope, depth, top_k, &ScarLog::default());

    // Run activation at current time
    let current_scope = TemporalScope::Current;
    let current_recall =
        network.spread_activation(seeds, &current_scope, depth, top_k, &ScarLog::default());

    // Build lookup maps for activation levels
    let historical_map: std::collections::HashMap<Seed, &ActivatedNode> = historical_recall
        .activated
        .iter()
        .map(|a| (a.seed, a))
        .collect();

    let current_map: std::collections::HashMap<Seed, &ActivatedNode> = current_recall
        .activated
        .iter()
        .map(|a| (a.seed, a))
        .collect();

    // Collect all unique seeds from both results
    let mut all_seeds: std::collections::HashSet<Seed> = std::collections::HashSet::new();
    for a in &historical_recall.activated {
        all_seeds.insert(a.seed);
    }
    for a in &current_recall.activated {
        all_seeds.insert(a.seed);
    }

    // Compute diffs
    let mut diffs: Vec<NodeDiff> = Vec::new();
    let mut emerged = 0usize;
    let mut faded = 0usize;
    let mut strengthened = 0usize;
    let mut weakened = 0usize;
    let mut stable = 0usize;
    let mut total_abs_delta = 0.0f32;

    for seed in &all_seeds {
        let hist_activation = historical_map
            .get(seed)
            .map(|a| a.activation)
            .unwrap_or(0.0);
        let curr_activation = current_map
            .get(seed)
            .map(|a| a.activation)
            .unwrap_or(0.0);

        let delta = curr_activation - hist_activation;
        total_abs_delta += delta.abs();

        let content = historical_map
            .get(seed)
            .map(|a| a.content.clone())
            .or_else(|| current_map.get(seed).map(|a| a.content.clone()))
            .unwrap_or_else(|| format!("{}", seed));

        let change_type = if hist_activation == 0.0 && curr_activation > 0.0 {
            emerged += 1;
            ChangeType::Emerged
        } else if hist_activation > 0.0 && curr_activation == 0.0 {
            faded += 1;
            ChangeType::Faded
        } else if delta > SIGNIFICANCE_THRESHOLD {
            strengthened += 1;
            ChangeType::Strengthened
        } else if delta < -SIGNIFICANCE_THRESHOLD {
            weakened += 1;
            ChangeType::Weakened
        } else {
            stable += 1;
            ChangeType::Stable
        };

        diffs.push(NodeDiff {
            seed: *seed,
            content,
            historical_activation: hist_activation,
            current_activation: curr_activation,
            delta,
            change_type,
        });
    }

    // Sort by absolute delta (largest changes first)
    diffs.sort_by(|a, b| {
        b.delta
            .abs()
            .partial_cmp(&a.delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_nodes = diffs.len();
    let avg_abs_delta = if total_nodes > 0 {
        total_abs_delta / total_nodes as f32
    } else {
        0.0
    };

    TemporalDiff {
        historical_time: at,
        diffs,
        historical_recall,
        current_recall,
        summary: DiffSummary {
            emerged,
            faded,
            strengthened,
            weakened,
            stable,
            avg_abs_delta,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, KeyframeSource, TemporalBond};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};

    #[test]
    fn test_time_travel_detects_changes() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("TimeA");
        let sb = Seed::from_label("TimeB");

        net.add_node(KnowledgeNode::with_seed(
            sa, "TimeA alpha".into(),
            KnowledgeContent::Raw("Time node A".into()),
            Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            sb, "TimeB beta".into(),
            KnowledgeContent::Raw("Time node B".into()),
            Confidence::Verified, 0,
        ));

        // Bond starts weak, gets stronger over time
        let mut bond = TemporalBond::new(sa, sb, BondType::RelatedTo, 0.1, 100);
        bond.add_keyframe(5000, 0.9, KeyframeSource::Session(1));
        net.add_bond(bond).unwrap();

        // Time travel: compare t=200 (weak bond) vs current (strong bond)
        let diff = time_travel_with_seeds(&[sa], 200, &net);

        // Should detect that the relationship has changed
        assert!(!diff.diffs.is_empty());
        // The bond was weak at t=200, strong now
        let total_change: f32 = diff.diffs.iter().map(|d| d.delta.abs()).sum();
        assert!(total_change > 0.0 || diff.summary.emerged > 0 || diff.summary.strengthened > 0);
    }

    #[test]
    fn test_time_travel_keyword_query() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("Keyword1");

        net.add_node(KnowledgeNode::with_seed(
            sa, "temporal knowledge".into(),
            KnowledgeContent::Fact("Knowledge about time".into()),
            Confidence::Verified, 0,
        ));

        let diff = time_travel_query("temporal", 500, &net);
        // Should find the node via keyword search
        assert!(diff.current_recall.activated.len() >= 1 || diff.historical_recall.activated.len() >= 1);
    }

    #[test]
    fn test_change_type_classification() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("ClassA");
        let sb = Seed::from_label("ClassB");
        let sc = Seed::from_label("ClassC");

        for (s, label) in [(sa, "ClassA"), (sb, "ClassB"), (sc, "ClassC")] {
            net.add_node(KnowledgeNode::with_seed(
                s, label.into(),
                KnowledgeContent::Raw(label.into()),
                Confidence::Verified, 0,
            ));
        }

        // Bond A→B exists from t=0, bond A→C only from t=5000
        net.add_bond(TemporalBond::new(sa, sb, BondType::RelatedTo, 0.8, 0)).unwrap();
        net.add_bond(TemporalBond::new(sa, sc, BondType::RelatedTo, 0.8, 5000)).unwrap();

        let diff = time_travel_with_seeds(&[sa], 100, &net);

        // C should be "Emerged" (not present at t=100, present now)
        let c_diff = diff.diffs.iter().find(|d| d.seed == sc);
        if let Some(cd) = c_diff {
            assert_eq!(cd.change_type, ChangeType::Emerged);
        }
    }

    #[test]
    fn test_diff_summary_counts() {
        let net = AssociativeNetwork::new();
        let diff = time_travel_with_seeds(&[], 0, &net);
        assert_eq!(diff.summary.emerged, 0);
        assert_eq!(diff.summary.faded, 0);
        assert_eq!(diff.summary.avg_abs_delta, 0.0);
    }
}
