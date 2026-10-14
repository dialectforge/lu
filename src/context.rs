//! Context selector — determines which temporal scope produces the
//! most coherent activation neighborhood.
//!
//! Given a set of seed nodes, the context selector tries multiple
//! temporal scopes and picks the one that maximizes subgraph coherence
//! (average bond weight among activated nodes).

use crate::network::{AssociativeNetwork, RecallResult};
use crate::scope::TemporalScope;
use crate::scar::ScarLog;
use crate::seed::Seed;

/// Result of context selection, containing the best scope and its coherence score.
#[derive(Debug, Clone)]
pub struct ContextResult {
    /// The temporal scope that produced the highest coherence.
    pub best_scope: TemporalScope,
    /// The coherence score (average bond weight in activated subgraph).
    pub coherence: f32,
    /// The recall result from the best scope.
    pub recall: RecallResult,
    /// All evaluated scopes with their coherence scores, for inspection.
    pub evaluated: Vec<(TemporalScope, f32)>,
}

/// Context selector that evaluates multiple temporal scopes to find
/// the most coherent one for a given query.
pub struct ContextSelector {
    /// The activation depth to use during evaluation.
    pub depth: usize,
    /// The number of top-k nodes to activate.
    pub top_k: usize,
}

impl ContextSelector {
    /// Create a new context selector with default parameters.
    pub fn new() -> Self {
        ContextSelector {
            depth: 3,
            top_k: 20,
        }
    }

    /// Create a context selector with custom parameters.
    pub fn with_params(depth: usize, top_k: usize) -> Self {
        ContextSelector { depth, top_k }
    }

    /// Select the best temporal scope from a set of candidates.
    ///
    /// Runs spread activation under each scope, computes subgraph coherence,
    /// and returns the scope with the highest coherence score.
    pub fn select_best_scope(
        &self,
        network: &AssociativeNetwork,
        seeds: &[Seed],
        candidate_scopes: &[TemporalScope],
    ) -> ContextResult {
        let mut best_scope = TemporalScope::Current;
        let mut best_coherence = f32::NEG_INFINITY;
        let mut best_recall: Option<RecallResult> = None;
        let mut evaluated = Vec::new();

        for scope in candidate_scopes {
            let recall = network.spread_activation(seeds, scope, self.depth, self.top_k, &ScarLog::default());

            // Compute coherence: average bond weight in the activated subgraph
            let activated_seeds: Vec<Seed> = recall
                .activated
                .iter()
                .map(|a| a.seed)
                .collect();

            let coherence = if activated_seeds.len() < 2 {
                0.0
            } else {
                network.subgraph_coherence(&activated_seeds, scope, &ScarLog::default())
            };

            evaluated.push((scope.clone(), coherence));

            if coherence > best_coherence {
                best_coherence = coherence;
                best_scope = scope.clone();
                best_recall = Some(recall);
            }
        }

        // If no candidate was evaluated, fall back to Current scope
        let recall = best_recall.unwrap_or_else(|| {
            network.spread_activation(seeds, &TemporalScope::Current, self.depth, self.top_k, &ScarLog::default())
        });

        ContextResult {
            best_scope,
            coherence: best_coherence.max(0.0),
            recall,
            evaluated,
        }
    }

    /// Generate a set of candidate scopes based on the network's bond history.
    ///
    /// Produces: Current, plus Historical scopes at significant timestamps
    /// found in the bonds connected to the seed nodes.
    pub fn generate_candidate_scopes(
        &self,
        network: &AssociativeNetwork,
        seeds: &[Seed],
    ) -> Vec<TemporalScope> {
        let mut timestamps: Vec<i64> = Vec::new();
        let mut scopes = vec![TemporalScope::Current];

        // Collect significant timestamps from bonds connected to seed nodes
        for seed in seeds {
            // OP-BRAIN-MapRAM P5-batch2: source the seed's outgoing +
            // incoming edges (and their keyframes) from the canonical,
            // uncapped, edge-carrying seam. In cache mode the bond -- and
            // its keyframes -- come from the carried `BondPayload`
            // (reconstructed byte-identically to rehydrate), never the
            // resident `get_bond`. Output is sorted+deduped below, so it
            // is order-independent; the canonical seam order also makes it
            // deterministic across both paths.
            for (_neighbor, bond) in network.seam_walk_outgoing(seed) {
                if let Some(bond) = bond {
                    // Audit fix internal notes: keyframes() panics on
                    // archive-backed bonds. Use the archive-aware iterator.
                    for kf in bond.view().keyframes_iter() {
                        timestamps.push(kf.timestamp);
                    }
                }
            }
            for (_neighbor, bond) in network.seam_walk_incoming(seed) {
                if let Some(bond) = bond {
                    // Audit fix internal notes: keyframes() panics on
                    // archive-backed bonds. Use the archive-aware iterator.
                    for kf in bond.view().keyframes_iter() {
                        timestamps.push(kf.timestamp);
                    }
                }
            }
        }

        timestamps.sort();
        timestamps.dedup();

        // Add Historical scopes at each unique timestamp
        for t in timestamps {
            scopes.push(TemporalScope::Historical(t));
        }

        scopes
    }

    /// Convenience: auto-generate candidates and select the best scope.
    pub fn auto_select(
        &self,
        network: &AssociativeNetwork,
        seeds: &[Seed],
    ) -> ContextResult {
        let candidates = self.generate_candidate_scopes(network, seeds);
        self.select_best_scope(network, seeds, &candidates)
    }
}

impl Default for ContextSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{BondType, TemporalBond};
    use crate::node::{Confidence, KnowledgeContent, KnowledgeNode};

    #[test]
    fn test_context_selector_basic() {
        let mut net = AssociativeNetwork::new();
        let sa = Seed::from_label("Alpha");
        let sb = Seed::from_label("Beta");

        net.add_node(KnowledgeNode::with_seed(
            sa, "Alpha".into(),
            KnowledgeContent::Raw("Alpha concept".into()),
            Confidence::Verified, 0,
        ));
        net.add_node(KnowledgeNode::with_seed(
            sb, "Beta".into(),
            KnowledgeContent::Raw("Beta concept".into()),
            Confidence::Verified, 0,
        ));
        net.add_bond(TemporalBond::new(sa, sb, BondType::RelatedTo, 0.8, 100))
            .unwrap();

        let selector = ContextSelector::new();
        let result = selector.auto_select(&net, &[sa]);

        assert!(result.coherence >= 0.0);
        assert!(!result.evaluated.is_empty());
    }

    #[test]
    fn test_generate_candidates_includes_current() {
        let net = AssociativeNetwork::new();
        let selector = ContextSelector::new();
        let candidates = selector.generate_candidate_scopes(&net, &[]);
        assert!(candidates.iter().any(|s| matches!(s, TemporalScope::Current)));
    }
}
