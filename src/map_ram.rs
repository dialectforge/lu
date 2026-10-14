//! OP-BRAIN-MapRAM P0 — the resident MAP ("map in RAM, brain on disk").
//!
//! The architecture goal (see `docs/OP-BRAIN-MapRAM-Design-2026-06-12.md`)
//! is to make resident RAM track the *working set* being traversed, not
//! the *total* brain, losslessly, with the exact graph on disk. Phase
//! 3.8.1's [`crate::brain::Brain::rehydrate_adjacency`] today rehydrates
//! ALL adjacency resident (`HashMap<Seed, Vec<NodeId>>`, O(bonds), the
//! ~46 GiB cost). This module is **P0**: the foundational resident
//! structure that later phases (the cache-backed reader in P1, the
//! consumer rerouting in P2, the prefetch router in P3) build on.
//!
//! [`ResidentMap`] is an INDEX / LOCATOR only — it answers two questions
//! in O(1) for any [`NodeId`]:
//!   * does this node exist (is it within the interned id space), and
//!   * what is its out-degree and in-degree?
//!
//! It holds NO edges and NO payloads; the exact bonds stay on disk in the
//! v5 bond-store SSTs and are read on demand via
//! [`crate::brain::Brain::get_outgoing_bonds_v5`] /
//! `get_incoming_bonds_v5`. Reintroducing edge lists here would defeat
//! the entire purpose (it would be `rehydrate_adjacency` again) — so the
//! map deliberately stores only degrees.
//!
//! ## Why degree, and why O(1)
//!
//! The hot traversal path (`spread_activation_weighted_inner`,
//! `network.rs`) needs each neighbour's degree to compute the
//! degree-normalisation factor `deg_norm = sqrt(degree_p90 / deg)` for
//! hub down-weighting, on EVERY expanded edge. The design doc requires
//! this per-neighbour degree to be served from the resident map in O(1)
//! (it must NOT trigger a disk/decode read), otherwise the
//! query→seed path's `out_degree` / `in_degree` lookups would dominate.
//! Backing the degrees with a `Vec` indexed directly by `NodeId.0`
//! (NodeIds are dense and append-only — `node_intern.rs`) gives that O(1)
//! with no hash overhead.
//!
//! ## Size
//!
//! Two `u32`s per node = 8 B/node, O(nodes), **flat in bond count**. At
//! 15.6M nodes that is ~125 MiB; at 30M nodes (full Wikipedia) ~240 MiB
//! — versus the tens of GiB of resident adjacency that grows with bonds.
//! That fixed-RAM win is the architecture's purpose; P0 builds the
//! structure, later phases route consumers onto it so the 46 GiB can go.
//!
//! The optional per-direction on-disk *locator hint* described in the
//! design doc (§4.1) is deliberately NOT built in P0: it is an explicit
//! optimisation to skip the manifest binary-search on hot subjects, it
//! relates to the deferred 242µs decode work, and the doc marks it
//! optional. P0 ships the degree/existence map only — the minimal
//! correct foundation — and does not over-build.

use crate::node_intern::NodeId;

/// Resident MAP: `NodeId → (out_degree, in_degree)`, O(1) lookups,
/// flat in bond count. See the module docs.
///
/// Indexed densely by `NodeId.0`: `out_degree[id.0]` / `in_degree[id.0]`.
/// A node is considered *present* iff its id is within the vectors'
/// length (which is sized to the interned node count at build time), so
/// an isolated/degree-0 node is still present with both degrees 0.
#[derive(Debug, Clone, Default)]
pub struct ResidentMap {
    /// `out_degree[id.0]` — number of distinct outgoing-edge objects for
    /// the subject `id`, matching `AssociativeNetwork::out_degree`.
    out_degree: Vec<u32>,
    /// `in_degree[id.0]` — number of distinct incoming-edge subjects for
    /// the object `id`, matching `AssociativeNetwork::in_degree`.
    in_degree: Vec<u32>,
}

impl ResidentMap {
    /// Construct from pre-counted degree vectors. The two vectors MUST be
    /// the same length (one slot per interned node); this is the only
    /// invariant the type enforces.
    pub fn from_degrees(out_degree: Vec<u32>, in_degree: Vec<u32>) -> Self {
        assert_eq!(
            out_degree.len(),
            in_degree.len(),
            "ResidentMap: out_degree and in_degree must have one slot per node",
        );
        Self { out_degree, in_degree }
    }

    /// Number of nodes the map covers (its dense id space). Equals the
    /// interned node count at build time.
    pub fn node_count(&self) -> usize {
        self.out_degree.len()
    }

    /// True iff `id` is within the map's id space (a known node). O(1).
    pub fn contains(&self, id: NodeId) -> bool {
        (id.0 as usize) < self.out_degree.len()
    }

    /// Out-degree of `id`. O(1). Out-of-range ids return 0 (the same
    /// answer `AssociativeNetwork::out_degree` gives for an absent seed).
    pub fn out_degree(&self, id: NodeId) -> u32 {
        self.out_degree.get(id.0 as usize).copied().unwrap_or(0)
    }

    /// In-degree of `id`. O(1). Out-of-range ids return 0.
    pub fn in_degree(&self, id: NodeId) -> u32 {
        self.in_degree.get(id.0 as usize).copied().unwrap_or(0)
    }

    /// Total (out + in) degree of `id` — the quantity the spread path's
    /// hub classification / `deg_norm` uses. O(1). Widened to `u64` so a
    /// pathological hub (max measured degree ~99,879) cannot overflow.
    pub fn total_degree(&self, id: NodeId) -> u64 {
        self.out_degree(id) as u64 + self.in_degree(id) as u64
    }

    /// Sum of all out-degrees — the counted bond total. For a closed
    /// graph this equals [`Self::total_in_bonds`]; both are handy for
    /// build-time sanity checks and telemetry.
    pub fn total_out_bonds(&self) -> u64 {
        self.out_degree.iter().map(|&d| d as u64).sum()
    }

    /// Sum of all in-degrees.
    pub fn total_in_bonds(&self) -> u64 {
        self.in_degree.iter().map(|&d| d as u64).sum()
    }

    /// OP-BRAIN-MapRAM persist -- borrow the dense out-degree slice for
    /// serialisation to the degree-map sidecar. Indexed by `NodeId.0`.
    pub fn out_degrees(&self) -> &[u32] {
        &self.out_degree
    }

    /// OP-BRAIN-MapRAM persist -- borrow the dense in-degree slice for
    /// serialisation to the degree-map sidecar. Indexed by `NodeId.0`.
    pub fn in_degrees(&self) -> &[u32] {
        &self.in_degree
    }
    /// Resident heap footprint of the degree vectors, in bytes
    /// (excludes the struct header). ~8 B × node_count once built.
    pub fn resident_bytes(&self) -> usize {
        (self.out_degree.capacity() + self.in_degree.capacity())
            * std::mem::size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_intern::NodeId;

    #[test]
    fn degrees_and_existence_are_o1_and_correct() {
        // ids 0..4: out=[2,0,1,3], in=[0,1,2,1]
        let map = ResidentMap::from_degrees(vec![2, 0, 1, 3], vec![0, 1, 2, 1]);
        assert_eq!(map.node_count(), 4);

        assert_eq!(map.out_degree(NodeId(0)), 2);
        assert_eq!(map.in_degree(NodeId(0)), 0);
        assert_eq!(map.total_degree(NodeId(0)), 2);

        // isolated-but-present node (degree 0, still contained)
        assert_eq!(map.out_degree(NodeId(1)), 0);
        assert_eq!(map.in_degree(NodeId(1)), 1);
        assert!(map.contains(NodeId(1)));

        assert_eq!(map.total_degree(NodeId(3)), 4);
    }

    #[test]
    fn out_of_range_id_is_absent_and_zero() {
        let map = ResidentMap::from_degrees(vec![1, 1], vec![1, 1]);
        assert!(!map.contains(NodeId(2)));
        assert!(!map.contains(NodeId(9_999)));
        // absent ids read as degree 0 (not a panic), matching the
        // network accessors' "unwrap_or(0)".
        assert_eq!(map.out_degree(NodeId(2)), 0);
        assert_eq!(map.in_degree(NodeId(9_999)), 0);
    }

    #[test]
    fn bond_totals_sum_degrees() {
        let map = ResidentMap::from_degrees(vec![2, 0, 1, 3], vec![1, 2, 2, 1]);
        assert_eq!(map.total_out_bonds(), 6);
        assert_eq!(map.total_in_bonds(), 6);
    }

    #[test]
    fn empty_map_is_well_formed() {
        let map = ResidentMap::default();
        assert_eq!(map.node_count(), 0);
        assert_eq!(map.out_degree(NodeId(0)), 0);
        assert!(!map.contains(NodeId(0)));
        assert_eq!(map.resident_bytes(), 0);
    }
}
