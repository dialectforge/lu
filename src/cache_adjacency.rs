//! OP-BRAIN-MapRAM P1 -- the cache-backed adjacency READ PATH
//! ("map -> cache -> v5 reader").
//!
//! See `docs/OP-BRAIN-MapRAM-Design-2026-06-12.md` (P1, sections 4.2 /
//! 4.3 / 5). P0 ([`crate::map_ram::ResidentMap`]) made degree and
//! existence O(1) and resident; P1 turns that map into a neighbour
//! READER: [`CacheBackedAdjacency::neighbors_outgoing`] /
//! [`neighbors_incoming`](CacheBackedAdjacency::neighbors_incoming)
//! resolve a node's adjacency list through
//!
//!   resident MAP (existence + degree, O(1), no I/O)
//!     -> bounded HOT cache (decoded list, ~1us hit, skips the 242us decode)
//!       -> on miss, the v5 reader (`get_(out|in)going_bonds_v5`, the
//!          decrypt + zstd + cross-level merge that costs ~242us),
//!          inserting the decoded list into the cache.
//!
//! The exact graph stays on disk in the v5 bond-store SSTs. This cache
//! is an ACCELERATOR, never the store of record: every list it serves
//! is byte-identical to what the v5 reader (and hence
//! [`crate::brain::Brain::rehydrate_adjacency`]) would produce, and a
//! miss is always satisfiable from disk. A wrong/empty cache can only
//! cost a re-decode, never a wrong answer.
//!
//! ## Scope (P1, read carefully)
//!
//! P1 builds and PROVES the read path behind the `LU_MAPRAM_CACHE`
//! flag. It does **not** reroute the production consumers
//! (`spread_activation`, hub-fill, curiosity, mcp_cat3b) off
//! `network.outgoing`/`incoming`, and it does **not** remove
//! `rehydrate_adjacency` -- that resident adjacency stays the source of
//! truth for parity. The RAM win lands in P2 when consumers move onto
//! this reader and the 46 GiB resident adjacency can go. P1's job is
//! the correct, cache-backed reader and the parity proof.
//!
//! ## What is built here vs. deferred (no over-building)
//!
//! Built: the decoded per-`(node, direction)` adjacency-LIST cache with
//! a scan-resistant **2Q** admission/eviction policy + a PINNED set of
//! the highest-degree hubs (top ~0.1% from the P0 map) that never
//! evict. This is the increment that proves the map->cache->v5 read
//! path and delivers the doc's stated value: "a hit costs ~1us (avoids
//! the 242us decode); this is the cache's real value -- skip decode,
//! not skip I/O" (sec 4.2).
//!
//! Deferred (a distinct, invasive increment): the decoded-BLOCK cache
//! (sec 4.2 "strongly recommended addition"). It would cache
//! decompressed SST blocks so one block-decode serves many subjects --
//! but the v5 read path streams blocks INTERNALLY (`iter_subject` /
//! `iter_all`) and [`crate::bond_store::BondSstReaderCache`]
//! deliberately caches only FD+index, not decoded blocks. A real block
//! cache therefore needs a new block-decode seam threaded through
//! `bond_store::read_path` plus an out-of-reader subject->block index --
//! its own correctness/eviction surface and clearly a SEPARATE clean
//! increment. P1 ships the list cache (which already skips the full
//! per-call decode) and leaves the block cache as a follow-up, exactly
//! as P0 deferred the optional locator hint with an explicit rationale
//! rather than over-build.
//!
//! ## Correctness (the gate's heart)
//!
//! [`V5AdjacencySource`] replays the EXACT filtering
//! `rehydrate_adjacency` applies before a bond lands in adjacency:
//! collapse the v5 reader's `(subject, predicate, object)` rows to one
//! winner per `(subject, object)` pair by highest `seq` (the reader has
//! already merged levels and dropped tombstones), THEN drop the pair if
//! the winning predicate discriminant is invalid
//! (`bond_type_from_discriminant` -> `None`, rehydrate's
//! `skipped_bad_predicate`) or either endpoint is not interned
//! (rehydrate's `skipped_no_seed`). The surviving set of objects (for
//! outgoing) / subjects (for incoming) is therefore the SAME SET
//! `AssociativeNetwork::neighbors_outgoing`/`incoming` holds, with the
//! same winning predicate + payload per neighbour. The order differs
//! (adjacency insertion order vs. reader order), so parity is asserted
//! as a SET, exactly as the P1 brief specifies.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::brain::BrainError;
use crate::map_ram::ResidentMap;
use crate::node_intern::NodeId;

/// Edge direction for a neighbour lookup. `Outgoing` reads a subject's
/// objects; `Incoming` reads an object's subjects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Outgoing,
    Incoming,
}

/// One decoded neighbour: the other endpoint plus the winning bond's
/// predicate discriminant and full [`crate::bond_store::BondPayload`]
/// (weight / confidence / keyframes / seq). "Lists + payloads" per the
/// P1 brief -- so a future spread loop reads weights WITH the neighbour
/// list and never needs a separate `get_bond` per edge.
#[derive(Debug, Clone)]
pub struct DecodedNeighbor {
    /// The neighbour node: the OBJECT for an outgoing read, the SUBJECT
    /// for an incoming read.
    pub node: NodeId,
    /// Winning predicate discriminant (`bond_type_from_discriminant`
    /// maps it back to a [`crate::bond::BondType`]).
    pub predicate: u8,
    /// The winning bond payload (highest `seq` for this `(subject,
    /// object)` pair), carried alongside the list.
    pub payload: crate::bond_store::BondPayload,
}

/// The on-miss resolver: decode a node's neighbour list for a
/// direction, applying the EXACT `rehydrate_adjacency` filtering so the
/// result is set-identical to the resident adjacency. Abstracted as a
/// trait so the cache can be unit-tested with an instrumented mock that
/// counts decode calls, while production uses [`V5AdjacencySource`].
pub trait AdjacencySource {
    /// Decode the live neighbours of `node` in `dir`. Returns one
    /// [`DecodedNeighbor`] per distinct other-endpoint, post-filter.
    fn read_neighbors(
        &self,
        node: NodeId,
        dir: Direction,
    ) -> Result<Vec<DecodedNeighbor>, BrainError>;
}

/// Production [`AdjacencySource`]: the v5 bond-store reader + the exact
/// `rehydrate_adjacency` filter. Holds only cheap `Arc`/clone handles
/// (manifest snapshot, bond reader cache, intern), so
/// [`crate::brain::Brain::v5_adjacency_source`] can hand a fresh one to
/// the cache per call without borrowing the `Brain`.
pub struct V5AdjacencySource {
    manifest: Arc<crate::manifest::Manifest>,
    bond_cache: Arc<crate::bond_store::BondSstReaderCache>,
    intern: crate::node_intern::NodeIntern,
}

impl V5AdjacencySource {
    /// Construct from the handles `Brain` already holds post-flush-worker.
    pub fn new(
        manifest: Arc<crate::manifest::Manifest>,
        bond_cache: Arc<crate::bond_store::BondSstReaderCache>,
        intern: crate::node_intern::NodeIntern,
    ) -> Self {
        Self { manifest, bond_cache, intern }
    }
}

impl AdjacencySource for V5AdjacencySource {
    fn read_neighbors(
        &self,
        node: NodeId,
        dir: Direction,
    ) -> Result<Vec<DecodedNeighbor>, BrainError> {
        // 1. v5 reader rows. The reader has ALREADY done the cross-level
        //    triple merge, within-L0 seq dedup, and tombstone drop
        //    (callers never see confidence == 0xFF), ascending by
        //    (predicate, other-endpoint). Rows are (s, p, o, payload)
        //    preserving the bond's original orientation.
        let rows: Vec<(NodeId, u8, NodeId, crate::bond_store::BondPayload)> = match dir {
            Direction::Outgoing => {
                crate::bond_store::get_outgoing_bonds_v5(&self.manifest, &self.bond_cache, node)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        BrainError::Format(format!("cache_adjacency outgoing read: {}", e))
                    })?
            }
            Direction::Incoming => {
                crate::bond_store::get_incoming_bonds_v5(&self.manifest, &self.bond_cache, node)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        BrainError::Format(format!("cache_adjacency incoming read: {}", e))
                    })?
            }
        };

        // 2. Collapse to one winner per (subject, object) pair by
        //    highest seq -- exactly as rehydrate's `pair_winners`. The
        //    pair's fixed endpoint is `node`; the varying endpoint is
        //    the OBJECT for outgoing, the SUBJECT for incoming.
        //    Tombstones are already gone (the reader dropped them), so
        //    a plain max-seq over reader rows matches rehydrate, which
        //    drops tombstone winners BEFORE the pair collapse.
        let mut winners: HashMap<u64, (u8, crate::bond_store::BondPayload)> = HashMap::new();
        for (s, p, o, payload) in rows {
            let other = match dir {
                Direction::Outgoing => o.0,
                Direction::Incoming => s.0,
            };
            match winners.entry(other) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((p, payload));
                }
                std::collections::hash_map::Entry::Occupied(mut occ) => {
                    if payload.seq > occ.get().1.seq {
                        occ.insert((p, payload));
                    }
                }
            }
        }

        // 3. Apply rehydrate's post-collapse filters on the WINNER:
        //    - invalid predicate discriminant => skip the pair
        //      (rehydrate: `skipped_bad_predicate`).
        //    - either endpoint not interned => skip the pair
        //      (rehydrate: `skipped_no_seed`). `node` is the fixed
        //      endpoint; `other` the varying one.
        let mut out: Vec<DecodedNeighbor> = Vec::with_capacity(winners.len());
        let self_interned = self.intern.reverse_lookup(node).is_some();
        for (other_id, (p, payload)) in winners {
            if crate::memtable::bond_type_from_discriminant(p).is_none() {
                continue;
            }
            if !self_interned || self.intern.reverse_lookup(NodeId(other_id)).is_none() {
                continue;
            }
            out.push(DecodedNeighbor { node: NodeId(other_id), predicate: p, payload });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// Scan-resistant 2Q cache internals.
//
// Classic 2Q (Johnson & Shasha, VLDB'94, the full variant):
//   * A1in -- a FIFO of first-time references. A scan that touches each
//     node once flows THROUGH A1in and out to the ghost queue without
//     ever entering the hot set, so a single hub query cannot evict the
//     working set. (No reorder on an A1in hit -- FIFO, not LRU.)
//   * A1out -- a ghost queue of KEYS only (no data) evicted from A1in.
//     A reference that hits a ghost is a PROVEN reuse and is promoted to
//     Am.
//   * Am -- the hot LRU of reused items.
// This is the scan-resistance the design doc (sec 4.2) requires: "a
// plain LRU would be churned by [100k+ one-shot] scans and evict the
// useful pinned/working set ... LRU alone is fragile here."
// ---------------------------------------------------------------------

type Key = (u64, Direction);

/// One slot in an intrusive doubly linked list keyed by [`Key`].
struct LinkNode {
    data: Arc<[DecodedNeighbor]>,
    prev: Option<Key>,
    next: Option<Key>,
}

/// O(1) intrusive doubly linked list + hash index. `head` is MRU,
/// `tail` is LRU. Serves both the LRU `Am` and the FIFO `A1in` (the
/// FIFO simply never calls [`Self::move_to_front`] on a hit).
#[derive(Default)]
struct LinkedMap {
    map: HashMap<Key, LinkNode>,
    head: Option<Key>,
    tail: Option<Key>,
}

impl LinkedMap {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn get_clone(&self, key: &Key) -> Option<Arc<[DecodedNeighbor]>> {
        self.map.get(key).map(|n| n.data.clone())
    }

    fn push_front(&mut self, key: Key, data: Arc<[DecodedNeighbor]>) {
        let old_head = self.head;
        self.map.insert(key, LinkNode { data, prev: None, next: old_head });
        if let Some(h) = old_head {
            if let Some(n) = self.map.get_mut(&h) {
                n.prev = Some(key);
            }
        }
        self.head = Some(key);
        if self.tail.is_none() {
            self.tail = Some(key);
        }
    }

    fn unlink(&mut self, key: &Key) -> Option<Arc<[DecodedNeighbor]>> {
        let node = self.map.remove(key)?;
        match node.prev {
            Some(p) => {
                if let Some(n) = self.map.get_mut(&p) {
                    n.next = node.next;
                }
            }
            None => self.head = node.next,
        }
        match node.next {
            Some(nx) => {
                if let Some(n) = self.map.get_mut(&nx) {
                    n.prev = node.prev;
                }
            }
            None => self.tail = node.prev,
        }
        Some(node.data)
    }

    fn move_to_front(&mut self, key: &Key) {
        if self.head == Some(*key) {
            return;
        }
        if let Some(data) = self.unlink(key) {
            self.push_front(*key, data);
        }
    }

    fn pop_tail(&mut self) -> Option<Key> {
        let t = self.tail?;
        self.unlink(&t);
        Some(t)
    }
}

/// Ghost queue: bounded set of recently-evicted-from-A1in KEYS (no
/// data). Lazy removal -- a promoted key is dropped from `set` and the
/// stale `queue` slot is skipped on eviction, keeping every op O(1)
/// amortized.
#[derive(Default)]
struct GhostQueue {
    queue: VecDeque<Key>,
    set: HashSet<Key>,
    cap: usize,
}

impl GhostQueue {
    fn contains(&self, key: &Key) -> bool {
        self.set.contains(key)
    }

    fn remove(&mut self, key: &Key) {
        self.set.remove(key); // queue slot left stale, skipped on evict
    }

    fn push(&mut self, key: Key) {
        if self.cap == 0 || self.set.contains(&key) {
            return;
        }
        self.queue.push_back(key);
        self.set.insert(key);
        while self.set.len() > self.cap {
            match self.queue.pop_front() {
                Some(old) => {
                    self.set.remove(&old); // may be a no-op for a stale slot
                }
                None => break,
            }
        }
    }
}

/// Result of probing the 2Q cache for a key.
enum Probe {
    Hit(Arc<[DecodedNeighbor]>),
    /// Miss, but the key is a ghost (proven reuse): admit to `Am`.
    Ghost,
    /// Cold miss: admit to `A1in`.
    Cold,
}

/// The 2Q cache: `Am` (hot LRU) + `A1in` (FIFO) + `A1out` (ghost keys).
#[derive(Default)]
struct TwoQCache {
    am: LinkedMap,
    a1in: LinkedMap,
    a1out: GhostQueue,
    am_cap: usize,
    a1in_cap: usize,
}

impl TwoQCache {
    fn new(cap: usize) -> Self {
        // Classic 2Q split: A1in (= A1out) ~ 1/4 of the data budget,
        // Am the remainder. Both floored at 1 so a tiny cap still
        // caches.
        let a1in_cap = (cap / 4).max(1);
        let am_cap = cap.saturating_sub(a1in_cap).max(1);
        TwoQCache {
            am: LinkedMap::default(),
            a1in: LinkedMap::default(),
            a1out: GhostQueue { cap: a1in_cap, ..GhostQueue::default() },
            am_cap,
            a1in_cap,
        }
    }

    fn probe(&mut self, key: &Key) -> Probe {
        if let Some(d) = self.am.get_clone(key) {
            self.am.move_to_front(key); // Am is LRU
            return Probe::Hit(d);
        }
        if let Some(d) = self.a1in.get_clone(key) {
            return Probe::Hit(d); // A1in is FIFO -- no reorder on hit
        }
        if self.a1out.contains(key) {
            self.a1out.remove(key);
            return Probe::Ghost;
        }
        Probe::Cold
    }

    /// Admit `data` after a miss. `ghost` => promote to `Am` (proven
    /// reuse); else => first-reference into `A1in`. Idempotent under
    /// races: any prior slot for `key` is unlinked first, so a duplicate
    /// concurrent miss cannot corrupt the lists.
    fn admit(&mut self, key: Key, data: Arc<[DecodedNeighbor]>, ghost: bool) {
        self.am.unlink(&key);
        self.a1in.unlink(&key);
        self.a1out.remove(&key);
        if ghost {
            self.am.push_front(key, data);
            while self.am.len() > self.am_cap {
                self.am.pop_tail();
            }
        } else {
            self.a1in.push_front(key, data);
            while self.a1in.len() > self.a1in_cap {
                if let Some(evicted) = self.a1in.pop_tail() {
                    self.a1out.push(evicted);
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.am.len() + self.a1in.len()
    }
}

struct Inner {
    cache: TwoQCache,
    /// Pinned hubs: never evicted, never counted against the 2Q caps.
    /// Keyed per `(node, direction)`; filled lazily (or via
    /// [`CacheBackedAdjacency::prewarm_pins`]) and then resident for the
    /// brain's life.
    pinned: HashMap<Key, Arc<[DecodedNeighbor]>>,
}

/// P1's cache-backed adjacency reader. See the module docs.
pub struct CacheBackedAdjacency {
    map: Arc<ResidentMap>,
    /// Node ids pinned in BOTH directions (top-degree hubs).
    pinned_ids: HashSet<u64>,
    inner: Mutex<Inner>,
    cap: usize,
    // Telemetry (P4 will surface these; useful for the P1 tests too).
    hits: AtomicU64,
    misses: AtomicU64,
    reader_calls: AtomicU64,
    map_short_circuits: AtomicU64,
}

fn empty_list() -> Arc<[DecodedNeighbor]> {
    Arc::from(Vec::<DecodedNeighbor>::new().into_boxed_slice())
}

impl CacheBackedAdjacency {
    /// Construct with an explicit pin COUNT (top-`pin_count` nodes by
    /// total degree from the resident map). Used by tests for
    /// deterministic pinning; production uses [`Self::from_fraction`].
    pub fn with_pin_count(map: Arc<ResidentMap>, cap: usize, pin_count: usize) -> Self {
        let pinned_ids = Self::select_top_hubs(&map, pin_count);
        CacheBackedAdjacency {
            map,
            pinned_ids,
            inner: Mutex::new(Inner {
                cache: TwoQCache::new(cap.max(1)),
                pinned: HashMap::new(),
            }),
            cap: cap.max(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            reader_calls: AtomicU64::new(0),
            map_short_circuits: AtomicU64::new(0),
        }
    }

    /// Construct pinning the top `pin_fraction` of nodes by degree
    /// (e.g. `0.001` = top 0.1% hubs, per the design doc). Production
    /// entry point.
    pub fn from_fraction(map: Arc<ResidentMap>, cap: usize, pin_fraction: f64) -> Self {
        let n = map.node_count();
        let pin_count = ((n as f64) * pin_fraction.max(0.0)).round() as usize;
        Self::with_pin_count(map, cap, pin_count.min(n))
    }

    /// Pick the `count` highest-total-degree node ids from the map.
    /// O(nodes) scan keeping a bounded min-heap of size `count`.
    fn select_top_hubs(map: &ResidentMap, count: usize) -> HashSet<u64> {
        if count == 0 {
            return HashSet::new();
        }
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        // Min-heap on degree (via Reverse) so the smallest sits on top
        // and is the one popped when a bigger hub arrives.
        let mut heap: BinaryHeap<Reverse<(u64, u64)>> = BinaryHeap::with_capacity(count + 1);
        for i in 0..map.node_count() as u64 {
            let id = NodeId(i);
            let deg = map.total_degree(id);
            if deg == 0 {
                continue; // an isolated node is never a hub
            }
            if heap.len() < count {
                heap.push(Reverse((deg, i)));
            } else if let Some(Reverse((min_deg, _))) = heap.peek().copied() {
                if deg > min_deg {
                    heap.pop();
                    heap.push(Reverse((deg, i)));
                }
            }
        }
        heap.into_iter().map(|Reverse((_, id))| id).collect()
    }

    /// Decoded outgoing neighbours of `subject` (objects + payloads),
    /// served via map -> cache -> v5 reader. See [`Self::neighbors`].
    pub fn neighbors_outgoing(
        &self,
        subject: NodeId,
        source: &dyn AdjacencySource,
    ) -> Result<Arc<[DecodedNeighbor]>, BrainError> {
        self.neighbors(subject, Direction::Outgoing, source)
    }

    /// Decoded incoming neighbours of `object` (subjects + payloads),
    /// served via map -> cache -> v5 reader. See [`Self::neighbors`].
    pub fn neighbors_incoming(
        &self,
        object: NodeId,
        source: &dyn AdjacencySource,
    ) -> Result<Arc<[DecodedNeighbor]>, BrainError> {
        self.neighbors(object, Direction::Incoming, source)
    }

    /// Core read path: resident MAP (existence + degree, O(1), no I/O)
    /// -> hot 2Q cache (decoded list) -> `source` (v5 reader) on miss,
    /// inserting the decoded list. Pinned hubs are checked first and
    /// never evict.
    pub fn neighbors(
        &self,
        node: NodeId,
        dir: Direction,
        source: &dyn AdjacencySource,
    ) -> Result<Arc<[DecodedNeighbor]>, BrainError> {
        // --- Resident-map short-circuits (no cache, no I/O). ---
        // An absent node has no edges (matches the network accessors'
        // empty answer for an unknown seed).
        if !self.map.contains(node) {
            self.map_short_circuits.fetch_add(1, Ordering::Relaxed);
            return Ok(empty_list());
        }
        // Degree 0 in this direction => no neighbours, provably (P0:
        // map degree is byte-exact to the resident adjacency degree).
        // Serve empty WITHOUT touching the cache or disk -- pure
        // sinks/sources never decode.
        let degree = match dir {
            Direction::Outgoing => self.map.out_degree(node),
            Direction::Incoming => self.map.in_degree(node),
        };
        if degree == 0 {
            self.map_short_circuits.fetch_add(1, Ordering::Relaxed);
            return Ok(empty_list());
        }

        let key = (node.0, dir);

        // --- Probe pinned set + 2Q cache under the lock. ---
        let ghost;
        {
            let mut inner = self.inner.lock().expect("cache_adjacency lock");
            if let Some(d) = inner.pinned.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(d.clone());
            }
            match inner.cache.probe(&key) {
                Probe::Hit(d) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(d);
                }
                Probe::Ghost => ghost = true,
                Probe::Cold => ghost = false,
            }
        }

        // --- Miss: decode via the v5 reader (lock released). ---
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.reader_calls.fetch_add(1, Ordering::Relaxed);
        let list = source.read_neighbors(node, dir)?;
        // Internal consistency: the post-filter neighbour count MUST
        // equal the resident map's degree (P0 parity). A mismatch means
        // the filter and the map disagree -- a correctness bug, caught
        // here in debug builds rather than silently served.
        debug_assert_eq!(
            list.len(),
            degree as usize,
            "cache_adjacency: decoded count {} != map degree {} for node {} dir {:?}",
            list.len(),
            degree,
            node.0,
            dir,
        );
        let arc: Arc<[DecodedNeighbor]> = Arc::from(list.into_boxed_slice());

        // --- Insert: pinned hubs go to the immune pinned map; everyone
        //     else into the 2Q cache. ---
        {
            let mut inner = self.inner.lock().expect("cache_adjacency lock");
            if self.pinned_ids.contains(&node.0) {
                inner.pinned.insert(key, arc.clone());
            } else {
                inner.cache.admit(key, arc.clone(), ghost);
            }
        }
        Ok(arc)
    }

    /// Eagerly load every pinned hub (both directions) into the immune
    /// pinned map so a pinned hub is never a first-access miss -- the
    /// boot pre-warm the design doc describes (sec 4.4). Idempotent.
    pub fn prewarm_pins(&self, source: &dyn AdjacencySource) -> Result<usize, BrainError> {
        let ids: Vec<u64> = self.pinned_ids.iter().copied().collect();
        let mut warmed = 0usize;
        for id in ids {
            for dir in [Direction::Outgoing, Direction::Incoming] {
                self.neighbors(NodeId(id), dir, source)?;
                warmed += 1;
            }
        }
        Ok(warmed)
    }

    /// Number of pinned hub node ids (each pinned in both directions).
    pub fn pin_count(&self) -> usize {
        self.pinned_ids.len()
    }

    /// True iff `node` is a pinned hub.
    pub fn is_pinned(&self, node: NodeId) -> bool {
        self.pinned_ids.contains(&node.0)
    }

    /// Configured non-pinned data capacity (decoded lists).
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// OP-BRAIN-MapRAM P2 -- O(1) out-degree from the resident map
    /// (byte-exact to the resident adjacency degree by P0). Lets the
    /// network's adjacency seam read `deg_norm` without touching the
    /// 2Q cache or disk.
    pub fn out_degree(&self, node: NodeId) -> u32 {
        self.map.out_degree(node)
    }

    /// OP-BRAIN-MapRAM P2 -- O(1) in-degree from the resident map. See
    /// [`Self::out_degree`].
    pub fn in_degree(&self, node: NodeId) -> u32 {
        self.map.in_degree(node)
    }

    /// OP-BRAIN-MapRAM P3 -- O(1) existence in the resident MAP. Lets
    /// the spread seam reproduce `view_node`'s rehydration-stub answer
    /// (present -> stub, absent -> None) WITHOUT reading the resident
    /// node structures.
    pub fn map_contains(&self, node: NodeId) -> bool {
        self.map.contains(node)
    }

    /// OP-RECALL-1 (R5) -- O(1) NODE TOTAL from the resident map.
    ///
    /// The resident map is sized to the interned node count at build
    /// time (one dense slot per `NodeId`), so its length IS the node
    /// count of the brain the cache is backing. Cache-backed v5 brains
    /// keep `AssociativeNetwork::nodes` empty and have no v3 archive,
    /// so this is the ONLY correct answer for `node_count()` in that
    /// mode -- without it the whole TF-IDF layer sees `n = 0`.
    pub fn node_count(&self) -> usize {
        self.map.node_count()
    }

    /// OP-RECALL-1 (R5) -- O(N) BOND TOTAL from the resident map.
    ///
    /// Sum of every out-degree, i.e. the counted bond total of the
    /// backing graph. Deliberately NOT on a hot path: the network
    /// memoizes it in `bond_count_cache` after the first call.
    pub fn total_bonds(&self) -> u64 {
        self.map.total_out_bonds()
    }

    /// OP-BRAIN-MapRAM P3 -- `degree_p90` from the resident MAP (the
    /// floor(0.9*n) order statistic of per-node total degree, floored
    /// at 1). Byte-identical to `AssociativeNetwork::recompute_degree_p90`
    /// over the rehydrated adjacency (same statistic of the same
    /// degrees), so the hub threshold can be sourced from the map.
    pub fn degree_p90(&self) -> usize {
        let n = self.map.node_count();
        if n == 0 {
            return crate::network::DEGREE_NORM_THRESHOLD;
        }
        let mut degs: Vec<usize> = (0..n as u64)
            .map(|i| self.map.total_degree(NodeId(i)) as usize)
            .collect();
        degs.sort_unstable();
        let idx = (((degs.len() as f64) * 0.9).floor() as usize).min(degs.len() - 1);
        degs[idx].max(1)
    }

    /// Cache hits (pinned + 2Q). Telemetry.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cache misses that fell through to the reader. Telemetry.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Total v5-reader decode calls. Telemetry + the pin/eviction tests
    /// assert on this (a served-from-cache read must NOT bump it).
    pub fn reader_calls(&self) -> u64 {
        self.reader_calls.load(Ordering::Relaxed)
    }

    /// Lookups answered purely from the resident map (absent or
    /// degree-0), never touching cache or disk. Telemetry.
    pub fn map_short_circuits(&self) -> u64 {
        self.map_short_circuits.load(Ordering::Relaxed)
    }

    /// Current resident decoded-list count: 2Q entries + pinned entries.
    pub fn resident_lists(&self) -> usize {
        let inner = self.inner.lock().expect("cache_adjacency lock");
        inner.cache.len() + inner.pinned.len()
    }

    /// Current count of resident PINNED decoded lists.
    pub fn pinned_lists(&self) -> usize {
        let inner = self.inner.lock().expect("cache_adjacency lock");
        inner.pinned.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond_store::BondPayload;

    fn payload(seq: u64) -> BondPayload {
        BondPayload { weight: 200, confidence: 1, keyframes: Vec::new(), seq }
    }

    fn nb(id: u64, seq: u64) -> DecodedNeighbor {
        DecodedNeighbor { node: NodeId(id), predicate: 5, payload: payload(seq) }
    }

    /// An instrumented in-memory source: a fixed adjacency, plus a call
    /// counter per (node, direction) so the cache tests can prove a
    /// served-from-cache read did NOT re-decode.
    struct MockSource {
        out: HashMap<u64, Vec<DecodedNeighbor>>,
        inc: HashMap<u64, Vec<DecodedNeighbor>>,
        calls: Mutex<HashMap<Key, u64>>,
    }

    impl MockSource {
        fn calls_for(&self, node: u64, dir: Direction) -> u64 {
            *self.calls.lock().unwrap().get(&(node, dir)).unwrap_or(&0)
        }
        fn total_calls(&self) -> u64 {
            self.calls.lock().unwrap().values().sum()
        }
    }

    impl AdjacencySource for MockSource {
        fn read_neighbors(
            &self,
            node: NodeId,
            dir: Direction,
        ) -> Result<Vec<DecodedNeighbor>, BrainError> {
            *self.calls.lock().unwrap().entry((node.0, dir)).or_insert(0) += 1;
            let m = match dir {
                Direction::Outgoing => &self.out,
                Direction::Incoming => &self.inc,
            };
            Ok(m.get(&node.0).cloned().unwrap_or_default())
        }
    }

    /// Build a map whose degrees match the mock adjacency exactly (so
    /// the cache's degree short-circuit and debug_assert agree).
    fn map_for(out: &HashMap<u64, Vec<DecodedNeighbor>>, inc: &HashMap<u64, Vec<DecodedNeighbor>>, n: u64) -> Arc<ResidentMap> {
        let mut od = vec![0u32; n as usize];
        let mut idg = vec![0u32; n as usize];
        for (k, v) in out {
            od[*k as usize] = v.len() as u32;
        }
        for (k, v) in inc {
            idg[*k as usize] = v.len() as u32;
        }
        Arc::new(ResidentMap::from_degrees(od, idg))
    }

    fn sorted_ids(list: &[DecodedNeighbor]) -> Vec<u64> {
        let mut v: Vec<u64> = list.iter().map(|n| n.node.0).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn cold_miss_then_cache_hit_no_second_decode() {
        let mut out = HashMap::new();
        out.insert(1u64, vec![nb(2, 1), nb(3, 1)]);
        let inc = HashMap::new();
        let map = map_for(&out, &inc, 8);
        let src = MockSource { out, inc, calls: Mutex::new(HashMap::new()) };
        let cache = CacheBackedAdjacency::with_pin_count(map, 16, 0);

        // Cold: one decode.
        let a = cache.neighbors_outgoing(NodeId(1), &src).unwrap();
        assert_eq!(sorted_ids(&a), vec![2, 3]);
        assert_eq!(src.calls_for(1, Direction::Outgoing), 1);
        assert_eq!(cache.reader_calls(), 1);

        // Warm: served from cache, NO second decode.
        let b = cache.neighbors_outgoing(NodeId(1), &src).unwrap();
        assert_eq!(sorted_ids(&b), vec![2, 3]);
        assert_eq!(src.calls_for(1, Direction::Outgoing), 1, "must not re-decode");
        assert_eq!(cache.reader_calls(), 1);
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn degree_zero_and_absent_short_circuit_without_decode() {
        let out: HashMap<u64, Vec<DecodedNeighbor>> = HashMap::new();
        let inc: HashMap<u64, Vec<DecodedNeighbor>> = HashMap::new();
        let map = map_for(&out, &inc, 4); // all degrees 0
        let src = MockSource { out, inc, calls: Mutex::new(HashMap::new()) };
        let cache = CacheBackedAdjacency::with_pin_count(map, 8, 0);

        // Degree-0 node: empty, no decode.
        let a = cache.neighbors_outgoing(NodeId(0), &src).unwrap();
        assert!(a.is_empty());
        // Absent node (id beyond map): empty, no decode.
        let b = cache.neighbors_incoming(NodeId(999), &src).unwrap();
        assert!(b.is_empty());
        assert_eq!(src.total_calls(), 0, "map short-circuit must not decode");
        assert_eq!(cache.reader_calls(), 0);
        assert_eq!(cache.map_short_circuits(), 2);
    }

    #[test]
    fn eviction_then_refetch_is_still_correct() {
        // Tiny cap forces eviction; a one-shot scan of many nodes pushes
        // earlier entries out. The refetched list must still be exact.
        let mut out = HashMap::new();
        for id in 1u64..=20 {
            out.insert(id, vec![nb(id + 100, 1)]);
        }
        let inc = HashMap::new();
        let map = map_for(&out, &inc, 200);
        let src = MockSource { out, inc, calls: Mutex::new(HashMap::new()) };
        let cache = CacheBackedAdjacency::with_pin_count(map, 4, 0); // cap 4

        // Scan all 20 once -> 20 cold decodes, cache holds <= 4 + ghosts.
        for id in 1u64..=20 {
            let l = cache.neighbors_outgoing(NodeId(id), &src).unwrap();
            assert_eq!(sorted_ids(&l), vec![id + 100]);
        }
        assert_eq!(cache.reader_calls(), 20);
        assert!(cache.resident_lists() <= 4, "2Q must bound resident lists to cap");

        // Node 1 was evicted long ago: refetch re-decodes, still exact.
        let again = cache.neighbors_outgoing(NodeId(1), &src).unwrap();
        assert_eq!(sorted_ids(&again), vec![101]);
        assert!(
            src.calls_for(1, Direction::Outgoing) >= 2,
            "evicted entry must have been re-decoded",
        );
    }

    #[test]
    fn pinned_hub_survives_eviction_pressure() {
        // Node 1 is a high-degree hub; pin the top 1. Fill the cache with
        // a long one-shot scan, then assert the hub is STILL served from
        // the pinned map with no extra decode.
        let mut out = HashMap::new();
        out.insert(1u64, (50u64..60).map(|o| nb(o, 1)).collect()); // hub: deg 10
        for id in 2u64..=40 {
            out.insert(id, vec![nb(id + 100, 1)]); // deg 1 each
        }
        let inc = HashMap::new();
        let map = map_for(&out, &inc, 200);
        let src = MockSource { out, inc, calls: Mutex::new(HashMap::new()) };
        let cache = CacheBackedAdjacency::with_pin_count(map, 4, 1); // pin top 1 = node 1
        assert!(cache.is_pinned(NodeId(1)));

        // Warm the hub (pinned), then scan 39 other nodes (eviction
        // pressure that would churn a plain LRU).
        let h0 = cache.neighbors_outgoing(NodeId(1), &src).unwrap();
        assert_eq!(h0.len(), 10);
        let after_hub = src.calls_for(1, Direction::Outgoing);
        for id in 2u64..=40 {
            cache.neighbors_outgoing(NodeId(id), &src).unwrap();
        }
        // Hub still resident in the pinned map: served with NO new decode.
        let h1 = cache.neighbors_outgoing(NodeId(1), &src).unwrap();
        assert_eq!(sorted_ids(&h1), (50u64..60).collect::<Vec<_>>());
        assert_eq!(
            src.calls_for(1, Direction::Outgoing),
            after_hub,
            "pinned hub must not be re-decoded after eviction pressure",
        );
        assert_eq!(cache.pinned_lists(), 1);
    }

    #[test]
    fn two_q_is_scan_resistant_for_reused_keys() {
        // A reused key, once promoted to Am via a ghost hit, survives a
        // subsequent one-shot scan that would evict it from a plain FIFO.
        let mut out = HashMap::new();
        out.insert(7u64, vec![nb(700, 1)]); // the reused key
        for id in 10u64..=60 {
            out.insert(id, vec![nb(id + 100, 1)]);
        }
        let inc = HashMap::new();
        let map = map_for(&out, &inc, 200);
        let src = MockSource { out, inc, calls: Mutex::new(HashMap::new()) };
        let cache = CacheBackedAdjacency::with_pin_count(map, 8, 0);

        // Reference 7 (A1in), then evict it to the ghost queue with
        // EXACTLY a1in_cap (= cap/4 = 2) further first-references so 7's
        // ghost entry SURVIVES -- a longer scan would also evict the
        // ghost and 7 could never be promoted. Re-referencing 7 is then
        // a ghost hit, which promotes it to the hot Am queue.
        cache.neighbors_outgoing(NodeId(7), &src).unwrap();
        cache.neighbors_outgoing(NodeId(10), &src).unwrap();
        cache.neighbors_outgoing(NodeId(11), &src).unwrap();
        cache.neighbors_outgoing(NodeId(7), &src).unwrap(); // ghost hit -> Am
        let calls_after_promote = src.calls_for(7, Direction::Outgoing);

        // Another one-shot scan; an Am resident should survive it.
        for id in 31u64..=60 {
            cache.neighbors_outgoing(NodeId(id), &src).unwrap();
        }
        let l = cache.neighbors_outgoing(NodeId(7), &src).unwrap();
        assert_eq!(sorted_ids(&l), vec![700]);
        assert_eq!(
            src.calls_for(7, Direction::Outgoing),
            calls_after_promote,
            "a promoted (Am) key must survive a one-shot scan -- scan resistance",
        );
    }

    #[test]
    fn top_hub_selection_picks_highest_degree() {
        // degrees: id3 total 5, id1 total 3, id0 total 1 -> top 2 = {3,1}.
        let map = Arc::new(ResidentMap::from_degrees(
            vec![1, 2, 0, 3],
            vec![0, 1, 0, 2],
        ));
        let cache = CacheBackedAdjacency::with_pin_count(map, 8, 2);
        assert!(cache.is_pinned(NodeId(3)));
        assert!(cache.is_pinned(NodeId(1)));
        assert!(!cache.is_pinned(NodeId(0)));
        assert!(!cache.is_pinned(NodeId(2)));
        assert_eq!(cache.pin_count(), 2);
    }
}
