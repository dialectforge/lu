# OP-BRAIN — "Map in RAM, Brain on Disk" with Predictive Routing — Design + Measurement

- Status: DESIGN / CHECKPOINT (measurement complete, not implemented)
- Date: 2026-06-12
- Author: John (dialectforge) — measurement + spec drafted in-session
- Repo HEAD: 750eb9c
- Measured against: /var/lib/lu/brain-v5-20260611-enc-pretrain (15,592,593 nodes / 38,502,866 bonds), read-only, training stopped.
- Box: Hetzner AX41, 12 cores, 62 GiB RAM, 2x Toshiba KXG60ZNV512G NVMe (md RAID1).

---

## 0. TL;DR / Verdict

Goal: make resident RAM track the WORKING SET being traversed, not the TOTAL brain, losslessly, with the exact graph on disk. Today Phase 3.8.1 rehydrates ALL adjacency resident (~46 GiB choice); the resident adjacency is the structural ceiling (crosses 56 GiB at ~82M bonds; full Wikipedia ~111M needs ~76 GiB). The bonds already live on disk (v5 bond-store SSTs) with an on-demand reader (get_outgoing_bonds_v5 + BondSstReaderCache, Phase 3.7).

Measured verdict on traversal locality: **LOW. A simple bounded cache does NOT keep traversal fast on hits — the access stream is dominated by COMPULSORY (first-touch) misses (79-98%), and what reuse exists sits at very large reuse distances (>=100k).** Realistic correlated (curiosity-session) traffic tops out at ~7% hit at a 1%-of-nodes cache and ~20-22% even at a 5-10% cache. So:

- The **simple map + bounded cache + on-demand disk reads is VIABLE** (it delivers the fixed-RAM win and scales to any brain size) and is acceptable under John's explicit decision to trade traversal speed for RAM. It is viable not because the cache hits a lot (it does not) but because (a) per-query working sets are small (median 20-27 adjacency reads) and (b) the encoded bond store is tiny (2.1 GiB; ~6 GiB at full Wiki) so it lives in OS page cache.
- **The cache is NOT the speed mechanism. The PREDICTIVE PREFETCH ROUTER is.** Because misses are overwhelmingly compulsory, a reactive cache cannot help them by definition — **prefetch is the only lever for compulsory misses.** The router is therefore correctly framed (per John) as the OPTIMIZATION that claws speed back, not a precondition. v1 (cheap heuristic + parallel batch prefetch) ships the win; v2 (learned co-activation predictor) is upside.

The dominant per-read cost is NOT disk I/O — it is the CPU of the on-demand decode path: **measured ~242 us per outgoing lookup warm (page-cache resident bytes), vs ~1 us for the resident HashMap path today** — a ~240x per-lookup penalty (manifest level-walk + decrypt + zstd + LSM merge/shadow/tombstone). This reframes the cache and the router: cache DECODED blocks/lists (avoid the 242 us), and PARALLELIZE prefetch across the 12 cores (the cost is CPU, so it parallelizes well).

---

## 1. Part A — Traversal locality measurement

### 1.1 Method (faithful to real traversal)

The simulation replicates spread_activation_weighted_inner (network.rs:2431) exactly, driven over the REAL resident graph of the loaded pretrain snapshot via public accessors:
- depth = 3 (production default; mcp.rs:405/586), seed weight 1.0, initial activation = weight*confidence.
- FIFO frontier BFS; per popped node read outgoing+incoming neighbor lists (neighbors_outgoing_capped / neighbors_incoming_capped — exactly what the V3Only read path uses).
- per-edge activation = cur_act * bond_weight * DECAY_PER_HOP(0.7) * nbr_conf * deg_norm * stopword_decay; ACTIVATION_THRESHOLD 0.001; incoming weight halved; degree-normalisation deg_norm = sqrt(degree_p90/deg) for hubs; stopwords (is_hub_stopword, network.rs:51) visited but never expanded; activation cap = effective_cap_activations().
- Real bond weights (latest_raw_weight), real per-node confidence, real degree (out+in), real degree_p90, real stopword set. Only approximation: scope/scar/decay/sense runtimes left at defaults (none set on a freshly-loaded brain), i.e. effective_weight == stored weight — the production default for a recall.
- An adjacency READ = a node being popped/expanded (its full neighbor list is materialised once). Per-neighbor DEGREE lookups are O(1) and in the target design are served from the resident MAP, so they are NOT counted as cache/disk accesses.
- Cache metrics computed by exact LRU stack-distance (Mattson, Fenwick tree): reuse-distance distribution and LRU hit-rate at any cache size in one pass. Hub-pinning modelled by removing the pinned set (top 0.1% by degree) from the LRU stream (pinned = always-hit) and running LRU over the remainder with capacity (C - pinned).
- Harness (throwaway, OUTSIDE the repo): /opt/lu/op_brain_measure/ (src/main.rs locality, src/bench.rs latency, nvme_lat.py). Repo untouched except this doc.

Three workloads bracket reality:
- SCATTERED: 800 independent random single-seed queries (pessimistic — no session locality).
- HUB_SEEDED: 200 queries seeded from random top-1% hubs.
- SESSION: 30 curiosity sessions x 25 queries; seed drifts along outgoing bonds within a topic (realistic correlated traffic).

### 1.2 Graph shape / hub concentration

- nodes = 15,592,593 ; bonds = 38,502,866 ; degree_p90 = 10 (90% of nodes have total degree <= 10 — sparse long tail).
- max degree = 99,879 ; mean of top-10 = 59,030.
- Degree concentration (share of all (out+in) degree):
  - top 0.01% (1,560 nodes): 10.1%
  - top 0.1%  (15,593 nodes): 18.4%
  - top 1%    (155,926 nodes): 33.2%
  - top 5%    (779,630 nodes): 56.1%
- Stopwords among the top-1000 hubs: 0 (the function-word stopword sink does NOT coincide with the structural hubs; hub-pinning and stopword-skip are independent levers).

Interpretation: a MODERATE power law. There is no handful of mega-hubs that dominate all traversal — the top 1% holds only ~1/3 of degree. Hub-pinning helps a little, not a lot.

### 1.3 Working set per query and per session

| Workload  | mean | p50 | p90  | max     | distinct touched (over workload) |
|-----------|------|-----|------|---------|----------------------------------|
| SCATTERED | 173  | 20  | 423  | 8,866   | 136,158  (0.87% of brain)        |
| HUB_SEEDED| 4,545| 3,224|6,983| 100,936 | 778,767  (4.99% of brain)        |
| SESSION   | 1,935| 27  | 1,324| 113,965 | 1,148,274 (7.36% of brain)       |

Typical queries are SMALL (median 20-27 adjacency reads). Means are pulled up by occasional hub-crossing traversals; the max (100k+) is the pathological hub traversal.

### 1.4 Reuse-distance distribution (fraction of accesses)

| Workload  | dist 0 | 1-9 | 10-99 | 100-999 | 1k-10k | 10k-100k | >=100k | compulsory |
|-----------|--------|-----|-------|---------|--------|----------|--------|------------|
| SCATTERED | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 1.4 | 0.1 | 98.3 |
| HUB_SEEDED| 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 3.6 | 10.3 | 85.7 |
| SESSION   | 0.0 | 0.0 | 0.0 | 0.3 | 0.7 | 3.8 | 16.0 | 79.1 |

The access stream is compulsory-miss dominated, and the reuse that exists is at LARGE distances. There is essentially no short-range temporal locality to exploit.

### 1.5 LRU hit-rate at bounded cache sizes (cache as % of nodes)

| Workload  | c=0.5% | c=1% | c=5% | c=10% | +pin0.1% c=0.5/1/5/10% |
|-----------|--------|------|------|-------|------------------------|
| SCATTERED | 1.4 | 1.7 | 1.7 | 1.7 | 2.4 / 2.9 / 2.9 / 2.9 |
| HUB_SEEDED| 3.3 | 5.9 | 14.3| 14.3 | 4.3 / 7.1 / 15.4 / 15.4 |
| SESSION   | 3.9 | 6.9 | 19.8| 20.9 | 5.9 / 8.7 / 20.7 / 21.8 |

(c=0.5% ~ 78k entries; c=1% ~ 156k; c=5% ~ 780k; c=10% ~ 1.56M.)

### 1.6 Verdict

Locality is LOW and miss-dominated. A modest cache (<= ~1% of nodes, a few hundred MB of decoded lists) buys ~4-9% hit on realistic traffic; a large 5-10% cache only reaches ~20-22% and is not worth the RAM. **Do not size the cache to chase a hit rate that is not there.** The simple map+cache stays correct and bounded but will NOT be fast on cache hits — most reads go to the decode path. That is acceptable per the speed-for-RAM decision; the prefetch router (Section 4) is the speed mechanism because the misses are compulsory.

---

## 2. Speed / RAM trade (the decision artifact)

### 2.1 Measured per-lookup latencies

- Resident path TODAY (neighbors_outgoing_capped over rehydrated HashMap): p50 0.81 us, mean 0.99 us, p99 3.09 us.
- On-demand v5 path (Brain::get_outgoing_bonds_v5), WARM (bond store page-cached): **p50 242 us, mean 241 us, p90 259 us, p99 295 us** (avg 9.6 bonds/subject). Cost is CPU-bound: manifest level-walk + per-level binary search + block decrypt(XOR) + zstd decode + cross-level merge/shadow/tombstone — NOT I/O (bytes were already in page cache).
- Raw NVMe random read (O_DIRECT, cold): 4K p50 96.6 us / mean 102.9 us ; 16K p50 123.9 us / mean 125.1 us. Page-cache hit: 4K 1.6 us / 16K 2.6 us. => cold (post-boot, bytes not yet cached) adds ~100-125 us per uncached block on top of the ~242 us decode.

So per adjacency read: ~1 us today -> ~242 us warm (~340-370 us cold). ~240x slower per lookup. A read returns the full neighbor list WITH bond payloads (weights/confidence), so it is 1 read per expanded node per direction; per-neighbor degree is served O(1) from the resident map. => reads/query = ~2 x expanded_nodes (outgoing + incoming).

### 2.2 Projected added latency per query (warm; SESSION hit ~7% at 1% cache; effective per-read ~225 us)

| Query class            | expanded | reads (~2x) | TODAY (~1us) | NEW simple cache | NEW + v1 parallel prefetch (~10x on large fan-out, 12 cores) |
|------------------------|----------|-------------|--------------|------------------|---------------------------------------------|
| typical (SESSION p50)  | 27       | ~54         | ~54 us       | ~12 ms           | ~12 ms (already small; little to parallelise) |
| SESSION p90            | 1,324    | ~2,650      | ~2.6 ms      | ~0.6 s           | ~60 ms                                       |
| SESSION mean           | 1,935    | ~3,870      | ~3.9 ms      | ~0.87 s          | ~90 ms                                       |
| HUB p50                | 3,224    | ~6,450      | ~6.5 ms      | ~1.45 s          | ~150 ms                                      |
| pathological max       | 113,965  | ~228k       | ~0.23 s      | ~50 s            | ~5 s                                         |

Reading of the trade: typical queries get ~12 ms slower (fine). Large/hub traversals are seconds slower WITHOUT the router and only acceptable WITH it. **"X% slower" framing: for a fixed-RAM brain that scales to any size, expect typical recall +~10 ms, and large/hub recall 1-2 orders of magnitude slower unless prefetched.** The router (parallel prefetch + decoded-block cache) is what converts the seconds back to tens-of-ms. The cost being CPU-bound (not I/O) is the good news: it parallelises across 12 cores and is attackable by caching DECODED blocks.

### 2.3 RAM curve (the win)

- Today (rehydrate_adjacency, brain.rs:10347): resident outgoing+incoming HashMap<Seed, Vec<NodeId>> — O(bonds). Crosses 56 GiB at ~82M bonds; ~76 GiB at ~111M.
- New: resident = MAP (degree array, O(nodes), 4-8 B/node) + bounded hot cache (fixed cap) + reader handles/index (tens of MB). At 15.6M nodes the degree map is ~62-125 MB; at 30M nodes (full Wiki) ~120-240 MB. **Essentially flat in bond count.** Total resident-for-adjacency ~0.2 GiB (map) + chosen cache cap (recommend ~0.3-1 GiB) + ~6 GiB of OS page cache for the encoded bond store (reclaimable, shared, not process RSS).
- Net: adjacency RAM goes from tens of GiB growing with bonds, to a few hundred MB + a fixed cache cap, independent of brain size. This is the architecture's entire purpose and it is achieved.

---

## 3. Part B — Code map (file:line)

### 3.1 Substrate consumers that read adjacency directly (above the Brain trait)

These read network.outgoing / network.incoming (or neighbors_* / network.bonds()) directly and must be routed through the cache-backed reader:
- Spread activation (the hot path): network.rs:2431 spread_activation_weighted_inner.
  - outgoing expand: network.rs:2516 neighbors_outgoing_for_read_mode(&cur) (==neighbors_outgoing_capped in V3Only).
  - incoming expand: network.rs:2589 neighbors_incoming_capped(&cur).
  - degree for deg_norm: network.rs:2546 / 2617 self.outgoing.get(&nbr).len() + self.incoming.get(&nbr).len()  -> becomes resident-MAP O(1) degree lookup.
  - curiosity hub classification degree: network.rs:2665 self.outgoing.get(&cur)... + self.incoming...
  - second weighted variant (typed_recall_weighted): network.rs:2860 / 2923 / 2989 (same shape).
- Resident adjacency primitives: network.rs:3044 neighbors_outgoing / 3053 neighbors_incoming ; 6925 in_degree / 6930 out_degree ; 6956 neighbors_outgoing_capped / 6981 neighbors_incoming_capped ; 7055 neighbors_outgoing_for_read_mode. Backing fields: network.rs:334 outgoing: HashMap<Seed,Vec<NodeId>>, network.rs:337 incoming: HashMap<Seed,Vec<NodeId>>.
- Curiosity: curiosity.rs:196-204 degree() / is_hub() via neighbors_outgoing/incoming; curiosity expansion rides the spread path (spread_activation_weighted_with_curiosity).
- Hub-fill / consolidation: consolidation.rs:202 consolidate, 766 internal_fill_at_seed, 893 internal_fill_at_hub; direct neighbour/bond walks at consolidation.rs:431-432, 478, 494, 653 (network.bonds().iter()), 779-819, 909-946.
- MCP neighborhood read: mcp_cat3b.rs:305 brain.network.neighbors_outgoing(&seed).
- (Serde/io counts at brain_io.rs / main.rs are save/verify paths, not traversal — out of scope.)

### 3.2 The on-demand disk read path that already exists (Phase 3.7)

- Brain::get_outgoing_bonds_v5 (brain.rs:11250) and Brain::get_incoming_bonds_v5 (brain.rs:11291) -> bond_store::read_path (src/bond_store/read_path.rs).
- BondSstReaderCache (read_path.rs:175): RwLock<HashMap<(BondSstKind,u64), Arc<BondSstReader>>> + optional MasterKey. get_or_open_primary / get_or_open_reverse open-and-cache READER HANDLES (FD + SST index) lazily; it does NOT cache decoded bonds or decompressed blocks. evict/evict_reverse exist but are not wired to compaction-retire yet.
- get_outgoing_bonds_v5 walks manifest.bond_v5_levels L0->LN, binary-searches each level for the subject's SST, reads+decrypts+zstd-decodes the block, merges with cross-level shadowing (merge_hit, read_path.rs) and drops tombstones (BOND_TOMBSTONE_CONFIDENCE) at the end. This per-call work is the measured ~242 us.
- KNOWN RISK (read_path.rs module docs): get_incoming_bonds_v5 uses .bsst.reverse indexes for L1+, but reverse-LESS SSTs (all of L0, plus migration-promoted) fall back to a FULL PRIMARY SCAN — O(L0 bonds) per incoming lookup. Spread reads incoming on EVERY expansion, so this is a latency landmine for the new design (see Open Question 3). Not separately benchmarked this session.
- Consistency caveat (read_path.rs module docs): get_*_bonds_v5 is SST-only; un-flushed MemTable bonds are invisible until the next flush. Fine while training is stopped; matters if writes resume.

### 3.3 What builds the full resident adjacency today

- Brain::rehydrate_adjacency (brain.rs:10347), called from load_v5 (brain.rs:11555). Iterates ALL NodeSSTs (node stubs) then ALL BondSSTs, triple-level merge with shadowing/tombstones, collapses to (subject,object) pairs, and inserts into network.outgoing / network.incoming. Measured this session: 253 s, builds 38.5M bonds + 15.6M stubs; this is the ~46 GiB resident cost. The new design REPLACES this with a thin map build (degree + existence + optional locator), leaving bonds on disk.

### 3.4 What must change

1. Replace rehydrate_adjacency's full outgoing/incoming build with a MAP build (resident degree[] + existence; optional per-subject level/offset hint).
2. Introduce a CacheBackedAdjacency that serves neighbors_outgoing/incoming via map -> hot cache -> get_(out|in)going_bonds_v5, returning decoded Vec<NodeId> (+ payload for weights, so the spread loop no longer needs a separate get_bond per neighbour).
3. Route the 3.1 consumers through it: spread reads neighbour LISTS from the cache reader, and degree from the resident map (deg_norm stays O(1)). Hub-fill/consolidation and curiosity likewise.
4. Provide a cheap INCOMING path (reverse indexes on all levels) or the incoming reads dominate (Open Q3).

---

## 4. Part C — Architecture design

### 4.1 Resident MAP (node -> where its edges live)

- Keyed by NodeId (dense u64, node_intern.rs:85 NodeId(pub u64)); store as Vec indexed by id (no hash overhead):
  - out_degree: u32, in_degree: u32 (serves deg_norm + hub-pin decisions O(1)). (8 B/node.)
  - optional locator hint per direction: (level_bitset:u8, primary_sst_uid:u32, block_off:u32) to skip the manifest binary-search on hot subjects. (~12 B/node if used.)
- Size: degree-only ~62-125 MB at 15.6M nodes; with locator ~250-360 MB. Matches the ~150-200 MB target band. O(nodes), flat in bonds.
- Built at load by a stripped rehydrate (iterate BondSSTs counting per-subject degree, or read degree from NodeSST stubs if available) — far cheaper than building the full adjacency Vecs.

### 4.2 HOT-ADJACENCY CACHE (bounded)

- Cache DECODED neighbour lists keyed by (NodeId, direction) -> Arc<[NodeId+payload]>. A hit costs ~1 us (avoids the 242 us decode); this is the cache's real value (skip decode, not skip I/O).
- Eviction: LRU (or CLOCK/2Q for scan-resistance — see below) + PINNED set of high-degree hubs (top 0.1% by degree from the resident map), pinned at warm-up and never evicted.
- Sizing (data-driven, Section 1.5): locality is low, so size SMALL — recommend cache cap ~0.5-1% of nodes (~78-156k decoded lists, a few hundred MB) plus the pinned hubs. A bigger cache is not worth the RAM (20% ceiling at 5-10%).
- Scan-resistance: hub/large traversals touch 100k+ unique nodes once each — a plain LRU would be churned by these one-shot scans and evict the useful pinned/working set. Use 2Q/CLOCK-Pro or simply mark prefetched/one-shot reads as low-priority so a single hub query cannot evict the pinned hubs. (Recommended; LRU alone is fragile here.)
- STRONGLY RECOMMENDED ADDITION — DECODED-BLOCK CACHE: because the 242 us is per-call decode and an SST block holds MANY subjects, cache DECOMPRESSED SST BLOCKS (not just per-subject lists). One block decode then serves every subject in it. Given low per-subject reuse but spatial locality within blocks (subjects sorted by id), a block cache likely yields a much higher EFFECTIVE decode-hit rate than per-subject caching for the same RAM. This sits alongside BondSstReaderCache (which only caches FD+index today).

### 4.3 On-demand read path

- neighbour lookup: resident map (existence + degree + optional locator) -> hot cache (decoded list) -> decoded-block cache -> get_(out|in)going_bonds_v5 (decrypt+zstd+merge) -> insert into caches. Weights/confidence come WITH the decoded list (no separate get_bond per neighbour).
- degree (deg_norm) NEVER hits the cache/disk — served from the resident map.
- Wire BondSstReaderCache.evict to compaction-retire while here (currently a known no-op leak of FDs).

### 4.4 PREDICTIVE ROUTER (the speed mechanism)

Because misses are compulsory, prefetch is the only lever. Used for PREFETCH/ROUTING ONLY — never the store of record; exact bonds always come from disk.

v1 — cheap heuristic prefetch (nearly free; ships the win):
- Frontier look-ahead: when expanding cur, we already hold cur's decoded neighbour list with weights + each neighbour's degree (resident map). Compute the pre-threshold predicted activation (cur_act * weight * 0.7 * deg_norm) for each neighbour and ASYNC-prefetch the outgoing+incoming lists of those predicted to clear ACTIVATION_THRESHOLD (i.e. the next frontier), into the cache, BEFORE the BFS pops them. Signal: bond weight + degree — both already in hand.
- Parallel batch: issue the predicted next-frontier reads across a small worker pool (12 cores). Since the 242 us is CPU-bound decode, N-way parallelism cuts wall-clock ~N x on large fan-outs — the main reason hub queries drop from seconds to tens/hundreds of ms.
- Hub pin/prefetch: pinned top-degree hubs are pre-warmed at boot so they are never a miss.
- Depth pipelining: prefetch depth d+1 while computing depth d (BFS is FIFO, so the next frontier is known one level ahead).

v2 — learned / compressed predictor (upside):
- Signal: co-activation history — the SEQUENCE/SET of nodes activated per query (exactly the access logs this harness produces). Log (seed-set -> activated-set) over real sessions.
- Model: a COMPACT, lossy predictor (e.g. per-node co-activation top-k table, or a small learned embedding / product-quantised "region" code per node, or an LLM-style next-region token model) that, given the query seeds, predicts the likely WORKING SET / next region and BULK-prefetches it at query start. This is the "compressed router": a lossy map of where traversal tends to go, used only to prefetch.
- Invariant: the predictor can be wrong with zero correctness cost — a mispredict is just a wasted prefetch (or a cache miss served from disk). Exactness stays in the SSTs. This is the safe place for ML in the system.
- Why it fits the data: reuse is compulsory-dominated and at large distances; a reactive cache cannot catch it, but a predictor that front-loads the whole likely working set converts those compulsory misses into prefetch hits. This is the highest-leverage speed work IF v1 is not enough.

### 4.5 Projected resident-RAM curve

Resident(adjacency) = map(~8 B/node) + cache_cap(fixed) + reader_handles(~tens MB), independent of bonds. ~0.2 GiB map + ~0.3-1 GiB cache at any size. Encoded bond store (~2.1 GiB now, ~6 GiB at full Wiki) lives in reclaimable OS page cache, not RSS. Contrast: today's resident adjacency is ~tens of GiB and grows with bonds (76 GiB at 111M). The brain now scales to any size at fixed RAM.

---

## 5. Sub-phased implementation plan + LOC (padded ~2x per recent calibration)

- **[DONE 2026-06-12]** P0 Resident MAP + thin load: degree[]/existence (+optional locator) struct; replace rehydrate_adjacency full build with map build; keep both behind a flag for parity. ~500-700 LOC. (Shipped: `src/map_ram.rs` + `Brain::build_resident_map`/`enable_resident_map`, `LU_MAPRAM_BUILD` flag in `load_v5`; locator hint deferred as optional. See §9.)
- **[DONE 2026-06-12]** P1 CacheBackedAdjacency: decoded hot cache (2Q + hub pin); neighbors_outgoing/incoming via map->cache->v5 reader; returns lists+payloads. (Shipped: `src/cache_adjacency.rs` + `Brain::enable_cache_backed_adjacency`/`v5_adjacency_source`/`cache_neighbors_*`, `LU_MAPRAM_CACHE` flag in `load_v5`; the decoded-BLOCK cache deferred as a distinct increment needing a `bond_store::read_path` block-decode seam. See §10.)
- **[REROUTE+PARITY DONE 2026-06-12; rehydrate-removal + default-flip DEFERRED to next gated step]** P2 Route consumers: spread_activation_weighted_inner + typed_recall variant + consolidation hub-fill + curiosity + mcp_cat3b through a SELECTABLE adjacency seam (legacy resident adjacency vs P1 cache-backed map+cache+reader), flag-gated `LU_MAPRAM_ROUTE`, with BYTE-EQUIVALENT activation parity proven across both paths. `rehydrate_adjacency` kept intact + default unchanged so the cache path can be A/B-validated at scale before the irreversible removal. (See §11.)
- P3 Router v1: frontier look-ahead prefetch + parallel worker pool + hub pre-warm + depth pipelining + scan-resistant eviction priority. ~600-900 LOC.
- P4 Telemetry + tuning: hit rate, miss/decode latency, prefetch hit/waste, cache-cap + pin-count knobs (runtime_config); incoming-reverse coverage check. ~350-550 LOC.
- P5 (later / upside) Router v2 learned predictor: co-activation logging, compact model build/serve, bulk prefetch, mispredict accounting. ~1200-2200 LOC.

Core P0-P4 ~3,150-4,650 LOC padded. (Excludes P5.)

---

## 6. Open questions (need John before implementation)

1. Cache sizing: accept SMALL cache (~0.5-1% nodes) + page cache + prefetch (recommended, data-driven), or spend RAM chasing the ~20% ceiling? Confirm we are not optimising for hit rate.
2. Page-cache dependency: the warm-path win assumes the encoded bond store stays in OS page cache (2.1 -> ~6 GiB). Acceptable to depend on the OS, or do we want an explicit in-process decoded-block cache as the primary mechanism (Section 4.2)? (I recommend the explicit decoded-block cache, since the cost is decode not I/O.)
3. INCOMING bonds (highest risk): get_incoming_bonds_v5 full-scans reverse-less L0 SSTs (read_path.rs docs). Spread reads incoming every expansion. Do we (a) build .bsst.reverse on ALL levels incl. L0, (b) keep incoming adjacency resident (it is the smaller half but still O(bonds)), or (c) change spread to avoid incoming reads? This decides whether the design is viable for spread at all. Needs its own benchmark.
4. Decode cost itself (~242 us): is it worth optimising the v5 read path (fewer levels via more aggressive compaction; cheaper merge; larger/columnar blocks) so every miss is cheaper, independent of caching?
5. Per-read 242 us measured WARM only. Do we need a cold/post-boot SLO (adds ~100 us/block), or is a boot-time hub/region pre-warm acceptable?
6. Router v2: build the learned predictor, or is v1 heuristic + parallel prefetch + pinning sufficient? Gate on measured v1 effectiveness.
7. MemTable consistency: SST-only reads are fine with training stopped; if writes resume, do we wire the MemTable half into the cache reader or keep flush-before-read?

---

## 7. Appendix — reproducibility

- Harness: /opt/lu/op_brain_measure/ (external to repo). main.rs = locality sim; bench.rs = latency; nvme_lat.py = device latency. Built against lu4d via path dep, CARGO_TARGET_DIR shared with repo target.
- Snapshot: /var/lib/lu/brain-v5-20260611-enc-pretrain ; key /var/lib/lu/master.key. Read-only; brain loaded ~267 s (rehydrate 253 s).
- All numbers in Sections 1-2 are from this session.
---

## 8. Update (2026-06-12) — v5 bond-compaction scheduler landed (prerequisite for this design)

**Bug found while grounding this doc in the code:** at HEAD `750eb9c`, `Brain::load_v5` started compaction via `enable_compaction`, which spawns only the **v4** `CompactionWorker`. That worker's `decide_compaction` reads `manifest.levels` (the v4 bond store), which is **empty** for a v5 brain — so the Phase 3.6 `BondCompactionWorker` (the only code that emits the on-disk reverse index `<uid>.bsst.reverse`) **never ran in production**. Consequence during v5 training: L0 BondSSTs piled up unbounded, reverse indexes were never built, and incoming-edge reads (`get_incoming_bonds_v5`) degraded toward the full-scan cliff flagged in Open Question #3 above.

**Fix:** new production scheduler `BondV5CompactionScheduler` (`src/bond_store/scheduler.rs`), a background worker mirroring the v4 `CompactionWorker` structure (start / poll-loop / `compact_once` / stop+Drop) but driving the **real** Phase 3.6 compactor over `bond_v5_levels`. When live L0 count reaches the trigger (default 4, same as v4), it runs the canonical minor compaction (all L0 → L1) — emitting the paired `.bsst.reverse`, committing the manifest atomically (`save_atomic`), swapping the live snapshot, and running `cleanup_orphan_bond_ssts` — exactly the sequence the `examples/derisk_drive_compaction.rs` harness validated by hand, now promoted to a looping background worker. `load_v5` now calls `enable_bond_v5_compaction` **in place of** the dead v4 path, so a v5 brain runs the v5 scheduler and not the v4 worker (no double-run). Compaction is LSM maintenance and runs regardless of `--no-save`, matching the existing convention.

**Test:** `bond_v5_compaction_scheduler_tests::bond_v5_scheduler_auto_compacts_and_builds_reverse_index` boots via `load_v5`, piles L0 above the trigger, and asserts the background scheduler (no manual call) drains L0 → L1, the `.bsst.reverse` exists on disk, and `get_incoming_bonds_v5` returns the correct incoming edges (now served by the L1 reverse index). Full suite green.

**Relevance to this design:** this directly de-risks Open Question #3 — with the scheduler running, L0 stays small and L1 carries a real reverse index, so incoming reads are reverse-backed rather than full-scan. This is the **P0 prerequisite** for the map-in-RAM architecture; the resident-map work can now assume a compacting v5 bond store with built reverse indexes. (Building `.bsst.reverse` on L0 too, vs. keeping incoming adjacency resident, remains the open follow-up.)

**Note for the node side:** `NodeCompactionWorker` (Phase 2.7) has the **same gap** — it is a one-shot compactor with no production scheduler, and `load_v5` does not schedule it either. Node L0 NodeSSTs therefore also pile up. Left as a separate follow-up (not fixed here); the bond fix mirrors a pattern the node side can reuse.

**Update (2026-06-13):** node side now scheduled too -- `NodeV5CompactionScheduler` (`src/node_store/scheduler.rs`) mirrors the bond scheduler and is wired into `load_v5` via `Brain::enable_node_v5_compaction`, draining node L0->L1 over `node_levels`. Both node and bond compaction now run automatically; clear for MapRAM P0. Test: `node_v5_scheduler_auto_compacts_l0`.

---

## 9. Update (2026-06-12) — P0 landed (resident MAP + thin load)

**Built:** `src/map_ram.rs` — `ResidentMap`, a resident index keyed densely by `NodeId.0`: two parallel `Vec<u32>` (`out_degree`, `in_degree`), O(1) degree/existence lookups, 8 B/node, flat in bond count (~125 MiB at 15.6M nodes; ~240 MiB at 30M). It is an INDEX/LOCATOR only — no edges, no payloads; the exact graph stays in the v5 bond SSTs. The optional per-direction on-disk locator hint (§4.1) was deliberately NOT built — it is an explicit optimisation tied to the deferred 242µs-decode work and the doc marks it optional; P0 ships the minimal correct foundation.

**Build path:** `Brain::build_resident_map()` is the "stripped rehydrate" — it replays the EXACT BondSST triple-level merge / `(subject,object)` collapse / tombstone-drop / predicate-validity / seed-existence filtering that `rehydrate_adjacency` applies before a bond lands in adjacency, but COUNTS per-subject out-degree and per-object in-degree instead of inserting bonds. Result: degrees byte-identical to `AssociativeNetwork::out_degree`/`in_degree`. `Brain::enable_resident_map()` retains the map on the new `Brain::resident_map` field. `load_v5` builds it behind the `LU_MAPRAM_BUILD` flag (default OFF) so production load is byte-for-byte unchanged and `rehydrate_adjacency` stays the source of truth — the two coexist for parity, exactly as the P0 plan specified.

**Not done in P0 (by design):** `rehydrate_adjacency` (the 46 GiB) is NOT removed and consumers are NOT rerouted — the RAM win lands when consumers move onto the map (P2+). O(1) degree from the map is in place for the spread path's `deg_norm` to consume later.

**Tests (6 new, all green):** `map_ram::tests` (4 unit — O(1) degree/existence, out-of-range→0, bond-total sums, empty map) + `brain::mapram_p0_tests` (2 integration over a real v5 brain booted via `load_v5`: a universal sweep asserting map degree == rehydrated adjacency degree for EVERY interned node, with a duplicate `(s0,hub)` pair spanning two L0 SSTs to exercise cross-SST collapse; and the `LU_MAPRAM_BUILD` flag on/off wiring).

**Gate:** `cargo build --workspace` clean; full `cargo test --workspace` green — **lib 1984 passed / 0 failed** (baseline 1978 + 6), **workspace 2002 passed / 0 failed**. (Note: `node_v5_scheduler_auto_compacts_l0` — the 2026-06-13 node-compaction scheduler test — is intermittently flaky: fails ~3/5 in isolation on the CLEAN pre-P0 baseline due to a compact-vs-cleanup file race; unrelated to and unaffected by P0. A fully-green full-suite run was confirmed.)

**Status:** P0 DONE. **Next — P1:** `CacheBackedAdjacency`: decoded hot cache (2Q/CLOCK + hub pin) + decoded-block cache; `neighbors_outgoing`/`incoming` served via map → cache → v5 reader, returning lists + payloads (§5 P1, ~800-1100 LOC).


---

## 10. Update (2026-06-12) — P1 landed (cache-backed adjacency read path)

**Scope as built (P1 = ONE clean increment).** P1 builds and PROVES the
cache-backed adjacency READ PATH — neighbour lookups going map → cache →
disk (the v5 readers) instead of the full resident adjacency — behind the
`LU_MAPRAM_CACHE` flag. It does NOT reroute the production consumers
(`spread_activation`, hub-fill, curiosity, mcp_cat3b) off `network.bonds`
and does NOT remove `rehydrate_adjacency`: that resident adjacency stays
the source of truth for parity. The RAM win lands in P2.

**Built:** `src/cache_adjacency.rs` — `CacheBackedAdjacency`,
`V5AdjacencySource`, `AdjacencySource` (trait), `DecodedNeighbor`,
`Direction`. `neighbors_outgoing(node)` / `neighbors_incoming(node)`
resolve via:
  1. resident MAP — existence + degree, O(1), NO I/O. An absent node or a
     degree-0 direction returns empty WITHOUT touching cache or disk
     (pure sinks/sources never decode); correct because P0 proved map
     degree is byte-exact to the resident-adjacency degree.
  2. bounded HOT cache — decoded per-`(node,direction)` neighbour list
     keyed `(NodeId, Direction)` -> `Arc<[DecodedNeighbor]>` (the other
     endpoint + winning predicate + full `BondPayload`: "lists + payloads"
     so a future spread loop reads weights WITH the list). A hit is ~1µs
     and skips the ~242µs decode — the cache's real value is skip-decode,
     not skip-I/O.
  3. on miss — the v5 reader (`get_(out|in)going_bonds_v5`), then insert.

**Cache structure + eviction + hub-pin.** Scan-resistant **2Q** (Johnson
& Shasha, full variant): `A1in` (FIFO of first references) → `A1out`
(ghost keys only) → `Am` (hot LRU). A one-shot hub scan flows THROUGH
`A1in` to the ghost queue without entering `Am`, so it cannot churn the
working set — the scan-resistance the doc (§4.2) requires over a plain
LRU. Plus a PINNED set of the top ~0.1% hubs by degree (selected from
the P0 map via a bounded min-heap), held in an immune map that never
evicts and is never counted against the 2Q cap; `prewarm_pins` warms
them at boot. Cap is configurable: `LU_MAPRAM_CACHE_CAP` (default 65,536
decoded lists — the data-driven "start small" sizing; rely on OS page
cache, prefetch is the P3 router) and `LU_MAPRAM_PIN_FRACTION` (default
0.001). Internally `Mutex<Inner>` for `&self` interior mutability; the
v5 decode runs with the lock RELEASED.

**Deferred within P1 (no over-building, mirrors P0's locator-hint
call).** The decoded-BLOCK cache (§4.2 "strongly recommended addition")
is NOT built: the v5 read path streams blocks INTERNALLY
(`iter_subject`/`iter_all`) and `BondSstReaderCache` caches only
FD+index, so a real block cache needs a new block-decode seam threaded
through `bond_store::read_path` plus an out-of-reader subject→block
index — its own correctness/eviction surface and clearly a SEPARATE
clean increment. The per-list cache already skips the full per-call
decode (the stated P1 value) and proves the read path; the block cache
is a follow-up.

**Correctness-parity proof (the gate's heart).** `V5AdjacencySource`
replays the EXACT `rehydrate_adjacency` filtering: collapse the v5
reader rows to one winner per `(subject,object)` pair by highest seq
(reader already merged levels + dropped tombstones), THEN drop the pair
if the winning predicate discriminant is invalid
(`bond_type_from_discriminant` -> None) or either endpoint is
un-interned. So the served set is set-identical to
`AssociativeNetwork::neighbors_outgoing`/`incoming`, with the same
winning predicate per pair. A `debug_assert` in the hot path also checks
decoded-count == map degree on every miss. Integration tests over a real
`load_v5` brain assert: a universal sweep (every interned node, both
directions) cache-set == adjacency-set + predicate parity + degree
agreement (map == adjacency == cache.len); the multi-predicate
`(s0,a)`->IsA collapse winner; cold-miss→reader→warm-hit (no
re-decode); eviction-then-refetch still exact; hub-pin survives eviction
pressure served without re-decode; and incoming reads served via the
on-disk `.bsst.reverse` index after a forced minor compaction (still
exact). 6 unit tests (2Q eviction, scan-resistance via ghost-promotion,
hub selection, degree/absent short-circuit) round it out.

**Tests (11 new, all green):** `cache_adjacency::tests` (6 unit) +
`brain::mapram_p1_tests` (5 integration over real v5 brains).

**Gate:** `cargo build --workspace` clean (no warnings); full
`cargo test --workspace` green — **lib 1995 passed / 0 failed** (baseline
1984 + 11), **workspace 2013 passed / 0 failed** (baseline 2002 + 11),
6 ignored.

**Status:** P1 DONE. **Next — P2:** route the §3.1 consumers
(`spread_activation_weighted_inner` + the `typed_recall` variant to read
lists from the cache reader and degree from the map; consolidation
hub-fill; curiosity; mcp_cat3b) off `network.outgoing`/`incoming` and
onto this cache-backed path + the resident map, with byte-equivalent
activation parity tests on a fixed seed set — the invasive phase where
`rehydrate_adjacency` (the ~46 GiB) can finally be removed and the
fixed-RAM win lands (§5 P2, ~900-1400 LOC).


---

## 11. Update (2026-06-12) -- P2 landed (reroute + byte-equivalent parity; rehydrate-removal DEFERRED)

**Scope as built (the REVERSIBLE core of P2).** The doc's P2 (§5) bundles
the consumer reroute, the byte-equivalent parity proof, AND the
`rehydrate_adjacency` removal + RAM win into one phase. Per the safety
split this task landed ONLY the reversible core -- reroute + parity --
and **defers the `rehydrate_adjacency` removal and the default-flip to a
follow-up gated step**. BOTH adjacency paths stay live so the cache path
can be A/B'd and validated at scale before the irreversible deletion.

**The seam.** A single selectable adjacency access point on
`AssociativeNetwork` (`mapram_mode` atomic + installed cache / manifest /
bond-reader handles, mirroring the `read_path_mode` pattern). Every
rerouted consumer calls the seam instead of touching
`outgoing`/`incoming` directly:
  * `seam_spread_outgoing` / `seam_spread_incoming` -- the spread hot
    path's neighbour lists (with the hub-cap);
  * `seam_neighbors_outgoing` -- the raw uncapped walk view
    (consolidation hub-fill, mcp_cat3b);
  * `seam_out_degree` / `seam_in_degree` / `seam_total_degree` --
    `deg_norm` + hub classification, served O(1) from the resident MAP
    in cache mode.
In LEGACY mode (`mapram_mode == 0`, the default) each method is the
prior resident-adjacency behaviour; in CACHE mode neighbour LISTS come
from the P1 `CacheBackedAdjacency` (map -> 2Q cache -> v5 reader) and
DEGREE from the resident map. `Brain::enable_adjacency_seam` installs it;
`Brain::set_adjacency_cache_backed` flips the mode; `load_v5` wires both
behind `LU_MAPRAM_ROUTE` (implies `LU_MAPRAM_CACHE`).

**Rerouted consumers (§3.1):** `spread_activation_weighted_inner`
(network.rs), `typed_recall_weighted_inner` (network.rs), curiosity
`is_hub`/`degree` (curiosity.rs), consolidation `consolidate` /
`internal_fill_at_seed` / `internal_fill_at_hub` (consolidation.rs), and
`mcp_cat3b` neighbourhood read + degrees. `view_bond`/`view_node` are
LEFT on the resident structures (rehydrate stays), so per-neighbour
weights are identical across modes by construction -- the seam changes
only which neighbour ids are enumerated and where degree is read. That
is exactly why the reversible reroute is byte-equivalent and the
`view_bond` migration belongs with the (deferred) removal.

**Byte-equivalent parity proof + a REAL determinism gap found.** New
integration tests over real `load_v5` brains run the SAME spread /
typed_recall queries through BOTH paths (flag off vs on) across seeds,
hub seeds, multi-hop depths, bond filters, the hub-cap, and cache
eviction pressure, asserting the FULL ranked activation result -- node
ids + activation BIT-PATTERNS + depth + ordering -- is IDENTICAL, plus a
curiosity-signal parity test. The first run DIVERGED, and the cause was
NOT a cache bug but a pre-existing engine determinism gap the cache's
different iteration order surfaced:
  1. **Convergence-node propagation.** The spread propagates each node's
     FIRST-REACH activation (the value captured when it was pushed to
     the frontier), not its stored max. So a convergence node (reached
     by two routes) propagates a value that depends on neighbour
     iteration ORDER.
  2. **Hub-cap tie-break.** When the read-side cap keeps top-K by weight
     and weights TIE, which neighbours survive depends on input order.
Both were already non-deterministic in the LEGACY path, because
`rehydrate_adjacency` builds the resident adjacency from a
`HashMap::drain` (random order under RandomState) -- i.e. a convergence
node's recall activation already varied PER LOAD before this work. Per
the brief's bar ("made order-independent; report whichever") the fix
makes iteration order DETERMINISTIC: the seam canonicalises every
neighbour list by seed-bytes ascending in BOTH modes, and the hub-cap
sorts over that canonical input so its tie-break is deterministic too.
This is the same defence the final-result sort already uses (the
BR.C-Hyp seed-byte tiebreak) and it closes the per-load non-determinism
for convergence nodes as a side benefit. With canonical order both paths
are bit-identical; all parity tests pass.

**Gate:** `cargo build` clean (no warnings); full `cargo test
--workspace` green -- **lib 2000 passed / 0 failed** (baseline 1995 + 5
P2), **workspace 2018 passed / 0 failed** (baseline 2013 + 5). The 5 new
tests: `brain::mapram_p2_tests` (spread parity, typed_recall parity,
hub-cap parity, cache-eviction parity, curiosity-signal parity). No
existing test depended on the old random adjacency order -- the
canonical-order change left the entire suite green.

**Status:** P2 reroute + parity DONE. **Next -- the gated, irreversible
step:** remove `rehydrate_adjacency` (the ~46 GiB resident adjacency),
move `view_bond`/`view_node` weight reads onto the cache payloads the
seam already carries, and FLIP the default to the cache-backed path --
at which point the fixed-RAM win (§4.5) lands. Gate it behind the same
byte-equivalent parity at scale this phase established.
---

## 12. Update (2026-06-13) -- P3 landed (recall self-sufficient off resident adjacency; weights+payload+degree_p90 rerouted; rehydrate-removal STILL gated)

**Scope as built.** P3 made the CORE recall hot path
(`spread_activation_weighted_inner` + `typed_recall_weighted_inner` and
their typed / bridge / curiosity wrappers) read NOTHING from the
resident adjacency, while keeping BOTH paths live, `rehydrate_adjacency`
intact, and legacy as the default (no removal, no default-flip -- that
is still the next gated step). Cache-mode only; legacy unchanged.

**What P3 rerouted (cache mode):**
- **Edge WEIGHTS from the carried payload.** The P1 `DecodedNeighbor`
  already carries the bond `BondPayload`; the seam now threads it
  through new edge-carrying accessors (`seam_spread_outgoing_edges` /
  `seam_spread_incoming_edges`, `SeamEdge`/`SeamBondSrc`). The hot loop
  reconstructs the heap `TemporalBond` from the carried payload
  *byte-identically* to what `rehydrate_adjacency` would have inserted
  (empty-keyframe baseline fallback, `bond_type_from_discriminant`,
  `DEFAULT_HALF_LIFE_MS`) and builds a `BondView::from_heap` over it --
  so `weight_in_scope` / `bond_type` / decay all read the carried edge,
  never `view_bond`. The seam hub-cap (`mapram_cap_edges_by_weight`)
  caps by the carried weight, not a per-edge `get_bond`.
- **NODE confidence/content via map-stub (decision A).** `view_node`
  reads in the hot loop (seed seeding, neighbour conf/content, curiosity
  hub test, final activated collection) now go through `seam_recall_node`
  -> `SeamNode`. In cache mode it reproduces the rehydration STUB
  (`Confidence::Unverified` = 0.6, empty content) for any node present
  in the resident MAP -- byte-identical to what `view_node` returns on a
  loaded v5 brain (every node is a stub), WITHOUT reading the resident
  node structures. Real on-disk payloads from `get_node_v5` are
  deliberately NOT used (that diverges from legacy -- it is the B
  thread, below).
- **`degree_p90` from the map.** `install_mapram_seam` now sets
  `degree_p90 = cache.degree_p90()` (the floor(0.9*n) order statistic of
  the map's per-node total degree, floored at 1) -- byte-identical to
  rehydrate's `recompute_degree_p90` over the same brain, so `hub_t` no
  longer depends on rehydrate.

**Byte-equivalence + independence proof.** The P2 byte-equivalence gate
(`brain::mapram_p2_tests`, 5 tests: spread / typed / hub-cap /
cache-eviction / curiosity-signal parity) STILL passes unchanged -- the
cache path is byte-identical to legacy with weights/payloads/degree now
from carried-payload + map instead of resident structures. New
`brain::mapram_p3_tests` (2 tests): (a) `degree_p90_map_sourced_matches_recompute`
proves the map-sourced threshold equals `recompute_degree_p90`; (b)
`cache_recall_independent_of_resident_structures` is the STRONG
independence proof -- it runs cache-mode recall, then POISONS every
resident structure the legacy path reads (clears `outgoing`/`incoming`,
zeroes every resident bond weight, sets every resident node to
Contested/"POISON"), and asserts cache-mode recall is byte-identical
before/after; a sanity assert confirms legacy mode DOES diverge (the
poison bit). No determinism/order gap surfaced (the P2 canonicalisation
already closed that).

**Non-rerouted resident-reader audit (the 5 from internal notes.**
None are on the core spread/typed recall hot loop:
- `update_typed_ratio_for` -- INGEST (called from `add_bond`). Reads
  resident adjacency via `neighbors_outgoing` + `get_bond`. Ratio is
  order-insensitive -> seam-rerouteable, but not recall. **Deferred**
  (must-handle-before-removal if a loaded brain ingests with rehydrate
  gone).
- `recompute_all_typed_ratios` -- LOAD-time (`from_snapshot`). Only runs
  when resident adjacency is present. **Deferred** (only-runs-with-
  adjacency-present).
- `disambiguate_polysemous_with_anchors`, `polysemy_label_prop_clusters`,
  `score_sense_clusters` -- the SENSE/polysemy subsystem in the
  narration + `mcp_sense` layer (reached by `ask()`'s narration, so
  recall-ADJACENT). They read resident adjacency via **raw,
  uncanonicalised** `neighbors_outgoing` + `get_bond`. Rerouting them to
  the existing seam would change their legacy neighbour ORDER (the seam
  sorts by seed bytes; raw adjacency does not), so it is NOT a
  byte-equivalent swap. **Deferred to the removal phase with its own
  order-equivalence proof.** None block the core recall path, so none
  were rerouted in P3 (per "fix only those that block recall").

**B thread (recorded for later, internal notes.** On a loaded v5
brain, recall is STUB-BASED for node confidence (flat
`Unverified`=0.6) and content (empty): `view_node` returns the
rehydration stub and SHADOWS the real on-disk node payload (proven:
a node flushed as Verified/0.9/"hello" reads back via `view_node` as
0.6/"" while `get_node_v5` returns 0.9/"hello"). P3 preserved this
byte-for-byte (A). **B = feed real per-node confidence/content via
`get_node_v5` into recall and evaluate ranking** -- a deliberate
semantic change requiring a re-baselined parity bar; tracked separately.

**Gate.** `cargo build` clean; full `cargo test --workspace` =
**2020 passed / 0 failed** across 29 suites (baseline 2018 + 2 new P3
independence tests; lib 2000 -> 2002). LOC: +462 / -49 across
`src/network.rs`, `src/brain.rs`, `src/cache_adjacency.rs`.

**Post-P3 removal-readiness verdict.** rehydrate removal is now CLEAN
*for the core spread/typed recall path* -- it reads neighbours/degree
from the map+cache+reader, weights from the carried payload, and node
confidence/content from the map-stub, proven byte-identical to legacy
and independent of the resident adjacency. It is NOT yet GLOBALLY clean:
before `rehydrate_adjacency` can be deleted, the sense/polysemy readers
(item 1 above) and the ingest/load typed-ratio readers (item 2) must be
rerouted with their own equivalence proofs, otherwise removal would
silently change narration/sense output and ingest-time typed ratios.
The RAM win is UNBLOCKED for the recall hot path; the irreversible
removal + default-flip stays gated on those remaining readers.


## 13. Update (2026-06-13) -- P4 landed (sense/polysemy + typed-ratio readers rerouted off resident adjacency; byte-equivalent; rehydrate-removal STILL gated)

**Scope as built.** P4 reroutes the FIVE peripheral resident-adjacency
readers the P3 audit (§12) deferred, so each reads neighbours / bond
type from the cache seam (map -> 2Q cache -> v5 reader) in cache mode
instead of `neighbors_outgoing` + `get_bond`. Cache-mode only; the legacy
branch of every reader is byte-for-byte the prior code. `rehydrate_adjacency`
stays intact, both paths live, legacy the default -- no removal, no
default-flip.

**The seam grew two raw, uncapped accessors** (`src/network.rs`):
- `seam_outgoing_set(seed) -> Vec<Seed>` -- outgoing neighbour SEEDS
  (legacy = `neighbors_outgoing`; cache = `mapram_raw_neighbors`). RAW /
  UNCANONICALISED on purpose: every consumer folds it into a `HashSet`
  or a per-edge COUNT, so order is irrelevant and the legacy branch is
  byte-identical.
- `seam_outgoing_typed(seed) -> Vec<(Seed, Option<BondType>)>` -- the
  same neighbours paired with their bond TYPE. In CACHE mode the type is
  the carried predicate (`bond_type_from_discriminant`) off the P1
  `DecodedNeighbor`, so it reads NEITHER the resident adjacency NOR the
  resident `bonds`; in LEGACY mode it is exactly `neighbors_outgoing` +
  per-neighbour `get_bond().bond_type()` (same order, same Some/None).

**The 5 readers + per-reader order/equivalence approach.** The decisive
fact: `rehydrate_adjacency` builds `outgoing`/`incoming` by draining
`pair_winners` (a `HashMap`), so the resident neighbour ORDER is
`HashMap::drain`-random per load. Any reader whose OUTPUT depended on
neighbour order would therefore already be per-load nondeterministic in
legacy -- byte-equivalence is only well-defined for ORDER-INDEPENDENT
outputs. All 5 were rerouted via option **(a) proven order-independent**:
- `update_typed_ratio_for` (ingest) and `recompute_all_typed_ratios`
  (load): output is a pure typed/total COUNT over the outgoing edge set
  -> order-independent. Now enumerate via `seam_outgoing_typed`; the
  whole-graph variant keeps iterating the node OVERLAY
  (`self.nodes.keys()`, unaffected by removal) and only its inner
  neighbour/type enumeration moved to the seam.
- `polysemy_label_prop_clusters` (shared by `disambiguate_polysemous*`
  and `score_sense_clusters`): the cluster PARTITION is the
  connected-component decomposition of the (order-independent)
  co-neighbourhood Jaccard graph; cluster memberships, typed scores, and
  the elected winner are order-independent functions of that partition.
  Only the integer cluster LABEL (= a neighbour index) and the
  score-tie vector ORDER are positional -- and those are ALREADY per-load
  nondeterministic under the random resident order, so they are not a
  meaningful output and are deliberately NOT asserted (and not changed:
  legacy untouched). Neighbours + types now come from
  `seam_outgoing_typed`, neighbour SETS from `seam_outgoing_set`.
- `disambiguate_polysemous_with_anchors` anchor path: a per-neighbour
  Jaccard filter into a `HashSet` -- order-independent by construction.

**Byte-equivalence + independence proof** (`brain::mapram_p4_tests`, 3
tests, all on real `load_v5` brains):
- `typed_ratio_byte_equivalent_both_paths` -- `update_typed_ratio_for`
  and `recompute_all_typed_ratios` produce bit-identical f32 ratios in
  legacy vs cache mode (and the concrete 2/3 value checks correctness).
- `sense_readers_byte_equivalent_both_paths` -- on a polysemous two-sense
  hub, `disambiguate_polysemous`, `disambiguate_polysemous_multi_sense`,
  `disambiguate_polysemous_with_anchors`, and `score_sense_clusters` are
  byte-identical legacy vs cache on their MEANINGFUL projection (sorted
  member seed-bytes + score bit-patterns + typed counts + winner flag);
  non-vacuous asserts confirm the readers actually fire (>=2 clusters,
  Some disambiguation). The cache's different (v5-reader) order yields
  identical output -- the order-independence argument, confirmed.
- `sense_and_typed_independent_of_resident` -- the STRONG independence
  proof: in cache mode, compute the sense + typed-ratio outputs, then
  POISON every resident structure the legacy path reads
  (`mapram_test_poison_resident` clears `outgoing`/`incoming`, zeroes
  bond weights, Contests/"POISON"s nodes) AND flip every resident bond
  TYPE (new `mapram_test_corrupt_resident_bond_types`, so a leaked
  `get_bond` type read would change the typed/related split), and assert
  the cache-mode outputs are byte-unchanged; a sanity assert confirms
  the legacy path DOES diverge. This proves the rerouted readers source
  neighbours from the cache and types from the carried predicate, never
  the resident adjacency/bonds.

**Gate.** `cargo build --workspace --all-targets` clean (only pre-existing
example warnings); full `cargo test --workspace` = **2023 passed / 0
failed / 6 ignored** across 29 suites (baseline 2020 + 3 new P4 tests;
lib 2002 -> 2005). One pre-existing flake surfaced on the first run
(`mapram_p1_tests::cache_incoming_via_reverse_index_is_exact` -- a
background bond-compaction / orphan-cleanup file race in `load_v5`,
code P4 does not touch); it passed 3/3 in isolation and on the green
re-run. LOC: see `git show --stat` for the commit.

**Post-P4 removal-readiness verdict -- NOT YET GLOBALLY CLEAN.** The five
P4-scoped readers are now clean (rerouted + byte-equivalent +
resident-independent), as are the P3 core recall hot path, curiosity,
mcp_cat3b, and `internal_fill_at_seed`. HOWEVER, a fresh grep of every
`neighbors_outgoing` / `neighbors_incoming` / `get_bond` caller shows
ADDITIONAL direct raw-adjacency consumers that are NOT routed through
the seam and would read empty adjacency once `rehydrate_adjacency` is
removed and the default flips to cache:
- `consolidation::internal_fill_at_hub` (consolidation.rs:~908) -- its
  degree checks were migrated to `seam_total_degree`, but the hub
  neighbour-LIST gather still calls raw `neighbors_outgoing`. PARTIALLY
  migrated; this is exactly the hub-fill consumer the Phase 3.8.1 audit
  (brain.rs:~10330) names as a raw-adjacency reader.
- `network::typed_recall` (network.rs:~3616/3705) -- the non-`_inner`
  typed-recall variant still walks raw `neighbors_outgoing`/`incoming`.
- `bridge::is_hub` (bridge.rs:~182) and the bridge candidate gather
  (bridge.rs:~256/257) -- raw degree + neighbour reads.
- `idk_gate` neighbour-coverage (idk_gate.rs:~401/404).
- `narration` hub degree gate (narration.rs:~669).
- `reasoning` chain/analogy walks (reasoning.rs:~660, 773, 2198, 2516,
  2573, 2630, 2955) -- `neighbors_outgoing`/`incoming` + `get_bond`.
- `variants` degree/walk (variants.rs:~201/210/215).

These are the SAME consumer class `rehydrate_adjacency` was created
(Phase 3.8.1) to feed. The task's premise of "5 remaining readers per the
P3 audit" undercounts the live code: the 5 sense/typed-ratio readers are
done, but the consumers above must each be rerouted to the seam (or
proven unreachable on a loaded-v5 cache-mode brain) WITH their own
equivalence proofs before `rehydrate_adjacency` can be deleted. Until
then the irreversible removal + default-flip stays gated. RECOMMEND:
reconcile the internal notes "remaining readers" list against a fresh
full grep before scheduling the removal.

**P5-batch1 status (2026-06-13, landed -- see §14).** Of the
post-P4 remaining-reader list above, the ORDER-INDEPENDENT subset
(degree gates + set-membership) is now rerouted through the seam and
proven byte-equivalent: `bridge::is_hub` (total-degree) +
`bridge::jaccard_neighbors` (out+in SET); `idk_gate` anchor coverage
(out+in SET); `narration` hub DEGREE gate (~669, out-degree -- the
answer-gen WALKS at ~847/865/1564/1618 stay batch 2);
`variants::degree` (~201 -- `neighbors_ordered` ~210/215 stays batch 2);
`research::chain` `endpoint_gap_intensity`/`endpoint_is_hub` (in/out
degree); and `network::collect_seeds_with_filter` majority-scan degree.
Still batch 2 (order-SENSITIVE bond-carrying walks, need the new
uncapped-canonical edge accessor): `consolidation::internal_fill_at_hub`
list gather, `network::typed_recall` non-`_inner`,
`reasoning` chain/analogy walks, narration answer-gen walks, the agents
triangle walk, bare `spread_activation`/`timetravel`, `neighbors_capped`.
`context::generate_candidate_scopes` is DEFERRED to batch 2: its OUTPUT
is order-independent (sorted+deduped keyframe timestamps), but it reads
bond KEYFRAMES (not degree/set), so byte-equivalence needs the uncapped
edge-carrying accessor + a keyframe/timestamp poison dimension -- i.e.
bond-carrying work, not trivially byte-equivalent through the canonical
seam.

## 14. Update (2026-06-13) -- P5 batch 1 landed (order-independent adjacency consumers rerouted off resident adjacency; byte-equivalent; rehydrate-removal STILL gated)

**Scope as built.** P5 batch 1 reroutes the FIRST iterative pass of the
post-P4 audit (§13): the consumers whose output is ORDER-INDEPENDENT
(degree gates / set membership), each trivially byte-equivalent through
the canonical seam. Cache-mode only; the legacy branch of every consumer
is byte-for-byte the prior code. `rehydrate_adjacency` stays intact, both
paths live, legacy the default -- no removal, no default-flip.

**The seam grew one accessor** (`src/network.rs`): `seam_incoming_set`
-- the in-direction twin of the P4 `seam_outgoing_set`. RAW + UNCAPPED +
UNCANONICALIZED (consumers set-fold it; order irrelevant). Legacy =
`neighbors_incoming`; cache = `mapram_raw_neighbors(Incoming)` (the P1
cache reverse-index set, proven exact by
`mapram_p1_tests::cache_incoming_via_reverse_index_is_exact`). All other
batch-1 reads reuse the existing P2/P3/P4 accessors: `seam_out_degree`,
`seam_in_degree`, `seam_total_degree`, `seam_outgoing_set`.

**Consumers rerouted (6).**
- `bridge::is_hub` (bridge.rs) -- total-degree gate -> `seam_total_degree`.
- `bridge::jaccard_neighbors` (bridge.rs) -- out+in neighbour HashSet ->
  `seam_outgoing_set` + `seam_incoming_set`.
- `idk_gate` anchor-coverage (idk_gate.rs ~401/404) -- out+in HashSet ->
  `seam_outgoing_set` + `seam_incoming_set`.
- `narration` hub degree gate (narration.rs ~669) -- out-degree ->
  `seam_out_degree`.
- `variants::degree` (variants.rs ~201) -- total-degree ->
  `seam_total_degree`.
- `research::chain::endpoint_gap_intensity` + `endpoint_is_hub`
  (research/chain.rs ~183/206) -- in/out-degree -> `seam_in_degree` /
  `seam_out_degree`.
- `network::collect_seeds_with_filter` (network.rs, majority-scan degree
  sort key) -- `out_degree+in_degree` -> `seam_total_degree`.

**Why byte-equivalent.** In legacy mode each seam read is the prior call
BYTE-FOR-BYTE: `neighbors_outgoing(s).len() == out_degree(s)` (both read
`self.outgoing`, 1:1 intern map), likewise incoming; `seam_*_set ==
neighbors_*`. Every consumer is a PURE, DETERMINISTIC function of those
reads (a count, or a HashSet fold) -- ORDER-INDEPENDENT -- so legacy
output is unchanged. In cache mode the degree comes from the resident
MAP (byte-exact to adjacency degree by P0) and the neighbour sets from
the P1 cache (exact as SETS by P1/P4), so cache output equals legacy.

**Byte-equivalence + independence proof** (`brain::mapram_p5_tests`, 3
tests, all on real `load_v5` brains):
- `p5_seam_degree_and_sets_byte_equivalent` -- the seam reads ALL six
  consumers are functions of (out/in/total degree; sorted out/in sets)
  are byte-identical legacy vs cache; this is the property that
  propagates byte-equivalence to the consumers, including the private
  ones (`jaccard_neighbors`, `variants::degree`, the narration gate,
  `collect_seeds_with_filter`) not directly callable from a test.
- `p5_public_consumers_byte_equivalent` -- the reachable public consumers
  (`bridge::is_hub`, `research::chain::endpoint_gap_intensity`,
  `endpoint_is_hub`) are byte-identical legacy vs cache and discriminate
  hub from leaf (non-vacuous).
- `p5_batch1_reads_independent_of_resident` -- the STRONG independence
  proof: in cache mode the seam reads + consumers are byte-invariant
  after POISONING every resident structure the legacy path reads
  (`mapram_test_poison_resident` clears adjacency, zeroes bond
  weights+keyframes, nukes nodes; + `mapram_test_corrupt_resident_bond_types`
  flips every bond TYPE); a sanity assert confirms the legacy reads DO
  diverge.

**Gate.** `cargo build --workspace --all-targets` clean (only the
pre-existing `migration_verify_v5.rs` + `probe_anchor_coverage` example
warnings); full `cargo test --workspace` = **2027 passed / 0 failed / 6
ignored across 29 suites** (lib 2009 / 0 / 4); the 3 new `mapram_p5_tests`
included.

**Remaining batch 2 (order-SENSITIVE bond-carrying walks; need the new
uncapped-canonical edge accessor).** `consolidation::internal_fill_at_hub`
neighbour-list gather; `network::typed_recall` non-`_inner`;
`reasoning` chain/analogy walks (reasoning.rs ~660/773/2198/2516/2573/
2630/2955); narration answer-gen walks (narration.rs ~847/865/1564/1618);
the agents triangle walk; bare `spread_activation`/`timetravel`;
`neighbors_capped`; `variants::neighbors_ordered` (~210/215); and
`context::generate_candidate_scopes` (output order-independent but
keyframe-CARRYING, so it needs the edge accessor + a keyframe/timestamp
poison dimension). Until these are rerouted with their own equivalence
proofs, the irreversible `rehydrate_adjacency` removal + default-flip
stays gated.

## 15. Update (2026-06-13) -- P5 batch 2 (sub-batch A) landed (uncapped canonical edge-carrying seam + first bond-payload-carrying walks rerouted off resident adjacency; byte-equivalent; rehydrate-removal STILL gated)

**Scope as built.** Batch 2 (the order-SENSITIVE bond-carrying walks) is
LARGE; this commit lands a coherent, fully-proven SUB-BATCH -- the new
edge-carrying seam INFRASTRUCTURE plus the consumers provable end-to-end
through a public entry point. Cache-mode only; the legacy branch of every
consumer is byte-for-byte the prior code apart from the deliberate
canonical neighbour ORDER (legacy resident order was `HashMap::drain`-
random per load -- already non-deterministic -- so canonicalising BOTH
modes makes the output deterministic AND identical; same defence P2
applied to the spread path). `rehydrate_adjacency` stays intact, both
paths live, legacy the default -- no removal, no default-flip.

**The seam grew the UNCAPPED, CANONICAL, edge-carrying accessor** (`src/network.rs`):
- `seam_outgoing_edges` / `seam_incoming_edges` -- the uncapped twin of
  the P3 `seam_spread_*_edges`, BOTH directions, returning `Vec<SeamEdge>`
  in canonical seed-byte order in BOTH modes. CACHE yields `Carried`
  edges holding the predicate + full `BondPayload` (weight + type +
  KEYFRAMES); LEGACY yields `Resident` edges. No hub-cap (the walks
  impose their own `take(cap)` / `take(scan_limit)`).
- `SeamWalkBond<'a>` ( `Resident(BondView)` | `Carried(TemporalBond)` )
  with `.view() -> BondView`, plus `seam_walk_outgoing` /
  `seam_walk_incoming` (canonical `(nbr, Option<bond>)` walk pairs) and
  `seam_directed_bond` (single directed `a->b` lookup, for the cross-edge
  single-pair reads). In cache mode the bond -- and its keyframes -- are
  reconstructed byte-identically from the carried payload
  (`reconstruct_carried_bond`, the same P3 routine `rehydrate_adjacency`
  matches), never resident `get_bond`/`view_bond`.

**Consumers rerouted (2 functions, both proven through a PUBLIC entry).**
- `context::ContextSelector::generate_candidate_scopes` (`context.rs`) --
  reads bond KEYFRAME timestamps on the seed's outgoing + incoming edges
  (the batch-1-DEFERRED keyframe-carrying reader, §14). Now walks
  `seam_walk_outgoing` / `seam_walk_incoming`; in cache mode keyframes
  come from the carried payload. Output (sorted+deduped Historical
  scopes) is order-independent.
- `consolidation::internal_fill_at_hub` (`consolidation.rs`) -- the hub
  neighbour-list gather (`take(max_neighbors)` -- the partially-migrated
  consumer §13 named), the A->B / A->C legs (last-keyframe weight+source),
  and the B->C idempotency leg (`seam_directed_bond`). Its degree gates
  were already on `seam_total_degree` (P5-b1); only the neighbour/bond
  reads moved. Node content (`is_hub_stopword`) stays resident -- out of
  scope (the separate node "B" thread), and inert under the poison test
  (non-stopword labels).

**Byte-equivalence + independence proof** (`brain::mapram_p5b_tests`, 3
tests, all on real `load_v5` brains):
- `p5b_candidate_scopes_byte_equivalent` -- `generate_candidate_scopes`
  is byte-identical legacy vs cache (full scope-list Debug projection),
  and the four DISTINCT inserted keyframe timestamps (outgoing 100/300,
  incoming 150/400) appear as Historical scopes (keyframe-carrying,
  non-vacuous).
- `p5b_internal_fill_at_hub_byte_equivalent` -- internal_fill proposals
  are byte-identical legacy vs cache (sorted a / c / new_weight-bits /
  path / create_if_missing projection) and fire (3 create-mode
  proposals).
- `p5b_batch2_independent_of_resident` -- the STRONG independence proof:
  in cache mode BOTH outputs are byte-invariant after POISONING every
  resident structure the legacy path reads -- adjacency cleared, weights+
  keyframes zeroed (`mapram_test_poison_resident`), every bond TYPE
  flipped (`mapram_test_corrupt_resident_bond_types`), and every keyframe
  timestamp/weight/source overwritten with a distinctive non-zero
  sentinel (NEW `mapram_test_corrupt_resident_keyframes` -- the
  keyframe/timestamp poison dimension batch 2 required); a sanity assert
  confirms the legacy reads DO diverge.

**Gate.** `cargo build --workspace --all-targets` clean (only the
pre-existing `migration_verify_v5.rs` + `probe_anchor_coverage.rs` example
warnings); full `cargo test --workspace` = **2030 passed / 0 failed / 6
ignored across 29 suites** (baseline 2027 + 3 new `mapram_p5b_tests`; lib
2009 -> 2012). No flake this run.

**Remaining batch 2 (NOT in this commit).** The seam infra every
remaining walk needs is now in place, so each is a mechanical reroute to
`seam_walk_*` / `seam_directed_bond` + its OWN equivalence proof (they
all share the `neighbors_*` + per-edge `get_bond` + `take(cap)` shape):
- `network::typed_recall` non-`_inner` (network.rs ~3616/3705).
- `reasoning` chain/analogy walks (reasoning.rs ~660/773/2198/2516/2573/
  2630/2955) -- single-anchor outgoing/incoming walks.
- `narration` answer-gen walks (narration.rs ~865/872, 1024, 1564, 1618)
  + single-pair lookups (504/507, 1336, 1504, 3078 -> `seam_directed_bond`);
  847/848 is a degree count (-> `seam_*_degree`).
- `agents` triangle walk (~346/377, 351/396).
- bare `network::spread_activation` (~3247, reachable from Brain timetravel).
- `network::neighbors_outgoing_capped` / `neighbors_incoming_capped`
  (~7563/7588) -- need the carried-weight cap (the `mapram_cap_edges_by_weight`
  twin over the uncapped edge accessor).
- `variants::neighbors_ordered` (~207/210/215) + `bond_weight_directed`
  (~225) -- private fns; need an arena-level public proof
  (`generate_variants_for_goal` / `resolve_anchor`).

**FINAL removal-readiness verdict -- STILL GATED.** `rehydrate_adjacency`
removal + default-flip remains BLOCKED. The seam now has the uncapped
edge-carrying accessor every remaining batch-2 walk needs, and the
context keyframe reader + the hub-fill consumer are clean (rerouted +
byte-equivalent + resident-independent incl. keyframes). But the
remaining batch-2 walks above STILL read raw `neighbors_*` + resident
`get_bond` and would read empty adjacency / wrong bonds once rehydrate is
removed and the default flips. Removal stays gated until every one of
them is rerouted with its own equivalence proof. NO blocker found: every
consumer rerouted in this sub-batch was made byte-equivalent in both
modes with no genuine order/semantic obstruction.

## 16. Update (2026-06-13) -- P5 batch 2 (sub-batch B.1) landed (bare recall walks + carried-weight CAP TWIN rerouted off resident adjacency; byte-equivalent; rehydrate-removal STILL gated)

**Scope as built.** Batch 2 is the order-SENSITIVE bond-carrying walks;
sub-batch A (§15) landed the uncapped edge-carrying seam + the
context/keyframe + hub-fill consumers. This sub-batch (B.1) lands a
coherent, fully-proven slice of the remaining walks: the seam's missing
CAPPED accessor (the carried-weight hub-cap twin) plus the two bare
public recall walks that drive it / the uncapped walk. Cache-mode only;
the legacy branch of every consumer is byte-for-byte the prior reads
apart from the deliberate canonical neighbour ORDER (legacy resident
order was `HashMap::drain`-random per load -- already non-deterministic --
so canonicalising BOTH modes makes the output deterministic AND
identical; same defence P2/sub-batch-A applied). `rehydrate_adjacency`
stays intact, both paths live, legacy the default -- no removal, no
default-flip.

**The seam grew the CAPPED edge-carrying accessor** (`src/network.rs`):
- `seam_outgoing_edges_capped` / `seam_incoming_edges_capped` -- the
  carried-weight hub-cap twin of `neighbors_outgoing_capped` /
  `neighbors_incoming_capped`, built OVER the uncapped, canonical
  `seam_outgoing_edges` / `seam_incoming_edges` (whose legacy source is
  `neighbors_outgoing` / `neighbors_incoming` -- exactly matching
  `neighbors_*_capped`'s source, unlike `seam_spread_*_edges` which reads
  `neighbors_outgoing_read_mode_uncapped`). When hub-read enforcement is
  off or degree is within `hub_read_max_{out,in}`, returns the full
  canonical edge list; otherwise keeps top-K by latest raw bond WEIGHT
  via the existing `mapram_cap_edges_by_weight` (CACHE weight = carried
  payload; LEGACY weight = `get_bond`) -- the SAME stable sort + drop
  counter (`hub_cap_filtered_at_query_total`) as `neighbors_*_capped`.
  The enforce/pass threshold uses `raw.len()` (= out/in-degree in both
  modes), byte-identical to the `out_degree`-based gate.

**Consumers rerouted (3 of the 7 batch-2 groups; 2 functions + the cap
twin).**
- `network::spread_activation` (bare, non-`_inner`; reachable from Brain
  timetravel) -- the CAPPED spread walk. Now reads neighbours + edge
  weight via `seam_outgoing_edges_capped` / `seam_incoming_edges_capped`,
  the bond (incl. keyframes) reconstructed from the carried payload
  (`reconstruct_carried_bond`), neighbour DEGREE via `seam_total_degree`
  (was `self.outgoing/incoming.get(..).len()`), and the node stub via
  `seam_recall_node` (seed init + per-neighbour + result assembly; was
  `view_node` + `self.nodes.get`). No resident `neighbors_*_capped` /
  `view_bond` / `self.outgoing` / `self.nodes` read in cache mode.
- `network::typed_recall` (non-`_inner` public variant) -- the UNCAPPED
  typed walk. Now reads neighbours via `seam_outgoing_edges` /
  `seam_incoming_edges`, the bond from the carried payload (the bond
  TYPE that drives `bond_filter` is the carried predicate, never
  resident `get_bond`), degree via `seam_total_degree`, node stub via
  `seam_recall_node`.

**Byte-equivalence + independence proof** (`brain::mapram_p5c_tests`, 4
tests, all on real `load_v5` brains; the `ranked` projection =
seed bytes + activation f32 BIT pattern + depth -- the adjacency/bond-
derived output; node CONTENT is the separate resident "B" thread, out of
scope and excluded from the projection, exactly as the P3 recall
independence test does):
- `p5c_spread_activation_byte_equivalent` -- bare spread is byte-identical
  legacy vs cache across seeds x depths, non-vacuous (propagates past the
  seed).
- `p5c_typed_recall_byte_equivalent` -- typed recall is byte-identical
  legacy vs cache; the bond-TYPE filter discriminates (RelatedTo-only
  differs from unfiltered, since `q->ia` is IsA) -- proving the carried
  predicate drives the filter.
- `p5c_spread_cap_twin_byte_equivalent` -- with enforcement on + cap=2 on
  a 6-leaf hub, the carried-weight cap is byte-identical legacy vs cache
  AND the drop counter advances in BOTH paths (the cap twin is actually
  exercised).
- `p5c_recall_independent_of_resident` -- the STRONG independence proof:
  in cache mode spread + typed (filtered + unfiltered) are byte-invariant
  after POISONING every resident structure the legacy path reads --
  adjacency cleared, bond weights+keyframes zeroed
  (`mapram_test_poison_resident`), every bond TYPE flipped
  (`mapram_test_corrupt_resident_bond_types`), every keyframe overwritten
  with a sentinel (`mapram_test_corrupt_resident_keyframes`); a sanity
  assert confirms the legacy reads DO diverge.

**Gate.** `cargo build --workspace --all-targets` clean (only the
pre-existing `migration_verify_v5.rs` + `probe_anchor_coverage.rs` example
warnings); full `cargo test --workspace` = **2034 passed / 0 failed / 6
ignored across 29 suites** (baseline 2030 + 4 new `mapram_p5c_tests`; lib
2012 -> 2016). No flake this run.

**Remaining batch 2 (NOT in this commit -- the 4 walk groups still to do).**
Each is a mechanical reroute to `seam_walk_outgoing` / `seam_walk_incoming`
(uncapped `(nbr, Option<bond>)` pairs, `.view()` -> `BondView`) +
`seam_directed_bond` for single-pair lookups + its OWN equivalence proof.
They emit node CONTENT into their output, so their independence proof
must project to the adjacency/bond-derived fields (or poison adjacency +
bonds only), the same way `ranked` does here:
- `reasoning` chain/analogy walks (reasoning.rs ~660/773/2198/2516/2573/
  2630/2955) -- single-anchor outgoing/incoming `neighbors_* + get_bond`
  with `take(cap)`/`take(scan_limit)` (canonical order makes the take
  deterministic).
- `narration` answer-gen walks (narration.rs ~865/872, 1024, 1564/1567,
  1618) + single-pair lookups (504/507, 1336, 1504, 3078 ->
  `seam_directed_bond`); 847/848 is a degree count (-> `seam_*_degree`).
- `agents` triangle walk (agents.rs ~346/377 outgoing, 351/396 get_bond).
- `variants::neighbors_ordered` (~207/210/215) + `bond_weight_directed`
  (~226) -- private fns; need an arena-level public proof
  (`generate_variants_for_goal` / `resolve_anchor`).

**FINAL removal-readiness verdict -- STILL GATED.** `rehydrate_adjacency`
removal + default-flip remains BLOCKED. Now CLEAN (rerouted +
byte-equivalent + resident-independent): the P3 core recall hot path,
curiosity, mcp_cat3b, `internal_fill_at_seed`, the P4 sense/typed-ratio
readers, the P5-b1 degree/set consumers, the sub-batch-A context-keyframe
+ `internal_fill_at_hub` consumers, and -- as of this commit -- the bare
`spread_activation` + `typed_recall` recall walks and the carried-weight
CAP TWIN. STILL serving-path BLOCKERS (read raw `neighbors_*` + resident
`get_bond`, would read empty adjacency / wrong bonds once rehydrate is
removed and the default flips): the 4 walk groups above (reasoning,
narration, agents, variants). Removal stays gated until every one of them
is rerouted with its own equivalence proof. Maintenance/ingest-only
resident readers (Hebbian/learning `get_bond_mut` writes in
`thinking.rs`/`agents.rs`, consolidation proposal writers, ingest) are
NOT serving-path and are safe -- they run on the resident ingest brain,
not a loaded cache-mode serving brain. NO blocker found WITHIN this
sub-batch: every consumer rerouted here was made byte-equivalent in both
modes with no genuine order/semantic obstruction.

## 17. Update (2026-06-13) -- P5 batch 2 (sub-batch B.2) landed (the FINAL 4 serving-path walk groups rerouted off resident adjacency; byte-equivalent; rehydrate-removal now FULLY CLEAN)

**Scope as built.** This sub-batch lands the FINAL 4 order-sensitive
bond-carrying walk groups that §15/§16 left as the only remaining
serving-path blockers -- reasoning, narration, agents, variants -- each a
mechanical reroute to the existing edge-carrying seam
(`seam_walk_outgoing`/`seam_walk_incoming` for the `(nbr, Option<bond>)`
walks, `seam_directed_bond` for single-pair lookups, `seam_total_degree`
for the degree count, `seam_outgoing_set` for the set-folded neighbour
read, `seam_outgoing_edges`/`seam_incoming_edges` for the canonical
ordered neighbour gather). The whole-`src/` final audit additionally
caught ONE serving-callable diagnostic reader the prior batches missed
(`mcp_cat3b::tool_hub_inspect`); it is rerouted here too. Cache-mode
only; the legacy branch of every consumer is byte-for-byte the prior
reads apart from the deliberate canonical neighbour ORDER (legacy
resident order was `HashMap::drain`-random per load -- canonicalising
BOTH modes makes the output deterministic AND identical; same P2 /
sub-batch-A / B.1 defence). `rehydrate_adjacency` stays intact, both
paths live, legacy the default -- NO removal, NO default-flip in this
commit (that is the next, final step).

**No new seam surface.** Every accessor these walks need already shipped
in sub-batches A (uncapped edge-carrying seam + `seam_walk_*` +
`seam_directed_bond`) and B.1 (capped twin). This commit is pure consumer
reroutes + their proofs.

**Consumers rerouted (the final 4 groups + 1 audit-found diagnostic).**
- `reasoning.rs` -- all 7 chain/analogy/procedure/sibling/comparison
  walks: `best_chain` (~660 extension walk), `depth_one_chains` (~773),
  `assemble_procedure` (~2198), `pick_differentiating_phrase` (~2516),
  `gather_siblings` incoming walk (~2573) + its prop-count walk (~2630),
  `gather_comparison` (~2955). All now `seam_walk_outgoing` /
  `seam_walk_incoming`; the bond (weight/type/keyframes) comes from the
  carried payload via `.view()` in cache mode. Node CONTENT (`get_node`)
  stays resident -- the separate "B" thread, excluded from the proof
  projection.
- `narration.rs` -- `bond_label_between` (504/507) + `bond_priority`
  (1336) + the typed_score (1504), definition_prose_override (1564/1567)
  and source-attribution (3078) single-pair lookups -> `seam_directed_bond`;
  `Walker::collect_outgoing` (1024), `has_typed_grounding` (865/872) and
  `definition_prose_override` walk -> `seam_walk_*`; the degree count
  (847/848) -> `seam_total_degree`; the polysemy out-neighbour set (1618)
  -> `seam_outgoing_set`.
- `agents.rs` -- the SYNTHESIS triangle walk in `run_synthesis`: A->B
  (~346) and B->C (~377) via `seam_walk_outgoing`, the r_ab (351) / r_bc
  (396) bond readings reconstructed from the carried payload.
- `variants.rs` -- `neighbors_ordered` (207/210/215) via the canonical
  `seam_outgoing_edges` + `seam_incoming_edges` (out-then-in, deduped),
  `bond_weight_directed` (226) via `seam_directed_bond`.
- `mcp_cat3b.rs` -- `tool_hub_inspect` (308): the per-neighbour weight
  read moved from resident `get_bond` to `seam_directed_bond` (its degree
  + neighbour list were already on the seam). The last serving-callable
  resident bond read the whole-`src/` audit found.

**Byte-equivalence + independence proof** (8 new tests, one
`mapram_p5d_*` module per consumer file, all on real `load_v5` brains;
projections capture only the ADJACENCY/BOND-derived output -- seeds +
bond_type + weight/score bits -- and EXCLUDE node content, which the
poison helpers overwrite with "POISON"):
- `reasoning::mapram_p5d_reasoning_tests` -- `p5d_reasoning_walks_byte_equivalent`
  (best_chain / depth_one_chains / assemble_procedure / gather_siblings /
  gather_comparison byte-identical legacy vs cache, all non-vacuous) +
  `p5d_reasoning_independent_of_resident` (all invariant under
  poison_resident + corrupt_bond_types + corrupt_keyframes; legacy diverges).
- `narration::mapram_p5d_narration_tests` -- bond_label_between (both
  directions), bond_priority, Walker::collect_outgoing (DISABLED noise
  filter so its content-based rejection cannot perturb the bond
  projection), has_typed_grounding: byte-equivalent + resident-independent.
- `agents::mapram_p5d_agents_tests` -- the synthesis triangle beliefs
  byte-equivalent (fresh brains per mode, since run_synthesis does a
  resident provenance write) + resident-independent.
- `variants::mapram_p5d_variants_tests` -- neighbors_ordered (content-free
  seed list) + bond_weight_directed byte-equivalent + resident-independent.
The narration readers reachable ONLY through `extract_slots` / `narrate`
(typed_score, definition_prose_override, polysemy_blocked, source-attr)
read adjacency/bonds EXCLUSIVELY through the SAME seam accessors proven
byte-equivalent + resident-independent here and in p5c; the full
workspace suite + the final audit confirm no residual resident adjacency
read remains in narration serving code.

**Gate.** `cargo build --lib` clean; full `cargo test --workspace` =
**2042 passed / 0 failed / 6 ignored across 29 suites** (baseline 2034 +
8 new `mapram_p5d_*` tests; lib 2016 -> 2024). No flake.

**FINAL whole-`src/` re-grep audit (every remaining resident
adjacency/bond reader classified).** Patterns swept: `neighbors_outgoing`/
`neighbors_incoming`, `get_bond`, `view_bond`, `self.outgoing`/`incoming`/
`nodes`.
- `view_bond` / `view_node` OUTSIDE `network.rs`: NONE.
- `self.outgoing` / `self.incoming` / `self.nodes` OUTSIDE `network.rs` /
  `cache_adjacency.rs`: NONE.
- SERVING-path readers (recall / reasoning / narration / agents /
  variants ANSWER path): **NONE remain.** All four walk groups +
  tool_hub_inspect are rerouted.
- Remaining resident readers, all NON-serving (safe for removal):
  - `brain::rehydrate_adjacency` -- the load-time builder itself (the
    thing the next step removes; it `add_bond`s, populating BOTH the
    adjacency lists AND `self.bonds`).
  - INGEST / WRITE-side maintenance, all on the resident ingest brain:
    `brain::decide_hub_cap_admission` (hub-cap write enforcement,
    8747/8773/8808/8834 + its get_bond), `brain::teach_with_kind` (4430
    Hebbian reinforce / insert), `agents::write_agent_keyframe` (610
    provenance write), `variants::generate_variants_for_goal` (369
    Hebbian write guard), `session::apply_hebbian` (83/91),
    `pending::apply::apply_change` (84 ingest insert).
  - `thinking::think` (593 hypothesis-write guard) + `propose_hypotheses`
    (754 strength read): the thinking/learning pipeline -- it MUTATES the
    brain (adds nodes+bonds), so it inherently runs on a writable
    resident/ingest brain, not a read-only cache-mode serving brain
    (consistent with the §16 classification).
  - `brain::bond_evidence` (3659) -> `has_contradiction`: public helper
    whose ONLY in-tree callers are tests; NOT wired into the recall/answer
    serving path. Latent resident reader; not a serving blocker.
  - `network.rs` seam LEGACY branches + accessor definitions, and
    `cache_adjacency.rs` cache implementation -- the seam itself, not
    consumers.
  - all `#[cfg(test)]` hits (reasoning 4679/4912, agents 1150/1190/1195/
    1266, variants 724/787/829, mcp.rs 6538/6540, archive.rs, the
    `mapram_*` suites, etc.).

**DEFINITIVE removal-readiness verdict -- FULLY CLEAN.** Every
serving-path consumer of the resident adjacency is now rerouted through
the seam (reads neighbours from the resident MAP and bonds --
weight/type/keyframes -- from the carried cache payload, never resident
`neighbors_*` / `get_bond` / `view_bond` in cache mode), each proven
byte-equivalent in BOTH modes (canonical order both modes) AND invariant
under full resident poisoning. The whole-`src/` audit finds NO remaining
serving-path resident adjacency/bond reader. `rehydrate_adjacency`
removal + default-flip is therefore UNBLOCKED: the only resident
adjacency/bond readers left are `rehydrate_adjacency` itself (to be
removed), ingest/write-side maintenance + the thinking/learning write
pipeline (which run on the resident ingest brain, not a loaded cache-mode
serving brain), one latent test-only public helper, the seam's own legacy
branches, and tests. NO blocker found: every consumer rerouted in this
sub-batch was made byte-equivalent with no genuine order/semantic
obstruction. The next step (separate commit) is the actual
`rehydrate_adjacency` removal + default-flip.

## 18. Update (2026-06-13) -- FINAL STEP landed (default flipped to cache-backed; rehydrate SKIPPED at load; legacy kept behind flag)

**What shipped.** The MapRAM default is now flipped: a v5 brain loads
cache-backed by default, and `load_v5` **no longer calls
`rehydrate_adjacency`** on the default path. This is the RAM win the whole
operation was built toward -- the ~31 GiB resident outgoing/incoming
adjacency (+ node stubs) that the 253 s boot-time rehydrate produced is
simply not built. `rehydrate_adjacency` is **NOT deleted** -- it stays
reachable behind the flag for the parity harness and for rollback.

**Flag model (all default ON for v5; force OFF with `0`/`false`/`off`/`no`).**
Read once at load via the new `Brain::mapram_flag_on(name, default_on)`
helper; the three flags are INDEPENDENT so each is observable on its own
(race-free single-writer tests):
- `LU_MAPRAM_ROUTE` (default ON) -- the master switch. ON => skip
  rehydrate and route consumers (spread / typed_recall / curiosity /
  hub-fill / mcp) through the cache-backed seam. OFF => legacy load
  (rehydrate the full resident adjacency, byte-for-byte as before).
- `LU_MAPRAM_BUILD` (default ON) -- retain the cheap resident MAP
  (`Brain::resident_map`, ~0.2 GiB; NodeId -> out/in degree).
- `LU_MAPRAM_CACHE` (default ON) -- in legacy mode, also build the
  coexisting cache for parity. In routed mode the cache is mandatory and
  always built (the seam needs it).
- `LU_MAPRAM_CACHE_CAP` (default 65 536) / `LU_MAPRAM_PIN_FRACTION`
  (default 0.001) unchanged.

Rollback to the pre-flip behaviour is a pure env flip:
`LU_MAPRAM_ROUTE=0 LU_MAPRAM_CACHE=0 LU_MAPRAM_BUILD=0` restores the
legacy rehydrate path with no resident map/cache built. Git-revertable +
flag-gated, as specified.

**`load_v5` default (cache-backed) path now does:** build resident MAP
(BUILD) -> build cache-backed adjacency (map -> 2Q hot cache -> v5
reader) -> install the seam -> select cache-backed. `degree_p90` is
sourced from the map at seam-install (`install_mapram_seam`), so it no
longer depends on rehydrate's `recompute_degree_p90`. The legacy branch
(`else`) calls `rehydrate_adjacency()` then optionally the map/cache for
parity.

**Downstream wiring confirmed without the resident adjacency:** the v5
bond + node compaction schedulers, the v5 save path (SST/manifest flush
+ overlay snapshot; it never serialised the resident adjacency for a v5
brain), and ingest are independent of the resident adjacency. The
`lu4d_status` tool already computed counts from the memtable + SST entry
counts (Phase 9), not the resident adjacency. The MCP startup banner was
updated to report the honest counts (node intern len; resident-MAP bond
total, falling back to the resident count only in forced-legacy mode).
`migration_verify_v5::verify` (an offline migration check, not the
serving path) rehydrates explicitly since it needs the full in-RAM
adjacency to count bonds + compare recall.

**Test-harness updates.** The legacy-vs-cache parity harness compares the
cache against the resident adjacency, so each `load_routed` helper and
the P0/P1/P3 comparison tests now rehydrate the resident adjacency
EXPLICITLY after `load_v5` (the default load no longer builds it). The
two flag tests were rewritten for the new default semantics
(default => built; forced-off => absent), keeping each module the sole
writer of its env var (race-free). No production parity guarantee
changed: both modes are still proven byte-equivalent.

**Gate.** `cargo build` clean; full `cargo test --workspace` green --
**lib 2024 passed / 0 failed**, **workspace 2042 passed / 0 failed /
6 ignored** (matches the pre-change baseline exactly). All MapRAM P0-P5b
parity suites pass in both modes.

**Status:** FINAL STEP DONE -- default is cache-backed, rehydrate is
skipped at load (RAM win realised), legacy + rollback preserved behind
the flag, `rehydrate_adjacency` retained. At-scale RAM before/after +
serving confirmation on a real-brain copy are recorded in the operation
report.
