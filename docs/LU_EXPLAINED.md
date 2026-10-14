# How Lu works

Lu is an experimental AI written from scratch in Rust. It has no neural network, no transformer and no learned weights. Its "brain" is a typed knowledge graph: discrete knowledge nodes joined by weighted, typed bonds. All of its "thinking" is graph work (spreading activation, typed traversal and template-driven narration), not matrix multiplication.

This document explains how the code in this repository works, what it can and can't do, and what went wrong along the way.

**Status.** Active development mostly stopped in September 2026. The server that ran Lu, and the trained brain data on it, were lost in October 2026. The code survives in full in this repository. You can build it and run the tests (2,388 passing), but you have to ingest your own data to get a working brain.

---

## The idea

An LLM answers by running a prompt through fixed learned weights. Its knowledge is smeared across billions of parameters, so you can't point at one fact, edit it or remove it.

Lu stores knowledge as explicit nodes and explicit edges that you can read, point at, strengthen, weaken and sever. That makes some things possible that are hard for an LLM:

- **Remove one fact without retraining.** Revoke a node or scar a bond, and it stops being reachable.
- **Show its work.** Every answer comes from a chain of bonds that you can print.
- **Say "I don't know."** If nothing in the graph supports an answer, Lu refuses instead of guessing.
- **Run on a CPU.** No GPU is needed.

The cost is just as structural. Lu only knows what has been ingested into its graph. It has no pre-trained prior, so its breadth depends on how much clean knowledge you load.

Lu is named after a cattle dog, Luna. Her photo is the logo.

---

## The building blocks

### Seeds, nodes and bonds

Every node has a 32-byte identity, a `Seed` (`src/seed.rs`). A node holds a piece of knowledge: a definition, a mechanism, a part-of relation, an example, a code snippet, raw text and so on. It also holds a confidence level.

Bonds are typed and weighted relations between two nodes (`src/bond.rs`), such as `IsA`, `HasProperty`, `Causes`, `PartOf`, `UsedFor` and `RelatedTo`. The full enum has 44 variants. It is append-only, so saved brains stay readable when new types are added. Bond weights carry a full time history (keyframes with exponential decay), which lets you query the graph as it was at an earlier time (`src/timetravel.rs`).

### Learning: "the model talks, Lu eats"

Lu learns by reading text, not by gradient descent. Extractors in `src/extractor/` split text into sentences and run typed pattern matchers in priority order: definition, mechanism, composition, purpose, example, contrast, causal. Anything that doesn't match falls back to raw text. Ingestion scripts in `scripts/` load dictionaries, WordNet, Wikidata, arXiv abstracts and more.

An LLM can be used as a *source*: its output gets broken into nodes and bonds. Its weights are never imported.

One lesson learned the hard way: node identity originally hashed the raw text with no case-folding, so "Albedo" and "albedo" became two nodes. The graph ended up over-noded, and that was its single biggest quality problem.

### Storage

Storage went through several generations. The current one (v5) is an SST-style store: a node store, a bond store with a reverse index, a write-ahead log, a manifest and a keyword index. Node IDs are append-only u64 with tombstones, and compaction is streaming (`src/sst.rs`, `src/node_store/`, `src/bond_store/`, `src/wal.rs`, `src/compaction.rs`). Optional encryption at rest is in `src/encrypt.rs`. A fixed-RAM adjacency plan ("MapRAM", `src/map_ram.rs`) keeps a degree map plus a hot cache of the busiest hub nodes. In measurements it cut steady-state RAM from about 26 GiB to about 7 GiB.

---

## How a question is answered

1. **Find seeds.** The keyword index picks candidate nodes that match the query terms.
2. **Spread activation.** Activation flows outward along bonds, with decay at each hop, a ceiling, and normalization so that big hub nodes don't swamp the result (`src/network.rs`). A scarred bond carries zero weight, so activation can't pass through it.
3. **Two-pass recall.** The first pass uses only the query, so understanding it isn't biased by the conversation. The second pass mixes in topic seeds from session memory (`src/session_memory.rs`, `src/lb_session/`).
4. **Honesty gate.** `src/idk_gate.rs` decides whether the recall is good enough to answer (see below).
5. **Reasoning.** A ReAct-style loop (thought, action, observation) and a recall, synthesis and critic agent chain (`src/thinking.rs`, `src/agents.rs`). The critic scores a draft and can force a retry.
6. **Narration.** Templates turn the reached nodes and bonds into prose: definitions, causal "why" answers, side-by-side comparisons and more (`src/narration.rs`, `src/synthesis.rs`). There is no LLM in the output path. The standing rule is "honest over eloquent".
7. **Self-check.** `src/offness.rs` runs after the answer is drafted and checks whether it looks "off" compared with what the graph supports.

---

## Honesty: "I don't know"

The IDK gate (`src/idk_gate.rs`) refuses in several situations:

- **Total IDK:** the query has no real traction (low total activation, no reasoning chain and no keyword hits).
- **Partial match:** some query words match the brain and others don't. Lu won't invent a connection between what it knows and what it doesn't.
- **Single-anchor dominance:** one word has a rich neighborhood and the others contribute nothing. An answer would really be about that one word, so Lu refuses.

In 100-question adversarial runs, Lu made **zero fabrications**, including on invented-name bait questions. Its failures were wrong routing or missing coverage, not made-up facts. These runs were done by the author on the author's own brain and haven't been reproduced independently.

The honest limit: Lu won't invent facts it wasn't taught, but it will faithfully repeat a falsehood it *was* taught. What goes into the brain matters as much as the gate.

## Scars: memory of past mistakes

A scar is a record on a bond that says "don't act on this connection", with a reason and a date (`src/scar.rs`). Scars live in their own log, separate from the bond history:

- **Recall enforces.** Spreading activation uses `effective_weight = historical × (scarred ? 0 : 1)`, so a scarred bond carries no activation.
- **Narration explains.** The observation layers can see both the history and the scar, so Lu can say "I know that connection exists, but I was scarred against acting on it on date X, for reason Y."
- History is kept. A query about the past still shows that the bond used to be active.

```rust
// src/scar.rs (abridged)
pub struct Scar {
    pub scar_id: u64,
    /// The directed bond this scar is on: (from, to).
    pub bond_seeds: (Seed, Seed),
    pub scar_seed: Seed,
    // ... reason, timestamp
}
```

Revoking a node works the same way. The data stays, but every path to it is cut.

## Curiosity

Lu is meant to *want* to fill gaps. During recall, `src/curiosity.rs` watches for four kinds of gap:

```rust
pub enum SignalType {
    DeadEnd,         // hit a hub but went nowhere
    IsolatedNode,    // active but barely connected
    KnowledgeDesert, // spread died before finding anything strong
    HubBoundary,     // touched a hub it couldn't traverse
}
```

Confirmed gaps go into a capped **wonder queue**. A separate curiosity worker (`src/bin/lu_curiosity_worker.rs`) drains the queue and researches the gaps later, off the hot path.

## Self-improvement and memory

- **Hebbian learning.** Bonds that get used are strengthened, and session-tagged keyframes keep one conversation from polluting the baseline. Bonds that are strengthened across several separate sessions get promoted into the baseline (consolidation, `src/consolidation.rs`).
- **Writeback.** Derived conclusions can be written back as *inferred* bonds, but only when they're corroborated. A bond that was just written can't count as its own confirmation (`src/writeback.rs`).
- **Session memory.** Per-conversation beliefs expire and are never promoted to facts automatically.

## Serving and safety boundaries

The serving path is split by privilege. An admin MCP binary has every tool; a guest HTTP binary (`crates/lu_guest_http`) exposes only "ask" and the session tools. The admin tools aren't compiled into the guest binary at all. Code execution goes through a separate broker (`src/bin/lu_codeexec.rs`) that runs in a sandbox.

---

## What works, and what doesn't

**Tested:**
- 2,388 Rust tests pass (`cargo test`).
- In the author's runs: dictionary-grounded definitions, entity overviews, attribute lookups (capital, language, author), comparison tables, causal "why" answers, a calculator, the IDK gate, a bounded code writer in a sandbox, and exact-or-IDK geometry answers.

**Unproven or weak:**
- Breadth never came close to an LLM. An audit of 20,000 nodes found only about 23% were clean concepts; the rest were sentence fragments and code symbols.
- Lu lists facts well but doesn't reliably chain premises into a conclusion.
- The vision "describe" path was a stub.
- Several planned features exist only as design docs in `docs/`.
- All scorecards are the author's own and haven't been reproduced independently.

## What went wrong

- **The breadth gap was structural.** Every node has to be earned, and the clean ones plateau. One large ingestion batch moved zero test questions from fail to pass.
- **RAM scaling under encryption hit a wall.** Resident memory grew with node count. Decrypting bond blocks took a large share of CPU, and production ran out of memory more than once.
- **Silently disconnected subsystems.** The recurring failure was a component that returned a plausible empty value (`0`, `""`, `None`) on failure instead of an error, so everything looked green while a feature was actually off. The rule that came out of it: *a component that returns a plausible empty value on failure is worse than one that errors.*
- **New facts aren't verified as true.** Misinformation could pass corroboration and had to be blocked with a curated denylist.
- **Timelines were optimistic.** Estimates slipped again and again.
- **The server was lost.** It ran on a single server with no off-box backup of the trained brain. When the server was lost in October 2026, the brain data went with it. The code was in Git and survived.

## What carried forward

The parts that held up:

- **Structural revocation:** cut a fact or capability by making it unreachable.
- **The honesty floor:** refuse instead of guessing.
- **Scars.**

These became **Cattledog**, a separate Apache-2.0 project that wraps any LLM with Lu's honesty and scars: https://github.com/dialectforge/cattledog

## License

AGPL-3.0, with a commercial license available from DialectForge LLC (info@dialectforge.com). See `LICENSE` and `COMMERCIAL.md`.
