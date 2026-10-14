# Lu 4D — Temporal Knowledge Graph

A neuro-symbolic AI system built in Rust that extends the Lu v2 associative knowledge graph with a fourth temporal dimension. Every bond carries a full weight history (keyframes + exponential decay), making relationships time-aware.

## Core Concepts

- **Temporal Bonds**: Bond weights are no longer single values — they're append-only sequences of timestamped keyframes with exponential decay between them.
- **Session Isolation**: Hebbian learning during a session writes keyframes tagged with that session ID. Sessions never contaminate the baseline graph.
- **Baseline Consolidation**: Bonds strengthened across multiple distinct sessions get promoted to baseline knowledge.
- **Time Travel**: Query the graph at any historical point and diff the results against the present to see what changed.
- **Spread Activation**: Temporal-aware activation spreading through the associative network, scoped to any time window.

## Building

```bash
cargo build
cargo test
```
