# AGENTS.md: notes for coding AIs (and the humans directing them)

Lu was built by John Martin directing AI coding agents. Vibe coders are welcome here: if you work by pointing an AI at a repo, this file is for your AI.

## What this is
Lu: an experimental, from-scratch associative-memory AI in Rust. It uses a knowledge graph, not a neural network. Read `docs/LU_EXPLAINED.md` first.

## Layout
- `src/`: the main crate (`lu4d`). Key modules:
  - `network.rs` (spreading activation), `idk_gate.rs` (honest "I don't know"), `scar.rs` (scars), `curiosity.rs` (gaps and the wonder queue), `narration.rs` / `synthesis.rs` (answers), `offness.rs` (post-answer self-check), `mcp*.rs` (MCP tools)
  - storage in `sst*.rs`, `node_store/`, `bond_store/`, `wal.rs`
- `src/bin/`: the binaries (admin MCP, curiosity worker, code-exec broker and others).
- `crates/lu_guest_http`: the guest-only HTTP server. `crates/lu_brain_bench`: benchmarks.
- `tests/`: integration tests. `scripts/`: ingestion and eval scripts (Python and shell).
- `docs/`: design docs. Many describe plans, not shipped features.

## Build and test
```bash
cargo build
cargo test          # expect 2,388 passing, 0 failing
```
There's no trained brain in the repo, so you need to ingest your own data (`scripts/`).

## Rules
- **Never commit secrets**: no keys, tokens, `.env`, `master.key`, brain files (`*.lbr`) or personal data. Check `.gitignore`.
- **Make honest claims.** Don't say something works unless a test or a real run shows it. Watch for "plausible empty value" bugs: return an error, not `0`, `""` or `None`, when something fails.
- Keep `cargo test` green. Add a test with every fix.
- `BondType` is append-only. Never reorder or remove variants.

## License
AGPL-3.0, with a commercial license from DialectForge LLC. Contributions need a CLA (see `CONTRIBUTING.md`).

## Where to start
- Make the build and tests pass on your platform, and report what breaks.
- Write a small, tested quickstart that ingests a tiny dataset and asks a question.
- Node identity case-folding and lemmatization (the "Albedo" vs "albedo" over-noding problem).
- Turn the plans in `docs/` into issues, or delete the stale ones.
