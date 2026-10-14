# Lu (Lu-4.0)

Lu is an experimental AI system written mostly in Rust. It is built around a knowledge graph, not a large neural network. The goals were:

- **Persistent and always on**: Lu keeps running and keeps what it has learned between sessions.
- **Honest**: when Lu doesn't know something, it is meant to say "I don't know" instead of guessing.
- **Scars**: Lu stores memories of past mistakes ("scars") and checks new actions and answers against them.

This repository holds the Lu 4.0 / v4 server code: about 110 Rust modules, a time-aware bond layer ("Lu 4D", see `docs/LU4D_README.md`), ingestion scripts, eval scripts and design docs. The server that ran it and its trained brain data were lost in October 2026. The code is all here.

## What works
- The Rust workspace builds with `cargo build`.
- Unit and integration tests are under `tests/` and `crates/*/tests`.
- Eval scripts (`eval_100.py`, `eval_100_strict.py`) produce 100-question scorecards.

## What is unproven
- The scorecards were run by the author on the author's own data. Nobody else has reproduced them.
- The trained brain (the graph data) is not included. Without it, Lu starts close to empty and you have to ingest your own data (see `scripts/`).
- Many design docs in `docs/` describe plans, not finished features.
- Lu cost its author a lot, personally and financially, and it never became a product. Treat it as research code.

## How to run it
```bash
cargo build --release
cargo test
```
Some scripts and docs assume a Linux server layout under `/opt/lu` and a brain file you build yourself. Adjust paths for your machine.

## License
Copyright 2026 DialectForge LLC. Created by John Dean Martin.

Lu is licensed under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)). A commercial license is also available from DialectForge LLC; see [COMMERCIAL.md](COMMERCIAL.md).

Cattledog, a smaller spinoff that wraps Lu's honesty and scars around any LLM, is released separately under Apache-2.0.
