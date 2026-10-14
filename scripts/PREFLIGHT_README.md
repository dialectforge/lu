# Lu BR.0 Preflight Harness

Pre-flight verification gate that runs BEFORE any 1+ hour ingest job
on Lu. Its job is to refuse to launch the ingest if any contract
between Python (extractor / harness) and Rust (binary), or between
the locally-built and deployed binaries, or between the daemon
launch script and the safety flags, is misaligned.

Designed to be cheap (seconds) and HERMETIC -- never touches the
canonical brain at `/var/lib/lu/brain.lbr`. Every MCP probe runs
`lu_4_mcp --no-save` against a `mktemp -d` tmp brain that is
deleted on exit.

## Usage

```
python3 scripts/preflight.py                # all phases (A + B + C)
python3 scripts/preflight.py --phase A      # vocab contracts only
python3 scripts/preflight.py --phase B      # deploy / config only
python3 scripts/preflight.py --phase C      # integration smoke only
python3 scripts/preflight.py --keep-going   # don't stop on first FAIL
python3 scripts/preflight.py --json         # emit JSON summary at end
```

Exit code:
  - `0` if every executed check PASSes (STUBs do not count as failures).
  - First-FAIL stop: `1`.
  - With `--keep-going`: number of FAILs.

## Phases and checks

### Phase A -- Vocabulary contract checks

These assert the Python extractor and the Rust binary speak the same
vocabulary. Each is the canonical "will the extractor's output even
parse?" gate.

| ID  | Check                          | Asserts                                                                                            | Audit punch item |
|-----|--------------------------------|----------------------------------------------------------------------------------------------------|------------------|
| A.1 | BondType                       | Every relation string emitted by `smart_extractor.py` parses via the binary's `parse_bond_type`.  | #1 (BondType vocabulary mismatch) |
| A.2 | NodeType                       | Every NodeType the extractor emits is in the Rust enum (today: extractor emits none -> no-op).    | #2 (NodeType vocabulary mismatch) |
| A.3 | KnowledgeContent kind presence | Distribution of `kind` field on emitted bonds is healthy. **STUB until BR.3** (kind not emitted yet). | #3 (KnowledgeContent kind always emitted) |
| A.4 | SourceKind                     | Producer variant tags are a subset of consumer variants in `SourceKind` enum.                      | #4 (SourceKind producer-consumer subset) |
| A.5 | Extension fields               | Documents and asserts current policy: extractor extension fields (`layer`, `frequency`) are accepted no-ops on Rust side. Fails if either side flips. | #5 (Cross-language extension fields) |

#### A.1 BondType detail

Relation strings are scraped dynamically from `scripts/smart_extractor.py`
(both the literal `"relation": "X"` style and the tuple-row
`(_RX_..., "X", weight, ...)` style in `_L3_PATTERNS` / `_L5_PATTERNS`).
Each is then probed against the binary by issuing a `lu4d_teach` call
with throwaway subject/object inside an ephemeral `--no-save` brain.
A `-32602` response whose error text mentions the relation is
interpreted as a parse rejection.

On the production binary today (commit `5b71af7`, BondType has 11
variants + Q.1 typed aliases), about 17 of the ~23 emitted relations
fail to parse. **A.1 is EXPECTED to fail until the BondType-extension
rebuild lands.** That's the proof preflight is doing its job.

#### A.5 Extension-fields policy

The Python extractor emits `layer` and `frequency` per bond. The
Rust MCP tool input schema for `lu4d_teach_corpus[_batch]` does not
declare these fields. Today they are silently dropped because the
Rust deserializer does not use `serde(deny_unknown_fields)`.

We pick **policy (a)**: extension fields are accepted no-ops. The
check asserts:
  1. The extractor still emits `layer` and `frequency`.
  2. No `serde(deny_unknown_fields)` directive exists anywhere under `src/`.

If either invariant flips (Rust gets strict, or the extractor stops
emitting), this check fails so we know the contract drifted.

### Phase B -- Configuration & deployment checks

| ID  | Check                       | Asserts                                                                                                                                  | Audit punch item |
|-----|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| B.1 | Binary md5                  | `md5(target/release/lu4d) == md5(/usr/local/bin/lu_4_mcp)`. Production binary matches the source we just built.                          | #6 |
| B.2 | Autosave audit              | Every launch path that runs `lu_4_mcp` (systemd unit, `/opt/lu/start_*.sh`, `lu_train_v*.py`) passes `--no-save`.              | #7 |
| B.3 | RuntimeConfig setter coverage | Every field of `RuntimeConfigSnapshot` (in `src/runtime_config.rs`) is reachable via at least one MCP setter (`lu4d_configure` or per-port). | #8 |
| B.4 | Working-tree cleanliness     | `git status --short` empty AND no `*.bak` files anywhere under the repo (excluding `.git/` and `target/`).                                 | #9 |

#### B.1 detail

The binary built by `cargo build --release` is named `lu4d` per
`Cargo.toml`. Production deploys it as `/usr/local/bin/lu_4_mcp` via
`sudo cp + chown root:lu`. If the deployed file's md5 does not match
the local build, you're about to ingest against a different binary
than the one you tested -- the check refuses to launch.

#### B.2 detail

`save_on_every_write` in `src/main.rs` defaults to `true` when the
brain has a path. The `--no-save` flag flips it to `false`. A 1+
hour ingest must use `--no-save` because mid-flight save fanout
adds multi-second pauses on every write. The check audits:
  - `/etc/systemd/system/lu.service`
  - `/opt/lu/start_*.sh`
  - `/opt/lu/lu_train_v*.py`

Any audited path that references `lu_4_mcp` or `target/release/lu4d`
but does not pass `--no-save` is reported as a failure with the
path name.

#### B.3 detail

Fields of `RuntimeConfigSnapshot` that are nested objects
(`trust_table`, `hub_degree`) are treated as covered when their
setter sub-tools exist (`lu4d_trust_set`, `hub_max_in_degree`,
etc.). New top-level fields without a setter (today: `sense_*`,
`chained_research_*`, `image_*`, `file_io_*`, `ingest_*`,
`tier2_prompt_*`, `max_chain_depth`, etc.) flag a failure so the
rebuild's per-port setter sweep is forced.

### Phase C -- Integration smoke

| ID  | Check                | Asserts                                                                                                                | Audit punch item |
|-----|----------------------|------------------------------------------------------------------------------------------------------------------------|------------------|
| C.1 | 50-article smoke     | 50 hand-crafted Wikipedia-shaped articles ingest cleanly: errors/article < 0.1, >=8 distinct relations, no relation > 70%, degree_p90 >= 5. | #10 |
| C.2 | Dual-mode typed-KU   | The smoke ingest passes in BOTH `recall.typed_ku.enabled = false` and `= true`. **STUB until BR.4** -- the runtime flag does not exist yet. | #11 |

#### C.1 detail

The 50 articles are EMBEDDED in `preflight.py` rather than streamed
from HuggingFace -- the check is hermetic, runs offline, and
finishes in seconds. Each article is hand-crafted to exercise
every extractor layer (L1 See also, L2 headings, L2b categories,
L3 lead patterns, L4 body co-mentions, L5 numerical attrs).

The brain is created with `mktemp -d` and `--no-save`, and is
`shutil.rmtree`'d in a `finally` clause. If the harness crashes
mid-flight the worst case is a leftover tmp dir under `/tmp/`.

C.1 will mostly fail today because A.1 fails -- most extractor
relations don't parse, so they error out instead of landing.
That's the gate's job: refuse the real ingest until A.1 + C.1
both pass.

## Architectural decisions encoded

1. **BondType extension** -- the rebuild adds first-class variants for
   every extractor relation (no aliasing). A.1 is the gate.
2. **KnowledgeContent kind always emitted** -- BR.3 makes the extractor
   emit `kind` on every bond. A.3 STUB upgrades to a healthy-distribution
   check at that point.
3. **Per-port setters + `tool_configure_batch`** -- B.3 forces every
   `RuntimeConfigSnapshot` field to have an exposed setter. New fields
   without a setter break the build.
4. **Q.1 typed-KU activation gated on runtime flag** -- C.2 STUB upgrades
   to a dual-mode smoke when `recall.typed_ku.enabled` lands.

## Hermetic guarantees

  - Canonical brain `/var/lib/lu/brain.lbr` is NEVER read or written.
  - Every MCP probe uses `--no-save` and a `tempfile.mkdtemp` brain dir.
  - Tmp dirs are `shutil.rmtree`'d in `finally` clauses.
  - No process is left behind (MCP subprocesses are explicitly
    `shutdown`+`wait`'d, then killed if they don't exit).
  - Read-only inspection of `src/*.rs`, `scripts/smart_extractor.py`,
    `Cargo.toml`, and the launch scripts -- no edits.

## When to update

  - When BR.3 lands (KnowledgeContent kind), upgrade A.3 from STUB
    to a real distribution check. See the `# TODO: requires BR.3`
    comment in `preflight.py`.
  - When BR.4 lands (`recall.typed_ku.enabled`), upgrade C.2 from
    STUB to a dual-mode smoke. See the `# TODO: requires BR.4`
    comment.
  - When the extractor begins emitting a `node_type` field, A.2's
    no-op branch upgrades automatically to a subset assertion -- no
    code change needed.
  - When the extractor adds new layers / patterns, A.1 picks them up
    automatically (relations are scraped dynamically from the source).

## BR.10b retrain invocation

Production retrain MUST pass `--seed-trust` to `scripts/ingest_curriculum_full.py`:

```
python3 scripts/ingest_curriculum_full.py --seed-trust
```

The flag invokes `lu4d_trust_reset_to_defaults { seed_curriculum_corpora: true }`
before any ingest call. Without it, `corpus_multipliers` stays empty and every
bond resolves to `default_multiplier=0.4` rather than the seeded curriculum
hierarchy (wikimedia/wikipedia=1.0, ranjaykrishna/visual_genome=1.0,
Voxel51/GQA-Scene-Graph=0.9, arxiv=0.9, stackexchange=0.8, grokipedia=1.0,
HuggingFaceH4/no_robots=0.7, HuggingFaceH4/instruction-dataset=0.7).

The gating is opt-in (default off) so existing operator behavior is unchanged.
The `t_ingest_1_default_does_not_seed_curriculum` Rust test continues to assert
that `TrustTable::default()` has empty `corpus_multipliers`; only the
runbook layer adds the seed.
