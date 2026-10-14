# The curiosity engine: how the loop closes

*OP-CURIOSITY-1, 2026-07-22. Implements the diagnosis in internal notes-1319.*

## The problem this document exists to explain

Lu's curiosity engine was never missing. Every part of it was built, correct, and live code. It was **unplugged** -- split in half across two processes, with no surface between them.

- **DETECT** runs inline in spread activation. It fires on essentially every real user query: 12 live probes produced 24 signals with genuine discrimination between them. It lives in the public `lu_guest_http` process.
- **FILL** -- hub-fill, triangle closure, single-hop research, chained research -- needs to **write bonds into the canonical brain**. Its only driver was a 2-second auto-tick thread inside `src/mcp.rs`, which is gated behind the `admin_mcp` feature. `lu_guest_http` builds `lu4d` with `default-features = false`, so dead-code elimination strips the entire filling half out of the serving binary.

So serving had the producer and no consumer. Training had the consumer and no producer -- no ingest script ever calls `ask`, so its queue was always empty. And the queue itself was a `BinaryHeap` in RAM that died with the process.

The result: **no evidence curiosity had ever filled a gap on the canonical brain.**

## What was built instead

### 1. A durable journal (`src/wonder_sidecar.rs`)

An append-only, line-delimited JSON file on a path both processes can name, **outside the brain directory**:

```
/var/lib/lu-wonder/wonder_queue.jsonl
```

One `WonderRecord` per line: the signal verbatim, the wall-clock append time, and the human-readable label the producer resolved (the consumer runs against a brain whose `gap_label_map` is empty, so a label not carried here is a label that is gone).

JSON-lines rather than a binary format on purpose: whole-line `O_APPEND` writes do not interleave, a torn tail costs exactly one record instead of the whole file, and -- for a demo-critical mechanism whose entire value is *showing* that a gap was detected and later filled -- it is greppable.

### 2. The guest is a PRODUCER and nothing more

`lu_guest_http --wonder-queue <dir>` attaches the journal. Every signal pushed to the in-RAM wonder queue is also appended.

This does not make the public process a writer to the brain:

- `Brain::load_readonly` still refuses `save` and `flush_now`.
- systemd still mounts `ReadOnlyPaths=/var/lib/lu` for that unit.
- The journal directory is the *single* narrow write capability it gains.
- `rewrite` (compaction) is a consumer operation the guest never calls.

A hostile or buggy guest can do exactly one thing to this system: append junk to a journal. The consumer treats every line as untrusted input and skips what it cannot parse.

Verified on the shipped artifact: `strings` finds `wonder_sidecar` present and `src/idle/loop_.rs`, `src/idle/auto_tick.rs`, `src/research/loop_.rs`, `src/consolidation.rs` and `src/mcp.rs` all **absent** (count 0). The OP-007 privsep property survives.

### 3. The FILL half is its own process (`src/bin/lu_curiosity_worker.rs`)

No listening socket, no auth surface, unreachable from the network. It loads the brain read-write, hydrates its wonder queue from the journal, runs N `idle_tick`s -- the real filling code, unchanged -- flushes, and retires the wonders it consumed.

The obvious alternative was to spawn the auto-tick driver inside the guest. That was rejected: it would make the internet-facing process a writer to the canonical brain and discard the privilege separation shipped in OP-007.

## The operational constraint: it is a SLEEP CYCLE

**The worker and `lu-guest-http` cannot run at the same time on this box.** Two reasons, both hard:

1. **RAM.** Loading the 233 GiB v5 brain peaks around 54 GiB and settles near 26 GiB, on 62 GiB total. Two loaders do not fit. Earlier attempts to co-load a second reader were OOM-killed at 22 GiB and 40 GiB while still loading.
2. **The refresh loop.** The guest polls the committed MANIFEST frontier and re-opens a *fresh* reader when it advances. A concurrent writer would therefore trigger a second full load *inside the guest itself*.

Training is a third tenant of the same brain and the same RAM. So:

> **Serving, training, and curiosity-fill are three mutually exclusive tenants. Any two of them together will OOM this box.**

This is enforced, not merely documented: the worker unit declares `Conflicts=lu-guest-http.service lu4d-*-ingest.service`, and the binary itself refuses to start if it sees `lu_guest_http` running.

The shape that falls out is a **sleep cycle**, and it is a better story than a background thread anyway: Lu wonders while awake, and consolidates while asleep.

```
sudo /usr/local/bin/lu-curiosity-cycle.sh --ticks 10
```

stops the guest, runs the worker, and brings the guest back on the enriched brain -- with a trap so the guest returns even if the worker fails.

## The approval gate

Everything the fill half produces lands in the `PendingLog` as `PendingStatus::Pending`. Until now the only way for a pending entry to become a bond was `lu4d_approve_changes`, an admin MCP tool that is physically absent from the guest binary. Combined with a non-durable PendingLog, this meant **"self-improving" was "self-proposing to nobody"**.

`CuriosityConfig::auto_apply` is the switch. It is **OFF by default**; turning it on is a product decision, not a default.

| knob | default | meaning |
|---|---|---|
| `auto_apply` | `false` | every fill waits for a human |
| `auto_apply_min_confidence` | `0.75` | confidence floor when on |
| `auto_apply_max_per_tick` | `8` | blast-radius bound |

**What the safety floor actually is** -- stated plainly, because it is easy to over-claim:

- Every candidate has already passed `push_pending_with_filter`, the Port T.8 learned-rejection pre-filter. Shapes rejected before never reach the applier.
- The 0.75 floor admits well-sourced research and **excludes bare internal synthesis by construction**, because triangle-closure proposals are hard-coded to confidence 0.5. The graph does not get to promote its own guesses into facts without external corroboration.
- `auto_apply_max_per_tick` bounds the damage if something goes wrong.

**What it is not.** It is not the K-gate. `crate::idk_gate` governs whether Lu *answers*, on the recall path; it has no opinion about writes and is not consulted here. Nor is it `ingest_config.auto_approve_threshold`, which is the `teach_corpus` gate and has never had any effect on idle output.

Every autonomous application is announced on stderr with its id, confidence and reason, so the journal shows exactly what the brain did to itself and when.

## Zero-reads that would have made this silently useless

Wiring the loop up is not enough if the filling code cannot see the graph. Two cache-backed-v5 zero-reads were fixed in the same pass:

- `idle/loop_.rs` resolved proposal subject/object via `network.get_node`, which reads the **resident** node map -- empty by design on a v5 brain. Every proposal hit the `None => continue` arm, so internal fill would have emitted *nothing at all*, silently. Now routed through `node_label` (resident -> memo -> node store).
- `network.rs` hard-coded the curiosity hub predicate's content to `""` in cache mode, so `is_hub_stopword` was dead there and hub classification was degree-threshold-only. Curiosity could not tell a function-word hub from a topical one.

## Files

| path | role |
|---|---|
| `src/wonder_sidecar.rs` | the journal format and IO |
| `src/brain.rs` | `attach_wonder_sidecar`, `hydrate_wonder_queue_from_sidecar`, `auto_apply_curiosity_pending` |
| `crates/lu_guest_http/src/main.rs` | producer wiring (`--wonder-queue`) |
| `src/bin/lu_curiosity_worker.rs` | the consumer |
| `scripts/lu-curiosity-cycle.sh` | the sleep cycle |
| `scripts/lu-curiosity-worker.service` | systemd unit with the exclusion invariant |
