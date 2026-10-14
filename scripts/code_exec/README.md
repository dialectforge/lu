# Code-execution + code-writing tooling (X1 / S1 / free-form)

Standalone tools for Lu's code capabilities. Prototype tier — the sandbox is
production-grade; the planners are Python prototypes proven end-to-end on the
C2 coding brain (not yet ported into `synthesis.rs`).

- `x1_broker.py` + `launcher.py` — **X1 sandbox** gatekeeper. One fresh sandbox
  per run: root-launched bwrap dropped to the unprivileged `lu-sandbox` user,
  all namespaces (empty netns = no network), read-only minimal rootfs (NO
  brain/master.key/repo), ephemeral per-run FS, cgroups + rlimits + seccomp +
  wall-clock. Returns stdout/stderr/exit + one produced file (downloadable).
  G-X1 security battery: 15/15.
- `c2_handles.py` — **option-2** distinctive seedable coding handles
  (`pyfn<op>`/`jsfn<op>` -> canonical `the <op> function` node). Requires the
  C2-ingested brain.
- `c2_depth.py` — C2-deep op-semantics depth pass (HasParameter/Returns/Causes).
- `s1_planner_c2.py` — S1 synthesis planner (fixed-task demonstrator) grounding
  on real C2 op-facts, verifying via X1 (run->observe->fix), cases -> separate
  store.
- `freeform_planner.py` — **free-form** planner: NL intent parse + data-plumbing
  composer + pattern library (retrieve+adapt known patterns) + X1 run/fix +
  honesty floor (novel logic -> attempt/run/fail/say-so or honest abort).
  Verified 17/17 solvable prompts end-to-end + 3/3 honest on research-hard.

Serve the sense-router demo brain with `LU_SENSE_ROUTER=1 LU_SENSE_ROUTER_BOOST=4`.

NOT wired into the live guest daemon yet, and NOT reimplemented in `synthesis.rs`
(that production landing is the recommended focused follow-up).
