# Phase 3 prototype — automatic region decomposition

Closes the Phase-2 gap: the region decomposition is no longer hand-written.
Inputs are ONLY the figure primitives (bounding square + circular arcs). The
code builds the planar ARRANGEMENT of the curves, extracts the atomic FACES
(half-edge / DCEL walk), and computes each face area EXACTLY via Green's
theorem (a + b*pi), keeping the sector angle symbolic so intermediate angles
cancel. A sum-to-whole check verifies all faces reconstruct the figure (144).

Prototype only (Python). Not wired into the brain/service. Rust port TBD,
behind the existing IDK gate.

## Run
    python3 redleaf_demo.py

## Red-leaf result (no hand cuts)
- Arc-only arrangement -> 4 atomic faces summing to 144 exactly.
- LEFT LOBE = 32*pi - 64 falls out as a SINGLE atomic face (matches Phase-1).
- FULL LEAF = 48*pi - 96 = left lobe + right wedge (16*pi-32); the wedge needs
  the tip-to-tip chord (0,0)-(12,12) because arc C is internally tangent to
  arc A, so the arcs alone never cut it off.
- Honesty gate: corrupting any face breaks the sum (!=144) -> flag, not a wrong
  number.

## Files
- auto_partition.py  — the engine (Area algebra, intersections, arrangement,
  faces, Green areas, point-in-face identification).
- redleaf_demo.py    — the acceptance run.
