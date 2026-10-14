# Metric/constraint geometry solver (solver.py) — angle & length problems

A second geometry engine, complementary to the Phase-3 shaded-area engine.

## How it works
1. Model the figure as points + exact constraints: distance, angle, perpendicular,
   collinear, midpoint, on-circle / diameter, equal-distance.
2. Place it numerically (hand-rolled Levenberg-Marquardt, central-difference
   Jacobian) from several RANDOM seeds. Gauge DOFs (e.g. an unconstrained scale)
   are left free on purpose.
3. Read the queried length or angle; RECOVER its exact closed form (small-
   denominator rational, sqrt(rational), or simple-fraction degrees).

## K-gate (refuse, don't fabricate)
- INVARIANCE: the answer must be identical across all random valid placements.
  If it varies, the figure is under-determined -> REFUSE.
- SMALL-DENOMINATOR ONLY: recovery accepts only small-denominator closed forms
  verified to ~1e-7. A value with no such form (e.g. 2*sin20deg) is REFUSED,
  not approximated by a large fraction.

## Verified on the three sample images
- Problem 1 (angle DBE, two circles on a triangle): **110 deg**
- Problem 2 (semicircle, two 60deg rays, given 8): **x = 16**
- Problem 3 (angle B = 2*angle DAC, BD=4, DC=3): **AB = 7**
Plus refusal tests: under-determined length -> refuse; 2*sin20 -> refuse.

## Boundary / next
- Recovers rational / sqrt(rational) / simple-degree answers; exotic forms
  (pi-multiples for lengths, nested radicals) currently refuse (honest).
- Modeling (problem text/image -> constraints) is still by hand; wiring
  perception + NL parsing into constraint generation is the next layer.
- Prototype is Python; a Rust port would mirror the Phase-3 approach.
