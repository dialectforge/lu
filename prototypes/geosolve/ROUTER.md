# Geometry solver-type router (router.py)

Lu has multiple exact engines; the wrong one yields no answer. The router decides
from the PROMPT which to use, non-LLM (feature/keyword), and refuses/clarifies
rather than guessing. Intended to fold into the existing query_router.

## Decision
- bare or word-form arithmetic ("12*(3+4)", "7 plus 5")      -> arithmetic
- "shaded" + area                                            -> shaded_area  (Phase-3 arrangement/Green)
- explicit angle query (find/measure angle, ∠, how many deg) -> metric_geometry (angle)
- length target (find x, find AB, length, perimeter, radius) -> metric_geometry (length)
- area of a dimensioned shape (no "shaded")                  -> metric_geometry (area / formula)
- geometry words but unclear target                          -> CLARIFY (refuse to guess)
- no geometry, no arithmetic                                 -> recall (associative, not a solver)

A strong query target (find x / find AB / angle) signals geometry even with no
shape noun (the figure usually supplies the rest via perception).

## Result: 14/14 routing tests; end-to-end (end_to_end.py)
prompt -> router -> solver -> exact answer:
  "Find angle DBE ... 40"   -> metric/angle  -> 110 deg
  "Find AB ... BD=4 DC=3"   -> metric/length -> 7
  "length x ... 60deg ... 8"-> metric/length -> 16
  "area of the shaded ..."  -> shaded_area    -> (Phase-3) e.g. 32*pi-64

## Next
- Subtype is a hint; the metric solver handles angle+length either way.
- Real integration: port into src/query_router.rs alongside the existing routes,
  and connect perception output as an additional geometry signal.
