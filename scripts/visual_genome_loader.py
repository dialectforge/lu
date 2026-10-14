#!/usr/bin/env python3
"""Visual Genome scene-graph triplet streamer.

Reads the canonical VG relationships dump
(`relationships.json.zip` from
https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/),
yields normalized (subject, predicate, object, image_id) triplets.

Written because `ranjaykrishna/visual_genome` is script-backed and broken
under `datasets>=4.0`, and `Voxel51/GQA-Scene-Graph` only ships JPEGs.

The dump is one giant top-level JSON array (~743 MB uncompressed,
~108k images, ~2.3M relationships). We stream with ijson so RSS stays small.
"""

from __future__ import annotations
import os
import zipfile
from typing import Iterator, Tuple, Optional

DEFAULT_CACHE = "/var/lib/lu/data/visual_genome/relationships.json.zip"
CANONICAL_URL = (
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
    "relationships.json.zip"
)

# VG predicate -> Lu relation type. Three-bucket mapping:
#   HasProperty: attributes / possessions / wearing
#   PartOf:      containment / composition / spatial "on/in/of"
#   RelatedTo:   spatial neighborhoods, actions, fallback
VG_PRED_MAP = {
    "has": "HasProperty", "have": "HasProperty",
    "has a": "HasProperty", "have a": "HasProperty",
    "has an": "HasProperty", "have an": "HasProperty",
    "with": "HasProperty", "wearing": "HasProperty",
    "wears": "HasProperty", "wear": "HasProperty",
    "holding": "HasProperty", "holds": "HasProperty",
    "carrying": "HasProperty", "carries": "HasProperty",
    "covered with": "HasProperty", "covered in": "HasProperty",
    "on": "PartOf", "in": "PartOf", "of": "PartOf",
    "part of": "PartOf", "made of": "PartOf", "inside": "PartOf",
    "on top of": "PartOf", "attached to": "PartOf",
    "belonging to": "PartOf", "belongs to": "PartOf",
}


def map_predicate(p: str) -> str:
    """Map a raw VG predicate string to a Lu relation type."""
    return VG_PRED_MAP.get((p or "").lower().strip(), "RelatedTo")


def _node_name(node) -> str:
    """VG nodes alternately use `name` (singular str) or `names` (list)."""
    if not isinstance(node, dict):
        return ""
    n = node.get("name")
    if isinstance(n, str) and n.strip():
        return n.strip().lower()
    ns = node.get("names")
    if isinstance(ns, list) and ns:
        first = ns[0]
        if isinstance(first, str) and first.strip():
            return first.strip().lower()
    return ""


def ensure_cached(path: str = DEFAULT_CACHE) -> str:
    """Verify the canonical dump is on disk. Returns path or raises."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VG relationships dump not at {path!r}. Fetch with:\n"
            f"  curl -sSL -o {path} {CANONICAL_URL}"
        )
    return path


def stream_triplets(
    cache_path: str = DEFAULT_CACHE,
    max_triplets: Optional[int] = None,
    max_per_image: int = 25,
) -> Iterator[Tuple[str, str, str, int]]:
    """Yield (subject, raw_predicate, object, image_id) tuples.

    `raw_predicate` is the un-mapped VG string; callers run `map_predicate`
    on it. We yield the raw form so the harness can attribute against the
    original token while still landing a normalized Lu relation.

    Skips empty-name records and applies a per-image cap so a few densely
    annotated images don't dominate the stream.
    """
    import ijson
    ensure_cached(cache_path)
    yielded = 0
    with zipfile.ZipFile(cache_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".json")]
        if not members:
            raise RuntimeError(f"no .json in {cache_path}")
        with zf.open(members[0]) as fh:
            for image in ijson.items(fh, "item"):
                if max_triplets is not None and yielded >= max_triplets:
                    return
                image_id = image.get("image_id") if isinstance(image, dict) else None
                rels = image.get("relationships") if isinstance(image, dict) else None
                if not isinstance(rels, list):
                    continue
                emitted = 0
                for r in rels:
                    if emitted >= max_per_image:
                        break
                    if not isinstance(r, dict):
                        continue
                    pred = r.get("predicate") or ""
                    if not isinstance(pred, str) or not pred.strip():
                        continue
                    s = _node_name(r.get("subject"))
                    o = _node_name(r.get("object"))
                    if not s or not o:
                        continue
                    yield (s, pred.strip(), o, int(image_id) if image_id is not None else 0)
                    emitted += 1
                    yielded += 1
                    if max_triplets is not None and yielded >= max_triplets:
                        return


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    args = ap.parse_args()
    n = 0
    for s, p, o, img in stream_triplets(args.cache, max_triplets=args.limit):
        print(f"{img}\t{s}\t{p}\t{map_predicate(p)}\t{o}")
        n += 1
    print(f"# yielded={n}", file=sys.stderr)
