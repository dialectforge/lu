#!/usr/bin/env python3
"""Generate NOVEL geometry probe images with an INDEPENDENT renderer (PIL/numpy)
so the Rust perceiver reads pixels it has never seen. Also transcode John's real
redleaf.jpeg -> PNG (a pure format conversion; perception still happens in Rust).
Nothing here shares code with the Rust extractor."""
import os, numpy as np
from PIL import Image

OUT = "/tmp/geoprobe"
os.makedirs(OUT, exist_ok=True)


def figure(side_px, pad, mode="leaf"):
    N = side_px + 2 * pad
    img = np.full((N, N, 3), 255, np.uint8)
    ys, xs = np.mgrid[0:N, 0:N]
    x0, y0, x1, y1 = pad, pad, pad + side_px, pad + side_px
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    dtl = np.hypot(xs - x0, ys - y0)
    dbr = np.hypot(xs - x1, ys - y1)
    if mode == "leaf":
        shaded = inside & (dtl <= side_px) & (dbr <= side_px)
    else:  # quarter
        shaded = inside & (dtl <= side_px)
    img[shaded] = [220, 30, 30]
    border = inside & (
        (np.abs(xs - x0) < 2) | (np.abs(xs - x1) < 2)
        | (np.abs(ys - y0) < 2) | (np.abs(ys - y1) < 2)
    )
    img[border] = [0, 0, 0]
    return Image.fromarray(img)


# Deliberately odd pixel sizes the Rust code has never seen.
figure(263, 31, "leaf").save(f"{OUT}/leaf_novel_A.png")   # pass side=11
figure(197, 27, "leaf").save(f"{OUT}/leaf_novel_B.png")   # pass side=6
figure(263, 31, "quarter").save(f"{OUT}/quarter_novel.png")  # pass side=8

# John's real photo -> PNG (format transcode only; not perception).
src = "prototypes/perception/redleaf.jpeg"
if os.path.exists(src):
    Image.open(src).convert("RGB").save(f"{OUT}/redleaf_real.png")
    print("transcoded", src, "->", f"{OUT}/redleaf_real.png")

for f in sorted(os.listdir(OUT)):
    print(f, Image.open(f"{OUT}/{f}").size)
