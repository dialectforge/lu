#!/usr/bin/env python3
"""Generate the Lu PWA icon set from luAI.jpeg.

No committed icon spec ever existed (internal notes: only "if visual identity
 changes (logo, color palette), templates require update"), and the source
 art is 960x1088 -- non-square -- so this derives square icons
 deterministically rather than hand-cropping:

  icon-192.png / icon-512.png   center square crop, full bleed ('any')
  icon-maskable-512.png         same art at 76% on the app background, so
                                Android's circle/squircle mask cannot clip it
                                (maskable safe zone is the inner 80% circle)
  apple-touch-icon.png          180x180, opaque, no transparency (iOS)

Re-runnable: same input always produces the same output.
"""

import os
import sys

from PIL import Image

BG = (11, 11, 12)  # --bg  #0b0b0c


def center_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else 'luAI.jpeg'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'icons'

    os.makedirs(out_dir, exist_ok=True)
    src = Image.open(src_path).convert('RGB')
    square = center_square(src)

    for size in (192, 512):
        square.resize((size, size), Image.LANCZOS).save(
            os.path.join(out_dir, 'icon-%d.png' % size), 'PNG', optimize=True
        )

    # Maskable: art inset inside the safe zone on the app background.
    inner = int(512 * 0.76)
    canvas = Image.new('RGB', (512, 512), BG)
    canvas.paste(square.resize((inner, inner), Image.LANCZOS),
                 ((512 - inner) // 2, (512 - inner) // 2))
    canvas.save(os.path.join(out_dir, 'icon-maskable-512.png'), 'PNG', optimize=True)

    square.resize((180, 180), Image.LANCZOS).save(
        os.path.join(out_dir, 'apple-touch-icon.png'), 'PNG', optimize=True
    )

    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        print('%-28s %7d bytes  %s' % (name, os.path.getsize(path),
                                       Image.open(path).size))


if __name__ == '__main__':
    main()
