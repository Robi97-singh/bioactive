"""
Computes real per-channel mean/std over a random sample of downloaded
brightfield PNGs, to replace the placeholder (0.5,)/(0.25,) in
BioActBrightfield.init_stats() (defaults/datasets.py). Images are already
percentile-normalized uint8 grayscale, so this just measures their actual
pixel statistics on the [0, 1] scale used by the dataset's ToTensor step.

    python3 scripts/brightfield/04_compute_brightfield_stats.py
    python3 scripts/brightfield/04_compute_brightfield_stats.py --n 5000 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_ROOT = Path("/shared/hdd/data/bioactive/brightfield_images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of images to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    all_pngs = list(IMAGES_ROOT.glob("*/*.png"))
    if not all_pngs:
        print(f"[stats] no PNGs found under {IMAGES_ROOT} -- has the download finished?",
              file=sys.stderr)
        sys.exit(1)

    print(f"[stats] found {len(all_pngs)} total images under {IMAGES_ROOT}")
    sample = random.sample(all_pngs, min(args.n, len(all_pngs)))
    print(f"[stats] sampling {len(sample)} images (seed={args.seed})")

    sums = 0.0
    sums_sq = 0.0
    n_pixels = 0

    for i, p in enumerate(sample, 1):
        arr = np.asarray(Image.open(p), dtype=np.float64) / 255.0
        sums += arr.sum()
        sums_sq += (arr ** 2).sum()
        n_pixels += arr.size
        if i % 1000 == 0:
            print(f"[stats]   {i}/{len(sample)} images processed", flush=True)

    mean = sums / n_pixels
    var = (sums_sq / n_pixels) - mean ** 2
    std = var ** 0.5

    print()
    print(f"[stats] n_images={len(sample)}  n_pixels={n_pixels}")
    print(f"[stats] mean = {mean:.6f}")
    print(f"[stats] std  = {std:.6f}")
    print()
    print("Paste these into BioActBrightfield.init_stats() in defaults/datasets.py:")
    print(f"    self.mean = ({mean:.4f},)")
    print(f"    self.std  = ({std:.4f},)")


if __name__ == "__main__":
    main()
