#!/usr/bin/env python3
"""contact_sheet.py -- numbered contact sheets of the random-target images, to label cells vs noisy by eye.
Needs make_panel.py (rgb, FN) in the same folder.
Usage: python contact_sheet.py --dir $L/xai_scratch_r65/vitcx_dinov3_r448_r448 --out $L/xai_scratch_r65/sheets [--per_page 25]
Writes sheet_1.png ... and targets_index.csv (id, fold, compound, assay)."""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_panel import rgb, FN

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per_page", type=int, default=25)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
items = []
for f in glob.glob(os.path.join(a.dir, "f*.npz")):
    m = FN.match(os.path.basename(f))
    if m: items.append((int(m.group(1)), m.group(2), m.group(3), f))
items.sort()
idx = pd.DataFrame([dict(id=i + 1, fold=t[0], compound=t[1], assay=t[2]) for i, t in enumerate(items)])
idx.to_csv(os.path.join(a.out, "targets_index.csv"), index=False)
n, cols = len(items), 5
for p in range(0, n, a.per_page):
    chunk = items[p:p + a.per_page]
    rows = int(np.ceil(len(chunk) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
    for ax in axs.ravel(): ax.axis("off")
    for k, (fold, comp, assay, f) in enumerate(chunk):
        x = np.load(f)["x"].astype(np.float32)
        ax = axs[k // cols][k % cols]
        ax.imshow(rgb(x)); ax.set_title(f"#{p + k + 1}  f{fold}  {comp[-6:]}/{assay}", fontsize=9)
    fig.tight_layout()
    out = os.path.join(a.out, f"sheet_{p // a.per_page + 1}.png")
    fig.savefig(out, dpi=110); plt.close(fig); print("wrote", out)
print(n, "images; index in", os.path.join(a.out, "targets_index.csv"))
