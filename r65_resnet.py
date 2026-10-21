#!/usr/bin/env python3
"""r65_resnet.py -- cell enrichment of the ResNet Grad-CAM maps on the 65 random targets (same test as r65_enrich.py).
Images come from the DINOv3 run (same compounds). Needs make_panel.py in the same folder.
Usage: python r65_resnet.py --root $L/xai_scratch_r65 --gradcam $L/xai_scratch_r65/gradcam_resnet --index $L/xai_scratch_r65/sheets/targets_index.csv [--out file.csv]"""
import argparse, glob, os
import numpy as np, pandas as pd
from make_panel import FN, fit, enrich, boot, resnet_map

SPARSE = {6, 35, 44, 55, 62, 65}
NOISY = {45, 57}
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--gradcam", required=True)
ap.add_argument("--index", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
idx = pd.read_csv(a.index, dtype=str)
grp = {(int(r["fold"]), r["compound"], r["assay"]): ("noisy" if int(r["id"]) in NOISY else "sparse" if int(r["id"]) in SPARSE else "normal")
       for _, r in idx.iterrows()}
ref = sorted(glob.glob(os.path.join(a.root, "vitcx_dinov3*r448_r448")))[0]
rows, miss = [], 0
for f in sorted(glob.glob(os.path.join(ref, "f*.npz"))):
    m = FN.match(os.path.basename(f))
    if not m: continue
    fold, comp, assay = int(m.group(1)), m.group(2), m.group(3)
    mp = resnet_map(a.gradcam, assay, comp)
    if mp is None: miss += 1; continue
    x = np.load(f)["x"].astype(np.float32)
    rows.append(dict(group=grp.get((fold, comp, assay), "?"), enrich=enrich(x, fit(mp, x.shape[1:]))))
R = pd.DataFrame(rows)
print(len(R), "maps scored,", miss, "missing")
for name, sub in (("all", R), ("normal", R[R.group == "normal"]), ("sparse+noisy", R[R.group != "normal"])):
    if len(sub) >= 3:
        b = boot(sub.enrich.values, n=2000)
        print(f"ResNet-50 Grad-CAM  {name:13s} n={len(sub):2d}  enrich {b[0]:.2f} ({b[1]:.2f} to {b[2]:.2f})")
if a.out: R.to_csv(a.out, index=False)
