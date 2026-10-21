#!/usr/bin/env python3
"""r65_enrich.py -- cell enrichment on the 65 random targets, overall and by image group.
Groups (labelled by eye from the contact sheets): sparse = few cells in frame, noisy = no usable cells, rest = normal.
Needs make_panel.py in the same folder.
Usage: python r65_enrich.py --root $L/xai_scratch_r65 --index $L/xai_scratch_r65/sheets/targets_index.csv [--out file.csv]"""
import argparse, glob, os
import numpy as np, pandas as pd
from make_panel import COLS, FN, pick, fit, enrich, boot

SPARSE = {6, 35, 44, 55, 62, 65}
NOISY = {45, 57}
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--index", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
idx = pd.read_csv(a.index, dtype=str)
grp = {}
for _, r in idx.iterrows():
    i = int(r["id"])
    grp[(int(r["fold"]), r["compound"], r["assay"])] = "noisy" if i in NOISY else "sparse" if i in SPARSE else "normal"
rows = []
for title, g, sub in COLS:
    ds = sorted(glob.glob(os.path.join(a.root, g)))
    if not ds: continue
    for f in sorted(glob.glob(os.path.join(ds[0], "f*.npz"))):
        m = FN.match(os.path.basename(f))
        if not m: continue
        key = (int(m.group(1)), m.group(2), m.group(3))
        z = np.load(f); x = z["x"].astype(np.float32)
        mp = fit(pick(z, sub), x.shape[1:])
        rows.append(dict(method=title.replace("\n", " "), group=grp.get(key, "?"), enrich=enrich(x, mp)))
R = pd.DataFrame(rows)
out = []
for meth, g in R.groupby("method"):
    for name, sub in (("all", g), ("normal", g[g.group == "normal"]), ("sparse+noisy", g[g.group != "normal"])):
        if len(sub) < 3: continue
        b = boot(sub.enrich.values, n=2000)
        out.append(dict(method=meth, group=name, n=len(sub), enrich=b[0], lo=b[1], hi=b[2]))
O = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(O.round(2).to_string(index=False))
if a.out: O.to_csv(a.out, index=False); print("wrote", a.out)
