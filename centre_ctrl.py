#!/usr/bin/env python3
"""centre_ctrl.py -- compares the image-independent centre / border maps with the real methods, per image class.
Usage: python centre_ctrl.py --root $L/xai_scratch   (needs strat_panel.py + make_panel.py in the same folder)"""
import argparse, glob, os
import numpy as np, pandas as pd
from make_panel import boot
from strat_panel import CELLS

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
rows = []
for d in sorted(glob.glob(os.path.join(a.root, "vitcx_*"))):
    nm = os.path.basename(d)
    if "centre" in nm: kind = "CONTROL"
    elif "gradcam" in nm: kind = "gradcam"
    elif "occlusion" in nm: kind = "occlusion"
    else: kind = "vitcx"
    model = "dinov3" if "dinov3" in nm else "celldino"
    for f in glob.glob(os.path.join(d, "metrics_fold*.csv")):
        M = pd.read_csv(f)
        M["dir"] = nm; M["kind"] = kind; M["model"] = model
        rows.append(M)
M = pd.concat(rows)
M["cls"] = ["cells" if (str(c), str(s)) in CELLS else "noisy" for c, s in zip(M.compound, M.assay)]
M = M[M.variant.isin(["centre", "border", "paper", "occlusion", "gradxact"])]
out = []
for (model, var, cls), d in M.groupby(["model", "variant", "cls"]):
    for col in ("z_del", "z_ins"):
        b = boot(d[col].values, n=3000)
        out.append(dict(model=model, map=var, cls=cls, metric=col, n=len(d), mean=b[0], lo=b[1], hi=b[2]))
R = pd.DataFrame(out).sort_values(["model", "cls", "metric", "map"])
pd.set_option("display.width", 200); print(R.round(2).to_string(index=False))
