#!/usr/bin/env python3
"""agg_r65.py -- aggregate the random-target runs: per method, all targets and split by label / by score sign.
Needs make_panel.py in the same folder.
Usage: python agg_r65.py --root $L/xai_scratch_r65 --targets $L/results/cv/resnet50_r224/targets_random65.csv [--out file.csv]"""
import argparse, glob, os
import numpy as np, pandas as pd
from make_panel import boot

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
T = pd.read_csv(a.targets)[["compound_id", "true_label"]].drop_duplicates("compound_id")
rows = []
for d in sorted(glob.glob(os.path.join(a.root, "vitcx_*"))):
    fs = glob.glob(os.path.join(d, "metrics_fold*.csv"))
    if not fs: continue
    M = pd.concat([pd.read_csv(f) for f in fs])
    M["dir"] = os.path.basename(d).replace("vitcx_", "").replace("_r224_r224", "").replace("_r448_r448", "")
    rows.append(M)
M = pd.concat(rows).merge(T, left_on="compound", right_on="compound_id", how="left")
M["score_sign"] = np.where(M["prob"] > 0, "logit>0", "logit<=0")   # 'prob' holds the explained score (logit)
out = []
for (d, v), g in M.groupby(["dir", "variant"]):
    for name, sub in (("all", g), ("label=1", g[g.true_label == 1]), ("label=0", g[g.true_label == 0]),
                      ("logit>0", g[g.score_sign == "logit>0"]), ("logit<=0", g[g.score_sign == "logit<=0"])):
        if len(sub) < 3: continue
        zd, zi = boot(sub.z_del.values, n=2000), boot(sub.z_ins.values, n=2000)
        out.append(dict(run=d, variant=v, group=name, n=len(sub), z_del=zd[0], del_lo=zd[1], del_hi=zd[2],
                        z_ins=zi[0], ins_lo=zi[1], ins_hi=zi[2]))
R = pd.DataFrame(out)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
print(R.round(2).to_string(index=False))
if a.out: R.to_csv(a.out, index=False); print("wrote", a.out)
