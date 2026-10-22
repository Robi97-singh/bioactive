#!/usr/bin/env python3
"""cv_rescore.py -- compound-level test ROC-AUC per fold and assay from the oof_predictions.csv written by dump_predictions.py.
Per compound: mean of the per-image predictions and of the labels (1 = positive, -1 = negative, 0 = not measured). ROC-AUC per assay on measured labels (rank formula, ties averaged); mean over assays that have both classes.
Usage: python3 cv_rescore.py --eval DIR --out DIR [--reported DIR] --arm NAME SUBDIR [REPORTED.txt] [--arm ...]
Writes OUT/NAME/cv_summary.csv and cv_per_assay.csv; if REPORTED.txt is given (lines 'foldN: value'), prints the difference."""
import argparse, glob, os, re
import numpy as np, pandas as pd

def auc(s, y):
    r = s.rank(method="average"); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True); ap.add_argument("--reported", default=".")
ap.add_argument("--arm", nargs="+", action="append", required=True, metavar="NAME SUBDIR [TXT]")
a = ap.parse_args()
for arm in a.arm:
    name, sub, txt = arm[0], arm[1], (arm[2] if len(arm) > 2 else None)
    hits = glob.glob(os.path.join(a.eval, sub, "**", "oof_predictions.csv"), recursive=True)
    assert len(hits) == 1, (name, hits)
    D = pd.read_csv(hits[0]); od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
    assert sorted(D.fold.unique()) == list(range(6)), sorted(D.fold.unique())
    pc = [c for c in D.columns if c.startswith("pred_")]; lc = ["label_" + c[5:] for c in pc]
    per, rows = {}, []
    for f in range(6):
        d = D[D.fold == f]
        P = d.groupby("compound_id")[pc].mean(); L = d.groupby("compound_id")[lc].mean()
        r = {}
        for c, l in zip(pc, lc):
            t = L[l]; m = t != 0
            y = ((t[m] + t[m] ** 2) / 2).round().astype(int); s = P.loc[m, c]
            if (y == 1).sum() > 0 and (y == 0).sum() > 0: r[c[5:]] = auc(s, y)
        per[f] = pd.Series(r); rows.append((f, per[f].mean(), len(r)))
        print(name, "fold", f, "| images", len(d), "| compounds", P.shape[0], "| assays", len(r))
    S = pd.DataFrame(rows, columns=["fold", "mean_test_roc_auc", "n_assays"]); v = S.mean_test_roc_auc.values
    S.loc[len(S)] = ["MEAN", v.mean(), np.nan]; S.loc[len(S)] = ["SD", v.std(ddof=1), np.nan]
    S.to_csv(os.path.join(od, "cv_summary.csv"), index=False)
    A = pd.DataFrame({"fold%d" % f: per[f] for f in range(6)}); A["mean"] = A.mean(axis=1)
    A.index.name = "assay"; A.to_csv(os.path.join(od, "cv_per_assay.csv"))
    print(name, "recomputed:", " ".join("%.4f" % x for x in v), "| mean %.4f SD %.4f" % (v.mean(), v.std(ddof=1)))
    if txt:
        rep = {int(k): float(x) for k, x in re.findall(r"fold(\d):\s*([0-9.]+)", open(os.path.join(a.reported, txt)).read())}
        rv = np.array([rep[f] for f in range(6)])
        print(name, "his text:  ", " ".join("%.4f" % x for x in rv), "| max abs diff %.1e" % np.abs(rv - v).max())
