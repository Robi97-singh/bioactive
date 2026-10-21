#!/usr/bin/env python3
"""efficientnet_cv.py -- compound-level test ROC-AUC per fold and assay for the two EfficientNet-B3 arms (448 px and 224 px; runs by B. Parihar).
Reads test_preds.csv and test_labels.csv (rows = compounds, columns = assays, label -1 = not measured) from each fold folder, recomputes ROC-AUC per assay (rank formula, ties averaged), averages over assays and compares with the numbers in his text files.
Usage: python3 efficientnet_cv.py --src /shared/ssd/logs/b-b-parihar --out results_showcase/efficientnet_b3"""
import argparse, os, re
import numpy as np, pandas as pd

def auc(s, y):
    r = s.rank(method="average"); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

ARMS = {"efficientnet_b3_r448": ("bioact_efficientnet_b3_fold{}", "effnet_b3_448_cv_results.txt"),
        "efficientnet_b3_r224": ("bioact_efficientnet_b3_r224_fold{}", "effnet_b3_r224_cv_results.txt")}
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()


for arm, (sub, txt) in ARMS.items():
    od = os.path.join(a.out, arm); os.makedirs(od, exist_ok=True)
    D = pd.read_csv(os.path.join(a.eval, sub, "oof_predictions.csv"))
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
        print(arm, "fold", f, "| images", len(d), "| compounds", P.shape[0], "| assays", len(r))
    S = pd.DataFrame(rows, columns=["fold", "mean_test_roc_auc", "n_assays"])
    v = S.mean_test_roc_auc.values
    S.loc[len(S)] = ["MEAN", v.mean(), np.nan]; S.loc[len(S)] = ["SD", v.std(ddof=1), np.nan]
    S.to_csv(os.path.join(od, "cv_summary.csv"), index=False)
    A = pd.DataFrame({"fold%d" % f: per[f] for f in range(6)}); A["mean"] = A.mean(axis=1)
    A.index.name = "assay"; A.to_csv(os.path.join(od, "cv_per_assay.csv"))
    rep = [float(x) for _, x in re.findall(r"fold(\d):\s*([0-9.]+)", open(os.path.join(a.reported, txt)).read())]
    print(arm, "recomputed:", " ".join("%.4f" % x for x in v), "| mean %.4f SD %.4f" % (v.mean(), v.std(ddof=1)))
    print(arm, "his text:  ", " ".join("%.4f" % x for x in rep), "| max abs diff %.1e" % np.abs(np.array(rep) - v).max())
