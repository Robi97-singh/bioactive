#!/usr/bin/env python3
"""sample_targets.py -- random (compound, assay) targets from out-of-fold predictions, independent of confidence.
Writes a targets csv readable by run_vitcx.py. Assays are written as '#<column index>' (run_vitcx maps them to names).
Usage: python sample_targets.py --predictions $L/results/cv/resnet50_r224/oof_predictions.csv --out $L/results/cv/resnet50_r224/targets_random64.csv
         [--n_per_fold 13] [--folds 0 1 2 3 4] [--pos_frac 0.5] [--seed 0]"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_per_fold", type=int, default=13); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--pos_frac", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
P = pd.read_csv(a.predictions)
cc = "compound_id" if "compound_id" in P else "compound"
K = sorted(int(c.split("_")[1]) for c in P.columns if c.startswith("pred_"))
rng = np.random.default_rng(a.seed)
rows = []
for f in a.folds:
    D = P[P.fold == f].reset_index(drop=True)
    pairs = []
    for k in K:
        lab = D["label_%d" % k].values
        for i in np.where(lab >= 0)[0]:
            pairs.append((i, k, int(lab[i])))
    pairs = pd.DataFrame(pairs, columns=["i", "k", "y"])
    n_pos = int(round(a.n_per_fold * a.pos_frac)); n_neg = a.n_per_fold - n_pos
    used = set()
    for y, n in ((1, n_pos), (0, n_neg)):
        pool = pairs[pairs.y == y].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        got = 0
        for _, r in pool.iterrows():
            comp = D.loc[r.i, cc]
            if comp in used: continue
            used.add(comp); got += 1
            rows.append(dict(compound_id=comp, assay="#%d" % r.k, true_label=y, pred_prob=float(D.loc[r.i, "pred_%d" % r.k]),
                             fold=f, selection_reason="random_pos" if y == 1 else "random_neg"))
            if got == n: break
        if got < n: print("fold %d label %d: only %d of %d available" % (f, y, got, n))
O = pd.DataFrame(rows)
O.to_csv(a.out, index=False)
print("wrote", a.out, "| rows", len(O), "| compounds", O.compound_id.nunique(), "| assays", O.assay.nunique(), "| positives", int(O.true_label.sum()))
print(O.groupby("fold").size().to_dict())
