#!/usr/bin/env python3
"""
recompute_clip_per_assay.py

Recompute CLIP ViT-L/14 (frozen) per-assay ROC-AUCs from the teammate's saved
test_preds.csv / test_labels.csv (6 folds), using the SAME masking convention as
the project's mean_roc_auc metric, then aggregate to cv/clip_vitl14_r224/cv_per_assay.csv
in the standard format (so it slots into the benchmark figures exactly like the other arms).

Masking (matches utils.metrics.mean_roc_auc):
  labels are in {-1, 0, +1};  target = (y + y**2)/2  -> maps -1->0, +1->1;  mask = y**2 > 0
  per-assay AUC computed only on masked (tested) compounds, only if both classes present.

Validates the recomputed per-fold means against the teammate's reported values (edit
REPORTED below if the numbers have changed / this is a fresh run).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLIP_RESULTS = "/shared/ssd/logs/b-b-parihar/results"
CLIP_FOLD_PREFIX = "bioact_clip_vitl14_frozen_fold"
OUT_DIR = "/shared/ssd/logs/b-r-singh1/results/cv/clip_vitl14_r224"
os.makedirs(OUT_DIR, exist_ok=True)

# teammate's previously reported per-fold means, for validation (update if he re-ran)
REPORTED = {0: 0.6031105552448568, 1: 0.6322470379545461, 2: 0.6152166071204336,
            3: 0.6537856178407845, 4: 0.6400149755715235, 5: 0.6429076777625257}

ASSAY_COLS = [f"assay_{i}" for i in range(29)]


def per_assay_auc_for_fold(fold):
    d = os.path.join(CLIP_RESULTS, f"{CLIP_FOLD_PREFIX}{fold}")
    labels = pd.read_csv(os.path.join(d, "test_labels.csv"), index_col=0)
    preds = pd.read_csv(os.path.join(d, "test_preds.csv"), index_col=0)
    labels = labels[ASSAY_COLS].values
    preds = preds[ASSAY_COLS].values

    aucs = {}
    for c in range(labels.shape[1]):
        y = labels[:, c]
        tar = (y + y ** 2) / 2.0        # -1 -> 0, +1 -> 1
        mask = (y ** 2) > 0              # tested compounds only
        if mask.sum() == 0:
            continue
        yt = tar[mask]
        yp = preds[mask, c]
        if yt.sum() > 0 and (mask.sum() - yt.sum()) > 0:   # need both classes present
            aucs[f"assay_{c}"] = roc_auc_score(yt, yp)
    return aucs


per_fold = {}
fold_means = {}
for f in range(6):
    a = per_assay_auc_for_fold(f)
    per_fold[f] = a
    fold_means[f] = np.mean(list(a.values()))
    if f in REPORTED:
        match = "OK" if abs(fold_means[f] - REPORTED[f]) < 0.002 else "MISMATCH!"
        print(f"fold{f}: recomputed {fold_means[f]:.4f} | reported {REPORTED[f]:.4f}  [{match}]")
    else:
        print(f"fold{f}: recomputed {fold_means[f]:.4f}  (no reported value to check against)")

overall = np.mean(list(fold_means.values()))
overall_std = np.std(list(fold_means.values()), ddof=1)
print(f"\nrecomputed CV mean: {overall:.4f} +/- {overall_std:.4f}")

all_assays = sorted(set().union(*[set(per_fold[f].keys()) for f in range(6)]),
                    key=lambda s: int(s.split("_")[1]))
rows = {}
for assay in all_assays:
    rows[assay] = [per_fold[f].get(assay, np.nan) for f in range(6)]
df = pd.DataFrame.from_dict(rows, orient="index",
                            columns=[f"fold{f}" for f in range(6)])
df["mean"] = df[[f"fold{f}" for f in range(6)]].mean(axis=1)
df["std"] = df[[f"fold{f}" for f in range(6)]].std(axis=1)
df.index.name = ""
df.to_csv(os.path.join(OUT_DIR, "cv_per_assay.csv"))
print(f"\nwrote {OUT_DIR}/cv_per_assay.csv ({len(df)} assays)")
print(f"per-assay mean AUC across assays: {df['mean'].mean():.4f}")
