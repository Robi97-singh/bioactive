#!/usr/bin/env python3
"""plot_cv_extra.py -- cv_fold_spread.png and cv_per_assay.png for result folders written by cv_rescore.py (same style as aggregate_cv.py).
Usage: python3 plot_cv_extra.py DIR TITLE [DIR TITLE ...]   e.g.  results_showcase/resnet18/resnet18_r448 "resnet18 @ 448" """
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_MEAN = 0.660
BLUE, NAVY, GREY = "#2E6DA4", "#1F3B5B", "#888888"

for d, title in zip(sys.argv[1::2], sys.argv[2::2]):
    S = pd.read_csv(os.path.join(d, "cv_summary.csv"), dtype={"fold": str})
    S = S[S.fold.str.isdigit()]
    fold_means = pd.Series(S.mean_test_roc_auc.values, index=["fold " + f for f in S.fold])
    cv_mean, cv_std = fold_means.mean(), fold_means.std(ddof=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = range(len(fold_means))
    ax.bar(xs, fold_means.values, color=BLUE, width=0.6, zorder=3)
    ax.axhline(cv_mean, color=NAVY, lw=1.6, zorder=4, label=f"CV mean {cv_mean:.3f}")
    ax.axhspan(cv_mean - cv_std, cv_mean + cv_std, color=NAVY, alpha=0.12, zorder=2, label=f"+/- 1 std ({cv_std:.3f})")
    ax.axhline(PAPER_MEAN, color=GREY, ls="--", lw=1.4, zorder=4, label=f"paper public {PAPER_MEAN:.3f}")
    ax.set_xticks(list(xs)); ax.set_xticklabels(fold_means.index)
    ax.set_ylabel("mean test ROC-AUC"); ax.set_ylim(0.5, max(0.80, fold_means.max() + 0.05))
    ax.set_title(f"{title} -- per-fold test performance")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(d, "cv_fold_spread.png"), dpi=150); plt.close(fig)

    pa = pd.read_csv(os.path.join(d, "cv_per_assay.csv"), index_col=0)
    cols = [c for c in pa if c.startswith("fold")]
    pa["mean"] = pa[cols].mean(axis=1); pa["std"] = pa[cols].std(axis=1, ddof=1)
    pa = pa.sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.22 * len(pa))))
    ax.barh(range(len(pa)), pa["mean"].values, xerr=pa["std"].values, color=BLUE, ecolor=GREY, capsize=2, height=0.65, zorder=3)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2, zorder=4)
    ax.axvline(cv_mean, color=NAVY, lw=1.4, zorder=4, label=f"overall mean {cv_mean:.3f}")
    ax.set_yticks(range(len(pa))); ax.set_yticklabels(pa.index, fontsize=7)
    ax.set_xlabel("test ROC-AUC (mean over folds, error bar = std)"); ax.set_xlim(0.4, 1.0)
    ax.set_title(f"{title} -- per-assay performance")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(d, "cv_per_assay.png"), dpi=150); plt.close(fig)
    print(title, "| CV mean %.4f std %.4f | wrote cv_fold_spread.png, cv_per_assay.png in %s" % (cv_mean, cv_std, d))
