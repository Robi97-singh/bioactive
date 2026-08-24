#!/usr/bin/env python3
"""Aggregate the 6 CV folds into mean +/- std of test ROC-AUC, plus graphs."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = "/shared/ssd/logs/b-r-singh1"
FOLDS = [0, 1, 2, 3, 4, 5]
PAPER_MEAN, PAPER_STD = 0.660, 0.094

BLUE = "#2E6DA4"
NAVY = "#1F3B5B"
GREY = "#888888"


def fold_name(model, res, fold):
    return f"bioact_{model}_r{res}_fold{fold}"


def load_per_assay(save_dir, model, res, fold):
    name = fold_name(model, res, fold)
    p = os.path.join(save_dir, "results", name, "plots", "per_assay_auc.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0)
    col = df.columns[0]
    return df[col].rename(f"fold{fold}")


def load_metrics(save_dir, model, res, fold):
    name = fold_name(model, res, fold)
    p = os.path.join(save_dir, "checkpoints", f"{name}_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("res", type=int)
    ap.add_argument("--save_dir", default=SAVE_DIR)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(
        args.save_dir, "results", "cv", f"{args.model}_r{args.res}")
    os.makedirs(outdir, exist_ok=True)

    series, missing = [], []
    for f in FOLDS:
        s = load_per_assay(args.save_dir, args.model, args.res, f)
        if s is None:
            missing.append(f)
        else:
            series.append(s)

    if missing:
        print(f"WARNING: no test results for fold(s) {missing}.")
        print("         Aggregating over the remaining folds only.\n")
    if not series:
        sys.exit("No per-assay results found at all. Nothing to aggregate.")

    per_assay = pd.concat(series, axis=1)
    fold_means = per_assay.mean(axis=0)
    cv_mean = fold_means.mean()
    cv_std = fold_means.std(ddof=1) if len(fold_means) > 1 else float("nan")

    rows = [{"fold": c.replace("fold", ""), "mean_test_roc_auc": v,
             "n_assays": per_assay[c].notna().sum()}
            for c, v in fold_means.items()]
    rows.append({"fold": "MEAN", "mean_test_roc_auc": cv_mean, "n_assays": per_assay.shape[0]})
    rows.append({"fold": "STD", "mean_test_roc_auc": cv_std, "n_assays": per_assay.shape[0]})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "cv_summary.csv"), index=False)

    per_assay["mean"] = per_assay[[c for c in per_assay if c.startswith("fold")]].mean(axis=1)
    per_assay["std"] = per_assay[[c for c in per_assay if c.startswith("fold")]].std(axis=1, ddof=1)
    per_assay.sort_values("mean", ascending=False).to_csv(os.path.join(outdir, "cv_per_assay.csv"))

    print(f"{args.model} @ {args.res}  --  {len(series)}-fold cross-validation")
    print("-" * 56)
    for c, v in fold_means.items():
        print(f"  {c:<10} mean test ROC-AUC = {v:.4f}")
    print("-" * 56)
    print(f"  CV result: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  paper (public JUMP-CP): {PAPER_MEAN:.3f} +/- {PAPER_STD:.3f}")
    print()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = range(len(fold_means))
    ax.bar(xs, fold_means.values, color=BLUE, width=0.6, zorder=3)
    ax.axhline(cv_mean, color=NAVY, lw=1.6, zorder=4, label=f"CV mean {cv_mean:.3f}")
    if not np.isnan(cv_std):
        ax.axhspan(cv_mean - cv_std, cv_mean + cv_std, color=NAVY, alpha=0.12, zorder=2,
                   label=f"+/- 1 std ({cv_std:.3f})")
    ax.axhline(PAPER_MEAN, color=GREY, ls="--", lw=1.4, zorder=4, label=f"paper public {PAPER_MEAN:.3f}")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c.replace("fold", "fold ") for c in fold_means.index])
    ax.set_ylabel("mean test ROC-AUC")
    ax.set_ylim(0.5, max(0.80, fold_means.max() + 0.05))
    ax.set_title(f"{args.model} @ {args.res} -- per-fold test performance")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cv_fold_spread.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    any_curve = False
    for f in FOLDS:
        d = load_metrics(args.save_dir, args.model, args.res, f)
        if not d or not d.get("epochs"):
            continue
        eps = [e["epoch"] for e in d["epochs"]]
        vals = [e["val_roc_auc"] for e in d["epochs"]]
        ax.plot(eps, vals, marker="o", ms=3, lw=1.2, label=f"fold {f}")
        any_curve = True
    if any_curve:
        ax.axhline(PAPER_MEAN, color=GREY, ls="--", lw=1.2, label=f"paper public {PAPER_MEAN:.3f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation ROC-AUC")
        ax.set_title(f"{args.model} @ {args.res} -- validation trajectories")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "cv_val_curves.png"), dpi=150)
    plt.close(fig)

    pa = per_assay.sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.22 * len(pa))))
    ax.barh(range(len(pa)), pa["mean"].values, xerr=pa["std"].values,
            color=BLUE, ecolor=GREY, capsize=2, height=0.65, zorder=3)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2, zorder=4)
    ax.axvline(cv_mean, color=NAVY, lw=1.4, zorder=4, label=f"overall mean {cv_mean:.3f}")
    ax.set_yticks(range(len(pa)))
    ax.set_yticklabels(pa.index, fontsize=7)
    ax.set_xlabel("test ROC-AUC (mean over folds, error bar = std)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(f"{args.model} @ {args.res} -- per-assay performance")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cv_per_assay.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote to {outdir}:")
    for f in sorted(os.listdir(outdir)):
        print("  ", f)


if __name__ == "__main__":
    main()
