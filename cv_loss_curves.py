#!/usr/bin/env python3
"""Combined training-vs-validation loss across all 6 CV folds, in one figure."""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = "/shared/ssd/logs/b-r-singh1"
FOLDS = [0, 1, 2, 3, 4, 5]

GREEN = "#3C8C5A"
ORANGE = "#E08A1E"
GREY = "#888888"


def load_fold(save_dir, model, res, fold):
    p = os.path.join(save_dir, "checkpoints",
                     f"bioact_{model}_r{res}_fold{fold}_metrics.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    eps, tr, va = [], [], []
    for e in d.get("epochs", []):
        t = e.get("train_loss")
        v = e.get("val_loss")
        if t is None or v is None:
            continue
        eps.append(e["epoch"]); tr.append(t); va.append(v)
    if not eps:
        return None
    return np.array(eps), np.array(tr), np.array(va)


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

    folds = {}
    for f in FOLDS:
        r = load_fold(args.save_dir, args.model, args.res, f)
        if r is not None:
            folds[f] = r
    if not folds:
        sys.exit("No fold metrics with train_loss found. "
                 "Were these trained after the train_loss patch?")

    print(f"{args.model} @ {args.res}: found loss curves for folds {sorted(folds)}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (f, (eps, tr, va)) in enumerate(sorted(folds.items())):
        ax.plot(eps, tr, "-", color=GREEN, alpha=0.45, lw=1.3,
                label="train (per fold)" if i == 0 else None)
        ax.plot(eps, va, "-", color=ORANGE, alpha=0.45, lw=1.3,
                label="val (per fold)" if i == 0 else None)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{args.model} @ {args.res} -- train vs val loss, all {len(folds)} folds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cv_loss_overlay.png"), dpi=150)
    plt.close(fig)

    all_eps = sorted(set(int(e) for (eps, _, _) in folds.values() for e in eps))
    tr_mean, tr_std, va_mean, va_std, xs = [], [], [], [], []
    for ep in all_eps:
        trs = [tr[eps == ep][0] for (eps, tr, va) in folds.values() if ep in eps]
        vas = [va[eps == ep][0] for (eps, tr, va) in folds.values() if ep in eps]
        if not trs:
            continue
        xs.append(ep)
        tr_mean.append(np.mean(trs)); tr_std.append(np.std(trs, ddof=0))
        va_mean.append(np.mean(vas)); va_std.append(np.std(vas, ddof=0))
    xs = np.array(xs)
    tr_mean, tr_std = np.array(tr_mean), np.array(tr_std)
    va_mean, va_std = np.array(va_mean), np.array(va_std)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, tr_mean, "-o", color=GREEN, lw=2, ms=3, label="train loss (mean)")
    ax.fill_between(xs, tr_mean - tr_std, tr_mean + tr_std, color=GREEN, alpha=0.15)
    ax.plot(xs, va_mean, "-s", color=ORANGE, lw=2, ms=3, label="val loss (mean)")
    ax.fill_between(xs, va_mean - va_std, va_mean + va_std, color=ORANGE, alpha=0.15)
    ax.fill_between(xs, tr_mean, va_mean, where=(va_mean >= tr_mean),
                    color=GREY, alpha=0.10, label="generalisation gap")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{args.model} @ {args.res} -- train vs val loss "
                 f"(mean of {len(folds)} folds, band = +/-1 std)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cv_loss_meanband.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote to {outdir}:")
    print("   cv_loss_overlay.png   (all folds' curves)")
    print("   cv_loss_meanband.png  (mean +/- std across folds)")


if __name__ == "__main__":
    main()
