#!/usr/bin/env python3
"""
plot_head_curves.py

Plot the val ROC-AUC learning curves for a frozen linear-probe arm, reading the
per-epoch history saved by train_head.py in each fold's
{save_dir}/checkpoints/bioact_{model}_r{res}_fold{F}_head_metrics.json.

For a linear probe the *loss* curve is nearly flat and uninformative (the head
starts near-converged on frozen features). The val ROC-AUC curve is the
meaningful learning signal — it climbs and plateaus — so that is what we plot.

Outputs two PNGs to --outdir:
  head_val_auc_overlay.png   — one line per fold
  head_val_auc_meanband.png  — mean across folds with ±1 std band

Usage (in a pod; needs matplotlib + numpy):
  python plot_head_curves.py celldino 224 \
      --save_dir /shared/ssd/logs/b-r-singh1 \
      --outdir /shared/ssd/logs/b-r-singh1/results/cv/celldino_r224
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("res", type=int)
    ap.add_argument("--save_dir", default="/shared/ssd/logs/b-r-singh1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()

    ckpt = os.path.join(a.save_dir, "checkpoints")
    outdir = a.outdir or os.path.join(a.save_dir, "results", "cv",
                                      f"{a.model}_r{a.res}")
    os.makedirs(outdir, exist_ok=True)

    curves = {}   # fold -> (epochs, val_auc)
    for f in a.folds:
        p = os.path.join(ckpt, f"bioact_{a.model}_r{a.res}_fold{f}_head_metrics.json")
        if not os.path.exists(p):
            print(f"  fold {f}: MISSING {p}")
            continue
        d = json.load(open(p))
        hist = d.get("history", [])
        if not hist:
            print(f"  fold {f}: no history array in {p}")
            continue
        ep = np.array([h["epoch"] for h in hist])
        va = np.array([h["val_roc_auc"] for h in hist])
        curves[f] = (ep, va)
        print(f"  fold {f}: {len(ep)} epochs, best val {va.max():.4f}")

    if not curves:
        raise SystemExit("No head_metrics history found — nothing to plot.")

    # ---- overlay: one line per fold ----
    plt.figure(figsize=(7, 4.5))
    for f, (ep, va) in sorted(curves.items()):
        plt.plot(ep, va, linewidth=1.4, alpha=0.85, label=f"fold {f}")
    plt.xlabel("epoch")
    plt.ylabel("validation ROC-AUC")
    plt.title(f"{a.model} r{a.res} frozen linear probe — val ROC-AUC by fold")
    plt.legend(fontsize=8, ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    o1 = os.path.join(outdir, "head_val_auc_overlay.png")
    plt.savefig(o1, dpi=130)
    plt.close()
    print(f"  wrote {o1}")

    # ---- mean +/- std band (align on shortest common epoch range) ----
    min_len = min(len(va) for _, va in curves.values())
    stack = np.vstack([va[:min_len] for _, va in curves.values()])
    ep = np.arange(1, min_len + 1)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, mean, color="#1F3864", linewidth=2, label="mean val ROC-AUC")
    plt.fill_between(ep, mean - std, mean + std, color="#1F3864", alpha=0.18,
                     label="±1 std")
    plt.xlabel("epoch")
    plt.ylabel("validation ROC-AUC")
    plt.title(f"{a.model} r{a.res} frozen linear probe — mean val ROC-AUC "
              f"(n={stack.shape[0]} folds)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    o2 = os.path.join(outdir, "head_val_auc_meanband.png")
    plt.savefig(o2, dpi=130)
    plt.close()
    print(f"  wrote {o2}")


if __name__ == "__main__":
    main()
