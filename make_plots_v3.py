#!/usr/bin/env python3
"""
Per-run plots for Project Bioactive.  Usage:  python make_plots_v3.py <model_name>

Differences from make_plots_v2.py:
  1. No hardcoded /mnt/ssd8/bioactive. Paths come from BIOACT_SAVE_DIR, so the
     same script works on the 4080 and inside a cluster pod.
  2. Test CSVs are read from  <save_dir>/results/<model_name>/  (where
     trainer.test() now writes them, via BIOACT_RESULTS_DIR) rather than from a
     single shared src/data/ file that concurrent CV folds would overwrite.
     Falls back to the old location if the new one is absent.
  3. NEW graph7: training vs validation loss, with the generalisation gap
     shaded. This is the over-/under-fitting diagnostic. It requires the
     'train_loss' field added to the metrics JSON by apply_cluster_fixes.py;
     if that field is missing (older runs) the graph is skipped with a warning.

Environment:
  BIOACT_SAVE_DIR   default /mnt/ssd8/bioactive/src/models
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---------- model selection ----------
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "bioact_resnet50_v2_faithful"
PRETTY = {
    "bioact_resnet50_v2_faithful": "ResNet50 (baseline)",
    "bioact_dinov2_base": "DINOv2-Base",
    "bioact_dinov2_large": "DINOv2-Large",
    "bioact_biomedclip": "BiomedCLIP",
    "bioact_clip_vitl14": "CLIP ViT-L/14",
    "bioact_convnext_base": "ConvNeXt-Base",
}.get(MODEL_NAME, MODEL_NAME)

# ---------- paths ----------
SAVE_DIR = os.environ.get("BIOACT_SAVE_DIR", "/mnt/ssd8/bioactive/src/models")
RESULTS_DIR = os.path.join(SAVE_DIR, "results", MODEL_NAME)
METRICS_JSON = os.path.join(SAVE_DIR, "checkpoints", f"{MODEL_NAME}_metrics.json")
OUTDIR = os.path.join(RESULTS_DIR, "plots")

# test CSVs: new per-model location first, legacy shared location second
LABELS_CSV = os.path.join(RESULTS_DIR, "test_labels.csv")
PREDS_CSV = os.path.join(RESULTS_DIR, "test_preds.csv")
if not os.path.exists(PREDS_CSV):
    legacy = "/mnt/ssd8/bioactive/src/data"
    lc, pc = os.path.join(legacy, "test_labels.csv"), os.path.join(legacy, "test_preds.csv")
    if os.path.exists(pc):
        print(f"NOTE: using legacy test CSVs in {legacy} (not per-model)")
        LABELS_CSV, PREDS_CSV = lc, pc
    else:
        sys.exit(f"ERROR: no test predictions found.\n  looked in {RESULTS_DIR}\n  and {legacy}")

# ---------- reference lines ----------
RESNET_BASELINE = 0.702   # our ResNet50 single-fold test
PAPER_AUC = 0.660         # paper public 6-fold CV mean
PAPER_STD = 0.094
os.makedirs(OUTDIR, exist_ok=True)

# ---------- load test data ----------
labs = pd.read_csv(LABELS_CSV, index_col=0)
preds = pd.read_csv(PREDS_CSV, index_col=0)
assays = [c for c in labs.columns if c.startswith("assay_")]
labs, preds = labs[assays], preds[assays]

# ---------- per-assay AUC + ROC data ----------
per_assay, roc_data = {}, {}
for a in assays:
    y, p = labs[a].values, preds[a].values
    mask = y != 0
    yk = (y[mask] == 1).astype(int)
    pk = p[mask]
    if yk.sum() > 0 and (yk == 0).sum() > 0:
        auc = roc_auc_score(yk, pk)
        per_assay[a] = auc
        fpr, tpr, _ = roc_curve(yk, pk)
        roc_data[a] = (fpr, tpr, auc, int(yk.sum()), int((yk == 0).sum()))

auc_series = pd.Series(per_assay).sort_values()
mean_auc = auc_series.mean()
print(f"{PRETTY}: mean test ROC-AUC = {mean_auc:.4f} over {len(auc_series)} assays")

# ---------- training metrics ----------
ep = ep_roc = ep_loss = ep_train_loss = None
if os.path.exists(METRICS_JSON):
    m = json.load(open(METRICS_JSON))
    ep = [e["epoch"] for e in m["epochs"]]
    ep_roc = [e["val_roc_auc"] for e in m["epochs"]]
    ep_loss = [e["val_loss"] for e in m["epochs"]]
    tl = [e.get("train_loss") for e in m["epochs"]]
    ep_train_loss = tl if any(v is not None for v in tl) else None
    if ep_train_loss is None:
        print("WARNING: metrics JSON has no 'train_loss' -- graph7 skipped. "
              "Was apply_cluster_fixes.py run before this training?")
else:
    print("WARNING: metrics JSON not found, skipping training curves.")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2E5E8C", "#E08A1E", "#3C8C5A", "#C0392B", "#888888", "#6A4C93"


def add_ref_lines_v(ax):
    ax.axvline(RESNET_BASELINE, ls="-", color=PURPLE, lw=2, label=f"ResNet50 baseline {RESNET_BASELINE}")
    ax.axvline(PAPER_AUC, ls="--", color=RED, lw=1.5, label=f"Paper 6-fold CV {PAPER_AUC}")
    ax.axvspan(PAPER_AUC - PAPER_STD, PAPER_AUC + PAPER_STD, color=RED, alpha=0.07)
    ax.axvline(mean_auc, ls="-", color=ORANGE, lw=2, label=f"{PRETTY} {mean_auc:.3f}")
    ax.axvline(0.5, ls=":", color=GREY, lw=1, label="Chance 0.5")


# ---------- graph1: training curve ----------
if ep:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(ep, ep_roc, "-o", color=BLUE, lw=2, label="Val ROC-AUC")
    ax1.axhline(RESNET_BASELINE, ls="-", color=PURPLE, lw=1.5, label=f"ResNet50 baseline {RESNET_BASELINE}")
    ax1.axhline(PAPER_AUC, ls="--", color=RED, lw=1.5, label=f"Paper 6-fold CV {PAPER_AUC}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val ROC-AUC", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0.5, 0.78)
    ax2 = ax1.twinx()
    ax2.plot(ep, ep_loss, "-s", color=ORANGE, lw=1.5, alpha=0.6, label="Val loss")
    ax2.set_ylabel("Val loss", color=ORANGE); ax2.tick_params(axis="y", labelcolor=ORANGE); ax2.grid(False)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="lower right", fontsize=8)
    plt.title(f"Training Curve — {PRETTY}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph1_training_curve.png"); plt.close()

# ---------- graph2: per-assay ROC-AUC bars ----------
fig, ax = plt.subplots(figsize=(10, 7.5))
colors = [GREEN if v >= RESNET_BASELINE else (BLUE if v >= 0.6 else RED) for v in auc_series.values]
ax.barh(range(len(auc_series)), auc_series.values, color=colors, edgecolor="black", lw=0.4)
ax.set_yticks(range(len(auc_series))); ax.set_yticklabels(auc_series.index, fontsize=7)
add_ref_lines_v(ax)
ax.set_xlabel("Test ROC-AUC"); ax.set_xlim(0, 1.05)
ax.set_title(f"Per-Assay Test ROC-AUC — {PRETTY} ({len(auc_series)} assays)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph2_per_assay_roc.png"); plt.close()

# ---------- graph3: distribution ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))
axa.hist(auc_series.values, bins=12, color=BLUE, edgecolor="black", alpha=0.8)
axa.axvline(mean_auc, color=ORANGE, lw=2, label=f"{PRETTY} {mean_auc:.3f}")
axa.axvline(RESNET_BASELINE, color=PURPLE, lw=2, label=f"ResNet50 {RESNET_BASELINE}")
axa.axvline(PAPER_AUC, color=RED, ls="--", lw=1.5, label=f"Paper {PAPER_AUC}")
axa.set_xlabel("ROC-AUC"); axa.set_ylabel("# assays"); axa.set_title("AUC Histogram"); axa.legend(fontsize=8)
bp = axb.boxplot(auc_series.values, patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor(BLUE); bp["boxes"][0].set_alpha(0.6)
axb.axhline(RESNET_BASELINE, color=PURPLE, lw=2, label=f"ResNet50 {RESNET_BASELINE}")
axb.axhline(PAPER_AUC, color=RED, ls="--", lw=1.5, label=f"Paper {PAPER_AUC}")
axb.scatter(np.random.normal(1, 0.04, len(auc_series)), auc_series.values, color=ORANGE, s=18, alpha=0.7, zorder=3)
axb.set_ylabel("ROC-AUC"); axb.set_xticks([]); axb.set_title("AUC Spread"); axb.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph3_roc_distribution.png"); plt.close()

# ---------- graph4: ROC curves (best/median/worst) ----------
ordered = sorted(roc_data.items(), key=lambda kv: kv[1][2])
worst, median, best = ordered[0], ordered[len(ordered) // 2], ordered[-1]
fig, ax = plt.subplots(figsize=(7, 7))
for (name, (fpr, tpr, auc, npos, nneg)), col, lab in [
        (best, GREEN, "Best"), (median, BLUE, "Median"), (worst, RED, "Worst")]:
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{lab}: {name} (AUC={auc:.3f}, +{npos}/-{nneg})")
ax.plot([0, 1], [0, 1], ls=":", color=GREY, label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — {PRETTY} (mean {mean_auc:.3f})")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph4_roc_curves.png"); plt.close()

# ---------- graph5: score distributions ----------
act_scores, inact_scores = [], []
for a in assays:
    y, p = labs[a].values, preds[a].values
    act_scores += list(p[y == 1]); inact_scores += list(p[y == -1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(inact_scores, bins=40, color=RED, alpha=0.55, density=True, label=f"Inactive (n={len(inact_scores)})")
ax.hist(act_scores, bins=40, color=GREEN, alpha=0.55, density=True, label=f"Active (n={len(act_scores)})")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
ax.set_title(f"Score Distributions — {PRETTY} (actives vs inactives)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph5_score_distributions.png"); plt.close()

# ---------- graph6: summary panel ----------
fig, ax = plt.subplots(figsize=(9, 5.5)); ax.axis("off")
delta = mean_auc - RESNET_BASELINE
lines = [
    f"{PRETTY} — Test Results",
    "",
    f"Mean test ROC-AUC:        {mean_auc:.4f}",
    f"ResNet50 baseline (ours): {RESNET_BASELINE}   (delta {delta:+.3f})",
    f"Paper public (6-fold CV): {PAPER_AUC} +/- {PAPER_STD}",
    f"Assays evaluated:         {len(auc_series)}",
    f"Assays >= ResNet (0.702): {(auc_series >= RESNET_BASELINE).sum()} / {len(auc_series)}",
    f"Assays >= 0.80:           {(auc_series >= 0.8).sum()} / {len(auc_series)}",
    f"Assays <  0.50:           {(auc_series < 0.5).sum()} / {len(auc_series)}",
    f"Best assay:               {auc_series.idxmax()} = {auc_series.max():.3f}",
    f"Worst assay:              {auc_series.idxmin()} = {auc_series.min():.3f}",
    f"Beats ResNet baseline:    {'YES' if mean_auc > RESNET_BASELINE else 'NO'}",
]
ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=12,
        bbox=dict(boxstyle="round", fc="#EEF3F8", ec=BLUE))
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph6_summary.png"); plt.close()

# ---------- graph7: train vs val loss (over/under-fitting) ----------
if ep and ep_train_loss:
    tr = np.array([np.nan if v is None else v for v in ep_train_loss], dtype=float)
    va = np.array(ep_loss, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ep, tr, "-o", color=GREEN, lw=2, label="Train loss")
    ax.plot(ep, va, "-s", color=ORANGE, lw=2, label="Val loss")
    ax.fill_between(ep, tr, va, where=~np.isnan(tr), color=RED, alpha=0.10,
                    label="Generalisation gap")

    # a plain-language read of the curves, stated as evidence not verdict
    valid = ~np.isnan(tr)
    note = "insufficient points to judge"
    if valid.sum() >= 3:
        t, v = tr[valid], va[valid]
        gap = v - t
        # how much of the initial training loss was removed
        train_drop = (t[0] - t[-1]) / t[0] if t[0] else 0.0
        # gap must be both widening and non-trivial relative to the loss scale
        gap_widening = gap[-1] > gap[0] * 1.5 and gap[-1] > 0.05 * abs(v[-1])
        val_rising = v[-1] > np.nanmin(v) * 1.02

        if train_drop < 0.20:
            note = f"train loss fell only {train_drop:.0%} -> underfitting / LR too low"
        elif val_rising and gap_widening:
            note = "val loss rising while gap widens -> overfitting"
        elif gap_widening:
            note = "gap widening, val loss still falling -> watch for overfitting"
        else:
            note = "train and val fall together -> fitting healthily"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, family="monospace",
            va="bottom", bbox=dict(boxstyle="round", fc="#F5F5F5", ec=GREY))

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Train vs Validation Loss — {PRETTY}")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph7_train_val_loss.png"); plt.close()

# ---------- per-assay CSV ----------
auc_series.sort_values(ascending=False).to_csv(f"{OUTDIR}/per_assay_auc.csv", header=["test_roc_auc"])
n = 7 if (ep and ep_train_loss) else 6
print(f"Saved {n} graphs + per_assay_auc.csv to: {OUTDIR}")