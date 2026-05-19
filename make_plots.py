"""
make_plots.py — Generate the full test-results figure pack for
ResNet50 v2 (paper-faithful) on the JUMP-CP + ChEMBL 29-assay dataset.

Reads:
  - test_labels.csv   (true labels: +1 active, -1 inactive, 0 unknown)
  - test_preds.csv    (predicted probabilities, per compound x assay)
  - <model>_metrics.json  (training curve: val ROC-AUC + val loss per epoch)

Writes 6 PNGs + a per-assay CSV into:
  src/models/results/bioact_resnet50_v2_faithful/plots/

Run on the server:  python3 make_plots.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---------- paths ----------
LABELS_CSV   = "/mnt/ssd8/bioactive/src/data/test_labels.csv"
PREDS_CSV    = "/mnt/ssd8/bioactive/src/data/test_preds.csv"
METRICS_JSON = "/mnt/ssd8/bioactive/src/models/checkpoints/bioact_resnet50_v2_faithful_metrics.json"
OUTDIR       = "/mnt/ssd8/bioactive/src/models/results/bioact_resnet50_v2_faithful/plots"
PAPER_AUC    = 0.660
PAPER_STD    = 0.094
os.makedirs(OUTDIR, exist_ok=True)

# ---------- load test data ----------
labs  = pd.read_csv(LABELS_CSV, index_col=0)
preds = pd.read_csv(PREDS_CSV,  index_col=0)
assays = [c for c in labs.columns if c.startswith("assay_")]
labs, preds = labs[assays], preds[assays]

# ---------- per-assay AUC + ROC data (known labels only; +1 active, -1 inactive) ----------
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
print(f"Mean test ROC-AUC: {mean_auc:.4f} over {len(auc_series)} assays")

# ---------- training metrics ----------
ep = ep_roc = ep_loss = None
if os.path.exists(METRICS_JSON):
    m = json.load(open(METRICS_JSON))
    ep      = [e["epoch"] for e in m["epochs"]]
    ep_roc  = [e["val_roc_auc"] for e in m["epochs"]]
    ep_loss = [e["val_loss"] for e in m["epochs"]]
else:
    print("WARNING: metrics JSON not found, skipping graph1 training curve.")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE, GREEN, RED, GREY = "#2E5E8C", "#E08A1E", "#3C8C5A", "#C0392B", "#888888"

# ---------- graph1: training curve ----------
if ep:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(ep, ep_roc, "-o", color=BLUE, lw=2, label="Val ROC-AUC")
    ax1.axhline(PAPER_AUC, ls="--", color=RED, lw=1.5, label=f"Paper public {PAPER_AUC}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val ROC-AUC", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0.5, 0.75)
    ax2 = ax1.twinx(); ax2.plot(ep, ep_loss, "-s", color=ORANGE, lw=1.5, alpha=0.7, label="Val loss")
    ax2.set_ylabel("Val loss", color=ORANGE); ax2.tick_params(axis="y", labelcolor=ORANGE); ax2.grid(False)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="lower right", fontsize=9)
    plt.title("Training Curve — ResNet50 v2 (paper-faithful)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph1_training_curve.png"); plt.close()

# ---------- graph2: per-assay ROC-AUC bars ----------
fig, ax = plt.subplots(figsize=(10, 7))
colors = [GREEN if v >= 0.7 else (BLUE if v >= 0.6 else RED) for v in auc_series.values]
ax.barh(range(len(auc_series)), auc_series.values, color=colors, edgecolor="black", lw=0.4)
ax.set_yticks(range(len(auc_series))); ax.set_yticklabels(auc_series.index, fontsize=7)
ax.axvline(PAPER_AUC, ls="--", color="black", lw=1.5, label=f"Paper {PAPER_AUC}")
ax.axvline(mean_auc, ls="-", color=ORANGE, lw=2, label=f"Our mean {mean_auc:.3f}")
ax.axvline(0.5, ls=":", color=GREY, lw=1, label="Chance 0.5")
ax.set_xlabel("Test ROC-AUC"); ax.set_xlim(0, 1.05)
ax.set_title(f"Per-Assay Test ROC-AUC ({len(auc_series)} assays)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph2_per_assay_roc.png"); plt.close()

# ---------- graph3: distribution ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))
axa.hist(auc_series.values, bins=12, color=BLUE, edgecolor="black", alpha=0.8)
axa.axvline(mean_auc, color=ORANGE, lw=2, label=f"Mean {mean_auc:.3f}")
axa.axvline(PAPER_AUC, color=RED, ls="--", lw=1.5, label=f"Paper {PAPER_AUC}")
axa.set_xlabel("ROC-AUC"); axa.set_ylabel("# assays"); axa.set_title("AUC Histogram"); axa.legend(fontsize=8)
bp = axb.boxplot(auc_series.values, vert=True, patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor(BLUE); bp["boxes"][0].set_alpha(0.6)
axb.axhline(PAPER_AUC, color=RED, ls="--", lw=1.5, label=f"Paper {PAPER_AUC}")
axb.scatter(np.random.normal(1, 0.04, len(auc_series)), auc_series.values, color=ORANGE, s=18, alpha=0.7, zorder=3)
axb.set_ylabel("ROC-AUC"); axb.set_xticks([]); axb.set_title("AUC Spread"); axb.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph3_roc_distribution.png"); plt.close()

# ---------- graph4: real ROC curves (best/median/worst) ----------
ordered = sorted(roc_data.items(), key=lambda kv: kv[1][2])
worst, median, best = ordered[0], ordered[len(ordered) // 2], ordered[-1]
fig, ax = plt.subplots(figsize=(7, 7))
for (name, (fpr, tpr, auc, npos, nneg)), col, lab in [
        (best, GREEN, "Best"), (median, BLUE, "Median"), (worst, RED, "Worst")]:
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{lab}: {name} (AUC={auc:.3f}, +{npos}/-{nneg})")
ax.plot([0, 1], [0, 1], ls=":", color=GREY, label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — best/median/worst (overall mean {mean_auc:.3f})")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph4_roc_curves.png"); plt.close()

# ---------- graph5: score distributions (actives vs inactives, pooled) ----------
act_scores, inact_scores = [], []
for a in assays:
    y, p = labs[a].values, preds[a].values
    act_scores += list(p[y == 1]); inact_scores += list(p[y == -1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(inact_scores, bins=40, color=RED, alpha=0.55, density=True, label=f"Inactive (n={len(inact_scores)})")
ax.hist(act_scores, bins=40, color=GREEN, alpha=0.55, density=True, label=f"Active (n={len(act_scores)})")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
ax.set_title("Score Distributions — actives vs inactives (all assays pooled)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph5_score_distributions.png"); plt.close()

# ---------- graph6: summary panel ----------
fig, ax = plt.subplots(figsize=(9, 5.5)); ax.axis("off")
lines = [
    "ResNet50 v2 (paper-faithful) - Test Results",
    "",
    f"Mean test ROC-AUC:        {mean_auc:.4f}",
    f"Paper public benchmark:   {PAPER_AUC} +/- {PAPER_STD}",
    f"Assays evaluated:         {len(auc_series)}",
    f"Assays >= 0.70:           {(auc_series >= 0.7).sum()} / {len(auc_series)}",
    f"Assays >= 0.80:           {(auc_series >= 0.8).sum()} / {len(auc_series)}",
    f"Assays <  0.50:           {(auc_series < 0.5).sum()} / {len(auc_series)}",
    f"Best assay:               {auc_series.idxmax()} = {auc_series.max():.3f}",
    f"Worst assay:              {auc_series.idxmin()} = {auc_series.min():.3f}",
    f"Within paper band:        {'YES' if abs(mean_auc - PAPER_AUC) <= PAPER_STD else 'NO'}",
]
ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=12,
        bbox=dict(boxstyle="round", fc="#EEF3F8", ec=BLUE))
plt.tight_layout(); plt.savefig(f"{OUTDIR}/graph6_summary.png"); plt.close()

# ---------- per-assay CSV ----------
auc_series.sort_values(ascending=False).to_csv(f"{OUTDIR}/per_assay_auc.csv", header=["test_roc_auc"])
print("Saved 6 graphs + per_assay_auc.csv to:", OUTDIR)