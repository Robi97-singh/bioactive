#!/usr/bin/env python3
"""
make_pr_calibration_pareto.py

Three additional benchmark figures, built entirely from already-saved
per-sample test predictions (no retraining, no re-inference):

  fig_F_pr_curves.png          -- Precision-Recall overlay, all arms (224)
  fig_G_calibration.png        -- Reliability / calibration overlay, all arms
  fig_H_pareto_efficiency.png  -- Test ROC-AUC vs. % trainable parameters

For each arm, all 6 folds' test_preds.csv / test_labels.csv are pooled
(masked entries, label == -1, excluded) into one flat set of
(y_true, y_score) pairs, then a single micro-averaged curve is computed
per arm. This matches the same "pool across folds and assays" approach
implicit in the project's existing mean_roc_auc metric.

Usage (in a pod with pandas/numpy/matplotlib/scikit-learn):
    python make_pr_calibration_pareto.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

RESULTS = os.environ.get("BIOACT_RESULTS", "/shared/ssd/logs/b-r-singh1/results")
OUTDIR = os.environ.get("BIOACT_FIGDIR", os.path.join(RESULTS, "figures"))
os.makedirs(OUTDIR, exist_ok=True)

FOLDS = [0, 1, 2, 3, 4, 5]

# (results_dir_stem, pretty_label, color, path_style)
# path_style: "plots" -> {stem}_fold{F}/plots/test_preds.csv  (train_head.py arms)
#             "direct" -> {stem}_fold{F}/test_preds.csv        (classification.py arms)
ARMS = [
    ("bioact_celldino_r224",   "Cell-DINO",             "#2E6FB0", "plots"),
    ("bioact_dino_r224",       "DINOv2-Base",           "#5BA3D0", "plots"),
    ("bioact_dino_large_r224", "DINOv2-Large",          "#1A4A7A", "plots"),
    ("bioact_biomedclip_r224", "BiomedCLIP",            "#8FC3E0", "plots"),
    ("bioact_lora_vit_s_r224", "DINOv2+LoRA",           "#7B5EA7", "direct"),
    ("bioact_resnet_r224",     "ResNet50 (fine-tuned)", "#C0603A", "direct"),
]


def load_pooled(stem, path_style):
    """Load and pool (y_true, y_score) across all 6 folds, masking label==-1."""
    all_y, all_s = [], []
    for f in FOLDS:
        if path_style == "plots":
            base = os.path.join(RESULTS, f"{stem}_fold{f}", "plots")
        else:
            base = os.path.join(RESULTS, f"{stem}_fold{f}")
        pred_path = os.path.join(base, "test_preds.csv")
        label_path = os.path.join(base, "test_labels.csv")
        if not (os.path.exists(pred_path) and os.path.exists(label_path)):
            print(f"  MISSING fold {f} for {stem} -- skipping this fold")
            continue
        preds_df = pd.read_csv(pred_path)
        labels_df = pd.read_csv(label_path)
        assay_cols = [c for c in preds_df.columns if c.startswith("assay_")]
        preds = preds_df[assay_cols].values.astype(float)
        labels = labels_df[assay_cols].values.astype(float)
        mask = labels != -1.0
        all_y.append(labels[mask])
        all_s.append(preds[mask])
    if not all_y:
        return None, None
    return np.concatenate(all_y), np.concatenate(all_s)


# ---------- load everything once ----------
pooled = {}
for stem, label, color, style in ARMS:
    y, s = load_pooled(stem, style)
    if y is None:
        print(f"SKIP {label}: no data found")
        continue
    pooled[label] = (y, s, color)
    print(f"{label}: pooled {len(y)} scored (sample, assay) pairs")


# ================= FIG F: Precision-Recall overlay =================
fig, ax = plt.subplots(figsize=(7.5, 6))
for stem, label, color, style in ARMS:
    if label not in pooled:
        continue
    y, s, c = pooled[label]
    precision, recall, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax.plot(recall, precision, color=c, lw=2, label=f"{label} (AP={ap:.3f})")
base_rate = np.mean([pooled[l][0].mean() for l in pooled])
ax.axhline(base_rate, ls="--", color="grey", lw=1, label=f"random baseline (AP={base_rate:.3f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall curves, pooled across folds and assays\n(JUMP-CP source_11, 224 resolution)", fontsize=12)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_F_pr_curves.png"), dpi=150)
plt.close()
print("wrote fig_F_pr_curves.png")


# ================= FIG G: Calibration overlay =================
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfectly calibrated")
for stem, label, color, style in ARMS:
    if label not in pooled:
        continue
    y, s, c = pooled[label]
    frac_pos, mean_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", ms=4, color=c, lw=1.5, label=label)
ax.set_xlabel("Mean predicted probability (per bin)", fontsize=11)
ax.set_ylabel("Observed fraction positive (per bin)", fontsize=11)
ax.set_title("Calibration (reliability), pooled across folds and assays\n(quantile-binned, 10 bins)", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_G_calibration.png"), dpi=150)
plt.close()
print("wrote fig_G_calibration.png")


# ================= FIG H: Pareto efficiency (ROC-AUC vs % trainable params) =================
# (mean_test_roc_auc, pct_trainable_params, group)
# Frozen-probe arms: head-only Linear(D,29), computed exactly from each backbone's
# known embedding dim and total param count.
def head_params(dim, n_classes=29):
    return dim * n_classes + n_classes

# (label, auc, pct_trainable, group, label_offset_xy)
# Offsets are hand-tuned so the three closely-clustered frozen-probe points
# (DINOv2-Base, DINOv2-Large, BiomedCLIP -- all near 0.01-0.03% trainable)
# don't overlap each other or their own markers.
PARETO = [
    ("Cell-DINO",             0.5932, 100 * head_params(384)  / 21_500_000, "frozen",    (10, 4)),
    ("DINOv2-Base",           0.5796, 100 * head_params(768)  / 86_600_000, "frozen",    (-15, 12)),
    ("DINOv2-Large",          0.5790, 100 * head_params(1024) / 304_800_000, "frozen",   (12, -18)),
    ("BiomedCLIP",            0.5774, 100 * head_params(768)  / 86_000_000, "frozen",    (10, -14)),  # ViT-B/16 vision tower, ~86M
    ("DINOv2+LoRA",           0.6373, 100 * 0.54 / 22.4, "lora",                          (10, 4)),
    ("ResNet50 (fine-tuned)", 0.6638, 100.0, "finetuned",                                 (-155, 4)),
]
GROUP_COLOR = {"frozen": "#5BA3D0", "lora": "#7B5EA7", "finetuned": "#C0603A"}

fig, ax = plt.subplots(figsize=(9, 6.5))
for label, auc, pct, grp, (dx, dy) in PARETO:
    ax.scatter(pct, auc, s=140, color=GROUP_COLOR[grp], edgecolor="black", zorder=3)
    ax.annotate(label, (pct, auc), textcoords="offset points", xytext=(dx, dy),
                fontsize=9, arrowprops=dict(arrowstyle="-", color="grey", lw=0.6,
                shrinkA=0, shrinkB=6))
ax.set_xscale("log")
ax.set_xlim(0.006, 200)
ax.set_xlabel("Trainable parameters (%, log scale)", fontsize=11)
ax.set_ylabel("Test ROC-AUC (6-fold mean)", fontsize=11)
ax.set_title("Parameter efficiency: performance vs. training cost\n(JUMP-CP source_11, 224 resolution)", fontsize=12)
ax.grid(alpha=0.3, which="both")
ax.set_ylim(0.55, 0.70)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_H_pareto_efficiency.png"), dpi=150)
plt.close()
print("wrote fig_H_pareto_efficiency.png")

print("\nAll three figures written to", OUTDIR)
