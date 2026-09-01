#!/usr/bin/env python3
"""
make_benchmark_figures.py

Publication figures for the frozen-backbone bioactivity benchmark, styled after
Fredin Haslum et al. (Nat Commun 2024) Fig 1c (per-assay ROC-AUC boxplot) and
Fig 2a (model comparison with Friedman + Nemenyi).

Reads each arm's aggregated per-assay CV file:
  {RESULTS}/cv/{arm}/cv_per_assay.csv   (index=assay, cols: fold0..fold5, mean, std)

Outputs to {OUTDIR}:
  fig_A_per_assay_boxplot.png     - one box per model, distribution of 29 assay AUCs
  fig_B_model_comparison.png      - boxplot ranked, with Friedman p + Nemenyi CD note
  fig_C_resolution.png            - 224 vs 448 for the DINO-family arms
  benchmark_summary.csv           - mean/median/std per arm for the writeup

Usage (in a pod with pandas/matplotlib/scipy/scikit-posthocs):
  python make_benchmark_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

RESULTS = os.environ.get("BIOACT_RESULTS", "/shared/ssd/logs/b-r-singh1/results")
OUTDIR = os.environ.get("BIOACT_FIGDIR", os.path.join(RESULTS, "figures"))
os.makedirs(OUTDIR, exist_ok=True)

FOLD_COLS = ["fold0", "fold1", "fold2", "fold3", "fold4", "fold5"]

# ---- which arms to load: (results-dir key, pretty label, group) ----
ARMS = [
    ("celldino_r224",   "Cell-DINO",   "frozen"),
    ("dino_r224",       "DINOv2",      "frozen"),
    ("biomedclip_r224", "BiomedCLIP",  "frozen"),
    ("resnet_r224",     "ResNet50\n(fine-tuned)", "finetuned"),
]
# resolution comparison arms
RES_ARMS = [
    ("celldino_r224", "Cell-DINO 224"), ("celldino_r448", "Cell-DINO 448"),
    ("dino_r224", "DINOv2 224"), ("dino_r448", "DINOv2 448"),
    ("resnet_r224", "ResNet 224"), ("resnet_r448", "ResNet 448"),
]

# colors: frozen = blue shades, fine-tuned = warm
COLORS = {
    "Cell-DINO": "#2E6FB0", "DINOv2": "#5BA3D0", "BiomedCLIP": "#8FC3E0",
    "ResNet50\n(fine-tuned)": "#C0603A",
}
PUBLIC_REF = 0.660  # Haslum public JUMP-CP benchmark


def load_per_assay(arm_key):
    """Return DataFrame indexed by assay with fold cols + mean, or None."""
    path = os.path.join(RESULTS, "cv", arm_key, "cv_per_assay.csv")
    if not os.path.exists(path):
        print(f"  MISSING {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    return df


def per_assay_means(arm_key):
    """29 per-assay mean AUCs for one arm (as a Series)."""
    df = load_per_assay(arm_key)
    if df is None:
        return None
    if "mean" in df.columns:
        return df["mean"]
    return df[FOLD_COLS].mean(axis=1)


# =====================================================================
# FIGURE A + B : per-assay boxplot + model comparison (they share data)
# =====================================================================
labels, data, means = [], [], []
for key, label, group in ARMS:
    s = per_assay_means(key)
    if s is None:
        continue
    labels.append(label)
    data.append(s.values)
    means.append(np.mean(s.values))

# ---------- FIGURE A: per-assay distribution boxplot ----------
fig, ax = plt.subplots(figsize=(8, 5.5))
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))
for patch, label in zip(bp["boxes"], labels):
    patch.set_facecolor(COLORS.get(label, "#999999"))
    patch.set_alpha(0.85)
# overlay individual assay points (jittered)
for i, d in enumerate(data):
    x = np.random.normal(i + 1, 0.05, size=len(d))
    ax.scatter(x, d, s=12, color="black", alpha=0.35, zorder=3)
ax.axhline(PUBLIC_REF, ls="--", color="grey", lw=1,
           label=f"public JUMP-CP ref ({PUBLIC_REF})")
ax.axhline(0.5, ls=":", color="red", lw=0.8, alpha=0.6, label="random (0.5)")
ax.set_ylabel("Per-assay test ROC-AUC (6-fold mean)", fontsize=11)
ax.set_title("Per-assay bioactivity prediction across backbones\n"
             "(29 assays, JUMP-CP source_11, 6-fold CV)", fontsize=12)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.45, 0.85)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_A_per_assay_boxplot.png"), dpi=150)
plt.close()
print("wrote fig_A_per_assay_boxplot.png")

# ---------- FIGURE B: model comparison + Friedman/Nemenyi ----------
# Friedman needs the paired matrix: rows = assays, cols = models (per-assay means)
paired = pd.DataFrame({label: s for (key, label, g) in ARMS
                       if (s := per_assay_means(key)) is not None})
paired = paired.dropna()  # assays present in all models (should be all 29)

friedman_stat, friedman_p = friedmanchisquare(*[paired[c].values for c in paired.columns])

# Nemenyi post-hoc (needs scikit-posthocs; fall back gracefully)
nemenyi = None
try:
    import scikit_posthocs as sp
    nemenyi = sp.posthoc_nemenyi_friedman(paired.values)
    nemenyi.index = paired.columns
    nemenyi.columns = paired.columns
except Exception as e:
    print(f"  (scikit-posthocs unavailable: {e}; skipping Nemenyi matrix)")

# order models by mean AUC (best first)
order = paired.mean().sort_values(ascending=False).index.tolist()
data_ord = [paired[c].values for c in order]

fig, ax = plt.subplots(figsize=(8, 5.5))
bp = ax.boxplot(data_ord, labels=order, patch_artist=True, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5))
for patch, label in zip(bp["boxes"], order):
    patch.set_facecolor(COLORS.get(label, "#999999"))
    patch.set_alpha(0.85)
for i, d in enumerate(data_ord):
    x = np.random.normal(i + 1, 0.05, size=len(d))
    ax.scatter(x, d, s=12, color="black", alpha=0.35, zorder=3)
ax.axhline(PUBLIC_REF, ls="--", color="grey", lw=1, label=f"public ref ({PUBLIC_REF})")
ax.set_ylabel("Per-assay test ROC-AUC (6-fold mean)", fontsize=11)
sig = "significant" if friedman_p < 0.05 else "n.s."
ax.set_title(f"Backbone comparison, ranked by mean ROC-AUC\n"
             f"Friedman chi2={friedman_stat:.1f}, p={friedman_p:.2e} ({sig})",
             fontsize=12)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.45, 0.85)
# annotate means just under each box's whisker region
for i, c in enumerate(order):
    ax.text(i + 1, 0.465, f"mean\n{paired[c].mean():.3f}", ha="center",
            fontsize=8, color="black")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_B_model_comparison.png"), dpi=150)
plt.close()
print("wrote fig_B_model_comparison.png")

if nemenyi is not None:
    nemenyi.to_csv(os.path.join(OUTDIR, "nemenyi_pvalues.csv"))
    print("wrote nemenyi_pvalues.csv")
    print("\nNemenyi pairwise p-values:")
    print(nemenyi.round(4).to_string())

# =====================================================================
# FIGURE C : resolution comparison (224 vs 448) for DINO-family + ResNet
# =====================================================================
res_labels, res_data = [], []
for key, label in RES_ARMS:
    s = per_assay_means(key)
    if s is None:
        continue
    res_labels.append(label)
    res_data.append(s.values)

if res_data:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(res_data, labels=res_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))
    # color 224 vs 448 pairs
    pair_colors = ["#5BA3D0", "#2E6FB0"] * 3
    for patch, col in zip(bp["boxes"], pair_colors[:len(bp["boxes"])]):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    ax.axhline(PUBLIC_REF, ls="--", color="grey", lw=1)
    ax.set_ylabel("Per-assay test ROC-AUC (6-fold mean)", fontsize=11)
    ax.set_title("Resolution effect (224 vs 448) by backbone", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.45, 0.85)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_C_resolution.png"), dpi=150)
    plt.close()
    print("wrote fig_C_resolution.png")

# =====================================================================
# summary CSV
# =====================================================================
rows = []
for key, label, group in ARMS + [(k, l, "res") for k, l in RES_ARMS]:
    s = per_assay_means(key)
    if s is None:
        continue
    rows.append({"arm": key, "label": label.replace("\n", " "),
                 "mean_auc": s.mean(), "median_auc": s.median(),
                 "std_across_assays": s.std(), "min": s.min(), "max": s.max()})
pd.DataFrame(rows).drop_duplicates("arm").to_csv(
    os.path.join(OUTDIR, "benchmark_summary.csv"), index=False)
print("wrote benchmark_summary.csv")
print(f"\nFriedman: chi2={friedman_stat:.2f}, p={friedman_p:.3e}")
print("All figures in:", OUTDIR)
