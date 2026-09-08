#!/usr/bin/env python3
"""
make_adaptation_figure.py

Figure D — the adaptation spectrum: frozen linear probe -> LoRA -> full fine-tuning.
Places the LoRA-adapted DINOv2 (partial CV, folds as available) between the frozen backbones (6-fold CV)
and the fine-tuned ResNet (6-fold CV), showing how much of the frozen->fine-tuned gap
cheap adaptation (2.4% of parameters) recovers.

Also writes the LoRA per-assay AUCs to results/cv/lora_vit_s_r224/cv_per_assay.csv in the
standard format (for reference / optional inclusion elsewhere).

LoRA per-assay AUCs are auto-discovered from whatever bioact_lora_vit_s_r224_fold*
test evaluations exist on disk (results/bioact_lora_vit_s_r224_fold*/plots/per_assay_auc.csv),
averaged per assay, and written to results/cv/lora_vit_s_r224/cv_per_assay.csv in the
standard format. LoRA is shown as a distinct regime point (frozen probe -> LoRA -> full
fine-tuning), not mixed into the 6-fold statistical comparison in Figure B. As more LoRA
folds complete, re-running this script picks them up automatically -- no manual edits needed.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.environ.get("BIOACT_RESULTS", "/shared/ssd/logs/b-r-singh1/results")
OUTDIR = os.environ.get("BIOACT_FIGDIR", os.path.join(RESULTS, "figures"))
os.makedirs(OUTDIR, exist_ok=True)
FOLD_COLS = ["fold0", "fold1", "fold2", "fold3", "fold4", "fold5"]

# --- auto-discover LoRA fold per-assay AUCs from whatever folds exist ---
lora_fold_dfs = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "bioact_lora_vit_s_r224_fold*", "plots", "per_assay_auc.csv"))):
    m = re.search(r"fold(\d+)", path)
    if not m:
        continue
    fold_n = int(m.group(1))
    df = pd.read_csv(path, index_col=0)
    lora_fold_dfs[fold_n] = df["test_roc_auc"]

if not lora_fold_dfs:
    raise SystemExit("No LoRA fold per_assay_auc.csv files found -- run make_plots_v3.py on at least one fold first.")

lora_wide = pd.DataFrame(lora_fold_dfs).sort_index(axis=1)
lora_wide.columns = [f"fold{c}" for c in lora_wide.columns]
lora_wide = lora_wide.sort_index()  # sort by assay label for a stable, readable CSV
lora_wide["mean"] = lora_wide.mean(axis=1)
lora_wide["std"] = lora_wide[[c for c in lora_wide.columns if c.startswith("fold")]].std(axis=1)

n_lora_folds = len([c for c in lora_wide.columns if c.startswith("fold")])
lora_vals = lora_wide["mean"].values

lora_dir = os.path.join(RESULTS, "cv", "lora_vit_s_r224")
os.makedirs(lora_dir, exist_ok=True)
lora_wide.to_csv(os.path.join(lora_dir, "cv_per_assay.csv"))
fold_level_means = [lora_wide[c].mean() for c in lora_wide.columns if c.startswith("fold")]
fold_level_std = np.std(fold_level_means)
print(f"wrote {lora_dir}/cv_per_assay.csv  "
      f"(n_folds={n_lora_folds}, mean {lora_vals.mean():.4f}, "
      f"fold-level std {fold_level_std:.4f})")


def per_assay_means(arm_key):
    path = os.path.join(RESULTS, "cv", arm_key, "cv_per_assay.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df["mean"] if "mean" in df.columns else df[FOLD_COLS].mean(axis=1)


# ---- the spectrum: ordered by degree of adaptation ----
SPECTRUM = [
    ("biomedclip_r224", "BiomedCLIP\n(frozen)",  "frozen"),
    ("dino_r224",       "DINOv2\n(frozen)",      "frozen"),
    ("celldino_r224",   "Cell-DINO\n(frozen)",   "frozen"),
    ("lora_vit_s_r224", "DINOv2+LoRA\n(2.4% params)", "lora"),
    ("resnet_r224",     "ResNet50\n(fine-tuned)","finetuned"),
]
GROUP_COLOR = {"frozen": "#5BA3D0", "lora": "#7B5EA7", "finetuned": "#C0603A"}

labels, means, spreads, colors, groups = [], [], [], [], []
for key, label, grp in SPECTRUM:
    s = per_assay_means(key)
    if s is None:
        print(f"  MISSING {key}")
        continue
    labels.append(label)
    means.append(s.mean())
    spreads.append(s.std())
    colors.append(GROUP_COLOR[grp])
    groups.append(grp)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.6,
              yerr=spreads, capsize=4, error_kw=dict(alpha=0.5))
# value labels
for xi, m in zip(x, means):
    ax.text(xi, m + 0.004, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0.660, ls="--", color="grey", lw=1, label="public JUMP-CP ref (0.660)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean test ROC-AUC (per-assay)", fontsize=11)
ax.set_ylim(0.50, 0.72)
ax.set_title("Adaptation spectrum: frozen probe -> LoRA -> full fine-tuning\n"
             "(DINOv2-family where possible; JUMP-CP source_11, 29 assays)", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)

# annotate the gap LoRA recovers
frozen_dino = per_assay_means("dino_r224").mean()
lora_m = per_assay_means("lora_vit_s_r224").mean()
ft = per_assay_means("resnet_r224").mean()
recovered = 100 * (lora_m - frozen_dino) / (ft - frozen_dino)
ax.text(0.5, 0.02,
        f"LoRA recovers ~{recovered:.0f}% of the frozen->fine-tuned gap "
        f"with 2.4% of parameters trainable\n(LoRA: {n_lora_folds}-fold CV; frozen/fine-tuned: 6-fold CV)",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic",
        bbox=dict(boxstyle="round", fc="#F2F5FA", ec="grey", alpha=0.8))

plt.tight_layout()
out = os.path.join(OUTDIR, "fig_D_adaptation_spectrum.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"wrote {out}")
print(f"\nspectrum: " + " -> ".join(f"{l.split(chr(10))[0]}={m:.3f}"
      for l, m in zip(labels, means)))
