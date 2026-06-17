#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = "src/models/results"
models = {
    "ResNet50":      f"{BASE}/bioact_resnet50_v2_faithful/plots/per_assay_auc.csv",
    "DINOv2-Base":   f"{BASE}/bioact_dinov2_base/plots/per_assay_auc.csv",
    "CLIP ViT-L/14": f"{BASE}/bioact_clip_vitl14/plots/per_assay_auc.csv",
    "BiomedCLIP":    f"{BASE}/bioact_biomedclip/plots/per_assay_auc.csv",
}
def load(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if 'auc' in c.lower()]
    auc_col = cols[0] if cols else df.columns[-1]
    return df.set_index(df.columns[0])[auc_col]

series = {n: load(p) for n,p in models.items() if os.path.exists(p)}
ref = "ResNet50"
others = [m for m in series if m != ref]
colors = {"DINOv2-Base":"#1f78b4","CLIP ViT-L/14":"#33a02c","BiomedCLIP":"#e31a1c"}

fig, axes = plt.subplots(1, len(others), figsize=(5*len(others), 5), sharex=True, sharey=True)
if len(others)==1: axes=[axes]
for ax, name in zip(axes, others):
    common = series[ref].index.intersection(series[name].index)
    a = series[ref].loc[common].values
    b = series[name].loc[common].values
    n = len(common)
    wins_ref = int((a>b).sum())
    try: stat, p = stats.wilcoxon(a, b)
    except: stat, p = float('nan'), float('nan')
    ax.scatter(a, b, c=colors.get(name,"#666"), s=55, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
    ax.plot([0.2,1],[0.2,1],"k--",lw=1,alpha=0.6,zorder=2)
    ax.set_title(f"{ref} vs {name}\n{ref} higher on {wins_ref}/{n} assays\nWilcoxon p={p:.4f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel(f"{ref} per-assay AUC", fontsize=9)
    ax.set_ylabel(f"{name} per-assay AUC", fontsize=9)
    ax.set_xlim(0.25,1.02); ax.set_ylim(0.25,1.02); ax.set_aspect("equal")
    ax.grid(alpha=0.2, zorder=0)

fig.suptitle("Paired per-assay comparison with Wilcoxon signed-rank test (29 assays, single fold)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = f"{BASE}/comparison/comparison_paired_wilcoxon.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved -> {out}")
