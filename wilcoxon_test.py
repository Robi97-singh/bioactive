#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = "src/models/results"
models = {
    "ResNet50":      f"{BASE}/bioact_resnet50_v2_faithful/plots/per_assay_auc.csv",
    "DINOv2-Base":   f"{BASE}/bioact_dinov2_base/plots/per_assay_auc.csv",
    "CLIP-ViT-L/14": f"{BASE}/bioact_clip_vitl14/plots/per_assay_auc.csv",
    "BiomedCLIP":    f"{BASE}/bioact_biomedclip/plots/per_assay_auc.csv",
}

def load(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if 'auc' in c.lower()]
    auc_col = cols[0] if cols else df.columns[-1]
    return df.set_index(df.columns[0])[auc_col]

series = {}
for name, path in models.items():
    if os.path.exists(path):
        series[name] = load(path)
    else:
        print(f"WARNING: missing {path}")

print("="*70)
print("WILCOXON SIGNED-RANK TEST -- per-assay paired comparison")
print("="*70)

ref = "ResNet50"
if ref in series:
    for name in series:
        if name == ref:
            continue
        common = series[ref].index.intersection(series[name].index)
        a = series[ref].loc[common].values
        b = series[name].loc[common].values
        n = len(common)
        wins_ref = int((a > b).sum())
        wins_other = int((b > a).sum())
        mean_diff = float(np.mean(a - b))
        try:
            stat, p = stats.wilcoxon(a, b)
        except Exception:
            stat, p = float('nan'), float('nan')
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"\n{ref} vs {name}  (n={n} assays)")
        print(f"  Mean AUC: {ref} {np.mean(a):.3f} | {name} {np.mean(b):.3f}  (mean paired diff {mean_diff:+.3f})")
        print(f"  {ref} higher on {wins_ref}/{n} assays | {name} higher on {wins_other}/{n}")
        print(f"  Wilcoxon W={stat:.1f}, p={p:.4f}  -> {sig} at alpha=0.05")

print("\n" + "="*70)
print("p<0.05 means the per-assay difference is unlikely by chance.")
print("="*70)
