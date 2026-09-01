#!/usr/bin/env python3
"""
make_adaptation_figure.py

Figure D — the adaptation spectrum: frozen linear probe -> LoRA -> full fine-tuning.
Places the LoRA-adapted DINOv2 (single fold) between the frozen backbones (6-fold CV)
and the fine-tuned ResNet (6-fold CV), showing how much of the frozen->fine-tuned gap
cheap adaptation (2.4% of parameters) recovers.

Also writes the LoRA per-assay AUCs to results/cv/lora_vit_s_r224/cv_per_assay.csv in the
standard format (for reference / optional inclusion elsewhere).

LoRA per-assay AUCs are hard-coded from the fold-0 test evaluation (single fold; the other
arms are 6-fold means, so LoRA is shown as a distinct regime point, not mixed into the
6-fold statistical comparison in Figure B).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.environ.get("BIOACT_RESULTS", "/shared/ssd/logs/b-r-singh1/results")
OUTDIR = os.environ.get("BIOACT_FIGDIR", os.path.join(RESULTS, "figures"))
os.makedirs(OUTDIR, exist_ok=True)
FOLD_COLS = ["fold0", "fold1", "fold2", "fold3", "fold4", "fold5"]

# --- LoRA fold-0 per-assay AUCs (from test evaluation) ---
LORA_ASSAYS = {
    '688128': 0.5881481481481482, '688238': 0.5557894736842105, '688360': 0.7482517482517483,
    '688546': 0.33777777777777773, '688549': 0.6307692307692309, '688612': 0.5879699248120301,
    '688687': 0.5666666666666667, '688812': 0.5527426160337553, '688816': 0.6478899952584163,
    '736947': 0.5620253164556962, '737187': 0.5719298245614036, '737287': 0.6006944444444444,
    '737344': 0.5457128099173554, '752347': 0.4876543209876543, '752407': 0.5866012440598076,
    '752434': 0.79375, '752493': 0.5442424242424242, '752563': 0.5398166023166022,
    '752590': 0.6918441558441559, '752594': 0.7204116638078902, '845045': 0.6793032786885245,
    '845102': 0.5958333333333333, '845164': 0.6725352112676056, '845169': 0.7705882352941177,
    '845173': 0.43103448275862066, '845177': 0.7632432432432433, '845196': 0.6958333333333333,
    '954338': 0.5108225108225108, '1495346': 0.658102766798419,
}
lora_vals = np.array(list(LORA_ASSAYS.values()))

# write LoRA per-assay CSV (single fold -> one column + mean)
lora_dir = os.path.join(RESULTS, "cv", "lora_vit_s_r224")
os.makedirs(lora_dir, exist_ok=True)
lora_df = pd.DataFrame({"assay": [f"assay_{i}" for i in range(len(lora_vals))],
                        "fold0": lora_vals, "mean": lora_vals,
                        "std": np.zeros_like(lora_vals)}).set_index("assay")
lora_df.to_csv(os.path.join(lora_dir, "cv_per_assay.csv"))
print(f"wrote {lora_dir}/cv_per_assay.csv  (mean {lora_vals.mean():.4f})")


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
        f"with 2.4% of parameters trainable\n(LoRA is single-fold; frozen/fine-tuned are 6-fold CV)",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic",
        bbox=dict(boxstyle="round", fc="#F2F5FA", ec="grey", alpha=0.8))

plt.tight_layout()
out = os.path.join(OUTDIR, "fig_D_adaptation_spectrum.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"wrote {out}")
print(f"\nspectrum: " + " -> ".join(f"{l.split(chr(10))[0]}={m:.3f}"
      for l, m in zip(labels, means)))
