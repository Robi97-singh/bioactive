# DINOv3-Base -- Frozen Linear-Probe Arm @ 224

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv3-Base (Meta, ImageNet-pretrained self-supervised ViT, next generation of the DINOv2
family) evaluated **frozen** with the *exact same* precompute-then-probe pipeline and locked
linear-probe recipe used for every other frozen backbone in this benchmark (Cell-DINO,
DINOv2-Base, DINOv2-Large, BiomedCLIP, CLIP ViT-L/14), so it is directly comparable to them.
Training and extraction were run by Bhargav Parihar on his cluster account, using the project's
shared `extract_embeddings.py` / `train_head.py` scripts and the same source_11 CSV (identical,
fixed 6-fold splits). Public benchmark reference: **0.660 +/- 0.094**.

## Result

| Frozen backbone @ 224 | Pretraining | Test ROC-AUC (6-fold) |
|-----|-------------|------------------------|
| Cell-DINO | Cell Painting | 0.5932 +/- 0.0065 |
| CLIP ViT-L/14 (locked recipe) | Web image-text | 0.5798 +/- 0.0038 |
| DINOv2-Base | ImageNet | 0.5796 +/- 0.0067 |
| **DINOv3-Base** (this arm) | ImageNet | **0.5804 +/- 0.0060** |
| DINOv2-Large | ImageNet | 0.5790 +/- 0.0066 |
| BiomedCLIP | Biomedical image-text | 0.5774 +/- 0.0060 |
| ResNet50 (fine-tuned, ref) | ImageNet | 0.6638 +/- 0.0153 |

DINOv3-Base per-fold test ROC-AUC: 0.5802, 0.5837, 0.5736, 0.5812, 0.5895, 0.5740.

## What this shows

**1. A newer DINO generation does not escape the domain-vs-generic pattern already established
in this benchmark.** DINOv3-Base (0.5804) lands in the same tight cluster as DINOv2-Base
(0.5796), DINOv2-Large (0.5790), CLIP-L under the locked recipe (0.5798), and BiomedCLIP
(0.5774) -- all generically-pretrained backbones sit within about 0.003 ROC-AUC of each other,
well below the domain-matched Cell-DINO (0.5932). Whatever architectural or training-recipe
improvements DINOv3 brings over DINOv2 at a generic-vision level, they do not close the gap that
domain-specific (Cell Painting) pretraining opens on this task -- reinforcing this benchmark's
central finding that pretraining domain, not backbone generation or scale, is the dominant
factor for frozen linear-probe performance here.

**2. This is a fifth independent confirmation of the same pattern.** Cell-DINO's advantage over
generically-pretrained backbones now holds across DINOv2-Base, DINOv2-Large, BiomedCLIP,
CLIP ViT-L/14 (locked recipe), and now DINOv3-Base -- five different architectures, three
different pretraining paradigms (self-supervised vision-only, vision-language contrastive,
and DINOv3's improved self-supervised recipe), all converging on the same generically-pretrained
cluster below Cell-DINO.

## Method

Identical locked recipe to every other frozen arm: frozen backbone, cached CLS embeddings,
Linear(D, 29) head (D auto-sized to DINOv3-Base's embedding dimension), masked focal-BCE loss,
SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature standardization, early
stopping on validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6
folds, same masked labels as every other arm -- only the backbone differs. Verified via
`bioact_dinov3_r224_fold{0..5}_head_metrics.json`: `lr: 0.02, standardize: true`, matching the
project's recipe signature exactly.

## Provenance and independent verification

Embedding extraction and the original linear-probe training were run by Bhargav Parihar
(`b-b-parihar` cluster account) on the same `training_paper.csv` and fixed 6-fold splits as
every other arm in this benchmark. His account's saved cached embeddings (`.pt` files) were
copied into this account and the linear head was **independently retrained from scratch** here,
using the same locked recipe, to obtain the raw per-sample test predictions
(`test_preds.csv` / `test_labels.csv`) needed for the project's pooled Precision-Recall,
calibration, and parameter-efficiency figures (Figs F, G, H) -- these were not saved by the
original run. The retrained results matched the originally-reported per-fold means to within
floating-point precision (all 6 folds agreeing to 6+ decimal places), confirming the retrain is
equivalent to the original run rather than a divergent result.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
