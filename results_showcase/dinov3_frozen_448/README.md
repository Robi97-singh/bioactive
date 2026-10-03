# DINOv3-Base -- Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv3-Base @ 224, run at 448 input. Together with the
DINOv2-Base, DINOv2-Large, and Cell-DINO resolution arms, this checks whether the 224->448
resolution benefit already established for the DINOv2 family also holds for DINOv3. Public
benchmark reference: **0.660 +/- 0.094**.

## Result -- DINOv3-Base is the only frozen backbone that benefits from 448

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| DINOv2-Base (frozen) | ImageNet | 0.6345 +/- 0.0162 | 0.6211 +/- 0.0084 | -0.0134 |
| DINOv2-Large (frozen) | ImageNet | 0.6313 +/- 0.0202 | 0.6216 +/- 0.0195 | -0.0097 |
| **DINOv3-Base** (frozen) | ImageNet | **0.6332 +/- 0.0142** | **0.6448 +/- 0.0180** | **+0.0116** |
| Cell-DINO (frozen) | Cell Painting | 0.6373 +/- 0.0119 | 0.6203 +/- 0.0115 | -0.0170 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +0.0154 |

DINOv3-Base 448 per-fold test ROC-AUC: 0.6374, 0.6519, 0.6148, 0.6658, 0.6573, 0.6415.

Paired 224 -> 448 change: +0.0214, +0.0172, -0.0034, +0.0215, +0.0052, +0.0076.
**DINOv3-Base improves at 448 in 5 of 6 folds** (+0.0116 on average).

> **Note (2026-10-03):** before the fix, this arm was reported as following the same modest
> positive resolution trend as DINOv2-Base/Large (+1.03 pt, "replicates the DINOv2-family
> effect"). Corrected, DINOv2-Base and DINOv2-Large both *reverse* to a resolution penalty
> (see their own READMEs), while DINOv3-Base is the only frozen backbone in this comparison
> whose 448 result still improves on its 224 result. The finding is now the opposite framing:
> DINOv3-Base is the exception, not a third confirmation of a DINOv2-family pattern.

## What this shows

**1. DINOv3-Base is now the only frozen backbone (of four examined) that benefits from
higher resolution.** DINOv2-Base, DINOv2-Large, and Cell-DINO all score lower at 448 than
224 once compound-level aggregation is applied correctly; only DINOv3-Base, only fine-tuned
ResNet improve. This is a materially different claim than "the DINOv2-family resolution
effect replicates in DINOv3" -- there is no DINOv2-family resolution benefit left to
replicate.

**2. Whatever distinguishes DINOv3-Base from DINOv2-Base/Large here is not explained by this
benchmark's existing mechanistic story.** The architecture family (ViT), patch size, and
ImageNet pretraining are all similar across DINOv2-Base/Large and DINOv3-Base, yet only
DINOv3-Base shows a resolution benefit. Pinning down what DINOv3's training recipe or
architecture changes actually did here would need investigation beyond this benchmark's
scope (e.g. comparing position-embedding handling or training resolution curricula between
DINOv2 and DINOv3 specifically) -- it is flagged as an open question, not answered here.

**3. Fine-tuning remains the more reliable resolution lever.**
Only the fine-tuned ResNet shows a resolution benefit that holds regardless of backbone
choice. For frozen features, resolution's effect is backbone-specific and, for 3 of the 4
backbones examined, negative.

## Method

Identical locked recipe to every other frozen arm, and identical to the DINOv3-Base @ 224 arm in
every respect except input resolution: frozen backbone, cached CLS embeddings, Linear(D, 29)
head, masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature
standardization, early stopping on validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same
source_11 CSV, same 6 folds, same masked labels. Resolution mechanics (FOV-matching via
`_apply_resolution`) are identical to every other model's 224/448 comparison in this project.

## Provenance and independent verification

Embedding extraction and the original linear-probe training were run by Bhargav Parihar
(`b-b-parihar` cluster account). His account's cached embeddings were copied into this account
and the linear head was independently retrained from scratch here, using the same locked recipe,
to obtain raw per-sample test predictions (`test_preds.csv` / `test_labels.csv`) for the
project's pooled Precision-Recall, calibration, and parameter-efficiency figures. The retrained
results matched the originally-reported per-fold means to within floating-point precision for
all 6 folds, confirming equivalence with the original run.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
