# DINOv3-Base -- Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv3-Base @ 224, run at 448 input. Together with the
DINOv2-Base, DINOv2-Large, and Cell-DINO resolution arms, this checks whether the 224->448
resolution benefit already established for the DINOv2 family also holds for DINOv3. Public
benchmark reference: **0.660 +/- 0.094**.

## Result -- resolution helps DINOv3-Base too, though less uniformly than DINOv2

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| DINOv2-Base (frozen) | ImageNet | 0.5796 +/- 0.0067 | 0.5890 +/- 0.0046 | +0.94 pt |
| DINOv2-Large (frozen) | ImageNet | 0.5790 +/- 0.0066 | 0.5888 +/- 0.0071 | +0.98 pt |
| **DINOv3-Base** (frozen) | ImageNet | **0.5804 +/- 0.0060** | **0.5907 +/- 0.0093** | **+1.03 pt** |
| Cell-DINO (frozen) | Cell Painting | 0.5932 +/- 0.0065 | 0.5928 +/- 0.0067 | ~0 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +1.5 pt |

DINOv3-Base 448 per-fold test ROC-AUC: 0.5928, 0.5967, 0.5736, 0.5970, 0.5974, 0.5867.

Paired 224 -> 448 change: +0.0126, +0.0130, +0.0000, +0.0158, +0.0079, +0.0127. DINOv3-Base
improves in 5 of 6 folds, with one fold (fold 2) essentially unchanged (0.5736 at both
resolutions, to 4 decimal places) rather than the clean 6-of-6 improvement seen for both DINOv2
variants -- a genuine, minor departure worth noting rather than smoothing over.

## What this shows

**1. The DINOv2-family resolution effect replicates in DINOv3-Base, at a similar magnitude.**
+1.03 points at 448, closely matching DINOv2-Base's +0.94 and DINOv2-Large's +0.98 -- three
different model sizes/generations within the DINO self-supervised lineage all show essentially
the same, modest resolution benefit, reinforcing the project's finding that this effect tracks
architecture family and pretraining resolution proximity, not any single model's specific scale
or generation.

**2. Unlike the two DINOv2 variants, the gain is not uniform across every fold.** Fold 2 shows
no improvement at all (0.5736 -> 0.5736), while every other fold gains 0.8-1.6 points. This is
the first resolution-arm result in this benchmark to show a genuine per-fold exception rather
than a clean sweep, and is reported as such rather than folded into an overstated "consistent
improvement" claim -- the aggregate effect is still real and positive, but slightly less uniform
than DINOv2's.

**3. Domain-matched pretraining still resists the resolution effect entirely.** Cell-DINO
remains flat at both resolutions regardless of which ImageNet-pretrained backbone it is compared
against -- DINOv3-Base joining DINOv2-Base and DINOv2-Large as a third generically-pretrained
backbone that *does* benefit from 448 further sharpens the contrast with Cell-DINO's
pretraining-resolution-bound behavior (see the DINOv2 resolution showcase for the underlying
mechanism).

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
