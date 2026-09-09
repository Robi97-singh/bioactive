# DINOv2-Large -- Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv2-Large @ 224, run at 448 input. Together with the
DINOv2-Base and Cell-DINO resolution arms, this confirms whether the 224->448 resolution
benefit seen in DINOv2-Base also holds at larger scale (DINOv2-Large, 304M params). Public
benchmark reference: **0.660 +/- 0.094**.

## Result -- resolution helps DINOv2-Large too, same pattern as DINOv2-Base

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| DINOv2-Base (frozen) | ImageNet | 0.5796 +/- 0.0067 | 0.5890 +/- 0.0046 | +0.94 pt |
| **DINOv2-Large** (frozen) | ImageNet | **0.5790 +/- 0.0066** | **0.5888 +/- 0.0071** | **+0.98 pt** |
| Cell-DINO (frozen) | Cell Painting | 0.5932 +/- 0.0065 | 0.5928 +/- 0.0067 | ~0 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +1.5 pt |

DINOv2-Large 448 per-fold test ROC-AUC: 0.5858, 0.5874, 0.5797, 0.5917, 0.6011, 0.5870.

Paired 224 -> 448 change: +0.0077, +0.0111, +0.0114, +0.0039, +0.0188, +0.0058.
**DINOv2-Large improves at 448 in all 6 of 6 folds** (+0.98 ROC-AUC points on average) --
matching DINOv2-Base's per-fold consistency almost exactly.

## What this shows

**1. The resolution benefit for DINOv2-family backbones is scale-independent.**
DINOv2-Base gains +0.94 points going 224 -> 448; DINOv2-Large gains +0.98 points -- the same
effect size, within noise, despite a 3.5x difference in parameter count. Whatever makes higher
resolution help this architecture family (finer patch-level detail relative to native
pretraining resolution) does not depend on model size.

**2. This reinforces, rather than complicates, the earlier scale-ladder finding.**
DINOv2-Large and DINOv2-Base remain statistically indistinguishable at 224 (Nemenyi p=0.997)
and now show near-identical resolution sensitivity at 448 as well. Two independent axes (raw
performance, and response to resolution) both show scale alone changes nothing about this
backbone's behaviour -- consistent with finding #2 in the main benchmark README (scale alone
does not help, frozen).

**3. Resolution's benefit is still model-dependent overall, not just size-dependent.**
Both DINOv2 variants gain about a point at 448; Cell-DINO (pretrained on 128px crops) gains
essentially nothing. The determining factor remains proximity to the backbone's own
pretraining resolution, not parameter count, architecture family alone, or frozen-vs-fine-tuned
status.

## Method

Identical locked recipe to every other frozen arm, and identical to the DINOv2-Large @ 224 arm
in every respect except input resolution: frozen backbone, cached CLS embeddings, Linear(D, 29)
head (D = 1024, auto-sized), masked focal-BCE loss, SGD momentum 0.9 + cosine schedule,
lr = 0.02, train-only feature standardization, early stopping on validation ROC-AUC
(patience 6), mean_roc_auc metric. Same source_11 CSV, same 6 folds, same masked labels.

**Resolution mechanics**: per `_apply_resolution(res=448)`, the Resize step is skipped entirely
and the crop is taken directly from the native 1080x1080 image at 448x448 -- giving
FOV = 448/1080 = 0.415, matching the 224 arm's FOV (224/540 = 0.415) via a different path
(224 resizes to 540 first, then crops; 448 crops straight from native resolution). This FOV-
matching convention is identical to every other model's 224/448 comparison in this project.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` -- validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` -- mean validation ROC-AUC across folds (+/-1 std)
