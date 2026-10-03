# DINOv2-Large -- Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv2-Large @ 224, run at 448 input. Together with the
DINOv2-Base and Cell-DINO resolution arms, this confirms whether the 224->448 resolution
benefit seen in DINOv2-Base also holds at larger scale (DINOv2-Large, 304M params). Public
benchmark reference: **0.660 +/- 0.094**.

## Result -- resolution HURTS DINOv2-Large too (reversed from the pre-fix finding)

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| DINOv2-Base (frozen) | ImageNet | 0.6345 +/- 0.0162 | 0.6211 +/- 0.0084 | -0.0134 |
| **DINOv2-Large** (frozen) | ImageNet | **0.6313 +/- 0.0202** | **0.6216 +/- 0.0195** | **-0.0097** |
| Cell-DINO (frozen) | Cell Painting | 0.6373 +/- 0.0119 | 0.6203 +/- 0.0115 | -0.0170 |
| DINOv3-Base (frozen) | ImageNet | 0.6332 +/- 0.0142 | 0.6448 +/- 0.0180 | +0.0116 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +0.0154 |

DINOv2-Large 448 per-fold test ROC-AUC: 0.5993, 0.6182, 0.6109, 0.6338, 0.6540, 0.6133.

Paired 224 -> 448 change: -0.0184, -0.0089, +0.0103, -0.0205, +0.0060, -0.0272.
**DINOv2-Large is worse at 448 in 4 of 6 folds** (-0.0097 on average), closely matching
DINOv2-Base's reversal.

> **Note (2026-10-03):** before the fix, this arm was reported matching DINOv2-Base's clean
> resolution gain ("+0.98 points, 6 of 6 folds"). Corrected, the effect reverses in the same
> direction and magnitude as DINOv2-Base -- both ImageNet-pretrained DINOv2 sizes now show a
> modest resolution *penalty*, not a benefit.

## What this shows

**1. The resolution reversal is scale-independent, same as the original (now-superseded)
resolution benefit was claimed to be.**
DINOv2-Base loses 0.0134 points at 448; DINOv2-Large loses 0.0097 -- the same effect, within
noise, despite a 3.5x parameter difference. Whatever drives this (now negative) resolution
sensitivity for DINOv2-family backbones does not depend on model size, same as before, just
with the opposite sign.

**2. This still reinforces the scale-ladder finding from the 224 comparison, from the other
direction.** DINOv2-Base and DINOv2-Large remain statistically indistinguishable at 224
(Nemenyi p=1.0000) and now show near-identical resolution sensitivity at 448 as well --
two independent axes both confirm scale alone changes nothing about this backbone family's
behavior on this task.

**3. DINOv3-Base, not either DINOv2 size, is the exception that needs explaining.**
With both DINOv2 sizes and Cell-DINO now agreeing that 448 is a net loss for frozen features
here, DINOv3-Base's improvement at 448 is the finding that actually needs a mechanistic
account -- not DINOv2's (illusory) 224-vs-448 gain, which this correction has removed.

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
