# DINOv2-Base — Frozen Linear-Probe Arm (ImageNet baseline)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Base (Meta, ViT-B/14, ImageNet-pretrained, 768-dim) evaluated **frozen** with the same
linear-probe recipe as Cell-DINO. This is the **generic-pretraining baseline**: it answers
whether a strong general-purpose self-supervised backbone matches domain-specific (Cell Painting)
pretraining on cell images. Public benchmark reference: **0.660 ± 0.094**.

## Result -- domain pretraining has a small, non-significant edge, not a clean win

| Backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| **Cell-DINO** (frozen) | Cell Painting (domain) | 21.5M | **0.6373 +/- 0.0119** |
| **DINOv2-Base** (frozen) | ImageNet (generic) | 86M | **0.6345 +/- 0.0162** |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

DINOv2 per-fold test ROC-AUC: 0.6252, 0.6237, 0.6154, 0.6528, 0.6548, 0.6349.

Paired per-fold comparison (Cell-DINO - DINOv2): +0.0001, +0.0217, +0.0087, -0.0032, -0.0061, -0.0043.
**Cell-DINO wins 3 of 6 folds, DINOv2 wins 3 of 6** -- an even split. Nemenyi post-hoc test:
Cell-DINO vs DINOv2-Base p=0.7693 (not significant).

> **Note (2026-10-03):** before the frozen-backbone evaluation fix, this arm was reported as
> "Cell-DINO wins in all 6 of 6 folds" with a 1.4-point mean gap. Under correct compound-level
> aggregation, the mean gap shrinks to 0.3 points, the fold-win count is an even 3-3 split, and
> the difference is not statistically significant (Nemenyi p=0.77). This is a substantial
> softening of the original finding, not just a number update.

## What this shows

**1. Domain-specific pretraining shows a small numerical edge over generic ImageNet
pretraining, frozen -- but it is not a consistent, significant win.**
Cell-DINO's mean (0.6373) is marginally above DINOv2's (0.6345), but the two split folds
evenly and are statistically indistinguishable by Nemenyi post-hoc test. The original claim
that domain pretraining "wins on every fold" does not survive correction.

**2. Scale still doesn't explain the (now much smaller) gap either way.**
DINOv2-Base is the larger model (86M params, 768-dim) and Cell-DINO is smaller (21.5M, 384-dim);
whatever tiny edge Cell-DINO has is not attributable to capacity, but given the edge is itself
not significant, this is a weaker observation than previously framed.

**3. Both frozen backbones sit modestly, but not always significantly, below the fine-tuned
ResNet.** The gap to ResNet (0.6638) is now ~2.65-2.93 points, down from ~7 points pre-fix.
Nemenyi shows Cell-DINO vs ResNet is *not* significant (p=0.7693) -- Cell-DINO's frozen score
is statistically indistinguishable from the fine-tuned model -- while DINOv2-Base vs ResNet is
borderline (p=0.0546). The clean "frozen clearly underperforms fine-tuning" story from the
original README does not hold uniformly across backbones.

## Method

Identical locked recipe to the Cell-DINO arm, applied without change so the comparison is clean:
frozen backbone, cached CLS embeddings, Linear(D, 29) head (D = 768 for DINOv2, auto-sized),
masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature
standardization, early stopping on validation ROC-AUC (patience 6), `mean_roc_auc` metric.
Same source_11 CSV, same 6 folds, same masked labels, same loss and metric as every other arm -
only the backbone differs.

DINOv2 specifics: loaded from HuggingFace (`facebook/dinov2-base`); the 3-channel ImageNet
patch embedding is adapted to 5 channels by repeating and slicing the RGB projection weights
(the same channel-repeat strategy the ResNet uses). ViT-B/14 at 224 = 256 patch tokens.

Note on backbone differences: Cell-DINO (patch-8, 5-channel-native, 384-dim) and DINOv2
(patch-14, 3->5 adapted, 768-dim) differ in more than pretraining domain. The comparison is an
"off-the-shelf frozen backbone" benchmark - each model is taken as it ships - rather than a
controlled single-variable study of pretraining domain alone.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)
