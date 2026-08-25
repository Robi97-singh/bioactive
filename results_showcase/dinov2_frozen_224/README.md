# DINOv2-Base — Frozen Linear-Probe Arm (ImageNet baseline)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Base (Meta, ViT-B/14, ImageNet-pretrained, 768-dim) evaluated **frozen** with the same
linear-probe recipe as Cell-DINO. This is the **generic-pretraining baseline**: it answers
whether a strong general-purpose self-supervised backbone matches domain-specific (Cell Painting)
pretraining on cell images. Public benchmark reference: **0.660 ± 0.094**.

## Result — domain pretraining beats generic ImageNet, frozen

| Backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| **Cell-DINO** (frozen) | Cell Painting (domain) | 21.5M | **0.5932 +/- 0.0065** |
| **DINOv2-Base** (frozen) | ImageNet (generic) | 86M | **0.5796 +/- 0.0067** |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

DINOv2 per-fold test ROC-AUC: 0.5830, 0.5749, 0.5688, 0.5831, 0.5873, 0.5803.

Paired per-fold comparison (Cell-DINO - DINOv2): +0.0091, +0.0218, +0.0184, +0.0160, +0.0126, +0.0037.
**Cell-DINO wins in all 6 of 6 folds.**

## What this shows

**1. Domain-specific pretraining beats generic ImageNet pretraining - frozen, on every fold.**
Cell-DINO (pretrained on Cell Painting) outperforms DINOv2 (pretrained on ImageNet) by ~1.4
ROC-AUC points, and the win is consistent across all six folds. For frozen feature extraction
on cell images, *what the backbone was pretrained on* matters more than raw model quality.

**2. Domain beats scale.**
DINOv2-Base is the larger model (86M params, 768-dim embedding) yet loses to the smaller
Cell-DINO (21.5M params, 384-dim). The advantage is not capacity - it is representational fit.
Cell-morphology pretraining produces features better aligned to bioactivity than generic
natural-image pretraining, even from a smaller network.

**3. Both frozen backbones sit below the fine-tuned ResNet.**
Neither frozen backbone reaches the fine-tuned ResNet-224 (0.6638); frozen features leave
performance on the table versus full adaptation (see the Cell-DINO arm). But among frozen
options, domain pretraining is the better starting point.

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
