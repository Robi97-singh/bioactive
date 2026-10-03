# BiomedCLIP — Frozen Linear-Probe Arm (biomedical image-text baseline)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

BiomedCLIP (Microsoft, ViT-B/16 vision tower of a PubMedBERT image-text CLIP, 512-dim projected
embedding) evaluated **frozen** with the same linear-probe recipe as the other backbones. It is
the **biomedical-domain, image-text** point in the comparison: pretrained on biomedical figure-
caption pairs (pathology, radiology, microscopy figures from PubMed), not on Cell Painting.
Public benchmark reference: **0.660 ± 0.094**.

## Result -- biomedical-but-mismatched pretraining still does not beat generic ImageNet, though margins are smaller and not significant

| Frozen backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| Cell-DINO | Cell Painting (5 studies) | 21.5M | 0.6373 +/- 0.0119 |
| DINOv2-Base | ImageNet (generic) | 86M | 0.6345 +/- 0.0162 |
| **BiomedCLIP** | Biomedical image-text (PMB) | 86M | **0.6245 +/- 0.0166** |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

BiomedCLIP per-fold test ROC-AUC: 0.6154, 0.6296, 0.6088, 0.6551, 0.6229, 0.6153.

> **Note (2026-10-03):** numbers corrected for compound-level aggregation. The ordering
> (Cell-DINO > DINOv2-Base > BiomedCLIP) is unchanged, but margins shrink and, per the
> project-wide Nemenyi test, none of these pairwise differences are statistically
> significant (BiomedCLIP vs DINOv2-Base p=0.91, vs Cell-DINO p=0.12).

## What this shows

**1. "Biomedical" pretraining is still not the same as domain-matched pretraining, though the
evidence for this is weaker than previously reported.**
BiomedCLIP lands numerically below generic ImageNet DINOv2 (0.6245 vs 0.6345) and below
Cell-DINO (0.6373), preserving the original ordering -- but none of these gaps are
statistically significant under Nemenyi post-hoc testing. The direction is consistent with
"biomedical figure/pathology pretraining is not domain-matched to Cell Painting," but this
benchmark alone cannot establish it conclusively.

**2. The image-text (CLIP) objective remains a plausible, unconfirmed handicap.**
BiomedCLIP's 512-dim embedding, optimized for caption-matching rather than preserving fine
morphological detail, is still a reasonable candidate explanation for its last-place finish
among the three frozen backbones compared here -- domain mismatch and the caption-aligned
projection remain confounded and neither can be isolated from an off-the-shelf backbone alone.

**3. Domain-matched self-supervised pretraining is still numerically ahead, but not proven
best.** Across the three frozen backbones, the ordering Cell-DINO > DINOv2 > BiomedCLIP holds,
but the project-wide Nemenyi results (see the root README and the DINOv3 README) show none of
these differences reach significance -- treat the ordering as a numerical trend, not an
established effect.

## Method

Identical locked recipe to every other frozen arm: frozen backbone, cached embeddings, Linear(D, 29)
head (D = 512 for BiomedCLIP, auto-sized), masked focal-BCE loss, SGD momentum 0.9 + cosine schedule,
lr = 0.02, train-only feature standardization, early stopping on validation ROC-AUC (patience 6),
`mean_roc_auc` metric. Same source_11 CSV, same 6 folds, same masked labels — only the backbone differs.

BiomedCLIP specifics: loaded from HuggingFace via open_clip
(`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`); only the vision tower (`.visual`)
is used. The 3-channel patch embedding is adapted to 5 channels by repeating and slicing the RGB
projection weights (same channel-repeat strategy as the ResNet and DINOv2). The wrapper interpolates
all input to 224 internally, so BiomedCLIP is evaluated at 224 only — a genuine 448 arm is not possible
without positional-embedding surgery and off-distribution extrapolation of a CLIP image-text projection.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)
