# BiomedCLIP — Frozen Linear-Probe Arm (biomedical image-text baseline)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

BiomedCLIP (Microsoft, ViT-B/16 vision tower of a PubMedBERT image-text CLIP, 512-dim projected
embedding) evaluated **frozen** with the same linear-probe recipe as the other backbones. It is
the **biomedical-domain, image-text** point in the comparison: pretrained on biomedical figure-
caption pairs (pathology, radiology, microscopy figures from PubMed), not on Cell Painting.
Public benchmark reference: **0.660 ± 0.094**.

## Result — biomedical-but-mismatched pretraining does not beat generic ImageNet

| Frozen backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| Cell-DINO | Cell Painting (5 studies) | 21.5M | 0.5932 +/- 0.0065 |
| DINOv2-Base | ImageNet (generic) | 86M | 0.5796 +/- 0.0067 |
| **BiomedCLIP** | Biomedical image-text (PMB) | 86M | **0.5774 +/- 0.0060** |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

BiomedCLIP per-fold test ROC-AUC: 0.5749, 0.5776, 0.5743, 0.5877, 0.5797, 0.5701.

## What this shows

**1. "Biomedical" pretraining is not the same as domain-matched pretraining.**
BiomedCLIP lands slightly below generic ImageNet DINOv2 (0.5774 vs 0.5796) and clearly below
Cell-DINO (0.5932). Only the model actually pretrained on Cell Painting morphology (Cell-DINO)
beats generic features. BiomedCLIP's "biomedical" data is largely pathology/radiology and figure
panels from PubMed — visually far from 5-channel fluorescence Cell Painting — so the nominal
domain label overstates the true domain match for this task.

**2. The image-text (CLIP) objective is a likely handicap.**
BiomedCLIP's vision tower outputs a 512-dim embedding projected into a shared image-text space,
optimized for matching captions rather than preserving fine morphological detail. That projection
appears to discard exactly the subtle texture/intensity signal bioactivity prediction relies on.
The two effects (domain mismatch + caption-aligned projection) are confounded here — both plausibly
contribute, and neither can be isolated from an off-the-shelf backbone alone.

**3. Domain-matched self-supervised pretraining remains the winner.**
Across the three frozen backbones, the ordering is Cell-DINO > DINOv2 > BiomedCLIP. Pretraining on
the actual assay modality (Cell-DINO) is the only thing that beats generic ImageNet; broad biomedical
pretraining via an image-text objective does not.

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
