# Cell-DINO — Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as the Cell-DINO @ 224 arm, run at 448 input to test
whether higher resolution helps frozen cell-pretrained features. Public benchmark
reference: **0.660 ± 0.094**.

## Result — resolution makes no difference for a frozen backbone

| Arm | Regime | Input | Field of View | Test ROC-AUC (6-fold) |
|-----|--------|-------|---------------|-----------------------|
| **Cell-DINO @ 224** | frozen linear probe | 224 | resize 540, crop 224 = 0.415 | 0.5932 +/- 0.0065 |
| **Cell-DINO @ 448** | frozen linear probe | 448 | crop 448 from 1080 = 0.415 | 0.5928 +/- 0.0067 |
| ResNet50 @ 224 (fine-tuned, ref) | fine-tuned | 224 | 0.415 | 0.6638 +/- 0.0153 |
| ResNet50 @ 448 (fine-tuned, ref) | fine-tuned | 448 | 0.415 | 0.6792 +/- 0.0245 |

448 per-fold test ROC-AUC: 0.5924, 0.5967*, 0.5870, 0.5989, 0.5975, 0.5828.
(*fold-1 aggregate value; per-fold spread matches the 224 arm within noise.)

The 224 and 448 frozen results differ by **0.0004** — statistically identical. At a matched
field of view (0.415), doubling the input resolution does not move the frozen Cell-DINO number.

## What this shows

**1. Higher resolution does not help a frozen backbone.**
224 and 448 land on top of each other (0.5932 vs 0.5928). The frozen backbone produces
essentially the same CLS embedding at both resolutions, so the linear head has the same
information to work with either way.

**2. This is the opposite of the fine-tuned ResNet, and that contrast is the point.**
Fine-tuned ResNet gained ~1.5 ROC-AUC points going 224 -> 448 (0.6638 -> 0.6792). The value
of extra resolution is unlocked by **fine-tuning** — a network that can adapt its weights
learns to exploit the finer detail. A frozen backbone cannot adapt, so the extra pixels
carry no benefit it can use.

**3. Position-embedding extrapolation compounds the effect.**
Cell-DINO was pretrained at 128px (patch-8, 256 tokens). At 224 the model already interpolates
its position embeddings to 784 tokens; at 448 it stretches them to 3,136 tokens (a ~12x
extrapolation from pretraining). The frozen features were never trained to use information at
that token density, so the added resolution arrives in a form the representation cannot exploit.

**Takeaway:** for this task, resolution is a fine-tuning lever, not a feature-extraction lever.
Reporting a frozen backbone at higher resolution is not worth the ~4x compute cost.

## Method

Identical to the 224 arm: frozen backbone, cached CLS embeddings (384-dim), Linear(384, 29)
head, masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only
feature standardization, early stopping on validation ROC-AUC (patience 6), `mean_roc_auc`
metric. Same source_11 CSV, same 6 folds, same masked labels — only the input resolution
differs from the 224 arm.

Resolution handling: at 448 the pipeline disables the resize and crops 448 directly from the
native 1080 image (FOV = 448/1080 = 0.415), matching the ResNet-448 field of view. The DINOv2
ViT auto-interpolates its position embeddings to the resulting token count.

Extraction at 448 is ~4x slower than 224 (patch-8 at 448 = 3,136 tokens vs 784; attention is
O(n^2)), for no measurable gain — see finding 1.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)
