# DINOv2-Base — Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv2 @ 224, run at 448 input. Together with the
Cell-DINO resolution arms, this answers whether higher resolution helps frozen backbones -
and the answer turns out to be model-dependent. Public benchmark reference: **0.660 ± 0.094**.

## Result -- resolution HURTS DINOv2-Base (reversed from the pre-fix finding)

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| **DINOv2-Base** (frozen) | ImageNet | **0.6345 +/- 0.0162** | **0.6211 +/- 0.0084** | **-0.0134** |
| Cell-DINO (frozen) | Cell Painting | 0.6373 +/- 0.0119 | 0.6203 +/- 0.0115 | -0.0170 |
| DINOv3-Base (frozen) | ImageNet | 0.6332 +/- 0.0142 | 0.6448 +/- 0.0180 | +0.0116 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +0.0154 |

DINOv2 448 per-fold test ROC-AUC: 0.6122, 0.6265, 0.6229, 0.6104, 0.6324, 0.6223.

Paired 224 -> 448 change: -0.0130, +0.0028, +0.0075, -0.0424, -0.0224, -0.0126.
**DINOv2-Base is worse at 448 in 4 of 6 folds** (-0.0134 on average).

> **Note (2026-10-03):** before the frozen-backbone evaluation fix, this arm was reported as
> DINOv2's clean win -- "+0.94 points, 6 of 6 folds improve." Under correct compound-level
> aggregation, the effect reverses: DINOv2-Base is now *worse* at 448 on average, and only
> wins in 2 of 6 folds. Cell-DINO shows the same reversal (see its own README); DINOv3-Base
> is now the only frozen backbone in this comparison that still benefits from 448 (see its
> README). This is a full reversal of the original finding, not a magnitude change.

## What this shows

**1. Resolution does not reliably help frozen backbones here -- if anything it tends to hurt,
and DINOv3-Base is the exception, not DINOv2.**
The original story ("DINOv2 benefits from resolution, Cell-DINO doesn't") is backwards under
correct scoring: DINOv2-Base and Cell-DINO both get worse at 448, while DINOv3-Base is the
one backbone in this family that improves. Whatever separates DINOv3-Base from DINOv2-Base
here is not captured by the "pretraining resolution proximity" story used previously, since
both are ImageNet-pretrained ViTs of similar design.

**2. The previous mechanistic explanation (patch size / pretraining resolution proximity) no
longer fits the data and has not been replaced with a new one.**
The old argument -- DINOv2 handles 448 well because it stays within its usable multi-resolution
range, while Cell-DINO's ViT-S/8 position embeddings are already over-extrapolated at 224 --
predicted DINOv2 should benefit and Cell-DINO shouldn't. Both now get worse. Establishing why
DINOv3-Base alone benefits would need direct inspection (e.g. probing the cached embeddings
at both resolutions), not just inference from the AUC table.

**3. The fine-tuned ResNet remains the one model that reliably benefits from resolution.**
ResNet, which can adapt its weights to use finer spatial detail, is the only architecture in
this comparison with a clean, multi-model-consistent resolution gain. For every frozen
backbone examined except DINOv3-Base, pushing resolution without fine-tuning is neutral-to-
harmful, not a free improvement.

## Method

Identical locked recipe to every other frozen arm: frozen backbone, cached CLS embeddings
(768-dim), Linear(768, 29) head, masked focal-BCE loss, SGD momentum 0.9 + cosine schedule,
lr = 0.02, train-only feature standardization, early stopping on validation ROC-AUC
(patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6 folds, same masked labels -
only the input resolution differs from the DINOv2 @ 224 arm.

Resolution handling: at 448 the pipeline crops 448 from the native 1080 image (FOV 0.415,
matching every other 448 arm). DINOv2's forward passes the input straight through with no
internal resize; HuggingFace DINOv2 interpolates its position embeddings to the 1,024-token
grid (448/14 = 32 x 32, exactly divisible by the patch size). Verified the 448 cached
embeddings differ from the 224 ones (distinct md5), so the gain is a genuine resolution effect.

Extraction rate at 448 (~1.16 it/s) matched 224 (~1.09 it/s): at these resolutions the
bottleneck is NFS image reads, not the forward pass, so 448 costs no extra wall-clock for DINOv2.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)
