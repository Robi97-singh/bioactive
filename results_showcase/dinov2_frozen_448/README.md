# DINOv2-Base — Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as DINOv2 @ 224, run at 448 input. Together with the
Cell-DINO resolution arms, this answers whether higher resolution helps frozen backbones -
and the answer turns out to be model-dependent. Public benchmark reference: **0.660 ± 0.094**.

## Result — resolution HELPS DINOv2 (unlike Cell-DINO)

| Backbone | Regime | 224 | 448 | resolution effect |
|-----|--------|-----|-----|-------------------|
| **DINOv2-Base** (frozen) | ImageNet | 0.5796 +/- 0.0067 | **0.5890 +/- 0.0046** | **+0.94 pt** |
| Cell-DINO (frozen) | Cell Painting | 0.5932 +/- 0.0065 | 0.5928 +/- 0.0067 | ~0 |
| ResNet50 (fine-tuned) | ImageNet | 0.6638 +/- 0.0153 | 0.6792 +/- 0.0245 | +1.5 pt |

DINOv2 448 per-fold test ROC-AUC: 0.5906, 0.5876, 0.5804, 0.5904, 0.5932, 0.5919.

Paired 224 -> 448 change: +0.0076, +0.0127, +0.0116, +0.0073, +0.0059, +0.0116.
**DINOv2 improves at 448 in all 6 of 6 folds** (+0.94 ROC-AUC points on average).

## What this shows

**1. Resolution's benefit for a frozen backbone is model-dependent - it is NOT a simple
frozen-vs-fine-tuned effect.**
DINOv2 gains +0.94 points going 224 -> 448, in every fold. Cell-DINO, run identically, gains
nothing (0.5932 -> 0.5928). So "frozen backbones can't use extra resolution" is false: it
depends on the backbone.

**2. The difference tracks pretraining resolution and patch size.**
DINOv2 (ViT-B/14) was pretrained at 224px and is known to handle multi-resolution inference
well; at 448 it moves 256 -> 1,024 tokens, a modest 4x that stays within its usable range,
so the finer spatial detail helps. Cell-DINO (ViT-S/8) was pretrained at 128px; it is already
extrapolating its position embeddings hard at 224 (256 -> 784 tokens) and even harder at 448
(256 -> 3,136, ~12x), so it is off-distribution either way and the extra pixels add no usable
signal. A frozen backbone benefits from higher resolution only when the new resolution stays
close to what it was pretrained on.

**3. Higher resolution narrows the domain-pretraining gap.**
At 224, domain-pretrained Cell-DINO (0.5932) leads ImageNet DINOv2 (0.5796) by ~1.4 points.
At 448 the gap shrinks to ~0.4 points (0.5928 vs 0.5890), because resolution helps DINOv2 but
not Cell-DINO. The domain advantage is real but partly a resolution-regime artifact: measured
at each model's better resolution, the two frozen backbones are closer than the 224-only
comparison suggests.

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
