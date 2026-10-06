# Cell-DINO — Frozen Linear-Probe Arm @ 448 (Resolution Axis)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

Same frozen linear-probe pipeline as the Cell-DINO @ 224 arm, run at 448 input to test
whether higher resolution helps frozen cell-pretrained features. Public benchmark
reference: **0.660 ± 0.094**.

## Result -- resolution modestly HURTS this frozen backbone

| Arm | Regime | Input | Field of View | Test ROC-AUC (6-fold) |
|-----|--------|-------|---------------|-----------------------|
| **Cell-DINO @ 224** | frozen linear probe | 224 | resize 540, crop 224 = 0.415 | **0.6373 +/- 0.0119** |
| **Cell-DINO @ 448** | frozen linear probe | 448 | crop 448 from 1080 = 0.415 | **0.6203 +/- 0.0115** |
| ResNet50 @ 224 (fine-tuned, ref) | fine-tuned | 224 | 0.415 | 0.6638 +/- 0.0153 |
| ResNet50 @ 448 (fine-tuned, ref) | fine-tuned | 448 | 0.415 | 0.6792 +/- 0.0245 |

448 per-fold test ROC-AUC: 0.6157, 0.6284, 0.6160, 0.6330, 0.6269, 0.6017.

Paired 224 -> 448 change: -0.0096, -0.0170, -0.0081, -0.0166, -0.0218, -0.0289.
**448 is worse than 224 in all 6 of 6 folds** (-0.0170 on average).

> **Note (2026-10-03):** before the frozen-backbone evaluation fix, this arm was reported
> as "224 and 448 are statistically identical" (0.5932 vs 0.5928). Under correct
> compound-level aggregation, 448 is consistently *worse*, not equal -- the direction of
> this finding has changed, not just its magnitude.

## What this shows

**1. Higher resolution does not help this frozen backbone -- if anything, it modestly hurts.**
Every one of the 6 folds favors 224 over 448, by 1.7 ROC-AUC points on average. This is the
opposite of what the fine-tuned ResNet shows, and also the opposite of what DINOv3-Base shows
at the same two resolutions (see that arm's README) -- resolution sensitivity is backbone-
specific, and for Cell-DINO specifically, 448 is a net loss, not a neutral non-effect.

**2. The contrast with fine-tuning is still the headline, just sharper than before.**
Fine-tuned ResNet gains ~1.5 points going 224 -> 448 (0.6638 -> 0.6792); frozen Cell-DINO
*loses* ~1.7 points over the same change. A network that can adapt its weights learns to
exploit the finer detail; a frozen backbone cannot adapt, and here the extra resolution
appears to actively work against the fixed representation rather than simply being unused.

**3. Position-embedding extrapolation is a plausible contributor, but why 448 actively hurts
(not just fails to help) isn't established by this experiment alone.**
Cell-DINO was pretrained at 128px (patch-8, 256 tokens); at 224 it already interpolates to
784 tokens, at 448 to 3,136 (~12x). That this produces features the linear head finds
*harder* to use than the 224 features -- not just equally good -- would need direct inspection
of the embeddings to confirm, not just inferred from the AUC gap.

**Takeaway:** for this task, resolution remains a fine-tuning lever, not a feature-extraction
lever -- and for this particular frozen backbone, pushing resolution without fine-tuning is
actively counterproductive, not merely wasted compute.

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
