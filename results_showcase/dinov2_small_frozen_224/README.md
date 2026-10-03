> **[EVALUATION FIX, 2026-10-03]** The ROC-AUC numbers below predate a correction to the frozen-backbone evaluation pipeline (predictions were scored per-image instead of aggregated per-compound, underestimating AUC by ~0.02-0.05). Corrected `cv_summary.csv`/`cv_per_assay.csv` (and plots, where applicable) are in this folder; the tables/prose below are not yet updated. See the root README's "Evaluation protocol fix" section for corrected numbers and full explanation.

# DINOv2-Small — Frozen Linear-Probe Arm (same-backbone baseline for LoRA)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Small (Meta, ViT-S/14, ImageNet-pretrained, 384-dim) evaluated **frozen** with the same
linear-probe recipe as every other frozen arm. This is the **same-backbone baseline for the
DINOv2+LoRA arm**: LoRA is built on top of DINOv2-Small, not DINOv2-Base, so comparing LoRA's
gains against DINOv2-Base's frozen score would confound adaptation method with backbone size.
This arm closes that confound and gives a genuine apples-to-apples baseline. It also extends the
scale-invariance finding (see DINOv2-Base's README) to a third size point. Public benchmark
reference: **0.660 ± 0.094**.

## Result — a third data point confirming scale doesn't matter, frozen

| Backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| **DINOv2-Small** (frozen) | ImageNet (generic) | 22M | **0.5786 +/- 0.0051** |
| **DINOv2-Base** (frozen) | ImageNet (generic) | 86M | 0.5796 +/- 0.0067 |
| DINOv2-Large (frozen, ref) | ImageNet (generic) | 305M | 0.5790 +/- 0.0071 |
| **DINOv2-Small + LoRA** (adapted) | ImageNet (generic) | 22M (2.4% trained) | 0.6373 +/- 0.0203 |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

DINOv2-Small per-fold test ROC-AUC: 0.5746, 0.5779, 0.5721, 0.5847, 0.5841, 0.5780.

Paired per-fold comparison (DINOv2-Base − DINOv2-Small): +0.0084, −0.0030, −0.0033, −0.0016,
+0.0032, +0.0023. **Base wins 3 of 6 folds, Small wins 3 of 6 folds** — a coin flip, confirming
the two sizes are not meaningfully different despite a 4x parameter gap.

## What this shows

**1. Scale still doesn't matter, frozen — now across three sizes, not two.**
DINOv2-Small (22M), DINOv2-Base (86M), and DINOv2-Large (305M) land within 0.001 ROC-AUC of
each other (0.5786 / 0.5796 / 0.5790). A 14x range in parameter count, same architecture family,
same frozen recipe, produces no meaningful difference in frozen feature quality. This is the same
conclusion the Base-vs-Large comparison already showed, now confirmed at a third point.

**2. This closes a real confound in the LoRA gap-recovery number.**
LoRA adapts DINOv2-Small specifically, not DINOv2-Base. With a genuine same-size frozen baseline
(0.5786) instead of DINOv2-Base's (0.5796), the gap-recovery calculation is no longer mixing
"benefit of adaptation" with "benefit of a different base model size" — both are DINOv2-Small,
only the training regime (frozen vs. LoRA-adapted) differs. Recovered fraction of the
frozen-to-fine-tuned gap: (0.6373 − 0.5786) / (0.6638 − 0.5786) = **68.9%**, at 2.4% of the
trainable parameters.

**3. Frozen DINOv2-Small still sits well below both LoRA and full fine-tuning.**
As with every frozen arm, the ceiling is adaptation, not backbone choice: moving from frozen
(0.5786) to LoRA-adapted (0.6373) on the *same* backbone recovers most of the remaining gap to
full fine-tuning (0.6638) — evidence that *how much* of the model is allowed to train matters far
more than *which* frozen backbone is chosen.

## Method

Identical locked recipe to every other frozen arm, applied without change: frozen backbone,
cached CLS embeddings, Linear(384, 29) head (~11.2K trainable parameters), masked focal-BCE loss,
SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature standardization, early stopping
on validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6 folds, same
masked labels, same loss and metric as every other arm — only the backbone differs.

DINOv2-Small specifics: loaded from HuggingFace (`facebook/dinov2-small`); the 3-channel
ImageNet patch embedding is adapted to 5 channels using the same channel-repeat-and-slice
strategy as DINOv2-Base and DINOv2-Large. ViT-S/14 at 224 = 256 patch tokens, 384-dim CLS
embedding (vs. 768-dim for Base, 1024-dim for Large).

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)
