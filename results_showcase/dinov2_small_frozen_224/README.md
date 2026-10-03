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

## Result -- scale still doesn't clearly separate the three sizes, but LoRA's gap-recovery number changes substantially

| Backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| **DINOv2-Small** (frozen) | ImageNet (generic) | 22M | **0.6283 +/- 0.0144** |
| **DINOv2-Base** (frozen) | ImageNet (generic) | 86M | 0.6345 +/- 0.0162 |
| DINOv2-Large (frozen, ref) | ImageNet (generic) | 305M | 0.6313 +/- 0.0202 |
| **DINOv2-Small + LoRA** (adapted) | ImageNet (generic) | 22M (2.4% trained) | 0.6373 +/- 0.0185 |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

DINOv2-Small per-fold test ROC-AUC: 0.6150, 0.6266, 0.6120, 0.6513, 0.6364, 0.6284.

Paired per-fold comparison (DINOv2-Base - DINOv2-Small): +0.0102, -0.0029, +0.0034, +0.0015,
+0.0184, +0.0065. **Base now wins 5 of 6 folds** (mean +0.0062) -- more consistent than the
pre-fix "coin flip" (3-3), though the gap remains modest.

> **Note (2026-10-03):** before the fix, Base vs Small was reported as an even 3-3 split
> (differences within +/-0.001). Corrected, Base wins 5 of 6 folds, though the margin per fold
> is still small. More importantly: the LoRA gap-recovery calculation below changes from
> **68.9% to approximately 25%** -- LoRA's own number was unaffected by the bug (it was already
> compound-aggregated), but the frozen DINOv2-Small baseline it's compared against rose sharply
> (0.5786 -> 0.6283), which mechanically shrinks the "gap left to recover."

## What this shows

**1. Scale still shows no clean, monotonic trend across the three DINOv2 sizes.**
DINOv2-Small (22M) = 0.6283, DINOv2-Large (305M) = 0.6313, DINOv2-Base (86M) = 0.6345 --
Base is numerically highest despite being neither the smallest nor the largest model, so
there is still no evidence that scale alone drives frozen performance here. The spread
(0.0062) is wider than pre-fix (0.001) but still modest relative to other effects in this
benchmark.

**2. LoRA's gap-recovery claim drops from 68.9% to ~25% -- this is the most consequential
single change from the evaluation fix anywhere in this project.**
LoRA (0.6373) was already evaluated correctly before the fix (its CNN-style pipeline always
aggregated by compound), so its number is unchanged. What changed is the frozen DINOv2-Small
baseline it's compared against, which rose from 0.5786 to 0.6283 once its own evaluation was
corrected. Recovered fraction of the frozen-to-fine-tuned gap is now:
(0.6373 - 0.6283) / (0.6638 - 0.6283) = **25.4%**, at 2.4% of the trainable parameters --
not the 68.9% previously reported. LoRA still closes *some* of the gap, but the originally-
reported "LoRA recovers most of the frozen-to-fine-tuned gap" headline does not survive
correction; adaptation still helps, but the frozen baseline was simply much closer to LoRA
than previously measured.

**3. The adaptation spectrum (frozen -> LoRA -> fine-tuned) is now much more compressed.**
Frozen DINOv2-Small (0.6283) to LoRA (0.6373) is a gap of 0.009; LoRA to full fine-tuning
(0.6638) is a gap of 0.0265. The overall frozen-to-fine-tuned spread (0.6283 to 0.6638 =
0.0355) is itself much smaller than the pre-fix spread (0.5786 to 0.6638 = 0.0852) -- most of
what looked like "headroom for adaptation" in the original analysis was actually the
evaluation bug depressing the frozen baseline, not real headroom LoRA or fine-tuning recover.

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
