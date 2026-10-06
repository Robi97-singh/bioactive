# Cell-DINO — Frozen Linear-Probe Arm

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

All results are **6-fold cross-validation test ROC-AUC**, structure-aware splits (ECFP4 + Butina),
on the **same CSV and folds** as the ResNet arms. Public benchmark reference: **0.660 ± 0.094**.

Cell-DINO (Meta FAIR): a ViT-S/8 self-supervised backbone pretrained on Cell Painting
imagery, 384-dim CLS embedding, native 5-channel input. Here it is evaluated **frozen** —
the backbone is never fine-tuned; only a `Linear(384, 29)` head is trained on cached
embeddings. This answers *how good are the off-the-shelf features*, in contrast to the
fully fine-tuned ResNet.

## Result

| Arm | Regime | Input | Field of View | Test ROC-AUC (6-fold) |
|-----|--------|-------|---------------|-----------------------|
| **Cell-DINO @ 224** | frozen linear probe | 224 | resize 540, crop 224 = **0.415** | **0.6373 +/- 0.0119** |
| ResNet50 @ 224 (fine-tuned, ref) | fine-tuned | 224 | 224/540 = 0.415 | 0.6638 +/- 0.0153 |
| ResNet50 @ 448 (fine-tuned, ref) | fine-tuned | 448 | 448/1080 = 0.415 | 0.6792 +/- 0.0245 |

Per-fold test ROC-AUC: 0.6253, 0.6454, 0.6241, 0.6496, 0.6487, 0.6306.

> **Note (2026-10-03):** numbers corrected for compound-level aggregation. The frozen-to-
> fine-tuned gap shrinks substantially (was ~7 points, now ~2.65), and the CV spread
> comparison below is updated accordingly.

## What this shows

**1. Frozen domain-pretrained features still underperform full fine-tuning, but by a much
smaller margin than previously reported.**
At a matched field of view (0.415), the frozen Cell-DINO linear probe reaches 0.6373, about
2.65 ROC-AUC points below the fine-tuned ResNet at the same resolution (0.6638) -- down from
the previously-reported ~7-point gap. Fine-tuning still adapts the whole network while a
linear probe can only reweight fixed features, so a gap is still expected, just a more modest
one than originally measured.

**2. The frozen result remains fairly stable across folds, though not dramatically more so
than the fine-tuned ResNet.**
Cell-DINO's CV spread is now +/-0.0119, about three-quarters of the fine-tuned ResNet's
+/-0.0153 -- still somewhat tighter, but not the "less than half" previously reported. Some of
the apparent extra stability in the original per-image numbers may itself have been an
artifact of how the bug affected variance estimation, not a genuine property being
systematically under- or overstated in one consistent direction.

**3. The hyperparameter-robustness checks described below were run during original,
pre-correction experimentation** (feature standardization and learning-rate sweep on fold 0),
and describe the exploratory process rather than the final reported numbers above -- they are
retained for methodological context, not as a claim about the current 0.6373 figure.

## Method (linear-probe recipe, applied identically to every frozen backbone)

- **Features:** frozen backbone forward pass, cached once per fold (train/val/test), CLS token (384-dim).
- **Standardization:** subtract train mean, divide by train std (train-only stats; matches Cell-DINO's own linear-eval protocol).
- **Head:** `Linear(384, 29)`, masked focal-BCE loss (`BCEMASKEDLoss`) — the same loss the full pipeline uses.
- **Optimizer:** weight SGD, momentum 0.9, cosine LR schedule, lr = 0.02 (DINO-family linear-probe convention).
- **Early stopping:** on validation ROC-AUC, patience 6 (matches the ResNet regime).
- **Metric:** `mean_roc_auc` — per-assay masked ROC-AUC, the same metric as ResNet.

Comparability to ResNet: same source_11 CSV, same 6 folds, same masked labels, same loss,
same metric. Only the features differ (frozen Cell-DINO vs fine-tuned ResNet).

Note: fine-tuned ResNet used RandomCrop augmentation; the frozen probe uses a deterministic
center crop (one embedding per image), which is standard for linear probes.

## Pipeline

Two standalone scripts keep the frozen arm fast (the image-based path recomputed identical
frozen features every epoch at ~53 min/epoch with the GPU idle; caching pays that cost once):

1. `extract_embeddings.py` — run the frozen backbone over each split once, cache `[N, 384]`
   embeddings + labels (the only NFS-bound step, ~1 pass per fold).
2. `train_head.py` — train the linear head on cached embeddings (seconds/epoch, no images).

Then `aggregate_cv.py celldino 224` produces the CV summary, and `plot_head_curves.py celldino 224`
the val ROC-AUC learning curves.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
- `head_val_auc_overlay.png` — validation ROC-AUC per epoch, one line per fold
- `head_val_auc_meanband.png` — mean validation ROC-AUC across folds (+/-1 std)

For a linear probe the *loss* curve is nearly flat and uninformative (the head starts
near-converged on frozen features), so the meaningful learning signal shown here is the
validation ROC-AUC curve.
