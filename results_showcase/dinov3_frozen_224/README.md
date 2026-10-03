# DINOv3-Base -- Frozen Linear-Probe Arm @ 224

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv3-Base (Meta, ImageNet-pretrained self-supervised ViT, next generation of the DINOv2
family) evaluated **frozen** with the *exact same* precompute-then-probe pipeline and locked
linear-probe recipe used for every other frozen backbone in this benchmark (Cell-DINO,
DINOv2-Base, DINOv2-Large, BiomedCLIP, CLIP ViT-L/14), so it is directly comparable to them.
Training and extraction were run by Bhargav Parihar on his cluster account, using the project's
shared `extract_embeddings.py` / `train_head.py` scripts and the same source_11 CSV (identical,
fixed 6-fold splits). Public benchmark reference: **0.660 +/- 0.094**.

## Result -- a reordered cluster, with no pairwise Nemenyi significance from Cell-DINO

| Frozen backbone @ 224 | Pretraining | Test ROC-AUC (6-fold) |
|-----|-------------|------------------------|
| Cell-DINO | Cell Painting | 0.6373 +/- 0.0119 |
| DINOv2-Base | ImageNet | 0.6345 +/- 0.0162 |
| **DINOv3-Base** (this arm) | ImageNet | **0.6332 +/- 0.0142** |
| DINOv2-Large | ImageNet | 0.6313 +/- 0.0202 |
| CLIP ViT-L/14 (locked recipe) | Web image-text | 0.6267 +/- 0.0164 |
| BiomedCLIP | Biomedical image-text | 0.6245 +/- 0.0166 |
| ResNet50 (fine-tuned, ref) | ImageNet | 0.6638 +/- 0.0153 |

DINOv3-Base per-fold test ROC-AUC: 0.6160, 0.6347, 0.6182, 0.6443, 0.6521, 0.6339.

Nemenyi post-hoc p-values vs Cell-DINO: DINOv2-Base 0.7693, DINOv2-Large 0.7693, DINOv3-Base
0.3018, CLIP-L 0.1047, BiomedCLIP 0.1217 -- **none significant at p<0.05.**

> **Note (2026-10-03):** before the fix, this README reported "five independent confirmations"
> of Cell-DINO's advantage over every generically-pretrained backbone, each nominally
> significant under the (buggy, per-image) Nemenyi test. Under correct compound-level
> aggregation, Cell-DINO's mean is still numerically highest, but **no pairwise comparison
> against any generically-pretrained backbone reaches significance.** The overall Friedman
> test across all 7 arms is still significant (chi2=27.50, p=1.17e-4) -- some arms do differ
> from each other (chiefly the fine-tuned ResNet vs the weaker frozen arms) -- but that
> omnibus signal is not evidence that Cell-DINO specifically beats the generic-pretraining
> cluster.

## What this shows

**1. Cell-DINO is numerically ahead of every generically-pretrained backbone, but none of
these differences are statistically distinguishable from chance under Nemenyi post-hoc
testing.** DINOv3-Base, at 0.6332, sits in the middle of a cluster running from 0.6245
(BiomedCLIP) to 0.6345 (DINOv2-Base) -- a 0.01 spread -- with Cell-DINO at 0.6373 only
marginally above the top of that cluster. The previous framing ("five architectures, three
pretraining paradigms, all converging below Cell-DINO, mostly significantly") overstated the
strength of this evidence; the direction is consistent, but the statistical support for a
real domain-pretraining advantage, at least at this sample size (29 assays), is weak.

**2. This does not mean domain pretraining doesn't matter -- it means this particular
comparison can't establish that it does, at this significance threshold.**
A larger assay panel, more folds, or a different significance test might resolve the
numerically-consistent-but-not-significant pattern one way or the other. As reported here,
the honest claim is "Cell-DINO is directionally ahead of generic-pretraining backbones, not
provably better than them."

## Method

Identical locked recipe to every other frozen arm: frozen backbone, cached CLS embeddings,
Linear(D, 29) head (D auto-sized to DINOv3-Base's embedding dimension), masked focal-BCE loss,
SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature standardization, early
stopping on validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6
folds, same masked labels as every other arm -- only the backbone differs. Verified via
`bioact_dinov3_r224_fold{0..5}_head_metrics.json`: `lr: 0.02, standardize: true`, matching the
project's recipe signature exactly.

## Provenance and independent verification

Embedding extraction and the original linear-probe training were run by Bhargav Parihar
(`b-b-parihar` cluster account) on the same `training_paper.csv` and fixed 6-fold splits as
every other arm in this benchmark. His account's saved cached embeddings (`.pt` files) were
copied into this account and the linear head was **independently retrained from scratch** here,
using the same locked recipe, to obtain the raw per-sample test predictions
(`test_preds.csv` / `test_labels.csv`) needed for the project's pooled Precision-Recall,
calibration, and parameter-efficiency figures (Figs F, G, H) -- these were not saved by the
original run. The retrained results matched the originally-reported per-fold means to within
floating-point precision (all 6 folds agreeing to 6+ decimal places), confirming the retrain is
equivalent to the original run rather than a divergent result.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
