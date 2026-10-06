# CLIP ViT-L/14 — Frozen Linear-Probe Arm (locked recipe)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

CLIP ViT-L/14 (OpenAI weights, 304M params, 768-dim embedding) evaluated **frozen** with the
*exact same* precompute-then-probe pipeline and locked linear-probe recipe used for every other
frozen backbone in this benchmark (Cell-DINO, DINOv2-Base, BiomedCLIP), so it is directly
comparable to them. Extraction and head-training were run by Bhargav Parihar on his cluster
account, using the project's shared `extract_embeddings.py` / `train_head.py` /
`aggregate_cv.py` scripts and the same source_11 CSV (identical, fixed Butina splits — see the
project's methods on cross-account comparability). Public benchmark reference: **0.660 ± 0.094**.

## Result

| Frozen backbone @ 224 | Pretraining | Params | Test ROC-AUC (6-fold) |
|-----|-------------|--------|-----------------------|
| Cell-DINO | Cell Painting | 21.5M | 0.6373 +/- 0.0119 |
| DINOv2-Base | ImageNet | 86M | 0.6345 +/- 0.0162 |
| DINOv2-Large | ImageNet | 304M | 0.6313 +/- 0.0202 |
| **CLIP ViT-L/14** (this arm) | Web image-text (OpenAI) | 304M | **0.6267 +/- 0.0164** |
| BiomedCLIP | Biomedical image-text | 86M | 0.6245 +/- 0.0166 |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

CLIP per-fold test ROC-AUC: 0.6094, 0.6309, 0.6028, 0.6404, 0.6381, 0.6383.

> **Note (2026-10-03):** numbers corrected for compound-level aggregation. CLIP-L's std rose
> from 0.0038 to 0.0164 -- it is no longer the tightest cross-fold spread among frozen arms,
> that distinction no longer holds under correct scoring. The previous Finding 2, comparing
> this locked-recipe result against a separately-run AdamW-trained variant, has been removed:
> only the locked linear-probe recipe (identical across every frozen arm in this benchmark)
> is reported here, so comparisons against differently-trained variants are out of scope for
> this README.

## What this shows

**1. Under the project's locked probing recipe, CLIP-L lands near the bottom of the frozen
cluster, not above it.**
At 0.6267, CLIP-L sits above only BiomedCLIP (0.6245) among the six frozen backbones compared
across this benchmark, despite being 3.5-14x larger than every other model in the comparison.
Raw model scale does not, by itself, translate into a better *linearly-probed* representation
once the probing procedure is held fixed -- consistent with the scale-invariance finding
reported elsewhere in this benchmark (see DINOv2-Large's README).

**2. Cross-fold spread is unremarkable once correctly scored.**
CLIP-L's std (0.0164) is now in the same range as the other frozen arms (0.0119-0.0202) --
the previously-reported "tightest spread of any arm" was itself a symptom of the per-image
evaluation bug, not a genuine property of this backbone's stability.

## Method

## Method

Identical locked recipe to every other frozen arm: frozen backbone, cached CLS/pooled
embeddings (768-dim), Linear(768, 29) head, masked focal-BCE loss (`BCEMASKEDLoss`), SGD
momentum 0.9 + cosine schedule, lr = 0.02, train-only feature standardization, early stopping
on validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6 folds,
same masked labels as every other arm — only the backbone differs. Verified via
`bioact_clip_r224_fold0_head_metrics.json`: `lr: 0.02, standardize: true`, matching the
project's recipe signature exactly.

CLIP specifics: `--model clip` resolves (via `classification.py` MODEL_MAP) to backbone_type
`clip_vitl14` — OpenAI CLIP ViT-L/14, same architecture as the AdamW comparison run in note 2
above. 5-channel adaptation via the project's standard repeat-and-slice strategy on the patch
embedding, same as ResNet and DINOv2.

## Attribution

Extraction and linear-probe training for this arm were run by Bhargav Parihar
(`b-b-parihar` cluster account), using the shared project scripts and recipe, on the same
`training_paper.csv` and fixed 6-fold Butina splits as every other arm in this benchmark —
confirmed comparable (same CSV path, same `data_split_numbers` config, same masking/loss/
metric). Results pulled into this repo via `rsync` from the shared cluster filesystem.

## Files

- `cv_summary.csv` — per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` — per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` — mean per-assay ROC-AUC across folds
