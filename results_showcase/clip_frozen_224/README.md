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
| Cell-DINO | Cell Painting | 21.5M | 0.5932 +/- 0.0065 |
| **CLIP ViT-L/14** (this arm) | Web image-text (OpenAI) | 304M | **0.5798 +/- 0.0038** |
| DINOv2-Base | ImageNet | 86M | 0.5796 +/- 0.0067 |
| BiomedCLIP | Biomedical image-text | 86M | 0.5774 +/- 0.0060 |
| ResNet50 (fine-tuned, ref) | ImageNet | 25M | 0.6638 +/- 0.0153 |

CLIP per-fold test ROC-AUC: 0.5770, 0.5806, 0.5741, 0.5801, 0.5849, 0.5818 (tightest spread of
any frozen arm: std 0.0038).

## What this shows

**1. Under the project's locked probing recipe, CLIP-L lands with the other small/mid frozen
backbones, not above them.**
At 0.5798, CLIP-L is statistically indistinguishable from DINOv2-Base (0.5796) and BiomedCLIP
(0.5774), and below the domain-matched Cell-DINO (0.5932) — despite being 3.5-14x larger than
every other model in the comparison. Raw model scale does not, by itself, translate into a
better *linearly-probed* representation once the probing procedure is held fixed.

**2. The probing recipe matters as much as the backbone (recipe-sensitivity finding).**
The same CLIP ViT-L/14 backbone, probed with a different training procedure (AdamW, lr=1e-4,
full image-based training rather than the precompute-then-probe SGD/cosine/standardized
recipe used here), reaches **0.6312 +/- 0.0188** — a ~5 ROC-AUC point difference from the
*identical frozen features*. This is a larger swing than most of the between-backbone
differences reported elsewhere in this benchmark, and it is a genuine methodological caution:
comparing frozen backbones is only meaningful when the probing recipe is held constant. Every
other arm in this benchmark uses the single locked recipe (SGD momentum 0.9, cosine schedule,
lr=0.02, train-only feature standardization, patience 6) specifically to avoid this confound;
this CLIP arm is the direct evidence for why that discipline matters.

**3. Tightest cross-fold spread of any arm (std 0.0038).**
Consistent with CLIP-L's scale, its frozen features give a highly stable linear-probe result
across folds, even though the recipe used here does not surface its full capacity relative to
the AdamW-trained variant.

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
