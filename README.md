# Project Bioactive

**Benchmarking pretrained vision models for Cell Painting compound-bioactivity prediction.**

A faithful replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024). We reproduce the supervised ResNet-50 baseline and benchmark three pretrained models from three distinct learning paradigms against it, on an identical data split and evaluation protocol.

## Results

| Model | Paradigm | Resolution | Test ROC-AUC |
|---|---|---|---|
| ResNet-50 | Supervised CNN | 448 | **0.702** |
| DINOv2-Base | Self-supervised ViT | 448 | 0.660 |
| CLIP ViT-L/14 | Vision-Language (natural) | 224 | 0.643 |
| BiomedCLIP | Vision-Language (biomedical) | 224 | 0.605 |

Mean ROC-AUC over 29 assays on a single fixed test fold. Public paper benchmark: 0.660 ± 0.094 (6-fold CV).

**Key finding:** performance tracks how closely each model's pretraining matches the supervised target. A supervised CNN trained directly on the assay labels outperforms all general-purpose pretrained models. See `docs/Project_Bioactive_Methodology.docx` for the full methodology, pipeline description, and discussion.

## Task

Cell Painting five-channel fluorescence microscopy images to compound bioactivity, framed as masked multi-label classification over 29 assays (labels +1 active / -1 inactive / 0 not-tested; the loss masks untested entries).

## Repository layout

- `classification.py` — entry point (`--params_path`, `--test`)
- `defaults/` — model definitions, trainer, optimizer wrappers, datasets
- `utils/` — masked BCE loss, per-assay ROC-AUC metrics, helpers
- `params/` — one JSON config per benchmarked model
- `data_prep/` — paper-faithful data preparation scripts + data-download docs
- `make_plots_v2.py`, `make_comparison_figures.py` — figure generation
- `apply_early_stopping.py` — trainer early-stopping patch utility
- `results/` and comparison figures — test result logs, per-assay AUC CSVs
- `docs/` — methodology document

## Not included (live on the compute server, excluded via .gitignore)

- Images — five-channel Cell Painting microscopy (~2 TB)
- Training CSV — produced by the paper-faithful preparation scripts in `data_prep/`
- Checkpoints — trained model weights (1.4–4.8 GB each)
- Pretrained weights — loaded from a local Hugging Face cache (server runs offline)

## Reproducing a run

Each model uses the same recipe; only the params file differs.

**Train:**

export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export HF_HOME=~/.cache/huggingface

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 classification.py --params_path params/params_<model>.json

**Test** (restores the best checkpoint, evaluates the test fold):
python3 classification.py --params_path params/params_<model>.json --test

**Generate figures:**
python3 make_plots_v2.py bioact_<model>        # per-model

python3 make_comparison_figures.py             # cross-model

## Experimental design (summary)

- **Single fixed fold** (folds 0–3 train / 4 val / 5 test) — identical across all models, so model-to-model comparison is internally valid. Comparison to the paper's 6-fold CV mean is indicative.
- **Per-architecture hyperparameters** — ResNet-50: SGD, lr 1e-3, batch 64. All three ViTs: AdamW, lr 1e-4, batch 16 (matched). ReduceLROnPlateau scheduling for all.
- **Resolution** — ResNet-50 and DINOv2 at 448; CLIP-family at native 224 (positional embeddings do not transfer to 448). Field of view held identical across all models.
- **Single seed** — run-to-run variance ≈ ±0.02 ROC-AUC; only differences above this scale are interpreted. Model ordering is robust to it.

Full reasoning and limitations are in `docs/Project_Bioactive_Methodology.docx`.

## Reference

Fredin Haslum, J. et al. *Cell Painting-based bioactivity prediction boosts high-throughput screening hit-rates and compound diversity.* Nature Communications 15, 3470 (2024).

## Evaluation protocol fix — frozen-backbone models (2026-10-03)

### What was wrong

All six-fold CV results for frozen-backbone models (DINOv2 small/base/large,
DINOv3-Base, Cell-DINO, CLIP ViT-L/14, BiomedCLIP — anything trained via
`extract_embeddings.py` + `train_head.py`) were computed on **raw per-image
predictions, with no aggregation across a compound's multiple site images**.

The fine-tuned CNN pipeline (`classification.py` → `defaults/trainer.py` →
`utils/metrics.py`'s `evaluate_predictions()`) has always aggregated
predictions to compound level before scoring:

```python
preds[assays] = sigmoid(preds[assays].values)
preds_combined = preds.groupby("compound").mean()
```

`train_head.py` — which evaluates a linear head on cached, frozen-backbone
embeddings — never had an equivalent step. It scored every image
independently:

```python
test_logits = head(Xte.to(device)).cpu().numpy()
test_mean, _ = mean_roc_auc(Yte.numpy(), test_logits, do_sigmoid=True)
```

### Why that matters

Every assay label is a compound-level property (ChEMBL potency annotation,
not an image-level one) — all site images of the same well carry an
identical label. Scoring each image as an independent test case is
pseudo-replication: correlated, repeated observations of one ground truth
were fed into the AUC calculation as if they were independent samples, and
skipping the averaging step left in per-image noise (site-to-site cell
confluence, debris, illumination variation) that the multi-site imaging
protocol is specifically designed to be averaged away. Single-image scoring
systematically under-estimates a model's true compound-level discriminative
power.

### Impact

Found while building a matched fluorescence baseline for a brightfield
imaging ablation and noticing the evaluation code paths diverged. A
retrospective audit of all frozen-backbone results showed every one
under-reporting AUC by ~0.04–0.05:

| Model | Per-image (uncorrected) | Compound-level (corrected) | Δ |
|---|---|---|---|
| DINOv3-Base r448 | 0.5907 | **0.6448** | +0.0541 |
| Cell-DINO r224 | 0.5932 | 0.6373 | +0.0441 |
| DINOv2-Base r224 | 0.5796 | 0.6345 | +0.0549 |
| DINOv3-Base r224 | 0.5804 | 0.6332 | +0.0528 |
| DINOv2-Large r224 | 0.5790 | 0.6313 | +0.0523 |
| DINOv2-Small r224 | 0.5786 | 0.6283 | +0.0497 |
| CLIP ViT-L/14 r224 | 0.5798 | 0.6267 | +0.0469 |
| BiomedCLIP r224 | 0.5774 | 0.6245 | +0.0471 |

Mean over 6 folds, all 29 assays, all `b-r-singh1` runs. (DINOv3 r224/r448
and CLIP r224 results under `b-b-parihar` could not be corrected
retroactively — raw per-image predictions were never saved for those runs,
only the aggregate `per_assay_auc.csv`.)

**The best frozen model changes as a result: DINOv3-Base r448 (0.6448), not
Cell-DINO r224 (0.6373).** Fine-tuned CNN results (ResNet-50, ResNet-18,
EfficientNet-B3) and the LoRA parameter-efficient arm were unaffected — both
go through `trainer.py`/`metrics.py` and already aggregated by compound
correctly.

### Fix applied

- `train_head.py` now builds a `(plate, well) → compound` lookup from
  `training_paper.csv`, maps each cached embedding's `uid`
  (`PLATE/WELL_SITE.png`) back to its compound, averages sigmoid
  probabilities per compound, and computes AUC on the aggregated values —
  matching `evaluate_predictions()`'s logic exactly. The old per-image mean
  is still recorded (as `test_mean_roc_auc_per_image_UNCORRECTED` in
  `head_metrics.json`) for audit purposes, but is no longer the reported
  metric.
- All historical results for the 8 affected models (48 fold-runs) were
  corrected in place by re-aggregating their already-saved per-image
  `test_preds.csv`/`test_labels.csv` — no retraining was needed, since the
  frozen backbone and trained head weights were untouched by this fix. The
  original per-image files are preserved alongside the corrected ones as
  `test_preds_per_image_UNCORRECTED.csv`, `test_labels_per_image_UNCORRECTED.csv`,
  and `per_assay_auc_per_image_UNCORRECTED.csv` in each fold's `plots/`
  directory.

### Still open

- The top "Results" table in this README predates this fix and this
  project's full 6-fold CV benchmark (it reports a single fixed test fold).
  It should be regenerated from the corrected 6-fold numbers above.
- Any figures/tables produced by `make_benchmark_figures.py`,
  `make_comparison_figures.py`, `make_pr_calibration_pareto.py`, or the
  Wilcoxon/Friedman–Nemenyi comparison scripts that used the old frozen-model
  numbers need to be regenerated from the corrected `per_assay_auc.csv`
  files.
- Parihar's `dinov3_r224`/`dinov3_r448`/`clip_r224` runs need either a
  version of `train_head.py` with this fix re-run, or their raw per-image
  predictions located if they exist elsewhere, before they can be trusted
  alongside the corrected numbers above.
