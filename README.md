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
