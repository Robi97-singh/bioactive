# DINOv2-Small -- Frozen Linear-Probe Arm @ 224

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024), on the
public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

## Result

- **Test ROC-AUC (6-fold mean +/- SD): 0.6259 +/- 0.0121** (compound-level aggregation)
- Per-fold: 0.6134, 0.6231, 0.6125, 0.6427, 0.6359, 0.6276
- Per-image (uncorrected) mean: 0.5761 -> compound-level gain +0.0498
- Rank among the 11 frozen configurations: 9 of 11 (best: dino_large_r448, 0.6412; worst: biomedclip_r224, 0.6214)
- Head training: lr 0.02, stopped after 33-77 epochs per fold (early stopping on compound-level validation ROC-AUC)

## Benchmark context

| Reference | Test ROC-AUC (6-fold) |
|---|---|
| ResNet-50 fine-tuned, 448 | 0.6792 +/- 0.0245 |
| ResNet-50 fine-tuned, 224 | 0.6638 +/- 0.0153 |
| DINOv2-Small + LoRA, 224 | 0.6373 +/- 0.0203 |

Frozen probes are statistically tied with each other apart from one pair (Cell-DINO over BiomedCLIP), and
fine-tuned ResNet-50 is consistently best; see `../benchmark_figures/README.md` for the Friedman/Nemenyi tests.

## Method

Identical locked recipe for every frozen arm: frozen backbone, cached CLS embeddings, `Linear(D, 29)` head,
masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr 0.02, train-only feature standardization,
early stopping on compound-level validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6 folds,
same masked labels. The learning rate was chosen in a fold-0 sweep on Cell-DINO and applied unchanged to all frozen backbones.

## Verification

Results re-run on 2026-10-06 with the committed `train_head.py` and an explicit `--lr 0.02`. Per-assay AUC recomputed
independently from the saved compound-level `test_preds.csv` / `test_labels.csv` matches the reported values to ~1e-16
on spot-checked runs. See the root README, section "Recipe and re-run verification".
