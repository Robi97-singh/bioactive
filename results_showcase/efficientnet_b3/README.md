# EfficientNet-B3 (fine-tuned end to end), 6-fold CV

Configuration, training and checkpoints: **Bhartendu Parihar** (collaborator). Scoring: re-computed here from his checkpoints with `cv_rescore.py` (compound level, same protocol as every other model in this folder).

| Arm | Field of view | Mean test ROC-AUC ± SD (6 folds) | Folds 0-5 |
|---|---|---|---|
| EfficientNet-B3, 448 px | 448/540 = 0.83 | 0.6692 ± 0.0118 | 0.6467 0.6708 0.6709 0.6802 0.6768 0.6701 |
| EfficientNet-B3, 224 px | 224/540 = 0.415 | 0.6622 ± 0.0254 | 0.6293 0.6462 0.6583 0.7032 0.6756 0.6609 |

Files: `efficientnet_b3_r448/` and `efficientnet_b3_r224/`, each with `cv_summary.csv` and `cv_per_assay.csv`. Only these two arms are included (the 224 px arm has the same field of view as the ResNet-50 224 px FOV 0.415 run).

## Checks
- 224 px: reproduces the numbers in his text file (`effnet_b3_r224_cv_results.txt`) to within 4.6e-5.
- 448 px: does **not** match his text file (`effnet_b3_448_cv_results.txt`):

| | Fold 0 | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|---|
| Re-computed from his checkpoints | 0.6467 | 0.6708 | 0.6709 | 0.6802 | 0.6768 | 0.6701 | 0.6692 |
| His text file | 0.6325 | 0.6576 | 0.6618 | 0.6748 | 0.6694 | 0.6675 | 0.6606 |

The text file is lower in every fold (by 0.003 to 0.014). Predictions saved in his result folders agree with the re-computed values at the two folds we compared, so the text file appears to come from a different run than the checkpoints now on disk. The reason has not been established. This README and the verifier use the checkpoint-derived numbers.

## Caveats
- The field of view of each arm is the one applied by `dump_predictions.py` at evaluation; the training transforms are those in his params files and were not independently checked.
- Params: `params/params_efficientnet_b3_fold*.json` (448 px) and `params_efficientnet_b3_r224_fold*.json` (224 px), copied from his repository.
- Re-scoring job: `jobs/robin-bioact-effnet-eval.yaml`.

Per-fold (`cv_fold_spread.png`) and per-assay (`cv_per_assay.png`) plots for each resolution are in the arm folders; made with `plot_cv_extra.py`.
