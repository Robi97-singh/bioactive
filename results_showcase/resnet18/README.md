# ResNet-18 (fine-tuned end to end), 6-fold CV

Configuration, training and checkpoints: **Bhartendu Parihar** (collaborator). Scoring: re-computed here from his checkpoints with `cv_rescore.py` (compound level, same protocol as every other model in this folder).

| Arm | Field of view | Mean test ROC-AUC ± SD (6 folds) | Folds 0-5 |
|---|---|---|---|
| ResNet-18, 448 px | 448/540 = 0.83 | 0.6759 ± 0.0218 | 0.6450 0.6744 0.6583 0.7021 0.6795 0.6959 |
| ResNet-18, 224 px | 224/540 = 0.415 | 0.6611 ± 0.0145 | 0.6452 0.6602 0.6448 0.6815 0.6717 0.6628 |

Files: `resnet18_r448/` and `resnet18_r224/`, each with `cv_summary.csv` (per fold) and `cv_per_assay.csv` (per fold and assay).

## Checks
- 448 px: reproduces the numbers in his own text file (`resnet18_cv_results.txt`) to within 3.5e-4 (largest difference, fold 1; fold 0 agrees to about 10 digits).
- 224 px: his text file has no numbers for this arm, so these values are our re-computation only.
- The field of view of each arm is the one applied by `dump_predictions.py` at evaluation (224 px: resize to 540, crop 224). The training transforms are those in his params files; they were not independently checked.

## Caveats
- His ResNet-18 runs use early stopping (patience 6). The ResNet-50 reference setup has none. All other settings compared (learning rate, batch size, head, pretrained weights) are identical. The two models are therefore not a perfectly controlled comparison.
- Params: `params/params_resnet18_fold*.json` and `params_resnet18_r224_fold*.json` (copied from his repository; they keep his paths).
- Re-scoring job: `jobs/robin-bioact-r18-eval.yaml`.
