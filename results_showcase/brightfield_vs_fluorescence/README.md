# Brightfield vs fluorescence (ResNet-50, 448 px, same 102 plates)

A one-channel brightfield ResNet-50 and a five-channel fluorescence ResNet-50 were trained at 448 px on the same 102 plates, one job per fold, over all six folds (same split and compound-level aggregation as the main benchmark). The question: do the fluorescence channels carry information that brightfield lacks?

Folders: `brightfield_resnet_r448/` and `fluorescence_resnet_r448_102plates/` each hold cv_summary.csv (per-fold test ROC-AUC), cv_per_assay.csv, cv_fold_spread.png, cv_per_assay.png and cv_val_curves.png (produced by aggregate_cv.py). `compare_resnet_output.txt` is the saved output of compare_resnet.py, including the paired tests below.

These numbers use the 102-plate subset and are not comparable with the full-dataset ResNet-50 rows of the main benchmark. Folds 2 and 5 have 28 evaluable assays instead of 29 for both models.

| Fold | Brightfield | Fluorescence | Brightfield minus fluorescence |
|---|---|---|---|
| 0 | 0.6973 | 0.6852 | +0.0122 |
| 1 | 0.6754 | 0.6657 | +0.0096 |
| 2 | 0.6496 | 0.6624 | -0.0128 |
| 3 | 0.7071 | 0.7275 | -0.0205 |
| 4 | 0.7010 | 0.7285 | -0.0275 |
| 5 | 0.7335 | 0.7890 | -0.0555 |
| Mean (SD) | 0.6940 (0.0287) | 0.7097 (0.0484) | -0.0157 |

Paired t-test on the six folds: difference -0.0157 (SD 0.0252), p = 0.1864, brightfield better on 2 of 6 folds. Not significant; fluorescence is ahead on four folds (2 to 5), so a small fluorescence advantage is not ruled out.

The same comparison file also shows that fine-tuned brightfield ResNet-50 at 448 px beats frozen brightfield DINOv3 at 448 px (+0.0353, p = 0.017, better on 6 of 6 folds) and frozen brightfield Cell-DINO at 224 px (+0.1058, p = 0.0001, 6 of 6 folds).
