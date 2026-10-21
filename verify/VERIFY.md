# How to verify the claims in this repository

Run `python3 verify/verify_claims.py` from the repository root. It needs only Python with numpy and scipy. It reads the CSV files committed under `results_showcase/`, recomputes each headline number and prints PASS, FAIL or MISSING next to the value claimed in the README and the documents. Current state: 28 PASS, 0 FAIL, 0 MISSING.

## What the script checks

| Claim | Input file | Check |
|---|---|---|
| Brightfield ROC-AUC 0.6940 (SD 0.0287) | results_showcase/brightfield_vs_fluorescence/brightfield_resnet_r448/cv_summary.csv | mean and sample SD of the six fold values |
| Fluorescence ROC-AUC 0.7097 (SD 0.0484) | .../fluorescence_resnet_r448_102plates/cv_summary.csv | same |
| Difference -0.0157, p = 0.1864, brightfield better on 2 of 6 folds | both files above | paired t-test over the six folds, fold-by-fold sign count |
| Enrichment of five frozen-model methods on 65 random targets, with 95% intervals | results_showcase/explainability/random_65_targets/enrich_r65.csv | values within rounding (0.005) |
| ResNet-50 Grad-CAM enrichment 1.07 (0.90 to 1.28), 58 of 65 targets | .../enrich_resnet_per_target.csv | mean of the non-empty rows, own bootstrap for the interval (tolerance 0.03 on the interval) |

## What it does not check

* The cross-validation benchmark table (ResNet-50 0.6792 and the frozen-probe rows). The per-fold prediction files are not in this repository, so those numbers can be checked only against the cluster logs.
* Faithfulness numbers (deletion and insertion z-scores) are in agg_r65.csv; the script does not recompute them from the raw explanation maps, which are too large to commit.
* Paired differences against the ResNet reference are computed inside make_panel.py (on the targets where the ResNet value exists) and are not recomputed by the script.
* Training itself: it needs the Cell Painting images and a GPU cluster. The job specs in jobs/ and the parameter files in params/ document how each run was launched.

## Notes for a reviewer

* The intervals in bootstrap_ci.csv and enrich_r65.csv differ in the second decimal for some methods (for example Cell-DINO grad x act, 2.26 vs 2.29). Both are bootstrap estimates of the same quantity with different random draws. The README and documents quote enrich_r65.csv.
* Known errors found and corrected during the project are listed in CORRECTIONS.md.
