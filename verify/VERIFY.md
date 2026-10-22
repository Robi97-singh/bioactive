# How to verify the claims in this repository

Run `python3 verify/verify_claims.py` from the repository root. It needs only Python with numpy and scipy. It reads the CSV files committed under `results_showcase/`, recomputes each headline number and prints PASS, FAIL or MISSING next to the value claimed in the README and the documents. Current state: 65 PASS, 0 FAIL, 0 MISSING.

## What the script checks

| Claim | Input file | Check |
|---|---|---|
| Brightfield ROC-AUC 0.6940 (SD 0.0287) | results_showcase/brightfield_vs_fluorescence/brightfield_resnet_r448/cv_summary.csv | mean and sample SD of the six fold values |
| Fluorescence ROC-AUC 0.7097 (SD 0.0484) | .../fluorescence_resnet_r448_102plates/cv_summary.csv | same |
| Difference -0.0157, p = 0.1864, brightfield better on 2 of 6 folds | both files above | paired t-test over the six folds, fold-by-fold sign count |
| Enrichment of five frozen-model methods on 65 random targets, with 95% intervals | results_showcase/explainability/random_65_targets/enrich_r65.csv | values within rounding (0.005) |
| ResNet-50 Grad-CAM enrichment 1.07 (0.90 to 1.28), 58 of 65 targets | .../enrich_resnet_per_target.csv | mean of the non-empty rows, own bootstrap for the interval (tolerance 0.03 on the interval) |
| Benchmark table in README.md (18 rows: ResNet-50, ResNet-18 and EfficientNet-B3 at 448 and 224 px, LoRA, frozen probes) | results_showcase/<model>/cv_summary.csv, one file per row | mean and sample SD of the six fold values, compared with the README row (tolerance 0.00006, i.e. rounding to four decimals); a row without a matching file is reported as MISSING |

## What it does not check

* The per-fold scores themselves. The benchmark table is checked against the per-fold scores in the cv_summary.csv files, but those scores were computed from per-image prediction files that are too large to commit, so they are not re-derived from predictions here. They can be checked against the cluster logs.
* Faithfulness numbers (deletion and insertion z-scores) are in agg_r65.csv; the script does not recompute them from the raw explanation maps, which are too large to commit.
* Paired differences against the ResNet reference are computed inside make_panel.py (on the targets where the ResNet value exists) and are not recomputed by the script.
* Training itself: it needs the Cell Painting images and a GPU cluster. The job specs in jobs/ and the parameter files in params/ document how each run was launched.

## Authorship and limits

This verifier and the documentation around it were written with AI assistance (Claude, Anthropic) together with the project author. It checks internal consistency: that the numbers stated in the README and the documents agree with the result files committed here. It does not prove independently that the result files are correct; for that the training would have to be repeated (see above). The collaborator models (EfficientNet-B3 and ResNet-18, by Bhartendu Parihar) were re-scored from his checkpoints with cv_rescore.py; the checks done and the caveats are in results_showcase/efficientnet_b3/README.md and results_showcase/resnet18/README.md (ResNet-18 used early stopping and ResNet-50 did not; his EfficientNet-B3 448 px text file does not match his checkpoints).

## Notes for a reviewer

* Label coding in test_labels.csv: 1 = positive, -1 = negative, 0 = not measured (see mean_roc_auc in utils/metrics.py). Per-compound scores are the mean of the sigmoid probabilities over a compound's images; ROC-AUC is computed per assay on measured labels and averaged over assays that have both classes. verify_eval.py recomputes this for one model and fold and compares it with the reported per-assay values.

* The intervals in bootstrap_ci.csv and enrich_r65.csv differ in the second decimal for some methods (for example Cell-DINO grad x act, 2.26 vs 2.29). Both are bootstrap estimates of the same quantity with different random draws. The README and documents quote enrich_r65.csv.
* Known errors found and corrected during the project are listed in CORRECTIONS.md.
