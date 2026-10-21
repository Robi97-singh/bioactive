# Corrections log

Errors and overstated claims found in this project's own work, with what changed. Listed so that a reviewer finds them here first.

| Date | What was wrong | Effect | Fix |
|---|---|---|---|
| 2026-10-03 | train_head.py scored frozen-model heads per image instead of per compound. | Frozen-probe ROC-AUC was under-reported by about 0.04 to 0.05. | Scoring fixed; all frozen heads re-run; re-run verified 2026-10-06. README and documents use the corrected numbers. |
| 2026-10 | Explainability pilot used 16 targets that were the ResNet's most confident cases, half of them noise images. | The pilot overstated every explanation method. | The 65 randomly sampled targets are the primary result; the pilot is kept for transparency only. |
| 2026-10 | An automatic rule for labelling image types (normal, sparse, noise). | Not reliable. | Dropped; the 65 images were labelled by eye from the contact sheets. |
| 2026-10 | A statement that the frozen head uses only intensity. | Not supported by the experiments. | Retracted. |
| 2026-10 | A runtime estimate of 1 to 2 hours per fold. | Wrong guess. | Corrected in the documents. |
| 2026-10-21 | README said fluorescence was ahead on "the last three folds". | Understated: fluorescence leads on four of six folds (folds 2 to 5, by 0.013 to 0.056). | Corrected in README and both documents. The difference is still not significant (p = 0.1864). |
| 2026-10-21 | ResNet Grad-CAM enrichment was reported for n = 65 targets. | The value is undefined for 7 targets (empty Grad-CAM map); the 1.07 (0.90 to 1.28) rests on 58. | Corrected in the explainability README and both documents; enrich_resnet_per_target.csv committed; verify_claims.py checks 65 rows and 58 valid values. |

Open item: the GitHub repository description still quotes a ResNet-50 baseline of 0.702, while the README reports 0.6792 +- 0.0245 for the full benchmark.
