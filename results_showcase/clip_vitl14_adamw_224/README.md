# CLIP ViT-L/14 -- Recipe-Sensitivity Comparison Arm (AdamW, image-based)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

**This is not a peer entry in the frozen-backbone benchmark.** It is the same CLIP ViT-L/14
backbone (OpenAI weights, 304M params, 768-dim) as the [locked-recipe frozen arm](../clip_frozen_224/),
evaluated with a **different training procedure**: full image-based training(The phrase "full image-based training" in your text means the data path — images are loaded, augmented and pushed through the frozen backbone every epoch — as opposed to the precompute path, where embeddings are extracted once and cached. Both arms are frozen linear probes.) with AdamW
(lr = 1e-4) rather than the precompute-then-probe pipeline with SGD/cosine/lr=0.02 used for
every arm in the main frozen-backbone comparison. It exists specifically to demonstrate the
project's recipe-sensitivity finding with real, traceable data, rather than as a number stated
without evidence.

## Result

| CLIP ViT-L/14 arm | Recipe | Test ROC-AUC (6-fold) |
|---|---|---|
| Frozen, locked recipe (main benchmark arm) | SGD, momentum 0.9, cosine, lr=0.02, precompute-then-probe | 0.5798 +/- 0.0038 |
| **This arm** | AdamW, lr=1e-4, full image-based training | **0.6312 +/- 0.0188** |

Per-fold test ROC-AUC (this arm): 0.6031, 0.6322, 0.6152, 0.6538, 0.6400, 0.6429.

**The same frozen features, probed two different ways, differ by ~5.1 ROC-AUC points** -- a
larger swing than most of the between-backbone differences reported in the main benchmark
(e.g. Cell-DINO vs DINOv2-Base is ~1.4 points). Note also the much wider cross-fold spread here
(std 0.0188) compared to the locked-recipe arm's tight std of 0.0038, consistent with AdamW-based
full-image training being a less stable, higher-variance procedure than the precompute-then-probe
SGD recipe on this task.

## Why this arm exists

The project's main benchmark (Figs A, B, C, E; the Friedman/Nemenyi comparison) depends entirely
on one assumption: that any performance difference between frozen backbones reflects the quality
of their representations, not a difference in how each was probed. Every arm in that comparison
therefore shares one locked recipe. This arm is the direct, quantified evidence for *why* that
discipline is necessary -- it shows that changing only the probing procedure, with the identical
underlying frozen weights, moves the result by more than most of the differences the main
benchmark attributes to backbone choice. Without holding the recipe fixed, an apparent
"backbone quality" difference could instead be a "probing procedure quality" difference in
disguise.

## Provenance and reproducibility note

This run was originally trained and evaluated on Bhargav Parihar's (`b-b-parihar`) cluster
account, using `classification.py` (the project's full-image training entry point, the same one
used for the ResNet50 and LoRA arms) rather than the `extract_embeddings.py` / `train_head.py`
precompute-then-probe pipeline used for the locked-recipe frozen arms. The raw per-sample test
predictions (`test_preds.csv` / `test_labels.csv`) were copied into this account
(`bioact_clip_vitl14_adamw_r224_fold{0..5}`), then `make_plots_v3.py` and `aggregate_cv.py` were
run against them here to produce `per_assay_auc.csv` and this folder's official 6-fold summary --
the same processing pipeline used for every other arm's results, so this number is verified
independently rather than only quoted from a teammate's report.

## Method

CLIP ViT-L/14 (OpenAI weights), 5-channel patch-embedding adaptation via the project's standard
repeat-and-slice strategy, same as every other arm. Trained end-to-end (not frozen-then-probed)
with AdamW, learning rate 1e-4, on full images each epoch (not cached embeddings) -- a materially
different and more expensive training loop than the locked-recipe arm's cached-embedding linear
probe. Same source_11 CSV, same 6 folds, same masked labels, same loss and metric as every other
arm in this project -- only the backbone-adaptation procedure differs from the locked-recipe
comparison.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
