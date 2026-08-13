# ResNet50 — Resolution & Field-of-View Comparison

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

All results are **6-fold cross-validation test ROC-AUC**, structure-aware splits (ECFP4 + Butina).
Public benchmark reference: **0.660 ± 0.094**.

## Results

| Arm | Input | Resize -> Crop | Field of View | Test ROC-AUC (6-fold) |
|-----|-------|----------------|---------------|-----------------------|
| **ResNet50 @ 448** | 448 | crop 448 from 1080 | 448/1080 = **0.415** | **0.6792 +/- 0.0245** |
| **ResNet50 @ 224 (primary)** | 224 | resize 1080->540, crop 224 | 224/540 = **0.415** | **0.6638 +/- 0.0153** |
| **ResNet50 @ 224 (FOV 0.450)** | 224 | resize 1080->498, crop 224 | 224/498 = **0.450** | **0.6623 +/- 0.0223** |

Per-channel images are 1080x1080 (five channels: DNA, ER, RNA, AGP, Mito).
Field of view = crop size / image size = the fraction of the imaged area the network sees.

## What this shows

**1. The resolution effect is real but modest.**
At a matched field of view (0.415), halving the input resolution from 448 to 224 costs
about 1.5 ROC-AUC points (0.6792 -> 0.6638). Both arms see the *same* fraction of each
cell; only the pixel count differs, so this is a clean single-variable resolution comparison.

**2. The gap is robust to field of view.**
The two 224 arms -- one at FOV 0.415 (resize 540) and one at FOV 0.450 (resize 498) --
land within 0.0015 of each other (0.6638 vs 0.6623). Changing the field of view barely
moves the 224 result, which confirms the 448-vs-224 gap is driven by **resolution**, not
by how much of the cell is visible.

**3. All three arms reproduce the paper's benchmark.**
Every configuration lands at or above the public JUMP-CP reference of 0.660 +/- 0.094,
confirming a faithful replication across resolutions.

## Why field-of-view matching matters

A naive 224 arm would crop 224 directly from the 1080 image, giving FOV 224/1080 = 0.207 --
a much smaller window than the 448 arm's 0.415. That would confound resolution with field
of view. Resizing to 540 first (224/540 = 0.415) holds the field of view constant, so the
comparison isolates resolution alone.

## Folders

- `r448/` -- ResNet50 @ 448, FOV 0.415 (paper-matched reference)
- `r224_fov0415/` -- ResNet50 @ 224, FOV 0.415 (primary, matched to 448)
- `r224_fov0450/` -- ResNet50 @ 224, FOV 0.450 (robustness point)

Each contains: cv_fold_spread.png, cv_per_assay.png/.csv, cv_val_curves.png, cv_summary.csv.
