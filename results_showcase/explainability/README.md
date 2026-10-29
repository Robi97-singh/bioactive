# Explainability of the frozen DINOv3 and Cell-DINO heads

Explanations of the frozen DINOv3-Base (448 px) and Cell-DINO (224 px) linear heads, compared with ResNet-50 Grad-CAM as the reference. Methods: ViT-CX, occlusion, ViT Grad-CAM, gradient x activation (grad x act). Score explained: the assay logit.

Metrics: deletion and insertion AUC (20 steps), each standardized (z) against 10 smooth random maps (deletion: lower is better; insertion: higher is better); two image-independent control maps (centre blob, border) scored the same way; cell enrichment = mean of the map on the brightest 3% of pixels divided by its mean (1.0 = chance). Intervals are 95% bootstrap intervals over targets.

## Folders

* `random_65_targets/` is the main result. 65 randomly sampled (compound, assay) targets (13 per fold, folds 0-4, 30 positives, 35 negatives, 27 assays). panel_1 to panel_9 (jpg) show side-by-side heat maps (ResNet Grad-CAM and six frozen-model methods, 8 targets per panel). agg_r65.csv has faithfulness by method and group, enrich_r65.csv and enrichment_per_target.csv have cell enrichment, bootstrap_ci.csv the intervals, targets_random65.csv the target list, and contact_sheets/ the numbered images used to label image types by eye (57 normal, 6 sparse, 2 noise).
* `pilot_16_targets/` is an earlier pilot on 16 targets that were the ResNet's most confident cases (half were noise images). It overstated every method and is kept for transparency only. It holds report_figure.png (three example images), panel_1 and panel_2 (jpg; heat maps of ResNet Grad-CAM and six frozen-model methods: DINOv3 occlusion, ViT-CX and grad x act, Cell-DINO occlusion, ViT-CX and grad x act) and CSVs with intervals and the image-type split. DINOv3 occlusion on the 16 pilot targets: deletion z -4.85 (-8.01 to -2.22), insertion z -1.01 (-4.48 to 2.09), enrichment 0.96 (0.84 to 1.08).

## Main result (65 random targets)

| Model | Method | Deletion z | Insertion z | Cell enrichment |
|---|---|---|---|---|
| DINOv3 | ViT-CX | -0.95 (-1.74 to -0.22) | -0.84 (-1.58 to -0.11) | 1.30 (1.07 to 1.53) |
| DINOv3 | grad x act | -1.32 (-2.00 to -0.69) | 1.20 (0.64 to 1.77) | 3.27 (2.77 to 3.81) |
| DINOv3 | occlusion | -2.63 (-3.42 to -1.88) | 2.57 (1.87 to 3.28) | 0.94 (0.89 to 0.99) |
| DINOv3 | centre control | -0.40 (-1.38 to 0.58) | 0.32 (-0.43 to 1.08) | not applicable |
| Cell-DINO | ViT-CX | -1.21 (-2.22 to -0.01) | 0.73 (-0.17 to 1.65) | 1.01 (0.93 to 1.09) |
| Cell-DINO | grad x act | -1.53 (-2.33 to -0.77) | 0.74 (0.15 to 1.37) | 2.29 (1.62 to 3.14) |
| Cell-DINO | occlusion | -5.03 (-6.11 to -4.08) | 2.21 (1.46 to 2.99) | 1.07 (0.98 to 1.16) |
| Cell-DINO | centre control | -0.51 (-1.57 to 0.60) | 0.09 (-0.59 to 0.78) | not applicable |
| ResNet-50 | Grad-CAM (reference) | not evaluated | not evaluated | 1.07 (0.90 to 1.28), 58 of 65 targets |

Paired difference in enrichment against ResNet Grad-CAM, on the 58 targets where the ResNet map is not empty (mean, 95% CI): DINOv3 ViT-CX +0.27 (-0.07 to +0.59); Cell-DINO ViT-CX -0.08 (-0.27 to +0.11); Cell-DINO occlusion 0.00 (-0.19 to +0.17); DINOv3 occlusion -0.12 (-0.33 to +0.07); DINOv3 grad x act +2.14 (+1.59 to +2.74); Cell-DINO grad x act +1.16 (+0.50 to +1.94).

Reading: only occlusion clearly beats the control maps on both deletion and insertion (the metric favours perturbation methods by construction). ViT-CX is not clearly better than a centre blob on deletion. Grad x act has the highest enrichment, but this is partly mechanical because the map is weighted by activation. ResNet Grad-CAM is at chance on enrichment (defined for 58 of 65 targets; in 7 the Grad-CAM map was empty); its deletion and insertion scores were not evaluated.

## How to regenerate

All scripts are in the repository root and describe their arguments in their docstrings. Inputs (images, embeddings, checkpoints) are not in the repository. Order: sample_targets.py (target sampling), run_vitcx.py (explanations and metrics, launched by jobs/r65-*.yaml with METHOD=vitcx, occlusion, gradcam or centre), run_gradcam_r65.py (ResNet reference, jobs/r65-gcam-resnet.yaml), contact_sheet.py (numbered images for labelling), agg_r65.py (faithfulness tables), r65_enrich.py and r65_resnet.py (enrichment tables), make_panel.py (heat-map panels and intervals).

Image groups (normal, sparse, noise) were labelled by eye from the contact sheets and are encoded in r65_enrich.py and r65_resnet.py.
