DINOv2-Large — Frozen Linear-Probe Arm (Scale Ladder)
Replication and extension of Fredin Haslum et al., Nature Communications 15:3470 (2024), on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Large (Meta, ViT-L/14, ImageNet-pretrained, 1024-dim) evaluated frozen with the same linear-probe recipe as every other frozen arm. This is the scale-ladder control: DINOv2-Base and DINOv2-Large share the same architecture family and the same ImageNet pretraining, differing only in parameter count (86M -> 304M, 3.5x). It answers whether scale alone - independent of pretraining domain or data diversity - improves frozen linear-probe performance. Public benchmark reference: 0.660 +/- 0.094.

Result -- scale alone still does not help, frozen

Backbone @ 224 | Pretraining | Params | Embedding dim | Test ROC-AUC (6-fold)
--- | --- | --- | --- | ---
DINOv2-Base (frozen) | ImageNet (generic) | 86M | 768 | 0.6345 +/- 0.0162
DINOv2-Large (frozen) | ImageNet (generic) | 304M | 1024 | 0.6313 +/- 0.0202
Cell-DINO (frozen, ref) | Cell Painting (domain) | 21.5M | 384 | 0.6373 +/- 0.0119
ResNet50 (fine-tuned, ref) | ImageNet | 25M | -- | 0.6638 +/- 0.0153

DINOv2-Large per-fold test ROC-AUC: 0.6177, 0.6271, 0.6006, 0.6543, 0.6480, 0.6405.

Paired per-fold comparison (DINOv2-Base - DINOv2-Large): +0.0075, -0.0034, +0.0148, -0.0015, +0.0068, -0.0056. Base wins 3 of 6 folds, Large wins 3 of 6 -- still an even split, mean difference +0.0032 (Base marginally ahead, well within noise). Nemenyi: Base vs Large p=1.0000 (not significant).

Note (2026-10-03): numbers corrected for compound-level aggregation. The even 3-3 fold split and "scale doesn't matter" conclusion are unchanged; only the absolute numbers and the Nemenyi p-value (now 1.0000, even more clearly non-significant) are updated.

What this shows
1. Scale alone still does not improve frozen linear-probe performance within a fixed architecture and pretraining family. A 3.5x increase in parameters (86M -> 304M) produces no measurable gain -- the two models remain statistically indistinguishable (Nemenyi p=1.0000) and still split folds evenly.
2. The comparison to CLIP-L's scale finding still holds, with updated numbers. CLIP-L (304M, 0.6267) remains behind Cell-DINO (21.5M, 0.6373) despite the large size difference, though neither gap is statistically significant under the corrected Nemenyi results -- scale in isolation still does not appear to be the active ingredient behind any backbone's ranking in this benchmark.
3. Neither scale nor size closes the gap to domain-matched pretraining or to fine-tuning, though these gaps are now much smaller. DINOv2-Large still trails Cell-DINO by ~0.6 points (was ~1.4) despite 14x more parameters -- and this difference is not significant (Nemenyi p=0.7693) -- and both remain modestly below the fine-tuned ResNet50 (0.6638), with DINOv2-Large vs ResNet itself only borderline significant (p=0.0546).

Method
Identical locked recipe to every other frozen arm, applied without change: frozen backbone, cached CLS embeddings, Linear(D, 29) head (D = 1024, auto-sized), masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr = 0.02, train-only feature standardization, early stopping on validation ROC-AUC (patience 6), mean_roc_auc metric. Same source_11 CSV, same 6 folds, same masked labels, same loss and metric as every other arm -- only the backbone differs.

DINOv2-Large specifics: loaded from HuggingFace (facebook/dinov2-large), sharing the same DINOv2Classifier class and forward pass as DINOv2-Base -- only the backbone_type string differs, so all wiring (patch-embedding channel adaptation, CLS-token extraction, position-embedding handling) is identical and already-validated code, not a new implementation. ViT-L/14 at 224 = 256 patch tokens, same grid as Base. 3-channel ImageNet patch embedding adapted to 5 channels by repeating and slicing the RGB projection weights, identical strategy to every other adapted backbone in this project.

Note on comparability: unlike the Cell-DINO vs DINOv2-Base comparison (which differs in domain, architecture family, patch size, and embedding dim all at once), DINOv2-Base vs DINOv2-Large is a clean single-variable comparison -- same architecture family, same patch size (14), same pretraining data (ImageNet), only parameter count and embedding dimension differ as a consequence of scale.

Files
cv_summary.csv -- per-fold and mean +/- std test ROC-AUC
cv_fold_spread.png -- per-fold test ROC-AUC
cv_per_assay.png / .csv -- mean per-assay ROC-AUC across folds
head_val_auc_overlay.png -- validation ROC-AUC per epoch, one line per fold
head_val_auc_meanband.png -- mean validation ROC-AUC across folds (+/-1 std)
