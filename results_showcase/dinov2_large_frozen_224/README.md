DINOv2-Large — Frozen Linear-Probe Arm (Scale Ladder)
Replication and extension of Fredin Haslum et al., Nature Communications 15:3470 (2024), on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Large (Meta, ViT-L/14, ImageNet-pretrained, 1024-dim) evaluated frozen with the same linear-probe recipe as every other frozen arm. This is the scale-ladder control: DINOv2-Base and DINOv2-Large share the same architecture family and the same ImageNet pretraining, differing only in parameter count (86M -> 304M, 3.5x). It answers whether scale alone - independent of pretraining domain or data diversity - improves frozen linear-probe performance. Public benchmark reference: 0.660 +/- 0.094.

Result -- scale alone does not help, frozen
Backbone @ 224 | Pretraining | Params | Embedding dim | Test ROC-AUC (6-fold)
--- | --- | --- | --- | ---
DINOv2-Base (frozen) | ImageNet (generic) | 86M | 768 | 0.5796 +/- 0.0067
DINOv2-Large (frozen) | ImageNet (generic) | 304M | 1024 | 0.5790 +/- 0.0066
Cell-DINO (frozen, ref) | Cell Painting (domain) | 21.5M | 384 | 0.5932 +/- 0.0065
ResNet50 (fine-tuned, ref) | ImageNet | 25M | -- | 0.6638 +/- 0.0153

DINOv2-Large per-fold test ROC-AUC: 0.5781, 0.5763, 0.5683, 0.5878, 0.5823, 0.5812.

Paired per-fold comparison (DINOv2-Base - DINOv2-Large): +0.0049, -0.0014, +0.0005, -0.0047, +0.0050, -0.0009. Base wins 3 of 6 folds, Large wins 3 of 6 -- an even split, mean difference +0.0006 (Base marginally ahead, well within noise).

What this shows
1. Scale alone does not improve frozen linear-probe performance within a fixed architecture and pretraining family. A 3.5x increase in parameters (86M -> 304M) produces no measurable gain -- the two models are statistically indistinguishable (Nemenyi p=0.997, see the project-wide benchmark figures) and split folds evenly.
2. This refines the earlier scale finding from the CLIP-L comparison. CLIP-L (304M) beats Cell-DINO (21.5M) despite the domain-matching result favoring Cell-DINO over DINOv2-Base at comparable scale -- but CLIP-L differs from Cell-DINO in scale, pretraining domain, AND pretraining data volume simultaneously, so that comparison could not isolate which factor mattered. DINOv2-Large holds architecture and pretraining domain fixed and only varies size, cleanly showing that parameter count in isolation is not the active ingredient -- CLIP-L's advantage more likely comes from its far larger and more diverse pretraining data, not raw capacity.
3. Neither scale nor size closes the gap to domain-matched pretraining or to fine-tuning. DINOv2-Large still trails Cell-DINO (21.5M, cell-pretrained) by ~1.4 points despite having 14x more parameters, and both remain well below the fine-tuned ResNet50 (0.6638). For frozen feature extraction on cell images, what the backbone was pretrained on continues to matter more than how big it is.

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
