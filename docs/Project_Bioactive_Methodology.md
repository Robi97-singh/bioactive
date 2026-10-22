# Project Bioactive: Methodology and Results

Oct 21, 2026 · Robin Singh

## Task and data

The project predicts compound bioactivity from Cell Painting microscopy images, as a replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024). The task is masked multi-label classification over 29 assays: each compound-assay label is +1 (active), -1 (inactive) or 0 (not tested), and the loss ignores untested entries.

- **Data:** five-channel fluorescence images from the public JUMP-CP collection, joined with ChEMBL bioactivity labels by `data_prep/prep_paper_faithful.py`.
- **Split:** compounds are clustered with the Butina algorithm and assigned to six folds at compound level by `data_prep/split_paper_faithful.py`, so similar compounds never sit in both training and test.
- **Brightfield extension:** brightfield TIFFs are located from per-plate load-data files, downloaded from the public JUMP-CP S3 bucket, percentile-normalized and saved as one-channel PNGs, with real per-channel mean and standard deviation computed from the downloaded images.
- **Reference point:** the public benchmark reports a mean ROC-AUC of 0.660 ± 0.094 over six folds. It is kept as an external reference and is not one of this project's own rows.

## Models and training recipes

Three regimes are compared on an identical split and evaluation protocol.

- **Fine-tuned ResNet-50** at 224 and 448 px, trained and then tested separately for each fold with `classification.py`. This is the supervised baseline of the reference paper.
- **Frozen backbones with a linear probe.** Each backbone is run once per split by `extract_embeddings.py` and the embeddings are cached. `train_head.py` then trains a linear head on the cached embeddings. Backbones: DINOv2 (Small, Base, Large), DINOv3-Base, Cell-DINO, CLIP ViT-L/14 and BiomedCLIP, at 224 px and, for the DINO family, 448 px. The head uses the standardized CLS embedding.
- **LoRA adaptation** of DINOv2-Small at 224 px, with 2.4% of the parameters trainable.

One locked linear-probe recipe is used for every frozen backbone. Its learning rate (0.02) was selected in a fold-0 sweep on Cell-DINO and applied unchanged to all the others. ResNet-50 and LoRA use individually tuned recipes, which favours them; this is stated as a caveat wherever the results are compared.

## Evaluation protocol

Every result is the mean ROC-AUC over the 29 assays on the held-out test fold, averaged over six cross-validation folds, with predictions aggregated to compound level to match the reference paper.

- **Aggregation:** several images per compound are combined into one prediction per compound before the AUC is computed.
- **Significance:** models are compared with a Friedman test and Nemenyi post-hoc comparisons. Two comparisons are reported: a 9-arm test at 224 px (six frozen probes plus ResNet-50, ResNet-18 and EfficientNet-B3; the 29 assays are the blocks, each averaged over the six folds) and a more conservative 16-arm test over the six folds that adds the 448 px arms (Nemenyi critical difference 9.42 ranks).
- **Correction of the frozen-backbone protocol:** the frozen-backbone evaluation was corrected on 3 October 2026, and the full benchmark was re-run at the locked recipe and verified on 6 October 2026. All numbers in this document come from the re-run and supersede the earlier ones. Before the correction, every frozen-backbone result was scored on raw per-image predictions, without averaging the site images of a compound, while the fine-tuned CNN pipeline had always aggregated to compound level. Because every assay label is a compound-level property, scoring each image as an independent case was pseudo-replication and left per-image noise in the AUC, so all frozen-backbone results were under-reported by about 0.04 to 0.05. The problem was found on 3 October 2026 while building the matched fluorescence baseline for the brightfield ablation and noticing that the two evaluation code paths diverged.

Size of the correction for the frozen probes (mean test ROC-AUC, per-image scoring against compound-level scoring):

| Model | Per-image (uncorrected) | Compound-level (corrected) | Difference |
| --- | --- | --- | --- |
| DINOv2-Large, 448 px | 0.5868 | 0.6412 | +0.0544 |
| DINOv3-Base, 448 px | 0.5863 | 0.6398 | +0.0535 |
| DINOv2-Base, 448 px | 0.5842 | 0.6371 | +0.0529 |
| Cell-DINO, 224 px | 0.5892 | 0.6333 | +0.0441 |
| Cell-DINO, 448 px | 0.5899 | 0.6333 | +0.0434 |
| DINOv2-Large, 224 px | 0.5783 | 0.6299 | +0.0516 |
| DINOv3-Base, 224 px | 0.5761 | 0.6281 | +0.0520 |
| DINOv2-Base, 224 px | 0.5749 | 0.6272 | +0.0523 |
| DINOv2-Small, 224 px | 0.5761 | 0.6259 | +0.0498 |
| CLIP ViT-L/14, 224 px | 0.5785 | 0.6257 | +0.0472 |

BiomedCLIP is omitted from this table; its corrected value (0.6214) is in the benchmark table.

## Benchmark results

Fine-tuned CNNs are the strongest arms: ResNet-50 at 448 px reaches a mean test ROC-AUC of 0.679, with ResNet-18 (0.676) and EfficientNet-B3 (0.669) at 448 px within one fold-to-fold SD of it, against 0.621 to 0.641 for the frozen probes. Mean test ROC-AUC over 29 assays, six-fold cross-validation, compound-level aggregation:

| Regime | Model | Resolution (px) | Test ROC-AUC (mean ± SD) |
| --- | --- | --- | --- |
| Fine-tuned | ResNet-50 | 448 | 0.6792 ± 0.0245 |
| Fine-tuned | ResNet-18 | 448 | 0.6759 ± 0.0218 |
| Fine-tuned | EfficientNet-B3 | 448 | 0.6692 ± 0.0118 |
| Fine-tuned | ResNet-50 | 224 | 0.6638 ± 0.0153 |
| Fine-tuned | EfficientNet-B3 | 224 | 0.6622 ± 0.0254 |
| Fine-tuned | ResNet-18 | 224 | 0.6611 ± 0.0145 |
| Frozen probe | DINOv2-Large | 448 | 0.6412 ± 0.0170 |
| Frozen probe | DINOv3-Base | 448 | 0.6398 ± 0.0192 |
| LoRA (2.4% trainable) | DINOv2-Small | 224 | 0.6373 ± 0.0203 |
| Frozen probe | DINOv2-Base | 448 | 0.6371 ± 0.0124 |
| Frozen probe | Cell-DINO | 224 | 0.6333 ± 0.0115 |
| Frozen probe | Cell-DINO | 448 | 0.6333 ± 0.0132 |
| Frozen probe | DINOv2-Large | 224 | 0.6299 ± 0.0218 |
| Frozen probe | DINOv3-Base | 224 | 0.6281 ± 0.0159 |
| Frozen probe | DINOv2-Base | 224 | 0.6272 ± 0.0190 |
| Frozen probe | DINOv2-Small | 224 | 0.6259 ± 0.0121 |
| Frozen probe | CLIP ViT-L/14 | 224 | 0.6257 ± 0.0160 |
| Frozen probe | BiomedCLIP | 224 | 0.6214 ± 0.0127 |

ResNet-18 and EfficientNet-B3 were configured and trained by Bhartendu Parihar; their numbers here are re-computed from his saved checkpoints at compound level with cv\_rescore.py. His ResNet-18 runs used early stopping, the ResNet-50 reference did not. For EfficientNet-B3 at 448 px his own text file reports a lower mean (about 0.661) than his checkpoints give (0.669); the checkpoint-derived value is used and the cause of the difference is not established. Both significance tests below include ResNet-18 and EfficientNet-B3 (the 9-arm and 16-arm tests); the statement that ResNet-50 ranks first in every fold holds for the original arms only.

The public benchmark of the reference paper reports 0.660 ± 0.094 and is a different number from every row above.

- **Fine-tuned CNNs** are the strongest arms and are not separable from one another. Among the original arms ResNet-50 at 448 px ranks first in all six folds; with ResNet-18 and EfficientNet-B3 included the best of the 16 arms varies by fold (ResNet-50 ranks 4, 3, 2, 3, 1 and 1). In the 9-arm test at 224 px (Friedman chi-squared 67.2, p = 2e-11) all three are significantly better than DINOv2-Base (p <= 0.002), DINOv3-Base (p <= 0.001), CLIP ViT-L/14 (p <= 0.001) and BiomedCLIP (p < 0.001), marginally better than DINOv2-Large (p = 0.019 to 0.048), and not significantly different from Cell-DINO (p = 0.40 to 0.60) or from each other (p = 1.0). In the 16-arm fold test (chi-squared 71.8, p = 2e-9) the three 448 px CNNs rank 1 to 3 and separate from the weakest frozen probes (DINOv2 r224, DINOv3-Base r224, DINOv2-Small, CLIP, BiomedCLIP) but not from the best frozen 448 px probes.
- **Frozen probes** are statistically tied (no pair separates in the 9-arm test; the largest gap, Cell-DINO vs BiomedCLIP, is p = 0.15), so neither pretraining domain nor model scale gives a reliable advantage.
- **The 16-arm test** over the six folds (chi-squared 71.8, p = 2e-9, critical difference 9.42 ranks) is more conservative. ResNet-18, ResNet-50 and EfficientNet-B3 at 448 px rank 1 to 3 (mean ranks 2.17, 2.33, 2.50) and each separates from DINOv2 at 224 px, DINOv3-Base at 224 px, DINOv2-Small, CLIP and BiomedCLIP; ResNet-18 at 448 px also separates from DINOv2-Large at 224 px, and ResNet-18 and EfficientNet-B3 at 224 px (rank 4.0 each) separate from BiomedCLIP. None of the CNNs separates from the best frozen 448 px probes or Cell-DINO.
- **Resolution** has a small positive effect, 224 to 448 px: DINOv2-Base +0.010, DINOv2-Large +0.011, DINOv3-Base +0.012, Cell-DINO 0.000 and fine-tuned ResNet-50 +0.015, all within the fold-to-fold standard deviation of 0.01 to 0.02.
- **LoRA** DINOv2-Small (0.6373) is 0.011 above its frozen backbone and level with the best frozen 448 px probes. It is not part of the significance tests.

## Brightfield vs fluorescence

Brightfield and fluorescence ResNet-50 models reach statistically indistinguishable accuracy: the six-fold mean test AUC is 0.694 for brightfield and 0.710 for fluorescence (p = 0.19).

- **Setup:** a one-channel brightfield ResNet-50 and a five-channel fluorescence ResNet-50 were trained at 448 px on the same 102 plates, one job per fold, over all six folds, with the same split and compound-level aggregation. Each fluorescence fold took roughly 27 to 41 hours.
- **Difference:** brightfield minus fluorescence is -0.016 AUC (paired t-test p = 0.19; brightfield better on 2 of 6 folds; rough 95% range -0.042 to +0.011).
- **Reading:** fluorescence is ahead on four of the six folds (folds 2 to 5, by 0.013 to 0.056) and brightfield on folds 0 and 1 (by 0.010 and 0.012), so a small fluorescence advantage is not ruled out. Against the frozen models on brightfield, the fine-tuned ResNet-50 at 448 px is ahead of frozen DINOv3 at 448 px by 0.0353 (p = 0.017, better on 6 of 6 folds) and of frozen Cell-DINO at 224 px by 0.1058 (p = 0.0001, 6 of 6 folds).
- **Comparability:** these numbers use the 102-plate subset and are not comparable with the full-dataset ResNet-50 rows in the benchmark table.

## Explainability of the frozen models

On 65 randomly sampled targets, occlusion is the only explanation method that clearly beats image-independent control maps, and none of the ViT methods is both faithful and focused on cells.

- **Models and methods:** the frozen DINOv3-Base (448 px) and Cell-DINO (224 px) linear heads were explained with ViT-CX (Xie et al. 2023, the approved primary method), occlusion, ViT Grad-CAM and gradient-times-activation, and compared with ResNet-50 Grad-CAM as the reference. The explained score is the assay logit, because the head's probability varies only about 0.72 to 0.74 across images.
- **Faithfulness:** deletion and insertion AUC over 20 steps, each standardized (z) against 10 smooth random maps. Lower is better for deletion and higher for insertion. Two image-independent control maps (a fixed centre blob and its inverse) are scored the same way, because large contiguous deletions hurt the score more than scattered random ones.
- **Cell focus:** enrichment of the map on the brightest 3% of pixels relative to its mean, where 1.0 is chance. Gradient-times-activation maps are mechanically favoured by it.
- **Targets:** 65 (compound, assay) pairs sampled from the ResNet's out-of-fold predictions without looking at confidence, 13 per fold over folds 0 to 4, 30 positives and 35 negatives, 27 assays. By eye, 57 images show normal cell fields, 6 are sparse and 2 are noise. Intervals are 95% bootstrap intervals over targets.

| Model | Method | Deletion z | Insertion z | Cell enrichment |
| --- | --- | --- | --- | --- |
| DINOv3 | ViT-CX | -0.95 (-1.74 to -0.22) | -0.84 (-1.58 to -0.11) | 1.30 (1.07 to 1.53) |
| DINOv3 | grad x act | -1.32 (-2.00 to -0.69) | 1.20 (0.64 to 1.77) | 3.27 (2.77 to 3.81) |
| DINOv3 | occlusion | -2.63 (-3.42 to -1.88) | 2.57 (1.87 to 3.28) | not computed |
| DINOv3 | centre control | -0.40 (-1.38 to 0.58) | 0.32 (-0.43 to 1.08) | not applicable |
| Cell-DINO | ViT-CX | -1.21 (-2.22 to -0.01) | 0.73 (-0.17 to 1.65) | 1.01 (0.93 to 1.09) |
| Cell-DINO | grad x act | -1.53 (-2.33 to -0.77) | 0.74 (0.15 to 1.37) | 2.29 (1.62 to 3.14) |
| Cell-DINO | occlusion | -5.03 (-6.11 to -4.08) | 2.21 (1.46 to 2.99) | 1.07 (0.98 to 1.16) |
| Cell-DINO | centre control | -0.51 (-1.57 to 0.60) | 0.09 (-0.59 to 0.78) | not applicable |
| ResNet-50 | Grad-CAM (reference) | not evaluated | not evaluated | 1.07 (0.90 to 1.28), 58 of 65 targets |

- **Occlusion** is the most faithful under this metric, but deletion and insertion perturb pixels the same way occlusion does, so the metric favours it by construction. Its Cell-DINO maps are at chance on cell enrichment.
- **ViT-CX** is not clearly better than a plain centre blob on deletion, and its DINOv3 insertion score is worse than random.
- **Gradient-times-activation** has the highest enrichment, but this is partly mechanical.
- **ResNet Grad-CAM** is at chance on enrichment. Its faithfulness was not evaluated, so the comparison with it is limited to cell enrichment.
- **A 16-target pilot** gave much stronger numbers (DINOv3 ViT-CX deletion z -7.5). Those targets were the ResNet's most confident cases and half were noise images, so the pilot overstated the methods and is not used for conclusions.

## Limitations and reproduction

The benchmark ranking is robust at the level of "fine-tuned CNNs first, frozen probes tied", but several design choices limit what can be concluded.

- **Unequal tuning:** the frozen recipe (learning rate 0.02) was selected on fold 0 with Cell-DINO and applied unchanged to all frozen backbones, whereas ResNet-50 and LoRA use individually tuned recipes.
- **Cell-DINO input:** its wrapper resizes images to 128 px internally, so its explanation maps are coarse.
- **DINOv3 embeddings:** they were extracted under another cluster account, and the training-set count differs slightly (111,968 against 111,984).
- **Explainability intervals** cover target sampling only. They do not reflect the choice of images or of method settings, and image groups were labelled by eye.
- **Metrics disagree:** perturbation scores favour perturbation methods, and activation-weighted maps are favoured by the enrichment score, in line with Stassin et al. 2024, who found that explanation metrics do not converge.
- **Not evaluated:** deletion and insertion scores for the ResNet Grad-CAM reference.
- **Brightfield comparison** uses a 102-plate subset and is not comparable with the full-dataset rows.

All code is in the repository `Robi97-singh/bioactive` (explainability pipeline and 65-target replication added in commit `ee54038`). The README lists every script, its role and the exact commands for data preparation, training, embedding extraction, head training, aggregation and figures. The headline numbers and every row of the benchmark table can be re-checked from the committed result files with verify/verify\_claims.py (65 checks; an AI-assisted consistency check, not an independent proof, see verify/VERIFY.md), and errors found in our own work are listed in verify/CORRECTIONS.md.
