# DINOv2 + LoRA -- Parameter-Efficient Adaptation Arm @ 224 (Full 6-fold CV)

Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024),
on the public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).

DINOv2-Small (ViT-S/14, ImageNet-pretrained) adapted with LoRA (rank=8, alpha=16, dropout=0.05,
target_modules=[query, value]) on the query/value attention projections, plus a re-adapted
5-channel patch embedding kept fully trainable. Only 0.54M of 22.4M parameters (2.4%) are
trained; the rest of the backbone stays frozen. This is the middle point of the adaptation
spectrum: frozen probe -> LoRA -> full fine-tuning. Public benchmark reference:
**0.660 +/- 0.094**.

## Result -- cheap adaptation closes most of the frozen-to-fine-tuned gap

| Regime | Model | Trainable params | Test ROC-AUC (6-fold) |
|--------|-------|-------------------|------------------------|
| Frozen probe | DINOv2-Base | head only | 0.5796 +/- 0.0067 |
| Frozen probe | Cell-DINO | head only | 0.5932 +/- 0.0065 |
| **LoRA** | **DINOv2 + LoRA** | **0.54M (2.4%)** | **0.6373 +/- 0.0203** |
| Fine-tuned | ResNet50 | all (25M) | 0.6638 +/- 0.0153 |

LoRA per-fold test ROC-AUC: 0.6082, 0.6426, 0.6233, 0.6677, 0.6456, 0.6364.

**LoRA recovers ~68.5% of the frozen (DINOv2-Base) -> fine-tuned (ResNet50) gap**, closing to
within 0.0265 of full fine-tuning while training only 2.4% of parameters.

## What this shows

**1. Cheap adaptation gets most of the way to full fine-tuning.** Adapting just the
query/value attention projections and the patch embedding -- 2.4% of the backbone's
parameters -- recovers roughly two-thirds of the gap between a frozen DINOv2 probe and a
fully fine-tuned ResNet50. This is a strong parameter-efficiency result: most of the
performance benefit of full adaptation is available at a fraction of the training cost and
memory footprint.

**2. LoRA has meaningfully wider fold-to-fold variance than the frozen arms.** Std = 0.0203,
roughly 3x wider than any frozen-probe arm (~0.006-0.007) and closer to (though still tighter
than) ResNet50's fine-tuned spread (0.0153, on a smaller sample here since ResNet is 6-fold vs
LoRA also 6-fold -- comparable scale). The per-fold range (0.608 to 0.668, a 6-point spread)
is wide enough that a single fold would have been a materially unreliable estimate of the
regime's true performance -- as the project's own early single-fold (0.608) and 3-fold
(0.626) estimates undersold the eventual 6-fold mean by a wide margin. This is the concrete
reason the frozen and fine-tuned arms use 6-fold CV: with less stable adaptation-regime
methods, folds genuinely differ from each other by more than sampling noise alone would
suggest, likely reflecting run-to-run optimization variance under a low-parameter-count
regime with high sensitivity to which folds end up in the training/val/test split.

**3. This refines the adaptation-spectrum picture.** With only a single fold (0.608), LoRA
looked barely better than the best frozen probe (Cell-DINO, 0.593) and far from fine-tuning
(0.664). With the full 6-fold estimate (0.637), LoRA sits much closer to fine-tuning than to
any frozen arm -- a materially different and more optimistic picture of what cheap adaptation
buys you on this task.

## Method

DINOv2-Small (ViT-S/14) loaded from HuggingFace (facebook/dinov2-small); LoRA adapters
injected on the `query` and `value` projections of every attention block via `peft`
(`inject_adapter_in_model`), rank=8, alpha=16, dropout=0.05, bias=none. The base backbone is
frozen; only the LoRA adapter matrices, the re-adapted 5-channel patch embedding (never seen
this input distribution during pretraining), and the linear classification head are
trainable. Training uses AdamW with a cosine-decaying learning rate (starting at 1e-4, halving
periodically down to ~3e-6 by late epochs), masked focal-BCE loss, early stopping on
validation ROC-AUC (patience 6 checks, `val_every=5` epochs). This is a deliberately different
recipe from the frozen linear-probe arms (SGD, lr=0.02) -- LoRA is a different regime
(adapting weights inside the backbone through a much deeper computational graph), and using
the linear-probe's higher SGD learning rate here would very likely destabilize training. Each
fine-tuned/adapted regime in this project uses its own architecture-appropriate recipe; only
the frozen linear-probe arms share one fixed recipe (see main benchmark README).

Same source_11 CSV, same 6 folds, same masked labels, same loss and metric as every other arm
in this project -- training regime and recipe are the only differences.

## Note on convergence time and variance

Each fold took 66-90 epochs to reach its early-stopping point, at roughly 68-115 minutes per
epoch depending on cluster NFS load at the time (LoRA training is I/O-bound reading full
images every epoch, unlike the frozen arms which train a linear head on cached embeddings in
minutes) -- roughly 2-4.5 days of wall-clock time per fold. Folds were trained in two batches
of three (0/2/4, then 1/3/5) rather than all six simultaneously, to stay within the cluster's
practical concurrent-pod ceiling before throughput degrades.

## Files

- `cv_summary.csv` -- per-fold and mean +/- std test ROC-AUC
- `cv_fold_spread.png` -- per-fold test ROC-AUC
- `cv_per_assay.png` / `.csv` -- mean per-assay ROC-AUC across folds
- `cv_val_curves.png` -- validation ROC-AUC trajectories across all 6 folds during training
