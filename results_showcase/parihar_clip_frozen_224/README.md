# CLIP ViT-L/14 @ 224 (frozen linear probe) -- Parihar's run

These 6-fold CV test ROC-AUC results use **Parihar's own cached embeddings**
(his `extract_embeddings.py` run), not Robin's. They are evaluated with the
same corrected, compound-level-aggregation `train_head.py` used for every
other frozen-backbone result in this repo (see the root README's
"Evaluation protocol fix" section) -- re-run directly against his cached
`.pt` embeddings via a symlinked read-only mount, writing output only to
Robin's own directory. Nothing in Parihar's own result files was read from
or written to beyond the cached embeddings themselves.

Kept in a separate `parihar_*` folder, rather than merged into
Robin's own CLIP ViT-L/14 @ 224 result, because the two runs used
independently-extracted embeddings (different extraction run, possibly
different seed/checkpoint) and are not directly interchangeable --
run-to-run variance between them should be expected and is not a
discrepancy to resolve.

**Corrected 6-fold mean test ROC-AUC: 0.6104**

See `cv_summary.csv` for the per-fold breakdown and `cv_per_assay.csv`
for per-assay detail.
