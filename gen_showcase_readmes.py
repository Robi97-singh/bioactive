import os, json, subprocess, numpy as np, pandas as pd
L = "/shared/ssd/logs/b-r-singh1"
MAP = [("biomedclip_frozen_224","biomedclip_r224"),("celldino_frozen_224","celldino_r224"),
       ("celldino_frozen_448","celldino_r448"),("clip_frozen_224","clip_r224"),
       ("dinov2_frozen_224","dino_r224"),("dinov2_frozen_448","dino_r448"),
       ("dinov2_large_frozen_224","dino_large_r224"),("dinov2_large_frozen_448","dino_large_r448"),
       ("dinov2_small_frozen_224","dino_small_r224"),("dinov3_frozen_224","dinov3_r224"),
       ("dinov3_frozen_448","dinov3_r448")]
LABEL = {"biomedclip":"BiomedCLIP","celldino":"Cell-DINO","clip":"CLIP ViT-L/14 (locked recipe)",
         "dino":"DINOv2-Base","dino_large":"DINOv2-Large","dino_small":"DINOv2-Small","dinov3":"DINOv3-Base"}
PARIHAR = {"dinov3","clip"}
def split(stem): m, r = stem.rsplit("_r", 1); return m, int(r)
def load(stem):
    d = pd.read_csv(f"{L}/results/cv/{stem}/cv_summary.csv"); f = d["fold"].astype(str)
    folds = d[f.isin(list("012345"))]["mean_test_roc_auc"].astype(float).values
    mean = float(d[f == "MEAN"]["mean_test_roc_auc"].iloc[0]); sd = float(d[f == "STD"]["mean_test_roc_auc"].iloc[0])
    js = [json.load(open(f"{L}/checkpoints/bioact_{stem}_fold{i}_head_metrics.json")) for i in range(6)]
    return dict(folds=folds, mean=mean, std=sd,
                pi=np.mean([j["test_mean_roc_auc_per_image_UNCORRECTED"] for j in js]),
                ep=[len(j["history"]) for j in js], lr=sorted({j["lr"] for j in js}))
D = {st: load(st) for _, st in MAP}
RANK = sorted(D, key=lambda s: -D[s]["mean"])
def fmt(x): return f"{x:.4f}"
def readme(stem):
    m, res = split(stem); d = D[stem]; name = LABEL[m]
    o = [f"# {name} -- Frozen Linear-Probe Arm @ {res}", "",
         "Replication and extension of Fredin Haslum et al., *Nature Communications* 15:3470 (2024), on the",
         "public JUMP-CP Cell Painting bioactivity benchmark (source_11, 29 assays, 6-fold CV).", "",
         "## Result", "",
         f"- **Test ROC-AUC (6-fold mean +/- SD): {fmt(d['mean'])} +/- {fmt(d['std'])}** (compound-level aggregation)",
         "- Per-fold: " + ", ".join(fmt(x) for x in d["folds"]),
         f"- Per-image (uncorrected) mean: {fmt(d['pi'])} -> compound-level gain {d['mean']-d['pi']:+.4f}",
         f"- Rank among the 11 frozen configurations: {RANK.index(stem)+1} of 11 (best: {RANK[0]}, {fmt(D[RANK[0]]['mean'])}; worst: {RANK[-1]}, {fmt(D[RANK[-1]]['mean'])})",
         f"- Head training: lr {d['lr'][0]}, stopped after {min(d['ep'])}-{max(d['ep'])} epochs per fold (early stopping on compound-level validation ROC-AUC)", ""]
    other = f"{m}_r{448 if res == 224 else 224}"
    if other in D:
        a, b = (stem, other) if res == 224 else (other, stem)
        da, db = D[a], D[b]; dl = db["folds"] - da["folds"]
        o += ["## Resolution (224 -> 448)", "", "| 224 | 448 | effect | folds improved at 448 |", "|---|---|---|---|",
              f"| {fmt(da['mean'])} +/- {fmt(da['std'])} | {fmt(db['mean'])} +/- {fmt(db['std'])} | {db['mean']-da['mean']:+.4f} | {int((dl>0).sum())} of 6 |", "",
              "Paired per-fold change: " + ", ".join(f"{x:+.4f}" for x in dl) + ".",
              "The effect is small relative to the 0.01-0.02 fold-to-fold standard deviation; see the benchmark README for the cross-model comparison.", ""]
    o += ["## Benchmark context", "",
          "| Reference | Test ROC-AUC (6-fold) |", "|---|---|",
          "| ResNet-50 fine-tuned, 448 | 0.6792 +/- 0.0245 |", "| ResNet-50 fine-tuned, 224 | 0.6638 +/- 0.0153 |",
          "| DINOv2-Small + LoRA, 224 | 0.6373 +/- 0.0203 |", "",
          "Frozen probes are statistically tied with each other apart from one pair (Cell-DINO over BiomedCLIP), and",
          "fine-tuned ResNet-50 is consistently best; see `../benchmark_figures/README.md` for the Friedman/Nemenyi tests.", "",
          "## Method", "",
          "Identical locked recipe for every frozen arm: frozen backbone, cached CLS embeddings, `Linear(D, 29)` head,",
          "masked focal-BCE loss, SGD momentum 0.9 + cosine schedule, lr 0.02, train-only feature standardization,",
          "early stopping on compound-level validation ROC-AUC (patience 6), `mean_roc_auc` metric. Same source_11 CSV, same 6 folds,",
          "same masked labels. The learning rate was chosen in a fold-0 sweep on Cell-DINO and applied unchanged to all frozen backbones.", "",
          "## Verification", "",
          "Results re-run on 2026-10-06 with the committed `train_head.py` and an explicit `--lr 0.02`. Per-assay AUC recomputed",
          "independently from the saved compound-level `test_preds.csv` / `test_labels.csv` matches the reported values to ~1e-16",
          "on spot-checked runs. See the root README, section \"Recipe and re-run verification\"."]
    if m in PARIHAR:
        o += ["", "Embedding extraction for this backbone was originally run by Bhargav Parihar (`b-b-parihar` cluster account);",
              "his cached embeddings were copied into this account and the linear head was trained here from them."]
    return "\n".join(o) + "\n"
APPLY = os.environ.get("APPLY") == "1"
if not APPLY:
    print(readme("dinov3_r448")); print("(dry run: sample only; set APPLY=1 to write all 11)")
else:
    S = "results_showcase/_superseded_20261003"
    for folder, stem in MAP:
        old = f"results_showcase/{folder}/README.md"
        if os.path.exists(old):
            os.makedirs(f"{S}/{folder}", exist_ok=True)
            subprocess.run(["git", "mv", old, f"{S}/{folder}/README_pre_correction.md"], check=True)
        open(old, "w", encoding="utf-8").write(readme(stem)); print("wrote", old)
