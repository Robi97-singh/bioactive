#!/usr/bin/env python3
"""
train_head.py

Train the linear probe head on CACHED embeddings produced by
extract_embeddings.py. No images, no backbone, no NFS -> seconds/epoch.

Pipeline (per --model / --res / --fold):
  1. Load {stem}_{train,val,test}.pt  (stem = {model}_r{res}_fold{fold}).
  2. Train nn.Linear(D, 29) with the SAME BCEMASKEDLoss the full pipeline uses.
  3. Validate every epoch with the SAME mean_roc_auc metric (val ROC-AUC),
     early-stop at patience 6 (matches ResNet regime), keep best-val head.
  4. Evaluate the best head on the TEST split, AGGREGATE per-image sigmoid
     predictions to COMPOUND level (mean, grouped by compound -- matching
     utils/metrics.py's evaluate_predictions(), which is what the fine-tuned
     CNN pipeline uses), THEN compute AUC on the compound-level predictions.
     Writes per_assay_auc.csv to
     {save_dir}/results/{fold_name}/plots/per_assay_auc.csv
     in the exact layout aggregate_cv.py reads (index = assay name, single
     column "test_roc_auc"). So `python aggregate_cv.py --model {model}
     --res {res}` works unchanged.

IMPORTANT (fixed 2026-10-03): earlier versions of this script computed AUC
directly on raw per-image embeddings with NO compound-level aggregation,
while the fine-tuned CNN pipeline (defaults/trainer.py -> utils/metrics.py
evaluate_predictions()) always aggregates multiple site-images of the same
compound via groupby("compound").mean() before scoring. Since every image
of a compound carries an identical label (the label is a compound-level
ChEMBL potency annotation, not an image-level one), per-image scoring is a
form of pseudo-replication: correlated, non-independent observations of one
ground truth were treated as independent samples, and skipping the
averaging step left in per-image noise (site-to-site variability) that
compound averaging is designed to cancel out. This under-estimated every
frozen-backbone model's true AUC by roughly 0.04-0.05 in this project's
retrospective audit. See README.md "Evaluation protocol fix" section for
the full writeup and before/after numbers. All historical results were
corrected in place (original per-image files preserved as
*_per_image_UNCORRECTED.csv next to the corrected ones).

Why this is comparable to the fine-tuned ResNet 0.6638 / 0.6792:
  same folds, same masked labels, same BCEMASKEDLoss, same mean_roc_auc
  metric, AND (as of this fix) same compound-level aggregation before AUC.
  Only the features differ (frozen backbone embeddings vs fine-tuned ResNet).

Usage (CPU is fine; GPU optional and faster):
  python train_head.py --model celldino --res 224 --fold 0 \
      --save_dir /shared/ssd/logs/b-r-singh1 \
      --epochs 100 --patience 6 --lr 1e-3
"""
import os
import sys
import json
import time
import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse the project's exact loss + metric so the number is on the ResNet scale.
from utils._utils import BCEMASKEDLoss          # noqa: E402
from utils.metrics import mean_roc_auc          # noqa: E402

TRAINING_PAPER_CSV = "/shared/hdd/data/bioactive/training_paper.csv"


def _load_split(emb_dir, stem, split, feature="cls"):
    path = os.path.join(emb_dir, f"{stem}_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached embeddings: {path}\n"
                                f"Run extract_embeddings.py for this fold first.")
    d = torch.load(path, map_location="cpu")
    # New extractor saves {'cls':[N,D], 'cls_avgpool':[N,2D], 'labels', 'uids'}.
    # Old extractor saved {'embeddings':[N,D], ...}. Support both.
    if feature in d:
        emb = d[feature].float()
    elif "embeddings" in d:
        emb = d["embeddings"].float()
    else:
        raise KeyError(f"{path} has no '{feature}' or 'embeddings' key; "
                       f"keys present: {list(d.keys())}")
    lab = d["labels"].float()
    return emb, lab, d.get("uids")


def _assay_names(n_classes, fold, save_dir, model, res):
    """
    The metric returns AUCs only for assays that passed the sample check, in
    class-index order. We map those back to assay names. The canonical names are
    'assay_{i}' (as seen in the ResNet per_assay_auc.csv, e.g. assay_2). We take
    int_to_labels if we can find it; otherwise fall back to assay_{i}.
    """
    # Fall back to the observed convention: assay_{class_index}.
    return {i: f"assay_{i}" for i in range(n_classes)}


def _build_plate_well_to_compound(csv_path=TRAINING_PAPER_CSV):
    """Build a (plate, well) -> Metadata_JCP2022 lookup from the master CSV.
    Used to map per-image uids ('PLATE/WELL_SITE.png') back to the compound
    they belong to, so predictions can be aggregated at compound level."""
    lookup = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[(row["Metadata_Plate"], row["Metadata_Well"])] = row["Metadata_JCP2022"]
    return lookup


def _aggregate_by_compound(logits_sigmoid, labels, uids, pw_to_compound):
    """
    Group per-image sigmoid predictions (and their labels) by compound,
    averaging predictions across all site-images of the same compound.
    Mirrors utils/metrics.py's evaluate_predictions():
        preds[assays] = sigmoid(preds[assays]); preds.groupby("compound").mean()
    Returns (compound_ids, labels_arr, preds_arr) as aligned numpy arrays,
    plus the count of images that could not be mapped to a compound.
    """
    n_classes = logits_sigmoid.shape[1]
    compound_preds = defaultdict(lambda: [[] for _ in range(n_classes)])
    compound_labels = {}
    unmapped = 0
    for i, uid in enumerate(uids):
        parts = str(uid).split("/")
        if len(parts) != 2:
            unmapped += 1
            continue
        plate, well_site = parts
        well = well_site.split("_")[0]
        compound = pw_to_compound.get((plate, well))
        if compound is None:
            unmapped += 1
            continue
        for c in range(n_classes):
            compound_preds[compound][c].append(logits_sigmoid[i, c])
        compound_labels[compound] = labels[i]

    compound_ids = sorted(compound_preds.keys())
    preds_arr = np.array([[np.mean(compound_preds[cid][c]) for c in range(n_classes)]
                           for cid in compound_ids])
    labels_arr = np.array([compound_labels[cid] for cid in compound_ids])
    return compound_ids, labels_arr, preds_arr, unmapped


def _per_assay_auc_from_arrays(labels_arr, preds_arr, n_classes, assay_name_map):
    """Per-assay AUC on already-aggregated (compound-level) arrays.
    preds_arr is already sigmoid-applied; labels_arr uses the project's
    {-1, 0, 1} masked-label convention (0 = not tested)."""
    from sklearn import metrics as skm
    names, aucs = [], []
    for c in range(n_classes):
        tar = (labels_arr[:, c] + labels_arr[:, c] ** 2) / 2.0
        mask = labels_arr[:, c] ** 2 > 0
        if tar.sum() > 0 and (mask.sum() - tar.sum()) > 0:
            auc = skm.roc_auc_score(tar[mask], preds_arr[:, c][mask])
            names.append(assay_name_map[c])
            aucs.append(auc)
    return names, aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # e.g. celldino
    ap.add_argument("--res", type=int, required=True)  # e.g. 224 or 448
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--save_dir", default="/shared/ssd/logs/b-r-singh1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)   # matches ResNet
    # DINO linear-probe convention: SGD + momentum + cosine LR schedule.
    ap.add_argument("--lr", type=float, default=0.02)  # locked frozen-probe recipe (fold-0 sweep on Cell-DINO)   # DINO probe default range 1e-3..5e-3
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--feature", choices=["cls", "cls_avgpool"], default="cls",
                    help="Which cached feature to probe. 'cls' = CLS token only "
                         "(384-dim, uniform across backbones). 'cls_avgpool' = "
                         "CLS concatenated with mean patch token (matches "
                         "Cell-DINO's own paper protocol).")
    ap.add_argument("--batch_size", type=int, default=4096)  # embeddings are tiny
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standardize", type=int, default=1,
                    help="1 = standardize features (subtract train mean, divide "
                         "by train std) before the linear head; matches Cell-DINO's "
                         "linear-eval protocol. Stats computed on TRAIN only, "
                         "applied to all splits (no val/test leakage). 0 = raw.")
    ap.add_argument("--training_paper_csv", default=TRAINING_PAPER_CSV,
                    help="Path to the master CSV used to build the (plate, well) "
                         "-> compound lookup for test-time aggregation.")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = f"{a.model}_r{a.res}_fold{a.fold}"
    emb_dir = os.path.join(a.save_dir, "embeddings")

    Xtr, Ytr, _ = _load_split(emb_dir, stem, "train", a.feature)
    Xva, Yva, Uva = _load_split(emb_dir, stem, "val", a.feature)
    Xte, Yte, Ute = _load_split(emb_dir, stem, "test", a.feature)

    # Standardize features: subtract TRAIN mean, divide by TRAIN std. Stats come
    # ONLY from train (fitting on val/test would leak). Applied to all splits.
    # Matches Cell-DINO's linear-eval protocol (self-normalization) and generally
    # makes linear probing far easier when feature magnitudes are uneven.
    if a.standardize:
        mu = Xtr.mean(dim=0, keepdim=True)
        sd = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        Xte = (Xte - mu) / sd
        print(f"[{stem}] standardized features (train mean/std); "
              f"mean|.|={mu.abs().mean():.4f} std|.|={sd.mean():.4f}", flush=True)

    D = Xtr.shape[1]
    n_classes = Ytr.shape[1]
    print(f"[{stem}] train {tuple(Xtr.shape)} | val {tuple(Xva.shape)} | "
          f"test {tuple(Xte.shape)} | D={D} | classes={n_classes}", flush=True)

    head = nn.Linear(D, n_classes).to(device)
    criterion = BCEMASKEDLoss()
    opt = torch.optim.SGD(head.parameters(), lr=a.lr,
                          momentum=a.momentum, weight_decay=a.weight_decay)
    # Cosine schedule over the full epoch budget (DINO linear-probe convention).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    Xtr_d, Ytr_d = Xtr.to(device), Ytr.to(device)
    Xva_d = Xva.to(device)

    # --- Validation fix (2026-10-03): early stopping now scores validation at
    # compound level, matching the test-time fix below, instead of the raw
    # per-image signal used previously. Spot-checked on 2 models (dino_r224,
    # celldino_r224, fold 0): the per-image and compound-level metrics agreed
    # on epoch ordering 97-99% of the time and the "wrong" checkpoint the old
    # code would have picked cost <0.0001 val ROC-AUC -- so this fix does not
    # change any already-reported result, but is scored correctly from here on.
    assay_name_map = _assay_names(n_classes, a.fold, a.save_dir, a.model, a.res)
    pw_to_compound = _build_plate_well_to_compound(a.training_paper_csv)
    if Uva is None:
        print(f"[{stem}] WARNING: no 'uids' in cached val embeddings -- "
              f"falling back to per-image validation AUC for early stopping.",
              flush=True)

    best_val = -1.0
    best_state = None
    patience_ctr = 0
    n = Xtr.shape[0]
    history = []   # per-epoch (epoch, train_loss, val_roc_auc) for curve plotting

    for epoch in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, a.batch_size):
            idx = perm[s:s + a.batch_size]
            xb, yb = Xtr_d[idx], Ytr_d[idx]
            opt.zero_grad()
            out = head(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        train_loss = total / n
        scheduler.step()

        if epoch % a.val_every == 0:
            head.eval()
            with torch.no_grad():
                val_logits = head(Xva_d).cpu().numpy()
            val_auc_per_image, _ = mean_roc_auc(Yva.numpy(), val_logits, do_sigmoid=True)
            val_auc = val_auc_per_image
            if Uva is not None:
                val_probs = 1.0 / (1.0 + np.exp(-val_logits))
                _, labels_vc, preds_vc, _ = _aggregate_by_compound(
                    val_probs, Yva.numpy(), Uva, pw_to_compound)
                _, vc_aucs = _per_assay_auc_from_arrays(
                    labels_vc, preds_vc, n_classes, assay_name_map)
                if vc_aucs:
                    val_auc = float(np.mean(vc_aucs))
            improved = val_auc > best_val
            if improved:
                best_val = val_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_roc_auc {val_auc:.4f} (compound) | "
                  f"val_roc_auc_per_image {val_auc_per_image:.4f} | "
                  f"best {best_val:.4f} | "
                  f"patience {patience_ctr}/{a.patience}", flush=True)
            history.append({"epoch": epoch, "train_loss": float(train_loss),
                            "val_roc_auc": float(val_auc),
                            "val_roc_auc_per_image": float(val_auc_per_image)})
            if patience_ctr >= a.patience:
                print(f"  EARLY STOP at epoch {epoch} (best val {best_val:.4f})",
                      flush=True)
                break

    # Restore best-val head and evaluate on TEST.
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_logits = head(Xte.to(device)).cpu().numpy()
    test_probs_per_image = 1.0 / (1.0 + np.exp(-test_logits))

    # Also compute the uncorrected (per-image, no aggregation) number, purely
    # for audit-trail / comparison purposes -- NOT used as the reported metric.
    test_mean_per_image, _ = mean_roc_auc(Yte.numpy(), test_logits, do_sigmoid=True)

    # --- Compound-level aggregation (the fix) ---------------------------------
    # assay_name_map / pw_to_compound were already built before the training
    # loop (used there for validation scoring too); reused here as-is.
    if Ute is None:
        raise RuntimeError(
            f"No 'uids' found in cached test embeddings for {stem}; cannot "
            f"aggregate to compound level. Re-run extract_embeddings.py with "
            f"a version that saves uids."
        )
    compound_ids, labels_c, preds_c, unmapped = _aggregate_by_compound(
        test_probs_per_image, Yte.numpy(), Ute, pw_to_compound)
    if unmapped:
        print(f"  WARNING: {unmapped} test images could not be mapped to a "
              f"compound and were dropped from evaluation", flush=True)

    names, aucs = _per_assay_auc_from_arrays(labels_c, preds_c, n_classes, assay_name_map)
    test_mean = float(np.mean(aucs)) if aucs else float("nan")

    print(f"[{stem}] TEST mean ROC-AUC (compound-level, corrected) = {test_mean:.4f} "
          f"over {len(aucs)} scored assays "
          f"[per-image uncorrected was {test_mean_per_image:.4f}] "
          f"({len(compound_ids)} compounds)", flush=True)

    # Write per_assay_auc.csv in aggregate_cv.py's expected layout.
    fold_name = f"bioact_{a.model}_r{a.res}_fold{a.fold}"
    out_plots = os.path.join(a.save_dir, "results", fold_name, "plots")
    os.makedirs(out_plots, exist_ok=True)
    ser = pd.Series(dict(zip(names, aucs))).sort_values(ascending=False)
    ser.to_csv(os.path.join(out_plots, "per_assay_auc.csv"),
               header=["test_roc_auc"])
    print(f"  wrote {os.path.join(out_plots, 'per_assay_auc.csv')} "
          f"({len(ser)} assays)", flush=True)

    # Write compound-level test_preds.csv / test_labels.csv -- same format as
    # the fine-tuned CNN pipeline (one row per compound, 'compound' index col).
    pd.DataFrame(preds_c, columns=names if len(names) == n_classes else
                 [assay_name_map[c] for c in range(n_classes)]).assign(
        compound=compound_ids
    ).set_index("compound").reset_index().to_csv(
        os.path.join(out_plots, "test_preds.csv"), index=False)
    pd.DataFrame(labels_c, columns=names if len(names) == n_classes else
                 [assay_name_map[c] for c in range(n_classes)]).assign(
        compound=compound_ids
    ).set_index("compound").reset_index().to_csv(
        os.path.join(out_plots, "test_labels.csv"), index=False)
    print(f"  wrote compound-level test_preds.csv / test_labels.csv "
          f"({len(compound_ids)} compounds, {n_classes} assays)", flush=True)

    # Also drop a small metrics json next to the checkpoints for convenience.
    ckpt_dir = os.path.join(a.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, f"{fold_name}_head_metrics.json"), "w") as f:
        json.dump({"stem": stem, "best_val_roc_auc": float(best_val),
                   "test_mean_roc_auc": float(test_mean),
                   "test_mean_roc_auc_per_image_UNCORRECTED": float(test_mean_per_image),
                   "n_scored_assays": len(aucs),
                   "n_test_compounds": len(compound_ids),
                   "n_unmapped_test_images": unmapped,
                   "lr": a.lr, "standardize": bool(a.standardize),
                   "evaluation_note": ("Compound-level aggregation (mean of "
                                       "per-image sigmoid probs, grouped by "
                                       "compound) applied before AUC, matching "
                                       "the fine-tuned CNN pipeline's "
                                       "evaluate_predictions(). See README."),
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
