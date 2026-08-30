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
  4. Evaluate the best head on the TEST split, and write per_assay_auc.csv to
     {save_dir}/results/{fold_name}/plots/per_assay_auc.csv
     in the exact layout aggregate_cv.py reads (index = assay name, single
     column "test_roc_auc"). So `python aggregate_cv.py --model {model}
     --res {res}` works unchanged.

Why this is comparable to the fine-tuned ResNet 0.6638:
  same folds, same masked labels, same BCEMASKEDLoss, same mean_roc_auc metric.
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


def _per_assay_from_metric(truths_np, preds_np, n_classes, assay_name_map):
    """
    Reproduce mean_roc_auc's per-assay selection AND capture which class indices
    were scored, so we can name them correctly. mean_roc_auc returns
    (mean, aucs_array) but aucs_array is only the *scored* assays, in order.
    We recompute the same selection here to pair name<->auc robustly.
    """
    from sklearn import metrics as skm

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    preds = _sigmoid(preds_np)
    names, aucs = [], []
    for c in range(n_classes):
        tar = (truths_np[:, c] + truths_np[:, c] ** 2) / 2.0
        mask = truths_np[:, c] ** 2 > 0
        if tar.sum() > 0 and (mask.sum() - tar.sum()) > 0:
            auc = skm.roc_auc_score(tar[mask], preds[:, c][mask])
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
    ap.add_argument("--lr", type=float, default=0.005)   # DINO probe default range 1e-3..5e-3
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
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = f"{a.model}_r{a.res}_fold{a.fold}"
    emb_dir = os.path.join(a.save_dir, "embeddings")

    Xtr, Ytr, _ = _load_split(emb_dir, stem, "train", a.feature)
    Xva, Yva, _ = _load_split(emb_dir, stem, "val", a.feature)
    Xte, Yte, _ = _load_split(emb_dir, stem, "test", a.feature)

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
            val_auc, _ = mean_roc_auc(Yva.numpy(), val_logits, do_sigmoid=True)
            improved = val_auc > best_val
            if improved:
                best_val = val_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_roc_auc {val_auc:.4f} | best {best_val:.4f} | "
                  f"patience {patience_ctr}/{a.patience}", flush=True)
            history.append({"epoch": epoch, "train_loss": float(train_loss),
                            "val_roc_auc": float(val_auc)})
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

    test_mean, _ = mean_roc_auc(Yte.numpy(), test_logits, do_sigmoid=True)
    assay_name_map = _assay_names(n_classes, a.fold, a.save_dir, a.model, a.res)
    names, aucs = _per_assay_from_metric(Yte.numpy(), test_logits,
                                         n_classes, assay_name_map)
    print(f"[{stem}] TEST mean ROC-AUC = {test_mean:.4f} "
          f"over {len(aucs)} scored assays", flush=True)

    # Write per_assay_auc.csv in aggregate_cv.py's expected layout.
    fold_name = f"bioact_{a.model}_r{a.res}_fold{a.fold}"
    out_plots = os.path.join(a.save_dir, "results", fold_name, "plots")
    os.makedirs(out_plots, exist_ok=True)
    ser = pd.Series(dict(zip(names, aucs))).sort_values(ascending=False)
    ser.to_csv(os.path.join(out_plots, "per_assay_auc.csv"),
               header=["test_roc_auc"])
    print(f"  wrote {os.path.join(out_plots, 'per_assay_auc.csv')} "
          f"({len(ser)} assays)", flush=True)

    # Also drop a small metrics json next to the checkpoints for convenience.
    ckpt_dir = os.path.join(a.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, f"{fold_name}_head_metrics.json"), "w") as f:
        json.dump({"stem": stem, "best_val_roc_auc": float(best_val),
                   "test_mean_roc_auc": float(test_mean),
                   "n_scored_assays": len(aucs),
                   "lr": a.lr, "standardize": bool(a.standardize),
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
