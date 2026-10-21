#!/usr/bin/env python3
"""
select_gradcam_targets.py — Pick (compound, assay) pairs for Grad-CAM attribution.

Strategy:
  1. Stratify assays by per-assay test ROC-AUC (from cv_per_assay.csv):
     - 2 high-AUC assays, 1 mid-AUC assay, 1 low-AUC assay
  2. Within each chosen assay, pick compounds from the out-of-fold predictions:
     - 3 true positives with highest predicted probability (confident correct)
     - 1 true negative with highest predicted probability (confident false positive)
     This yields ~16 (compound, assay) pairs for the pilot.

Input:
  - oof_predictions.csv (from dump_predictions.py)
  - cv_per_assay.csv    (from aggregate_cv.py)

Output:
  - gradcam_targets.csv with columns: compound_id, assay, true_label, pred_prob,
    fold, selection_reason, assay_auc_tier

Usage:
    python select_gradcam_targets.py \
        --predictions /shared/ssd/logs/b-r-singh1/results/cv/resnet50_r224/oof_predictions.csv \
        --per_assay   /shared/ssd/logs/b-r-singh1/results/cv/resnet50_r224/cv_per_assay.csv \
        --out         /shared/ssd/logs/b-r-singh1/results/cv/resnet50_r224/gradcam_targets.csv
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Path to oof_predictions.csv")
    p.add_argument("--per_assay", required=True, help="Path to cv_per_assay.csv")
    p.add_argument("--out", required=True, help="Output path for gradcam_targets.csv")
    p.add_argument("--n_tp", type=int, default=3,
                   help="True positives per assay (default: 3)")
    p.add_argument("--n_fp", type=int, default=1,
                   help="False positives per assay (default: 1)")
    p.add_argument("--n_high", type=int, default=2,
                   help="Number of high-AUC assays to select (default: 2)")
    p.add_argument("--n_mid", type=int, default=1,
                   help="Number of mid-AUC assays (default: 1)")
    p.add_argument("--n_low", type=int, default=1,
                   help="Number of low-AUC assays (default: 1)")
    return p.parse_args()


def select_assays(per_assay_df, n_high=2, n_mid=1, n_low=1):
    """Stratify assays into high/mid/low AUC tiers and pick representatives."""

    # per_assay_df should have columns like 'assay' and 'mean' (or similar)
    # Try to detect column names
    if "mean" in per_assay_df.columns:
        auc_col = "mean"
    elif "mean_auc" in per_assay_df.columns:
        auc_col = "mean_auc"
    elif "roc_auc" in per_assay_df.columns:
        auc_col = "roc_auc"
    else:
        # Assume the second column is the AUC
        auc_col = per_assay_df.columns[1]
        print(f"  Auto-detected AUC column: '{auc_col}'")

    if "assay" in per_assay_df.columns:
        name_col = "assay"
    else:
        name_col = per_assay_df.columns[0]
        print(f"  Auto-detected assay name column: '{name_col}'")

    df = per_assay_df[[name_col, auc_col]].copy()
    df.columns = ["assay", "auc"]
    df = df.sort_values("auc", ascending=False).reset_index(drop=True)

    n = len(df)
    # Tercile boundaries
    t1 = n // 3
    t2 = 2 * n // 3

    high_pool = df.iloc[:t1]
    mid_pool  = df.iloc[t1:t2]
    low_pool  = df.iloc[t2:]

    selected = []

    # Pick from each tier (highest AUC in high, median in mid, lowest in low)
    for pool, tier, count in [(high_pool, "high", n_high),
                               (mid_pool, "mid", n_mid),
                               (low_pool, "low", n_low)]:
        picks = pool.head(count) if tier == "high" else (
                pool.iloc[len(pool)//2 - count//2 : len(pool)//2 - count//2 + count]
                if tier == "mid" else pool.tail(count))
        for _, row in picks.iterrows():
            selected.append({"assay": row["assay"], "auc": row["auc"], "tier": tier})

    return pd.DataFrame(selected)


def select_compounds(predictions_df, assay_name, n_tp=3, n_fp=1):
    """For one assay, pick high-confidence TPs and FPs."""

    pred_col  = f"pred_{assay_name}"
    label_col = f"label_{assay_name}"

    if pred_col not in predictions_df.columns:
        print(f"  [!] Column {pred_col} not found — skipping assay {assay_name}")
        return pd.DataFrame()

    df = predictions_df[["compound_id", "fold", pred_col, label_col]].copy()
    df.columns = ["compound_id", "fold", "pred_prob", "true_label"]

    # Exclude masked labels (-1)
    df = df[df["true_label"] >= 0].copy()

    if len(df) == 0:
        print(f"  [!] No valid labels for assay {assay_name}")
        return pd.DataFrame()

    results = []

    # True positives: label == 1, sorted by pred_prob descending
    tp = df[df["true_label"] == 1].sort_values("pred_prob", ascending=False)
    for _, row in tp.head(n_tp).iterrows():
        results.append({**row.to_dict(), "selection_reason": "true_positive_high_conf"})

    # False positives: label == 0, sorted by pred_prob descending (model thinks active, but isn't)
    fp = df[df["true_label"] == 0].sort_values("pred_prob", ascending=False)
    for _, row in fp.head(n_fp).iterrows():
        results.append({**row.to_dict(), "selection_reason": "false_positive_high_conf"})

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["assay"] = assay_name
    return out


def main():
    args = parse_args()

    print("Loading predictions...")
    preds = pd.read_csv(args.predictions)
    print(f"  {len(preds)} compounds, {preds.filter(like='pred_').shape[1]} assays")

    print("\nLoading per-assay AUC...")
    per_assay = pd.read_csv(args.per_assay)
    print(f"  {len(per_assay)} assays")

    print("\nSelecting assays by AUC tier...")
    assay_sel = select_assays(per_assay, args.n_high, args.n_mid, args.n_low)
    for _, row in assay_sel.iterrows():
        print(f"  [{row['tier']:>4s}] {row['assay']:40s}  AUC = {row['auc']:.4f}")

    print("\nSelecting compounds per assay...")
    all_targets = []
    for _, arow in assay_sel.iterrows():
        assay_name = arow["assay"]
        tier = arow["tier"]
        targets = select_compounds(preds, assay_name, args.n_tp, args.n_fp)
        if len(targets):
            targets["assay_auc_tier"] = tier
            targets["assay_auc"] = arow["auc"]
            all_targets.append(targets)
            n_tp = (targets["selection_reason"] == "true_positive_high_conf").sum()
            n_fp = (targets["selection_reason"] == "false_positive_high_conf").sum()
            print(f"  {assay_name}: {n_tp} TPs + {n_fp} FPs")

    if not all_targets:
        print("\n[ERROR] No targets selected!")
        return

    final = pd.concat(all_targets, ignore_index=True)
    final = final[["compound_id", "assay", "true_label", "pred_prob", "fold",
                    "selection_reason", "assay_auc_tier", "assay_auc"]]

    final.to_csv(args.out, index=False)
    print(f"\nSaved {len(final)} (compound, assay) targets to {args.out}")
    print(f"  {final['assay'].nunique()} assays, "
          f"{final['compound_id'].nunique()} unique compounds")


if __name__ == "__main__":
    main()
