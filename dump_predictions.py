#!/usr/bin/env python3
"""
dump_predictions.py — Save per-compound out-of-fold predicted probabilities.
"""

import os, sys, argparse, json, torch, numpy as np, pandas as pd
from copy import deepcopy
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classification import MODEL_MAP, _fold_splits, _apply_resolution
from defaults.wrappers import DefaultWrapper
from defaults.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Dump out-of-fold predictions to CSV")
    p.add_argument("--model", type=str, default="resnet", choices=list(MODEL_MAP.keys()))
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--params", type=str, default="params.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_name", type=str, default=None,
                   help="Override model_name base (e.g. bioact_resnet). Default: from MODEL_MAP.")
    return p.parse_args()


def run_fold(fold, args, base_params):
    backbone_type, model_name_base = MODEL_MAP[args.model]
    model_name_base_eff = args.model_name if args.model_name else model_name_base
    model_name = f"{model_name_base_eff}_r{args.res}_fold{fold}"

    params = edict(deepcopy(base_params))
    params.model_params.backbone_type = backbone_type
    params.training_params.model_name = model_name

    # Set fold splits into dataset_params.data_split_numbers
    splits = _fold_splits(fold)
    params.dataset_params.data_split_numbers = splits

    _apply_resolution(params, args.res)

    params.dataloader_params.testloader.batch_size = args.batch_size
    params.dataloader_params.testloader.shuffle = False

    params.training_params.epochs = 0
    params.training_params.restore_session = True
    params.training_params.restore_only_model = True

    wrapper = DefaultWrapper(params)
    wrapper.instantiate()
    trainer = Trainer(wrapper)

    trainer.restore_session = True
    trainer.restore_only_model = True
    trainer.set_models_precision(False)
    try:
        trainer.load_session(restore_only_model=True)
    except Exception as e:
        print(f"  [!] Could not load checkpoint for {model_name}: {e}")
        return None

    trainer.model.eval()
    dataloader = trainer.testloader

    if not len(dataloader):
        print(f"  [!] Empty test dataloader for fold {fold}")
        return None

    int_to_labels = dataloader.dataset.int_to_labels

    all_preds = []
    all_labels = []
    all_compounds = []

    with torch.no_grad():
        for images, labels, ids, compound in tqdm(dataloader,
                                                   desc=f"  Fold {fold} inference",
                                                   leave=True):
            images = images.to(trainer.device_id, non_blocking=True)
            outputs, _ = trainer.model(images, return_embedding=True)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor)
                              else np.array(labels))
            all_compounds.extend(compound)

    all_preds  = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    n_samples, n_assays = all_preds.shape
    print(f"  Fold {fold}: {n_samples} samples, {n_assays} assays")

    assay_names = [int_to_labels[i] for i in range(n_assays)]
    pred_cols  = [f"pred_{name}" for name in assay_names]
    label_cols = [f"label_{name}" for name in assay_names]

    df = pd.DataFrame(all_preds, columns=pred_cols)
    df_labels = pd.DataFrame(all_labels, columns=label_cols)
    df.insert(0, "compound_id", all_compounds)
    df.insert(1, "fold", fold)
    df = pd.concat([df, df_labels], axis=1)
    return df


def main():
    args = parse_args()
    backbone_type, model_name_base = MODEL_MAP[args.model]
    print(f"Dumping predictions for: {args.model} ({backbone_type}), resolution {args.res}")

    with open(args.params) as f:
        base_params = json.load(f)

    folds = [args.fold] if args.fold is not None else list(range(6))
    all_dfs = []

    for fold in folds:
        print(f"\n{'='*60}\nFold {fold}\n{'='*60}")
        df = run_fold(fold, args, base_params)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\n[ERROR] No predictions were generated.")
        sys.exit(1)

    pooled = pd.concat(all_dfs, ignore_index=True)

    if args.save_dir:
        out_dir = args.save_dir
    else:
        log_dir = base_params.get("training_params", {}).get("save_dir",
                  base_params.get("save_dir", "/shared/ssd/logs/b-r-singh1"))
        out_dir = os.path.join(log_dir, "results", "cv", f"{backbone_type}_r{args.res}")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "oof_predictions.csv")
    pooled.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(pooled)} rows to {out_path}")
    print(f"Unique compounds: {pooled['compound_id'].nunique()}")
    print(f"Folds covered: {sorted(pooled['fold'].unique().tolist())}")

    dup = pooled.groupby("compound_id")["fold"].nunique()
    multi = dup[dup > 1]
    if len(multi):
        print(f"\n[WARNING] {len(multi)} compounds in multiple folds!")
    else:
        print("Sanity check passed: every compound in exactly one fold.")


if __name__ == "__main__":
    main()
