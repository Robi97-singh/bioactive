#!/usr/bin/env python3
"""
extract_embeddings.py

Run a FROZEN backbone over each CV split ONCE and cache the embeddings, so the
linear head can then be trained on cached vectors (seconds/epoch) instead of
re-running the backbone every epoch (the ~53 min/epoch, 0% GPU, NFS-bound path).

For a given --model / --res / --fold it:
  1. Builds the dataset + model EXACTLY as classification.py does
     (same fold split, z-norm, res-224 center crop) by reusing DefaultWrapper.
  2. For each of train / val / test, loops once under torch.no_grad() calling
     model(images, return_embedding=True) and collects the [B, D] embeddings.
  3. Saves {embeddings:[N,D], labels:[N,29], uids:[N]} per split to:
     {save_dir}/embeddings/{model_name_base}_r{res}_fold{fold}_{split}.pt

Labels are saved RAW (the +1 / -1 / 0 encoding). Do NOT convert them: the mask
lives inside the label (mask = label**2 > 0), and both BCEMASKEDLoss and
mean_roc_auc depend on that encoding. train_head.py consumes them as-is.

Usage (inside a pod, venv activated):
  python extract_embeddings.py --params_path params/params_celldino_cluster.json \
      --model celldino --res 224 --fold 0

Notes:
  * This is the ONLY NFS-bound step. It still reads every image once, so it is
    as slow as ~one training epoch for that split's size. That's expected and
    unavoidable: you must see each image at least once to embed it.
  * We force a DETERMINISTIC transform: the val/test transforms (CenterCrop) are
    used for ALL splits, so each image maps to exactly one vector. This is the
    standard linear-probe setup (no train-time augmentation). Note this in the
    thesis: fine-tuned ResNet used RandomCrop augmentation; frozen probes use a
    fixed center crop.
"""
import os
import sys
import time
import json
import argparse

import torch

# --- make the repo importable regardless of where we're launched from ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Cell-DINO needs xformers disabled BEFORE the model is built.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from easydict import EasyDict as edict                # noqa: E402
from utils.launch import launch                       # noqa: E402
from utils.helpfuns import load_params                 # noqa: E402
from utils.system_def import define_system_params      # noqa: E402
from defaults.wrappers import DefaultWrapper           # noqa: E402

# classification.py owns the arg parser + the arg->params mutators we reuse.
import classification as C                             # noqa: E402


def _deterministic_transforms(params):
    """
    Force every split to use the val_transforms (deterministic CenterCrop), so
    each image yields ONE embedding. We copy val_transforms onto train_transforms
    so the trainloader stops doing RandomCrop / augmentation.

    _apply_resolution has already run (res=224 -> Resize 540, crop 224) before
    this, so the sizes are correct; we only swap the crop *type* to center.
    """
    dp = params['dataset_params']
    if 'val_transforms' in dp:
        # Deep-ish copy: transforms are plain dicts of dicts.
        import copy
        dp['train_transforms'] = copy.deepcopy(dp['val_transforms'])
    # Also make sure we don't request multiple augmentations per image.
    for k in ('num_augmentations',):
        if k in dp:
            dp[k] = 1
    return params


@torch.no_grad()
def _embed_split(model, loader, device, split_name):
    """Run the frozen backbone over one loader once; return (emb, lab, uids)."""
    model.eval()
    embs, labs, uids_all = [], [], []
    n_batches = len(loader)
    t0 = time.time()
    for i, batch in enumerate(loader):
        # dataset returns (img, label, uid) for train/val and
        # (img, label, uid, cmpd) for test (mode == 'test').
        images, labels, uids = batch[0], batch[1], batch[2]
        images = images.to(device, non_blocking=True)

        out = model(images, return_embedding=True)
        # model(..., return_embedding=True) returns (logits, features).
        # We only want the features (the frozen embedding).
        if isinstance(out, (tuple, list)):
            feats = out[1]
        else:
            feats = out
        embs.append(feats.detach().cpu())
        labs.append(labels.detach().cpu() if torch.is_tensor(labels)
                    else torch.as_tensor(labels))
        # uids can be a list/tuple of strings or a tensor
        if torch.is_tensor(uids):
            uids_all.extend([str(u) for u in uids.tolist()])
        else:
            uids_all.extend([str(u) for u in uids])

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            print(f"    [{split_name}] {i+1}/{n_batches} batches "
                  f"({rate:.2f} it/s, {dt:.0f}s)", flush=True)

    emb = torch.cat(embs, dim=0)
    lab = torch.cat(labs, dim=0)
    return emb, lab, uids_all


def main(parameters, args):
    define_system_params(parameters.system_params)

    # Deterministic transforms for embedding (center crop on all splits).
    _deterministic_transforms(parameters)

    wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()

    device = wrapper.model.device_id if hasattr(wrapper.model, "device_id") \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        wrapper.model.to(device)
    except Exception:
        pass

    dl = wrapper.dataloaders
    splits = {
        "train": dl.trainloader,
        "val":   dl.valloader,
        "test":  dl.testloader,
    }

    save_dir = parameters.training_params.save_dir
    out_dir = os.path.join(save_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # model_name currently looks like bioact_celldino_r224_fold0 (fold suffix
    # added by _fold_splits handling). Strip it back to a stable stem for the
    # embedding filenames: {model}_r{res}_fold{fold}.
    stem = f"{args.model}_r{args.res}_fold{args.fold}"

    print(f"\n=== Extracting embeddings: {stem} ===", flush=True)
    print(f"    save -> {out_dir}", flush=True)

    meta = {"model": args.model, "res": args.res, "fold": args.fold,
            "splits": {}}

    for split_name, loader in splits.items():
        if loader is None:
            print(f"    [{split_name}] loader is None, skipping", flush=True)
            continue
        emb, lab, uids = _embed_split(wrapper.model, loader, device, split_name)
        out_path = os.path.join(out_dir, f"{stem}_{split_name}.pt")
        torch.save({"embeddings": emb, "labels": lab, "uids": uids}, out_path)
        print(f"    [{split_name}] saved {tuple(emb.shape)} embeddings, "
              f"{tuple(lab.shape)} labels -> {out_path}", flush=True)
        meta["splits"][split_name] = {
            "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
            "path": out_path,
        }

    meta_path = os.path.join(out_dir, f"{stem}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    meta -> {meta_path}", flush=True)
    print("=== extraction done ===\n", flush=True)


if __name__ == "__main__":
    # Mirror classification.py's __main__ EXACTLY so --params_path/--model/--res
    # /--fold behave identically (MODEL_MAP, _apply_resolution, _fold_splits).
    args = C.parse_arguments()
    parameters = edict(C.load_params(args)) if hasattr(C, "load_params") \
        else edict(load_params(args))
    C.update_params_from_args(parameters, args)

    launch(main, (parameters, args))
