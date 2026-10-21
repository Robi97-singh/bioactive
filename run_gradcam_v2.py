#!/usr/bin/env python3
"""
run_gradcam_v2.py — Single-pass Grad-CAM for ResNet50.
Same as run_gradcam.py but scans each fold's dataloader ONCE
instead of once per target compound.
"""

import os, sys, argparse, json, functools
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from easydict import EasyDict as edict

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classification import MODEL_MAP, _fold_splits, _apply_resolution
from defaults.wrappers import DefaultWrapper
from defaults.trainer import Trainer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class AssayTarget:
    def __init__(self, assay_idx):
        self.assay_idx = assay_idx
    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output[self.assay_idx]
        return model_output[0, self.assay_idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params", default="params.json")
    p.add_argument("--model", default="resnet")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--model_name", type=str, default=None)
    return p.parse_args()


def get_target_layer(model, model_type):
    """Return the last convolutional layer for Grad-CAM."""
    if model_type in ("resnet", "resnet18"):
        return model.backbone.layer4[-1]
    elif model_type == "effnet":
        return model.backbone.features[-1]
    elif model_type == "convnext":
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unsupported model type for Grad-CAM: {model_type}")


def load_fold_model(fold, args, base_params):
    backbone_type, model_name_base = MODEL_MAP[args.model]
    model_name_base_eff = args.model_name if args.model_name else model_name_base
    model_name = model_name_base_eff + "_r" + str(args.res) + "_fold" + str(fold)

    params = edict(deepcopy(base_params))
    params.model_params.backbone_type = backbone_type
    params.training_params.model_name = model_name

    splits = _fold_splits(fold)
    params.dataset_params.data_split_numbers = splits

    _apply_resolution(params, args.res)
    params.dataloader_params.testloader.batch_size = 1
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
    trainer.load_session(restore_only_model=True)
    trainer.model.eval()
    return trainer


def make_rgb_thumbnail(image_5ch):
    img = image_5ch.cpu().numpy()
    rgb = np.stack([img[0], img[2], img[4]], axis=-1)
    for c in range(3):
        ch = rgb[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            rgb[:, :, c] = (ch - mn) / (mx - mn)
        else:
            rgb[:, :, c] = 0.0
    return rgb.astype(np.float32)


def run_gradcam_direct(trainer, image, label, compound_id, assay_name,
                       assay_idx, out_dir, reason, model_type="resnet"):
    """Run Grad-CAM with pre-loaded image tensor."""
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    target_layer = get_target_layer(model, model_type)
    original_forward = model.forward

    def forward_no_emb(x):
        return original_forward(x, return_embedding=False)

    model.forward = forward_no_emb
    try:
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [AssayTarget(assay_idx)]
        grayscale_cam = cam(input_tensor=image, targets=targets)
        grayscale_cam = grayscale_cam[0]
    finally:
        model.forward = original_forward

    assay_dir = os.path.join(out_dir, str(assay_name).replace("/", "_"))
    os.makedirs(assay_dir, exist_ok=True)

    raw_path = os.path.join(assay_dir, compound_id + "_" + reason + "_raw.npy")
    np.save(raw_path, grayscale_cam)

    rgb = make_rgb_thumbnail(image[0])
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (pseudo-RGB)\n" + compound_id)
    axes[0].axis("off")

    true_lbl = label[0, assay_idx].item() if label.dim() > 1 else label.item()
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM: " + str(assay_name) + "\nlabel=" + str(int(true_lbl)) + ", reason=" + reason)
    axes[1].axis("off")

    fig.tight_layout()
    overlay_path = os.path.join(assay_dir, compound_id + "_" + reason + "_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: " + overlay_path)
    return True


def main():
    args = parse_args()

    print("Loading Grad-CAM targets from " + args.targets)
    targets_df = pd.read_csv(args.targets)
    print("  " + str(len(targets_df)) + " (compound, assay) pairs across " +
          str(targets_df['fold'].nunique()) + " folds")

    with open(args.params) as f:
        base_params = json.load(f)

    folds_needed = sorted(targets_df["fold"].unique())

    # Load assay mapping from first fold
    print("\nLoading assay name mapping...")
    first_fold = folds_needed[0]
    trainer_tmp = load_fold_model(first_fold, args, base_params)
    int_to_labels = trainer_tmp.testloader.dataset.int_to_labels
    label_to_int = {}
    for k, v in int_to_labels.items():
        label_to_int[v] = k
    print("  " + str(len(int_to_labels)) + " assays: " + str(list(int_to_labels.values())[:5]) + "...")
    print("  label_to_int keys: " + str(list(label_to_int.keys())))

    total_done = 0
    total_failed = 0

    for fold in folds_needed:
        fold_targets = targets_df[targets_df["fold"] == fold]
        print("\n" + "=" * 60)
        print("Fold " + str(fold) + ": " + str(len(fold_targets)) + " targets")
        print("=" * 60)

        # Build lookup: compound_id -> [(assay_name, assay_idx, reason), ...]
        targets_lookup = {}
        for _, row in fold_targets.iterrows():
            compound_id = row["compound_id"]
            assay_name = row["assay"]
            reason = row["selection_reason"].replace("_high_conf", "")
            assay_key = str(assay_name)

            if assay_key not in label_to_int:
                print("  [!] Assay '" + assay_key + "' not in label map - skipping")
                total_failed += 1
                continue

            assay_idx = label_to_int[assay_key]
            if compound_id not in targets_lookup:
                targets_lookup[compound_id] = []
            targets_lookup[compound_id].append((assay_name, assay_idx, reason))

        if not targets_lookup:
            print("  No valid targets for this fold, skipping.")
            continue

        print("  Looking for " + str(len(targets_lookup)) + " compounds...")

        # Load model
        if fold == first_fold:
            trainer = trainer_tmp
        else:
            trainer = load_fold_model(fold, args, base_params)

        dataloader = trainer.testloader
        found = set()

        # SINGLE PASS through the dataloader
        for batch_idx, batch_data in enumerate(dataloader):
            images, labels, ids, compounds = batch_data
            if batch_idx % 200 == 0:
                print("    Batch " + str(batch_idx) + ", found " + str(len(found)) + "/" + str(len(targets_lookup)))
            for i, cmpd in enumerate(compounds):
                if cmpd in targets_lookup and cmpd not in found:
                    found.add(cmpd)
                    img = images[i:i+1].to(trainer.device_id)
                    lbl = labels[i:i+1]
                    for assay_name, assay_idx, reason in targets_lookup[cmpd]:
                        print("  FOUND " + cmpd + " x " + str(assay_name) + " (idx=" + str(assay_idx) + ", " + reason + ")")
                        ok = run_gradcam_direct(trainer, img, lbl, cmpd,
                                                assay_name, assay_idx, args.out, reason,
                                                model_type=args.model)
                        if ok:
                            total_done += 1
                        else:
                            total_failed += 1
            if len(found) == len(targets_lookup):
                print("    All " + str(len(targets_lookup)) + " compounds found!")
                break

        for cmpd in targets_lookup:
            if cmpd not in found:
                print("    [!] " + cmpd + " NOT found in fold " + str(fold))
                total_failed += len(targets_lookup[cmpd])

    print("\n" + "=" * 60)
    print("Done. " + str(total_done) + " heatmaps, " + str(total_failed) + " failed.")
    print("Output: " + args.out)


if __name__ == "__main__":
    main()
