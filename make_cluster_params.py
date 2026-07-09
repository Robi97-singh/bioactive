#!/usr/bin/env python3
"""Derive params_cluster_base.json from a 4080 params file: paths only.
Resolution and fold stay as runtime flags."""
import argparse, json

IMAGES   = "/shared/hdd/data/bioactive/images"
CSV      = "/shared/hdd/data/bioactive/training_paper.csv"
SAVE_DIR = "/shared/ssd/logs/b-r-singh1/"
OUT      = "params/params_cluster_base.json"

def setp(d, dotted, value, rep):
    keys = dotted.split("."); node = d
    for k in keys[:-1]:
        if k not in node:
            rep.append(f"  MISSING section '{k}' (skipped {dotted})"); return
        node = node[k]
    last = keys[-1]
    rep.append(f"  {dotted}\n      old: {node.get(last)}\n      new: {value}")
    node[last] = value

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--out", default=OUT)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

p = json.load(open(a.base)); rep = []
setp(p, "dataset_params.data_location", IMAGES, rep)

dp = p.get("dataset_params", {})
csv_keys = [k for k in dp if "csv" in k.lower()]
if csv_keys:
    for k in csv_keys: setp(p, f"dataset_params.{k}", CSV, rep)
else:
    rep.append("  WARNING: no *csv* key in dataset_params -- set it manually")

setp(p, "training_params.save_dir", SAVE_DIR, rep)
for l in ("trainloader", "valloader", "testloader"):
    if l in p.get("dataloader_params", {}) and "num_workers" in p["dataloader_params"][l]:
        setp(p, f"dataloader_params.{l}.num_workers", 8, rep)

print(f"Deriving from {a.base}\n"); print("\n".join(rep))
cc = dp.get("test_transforms", {}).get("CenterCrop", {})
print(f"\nSanity:\n  base crop {cc.get('height')} (overridden by --res)")
print(f"  split key present: {'data_split_numbers' in dp} (overridden by --fold)")

if a.dry:
    print("\n[dry run] nothing written")
else:
    json.dump(p, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")
