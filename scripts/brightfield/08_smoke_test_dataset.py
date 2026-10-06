"""Smoke test: real batch through BioActBrightfield with train/test transforms."""
import json, os, random, sys
import numpy as np, torch
sys.path.insert(0, ".")
from defaults.datasets import BioActBrightfield

P = "params/params_brightfield_dinov3_base_cluster.json"
params = json.load(open(P))
print("top-level keys:", list(params.keys()))
dp = dict(params["dataset_params"])
print("dataset_params:", {k: v for k, v in dp.items() if not isinstance(v, (dict, list))})
for k in ("data_split_number", "split_number", "fold"):
    if k in dp:
        dp[k] = 0
        print(f"set {k}=0")

ok = True
for mode in ("train", "val", "test"):
    ds = BioActBrightfield(dp, mode=mode)
    print(f"\n[{mode}] n_items={len(ds)} n_classes={getattr(ds,'n_classes',None)}")
    random.seed(0)
    idx = random.sample(range(len(ds)), 64)
    imgs, labs = [], []
    for i in idx:
        it = ds[i]
        imgs.append(it[0]); labs.append(it[1])
        if mode == "test" and len(it) != 4:
            ok = False; print("  FAIL: test item should have 4 fields (img,label,uid,cmpd)")
    x = torch.stack([torch.as_tensor(a) for a in imgs]).float()
    y = torch.stack([torch.as_tensor(a) for a in labs]).float()
    print(f"  img batch {tuple(x.shape)} dtype={x.dtype} min={x.min():.3f} max={x.max():.3f}")
    print(f"  post-normalize mean={x.mean():.3f} std={x.std():.3f}  (expect ~0 / ~1)")
    print(f"  labels {tuple(y.shape)} unique values={torch.unique(y)[:6].tolist()}")
    if x.shape[1] != 1: ok = False; print("  FAIL: channel dim != 1")
    if abs(x.mean()) > 0.35 or not (0.6 < x.std() < 1.5):
        ok = False; print("  FAIL: normalization looks off")
    if torch.isnan(x).any(): ok = False; print("  FAIL: NaN in images")
print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
