#!/usr/bin/env python3
"""analyze_vitcx.py -- (1) aggregate ViT-CX deletion/insertion z-scores, (2) cell-overlap
enrichment of ViT-CX maps vs the ResNet Grad-CAM maps, (3) map agreement. numpy/scipy only.
  python3 analyze_vitcx.py  [XAI=/shared/ssd/logs/b-r-singh1/xai_scratch]
                            [GC=/shared/ssd/logs/b-r-singh1/results/cv/resnet50_r224/gradcam]"""
import os, glob, csv
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

XAI = os.environ.get("XAI", "/shared/ssd/logs/b-r-singh1/xai_scratch")
GC = os.environ.get("GC", "/shared/ssd/logs/b-r-singh1/results/cv/resnet50_r224/gradcam")
FG_FRAC = float(os.environ.get("FG_FRAC", "0.03"))     # brightest 3% of pixels = "cells"


def mm(a):
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def resize(a, hw):
    if a.shape == tuple(hw):
        return a
    return zoom(a, (hw[0] / a.shape[0], hw[1] / a.shape[1]), order=1)[:hw[0], :hw[1]]


def foreground(x):
    """x [C,H,W] -> bool mask of the brightest FG_FRAC of pixels (robust z across channels)."""
    z = []
    for c in range(x.shape[0]):
        ch = gaussian_filter(x[c].astype(np.float64), 1.0)
        med = np.median(ch)
        mad = np.median(np.abs(ch - med)) * 1.4826 + 1e-9
        z.append((ch - med) / mad)
    s = np.max(np.stack(z), axis=0)
    return s >= np.quantile(s, 1.0 - FG_FRAC)


def enrich(sal, fg):
    """mean saliency on foreground / mean saliency overall (1.0 = no better than chance)."""
    return float(sal[fg].mean() / max(sal.mean(), 1e-12))


def corr(a, b, n=56):
    a, b = resize(mm(a), (n, n)).ravel(), resize(mm(b), (n, n)).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    # ---- (1) deletion / insertion aggregate --------------------------------------
    print("=== (1) deletion/insertion vs 10 random maps (z<-2 / z>+2 = clearly better than random) ===")
    for d in sorted(glob.glob(os.path.join(XAI, "vitcx_*"))):
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "metrics_fold*.csv"))):
            rows += list(csv.DictReader(open(f)))
        if not rows:
            continue
        print("\n%s  (%d rows)" % (os.path.basename(d), len(rows)))
        for v in sorted({r["variant"] for r in rows}):
            r = [q for q in rows if q["variant"] == v]
            zd = np.array([float(q["z_del"]) for q in r])
            zi = np.array([float(q["z_ins"]) for q in r])
            print("  %-12s n=%2d | mean z_del %+.2f (frac<-2: %.2f) | mean z_ins %+.2f (frac>+2: %.2f)"
                  % (v, len(r), zd.mean(), (zd < -2).mean(), zi.mean(), (zi > 2).mean()))

    # ---- (2)+(3) cell-overlap enrichment and agreement with ResNet Grad-CAM ------
    print("\n=== (2) cell-overlap enrichment (mean map on brightest %.0f%% pixels / mean map; 1.0 = chance) ===" % (100 * FG_FRAC))
    acc = {}
    for d in sorted(glob.glob(os.path.join(XAI, "vitcx_*"))):
        for f in sorted(glob.glob(os.path.join(d, "f*_*.npz"))):
            base = os.path.basename(f)[:-4]          # f<fold>_<compound>_<assay>
            parts = base.split("_", 1)[1].rsplit("_", 1)
            comp, assay = parts[0], parts[1]
            z = np.load(f)
            x = z["x"].astype(np.float32)
            fg = foreground(x)
            tag = os.path.basename(d)
            for v in [k for k in z.files if k != "x"]:
                s = z[v].astype(np.float32)
                acc.setdefault((tag, v), []).append(enrich(s, fg))
            gcs = glob.glob(os.path.join(GC, assay, comp + "_*_raw.npy"))
            if gcs:
                g = resize(mm(np.load(gcs[0]).squeeze()), x.shape[1:])
                acc.setdefault(("resnet50_gradcam", "gradcam"), []).append(enrich(g, fg))
                for v in [k for k in z.files if k != "x"]:
                    acc.setdefault((tag, v + "|corr_vs_gradcam"), []).append(corr(z[v].astype(np.float32), g))
    seen = set()
    for (tag, v), vals in sorted(acc.items()):
        key = (tag, v)
        if key in seen:
            continue
        seen.add(key)
        vals = np.array(vals)
        if "corr" in v:
            print("  %-26s %-28s n=%2d mean corr %+.2f" % (tag, v, len(vals), vals.mean()))
        else:
            print("  %-26s %-14s n=%2d enrichment mean %.2f median %.2f (frac>1.5: %.2f)"
                  % (tag, v, len(vals), vals.mean(), np.median(vals), (vals > 1.5).mean()))
    print("\nNote: ResNet Grad-CAM enrichment is computed on the same images (matched compound/assay).")


if __name__ == "__main__":
    main()
