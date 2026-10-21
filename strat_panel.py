#!/usr/bin/env python3
"""strat_panel.py -- dense vs sparse split + centre-bias check. Needs make_panel.py in the same folder.
Usage: python strat_panel.py --root $L/xai_scratch --gradcam $L/results/cv/resnet50_r224/gradcam --out $L/xai_scratch/panel
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from make_panel import COLS, FN, pick, enrich, fit, boot, resnet_map

# labels read by eye from the 16-target figure (individual cells clearly visible = "cells"; mostly noise + a few specks = "noisy")
CELLS = {("JCP2022_015965","688687"),("JCP2022_024601","688687"),("JCP2022_042261","688687"),("JCP2022_048713","845173"),
         ("JCP2022_084669","845173"),("JCP2022_096456","845173"),("JCP2022_053273","737187"),("JCP2022_053273","845173")}
THR, DENSE_FRAC = 0.30, 0.10   # fixed rule: foreground = smoothed normalised intensity > THR; dense if > 10% of pixels


def fg_frac(x):
    s = []
    for c in range(x.shape[0]):
        lo, hi = np.percentile(x[c], [1, 99.9])
        s.append((x[c] - lo) / (hi - lo + 1e-9))
    s = gaussian_filter(np.max(s, 0), 2.0)
    return float((s > THR).mean())


def small(m, n=64):
    m = fit(m, (n * (m.shape[0] // n + 1), n * (m.shape[1] // n + 1))) if min(m.shape) < n else m
    h, w = m.shape; m = m[:h // n * n, :w // n * n]
    m = m.reshape(n, h // n, n, w // n).mean((1, 3))
    return (m - m.mean()) / (m.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--gradcam", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = {t: (sorted(glob.glob(os.path.join(a.root, g))) or [None])[0] for t, g, k in COLS}
    ref = next(d for d in dirs.values() if d)
    tg = sorted(FN.match(os.path.basename(f)).groups() for f in glob.glob(os.path.join(ref, "f*.npz"))
                if FN.match(os.path.basename(f)))
    lab, maps, rows = {}, {}, []
    for fold, comp, assay in tg:
        nm = f"f{fold}_{comp}_{assay}.npz"; x = None
        for t, g, k in COLS:
            if dirs[t] and os.path.exists(os.path.join(dirs[t], nm)):
                x = np.load(os.path.join(dirs[t], nm))["x"].astype(np.float32); break
        if x is None: continue
        ff = fg_frac(x); cls = "cells" if (comp, assay) in CELLS else "noisy"
        lab[(comp, assay)] = cls
        print(f"{comp} {assay} fold{fold}  fg={ff:.3f}  {cls}")
        ms = {"ResNet Grad-CAM": resnet_map(a.gradcam, assay, comp)}
        for t, g, k in COLS:
            p = os.path.join(dirs[t], nm) if dirs[t] else None
            ms[t.replace("\n", " ")] = pick(np.load(p), k) if p and os.path.exists(p) else None
        for meth, m in ms.items():
            if m is None: continue
            m = fit(m, x.shape[1:]); m = (m - m.min()) / (np.ptp(m) + 1e-9)
            maps[(meth, comp, assay)] = small(m)
            rows.append(dict(method=meth, compound=comp, assay=assay, cls=cls, enr=enrich(x, m)))
    E = pd.DataFrame(rows)
    print("\nclass counts:", pd.Series(list(lab.values())).value_counts().to_dict())
    # centre bias: corr with a centred gaussian, and with the mean map of the OTHER targets
    n = 64; yy, xx = np.mgrid[:n, :n]; g = np.exp(-(((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (2 * (n / 4) ** 2)))
    g = (g - g.mean()) / g.std()
    cb = []
    for meth in E.method.unique():
        keys = [k for k in maps if k[0] == meth]
        for k in keys:
            oth = np.mean([maps[o] for o in keys if o != k], 0)
            cb.append(dict(method=meth, cls=lab[(k[1], k[2])],
                           r_centre=float((maps[k] * g).mean()), r_others=float((maps[k] * oth).mean() / (oth.std() + 1e-9))))
    C = pd.DataFrame(cb)
    pd.set_option("display.width", 200)
    print("\nCENTRE / SAMEness check (mean r; high = map resembles a centre blob / the other images' maps)")
    print(C.groupby(["method", "cls"]).agg(n=("r_centre", "size"), r_centre=("r_centre", "mean"), r_others=("r_others", "mean")).round(2).to_string())
    # stratified enrichment CIs
    out = []
    for (m, c), d in E.groupby(["method", "cls"]):
        b = boot(d.enr.values, n=3000)
        out.append(dict(method=m, cls=c, metric="enrichment", n=d.enr.notna().sum(), mean=b[0], lo=b[1], hi=b[2], median=b[3]))
    # stratified z_del / z_ins if metrics rows can be matched to targets
    for t, g_, k in COLS:
        d = dirs[t]
        fs = glob.glob(os.path.join(d, "metrics_fold*.csv")) if d else []
        if not fs: continue
        M = pd.concat([pd.read_csv(f) for f in fs])
        cc = next((c for c in ("compound", "compound_id") if c in M), None)
        ac = next((c for c in ("assay", "assay_id") if c in M), None)
        if cc is None or ac is None:
            print("metrics csv has no compound/assay columns:", list(M.columns)); break
        if "variant" in M and M.variant.astype(str).str.contains(k.replace("occl", "occ")).any():
            M = M[M.variant.astype(str).str.contains(k.replace("occl", "occ"))]
        M["cls"] = [lab.get((str(c), str(a_)), None) for c, a_ in zip(M[cc], M[ac])]
        for c, dd in M.dropna(subset=["cls"]).groupby("cls"):
            for col in ("z_del", "z_ins"):
                if col in dd:
                    b = boot(dd[col].values, n=3000)
                    out.append(dict(method=t.replace("\n", " "), cls=c, metric=col, n=dd[col].notna().sum(), mean=b[0], lo=b[1], hi=b[2], median=b[3]))
    R = pd.DataFrame(out).sort_values(["metric", "method", "cls"])
    R.to_csv(os.path.join(a.out, "stratified_ci.csv"), index=False)
    print("\nSTRATIFIED (95% bootstrap CI of the mean over targets)")
    print(R.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
