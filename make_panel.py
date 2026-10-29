#!/usr/bin/env python3
"""make_panel.py -- 16-target figure + bootstrap CIs. numpy/scipy/matplotlib/pandas only.
Usage: python make_panel.py --root $L/xai_scratch --gradcam $L/results/cv/resnet50_r224/gradcam --out $L/xai_scratch/panel
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter, zoom

# (column title, dir glob under root, substring of npz key)
COLS = [
    ("DINOv3\nocclusion", "vitcx_occlusion-dinov3*", "occl"),
    ("DINOv3\nViT-CX", "vitcx_dinov3*r448_r448", "paper"),
    ("DINOv3\ngrad x act", "vitcx_gradcam-dinov3*", "gradxact"),
    ("Cell-DINO\nocclusion", "vitcx_occlusion-celldino*", "occl"),
    ("Cell-DINO\nViT-CX", "vitcx_celldino*r224_r224", "paper"),
    ("Cell-DINO\ngrad x act", "vitcx_gradcam-celldino*", "gradxact"),
]
FN = re.compile(r"^f(\d+)_(.+)_(\d+)\.npz$")


def pick(z, sub):
    ks = [k for k in z.files if k != "x" and sub in k]
    if not ks:
        ks = [k for k in z.files if k != "x"][:1]
    return z[ks[0]].astype(np.float32)


def rgb(x, ch=(0, 2, 4)):
    ch = [c for c in ch if c < x.shape[0]] or [0, 0, 0]
    while len(ch) < 3: ch.append(ch[-1])
    im = np.stack([x[c] for c in ch], -1).astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(im[..., c], [1, 99.5])
        im[..., c] = np.clip((im[..., c] - lo) / (hi - lo + 1e-9), 0, 1)
    return im


def enrich(x, m, frac=0.03):
    """mean map on brightest 3% pixels / mean map; 1.0 = chance."""
    z = []
    for c in range(x.shape[0]):
        med = np.median(x[c]); mad = np.median(np.abs(x[c] - med)) * 1.4826 + 1e-9
        z.append((x[c] - med) / mad)
    b = gaussian_filter(np.max(z, 0), 1.0)
    t = np.quantile(b, 1 - frac)
    mk = b >= t
    mm = m.mean()
    return float(m[mk].mean() / mm) if mm > 0 and mk.any() else np.nan


def fit(m, hw):
    if m.shape == hw: return m
    m = zoom(m, (hw[0] / m.shape[0], hw[1] / m.shape[1]), order=1)
    return m[:hw[0], :hw[1]]


def boot(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan,) * 6
    r = np.random.default_rng(seed)
    bs = np.array([np.median(r.choice(v, len(v))) for _ in range(n)])
    bm = np.array([np.mean(r.choice(v, len(v))) for _ in range(n)])
    return float(np.mean(v)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), \
        float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def resnet_map(gdir, assay, comp):
    f = sorted(glob.glob(os.path.join(gdir, str(assay), f"{comp}_*_raw.npy")))
    return np.load(f[0]).astype(np.float32).squeeze() if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--gradcam", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = {}
    for t, g, k in COLS:
        h = sorted(glob.glob(os.path.join(a.root, g)))
        if not h: print("MISSING dir for", t.replace("\n", " "), g)
        dirs[t] = h[0] if h else None
        print(t.replace("\n", " "), "->", dirs[t])
    # targets: union of npz names in the first available dir
    ref = next(d for d in dirs.values() if d)
    tg = sorted(FN.match(os.path.basename(f)).groups() for f in glob.glob(os.path.join(ref, "f*.npz"))
                if FN.match(os.path.basename(f)))
    print(len(tg), "targets")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ncol = 2 + len(COLS)
    per = 8
    rows = []
    for pg in range(0, len(tg), per):
        sub = tg[pg:pg + per]
        fig, ax = plt.subplots(len(sub), ncol, figsize=(2.6 * ncol, 2.6 * len(sub)), squeeze=False)
        for i, (fold, comp, assay) in enumerate(sub):
            nm = f"f{fold}_{comp}_{assay}.npz"
            x = None
            for t, g, k in COLS:
                p = os.path.join(dirs[t], nm) if dirs[t] else None
                if p and os.path.exists(p):
                    x = np.load(p)["x"].astype(np.float32); break
            if x is None: continue
            hw = x.shape[1:]; im = rgb(x)
            ax[i, 0].imshow(im); ax[i, 0].set_ylabel(f"{comp[-6:]}\n{assay}", fontsize=8)
            rm = resnet_map(a.gradcam, assay, comp)
            ax[i, 1].imshow(im)
            if rm is not None:
                rm = fit(rm, hw); rm = (rm - rm.min()) / (np.ptp(rm) + 1e-9)
                ax[i, 1].imshow(rm, cmap="jet", alpha=.5, vmin=0, vmax=1)
                rows.append(dict(method="ResNet Grad-CAM", compound=comp, assay=assay, enr=enrich(x, rm)))
            for j, (t, g, k) in enumerate(COLS):
                p = os.path.join(dirs[t], nm) if dirs[t] else None
                ax[i, 2 + j].imshow(im)
                if p and os.path.exists(p):
                    m = fit(pick(np.load(p), k), hw); m = (m - m.min()) / (np.ptp(m) + 1e-9)
                    ax[i, 2 + j].imshow(m, cmap="jet", alpha=.5, vmin=0, vmax=1)
                    rows.append(dict(method=t.replace("\n", " "), compound=comp, assay=assay, enr=enrich(x, m)))
            if i == 0:
                for j, tt in enumerate(["Input", "ResNet\nGrad-CAM"] + [c[0] for c in COLS]):
                    ax[0, j].set_title(tt, fontsize=9)
        for q in ax.ravel(): q.set_xticks([]); q.set_yticks([])
        fig.tight_layout(); fp = os.path.join(a.out, f"panel_{pg // per + 1}.png")
        fig.savefig(fp, dpi=110); plt.close(fig); print("wrote", fp)
    # bootstrap: enrichment
    E = pd.DataFrame(rows); E.to_csv(os.path.join(a.out, "enrichment_per_target.csv"), index=False)
    out = []
    for m, d in E.groupby("method"):
        b = boot(d.enr.values); out.append(dict(method=m, metric="enrichment", n=d.enr.notna().sum(),
            mean=b[0], mean_lo=b[1], mean_hi=b[2], median=b[3], med_lo=b[4], med_hi=b[5]))
    # bootstrap: z_del / z_ins from metrics csvs
    for t, g, k in COLS:
        d = dirs[t]
        if not d: continue
        fs = glob.glob(os.path.join(d, "metrics_fold*.csv"))
        if not fs: continue
        M = pd.concat([pd.read_csv(f) for f in fs])
        if "variant" in M:
            M = M[M.variant.astype(str).str.contains(k.replace("occl", "occ")) | (M.variant.astype(str).nunique() == 1)] \
                if M.variant.astype(str).str.contains(k.replace("occl", "occ")).any() else M
        for col in ("z_del", "z_ins"):
            if col in M:
                b = boot(M[col].values)
                out.append(dict(method=t.replace("\n", " "), metric=col, n=M[col].notna().sum(),
                    mean=b[0], mean_lo=b[1], mean_hi=b[2], median=b[3], med_lo=b[4], med_hi=b[5]))
    R = pd.DataFrame(out); R.to_csv(os.path.join(a.out, "bootstrap_ci.csv"), index=False)
    pd.set_option("display.width", 200); print(R.round(2).to_string(index=False))
    # paired: each method vs ResNet enrichment
    P = E.pivot_table(index=["compound", "assay"], columns="method", values="enr")
    if "ResNet Grad-CAM" in P:
        r = np.random.default_rng(1)
        for m in P.columns:
            if m == "ResNet Grad-CAM": continue
            dd = (P[m] - P["ResNet Grad-CAM"]).dropna().values
            if len(dd):
                bs = [np.mean(r.choice(dd, len(dd))) for _ in range(5000)]
                print(f"{m} minus ResNet enrichment: mean {dd.mean():+.2f}  95% CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")

if __name__ == "__main__":
    main()
