#!/usr/bin/env python3
"""report_fig.py -- clean report figure for chosen examples. Needs make_panel.py + strat_panel.py in the same folder.
Usage: python report_fig.py --root $L/xai_scratch --gradcam $L/results/cv/resnet50_r224/gradcam --out $L/xai_scratch/panel \
         [--pick JCP2022_024601:688687 JCP2022_084669:845173 JCP2022_000099:752594]"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_panel import COLS, FN, pick, fit, rgb, resnet_map
from strat_panel import CELLS

DEFAULT = ["JCP2022_024601:688687", "JCP2022_084669:845173", "JCP2022_000099:752594"]


def norm(m):
    return (m - m.min()) / (np.ptp(m) + 1e-9)


def zdel(d, comp, assay, key):
    fs = glob.glob(os.path.join(d, "metrics_fold*.csv")) if d else []
    if not fs: return None
    M = pd.concat([pd.read_csv(f) for f in fs])
    M = M[(M.compound.astype(str) == comp) & (M.assay.astype(str) == assay)]
    sub = key.replace("occl", "occ")
    if "variant" in M and M.variant.astype(str).str.contains(sub).any():
        M = M[M.variant.astype(str).str.contains(sub)]
    return float(M.z_del.iloc[0]) if len(M) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--gradcam", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--pick", nargs="*", default=DEFAULT)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dirs = {t: (sorted(glob.glob(os.path.join(a.root, g))) or [None])[0] for t, g, k in COLS}
    ref = next(d for d in dirs.values() if d)
    avail = {}
    for f in glob.glob(os.path.join(ref, "f*.npz")):
        m = FN.match(os.path.basename(f))
        if m: avail[(m.group(2), m.group(3))] = os.path.basename(f)
    picks = [tuple(p.split(":")) for p in a.pick]
    ncol = 2 + len(COLS)
    groups = [("Reference", 1, 2), ("DINOv3 (frozen)", 2, 4), ("Cell-DINO (frozen)", 4, 7)]
    fig, ax = plt.subplots(len(picks), ncol, figsize=(2.5 * ncol + 0.6, 2.55 * len(picks) + 1.0), squeeze=False)
    heads = ["Input", "ResNet-50\nGrad-CAM"] + [c[0].replace("\n", "\n") for c in COLS]
    for i, (comp, assay) in enumerate(picks):
        nm = avail.get((comp, assay))
        if nm is None: print("not found:", comp, assay); continue
        x = None
        for t, g, k in COLS:
            p = os.path.join(dirs[t], nm) if dirs[t] else None
            if p and os.path.exists(p): x = np.load(p)["x"].astype(np.float32); break
        hw = x.shape[1:]; im = rgb(x)
        kind = "Cell image" if (comp, assay) in CELLS else "Noisy image"
        ax[i, 0].imshow(im); ax[i, 0].set_ylabel("%s\n%s\nassay %s" % (kind, comp.replace("JCP2022_", "JCP2022\n"), assay), fontsize=8)
        rm = resnet_map(a.gradcam, assay, comp)
        ax[i, 1].imshow(im)
        if rm is not None:
            ax[i, 1].imshow(norm(fit(rm, hw)), cmap="inferno", alpha=.55, vmin=0, vmax=1)
        for j, (t, g, k) in enumerate(COLS):
            ax[i, 2 + j].imshow(im)
            p = os.path.join(dirs[t], nm) if dirs[t] else None
            if p and os.path.exists(p):
                m = norm(fit(pick(np.load(p), k), hw))
                h = ax[i, 2 + j].imshow(m, cmap="inferno", alpha=.55, vmin=0, vmax=1)
                z = zdel(dirs[t], comp, assay, k)
                if z is not None:
                    ax[i, 2 + j].text(0.03, 0.04, "deletion z %+.1f" % z, transform=ax[i, 2 + j].transAxes, fontsize=7.5,
                                      color="white", bbox=dict(facecolor="black", alpha=.6, pad=1.5, lw=0))
    for j, h in enumerate(heads):
        ax[0, j].set_title(h, fontsize=9)
    for q in ax.ravel(): q.set_xticks([]); q.set_yticks([])
    fig.tight_layout(rect=(0, 0.11, 1, 0.94), h_pad=1.6)
    for name, c0, c1 in groups:
        b0 = ax[0, c0].get_position(); b1 = ax[0, c1 - 1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, 0.955, name, ha="center", fontsize=10, fontweight="bold")
        fig.add_artist(plt.Line2D([b0.x0, b1.x1], [0.95, 0.95], color="#888", lw=1))
    cax = fig.add_axes([0.35, 0.075, 0.3, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(cmap="inferno"), cax=cax, orientation="horizontal", ticks=[0, 1]).ax.set_xticklabels(["low", "high"])
    fig.text(0.5, 0.012, "Relative importance, rescaled 0-1 within each map (pseudo-RGB input: channels 1, 3, 5). Deletion z: lower = more faithful; compare with control maps.",
             ha="center", fontsize=7.5, color="#444")
    for ext in ("png", "pdf"):
        fp = os.path.join(a.out, "report_figure." + ext); fig.savefig(fp, dpi=200); print("wrote", fp)

if __name__ == "__main__":
    main()
