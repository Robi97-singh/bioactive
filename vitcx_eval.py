#!/usr/bin/env python3
"""vitcx_eval.py -- numpy-only helpers: deletion/insertion curves (ViT-CX paper protocol),
map agreement, and the comparison figure. No torch."""
import numpy as np
from scipy.ndimage import gaussian_filter


def _auc(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1])) / 2.0)


def deletion_insertion(x, sal, score_fn, fill, steps=20, blur_sigma=8.0):
    """x [C,H,W]; sal [H,W]; score_fn(X [B,C,H,W]) -> [B] scores of ONE assay; fill [C].
    Deletion : remove the most salient pixels first (replace by fill)  -> lower AUC is better.
    Insertion: start from a blurred image, reveal the most salient first -> higher AUC is better."""
    C, H, W = x.shape
    order = np.argsort(-sal.reshape(-1), kind="stable")
    fr = np.linspace(0.0, 1.0, steps + 1)
    fill = np.asarray(fill, np.float32).reshape(C, 1)
    flat_x = x.reshape(C, -1)
    blur = np.stack([gaussian_filter(x[c], blur_sigma) for c in range(C)]).astype(np.float32).reshape(C, -1)
    dels = np.empty((steps + 1, C, H * W), np.float32)
    inss = np.empty_like(dels)
    for s, f in enumerate(fr):
        idx = order[:int(round(f * H * W))]
        d = flat_x.copy()
        d[:, idx] = fill
        dels[s] = d
        i = blur.copy()
        i[:, idx] = flat_x[:, idx]
        inss[s] = i
    sd = np.asarray(score_fn(dels.reshape(steps + 1, C, H, W)))
    si = np.asarray(score_fn(inss.reshape(steps + 1, C, H, W)))
    return _auc(sd, fr), _auc(si, fr), sd, si


def random_baseline(x, score_fn, fill, n=3, seed=0, **kw):
    r = np.random.default_rng(seed)
    out = [deletion_insertion(x, r.random(x.shape[1:]), score_fn, fill, **kw)[:2] for _ in range(n)]
    return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))


def map_corr(a, b):
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def pseudo_rgb(x, chans=(0, 2, 4)):
    rgb = np.stack([x[c] for c in chans], axis=-1).astype(np.float32)
    for c in range(3):
        lo, hi = rgb[..., c].min(), rgb[..., c].max()
        rgb[..., c] = (rgb[..., c] - lo) / (hi - lo) if hi > lo else 0.0
    return rgb


def save_figure(x, maps, suptitle, path):
    """maps: list of (title, sal [H,W])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = pseudo_rgb(x)
    n = len(maps) + 1
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    ax[0].imshow(rgb)
    ax[0].set_title("Input (pseudo-RGB)")
    for a, (t, s) in zip(ax[1:], maps):
        a.imshow(rgb)
        a.imshow(s, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        a.set_title(t, fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
