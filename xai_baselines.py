#!/usr/bin/env python3
"""xai_baselines.py -- numpy-only baseline: sliding-window occlusion saliency."""
import numpy as np


def occlusion_map(x, a, fill, score_fn, win=None, stride=None, chunk=32):
    """x [C,H,W]; a = assay index; fill [C]. Slide a win x win window (stride) that replaces the
    pixels by `fill`; drop = score(real) - score(occluded) (positive = region supports the assay);
    each pixel gets the mean drop of all windows covering it; min-max normalised to [0,1]."""
    C, H, W = x.shape
    win = win or max(8, H // 7)
    stride = stride or max(4, win // 2)
    ys = list(range(0, H - win + 1, stride))
    xs = list(range(0, W - win + 1, stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)
    boxes = [(y, xx) for y in ys for xx in xs]
    fill = np.asarray(fill, np.float32).reshape(C, 1, 1)
    f0 = float(score_fn(x[None].astype(np.float32))[0, a])
    acc = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    for i in range(0, len(boxes), chunk):
        part = boxes[i:i + chunk]
        batch = np.repeat(x[None].astype(np.float32), len(part), axis=0)
        for j, (y, xx) in enumerate(part):
            batch[j, :, y:y + win, xx:xx + win] = fill
        drops = f0 - score_fn(batch)[:, a]
        for d, (y, xx) in zip(drops, part):
            acc[y:y + win, xx:xx + win] += d
            cnt[y:y + win, xx:xx + win] += 1
    sal = acc / np.maximum(cnt, 1)
    lo, hi = sal.min(), sal.max()
    return ((sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)).astype(np.float32)
