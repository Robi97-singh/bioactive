#!/usr/bin/env python3
"""
vitcx_core.py -- ViT-CX (Xie et al., IJCAI 2023, arXiv:2211.03064) core algorithm.

Model-agnostic and torch-free: the model is reached through two callables.

    feat_fn(x)  -> feature maps  [D, h, w]   patch embeddings at the output of the
                                             attention module of a chosen block,
                                             reshaped to the token grid (no CLS/
                                             register tokens).   x: [C, H, W]
    score_fn(X) -> scores [B, A]             target scores for A assays for a batch
                                             of images X [B, C, H, W]
                                             (sigmoid probability of each assay,
                                             as ViT-CX uses class probability).

Algorithm (paper Sec. 3):
  1. Masks: every channel of the feature map is min-max normalised to [0,1]
     (D masks). Agglomerative clustering (cosine distance, threshold delta) merges
     near-duplicate masks; each cluster's mean mask is one mask M_k (K of them).
     Masks are upsampled bilinearly to the image size.
  2. Debiased causal impact (Eq. 2):
         s_k = f(X*M_k + eps_k) + [ f(X) - f(X + eps_k) ],
         eps_k = (1 - M_k) * Z,  Z ~ N(0, sigma^2)
  3. Pixel-coverage-bias correction (Eq. 5):
         S(x) = sum_k s_k M_k(x) / rho(x),   rho(x) = mean_k M_k(x),
         then min-max normalised to [0,1].

Deviations from the paper (implementation choices, flagged):
  * clustering is computed on the token-grid masks (h x w), not on the upsampled
    masks; only the K cluster means are upsampled. Cosine similarity of bilinearly
    upsampled maps is close to that of the originals, and this keeps it cheap.
  * linkage = 'average' (the paper text does not state the linkage in the main body).
  * the masked-out fill value is configurable (default 0 in the model's input space,
    i.e. X (.) M as in the paper's formula).
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import zoom
from scipy.spatial.distance import pdist


def _minmax(a, axis=None, eps=1e-12):
    lo = a.min(axis=axis, keepdims=True)
    hi = a.max(axis=axis, keepdims=True)
    return (a - lo) / np.maximum(hi - lo, eps), (hi - lo)


def make_masks(feat, out_hw, delta=0.1, linkage_method="average"):
    """feat [D,h,w] -> (masks [K,H,W] in [0,1], n_raw, K)."""
    D, h, w = feat.shape
    flat = feat.reshape(D, -1).astype(np.float64)
    norm, rng = _minmax(flat, axis=1)
    keep = (rng.reshape(-1) > 1e-9)               # drop constant channels
    norm = norm[keep]
    n_raw = norm.shape[0]
    if n_raw == 0:
        raise ValueError("all feature maps are constant")
    nz = np.linalg.norm(norm, axis=1) > 1e-12
    norm = norm[nz]
    if len(norm) == 1:
        groups = np.array([1])
    else:
        dist = pdist(norm, metric="cosine")
        dist = np.nan_to_num(dist, nan=1.0)
        Z = linkage(dist, method=linkage_method)
        groups = fcluster(Z, t=delta, criterion="distance")
    K = groups.max()
    H, W = out_hw
    masks = np.empty((K, H, W), dtype=np.float32)
    for k in range(1, K + 1):
        mean_mask = norm[groups == k].mean(axis=0).reshape(h, w)
        up = zoom(mean_mask, (H / h, W / w), order=1)   # bilinear
        up = up[:H, :W]
        lo, hi = up.min(), up.max()
        masks[k - 1] = (up - lo) / max(hi - lo, 1e-12)
    return masks, n_raw, K


def vitcx_saliency(x, feat_fn, score_fn, delta=0.1, sigma=0.1, fill=None,
                   pcb=True, seed=0, rho_eps=1e-6, return_parts=False):
    """x [C,H,W] -> saliency [A,H,W] in [0,1] (one map per assay)."""
    rng = np.random.default_rng(seed)
    C, H, W = x.shape
    fill = np.zeros((C, 1, 1), np.float32) if fill is None else \
        np.asarray(fill, np.float32).reshape(C, 1, 1)

    # sigma may be a scalar or a per-channel array [C]
    sig = np.asarray(sigma, np.float32)
    sig = sig.reshape(C, 1, 1) if sig.ndim > 0 else sig

    feat = feat_fn(x)
    masks, n_raw, K = make_masks(feat, (H, W), delta=delta)

    f_x = score_fn(x[None].astype(np.float32))[0]            # [A]
    A = f_x.shape[0]

    # masked + noisy images, and unmasked + the same noise
    masked_noisy = np.empty((K, C, H, W), np.float32)
    plain_noisy = np.empty((K, C, H, W), np.float32)
    for k in range(K):
        m = masks[k][None]                                    # [1,H,W]
        eps = ((1.0 - m) * rng.normal(0.0, 1.0, size=(C, H, W)) * sig).astype(np.float32)
        masked_noisy[k] = fill + (x - fill) * m + eps
        plain_noisy[k] = x + eps
    f_masked = score_fn(masked_noisy)                         # [K,A]
    f_plain = score_fn(plain_noisy)                           # [K,A]
    s = f_masked + (f_x[None, :] - f_plain)                   # [K,A]  Eq.(2)

    num = np.tensordot(s.T, masks, axes=([1], [0]))           # [A,H,W] sum_k s_k M_k
    if pcb:
        rho = masks.mean(axis=0)                              # [H,W]
        sal = np.where(rho[None] > rho_eps, num / np.maximum(rho, rho_eps)[None], 0.0)
    else:
        sal = num
    lo = sal.reshape(A, -1).min(axis=1)[:, None, None]
    hi = sal.reshape(A, -1).max(axis=1)[:, None, None]
    sal = (sal - lo) / np.maximum(hi - lo, 1e-12)
    if return_parts:
        return sal.astype(np.float32), dict(K=K, n_raw=n_raw, s=s, f_x=f_x, masks=masks)
    return sal.astype(np.float32)
