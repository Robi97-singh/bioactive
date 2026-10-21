#!/usr/bin/env python3
"""
run_vitcx.py -- ViT-CX saliency (frozen ViT backbone + standardised-CLS linear head) for the
Grad-CAM target list. Run inside a pod. Same CLI as classification.py / check_vit_hooks.py.

  python run_vitcx.py --params_path params/params_cluster_base.json --model <backbone key> \
      --res 448 --fold 0 --save_dir /shared/ssd/logs/b-r-singh1/xai_scratch

Config via env vars (the CLI parser is shared with training, so no extra flags):
  HEADFILE   full path of the saved head (default $HEADDIR/bioact_$HEADSTEM_fold<F>_head.pt)
  HEADDIR    default /shared/ssd/logs/b-r-singh1/xai_heads/checkpoints
  HEADSTEM   e.g. dinov3 | celldino
  TARGETS    path of gradcam_targets.csv (columns auto-detected; printed)
  OUT        output dir (default $SAVE_DIR/vitcx_<HEADSTEM>_<res>)
  LIMIT      max number of targets (smoke test)           default: all
  VARIANTS   comma list from paper,sigma_scaled,fill_median,both   default: all four
  DELTA      clustering threshold                         default 0.1
  CHUNK      forward batch size                           default 32
  EVAL_STEPS deletion/insertion steps                     default 20
"""
import os
import sys
import math
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
from easydict import EasyDict as edict                    # noqa: E402
from utils.launch import launch                           # noqa: E402
from utils.helpfuns import load_params                    # noqa: E402
from utils.system_def import define_system_params         # noqa: E402
from defaults.wrappers import DefaultWrapper              # noqa: E402
import classification as C                                # noqa: E402
import extract_embeddings as X                            # noqa: E402
from vitcx_core import vitcx_saliency                     # noqa: E402
from vitcx_eval import deletion_insertion, random_baseline, map_corr, save_figure  # noqa: E402

ALL_VARIANTS = ["paper", "sigma_scaled", "fill_median", "both"]


def find_last_attention(bb):
    """Last attention module in module order == the attention of the last real block
    (works for HF DINOv3 `layer[-1].attention` and DINOv2-style `blocks[-1][-1].attn`)."""
    last = None
    for name, m in bb.named_modules():
        cn = type(m).__name__.lower()
        if cn.endswith("attention") or name.endswith(".attn") or name.endswith(".attention"):
            last = (name, m)
    return last


def load_head(path, device):
    ck = torch.load(path, map_location="cpu")
    sd = ck["state_dict"]
    wk = [k for k, v in sd.items() if k.endswith("weight") and v.ndim == 2][0]
    bk = wk[:-len("weight")] + "bias"
    W = sd[wk].float().to(device)
    b = sd[bk].float().to(device) if bk in sd else torch.zeros(W.shape[0], device=device)
    std = bool(ck.get("standardize", True))
    mu = ck["mu"].float().to(device) if ck.get("mu") is not None else None
    s = ck["sd"].float().to(device) if ck.get("sd") is not None else None
    return W, b, (mu, s) if std else (None, None), ck


def read_targets(path, fold):
    import csv
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys())
    low = {c: c.lower() for c in cols}
    comp = next((c for c in cols if "compound" in low[c]), None) or \
        next((c for c in cols if "inchi" in low[c]), None)
    assay = next((c for c in cols if "assay" in low[c]), None) or \
        next((c for c in cols if low[c] in ("target", "task", "label", "name")), None)
    fcol = next((c for c in cols if low[c] == "fold"), None)
    print("targets csv columns:", cols, "| using compound:", comp, "assay:", assay, "fold:", fcol, flush=True)
    if comp is None or assay is None:
        raise SystemExit("cannot auto-detect compound/assay columns in %s" % path)
    out = []
    for r in rows:
        if fcol is not None and str(r[fcol]).strip() != str(fold):
            continue
        out.append((str(r[comp]).strip(), str(r[assay]).strip()))
    return out


def main(parameters, args):
    define_system_params(parameters.system_params)
    X._deterministic_transforms(parameters)
    wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()
    model = wrapper.model
    device = model.device_id if hasattr(model, "device_id") else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fold = args.fold
    res = args.res
    env = os.environ.get
    stem = env("HEADSTEM", "dinov3")
    headfile = env("HEADFILE") or os.path.join(
        env("HEADDIR", "/shared/ssd/logs/b-r-singh1/xai_heads/checkpoints"),
        "bioact_%s_fold%s_head.pt" % (stem, fold))
    W, b, (mu, sd), ck = load_head(headfile, device)
    print("head:", headfile, "| W", tuple(W.shape), "| stored test AUC", ck.get("test_mean_roc_auc"),
          "| standardize", mu is not None, flush=True)
    METHOD = env("METHOD", "vitcx")      # vitcx | occlusion | gradcam
    tag_ = stem if METHOD == "vitcx" else "%s-%s" % (METHOD, stem)
    out_dir = env("OUT") or os.path.join(args.save_dir, "vitcx_%s_r%s" % (tag_, res))
    os.makedirs(out_dir, exist_ok=True)
    chunk = int(env("CHUNK", "32"))
    delta = float(env("DELTA", "0.1"))
    steps = int(env("EVAL_STEPS", "20"))
    _dv = {"vitcx": ",".join(ALL_VARIANTS), "occlusion": "occlusion", "gradcam": "vitgradcam,gradxact", "centre": "centre,border"}[METHOD]
    variants = [v for v in env("VARIANTS", _dv).split(",") if v]

    # ---- hook on the last attention module -------------------------------------------
    bb = model.backbone
    found = find_last_attention(bb)
    if found is None:
        raise SystemExit("no attention module found in backbone")
    aname, attn = found
    print("hooking:", aname, type(attn).__name__, flush=True)
    state = {"on": False, "tok": None}

    def _hook(m, i, o):
        if state["on"]:
            t = o[0] if isinstance(o, (tuple, list)) else o
            state["tok"] = t.detach().float().cpu()
    attn.register_forward_hook(_hook)

    reg = getattr(bb, "num_register_tokens", None)
    if reg is None:
        reg = getattr(getattr(bb, "config", None), "num_register_tokens", None)

    def embed(Xb):
        with torch.no_grad():
            o = model(Xb, return_embedding=True)
        e = (o[1] if isinstance(o, (tuple, list)) else o).float()
        return e

    def score_fn(Xnp):
        outs = []
        for i in range(0, len(Xnp), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(Xnp[i:i + chunk])).to(device)
            e = embed(xb)
            if mu is not None:
                e = (e - mu) / sd
            z_ = e @ W.T + b
            outs.append((z_ if env("SCORE", "logit") == "logit" else torch.sigmoid(z_)).cpu().numpy())
        return np.concatenate(outs, 0)

    grid = {}

    def feat_fn(x):
        state["on"] = True
        try:
            embed(torch.from_numpy(x[None].astype(np.float32)).to(device))
        finally:
            state["on"] = False
        t = state["tok"][0]                                  # [N, D]
        N, D = t.shape
        if "p" not in grid:
            cands = []
            if reg is not None:
                cands.append(1 + int(reg))
            cands += [1, 5, 0, 2, 3, 4]
            for p in cands:
                r = N - p
                s = math.isqrt(r) if r > 0 else 0
                if r > 0 and s * s == r:
                    grid["p"], grid["s"] = p, s
                    break
            if "p" not in grid:
                raise SystemExit("cannot infer token layout from N=%d" % N)
            print("tokens N=%d D=%d -> prefix %d + grid %dx%d (num_register_tokens=%s)"
                  % (N, D, grid["p"], grid["s"], grid["s"], reg), flush=True)
        pt = t[grid["p"]:]
        return pt.T.reshape(D, grid["s"], grid["s"]).numpy()

    # ---- baselines: occlusion (numpy) and ViT Grad-CAM (needs a gradient path) -----------
    from scipy.ndimage import zoom as _zoom2
    from xai_baselines import occlusion_map
    gstate = {"on": False, "tok": None}

    def _pre(m, a_, kw_):
        if not gstate["on"]:
            return None
        h = a_[0] if a_ else kw_["hidden_states"]
        h2 = h.detach().clone().requires_grad_(True)
        gstate["tok"] = h2
        if a_:
            return (h2,) + tuple(a_[1:]), kw_
        kw2 = dict(kw_)
        kw2["hidden_states"] = h2
        return a_, kw2
    attn.register_forward_pre_hook(_pre, with_kwargs=True)

    def gradcam_map(x, a, kind):
        """Grad-CAM for ViT on the INPUT tokens of the last attention block (post-norm):
        vitgradcam = ReLU(A . mean_tokens(G)); gradxact = ReLU(sum_c A*G) per patch token."""
        if "p" not in grid:
            feat_fn(x)
        p, s_ = grid["p"], grid["s"]
        gstate["on"] = True
        try:
            xb = torch.from_numpy(x[None].astype(np.float32)).to(device)
            with torch.enable_grad():
                if hasattr(model, "celldino_input_size"):
                    # Cell-DINO wrapper runs its frozen backbone under no_grad: repeat its two steps
                    # (resize to celldino_input_size, forward_features -> CLS) with gradients on.
                    import torch.nn.functional as F_
                    x_in = F_.interpolate(xb, size=(model.celldino_input_size, model.celldino_input_size),
                                          mode="bicubic", align_corners=False, antialias=True)
                    model.backbone.eval()
                    e = model.backbone.forward_features(x_in)["x_norm_clstoken"].float()
                else:
                    o = model(xb, return_embedding=True)
                    e = (o[1] if isinstance(o, (tuple, list)) else o).float()
                if mu is not None:
                    e = (e - mu) / sd
                z = (e @ W.T + b)[0, a]
                if not z.requires_grad:
                    raise SystemExit("NO GRADIENT PATH: the frozen backbone seems to run under no_grad")
                g = torch.autograd.grad(z, gstate["tok"])[0][0]
        finally:
            gstate["on"] = False
        if "chk" not in grid:
            with torch.no_grad():
                o2 = model(xb, return_embedding=True)
            e2 = (o2[1] if isinstance(o2, (tuple, list)) else o2).float()
            e1 = e if mu is None else (e * sd + mu)           # undo standardisation for the comparison
            d_ = float((e1.detach() - e2).abs().max())
            grid["chk"] = d_
            print("  gradcam forward check: max|e_grad - e_standard| = %.2e (|e| max %.2f)"
                  % (d_, float(e2.abs().max())), flush=True)
            if d_ > 1e-2 * max(1.0, float(e2.abs().max())):
                raise SystemExit("GRADCAM FORWARD DIFFERS from the standard forward; maps would be wrong")
        A_ = gstate["tok"][0].detach()[p:]
        g = g[p:]
        cam = torch.relu(A_ @ g.mean(0)) if kind == "vitgradcam" else torch.relu((A_ * g).sum(1))
        cam = cam.reshape(s_, s_).float().cpu().numpy()
        Hh, Ww = x.shape[1:]
        up = _zoom2(cam, (Hh / s_, Ww / s_), order=1)[:Hh, :Ww]
        lo, hi = up.min(), up.max()
        return ((up - lo) / (hi - lo) if hi > lo else np.zeros_like(up)).astype(np.float32)

    # ---- locate targets in this fold's test set --------------------------------------
    targets = read_targets(env("TARGETS", "gradcam_targets.csv"), fold)
    loader = wrapper.dataloaders.testloader
    il = getattr(loader.dataset, "int_to_labels", None)
    if isinstance(il, dict):
        if all(isinstance(k, (int, np.integer)) for k in il.keys()):
            names = {str(v): int(k) for k, v in il.items()}
        else:
            names = {str(k): int(v) for k, v in il.items()}
    else:
        names = {str(n): i for i, n in enumerate(il)} if il is not None else {}
    print("assay names known:", len(names), "| sample:", list(names.items())[:3], flush=True)
    want = {}
    for comp, assay in targets:
        want.setdefault(comp, []).append(assay)
    found_img = {}
    for batch in loader:
        imgs, labels, ids, comps = batch[0], batch[1], batch[2], batch[3]
        for j, c in enumerate(comps):
            c = str(c)
            if c in want and c not in found_img:
                found_img[c] = (imgs[j].float().numpy(), labels[j].numpy() if hasattr(labels[j], "numpy") else labels[j], str(ids[j]))
        if len(found_img) == len(want):
            break
    jobs = [(c, a) for c, a in targets if c in found_img]
    miss = [c for c in want if c not in found_img]
    print("targets: %d requested, %d found, missing compounds: %s" % (len(targets), len(jobs), miss), flush=True)
    lim = env("LIMIT")
    if lim:
        jobs = jobs[:int(lim)]

    rows = []
    for ti, (comp, assay) in enumerate(jobs):
        x, lab, uid = found_img[comp]
        if assay.startswith("#"):          # "#<column index>" -> resolve to the assay name
            assay = {v: k for k, v in names.items()}.get(int(assay[1:]), assay)
        if assay not in names:
            print("assay %r not in head index; skipping" % assay, flush=True)
            continue
        a = names[assay]
        C_, H, Wd = x.shape
        print("\n[%d/%d] fold %s compound %s assay %s (idx %d) uid %s | label %s"
              % (ti + 1, len(jobs), fold, comp, assay, a, uid,
                 (lab[a] if hasattr(lab, "__len__") and len(lab) > a else lab)), flush=True)
        print("  input p1/p50/p99 per channel:",
              [tuple(round(float(v), 2) for v in np.percentile(x[c], [1, 50, 99])) for c in range(C_)], flush=True)
        ch_std = x.reshape(C_, -1).std(axis=1)
        med = np.median(x.reshape(C_, -1), axis=1)
        zero = np.zeros(C_, np.float32)
        f0 = float(score_fn(x[None])[0, a])
        print("  assay prob on real image: %.4f" % f0, flush=True)
        ev_fill = med                                         # common neutral fill for evaluation
        # ---- sensitivity probes: does the score react to the image at all? ----
        r_ = np.random.default_rng(1)
        perm = r_.permutation(H * Wd)
        probes = {
            "real": x,
            "median-const": np.broadcast_to(med.reshape(C_, 1, 1), x.shape),
            "zeros": np.zeros_like(x),
            "noise(med,std)": (med.reshape(C_, 1, 1) + r_.normal(size=x.shape) * ch_std.reshape(C_, 1, 1)),
            "pixel-shuffle": x.reshape(C_, -1)[:, perm].reshape(x.shape),
        }
        for k_, (cid_, (xo_, _, _)) in enumerate(found_img.items()):
            if cid_ != comp:
                probes["other:" + cid_[-6:]] = xo_
        pr_ = score_fn(np.stack([np.asarray(v, np.float32) for v in probes.values()]))[:, a]
        print("  PROBES (prob of this assay):", {k_: round(float(v), 4) for k_, v in zip(probes, pr_)}, flush=True)
        pa_ = score_fn(np.stack([np.asarray(v, np.float32) for v in probes.values()]))
        print("  PROBES all-assay std across probes (mean over assays): %.4f" % float(pa_.std(axis=0).mean()), flush=True)
        sc1 = lambda Xb, a=a: score_fn(Xb)[:, a]              # noqa: E731
        nr_ = int(env("NRAND", "10"))
        rr_ = np.random.default_rng(123)
        from scipy.ndimage import zoom as _zoom
        G_ = int(env("RGRID", "14"))
        rmap_ = lambda: _zoom(rr_.random((G_, G_)), (H / G_, Wd / G_), order=1)[:H, :Wd]  # noqa: E731
        rnd_ = np.array([deletion_insertion(x, rmap_(), sc1, ev_fill, steps=steps)[:2]
                         for _ in range(nr_)])
        rd, ri = float(rnd_[:, 0].mean()), float(rnd_[:, 1].mean())
        rds, ris = float(rnd_[:, 0].std(ddof=1)) + 1e-12, float(rnd_[:, 1].std(ddof=1)) + 1e-12
        maps, tag = [], []
        stem_name = "f%s_%s_%s" % (fold, comp.replace("/", "_")[:40], assay.replace("/", "_")[:40])
        sal_store = {}
        for v in variants:
            sigma = (0.1 * ch_std) if v in ("sigma_scaled", "both") else 0.1
            fill = med if v in ("fill_median", "both") else zero
            t0 = time.time()
            if METHOD == "vitcx":
                sal, parts = vitcx_saliency(x, feat_fn, score_fn, delta=delta, sigma=sigma, fill=fill,
                                            pcb=True, seed=0, return_parts=True)
                s = sal[a]
            elif METHOD == "occlusion":
                s = occlusion_map(x, a, med, score_fn, chunk=chunk)
                parts = dict(K=0, n_raw=0)
            elif METHOD == "centre":      # control: fixed map independent of the image
                _yy, _xx = np.mgrid[:H, :Wd]
                s = np.exp(-(((_yy - H / 2.0) ** 2 + (_xx - Wd / 2.0) ** 2) / (2.0 * (min(H, Wd) / 4.0) ** 2))).astype(np.float32)
                if v == "border":
                    s = 1.0 - s
                parts = dict(K=0, n_raw=0)
            else:
                s = gradcam_map(x, a, v)
                parts = dict(K=0, n_raw=0)
            da, ia, sd_c, si_c = deletion_insertion(x, s, sc1, ev_fill, steps=steps)
            print("  %s deletion curve: %s" % (v, [round(float(q), 3) for q in sd_c[::5]]), flush=True)
            print("  %s insertion curve: %s" % (v, [round(float(q), 3) for q in si_c[::5]]), flush=True)
            sal_store[v] = s
            maps.append(("%s  del %.3f ins %.3f" % (v, da, ia), s))
            rows.append(dict(fold=fold, model=stem, res=res, compound=comp, assay=assay, variant=v,
                             K=int(parts["K"]), n_raw=int(parts["n_raw"]), prob=f0,
                             del_auc=da, ins_auc=ia, rand_del=rd, rand_ins=ri, rand_del_sd=rds, rand_ins_sd=ris,
                             z_del=(da - rd) / rds, z_ins=(ia - ri) / ris,
                             sec=round(time.time() - t0, 1)))
            print("  %-12s K=%3d/%3d del %.3f (rand %.3f sd %.3f) ins %.3f (rand %.3f sd %.3f) | z_del %+.1f z_ins %+.1f %.0fs"
                  % (v, parts["K"], parts["n_raw"], da, rd, rds, ia, ri, ris,
                     (da - rd) / rds, (ia - ri) / ris, time.time() - t0), flush=True)
        if "paper" in sal_store:
            for v in variants:
                if v != "paper":
                    print("  corr(paper,%s) = %.3f" % (v, map_corr(sal_store["paper"], sal_store[v])), flush=True)
        np.savez_compressed(os.path.join(out_dir, stem_name + ".npz"), x=x,
                            **{k: v_.astype(np.float16) for k, v_ in sal_store.items()})
        save_figure(x, maps, "%s r%s | fold %s | %s | %s | p=%.3f" % (stem, res, fold, comp, assay, f0),
                    os.path.join(out_dir, stem_name + ".png"))

    import csv
    mp = os.path.join(out_dir, "metrics_fold%s.csv" % fold)
    if rows:
        with open(mp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", mp, "| rows", len(rows), flush=True)
        for v in variants:
            r = [q for q in rows if q["variant"] == v]
            if r:
                print("  %-12s mean del %.3f ins %.3f | random del %.3f ins %.3f | mean z_del %+.2f z_ins %+.2f | n=%d" % (
                    v, np.mean([q["del_auc"] for q in r]), np.mean([q["ins_auc"] for q in r]),
                    np.mean([q["rand_del"] for q in r]), np.mean([q["rand_ins"] for q in r]),
                    np.mean([q["z_del"] for q in r]), np.mean([q["z_ins"] for q in r]), len(r)))


if __name__ == "__main__":
    args = C.parse_arguments()
    parameters = edict(C.load_params(args)) if hasattr(C, "load_params") \
        else edict(load_params(args))
    C.update_params_from_args(parameters, args)
    launch(main, (parameters, args))
