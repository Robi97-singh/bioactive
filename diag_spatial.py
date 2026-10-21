#!/usr/bin/env python3
"""diag_spatial.py -- does the frozen-probe score depend on spatial structure at all?
Same CLI as run_vitcx.py. Env: HEADSTEM, HEADDIR/HEADFILE, NB (batches, default 8).
For test images: image-level ROC-AUC (labels +1 vs -1, 0 = missing) per assay for
  real | pixel-shuffled | per-channel-mean constant, and corr(real logit, perturbed logit)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np, torch                                   # noqa: E402
from easydict import EasyDict as edict                      # noqa: E402
from utils.launch import launch                             # noqa: E402
from utils.helpfuns import load_params                      # noqa: E402
from utils.system_def import define_system_params           # noqa: E402
from defaults.wrappers import DefaultWrapper                # noqa: E402
import classification as C                                  # noqa: E402
import extract_embeddings as X                              # noqa: E402
from run_vitcx import load_head                             # noqa: E402


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def main(parameters, args):
    define_system_params(parameters.system_params)
    X._deterministic_transforms(parameters)
    wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()
    model = wrapper.model
    device = model.device_id if hasattr(model, "device_id") else "cuda"
    model.to(device).eval()
    env = os.environ.get
    stem = env("HEADSTEM", "dinov3")
    hf = env("HEADFILE") or os.path.join(env("HEADDIR", "/shared/ssd/logs/b-r-singh1/xai_heads/checkpoints"),
                                         "bioact_%s_fold%s_head.pt" % (stem, args.fold))
    W, b, (mu, sd), ck = load_head(hf, device)
    nb = int(env("NB", "8"))
    names = {}
    il = getattr(wrapper.dataloaders.testloader.dataset, "int_to_labels", None)
    if isinstance(il, dict):
        names = {int(k): str(v) for k, v in il.items()} if all(isinstance(k, (int, np.integer)) for k in il) \
            else {int(v): str(k) for k, v in il.items()}

    def logits(xb):
        with torch.no_grad():
            o = model(xb, return_embedding=True)
            e = (o[1] if isinstance(o, (tuple, list)) else o).float()
            if mu is not None:
                e = (e - mu) / sd
            return (e @ W.T + b).cpu().numpy()

    g = torch.Generator().manual_seed(0)
    R, S, M, Y = [], [], [], []
    for bi, batch in enumerate(wrapper.dataloaders.testloader):
        x = batch[0].to(device).float()
        B, Ch, H, Wd = x.shape
        perm = torch.randperm(H * Wd, generator=g).to(device)
        xs = x.reshape(B, Ch, -1)[:, :, perm].reshape(x.shape)
        xm = x.mean(dim=(2, 3), keepdim=True).expand_as(x).contiguous()
        R.append(logits(x)); S.append(logits(xs)); M.append(logits(xm))
        Y.append(batch[1].numpy())
        if bi + 1 >= nb:
            break
    R, S, M, Y = [np.concatenate(v) for v in (R, S, M, Y)]
    print("images:", len(R), "| assays:", R.shape[1], "| head:", hf, flush=True)
    rows = []
    for a in range(R.shape[1]):
        m = Y[:, a] != 0
        yb = (Y[m, a] == 1).astype(int)
        minn = int(env("MINN", "15"))
        if yb.sum() < minn or (1 - yb).sum() < minn:
            continue
        rows.append((a, int(m.sum()), auc(yb, R[m, a]), auc(yb, S[m, a]), auc(yb, M[m, a]),
                     np.corrcoef(R[:, a], S[:, a])[0, 1], np.corrcoef(R[:, a], M[:, a])[0, 1]))
    if not rows:
        print("no assay has enough labelled images; raise NB or lower MINN")
        return
    arr = np.array([r[2:] for r in rows])
    print("assays with >=20 pos & neg: %d" % len(rows))
    print("MEAN image-level AUC  real %.4f | shuffled %.4f | mean-const %.4f" % tuple(arr[:, :3].mean(0)))
    print("MEAN corr(real logit, shuffled) %.3f | (real, mean-const) %.3f" % tuple(arr[:, 3:].mean(0)))
    for r in rows:
        print("  assay %-8s idx %2d n=%5d  AUC real %.3f shuf %.3f const %.3f | corr shuf %.2f const %.2f" % (
            names.get(r[0], "?"), r[0], r[1], r[2], r[3], r[4], r[5], r[6]))


if __name__ == "__main__":
    args = C.parse_arguments()
    parameters = edict(C.load_params(args)) if hasattr(C, "load_params") else edict(load_params(args))
    C.update_params_from_args(parameters, args)
    launch(main, (parameters, args))
