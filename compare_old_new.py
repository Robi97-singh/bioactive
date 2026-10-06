import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
f = sys.argv[1]; stem = sys.argv[2] if len(sys.argv) > 2 else "dinov3_r448"
L = "/shared/ssd/logs/b-r-singh1"
od = f"{L}/results_pre_fix_backup_20261007/bioact_{stem}_fold{f}/plots/"
nd = f"{L}/results/bioact_{stem}_fold{f}/plots/"
def ld(d):
    P = pd.read_csv(d+"test_preds.csv").set_index("compound")
    Y = pd.read_csv(d+"test_labels.csv").set_index("compound")
    R = pd.read_csv(d+"per_assay_auc.csv", index_col=0).iloc[:, 0]
    return P, Y, R
Po, Yo, Ro = ld(od); Pn, Yn, Rn = ld(nd)
print(f"old rows {len(Po)}, new rows {len(Pn)}, reported old mean {Ro.mean():.4f}, new {Rn.mean():.4f}")
com = Po.index.intersection(Pn.index)
print(f"common compounds {len(com)}; only-old {len(Po.index.difference(Pn.index))}; only-new {len(Pn.index.difference(Po.index))}")
cols = [c for c in Yn.columns if c in Yo.columns]
same = (Yo.loc[com, cols].values == Yn.loc[com, cols].values).mean()
print(f"label agreement on common compounds: {same:.4f}")
def auc(P, Y, idx):
    out = []
    for a in cols:
        y = Y.loc[idx, a].values; m = y != 0
        t = (y[m] == 1).astype(int)
        if t.sum() > 0 and (len(t) - t.sum()) > 0:
            out.append(roc_auc_score(t, P.loc[idx, a].values[m]))
    return np.mean(out)
print(f"AUC on common compounds (labels = new): old preds {auc(Po, Yn, com):.4f} | new preds {auc(Pn, Yn, com):.4f}")
print(f"AUC old preds with old labels, own rows: {auc(Po, Yo, Po.index):.4f}")
cor = np.mean([np.corrcoef(Po.loc[com, a], Pn.loc[com, a])[0, 1] for a in cols])
print(f"mean per-assay correlation of old vs new predictions: {cor:.3f}")
