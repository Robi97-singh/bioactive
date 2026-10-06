import os, numpy as np, pandas as pd
L = "/shared/ssd/logs/b-r-singh1"
SRC = {"original (lr .02, per-image val)": f"{L}/results_pre_fix_backup_20261007",
       "lr .005 (swapped in)": f"{L}/results",
       "lr .02 (new, compound val)": f"{L}/rerun_all_lr002/results"}
STEMS = ["biomedclip_r224","celldino_r224","celldino_r448","clip_r224","dino_large_r224",
         "dino_large_r448","dino_r224","dino_r448","dino_small_r224","dinov3_r224","dinov3_r448"]
def mean_auc(base, stem):
    v = []
    for f in range(6):
        p = f"{base}/bioact_{stem}_fold{f}/plots/per_assay_auc.csv"
        if os.path.exists(p):
            v.append(pd.read_csv(p, index_col=0).iloc[:, 0].mean())
    return (np.mean(v), len(v)) if v else (np.nan, 0)
rows = []
for s in STEMS:
    r = {"stem": s}
    for k, b in SRC.items():
        m, n = mean_auc(b, s); r[k] = m; r[k.split(" ")[0] + "_n"] = n
    rows.append(r)
d = pd.DataFrame(rows)
cols = list(SRC)
out = d[["stem"] + cols].copy()
out["new-orig"] = out[cols[2]] - out[cols[0]]
print(out.round(4).to_string(index=False))
