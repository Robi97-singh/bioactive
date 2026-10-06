import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
fold, stem = sys.argv[1], sys.argv[2]          # e.g. 0 dinov3_r448
d = f"/shared/ssd/logs/b-r-singh1/rerun_all_lr002/results/bioact_{stem}_fold{fold}/plots/"
P = pd.read_csv(d+"test_preds.csv").set_index("compound")
L = pd.read_csv(d+"test_labels.csv").set_index("compound")
R = pd.read_csv(d+"per_assay_auc.csv", index_col=0).iloc[:,0]
rows=[]
for a in L.columns:
    m = L[a].values != 0                      # 0 = untested -> excluded
    y = (L[a].values[m] == 1).astype(int)     # +1 positive, -1 negative
    if y.sum()>0 and (len(y)-y.sum())>0:
        rows.append((a, roc_auc_score(y, P[a].values[m]), R.get(a, np.nan)))
df = pd.DataFrame(rows, columns=["assay","recomputed","reported"])
df["diff"] = (df.recomputed-df.reported).abs()
print(f"[{stem} fold {fold}] assays={len(df)} max|diff|={df['diff'].max():.2e} "
      f"mean recomputed={df.recomputed.mean():.4f} reported={df.reported.mean():.4f}")
