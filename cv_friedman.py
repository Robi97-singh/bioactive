import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata
CV = "/shared/ssd/logs/b-r-singh1/results/cv"
M = ["celldino_r224","celldino_r448","dino_r448","dino_large_r448","dino_r224",
     "dinov3_r448","dino_large_r224","dinov3_r224","clip_r224","dino_small_r224",
     "biomedclip_r224","resnet_r448"]
Q05 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164,11:3.219,12:3.268}
cols = {}
for m in M:
    d = pd.read_csv(f"{CV}/{m}/cv_summary.csv")
    d = d[d["fold"].astype(str).isin([str(i) for i in range(6)])]
    cols[m] = d["mean_test_roc_auc"].astype(float).values
A = pd.DataFrame(cols)
print(A.round(4).to_string()); print("folds:", len(A))
k, N = A.shape[1], A.shape[0]
stat, p = friedmanchisquare(*[A[m].values for m in A])
print(f"\nFriedman chi2={stat:.2f}, p={p:.3g}  (k={k} models, N={N} folds)")
R = np.vstack([rankdata(-A.iloc[i].values) for i in range(N)]).mean(axis=0)
cd = Q05[k]*np.sqrt(k*(k+1)/(6*N))
print(f"Nemenyi CD (alpha=0.05) = {cd:.2f} ranks\n")
rk = pd.Series(R, index=A.columns).sort_values()
print("mean rank (1 = best):"); print(rk.round(2).to_string())
print("\npairs differing by more than CD:")
n = 0
for i, a in enumerate(rk.index):
    for b in rk.index[i+1:]:
        if rk[b]-rk[a] > cd:
            print(f"  {a} > {b}  (diff {rk[b]-rk[a]:.2f})"); n += 1
if n == 0: print("  none")
