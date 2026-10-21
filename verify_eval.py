import sys
import pandas as pd
fold = int(sys.argv[1]); model = sys.argv[2]
d = f"/shared/ssd/logs/b-r-singh1/results/bioact_{model}_fold{fold}/plots"
P = pd.read_csv(f"{d}/test_preds.csv", index_col=0)
L = pd.read_csv(f"{d}/test_labels.csv", index_col=0)
R = pd.read_csv(f"{d}/per_assay_auc.csv", index_col=0).iloc[:, 0]
assert list(P.index) == list(L.index), "compound order differs"
def auc(s, y):
    r = s.rank(method="average"); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
mine = {}
for a in P.columns:
    m = L[a] != -1
    y, s = L.loc[m, a], P.loc[m, a]
    if (y == 1).sum() > 0 and (y == 0).sum() > 0:
        mine[a] = auc(s, y)
mine = pd.Series(mine)
common = R.index.intersection(mine.index)
print(f"[{model} fold {fold}] A. recomputed vs reported AUC: {len(common)} assays | max abs diff = {(R[common] - mine[common]).abs().max():.2e}")
print("   reported but not recomputable:", sorted(set(R.index) - set(mine.index)))
print("   computable but not reported:", sorted(set(mine.index) - set(R.index)))
print(f"   mean reported = {R.mean():.4f} | mean recomputed (same assays) = {mine[common].mean():.4f}")
T = pd.read_csv("/shared/hdd/data/bioactive/training_paper.csv", usecols=["Metadata_JCP2022", "split_number"])
n_sp = T.groupby("Metadata_JCP2022")["split_number"].nunique()
test_c = set(T.loc[T.split_number == fold, "Metadata_JCP2022"])
other_c = set(T.loc[T.split_number != fold, "Metadata_JCP2022"])
pc = set(P.index)
print(f"   B. compounds in more than one split: {(n_sp > 1).sum()} of {len(n_sp)} | test overlap with other splits: {len(test_c & other_c)}")
print(f"      predicted compounds: {len(pc)} | not in split {fold}: {len(pc - test_c)} | split-{fold} compounds with no prediction: {len(test_c - pc)}")
