import os, sys, pandas as pd
from scipy import stats
R = os.environ.get("L", "/shared/ssd/logs/b-r-singh1") + "/results/cv/"
def folds(stem):
    p = R + stem + "/cv_summary.csv"
    if not os.path.exists(p): return None
    d = pd.read_csv(p); d = d[d["fold"].astype(str).str.isdigit()]
    return d.set_index(d["fold"].astype(int))["mean_test_roc_auc"]
def cmp(a, b, la, lb):
    x, y = folds(a), folds(b)
    if x is None or y is None:
        print(f"== {la} vs {lb}: missing results"); return
    i = x.index.intersection(y.index); d = (x[i] - y[i])
    print(f"== {la} minus {lb} (folds {list(i)})")
    print(d.round(4).to_string())
    if len(i) >= 3:
        print(f"mean diff {d.mean():+.4f}  SD {d.std(ddof=1):.4f}  paired t p={stats.ttest_rel(x[i], y[i]).pvalue:.4f}  {la} better in {(d>0).sum()}/{len(i)}")
    print()
cmp("brightfield_resnet_r448", "bf_dinov3_base_r448", "BF ResNet r448", "BF DINOv3 r448 frozen")
cmp("brightfield_resnet_r448", "bf_celldino_r224", "BF ResNet r448", "BF Cell-DINO r224 frozen")
cmp("brightfield_resnet_r448", "f102_resnet_r448", "BF ResNet r448", "FL ResNet r448")
