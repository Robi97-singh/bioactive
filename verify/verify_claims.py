"""Recompute the headline numbers from the committed CSV files and compare
them with the values claimed in README.md and the documents.
Run from anywhere:  python3 verify/verify_claims.py
Prints PASS / FAIL / MISSING per claim; exit code 1 if any FAIL."""
import csv, os, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results_showcase")
out = []

def chk(name, got, want, tol):
    if got is None:
        out.append("MISSING"); print("MISSING", name); return
    ok = abs(got - want) <= tol
    out.append("PASS" if ok else "FAIL")
    print("PASS " if ok else "FAIL ", name, "| claimed", want, "| recomputed", round(got, 4))

def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# ---- brightfield vs fluorescence ----
B = os.path.join(R, "brightfield_vs_fluorescence")
def folds(d):
    r = rows(os.path.join(B, d, "cv_summary.csv"))
    return [float(x["mean_test_roc_auc"]) for x in sorted([y for y in r if y["fold"].isdigit()], key=lambda x: int(x["fold"]))]
bf = folds("brightfield_resnet_r448")
fl = folds("fluorescence_resnet_r448_102plates")
diff = [a - b for a, b in zip(bf, fl)]
chk("BF mean ROC-AUC (6 folds)", st.mean(bf), 0.6940, 6e-5)
chk("BF SD", st.stdev(bf), 0.0287, 6e-5)
chk("FL mean ROC-AUC (6 folds)", st.mean(fl), 0.7097, 6e-5)
chk("FL SD", st.stdev(fl), 0.0484, 6e-5)
chk("mean difference BF-FL", st.mean(diff), -0.0157, 6e-5)
chk("folds where BF is better (of 6)", float(sum(d > 0 for d in diff)), 2.0, 0)
chk("folds where FL is ahead (of 6)", float(sum(d < 0 for d in diff)), 4.0, 0)
try:
    from scipy import stats
    chk("paired t-test p-value", float(stats.ttest_rel(bf, fl).pvalue), 0.1864, 6e-5)
except ImportError:
    chk("paired t-test p-value (scipy missing)", None, 0.1864, 6e-5)

# ---- explainability, 65 random targets (enrichment, group "all") ----
X = os.path.join(R, "explainability", "random_65_targets")
E = {(r["method"], r["group"]): r for r in rows(os.path.join(X, "enrich_r65.csv"))}
CLAIMS = {  # method: (mean, lo, hi) as stated in the README and documents
    "DINOv3 ViT-CX": (1.30, 1.07, 1.53),
    "DINOv3 grad x act": (3.27, 2.77, 3.81),
    "Cell-DINO ViT-CX": (1.01, 0.93, 1.09),
    "Cell-DINO grad x act": (2.29, 1.62, 3.14),
    "Cell-DINO occlusion": (1.07, 0.98, 1.16),
}
for m, (mean, lo, hi) in CLAIMS.items():
    r = E.get((m, "all"))
    for lab, key, want in (("mean", "enrich", mean), ("CI low", "lo", lo), ("CI high", "hi", hi)):
        chk("enrichment " + m + " " + lab, float(r[key]) if r else None, want, 0.0051)

# ---- ResNet-50 Grad-CAM reference (per-target file, own bootstrap) ----
import numpy as np
pr = os.path.join(X, "enrich_resnet_per_target.csv")
if os.path.exists(pr):
    allr = rows(pr)
    v = np.array([float(r["enrich"]) for r in allr if r["enrich"] != ""])
    g = np.random.default_rng(0)
    bm = np.sort([g.choice(v, len(v)).mean() for _ in range(10000)])
    chk("ResNet-50 Grad-CAM rows (targets)", float(len(allr)), 65.0, 0)
    chk("ResNet-50 Grad-CAM valid values (map not empty)", float(len(v)), 58.0, 0)
    chk("enrichment ResNet-50 Grad-CAM mean", float(v.mean()), 1.07, 0.0051)
    chk("enrichment ResNet-50 Grad-CAM CI low", float(bm[250]), 0.90, 0.03)
    chk("enrichment ResNet-50 Grad-CAM CI high", float(bm[9750]), 1.28, 0.03)
else:
    chk("ResNet-50 Grad-CAM file missing", None, 0, 0)
print()
print("methods available in enrich_r65.csv:", sorted({k[0] for k in E}))
print("summary:", {k: out.count(k) for k in ("PASS", "FAIL", "MISSING")})
sys.exit(1 if "FAIL" in out else 0)
