import pandas as pd, numpy as np
from scipy import stats
R='/shared/ssd/logs/b-r-singh1/results/cv'
pairs=[('Cell-DINO r224','bf_celldino_r224','f102_celldino_r224'),
       ('DINOv3-Base r448','bf_dinov3_base_r448','f102_dinov3_base_r448')]
def load(stem):
    d=pd.read_csv(f'{R}/{stem}/cv_summary.csv')
    d['fold']=pd.to_numeric(d['fold'],errors='coerce')
    return d.dropna(subset=['fold']).astype({'fold':int})
for name,b,f in pairs:
    m=load(b).merge(load(f),on='fold',suffixes=('_bf','_fl'))
    m['diff']=m.mean_test_roc_auc_bf-m.mean_test_roc_auc_fl
    print(f'\n== {name}: brightfield minus fluorescence (same 102 plates, same test compounds)')
    print(m[['fold','mean_test_roc_auc_bf','mean_test_roc_auc_fl','diff','n_assays_bf','n_assays_fl']].round(4).to_string(index=False))
    d=m['diff'].values
    print(f"mean BF {m.mean_test_roc_auc_bf.mean():.4f}  mean FL {m.mean_test_roc_auc_fl.mean():.4f}  mean diff {d.mean():+.4f}  SD {d.std(ddof=1):.4f}")
    print(f"paired t p={stats.ttest_rel(m.mean_test_roc_auc_bf,m.mean_test_roc_auc_fl).pvalue:.4f}  wilcoxon p={stats.wilcoxon(d).pvalue:.4f} (n=6, min possible 0.03125)  BF better in {(d>0).sum()}/6 folds")
    if (m.n_assays_bf!=m.n_assays_fl).any(): print('WARNING: assay counts differ in some folds; matched-assay comparison needed')
