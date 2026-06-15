import pandas as pd
import numpy as np
import random as rd
from collections import defaultdict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

DATA_CSV  = "/mnt/ssd8/bioactive/src/data/data_paper.csv"
OUT_CSV   = "/mnt/ssd8/bioactive/src/data/split_data_paper.csv"
rd.seed(42); np.random.seed(42)

def ClusterFps(fps, cutoff=0.2):
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    return Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

data = pd.read_csv(DATA_CSV, index_col=0)
unique = data.drop_duplicates(subset="Metadata_InChI")
print("Unique compounds:", len(unique))

ms  = [Chem.MolFromInchi(unique.iloc[i].Metadata_InChI) for i in range(len(unique))]
ok  = [(m, unique.iloc[i].Metadata_JCP2022) for i, m in enumerate(ms) if m is not None]
print("Valid mols:", len(ok))
mols = [m for m, _ in ok]
jcps = [j for _, j in ok]

fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
clusters = list(ClusterFps(fps, cutoff=0.7))
print("Clusters:", len(clusters))

# assign clusters to 6 folds of ~equal compound count
rd.shuffle(clusters)
n_unique = len(jcps)
target_per_fold = n_unique / 6.0
folds = [[] for _ in range(6)]
ci = 0
for f in range(6):
    while len(folds[f]) <= target_per_fold and ci < len(clusters):
        folds[f].extend([jcps[idx] for idx in clusters[ci]])
        ci += 1
# dump any remainder into last fold
while ci < len(clusters):
    folds[5].extend([jcps[idx] for idx in clusters[ci]]); ci += 1

data["split_number"] = -1
for f in range(6):
    data.loc[data.Metadata_JCP2022.isin(folds[f]), "split_number"] = f

print("Per-fold compound counts:")
print(data.drop_duplicates("Metadata_JCP2022").split_number.value_counts().sort_index())
print("Rows still unassigned (-1):", (data.split_number == -1).sum())

data.to_csv(OUT_CSV)
print("Wrote", OUT_CSV, "shape", data.shape)
