import pandas as pd
import numpy as np
import random as rd
import os
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import inchi as rdinchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH     = "/mnt/ssd8/bioactive/"
DATA_PATH     = BASE_PATH + "src/data/data.csv"
METADATA_PATH = BASE_PATH + "src/data/metadata/"
OUTPUT_PATH   = BASE_PATH + "src/data/split_data.csv"

RANDOM_SEED   = 42
N_SPLITS      = 6
TANIMOTO_CUTOFF = 0.7   # Butina clustering cutoff (paper setting)

# ──────────────────────────────────────────────────────────────────────────────

def ClusterFps(fps, cutoff=0.2):
    """Butina clustering on ECFP4 fingerprints."""
    dists = []
    nfps  = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


def split_data_cross_validation_structure_clustering(
    clusters, unique_df, n_splits=6
):
    """
    Assign each cluster to one of n_splits folds.
    Folds are filled sequentially until each reaches ~1/n_splits of compounds.
    """
    unique_cmp  = list(unique_df["Metadata_JCP2022"].unique())
    target_size = len(unique_cmp) // n_splits

    clusters = list(clusters)
    rd.shuffle(clusters)

    data_splits   = []
    cluster_index = 0

    for i in range(n_splits):
        split_data = []
        while len(split_data) < target_size and cluster_index < len(clusters):
            split_data.extend(
                unique_df.iloc[list(clusters[cluster_index])]["Metadata_JCP2022"].values
            )
            cluster_index += 1
        data_splits.append(split_data)
        print(f"  Fold {i}: {len(split_data)} compounds  (clusters used so far: {cluster_index})")

    # Assign any remaining clusters to the last fold
    while cluster_index < len(clusters):
        data_splits[-1].extend(
            unique_df.iloc[list(clusters[cluster_index])]["Metadata_JCP2022"].values
        )
        cluster_index += 1

    return data_splits


def main():
    rd.seed(RANDOM_SEED)

    # 1. Load data
    print("Loading data.csv...")
    data = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Shape: {data.shape}")

    # 2. Join InChI strings from compound metadata (needed for scaffold/fingerprint)
    print("Joining InChI strings from compound metadata...")
    compond = pd.read_csv(METADATA_PATH + "compound.csv.gz")
    data    = data.merge(
        compond[["Metadata_JCP2022", "Metadata_InChI", "Metadata_InChIKey"]],
        on="Metadata_JCP2022", how="left"
    )
    print(f"  Shape after join: {data.shape}")
    print(f"  Missing InChI   : {data['Metadata_InChI'].isna().sum()}")

    # 3. Get unique compounds
    unique = data.drop_duplicates(subset="Metadata_JCP2022").dropna(subset=["Metadata_InChI"])
    print(f"  Unique compounds with InChI: {len(unique)}")

    # 4. Build RDKit molecules from InChI — FIX: use rdinchi.MolFromInchi
    print("\nBuilding molecules from InChI strings...")
    ms = []
    valid_idx = []
    for i in range(len(unique)):
        inchi_str = unique.iloc[i]["Metadata_InChI"]
        mol = rdinchi.MolFromInchi(inchi_str)
        if mol is not None:
            ms.append(mol)
            valid_idx.append(i)
        else:
            print(f"  WARNING: Could not parse InChI for row {i}")

    unique = unique.iloc[valid_idx].reset_index(drop=True)
    print(f"  Valid molecules: {len(ms)}")

    # 5. Compute ECFP4 fingerprints
    print("Computing ECFP4 fingerprints (radius=2, 1024 bits)...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in ms]

    # 6. Butina clustering at 0.7 Tanimoto cutoff (paper setting)
    print(f"Running Butina clustering (cutoff={TANIMOTO_CUTOFF})...")
    clusters = ClusterFps(fps, cutoff=TANIMOTO_CUTOFF)
    clu_list = list(clusters)
    print(f"  Total clusters: {len(clu_list)}")
    print(f"  Largest cluster size: {max(len(c) for c in clu_list)}")

    # 7. Assign clusters to folds
    print(f"\nAssigning clusters to {N_SPLITS} folds...")
    splits = split_data_cross_validation_structure_clustering(clu_list, unique, N_SPLITS)

    # 8. Label each row in data with its fold
    data["split_number"] = -1
    for i, fold_jcp_ids in enumerate(splits):
        data.loc[data["Metadata_JCP2022"].isin(fold_jcp_ids), "split_number"] = i

    unassigned = (data["split_number"] == -1).sum()
    print(f"\nUnassigned rows: {unassigned}")
    print("Split distribution:")
    print(data["split_number"].value_counts().sort_index())

    # 9. Save
    data.to_csv(OUTPUT_PATH)
    print(f"\nSaved -> {OUTPUT_PATH}")
    print(f"Shape : {data.shape}")


if __name__ == "__main__":
    main()