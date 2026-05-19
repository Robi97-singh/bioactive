import sqlite3
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_PATH     = "/mnt/ssd8/bioactive/"
METADATA_PATH = BASE_PATH + "src/data/metadata/"
OUTPUT_PATH   = BASE_PATH + "src/data/"
CHEMBL_DB     = BASE_PATH + "chembl/chembl_37/chembl_37_sqlite/chembl_37.db"

# ── Activity thresholds (paper-faithful) ─────────────────────────────────────
ACTIVE_THRESHOLD   = 6.0
INACTIVE_THRESHOLD = 5.0

# Lowered to match paper (~140 targets)
MIN_ACTIVES   = 3
MIN_INACTIVES = 3

# Use all JUMP-CP sources (None = all sources)
SOURCES_TO_USE = None

# ── SQL query ────────────────────────────────────────────────────────────────
ACTIVITY_QUERY = """
SELECT
    md.chembl_id                    AS molecule_chembl_id,
    cs.canonical_smiles             AS canonical_smiles,
    act.pchembl_value               AS pchembl_value,
    act.standard_type               AS standard_type,
    ass.chembl_id                   AS assay_chembl_id,
    td.chembl_id                    AS target_chembl_id,
    td.pref_name                    AS target_name,
    td.target_type                  AS target_type,
    td.organism                     AS organism
FROM
    activities          act
    JOIN assays         ass ON act.assay_id       = ass.assay_id
    JOIN target_dictionary td ON ass.tid           = td.tid
    JOIN molecule_dictionary md ON act.molregno    = md.molregno
    JOIN compound_structures cs ON act.molregno    = cs.molregno
WHERE
    act.standard_type  IN ('IC50', 'Ki', 'Kd', 'EC50')
    AND act.pchembl_value IS NOT NULL
    AND act.data_validity_comment IS NULL
    AND act.potential_duplicate = 0
    AND td.target_type = 'SINGLE PROTEIN'
    AND td.organism    = 'Homo sapiens'
    AND cs.canonical_smiles IS NOT NULL
"""


def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def load_chembl_activities():
    print(f"Connecting to ChEMBL SQLite: {CHEMBL_DB}")
    if not os.path.exists(CHEMBL_DB):
        raise FileNotFoundError(f"ChEMBL database not found at {CHEMBL_DB}")
    conn = sqlite3.connect(CHEMBL_DB)
    print("Running activity query (this may take 2-5 minutes)...")
    df = pd.read_sql_query(ACTIVITY_QUERY, conn)
    conn.close()
    print(f"  Raw records fetched : {len(df):,}")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])
    print(f"  After dropping nulls: {len(df):,}")
    print(f"  Unique targets      : {df['target_chembl_id'].nunique():,}")
    print(f"  Unique compounds    : {df['molecule_chembl_id'].nunique():,}")
    return df


def derive_inchikeys(df):
    print("\nDeriving InChIKeys from SMILES...")
    unique_smiles = df["canonical_smiles"].unique()
    print(f"  Computing InChIKeys for {len(unique_smiles):,} unique SMILES...")
    smiles_to_key = {}
    for i, smi in enumerate(unique_smiles):
        if i % 50000 == 0:
            print(f"    {i:,}/{len(unique_smiles):,}", end="\r")
        smiles_to_key[smi] = smiles_to_inchikey(smi)
    print(f"\n  Done.")
    df = df.copy()
    df["standard_inchi_key"] = df["canonical_smiles"].map(smiles_to_key)
    df = df.dropna(subset=["standard_inchi_key"])
    print(f"  Records with valid InChIKey: {len(df):,}")
    return df


def build_target_label_matrix(activity_df, jump_inchikeys):
    print("\nBuilding target-level label matrix...")
    df = activity_df[activity_df["standard_inchi_key"].isin(jump_inchikeys)].copy()
    print(f"  Records overlapping with JUMP-CP : {len(df):,}")
    print(f"  Unique compounds                 : {df['standard_inchi_key'].nunique():,}")
    print(f"  Unique targets                   : {df['target_name'].nunique():,}")

    if df.empty:
        raise ValueError("No overlapping compounds between ChEMBL and JUMP-CP.")

    agg = df.groupby(
        ["standard_inchi_key", "target_name", "target_chembl_id"]
    )["pchembl_value"].median().reset_index()
    agg.rename(columns={"pchembl_value": "median_pchembl"}, inplace=True)

    agg["target_label"] = 0
    agg.loc[agg["median_pchembl"] >= ACTIVE_THRESHOLD,   "target_label"] =  1
    agg.loc[agg["median_pchembl"] <= INACTIVE_THRESHOLD, "target_label"] = -1
    agg = agg[agg["target_label"] != 0]

    label_matrix = agg.pivot_table(
        values="target_label",
        index="standard_inchi_key",
        columns="target_name",
        aggfunc="first",
    ).fillna(0)

    print(f"\n  Label matrix shape (before filter): {label_matrix.shape}")

    valid_targets = [
        col for col in label_matrix.columns
        if (label_matrix[col] ==  1).sum() >= MIN_ACTIVES
        and (label_matrix[col] == -1).sum() >= MIN_INACTIVES
    ]
    label_matrix = label_matrix[valid_targets]

    print(f"  Targets passing quality filter   : {len(valid_targets)}")
    print(f"  Label matrix shape (after filter): {label_matrix.shape}")
    print(f"\n  Per-target class counts:")

    target_stats = []
    for col in label_matrix.columns:
        n_pos = (label_matrix[col] ==  1).sum()
        n_neg = (label_matrix[col] == -1).sum()
        target_stats.append((col, n_pos, n_neg))
    target_stats.sort(key=lambda x: -x[1])
    for col, n_pos, n_neg in target_stats:
        print(f"    {col[:50]:50s}  active={n_pos:4d}  inactive={n_neg:4d}")

    target_list_path = OUTPUT_PATH + "valid_targets.csv"
    pd.DataFrame(
        target_stats, columns=["target_name", "n_actives", "n_inactives"]
    ).to_csv(target_list_path, index=False)
    print(f"\n  Target list saved -> {target_list_path}")
    return label_matrix


def merge_with_jump(label_matrix, compound, wells, plate):
    print("\nMerging with JUMP-CP metadata...")

    lm = label_matrix.reset_index()
    lm["Metadata_JCP2022"] = lm["standard_inchi_key"].map(
        compound.set_index("Metadata_InChIKey")["Metadata_JCP2022"]
    )
    lm = lm.dropna(subset=["Metadata_JCP2022"])
    print(f"  Compounds mapped to JCP IDs: {len(lm):,}")

    if SOURCES_TO_USE is not None:
        plate_filtered = plate[
            (plate["Metadata_Source"].isin(SOURCES_TO_USE)) &
            (plate["Metadata_PlateType"] == "COMPOUND")
        ]
        print(f"  Using sources: {SOURCES_TO_USE}")
    else:
        plate_filtered = plate[plate["Metadata_PlateType"] == "COMPOUND"]
        print(f"  Using ALL sources ({plate['Metadata_Source'].nunique()} sources)")

    valid_plates = plate_filtered["Metadata_Plate"].values

    wells_filtered = wells[
        (wells["Metadata_Plate"].isin(valid_plates)) &
        (wells["Metadata_JCP2022"].isin(lm["Metadata_JCP2022"]))
    ].copy()
    print(f"  Matched wells: {len(wells_filtered):,}")

    if wells_filtered.empty:
        raise ValueError("No matching wells found.")

    wells_filtered["_key"] = 0
    df_sites = pd.DataFrame({"Metadata_Site": range(1, 10), "_key": 0})
    well_with_sites = wells_filtered[
        ["Metadata_Plate", "Metadata_Well", "Metadata_JCP2022",
         "Metadata_Source", "_key"]
    ].merge(df_sites, on="_key").drop(columns="_key")

    well_with_sites["Metadata_Path"] = (
        well_with_sites["Metadata_Plate"].astype(str) + "/" +
        well_with_sites["Metadata_Well"]              + "_" +
        well_with_sites["Metadata_Site"].astype(str)  + ".png"
    )

    target_cols = [
        c for c in lm.columns
        if c not in ["standard_inchi_key", "Metadata_JCP2022"]
    ]
    final = well_with_sites.merge(
        lm[["Metadata_JCP2022"] + target_cols],
        on="Metadata_JCP2022",
        how="left",
    )

    print(f"  Final rows        : {len(final):,}")
    print(f"  Unique compounds  : {final['Metadata_JCP2022'].nunique():,}")
    print(f"  Unique plates     : {final['Metadata_Plate'].nunique():,}")
    print(f"  Unique sources    : {final['Metadata_Source'].nunique():,}")
    print(f"  Target columns    : {len(target_cols)}")
    return final


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading JUMP-CP metadata...")
    compound = pd.read_csv(METADATA_PATH + "compound.csv.gz")
    wells    = pd.read_csv(METADATA_PATH + "well.csv.gz")
    plate    = pd.read_csv(METADATA_PATH + "plate.csv.gz")
    print(f"  Compounds: {compound.shape[0]:,}  Wells: {wells.shape[0]:,}  Plates: {plate.shape[0]:,}")
    jump_inchikeys = set(compound["Metadata_InChIKey"].dropna().unique())
    print(f"  Unique InChIKeys in JUMP-CP: {len(jump_inchikeys):,}")

    activity_df  = load_chembl_activities()
    activity_df  = derive_inchikeys(activity_df)
    label_matrix = build_target_label_matrix(activity_df, jump_inchikeys)
    final        = merge_with_jump(label_matrix, compound, wells, plate)

    out_path = OUTPUT_PATH + "data.csv"
    final.to_csv(out_path, index=False)
    print(f"\n✅ Saved  -> {out_path}")
    print(f"   Shape  : {final.shape}")
    print(f"   Columns: {list(final.columns[:8])} ... +{len(final.columns)-8} target cols")


if __name__ == "__main__":
    main()