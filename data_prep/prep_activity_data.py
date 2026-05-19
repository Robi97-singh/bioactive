import requests
import pandas as pd
import numpy as np
import os
import time
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

BASE_PATH     = os.path.expanduser("~/PycharmProjects/bioactive/")
METADATA_PATH = BASE_PATH + "src/data/metadata/"
OUTPUT_PATH   = BASE_PATH + "src/data/"
CHEMBL_API    = "https://www.ebi.ac.uk/chembl/api/data"

TARGET_IDS = {
    "EGFR":  "CHEMBL203",
    "AURKA": "CHEMBL2148",
    "CDK2":  "CHEMBL301",
    "BRAF":  "CHEMBL5145",
    "ABL1":  "CHEMBL1862",
}

# pchembl_value thresholds
# >= 6  →  IC50 <= 1µM  → active
# <= 5  →  IC50 >= 10µM → inactive
ACTIVE_THRESHOLD   = 6.0
INACTIVE_THRESHOLD = 5.0

# Per-TARGET (not per-assay) minimum class size
MIN_ACTIVES   = 10
MIN_INACTIVES = 10


def api_get(url, params=None, retries=3):
    params = params or {}
    params["format"] = "json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries} - {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_all_pages(url, params, result_key):
    results = []
    params  = {**params, "limit": 1000, "offset": 0}
    while True:
        data = api_get(url, params)
        if data is None:
            break
        batch = data.get(result_key, [])
        results.extend(batch)
        total = data.get("page_meta", {}).get("total_count", "?")
        print(f"    ...{len(results)}/{total}", end="\r")
        if not data.get("page_meta", {}).get("next"):
            break
        params["offset"] += params["limit"]
    print()
    return results


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


def get_activities_for_target(target_id, target_name):
    print(f"\n  [{target_name}] Fetching activities...")
    records = fetch_all_pages(
        f"{CHEMBL_API}/activity.json",
        {
            "target_chembl_id":      target_id,
            "standard_type__in":     "IC50,Ki,Kd,EC50",
            "pchembl_value__isnull": "false",
        },
        "activities",
    )
    if not records:
        print(f"    -> No records.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"    -> {len(df):,} raw records")

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])

    # Label per threshold
    df["activity_label"] = np.nan
    df.loc[df["pchembl_value"] >= ACTIVE_THRESHOLD,   "activity_label"] =  1
    df.loc[df["pchembl_value"] <= INACTIVE_THRESHOLD, "activity_label"] = -1
    df = df.dropna(subset=["activity_label"])
    print(f"    -> {len(df):,} labelled  (+:{(df.activity_label==1).sum()}  -:{(df.activity_label==-1).sum()})")

    if df.empty:
        return pd.DataFrame()

    # Derive InChIKey from SMILES
    df["standard_inchi_key"] = df["canonical_smiles"].apply(smiles_to_inchikey)
    df = df.dropna(subset=["standard_inchi_key"])
    print(f"    -> {df['standard_inchi_key'].nunique():,} unique compounds with InChIKey")

    df["target_name"] = target_name
    return df[["standard_inchi_key", "activity_label", "pchembl_value",
               "assay_chembl_id", "target_name"]]


def build_target_label_matrix(activity_df, jump_inchikeys):
    """
    Aggregate activity labels per compound per TARGET (not per assay).
    One column per kinase — much more data per column than per-assay approach.
    Compound is active against a target if median pchembl >= threshold across all assays.
    """
    print("\nBuilding target-level label matrix...")

    df = activity_df[activity_df["standard_inchi_key"].isin(jump_inchikeys)].copy()
    print(f"  Records overlapping with JUMP-CP: {len(df):,}")
    print(f"  Unique compounds                : {df['standard_inchi_key'].nunique():,}")

    if df.empty:
        raise ValueError("No overlapping compounds between ChEMBL and JUMP-CP.")

    # For each (compound, target) pair take median pchembl_value across all assays
    agg = df.groupby(["standard_inchi_key", "target_name"])["pchembl_value"].median().reset_index()
    agg.rename(columns={"pchembl_value": "median_pchembl"}, inplace=True)

    # Re-apply threshold on the per-target median
    agg["target_label"] = 0
    agg.loc[agg["median_pchembl"] >= ACTIVE_THRESHOLD,   "target_label"] =  1
    agg.loc[agg["median_pchembl"] <= INACTIVE_THRESHOLD, "target_label"] = -1
    # Compounds between thresholds stay 0 (unknown) — drop them
    agg = agg[agg["target_label"] != 0]

    # Pivot: rows = compounds, columns = target names
    label_matrix = agg.pivot_table(
        values="target_label",
        index="standard_inchi_key",
        columns="target_name",
        aggfunc="first",
    ).fillna(0)

    print(f"\n  Label matrix shape: {label_matrix.shape}")
    print(f"  Per-target class counts:")
    for col in label_matrix.columns:
        n_pos = (label_matrix[col] ==  1).sum()
        n_neg = (label_matrix[col] == -1).sum()
        print(f"    {col:8s}  active={n_pos:4d}  inactive={n_neg:4d}")

    # Drop targets with too few examples
    valid_targets = [
        col for col in label_matrix.columns
        if (label_matrix[col] ==  1).sum() >= MIN_ACTIVES
        and (label_matrix[col] == -1).sum() >= MIN_INACTIVES
    ]
    label_matrix = label_matrix[valid_targets]
    print(f"\n  Targets passing quality filter ({MIN_ACTIVES}+ per class): {valid_targets}")

    if label_matrix.empty:
        raise ValueError(
            f"No targets with >= {MIN_ACTIVES} actives AND >= {MIN_INACTIVES} inactives "
            f"in JUMP-CP overlap. Try lowering thresholds."
        )

    return label_matrix


def merge_with_jump(label_matrix, compond, wells, plate):
    print("\nMerging with JUMP-CP metadata...")
    lm = label_matrix.reset_index()
    lm["Metadata_JCP2022"] = lm["standard_inchi_key"].map(
        compond.set_index("Metadata_InChIKey")["Metadata_JCP2022"]
    )
    lm = lm.dropna(subset=["Metadata_JCP2022"])
    print(f"  Compounds mapped to JCP IDs: {len(lm):,}")

    source_4_plates = plate[
        (plate["Metadata_Source"]    == "source_4") &
        (plate["Metadata_PlateType"] == "COMPOUND")
    ]["Metadata_Plate"].values

    wells_4 = wells[
        (wells["Metadata_Source"] == "source_4") &
        (wells["Metadata_Plate"].isin(source_4_plates)) &
        (wells["Metadata_JCP2022"].isin(lm["Metadata_JCP2022"]))
    ].copy()
    print(f"  Matched wells in source_4: {len(wells_4):,}")

    if wells_4.empty:
        print("  WARNING: No wells in source_4. Falling back to all sources...")
        wells_4 = wells[wells["Metadata_JCP2022"].isin(lm["Metadata_JCP2022"])].copy()
        print(f"  Matched wells (all sources): {len(wells_4):,}")

    wells_4["sample"] = 0
    df_sites        = pd.DataFrame({"Metadata_Site": range(1, 10), "sample": 0})
    well_with_sites = wells_4[
        ["Metadata_Plate", "Metadata_Well", "Metadata_JCP2022", "sample"]
    ].merge(df_sites)

    well_with_sites["Metadata_Path"] = (
        well_with_sites["Metadata_Plate"] + "/" +
        well_with_sites["Metadata_Well"]  + "_" +
        well_with_sites["Metadata_Site"].astype(str) + ".png"
    )

    target_cols = [c for c in lm.columns if c not in ["standard_inchi_key", "Metadata_JCP2022"]]
    final = well_with_sites.merge(
        lm[["Metadata_JCP2022"] + target_cols], on="Metadata_JCP2022", how="left"
    )
    print(f"  Final rows       : {len(final):,}")
    print(f"  Unique compounds : {final['Metadata_JCP2022'].nunique():,}")
    print(f"  Target columns   : {target_cols}")
    return final


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading JUMP-CP metadata...")
    compond = pd.read_csv(METADATA_PATH + "compound.csv.gz")
    wells   = pd.read_csv(METADATA_PATH + "well.csv.gz")
    plate   = pd.read_csv(METADATA_PATH + "plate.csv.gz")
    print(f"  Compounds: {compond.shape[0]:,}  Wells: {wells.shape[0]:,}  Plates: {plate.shape[0]:,}")
    jump_inchikeys = set(compond["Metadata_InChIKey"].dropna().unique())

    print(f"\nFetching activities for {len(TARGET_IDS)} kinase targets...")
    all_activities = []
    for target_name, target_id in TARGET_IDS.items():
        df = get_activities_for_target(target_id, target_name)
        if not df.empty:
            all_activities.append(df)

    if not all_activities:
        raise RuntimeError("No activities fetched.")

    activity_df = pd.concat(all_activities, ignore_index=True)
    print(f"\nTotal labelled records : {len(activity_df):,}")
    print(f"Unique compounds       : {activity_df['standard_inchi_key'].nunique():,}")

    label_matrix = build_target_label_matrix(activity_df, jump_inchikeys)
    final        = merge_with_jump(label_matrix, compond, wells, plate)

    out_path = OUTPUT_PATH + "data.csv"
    final.to_csv(out_path)
    print(f"\nSaved  -> {out_path}")
    print(f"Shape  : {final.shape}")


if __name__ == "__main__":
    main()