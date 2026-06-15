import sqlite3
import pandas as pd
import numpy as np

# --- paths adapted to this server (logic identical to paper repo) ---
METADATA_PATH = "/mnt/ssd8/bioactive/src/data/"      # paper appends "metadata/..." below
PATH_ChEMBL   = "/mnt/ssd8/bioactive/chembl/chembl_37/chembl_37_sqlite/chembl_37.db"
OUTPUT_CSV    = "/mnt/ssd8/bioactive/src/data/data_paper.csv"

wells   = pd.read_csv(METADATA_PATH+"metadata/well.csv.gz")

con = sqlite3.connect(PATH_ChEMBL)
assay_df    = pd.read_sql_query("select * from assays", con)
compound_df = pd.read_sql_query("select * from compound_structures", con)

chembl_compounds = set(compound_df.standard_inchi_key.unique())

compond = pd.read_csv(METADATA_PATH+"metadata/compound.csv.gz")
jump_compounds = set(compond.Metadata_InChIKey.unique())

overlapping_compounds = chembl_compounds.intersection(jump_compounds)
df_overlap_compounds  = compound_df[compound_df.standard_inchi_key.isin(overlapping_compounds)]
molregno_overlapping  = df_overlap_compounds.molregno.values

mol_list = tuple(int(x) for x in molregno_overlapping)
activity_of_overlapping = pd.read_sql_query(
    "SELECT molregno, assay_id, activity_comment, standard_type FROM activities "
    "WHERE standard_type = 'Potency' AND molregno IN (%s)" % ",".join(map(str, mol_list)),
    con)

potency_subset = activity_of_overlapping[activity_of_overlapping.standard_type == "Potency"]
potency_subset = potency_subset[potency_subset.activity_comment.isin(['inactive','active','Active','Not Active'])]
compound_counts = potency_subset.assay_id.value_counts()
selected_subset = potency_subset[potency_subset.assay_id.isin(compound_counts[(compound_counts > 100)].index)].copy()

selected_subset.loc[:,"activity_based_on_comment"] = 0
selected_subset.loc[selected_subset.activity_comment.isin(["Active","active"]),    "activity_based_on_comment"] = 1
selected_subset.loc[selected_subset.activity_comment.isin(["Not Active","inactive"]),"activity_based_on_comment"] = -1

label_matrix = selected_subset.pivot_table(values="activity_based_on_comment", index="molregno", columns="assay_id", aggfunc=np.median)

label_matrix = label_matrix.fillna(0)

assay_ids = ((label_matrix == 1).sum() > 50)
subset_of_assays = assay_ids[assay_ids].index
subset_with_neg = ((label_matrix[subset_of_assays] == -1.0).sum() > 50)
subset_keep = subset_with_neg[subset_with_neg].index
label_matrix_subset = label_matrix[subset_keep]

print("Assays after global 50+/50- filter:", len(subset_keep))

# restrict to source_11
sel = compound_df[compound_df.molregno.isin(label_matrix_subset.index)]
subset_jcp = compond[compond.Metadata_InChIKey.isin(sel.standard_inchi_key)].Metadata_JCP2022.values
well_subset = wells[wells.Metadata_JCP2022.isin(subset_jcp)]
src11_jcp = well_subset[well_subset.Metadata_Source == "source_11"].Metadata_JCP2022.values
src11_molregno = compound_df[compound_df.standard_inchi_key.isin(
    compond[compond.Metadata_JCP2022.isin(src11_jcp)].Metadata_InChIKey.values)].molregno.values

pos_ids = ((label_matrix_subset.loc[src11_molregno] == 1.0).sum() > 50)
neg_ids = ((label_matrix_subset[pos_ids[pos_ids].index].loc[src11_molregno] == -1.0).sum() > 50)
label_matrix_filtered = label_matrix_subset[neg_ids[neg_ids].index].loc[src11_molregno]

print("Assays after source_11 50+/50- filter:", label_matrix_filtered.shape[1])

label_matrix_renamed = label_matrix_filtered.reset_index().merge(
    df_overlap_compounds[["molregno","standard_inchi_key"]], how="left", on="molregno")

wells_11 = wells[wells.Metadata_Source == "source_11"].copy()
wells_11_meta = wells_11.merge(compond, how="left", on="Metadata_JCP2022")
wells_11_meta_ov = wells_11_meta[wells_11_meta.Metadata_InChIKey.isin(label_matrix_renamed.standard_inchi_key.unique())]
combined = wells_11_meta_ov.merge(label_matrix_renamed, how="left",
                                  right_on="standard_inchi_key", left_on="Metadata_InChIKey")

wells_11.loc[:,"sample"] = 0
df_sites = pd.DataFrame([1,2,3,4,5,6,7,8,9], columns=["Metadata_Site"]); df_sites.loc[:,"sample"] = 0
well_11_sites = wells_11[["Metadata_Plate","Metadata_Well","sample"]].merge(df_sites)
well_11_sites["Metadata_Path"] = (well_11_sites["Metadata_Plate"] + "/" +
                                  well_11_sites["Metadata_Well"] + "_" +
                                  well_11_sites["Metadata_Site"].astype(str) + ".png")

merged = combined.merge(well_11_sites, how="left", on=["Metadata_Plate","Metadata_Well"])
merged.to_csv(OUTPUT_CSV)
print("Wrote", OUTPUT_CSV, "shape", merged.shape)
assay_cols = list(label_matrix_filtered.columns)
print("Assay columns:", len(assay_cols))
print("Label values present:", sorted(pd.unique(merged[assay_cols].values.ravel())[:10]))
