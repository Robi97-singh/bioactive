import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH    = os.path.expanduser("~/PycharmProjects/bioactive/")
DF_PATH      = BASE_PATH + "src/data/split_data.csv"
IMAGE_PATH   = BASE_PATH + "src/data/images/"
OUTPUT_CSV   = BASE_PATH + "src/data/training.csv"
DOWNLOAD_CSV = BASE_PATH + "src/data/to_download.csv"

# ──────────────────────────────────────────────────────────────────────────────

print("Loading split_data.csv...")
df = pd.read_csv(DF_PATH, index_col=0)
print(f"  Shape: {df.shape}")
print(f"  Unique compounds : {df['Metadata_JCP2022'].nunique():,}")
print(f"  Unique image paths: {df['Metadata_Path'].nunique():,}")

# ── Check which images exist on disk ──────────────────────────────────────────
print(f"\nChecking image files under: {IMAGE_PATH}")
os.makedirs(IMAGE_PATH, exist_ok=True)

missing_indices  = []
present_indices  = []
missing_paths    = []

for i, row in df.iterrows():
    full_path = os.path.join(IMAGE_PATH, str(row["Metadata_Path"]))
    if os.path.isfile(full_path):
        present_indices.append(i)
    else:
        missing_indices.append(i)
        missing_paths.append(row["Metadata_Path"])

print(f"  Present : {len(present_indices):,} rows")
print(f"  Missing : {len(missing_indices):,} rows")

# ── Summary of what needs to be downloaded ────────────────────────────────────
missing_df        = df.loc[missing_indices].copy()
unique_missing    = missing_df.drop_duplicates(subset="Metadata_Path")
unique_plates     = missing_df["Metadata_Plate"].nunique()
unique_compounds  = missing_df["Metadata_JCP2022"].nunique()

print(f"\n── Download summary ──────────────────────────────────────────")
print(f"  Unique image files to download : {len(unique_missing):,}")
print(f"  Unique plates involved         : {unique_plates:,}")
print(f"  Unique compounds involved      : {unique_compounds:,}")
print(f"  Est. size @ ~60MB/image        : ~{len(unique_missing) * 60 / 1024:.1f} GB")
print(f"  Est. size @ ~30MB/image        : ~{len(unique_missing) * 30 / 1024:.1f} GB")

# Save list of images to download for use in download_and_process.py
unique_missing[["Metadata_Plate", "Metadata_Well", "Metadata_Site",
                "Metadata_Path", "Metadata_JCP2022"]].to_csv(DOWNLOAD_CSV, index=False)
print(f"\n  Download manifest saved -> {DOWNLOAD_CSV}")

# ── Build training.csv ────────────────────────────────────────────────────────
# Since images aren't downloaded yet, training.csv = full split_data.csv
# (remove_missing_files will be re-run after download to prune truly missing)
if len(present_indices) > 0:
    df_present = df.loc[present_indices]
    print(f"\nImages already on disk — saving {len(df_present):,} rows to training.csv")
else:
    print("\nNo images on disk yet — saving full split_data.csv as training.csv")
    print("Re-run this script after download_and_process.py to prune missing files.")
    df_present = df.copy()

df_present.to_csv(OUTPUT_CSV)
print(f"Saved -> {OUTPUT_CSV}")
print(f"Shape : {df_present.shape}")

# ── Per-fold summary ──────────────────────────────────────────────────────────
print("\nPer-fold split distribution in training.csv:")
print(df_present["split_number"].value_counts().sort_index().to_string())
print("\nPer-fold CDK2 label distribution:")
print(df_present.groupby("split_number")["CDK2"].value_counts().unstack(fill_value=0))
print("\nPer-fold EGFR label distribution:")
print(df_present.groupby("split_number")["EGFR"].value_counts().unstack(fill_value=0))