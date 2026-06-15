import pandas as pd
import os

DF_PATH   = "/mnt/ssd8/bioactive/src/data/split_data_paper.csv"
BASE_PATH = "/mnt/ssd8/bioactive/src/data/images/"
OUT_CSV   = "/mnt/ssd8/bioactive/src/data/training_paper.csv"

df = pd.read_csv(DF_PATH, index_col=0)
print("Total rows:", len(df))
print("Rows with NaN Metadata_Path:", df.Metadata_Path.isna().sum())
df = df.dropna(subset=["Metadata_Path"])

present, missing = [], 0
for i, row in df.iterrows():
    if os.path.isfile(BASE_PATH + str(row.Metadata_Path)):
        present.append(i)
    else:
        missing += 1

print("Present:", len(present), " Missing:", missing)
out = df.loc[present]
out.to_csv(OUT_CSV)
print("Wrote", OUT_CSV, "shape", out.shape)

# per-fold sanity
print("\nPer-fold rows kept:")
print(out.split_number.value_counts().sort_index())
