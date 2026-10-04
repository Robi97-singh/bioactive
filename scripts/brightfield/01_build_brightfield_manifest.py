"""
Step 1 of the brightfield ablation pipeline.

Builds a per-row manifest that maps every (Plate, Well, Site) triple in the
102 brightfield-available plates (Batch1 / Batch2 / Batch5) to:
  - its existing label row in training_paper.csv (split_number, 29 assay
    columns, compound identifiers, etc. all carried over unchanged)
  - the exact S3 key of its brightfield TIFF (from load_data_with_illum.csv's
    PathName_OrigBrightfield / FileName_OrigBrightfield columns)

Output: /shared/hdd/data/bioactive/brightfield_training_paper.csv
        (same schema as training_paper.csv, plus:
         Metadata_Batch, S3_Brightfield_Key)

Run on the login node (needs network + awscli, no GPU):
    python3 scripts/brightfield/01_build_brightfield_manifest.py
"""
import io
import subprocess
import sys

import pandas as pd

BUCKET = "cellpainting-gallery"
PREFIX = "cpg0016-jump/source_11"
BATCHES_WITH_BRIGHTFIELD = ["Batch1", "Batch2", "Batch5"]

TRAINING_CSV = "/shared/hdd/data/bioactive/training_paper.csv"
OUT_CSV = "/shared/hdd/data/bioactive/brightfield_training_paper.csv"


def s3_cat(key: str) -> bytes:
    uri = f"s3://{BUCKET}/{key}"
    proc = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", uri, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def s3_ls(prefix: str) -> list[str]:
    uri = f"s3://{BUCKET}/{prefix}"
    proc = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = proc.stdout.decode().splitlines()
    return [ln.split()[-1].rstrip("/") for ln in lines if ln.strip()]


def main() -> None:
    df = pd.read_csv(TRAINING_CSV)
    print(f"[manifest] training_paper.csv: {len(df)} rows, "
          f"{df['Metadata_Plate'].nunique()} plates", file=sys.stderr)

    plate_to_batch: dict[str, str] = {}
    for batch in BATCHES_WITH_BRIGHTFIELD:
        plates = s3_ls(f"{PREFIX}/workspace/load_data_csv/{batch}/")
        for p in plates:
            plate_to_batch[p] = batch
    print(f"[manifest] {len(plate_to_batch)} plates have brightfield "
          f"across {BATCHES_WITH_BRIGHTFIELD}", file=sys.stderr)

    bf_df = df[df["Metadata_Plate"].isin(plate_to_batch)].copy()
    bf_df["Metadata_Batch"] = bf_df["Metadata_Plate"].map(plate_to_batch)
    n_plates = bf_df["Metadata_Plate"].nunique()
    print(f"[manifest] {len(bf_df)} rows / {n_plates} plates restricted to "
          f"brightfield-available batches", file=sys.stderr)
    if n_plates != 102:
        print(f"[manifest] WARNING: expected 102 plates, got {n_plates} "
              f"-- double check before continuing", file=sys.stderr)

    key_lookup: dict[tuple[str, str, int], str] = {}
    for batch in BATCHES_WITH_BRIGHTFIELD:
        plates_here = sorted(bf_df.loc[bf_df["Metadata_Batch"] == batch,
                                        "Metadata_Plate"].unique())
        for plate in plates_here:
            csv_key = (f"{PREFIX}/workspace/load_data_csv/{batch}/{plate}/"
                       f"load_data_with_illum.csv")
            raw = s3_cat(csv_key)
            ld = pd.read_csv(io.BytesIO(raw))
            if "URL_OrigBrightfield" not in ld.columns and (
                "PathName_OrigBrightfield" not in ld.columns
                or "FileName_OrigBrightfield" not in ld.columns
            ):
                print(f"[manifest] WARNING: {plate} ({batch}) has no "
                      f"brightfield columns, skipping", file=sys.stderr)
                continue
            for _, row in ld.iterrows():
                well = row["Metadata_Well"]
                site = int(row["Metadata_Site"])
                if "URL_OrigBrightfield" in ld.columns:
                    url = row["URL_OrigBrightfield"]
                    key = url.split(f"{BUCKET}/", 1)[-1] if "s3://" in str(url) else url
                else:
                    key = (row["PathName_OrigBrightfield"].rstrip("/") + "/"
                           + row["FileName_OrigBrightfield"])
                    key = key.split(f"{BUCKET}/", 1)[-1] if f"{BUCKET}/" in key else key
                key_lookup[(plate, well, site)] = key
            print(f"[manifest]   {plate} ({batch}): {len(ld)} sites mapped",
                  file=sys.stderr)

    bf_df["S3_Brightfield_Key"] = bf_df.apply(
        lambda r: key_lookup.get(
            (r["Metadata_Plate"], r["Metadata_Well"], int(r["Metadata_Site"]))
        ),
        axis=1,
    )
    n_missing = bf_df["S3_Brightfield_Key"].isna().sum()
    print(f"[manifest] {n_missing} / {len(bf_df)} rows could not be matched "
          f"to a brightfield key", file=sys.stderr)
    if n_missing:
        bf_df = bf_df[bf_df["S3_Brightfield_Key"].notna()].copy()
        print(f"[manifest] dropped unmatched rows, {len(bf_df)} remain",
              file=sys.stderr)

    bf_df.to_csv(OUT_CSV, index=False)
    print(f"[manifest] wrote {OUT_CSV} ({len(bf_df)} rows, "
          f"{bf_df['Metadata_Plate'].nunique()} plates, "
          f"{bf_df['molregno'].nunique() if 'molregno' in bf_df.columns else '?'} compounds)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
