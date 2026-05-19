"""
download_and_process.py
Full JUMP-CP image pipeline:
  1. Read data.csv to get all plates needed
  2. Download load_data CSV for each plate from S3
  3. For each site: download 5 channel TIFFs -> process -> save PNG -> delete TIFFs
  4. Parallel execution per plate using ThreadPoolExecutor
"""

import os
import io
import gzip
import boto3
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

BASE_PATH   = "/mnt/ssd8/bioactive/"
DATA_CSV    = BASE_PATH + "src/data/data.csv"
PLATE_CSV   = BASE_PATH + "src/data/metadata/plate.csv.gz"
OUTPUT_PATH = BASE_PATH + "src/data/images/"

S3_BUCKET   = "cellpainting-gallery"
S3_PREFIX   = "cpg0016-jump"

CHANNEL_SUFFIXES  = ["DNA", "ER", "RNA", "AGP", "Mito"]
MAX_PLATE_WORKERS = 4
MAX_SITE_WORKERS  = 16

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

def get_s3_client():
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )

def percentile_normalize(arr, low=1, high=99):
    p_low  = np.percentile(arr, low)
    p_high = np.percentile(arr, high)
    arr    = np.clip(arr, p_low, p_high)
    if p_high > p_low:
        arr = (arr - p_low) / (p_high - p_low) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)

def download_tiff_from_s3(s3_client, s3_key):
    buf = io.BytesIO()
    s3_client.download_fileobj(S3_BUCKET, s3_key, buf)
    buf.seek(0)
    with tifffile.TiffFile(buf) as tif:
        arr = tif.asarray().astype(np.float32)
    return arr

def process_site(s3_client, plate_id, well, site, channel_keys, out_path):
    if os.path.exists(out_path):
        return True, f"skip {os.path.basename(out_path)}"
    channels = []
    for key in channel_keys:
        try:
            arr = download_tiff_from_s3(s3_client, key)
            arr = percentile_normalize(arr)
            channels.append(arr)
        except Exception as e:
            return False, f"TIFF download failed {key}: {e}"
    combined = np.hstack(channels)
    img      = Image.fromarray(combined)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True, os.path.basename(out_path)

def get_load_data_s3_key(source, batch, plate_id):
    return f"{S3_PREFIX}/{source}/workspace/load_data_csv/{batch}/{plate_id}/load_data_with_illum.csv"

def fetch_load_data(s3_client, source, batch, plate_id):
    key = get_load_data_s3_key(source, batch, plate_id)
    try:
        buf = io.BytesIO()
        s3_client.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        df = pd.read_csv(buf)
        return df
    except Exception as e:
        tprint(f"  [{plate_id}] Failed to fetch load_data: {e}")
        return None

def get_channel_keys(load_data_row, source, batch, plate_id):
    keys = []
    for ch in CHANNEL_SUFFIXES:
        col = f"URL_Orig{ch}"
        if col not in load_data_row.index:
            return None
        url = load_data_row[col]
        key = url.replace(f"s3://{S3_BUCKET}/", "")
        keys.append(key)
    return keys

def process_plate(plate_id, source, batch, wells_needed):
    s3_client = get_s3_client()
    plate_dir = os.path.join(OUTPUT_PATH, str(plate_id))
    os.makedirs(plate_dir, exist_ok=True)

    tprint(f"\n[{plate_id}] Fetching load_data (source={source}, batch={batch})...")
    load_data = fetch_load_data(s3_client, source, batch, plate_id)
    if load_data is None:
        tprint(f"[{plate_id}] Could not fetch load_data — skipping")
        return plate_id, 0, len(wells_needed), 0

    load_data = load_data[load_data["Metadata_Well"].isin(wells_needed)]
    tprint(f"[{plate_id}] {len(load_data)} sites across {load_data['Metadata_Well'].nunique()} wells")

    if load_data.empty:
        return plate_id, 0, 0, 0

    n_success = 0
    n_failed  = 0
    n_skipped = 0

    def process_row(row):
        well     = row["Metadata_Well"]
        site     = str(row["Metadata_Site"])
        out_path = os.path.join(plate_dir, f"{well}_{site}.png")
        channel_keys = get_channel_keys(row, source, batch, plate_id)
        if channel_keys is None:
            return False, f"Missing channel URL columns for {well}_{site}"
        return process_site(s3_client, plate_id, well, site, channel_keys, out_path)

    rows = [row for _, row in load_data.iterrows()]
    with ThreadPoolExecutor(max_workers=MAX_SITE_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        for i, future in enumerate(as_completed(futures)):
            try:
                ok, msg = future.result()
                if ok:
                    if msg.startswith("skip"):
                        n_skipped += 1
                    else:
                        n_success += 1
                else:
                    n_failed += 1
                    tprint(f"  [{plate_id}] FAILED: {msg}")
            except Exception as e:
                n_failed += 1
                tprint(f"  [{plate_id}] Exception: {e}")
            if (i + 1) % 50 == 0:
                tprint(f"  [{plate_id}] {i+1}/{len(rows)} done (✅{n_success} ⏭{n_skipped} ❌{n_failed})")

    tprint(f"[{plate_id}] Done — success={n_success} skipped={n_skipped} failed={n_failed}")
    return plate_id, n_success, n_failed, n_skipped

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading data.csv...")
    data = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  Rows            : {len(data):,}")
    print(f"  Unique plates   : {data['Metadata_Plate'].nunique():,}")
    print(f"  Unique compounds: {data['Metadata_JCP2022'].nunique():,}")

    print("\nLoading plate metadata...")
    plate_meta = pd.read_csv(PLATE_CSV)
    plate_meta["Metadata_Plate"] = plate_meta["Metadata_Plate"].astype(str)
    data["Metadata_Plate"]       = data["Metadata_Plate"].astype(str)

    plates_needed = data["Metadata_Plate"].unique()
    plate_info = plate_meta[
        plate_meta["Metadata_Plate"].isin(plates_needed)
    ][["Metadata_Plate", "Metadata_Source", "Metadata_Batch"]].drop_duplicates()
    print(f"  Plates found in metadata: {len(plate_info):,}")

    jobs = []
    for _, row in plate_info.iterrows():
        plate_id = row["Metadata_Plate"]
        source   = row["Metadata_Source"]
        batch    = row["Metadata_Batch"]
        wells    = data[data["Metadata_Plate"] == plate_id]["Metadata_Well"].unique()
        jobs.append((plate_id, source, batch, set(wells)))

    already_done = []
    remaining    = []
    for job in jobs:
        plate_dir = os.path.join(OUTPUT_PATH, str(job[0]))
        if os.path.exists(plate_dir) and len(list(Path(plate_dir).glob("*.png"))) > 0:
            already_done.append(job[0])
        else:
            remaining.append(job)

    if already_done:
        print(f"  Skipping {len(already_done)} plates already processed")
    print(f"  Processing {len(remaining)} plates\n")
    print(f"Output directory: {OUTPUT_PATH}\n")

    total_success = 0
    total_failed  = 0
    total_skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_PLATE_WORKERS) as executor:
        futures = {executor.submit(process_plate, *job): job[0] for job in remaining}
        for future in as_completed(futures):
            plate_id = futures[future]
            try:
                _, n_s, n_f, n_sk = future.result()
                total_success += n_s
                total_failed  += n_f
                total_skipped += n_sk
            except Exception as e:
                print(f"[{plate_id}] Unhandled exception: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  Total sites processed : {total_success:,}")
    print(f"  Total sites skipped   : {total_skipped:,}")
    print(f"  Total sites failed    : {total_failed:,}")
    all_pngs = list(Path(OUTPUT_PATH).rglob("*.png"))
    print(f"  Total PNGs on disk    : {len(all_pngs):,}")
    statvfs = os.statvfs(OUTPUT_PATH)
    free_gb = statvfs.f_frsize * statvfs.f_bavail / 1e9
    print(f"  Disk free             : {free_gb:.1f} GB")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
