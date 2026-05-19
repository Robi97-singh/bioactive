import pandas as pd
import numpy as np
import os
from PIL import Image
import tifffile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH    = os.path.expanduser("~/PycharmProjects/bioactive/")
TIFF_PATH    = BASE_PATH + "src/data/images/1053597936/"
OUTPUT_PATH  = BASE_PATH + "src/data/images/"
LOAD_DATA    = BASE_PATH + "src/data/load_data.csv"
SPLIT_DATA   = BASE_PATH + "src/data/split_data.csv"
PLATE_ID     = "1053597936"

# 5 Cell Painting channels in order: DNA, ER, RNA, AGP, Mito
CHANNEL_COLS = [
    "URL_OrigDNA",
    "URL_OrigER",
    "URL_OrigRNA",
    "URL_OrigAGP",
    "URL_OrigMito",
]

# ──────────────────────────────────────────────────────────────────────────────

def load_tiff_channel(filepath):
    """Load a 16-bit TIFF and return as numpy array."""
    with tifffile.TiffFile(filepath) as tif:
        arr = tif.asarray().astype(np.float32)
    return arr


def percentile_normalize(arr, low=1, high=99):
    """Clip to 1st/99th percentile then scale to 0-255 (paper setting)."""
    p_low  = np.percentile(arr, low)
    p_high = np.percentile(arr, high)
    arr    = np.clip(arr, p_low, p_high)
    if p_high > p_low:
        arr = (arr - p_low) / (p_high - p_low) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def process_site(row, tiff_dir, out_dir):
    """
    Stack 5 channels for one site into a single horizontal PNG.
    Filename: {Well}_{Site}.png  e.g. B02_1.png
    """
    well = row["Metadata_Well"]
    site = str(row["Metadata_Site"])
    out_filename = f"{well}_{site}.png"
    out_path     = os.path.join(out_dir, PLATE_ID, out_filename)

    if os.path.exists(out_path):
        return True, out_filename  # already processed

    channels = []
    for col in CHANNEL_COLS:
        s3_url   = row[col]
        filename = os.path.basename(s3_url)
        filepath = os.path.join(tiff_dir, filename)

        if not os.path.exists(filepath):
            return False, f"Missing: {filename}"

        arr = load_tiff_channel(filepath)
        arr = percentile_normalize(arr)
        channels.append(arr)

    # Stack horizontally: shape (H, W*5)
    combined = np.hstack(channels)
    img      = Image.fromarray(combined)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True, out_filename


def main():
    print("Loading metadata...")
    ld    = pd.read_csv(LOAD_DATA)
    split = pd.read_csv(SPLIT_DATA, index_col=0)

    # Filter load_data to our labelled wells only
    our_wells = split[split["Metadata_Plate"] == PLATE_ID]["Metadata_Well"].unique()
    ld_subset = ld[ld["Metadata_Well"].isin(our_wells)].reset_index(drop=True)
    print(f"  Wells to process : {len(our_wells)}")
    print(f"  Sites to process : {len(ld_subset)}")

    out_dir = os.path.join(BASE_PATH, "src/data/images/")
    os.makedirs(os.path.join(out_dir, PLATE_ID), exist_ok=True)

    success = 0
    failed  = []

    print(f"\nProcessing {len(ld_subset)} sites...")
    for i, row in ld_subset.iterrows():
        ok, msg = process_site(row, TIFF_PATH, out_dir)
        if ok:
            success += 1
            if success % 20 == 0:
                print(f"  [{success}/{len(ld_subset)}] {msg}")
        else:
            failed.append(msg)
            print(f"  FAILED: {msg}")

    print(f"\nDone.")
    print(f"  Successful : {success}")
    print(f"  Failed     : {len(failed)}")

    # Verify output PNGs exist
    out_pngs = list(Path(os.path.join(out_dir, PLATE_ID)).glob("*.png"))
    print(f"  PNGs created: {len(out_pngs)}")

    if out_pngs:
        sample = out_pngs[0]
        img    = Image.open(sample)
        print(f"  Sample PNG  : {sample.name}  size={img.size}")

    # Check disk
    statvfs   = os.statvfs(BASE_PATH)
    free_gb   = statvfs.f_frsize * statvfs.f_bavail / 1e9
    print(f"  Disk free   : {free_gb:.1f} GB")

    # Update split_data paths to match actual PNG locations
    print("\nVerifying path alignment with split_data.csv...")
    split_plate = split[split["Metadata_Plate"] == PLATE_ID].copy()
    matched = 0
    for png in out_pngs:
        expected_path = f"{PLATE_ID}/{png.name}"
        if (split_plate["Metadata_Path"] == expected_path).any():
            matched += 1

    print(f"  Paths matching split_data: {matched}/{len(out_pngs)}")
    if matched == 0:
        print("  WARNING: No path matches — check Metadata_Path format in split_data.csv")
        print("  Sample split path :", split_plate["Metadata_Path"].iloc[0])
        print("  Sample actual path:", f"{PLATE_ID}/{out_pngs[0].name}")


if __name__ == "__main__":
    main()