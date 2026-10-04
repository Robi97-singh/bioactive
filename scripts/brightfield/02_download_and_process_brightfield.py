"""
Step 2 of the brightfield ablation pipeline.

Downloads the raw OrigBrightfield TIFF for every row in
brightfield_training_paper.csv, applies percentile_normalize(low=1, high=99)
-> uint8, writes each as a single-channel PNG. Safe to re-run: already
downloaded files are skipped. Parallelized (I/O bound), ~88k images.

Output: /shared/hdd/data/bioactive/brightfield_images/{Plate}/{Well}_{Site}.png
Adds Metadata_Path_Brightfield column to the manifest -- IMPORTANT: when
--limit is used (a dry run), the manifest is NEVER rewritten, so a partial
run can't truncate the full manifest. Only a full (--limit unset) run
updates the manifest on disk.

    python3 scripts/brightfield/02_download_and_process_brightfield.py --workers 16
"""
import argparse
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BUCKET = "cellpainting-gallery"
MANIFEST_CSV = "/shared/hdd/data/bioactive/brightfield_training_paper.csv"
OUT_ROOT = Path("/shared/hdd/data/bioactive/brightfield_images")


def percentile_normalize(arr: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def s3_bytes(key: str) -> bytes:
    uri = f"s3://{BUCKET}/{key}"
    proc = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", uri, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def process_row(plate: str, well: str, site, s3_key: str) -> tuple[str, bool, str]:
    out_dir = OUT_ROOT / plate
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{plate}/{well}_{site}.png"
    out_path = OUT_ROOT / rel_path
    if out_path.exists():
        return rel_path, True, ""
    try:
        raw = s3_bytes(s3_key)
        arr = np.array(Image.open(io.BytesIO(raw)))
        norm = percentile_normalize(arr, low=1, high=99)
        Image.fromarray(norm, mode="L").save(out_path)
        return rel_path, True, ""
    except Exception as exc:  # noqa: BLE001
        return rel_path, False, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                     help="process only first N rows, for a quick dry run "
                          "-- never writes back to the manifest CSV")
    args = ap.parse_args()

    full_df = pd.read_csv(MANIFEST_CSV)
    df = full_df.head(args.limit).copy() if args.limit else full_df
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[download] {len(df)} images to process with {args.workers} workers"
          + (" (DRY RUN -- manifest will NOT be updated)" if args.limit else ""),
          file=sys.stderr)

    rel_paths = [None] * len(df)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_row, row.Metadata_Plate, row.Metadata_Well,
                        row.Metadata_Site, row.S3_Brightfield_Key): i
            for i, row in enumerate(df.itertuples(index=False))
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            rel_path, ok, err = fut.result()
            rel_paths[i] = rel_path if ok else None
            if not ok:
                failures.append((i, err))
            done += 1
            if done % 2000 == 0:
                print(f"[download] {done}/{len(df)} done, "
                      f"{len(failures)} failures so far", file=sys.stderr)

    df = df.copy()
    df["Metadata_Path_Brightfield"] = rel_paths
    n_failed = df["Metadata_Path_Brightfield"].isna().sum()
    print(f"[download] finished: {len(df) - n_failed} ok, {n_failed} failed",
          file=sys.stderr)
    if failures[:10]:
        print("[download] sample failures:", file=sys.stderr)
        for i, err in failures[:10]:
            print(f"    row {i}: {err}", file=sys.stderr)

    if args.limit:
        print("[download] DRY RUN complete -- manifest NOT written", file=sys.stderr)
        return

    if n_failed:
        df = df[df["Metadata_Path_Brightfield"].notna()].copy()
        print(f"[download] dropped failed rows, {len(df)} remain -- "
              f"re-run this script again to retry transient S3 failures "
              f"before dropping, if n_failed is large", file=sys.stderr)

    df.to_csv(MANIFEST_CSV, index=False)
    print(f"[download] updated manifest written to {MANIFEST_CSV} "
          f"({len(df)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
