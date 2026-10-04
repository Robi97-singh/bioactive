"""
Applies the brightfield patch to defaults/datasets.py programmatically.
Run from the bioactive repo root:

    python3 apply_datasets_patch.py
"""
import pathlib
import sys

TARGET = pathlib.Path("defaults/datasets.py")

OLD_1 = '''        df = pd.read_csv(dataset_csv_path, index_col=0)
        df = df.dropna(subset="Metadata_Path")'''

NEW_1 = '''        path_column = getattr(self, "path_column", "Metadata_Path")
        df = pd.read_csv(dataset_csv_path, index_col=0)
        df = df.dropna(subset=path_column)'''

OLD_2 = '''        img_paths     = data["Metadata_Path"].values.tolist()'''

NEW_2 = '''        img_paths     = data[path_column].values.tolist()'''

NEW_CLASS = '''

class BioActBrightfield(BioAct):
    """Single-channel (brightfield) variant of BioAct. Reuses BioAct's
    compound-level split/label logic untouched -- same split_number folds,
    same 29 assays, same compounds (brightfield comes from the same
    acquisition rows). Only channel count and image loading differ."""

    img_channels = 1
    path_column  = "Metadata_Path_Brightfield"

    def init_stats(self):
        # PLACEHOLDER -- replace with real values from
        # 04_compute_brightfield_stats.py once images are downloaded.
        self.mean = (0.5,)
        self.std  = (0.25,)

    def get_image_data(self, path: str):
        img = Image.open(path)
        img = np.array(img)
        if img.ndim == 2:
            img = img[:, :, None]          # (H, W) -> (H, W, 1)
        return img
'''


def main() -> None:
    text = TARGET.read_text()

    if "class BioActBrightfield" in text:
        print("[patch] BioActBrightfield already present, skipping class insert", file=sys.stderr)
    else:
        if text.count(OLD_1) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of OLD_1, found {text.count(OLD_1)}", file=sys.stderr)
            sys.exit(1)
        if text.count(OLD_2) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of OLD_2, found {text.count(OLD_2)}", file=sys.stderr)
            sys.exit(1)

        text = text.replace(OLD_1, NEW_1, 1)
        text = text.replace(OLD_2, NEW_2, 1)

        marker = "class Hofmarcher(BaseSet):"
        if marker not in text:
            print(f"[patch] ERROR: could not find insertion marker {marker!r}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(marker, NEW_CLASS.strip("\n") + "\n\n\n" + marker, 1)

        TARGET.write_text(text)
        print("[patch] defaults/datasets.py patched successfully", file=sys.stderr)

    import ast
    ast.parse(TARGET.read_text())
    print("[patch] syntax OK", file=sys.stderr)


if __name__ == "__main__":
    main()
