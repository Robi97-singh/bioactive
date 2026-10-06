"""
Idempotent patch for defaults/datasets.py: writes the measured brightfield
mean/std (parsed from the stats script's log) into
BioActBrightfield.init_stats(), replacing the placeholder values.

    python3 scripts/brightfield/04_compute_brightfield_stats.py | tee brightfield_stats.log
    python3 apply_stats_patch.py brightfield_stats.log
"""
import ast
import pathlib
import re
import sys

TARGET = pathlib.Path("defaults/datasets.py")


def main() -> None:
    log = pathlib.Path(sys.argv[1]).read_text()
    m_mean = re.search(r"\[stats\]\s+mean\s*=\s*([0-9.eE+-]+)", log)
    m_std = re.search(r"\[stats\]\s+std\s*=\s*([0-9.eE+-]+)", log)
    assert m_mean and m_std, "could not find '[stats] mean/std' lines in the log"
    mean, std = float(m_mean.group(1)), float(m_std.group(1))
    assert 0.0 < mean < 1.0 and 0.0 < std < 1.0, f"implausible stats: {mean}, {std}"

    text = TARGET.read_text()
    start = text.index("class BioActBrightfield")
    end = text.find("\nclass ", start + 1)
    end = len(text) if end == -1 else end
    block = text[start:end]

    block, n1 = re.subn(r"self\.mean\s*=\s*\([^)]*\)", f"self.mean = ({mean:.4f},)", block)
    block, n2 = re.subn(r"self\.std\s*=\s*\([^)]*\)", f"self.std  = ({std:.4f},)", block)
    assert n1 == 1 and n2 == 1, f"expected one mean and one std assignment, got {n1}, {n2}"
    block = block.replace(
        "        # PLACEHOLDER -- replace with real values from\n"
        "        # 04_compute_brightfield_stats.py once images are downloaded.\n",
        "        # Measured by scripts/brightfield/04_compute_brightfield_stats.py\n"
        "        # (5000 random images, seed 0), on the [0, 1] scale after ToTensor.\n")

    new_text = text[:start] + block + text[end:]
    ast.parse(new_text)
    TARGET.write_text(new_text)
    print(f"[patch] BioActBrightfield stats set: mean=({mean:.4f},) std=({std:.4f},)")
    print("SYNTAX_OK")


if __name__ == "__main__":
    main()
