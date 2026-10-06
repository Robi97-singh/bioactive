"""
Idempotent patch for defaults/models.py: collapse pretrained first-layer
weights to 1 channel with SUM instead of MEAN.

Summing over input channels is mathematically identical to feeding the
grayscale image replicated across the pretrained channels (the standard
baseline, and what timm's adapt_input_conv does). MEAN made first-layer
activations 3x weaker for RGB models (DINOv3, ResNet) and 5x weaker for
Cell-DINO, pushing the frozen backbones off the signal scale they were
trained on.

    python3 apply_sum_adaptation_patch.py
"""
import ast
import pathlib

TARGET = pathlib.Path("defaults/models.py")

# (old, new, expected_count)
EDITS = [
    ("pretrained_weight = pretrained_weight.mean(dim=1, keepdim=True)",
     "pretrained_weight = pretrained_weight.sum(dim=1, keepdim=True)  # == gray replicated over RGB",
     1),   # Classifier.modify_first_layer (ResNet branch)
    ("new_weight = old_weight.mean(dim=1, keepdim=True)",
     "new_weight = old_weight.sum(dim=1, keepdim=True)  # == gray replicated over pretrained channels",
     2),   # DINOv3Classifier + CellDINOClassifier
]


def main() -> None:
    text = TARGET.read_text()
    for old, new, expected in EDITS:
        n_old = text.count(old)
        if n_old == 0 and text.count(new) == expected:
            print(f"[patch] already applied: {new[:60]}...")
            continue
        assert n_old == expected, f"expected {expected} x {old!r}, found {n_old}"
        text = text.replace(old, new)
        print(f"[patch] replaced {expected} x {old[:60]}...")
    ast.parse(text)
    TARGET.write_text(text)
    print("SYNTAX_OK")


if __name__ == "__main__":
    main()
