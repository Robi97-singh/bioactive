"""
Idempotent patch applier for defaults/wrappers.py: adds the missing
DINOv3Classifier branch to Trainer.init_model()'s backbone dispatch. Without
this, backbone_type="dinov3_small/base/large" silently falls through to the
generic (ResNet-style) Classifier instead of DINOv3Classifier.

    python3 apply_wrappers_patch.py
"""
import ast
import pathlib

TARGET = pathlib.Path("defaults/wrappers.py")

OLD = '''        elif backbone == "celldino":
            from defaults.models import CellDINOClassifier
            model = CellDINOClassifier(self.model_params)'''

NEW = '''        elif backbone in ("dinov3_small", "dinov3_base", "dinov3_large"):
            from defaults.models import DINOv3Classifier
            model = DINOv3Classifier(self.model_params)
        elif backbone == "celldino":
            from defaults.models import CellDINOClassifier
            model = CellDINOClassifier(self.model_params)'''


def main() -> None:
    text = TARGET.read_text()

    if "DINOv3Classifier" in text:
        print("[patch] DINOv3Classifier branch already present -- skipping")
    else:
        count = text.count(OLD)
        assert count == 1, f"expected exactly 1 occurrence of OLD block, found {count}"
        text = text.replace(OLD, NEW, 1)
        TARGET.write_text(text)
        print("[patch] defaults/wrappers.py updated")

    ast.parse(TARGET.read_text())
    print("SYNTAX_OK")


if __name__ == "__main__":
    main()
