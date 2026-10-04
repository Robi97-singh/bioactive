"""
Idempotent patch applier for classification.py: adds two new MODEL_MAP
entries ("bf_dinov3_base", "bf_celldino") used ONLY for the brightfield
frozen-probe runs. These are additive -- existing keys ("celldino", "dino",
etc.) and their results are untouched.

Why new keys instead of reusing "celldino": extract_embeddings.py/
train_head.py build their embeddings-file stem directly from the literal
--model string (not a dict lookup), and update_params_from_args() uses
--model to also set training_params.model_name (MODEL_MAP[args.model][1])
whenever --model_name isn't separately given. Reusing "celldino" for the
brightfield run would silently set model_name back to "bioact_celldino",
colliding with (and risking overwriting) the original fluorescence
Cell-DINO run's results. Distinct keys make this impossible.

    python3 apply_modelmap_patch.py
"""
import ast
import pathlib

TARGET = pathlib.Path("classification.py")

OLD = '''    "celldino":   ("celldino",        "bioact_celldino"),'''

NEW = '''    "celldino":   ("celldino",        "bioact_celldino"),
    "bf_dinov3_base": ("dinov3_base",  "bioact_brightfield_dinov3_base"),
    "bf_celldino":    ("celldino",     "bioact_brightfield_celldino"),'''


def main() -> None:
    text = TARGET.read_text()

    if "bf_dinov3_base" in text:
        print("[patch] brightfield MODEL_MAP entries already present -- skipping")
    else:
        count = text.count(OLD)
        assert count == 1, f"expected exactly 1 occurrence of OLD line, found {count}"
        text = text.replace(OLD, NEW, 1)
        TARGET.write_text(text)
        print("[patch] classification.py MODEL_MAP updated")

    ast.parse(TARGET.read_text())
    print("SYNTAX_OK")


if __name__ == "__main__":
    main()
