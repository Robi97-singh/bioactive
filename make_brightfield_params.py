"""
Generates the three brightfield params JSON files (one per model config in the
ablation), each cloned from its matching fluorescence template with:
  - dataset_params.dataset       -> "BioActBrightfield"
  - dataset_params.data_location -> brightfield images root (unused in
    practice since the manifest stores absolute paths, but kept for clarity)
  - dataset_params.dataset_csv_path -> brightfield manifest CSV
  - model_params.backbone_type   -> set to the right value for each config
  - training_params.model_name   -> a distinct name per config

    python3 make_brightfield_params.py
"""
import copy
import json
import pathlib

PARAMS_DIR = pathlib.Path("params")

BRIGHTFIELD_DATA_LOCATION = "/shared/hdd/data/bioactive/brightfield_images"
BRIGHTFIELD_CSV = "/shared/hdd/data/bioactive/brightfield_training_paper.csv"


def load(name: str) -> dict:
    return json.loads((PARAMS_DIR / name).read_text())


def set_brightfield_dataset(params: dict) -> None:
    params["dataset_params"]["dataset"] = "BioActBrightfield"
    params["dataset_params"]["data_location"] = BRIGHTFIELD_DATA_LOCATION
    params["dataset_params"]["dataset_csv_path"] = BRIGHTFIELD_CSV


def write(name: str, params: dict) -> None:
    out_path = PARAMS_DIR / name
    out_path.write_text(json.dumps(params, indent=2) + "\n")
    # round-trip validate
    json.loads(out_path.read_text())
    print(f"[ok] wrote {out_path}")


def main() -> None:
    # 1) DINOv3-Base, frozen probe -- clone the dinov2_base cluster config
    #    (same frozen-probe recipe shape) and swap the backbone.
    dinov3 = load("params_dinov2_base_cluster.json")
    set_brightfield_dataset(dinov3)
    dinov3["model_params"] = {
        "backbone_type": "dinov3_base",
        "pretrained": True,
        "freeze_backbone": True,
    }
    dinov3["training_params"]["model_name"] = "bioact_brightfield_dinov3_base"
    write("params_brightfield_dinov3_base_cluster.json", dinov3)

    # 2) Cell-DINO, frozen probe -- clone the celldino cluster config as-is
    #    (model_params already correct: backbone_type, celldino_repo/ckpt).
    celldino = load("params_celldino_cluster.json")
    set_brightfield_dataset(celldino)
    celldino["training_params"]["model_name"] = "bioact_brightfield_celldino"
    write("params_brightfield_celldino_cluster.json", celldino)

    # 3) ResNet-50, fine-tuned CNN -- clone the base cluster config.
    resnet = load("params_cluster_base.json")
    set_brightfield_dataset(resnet)
    resnet["training_params"]["model_name"] = "bioact_brightfield_resnet50"
    write("params_brightfield_resnet_cluster.json", resnet)


if __name__ == "__main__":
    main()
