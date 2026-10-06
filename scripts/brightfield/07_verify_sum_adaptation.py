"""
Checks that the 1-channel first-layer adaptation is numerically equivalent to
feeding the grayscale image replicated across the pretrained channels:

    first_layer_1ch(x)  ==  first_layer_native(x.repeat(1, C, 1, 1))

for DINOv3-Base (C=3), ResNet-50 (C=3) and Cell-DINO (C=5). Run as a GPU/venv
job: PYTHONPATH=. python3 scripts/brightfield/07_verify_sum_adaptation.py
"""
import torch

from defaults.models import Classifier, DINOv3Classifier, CellDINOClassifier

CELLDINO_REPO = "/shared/ssd/home/b-r-singh1/dinov2_repo"
CELLDINO_CKPT = "/shared/ssd/home/b-r-singh1/celldino_weights/cell_dino_vits8_pretrain_cp.pth"


def check(name, out_1ch, out_ref):
    max_err = (out_1ch - out_ref).abs().max().item()
    scale = out_ref.abs().max().item()
    print(f"{name}: max|diff|={max_err:.2e}  (reference max|.|={scale:.2e})")
    assert max_err <= 1e-4 * max(1.0, scale), f"{name}: 1-channel adaptation NOT equivalent"


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)

    base = {'freeze_backbone': True, 'pretrained': True, 'n_classes': 29,
            'use_mixed_precision': False}

    with torch.no_grad():
        m1 = DINOv3Classifier({**base, 'backbone_type': 'dinov3_base', 'img_channels': 1})
        m3 = DINOv3Classifier({**base, 'backbone_type': 'dinov3_base', 'img_channels': 3})
        check("DINOv3 patch-embed",
              m1.backbone.embeddings.patch_embeddings(x),
              m3.backbone.embeddings.patch_embeddings(x.repeat(1, 3, 1, 1)))

        r1 = Classifier({**base, 'freeze_backbone': False, 'backbone_type': 'resnet50',
                         'img_channels': 1})
        r3 = Classifier({**base, 'freeze_backbone': False, 'backbone_type': 'resnet50',
                         'img_channels': 3})
        check("ResNet-50 conv1",
              r1.backbone.conv1(x), r3.backbone.conv1(x.repeat(1, 3, 1, 1)))

        cd = {**base, 'backbone_type': 'celldino',
              'celldino_repo': CELLDINO_REPO, 'celldino_ckpt': CELLDINO_CKPT}
        c1 = CellDINOClassifier({**cd, 'img_channels': 1})
        c5 = CellDINOClassifier({**cd, 'img_channels': 5})
        check("Cell-DINO patch-embed",
              c1.backbone.patch_embed.proj(x),
              c5.backbone.patch_embed.proj(x.repeat(1, 5, 1, 1)))

    print()
    print("ALL ADAPTATION EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
