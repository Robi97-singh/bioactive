"""
Sanity check: builds each of the three brightfield models (DINOv3-Base,
Cell-DINO, ResNet-50) with img_channels=1 and runs one forward pass, to
confirm the channel-adaptation patches in defaults/models.py actually work
end-to-end before committing to real training jobs.

celldino_repo/celldino_ckpt are overridden here because this cluster account
keeps them under /shared/ssd/home/b-r-singh1/ rather than the hardcoded
/mnt/ssd8/bioactive/ default in CellDINOClassifier -- same override should
be passed in the real training job configs later.
"""
import torch

from defaults.models import Classifier, DINOv3Classifier, CellDINOClassifier

CELLDINO_REPO = "/shared/ssd/home/b-r-singh1/dinov2_repo"
CELLDINO_CKPT = "/shared/ssd/home/b-r-singh1/celldino_weights/cell_dino_vits8_pretrain_cp.pth"


def main() -> None:
    print("--- DINOv3Classifier, 1 channel, r448 ---")
    m = DINOv3Classifier({
        'backbone_type': 'dinov3_base', 'freeze_backbone': True,
        'pretrained': True, 'img_channels': 1, 'n_classes': 29,
        'use_mixed_precision': False,
    })
    out = m(torch.zeros(2, 1, 448, 448))
    assert out.shape == (2, 29), out.shape
    print("DINOv3 1ch OUTPUT SHAPE:", out.shape)

    print("--- CellDINOClassifier, 1 channel, r224 ---")
    m2 = CellDINOClassifier({
        'backbone_type': 'celldino', 'freeze_backbone': True,
        'pretrained': True, 'img_channels': 1, 'n_classes': 29,
        'use_mixed_precision': False,
        'celldino_repo': CELLDINO_REPO,
        'celldino_ckpt': CELLDINO_CKPT,
    })
    out2 = m2(torch.zeros(2, 1, 224, 224))
    assert out2.shape == (2, 29), out2.shape
    print("CellDINO 1ch OUTPUT SHAPE:", out2.shape)

    print("--- ResNet-50 (Classifier), 1 channel, r448 ---")
    m3 = Classifier({
        'backbone_type': 'resnet50', 'freeze_backbone': False,
        'pretrained': True, 'img_channels': 1, 'n_classes': 29,
        'use_mixed_precision': False,
    })
    out3 = m3(torch.zeros(2, 1, 448, 448))
    assert out3.shape == (2, 29), out3.shape
    print("ResNet50 1ch OUTPUT SHAPE:", out3.shape)

    print()
    print("ALL MODEL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
