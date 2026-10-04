"""
Applies the brightfield patch to defaults/models.py programmatically.
Run from the bioactive repo root:

    python3 apply_models_patch.py
"""
import pathlib
import sys

TARGET = pathlib.Path("defaults/models.py")

# --- 1) ResNet 1-channel branch ---------------------------------------------
OLD_RESNET = '''            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]'''

NEW_RESNET = '''            pretrained_weight = self.backbone.conv1.weight.data
            if img_channels == 1:
                # grayscale (brightfield): average the pretrained RGB weights
                # into a single input channel, rather than slicing off just
                # the red channel via repeat+slice.
                pretrained_weight = pretrained_weight.mean(dim=1, keepdim=True)
            else:
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]'''

# --- 2) new DINOv3Classifier -------------------------------------------------
DINOV3_CLASSIFIER = '''

class DINOv3Classifier(BaseModel):
    """DINOv3 backbone with patch embedding adapted to img_channels (5 for
    fluorescence, 1 for brightfield)."""

    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)

        from transformers import AutoModel
        model_id = {
            'dinov3_small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'dinov3_base':  'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        }[self.backbone_type]

        print(f"Loading {model_id}...")
        self.backbone = AutoModel.from_pretrained(model_id)
        hidden_size = self.backbone.config.hidden_size

        self._adapt_patch_embedding()

        self.fc = nn.Linear(hidden_size, self.n_classes)

        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)

        print(f"  Params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def _adapt_patch_embedding(self):
        """DINOv3: embeddings.patch_embeddings IS the Conv2d (no .projection wrapper)."""
        if self.img_channels == 3:
            return

        conv = self.backbone.embeddings.patch_embeddings
        old_weight = conv.weight.data                      # [embed_dim, 3, 16, 16]

        if self.img_channels == 1:
            # grayscale (brightfield): average the 3 pretrained input-channel
            # weights into one, instead of repeat+slice (which would just
            # keep the red-channel weights verbatim).
            new_weight = old_weight.mean(dim=1, keepdim=True)
        else:
            new_weight = old_weight.repeat(1, 2, 1, 1)[:, :self.img_channels, :, :]

        new_conv = nn.Conv2d(
            self.img_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None
        )
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        self.backbone.embeddings.patch_embeddings = new_conv
        self.backbone.config.num_channels = self.img_channels
        print(f"  Adapted patch embedding to {self.img_channels} channels")

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            outputs = self.backbone(x)
            x_emb = outputs.last_hidden_state[:, 0, :]     # CLS token
            x_out = self.fc(x_emb)
            if return_embedding:
                return x_out, x_emb
            return x_out

'''

INSERT_MARKER = "class CLIPVisionClassifier(BaseModel):"

# --- 3) Cell-DINO: load at native channels, adapt afterward -----------------
OLD_CELLDINO = '''        self.backbone = torch.hub.load(
            REPO, "cell_dino_cp_vits8",
            source="local",
            pretrained_path=CKPT,
            in_channels=self.img_channels,     # 5
        )'''

NEW_CELLDINO = '''        native_channels = 5   # Cell-DINO ViT-S/8 CP checkpoint's native input width
        self.backbone = torch.hub.load(
            REPO, "cell_dino_cp_vits8",
            source="local",
            pretrained_path=CKPT,
            in_channels=native_channels,
        )
        self._adapt_patch_embedding(native_channels)'''

CELLDINO_METHOD = '''    def _adapt_patch_embedding(self, native_channels: int):
        """Cell-DINO's hub loader does strict=True state-dict loading, so the
        model must be built+loaded at its native channel count first. If the
        requested img_channels differs (e.g. 1 for brightfield), swap in an
        averaged/adapted patch_embed.proj conv afterward."""
        if self.img_channels == native_channels:
            return

        conv = self.backbone.patch_embed.proj          # nn.Conv2d, confirmed
        old_weight = conv.weight.data                   # [384, native_channels, 8, 8]

        if self.img_channels == 1:
            new_weight = old_weight.mean(dim=1, keepdim=True)
        elif self.img_channels < native_channels:
            keep = self.img_channels - 1
            new_weight = torch.cat(
                [old_weight[:, :keep], old_weight[:, keep:].mean(dim=1, keepdim=True)],
                dim=1,
            )
        else:
            new_weight = old_weight.repeat(
                1, (self.img_channels + native_channels - 1) // native_channels, 1, 1
            )[:, :self.img_channels]

        new_conv = nn.Conv2d(
            self.img_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        self.backbone.patch_embed.proj = new_conv
        print(f"  Adapted Cell-DINO patch embedding from {native_channels} "
              f"to {self.img_channels} channel(s)")

'''

CELLDINO_METHOD_ANCHOR = '''    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            x_in = F.interpolate(
                x, size=(self.celldino_input_size, self.celldino_input_size),'''


def main() -> None:
    text = TARGET.read_text()

    # 1) ResNet
    if "if img_channels == 1:" in text:
        print("[patch] ResNet 1-channel branch already present, skipping", file=sys.stderr)
    else:
        if text.count(OLD_RESNET) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of OLD_RESNET, found {text.count(OLD_RESNET)}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(OLD_RESNET, NEW_RESNET, 1)
        print("[patch] ResNet 1-channel branch added", file=sys.stderr)

    # 2) DINOv3Classifier
    if "class DINOv3Classifier" in text:
        print("[patch] DINOv3Classifier already present, skipping", file=sys.stderr)
    else:
        if text.count(INSERT_MARKER) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of {INSERT_MARKER!r}, found {text.count(INSERT_MARKER)}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(INSERT_MARKER, DINOV3_CLASSIFIER.strip("\n") + "\n\n\n" + INSERT_MARKER, 1)
        print("[patch] DINOv3Classifier inserted", file=sys.stderr)

    # 3) Cell-DINO
    if "_adapt_patch_embedding(native_channels)" in text and "def _adapt_patch_embedding(self, native_channels" in text:
        print("[patch] Cell-DINO channel adaptation already present, skipping", file=sys.stderr)
    else:
        if text.count(OLD_CELLDINO) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of OLD_CELLDINO, found {text.count(OLD_CELLDINO)}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(OLD_CELLDINO, NEW_CELLDINO, 1)

        if text.count(CELLDINO_METHOD_ANCHOR) != 1:
            print(f"[patch] ERROR: expected 1 occurrence of CELLDINO_METHOD_ANCHOR, found {text.count(CELLDINO_METHOD_ANCHOR)}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(CELLDINO_METHOD_ANCHOR, CELLDINO_METHOD + CELLDINO_METHOD_ANCHOR, 1)
        print("[patch] Cell-DINO channel adaptation added", file=sys.stderr)

    TARGET.write_text(text)

    import ast
    ast.parse(TARGET.read_text())
    print("[patch] syntax OK", file=sys.stderr)


if __name__ == "__main__":
    main()
