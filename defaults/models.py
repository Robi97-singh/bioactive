import os

from .bases import *
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.
    
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)     

        if hasattr(cnn_models, self.backbone_type):
            incargs = {"aux_logits":False, "transform_input":False} if self.backbone_type == "inception_v3" else {}
            if self.pretrained:
                weight_pretrained = "IMAGENET1K_V1"
                print("using pretrained weights, ", weight_pretrained, " for ", self.backbone_type)
            else:
                weight_pretrained = None
            self.backbone = cnn_models.__dict__[self.backbone_type](weights=weight_pretrained, **incargs)
            fc_in_channels = self.backbone.fc.in_features
        else:
            raise NotImplementedError                
        self.backbone.fc = Identity()  # removing the fc layer from the backbone (which is manually added below)

        # modify stem and last layer
        self.fc = nn.Linear(fc_in_channels, self.n_classes)
        self.modify_first_layer(self.img_channels, self.pretrained)            
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)   

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            if self.freeze_backbone:
                self.backbone.eval()

            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1], 0)
                
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx             
            else:
                x_emb = self.backbone(x)
                
            x = self.fc(x_emb)
            

            if return_embedding:
                return x, x_emb        
            else:
                return x
        
    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return

        if backbone_type == 'ResNet':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight 
            print(f"Adapting first chanel of network to {img_channels} channels")

        elif backbone_type == 'Inception3':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.Conv2d_1a_3x3.conv, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.Conv2d_1a_3x3.conv.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.Conv2d_1a_3x3.conv.weight.data = pretrained_weight                 
                
        else:
            raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))


class DINOv2Classifier(BaseModel):
    """DINOv2 backbone with 5-channel patch embedding adaptation."""
    
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)
        
        from transformers import AutoModel
        model_id = {
            'dinov2_base':  'facebook/dinov2-base',
            'dinov2_large': 'facebook/dinov2-large',
            'dinov2_giant': 'facebook/dinov2-giant',
        }[self.backbone_type]
        
        print(f"Loading {model_id}...")
        self.backbone = AutoModel.from_pretrained(model_id)
        
        # Get embedding dimension
        hidden_size = self.backbone.config.hidden_size
        
        # Adapt patch embedding from 3 to 5 channels
        self._adapt_patch_embedding()
        
        # Classification head
        self.fc = nn.Linear(hidden_size, self.n_classes)
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)
        
        print(f"  Params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def _adapt_patch_embedding(self):
        """Adapt patch embedding from 3 to 5 channels using weight averaging."""
        if self.img_channels == 3:
            return
        
        proj = self.backbone.embeddings.patch_embeddings.projection
        old_weight = proj.weight.data  # [embed_dim, 3, patch_size, patch_size]
        
        # Repeat and slice to get 5-channel weights
        new_weight = old_weight.repeat(1, 2, 1, 1)[:, :self.img_channels, :, :]
        
        # Create new conv with 5 input channels
        new_proj = nn.Conv2d(
            self.img_channels,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None
        )
        new_proj.weight.data = new_weight
        if proj.bias is not None:
            new_proj.bias.data = proj.bias.data
            
        self.backbone.embeddings.patch_embeddings.projection = new_proj
        self.backbone.config.num_channels = self.img_channels
        self.backbone.embeddings.patch_embeddings.num_channels = self.img_channels
        print(f"  Adapted patch embedding to {self.img_channels} channels")

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            outputs = self.backbone(x)
            # Use CLS token as embedding
            x_emb = outputs.last_hidden_state[:, 0, :]
            x_out = self.fc(x_emb)
            
            if return_embedding:
                return x_out, x_emb
            return x_out


class CLIPVisionClassifier(BaseModel):
    """CLIP/BiomedCLIP vision encoder with 5-channel adaptation."""
    
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)
        
        import open_clip
        
        model_configs = {
            'clip_vitl14':  ('ViT-L-14', 'openai'),
            'biomedclip':   ('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', None),
        }
        
        model_id, pretrained = model_configs[self.backbone_type]
        print(f"Loading {model_id}...")
        
        if pretrained:
            full_model, _, _ = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        else:
            full_model, _, _ = open_clip.create_model_and_transforms(model_id)
        
        self.backbone = full_model.visual
        
        # Get output dimension
        embed_dim = self.backbone.output_dim if hasattr(self.backbone, 'output_dim') else self.backbone.head.in_features
        
        # Adapt first conv to 5 channels
        self._adapt_first_conv()
        
        # Classification head
        self.fc = nn.Linear(embed_dim, self.n_classes)
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)
            
        print(f"  Params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def _adapt_first_conv(self):
        """Adapt first conv layer from 3 to 5 channels."""
        if self.img_channels == 3:
            return
            
        # Find first conv layer in vision transformer patch embedding
        conv = self.backbone.conv1 if hasattr(self.backbone, 'conv1') else \
               self.backbone.trunk.patch_embed.proj
               
        old_weight = conv.weight.data
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
            
        if hasattr(self.backbone, 'conv1'):
            self.backbone.conv1 = new_conv
        else:
            self.backbone.trunk.patch_embed.proj = new_conv
            
        print(f"  Adapted first conv to {self.img_channels} channels")

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            x_emb = self.backbone(x)
            x_out = self.fc(x_emb)
            
            if return_embedding:
                return x_out, x_emb
            return x_out


class ConvNeXtClassifier(BaseModel):
    """ConvNeXt backbone with 5-channel adaptation using timm."""
    
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)
        
        import timm
        
        model_configs = {
            'convnext_base':  'convnext_base.fb_in22k',
            'convnext_large': 'convnext_large.fb_in22k',
        }
        
        model_id = model_configs[self.backbone_type]
        print(f"Loading {model_id}...")
        
        self.backbone = timm.create_model(
            model_id,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classifier head
            in_chans=self.img_channels  # timm handles channel adaptation natively
        )
        
        # Get embedding dimension
        embed_dim = self.backbone.num_features
        
        # Classification head
        self.fc = nn.Linear(embed_dim, self.n_classes)
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)
            
        print(f"  Adapted to {self.img_channels} channels")
        print(f"  Params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            x_emb = self.backbone(x)
            x_out = self.fc(x_emb)
            
            if return_embedding:
                return x_out, x_emb
            return x_out
