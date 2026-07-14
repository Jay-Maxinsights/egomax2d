# ViT backbone for EPFv2 (Stage1 heatmap pretraining + Stage2 pose decoding).
#
# Loading paths:
#   1. timm model name (default "vit_base_patch16_224.augreg_in21k"); timm
#      interpolates the position embedding to img_size at creation time.
#   2. weights_path: an optional local state_dict (e.g. converted DINOv3
#      weights, whose HF repo is gated) loaded on top with strict=False.
#
# Output modes:
#   out_stride=4 : tokens are reshaped to a stride-16 map then upsampled by a
#                  2-level deconv neck to stride-4 [B, V, out_channels, 64, 64],
#                  matching what the heatmap head / EPFv1 decoder expect.
#   out_stride=16: raw projected token map [B, V, embed_dim, 16, 16] for the
#                  EPFv2 cross-attention decoder (256 tokens per view).

import torch
import torch.nn as nn

# Dataset images arrive as grayscale replicated to 3 channels in [0, 1] and are
# NOT normalized by the dataloader; ImageNet stats are applied here instead.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HeatmapUpsampleNeck(nn.Module):
    """stride-16 ViT feature map -> stride-4 map via two deconv blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.up2(self.up1(x))


class ViTBackbone(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224.augreg_in21k",
        pretrained=True,
        img_size=256,
        out_stride=4,
        out_channels=128,
        neck_mid_channels=256,
        drop_path_rate=0.0,
        weights_path=None,
        grad_checkpointing=False,
        imagenet_norm=True,
    ):
        super().__init__()
        assert out_stride in (4, 16)
        import timm  # deferred: resnet-only setups don't need timm installed

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = self.vit.embed_dim
        self.grid_size = self.vit.patch_embed.grid_size  # (h, w), e.g. (16, 16)
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)

        if weights_path is not None:
            self._load_external_weights(weights_path)
        if grad_checkpointing:
            self.vit.set_grad_checkpointing(True)

        self.out_stride = out_stride
        if out_stride == 4:
            self.neck = HeatmapUpsampleNeck(self.embed_dim, out_channels, neck_mid_channels)
            self._out_channels = out_channels
        else:
            self.neck = None
            self._out_channels = self.embed_dim

        self.imagenet_norm = imagenet_norm
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _load_external_weights(self, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        print(f"[ViTBackbone] external weights {weights_path}: "
              f"loaded={len(state) - len(unexpected)}, missing={len(missing)}, unexpected={len(unexpected)}")

    def get_output_channel(self):
        return self._out_channels

    def forward(self, image):
        # Same input contract as ResnetBackbone: [B, V, 3, H, W] or [B, V, H, W].
        if image.dim() == 4:
            B, V, H, W = image.shape
            x = image.reshape(B * V, 1, H, W).repeat(1, 3, 1, 1)
        else:
            B, V, C, H, W = image.shape
            x = image.reshape(B * V, C, H, W)

        if self.imagenet_norm:
            x = (x - self.pixel_mean) / self.pixel_std

        tokens = self.vit.forward_features(x)              # [B*V, prefix+L, C]
        tokens = tokens[:, self.num_prefix_tokens:]        # drop cls/register tokens
        h, w = self.grid_size
        feat = tokens.permute(0, 2, 1).reshape(B * V, self.embed_dim, h, w)

        if self.neck is not None:
            feat = self.neck(feat)
        return feat.reshape(B, V, *feat.shape[1:])
