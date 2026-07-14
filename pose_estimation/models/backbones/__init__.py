from pose_estimation.models.backbones.resnet import ResnetBackbone


def build_encoder(encoder_cfg):
    """Dispatch on encoder_cfg["type"]; legacy configs without "type" are ResNet."""
    cfg = dict(encoder_cfg)
    enc_type = cfg.pop("type", "resnet")
    if enc_type == "resnet":
        return ResnetBackbone(**cfg)
    if enc_type == "vit":
        from pose_estimation.models.backbones.vit import ViTBackbone  # needs timm
        return ViTBackbone(**cfg)
    raise NotImplementedError(f"Unknown encoder type: {enc_type}")
