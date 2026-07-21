"""Model and inference constants shared by the EgoMax2D pipeline."""

# ── model config (must match configs/egobody3m_vit_heatmap.yaml) ──────────────
_ENCODER_CFG = dict(
    type="vit",
    model_name="vit_base_patch16_224.augreg_in21k",
    pretrained=False,   # overwritten by ckpt state_dict below
    img_size=256,
    out_stride=4,
    out_channels=128,
    neck_mid_channels=256,
    drop_path_rate=0.1,
    grad_checkpointing=False,
    weights_path=None,
)
MODEL_CFG = dict(num_heatmap=26, encoder_cfg=_ENCODER_CFG, train_cfg=dict(w_heatmap=10.0))

DEFAULT_CKPT = (
    "work_dirs/egomax2d_vit_heatmap_ft/checkpoints"
    "/epoch=2-val_kp_px_error=4.89.ckpt"
)

# ── constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 256
HM_SIZE = 64
HM_SCALE = IMG_SIZE / HM_SIZE   # 4.0
CONF_THRESH = 0.01

CAMS = [("head_front_left", "head-front-left"), ("head_front_right", "head-front-right")]
