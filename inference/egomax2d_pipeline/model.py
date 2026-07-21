"""Model construction and checkpoint loading for the EgoMax2D pipeline."""

import torch

from pose_estimation.models.estimator import EgoPoseFormerHeatmap

from .configs.constant import MODEL_CFG


def load_model(ckpt_path: str) -> EgoPoseFormerHeatmap:
    model = EgoPoseFormerHeatmap(**MODEL_CFG)
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
