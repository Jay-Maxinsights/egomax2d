import torch
from pytorch_lightning.cli import LightningCLI

from pose_estimation.pl_wrappers import PoseHeatmapFTLightningModel  # noqa: F401


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    LightningCLI()
