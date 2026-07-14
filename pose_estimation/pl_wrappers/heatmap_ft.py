# EgoMax2D fine-tuning wrapper for the Stage1 heatmap model.
#
# Differences vs PoseHeatmapLightningModel:
#   - pretrained_ckpt: warm-starts the WHOLE EgoPoseFormerHeatmap
#     (encoder + heatmap head) from an EgoBody3M Stage1 checkpoint, then
#     trains with a fresh optimizer and this config's hyperparameters
#     (same pattern as Pose3DV2LightningModel.pretrained_ckpt).
#   - masked loss: the 26-channel MSE is weighted by the per-view
#     per-channel hm_weight from the dataset, so EgoMax2D samples only
#     backprop through the 5 labeled channels {2, 3, 6, 7, 10} and only
#     where that point actually has usable human coordinates.
#   - validation metric: direct pixel error of the 5 keypoints on the
#     256 canvas (val_kp_px_error, argmax decode x4), not heatmap L1.

from typing import Optional

import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from pose_estimation.datasets import EgoMax2DHeatmapDataset
from pose_estimation.datasets.egomax2d.remap import KPS, KP_JOINT_IDS
from pose_estimation.models.estimator import EgoPoseFormerHeatmap

_HM_SCALE = 4  # 256 canvas / 64 heatmap


class PoseHeatmapFTLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        lr: float,
        weight_decay: float,
        lr_decay_epochs: tuple,
        warmup_iters: int,
        batch_size: int,
        workers: int,
        dataset_kwargs: dict = {},
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        assert dataset_type == "egomax2d"
        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.model = EgoPoseFormerHeatmap(**model_cfg)
        if pretrained_ckpt is not None:
            self._load_pretrained_full(pretrained_ckpt)
        self.w_heatmap = self.model.train_cfg.get("w_heatmap", 1.0)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def _load_pretrained_full(self, ckpt_path: str):
        """Load the entire heatmap model (encoder + head) from a Stage1
        checkpoint; strict=True since the architecture is unchanged."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {
            k[len("model."):]: v for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        print(f"[pretrained_ckpt] loaded {ckpt_path}: "
              f"{len(state_dict) - len(unexpected)} matched, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    # ── forward pieces ─────────────────────────────────────────────────────────

    def _predict_heatmaps(self, img):
        """img [B, V, 3, H, W] -> heatmaps [B, V, 26, 64, 64]."""
        B, V = img.shape[:2]
        feats = self.model.forward_backbone(img)
        hm = self.model.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
        return hm.view(B, V, *hm.shape[1:])

    def _masked_loss(self, pred, gt, weight):
        """MSE over supervised channels only.

        pred/gt [B, V, 26, h, w], weight [B, V, 26] in {0, 1}.
        Mean over the supervised channels' pixels, scaled by w_heatmap —
        same scale as the unmasked pretrain loss on its supervised set.
        """
        with autocast(False):
            pred = pred.float()
            gt = gt.float()
            w = weight.float()[..., None, None]
            h, wpx = pred.shape[-2:]
            denom = (w.sum() * h * wpx).clamp(min=1.0)
            loss = ((pred - gt) ** 2 * w).sum() / denom
            return loss * self.w_heatmap

    def _kp_pixel_errors(self, pred_hm, kp2d, kp_mask):
        """Argmax-decode the 5 mapped channels and compare on the 256 canvas.

        pred_hm [B, V, 26, h, w], kp2d [B, V, 5, 2], kp_mask [B, V, 5].
        Returns (per_kp_error_sum [5], per_kp_count [5]).
        """
        B, V = pred_hm.shape[:2]
        h, w = pred_hm.shape[-2:]
        hm = pred_hm[:, :, list(KP_JOINT_IDS)].reshape(B, V, len(KP_JOINT_IDS), h * w)
        flat_idx = hm.argmax(dim=-1)                              # [B, V, 5]
        pred_u = (flat_idx % w).float() * _HM_SCALE
        pred_v = torch.div(flat_idx, w, rounding_mode="floor").float() * _HM_SCALE
        pred_xy = torch.stack([pred_u, pred_v], dim=-1)           # [B, V, 5, 2]
        err = torch.linalg.norm(pred_xy - kp2d, dim=-1) * kp_mask  # [B, V, 5]
        return err.sum(dim=(0, 1)), kp_mask.sum(dim=(0, 1))

    # ── lightning hooks ────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        assert self.model.training
        pred = self._predict_heatmaps(batch["img"])
        loss = self._masked_loss(pred, batch["heatmap_gt"], batch["hm_weight"])
        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def eval_step(self, batch, batch_idx, prefix):
        assert not self.model.training
        with torch.no_grad():
            pred = self._predict_heatmaps(batch["img"])
            loss = self._masked_loss(pred, batch["heatmap_gt"], batch["hm_weight"])
            err_sum, err_cnt = self._kp_pixel_errors(
                pred, batch["kp2d"], batch["kp_mask"])
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        # Weighted running mean over labeled points (batch_size=count).
        total_cnt = err_cnt.sum().clamp(min=1.0)
        self.log(f"{prefix}_kp_px_error", err_sum.sum() / total_cnt,
                 sync_dist=True, prog_bar=True, batch_size=int(total_cnt))
        for k, name in enumerate(KPS):
            if err_cnt[k] > 0:
                self.log(f"{prefix}_px_{name}", err_sum[k] / err_cnt[k],
                         sync_dist=True, batch_size=int(err_cnt[k]))
        return None

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        self.log("lr", optimizer.param_groups[0]["lr"],
                 on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)
        return [optimizer], [scheduler]

    # ── data ───────────────────────────────────────────────────────────────────

    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage == "fit":
            self.train_dataset = EgoMax2DHeatmapDataset(
                data_root=self.data_root, split="train", **self.dataset_kwargs)

        eval_split = "test" if stage in ("test", "predict") else "validation"
        self.eval_dataset = EgoMax2DHeatmapDataset(
            data_root=self.data_root, split=eval_split, **self.dataset_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            prefetch_factor=1,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=1,
        )

    def test_dataloader(self):
        return self.val_dataloader()
