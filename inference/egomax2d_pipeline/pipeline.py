"""Calibration-bound, batch-stateless processing stages for EgoMax2D inference.

``Pipeline`` exposes the existing numerical path (`preprocess_gpu` /
`batch_preprocess` -> model forward -> `decode_heatmap`) as three stages:
``preprocess`` -> ``inference`` -> ``decode``. Remaps and the model are built once
in ``__init__``; each stage is stateless across batches and carries no monitoring
code (see ``../../docs/specs/changes/EgoMax2D-inference-pipeline-refactor``).
"""

from __future__ import annotations

import numpy as np
from torch import Tensor

from pose_estimation.datasets.egomax2d.remap import build_remap
from pose_estimation.models.utils.camera_models import _EGOBODY3M_DS_PARAMS

# Reuse the legacy numerical helpers rather than reimplementing them. The
# dependency is one-way: the legacy script never imports this package.
from inference.inference_heatmap_egomax2d_dev import (
    batch_preprocess,
    decode_heatmap,
    preprocess_gpu,
)

from .configs.constant import CONF_THRESH, IMG_SIZE
from .io.reader import FrameBatch
from .model import load_model

# Two stereo views per frame (left, right).
_VIEWS = 2


class Pipeline:
    """Holds the model and per-camera remaps; runs the three inference stages."""

    def __init__(
        self,
        calibration: dict,
        ckpt: str,
        device: str = "cuda",
        rotate: str = "right",
        preprocess: str = "gpu",
        conf_thresh: float = CONF_THRESH,
    ) -> None:
        self._device = device
        self._preprocess_mode = preprocess
        self._conf_thresh = conf_thresh

        # Build both remaps once. EgoBody3M params first, session calib second;
        # left -> camera 1 + videoFL, right -> camera 2 + videoFR.
        self._left_map = build_remap(
            dict(_EGOBODY3M_DS_PARAMS[1]), calibration["videoFL"], rotate
        )
        self._right_map = build_remap(
            dict(_EGOBODY3M_DS_PARAMS[2]), calibration["videoFR"], rotate
        )

        # Load the model exactly once.
        self.model = load_model(ckpt).to(device)

    def preprocess(self, batch: FrameBatch) -> Tensor:
        """Return the batch as a ``(B, 2, 3, 256, 256)`` tensor on the device."""
        # Flatten per frame as [f0_L, f0_R, f1_L, f1_R, ...] with aligned maps.
        paths: list = []
        maps: list = []
        for left, right in zip(batch.left, batch.right):
            paths.append(left)
            maps.append(self._left_map)
            paths.append(right)
            maps.append(self._right_map)

        if self._preprocess_mode == "cpu":
            flat = batch_preprocess(paths, maps)
        else:
            flat = preprocess_gpu(paths, maps, device=self._device)

        b = len(batch.indices)
        return flat.view(b, _VIEWS, 3, IMG_SIZE, IMG_SIZE).to(self._device)

    def inference(self, x: Tensor) -> np.ndarray:
        """Run the model forward, returning ``(B*2, 26, 64, 64)`` NumPy on CPU."""
        b, v = x.shape[0], x.shape[1]
        feats = self.model.forward_backbone(x)
        hm = self.model.conv_heatmap(feats.view(b * v, *feats.shape[2:]))
        return hm.cpu().numpy()

    def decode(self, hm: np.ndarray, batch: FrameBatch, result: "InferenceResult") -> None:
        """Decode left/right heatmaps and store both under the original frame index."""
        for i, idx in enumerate(batch.indices):
            left = self._decode_view(hm[i * _VIEWS + 0])
            right = self._decode_view(hm[i * _VIEWS + 1])
            result.add(idx, left, right)

    def _decode_view(self, hm: np.ndarray):
        """Decode one view, applying the configurable confidence threshold.

        ``decode_heatmap`` already drops peaks below the ``CONF_THRESH`` decode
        floor, so with the default threshold this filter is a no-op and behavior
        is unchanged; a higher ``conf_thresh`` tightens the kept joints.
        """
        joints, confs = decode_heatmap(hm)
        if self._conf_thresh != CONF_THRESH:
            drop = confs < self._conf_thresh
            joints[drop] = -1.0
            confs[drop] = 0.0
        return joints, confs
