"""Sequential (windowed) EgoBody3M 3D-pose dataset for EPFv2 temporal training.

Phase B of analysis/model_design_plan.md: EPFv2's causal temporal attention and
jerk loss both need windows of temporally consecutive frames, plus the headset
6DoF pose as an auxiliary decoder input. This dataset reuses all of
EgoBody3M3DPoseDataset's indexing/caching machinery (manifest, flat/zip image
cache, mmap'd 3D metadata) and adds a window index on top.

Output dict (T = seq_len, V = len(ego_cams))
--------------------------------------------
img         : FloatTensor [T, V, 3, 256, 256]
gt_pose     : FloatTensor [T, 26, 3]   pelvis-relative joints, headset frame, cm
origin_3d   : FloatTensor [T, V, 1, 3] pelvis in headset frame, repeated per view
aux_pose    : FloatTensor [T, 9]       headset rotation 6D + gravity dir in headset frame
frame_idxs  : LongTensor  [T]          raw frame indices (consecutive by construction)
seq_id      : str
dataset_idx : int

Windowing
---------
Samples are grouped into runs of temporally consecutive frames (same seq_id,
raw frame index difference exactly 1 — annotation-level filtering leaves gaps).
Windows never straddle a gap. Runs shorter than seq_len are dropped instead of
padded: on the level-0 manifests 99.7% of consecutive kept-frame pairs are
adjacent (median run length: validation 30 / test 60 frames, max 4212), so the
loss is negligible and no valid-mask plumbing is needed.

- train (eval_mode=False): window starts every `window_stride` frames within
  each run; DataLoader shuffle randomizes window order across the epoch.
- eval (eval_mode=True): non-overlapping windows (stride = seq_len), plus a
  right-aligned tail window per run so every frame of the run is covered.

aux_pose
--------
Per frame, from T_W_H (world_from_headset, cm):
- rot 6D: first two columns of R_W_H (Zhou et al. continuity representation);
- gravity: world down-direction rotated into the headset frame,
  g_h = R_W_H^T @ (0, -1, 0). EgoBody3M world is Y-up — verified empirically:
  pelvis world-Y ~= 88 cm and headset world-Y ~= 166 cm on validation data,
  while X/Z wander freely.

frame_stride is intentionally NOT supported here (windows must be internally
consecutive); use window_stride to thin out training windows instead.
"""

import json
import os

import numpy as np
import torch

from .egobody3m_3dpose import (
    _PELVIS_IDS,
    EgoBody3M3DPoseDataset,
    _compute_pelvis_in_headset,
    _transform_world_to_headset,
)
from .egobody3m_heatmap import (
    _open_zip,
    _read_ego_img,
    _read_ego_img_cached,
    _read_ego_img_flat,
    _zip_cache,
)

# World down-direction (Y-up world, see module docstring).
_GRAVITY_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def headset_aux_pose(T_W_H: np.ndarray) -> np.ndarray:
    """[4,4] world_from_headset -> [9] rotation 6D + gravity direction in headset frame."""
    R = T_W_H[:3, :3]
    rot6d = R[:, :2].reshape(-1, order="F")      # columns 0 and 1, column-major -> [6]
    g_head = R.T @ _GRAVITY_WORLD                # R_H_W @ g_world
    return np.concatenate([rot6d, g_head]).astype(np.float32)


class EgoBody3M3DPoseSeqDataset(EgoBody3M3DPoseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        seq_len: int = 16,
        window_stride: int = 4,
        eval_mode: bool = False,
        frame_stride: int = 1,
        **kwargs,
    ):
        if frame_stride != 1:
            raise ValueError(
                "EgoBody3M3DPoseSeqDataset requires frame_stride=1 (windows must be "
                "temporally consecutive); use window_stride to space out windows."
            )
        super().__init__(data_root=data_root, split=split, frame_stride=1, **kwargs)

        self.seq_len = seq_len
        self.window_stride = window_stride
        self.eval_mode = eval_mode

        # Runs of consecutive positions in self.samples: same sequence, raw frame
        # index advancing by exactly 1. self.samples is ordered per sequence by
        # _build_sequence_index, so a single pass suffices.
        runs = []  # [start, end) positions into self.samples
        n = len(self.samples)
        run_start = 0
        for i in range(1, n + 1):
            if (
                i == n
                or self.samples[i][0] != self.samples[i - 1][0]
                or self.samples[i][1] != self.samples[i - 1][1] + 1
            ):
                runs.append((run_start, i))
                run_start = i

        self.windows = []  # start positions; window = samples[start : start + seq_len]
        dropped_runs = 0
        for s, e in runs:
            if e - s < seq_len:
                dropped_runs += 1
                continue
            if eval_mode:
                starts = list(range(s, e - seq_len + 1, seq_len))
                if starts[-1] != e - seq_len:      # right-align tail to cover the run
                    starts.append(e - seq_len)
            else:
                starts = list(range(s, e - seq_len + 1, window_stride))
            self.windows.extend(starts)

        print(
            f"[Dataset3DSeq] split={split}  windows={len(self.windows)}  "
            f"seq_len={seq_len}  window_stride={window_stride}  eval_mode={eval_mode}  "
            f"runs={len(runs)} (dropped {dropped_runs} shorter than seq_len)"
        )

    def __len__(self):
        return len(self.windows)

    def _read_frame_images(self, seq_id: str, frame_idx: int) -> np.ndarray:
        # Mirrors the image-loading priority of the parent's __getitem__.
        if self.cached_split_dir:
            if seq_id in self._flat_seqs:
                return np.stack([
                    _read_ego_img_flat(self.cached_split_dir, seq_id, frame_idx, cam)
                    for cam in self.ego_cams
                ])
            img_zip = os.path.join(self.cached_split_dir, f"{seq_id}.images_256.zip")
            return np.stack([
                _read_ego_img_cached(img_zip, seq_id, frame_idx, cam)
                for cam in self.ego_cams
            ])
        img_zip = os.path.join(self.split_dir, f"{seq_id}.images.zip")
        return np.stack([
            _read_ego_img(img_zip, seq_id, frame_idx, cam)
            for cam in self.ego_cams
        ])

    def _frame_meta(self, pos: int, seq_id: str, frame_idx: int):
        if self._joints_world is not None:
            return (
                np.array(self._joints_world[pos]),   # (26,3) copy from mmap
                np.array(self._T_W_H[pos]),          # (4,4)
            )
        meta_zip = os.path.join(self.meta_split_dir, f"{seq_id}.metadata.zip")
        meta = json.loads(_open_zip(meta_zip).read(f"{seq_id}/frame{frame_idx:04d}.json"))
        return (
            np.array(meta["joint_positions_world_cm"], dtype=np.float32),
            np.array(meta["world_from_headset_xf_cm"], dtype=np.float32),
        )

    def _load_frame(self, pos: int):
        """Load one frame by position in self.samples. Returns (imgs [V,3,H,W],
        gt_pose [26,3], origin_3d [V,1,3], aux [9]) as numpy arrays."""
        seq_id, frame_idx = self.samples[pos]
        imgs = self._read_frame_images(seq_id, frame_idx)

        joints_world, T_W_H_val = self._frame_meta(pos, seq_id, frame_idx)
        pelvis_headset = _compute_pelvis_in_headset(joints_world, T_W_H_val)

        if self.pose_frame == "headset":
            joints_headset = _transform_world_to_headset(joints_world, T_W_H_val)
            gt_pose = joints_headset - pelvis_headset
        else:
            pelvis_world = joints_world[list(_PELVIS_IDS)].mean(axis=0)
            gt_pose = (joints_world - pelvis_world).astype(np.float32)

        V = len(self.ego_cams)
        origin_3d = np.stack([pelvis_headset] * V)[:, np.newaxis, :]
        aux = headset_aux_pose(T_W_H_val)
        return imgs, gt_pose, origin_3d, aux

    def __getitem__(self, idx):
        for attempt in range(len(self.windows)):
            widx = (idx + attempt) % len(self.windows)
            start = self.windows[widx]
            try:
                imgs, poses, origins, auxes, fidxs = [], [], [], [], []
                for pos in range(start, start + self.seq_len):
                    im, gp, og, ax = self._load_frame(pos)
                    imgs.append(im)
                    poses.append(gp)
                    origins.append(og)
                    auxes.append(ax)
                    fidxs.append(self.samples[pos][1])

                return {
                    "img":         torch.from_numpy(np.stack(imgs)),      # [T,V,3,256,256]
                    "gt_pose":     torch.from_numpy(np.stack(poses)),     # [T,26,3]
                    "origin_3d":   torch.from_numpy(np.stack(origins)),   # [T,V,1,3]
                    "aux_pose":    torch.from_numpy(np.stack(auxes)),     # [T,9]
                    "frame_idxs":  torch.tensor(fidxs, dtype=torch.long), # [T]
                    "seq_id":      self.samples[start][0],
                    "dataset_idx": idx,
                }
            except Exception:
                # Same recovery as the parent: drop a possibly-stale zip handle
                # once, then move on to the next window.
                if attempt == 0 and hasattr(_zip_cache, "d") and self.cached_split_dir:
                    seq_id = self.samples[start][0]
                    img_zip = os.path.join(self.cached_split_dir, f"{seq_id}.images_256.zip")
                    _zip_cache.d.pop(img_zip, None)
                continue
        raise RuntimeError(f"No valid window found starting at idx={idx}")
