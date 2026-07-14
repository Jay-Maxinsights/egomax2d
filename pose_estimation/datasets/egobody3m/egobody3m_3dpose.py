import json
import os
import time
import zipfile
from abc import ABCMeta
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .egobody3m_heatmap import (
    _open_zip,
    _build_sequence_index,
    _read_ego_img,
    _read_ego_img_cached,
    _read_ego_img_flat,
    _zip_cache,
)

_NUM_JOINTS = 26
# Camera IDs match image filenames (frame{idx}_cam{id}.jpg).
# cam1=videoFL (front-left fisheye, 1024×1280), cam2=videoFR (front-right fisheye, 1024×1280)
# cam0=videoL (left side, 480×636), cam3=videoR (right side, 480×636)
# Legacy _256 caches may contain only cam1/cam2. For 4-cam training, run
# preprocess/preprocess_images.py with the default cams=(0,1,2,3) first.
_EGO_CAMS_DEFAULT = (1, 2)  # legacy-safe front pair; 4-cam training passes ego_cams=(0,1,2,3)

# Joint indices used to compute the pelvis as the midpoint of left and right hips.
# Based on EgoBody3M 26-joint skeleton: joint 14 = right hip, joint 20 = left hip.
_PELVIS_IDS = (14, 20)


def _transform_world_to_headset(points_world: np.ndarray, T_W_H: np.ndarray) -> np.ndarray:
    """Transform points from world coordinates into the headset frame."""
    T_H_W = np.linalg.inv(T_W_H)
    points_headset = T_H_W[:3, :3] @ points_world.T + T_H_W[:3, 3:4]
    return points_headset.T.astype(np.float32)


def _compute_pelvis_in_headset(joints_world: np.ndarray, T_W_H: np.ndarray) -> np.ndarray:
    """Return the pelvis position expressed in the headset coordinate frame."""
    pelvis_world = joints_world[list(_PELVIS_IDS)].mean(axis=0, keepdims=True)  # [1, 3]
    return _transform_world_to_headset(pelvis_world, T_W_H)[0]


class EgoBody3M3DPoseDataset(Dataset, metaclass=ABCMeta):
    """Per-frame dataset that returns egocentric images, pelvis-relative 3D pose,
    and the pelvis origin in the headset frame.

    Output dict
    -----------
    img          : FloatTensor [V, 3, 256, 256]  V camera views
    gt_pose      : FloatTensor [26, 3]           pelvis-relative joints, unit cm
    origin_3d    : FloatTensor [V, 1, 3]         pelvis in headset frame, repeated per view (cm)
    dataset_idx  : int

    Image loading priority (fastest first):
      1. Flat files extracted by scripts/extract_zips.py  (cv2.imread, OS page cache)
      2. Zip files fallback                               (Python zipfile)

    3D metadata is pre-loaded into RAM-mapped numpy arrays at __init__ time so that
    workers never touch metadata zips during training.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        max_samples: int = None,
        cached_root: str = None,
        ego_cams: tuple = _EGO_CAMS_DEFAULT,
        meta_root: str = None,
        pose_frame: str = "headset",
        annotation_levels: tuple = None,
        frame_stride: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert split in ("train", "test", "validation")
        assert pose_frame in ("world", "headset")
        self.split_dir = os.path.join(data_root, split)
        self.meta_split_dir = os.path.join(meta_root, split) if meta_root else self.split_dir
        self.ego_cams = tuple(ego_cams)
        self.pose_frame = pose_frame
        self.cached_split_dir = os.path.join(cached_root, split) if cached_root else None
        ann_levels = tuple(annotation_levels) if annotation_levels is not None else None

        if self.cached_split_dir:
            self.samples = _build_sequence_index(
                self.cached_split_dir, suffix=".images_256.zip",
                meta_dir=self.meta_split_dir, annotation_levels=ann_levels,
                frame_stride=frame_stride)
        else:
            self.samples = _build_sequence_index(
                self.split_dir, meta_dir=self.meta_split_dir, annotation_levels=ann_levels,
                frame_stride=frame_stride)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"[Dataset3D] split={split}  samples={len(self.samples)}"
              f"  annotation_levels={ann_levels}  frame_stride={frame_stride}")

        # Detect sequences extracted to flat directories by scripts/extract_zips.py.
        self._flat_seqs: set = set()
        if self.cached_split_dir:
            unique_seqs = {seq_id for seq_id, _ in self.samples}
            self._flat_seqs = {
                s for s in unique_seqs
                if os.path.exists(os.path.join(self.cached_split_dir, s, ".done"))
            }
            pct = 100 * len(self._flat_seqs) / max(len(unique_seqs), 1)
            print(f"[Dataset3D] flat seqs: {len(self._flat_seqs)}/{len(unique_seqs)} ({pct:.0f}%)"
                  f"  {'← direct cv2.imread' if self._flat_seqs else '← run scripts/extract_zips.py to speed up IO'}")

        # Pre-load 3D metadata (joints_world + T_W_H) into memory-mapped numpy arrays.
        # Shared across workers via fork copy-on-write; zero metadata IO during training.
        self._joints_world: np.ndarray | None = None
        self._T_W_H: np.ndarray | None = None
        level_tag  = "_".join(str(l) for l in sorted(ann_levels)) if ann_levels else "all"
        stride_tag = f"_s{frame_stride}" if frame_stride > 1 else ""
        joints_npy = os.path.join(self.meta_split_dir, f".meta_3d_joints_level{level_tag}{stride_tag}.npy")
        twh_npy    = os.path.join(self.meta_split_dir, f".meta_3d_twh_level{level_tag}{stride_tag}.npy")

        if max_samples is not None:
            # Truncated debug run: never read/delete/overwrite the shared full-split cache.
            self._preload_3d_metadata(None, None)
        elif os.path.exists(joints_npy) and os.path.exists(twh_npy):
            j = np.load(joints_npy, mmap_mode="r")
            t = np.load(twh_npy,    mmap_mode="r")
            if j.shape == (len(self.samples), _NUM_JOINTS, 3) and t.shape == (len(self.samples), 4, 4):
                self._joints_world = j
                self._T_W_H = t
                size_gb = (j.nbytes + t.nbytes) / 1e9
                print(f"[Dataset3D] 3D meta cache loaded: {joints_npy}  ({size_gb:.2f} GB mmap)")
            else:
                print(f"[Dataset3D] 3D meta cache shape mismatch, regenerating …")
                os.remove(joints_npy)
                os.remove(twh_npy)

        if self._joints_world is None:
            self._preload_3d_metadata(joints_npy, twh_npy)

    def _preload_3d_metadata(self, joints_npy: str, twh_npy: str):
        N = len(self.samples)
        joints_world = np.zeros((N, _NUM_JOINTS, 3), dtype=np.float32)
        T_W_H        = np.zeros((N, 4, 4),           dtype=np.float32)

        seq_to_items: dict = defaultdict(list)
        for i, (seq_id, frame_idx) in enumerate(self.samples):
            seq_to_items[seq_id].append((i, frame_idx))

        num_seqs = len(seq_to_items)
        print(f"[Dataset3D] Pre-loading 3D metadata: {N} frames from {num_seqs} seqs …", flush=True)
        t0 = time.time()

        skipped_seqs = 0
        for done, (seq_id, items) in enumerate(seq_to_items.items()):
            meta_zip_path = os.path.join(self.meta_split_dir, f"{seq_id}.metadata.zip")
            try:
                with zipfile.ZipFile(meta_zip_path) as zf:
                    for sample_idx, frame_idx in items:
                        data = json.loads(zf.read(f"{seq_id}/frame{frame_idx:04d}.json"))
                        joints_world[sample_idx] = data["joint_positions_world_cm"]
                        T_W_H[sample_idx]        = data["world_from_headset_xf_cm"]
            except Exception as e:
                # Corrupted metadata zip: leave entries as zeros; __getitem__ retry loop will skip.
                skipped_seqs += 1
                print(f"[Dataset3D] warn: meta preload skip {seq_id}: {e}", flush=True)

            if (done + 1) % 200 == 0 or done + 1 == num_seqs:
                elapsed = time.time() - t0
                eta = elapsed / (done + 1) * (num_seqs - done - 1)
                print(f"[Dataset3D]   3D meta {done+1}/{num_seqs}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

        elapsed = time.time() - t0
        size_gb = (joints_world.nbytes + T_W_H.nbytes) / 1e9
        print(f"[Dataset3D] 3D metadata pre-loaded: {size_gb:.2f} GB in {elapsed:.0f}s"
              + (f"  ({skipped_seqs} seqs skipped due to corrupt metadata)" if skipped_seqs else ""))

        if joints_npy is None:
            self._joints_world = joints_world
            self._T_W_H        = T_W_H
            return
        try:
            np.save(joints_npy, joints_world)
            np.save(twh_npy,    T_W_H)
            self._joints_world = np.load(joints_npy, mmap_mode="r")
            self._T_W_H        = np.load(twh_npy,    mmap_mode="r")
            print(f"[Dataset3D] 3D meta cache saved → {joints_npy}")
        except OSError as e:
            print(f"[Dataset3D] Warning: could not save 3D meta cache: {e}")
            self._joints_world = joints_world
            self._T_W_H        = T_W_H

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(len(self.samples)):
            actual_idx = (idx + attempt) % len(self.samples)
            seq_id, frame_idx = self.samples[actual_idx]
            try:
                # --- Load images (flat files preferred, zip as fallback) ---
                if self.cached_split_dir:
                    if seq_id in self._flat_seqs:
                        imgs = np.stack([
                            _read_ego_img_flat(self.cached_split_dir, seq_id, frame_idx, cam)
                            for cam in self.ego_cams
                        ])
                    else:
                        img_zip = os.path.join(self.cached_split_dir, f"{seq_id}.images_256.zip")
                        imgs = np.stack([
                            _read_ego_img_cached(img_zip, seq_id, frame_idx, cam)
                            for cam in self.ego_cams
                        ])
                else:
                    img_zip = os.path.join(self.split_dir, f"{seq_id}.images.zip")
                    imgs = np.stack([
                        _read_ego_img(img_zip, seq_id, frame_idx, cam)
                        for cam in self.ego_cams
                    ])

                # --- Load 3D metadata (numpy cache or zip fallback) ---
                if self._joints_world is not None:
                    joints_world = np.array(self._joints_world[actual_idx])  # (26, 3), copy from mmap
                    T_W_H_val    = np.array(self._T_W_H[actual_idx])         # (4, 4)
                else:
                    meta_zip = os.path.join(self.meta_split_dir, f"{seq_id}.metadata.zip")
                    meta = json.loads(_open_zip(meta_zip).read(f"{seq_id}/frame{frame_idx:04d}.json"))
                    joints_world = np.array(meta["joint_positions_world_cm"], dtype=np.float32)
                    T_W_H_val    = np.array(meta["world_from_headset_xf_cm"], dtype=np.float32)

                pelvis_headset = _compute_pelvis_in_headset(joints_world, T_W_H_val)

                if self.pose_frame == "headset":
                    joints_headset = _transform_world_to_headset(joints_world, T_W_H_val)
                    gt_pose = joints_headset - pelvis_headset          # [26, 3], headset-relative
                else:
                    pelvis_world = joints_world[list(_PELVIS_IDS)].mean(axis=0)
                    gt_pose = (joints_world - pelvis_world).astype(np.float32)

                V = len(self.ego_cams)
                origin_3d = np.stack([pelvis_headset] * V)[:, np.newaxis, :]  # [V, 1, 3]

                return {
                    "img":         torch.from_numpy(imgs),
                    "gt_pose":     torch.from_numpy(gt_pose),
                    "origin_3d":   torch.from_numpy(origin_3d),
                    "dataset_idx": idx,
                }
            except Exception:
                if attempt == 0 and hasattr(_zip_cache, "d") and self.cached_split_dir:
                    img_zip = os.path.join(self.cached_split_dir, f"{seq_id}.images_256.zip")
                    _zip_cache.d.pop(img_zip, None)
                continue
        raise RuntimeError(f"No valid sample found starting at idx={idx}")
