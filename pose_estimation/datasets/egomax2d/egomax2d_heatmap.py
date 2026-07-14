# EgoMax2D heatmap fine-tuning dataset.
#
# Raw data: data/EgoMax2D/<ULID session>/ with images/head-front-{left,right}/
# (2592x1944 RGB fisheye), estimations.toon (5 human-annotated 2D keypoints:
# elbows, shoulders, pelvis) and calibration.json (per-session DS intrinsics).
#
# On first use, each session is preprocessed once into cached_root
# (default data/EgoMax2D_256/<session>/):
#   - images: calibration remap into EgoBody3M cam1/cam2 geometry (see
#     remap.py) + grayscale, saved as 256x256 JPEGs frame_XXXXXXXX_cam{1,2}.jpg
#     (cam1=left, cam2=right, matching EgoBody3M camera ids). Only frames with
#     at least one human label are cached (interpolated-only frames skipped).
#   - meta.npz: kp2d [N, V=2, K=5, 2] float32 (256-canvas coords, NaN when
#     unusable) and src [N, V, K] uint8 label-source codes.
# Later runs read the cache directly and skip all raw-data processing.
#
# Label-source codes (per point, per view):
#   0 = unusable  — no pixel_coords. This includes the ~37k out_of_frame
#       entries that still carry conf>0: they are filtered here by the
#       coords-is-None test, never by confidence.
#   1 = interpolated between human keyframes (coords, conf==0, status null)
#   2 = first-pass human keyframe annotation (coords, conf>0, status null)
#   3 = manual_annotated (second-pass human fix)
#   4 = occluded but human-annotated with coords
# Training default uses sources (2, 3, 4) — human labels only, weight 1.0.
#
# Split: sessions sorted by name, 8:1:1 — with 82 sessions that is
# train=first 66, validation=next 8, test=last 8 (later sessions held out).
#
# Output dict per sample (one frame):
#   img        : FloatTensor [V=2, 3, 256, 256]   grayscale x3 in [0, 1]
#   heatmap_gt : FloatTensor [V, 26, 64, 64]      Gaussians on mapped channels
#   hm_weight  : FloatTensor [V, 26]              1.0 on supervised channels
#   kp2d       : FloatTensor [V, 5, 2]            256-canvas coords (-1 invalid)
#   kp_mask    : FloatTensor [V, 5]               1.0 where kp2d is valid

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:
    from yaml import SafeLoader as _YamlLoader

from pose_estimation.datasets.egomax2d.remap import (
    IMG_SIZE, KPS, KP_JOINT_IDS, SIDE_SPECS,
    build_session_remaps, load_session_calib, remap_gt_point,
)
from pose_estimation.models.utils.camera_models import _EGOBODY3M_DS_PARAMS

_TARGET = 256
_HEATMAP = 64
_NUM_JOINTS = 26
_SIDES = ("left", "right")          # v=0 left->cam1, v=1 right->cam2
_CAM_ID = {"left": 1, "right": 2}

SRC_NONE, SRC_INTERP, SRC_HUMAN, SRC_MANUAL, SRC_OCCLUDED = 0, 1, 2, 3, 4
DEFAULT_LABEL_SOURCES = (SRC_HUMAN, SRC_MANUAL, SRC_OCCLUDED)


def _gaussian_heatmap(u: float, v: float, size: int, sigma: float) -> np.ndarray:
    """Gaussian blob centered at (u, v) on a size x size map (inlined from the
    EgoBody3M dataset so this module is self-contained)."""
    hm = np.zeros((size, size), dtype=np.float32)
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < size and 0 <= vi < size):
        return hm
    r = int(3 * sigma)
    x0, x1 = max(0, ui - r), min(size, ui + r + 1)
    y0, y1 = max(0, vi - r), min(size, vi + r + 1)
    xs, ys = np.arange(x0, x1) - ui, np.arange(y0, y1) - vi
    gx, gy = np.meshgrid(xs, ys)
    hm[y0:y1, x0:x1] = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
    return hm


def _classify_entry(entry: dict) -> int:
    if entry["pixel_coords"] is None:
        return SRC_NONE          # also catches out_of_frame with stale conf>0
    status = entry.get("status")
    if status == "manual_annotated":
        return SRC_MANUAL
    if status == "occluded":
        return SRC_OCCLUDED
    if (entry["confidence_score"] or 0) > 0:
        return SRC_HUMAN
    return SRC_INTERP


def build_session_cache(raw_session_dir: str, out_dir: str,
                        jpeg_quality: int = 92, rotate: str = "right") -> str:
    """Preprocess one session into out_dir (idempotent via .done marker)."""
    sid = os.path.basename(raw_session_dir.rstrip("/"))
    done_marker = os.path.join(out_dir, ".done")
    if os.path.exists(done_marker):
        return sid
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(raw_session_dir, "estimations.toon")) as f:
        toon = yaml.load(f, Loader=_YamlLoader)
    n = toon[SIDE_SPECS["left"]["toon_key"]]["metadata"]["number_of_frames"]

    calib = load_session_calib(raw_session_dir)
    remaps = build_session_remaps(raw_session_dir, rotate)

    kp2d = np.full((n, 2, len(KPS), 2), np.nan, dtype=np.float32)
    src = np.zeros((n, 2, len(KPS)), dtype=np.uint8)
    for v, side in enumerate(_SIDES):
        spec = SIDE_SPECS[side]
        frames = toon[spec["toon_key"]]["frames"]
        src_params = calib[spec["cam_name"]]
        eb_params = dict(_EGOBODY3M_DS_PARAMS[spec["eb_cam"]])
        for idx in range(n):
            fr = frames["%06d" % idx]
            for k, kp in enumerate(KPS):
                code = _classify_entry(fr[kp])
                if code == SRC_NONE:
                    continue
                coords = fr[kp]["pixel_coords"]
                p = remap_gt_point(coords[0], coords[1], src_params, eb_params, rotate)
                if p is None:
                    continue                       # outside DS domain -> unusable
                kp2d[idx, v, k] = p
                src[idx, v, k] = code

    # Cache images only for frames carrying at least one human label.
    human = np.isin(src, DEFAULT_LABEL_SOURCES) & np.isfinite(kp2d[..., 0])
    cache_frames = np.where(human.any(axis=(1, 2)))[0]
    for idx in cache_frames:
        for side in _SIDES:
            out_jpg = os.path.join(out_dir, f"frame_{idx:08d}_cam{_CAM_ID[side]}.jpg")
            if os.path.exists(out_jpg):
                continue
            raw_jpg = os.path.join(raw_session_dir, "images",
                                   SIDE_SPECS[side]["cam_dir"], f"frame_{idx:08d}.jpg")
            bgr = cv2.imread(raw_jpg)
            if bgr is None:
                raise FileNotFoundError(raw_jpg)
            mx, my = remaps[side]
            remapped = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            gray = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_jpg, gray, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    np.savez(os.path.join(out_dir, "meta.npz"), kp2d=kp2d, src=src)
    with open(done_marker, "w") as f:
        f.write(f"frames={n} cached_frames={len(cache_frames)} rotate={rotate}\n")
    return sid


def ensure_cache(raw_root: str, cached_root: str, sessions: list,
                 workers: int = 8, jpeg_quality: int = 92, rotate: str = "right"):
    """Build any missing per-session caches (parallel across sessions)."""
    todo = [s for s in sessions
            if not os.path.exists(os.path.join(cached_root, s, ".done"))]
    if not todo:
        return
    os.makedirs(cached_root, exist_ok=True)
    print(f"[EgoMax2D] Building cache for {len(todo)} sessions "
          f"({workers} workers) → {cached_root} …", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(build_session_cache, os.path.join(raw_root, s),
                        os.path.join(cached_root, s), jpeg_quality, rotate): s
            for s in todo
        }
        for done, fut in enumerate(as_completed(futures), 1):
            sid = fut.result()   # re-raises worker exceptions
            elapsed = time.time() - t0
            eta = elapsed / done * (len(todo) - done)
            print(f"[EgoMax2D]   {done}/{len(todo)} {sid}  "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
    print(f"[EgoMax2D] Cache build finished in {time.time() - t0:.0f}s")


def split_sessions(sessions: list, split: str,
                   ratio: tuple = (0.8, 0.1, 0.1)) -> list:
    """8:1:1 by sorted session name; the later sessions are held out
    (82 sessions -> train 66 / validation 8 / test 8)."""
    n = len(sessions)
    n_test = max(1, round(n * ratio[2]))
    n_val = max(1, round(n * ratio[1]))
    n_train = n - n_val - n_test
    if split == "train":
        return sessions[:n_train]
    if split == "validation":
        return sessions[n_train:n_train + n_val]
    return sessions[n_train + n_val:]


class EgoMax2DHeatmapDataset(Dataset):
    def __init__(
        self,
        data_root: str,                 # raw root, e.g. ./data/EgoMax2D
        split: str,
        sigma: float = 2.0,
        max_samples: int = None,
        cached_root: str = None,        # e.g. ./data/EgoMax2D_256
        label_sources: tuple = DEFAULT_LABEL_SOURCES,
        frame_step: int = 1,            # subsample labeled frames (1 = all)
        split_ratio: tuple = (0.8, 0.1, 0.1),
        cache_workers: int = 8,
        jpeg_quality: int = 92,
        rotate: str = "right",
        **kwargs,
    ):
        super().__init__()
        assert split in ("train", "validation", "test")
        self.sigma = sigma
        self.label_sources = tuple(label_sources)
        self.cached_root = cached_root or (data_root.rstrip("/") + "_256")

        if os.path.isdir(data_root):
            all_sessions = sorted(d for d in os.listdir(data_root)
                                  if os.path.isdir(os.path.join(data_root, d)))
        else:  # raw data gone; rely on a complete cache
            all_sessions = sorted(d for d in os.listdir(self.cached_root)
                                  if os.path.isdir(os.path.join(self.cached_root, d)))
        sessions = split_sessions(all_sessions, split, split_ratio)

        if os.path.isdir(data_root):
            ensure_cache(data_root, self.cached_root, sessions,
                         workers=cache_workers, jpeg_quality=jpeg_quality, rotate=rotate)

        # Load per-session meta and collect samples = frames with >=1 usable label.
        self.samples = []        # (session, frame_idx)
        self.kp2d = []           # [V, K, 2] float32 per sample
        self.kp_mask = []        # [V, K] float32 per sample
        n_pts = 0
        for sid in sessions:
            meta = np.load(os.path.join(self.cached_root, sid, "meta.npz"))
            kp2d, src = meta["kp2d"], meta["src"]
            usable = (np.isin(src, self.label_sources)
                      & np.isfinite(kp2d[..., 0])
                      & (kp2d[..., 0] >= 0) & (kp2d[..., 0] < _TARGET)
                      & (kp2d[..., 1] >= 0) & (kp2d[..., 1] < _TARGET))
            frame_ids = np.where(usable.any(axis=(1, 2)))[0][::frame_step]
            for idx in frame_ids:
                self.samples.append((sid, int(idx)))
                self.kp2d.append(np.where(usable[idx, ..., None], kp2d[idx], -1.0)
                                 .astype(np.float32))
                self.kp_mask.append(usable[idx].astype(np.float32))
                n_pts += int(usable[idx].sum())

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            self.kp2d = self.kp2d[:max_samples]
            self.kp_mask = self.kp_mask[:max_samples]
        print(f"[EgoMax2D] split={split}  sessions={len(sessions)}  "
              f"samples={len(self.samples)}  labeled_points={n_pts}  "
              f"sources={self.label_sources}  frame_step={frame_step}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, frame_idx = self.samples[idx]
        session_dir = os.path.join(self.cached_root, sid)

        imgs = []
        for side in _SIDES:
            path = os.path.join(session_dir, f"frame_{frame_idx:08d}_cam{_CAM_ID[side]}.jpg")
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(path)
            imgs.append(np.stack([gray] * 3, axis=0).astype(np.float32) / 255.0)
        imgs = np.stack(imgs)                                   # [V, 3, 256, 256]

        kp2d = self.kp2d[idx]                                   # [V, K, 2]
        kp_mask = self.kp_mask[idx]                             # [V, K]
        heatmaps = np.zeros((2, _NUM_JOINTS, _HEATMAP, _HEATMAP), dtype=np.float32)
        hm_weight = np.zeros((2, _NUM_JOINTS), dtype=np.float32)
        for v in range(2):
            for k, j in enumerate(KP_JOINT_IDS):
                if kp_mask[v, k] == 0:
                    continue
                u_hm = kp2d[v, k, 0] * _HEATMAP / _TARGET
                v_hm = kp2d[v, k, 1] * _HEATMAP / _TARGET
                heatmaps[v, j] = _gaussian_heatmap(u_hm, v_hm, _HEATMAP, self.sigma)
                hm_weight[v, j] = 1.0

        return {
            "img":        torch.from_numpy(imgs),
            "heatmap_gt": torch.from_numpy(heatmaps),
            "hm_weight":  torch.from_numpy(hm_weight),
            "kp2d":       torch.from_numpy(kp2d),
            "kp_mask":    torch.from_numpy(kp_mask),
        }
