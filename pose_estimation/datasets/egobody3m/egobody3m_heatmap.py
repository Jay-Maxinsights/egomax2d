import json
import os
import threading
import time
import zipfile
from abc import ABCMeta
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Thread-local ZipFile cache: one open handle per zip per worker, never closed during training.
_zip_cache = threading.local()

_TARGET    = 256             # model input size
_HEATMAP   = 64              # _TARGET / 4
_NUM_JOINTS = 26

# Per-camera native geometry (H, W) — matches preprocess/preprocess_images.py _CAM_SPECS.
# cam0=videoL  (left side,          480×636)
# cam1=videoFL (front-left fisheye, 1024×1280)
# cam2=videoFR (front-right fisheye,1024×1280)
# cam3=videoR  (right side,         480×636)
_CAM_SPECS = {
    0: (480,  636),
    1: (1024, 1280),
    2: (1024, 1280),
    3: (480,  636),
}

def _cam_pad_params(cam_id: int):
    """Return (square_side, pad_top, pad_bot) for pad-to-square preprocessing."""
    h, w = _CAM_SPECS[cam_id]
    sq = max(h, w)
    pad_top = (sq - h) // 2
    pad_bot = sq - h - pad_top
    return sq, pad_top, pad_bot

_EGO_CAMS_DEFAULT = (0, 1, 2, 3)  # all four cameras


def _open_zip(path: str) -> zipfile.ZipFile:
    if not hasattr(_zip_cache, "d"):
        _zip_cache.d = {}
    if path not in _zip_cache.d:
        _zip_cache.d[path] = zipfile.ZipFile(path, "r")
    return _zip_cache.d[path]


def _build_sequence_index(
    split_dir: str,
    suffix: str = ".images.zip",
    meta_dir: str = None,
    annotation_levels: tuple = None,
    frame_stride: int = 1,
) -> list:
    """Scan split_dir for *suffix files and return [(seq_id, frame_idx), ...].

    meta_dir: directory that holds *.metadata.zip and manifest files.
              Defaults to split_dir.
    annotation_levels: if given (e.g. (0,)), only include frames whose
              annotation_level is in this set. Filtered manifest is cached
              separately so subsequent runs are instant.
    frame_stride: take every Nth frame from each sequence (default 1 = all frames).
    """
    if meta_dir is None:
        meta_dir = split_dir

    # Choose manifest filename based on whether we're filtering by level.
    if annotation_levels is not None:
        level_tag = "_".join(str(l) for l in sorted(annotation_levels))
        manifest_path = os.path.join(meta_dir, f".manifest_level{level_tag}.json")
    else:
        manifest_path = os.path.join(meta_dir, ".manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        # keep only sequences whose image zip actually exists in split_dir
        manifest = {k: v for k, v in manifest.items()
                    if os.path.exists(os.path.join(split_dir, f"{k}{suffix}"))}
        return [(seq_id, i) for seq_id, frames in manifest.items() for i in frames[::frame_stride]]

    seq_ids = sorted(
        f[: -len(suffix)]
        for f in os.listdir(split_dir)
        if f.endswith(suffix)
    )

    def _scan_one(seq_id):
        meta_path = os.path.join(meta_dir, f"{seq_id}.metadata.zip")
        if not os.path.exists(meta_path):
            return seq_id, None
        with zipfile.ZipFile(meta_path) as zf:
            json_names = sorted(n for n in zf.namelist() if n.endswith(".json"))
            if annotation_levels is not None:
                valid_frames = []
                for name in json_names:
                    frame_data = json.loads(zf.read(name))
                    if frame_data.get("annotation_level", 0) in annotation_levels:
                        fname = os.path.basename(name)
                        idx = int(fname.replace("frame", "").replace(".json", ""))
                        valid_frames.append(idx)
                return seq_id, valid_frames
            else:
                return seq_id, list(range(len(json_names)))

    num_workers = min(32, os.cpu_count() or 4)
    print(f"[Dataset] Building manifest with {num_workers} threads for {len(seq_ids)} sequences …")
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_scan_one, sid): sid for sid in seq_ids}
        done = 0
        for fut in as_completed(futures):
            seq_id, frames = fut.result()
            done += 1
            if frames is not None:
                results[seq_id] = frames
            if done % 50 == 0 or done == len(seq_ids):
                print(f"[Dataset]   {done}/{len(seq_ids)} sequences scanned", flush=True)

    manifest = {sid: results[sid] for sid in seq_ids if sid in results}

    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        print(f"[Dataset] Manifest saved to {manifest_path}")
    except OSError:
        pass

    return [(seq_id, i) for seq_id, frames in manifest.items() for i in frames[::frame_stride]]


def _read_ego_img(img_zip_path: str, seq_id: str, frame_idx: int, cam_id: int) -> np.ndarray:
    """Read one egocentric JPEG from zip, pad to square then resize, return [3, H, W] float32 in [0, 1]."""
    key = f"{seq_id}/frame{frame_idx:04d}_cam{cam_id}.jpg"
    raw = _open_zip(img_zip_path).read(key)
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, pad_top, pad_bot = _cam_pad_params(cam_id)
    gray = cv2.copyMakeBorder(gray, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=0)
    gray = cv2.resize(gray, (_TARGET, _TARGET), interpolation=cv2.INTER_LINEAR)
    return np.stack([gray] * 3, axis=0).astype(np.float32) / 255.0


def _read_ego_img_cached(img_zip_path: str, seq_id: str, frame_idx: int, cam_id: int) -> np.ndarray:
    """Read a pre-processed 256×256 JPEG (no pad/resize needed), return [3, H, W] float32 in [0, 1]."""
    key = f"{seq_id}/frame{frame_idx:04d}_cam{cam_id}.jpg"
    raw = _open_zip(img_zip_path).read(key)
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.stack([gray] * 3, axis=0).astype(np.float32) / 255.0


def _read_ego_img_flat(split_dir: str, seq_id: str, frame_idx: int, cam_id: int) -> np.ndarray:
    """Read a pre-extracted 256×256 JPEG from flat directory, return [3, H, W] float32 in [0, 1]."""
    path = os.path.join(split_dir, seq_id, f"frame{frame_idx:04d}_cam{cam_id}.jpg")
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"flat image not found: {path}")
    return np.stack([gray] * 3, axis=0).astype(np.float32) / 255.0


def _gaussian_heatmap(u: float, v: float, size: int, sigma: float) -> np.ndarray:
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


def _build_heatmaps(meta: dict, cam_id: int, sigma: float) -> np.ndarray:
    """Build [NUM_JOINTS, HEATMAP, HEATMAP] float32 Gaussian heatmaps from projected joints."""
    proj = meta[f"projected_joint_positions_cam{cam_id}_px"]  # list of [u, v, depth_cm]
    sq, pad_top, _ = _cam_pad_params(cam_id)
    hms = np.zeros((_NUM_JOINTS, _HEATMAP, _HEATMAP), dtype=np.float32)
    for j, (u, v, depth) in enumerate(proj):
        if depth <= 0:  # joint behind camera
            continue
        u_hm = u * _HEATMAP / sq
        v_hm = (v + pad_top) * _HEATMAP / sq
        hms[j] = _gaussian_heatmap(u_hm, v_hm, _HEATMAP, sigma)
    return hms


def _build_heatmaps_from_proj(proj: np.ndarray, cam_id: int, sigma: float) -> np.ndarray:
    """Build [NUM_JOINTS, HEATMAP, HEATMAP] from pre-loaded (NUM_JOINTS, 3) numpy array [u, v, depth]."""
    sq, pad_top, _ = _cam_pad_params(cam_id)
    hms = np.zeros((_NUM_JOINTS, _HEATMAP, _HEATMAP), dtype=np.float32)
    for j in range(_NUM_JOINTS):
        u, v, depth = proj[j]
        if depth <= 0:
            continue
        u_hm = u * _HEATMAP / sq
        v_hm = (v + pad_top) * _HEATMAP / sq
        hms[j] = _gaussian_heatmap(u_hm, v_hm, _HEATMAP, sigma)
    return hms


class EgoBody3MHeatmapDataset(Dataset, metaclass=ABCMeta):
    """Per-frame dataset that returns egocentric images and Gaussian joint heatmaps.

    Output dict
    -----------
    img         : FloatTensor [V, 3, 256, 256]  V camera views
    heatmap_gt  : FloatTensor [V, 26, 64, 64]   Gaussian heatmaps per camera

    Parameters
    ----------
    ego_cams : tuple of int
        Camera IDs to load. Default (0, 1, 2, 3) uses all four cameras.
    cached_root : str, optional
        Root with pre-processed 256×256 zips. If a sequence directory exists
        (created by scripts/extract_zips.py), flat files are used instead of zip.
    meta_root : str, optional
        Root directory that contains per-split metadata zips (*.metadata.zip).
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        sigma: float = 2.0,
        max_samples: int = None,
        cached_root: str = None,
        ego_cams: tuple = _EGO_CAMS_DEFAULT,
        meta_root: str = None,
        annotation_levels: tuple = None,
        frame_stride: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert split in ("train", "test", "validation")
        self.split_dir = os.path.join(data_root, split)
        self.meta_split_dir = os.path.join(meta_root, split) if meta_root else self.split_dir
        self.sigma = sigma
        self.ego_cams = tuple(ego_cams)
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
        print(f"[Dataset] split={split}  samples={len(self.samples)}"
              f"  annotation_levels={ann_levels}  frame_stride={frame_stride}")

        # Detect sequences already extracted to flat directories by extract_zips.py.
        self._flat_seqs: set = set()
        if self.cached_split_dir:
            unique_seqs = {seq_id for seq_id, _ in self.samples}
            self._flat_seqs = {
                s for s in unique_seqs
                if os.path.exists(os.path.join(self.cached_split_dir, s, ".done"))
            }
            pct = 100 * len(self._flat_seqs) / max(len(unique_seqs), 1)
            print(f"[Dataset] flat seqs: {len(self._flat_seqs)}/{len(unique_seqs)} ({pct:.0f}%)"
                  f"  {'← will use direct cv2.imread' if self._flat_seqs else '← run scripts/extract_zips.py to speed up IO'}")

        # Pre-load all metadata projections into a memory-mapped numpy array.
        # Shape: (N, num_cams, 26, 3) float32. Loaded once at init, shared across workers via fork CoW.
        self._meta_proj: np.ndarray | None = None
        level_tag = "_".join(str(l) for l in sorted(ann_levels)) if ann_levels else "all"
        stride_tag = f"_s{frame_stride}" if frame_stride > 1 else ""
        meta_npy = os.path.join(self.meta_split_dir, f".meta_proj_level{level_tag}{stride_tag}.npy")

        if max_samples is not None:
            # Truncated debug run: never read/delete/overwrite the shared full-split cache.
            self._meta_proj = self._preload_metadata(None, ann_levels)
        elif os.path.exists(meta_npy):
            proj = np.load(meta_npy, mmap_mode="r")
            if proj.shape == (len(self.samples), len(self.ego_cams), _NUM_JOINTS, 3):
                self._meta_proj = proj
                print(f"[Dataset] meta cache loaded: {meta_npy}  ({proj.nbytes/1e9:.2f} GB mmap)")
            else:
                print(f"[Dataset] meta cache shape mismatch {proj.shape}, regenerating …")
                os.remove(meta_npy)

        if self._meta_proj is None:
            self._meta_proj = self._preload_metadata(meta_npy, ann_levels)

    def _preload_metadata(self, save_path: str, ann_levels) -> np.ndarray:
        """Read all frame JSONs from metadata zips and extract projection data into numpy array."""
        from collections import defaultdict

        N = len(self.samples)
        C = len(self.ego_cams)
        proj = np.zeros((N, C, _NUM_JOINTS, 3), dtype=np.float32)

        # Group samples by seq_id to read each metadata zip once sequentially.
        seq_to_items: dict = defaultdict(list)
        for i, (seq_id, frame_idx) in enumerate(self.samples):
            seq_to_items[seq_id].append((i, frame_idx))

        print(f"[Dataset] Pre-loading metadata: {N} frames from {len(seq_to_items)} seqs …", flush=True)
        t0 = time.time()
        num_seqs = len(seq_to_items)

        skipped_seqs = 0
        for done, (seq_id, items) in enumerate(seq_to_items.items()):
            meta_zip_path = os.path.join(self.meta_split_dir, f"{seq_id}.metadata.zip")
            try:
                with zipfile.ZipFile(meta_zip_path) as zf:
                    for sample_idx, frame_idx in items:
                        data = json.loads(zf.read(f"{seq_id}/frame{frame_idx:04d}.json"))
                        for c, cam_id in enumerate(self.ego_cams):
                            proj[sample_idx, c] = data[f"projected_joint_positions_cam{cam_id}_px"]
            except Exception as e:
                # Corrupted metadata zip: leave entries as zeros; __getitem__ retry loop will skip.
                skipped_seqs += 1
                print(f"[Dataset] warn: meta preload skip {seq_id}: {e}", flush=True)

            if (done + 1) % 200 == 0 or done + 1 == num_seqs:
                elapsed = time.time() - t0
                eta = elapsed / (done + 1) * (num_seqs - done - 1)
                print(f"[Dataset]   meta {done+1}/{num_seqs}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

        elapsed = time.time() - t0
        print(f"[Dataset] Metadata pre-loaded: {proj.nbytes/1e9:.2f} GB in {elapsed:.0f}s"
              + (f"  ({skipped_seqs} seqs skipped due to corrupt metadata)" if skipped_seqs else ""))

        if save_path is None:
            return proj
        try:
            np.save(save_path, proj)
            print(f"[Dataset] Meta cache saved → {save_path}")
            # Reload as mmap so workers share pages via fork CoW.
            return np.load(save_path, mmap_mode="r")
        except OSError as e:
            print(f"[Dataset] Warning: could not save meta cache: {e}")
            return proj

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(len(self.samples)):
            actual_idx = (idx + attempt) % len(self.samples)
            seq_id, frame_idx = self.samples[actual_idx]
            try:
                # --- Load images ---
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

                # --- Load heatmaps ---
                if self._meta_proj is not None:
                    heatmaps = np.stack([
                        _build_heatmaps_from_proj(self._meta_proj[actual_idx, c], cam, self.sigma)
                        for c, cam in enumerate(self.ego_cams)
                    ])
                else:
                    meta_zip = os.path.join(self.meta_split_dir, f"{seq_id}.metadata.zip")
                    meta = json.loads(_open_zip(meta_zip).read(f"{seq_id}/frame{frame_idx:04d}.json"))
                    heatmaps = np.stack([
                        _build_heatmaps(meta, cam, self.sigma)
                        for cam in self.ego_cams
                    ])

                return {
                    "img":        torch.from_numpy(imgs),
                    "heatmap_gt": torch.from_numpy(heatmaps),
                }
            except Exception:
                if attempt == 0 and hasattr(_zip_cache, "d") and self.cached_split_dir:
                    img_zip = os.path.join(self.cached_split_dir, f"{seq_id}.images_256.zip")
                    _zip_cache.d.pop(img_zip, None)
                continue
        raise RuntimeError(f"No valid sample found starting at idx={idx}")
