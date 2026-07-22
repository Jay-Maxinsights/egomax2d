"""ViT Stage1 heatmap inference on EgoMax2D raw stereo sequences.

Reads images/head-front-left + images/head-front-right from one EgoMax2D
session, runs the ViT heatmap model, and writes an mp4 with GT (top row,
from estimations.toon, 5 keypoints) vs Pred (bottom row, 26J skeleton)
side by side per camera.

Preprocessing is a calibration-aware ray-level remap into the EgoBody3M
training-camera geometry (left cam -> cam1, right cam -> cam2):
  256-canvas px -> EgoBody3M native px (undo pad+resize)
    -> unproject with the repo-estimated EgoBody3M Double Sphere params
    -> rotate the ray 90 deg about the optical axis (EgoBody3M images have
       the person horizontal; EgoMax2D raw frames are upright)
    -> project with the session's own DS calibration (calibration.json,
       differs per session) -> sample the raw 2592x1944 frame.
Frames are then grayscaled x3 like the training dataloader. GT keypoints go
through the same ray chain so overlays stay aligned.

Usage:
    cd /workspace/egomax2d
    python inference/inference_heatmap_egomax2d.py \
        --ckpt work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt \
        --session-idx 0 \
        --output-dir results/heatmap_egomax2d
"""

import argparse
import os
# import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, ImageReadMode
# import yaml

# try:
#     from yaml import CSafeLoader as _Loader
# except ImportError:
#     from yaml import SafeLoader as _Loader

from pose_estimation.models.estimator import EgoPoseFormerHeatmap
from pose_estimation.models.utils.camera_models import _EGOBODY3M_DS_PARAMS

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

# EgoMax2D raw resolution (same stream as max_data decoded_stereo)
# _RAW_H, _RAW_W = 1944, 2592
# _PAD_TOP = (_RAW_W - _RAW_H) // 2   # 324
# _PAD_BOT = _RAW_W - _RAW_H - _PAD_TOP

CAMS = [("head_front_left", "head-front-left"), ("head_front_right", "head-front-right")]

# ── skeleton (26J EgoBody3M) ───────────────────────────────────────────────────
# 0=Head 1=LCollar 2=LShoulder 3=LElbow 4=LWrist
# 5=RCollar 6=RShoulder 7=RElbow 8=RWrist 9=Neck 10=Pelvis
# 11=SpineLower 12=SpineMid 13=SpineUpper 14=RHip 15=RKnee
# 16=RAnkle 18=RFoot 20=LHip 21=LKnee 22=LAnkle 24=LFoot
# SKELETON_UPPER = [
#     (0, 9),
#     (9, 2), (2, 3), (3, 4),
#     (9, 6), (6, 7), (7, 8),
# ]
# 5-kp comparable view: shoulders(2/6), elbows(3/7), wrists(4/8), pelvis(10)
# PRED_JOINTS_5KP = {2, 3, 4, 6, 7, 8, 10}
# SKELETON_5KP = [(2, 3), (3, 4), (6, 7), (7, 8)]
# SKELETON_FULL = [
#     (0, 9),   (9, 13),  (13, 12), (12, 11), (11, 10),
#     (10, 20), (10, 14),
#     (20, 21), (21, 22), (22, 24),
#     (14, 15), (15, 16), (16, 18),
#     (9, 1),   (1, 2),   (2, 3),   (3, 4),
#     (9, 5),   (5, 6),   (6, 7),   (7, 8),
# ]
# PRED_JOINT_COLOR = (0, 255, 100)
# PRED_BONE_COLOR = (0, 200, 255)

# GT_BONES = [(2, 3), (6, 7)]   # shoulder -> elbow, both sides
# GT_JOINT_COLOR = (0, 230, 255)
# GT_BONE_COLOR = (0, 140, 255)


# ── calibration remap (shared with the EgoMax2D training dataloader) ──────────

from pose_estimation.datasets.egomax2d.remap import (  # noqa: E402
    KP2JOINT as GT_KP2JOINT,
    build_remap, load_session_calib, remap_gt_point,
)


def remap_preprocess(bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray):
    """Raw frame -> EgoBody3M-geometry 256x256 canvas, grayscale x3 like the
    training dataloader. Returns (canvas_bgr, tensor)."""
    remapped = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    return canvas, tensor


def preprocess(image_dir: str, map_x: np.ndarray, map_y: np.ndarray) -> torch.Tensor:
    """
    Load and preprocess an image for EgoPoseFormer.
    Args:
        - image_dir: str, path to the image
        - map_x: np.ndarray, remap x coordinates
        - map_y: np.ndarray, remap y coordinates
    Returns:
        - torch.Tensor, preprocessed image tensor of shape (3, 256, 256
    """
    bgr = cv2.imread(image_dir)
    remapped = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    return tensor


def batch_preprocess(image_dirs: list[str],
                     maps: list[tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
    """Batched CPU version of preprocess().

    Load + remap a list of frames and stack them into one tensor. Each path
    uses its aligned (map_x, map_y). The per-image transform (remap ->
    grayscale x3 -> /255) is identical to preprocess(), so outputs match the
    single-image path exactly.

    Args:
        - image_dirs: list[str], paths to the images.
        - maps: list of (map_x, map_y) tuples, aligned with image_dirs.
    Returns:
        - torch.Tensor, preprocessed batch of shape (N, 3, 256, 256).
    """
    tensors = []
    for path, (map_x, map_y) in zip(image_dirs, maps):
        bgr = cv2.imread(path)
        remapped = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gray = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1).copy()))
    return torch.stack(tensors, dim=0)  # (N, 3, 256, 256)


# BT.601 luma weights — identical to cv2.COLOR_BGR2GRAY
_GRAY_W = torch.tensor([0.299, 0.587, 0.114])


def preprocess_gpu(
    image_paths: list[str],
    maps: list[tuple[np.ndarray, np.ndarray]],
    device: str = "cuda",
    raw_wh: tuple[int, int] = (2592, 1944),
) -> torch.Tensor:
    """Batched GPU version of preprocess(): decode JPEGs on GPU (nvjpeg), remap
    to the 256 canvas with grid_sample, grayscale x3, /255.

    Args:
        image_paths: list[str] of JPEG paths.
        maps: list of (map_x, map_y) arrays from build_remap(), aligned with paths.
        device: CUDA device the model lives on.
        raw_wh: (W, H) of the raw frames the maps index into (EgoMax2D = 2592x1944).
    Returns:
        torch.Tensor (N, 3, 256, 256) float32 on `device`.
    """
    raw_w, raw_h = raw_wh
    dev = torch.device(device)

    byte_tensors = [torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.uint8)
                    for p in image_paths]
    imgs = decode_jpeg(byte_tensors, device=dev, mode=ImageReadMode.RGB)  # list of (3,H,W)

    grids = []
    for mx, my in maps:
        gx = torch.as_tensor(mx, dtype=torch.float32, device=dev)
        gy = torch.as_tensor(my, dtype=torch.float32, device=dev)
        grids.append(torch.stack([2 * gx / (raw_w - 1) - 1, 2 * gy / (raw_h - 1) - 1], dim=-1))
    grids = torch.stack(grids, dim=0)  # (N, 256, 256, 2)

    batch = torch.stack([im.float() for im in imgs], dim=0)  # (N, 3, H, W) RGB
    sampled = F.grid_sample(batch, grids, mode="bilinear",
                            padding_mode="zeros", align_corners=True)  # (N, 3, 256, 256)

    gray = (sampled * _GRAY_W.to(dev).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return gray.expand(-1, 3, -1, -1).contiguous() / 255.0


# ── model / IO helpers ──────────────────────────────────────────────────────────

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


def decode_heatmap(hm: np.ndarray):
    """26x64x64 -> joints (26,2) and confs (26,) in 256x256 space."""
    J = hm.shape[0]
    joints = np.full((J, 2), -1.0, dtype=np.float32)
    confs = np.zeros(J, dtype=np.float32)
    for j in range(J):
        flat_idx = hm[j].argmax()
        peak_val = float(hm[j].flat[flat_idx])
        if peak_val < CONF_THRESH:
            continue
        joints[j] = [float(flat_idx % HM_SIZE) * HM_SCALE,
                     float(flat_idx // HM_SIZE) * HM_SCALE]
        confs[j] = peak_val
    return joints, confs

# def preprocess_raw(bgr: np.ndarray, rotate: str = "right"):
#     """Same transform as inference_heatmap_maxdata.py: optional rotate ->
#     pad top/bottom by fixed constants -> resize to 256x256.
#     Returns (canvas_bgr, tensor).
#     """
#     if rotate == "right":
#         bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     elif rotate == "left":
#         bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

#     padded = cv2.copyMakeBorder(
#         bgr, _PAD_TOP, _PAD_BOT, 0, 0,
#         cv2.BORDER_CONSTANT, value=0,
#     )
#     resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#     rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
#     tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
#     return resized, tensor


# def transform_point(x: float, y: float, rotate: str = "right"):
#     """Map a raw-image pixel_coords point through the exact same
#     rotate -> pad -> resize pipeline as preprocess_raw, so GT dots land on
#     the same 256x256 canvas as the model input / predicted heatmap.
#     """
#     H, W = _RAW_H, _RAW_W
#     if rotate == "right":       # cv2.ROTATE_90_COUNTERCLOCKWISE
#         x, y = y, W - 1 - x
#         rh, rw = W, H
#     elif rotate == "left":      # cv2.ROTATE_90_CLOCKWISE
#         x, y = H - 1 - y, x
#         rh, rw = W, H
#     else:
#         rh, rw = H, W

#     y = y + _PAD_TOP
#     padded_h = rh + _PAD_TOP + _PAD_BOT
#     padded_w = rw
#     cx = x * (IMG_SIZE / padded_w)
#     cy = y * (IMG_SIZE / padded_h)
#     return cx, cy


# def get_gt_joints(toon_frame: dict, src_params: dict, eb_params: dict, rotate: str):
#     """EgoMax2D frame dict (5 kps) -> joints (26,2) / confs (26,) on the
#     same 256x256 canvas as decode_heatmap, indices per GT_KP2JOINT.
#     """
#     joints = np.full((26, 2), -1.0, dtype=np.float32)
#     confs = np.zeros(26, dtype=np.float32)
#     for kp, j in GT_KP2JOINT.items():
#         coords = toon_frame[kp]["pixel_coords"]
#         if coords is None:
#             continue
#         p = remap_gt_point(coords[0], coords[1], src_params, eb_params, rotate)
#         if p is None:
#             continue
#         joints[j] = p
#         confs[j] = 1.0
#     return joints, confs


# def draw_pose(canvas, joints, confs, joint_color, bone_color, skeleton,
#               only_joints=None) -> np.ndarray:
#     vis = canvas.copy()
#     for (a, b) in skeleton:
#         if confs[a] > CONF_THRESH and confs[b] > CONF_THRESH:
#             cv2.line(vis,
#                      (int(joints[a, 0]), int(joints[a, 1])),
#                      (int(joints[b, 0]), int(joints[b, 1])),
#                      bone_color, 2, cv2.LINE_AA)
#     for j in range(len(joints)):
#         if only_joints is not None and j not in only_joints:
#             continue
#         if confs[j] > CONF_THRESH:
#             cx, cy = int(joints[j, 0]), int(joints[j, 1])
#             cv2.circle(vis, (cx, cy), 4, joint_color, -1, cv2.LINE_AA)
#             cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)
#     return vis


# def add_label(img, text):
#     cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (255, 255, 255), 1, cv2.LINE_AA)
#     return img


# def make_frame(canvas_l, canvas_r, gt_l, gt_r, pred_l, pred_r, skeleton,
#                pred_joints=None):
#     r0_l = draw_pose(canvas_l, *gt_l, GT_JOINT_COLOR, GT_BONE_COLOR, GT_BONES)
#     r0_r = draw_pose(canvas_r, *gt_r, GT_JOINT_COLOR, GT_BONE_COLOR, GT_BONES)
#     r1_l = draw_pose(canvas_l, *pred_l, PRED_JOINT_COLOR, PRED_BONE_COLOR, skeleton,
#                      only_joints=pred_joints)
#     r1_r = draw_pose(canvas_r, *pred_r, PRED_JOINT_COLOR, PRED_BONE_COLOR, skeleton,
#                      only_joints=pred_joints)

#     n = "26J" if pred_joints is None else f"{len(pred_joints)}J"
#     add_label(r0_l, "GT   left  (5J)")
#     add_label(r0_r, "GT   right (5J)")
#     add_label(r1_l, f"Pred left  ({n})")
#     add_label(r1_r, f"Pred right ({n})")

#     row0 = np.concatenate([r0_l, r0_r], axis=1)
#     row1 = np.concatenate([r1_l, r1_r], axis=1)
#     return np.concatenate([row0, row1], axis=0)


# def _pick_ffmpeg_bin():
#     """Prefer a system ffmpeg with libx264 baked in. Inside the training
#     container, /opt/conda/bin/ffmpeg (first on PATH) has a broken
#     libopenh264 and no libx264 at all, while /usr/bin/ffmpeg has libx264.
#     """
#     candidates = ["/usr/bin/ffmpeg", "ffmpeg"]
#     for exe in candidates:
#         try:
#             encoders = subprocess.run(
#                 [exe, "-hide_banner", "-encoders"],
#                 capture_output=True, text=True,
#             ).stdout
#         except FileNotFoundError:
#             continue
#         if "libx264" in encoders:
#             return exe
#     raise RuntimeError("No ffmpeg with libx264 found (tried: " + ", ".join(candidates) + ")")


# def build_ffmpeg_cmd(out_path: str, w: int, h: int, fps: float):
#     exe = _pick_ffmpeg_bin()
#     return [
#         exe, "-y", "-hide_banner", "-loglevel", "error",
#         "-f", "rawvideo", "-vcodec", "rawvideo",
#         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:g}",
#         "-i", "pipe:0",
#         "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
#         "-pix_fmt", "yuv420p", out_path,
#     ]


def resolve_session(root: str, session: str, session_idx: int) -> str:
    all_sessions = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not all_sessions:
        raise FileNotFoundError(f"No session directories under {root}")
    if session:
        if session not in all_sessions:
            raise FileNotFoundError(f"Session {session} not found under {root}")
        return session
    return all_sessions[session_idx]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--root", default="data/EgoMax2D")
    parser.add_argument("--session", default=None, help="exact session ID; overrides --session-idx")
    parser.add_argument("--session-idx", type=int, default=0, help="sorted-index pick when --session not given")
    parser.add_argument("--output-dir", default="results/heatmap_egomax2d")
    parser.add_argument("--step", type=int, default=1, help="frame sampling step (default 1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=0)
    # parser.add_argument("--fps", type=float, default=None, help="default: 30/step (real-time)")
    # parser.add_argument("--with-leg", action="store_true", default=False, help="draw full-body pred skeleton")
    # EgoBody3M training images have the person horizontal; EgoMax2D raw
    # frames are upright, so the default 90-deg ray rotation ("right" = CCW
    # in image terms) is part of matching the training-camera geometry.
    parser.add_argument("--rotate", choices=["right", "left", "none"], default="right")
    # parser.add_argument("--save-frames", action="store_true", default=False, help="also dump per-frame PNGs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing-warmup", type=int, default=10,
                        help="first N forwards excluded from timing stats (CUDA init/JIT)")
    args = parser.parse_args()

    session = resolve_session(args.root, args.session, args.session_idx)
    session_dir = os.path.join(args.root, session)
    print(f"Session: {session}")

    calib = load_session_calib(session_dir)
    # left cam -> EgoBody3M cam1 geometry, right cam -> cam2
    remap_cfg = {
        "head-front-left": (calib["videoFL"], dict(_EGOBODY3M_DS_PARAMS[1])),
        "head-front-right": (calib["videoFR"], dict(_EGOBODY3M_DS_PARAMS[2])),
    }
    remaps = {cam_dir: build_remap(eb, src, args.rotate)
              for cam_dir, (src, eb) in remap_cfg.items()}

    # with open(os.path.join(session_dir, "estimations.toon")) as f:
    #     toon = yaml.load(f, Loader=_Loader)
    # n_frames = toon[CAMS[0][0]]["metadata"]["number_of_frames"]

    # Count total frames
    n_frames = len(os.listdir(os.path.join(session_dir, "images", CAMS[0][1])))
    idxs = list(range(0, n_frames, args.step))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]
    # fps = args.fps if args.fps is not None else 30.0 / args.step
    fps = 30

    # # default: only the joints comparable with EgoMax2D GT (arms + pelvis)
    # if args.with_leg:
    #     skeleton, pred_joints = SKELETON_FULL, None
    # else:
    #     skeleton, pred_joints = SKELETON_5KP, PRED_JOINTS_5KP

    # os.makedirs(args.output_dir, exist_ok=True)
    # frames_dir = os.path.join(args.output_dir, "frames")
    # if args.save_frames:
    #     os.makedirs(frames_dir, exist_ok=True)

    model = load_model(args.ckpt).to(args.device)

    # print(f"Frames: {len(idxs)}/{n_frames}  |  device: {args.device}  |  "
    #       f"skeleton: {'full' if args.with_leg else 'upper-body'}  |  rotate: {args.rotate}  |  fps: {fps:g}")
    print(f"Frames: {len(idxs)}/{n_frames}  |  device: {args.device}  |  "
          f"rotate: {args.rotate}  |  fps: {fps:g}")

    # out_video = os.path.join(args.output_dir, f"{session}_heatmap.mp4")
    # W, H = IMG_SIZE * 2, IMG_SIZE * 2
    # ffmpeg_cmd = build_ffmpeg_cmd(out_video, W, H, fps)
    # ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    B, V = 1, 2
    use_cuda = args.device.startswith("cuda")
    gpu_ms = []          # pure model forward (backbone + heatmap head), per stereo pair
    e2e_ms = []          # H2D copy + forward + D2H copy, per stereo pair
    predictions = {}
    with torch.no_grad():
        for fi, idx in enumerate(idxs):
            if fi % 50 == 0:
                print(f"  {fi}/{len(idxs)}  frame {idx:06d}")

            # # Original CPU version
            bgr_l = cv2.imread(os.path.join(session_dir, "images", CAMS[0][1], f"frame_{idx:08d}.jpg"))
            bgr_r = cv2.imread(os.path.join(session_dir, "images", CAMS[1][1], f"frame_{idx:08d}.jpg"))

            canvas_l, tensor_l = remap_preprocess(bgr_l, *remaps[CAMS[0][1]])
            canvas_r, tensor_r = remap_preprocess(bgr_r, *remaps[CAMS[1][1]])



            # tensor_l = preprocess(
            #     os.path.join(session_dir, "images", CAMS[0][1], f"frame_{idx:08d}.jpg"), 
            #     *remaps[CAMS[0][1]]
            # )
            # tensor_r = preprocess(
            #     os.path.join(session_dir, "images", CAMS[1][1], f"frame_{idx:08d}.jpg"), 
            #     *remaps[CAMS[1][1]]
            # )
            # tensor_l, tensor_r = preprocess_gpu(
            #     [
            #         os.path.join(session_dir, "images", CAMS[0][1], f"frame_{idx:08d}.jpg"),
            #         os.path.join(session_dir, "images", CAMS[1][1], f"frame_{idx:08d}.jpg"),
            #     ],
            #     [remaps[CAMS[0][1]], remaps[CAMS[1][1]]],
            #     device=args.device,
            # )

            t_e2e0 = time.perf_counter()
            img_t = torch.stack([tensor_l, tensor_r]).unsqueeze(0).to(args.device)
            if use_cuda:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            else:
                t_fwd0 = time.perf_counter()
          
            feats = model.forward_backbone(img_t)
            hm_gpu = model.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
            if use_cuda:
                ev1.record()
                torch.cuda.synchronize()
                gpu_ms.append(ev0.elapsed_time(ev1))
            else:
                gpu_ms.append((time.perf_counter() - t_fwd0) * 1000.0)
            heatmaps = hm_gpu.cpu().numpy()
            e2e_ms.append((time.perf_counter() - t_e2e0) * 1000.0)
            pred_l = decode_heatmap(heatmaps[0]) # Left camera prediction
            pred_r = decode_heatmap(heatmaps[1]) # Right camera prediction
            # Store predictions in a dict
            predictions[idx] = {
                "left": {"joints": pred_l[0], "confidences": pred_l[1]},
                "right": {"joints": pred_r[0], "confidences": pred_r[1]},
            }
    
    #         frame_l = toon[CAMS[0][0]]["frames"]["%06d" % idx]
    #         frame_r = toon[CAMS[1][0]]["frames"]["%06d" % idx]
    #         gt_l = get_gt_joints(frame_l, *remap_cfg[CAMS[0][1]], args.rotate)
    #         gt_r = get_gt_joints(frame_r, *remap_cfg[CAMS[1][1]], args.rotate)

    #         canvas = make_frame(canvas_l, canvas_r, gt_l, gt_r, pred_l, pred_r, skeleton,
    #                             pred_joints=pred_joints)
    #         cv2.putText(canvas, f"f{idx:06d}", (W - 90, 18),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    #         if args.save_frames:
    #             cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.png"), canvas)
    #         ffmpeg_proc.stdin.write(canvas.tobytes())

    # ffmpeg_proc.stdin.close()
    # ffmpeg_proc.wait()
    # print(f"\nVideo -> {out_video}")
    # if args.save_frames:
    #     print(f"Frames -> {frames_dir}/")

    # Save predictions to a .pt file
    predictions_path = os.path.join(args.output_dir, f"{session}_predictions.pt")
    torch.save(predictions, predictions_path)
    print(f"Predictions -> {predictions_path}")

    # ── timing report ──────────────────────────────────────────────────────────
    timing_csv = os.path.join(args.output_dir, f"{session}_timing.csv")
    with open(timing_csv, "w") as f:
        f.write("frame_idx,gpu_forward_ms_pair,e2e_ms_pair\n")
        for i, idx in enumerate(idxs):
            f.write(f"{idx},{gpu_ms[i]:.3f},{e2e_ms[i]:.3f}\n")

    warm = min(args.timing_warmup, max(len(gpu_ms) - 1, 0))
    g = np.array(gpu_ms[warm:])
    e = np.array(e2e_ms[warm:])
    dev_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"\n=== Inference timing ({dev_name}, warmup {warm} excluded, n={len(g)}) ===")
    print("Each forward = 1 stereo pair = 2 images (batch 2x3x256x256)")
    for name, a in [("GPU forward", g), ("e2e (H2D+fwd+D2H)", e)]:
        print(f"{name:>18}: mean {a.mean():6.2f} ms/pair  |  per-image {a.mean() / 2:6.2f} ms  |  "
              f"median {np.median(a):6.2f}  p95 {np.percentile(a, 95):6.2f}  "
              f"min {a.min():6.2f}  max {a.max():6.2f}")
    print(f"{'throughput':>18}: {1000.0 / g.mean():6.1f} pair/s  =  {2000.0 / g.mean():6.1f} img/s  (GPU forward only)")
    print(f"Timing CSV -> {timing_csv}")


# ── batched main ─────────────────────────────────────────────────────────────

def batch_main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--root", default="data/EgoMax2D")
    parser.add_argument("--session", default=None, help="exact session ID; overrides --session-idx")
    parser.add_argument("--session-idx", type=int, default=0, help="sorted-index pick when --session not given")
    parser.add_argument("--output-dir", default="results/heatmap_egomax2d")
    parser.add_argument("--step", type=int, default=1, help="frame sampling step (default 1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of stereo frames per model forward (model batch = batch_size*2)")
    parser.add_argument("--rotate", choices=["right", "left", "none"], default="right")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing-warmup", type=int, default=10,
                        help="first N forwards excluded from timing stats (CUDA init/JIT)")
    args = parser.parse_args()

    session = resolve_session(args.root, args.session, args.session_idx)
    session_dir = os.path.join(args.root, session)
    print(f"Session: {session}")

    calib = load_session_calib(session_dir)
    # left cam -> EgoBody3M cam1 geometry, right cam -> cam2
    remap_cfg = {
        "head-front-left": (calib["videoFL"], dict(_EGOBODY3M_DS_PARAMS[1])),
        "head-front-right": (calib["videoFR"], dict(_EGOBODY3M_DS_PARAMS[2])),
    }
    remaps = {cam_dir: build_remap(eb, src, args.rotate)
              for cam_dir, (src, eb) in remap_cfg.items()}

    # Count total frames
    n_frames = len(os.listdir(os.path.join(session_dir, "images", CAMS[0][1])))
    idxs = list(range(0, n_frames, args.step))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]
    fps = 30

    model = load_model(args.ckpt).to(args.device)

    bs = max(1, args.batch_size)
    n_batches = (len(idxs) + bs - 1) // bs
    print(f"Frames: {len(idxs)}/{n_frames}  |  device: {args.device}  |  "
          f"rotate: {args.rotate}  |  fps: {fps:g}  |  batch-size: {bs}  |  batches: {n_batches}")

    V = 2
    use_cuda = args.device.startswith("cuda")
    gpu_ms = []          # pure model forward (backbone + heatmap head), per batch
    e2e_ms = []          # H2D copy + forward + D2H copy, per batch
    batch_nframes = []   # frames in each batch (last may be partial)
    predictions = {}
    with torch.no_grad():
        for bi_batch in range(n_batches):
            chunk = idxs[bi_batch * bs: (bi_batch + 1) * bs]
            if bi_batch % 50 == 0:
                print(f"  batch {bi_batch}/{n_batches}  frames {chunk[0]:06d}..{chunk[-1]:06d}")

            # Build flat path/map lists grouped per frame: [f0_L, f0_R, f1_L, f1_R, ...]
            paths, maps_list = [], []
            for idx in chunk:
                for _cam_key, cam_dir in CAMS:      # left, then right
                    paths.append(os.path.join(session_dir, "images", cam_dir, f"frame_{idx:08d}.jpg"))
                    maps_list.append(remaps[cam_dir])

            # flat = batch_preprocess(paths, maps_list)   # (B*V, 3, 256, 256) CPU
            flat = preprocess_gpu(paths, maps_list, device=args.device)   # (B*V, 3, 256, 256) on device
            B = len(chunk)
            assert flat.shape == (B * V, 3, IMG_SIZE, IMG_SIZE), flat.shape

            t_e2e0 = time.perf_counter()
            img_t = flat.view(B, V, 3, IMG_SIZE, IMG_SIZE).to(args.device)
            if use_cuda:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            else:
                t_fwd0 = time.perf_counter()

            feats = model.forward_backbone(img_t)
            hm_gpu = model.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
            if use_cuda:
                ev1.record()
                torch.cuda.synchronize()
                gpu_ms.append(ev0.elapsed_time(ev1))
            else:
                gpu_ms.append((time.perf_counter() - t_fwd0) * 1000.0)
            heatmaps = hm_gpu.cpu().numpy()             # (B*V, 26, 64, 64)
            e2e_ms.append((time.perf_counter() - t_e2e0) * 1000.0)
            batch_nframes.append(B)
            assert heatmaps.shape == (B * V, MODEL_CFG["num_heatmap"], HM_SIZE, HM_SIZE), heatmaps.shape

            for bi, idx in enumerate(chunk):
                pred_l = decode_heatmap(heatmaps[bi * V + 0])   # Left camera prediction
                pred_r = decode_heatmap(heatmaps[bi * V + 1])   # Right camera prediction
                predictions[idx] = {
                    "left": {"joints": pred_l[0], "confidences": pred_l[1]},
                    "right": {"joints": pred_r[0], "confidences": pred_r[1]},
                }

    # Save predictions to a .pt file
    predictions_path = os.path.join(args.output_dir, f"{session}_predictions.pt")
    torch.save(predictions, predictions_path)
    print(f"Predictions -> {predictions_path}")

    # ── timing report ──────────────────────────────────────────────────────────
    timing_csv = os.path.join(args.output_dir, f"{session}_timing.csv")
    with open(timing_csv, "w") as f:
        f.write("batch_idx,n_frames,gpu_forward_ms_batch,e2e_ms_batch\n")
        for i in range(len(gpu_ms)):
            f.write(f"{i},{batch_nframes[i]},{gpu_ms[i]:.3f},{e2e_ms[i]:.3f}\n")

    warm = min(args.timing_warmup, max(len(gpu_ms) - 1, 0))
    g = np.array(gpu_ms[warm:])
    e = np.array(e2e_ms[warm:])
    imgs_per_batch = bs * V
    dev_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"\n=== Inference timing ({dev_name}, warmup {warm} excluded, n={len(g)}) ===")
    print(f"Each forward = 1 batch = {bs} stereo pairs = {imgs_per_batch} images "
          f"(batch {bs}x{V}x3x256x256)")
    for name, a in [("GPU forward", g), ("e2e (H2D+fwd+D2H)", e)]:
        print(f"{name:>18}: mean {a.mean():6.2f} ms/batch  |  per-image {a.mean() / imgs_per_batch:6.2f} ms  |  "
              f"median {np.median(a):6.2f}  p95 {np.percentile(a, 95):6.2f}  "
              f"min {a.min():6.2f}  max {a.max():6.2f}")
    print(f"{'throughput':>18}: {1000.0 / g.mean():6.1f} batch/s  =  "
          f"{1000.0 * imgs_per_batch / g.mean():6.1f} img/s  (GPU forward only)")
    print(f"Timing CSV -> {timing_csv}")


if __name__ == "__main__":
    batch_main()
    # main()
