"""Render EgoMax2D demos in the ORIGINAL pixel space (upright color frames).

The fine-tuned model consumes calibration-remapped grayscale 256x256 inputs
(person horizontal, EgoBody3M geometry). This script runs that inference,
then maps every predicted keypoint through the INVERSE chain back onto the
raw upright 2592x1944 color frames (person at the bottom):

    canvas px -> EgoBody3M native px (undo pad+resize)
      -> unproject with EgoBody3M DS intrinsics -> ray
      -> rotate 90 deg about the optical axis (same direction as the image
         remap: EgoBody3M frame -> raw frame, (x,y,z) -> (-y,x,z))
      -> project with the session's own videoFL/FR DS calibration -> raw px

GT (estimations.toon) is already in raw pixels and is drawn directly — no
transform. Output: one mp4 per session on the color frames, same layout as
the 256-canvas demos: top row GT (green, 5 kps), bottom row Pred (blue,
7 joints: shoulders/elbows/wrists/pelvis), left|right cameras side by side.

Usage:
    cd /workspace/egomax2d
    python inference/tran2org.py --sessions all
    python inference/tran2org.py --sessions 0 --max-frames 300 --scale 0.5
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

# same-directory import (inference/ is not a package; inference.py shadows it)
from inference_heatmap_egomax2d import (  # noqa: E402
    load_model, decode_heatmap, remap_preprocess, _pick_ffmpeg_bin,
    CONF_THRESH, PRED_JOINTS_5KP, SKELETON_5KP,
)
from pose_estimation.datasets.egomax2d.remap import (
    IMG_SIZE, KPS, SIDE_SPECS,
    build_session_remaps, load_session_calib, ds_unproject, ds_project, _rotate_ray,
)
from pose_estimation.models.utils.camera_models import _EGOBODY3M_DS_PARAMS

DEFAULT_CKPT = ("work_dirs/egomax2d_vit_heatmap_ft/checkpoints"
                "/epoch=2-val_kp_px_error=4.89.ckpt")

RAW_H, RAW_W = 1944, 2592
GT_COLOR = (0, 255, 0)        # green joints
GT_BONE_COLOR = (0, 210, 0)   # green bones
PRED_COLOR = (255, 160, 60)   # light-blue joints
PRED_BONE_COLOR = (255, 0, 0)  # blue bones
GT_BONES = [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")]


def canvas_to_raw(pts, eb_params, src_params, rotate="right"):
    """[N, 2] points on the 256 canvas -> raw-image pixels. Returns
    (raw_pts [N, 2], valid [N])."""
    pts = np.asarray(pts, dtype=np.float64)
    scale = eb_params["square"] / IMG_SIZE
    u_nat = pts[:, 0] * scale
    v_nat = pts[:, 1] * scale - eb_params["pad_top"]
    x, y, z, v1 = ds_unproject(u_nat, v_nat, eb_params["fx"], eb_params["fy"],
                               eb_params["cx"], eb_params["cy"],
                               eb_params["xi"], eb_params["alpha"])
    x, y, z = _rotate_ray(x, y, z, rotate)
    us, vs, v2 = ds_project(x, y, z, **src_params)
    valid = (v1 & v2 & (us >= 0) & (us < RAW_W) & (vs >= 0) & (vs < RAW_H))
    return np.stack([us, vs], axis=-1).astype(np.float32), valid


def parse_sessions(spec, n):
    if spec == "all":
        return list(range(n))
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), min(int(b), n - 1) + 1))
    return [int(spec)]


def _marker_geom(scale, line_width):
    radius = max(6, int(20 * scale))
    thickness = max(4, line_width)
    return radius, thickness


def draw_gt_panel(bgr_scaled, toon_frame, scale, label, line_width):
    """Top-row panel: GT (green) from toon, already raw pixels."""
    img = bgr_scaled.copy()
    r, t = _marker_geom(scale, line_width)
    gt_pts = {}
    for kp in KPS:
        coords = toon_frame[kp]["pixel_coords"]
        if coords is None:
            continue
        gt_pts[kp] = (int(round(coords[0] * scale)), int(round(coords[1] * scale)))
    for a, b in GT_BONES:
        if a in gt_pts and b in gt_pts:
            cv2.line(img, gt_pts[a], gt_pts[b], GT_BONE_COLOR, t, cv2.LINE_AA)
    for p in gt_pts.values():
        cv2.circle(img, p, r, GT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_pred_panel(bgr_scaled, pred_confs, pred_valid, raw_pred,
                    scale, label, line_width):
    """Bottom-row panel: predictions (blue) mapped back to raw pixels."""
    img = bgr_scaled.copy()
    r, t = _marker_geom(scale, line_width)
    pp = {}
    for j in PRED_JOINTS_5KP:
        if pred_confs[j] > CONF_THRESH and pred_valid[j]:
            pp[j] = (int(round(raw_pred[j, 0] * scale)),
                     int(round(raw_pred[j, 1] * scale)))
    for a, b in SKELETON_5KP:
        if a in pp and b in pp:
            cv2.line(img, pp[a], pp[b], PRED_BONE_COLOR, t, cv2.LINE_AA)
    for p in pp.values():
        cv2.circle(img, p, r, PRED_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render_session(model, root, sid, args):
    session_dir = os.path.join(root, sid)
    with open(os.path.join(session_dir, "estimations.toon")) as f:
        toon = yaml.load(f, Loader=_Loader)
    n = toon[SIDE_SPECS["left"]["toon_key"]]["metadata"]["number_of_frames"]

    calib = load_session_calib(session_dir)
    remaps = build_session_remaps(session_dir)   # for model input
    eb = {s: dict(_EGOBODY3M_DS_PARAMS[SIDE_SPECS[s]["eb_cam"]]) for s in ("left", "right")}
    src = {s: calib[SIDE_SPECS[s]["cam_name"]] for s in ("left", "right")}

    idxs = list(range(0, n, args.step))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]
    fps = args.fps if args.fps is not None else 30.0 / args.step

    out_path = os.path.join(args.output_dir, f"{sid}_org.mp4")
    proc = None
    with torch.no_grad():
        for fi, idx in enumerate(idxs):
            if fi % 100 == 0:
                print(f"  [{sid[:10]}…] {fi}/{len(idxs)}", flush=True)
            raw = {}
            tensors = []
            for side in ("left", "right"):
                p = os.path.join(session_dir, "images", SIDE_SPECS[side]["cam_dir"],
                                 f"frame_{idx:08d}.jpg")
                raw[side] = cv2.imread(p)
                if raw[side] is None:
                    raise FileNotFoundError(p)
                _, t = remap_preprocess(raw[side], *remaps[side])
                tensors.append(t)

            img_t = torch.stack(tensors).unsqueeze(0).to(args.device)
            feats = model.forward_backbone(img_t)
            hm = model.conv_heatmap(feats.view(2, *feats.shape[2:])).cpu().numpy()

            gt_row, pred_row = [], []
            for v, side in enumerate(("left", "right")):
                joints, confs = decode_heatmap(hm[v])          # 256 canvas
                raw_pred, valid = canvas_to_raw(joints, eb[side], src[side])
                frame_ann = toon[SIDE_SPECS[side]["toon_key"]]["frames"]["%06d" % idx]
                scaled = cv2.resize(raw[side], None, fx=args.scale, fy=args.scale,
                                    interpolation=cv2.INTER_AREA)
                gt_row.append(draw_gt_panel(
                    scaled, frame_ann, args.scale,
                    f"GT    {side}  f{idx:06d}", args.line_width))
                pred_row.append(draw_pred_panel(
                    scaled, confs, valid, raw_pred, args.scale,
                    f"Pred  {side}  f{idx:06d}", args.line_width))

            canvas = np.vstack([np.hstack(gt_row), np.hstack(pred_row)])
            h, w = canvas.shape[:2]
            canvas = canvas[: h - h % 16, : w - w % 16]        # x264-safe dims
            if proc is None:
                h, w = canvas.shape[:2]
                proc = subprocess.Popen(
                    [_pick_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                     "-r", f"{fps:g}", "-i", "pipe:0",
                     "-vcodec", "libx264", "-preset", "medium", "-crf", str(args.crf),
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")
    return out_path, len(idxs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--root", default="data/EgoMax2D")
    ap.add_argument("--sessions", default="all",
                    help="sorted-index range '0-3', single '0', or 'all'")
    ap.add_argument("--output-dir", default="results/heatmap_egomax2d_org")
    ap.add_argument("--scale", type=float, default=0.35,
                    help="render scale of the raw 2592x1944 frames (default 0.35)")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=None, help="default: 30/step")
    ap.add_argument("--crf", type=int, default=26,
                    help="x264 crf (default 26 keeps files small at this size)")
    ap.add_argument("--line-width", type=int, default=5,
                    help="bone line thickness in output pixels (default 5)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    all_sessions = sorted(d for d in os.listdir(args.root)
                          if os.path.isdir(os.path.join(args.root, d)))
    picks = parse_sessions(args.sessions, len(all_sessions))
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.ckpt).to(args.device)
    for i in picks:
        sid = all_sessions[i]
        print(f"session {i}: {sid}", flush=True)
        path, nf = render_session(model, args.root, sid, args)
        print(f"  -> {path} ({nf} frames)", flush=True)
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
