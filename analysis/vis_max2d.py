#!/usr/bin/env python3
"""Quick visualization of EgoMax2D estimations.toon 2D keypoints.

Draws the 5 keypoints (elbows, shoulders, pelvis) on the frames of one
session. Marker style encodes the label source:
  - solid circle + confidence text : detector keyframe (conf > 0)
  - hollow circle                  : interpolated (coords, conf == 0)
  - solid square                   : manual_annotated
  - missing (out_of_frame etc.)    : listed in the top-left status text

Usage examples
--------------
  # 10 frames of one session, both cameras side by side, half resolution
  python analysis/vis_max2d.py data/EgoMax2D/01KW3RYY4Z67HKAVYMAG8QG9WH \
      --frames 0:300:30 --out analysis/vis_max2d_out

  # single camera, every frame of a range, plus an mp4
  python analysis/vis_max2d.py data/EgoMax2D/01KW3RYY4Z67HKAVYMAG8QG9WH \
      --cam left --frames 0:150 --video --out analysis/vis_max2d_out
"""
import argparse
import os

import cv2
import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

KPS = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "pelvis"]
# BGR colors: left=orange-ish, right=cyan-ish, pelvis=magenta
KP_COLOR = {
    "left_elbow": (0, 128, 255),
    "left_shoulder": (0, 200, 255),
    "right_elbow": (255, 128, 0),
    "right_shoulder": (255, 200, 0),
    "pelvis": (255, 0, 255),
}
BONES = [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")]
CAM_KEY = {"left": "head_front_left", "right": "head_front_right"}
CAM_DIR = {"left": "head-front-left", "right": "head-front-right"}


def parse_frames(spec, n):
    """'0:300:30' / '0:150' / '12,40,99' / 'all' -> list of frame indices."""
    if spec == "all":
        return list(range(n))
    if ":" in spec:
        parts = [int(p) if p else None for p in spec.split(":")]
        start = parts[0] or 0
        stop = parts[1] if len(parts) > 1 and parts[1] is not None else n
        step = parts[2] if len(parts) > 2 and parts[2] is not None else 1
        return list(range(start, min(stop, n), step))
    return [int(p) for p in spec.split(",") if int(p) < n]


def draw_cam(session_dir, cam, ann, files, idx, scale):
    img = cv2.imread(os.path.join(session_dir, "images", CAM_DIR[cam], files[idx]))
    if img is None:
        raise FileNotFoundError(files[idx])
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    frame = ann["frames"]["%06d" % idx]
    pts = {}
    missing = []
    for kp in KPS:
        e = frame[kp]
        if e["pixel_coords"] is None:
            missing.append("%s:%s" % (kp, e.get("status")))
            continue
        x, y = e["pixel_coords"]
        pts[kp] = (int(round(x * scale)), int(round(y * scale)))

    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)

    r = max(4, int(10 * scale))
    for kp, p in pts.items():
        e = frame[kp]
        conf = e["confidence_score"] or 0
        color = KP_COLOR[kp]
        if e.get("status") == "manual_annotated":  # second-pass human fix
            cv2.rectangle(img, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, -1)
            cv2.putText(img, "M", (p[0] + r + 2, p[1] + r), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
        elif e.get("status") == "occluded":  # human-annotated but occluded
            cv2.drawMarker(img, p, color, cv2.MARKER_TRIANGLE_UP, 2 * r, 2)
            cv2.putText(img, "occ", (p[0] + r + 2, p[1] + r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        elif conf > 0:  # first-pass human keyframe annotation
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
            cv2.putText(img, "%.2f" % conf, (p[0] + r + 2, p[1] + r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:  # interpolated between keyframes
            cv2.circle(img, p, r, color, 2, cv2.LINE_AA)

    header = "%s  frame %06d (%s)" % (cam, idx, files[idx])
    cv2.putText(img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    if missing:
        cv2.putText(img, "missing: " + " ".join(missing), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def add_legend(img):
    y = img.shape[0] - 15
    cv2.putText(img, "solid=keyframe-anno  hollow=interp  square=manual-fix  triangle=occluded",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session directory (data/EgoMax2D/<ULID>)")
    ap.add_argument("--cam", choices=["left", "right", "both"], default="both")
    ap.add_argument("--frames", default="0::30",
                    help="'start:stop:step' | comma list | 'all' (default: every 30th)")
    ap.add_argument("--scale", type=float, default=0.5, help="output scale (default 0.5)")
    ap.add_argument("--out", default="analysis/vis_max2d_out")
    ap.add_argument("--video", action="store_true", help="also write an mp4 at 10 fps")
    args = ap.parse_args()

    session_dir = args.session.rstrip("/")
    sid = os.path.basename(session_dir)
    print("loading", os.path.join(session_dir, "estimations.toon"), "...")
    with open(os.path.join(session_dir, "estimations.toon")) as f:
        toon = yaml.load(f, Loader=_Loader)

    cams = ["left", "right"] if args.cam == "both" else [args.cam]
    files = {c: sorted(os.listdir(os.path.join(session_dir, "images", CAM_DIR[c])))
             for c in cams}
    n = len(files[cams[0]])
    idxs = parse_frames(args.frames, n)
    out_dir = os.path.join(args.out, sid)
    os.makedirs(out_dir, exist_ok=True)

    writer = None
    for i, idx in enumerate(idxs):
        panels = [draw_cam(session_dir, c, toon[CAM_KEY[c]], files[c], idx, args.scale)
                  for c in cams]
        canvas = add_legend(np.hstack(panels) if len(panels) > 1 else panels[0])
        out_path = os.path.join(out_dir, "vis_%06d.jpg" % idx)
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.video:
            if writer is None:
                vp = os.path.join(out_dir, "vis.mp4")
                writer = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        if (i + 1) % 20 == 0 or i + 1 == len(idxs):
            print("  %d/%d frames" % (i + 1, len(idxs)))
    if writer is not None:
        writer.release()
        print("video:", os.path.join(out_dir, "vis.mp4"))
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
