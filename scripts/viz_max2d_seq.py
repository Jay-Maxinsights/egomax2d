#!/usr/bin/env python3
"""Visualize EgoMax2D sequences as videos with keypoints drawn as big green dots.

Sessions are sorted by name; pick them by sorted index with --sessions
("0-2" = first three, "5" = single, "all"). Frames are sampled every 3rd
frame (0, 3, 6, ... — the human-annotation keyframe grid). Left and right
cameras are rendered side by side. One mp4 per session is written to
demo/viz_max_2d/<session_id>.mp4.

Usage:
  python3 scripts/viz_max2d_seq.py --sessions 0-2
  python3 scripts/viz_max2d_seq.py --sessions all --step 3 --scale 0.5
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

KPS = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "pelvis"]
GREEN = (0, 255, 0)
CAMS = [("head_front_left", "head-front-left"), ("head_front_right", "head-front-right")]


def parse_sessions(spec, n):
    if spec == "all":
        return list(range(n))
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), min(int(b), n - 1) + 1))
    return [int(spec)]


def draw_panel(session_dir, cam_dir, ann, idx, scale, radius):
    img = cv2.imread(os.path.join(session_dir, "images", cam_dir, "frame_%08d.jpg" % idx))
    if img is None:
        raise FileNotFoundError("%s/%s frame_%08d.jpg" % (session_dir, cam_dir, idx))
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame = ann["frames"]["%06d" % idx]
    for kp in KPS:
        coords = frame[kp]["pixel_coords"]
        if coords is None:
            continue
        p = (int(round(coords[0] * scale)), int(round(coords[1] * scale)))
        cv2.circle(img, p, radius, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, (0, 0, 0), 2, cv2.LINE_AA)  # outline for contrast
    cv2.putText(img, "%s  frame %06d" % (cam_dir, idx), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render_session(root, sid, out_dir, step, scale, radius, fps):
    session_dir = os.path.join(root, sid)
    with open(os.path.join(session_dir, "estimations.toon")) as f:
        toon = yaml.load(f, Loader=_Loader)
    n = toon[CAMS[0][0]]["metadata"]["number_of_frames"]
    idxs = range(0, n, step)

    out_path = os.path.join(out_dir, "%s.mp4" % sid)
    # Encode H.264 via an ffmpeg pipe: OpenCV's own mp4v fourcc is MPEG-4
    # Part 2, which browsers and most players refuse to play. Dimensions are
    # floored to a multiple of 16 (not just 2) so libx264 never needs an
    # internal macroblock-crop SEI, and profile/level/crf are pinned to
    # widely-supported values — VSCode's built-in video preview (esp. over
    # Remote-SSH) buffers the whole file into the webview and silently fails
    # above roughly 100MB, so file size is kept small deliberately.
    proc = None
    for idx in idxs:
        panels = [draw_panel(session_dir, cam_dir, toon[cam_key], idx, scale, radius)
                  for cam_key, cam_dir in CAMS]
        canvas = np.hstack(panels)
        h, w = canvas.shape[:2]
        canvas = canvas[: h - h % 16, : w - w % 16]
        if proc is None:
            h, w = canvas.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
                 "-r", "%g" % fps, "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "28",
                 "-profile:v", "main", "-level", "4.0",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                stdin=subprocess.PIPE)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed for %s" % out_path)
    return out_path, len(list(idxs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/EgoMax2D")
    ap.add_argument("--sessions", default="0-2",
                    help="sorted-index range '0-2', single '5', or 'all' (default 0-2)")
    ap.add_argument("--step", type=int, default=3, help="frame sampling step (default 3)")
    ap.add_argument("--scale", type=float, default=0.35,
                    help="render scale (default 0.35 — keeps mp4 small enough for "
                         "VSCode's built-in preview to load over Remote-SSH)")
    ap.add_argument("--radius", type=int, default=14,
                    help="dot radius in output pixels (default 14)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="video fps (default 10 = real-time for step 3 @ 30fps)")
    ap.add_argument("--out", default="demo/viz_max_2d")
    args = ap.parse_args()

    all_sessions = sorted(d for d in os.listdir(args.root)
                          if os.path.isdir(os.path.join(args.root, d)))
    picks = parse_sessions(args.sessions, len(all_sessions))
    os.makedirs(args.out, exist_ok=True)

    for i in picks:
        sid = all_sessions[i]
        print("[%d/%d] session %d: %s ..." % (picks.index(i) + 1, len(picks), i, sid),
              flush=True)
        path, nf = render_session(args.root, sid, args.out, args.step,
                                  args.scale, args.radius, args.fps)
        print("  -> %s (%d frames)" % (path, nf))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
