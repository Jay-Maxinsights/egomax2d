#!/usr/bin/env python3
"""Savitzky-Golay smoothing of the EgoMax2D pelvis 2D trajectory, with a
before/after comparison video.

Each camera's pelvis pixel trajectory (x(t), y(t)) is treated as an
independent time series over the full per-frame sequence (not just the
every-3rd-frame human keyframes — the toon file already fills the frames
in between via its own linear interpolation, which is exactly the source
of the jitter/zigzag noted in analysis/max2d.md's TODO). Missing frames
(out_of_frame) are linearly filled first so savgol_filter has a
continuous signal, then filtered.

Output: a 2x3 comparison video —
  top row    : RAW    left | RAW    right | RAW    trajectory (Cartesian)
  bottom row : SAVGOL left | SAVGOL right | SAVGOL trajectory (Cartesian)
— sampled every --step frames (default 3, the annotation keyframe grid),
written to demo/viz_max_2d/<session_id>_pelvis_savgol_compare.mp4.

The image panels draw the pelvis path so far as a trail (left camera in
blue, right camera in orange, matching the trajectory-plot column) plus
the current point as a big green dot. The 3rd column is a real Cartesian
(x, y in pixels) plot of both cameras' paths so far, y-axis inverted to
match image coordinates.

Only pelvis gets the trail + Savitzky-Golay treatment. The other 4
keypoints (elbows, shoulders) are also read from the toon file and drawn
as small static per-frame dots + shoulder-elbow bones for context — no
trail, no filtering.

Usage:
  python3 scripts/pelvis_savgol_compare.py                 # first session, defaults
  python3 scripts/pelvis_savgol_compare.py --session 01KW409V4KA9CMN8BT9KR0Q664
  python3 scripts/pelvis_savgol_compare.py --window 21 --polyorder 3
"""
import argparse
import os
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import savgol_filter

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

GREEN = (0, 255, 0)
CAMS = [("head_front_left", "head-front-left"), ("head_front_right", "head-front-right")]
# BGR (cv2) / matplotlib-name pairs, shared between the image trails and the plot column
LEFT_BGR, LEFT_MPL = (180, 119, 31), "tab:blue"
RIGHT_BGR, RIGHT_MPL = (14, 127, 255), "tab:orange"

# The other 4 keypoints are drawn as small static (unfiltered) context dots —
# only pelvis gets the trail + Savitzky-Golay treatment. Colors match
# scripts/vis_max2d.py so the two visualizations read consistently.
OTHER_KPS = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]
KP_COLOR = {
    "left_elbow": (0, 128, 255),
    "left_shoulder": (0, 200, 255),
    "right_elbow": (255, 128, 0),
    "right_shoulder": (255, 200, 0),
}
BONES = [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")]


def resolve_session(root, spec):
    sessions = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if spec.isdigit():
        return sessions[int(spec)]
    if spec not in sessions:
        raise ValueError("session %r not found under %s" % (spec, root))
    return spec


def load_kp_xy(toon, cam_key, n, kp_name):
    """Return (n, 2) float array, NaN where pixel_coords is missing."""
    xy = np.full((n, 2), np.nan, dtype=np.float64)
    frames = toon[cam_key]["frames"]
    for i in range(n):
        coords = frames["%06d" % i][kp_name]["pixel_coords"]
        if coords is not None:
            xy[i] = coords
    return xy


def fill_and_smooth(xy, window, polyorder):
    """Linearly fill NaN gaps, then apply Savitzky-Golay per axis."""
    n = xy.shape[0]
    t = np.arange(n)
    filled = xy.copy()
    for c in range(2):
        valid = ~np.isnan(xy[:, c])
        if valid.sum() < 2:
            raise ValueError("not enough valid pelvis points to filter (axis %d)" % c)
        filled[:, c] = np.interp(t, t[valid], xy[valid, c])
    win = min(window, n if n % 2 == 1 else n - 1)
    win = max(win, polyorder + 1 + (polyorder % 2 == 0))
    if win % 2 == 0:
        win += 1
    smoothed = np.stack(
        [savgol_filter(filled[:, c], win, polyorder, mode="interp") for c in range(2)],
        axis=1,
    )
    return filled, smoothed, win


def jitter_stats(xy):
    """Mean/max frame-to-frame displacement (px), ignoring NaN-adjacent steps."""
    d = np.diff(xy, axis=0)
    dist = np.linalg.norm(d, axis=1)
    dist = dist[~np.isnan(dist)]
    return dist.mean(), dist.max()


def draw_trail(img, xy_full, idx, scale, color):
    """Polyline of the path from frame 0..idx, skipping gaps where xy is NaN."""
    pts = xy_full[: idx + 1]
    valid = ~np.isnan(pts).any(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return
    breaks = np.where(np.diff(valid_idx) > 1)[0]
    runs = np.split(valid_idx, breaks + 1)
    for run in runs:
        if len(run) < 2:
            continue
        seg = (pts[run] * scale).round().astype(np.int32)
        cv2.polylines(img, [seg], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_other_kps(img, other_xy, idx, scale):
    """Static (unfiltered, no trail) context dots for elbows/shoulders + bones."""
    pts = {}
    for kp in OTHER_KPS:
        xy = other_xy[kp][idx]
        if not np.isnan(xy).any():
            pts[kp] = (int(round(xy[0] * scale)), int(round(xy[1] * scale)))
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
    r = 6
    for kp, p in pts.items():
        cv2.circle(img, p, r, KP_COLOR[kp], -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (0, 0, 0), 1, cv2.LINE_AA)


def draw_panel(session_dir, cam_dir, idx, scale, radius, panel_wh, trail_xy, trail_color,
              other_xy, label):
    img = cv2.imread(os.path.join(session_dir, "images", cam_dir, "frame_%08d.jpg" % idx))
    if img is None:
        raise FileNotFoundError("%s/%s frame_%08d.jpg" % (session_dir, cam_dir, idx))
    img = cv2.resize(img, panel_wh, interpolation=cv2.INTER_AREA)
    draw_other_kps(img, other_xy, idx, scale)
    draw_trail(img, trail_xy, idx, scale, trail_color)  # pelvis only: trail + big dot
    point = trail_xy[idx]
    if not np.isnan(point).any():
        p = (int(round(point[0] * scale)), int(round(point[1] * scale)))
        cv2.circle(img, p, radius, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "%s  %s  frame %06d" % (label, cam_dir, idx), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class TrajPlot:
    """Persistent matplotlib figure rendering the growing (x, y) path of both
    cameras on a real Cartesian axis (ticks, grid, inverted y to match image
    coordinates), reused across frames for speed."""

    def __init__(self, panel_wh, xlim, ylim, dpi=100):
        w, h = panel_wh
        self.panel_wh = panel_wh
        self.fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(111)
        (self.line_l,) = self.ax.plot([], [], "-", color=LEFT_MPL, lw=1.3, label="left")
        (self.line_r,) = self.ax.plot([], [], "-", color=RIGHT_MPL, lw=1.3, label="right")
        self.pt_l = self.ax.scatter([], [], color=LEFT_MPL, s=35, zorder=5, edgecolors="k")
        self.pt_r = self.ax.scatter([], [], color=RIGHT_MPL, s=35, zorder=5, edgecolors="k")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.invert_yaxis()  # image-style: y grows downward
        self.ax.set_xlabel("x (px)")
        self.ax.set_ylabel("y (px)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout()

    def render(self, idx, left_xy, right_xy, title):
        self.line_l.set_data(left_xy[: idx + 1, 0], left_xy[: idx + 1, 1])
        self.line_r.set_data(right_xy[: idx + 1, 0], right_xy[: idx + 1, 1])
        self.pt_l.set_offsets(left_xy[idx : idx + 1])
        self.pt_r.set_offsets(right_xy[idx : idx + 1])
        self.ax.set_title(title, fontsize=9)
        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())
        img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        w, h = self.panel_wh
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/EgoMax2D")
    ap.add_argument("--session", default="0",
                    help="sorted index (default '0' = first session) or session ULID")
    ap.add_argument("--window", type=int, default=15,
                    help="Savitzky-Golay window length in frames, odd (default 15 = 0.5s @30fps)")
    ap.add_argument("--polyorder", type=int, default=2, help="polynomial order (default 2)")
    ap.add_argument("--step", type=int, default=3, help="frame sampling step for video (default 3)")
    ap.add_argument("--scale", type=float, default=0.35,
                    help="render scale (default 0.35, keeps mp4 small for VSCode preview)")
    ap.add_argument("--radius", type=int, default=12, help="dot radius (default 12)")
    ap.add_argument("--fps", type=float, default=10.0, help="output video fps (default 10)")
    ap.add_argument("--out", default="demo/viz_max_2d")
    args = ap.parse_args()

    sid = resolve_session(args.root, args.session)
    session_dir = os.path.join(args.root, sid)
    print("session:", sid)
    with open(os.path.join(session_dir, "estimations.toon")) as f:
        toon = yaml.load(f, Loader=_Loader)
    n = toon[CAMS[0][0]]["metadata"]["number_of_frames"]

    raw, smoothed, other_xy, win_used = {}, {}, {}, None
    for cam_key, cam_dir in CAMS:
        xy = load_kp_xy(toon, cam_key, n, "pelvis")
        filled, sm, win_used = fill_and_smooth(xy, args.window, args.polyorder)
        raw[cam_dir] = xy
        smoothed[cam_dir] = sm
        raw_mean, raw_max = jitter_stats(filled)
        sm_mean, sm_max = jitter_stats(sm)
        n_missing = np.isnan(xy).any(axis=1).sum()
        print("  %-18s missing=%d/%d  jitter(px) raw mean=%.2f max=%.2f -> "
              "savgol mean=%.2f max=%.2f  (window=%d, polyorder=%d)"
              % (cam_dir, n_missing, n, raw_mean, raw_max, sm_mean, sm_max,
                 win_used, args.polyorder))
        other_xy[cam_dir] = {kp: load_kp_xy(toon, cam_key, n, kp) for kp in OTHER_KPS}

    raw_w, raw_h = int(2592 * args.scale), int(1944 * args.scale)
    panel_wh = (raw_w - raw_w % 16, raw_h - raw_h % 16)

    all_xy = np.concatenate([raw["head-front-left"], raw["head-front-right"],
                             smoothed["head-front-left"], smoothed["head-front-right"]])
    xpad = 0.05 * np.nanmax(all_xy[:, 0])
    ypad = 0.05 * np.nanmax(all_xy[:, 1])
    xlim = (np.nanmin(all_xy[:, 0]) - xpad, np.nanmax(all_xy[:, 0]) + xpad)
    ylim = (np.nanmin(all_xy[:, 1]) - ypad, np.nanmax(all_xy[:, 1]) + ypad)
    raw_plot = TrajPlot(panel_wh, xlim, ylim)
    savgol_plot = TrajPlot(panel_wh, xlim, ylim)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "%s_pelvis_savgol_compare.mp4" % sid)
    proc = None
    for idx in range(0, n, args.step):
        top = np.hstack([
            draw_panel(session_dir, "head-front-left", idx, args.scale, args.radius, panel_wh,
                      raw["head-front-left"], LEFT_BGR, other_xy["head-front-left"], "RAW"),
            draw_panel(session_dir, "head-front-right", idx, args.scale, args.radius, panel_wh,
                      raw["head-front-right"], RIGHT_BGR, other_xy["head-front-right"], "RAW"),
            raw_plot.render(idx, raw["head-front-left"], raw["head-front-right"],
                            "RAW pelvis trajectory  frame %06d" % idx),
        ])
        bottom = np.hstack([
            draw_panel(session_dir, "head-front-left", idx, args.scale, args.radius, panel_wh,
                      smoothed["head-front-left"], LEFT_BGR, other_xy["head-front-left"],
                      "SAVGOL w=%d" % win_used),
            draw_panel(session_dir, "head-front-right", idx, args.scale, args.radius, panel_wh,
                      smoothed["head-front-right"], RIGHT_BGR, other_xy["head-front-right"],
                      "SAVGOL w=%d" % win_used),
            savgol_plot.render(idx, smoothed["head-front-left"], smoothed["head-front-right"],
                               "SAVGOL pelvis trajectory  frame %06d" % idx),
        ])
        canvas = np.vstack([top, bottom])
        h, w = canvas.shape[:2]
        canvas = canvas[: h - h % 16, : w - w % 16]
        if proc is None:
            h, w = canvas.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
                 "-r", "%g" % args.fps, "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "28",
                 "-profile:v", "main", "-level", "4.0",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                stdin=subprocess.PIPE)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed for %s" % out_path)
    plt.close(raw_plot.fig)
    plt.close(savgol_plot.fig)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
