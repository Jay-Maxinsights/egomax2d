# Calibration-aware preprocessing for EgoMax2D: ray-level remap of the raw
# 2592x1944 Double Sphere fisheye frames into the EgoBody3M training-camera
# geometry (left cam -> cam1, right cam -> cam2), so that a model trained on
# EgoBody3M sees the same imaging geometry.
#
# Chain per target pixel (256 canvas):
#   canvas px -> EgoBody3M native px (undo pad+resize)
#     -> unproject with the repo-estimated EgoBody3M DS params
#     -> rotate the ray 90 deg about the optical axis ("right" = CCW image
#        rotation; EgoBody3M images have the person horizontal, EgoMax2D raw
#        frames are upright)
#     -> project with the session's own DS calibration (calibration.json,
#        differs per session — 20 distinct rigs) -> sample the raw frame.
#
# GT keypoints go through the same chain (remap_gt_point) so overlays and
# training targets stay aligned with the remapped images.
#
# Validated in scripts/_debug_remap_egomax2d.py (2026-07-12).

import json
import os

import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:
    from yaml import SafeLoader as _YamlLoader

from pose_estimation.models.utils.camera_models import _EGOBODY3M_DS_PARAMS

IMG_SIZE = 256

# EgoMax2D's 5 annotated keypoints -> EgoBody3M 26J channel indices.
KPS = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "pelvis"]
KP2JOINT = {
    "left_elbow": 3, "right_elbow": 7,
    "left_shoulder": 2, "right_shoulder": 6,
    "pelvis": 10,
}
KP_JOINT_IDS = tuple(KP2JOINT[k] for k in KPS)  # (3, 7, 2, 6, 10)

# EgoMax2D camera name per side, and the EgoBody3M camera id it maps onto.
SIDE_SPECS = {
    "left": dict(cam_name="videoFL", cam_dir="head-front-left",
                 toon_key="head_front_left", eb_cam=1),
    "right": dict(cam_name="videoFR", cam_dir="head-front-right",
                  toon_key="head_front_right", eb_cam=2),
}


# ── Double Sphere camera math (Usenko et al.), numpy ──────────────────────────

def ds_unproject(u, v, fx, fy, cx, cy, xi, alpha):
    """Pixel -> unit-scale ray in camera frame (x right, y down, z forward)."""
    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx * mx + my * my
    valid = np.ones_like(mx, dtype=bool)
    if alpha > 0.5:
        valid &= r2 <= 1.0 / (2 * alpha - 1)
    tmp = 1.0 - (2 * alpha - 1) * r2
    valid &= tmp >= 0
    mz = (1 - alpha * alpha * r2) / (alpha * np.sqrt(np.clip(tmp, 0, None)) + 1 - alpha)
    s = mz * mz + (1 - xi * xi) * r2
    valid &= s >= 0
    coef = (mz * xi + np.sqrt(np.clip(s, 0, None))) / (mz * mz + r2)
    return coef * mx, coef * my, coef * mz - xi, valid


def ds_project(x, y, z, fx, fy, cx, cy, xi, alpha):
    """Ray -> pixel."""
    d1 = np.sqrt(x * x + y * y + z * z)
    zxi = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zxi * zxi)
    denom = alpha * d2 + (1 - alpha) * zxi
    w1 = alpha / (1 - alpha) if alpha <= 0.5 else (1 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1)
    valid = (z > -w2 * d1) & (denom > 1e-6)
    safe = np.where(valid, denom, 1.0)
    return fx * x / safe + cx, fy * y / safe + cy, valid


def _rotate_ray(x, y, z, rotate, inverse=False):
    """90-deg rotation about the optical axis, EgoBody3M frame <-> raw frame.

    rotate="right" corresponds to a CCW image rotation (raw upright person ->
    horizontal, matching EgoBody3M's camera mount). Derivation: a CCW image
    rotation maps raw pixel offsets (dx, dy) -> (dy, -dx), so a ray (x, y, z)
    in the rotated (EgoBody3M-like) frame is (-y, x, z) in the raw frame.
    """
    if rotate == "none":
        return x, y, z
    if (rotate == "right") ^ inverse:
        return -y, x, z
    return y, -x, z


def load_session_calib(session_dir: str) -> dict:
    """Per-session DS intrinsics for videoFL/videoFR (calibrations differ
    across sessions — always load per session, never share globally)."""
    cal = json.load(open(os.path.join(session_dir, "calibration.json")))
    y = yaml.load(cal["calibrations"][0]["calibration_text"], Loader=_YamlLoader)
    out = {}
    for item in y["intrinsics"]:
        if item["camera_name"] in ("videoFL", "videoFR"):
            fx, fy, cx, cy, xi, alpha = item["params"]
            out[item["camera_name"]] = dict(fx=fx, fy=fy, cx=cx, cy=cy, xi=xi, alpha=alpha)
    return out


def build_remap(eb_params: dict, src_params: dict, rotate: str = "right"):
    """cv2.remap maps pulling raw EgoMax2D pixels onto the EgoBody3M 256
    canvas: canvas px -> native px -> EgoBody3M ray -> rotate -> raw px."""
    scale = eb_params["square"] / IMG_SIZE
    u256, v256 = np.meshgrid(np.arange(IMG_SIZE, dtype=np.float64),
                             np.arange(IMG_SIZE, dtype=np.float64))
    u_nat = u256 * scale
    v_nat = v256 * scale - eb_params["pad_top"]
    x, y, z, val1 = ds_unproject(u_nat, v_nat, eb_params["fx"], eb_params["fy"],
                                 eb_params["cx"], eb_params["cy"],
                                 eb_params["xi"], eb_params["alpha"])
    x, y, z = _rotate_ray(x, y, z, rotate)
    us, vs, val2 = ds_project(x, y, z, **src_params)
    valid = val1 & val2 & (v_nat >= 0) & (v_nat < eb_params["native_wh"][1])
    map_x = np.where(valid, us, -1).astype(np.float32)
    map_y = np.where(valid, vs, -1).astype(np.float32)
    return map_x, map_y


def build_session_remaps(session_dir: str, rotate: str = "right") -> dict:
    """{'left': (map_x, map_y), 'right': ...} for one session."""
    calib = load_session_calib(session_dir)
    return {
        side: build_remap(dict(_EGOBODY3M_DS_PARAMS[spec["eb_cam"]]),
                          calib[spec["cam_name"]], rotate)
        for side, spec in SIDE_SPECS.items()
    }


def remap_gt_point(x: float, y: float, src_params: dict, eb_params: dict,
                   rotate: str = "right"):
    """Raw GT px through the same chain: src ray -> un-rotate -> EgoBody3M
    native px -> 256 canvas. Returns None if outside either model's domain."""
    rx, ry, rz, v1 = ds_unproject(np.array([x], dtype=np.float64),
                                  np.array([y], dtype=np.float64), **src_params)
    rx, ry, rz = _rotate_ray(rx, ry, rz, rotate, inverse=True)
    u, v, v2 = ds_project(rx, ry, rz, eb_params["fx"], eb_params["fy"],
                          eb_params["cx"], eb_params["cy"],
                          eb_params["xi"], eb_params["alpha"])
    if not (v1[0] and v2[0]):
        return None
    scale = IMG_SIZE / eb_params["square"]
    return float(u[0]) * scale, (float(v[0]) + eb_params["pad_top"]) * scale
