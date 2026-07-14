# Author: Lu Dong 
# Time: 2026-07-13

import torch


def unrealego_proj(local_3d, local_origin):
    num_views = 2
    polynomial_w2c = (
        541.084422, 133.996745, -53.833198, 60.96083, -24.78051, 12.451492,
        -30.240511, 26.90122, 116.38499, -133.991117, -141.904687, 184.05592,
        107.45616, -125.552875, -55.66342, 44.209519, 18.234651, -6.410899, -2.737066
    )
    image_center = (511.1183388444314, 510.8730105600536)
    raw_image_size = (1024, 1024)

    with torch.no_grad():
        cam_3d = local_3d[:, None].repeat(1, num_views, 1, 1)  # [B, V, J, 3]
        cam_3d = cam_3d + local_origin

        x = cam_3d[..., 0]  # [B, V, J]
        y = cam_3d[..., 1]  # [B, V, J]
        z = cam_3d[..., 2]  # [B, V, J]

        norm = torch.sqrt(x * x + y * y)
        theta = torch.atan(-z / norm)

        rho = sum(a * theta ** i for i, a in enumerate(polynomial_w2c))

        u = x / norm * rho + image_center[0]
        v = y / norm * rho + image_center[1]

        u = u / raw_image_size[1]
        v = v / raw_image_size[0]

        image_coor_2d = torch.stack((u, v), dim=-1)  # [B, V, J, 2]
        in_fov = (
                (image_coor_2d[..., 0] > 0)
                & (image_coor_2d[..., 1] > 0)
                & (image_coor_2d[..., 0] < 1)
                & (image_coor_2d[..., 1] < 1)
        )  # [B, V, J]
        image_coor_2d = image_coor_2d.clamp(min=0.0, max=1.0)
        return image_coor_2d, in_fov


# ---------------------------------------------------------------------------
# Real camera parameters from 377d6e19bc5e6f0a.json (EgoCapture2 device)
# Scaramuzza / OCamCalib model.
#
# Projection convention (verified by round-trip against taylor_coefficient):
#   theta = atan2(r, z),  r = sqrt(x^2 + y^2),  z = depth toward body
#   rho   = polyval(inverse_poly, theta)   <- MATLAB descending-order polyval
#   u = x/r * rho + cx,  v = y/r * rho + cy
#   u_norm = u / W,  v_norm = v / H
# ---------------------------------------------------------------------------

# videoFL — left fisheye
_INV_POLY_L = (
    0.6753890702965489, -16.268338244436492, 179.6435030191337,
    -1202.9961885562325, 5444.819577872112, -17568.36926237116,
    41532.873144497236, -72808.98790985157, 94781.37945207143,
    -91013.26542095553, 63600.76086895691, -31608.8436190213,
    10611.098359317422, -2078.079617811596, 135.27377840345648,
    -23.136354349343293, 879.6705045602605, -0.0016147660601741293,
)
_CX_L, _CY_L = 1241.89453125, 965.35546875

# videoFR — right fisheye
_INV_POLY_R = (
    0.6711705262016818, -16.010696994584972, 174.94599098387053,
    -1158.3940635676615, 5180.818312184162, -16510.199747904964,
    38536.50083884841, -66689.98794401475, 85710.5627207632,
    -81290.595373141, 56159.71684902892, -27632.0497295537,
    9194.69242929093, -1786.4166097362768, 114.12333759733202,
    -19.709481516225527, 881.4354662254741, -0.0013596505097264386,
)
_CX_R, _CY_R = 1239.99609375, 967.25390625

_SENSOR_W, _SENSOR_H = 2592, 1944  # both cameras same sensor size

# Per-view calibration ordered by view index (0-based), matching ego_cams=(1,2,...) convention.
#
# Physical camera mapping (confirmed from 377d6e19bc5e6f0a.json calibration + image sizes):
#   image file cam0 → videoL  (left side,  480×636,  DS model — needs real calibration)
#   image file cam1 → videoFL (front-left,  1024×1280, OCamCalib — _INV_POLY_L, cx=1241.89)
#   image file cam2 → videoFR (front-right, 1024×1280, OCamCalib — _INV_POLY_R, cx=1240.00)
#   image file cam3 → videoR  (right side,  480×636,  DS model — needs real calibration)
#
# This table is indexed by *view index* = position in ego_cams tuple:
#   ego_cams=(1,2)     → view 0=cam1(FL), view 1=cam2(FR)
#   ego_cams=(0,1,2,3) → view 0=cam0(L),  view 1=cam1(FL), view 2=cam2(FR), view 3=cam3(R)
#                         ← requires reordering entries below and adding cam0/cam3 intrinsics
#
# Entries 2 & 3 below are OCamCalib placeholders for cam0/cam3 until their DS intrinsics
# are added and ego_cams is switched to (0,1,2,3).
_ORDERED_CAM_PARAMS = [
    (_INV_POLY_L, _CX_L, _CY_L, _SENSOR_W, _SENSOR_H),  # view 0 → cam1 (videoFL, front-left fisheye)
    (_INV_POLY_R, _CX_R, _CY_R, _SENSOR_W, _SENSOR_H),  # view 1 → cam2 (videoFR, front-right fisheye)
    (_INV_POLY_L, _CX_L, _CY_L, _SENSOR_W, _SENSOR_H),  # view 2 → placeholder (cam0/videoL, left side)
    (_INV_POLY_R, _CX_R, _CY_R, _SENSOR_W, _SENSOR_H),  # view 3 → placeholder (cam3/videoR, right side)
]


def _polyval_torch(coeffs_tuple, x):
    """MATLAB-style polyval: coeffs_tuple[0] is highest-degree coefficient."""
    result = torch.zeros_like(x)
    for c in coeffs_tuple:
        result = result * x + c
    return result


def real_egocam_proj(local_3d, local_origin):
    """Multi-view fisheye projection using real EgoCapture2 camera calibration.

    Args:
        local_3d:    [B, J, 3]     joint positions (pelvis-relative, in cm)
        local_origin:[B, V, 1, 3]  pelvis position in each camera frame (cm);
                                   V is inferred from this tensor — supports 2 or 4 views.

    Returns:
        image_coor_2d: [B, V, J, 2]  normalised u/v in [0, 1]
        in_fov:        [B, V, J]     bool mask — True if the joint is inside FOV
    """
    with torch.no_grad():
        V = local_origin.shape[1]
        assert V <= len(_ORDERED_CAM_PARAMS), \
            f"real_egocam_proj supports up to {len(_ORDERED_CAM_PARAMS)} views, got {V}"

        cam_3d = local_3d[:, None].repeat(1, V, 1, 1)  # [B, V, J, 3]
        cam_3d = cam_3d + local_origin  # shift to each camera frame

        x = cam_3d[..., 0]   # [B, V, J]
        y = cam_3d[..., 1]
        z = cam_3d[..., 2]   # positive = toward body (depth axis)

        r = torch.sqrt(x * x + y * y).clamp(min=1e-6)
        theta = torch.atan2(r, z)  # angle from optical axis; 0 = straight ahead

        # Per-view polynomial evaluation
        rho_per_view = []
        cx_vals, cy_vals, w_vals, h_vals = [], [], [], []
        for vi in range(V):
            inv_poly, cx, cy, w, h = _ORDERED_CAM_PARAMS[vi]
            rho_per_view.append(_polyval_torch(inv_poly, theta[:, vi]))  # [B, J]
            cx_vals.append(cx)
            cy_vals.append(cy)
            w_vals.append(w)
            h_vals.append(h)

        rho = torch.stack(rho_per_view, dim=1)  # [B, V, J]
        cx = torch.tensor(cx_vals, device=x.device, dtype=x.dtype)[None, :, None]  # [1, V, 1]
        cy = torch.tensor(cy_vals, device=x.device, dtype=x.dtype)[None, :, None]
        w_t = torch.tensor(w_vals, device=x.device, dtype=x.dtype)[None, :, None]
        h_t = torch.tensor(h_vals, device=x.device, dtype=x.dtype)[None, :, None]

        u = x / r * rho + cx   # [B, V, J]
        v_coord = y / r * rho + cy

        u_norm = u / w_t
        v_norm = v_coord / h_t

        image_coor_2d = torch.stack((u_norm, v_norm), dim=-1)  # [B, V, J, 2]
        in_fov = (
            (u_norm > 0) & (u_norm < 1)
            & (v_norm > 0) & (v_norm < 1)
            & (z > 0)   # only points in front of the camera
        )
        # NOTE: views whose _ORDERED_CAM_PARAMS entry is a placeholder (e.g. cam0/cam3 before
        # real calibration is added) will produce u_norm/v_norm far outside [0,1] because
        # cx/cy and sensor size are wrong.  As a result in_fov is all-False for those views,
        # and EgoPoseFormerTransformerLayer._run_cross_attn zero-masks their attention output
        # (anchors_valid[:, i] == False → masked_fill → 0).  The model silently degrades to
        # using only the correctly-calibrated views.  Replace placeholders with real intrinsics
        # to actually benefit from those cameras.
        image_coor_2d = image_coor_2d.clamp(min=0.0, max=1.0)
        return image_coor_2d, in_fov


# ---------------------------------------------------------------------------
# EgoBody3M calibration estimated from metadata 2D/3D GT.
#
# Source command:
#   python scripts/estimate_egobody3m_calibration.py \
#     --data-root /mnt/nas-q/LuDong/EgoBody3M/dataset \
#     --splits train validation test --cams 0 1 2 3 --max-seqs 170 \
#     --frame-stride 50 --max-points-per-cam 80000 --max-nfev 600 \
#     --fix-principal-point --refine-threshold-px 0 \
#     --out output/egobody3m_camera_calibration_auto_170seq.json
#
# Model:
#   Double Sphere projection, with a static camera_from_headset transform:
#       X_cam_cm = R_cam_from_headset @ X_headset_cm + t_cam_from_headset_cm
#
# Validation on the sampled 510-sequence subset:
#   cam0 RMS=6.17px, cam1 RMS=10.79px, cam2 RMS=11.39px, cam3 RMS=6.46px
# ---------------------------------------------------------------------------

_EGOBODY3M_DS_PARAMS = {
    0: {
        "fx": 700.0514895783737,
        "fy": 697.4069466140407,
        "cx": 317.99999999900007,
        "cy": 240.00000000099996,
        "xi": 1.888097142190858,
        "alpha": 5.484759957524661e-13,
        "R": (
            (-0.4805604901946838, -0.8686731923500686, 0.12028507869302019),
            (-0.015113148994583103, -0.1289373193484361, -0.9915375738753962),
            (0.8768313452376761, -0.47831166886412063, 0.048833794006886144),
        ),
        "t": (1.226944111056151, 3.337440293333688, -8.131303118114328),
        "native_wh": (636.0, 480.0),
        "square": 636.0,
        "pad_top": 78.0,
    },
    1: {
        "fx": 2097.817416582966,
        "fy": 2074.5274148037047,
        "cx": 639.9999999990001,
        "cy": 511.99999999900007,
        "xi": 2.898775481441713,
        "alpha": 2.2834031799614942e-10,
        "R": (
            (0.0014769778052491113, -0.936818656378978, -0.34981226907707064),
            (0.9999988112232201, 0.001228757862697969, 0.0009315075207742274),
            (-0.0004428190479218719, -0.34981322904430145, 0.9368193575588034),
        ),
        "t": (-0.7515030247520544, -3.197811963415324, -6.192491211229265),
        "native_wh": (1280.0, 1024.0),
        "square": 1280.0,
        "pad_top": 128.0,
    },
    2: {
        "fx": 2142.6333351590497,
        "fy": 2120.9676215804798,
        "cx": 639.9999999990001,
        "cy": 511.99999999900007,
        "xi": 2.960930518145045,
        "alpha": 1.4826303276414532e-11,
        "R": (
            (-7.955068376510818e-05, -0.937162696227296, -0.34889264032317097),
            (0.9999989366362879, 0.00043349118105460993, -0.0013924121837357029),
            (0.0014561586390844372, -0.34889238009073914, 0.9371616651979726),
        ),
        "t": (-0.8938857274421391, 3.1429902183206258, -6.102439696141204),
        "native_wh": (1280.0, 1024.0),
        "square": 1280.0,
        "pad_top": 128.0,
    },
    3: {
        "fx": 696.4600348473182,
        "fy": 691.4932228780577,
        "cx": 317.99999999900007,
        "cy": 239.99999999900004,
        "xi": 1.8577273354362285,
        "alpha": 2.3498939623338143e-13,
        "R": (
            (0.4748354389099339, -0.8732015755761575, 0.10977392389145135),
            (-0.022138246680925733, 0.11284186334186858, 0.993366302987689),
            (-0.8797961150113727, -0.47411572668374, 0.03425016382507293),
        ),
        "t": (1.2869986707466519, -3.3364557734334452, -8.0123714405678),
        "native_wh": (636.0, 480.0),
        "square": 636.0,
        "pad_top": 78.0,
    },
}


def _egobody3m_cam_order(num_views: int):
    # The projection function only receives tensor view count V, not ego_cams.
    # Keep this in sync with EgoBody3M3DPoseDataset.ego_cams:
    #   V=2 -> front pair [cam1, cam2]
    #   V=4 -> all cameras [cam0, cam1, cam2, cam3]
    if num_views == 2:
        return (1, 2)
    if num_views == 4:
        return (0, 1, 2, 3)
    raise AssertionError(f"egobody3m_ds supports 2 or 4 views, got {num_views}")


def egobody3m_ds_proj(local_3d, local_origin):
    """Project EgoBody3M joints with estimated Double Sphere calibration.

    `local_3d` is pelvis-relative pose in headset coordinates and
    `local_origin` is the pelvis position in the headset frame, matching the
    current EgoBody3M dataset output.  The camera-specific headset->camera
    extrinsic is applied inside this function.
    """
    with torch.no_grad():
        V = local_origin.shape[1]
        cam_ids = _egobody3m_cam_order(V)

        # Reconstruct absolute joint positions in headset coordinates for each
        # selected view. For V=4 this produces one copy per camera in cam0-3 order.
        pts_headset = local_3d[:, None].repeat(1, V, 1, 1) + local_origin
        uv_norm_all = []
        valid_all = []

        for vi, cam_id in enumerate(cam_ids):
            p = _EGOBODY3M_DS_PARAMS[cam_id]
            dtype = pts_headset.dtype
            device = pts_headset.device

            R = torch.tensor(p["R"], device=device, dtype=dtype)
            t = torch.tensor(p["t"], device=device, dtype=dtype)
            # Apply the static extrinsic for the physical camera assigned to
            # this view index, then project with that camera's DS intrinsics.
            xyz = torch.matmul(pts_headset[:, vi], R.T) + t

            x = xyz[..., 0]
            y = xyz[..., 1]
            z = xyz[..., 2]

            d1 = torch.sqrt(x * x + y * y + z * z).clamp(min=1e-6)
            z_xi = p["xi"] * d1 + z
            d2 = torch.sqrt(x * x + y * y + z_xi * z_xi).clamp(min=1e-6)
            denom = p["alpha"] * d2 + (1.0 - p["alpha"]) * z_xi
            denom = denom.clamp(min=1e-6)

            u = p["fx"] * x / denom + p["cx"]
            v_coord = p["fy"] * y / denom + p["cy"]

            native_w, native_h = p["native_wh"]
            square = p["square"]
            pad_top = p["pad_top"]

            u_norm = u / square
            v_norm = (v_coord + pad_top) / square
            uv_norm = torch.stack((u_norm, v_norm), dim=-1)

            valid = (
                (z > 0)
                & (u > 0) & (u < native_w)
                & (v_coord > 0) & (v_coord < native_h)
            )

            uv_norm_all.append(uv_norm)
            valid_all.append(valid)

        image_coor_2d = torch.stack(uv_norm_all, dim=1)
        in_fov = torch.stack(valid_all, dim=1)
        image_coor_2d = image_coor_2d.clamp(min=0.0, max=1.0)
        return image_coor_2d, in_fov


projection_funcs = {
    'unrealego':  unrealego_proj,
    'real_egocam': real_egocam_proj,
    'egobody3m_ds': egobody3m_ds_proj,
}
