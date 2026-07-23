"""Configurable preprocessing comparison helper for EgoMax2D sessions."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm

# Allow direct execution with ``python tests/test_preprocess.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.inference_heatmap_egomax2d_dev import (
    CAMS,
    IMG_SIZE,
    _EGOBODY3M_DS_PARAMS,
    remap_preprocess,
    resolve_session,
)
from pose_estimation.datasets.egomax2d.remap import build_remap, load_session_calib


@dataclass
class TestInput:
    function: callable
    batch_size: int


def gt_preprocess(
    left_image_path: str, right_image_path: str, remaps: dict
) -> torch.Tensor:
    """The ground-truth preprocessing function for a single stereo pair."""
    bgr_l = cv2.imread(left_image_path)
    bgr_r = cv2.imread(right_image_path)

    _, tensor_l = remap_preprocess(bgr_l, *remaps[CAMS[0][1]])
    _, tensor_r = remap_preprocess(bgr_r, *remaps[CAMS[1][1]])

    return torch.stack([tensor_l, tensor_r]).unsqueeze(0)


def test_preprocess(
    test_input: TestInput,
    root: str = "data/EgoMax2D",
    session_id: str = "01KWEDQ9HG6CSF6CNW0QVFV92E",
    session_idx: int = 0,
    rotate: str = "right",
    test_frames: int = 128,
) -> None:
    """Test a preprocessing function against ground truth for one session."""
    session = resolve_session(root, session_id, session_idx)
    session_dir = os.path.join(root, session)
    print("Testing episode: ", session_id)

    calib = load_session_calib(session_dir)
    remap_cfg = {
        "head-front-left": (calib["videoFL"], dict(_EGOBODY3M_DS_PARAMS[1])),
        "head-front-right": (calib["videoFR"], dict(_EGOBODY3M_DS_PARAMS[2])),
    }
    remaps = {
        cam_dir: build_remap(eb, src, rotate)
        for cam_dir, (src, eb) in remap_cfg.items()
    }

    n_frames = min(
        len(os.listdir(os.path.join(session_dir, "images", CAMS[0][1]))),
        test_frames,
    )
    batch_size = max(1, test_input.batch_size)

    error_list = []
    for idx in tqdm(
        range(0, n_frames, batch_size),
        desc=f"Frames: {n_frames} | batch-size: {batch_size}",
        unit="batch",
    ):
        current_batch_size = min(batch_size, n_frames - idx)

        paths, maps_list = [], []
        for i in range(idx, idx + current_batch_size):
            for _, cam_dir in CAMS:
                paths.append(
                    os.path.join(
                        session_dir, "images", cam_dir, f"frame_{i:08d}.jpg"
                    )
                )
                maps_list.append(remaps[cam_dir])

        process_res = test_input.function(paths, maps_list)
        gt_process_res = torch.stack(
            [
                gt_preprocess(
                    os.path.join(
                        session_dir, "images", CAMS[0][1], f"frame_{i:08d}.jpg"
                    ),
                    os.path.join(
                        session_dir, "images", CAMS[1][1], f"frame_{i:08d}.jpg"
                    ),
                    remaps,
                )
                for i in range(idx, idx + current_batch_size)
            ],
            dim=0,
        ).view(current_batch_size * 2, 3, IMG_SIZE, IMG_SIZE).to(process_res.device)

        error_list.append(torch.abs(process_res - gt_process_res))

    errors = torch.cat(error_list, dim=0)

    assert torch.equal(process_res[:, 0], process_res[:, 1]), "testing results: channel 0 != channel 1"
    assert torch.equal(process_res[:, 1], process_res[:, 2]), "testing results: channel 1 != channel 2"
    assert torch.equal(gt_process_res[:, 0], gt_process_res[:, 1]), "testing results: channel 0 != channel 1"
    assert torch.equal(gt_process_res[:, 1], gt_process_res[:, 2]), "testing results: channel 1 != channel 2"

    spatial_err = errors[:, 0]
    print(f"MAE:    {spatial_err.mean():.3f}")
    print(f"RMSE:   {spatial_err.square().mean().sqrt():.3f}")
    print(f"Median: {spatial_err.median():.3f}")
    print(f"P80:    {torch.quantile(spatial_err, 0.80):.3f}")
    print(f"P90:    {torch.quantile(spatial_err, 0.90):.3f}")
    print(f"P99:    {torch.quantile(spatial_err, 0.99):.3f}")
    print(f"P99.9:  {torch.quantile(spatial_err, 0.999):.3f}")
    print(f"Max:    {spatial_err.max():.3f}")


# This helper requires a caller-supplied preprocessing implementation and is
# therefore excluded from pytest's automatic test collection.
test_preprocess.__test__ = False


if __name__ == "__main__":
    from inference.inference_heatmap_egomax2d_dev import batch_preprocess

    test_input = TestInput(function=batch_preprocess, batch_size=32)
    # test_input = TestInput(function=preprocess_gpu, batch_size=32)
    test_preprocess(test_input)
