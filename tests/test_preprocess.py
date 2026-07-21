"""Compare GPU preprocessing with the CPU reference.

The default comparison runs these two implementations from
``inference/inference_heatmap_egomax2d_dev.py``:

* ground truth: ``preprocess`` (one image at a time on CPU)
* function under test: ``preprocess_gpu`` (all configured images in one batch)

For every output tensor value, the reported relative difference is

    abs(test - gt) / max(abs(gt), REL_EPS)

Reporting only -- no numerical threshold is enforced. Input paths and report
statistics are configured in the module globals below.

python -m pytest tests/test_preprocess.py -q
"""

import os

import numpy as np
import pytest
import torch

from inference.inference_heatmap_egomax2d_dev import (
    preprocess,
    preprocess_gpu,
)
from pose_estimation.datasets.egomax2d.remap import build_session_remaps


# Raw left-camera images and their session directory.
SESSION_DIR = "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E"
IMAGE_PATHS = [
    "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E/images/head-front-left/frame_00000000.jpg",
    "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E/images/head-front-left/frame_00000001.jpg",
]

DEVICE = "cuda"
RAW_WH = (2592, 1944)

# Calibration remap direction. This must match the inference pipeline.
ROTATE = "right"

# Relative-difference denominator floor. It keeps zero-valued reference pixels
# finite while still exposing nonzero target values at those positions.
REL_EPS = 1e-8

# Tokens: min, median, max, mean, pNN (any percentile, e.g. p90 or p99.9).
STATS = ["min", "median", "max", "mean", "p90", "p98", "p99"]


def _parse_stats(spec):
    """Parse a list or comma string into (display name, callable) pairs."""
    fixed = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "median": np.median,
    }
    tokens = spec.split(",") if isinstance(spec, str) else spec
    stats = []
    for token in tokens:
        token = str(token).strip().lower()
        if not token:
            continue
        if token in fixed:
            stats.append((token, fixed[token]))
        elif token.startswith("p"):
            try:
                q = float(token[1:])
            except ValueError:
                raise ValueError(
                    f"Unrecognized statistic {token!r} in stats config"
                ) from None
            if not 0 <= q <= 100:
                raise ValueError(
                    f"Percentile out of range in stats config: {token!r}"
                )
            stats.append((token, lambda values, q=q: np.percentile(values, q)))
        else:
            raise ValueError(f"Unrecognized statistic {token!r} in stats config")
    if not stats:
        raise ValueError("stats config parsed to an empty list")
    return stats


def test_preprocess_relative_diff(capsys):
    stats = _parse_stats(STATS)
    if not np.isfinite(REL_EPS) or REL_EPS <= 0:
        raise ValueError(f"REL_EPS must be finite and positive, got {REL_EPS!r}")
    if not IMAGE_PATHS:
        pytest.skip("no input images configured in IMAGE_PATHS")

    missing = [path for path in IMAGE_PATHS if not os.path.isfile(path)]
    if missing:
        pytest.skip("input image not found: " + ", ".join(missing))
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"configured CUDA device is unavailable: {DEVICE}")

    left_map = build_session_remaps(SESSION_DIR, rotate=ROTATE)["left"]
    maps = [left_map] * len(IMAGE_PATHS)

    with torch.no_grad():
        gt_output = torch.stack(
            [preprocess(path, *left_map) for path in IMAGE_PATHS], dim=0
        )
        test_output = preprocess_gpu(
            IMAGE_PATHS, maps, device=DEVICE, raw_wh=RAW_WH
        )

    gt_output = gt_output.detach().cpu()
    test_output = test_output.detach().cpu()
    expected_shape = (len(IMAGE_PATHS), 3, 256, 256)
    assert tuple(gt_output.shape) == expected_shape, (
        f"ground-truth output shape must be {expected_shape}, "
        f"got {tuple(gt_output.shape)}"
    )
    assert tuple(test_output.shape) == expected_shape, (
        f"test output shape must be {expected_shape}, "
        f"got {tuple(test_output.shape)}"
    )
    assert torch.isfinite(gt_output).all(), "ground-truth output contains non-finite values"
    assert torch.isfinite(test_output).all(), "test output contains non-finite values"

    gt_values = gt_output.numpy().astype(np.float64, copy=False)
    test_values = test_output.numpy().astype(np.float64, copy=False)
    abs_diffs = np.abs(test_values - gt_values)
    rel_diffs = abs_diffs / np.maximum(np.abs(gt_values), REL_EPS)
    assert np.isfinite(rel_diffs).all(), "relative differences contain non-finite values"

    total_values = int(rel_diffs.size)
    exact_matches = int(np.count_nonzero(abs_diffs == 0))
    zero_reference_values = int(np.count_nonzero(gt_values == 0))
    lines = [
        "",
        "=== Preprocess relative-diff report ===",
        "ground truth: preprocess (single)",
        "test:         preprocess_gpu (batch)",
        f"device:       {DEVICE}",
        f"rotation:     {ROTATE}",
        "inputs:",
    ]
    lines += [f"  {path}" for path in IMAGE_PATHS]
    lines += [
        f"output shape: {tuple(test_output.shape)}",
        f"values compared: {total_values}",
        f"exact matches: {exact_matches} ({exact_matches / total_values:.2%})",
        f"zero-valued GT values: {zero_reference_values}",
        f"relative denominator epsilon: {REL_EPS:g}",
        "--- relative difference statistics ---",
    ]
    lines += [f"{name:>8}: {fn(rel_diffs):.8e}" for name, fn in stats]
    with capsys.disabled():
        print("\n".join(lines))
