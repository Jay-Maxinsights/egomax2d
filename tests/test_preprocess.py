"""Compare ``Pipeline.preprocess`` with the legacy GPU fast path.

The pipeline stage flattens a ``FrameBatch`` into the interleaved order
``[f0_left, f0_right, f1_left, f1_right, ...]``, runs the existing
``preprocess_gpu`` implementation from
``inference/inference_heatmap_egomax2d_dev.py``, and reshapes to
``(B, 2, 3, 256, 256)``. This test asserts the pipeline stage reproduces the
legacy fast path exactly, including the interleaved view ordering:

* ground truth: ``preprocess_gpu`` over the same flat path/map lists, reshaped
* function under test: ``Pipeline.preprocess(FrameBatch)``

Model loading is monkeypatched so the test measures preprocessing rather than
checkpoint startup. Input paths and report statistics are configured in the
module globals below.

python -m pytest tests/test_preprocess.py -q
"""

import os

import numpy as np
import pytest
import torch

from inference.inference_heatmap_egomax2d_dev import preprocess_gpu
from inference.egomax2d_pipeline import pipeline as pipeline_mod
from inference.egomax2d_pipeline.configs.constant import IMG_SIZE
from inference.egomax2d_pipeline.configs.utils import load_calibration
from inference.egomax2d_pipeline.io.reader import FrameBatch
from inference.egomax2d_pipeline.pipeline import Pipeline
from pose_estimation.datasets.egomax2d.remap import build_session_remaps


# Raw left-camera images and their session directory. Right-camera paths are
# derived by swapping the camera folder so the pair is genuinely stereo.
SESSION_DIR = "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E"
LEFT_PATHS = [
    "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E/images/head-front-left/frame_00000000.jpg",
    "/home/max/jay/HIL_pose_annotation/egomax2d/data/EgoMax2D/01KWEDQ9HG6CSF6CNW0QVFV92E/images/head-front-left/frame_00000001.jpg",
]
RIGHT_PATHS = [
    path.replace("head-front-left", "head-front-right") for path in LEFT_PATHS
]
FRAME_INDICES = [0, 1]

DEVICE = "cuda"
RAW_WH = (2592, 1944)

# Calibration remap direction. This must match the inference pipeline.
ROTATE = "right"

# Relative-difference denominator floor. It keeps zero-valued reference pixels
# finite while still exposing nonzero target values at those positions.
REL_EPS = 1e-8

# Tokens: min, median, max, mean, pNN (any percentile, e.g. p90 or p99.9).
STATS = ["min", "median", "max", "mean", "p90", "p98", "p99"]


class _StubModel:
    """Stand-in for the checkpoint model so ``Pipeline`` builds without one."""

    def to(self, device):
        return self


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


def test_pipeline_preprocess_matches_fast_path(capsys, monkeypatch):
    stats = _parse_stats(STATS)
    if not np.isfinite(REL_EPS) or REL_EPS <= 0:
        raise ValueError(f"REL_EPS must be finite and positive, got {REL_EPS!r}")
    if not LEFT_PATHS:
        pytest.skip("no input images configured in LEFT_PATHS")

    all_paths = LEFT_PATHS + RIGHT_PATHS
    missing = [path for path in all_paths if not os.path.isfile(path)]
    if missing:
        pytest.skip("input image not found: " + ", ".join(missing))
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"configured CUDA device is unavailable: {DEVICE}")

    # Measure preprocessing, not checkpoint startup: the model is never used by
    # Pipeline.preprocess, so a stub keeps construction cheap and GPU-free.
    monkeypatch.setattr(pipeline_mod, "load_model", lambda ckpt: _StubModel())

    calibration = load_calibration(SESSION_DIR)
    batch = FrameBatch(
        indices=list(FRAME_INDICES),
        left=list(LEFT_PATHS),
        right=list(RIGHT_PATHS),
    )

    # Ground truth: the legacy fast path over the interleaved flat lists,
    # [f0_L, f0_R, f1_L, f1_R, ...], reshaped to (B, 2, 3, 256, 256).
    remaps = build_session_remaps(SESSION_DIR, rotate=ROTATE)
    flat_paths, flat_maps = [], []
    for left_path, right_path in zip(LEFT_PATHS, RIGHT_PATHS):
        flat_paths.append(left_path)
        flat_maps.append(remaps["left"])
        flat_paths.append(right_path)
        flat_maps.append(remaps["right"])

    with torch.no_grad():
        gt_output = preprocess_gpu(
            flat_paths, flat_maps, device=DEVICE, raw_wh=RAW_WH
        ).view(len(FRAME_INDICES), 2, 3, IMG_SIZE, IMG_SIZE)

        pipeline = Pipeline(
            calibration=calibration,
            ckpt="<stubbed>",
            device=DEVICE,
            rotate=ROTATE,
            preprocess="gpu",
        )
        test_output = pipeline.preprocess(batch)

    gt_output = gt_output.detach().cpu()
    test_output = test_output.detach().cpu()
    expected_shape = (len(FRAME_INDICES), 2, 3, 256, 256)
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
    # The pipeline stage must reproduce the fast path bit-for-bit, including the
    # interleaved left/right view ordering within each frame.
    assert torch.equal(test_output, gt_output), (
        "Pipeline.preprocess diverged from the legacy fast path — tensor values "
        "or view ordering differ"
    )

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
        "=== Pipeline.preprocess relative-diff report ===",
        "ground truth: preprocess_gpu (flat fast path, reshaped)",
        "test:         Pipeline.preprocess (FrameBatch)",
        f"device:       {DEVICE}",
        f"rotation:     {ROTATE}",
        "inputs:",
    ]
    lines += [f"  L {path}" for path in LEFT_PATHS]
    lines += [f"  R {path}" for path in RIGHT_PATHS]
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
