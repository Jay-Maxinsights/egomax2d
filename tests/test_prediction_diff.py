"""Payload parity report between two pose-prediction .pt files.

The report compares the refactored pipeline against ``batch_main()``.
It gates only on structural agreement of the two ``predictions.pt`` payloads:

- identical frame keys and, per frame, identical view keys;
- identical joint/confidence array shapes and dtypes.

Joint coordinates, joint validity, and confidences are *not* gated: they are not
bit-reproducible across runs/hardware (heatmap argmax can shift by a pixel or
land on a different peak). Instead they are reported:

- keypoint relative difference ``||gt - target|| / diag`` over jointly-valid slots;
- a validity-mismatch count (joints marked valid in one payload but not the other);
- confidence absolute difference ``|gt - target|`` over all slots (confidence is
  already in [0, 1], so the absolute difference is itself the rate).

The report is printed so divergences remain visible even though they no longer
fail the test. All configuration is in the module globals below.

python -m pytest tests/test_prediction_diff.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prediction files to compare (resolved against the repo root).
GT_PT = os.path.join(
    REPO_ROOT, "results/heatmap_egomax2d_lu_gt/01KWEDQ9HG6CSF6CNW0QVFV92E_predictions.pt"
)
TARGET_PT = os.path.join(
    REPO_ROOT, "results/heatmap_egomax2d/01KWEDQ9HG6CSF6CNW0QVFV92E_predictions.pt"
)

# Image size used in the normalization denominator sqrt(w^2 + h^2).
# Predictions are stored in 256x256 canvas space.
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Statistics reported over the relative differences.
# Tokens: min, median, max, mean, pNN (any percentile, e.g. p90, p99, p99.9).
STATS = ["min", "median", "max", "mean", "p90", "p98", "p99"]


def _parse_stats(spec):
    """Parse the 'stats' config (list or comma string) into (name, callable) pairs."""
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
                raise ValueError(f"Unrecognized statistic {token!r} in stats config")
            if not 0 <= q <= 100:
                raise ValueError(f"Percentile out of range in stats config: {token!r}")
            stats.append((token, lambda a, q=q: np.percentile(a, q)))
        else:
            raise ValueError(f"Unrecognized statistic {token!r} in stats config")
    if not stats:
        raise ValueError("stats config parsed to an empty list")
    return stats


def _valid_mask(joints):
    """A joint is valid unless it carries the [-1, -1] placeholder."""
    return ~np.all(joints == -1.0, axis=-1)


def _load_predictions(path):
    """Load prediction payloads written by either NumPy 1 or NumPy 2.

    NumPy 2 pickles refer to ``numpy._core``.  NumPy 1 exposes the same
    implementation as ``numpy.core``, so register those aliases only when a
    NumPy-2-authored fixture is loaded in a NumPy-1 runtime.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise

    # Import the concrete modules before registering aliases so pickle can
    # resolve both the package and common NumPy reconstruction helpers.
    import numpy.core as numpy_core
    import numpy.core.multiarray as numpy_multiarray
    import numpy.core.numeric as numpy_numeric

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
    return torch.load(path, map_location="cpu", weights_only=False)


def test_prediction_parity(capsys):
    gt_path = GT_PT
    target_path = TARGET_PT
    img_w = float(IMG_WIDTH)
    img_h = float(IMG_HEIGHT)
    stats = _parse_stats(STATS)

    for label, path in (("gt", gt_path), ("target", target_path)):
        if not os.path.isfile(path):
            pytest.skip(f"{label} predictions file not found: {path}")

    gt = _load_predictions(gt_path)
    target = _load_predictions(target_path)

    diag = float(np.hypot(img_w, img_h))
    assert diag > 0, "image diagonal must be positive"

    # Identical frame keys — no gt-only or target-only frames.
    assert set(gt) == set(target), (
        "frame key mismatch: "
        f"gt-only={sorted(set(gt) - set(target))[:10]}, "
        f"target-only={sorted(set(target) - set(gt))[:10]}"
    )
    frames = sorted(gt)
    assert frames, "gt and target share no frame indices"

    rel_diffs = []
    conf_abs_diffs = []
    views_compared = 0
    total_kps = 0
    both_invalid = 0
    validity_mismatches = 0

    for frame in frames:
        gt_views, t_views = gt[frame], target[frame]
        # Identical view keys per frame.
        assert set(gt_views) == set(t_views), (
            f"view key mismatch at frame {frame}: "
            f"gt={sorted(gt_views)} target={sorted(t_views)}"
        )
        for view in sorted(gt_views):
            gt_joints = np.asarray(gt_views[view]["joints"])
            t_joints = np.asarray(t_views[view]["joints"])
            gt_conf = np.asarray(gt_views[view]["confidences"])
            t_conf = np.asarray(t_views[view]["confidences"])

            # Shapes and dtypes must match exactly.
            assert gt_joints.shape == t_joints.shape, (
                f"joint shape mismatch at frame {frame} view {view}: "
                f"{gt_joints.shape} vs {t_joints.shape}"
            )
            assert gt_joints.dtype == t_joints.dtype, (
                f"joint dtype mismatch at frame {frame} view {view}: "
                f"{gt_joints.dtype} vs {t_joints.dtype}"
            )
            assert gt_conf.shape == t_conf.shape, (
                f"confidence shape mismatch at frame {frame} view {view}: "
                f"{gt_conf.shape} vs {t_conf.shape}"
            )
            assert gt_conf.dtype == t_conf.dtype, (
                f"confidence dtype mismatch at frame {frame} view {view}: "
                f"{gt_conf.dtype} vs {t_conf.dtype}"
            )
            assert gt_joints.ndim == 2 and gt_joints.shape[1] == 2

            # Joint coordinates are reported (not gated): their relative
            # difference is accumulated below alongside a validity-mismatch count.
            gt_valid = _valid_mask(gt_joints)
            t_valid = _valid_mask(t_joints)
            validity_mismatches += int(np.sum(gt_valid != t_valid))

            # Confidences are reported (not gated): absolute difference over all
            # slots (confidence is already in [0, 1], so |gt - target| is the rate).
            if gt_conf.size:
                conf_abs_diffs.append(
                    np.abs(gt_conf.astype(np.float64) - t_conf.astype(np.float64))
                )

            views_compared += 1
            total_kps += gt_joints.shape[0]
            both_invalid += int(np.sum(~gt_valid & ~t_valid))
            both_valid = gt_valid & t_valid
            dists = np.linalg.norm(
                gt_joints[both_valid].astype(np.float64)
                - t_joints[both_valid].astype(np.float64),
                axis=-1,
            )
            rel_diffs.append(dists / diag)

    rel_diffs = np.concatenate(rel_diffs) if rel_diffs else np.zeros(0)
    conf_abs_diffs = (
        np.concatenate(conf_abs_diffs) if conf_abs_diffs else np.zeros(0)
    )

    lines = [
        "",
        "=== Prediction parity report ===",
        f"gt:     {gt_path}",
        f"target: {target_path}",
        f"normalization diagonal: sqrt({img_w:g}^2 + {img_h:g}^2) = {diag:.4f}",
        f"frames compared: {len(frames)} (identical frame keys)",
        f"views compared: {views_compared}",
        f"keypoint slots: {total_kps}"
        f" (both-invalid, counted as 0: {both_invalid})",
        f"validity mismatches: {validity_mismatches} (reported, not gated)",
        "--- valid-joint relative difference statistics ---",
    ]
    if rel_diffs.size:
        lines += [f"{name:>8}: {fn(rel_diffs):.6f}" for name, fn in stats]
    else:
        lines.append("(no jointly-valid keypoints)")
    lines.append("--- confidence absolute difference statistics ---")
    if conf_abs_diffs.size:
        lines += [f"{name:>8}: {fn(conf_abs_diffs):.6f}" for name, fn in stats]
    else:
        lines.append("(no confidences)")
    with capsys.disabled():
        print("\n".join(lines))
