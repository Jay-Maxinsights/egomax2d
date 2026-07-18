"""Parity test between two pipelines' pose-prediction .pt files.

For every frame, view, and keypoint the normalized difference is

    sqrt((gt_x - t_x)^2 + (gt_y - t_y)^2) / sqrt(img_w^2 + img_h^2)

Keypoints invalid (coords == -1) in both files count as 0 difference;
keypoints invalid in exactly one file are excluded from the statistics
and reported as validity mismatches. Reporting only — no threshold.

All configuration is in the module globals below.

python -m pytest tests/test_prediction_diff.py -q
"""

import os

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prediction files to compare (resolved against the repo root).
GT_PT = os.path.join(
    REPO_ROOT, "results/heatmap_egomax2d_gt/01KWEDQ9HG6CSF6CNW0QVFV92E_predictions.pt"
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


def test_prediction_relative_diff(capsys):
    # load inputs
    gt_path = GT_PT
    target_path = TARGET_PT
    img_w = float(IMG_WIDTH)
    img_h = float(IMG_HEIGHT)
    stats = _parse_stats(STATS)

    for label, path in (("gt", gt_path), ("target", target_path)):
        if not os.path.isfile(path):
            pytest.skip(f"{label} predictions file not found: {path}")

    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)

    diag = float(np.hypot(img_w, img_h))
    assert diag > 0, "image diagonal must be positive"

    common_frames = sorted(set(gt) & set(target))
    assert common_frames, "gt and target share no frame indices"
    gt_only_frames = len(set(gt) - set(target))
    target_only_frames = len(set(target) - set(gt))

    rel_diffs = []
    views_compared = 0
    view_mismatches = 0
    total_kps = 0
    both_invalid = 0
    gt_only_valid = 0
    target_only_valid = 0

    for frame in common_frames:
        gt_views, t_views = gt[frame], target[frame]
        view_mismatches += len(set(gt_views) ^ set(t_views))
        for view in sorted(set(gt_views) & set(t_views)):
            gt_joints = np.asarray(gt_views[view]["joints"], dtype=np.float64)
            t_joints = np.asarray(t_views[view]["joints"], dtype=np.float64)
            assert gt_joints.shape == t_joints.shape, (
                f"joint shape mismatch at frame {frame} view {view}: "
                f"{gt_joints.shape} vs {t_joints.shape}"
            )
            assert gt_joints.ndim == 2 and gt_joints.shape[1] == 2

            gt_valid = _valid_mask(gt_joints)
            t_valid = _valid_mask(t_joints)

            views_compared += 1
            total_kps += gt_joints.shape[0]
            both_invalid += int(np.sum(~gt_valid & ~t_valid))
            gt_only_valid += int(np.sum(gt_valid & ~t_valid))
            target_only_valid += int(np.sum(~gt_valid & t_valid))

            both_valid = gt_valid & t_valid
            dists = np.linalg.norm(gt_joints[both_valid] - t_joints[both_valid], axis=-1)
            rel_diffs.append(dists / diag)
            # both-invalid joints count as zero difference
            rel_diffs.append(np.zeros(int(np.sum(~gt_valid & ~t_valid))))

    rel_diffs = np.concatenate(rel_diffs)
    mismatches = gt_only_valid + target_only_valid
    assert len(rel_diffs) + mismatches == total_kps

    lines = [
        "",
        "=== Prediction relative-diff report ===",
        f"gt:     {gt_path}",
        f"target: {target_path}",
        f"normalization diagonal: sqrt({img_w:g}^2 + {img_h:g}^2) = {diag:.4f}",
        f"frames compared: {len(common_frames)}"
        f" (gt-only: {gt_only_frames}, target-only: {target_only_frames})",
        f"views compared: {views_compared} (view mismatches: {view_mismatches})",
        f"keypoint slots: {total_kps}",
        f"  compared: {len(rel_diffs)} (of which both-invalid, counted as 0: {both_invalid})",
        f"  validity mismatches (excluded): {mismatches}"
        f" (valid in gt only: {gt_only_valid}, valid in target only: {target_only_valid})",
        "--- relative difference statistics ---",
    ]
    lines += [f"{name:>8}: {fn(rel_diffs):.6f}" for name, fn in stats]
    with capsys.disabled():
        print("\n".join(lines))
