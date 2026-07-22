"""CPU-only tests for deterministic paired-folder input."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.egomax2d_pipeline.io.reader import FolderInputSource


def _normalized_calibration() -> dict:
    params = {
        "fx": 1000.0,
        "fy": 1001.0,
        "cx": 1296.0,
        "cy": 972.0,
        "xi": -0.1,
        "alpha": 0.6,
    }
    return {"videoFL": dict(params), "videoFR": dict(params, fx=1002.0)}


def _make_frames(directory: Path, indices: set[int]) -> None:
    directory.mkdir()
    for index in indices:
        (directory / f"frame_{index:08d}.jpg").touch()


# Verify exact frame pairing and the step -> limit -> batch operation order.
@pytest.mark.parametrize(
    ("batch_size", "step", "max_frames", "expected_batches"),
    [
        (2, 1, 0, [[0, 1], [5]]),
        (2, 2, 0, [[0, 5]]),
        (1, 1, 2, [[0], [1]]),
        (2, 2, 1, [[0]]),
    ],
)
def test_folder_input_source_pairs_samples_and_batches(
    tmp_path, batch_size, step, max_frames, expected_batches
):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    _make_frames(left_dir, {0, 1, 2, 5})
    _make_frames(right_dir, {0, 1, 5, 9})

    # Exact, case-sensitive ``frame_%08d.jpg`` names are the only accepted inputs.
    (left_dir / "frame_00000003.JPG").touch()
    (right_dir / "frame_00000003.JPG").touch()
    (left_dir / "frame_000000004.jpg").touch()
    (right_dir / "frame_000000004.jpg").touch()

    calibration = _normalized_calibration()
    source = FolderInputSource(
        left_dir,
        right_dir,
        calibration,
        batch_size=batch_size,
        step=step,
        max_frames=max_frames,
    )
    batches = list(source)

    assert [batch.indices for batch in batches] == expected_batches
    for batch in batches:
        assert len(batch.indices) == len(batch.left) == len(batch.right)
        for index, left, right in zip(batch.indices, batch.left, batch.right):
            assert Path(left) == left_dir / f"frame_{index:08d}.jpg"
            assert Path(right) == right_dir / f"frame_{index:08d}.jpg"

    assert source.calibration() is calibration
    assert source.calibration() == _normalized_calibration()


# Reject invalid batching and sampling values at construction time.
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size must be an integer >= 1"),
        ("step", 0, "step must be an integer >= 1"),
        ("max_frames", -1, "max_frames must be an integer >= 0"),
    ],
)
def test_folder_input_source_rejects_invalid_sampling_values(
    tmp_path, field, value, message
):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    _make_frames(left_dir, {0})
    _make_frames(right_dir, {0})
    kwargs = {"batch_size": 1, "step": 1, "max_frames": 0, field: value}

    with pytest.raises(ValueError, match=message):
        FolderInputSource(
            left_dir, right_dir, _normalized_calibration(), **kwargs
        )


# Fail clearly when a frame folder is missing or the cameras share no frames.
def test_folder_input_source_rejects_missing_folder_and_empty_intersection(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    _make_frames(left_dir, {0})

    with pytest.raises(FileNotFoundError, match="Right frame folder does not exist"):
        FolderInputSource(
            left_dir, right_dir, _normalized_calibration(), batch_size=1
        )

    _make_frames(right_dir, {1})
    with pytest.raises(ValueError, match="no usable paired frames"):
        FolderInputSource(
            left_dir, right_dir, _normalized_calibration(), batch_size=1
        )
