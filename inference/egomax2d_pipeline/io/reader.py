"""Input sources for paired EgoMax2D stereo frames."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..configs.utils import load_calibration


_FRAME_NAME = re.compile(r"frame_(\d{8})\.jpg")


@dataclass
class FrameBatch:
    """One aligned batch of left/right stereo frame inputs."""

    indices: list[int]
    left: list[str | np.ndarray]
    right: list[str | np.ndarray]


class InputSource(ABC):
    """Abstract source of normalized calibration and aligned frame batches."""

    @abstractmethod
    def calibration(self) -> dict[str, Any]:
        """Return calibration normalized to ``videoFL`` and ``videoFR``."""

    @abstractmethod
    def __iter__(self) -> Iterator[FrameBatch]:
        """Yield aligned stereo frame batches."""


class FolderInputSource(InputSource):
    """Read deterministic batches from two completed JPEG frame folders."""

    def __init__(
        self,
        left_dir: str | PathLike[str],
        right_dir: str | PathLike[str],
        calibration: str | PathLike[str] | Mapping[str, Any],
        batch_size: int,
        step: int = 1,
        max_frames: int = 0,
    ) -> None:
        self._left_dir = _require_directory(left_dir, "Left frame folder")
        self._right_dir = _require_directory(right_dir, "Right frame folder")
        self._batch_size = _require_integer_at_least(batch_size, "batch_size", 1)
        self._step = _require_integer_at_least(step, "step", 1)
        self._max_frames = _require_integer_at_least(max_frames, "max_frames", 0)

        self._calibration = load_calibration(calibration)
        self._left_frames = _frame_paths_by_index(self._left_dir)
        self._right_frames = _frame_paths_by_index(self._right_dir)

        indices = sorted(self._left_frames.keys() & self._right_frames.keys())
        indices = indices[:: self._step]
        if self._max_frames > 0:
            indices = indices[: self._max_frames]
        if not indices:
            raise ValueError(
                "Left and right frame folders have no usable paired frames after "
                "applying step and max_frames"
            )
        self._indices = indices

    def calibration(self) -> dict[str, Any]:
        """Return the normalized calibration object loaded at construction."""
        return self._calibration

    def __iter__(self) -> Iterator[FrameBatch]:
        for start in range(0, len(self._indices), self._batch_size):
            indices = self._indices[start : start + self._batch_size]
            yield FrameBatch(
                indices=list(indices),
                left=[self._left_frames[index] for index in indices],
                right=[self._right_frames[index] for index in indices],
            )


def _require_directory(path: str | PathLike[str], name: str) -> Path:
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"{name} is not a directory: {directory}")
    return directory


def _require_integer_at_least(value: int, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _frame_paths_by_index(directory: Path) -> dict[int, str]:
    frames: dict[int, str] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = _FRAME_NAME.fullmatch(path.name)
        if match is not None:
            frames[int(match.group(1))] = str(path)
    return frames
