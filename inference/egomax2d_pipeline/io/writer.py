"""In-memory inference result and serialization to the unchanged ``predictions.pt``.

``InferenceResult`` collects per-frame decoded joints/confidences under their
original integer frame index and ``save_pt`` writes the exact ``dict[int, dict]``
shape that ``batch_main()`` produced (see
``../../docs/specs/changes/EgoMax2D-inference-pipeline-refactor/design/data-contracts.md``).
Unlike ``batch_main``, ``save_pt`` creates the output's parent directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InferenceResult:
    """Accumulates decoded stereo predictions keyed by integer frame index."""

    frames: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, idx: int, left, right) -> None:
        """Store one frame's left/right ``(joints, confidences)`` pair.

        ``left``/``right`` are the ``(joints, confidences)`` tuples returned by
        the decode stage; keys preserve the original NumPy dtypes and shapes.
        """
        self.frames[int(idx)] = {
            "left": {"joints": left[0], "confidences": left[1]},
            "right": {"joints": right[0], "confidences": right[1]},
        }

    def to_pt_dict(self) -> dict[int, dict[str, Any]]:
        """Return the exact mapping serialized to ``predictions.pt``."""
        return self.frames


def save_pt(result: InferenceResult, path: str) -> None:
    """Serialize ``result`` to ``path``, creating a parent directory if present."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(result.to_pt_dict(), path)
