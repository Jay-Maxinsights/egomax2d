"""Tests for the sequential orchestrator and the ``predictions.pt`` writer.

Uses fake source/pipeline/monitor doubles so the orchestrator contract is
exercised without a GPU or checkpoint, plus a temporary-path save/load round trip
that covers ``InferenceResult`` serialization (no standalone writer suite).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from inference.egomax2d_pipeline.io.reader import FrameBatch
from inference.egomax2d_pipeline.io.writer import InferenceResult, save_pt
from inference.egomax2d_pipeline.main import run_sequential


class _FakeSource:
    """Yields prebuilt batches; run_sequential only needs iteration."""

    def __init__(self, batches: list[FrameBatch]) -> None:
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class _FakePipeline:
    """Records stage calls and emits deterministic per-frame predictions."""

    def __init__(self, events: list, fail_on_inference: bool = False) -> None:
        self._events = events
        self._fail_on_inference = fail_on_inference
        self.grad_enabled_during_inference: list[bool] = []

    def preprocess(self, batch: FrameBatch):
        self._events.append(("preprocess", tuple(batch.indices)))
        return ("tensors", tuple(batch.indices))

    def inference(self, tensors):
        self._events.append(("inference", tensors[1]))
        self.grad_enabled_during_inference.append(torch.is_grad_enabled())
        if self._fail_on_inference:
            raise RuntimeError("boom")
        return ("heatmaps", tensors[1])

    def decode(self, hm, batch: FrameBatch, result: InferenceResult) -> None:
        self._events.append(("decode", tuple(batch.indices)))
        for idx in batch.indices:
            left = (
                np.full((26, 2), float(idx), dtype=np.float32),
                np.full((26,), float(idx) * 0.01, dtype=np.float32),
            )
            right = (
                np.full((26, 2), float(idx) + 0.5, dtype=np.float32),
                np.full((26,), float(idx) * 0.02, dtype=np.float32),
            )
            result.add(idx, left, right)


class _FakeMonitor:
    """Records lifecycle order and the cardinality passed to each timer."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.model_cardinalities: list[int] = []
        self.e2e_cardinalities: list[int] = []

    def start(self) -> None:
        self._events.append(("monitor_start",))

    def stop(self) -> None:
        self._events.append(("monitor_stop",))

    @contextmanager
    def e2e_timer(self, stereo_frames: int):
        self.e2e_cardinalities.append(stereo_frames)
        self._events.append(("e2e_enter", stereo_frames))
        try:
            yield
        finally:
            self._events.append(("e2e_exit", stereo_frames))

    @contextmanager
    def model_timer(self, stereo_frames: int):
        self.model_cardinalities.append(stereo_frames)
        yield


def _make_batches() -> list[FrameBatch]:
    # Non-contiguous indices, and a partial final batch of one stereo frame.
    return [
        FrameBatch(indices=[0, 3, 7], left=["l0", "l3", "l7"], right=["r0", "r3", "r7"]),
        FrameBatch(indices=[10], left=["l10"], right=["r10"]),
    ]


def test_run_sequential_orders_stages_and_brackets_the_monitor(tmp_path):
    events: list = []
    source = _FakeSource(_make_batches())
    pipeline = _FakePipeline(events)
    monitor = _FakeMonitor(events)

    result = run_sequential(source, pipeline, monitor)

    # start/stop bracket the whole run.
    assert events[0] == ("monitor_start",)
    assert events[-1] == ("monitor_stop",)

    # Stage order within each batch: e2e opens, then preprocess -> inference ->
    # decode, then e2e closes; batches run in source order.
    assert events[1:-1] == [
        ("e2e_enter", 3),
        ("preprocess", (0, 3, 7)),
        ("inference", (0, 3, 7)),
        ("decode", (0, 3, 7)),
        ("e2e_exit", 3),
        ("e2e_enter", 1),
        ("preprocess", (10,)),
        ("inference", (10,)),
        ("decode", (10,)),
        ("e2e_exit", 1),
    ]

    # Actual (partial) batch cardinality reaches both timers.
    assert monitor.e2e_cardinalities == [3, 1]
    assert monitor.model_cardinalities == [3, 1]

    # Inference always runs with autograd disabled.
    assert pipeline.grad_enabled_during_inference == [False, False]

    # Returned result preserves non-contiguous integer frame keys and shapes.
    assert list(result.frames.keys()) == [0, 3, 7, 10]
    for idx, frame in result.frames.items():
        assert set(frame) == {"left", "right"}
        for view in ("left", "right"):
            joints = frame[view]["joints"]
            confs = frame[view]["confidences"]
            assert joints.shape == (26, 2) and joints.dtype == np.float32
            assert confs.shape == (26,) and confs.dtype == np.float32

    # Save/load round trip into a not-yet-existing nested directory.
    out_path = tmp_path / "nested" / "predictions.pt"
    save_pt(result, str(out_path))
    assert out_path.exists()
    loaded = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert list(loaded.keys()) == [0, 3, 7, 10]
    assert all(isinstance(key, int) for key in loaded)
    for idx, frame in loaded.items():
        np.testing.assert_array_equal(
            frame["left"]["joints"], np.full((26, 2), float(idx), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["right"]["confidences"],
            np.full((26,), float(idx) * 0.02, dtype=np.float32),
        )
        assert frame["left"]["joints"].dtype == np.float32


def test_save_pt_writes_a_bare_filename_in_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = InferenceResult()
    result.add(
        5,
        (np.zeros((26, 2), np.float32), np.zeros((26,), np.float32)),
        (np.ones((26, 2), np.float32), np.ones((26,), np.float32)),
    )

    save_pt(result, "predictions.pt")

    written = tmp_path / "predictions.pt"
    assert written.exists()
    loaded = torch.load(str(written), map_location="cpu", weights_only=False)
    assert list(loaded.keys()) == [5]


def test_run_sequential_stops_the_monitor_when_a_stage_raises():
    events: list = []
    source = _FakeSource(_make_batches())
    pipeline = _FakePipeline(events, fail_on_inference=True)
    monitor = _FakeMonitor(events)

    with pytest.raises(RuntimeError, match="boom"):
        run_sequential(source, pipeline, monitor)

    # stop() still runs through try/finally after the raising stage.
    assert events[0] == ("monitor_start",)
    assert events[-1] == ("monitor_stop",)
    assert ("inference", (0, 3, 7)) in events
    assert ("decode", (0, 3, 7)) not in events
