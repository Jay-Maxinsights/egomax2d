"""CPU-only tests for decoupled inference resource monitoring."""

from __future__ import annotations

import json

import pytest

import inference.egomax2d_pipeline.monitor as monitor_module
from inference.egomax2d_pipeline.monitor import ResourceMonitor


class _Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_ms(self, milliseconds: float) -> None:
        self.seconds += milliseconds / 1000.0


class _FakeEvent:
    def __init__(self, cuda) -> None:
        self._cuda = cuda

    def record(self) -> None:
        pass

    def elapsed_time(self, end_event) -> float:
        assert isinstance(end_event, _FakeEvent)
        return self._cuda.event_durations_ms.pop(0)


class _FakeCuda:
    def __init__(self, event_durations_ms: list[float]) -> None:
        self.event_durations_ms = list(event_durations_ms)
        self.reset_calls = 0
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return True

    def Event(self, enable_timing: bool):
        assert enable_timing is True
        return _FakeEvent(self)

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return 4096

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeTorch:
    def __init__(self, event_durations_ms: list[float]) -> None:
        self.cuda = _FakeCuda(event_durations_ms)


def test_monitor_aggregates_actual_batch_sizes_and_writes_json(
    monkeypatch, tmp_path
):
    clock = _Clock()
    fake_torch = _FakeTorch([8.0, 12.0, 10.0])
    write_bytes = iter([100, 356])
    monkeypatch.setattr(monitor_module, "perf_counter", clock)
    monkeypatch.setattr(monitor_module, "_torch", fake_torch)
    monkeypatch.setattr(
        monitor_module, "_read_proc_io_write_bytes", lambda: next(write_bytes)
    )
    monkeypatch.setattr(monitor_module, "_read_proc_vm_hwm_bytes", lambda: 8192)

    report_path = tmp_path / "nested" / "monitor.json"
    monitor = ResourceMonitor(warmup=1, report_json=str(report_path))
    monitor.start()

    # The last batch is partial: one stereo frame (two views), not two frames.
    for duration_ms, stereo_frames in [(8.0, 2), (12.0, 2), (10.0, 1)]:
        with monitor.model_timer(stereo_frames):
            clock.advance_ms(duration_ms)
    for duration_ms, stereo_frames in [(20.0, 2), (24.0, 2), (20.0, 1)]:
        with monitor.e2e_timer(stereo_frames):
            clock.advance_ms(duration_ms)

    monitor.stop()
    report = monitor.report()

    model = report["timing"]["model"]
    assert model["sample_count"] == 3
    assert model["warmup_excluded"] == 1
    assert model["reported_sample_count"] == 2
    assert model["reported_stereo_frames"] == 3
    assert model["reported_views"] == 6
    assert model["per_batch_ms"] == pytest.approx(
        {"mean": 11.0, "median": 11.0, "p95": 11.9, "min": 10.0, "max": 12.0}
    )
    assert model["per_view_ms"] == pytest.approx(
        {"mean": 4.0, "median": 4.0, "p95": 4.9, "min": 3.0, "max": 5.0}
    )
    assert model["throughput_views_per_second"] == pytest.approx(6000.0 / 22.0)
    assert model["samples"][-1] == {
        "batch_ms": 10.0,
        "stereo_frames": 1,
        "views": 2,
        "per_view_ms": 5.0,
        "excluded_as_warmup": False,
    }

    e2e = report["timing"]["e2e"]
    assert e2e["per_batch_ms"]["mean"] == pytest.approx(22.0)
    assert e2e["per_view_ms"]["mean"] == pytest.approx(8.0)
    assert e2e["throughput_views_per_second"] == pytest.approx(6000.0 / 44.0)

    assert report["enabled"] is True
    assert report["total_wall_ms"] == pytest.approx(94.0)
    assert report["resources"] == {
        "peak_vram_bytes": 4096,
        "peak_ram_bytes": 8192,
        "disk_write_bytes": 256,
    }
    assert report["notes"] == []
    assert fake_torch.cuda.reset_calls == 1
    assert fake_torch.cuda.synchronize_calls == 3
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_monitor_retains_one_sample_and_degrades_without_cuda_or_proc(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(monitor_module, "perf_counter", clock)
    monkeypatch.setattr(monitor_module, "_torch", None)
    monkeypatch.setattr(monitor_module, "_read_proc_io_write_bytes", lambda: None)
    monkeypatch.setattr(monitor_module, "_read_proc_vm_hwm_bytes", lambda: None)

    monitor = ResourceMonitor(warmup=99)
    monitor.start()
    for milliseconds in (5.0, 7.0):
        with monitor.model_timer(1):
            clock.advance_ms(milliseconds)
        with monitor.e2e_timer(1):
            clock.advance_ms(milliseconds + 1.0)
    monitor.stop()
    report = monitor.report()

    for timing_name in ("model", "e2e"):
        timing = report["timing"][timing_name]
        assert timing["sample_count"] == 2
        assert timing["warmup_excluded"] == 1
        assert timing["reported_sample_count"] == 1
    assert report["timing"]["model"]["per_batch_ms"]["mean"] == pytest.approx(7.0)
    assert report["timing"]["e2e"]["per_batch_ms"]["mean"] == pytest.approx(8.0)
    assert report["resources"] == {}
    assert any("Peak VRAM unavailable" in note for note in report["notes"])
    assert any("CUDA event timing unavailable" in note for note in report["notes"])
    assert any("Peak RAM unavailable" in note for note in report["notes"])
    assert any("Disk write bytes unavailable" in note for note in report["notes"])


def test_disabled_monitor_is_a_true_no_op(monkeypatch, tmp_path, capsys):
    def fail_if_called():
        raise AssertionError("disabled monitor performed work")

    monkeypatch.setattr(monitor_module, "perf_counter", fail_if_called)
    monkeypatch.setattr(monitor_module, "_read_proc_io_write_bytes", fail_if_called)
    monkeypatch.setattr(monitor_module, "_read_proc_vm_hwm_bytes", fail_if_called)

    report_path = tmp_path / "unused" / "report.json"
    monitor = ResourceMonitor(enabled=False, report_json=str(report_path))
    monitor.start()
    with monitor.model_timer(0):
        pass
    with monitor.e2e_timer(0):
        pass
    monitor.stop()

    assert monitor.report() == {}
    assert not report_path.exists()
    assert capsys.readouterr().out == ""
