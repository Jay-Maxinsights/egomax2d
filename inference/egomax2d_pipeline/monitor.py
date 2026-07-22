"""Decoupled timing and process-resource monitoring for inference runs."""

from __future__ import annotations

import json
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

try:
    import torch as _torch
except ImportError:  # Keep CPU-only config and monitor tests lightweight.
    _torch = None


_STEREO_VIEWS = 2


@dataclass(frozen=True)
class _TimingSample:
    duration_ms: float
    stereo_frames: int

    @property
    def views(self) -> int:
        return self.stereo_frames * _STEREO_VIEWS


class ResourceMonitor:
    """Measure run-level resources and per-batch timing without pipeline coupling."""

    def __init__(
        self,
        enabled: bool = True,
        warmup: int = 10,
        report_json: str | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ValueError("warmup must be an integer >= 0")
        if report_json is not None and not isinstance(report_json, str):
            raise ValueError("report_json must be a string or null")

        self.enabled = enabled
        self.warmup = warmup
        self.report_json = report_json
        self._reset()

    def _reset(self) -> None:
        self._model_samples: list[_TimingSample] = []
        self._e2e_samples: list[_TimingSample] = []
        self._notes: list[str] = []
        self._wall_start: float | None = None
        self._total_wall_ms: float | None = None
        self._write_bytes_start: int | None = None
        self._vram_tracking = False
        self._peak_vram_bytes: int | None = None
        self._peak_ram_bytes: int | None = None
        self._disk_write_bytes: int | None = None
        self._started = False
        self._stopped = False

    def start(self) -> None:
        """Reset counters and begin monitoring the run."""
        if not self.enabled:
            return

        self._reset()
        self._started = True
        self._wall_start = perf_counter()
        self._write_bytes_start = _read_proc_io_write_bytes()
        if self._write_bytes_start is None:
            self._note(
                "Disk write bytes unavailable: /proc/self/io has no readable "
                "write_bytes field."
            )

        if not self._cuda_available():
            self._note("Peak VRAM unavailable: CUDA is not available.")
            return
        try:
            _torch.cuda.reset_peak_memory_stats()
            self._vram_tracking = True
        except Exception as exc:  # Monitoring must never abort inference.
            self._note(f"Peak VRAM unavailable: could not reset CUDA stats ({exc}).")

    @contextmanager
    def model_timer(self, stereo_frames: int) -> Iterator[None]:
        """Record model time for one batch containing ``stereo_frames`` pairs."""
        if not self.enabled:
            yield
            return

        stereo_frames = _require_stereo_frames(stereo_frames)
        wall_start = perf_counter()
        cuda_events = self._start_cuda_events()
        try:
            yield
        finally:
            wall_duration_ms = (perf_counter() - wall_start) * 1000.0
            duration_ms = self._finish_cuda_events(cuda_events, wall_duration_ms)
            self._model_samples.append(_TimingSample(duration_ms, stereo_frames))

    @contextmanager
    def e2e_timer(self, stereo_frames: int) -> Iterator[None]:
        """Record end-to-end time for one batch containing stereo frame pairs."""
        if not self.enabled:
            yield
            return

        stereo_frames = _require_stereo_frames(stereo_frames)
        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = (perf_counter() - start) * 1000.0
            self._e2e_samples.append(_TimingSample(duration_ms, stereo_frames))

    def stop(self) -> None:
        """Capture final wall-time and process-resource metrics."""
        if not self.enabled or self._stopped:
            return
        if not self._started or self._wall_start is None:
            self._note("Total wall time unavailable: monitor was stopped before start().")
            return

        self._total_wall_ms = (perf_counter() - self._wall_start) * 1000.0
        self._stopped = True

        if self._vram_tracking and self._cuda_available():
            try:
                self._peak_vram_bytes = int(_torch.cuda.max_memory_allocated())
            except Exception as exc:  # Monitoring must never abort inference.
                self._note(f"Peak VRAM unavailable: CUDA query failed ({exc}).")

        self._peak_ram_bytes = _read_proc_vm_hwm_bytes()
        if self._peak_ram_bytes is None:
            self._note(
                "Peak RAM unavailable: /proc/self/status has no readable VmHWM field."
            )

        write_bytes_stop = _read_proc_io_write_bytes()
        if self._write_bytes_start is not None and write_bytes_stop is not None:
            if write_bytes_stop >= self._write_bytes_start:
                self._disk_write_bytes = write_bytes_stop - self._write_bytes_start
            else:
                self._note(
                    "Disk write bytes unavailable: /proc/self/io write_bytes "
                    "decreased during the run."
                )
        elif self._write_bytes_start is not None:
            self._note(
                "Disk write bytes unavailable: /proc/self/io could not be read at stop."
            )

    def report(self) -> dict:
        """Print, optionally persist, and return the structured run report."""
        if not self.enabled:
            return {}

        report = {
            "enabled": True,
            "total_wall_ms": self._total_wall_ms,
            "resources": self._resource_report(),
            "timing": {
                "model": self._timing_report(self._model_samples),
                "e2e": self._timing_report(self._e2e_samples),
            },
            "notes": list(self._notes),
        }
        self._print_report(report)

        if self.report_json is not None:
            report_path = Path(self.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report

    def _cuda_available(self) -> bool:
        if _torch is None:
            return False
        try:
            return bool(_torch.cuda.is_available())
        except Exception as exc:  # Monitoring must never abort inference.
            self._note(f"CUDA metrics unavailable: CUDA availability query failed ({exc}).")
            return False

    def _start_cuda_events(self):
        if not self._cuda_available():
            self._note("CUDA event timing unavailable; model time uses perf_counter.")
            return None
        try:
            start_event = _torch.cuda.Event(enable_timing=True)
            end_event = _torch.cuda.Event(enable_timing=True)
            start_event.record()
            return start_event, end_event
        except Exception as exc:  # Monitoring must never abort inference.
            self._note(
                f"CUDA event timing unavailable ({exc}); model time uses perf_counter."
            )
            return None

    def _finish_cuda_events(self, events, wall_duration_ms: float) -> float:
        if events is None:
            return wall_duration_ms
        start_event, end_event = events
        try:
            end_event.record()
            _torch.cuda.synchronize()
            return float(start_event.elapsed_time(end_event))
        except Exception as exc:  # Monitoring must never abort inference.
            self._note(
                f"CUDA event timing failed ({exc}); model time uses perf_counter."
            )
            return wall_duration_ms

    def _resource_report(self) -> dict[str, int]:
        resources: dict[str, int] = {}
        if self._peak_vram_bytes is not None:
            resources["peak_vram_bytes"] = self._peak_vram_bytes
        if self._peak_ram_bytes is not None:
            resources["peak_ram_bytes"] = self._peak_ram_bytes
        if self._disk_write_bytes is not None:
            resources["disk_write_bytes"] = self._disk_write_bytes
        return resources

    def _timing_report(self, samples: list[_TimingSample]) -> dict:
        sample_count = len(samples)
        excluded = min(self.warmup, max(sample_count - 1, 0))
        reported = samples[excluded:]
        batch_ms = [sample.duration_ms for sample in reported]
        per_view_ms = [sample.duration_ms / sample.views for sample in reported]

        timing = {
            "sample_count": sample_count,
            "warmup_excluded": excluded,
            "reported_sample_count": len(reported),
            "reported_stereo_frames": sum(
                sample.stereo_frames for sample in reported
            ),
            "reported_views": sum(sample.views for sample in reported),
            "per_batch_ms": _statistics(batch_ms),
            "per_view_ms": _statistics(per_view_ms),
            "samples": [
                {
                    "batch_ms": sample.duration_ms,
                    "stereo_frames": sample.stereo_frames,
                    "views": sample.views,
                    "per_view_ms": sample.duration_ms / sample.views,
                    "excluded_as_warmup": index < excluded,
                }
                for index, sample in enumerate(samples)
            ],
        }
        total_duration_ms = math.fsum(batch_ms)
        if total_duration_ms > 0:
            timing["throughput_views_per_second"] = (
                1000.0 * timing["reported_views"] / total_duration_ms
            )
        return timing

    def _print_report(self, report: dict) -> None:
        print("\n=== EgoMax2D resource report ===")
        if report["total_wall_ms"] is not None:
            print(f"Total wall: {report['total_wall_ms']:.2f} ms")
        for name, value in report["resources"].items():
            print(f"{name}: {value} bytes")

        for name in ("model", "e2e"):
            timing = report["timing"][name]
            batch = timing["per_batch_ms"]
            per_view = timing["per_view_ms"]
            if not batch:
                print(f"{name}: no timing samples")
                continue
            print(
                f"{name}: mean {batch['mean']:.2f} ms/batch | "
                f"per-view {per_view['mean']:.2f} ms | "
                f"median {batch['median']:.2f} | p95 {batch['p95']:.2f} | "
                f"min {batch['min']:.2f} | max {batch['max']:.2f} "
                f"(warmup {timing['warmup_excluded']} excluded, "
                f"n={timing['reported_sample_count']})"
            )
        if report["notes"]:
            print("Notes:")
            for note in report["notes"]:
                print(f"- {note}")

    def _note(self, message: str) -> None:
        if message not in self._notes:
            self._notes.append(message)


def _require_stereo_frames(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("stereo_frames must be an integer >= 1")
    return value


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": math.fsum(values) / len(values),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _read_proc_vm_hwm_bytes() -> int | None:
    return _read_proc_field(Path("/proc/self/status"), "VmHWM", scale=1024)


def _read_proc_io_write_bytes() -> int | None:
    return _read_proc_field(Path("/proc/self/io"), "write_bytes", scale=1)


def _read_proc_field(path: Path, field: str, scale: int) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None

    prefix = f"{field}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        parts = line[len(prefix) :].strip().split()
        if not parts:
            return None
        try:
            value = int(parts[0])
        except ValueError:
            return None
        return value * scale if value >= 0 else None
    return None
