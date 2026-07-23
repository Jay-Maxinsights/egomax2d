# Design — Resource monitor (`monitor.py`)

## Requirements

- **Peak VRAM** and **peak RAM** — this process only.
- **Disk bytes written** by this process.
- **Time**: (1) per-batch/per-frame **model-inference** time, (2) per-batch/per-frame
  **E2E** time (preprocess + inference + decode), (3) total wall time.
- **Decoupled** from the pipeline: adding/changing a module (e.g. `FolderInputSource` →
  `MagicapInputSource`) must not require touching the monitor.

## Implementation — dependency-free on Linux

`psutil` is not installed and not needed; `/proc/self/*` is present (verified). Deploy
target is Linux.

| Metric | Source |
|---|---|
| Peak VRAM | `torch.cuda.reset_peak_memory_stats()` in `start()`; `torch.cuda.max_memory_allocated()` in `stop()` |
| Peak RAM | `VmHWM` from `/proc/self/status` in `stop()` — kernel-tracked peak RSS, **no sampler thread** |
| Disk written | `write_bytes` from `/proc/self/io`, delta `stop − start` |
| Model time | CUDA events + `torch.cuda.synchronize()` around the inference stage (per batch) |
| E2E time | `perf_counter` around the batch body (per batch) |
| Total wall | `perf_counter` across `start()`→`stop()` |

Per-frame = per-batch ÷ `(batch_size · 2)`. Exclude the first `warmup` batches from
timing stats. Reuse the mean/median/p95/min/max + per-image throughput formatting from
`batch_main` (`inference/inference_heatmap_egomax2d_dev.py`, lines 700–708).

Why `psutil` is not used: its `rss` is instantaneous (peak would need a sampler thread),
whereas `VmHWM` gives kernel-tracked peak RSS for free. Kept as an optional cross-platform
fallback only if a non-Linux environment ever appears.

## Interface

```python
class ResourceMonitor:
    def __init__(self, enabled: bool = True, warmup: int = 10, report_json: str | None = None): ...
    def start(self): ...                 # reset VRAM peak, record /proc baselines + wall t0
    @contextmanager
    def model_timer(self): ...           # records one per-batch model-ms (CUDA events)
    @contextmanager
    def e2e_timer(self): ...             # records one per-batch e2e-ms (perf_counter)
    def stop(self): ...                  # capture peak VRAM/RAM, disk delta, total wall
    def report(self) -> dict: ...        # print a table; dump JSON if report_json is set
```

## Decoupling guarantees

- `Pipeline` and `InputSource` contain **zero** monitoring code.
- VRAM / RAM / disk need **no** per-stage hooks — `start()`/`stop()` bracket the whole
  run and the numbers come from `/proc` + torch.
- Timing lives **only** as two `with` blocks in the orchestrator function
  (`run_sequential`, see [orchestrator.md](orchestrator.md)): `with monitor.e2e_timer()`
  around the batch body and `with monitor.model_timer()` around `inference`.
- When `enabled=False`, the context managers yield without recording and `start`/`stop`
  are no-ops — no separate code path in the pipeline.
- Consequences: swapping the input source or adding a module changes nothing here;
  adding a *stage* is at most one extra `with` line in the orchestrator.

## Notes / caveats

- Linux-specific (`/proc/self/status` VmHWM, `/proc/self/io` write_bytes). If a field is
  unavailable, skip that metric with a note rather than failing the run.
- Peak VRAM via `torch.cuda.max_memory_allocated` covers the torch allocator (not raw
  nvjpeg decode buffers).
- `VmHWM` is process-lifetime peak — fine here because the process *is* the run.
- The future `TaskManager` (feature 1) needs no monitor change: it already measures
  per-stage time, so it will report the overlap gains directly.
