# Commit 4 — add decoupled resource monitoring

**Depends on:** Commit 1. Apply after Commit 3 to keep a linear history.

## Objective

Measure required resources and timing without adding monitoring code to the input source
or processing pipeline.

## Changes

- [x] Add `monitor.py` with `start`, `model_timer`, `e2e_timer`, `stop`, and `report` as
  specified in `../design/resource-monitor.md`.
- [x] Record peak torch VRAM, `VmHWM`, `/proc/self/io` write-byte delta, per-batch model
  and E2E durations, and total wall time.
- [x] Associate every timing sample with its actual stereo frame/view count so a partial
  final batch is not divided by the configured full batch size.
- [x] Exclude at most `min(warmup, sample_count - 1)` samples so a non-empty run always
  retains at least one reportable timing sample.
- [x] Make missing CUDA or `/proc` fields omit the affected metric with a note rather
  than fail inference.
- [x] Make `enabled=False` a true no-op and create the parent directory when
  `report_json` is requested.

## Verification

- [x] Add `tests/test_monitor.py` using controlled clock/CUDA-event values.
- [x] Assert aggregation, per-batch/per-view math, partial-batch accounting, warmup
  exclusion, JSON shape, fallback behavior, and disabled mode. Do not assert live
  RAM/VRAM/disk values.
- [x] Run `python -m pytest tests/test_monitor.py -q`.

## Commit

`feat(inference): add decoupled resource monitoring`
