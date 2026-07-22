# Implementation logs

> Convention (see repo `CLAUDE.md`): log only what fell outside the plan, or bugs hit
> during implementation. Do not restate tasks from `tasks/commit-N.md`, and do not
> document the same thing twice. A commit with no deviations needs no entry.

## commit-1-logs

- Git commit: `0510d28 refactor(inference): establish egomax2d pipeline model package`.
- Constants and `load_model` were copied into the new package instead of removed from
  `inference_heatmap_egomax2d_dev.py`. The legacy script does not import the new package,
  keeping it independent for the final end-to-end comparison.
- A one-time AST script confirmed that neither legacy inference script imports
  `inference.egomax2d_pipeline`. The check was intentionally not added to the permanent
  test suite.
- Importing the new and legacy modules together and comparing their shared constants
  passed. `tests/test_preprocess.py` collected successfully; its data/CUDA-dependent
  parity case skipped in the available environment.

## commit-2-logs

- Git commit: `b396009 feat(inference): add pipeline config and calibration loading`.
- Added plain-value dataclass configuration, YAML loading, and explicit CLI override
  merging. Nested dataclasses use independent default factories.
- Added normalized calibration loading for dict, session-directory, and JSON-file
  inputs. The directory path delegates to `load_session_calib` when its dependencies are
  available and uses the identical JSON/YAML parser in a PyTorch-free environment.
- Kept core loading behavior separate from format-validation helpers in `configs/utils.py`
  so the static pipeline can simplify or remove defensive format checks later.
- `python -m pytest tests/test_pipeline_config.py -q`: `5 passed` in both the base and
  `posestudio` environments.

## commit-3-logs

- Git commit: `f40b45f feat(inference): add paired folder input source`.
- Added the `FrameBatch` and `InputSource` contracts plus a `FolderInputSource` that
  normalizes calibration once and snapshots paired frame paths during construction.
- Frame discovery accepts only exact `frame_%08d.jpg` names, intersects and sorts the
  two camera index sets, then applies `step`, `max_frames`, and batching in order. The
  final partial batch preserves left/right/index alignment.
- Added clear failures for invalid folders and sampling values, and for sources with no
  paired frames.
- `python -m pytest tests/test_folder_input_source.py -q`: `8 passed` in the base
  environment. The RTX 5090 Docker image ran the config, folder-input, and legacy CUDA
  preprocess tests together with `14 passed` and no skips.

## commit-4-logs

- Git commit: `6c13855 feat(inference): add decoupled resource monitoring`.
- Added a decoupled `ResourceMonitor` with CUDA-event model timing, `perf_counter` E2E
  and wall timing, torch peak-VRAM tracking, Linux `VmHWM`, and `/proc/self/io`
  write-byte deltas.
- Every model and E2E sample retains its actual stereo-frame and derived view count, so
  partial batches produce correct per-view statistics and throughput. Warmup exclusion
  always retains one sample when timing data is non-empty.
- Missing CUDA and `/proc` metrics are omitted with report notes; model timing falls
  back to `perf_counter`. Disabled monitoring performs no timing, resource, output, or
  validation work.
- Reports include raw samples, mean/median/p95/min/max aggregates, resource values, and
  optional JSON output whose parent directory is created automatically.
- `python -m pytest tests/test_monitor.py -q`: `3 passed` in the base environment using
  controlled clock, CUDA-event, and resource values. The combined config, folder-input,
  and monitor regression run completed with `16 passed`.

## commit-6-logs

- `writer.py` had to be kept out of `io/__init__.py`. It imports torch, and
  `io/__init__.py` runs on any `io.reader` import, which must stay torch-free so
  `test_folder_input_source` still collects in the base (no-torch) environment.
- `design/orchestrator.md`'s sketch calls `monitor.e2e_timer()` / `model_timer()` with no
  arguments, but the monitor implemented in commit-4 requires `stereo_frames`, so
  `run_sequential` passes `len(batch.indices)`.
