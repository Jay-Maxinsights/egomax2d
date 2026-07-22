# Commit 7 — wire the CLI and prove parity

**Depends on:** Commits 1–6.

## Objective

Deliver the standalone PoseAnnotation-compatible command and make output parity a real
pass/fail acceptance gate.

## Changes

- [x] Add `build_input_source`, `build_pipeline`, `build_monitor`, and `build_writer` to
  `main.py`. Each factory must implement only its current type and raise a clear
  `ValueError` for future or unknown types.
- [x] Add argparse support for `--config`, `--left-dir`, `--right-dir`, `--calibration`,
  `--ckpt`, `--out`, `--batch-size`, and `--device`. Explicit CLI values override YAML.
- [x] Make `main()` follow parse → build → run → write → report and return a nonzero exit
  for invalid config or input.
- [x] Support execution from the repository root with
  `python -m inference.egomax2d_pipeline.main`.
- [x] Upgrade existing `tests/test_prediction_diff.py` from a report-only test to an
  asserting test. Require identical frame/view keys, array shapes, and dtypes; exact
  decoded joint coordinates; zero validity mismatches; and confidence equality within
  `rtol=1e-6, atol=1e-7`.

## Verification

- [x] Run `python -m inference.egomax2d_pipeline.main --help`.
- [x] Generate baseline and new outputs for session
  `01KWEDQ9HG6CSF6CNW0QVFV92E` using identical `step`, `max_frames`, `rotate`, device,
  checkpoint, and batch size.
- [x] Feed the new command the explicit left folder, right folder, and
  `calibration.json`; do not reintroduce session-directory coupling.
- [x] Run `python -m pytest tests/test_prediction_diff.py -q` and confirm it executes
  rather than skips.
- [x] Run all six gates together.
- [x] Confirm the CLI writes `predictions.pt`; prints peak VRAM/RAM and disk-written
  metrics when available; prints model/E2E per-batch and per-view statistics plus total
  wall time; and writes valid JSON when configured.

## Commit

`feat(inference): expose config-driven cli with parity gate`
