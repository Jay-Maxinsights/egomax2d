# Commit 3 — implement paired folder input

**Depends on:** Commit 2.

## Objective

Produce deterministic `FrameBatch` objects from two completed image folders.

## Changes

- [x] Add `io/__init__.py` and `io/reader.py` with `FrameBatch`, `InputSource`, and
  `FolderInputSource` using the interfaces in `../design/components.md`.
- [x] Parse only `frame_%08d.jpg`, intersect the left/right index sets, sort ascending,
  then apply `step`, `max_frames`, and batching in that order.
- [x] Yield a partial final batch and keep `indices`, `left`, and `right` aligned. Never
  infer indices with `range(len(folder))`.
- [x] Load and normalize calibration once during source construction and return that
  same normalized object from `calibration()`.
- [x] Reject nonexistent folders, `batch_size < 1`, and `step < 1` with clear messages.
  An empty left/right intersection must fail rather than silently produce no work.

## Verification

- [x] Add `tests/test_folder_input_source.py` using left indices `{0,1,2,5}` and right
  indices `{0,1,5,9}`.
- [x] Assert sorted intersection, aligned paths, full plus partial batches, `step`,
  `max_frames`, and normalized calibration. Use a small parameterized test rather than
  separate tests for every permutation.
- [x] Run `python -m pytest tests/test_folder_input_source.py -q`.

## Commit

`feat(inference): add paired folder input source`
