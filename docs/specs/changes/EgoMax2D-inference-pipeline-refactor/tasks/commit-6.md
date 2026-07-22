# Commit 6 — add sequential execution and result writing

**Depends on:** Commits 3–5.

## Objective

Execute the three stages in order and preserve the exact in-memory and `.pt` output
contracts before adding CLI concerns.

## Changes

- [x] Add `io/writer.py` with `InferenceResult` and `save_pt`. Preserve integer frame
  keys, left/right names, NumPy dtypes and shapes, and confidence arrays.
- [x] Create the parent output directory when one is present. A filename in the current
  directory must also work.
- [x] Add `run_sequential` to `main.py` with explicit preprocess, inference, and decode
  boundaries under `torch.no_grad()`.
- [x] Keep timing brackets in the orchestrator, pass actual batch cardinality to the
  monitor, and guarantee `monitor.stop()` runs through `try/finally` if a stage raises.
- [x] Keep `Pipeline`, `InputSource`, and `InferenceResult` free of monitor references.

## Verification

- [x] Add one lightweight `tests/test_sequential_orchestrator.py` test using a fake
  source, pipeline, and monitor.
- [x] Assert stage order, non-contiguous indices, partial-batch cardinality passed to the
  monitor, returned result shape, and start/stop bracketing.
- [x] In the same test, use a temporary-path save/load round trip to cover serialization
  without creating a standalone writer suite.
- [x] Run `python -m pytest tests/test_sequential_orchestrator.py -q`.
- [x] Rerun `python -m pytest tests/test_pipeline_config.py tests/test_folder_input_source.py tests/test_monitor.py -q`.

## Commit

`feat(inference): add sequential execution and pt writer`
