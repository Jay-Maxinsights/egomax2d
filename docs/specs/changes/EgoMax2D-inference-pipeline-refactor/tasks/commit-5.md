# Commit 5 — implement the processing stages

**Depends on:** Commits 1 and 3.

## Objective

Expose the existing numerical path as three calibration-bound, batch-stateless methods.

## Changes

- [x] Add `pipeline.py` with `Pipeline.__init__`, `preprocess`, `inference`, and `decode`
  matching `../design/components.md`.
- [x] Import `batch_preprocess`, `preprocess_gpu`, and `decode_heatmap` from
  `inference.inference_heatmap_egomax2d_dev` into the new package. Do not move those
  helpers or add any import of the new package to either legacy inference script.
- [x] Build the two remaps and load the model exactly once. Preserve remap argument order:
  left is EgoBody3M camera 1 plus `videoFL`; right is camera 2 plus `videoFR`.
- [x] Flatten inputs as `[f0_left, f0_right, f1_left, f1_right, ...]`, call the existing
  GPU or CPU preprocessing implementation, and reshape to `(B, 2, 3, 256, 256)`.
- [x] Preserve the model forward and heatmap reshape exactly, returning
  `(B*2, 26, 64, 64)` on CPU as NumPy.
- [x] Decode `hm[i*2]` as left and `hm[i*2+1]` as right and store both under the original
  non-contiguous frame index.
- [x] Keep the confidence threshold configurable without changing its default behavior.
  Do not add monitoring or cross-batch state.

## Verification

- [x] Update existing `tests/test_preprocess.py` to construct a `FrameBatch` and call
  `Pipeline.preprocess` explicitly.
- [x] Monkeypatch model loading so the test measures preprocessing rather than checkpoint
  startup, then compare the tensor and view ordering against legacy `preprocess_gpu`.
- [x] Run `python -m pytest tests/test_preprocess.py -q` and confirm it executes rather
  than skips.

The unchanged model forward and decode path are covered by Commit 7's end-to-end parity
gate rather than duplicate unit tests.

## Commit

`feat(inference): add staged egomax2d processor`
