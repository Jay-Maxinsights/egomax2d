# Implementation logs

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
