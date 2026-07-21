# Commit 2 — add config and calibration loading

**Depends on:** Commit 1.

## Objective

Convert YAML plus optional CLI values into one validated, GPU-free configuration object.

## Changes

- [x] Add `InputConfig`, `ProcessorConfig`, `OrchestratorConfig`, `MonitorConfig`,
  `OutputConfig`, and `PipelineConfig` in `configs/schema.py`, using safe dataclass
  defaults for nested objects.
- [x] Implement `PipelineConfig.from_yaml` and `merge_cli` in `configs/utils.py`.
  Unspecified CLI arguments must preserve YAML values; explicitly supplied values must
  win.
- [x] Implement `load_calibration(path_or_dir_or_dict)`. Delegate directory parsing to
  `load_session_calib`; keep JSON-file parsing identical to it; pass normalized dicts
  through after validating `videoFL` and `videoFR`.
- [x] Raise actionable `ValueError`/`FileNotFoundError` messages for malformed YAML,
  missing calibration files, zero-byte JSON, unsupported calibration shapes, and unknown
  top-level config sections.
- [x] Keep `configs/` free of torch, model, and GPU imports.

## Verification

- [x] Add `tests/test_pipeline_config.py` covering a YAML round trip, one CLI override,
  dict/directory/file calibration inputs, and one malformed-calibration case. Avoid
  exhaustive tests of dataclass defaults.
- [x] Run `python -m pytest tests/test_pipeline_config.py -q`.

## Commit

`feat(inference): add pipeline config and calibration loading`
