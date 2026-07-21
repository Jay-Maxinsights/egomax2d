# Commit 1 — establish the package and model boundary

## Objective

Create the new package and copy model constants/loading behind it without changing
inference behavior or introducing a legacy-to-new dependency.

## Changes

- [x] Add `inference/egomax2d_pipeline/__init__.py`, `configs/__init__.py`,
  `configs/constant.py`, and `model.py`.
- [x] Copy `IMG_SIZE`, `HM_SIZE`, `HM_SCALE`, `CONF_THRESH`, `CAMS`, `_ENCODER_CFG`,
  `MODEL_CFG`, and `DEFAULT_CKPT` verbatim into `configs/constant.py`.
- [x] Copy `load_model` into `model.py`, preserving checkpoint loading, removal of the
  `model.` prefix, strict state-dict loading, and `.eval()` behavior.
- [x] Keep both legacy inference scripts independent of the new package: do not import
  or re-export `inference.egomax2d_pipeline` from either script. Existing public imports
  and execution paths must continue to resolve.
- [x] Do not move or alter preprocessing, remapping, decoding, or inference flow in this
  commit.

## Verification

- [x] Import the new constants/model modules and the legacy preprocessing module in the
  same Python process.
- [x] Assert the new and compatibility paths expose the same constant values.
- [x] Run a one-time CPU-only AST assertion that fails if either legacy inference script
  directly or relatively imports `inference.egomax2d_pipeline`; do not add it to the
  permanent test suite.
- [x] Run `python -m pytest tests/test_preprocess.py -q` and confirm the dependency check
  passes and imports collect successfully. A data/CUDA skip is acceptable for the
  data-dependent parity case.

## Commit

`refactor(inference): establish egomax2d pipeline model package`
