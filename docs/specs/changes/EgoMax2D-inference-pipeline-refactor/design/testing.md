# Design — Test plan

Deliberately minimal — mostly new + cheap (no GPU), two reusing existing infra. Each test
maps to a build step and guards a real failure mode. Do not add tests beyond these unless
a gate failure demands a targeted one.

Tests live under `tests/` (pytest; the repo's tests are report-style and `pytest.skip`
when data/CUDA is absent). Add `tests/conftest.py` only if `sys.path` needs the repo root
— existing tests rely on `inference.*` / `pose_estimation.*` imports resolving, so keep
that working.

## 1. Config parser round-trip — `tests/test_pipeline_config.py`
- YAML string → `PipelineConfig` → assert nested values are correct.
- Assert a CLI override wins in `merge_cli`.
- No GPU / checkpoint. **Guards the control surface.**

## 2. `FolderInputSource` enumeration/pairing/batching — `tests/test_folder_input_source.py`
- Tmp fixture: `left/` has frames `{0,1,2,5}`, `right/` has `{0,1,5,9}` (a gap + mismatches).
- Assert `__iter__` yields only `{0,1,5}`, ascending, in `batch_size`-sized `FrameBatch`es
  including a **partial last batch**; assert `calibration()` returns the parsed dict.
- No GPU. **The one piece of genuinely new logic.**

## 3. `ResourceMonitor` timing math — `tests/test_monitor.py`
- Feed known durations/values into `e2e_timer` / `model_timer`.
- Assert per-batch / per-frame / total stats and **warmup exclusion** are correct.
- Assert `enabled=False` makes the timers no-ops.
- No GPU. **Guards the stats math** (the memory/disk *values* are non-deterministic and
  are NOT asserted — eyeballed in the smoke run).

## 4. Preprocess parity — reuse `tests/test_preprocess.py`
- `Pipeline.preprocess` (gpu path) must match `_dev.preprocess_gpu` / `preprocess`.
- Keep the existing test green (Step 1 keeps its imports valid). **Reuse, don't duplicate.**
- Add a CPU-only AST assertion in this file that examines both legacy inference scripts
  and fails on absolute or relative imports of `inference.egomax2d_pipeline`. This guard
  must run even when the data/CUDA parity case skips.

## 5. Acceptance gate — `.pt` parity — reuse `tests/test_prediction_diff.py`
- Produce `results/heatmap_egomax2d/<SID>_predictions.pt` with the NEW `main` on session
  `01KWEDQ9HG6CSF6CNW0QVFV92E`, and compare to the existing GT `.pt`
  (`results/heatmap_egomax2d_gt/<SID>_predictions.pt`) — or to a fresh `batch_main()` run
  at the same `--batch-size`.
- Expect **~0 relative diff**. Needs GPU + `data/EgoMax2D/<SID>` + checkpoint.
- **The single gate that proves the refactor changed nothing.**

## Deliberately NOT unit-tested (avoid over-testing)
- Numeric *values* of VRAM/RAM/disk (non-deterministic — verified via the smoke run).
- `decode` and `save_pt` in isolation — covered by the end-to-end parity gate.
- The model forward — unchanged code.
- Factory / CLI glue — exercised by the gate.

If the gate ever needs debugging, add a targeted `decode` test *then*, not preemptively.
