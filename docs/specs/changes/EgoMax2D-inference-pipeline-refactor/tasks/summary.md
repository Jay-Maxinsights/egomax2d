# Tasks — implementation summary

Implement the commits below in order. Each `commit-<i>.md` file defines one reviewable
git commit and its verification gate. Complete every checkbox in that file before
starting the next commit.

## Commit sequence

| Commit | Scope | Depends on | Primary gate |
|---|---|---|---|
| [1](commit-1.md) | Package boundary, constants, model loader | — | Legacy preprocess import/test |
| [2](commit-2.md) | Config schema, YAML/CLI merge, calibration adapter | 1 | `test_pipeline_config.py` |
| [3](commit-3.md) | Paired folder input and deterministic batching | 2 | `test_folder_input_source.py` |
| [4](commit-4.md) | Decoupled resource monitor | 1 | `test_monitor.py` |
| [5](commit-5.md) | Calibration-bound pipeline stages | 1, 3 | Existing `test_preprocess.py` |
| [6](commit-6.md) | Sequential orchestration and `.pt` writer | 3–5 | `test_sequential_orchestrator.py` |
| [7](commit-7.md) | Factories, CLI, and end-to-end parity | 1–6 | Existing `test_prediction_diff.py` |

## Shared execution rules

- Do not edit `inference/inference_heatmap_egomax2d.py` or the PoseAnnotation repo.
- Keep `inference/inference_heatmap_egomax2d_dev.py` import-compatible, especially
  `preprocess` and `preprocess_gpu`, because existing tests import them directly.
- Enforce a one-way dependency boundary: code under `inference/egomax2d_pipeline/` may
  import legacy helpers, but neither legacy inference script may import or re-export the
  new package.
- Reuse the symbols identified in `../reference.md`, preferring stable lower-level
  modules before legacy-script imports; do not create second implementations of
  calibration parsing, remapping, preprocessing, or heatmap decoding.
- A skipped GPU/data test is useful during development but does not complete its commit.
  The preprocessing and final parity gates must execute with the required CUDA device,
  checkpoint, calibration, and session data.
- Do not implement speculative support for `magicap`, `taskmanager`, or `toon`. Their
  factory branches must fail clearly until sibling changes implement them.
- Keep each commit limited to its named scope. If a gate reveals a necessary exception,
  record it later in the reserved `../implementation-notes.md`.

## Minimal test budget

| Gate | Test | Purpose |
|---|---|---|
| 1 | `tests/test_pipeline_config.py` | Config parsing, CLI precedence, calibration loading |
| 2 | `tests/test_folder_input_source.py` | Frame pairing, ordering, sampling, batching |
| 3 | `tests/test_monitor.py` | Timing math, warmup, disabled/fallback behavior |
| 4 | Existing `tests/test_preprocess.py` | `Pipeline.preprocess` versus the legacy fast path |
| 5 | `tests/test_sequential_orchestrator.py` | One cheap orchestration contract test with fakes |
| 6 | Existing `tests/test_prediction_diff.py` | Asserting end-to-end payload parity gate |

Do not unit-test the unchanged model forward or every factory branch. Do not add
standalone decode or writer suites unless the end-to-end gate exposes a defect that
cannot be localized with the tests above.

## Definition of done

- [ ] All seven commits are individually reviewable and their stated gates pass.
- [ ] The final `.pt` payload satisfies Gate 6 against `batch_main()` at the same batch
  size. Compare payload values, not serialized `torch.save` archive bytes.
- [ ] The resource report contains the required metrics or an explicit availability note
  for platform-dependent metrics.
- [ ] No future-feature implementation, PoseAnnotation edit, or change to
  `inference/inference_heatmap_egomax2d.py` is included.
- [ ] Any implementation-time deviation is recorded later in the reserved
  `../implementation-notes.md`; do not expand the planning scope unless a gate proves it
  necessary.
