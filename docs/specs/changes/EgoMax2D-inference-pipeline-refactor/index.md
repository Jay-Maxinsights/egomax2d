# EgoMax2D Inference Pipeline Refactor — index

Index for this change folder. The EgoMax2D stereo pose inference is refactored from a
flat script (`batch_main()` in `inference/inference_heatmap_egomax2d_dev.py`) into small,
composable components under a new package `inference/egomax2d_pipeline/`, driven by one
config, with a decoupled resource logger. Behavior must stay **byte-identical** to
`batch_main()`.

- **Status:** Approved — planning complete, not yet implemented (see `proposal.md`).
- **Proposed:** 2026-07-21.

## Contents

- **[proposal.md](proposal.md)** — why the change, the goal, scope (in/out), and future
  work (sibling changes).
- **design/** — the design:
  - [architecture.md](design/architecture.md) — the four component concerns
    (`InputSource`, `Pipeline`, orchestrator, `ResourceMonitor`) and how config wires them.
  - [data-contracts.md](design/data-contracts.md) — input folder layout, `calibration.json`
    shape, and the unchanged `predictions.pt` output.
  - [components.md](design/components.md) — module layout, every class/function interface,
    and the config system (schema + parse→build).
  - [resource-monitor.md](design/resource-monitor.md) — the decoupled VRAM/RAM/disk/time
    logger.
  - [orchestrator.md](design/orchestrator.md) — the sequential run function with staged,
    commented CPU/GPU boundaries.
  - [testing.md](design/testing.md) — the minimal test plan and the acceptance gate.
- **[tasks.md](tasks.md)** — the incremental build order (checklist) + acceptance gate.
- **[reference.md](reference.md)** — map of existing code to reuse (paths + line numbers),
  gotchas, and assumptions.

## Suggested reading order

`proposal.md` → `design/architecture.md` → `design/data-contracts.md` →
`design/components.md` → `design/resource-monitor.md` → `design/orchestrator.md` →
`design/testing.md` → `tasks.md` (build) with `reference.md` open alongside.

## One-paragraph summary

Config (`configs/`) is parsed into plain-value dataclasses, then factories in `main.py`
build the wired components. An `InputSource` (now `FolderInputSource`) yields batched
frame references (`FrameBatch`) plus calibration; a `Pipeline` holds the model + remaps
and exposes three batch-stateless stages `preprocess` → `inference` → `decode`; a
sequential orchestrator function in `main.py` walks the stages per batch and collects an
in-memory `InferenceResult`, which `io/writer.py` serializes to the existing `.pt`
format. A `ResourceMonitor` measures peak VRAM/RAM, disk written, and per-batch/frame &
total timing, wired only at the orchestrator level so the pipeline components carry zero
monitoring code. The two future features — CPU/GPU overlap and streaming input — are each
a new config value plus one factory branch, with no change to the components.

## Reserved (implementation phase / future changes)

- Created when building (not now): `implementation-notes.md`, `results.md`.
- Future features become sibling change folders under `docs/specs/changes/`:
  `EgoMax2D-cpu-gpu-overlap/`, `EgoMax2D-streaming-input/`, `EgoMax2D-toon-output/`,
  `EgoMax2D-poseannotation-merge/`.
