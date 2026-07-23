# Proposal: EgoMax2D Inference Pipeline Refactor

| | |
|---|---|
| **Proposed** | 2026-07-21 |
| **Status** | Approved — planning complete, not yet implemented |
| **Owner** | Jay Hong |
| **Change dir** | `docs/specs/changes/EgoMax2D-inference-pipeline-refactor/` |

## Why

Today the EgoMax2D stereo pose inference lives as one flat procedural function,
`batch_main()` in `inference/inference_heatmap_egomax2d_dev.py` (lines 584–709). It
resolves a local session directory, loads `calibration.json`, builds calibration
remaps, loads the ViT heatmap model, loops over batches (GPU-decode + remap → forward →
decode heatmaps), `torch.save`s a `predictions.pt`, and prints an inline timing report —
all fused together.

This inference will become a **stage inside the PoseAnnotation pipeline**
(`/home/max/jay/HIL_pose_annotation/PoseAnnotation/run_pipeline.sh`), a bash orchestrator
of standalone Python CLI scripts that communicate over the local filesystem (download
frames+calibration from R2 → *[our inference here]* → upload results to R2 → mark done in
DynamoDB), and will eventually be **merged into that repo for cloud deployment**.

The flat script cannot cleanly plug into that pipeline, and it blocks two features we
know are coming. So we refactor it into small, composable, individually testable
components driven by one config, keeping behavior **byte-identical** to `batch_main()`.

## What (goal)

Refactor the inference into a staged, composable pipeline under a new package
`inference/egomax2d_pipeline/`, such that it:

- Takes a simple, explicit input: **calibration data + a left image folder + a right
  image folder** (matching PoseAnnotation's post-download episode layout; no session-dir
  coupling).
- Produces the **unchanged `.pt` predictions** output.
- Is driven by **one config** (parse YAML/CLI → build components).
- Reports **resources** — peak VRAM/RAM, disk written, per-batch/frame model & E2E time,
  total wall — via a logger **decoupled** from the pipeline.
- Plugs into `run_pipeline.sh` in a few lines.
- Is architected so two future features drop in without a rewrite.

Origin: the work began from the request to "make the current inference pipeline into a
reusable abstract component that can be part of a larger cloud pipeline, with images +
calibration as the input." Investigation of the target pipeline showed it integrates by
CLI + filesystem (not by a Python base class), so the design landed on **composable
components + a config-driven CLI**, not an abstract `Stage` hierarchy.

## Scope

**In scope**
- New package `inference/egomax2d_pipeline/` (see `design/`).
- `InputSource` seam with a `FolderInputSource` implementation.
- `Pipeline` with `preprocess` / `inference` / `decode` stages.
- Sequential orchestrator function + config system + decoupled `ResourceMonitor`.
- `.pt` output unchanged; parity vs `batch_main()` is the acceptance gate.
- Strict one-way dependency boundary: the new package may import existing helpers from
  the legacy inference scripts when needed, but neither legacy script may import
  `inference.egomax2d_pipeline`.

**Out of scope (do NOT build; only leave the seam)**
- CPU/GPU overlap orchestrator (`TaskManager`) — future feature 1.
- Streaming `MagicapInputSource` — future feature 2.
- Toon output writer; the PoseAnnotation merge; mkv/mp4 `calibration_json` parsing.
- **Do not edit `run_pipeline.sh` or touch the PoseAnnotation repo in this change.**

**Do not touch**
- `inference/inference_heatmap_egomax2d.py` (baseline parity reference).
- Keep `inference_heatmap_egomax2d_dev.py` importable — `tests/test_preprocess.py`
  imports from it.
- Do not add imports of `inference.egomax2d_pipeline` to either legacy inference script.

## Future work (separate changes / phases)

These become **sibling change folders** under `docs/specs/changes/` when picked up:

- `EgoMax2D-cpu-gpu-overlap/` — feature 1: overlap CPU (preprocess/decode) with GPU
  (inference) via a `TaskManager` reusing the same `Pipeline` methods
  (`orchestrator.type: taskmanager`).
- `EgoMax2D-streaming-input/` — feature 2: `MagicapInputSource` that yields batches as
  frames download/decode live (`input.type: magicap`).
- `EgoMax2D-toon-output/` — emit the `estimations.toon` format (`output.format: toon`).
- `EgoMax2D-poseannotation-merge/` — move the slim inference package into PoseAnnotation
  and wire it into `run_pipeline.sh` for cloud deploy.

## Document map

- `design/` — the design (architecture, data contracts, components, config, resource
  monitor, orchestrator, testing).
- `tasks.md` — incremental build steps + acceptance gate.
- `reference.md` — existing-code reuse map, gotchas, assumptions.
