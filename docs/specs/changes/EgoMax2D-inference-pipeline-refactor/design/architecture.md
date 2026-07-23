# Design — Architecture

Config drives everything; four component concerns are cleanly separated.

## 1. `InputSource` (`io/reader.py`)

The abstract contract for **"give me input batches."** No matter where inputs come from,
they simply *flow in* as a uniform stream; the rest of the pipeline neither knows nor
cares about the origin. Each subclass only defines **how frames flow in**:

- `FolderInputSource` → reads frames off disk (two folders, all present up front). Built
  in this change.
- `MagicapInputSource` *(future, feature 2)* → yields frames as they are downloaded /
  decoded live.

Both satisfy the same contract: `__iter__` → `FrameBatch`, plus `calibration()`. The
source also owns **batching policy** — a folder source always fills a full batch; a
streaming source may emit a partial batch when the download stalls rather than block.
That decision lives inside the source and is invisible downstream.

This seam exists **solely** to enable feature 2: swapping the source is the only change
needed to go from "batch a finished folder" to "stream while downloading."

## 2. `Pipeline` (`pipeline.py`)

The model + calibration-bound work, split into independent stages. Holds the remaps
(built once from calibration) and the loaded model. Three methods, each **stateless
across batches** (shared read-only remaps/model), so a scheduler can run them in any
order:

- `preprocess(FrameBatch) -> Tensor(B, 2, 3, 256, 256)` — CPU/decode (remap to the 256
  canvas, grayscale ×3).
- `inference(Tensor) -> heatmaps(B*2, 26, 64, 64)` — GPU (backbone + heatmap head).
- `decode(heatmaps, FrameBatch, result)` — CPU (argmax → joints/confidences).

(Named `Pipeline` after the file; it was called "Processor" during design discussion.)

## 3. Orchestrator (a function in `main.py`)

**Now:** a plain sequential function `run_sequential(source, pipeline, monitor)` that
walks the three stages per batch and collects an `InferenceResult`. See
[orchestrator.md](orchestrator.md).

**Swappable seam (future, feature 1):** a `TaskManager` runs the *same* `Pipeline`
methods asynchronously with bounded queues between stages, overlapping
`preprocess(N+1)` / `inference(N)` / `decode(N-1)`. No change to `Pipeline` or
`InputSource` — selected via `orchestrator.type: taskmanager` in config.

## 4. `ResourceMonitor` (`monitor.py`)

Decoupled observability — peak VRAM/RAM, disk written, per-batch/frame model & E2E time,
total wall. It carries **zero** coupling into the pipeline components (see
[resource-monitor.md](resource-monitor.md)).

## Config ties it together (`configs/`)

- **Parser** (`configs/utils.py`): YAML/CLI → `PipelineConfig` (plain-value dataclasses).
- **Factories** (`main.py`): `PipelineConfig` → wired components
  (`build_input_source`, `build_pipeline`, `build_monitor`, `build_writer`).

Nothing is constructed ad hoc; each factory is the single place a future variant plugs
in. Serialization (`io/writer.py`) is separate from inference — the run returns an
in-memory `InferenceResult`, and `save_pt()` writes the current `.pt` format (a toon
writer is a later swap via `output.format`).

## Data flow

```
InputSource ──FrameBatch──▶ Pipeline.preprocess ─Tensor─▶ Pipeline.inference
   (calibration())              (CPU)                        (GPU)
                                                               │ heatmaps
                                                               ▼
                                                     Pipeline.decode ──▶ InferenceResult
                                                        (CPU)                  │
                                                                               ▼
                                                                    io/writer.save_pt → .pt

ResourceMonitor brackets the whole run (start/stop) and wraps the inference &
batch-body timing at the orchestrator level only.
```

## Why not an abstract `Stage`/`Pipeline` base class?

The target PoseAnnotation pipeline integrates stages as **standalone CLI scripts
communicating over the filesystem** (bash orchestration), not by importing a Python
interface. A `Stage` ABC would be dead scaffolding. The abstraction that earns its keep
is `InputSource` (it enables streaming); everything else is plain composable classes plus
a config-driven CLI.
