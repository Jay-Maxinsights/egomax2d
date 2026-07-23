# Design — Sequential orchestrator (`main.py`)

The orchestrator is a plain function for now. Each stage is broken off by a **concise
comment** so the CPU/GPU split is obvious — and those boundaries are exactly where the
future `TaskManager` (feature 1) will parallelize. The monitor brackets are the only
instrumentation and live here, not in `Pipeline`.

```python
def run_sequential(source: InputSource, pipeline: Pipeline,
                   monitor: ResourceMonitor) -> InferenceResult:
    result = InferenceResult()
    monitor.start()                              # reset VRAM peak, /proc baselines, wall t0
    with torch.no_grad():
        for batch in source:
            with monitor.e2e_timer():            # E2E per batch (preprocess + infer + decode)
                # preprocess — CPU/decode (remap to 256 canvas, grayscale)
                tensors = pipeline.preprocess(batch)
                # inference — GPU (backbone + heatmap head)
                with monitor.model_timer():      # model-only time (CUDA events)
                    heatmaps = pipeline.inference(tensors)
                # decode — CPU (argmax -> joints/confidences, merged into result)
                pipeline.decode(heatmaps, batch, result)
    monitor.stop()                               # peak VRAM/RAM, disk delta, total wall
    return result
```

## Entrance (`main()`)

```python
def main():
    # 1. parse:   args = argparse(...); cfg = PipelineConfig.from_yaml(args.config); cfg = merge_cli(cfg, args)
    # 2. build:   calib  = load_calibration(cfg.input.calibration)
    #             source = build_input_source(cfg.input)
    #             pipe   = build_pipeline(cfg.processor, calib)
    #             mon    = build_monitor(cfg.monitor)
    #             write  = build_writer(cfg.output)
    # 3. run:     result = run_sequential(source, pipe, mon)
    # 4. write:   write(result, cfg.output.path)      # save_pt
    # 5. report:  mon.report()
```

Run it as `python -m inference.egomax2d_pipeline.main` from the repo root.

## Why the orchestrator is a function (not a class) now

The user's requirement: the orchestrator "right now should be just a sequential
function." Keeping it a function makes the sequential path trivial to read and keeps the
scheduling concern separate from the stages. Feature 1 introduces a `TaskManager`
(selected by `orchestrator.type: taskmanager`) that reuses the identical `Pipeline`
methods and `InputSource`; the sequential function stays as the simple baseline.

## Where future parallelism slots in

The three commented stages map directly to a 3-stage software pipeline:
`preprocess(N+1)` (CPU) ∥ `inference(N)` (GPU) ∥ `decode(N-1)` (CPU), connected by bounded
queues. Because the stages are batch-stateless and the monitor already times each, the
`TaskManager` is an additive change — no edits to `Pipeline`, `InputSource`, or the
monitor.
