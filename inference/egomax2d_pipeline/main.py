"""Sequential orchestrator and config-driven CLI for EgoMax2D stereo pose inference.

``run_sequential`` walks the ``preprocess`` -> ``inference`` -> ``decode`` stages
of a :class:`~inference.egomax2d_pipeline.pipeline.Pipeline` over every batch a
source yields, collecting an :class:`~inference.egomax2d_pipeline.io.writer.InferenceResult`.
Monitor brackets are the only instrumentation and live here, not in the pipeline
components; the three commented stage boundaries are exactly where the future
``TaskManager`` will parallelize CPU (preprocess/decode) against GPU (inference).

``main`` implements the parse -> build -> run -> write -> report flow. Factories
build one component each and are the single place a future variant (``magicap``
input, ``taskmanager`` orchestrator, ``toon`` output) plugs in; each implements
only its current type and raises a clear ``ValueError`` otherwise. Run it from the
repository root with ``python -m inference.egomax2d_pipeline.main``.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Callable

import torch

from .configs.schema import (
    InputConfig,
    MonitorConfig,
    OrchestratorConfig,
    OutputConfig,
    PipelineConfig,
    ProcessorConfig,
)
from .configs.utils import load_calibration, merge_cli
from .io.reader import FolderInputSource, InputSource
from .io.writer import InferenceResult, save_pt
from .monitor import ResourceMonitor
from .pipeline import Pipeline

if TYPE_CHECKING:  # Only needed for annotations.
    from typing import Any


def run_sequential(
    source: InputSource,
    pipeline: Pipeline,
    monitor: ResourceMonitor,
) -> InferenceResult:
    """Run every batch through the stages in order and collect the result.

    ``monitor.stop()`` is guaranteed to run even if a stage raises, so a partial
    run still reports the resources it consumed before failing.
    """
    result = InferenceResult()
    monitor.start()  # reset VRAM peak, /proc baselines, wall t0
    try:
        with torch.no_grad():
            for batch in source:
                # Actual batch cardinality (last batch may be partial).
                stereo_frames = len(batch.indices)
                with monitor.e2e_timer(stereo_frames):
                    # preprocess — CPU/decode (remap to 256 canvas, grayscale)
                    tensors = pipeline.preprocess(batch)
                    # inference — GPU (backbone + heatmap head)
                    with monitor.model_timer(stereo_frames):
                        heatmaps = pipeline.inference(tensors)
                    # decode — CPU (argmax -> joints/confidences, merged in place)
                    pipeline.decode(heatmaps, batch, result)
    finally:
        monitor.stop()  # peak VRAM/RAM, disk delta, total wall
    return result


# -----------------------------------------------------------------------------
# Factories — each builds one component and is the single place a future variant
# plugs in. Every factory implements only its current type and raises a clear
# ValueError for future or unknown types.
# -----------------------------------------------------------------------------


def build_input_source(config: InputConfig) -> InputSource:
    """Build the input source selected by ``input.type``."""
    if config.type == "folder":
        return FolderInputSource(
            left_dir=config.left_dir,
            right_dir=config.right_dir,
            calibration=config.calibration,
            batch_size=config.batch_size,
            step=config.step,
            max_frames=config.max_frames,
        )
    raise ValueError(
        f"Unsupported input.type {config.type!r}; only 'folder' is implemented "
        "(future: 'magicap')"
    )


def build_pipeline(config: ProcessorConfig, calibration: "dict[str, Any]") -> Pipeline:
    """Build the calibration-bound processing pipeline."""
    if config.preprocess not in ("gpu", "cpu"):
        raise ValueError(
            f"Unsupported processor.preprocess {config.preprocess!r}; only 'gpu' "
            "and 'cpu' are implemented"
        )
    return Pipeline(
        calibration=calibration,
        ckpt=config.ckpt,
        device=config.device,
        rotate=config.rotate,
        preprocess=config.preprocess,
        conf_thresh=config.conf_thresh,
    )


def build_monitor(config: MonitorConfig) -> ResourceMonitor:
    """Build the decoupled resource monitor."""
    return ResourceMonitor(
        enabled=config.enabled,
        warmup=config.warmup,
        report_json=config.report_json,
    )


def build_writer(config: OutputConfig) -> Callable[[InferenceResult, str], None]:
    """Build the result writer selected by ``output.format``."""
    if config.format == "pt":
        return save_pt
    raise ValueError(
        f"Unsupported output.format {config.format!r}; only 'pt' is implemented "
        "(future: 'toon')"
    )


def _run_orchestrator(
    config: OrchestratorConfig,
    source: InputSource,
    pipeline: Pipeline,
    monitor: ResourceMonitor,
) -> InferenceResult:
    """Dispatch to the orchestrator selected by ``orchestrator.type``."""
    if config.type == "sequential":
        return run_sequential(source, pipeline, monitor)
    raise ValueError(
        f"Unsupported orchestrator.type {config.type!r}; only 'sequential' is "
        "implemented (future: 'taskmanager')"
    )


# -----------------------------------------------------------------------------
# CLI — parse -> build -> run -> write -> report.
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.egomax2d_pipeline.main",
        description="Config-driven EgoMax2D stereo pose inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, help="path to a YAML pipeline config (optional)"
    )
    # Explicit overrides win over YAML. Defaults are None so merge_cli can tell an
    # omitted option (None) from a falsy-but-explicit one (e.g. --batch-size 0 is
    # rejected by validation rather than silently ignored).
    parser.add_argument("--left-dir", default=None, help="left-camera frame folder")
    parser.add_argument("--right-dir", default=None, help="right-camera frame folder")
    parser.add_argument(
        "--calibration",
        default=None,
        help="calibration.json path, session directory, or inline dict",
    )
    parser.add_argument("--ckpt", default=None, help="model checkpoint path")
    parser.add_argument("--out", default=None, help="output predictions.pt path")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="stereo frames per model forward (model batch = batch_size * 2)",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    return parser


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    """Parse YAML (if any) and apply explicit CLI overrides."""
    if args.config is not None:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    return merge_cli(config, args)


def main(argv: "list[str] | None" = None) -> int:
    """Entrance: parse -> build -> run -> write -> report. Returns an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 1. parse (invalid config -> nonzero exit)
    try:
        config = _load_config(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    if not config.output.path:
        print(
            "error: invalid configuration: no output path; set output.path or --out",
            file=sys.stderr,
        )
        return 2

    # 2. build (invalid input/component -> nonzero exit)
    try:
        calibration = load_calibration(config.input.calibration)
        source = build_input_source(config.input)
        pipeline = build_pipeline(config.processor, calibration)
        monitor = build_monitor(config.monitor)
        writer = build_writer(config.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: invalid input: {exc}", file=sys.stderr)
        return 2

    # 3. run
    result = _run_orchestrator(config.orchestrator, source, pipeline, monitor)

    # 4. write
    writer(result, config.output.path)
    print(f"Predictions -> {config.output.path}")

    # 5. report
    monitor.report()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrance
    raise SystemExit(main())
