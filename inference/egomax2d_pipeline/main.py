"""Sequential orchestrator for EgoMax2D stereo pose inference.

``run_sequential`` walks the ``preprocess`` -> ``inference`` -> ``decode`` stages
of a :class:`~inference.egomax2d_pipeline.pipeline.Pipeline` over every batch a
source yields, collecting an :class:`~inference.egomax2d_pipeline.io.writer.InferenceResult`.
Monitor brackets are the only instrumentation and live here, not in the pipeline
components; the three commented stage boundaries are exactly where the future
``TaskManager`` will parallelize CPU (preprocess/decode) against GPU (inference).
The config-driven CLI and factories are added in a later commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .io.writer import InferenceResult

if TYPE_CHECKING:  # Type-only imports keep this module light for now.
    from .io.reader import InputSource
    from .monitor import ResourceMonitor
    from .pipeline import Pipeline


def run_sequential(
    source: "InputSource",
    pipeline: "Pipeline",
    monitor: "ResourceMonitor",
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
