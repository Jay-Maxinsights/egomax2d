"""Plain-value configuration schema for the EgoMax2D inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from .constant import CONF_THRESH, DEFAULT_CKPT


CalibrationInput = str | PathLike[str] | dict[str, Any]


@dataclass
class InputConfig:
    type: str = "folder"
    left_dir: str = ""
    right_dir: str = ""
    calibration: CalibrationInput = ""
    batch_size: int = 32
    step: int = 1
    max_frames: int = 0


@dataclass
class ProcessorConfig:
    ckpt: str = DEFAULT_CKPT
    device: str = "cuda"
    rotate: str = "right"
    preprocess: str = "gpu"
    conf_thresh: float = CONF_THRESH


@dataclass
class OrchestratorConfig:
    type: str = "sequential"


@dataclass
class MonitorConfig:
    enabled: bool = True
    warmup: int = 10
    report_json: str | None = None


@dataclass
class OutputConfig:
    format: str = "pt"
    path: str = ""


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> "PipelineConfig":
        """Parse and validate a YAML configuration file.

        Parsing lives in ``configs.utils`` so this module remains a small schema-only
        dependency for callers that only need the dataclasses.
        """
        from .utils import _pipeline_config_from_yaml

        return _pipeline_config_from_yaml(path, config_type=cls)
