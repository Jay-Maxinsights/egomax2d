"""Config parsing, CLI merging, and calibration loading utilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .schema import (
    InputConfig,
    MonitorConfig,
    OrchestratorConfig,
    OutputConfig,
    PipelineConfig,
    ProcessorConfig,
)


_ConfigT = TypeVar("_ConfigT", bound=PipelineConfig)

_SECTION_TYPES = {
    "input": InputConfig,
    "processor": ProcessorConfig,
    "orchestrator": OrchestratorConfig,
    "monitor": MonitorConfig,
    "output": OutputConfig,
}

_CLI_OVERRIDES = {
    "left_dir": ("input", "left_dir"),
    "right_dir": ("input", "right_dir"),
    "calibration": ("input", "calibration"),
    "batch_size": ("input", "batch_size"),
    "ckpt": ("processor", "ckpt"),
    "device": ("processor", "device"),
    "out": ("output", "path"),
}

_CAMERAS = ("videoFL", "videoFR")
_DS_FIELDS = ("fx", "fy", "cx", "cy", "xi", "alpha")


# -----------------------------------------------------------------------------
# 1. Other important part
# Core YAML parsing, CLI merging, and calibration loading used by the pipeline.
# -----------------------------------------------------------------------------


def _pipeline_config_from_yaml(
    path: str | PathLike[str], config_type: type[_ConfigT] = PipelineConfig
) -> _ConfigT:
    """Implementation backing :meth:`PipelineConfig.from_yaml`."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Pipeline config path is not a file: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read pipeline config {config_path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"Pipeline config YAML is empty: {config_path}")

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed pipeline config YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Pipeline config {config_path} must contain a top-level mapping"
        )

    unknown_sections = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown_sections:
        raise ValueError(
            "Unknown top-level pipeline config section(s): "
            + ", ".join(map(str, unknown_sections))
        )

    section_values: dict[str, Any] = {}
    for section_name, section_type in _SECTION_TYPES.items():
        section_raw = raw.get(section_name, {})
        if section_raw is None:
            section_raw = {}
        if not isinstance(section_raw, Mapping):
            raise ValueError(
                f"Pipeline config section {section_name!r} must be a mapping"
            )
        allowed_fields = {item.name for item in fields(section_type)}
        unknown_fields = sorted(set(section_raw) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"Unknown field(s) in pipeline config section {section_name!r}: "
                + ", ".join(map(str, unknown_fields))
            )
        try:
            section_values[section_name] = section_type(**dict(section_raw))
        except TypeError as exc:
            raise ValueError(
                f"Invalid values in pipeline config section {section_name!r}: {exc}"
            ) from exc

    config = config_type(**section_values)
    _validate_config(config)
    return config


def merge_cli(config: _ConfigT, args: Any) -> _ConfigT:
    """Return a config with explicitly supplied CLI values applied.

    ``None`` means argparse did not receive that option. Falsy values such as ``0``
    and ``False`` remain explicit overrides.
    """
    if not isinstance(config, PipelineConfig):
        raise TypeError(f"config must be a PipelineConfig, got {type(config).__name__}")

    def cli_value(name: str) -> Any:
        if isinstance(args, Mapping):
            return args.get(name)
        return getattr(args, name, None)

    merged: PipelineConfig = config
    for cli_name, (section_name, field_name) in _CLI_OVERRIDES.items():
        value = cli_value(cli_name)
        if value is None:
            continue
        section = getattr(merged, section_name)
        merged = replace(
            merged,
            **{section_name: replace(section, **{field_name: value})},
        )

    _validate_config(merged)
    return merged  # type: ignore[return-value]


def load_calibration(
    path_or_dir_or_dict: str | PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Load calibration from a normalized mapping, directory, or JSON file."""
    if isinstance(path_or_dir_or_dict, Mapping):
        _validate_normalized_calibration(path_or_dir_or_dict, "calibration mapping")
        return path_or_dir_or_dict  # type: ignore[return-value]

    if not isinstance(path_or_dir_or_dict, (str, PathLike)):
        raise ValueError(
            "Calibration input must be a normalized dict, directory, or JSON file path"
        )

    path = Path(path_or_dir_or_dict)
    if not path.exists():
        raise FileNotFoundError(f"Calibration path does not exist: {path}")

    if path.is_dir():
        calibration_file = path / "calibration.json"
        _validate_calibration_file_path(calibration_file)
        try:
            calibration = _load_session_calibration(path, calibration_file)
        except (OSError, KeyError, IndexError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Malformed or unsupported calibration file {calibration_file}: {exc}"
            ) from exc
        _validate_normalized_calibration(calibration, str(calibration_file))
        return calibration

    if not path.is_file():
        raise ValueError(f"Calibration path is not a regular file: {path}")
    _validate_calibration_file_path(path)
    calibration = _load_calibration_json(path)
    _validate_normalized_calibration(calibration, str(path))
    return calibration


def _load_session_calibration(
    session_dir: Path, calibration_file: Path
) -> dict[str, Any]:
    """Use the repository helper without making config imports depend on torch.

    Importing ``pose_estimation.datasets.egomax2d.remap`` executes parent package
    initializers that import torch. Minimal config-only environments do not install
    torch, so the identical file parser is the fallback in that case.
    """
    try:
        from pose_estimation.datasets.egomax2d.remap import load_session_calib
    except ModuleNotFoundError:
        return _load_calibration_json(calibration_file)
    return load_session_calib(str(session_dir))


def _load_calibration_json(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed calibration JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read calibration file {path}: {exc}") from exc

    try:
        calibrations = document["calibrations"]
        if not isinstance(calibrations, list) or not calibrations:
            raise TypeError("'calibrations' must be a non-empty list")
        calibration_text = calibrations[0]["calibration_text"]
        if not isinstance(calibration_text, str) or not calibration_text.strip():
            raise TypeError("'calibration_text' must be a non-empty YAML string")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unsupported calibration shape in {path}; expected "
            "calibrations[0].calibration_text"
        ) from exc

    try:
        parsed = yaml.safe_load(calibration_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed calibration_text YAML in {path}: {exc}") from exc
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("intrinsics"), list):
        raise ValueError(
            f"Unsupported calibration_text shape in {path}; expected an intrinsics list"
        )

    calibration: dict[str, dict[str, Any]] = {}
    for item in parsed["intrinsics"]:
        if not isinstance(item, Mapping):
            continue
        camera_name = item.get("camera_name")
        if camera_name not in _CAMERAS:
            continue
        params = item.get("params")
        if (
            not isinstance(params, Sequence)
            or isinstance(params, (str, bytes))
            or len(params) != len(_DS_FIELDS)
        ):
            raise ValueError(
                f"Calibration camera {camera_name!r} in {path} must contain six DS params"
            )
        calibration[str(camera_name)] = dict(zip(_DS_FIELDS, params))
    return calibration


# -----------------------------------------------------------------------------
# 2. Format check related
# Kept separate because this pipeline has a static input/config format. These
# defensive checks can be reduced or removed later without changing the core flow.
# -----------------------------------------------------------------------------


def _validate_config(config: PipelineConfig) -> None:
    _require_nonempty_string(config.input.type, "input.type")
    _require_nonempty_string(config.processor.ckpt, "processor.ckpt")
    _require_nonempty_string(config.processor.device, "processor.device")
    _require_nonempty_string(config.processor.rotate, "processor.rotate")
    _require_nonempty_string(config.processor.preprocess, "processor.preprocess")
    _require_nonempty_string(config.orchestrator.type, "orchestrator.type")
    _require_nonempty_string(config.output.format, "output.format")

    _require_int_at_least(config.input.batch_size, "input.batch_size", 1)
    _require_int_at_least(config.input.step, "input.step", 1)
    _require_int_at_least(config.input.max_frames, "input.max_frames", 0)
    _require_int_at_least(config.monitor.warmup, "monitor.warmup", 0)

    if not isinstance(config.monitor.enabled, bool):
        raise ValueError("monitor.enabled must be a boolean")
    if config.monitor.report_json is not None and not isinstance(
        config.monitor.report_json, str
    ):
        raise ValueError("monitor.report_json must be a string or null")
    if not isinstance(config.processor.conf_thresh, Real) or isinstance(
        config.processor.conf_thresh, bool
    ):
        raise ValueError("processor.conf_thresh must be a finite non-negative number")
    if (
        not math.isfinite(float(config.processor.conf_thresh))
        or config.processor.conf_thresh < 0
    ):
        raise ValueError("processor.conf_thresh must be a finite non-negative number")
    if not isinstance(config.input.calibration, (str, PathLike, Mapping)):
        raise ValueError("input.calibration must be a path or normalized mapping")


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_int_at_least(value: Any, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_calibration_file_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Calibration path is not a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Could not inspect calibration file {path}: {exc}") from exc
    if size == 0:
        raise ValueError(f"Calibration file is empty: {path}")


def _validate_normalized_calibration(
    calibration: Mapping[str, Any], source: str
) -> None:
    missing_cameras = [camera for camera in _CAMERAS if camera not in calibration]
    if missing_cameras:
        raise ValueError(
            f"{source} is missing required camera calibration(s): "
            + ", ".join(missing_cameras)
        )

    for camera in _CAMERAS:
        params = calibration[camera]
        if not isinstance(params, Mapping):
            raise ValueError(f"{source} camera {camera!r} must be a mapping")
        missing_fields = [field for field in _DS_FIELDS if field not in params]
        if missing_fields:
            raise ValueError(
                f"{source} camera {camera!r} is missing DS parameter(s): "
                + ", ".join(missing_fields)
            )
        for field in _DS_FIELDS:
            value = params[field]
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"{source} camera {camera!r} parameter {field!r} "
                    "must be a finite number"
                )
