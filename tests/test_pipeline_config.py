"""CPU-only tests for pipeline configuration and calibration loading."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest
import yaml

from inference.egomax2d_pipeline.configs.schema import PipelineConfig
from inference.egomax2d_pipeline.configs.utils import load_calibration, merge_cli


def _normalized_calibration() -> dict:
    params = {"fx": 1000.0, "fy": 1001.0, "cx": 1296.0, "cy": 972.0,
              "xi": -0.1, "alpha": 0.6}
    return {"videoFL": dict(params), "videoFR": dict(params, fx=1002.0)}


def _calibration_document(calibration: dict) -> dict:
    intrinsics = []
    for camera_name in ("videoFL", "videoFR"):
        camera = calibration[camera_name]
        intrinsics.append(
            {
                "camera_name": camera_name,
                "model_type": "ds",
                "params": [camera[key] for key in ("fx", "fy", "cx", "cy", "xi", "alpha")],
            }
        )
    return {
        "calibrations": [
            {"calibration_text": yaml.safe_dump({"intrinsics": intrinsics})}
        ]
    }


def test_pipeline_config_yaml_round_trip_and_cli_override(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
input:
  type: folder
  left_dir: /yaml/left
  right_dir: /yaml/right
  calibration: /yaml/calibration.json
  batch_size: 16
  step: 2
processor:
  ckpt: /yaml/model.ckpt
  device: cuda:1
monitor:
  enabled: true
  warmup: 3
output:
  path: /yaml/predictions.pt
""",
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(config_path)
    assert config.input.left_dir == "/yaml/left"
    assert config.input.batch_size == 16
    assert config.input.step == 2
    assert config.processor.ckpt == "/yaml/model.ckpt"
    assert config.processor.device == "cuda:1"
    assert config.monitor.warmup == 3
    assert config.output.path == "/yaml/predictions.pt"

    merged = merge_cli(
        config,
        Namespace(
            left_dir=None,
            right_dir="/cli/right",
            calibration=None,
            ckpt=None,
            out=None,
            batch_size=8,
            device=None,
        ),
    )
    assert merged.input.left_dir == "/yaml/left"
    assert merged.input.right_dir == "/cli/right"
    assert merged.input.batch_size == 8
    assert merged.processor.ckpt == "/yaml/model.ckpt"
    assert merged.output.path == "/yaml/predictions.pt"
    assert config.input.right_dir == "/yaml/right"


def test_load_calibration_from_dict_directory_and_file(tmp_path):
    expected = _normalized_calibration()
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(_calibration_document(expected)), encoding="utf-8"
    )

    assert load_calibration(expected) is expected
    assert load_calibration(tmp_path) == expected
    assert load_calibration(calibration_path) == expected


def test_load_calibration_rejects_empty_file(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.touch()

    with pytest.raises(ValueError, match="empty"):
        load_calibration(calibration_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("input: [", "Malformed pipeline config YAML"),
        ("unexpected: {}", "Unknown top-level pipeline config section"),
    ],
)
def test_pipeline_config_rejects_invalid_yaml(tmp_path, text, message):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_yaml(config_path)
