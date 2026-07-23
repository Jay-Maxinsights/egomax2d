# Design — Data contracts (exact)

These are the exact shapes the pipeline reads and writes. Do not deviate — the output
shape in particular is consumed unchanged by `tests/test_prediction_diff.py`.

## Input — folder layout

What PoseAnnotation's download step (`grab_synced_frames.py`) produces, and what
`FolderInputSource` reads:

```
<left_dir>/frame_00000000.jpg     # 8-digit zero-padded index, JPEG (quality 90)
<right_dir>/frame_00000000.jpg    # left_dir  == ".../head-front-left"
calibration.json                  # right_dir == ".../head-front-right"
                                  # calibration.json is a sibling of the two folders
```

- Frame index comes from the filename `frame_%08d.jpg`.
- Indices may be **non-contiguous** — some frames are skipped upstream (unsyncable /
  decode-failed). **Enumerate files; never assume `range(n)`.**
- A frame is usable only if it exists in **both** folders. Take the intersection of the
  two folders' indices, sorted ascending.

## Input — `calibration.json` (mcap-normalized; the only shape supported)

```json
{
  "calibrations": [
    {
      "calibration_time": "...",
      "calibration_text": "intrinsics:\n- camera_name: videoFL\n  model_type: ds\n  params: [fx, fy, cx, cy, xi, alpha]\n  resolution: [2592, 1944]\n  rate: 30.0\n- camera_name: videoFR\n  ...\nextrinsics_reference_cam: imu\nextrinsics: ..."
    }
  ]
}
```

- `calibration_text` is a **YAML string**. `intrinsics` is a list; we use the entries
  with `camera_name` `videoFL` and `videoFR`.
- `params` order is `[fx, fy, cx, cy, xi, alpha]` (Double Sphere).
- `pose_estimation/datasets/egomax2d/remap.py::load_session_calib` already parses this
  into `{"videoFL": {fx,fy,cx,cy,xi,alpha}, "videoFR": {...}}`.
- **Not supported:** the mkv/mp4 path emits `calibrations[0].calibration_json` (a nested
  dict) instead of `calibration_text` — out of scope for this change.

## Output — `predictions.pt` (unchanged)

`torch.save(predictions, path)` where `predictions` is:

```python
predictions: dict[int, dict] = {
    frame_idx: {
        "left":  {"joints": np.ndarray(26, 2) float32, "confidences": np.ndarray(26,) float32},
        "right": {"joints": np.ndarray(26, 2) float32, "confidences": np.ndarray(26,) float32},
    },
    ...
}
```

- `frame_idx` is the integer frame index.
- Joints are in **256-canvas pixels** (exactly what `decode_heatmap` returns: below
  `CONF_THRESH` → `[-1, -1]`, `conf 0`; otherwise `[flat%64 * 4.0, flat//64 * 4.0]`,
  `conf = peak`).
- 26 joints per view (full model output, not the 5-keypoint subset — that subset only
  matters for the future toon writer).

This is byte-for-byte the shape produced by `batch_main()` (lines 675–685 of
`inference/inference_heatmap_egomax2d_dev.py`), which is why the acceptance gate can diff
the two `.pt` files directly.
