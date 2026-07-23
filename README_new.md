# EgoMax2D inference pipeline — how to run

The refactored stereo pose inference lives in `inference/egomax2d_pipeline/`. It is a
config-driven, composable replacement for `batch_main()` in
`inference/inference_heatmap_egomax2d_dev.py`, and produces a **byte-identical**
`predictions.pt`. Run it as a module from the **repository root**:

```bash
python -m inference.egomax2d_pipeline.main --help
```

## 1. Environment

You need a Python environment with **torch (CUDA build)** and **timm** installed. On this
machine, `sapien2_env` satisfies that (`posestudio` is missing `timm`):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sapien2_env
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Input layout

The pipeline reads two folders of completed JPEG frames plus a calibration file:

```
<left_dir>/frame_00000000.jpg     # 8-digit zero-padded index (left  camera / videoFL)
<right_dir>/frame_00000000.jpg    # same indices                (right camera / videoFR)
calibration.json                  # mcap-normalized calibration (sibling of the folders)
```

- Only frames present in **both** folders are used; the sorted intersection is processed.
- Indices may be non-contiguous — files are enumerated, never assumed to be `range(n)`.
- `--calibration` accepts a `calibration.json` path, a **session directory** (reads
  `calibration.json` inside it), or an inline dict (via YAML). There is **no** implicit
  session-directory coupling — you pass the left/right folders explicitly.

## 3. Quick start (CLI only)

Minimal run against an EgoMax2D session, writing `predictions.pt`:

```bash
SID=01KWEDQ9HG6CSF6CNW0QVFV92E
SESS=data/EgoMax2D/$SID

python -m inference.egomax2d_pipeline.main \
  --left-dir     $SESS/images/head-front-left \
  --right-dir    $SESS/images/head-front-right \
  --calibration  $SESS/calibration.json \
  --out          results/heatmap_egomax2d/${SID}_predictions.pt \
  --batch-size   32 \
  --device       cuda
```

The checkpoint defaults to
`work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt`; pass
`--ckpt <path>` to override.

> **Batch size:** on a 24 GB GPU, keep `--batch-size` around **32**. Larger sizes (e.g.
> the legacy default of 512) OOM during full-resolution JPEG decode (~14 GB for the decode
> stack alone). A 32 GB card can go higher.

### CLI options

| Flag | Overrides | Notes |
|---|---|---|
| `--config <yaml>` | — | Optional YAML config (see below). |
| `--left-dir <dir>` | `input.left_dir` | Left-camera frame folder. |
| `--right-dir <dir>` | `input.right_dir` | Right-camera frame folder. |
| `--calibration <path>` | `input.calibration` | `calibration.json`, session dir, or dict. |
| `--ckpt <path>` | `processor.ckpt` | Model checkpoint. |
| `--out <path>` | `output.path` | Output `predictions.pt` (parent dir auto-created). |
| `--batch-size <int>` | `input.batch_size` | Stereo frames per forward (model batch = ×2). |
| `--device <str>` | `processor.device` | e.g. `cuda`, `cuda:0`, `cpu`. |

Explicit CLI values **override** the YAML config. `--step`, `--max-frames`, and `--rotate`
are **not** CLI flags — set them in a YAML config (section 4).

`main()` returns a **nonzero exit code (2)** for invalid config or input (missing output
path, missing calibration, bad folder, etc.).

## 4. Config file

For anything beyond the CLI flags (frame sampling, rotation, preprocessing backend,
monitoring, JSON report), use a YAML config and pass it with `--config`. Full schema with
defaults:

```yaml
input:
  type: folder            # folder (only type implemented; magicap is future)
  left_dir:  ""           # or set via --left-dir
  right_dir: ""           # or set via --right-dir
  calibration: ""         # or set via --calibration
  batch_size: 32
  step: 1                 # frame sampling step (1 = every paired frame)
  max_frames: 0           # 0 = all; otherwise cap after step
processor:
  ckpt:   work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt
  device: cuda
  rotate: right           # right | left | none
  preprocess: gpu         # gpu (nvjpeg) | cpu (cv2)
  conf_thresh: 0.01
orchestrator:
  type: sequential        # sequential (only type implemented; taskmanager is future)
monitor:
  enabled: true
  warmup: 10              # first N batches excluded from timing stats
  report_json: null       # path to also dump the report as JSON, or null
output:
  format: pt              # pt (only format implemented; toon is future)
  path: ""                # or set via --out
```

Example — process the first 300 frames and dump a JSON report, with the paths supplied on
the CLI:

```yaml
# run.yaml
input:
  type: folder
  batch_size: 32
  step: 1
  max_frames: 300
processor:
  device: cuda
  rotate: right
  preprocess: gpu
monitor:
  enabled: true
  warmup: 9
  report_json: results/heatmap_egomax2d/report.json
output:
  format: pt
```

```bash
SID=01KWEDQ9HG6CSF6CNW0QVFV92E
SESS=data/EgoMax2D/$SID
python -m inference.egomax2d_pipeline.main \
  --config      run.yaml \
  --left-dir    $SESS/images/head-front-left \
  --right-dir   $SESS/images/head-front-right \
  --calibration $SESS/calibration.json \
  --out         results/heatmap_egomax2d/${SID}_predictions.pt
```

## 5. Output

`predictions.pt` is a `torch.save`d dict keyed by integer frame index (256×256 canvas
pixels; below-threshold joints are `[-1, -1]` with confidence `0`):

```python
{
  frame_idx: {
    "left":  {"joints": np.ndarray(26, 2) float32, "confidences": np.ndarray(26,) float32},
    "right": {"joints": np.ndarray(26, 2) float32, "confidences": np.ndarray(26,) float32},
  },
  ...
}
```

## 6. Resource report

When `monitor.enabled` is true, the run prints peak VRAM/RAM, disk-written bytes (when
available), per-batch and per-view model/E2E timing statistics, and total wall time. Set
`monitor.report_json` to also write the same report as a JSON file.

## 7. Parity gate (proving equivalence to the legacy path)

`tests/test_prediction_diff.py` asserts exact payload parity between two `predictions.pt`
files:

- `results/heatmap_egomax2d_gt/<SID>_predictions.pt` — the baseline (from `batch_main()`)
- `results/heatmap_egomax2d/<SID>_predictions.pt` — the new pipeline's output

Generate the baseline with the legacy script and the target with the new CLI at identical
params, then run the gate:

```bash
SID=01KWEDQ9HG6CSF6CNW0QVFV92E

# baseline (legacy, session-directory input)
python -m inference.inference_heatmap_egomax2d_dev \
  --session $SID --output-dir results/heatmap_egomax2d_gt \
  --step 1 --max-frames 300 --batch-size 32 --rotate right --device cuda

# target (new pipeline, explicit folder input) — use run.yaml with max_frames: 300
SID=01KWEDQ9HG6CSF6CNW0QVFV92E
python -m inference.egomax2d_pipeline.main --config run.yaml \
  --left-dir data/EgoMax2D/$SID/images/head-front-left \
  --right-dir data/EgoMax2D/$SID/images/head-front-right \
  --calibration data/EgoMax2D/$SID/calibration.json \
  --out results/heatmap_egomax2d/${SID}_predictions.pt

# gate: identical keys/shapes/dtypes, exact joints, 0 validity mismatches,
#       confidences equal within rtol=1e-6, atol=1e-7
python -m pytest tests/test_prediction_diff.py -q
```

## 8. Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` / `'timm'`** — wrong environment; use
  one with a CUDA torch build and timm (e.g. `sapien2_env`).
- **`CUDA out of memory` during preprocess** — lower `--batch-size` (32 is safe on 24 GB).
- **`Permission denied` writing the output** — the target directory may be owned by
  another user (e.g. created by a prior Docker run as root). Write to a directory you own,
  or fix ownership.
- **Exit code 2 with `error: invalid configuration/input`** — a required value is missing
  or a path does not exist; the message names the offending field.
