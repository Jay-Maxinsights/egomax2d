# Reference — existing code, gotchas, assumptions

All paths relative to the repo root `/home/max/jay/HIL_pose_annotation/egomax2d`. Treat
these symbols as the compatibility source. Commit 1 intentionally copies only the model
constants and loader into the new package; do not create further implementations when a
stable lower-level `pose_estimation.*` module or one-way legacy import is available.
Dependency flow is one-way: neither legacy inference script may import
`inference.egomax2d_pipeline`. Line numbers are as of 2026-07-21 — confirm before
editing.

## Existing code to reuse

### `inference/inference_heatmap_egomax2d_dev.py`
- **Constants** (lift into `configs/constant.py`):
  `IMG_SIZE=256` (L71), `HM_SIZE=64` (L72), `HM_SCALE=IMG_SIZE/HM_SIZE=4.0` (L73),
  `CONF_THRESH=0.01` (L74),
  `CAMS=[("head_front_left","head-front-left"),("head_front_right","head-front-right")]` (L81).
- **Model config** (L51–63): `_ENCODER_CFG = dict(type="vit",
  model_name="vit_base_patch16_224.augreg_in21k", pretrained=False, img_size=256,
  out_stride=4, out_channels=128, neck_mid_channels=256, drop_path_rate=0.1,
  grad_checkpointing=False, weights_path=None)`;
  `MODEL_CFG = dict(num_heatmap=26, encoder_cfg=_ENCODER_CFG, train_cfg=dict(w_heatmap=10.0))`.
  `DEFAULT_CKPT = "work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt"` (L65–68).
- **`load_model(ckpt_path) -> EgoPoseFormerHeatmap`** (L224–235): `EgoPoseFormerHeatmap(**MODEL_CFG)`,
  `torch.load(map_location="cpu")`, strip `"model."` prefix from `ckpt["state_dict"]`,
  `load_state_dict(strict=True)`, `.eval()`.
- **`preprocess_gpu(image_paths, maps, device="cuda", raw_wh=(2592,1944)) -> Tensor(N,3,256,256)`**
  (L183–219): `torchvision.io.decode_jpeg` on GPU (RGB) → `F.grid_sample` (grid from each
  `(map_x,map_y)` normalized `2*g/(dim-1)-1`) → BT.601 gray (`_GRAY_W=[0.299,0.587,0.114]`, L180)
  → expand 3ch → `/255`. The fast path `batch_main` uses.
- **`batch_preprocess(image_dirs, maps) -> Tensor(N,3,256,256)`** (L152–176): CPU
  `cv2.imread`+`cv2.remap`+gray+`/255`.
- **`remap_preprocess(bgr, map_x, map_y) -> (canvas, tensor)`** (L120–129).
- **`decode_heatmap(hm) -> (joints(26,2) f32, confs(26,))`** (L238–251): per-joint argmax
  over the `64×64` map; `peak < CONF_THRESH` → `[-1,-1]`/`0`; else `[flat%64*4.0, flat//64*4.0]`/`peak`.
- **`resolve_session(root, session, session_idx)`** (L390–398) — only if a session-dir
  convenience source is wanted; the episode input does not use it.
- **Remap build** in `batch_main` (L606–613): `calib = load_session_calib(session_dir)`;
  left `build_remap(dict(_EGOBODY3M_DS_PARAMS[1]), calib["videoFL"], rotate)`,
  right `build_remap(dict(_EGOBODY3M_DS_PARAMS[2]), calib["videoFR"], rotate)`.
- **Forward** (L662–663): `feats = model.forward_backbone(img_t)` with `img_t (B,2,3,256,256)`;
  `hm = model.conv_heatmap(feats.view(B*2, *feats.shape[2:]))` → `(B*2,26,64,64)` → `.cpu().numpy()`.
- **Predictions assembly** (L675–681), save (L684–685).
- **Timing** (reuse math/formatting): CUDA events (L656–667), `perf_counter` e2e (L653/671),
  warmup `warm = min(timing_warmup, len-1)` (L695), stats print (L700–708).

### `pose_estimation/datasets/egomax2d/remap.py`
- `IMG_SIZE=256` (L33).
- `KP2JOINT` (L37–41): `{"left_elbow":3,"right_elbow":7,"left_shoulder":2,"right_shoulder":6,"pelvis":10}`
  (only needed for the future toon writer).
- `SIDE_SPECS` (L45–50): left=`videoFL`/`head-front-left`/`head_front_left`/`eb_cam=1`;
  right=`videoFR`/`head-front-right`/`head_front_right`/`eb_cam=2`.
- `load_session_calib(session_dir) -> {"videoFL":{...},"videoFR":{...}}` (L100–110). Takes a
  **directory**, reads `calibration.json` inside it. Wrap for path/dict input (see
  `design/components.md` → `load_calibration`).
- `build_remap(eb_params, src_params, rotate="right") -> (map_x, map_y) f32` (L113–129).
  **EgoBody3M params first, session calib second.**
- `build_session_remaps(session_dir, rotate="right")` (L132–139).

### `pose_estimation/models/estimator/egoposeformer_heatmap.py`
- `EgoPoseFormerHeatmap(nn.Module)` (L10); `self.conv_heatmap = nn.Conv2d(enc_out_ch, num_heatmap, 1)` (L24).
- `forward_backbone(img)` (L34–38): `img [B,V,C,H,W]` → `feats [B,V,C,64,64]`. Inference
  calls `forward_backbone` then `conv_heatmap` directly (not `forward()`).

### `pose_estimation/models/utils/camera_models.py`
- `_EGOBODY3M_DS_PARAMS` (L199): keyed by cam id `0,1,2,3`; fields `fx,fy,cx,cy,xi,alpha,
  R,t,native_wh,square,pad_top`. Cam 1 & 2: `native_wh=(1280.0,1024.0)`, `square=1280.0`,
  `pad_top=128.0`.

### Existing tests
- `tests/test_preprocess.py` — imports `preprocess, preprocess_gpu` from
  `inference.inference_heatmap_egomax2d_dev` (L25); `build_session_remaps` from remap;
  `test_preprocess_relative_diff(capsys)` (L88). **Keep those imports valid.**
- `tests/test_prediction_diff.py` — `test_prediction_relative_diff(capsys)` (L78); globals
  (L25–29) `GT_PT = results/heatmap_egomax2d_gt/01KWEDQ9HG6CSF6CNW0QVFV92E_predictions.pt`,
  `TARGET_PT = results/heatmap_egomax2d/01KWEDQ9HG6CSF6CNW0QVFV92E_predictions.pt`. This is
  the parity harness for the acceptance gate.

## Gotchas (carry into implementation)

- **Create the output dir** in `save_pt` — `batch_main` never did (`os.makedirs` was
  commented out), so callers currently rely on the dir pre-existing.
- **Non-contiguous frame indices** — enumerate files, intersect left ∩ right, sort. Never
  assume `range(n)`.
- **0-byte `calibration.json`** ships in some sessions → `load_*` raises; surface a clear
  error, don't crash opaquely.
- **`inference/` is a namespace package** (tests import
  `inference.inference_heatmap_egomax2d_dev`). New package `inference/egomax2d_pipeline/`
  gets its own `__init__.py`; run the CLI as
  `python -m inference.egomax2d_pipeline.main` from the repo root.
- **`build_remap` arg order** — `build_remap(eb_params, src_params, rotate)`: EgoBody3M
  params first, session calib second. Left→cam 1, right→cam 2. `rotate` default `"right"`.
- **Batch reshape** — preprocess returns `(B,2,3,256,256)`; heatmaps come back
  `(B*2,26,64,64)`; `hm[i*2]`=left, `hm[i*2+1]`=right for `batch.indices[i]`.

## Assumptions

- Calibration is the mcap-normalized `calibrations[0].calibration_text` YAML with
  `videoFL`/`videoFR` DS intrinsics (what `load_session_calib` reads). mkv/mp4
  `calibration_json` shape is out of scope.
- Monitor is Linux-specific (`/proc/self/status` VmHWM, `/proc/self/io` write_bytes) —
  matches the deploy target; degrade gracefully (skip a metric with a note) if a field is
  unavailable. `psutil` deliberately not used (VmHWM gives free peak RSS).
- Peak VRAM = torch allocator peak (not raw nvjpeg buffers); VmHWM = process-lifetime peak
  (fine — the process *is* the run).
- Environment: `requirements.txt` has torch/torchvision, `opencv-python`, `PyYAML`,
  `numpy`; no `psutil`. Python with dataclasses / `str | None` syntax (3.10+).
