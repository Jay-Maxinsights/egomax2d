# EgoMax2D Release — 2D Keypoint Estimation (ViT-Based Model)

Release repository for the EgoMax2D 5-keypoint 2D estimation model (head-mounted forward-facing
stereo, in-the-wild). The model is a ViT-B/16 heatmap network (86.8M parameters), pretrained on
EgoBody3M and domain-adapted to EgoMax2D human annotations via fine-tuning.

**Held-out test (8 sessions, including outdoor scenes): 5.49 px mean keypoint error (on the 256 canvas)**

| Metric | Overall | L-elbow | R-elbow | L-shoulder | R-shoulder | Pelvis |
|---|---|---|---|---|---|---|
| px error | 5.49 | 4.1 | 5.0 | 6.1 | 6.4 | 6.5 |

Inference speed (H100, 256×256 stereo pair, fp32, unoptimized): 4.82 ms/pair GPU forward
(2.41 ms/image), 5.53 ms/pair end-to-end including copies ≈ 181 pairs/s — about 6× real-time
headroom against a 30 fps input.

Full dataset statistics and annotation taxonomy: `analysis/max2d.md`.

## Repository Layout

```
egomax2d_release
|-- configs
|   |-- egomax2d_vit_heatmap_ft.yaml   # fine-tune training recipe (for further fine-tuning)
|   `-- egomax2d_eval.yaml             # evaluation-only config (step 3)
|-- pose_estimation                    # Python package (installed via pip install -e .)
|   |-- datasets/egomax2d              # dataset + Double Sphere ray-level remap module
|   |-- models                         # ViT backbone, heatmap model, camera models
|   |-- pl_wrappers/heatmap_ft.py      # Lightning train/eval wrapper (masked loss)
|   `-- callbacks                      # training logger callback
|-- scripts
|   |-- max2D_Id_align.py              # preprocessing (a): 0-based frame-id alignment
|   |-- build_egomax2d_cache.py        # preprocessing (b): build the 256 remap cache
|   |-- viz_max2d_seq.py               # GT annotation visualization (QC)
|   `-- pelvis_savgol_compare.py       # interpolated-frame jitter analysis (QC)
|-- inference
|   |-- inference_heatmap_egomax2d.py  # inference: GT vs Pred comparison video + timing CSV
|   `-- tran2org.py                    # inference: demo video mapped back to raw pixel space
|-- eval
|   `-- eval_2D_egomax2d.py            # evaluation: per-session pixel error + PCK report
|-- analysis/max2d.md                  # dataset statistics and annotation-structure doc
|-- data                               # not in git; ships the 8 held-out test sessions
|   |-- EgoMax2D/<session ULID>        # raw data (images + toon + calibration)
|   `-- EgoMax2D_256/<session ULID>    # preprocessed cache (256 JPEGs + meta.npz)
|-- work_dirs                          # not in git; model weights distributed separately
|   `-- egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt
|-- results                            # inference/eval outputs land here by default
|-- run.py                             # Lightning CLI entry point (fit / test)
|-- Dockerfile.h100 / docker-compose.yml / run_h100_container.sh   # environment
`-- requirements.txt / setup.py
```

## 0. Environment & Model Weights

```shell
# Build the image (one-time) and enter the container
# (mounts the current directory as /workspace/egomax2d)
sudo docker build -f Dockerfile.h100 -t egomax2d:h100 .
bash run_h100_container.sh

# Or use docker compose (equivalent entry point)
docker compose build && docker compose run --rm egomax2d
```

Fine-tuned weights (not distributed via git; copy into place separately):

```
work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt
```

Data placement (this repository ships the 8 held-out test sessions):

```
data/EgoMax2D/<session ULID>/          # raw data: images/head-front-{left,right}/,
                                       #   estimations.toon, calibration.json
data/EgoMax2D_256/<session ULID>/      # preprocessed cache (built by step 1; shipped prebuilt)
```

## 1. Data Preprocessing

For new sessions, drop them under `data/EgoMax2D/` and run two steps (the 8 shipped test
sessions are already processed — skip this):

```shell
# (a) 0-based frame-id alignment — some raw sessions' image frame ids don't start at 0;
#     shift them into alignment first (idempotent; dry-run without --apply to check,
#     then run with --apply)
python scripts/max2D_Id_align.py --root data/EgoMax2D --apply

# (b) Build the 256 cache — ray-level remap from each session's own Double Sphere
#     calibration onto the EgoBody3M cam1/cam2 geometry (incl. the 90° optical-axis
#     rotation) + grayscale; only frames with human labels are cached. GT coordinates
#     go through the same transform chain, stored as meta.npz
python scripts/build_egomax2d_cache.py --raw-root data/EgoMax2D --cached-root data/EgoMax2D_256
```

Cache building is idempotent per session (`.done` markers); the full 82 sessions take ~3 minutes
(12 workers). Evaluation (step 3) reads the cache; inference (step 2) remaps the raw data on the
fly and does not depend on it.

## Further Fine-tuning (optional, when new annotated data is available)

```shell
python run.py fit --config configs/egomax2d_vit_heatmap_ft.yaml
# pretrained_ckpt already points at the shipped fine-tuned weights (warm start).
# To resume with optimizer state as well:
#   python run.py fit --config configs/egomax2d_vit_heatmap_ft.yaml --ckpt_path <ckpt>
```

Note: the 8 shipped sessions are the held-out **test** set — **do not train on them**. Put newly
annotated sessions in a separate directory (change `data_root` in the config) so they don't get
mixed with the test data by the 8:1:1 re-split. Training needs ~26 GB of GPU memory (bs 128);
if short, lower `batch_size` or enable `grad_checkpointing: True`.

## 2. Inference

Produce a GT vs Pred comparison video for one session (in the remapped 256 grayscale space):

```shell
python inference/inference_heatmap_egomax2d.py \
    --session 01KWEDQ9HG6CSF6CNW0QVFV92E \
    --output-dir results/heatmap_egomax2d
# --ckpt already defaults to the fine-tuned weights under work_dirs/egomax2d_vit_heatmap_ft/checkpoints/
# --session-idx N picks a session by sorted index; --step/--max-frames control frame sampling
# Also writes <session>_timing.csv (per-frame CUDA Event timing, first 10 warmup frames excluded)
```

For a quick look, limit to the first 300 frames (first 10 s at 30 fps) with `--max-frames`:

```shell
python inference/inference_heatmap_egomax2d.py \
    --session 01KWEDQ9HG6CSF6CNW0QVFV92E \
    --output-dir results/heatmap_egomax2d \
    --max-frames 300
```

Produce demo videos in the **raw pixel space** (2592×1944 upright color frames) — predictions
are mapped back onto the raw images through the inverse transform chain:

```shell
python inference/tran2org.py --sessions 0        # sorted index, ranges like '0-3', or 'all'
```

## 3. Evaluation (GT Metrics)

Computes per-keypoint pixel error (256 canvas, argmax decode) over **all** sessions under
`data/EgoMax2D/`. Recommended: the standalone eval script (per-session breakdown +
PCK@5/10px, writes a markdown report):

```shell
python eval/eval_2D_egomax2d.py --report results/eval_2D_egomax2d.md
# --ckpt defaults to the shipped fine-tuned weights; --fp32 disables bf16 autocast
```

Alternatively, run the Lightning test path (single aggregate view; results match):

```shell
python run.py test --config configs/egomax2d_eval.yaml \
    --ckpt_path "work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt"
```

Outputs the overall error (weighted by labeled-point count) and per-keypoint errors. Verified
reproduction on the 8 shipped test sessions: **overall 5.49 px**, PCK@5px 65.1% / PCK@10px 90.9%.

Notes:
- Errors are computed only at human-labeled points (first-pass keyframes / manual fixes /
  occluded-with-coordinates); out-of-frame and interpolated entries never count
  (`label_sources: [2, 3, 4]`).
- `split_ratio: [0, 0, 1]` in `configs/egomax2d_eval.yaml` assigns every session under data/
  to the test split. Since the release ships only test data, this reproduces exactly the test
  metrics of the original 82-session 8:1:1 split. If you place additional sessions under
  `data/EgoMax2D/`, they will be included in the evaluation as well.

## Fine-tune Part Reference

Whole-model warm start from the EgoBody3M Stage1 ViT heatmap checkpoint; the 26-channel head is
kept unchanged, with a masked MSE supervising only the 5 labeled channels {2,3,6,7,10};
lr 2e-5 / 3 epochs / bs 128 (~5 minutes on an H100, ~25.6 GB peak training memory).
Zero-shot ≈30 px → after fine-tuning: val 4.89 px / test 5.49 px.
