"""Standalone 2D evaluation of the fine-tuned ViT heatmap model on EgoMax2D.

Same metric semantics as pose_estimation/pl_wrappers/heatmap_ft.py eval_step
(this is what `run.py test --config configs/egomax2d_eval.yaml` reports):
  - argmax-decode the 5 mapped channels {2,3,6,7,10} of the 26-channel
    heatmap, x4 back to the 256 canvas;
  - pixel L2 error against the remapped human GT, only where a usable human
    label exists (label sources 2=first-pass, 3=manual, 4=occluded-w-coords;
    out_of_frame and interpolated entries never count);
  - overall mean is weighted by labeled-point count.
On top of the Lightning test path this script adds a per-session breakdown
and PCK@5px / PCK@10px (256-canvas thresholds).

Every session found under --data-root is evaluated (the release data/ ships
the 8 held-out test sessions, reproducing the original test-split numbers,
overall ~5.49 px). Reads the EgoMax2D_256 cache; missing session caches are
built on first use.

Usage:
    cd /home/ubuntu/project/egomax2d_release
    python eval/eval_2D_egomax2d.py \
        --ckpt "work_dirs/egomax2d_vit_heatmap_ft/checkpoints/epoch=2-val_kp_px_error=4.89.ckpt" \
        --report results/eval_2D_egomax2d.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_estimation.datasets.egomax2d.egomax2d_heatmap import EgoMax2DHeatmapDataset
from pose_estimation.datasets.egomax2d.remap import KPS, KP_JOINT_IDS
from pose_estimation.models.estimator import EgoPoseFormerHeatmap

_HM_SCALE = 4  # 256 canvas / 64 heatmap

# Must match configs/egomax2d_vit_heatmap_ft.yaml (the ckpt's architecture).
_ENCODER_CFG = dict(
    type="vit",
    model_name="vit_base_patch16_224.augreg_in21k",
    pretrained=False,   # weights come from --ckpt
    img_size=256,
    out_stride=4,
    out_channels=128,
    neck_mid_channels=256,
    drop_path_rate=0.1,
    grad_checkpointing=False,
    weights_path=None,
)
MODEL_CFG = dict(num_heatmap=26, encoder_cfg=_ENCODER_CFG, train_cfg=dict(w_heatmap=10.0))

DEFAULT_CKPT = (
    "work_dirs/egomax2d_vit_heatmap_ft/checkpoints"
    "/epoch=2-val_kp_px_error=4.89.ckpt"
)

PCK_THRESHOLDS_PX = (5.0, 10.0)


def load_model(ckpt_path: str, device: str) -> EgoPoseFormerHeatmap:
    model = EgoPoseFormerHeatmap(**MODEL_CFG)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {
        k[len("model."):]: v for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    print(f"loaded {ckpt_path}: {len(state_dict)} tensors")
    return model.to(device).eval()


@torch.no_grad()
def decode_kp(pred_hm: torch.Tensor) -> torch.Tensor:
    """[B, V, 26, h, w] -> argmax px of the 5 mapped channels on the 256
    canvas, [B, V, 5, 2] (same decode as heatmap_ft._kp_pixel_errors)."""
    B, V = pred_hm.shape[:2]
    h, w = pred_hm.shape[-2:]
    hm = pred_hm[:, :, list(KP_JOINT_IDS)].reshape(B, V, len(KP_JOINT_IDS), h * w)
    flat_idx = hm.argmax(dim=-1)
    pred_u = (flat_idx % w).float() * _HM_SCALE
    pred_v = torch.div(flat_idx, w, rounding_mode="floor").float() * _HM_SCALE
    return torch.stack([pred_u, pred_v], dim=-1)


class Accum:
    """Per-keypoint error sums / counts / PCK hits."""

    def __init__(self):
        k = len(KPS)
        self.err_sum = np.zeros(k)
        self.count = np.zeros(k)
        self.pck_hit = {t: np.zeros(k) for t in PCK_THRESHOLDS_PX}
        self.n_frames = 0

    def add(self, err, mask):
        """err/mask: [B, V, K] numpy, err already masked (0 where invalid)."""
        self.err_sum += err.sum(axis=(0, 1))
        self.count += mask.sum(axis=(0, 1))
        for t in PCK_THRESHOLDS_PX:
            self.pck_hit[t] += ((err <= t) & (mask > 0)).sum(axis=(0, 1))
        self.n_frames += err.shape[0]

    @property
    def overall_px(self):
        return self.err_sum.sum() / max(self.count.sum(), 1.0)

    def per_kp_px(self, k):
        return self.err_sum[k] / max(self.count[k], 1.0)

    def pck(self, t):
        return self.pck_hit[t].sum() / max(self.count.sum(), 1.0)


def fmt_row(name, acc: Accum) -> str:
    kp_cells = " | ".join(
        f"{acc.per_kp_px(k):.2f}" if acc.count[k] > 0 else "—"
        for k in range(len(KPS)))
    pck_cells = " | ".join(f"{acc.pck(t) * 100:.1f}" for t in PCK_THRESHOLDS_PX)
    return (f"| {name} | {acc.n_frames} | {int(acc.count.sum())} "
            f"| **{acc.overall_px:.2f}** | {kp_cells} | {pck_cells} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data-root", default="data/EgoMax2D")
    ap.add_argument("--cached-root", default="data/EgoMax2D_256")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (default matches training eval: bf16)")
    ap.add_argument("--report", default=None,
                    help="also write the markdown report to this path")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)

    # split_ratio (0,0,1): every session under data_root is the test split.
    dataset = EgoMax2DHeatmapDataset(
        data_root=args.data_root, split="test", cached_root=args.cached_root,
        label_sources=(2, 3, 4), split_ratio=(0.0, 0.0, 1.0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    use_amp = args.device.startswith("cuda") and not args.fp32
    per_session = {}
    overall = Accum()
    base = 0
    for bi, batch in enumerate(loader):
        img = batch["img"].to(args.device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            feats = model.forward_backbone(img)
            B, V = img.shape[:2]
            hm = model.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
            hm = hm.view(B, V, *hm.shape[1:]).float()
        pred_xy = decode_kp(hm).cpu()
        mask = batch["kp_mask"]                                      # [B, V, 5]
        err = (torch.linalg.norm(pred_xy - batch["kp2d"], dim=-1) * mask).numpy()
        mask = mask.numpy()

        # Route each sample of the batch to its session accumulator.
        for i in range(err.shape[0]):
            sid = dataset.samples[base + i][0]
            per_session.setdefault(sid, Accum()).add(err[i:i + 1], mask[i:i + 1])
        overall.add(err, mask)
        base += err.shape[0]
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(loader)}", flush=True)

    kp_hdr = " | ".join(KPS)
    pck_hdr = " | ".join(f"PCK@{t:g}px" for t in PCK_THRESHOLDS_PX)
    lines = [
        f"# EgoMax2D 2D eval — {os.path.basename(args.ckpt)}",
        "",
        f"- data_root: `{args.data_root}`  sessions: {len(per_session)}  "
        f"frames: {overall.n_frames}  labeled points: {int(overall.count.sum())}",
        f"- precision: {'bf16-autocast' if use_amp else 'fp32'}  "
        f"(256-canvas px error, argmax decode x{_HM_SCALE}; human labels only)",
        "",
        f"| session | frames | points | mean px | {kp_hdr} | {pck_hdr} |",
        "|---" * (4 + len(KPS) + len(PCK_THRESHOLDS_PX)) + "|",
    ]
    for sid in sorted(per_session):
        lines.append(fmt_row(sid, per_session[sid]))
    lines.append(fmt_row("**OVERALL**", overall))
    report = "\n".join(lines)
    print("\n" + report)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write(report + "\n")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
