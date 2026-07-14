#!/usr/bin/env python3
"""Pre-build the EgoMax2D_256 cache (remap + grayscale 256x256 + meta.npz).

The EgoMax2DHeatmapDataset builds missing sessions automatically on first
use; this script does the same thing ahead of time so the first training
run starts instantly. Idempotent — finished sessions (with .done) are
skipped, so it can be re-run after adding data.

Usage:
    python scripts/build_egomax2d_cache.py                 # all 82 sessions
    python scripts/build_egomax2d_cache.py --sessions 0-3  # subset by sorted index
    python scripts/build_egomax2d_cache.py --workers 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pose_estimation.datasets.egomax2d.egomax2d_heatmap import ensure_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="data/EgoMax2D")
    ap.add_argument("--cached-root", default="data/EgoMax2D_256")
    ap.add_argument("--sessions", default="all",
                    help="'all', sorted-index range '0-3', or single index '5'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    all_sessions = sorted(d for d in os.listdir(args.raw_root)
                          if os.path.isdir(os.path.join(args.raw_root, d)))
    if args.sessions == "all":
        picks = all_sessions
    elif "-" in args.sessions:
        a, b = args.sessions.split("-")
        picks = all_sessions[int(a): int(b) + 1]
    else:
        picks = [all_sessions[int(args.sessions)]]

    print(f"Sessions: {len(picks)}/{len(all_sessions)}  workers={args.workers}")
    ensure_cache(args.raw_root, args.cached_root, picks,
                 workers=args.workers, jpeg_quality=args.jpeg_quality)
    print("done")


if __name__ == "__main__":
    main()
