#!/usr/bin/env python3
"""Align EgoMax2D image filenames to be 0-based.

Some sessions' images start at frame_00000001 / 02 / 43 / 173 instead of
frame_00000000. The toon annotation keys are always 0-based ('000000'..'N-1'),
so this script shifts the filenames down by the session's start offset to make
`frame_%08d.jpg` == toon key. Filename format is unchanged.

Safety:
  - dry-run by default; pass --apply to actually rename
  - requires frame numbers to be contiguous and left/right cams to share the
    same start offset, else the session is skipped with an error
  - renames in ascending order (offset is negative, so targets are always
    already-vacated names); target collision is asserted before each rename
  - idempotent: sessions already starting at 0 are left untouched

Usage:
  python3 scripts/max2D_Id_align.py                 # dry-run, report only
  python3 scripts/max2D_Id_align.py --apply         # do the renames
  python3 scripts/max2D_Id_align.py --root data/EgoMax2D
"""
import argparse
import os
import re
import sys

CAMS = ["head-front-left", "head-front-right"]
FRAME_RE = re.compile(r"^frame_(\d{8})\.jpg$")


def scan_cam(cam_dir):
    """Return (start, count) after validating naming and contiguity."""
    nums = []
    for f in os.listdir(cam_dir):
        m = FRAME_RE.match(f)
        if not m:
            raise ValueError("unexpected filename %s in %s" % (f, cam_dir))
        nums.append(int(m.group(1)))
    nums.sort()
    if nums[-1] - nums[0] + 1 != len(nums):
        raise ValueError("frame numbers not contiguous in %s" % cam_dir)
    return nums[0], len(nums)


def align_session(session_dir, apply):
    """Return (start, n_frames, n_renamed)."""
    starts = {}
    counts = {}
    for cam in CAMS:
        starts[cam], counts[cam] = scan_cam(os.path.join(session_dir, "images", cam))
    if len(set(starts.values())) != 1 or len(set(counts.values())) != 1:
        raise ValueError("L/R start or count mismatch in %s: %s %s"
                         % (session_dir, starts, counts))
    start = starts[CAMS[0]]
    n = counts[CAMS[0]]
    if start == 0:
        return start, n, 0

    renamed = 0
    for cam in CAMS:
        cam_dir = os.path.join(session_dir, "images", cam)
        for old_num in range(start, start + n):  # ascending: no collisions
            src = os.path.join(cam_dir, "frame_%08d.jpg" % old_num)
            dst = os.path.join(cam_dir, "frame_%08d.jpg" % (old_num - start))
            if apply:
                assert not os.path.exists(dst), "target exists: %s" % dst
                os.rename(src, dst)
            renamed += 1
    return start, n, renamed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/EgoMax2D")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (default: dry-run report)")
    args = ap.parse_args()

    sessions = sorted(d for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d)))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print("[%s] %d sessions under %s" % (mode, len(sessions), args.root))

    total_renamed, shifted, errors = 0, 0, 0
    for s in sessions:
        try:
            start, n, renamed = align_session(os.path.join(args.root, s), args.apply)
        except (ValueError, AssertionError) as e:
            errors += 1
            print("  ERROR %s: %s" % (s, e))
            continue
        if start != 0:
            shifted += 1
            total_renamed += renamed
            print("  %s: start=%d n=%d -> %s %d files"
                  % (s, start, n, "renamed" if args.apply else "would rename", renamed))
    print("[%s] done: %d sessions shifted, %d files %s, %d errors"
          % (mode, shifted, total_renamed,
             "renamed" if args.apply else "to rename", errors))
    if not args.apply and shifted:
        print("re-run with --apply to execute")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
