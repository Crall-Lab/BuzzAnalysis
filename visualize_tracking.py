#!/usr/bin/env python3
"""
05_visualise_tracking.py
────────────────────────────────────────────────────────────────────────────
Create an overlay movie in which …

   • solid circles  – detections that came straight from the raw file
   • translucent circles – frames inserted by the interpolator
   • both Left & Right arenas are merged into one video

Folder layout assumed (what the earlier scripts produce)
─────────────────────────────────────────────────────────
project_root/
├── video1_raw.csv
├── video1.mnp4           ← original footage
└── video1/
    ├── video1_Left_clean.csv
    └── video1_Right_clean.csv

Invocation
──────────
python 05_visualise_tracking.py \
       -s <project_root> \
       --fps 30          \
       --ext .mp4       # extension to look for

The script looks for *_Left_clean.csv (and matching *_Right_clean.csv) under
<SOURCE>, finds the corresponding video, draws coloured circles, and writes

    video1/video1_tracked.mp4
"""
# ── std-lib ───────────────────────────────────────────────────────────────
import argparse, glob, os, random, sys
from pathlib import Path
from collections import defaultdict

# ── third-party ───────────────────────────────────────────────────────────
import cv2
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def iter_pairs(root):
    """
    Yield tuples  (stem, left_csv, right_csv_or_None)
    for every *_Left_clean.csv found beneath *root*.
    """
    left_files = glob.glob(os.path.join(root, "**", "*_Left_clean.csv"),
                           recursive=True)
    for left in left_files:
        stem = Path(left).stem.replace("_Left_clean", "")
        right = left.replace("_Left_clean.csv", "_Right_clean.csv")
        yield stem, left, (right if os.path.exists(right) else None)

# -------------------------------------------------------------------------
def load_track_csvs(left, right):
    """
    Merge Left + Right clean CSVs and return one DataFrame.

    Ensures the required columns exist and adds a helper 'csv_path' column
    containing the absolute path of the original CSV (used later to locate
    the video).
    """
    dfs = []
    df_left = pd.read_csv(left)
    df_left["csv_path"] = os.path.abspath(left)
    dfs.append(df_left)

    if right:
        df_right = pd.read_csv(right)
        df_right["csv_path"] = os.path.abspath(right)
        dfs.append(df_right)

    df = pd.concat(dfs, ignore_index=True)

    required = {"frame", "ID", "centroidX", "centroidY", "interpolated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s) {missing} in {left}")

    return df

# -------------------------------------------------------------------------
def stable_bgr(tag):
    """Deterministic bright BGR colour for a numeric tag (bee ID)."""
    rng = random.Random(int(tag))
    return (rng.randint(40, 255),
            rng.randint(40, 255),
            rng.randint(40, 255))

# -------------------------------------------------------------------------
def draw_points(img, rows, radius, alpha):
    """
    Draw circles for all rows (same frame) onto *img*.
    """
    # We separate opaque and translucent points so we don’t re-blend twice
    overlay = img.copy()

    for _, r in rows.iterrows():
        color   = stable_bgr(r.ID)
        center  = (int(r.centroidX), int(r.centroidY))
        cv2.circle(overlay, center, radius, color, -1)

    # Blend only where ANY point in this frame is interpolated
    blend = rows["interpolated"].astype(bool).any()
    if blend:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    else:
        img[:] = overlay

# -------------------------------------------------------------------------
def find_video(stem, csv_path, ext, root):
    """
    Return a fully-qualified path to the video that belongs to *stem*.

    Search order:
      1. same folder as clean CSV      (…/video1/video1.ext)
      2. parent folder                 (…/video1.ext)
      3. any <stem><ext> under *root*
    """
    csv_dir = Path(csv_path).parent
    # Case 1
    candidate = csv_dir / f"{stem}{ext}"
    if candidate.exists():
        return str(candidate)
    # Case 2
    candidate = csv_dir.parent / f"{stem}{ext}"
    if candidate.exists():
        return str(candidate)
    # Case 3
    hits = list(Path(root).rglob(f"{stem}{ext}"))
    return str(hits[0]) if hits else None

# -------------------------------------------------------------------------
def process_one_video(stem: str,
                      df: pd.DataFrame,
                      fps: float,
                      radius: int,
                      alpha: float,
                      codec: str,
                      ext: str,
                      root: str):
    """Overlay tracks onto video frames and save <stem>_tracked.mp4."""
    video_path = find_video(stem, df.iloc[0]["csv_path"], ext, root)
    if video_path is None:
        print(f"⚠  No video file found for {stem} (looked for *{ext})")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠  Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out_path = str(Path(video_path).with_name(f"{stem}_tracked.mp4"))
    writer  = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"▶  Writing {out_path}")

    grouped = df.groupby("frame")
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no in grouped.groups:
            draw_points(frame,
                        grouped.get_group(frame_no),
                        radius,
                        alpha)

        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()
    print("✅ Finished", out_path)

# ══════════════════════════════════════════════════════════════════════════
#  Main CLI entry-point
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True,
                    help="Parent folder that contains per-video sub-folders")
    ap.add_argument("--fps", type=float, required=True,
                    help="True FPS of the original recording")
    ap.add_argument("--radius", type=int, default=5,
                    help="Circle radius in pixels")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Transparency (0-1) for interpolated detections")
    ap.add_argument("--codec", default="mp4v",
                    help="FourCC codec for cv2.VideoWriter (default mp4v)")
    ap.add_argument("--ext", default=".mp4",
                    help="Extension of source video files (default .mp4)")
    args = ap.parse_args()

    # ── iterate over every recording ──────────────────────────────────────
    any_processed = False
    for stem, left_csv, right_csv in iter_pairs(args.source):
        try:
            df = load_track_csvs(left_csv, right_csv)
            process_one_video(stem, df,
                              args.fps,
                              args.radius,
                              args.alpha,
                              args.codec,
                              args.ext,
                              root=args.source)
            any_processed = True
        except Exception as e:
            print(f"⚠  {stem}: {e}")

    if not any_processed:
        sys.exit("No *_Left_clean.csv files found – nothing to do.")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
