#!/usr/bin/env python3
"""
02_clean_data.py
----------------
Duplicate removal, jump filtering, interpolation – now applied to
*_Left.csv, *_Right.csv (and *_Whole.csv) in a single run.

For each input file it writes  <same-name>_clean.csv   alongside it.
The data_cleaning.interpolate() function already appends an
'interpolated' column (0 = raw frame, 1 = filled frame).
"""
import argparse, os, pandas as pd
import json
from settings_io import read_settings, parse_with_overrides
import math
import sys
from pathlib import Path
from data_cleaning import (return_duplicate_bees,
                           drop_duplicates_clean,
                           remove_jumps,
                           summarize_jump_log,
                           interpolate)
from utils_io import iter_files, save_df

# ──────────────────────────────────────────────────────────────────────────
def apply_settings_file(args):
    if not args.settings:
        return args

    path = Path(args.settings).expanduser()
    if not path.exists():
        sys.exit(f"ERROR: Settings file not found: {path}")
    try:
        settings = read_settings(path)
    except Exception as exc:
        sys.exit(f"ERROR: Could not read settings file {path}: {exc}")

    if args.remove_jumps is None:
        max_speed = float(settings.get("max_speed_cutoff", 0) or 0)
        if max_speed > 0:
            args.remove_jumps = max_speed
            print(f"Using max_speed_cutoff from settings for spike removal: {args.remove_jumps} px/frame")

    return args


def clean_one(df, args, filename):
    # duplicates -----------------------------------------------------------
    df, flag = return_duplicate_bees(df)
    df       = drop_duplicates_clean(df, flag)

    # clear isolated tag-read spikes --------------------------------------
    if args.remove_jumps:
        df = remove_jumps(df, args, filename)

    # interpolation (adds 'interpolated' column automatically) ------------
    if args.interpolate:
        df = interpolate(df,
                         max_seconds_gap=args.max_interp_sec,
                         actual_frames_per_second=args.real_fps)
    return df
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--source", default=".",
                   help="Top-level folder; searched recursively")
    p.add_argument("--suffixes", default="_Left.csv,_Right.csv,_Whole.csv",
                   help="Comma-separated list of file endings to clean "
                        "(default processes Left, Right and Whole)")
    # interpolation --------------------------------------------------------
    p.add_argument("-i","--interpolate", action="store_true")
    p.add_argument("--real-fps", type=float,
                   help="Needed to convert max gap seconds → frames")
    p.add_argument("--max-interp-sec", type=float)
    # other cleaning -------------------------------------------------------
    p.add_argument("--settings", type=str, default=None,
                   help="JSON settings file exported from review_metrics_gui.py")
    p.add_argument("--remove-jumps", "--remove-spikes", dest="remove_jumps", type=float,
                   help=("Maximum plausible speed in px/frame. Removes only clear isolated "
                         "jump-out-and-back tag-read spikes; ambiguous one-way high-speed "
                         "transitions are logged but kept. Defaults to max_speed_cutoff "
                         "from --settings when available."))
    args = p.parse_args()
    args = apply_settings_file(args)

    if args.interpolate and args.real_fps is None:
        p.error("--interpolate requires --real-fps")

    if args.interpolate and args.max_interp_sec is None:
        p.error("--interpolate requires --max-interp-sec")
    if args.real_fps is not None and (not math.isfinite(args.real_fps) or args.real_fps <= 0):
        p.error("--real-fps must be finite and positive")
    if args.max_interp_sec is not None and (not math.isfinite(args.max_interp_sec) or args.max_interp_sec < 0):
        p.error("--max-interp-sec must be finite and nonnegative")
    if args.remove_jumps is not None and (not math.isfinite(args.remove_jumps) or args.remove_jumps < 0):
        p.error("--remove-jumps must be finite and nonnegative")

    suffix_list = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    seen = 0
    for suf in suffix_list:
        for file in iter_files(args.source, suf):
            seen += 1
            df = pd.read_csv(file)
            filename = Path(file).stem
            df_clean = clean_one(df, args, filename)

            out = os.path.join(os.path.dirname(file),
                               os.path.basename(file).replace(".csv","_clean.csv"))
            save_df(df_clean, out)
            print("✔", out)

    if seen == 0:
        print(f"[clean] No files matching {suffix_list} found under "
              f"{os.path.abspath(args.source)}")
    
    else:
        #give a terminal-printed summary of the number of jumps documented by bee ID
        summarize_jump_log(args)

if __name__ == "__main__":
    main()
