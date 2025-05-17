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
from data_cleaning import (return_duplicate_bees,
                           drop_duplicates_clean,
                           remove_jumps,
                           interpolate)
from utils_io import iter_files, save_df

# ──────────────────────────────────────────────────────────────────────────
def clean_one(df, args):
    # duplicates -----------------------------------------------------------
    df, flag = return_duplicate_bees(df)
    df       = drop_duplicates_clean(df, flag)

    # single-frame jumps ---------------------------------------------------
    if args.remove_jumps and args.remove_jumps > 0:
        df = remove_jumps(df)

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
    p.add_argument("--real-fps", type=float, default=None,
                   help="Needed to convert max gap seconds → frames")
    p.add_argument("--max-interp-sec", type=float, default=0.5)
    # other cleaning -------------------------------------------------------
    p.add_argument("--remove-jumps", type=int, default=500,
                   help="Pixel threshold; 0 disables")
    a = p.parse_args()

    suffix_list = [s.strip() for s in a.suffixes.split(",") if s.strip()]
    seen = 0
    for suf in suffix_list:
        for f in iter_files(a.source, suf):
            seen += 1
            df = pd.read_csv(f)
            df_clean = clean_one(df, a)

            out = os.path.join(os.path.dirname(f),
                               os.path.basename(f).replace(".csv","_clean.csv"))
            save_df(df_clean, out)
            print("✔", out)

    if seen == 0:
        print(f"[clean] No files matching {suffix_list} found under "
              f"{os.path.abspath(a.source)}")

if __name__ == "__main__":
    main()
