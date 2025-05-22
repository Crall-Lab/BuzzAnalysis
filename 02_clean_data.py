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
                           summarize_jump_log,
                           interpolate)
from utils_io import iter_files, save_df

# ──────────────────────────────────────────────────────────────────────────
def clean_one(df, args):
    # duplicates -----------------------------------------------------------
    df, flag = return_duplicate_bees(df)
    df       = drop_duplicates_clean(df, flag)

    # single-frame jumps ---------------------------------------------------
    if args.remove_jumps:
        df = remove_jumps(df, args)

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
    p.add_argument("--real-fps", type=(float or int),
                   help="Needed to convert max gap seconds → frames")
    p.add_argument("--max-interp-sec", type=(float or int))
    # other cleaning -------------------------------------------------------
    p.add_argument("--remove-jumps", type=int,
                   help="Pixel threshold distance to remove tag jumping between consecutive frames")
    args = p.parse_args()

    if args.interpolate and args.real_fps == None:
        print('''\nERROR!
        -------------------------------------------------------
        You have set interpolation to True (by using the --interpolate flag (-i for short))
        but haven't specified a value for --real-fps. Please set a number (of the type float)
        based on the actual framerate of your videos.

        Example command line implementation: --real-fps 4.5
        -------------------------------------------------------
        \n''')
        return

    if args.interpolate and args.max_interp_sec == None:
        print('''\nERROR!
        -------------------------------------------------------
        You have set interpolation to True (by using the --interpolate flag (-i for short))
        but haven't specified a value for --max-interp-sec. Please set a value for the maximum number of seconds
        you want to interpolate data between consecutive tag reads. Do this using the --max-interp-sec flag.
        You can enter either a float (ex. 3.5) or an integer (ex. 3).

        Example command line implementation: --max-interp-sec 3
        -------------------------------------------------------
        \n''')
        return
    
    suffix_list = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    seen = 0
    for suf in suffix_list:
        for file in iter_files(args.source, suf):
            seen += 1
            df = pd.read_csv(file)
            df_clean = clean_one(df, args)

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
