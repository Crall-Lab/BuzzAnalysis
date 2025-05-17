#!/usr/bin/env python3
"""
Aggregate per-frame intermediates into video-level means for every bee ID.

For each video it:
    • loads <basename>_pivot.feather
    • calls selected functions in baseFunctions.py and (optionally) broodFunctions.py
    • writes/updates a single CSV (default: Analysis.csv)
"""
import argparse, os, pandas as pd, numpy as np
from inspect import getmembers, isfunction
import baseFunctions, broodFunctions   # <-- available exactly as today
from utils_io import iter_files, save_df

def load_funcs(include_brood=False):
    f = [fn for fn in getmembers(baseFunctions)
         if isfunction(fn[1]) and fn[1].__module__ == 'baseFunctions']
    if include_brood:
        f += [fn for fn in getmembers(broodFunctions)
              if isfunction(fn[1]) and fn[1].__module__ == 'broodFunctions']
    return f

def summarise_one(feather_path, funcs):
    vid = os.path.basename(feather_path).split("_")[0]    # workerID_YYYY... prefix
    pivot = pd.read_feather(feather_path).set_index("frame")
    ordered_ids = pivot.columns.levels[1]                 # same trick as runMe13.py :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    summary = pd.DataFrame(index=ordered_ids)
    summary["video"] = vid
    summary["bee_ID"] = summary.index

    for name, fn in funcs:
        try:
            res = fn(pivot)
            if isinstance(res, (list, np.ndarray)):
                res = pd.Series(res, index=ordered_ids)
            summary[name] = res
        except Exception as e:
            print(f"⚠ {vid}   {name}: {e}")
            summary[name] = np.nan
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--source", default=".", help="folder with *_pivot.feather")
    p.add_argument("-o","--outfile", default="Analysis.csv")
    p.add_argument("-b","--brood", action="store_true")
    a = p.parse_args()

    funcs = load_funcs(include_brood=a.brood)
    all_vids = []

    for f in iter_files(a.source, "_pivot.feather"):
        all_vids.append(summarise_one(f, funcs))

    if not all_vids:
        print("No intermediate files found.")
        return

    full = pd.concat(all_vids, ignore_index=True)
    save_df(full, a.outfile)
    print("✅  Saved", a.outfile)

if __name__ == "__main__":
    main()
