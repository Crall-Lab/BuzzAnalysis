#!/usr/bin/env python3
"""
Compute frame-level derived metrics and per-bee summaries.
Outputs two files per input:
    • <basename>_pivot.feather        (pivoted X/Y dataframe)
    • <basename>_intermediate.npz     (NumPy archive of heavy arrays)
"""
import argparse, os, numpy as np, pandas as pd
from baseFunctions import interbee_distance_matrix           # heavy
from utils_io import iter_files, ensure_dir
from runMe13 import restructure_tracking_data                # reuse proven helper :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
import pickle, pathlib

def compute(one_clean_csv, args):
    raw = pd.read_csv(one_clean_csv)
    # Re-use proven pivot / restructure function
    # Supply dummy opts dict (only remove_jumps and interpolate flags needed)
    opts = dict(interpolate=False, remove_jumps=None)
    pivot = restructure_tracking_data(raw, opts,
                                      interpolated_path_name=None)
    # Heavy intermediates --------------------------------------------------
    ib_dist = interbee_distance_matrix(pivot)        # 3-D array (frames × bees × bees)
    npz_path = one_clean_csv.replace("_clean.csv","_intermediate.npz")
    np.savez_compressed(npz_path, interbee=ib_dist)
    # Light intermediate ---------------------------------------------------
    feather_path = one_clean_csv.replace("_clean.csv","_pivot.feather")
    ensure_dir(feather_path)
    pivot.reset_index().to_feather(feather_path)
    return pivot.shape[1]//2   # number of bees

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--source", default=".", help="*_clean.csv from step 2")
    p.add_argument("-c","--cores", type=int, default=1)
    a = p.parse_args()

    for f in iter_files(a.source, "_clean.csv"):
        n = compute(f, a)
        print(f"✔ {f}   ({n} bees)")

if __name__ == "__main__":
    main()
