#!/usr/bin/env python3
"""Write coordinate pivots and interbee distances for each *_clean.csv file."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from aux import interbee_distance_matrix
from utils_io import iter_files


def compute(one_clean_csv, args=None):
    raw = pd.read_csv(one_clean_csv)
    pivot = raw.pivot_table(index="frame", columns="ID", values=["centroidX", "centroidY"]).sort_index(axis=1)
    path = Path(one_clean_csv)
    stem = path.name.removesuffix("_clean.csv")
    np.savez_compressed(path.with_name(stem + "_intermediate.npz"), interbee=interbee_distance_matrix(pivot))
    pivot.to_feather(path.with_name(stem + "_pivot.feather"))
    return pivot["centroidX"].shape[1] if not pivot.empty else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", default=".")
    parser.add_argument("-c", "--cores", type=int, default=1)
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be positive")
    files = sorted(iter_files(args.source, "_clean.csv"))
    if not files:
        parser.error("No *_clean.csv files found")
    if args.cores > 1:
        with ProcessPoolExecutor(max_workers=args.cores) as pool:
            counts = list(pool.map(compute, files))
    else:
        counts = [compute(path) for path in files]
    for path, count in zip(files, counts):
        print(f"Saved intermediates: {path} ({count} bees)")


if __name__ == "__main__":
    main()
