#!/usr/bin/env python3
"""
01_split_lr.py
--------------
Split raw Bombus/BEEtag tracking CSVs into Left/Right (or Whole) halves.

For each source file
    xxx_raw.csv     ──►   xxx/
                         ├── xxx_Left.csv
                         └── xxx_Right.csv      (or xxx_Whole.csv)

Later stages will keep adding *_clean.csv, *_pivot.feather, …
inside the same xxx/ folder, so everything that belongs to a single
recording stays together.
"""
import argparse, os, sys, pandas as pd, numpy as np
from utils_io import iter_files, save_df

DEF_THRESHOLD = 2000          # X-coordinate separating arenas

# --------------------------------------------------------------------------
def split_df(df, threshold, whole):
    if whole:
        df["LR"] = "Whole"
        return {"Whole": df}
    df["LR"] = np.where(df["centroidX"] < threshold, "Left", "Right")
    return {k: v.drop(columns="LR") for k, v in df.groupby("LR")}
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--source", default=".",
                   help="Folder with raw tracking CSVs (searched recursively)")
    p.add_argument("-e","--extension", default=".csv",
                   help="Exact suffix to match, e.g. _raw.csv or .csv")
    p.add_argument("-t","--threshold", type=float, default=DEF_THRESHOLD)
    p.add_argument("-w","--whole", action="store_true",
                   help="Skip L/R split; keep full arena and label as Whole")
    args = p.parse_args()

    files = list(iter_files(args.source, args.extension))       # materialise first
    if not files:
        print(f"[split-lr] ❌  No files ending with “{args.extension}” found in "
              f"{os.path.abspath(a.source)}.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[split-lr] ⚠  Could not read “{f}”: {e}", file=sys.stderr)
            continue

        # ------------------------------------------------------------------
        stem = os.path.basename(f)[:-len(a.extension)]    # ‘video1’ from ‘video1_raw.csv’
        subdir = os.path.join(os.path.dirname(f), stem)
        os.makedirs(subdir, exist_ok=True)
        # ------------------------------------------------------------------

        for lr, part in split_df(df, a.threshold, a.whole).items():
            out = os.path.join(subdir, f"{stem}_{lr}.csv")
            part.sort_values(by=['ID', 'frame'], inplace=True)
            save_df(part, out)
            print("✔", out)

if __name__ == "__main__":
    main()
