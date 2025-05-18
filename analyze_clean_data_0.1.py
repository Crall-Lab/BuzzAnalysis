#!/usr/bin/env python3
"""
analyze_clean_data_0.1.py   ·   May-2025
────────────────────────────────────────────────────────────────────────────
Direct successor of the original runMe13.py, but it EXPECTS that your
tracking has already been split + cleaned, i.e. it starts from

    *_Left_clean.csv / *_Right_clean.csv / *_Whole_clean.csv

Key features
------------
• Computes all metrics in baseFunctions.py
• When you pass  -b  <brood directory>  it also runs every function in
  broodFunctions.py and the heavy distance matrices in processBroodFunctions.py
• Brood-nest map lookup is flexible:  <stem>*<ext>
      stem = "<colony>_<yyyy-mm-dd>"  (hyphens → underscores)
      ext  =  --broodExtension  (default "_nest_image.csv")
• Works single-core or multi-core (-c/--cores)
"""

# ── std-lib ───────────────────────────────────────────────────────────────
import argparse, os, sys, re, warnings
from inspect import getmembers, isfunction
from multiprocessing import Pool
from pathlib import Path
# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
# ── project modules ───────────────────────────────────────────────────────
import baseFunctions
import broodFunctions
import processBroodFunctions

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('-s','--source', default='.',
                   help='Root folder searched recursively for *_clean.csv')
    p.add_argument('-e','--extension', default='_clean.csv',
                   help='Suffix of clean files (default _clean.csv)')
    p.add_argument('-b','--brood', type=str, default=None,
                   help='Folder that contains brood map CSVs')
    p.add_argument('-x','--broodExtension', default='_nest_image.csv',
                   help='Suffix of brood maps (default _nest_image.csv)')
    p.add_argument('-o','--outFile', default='Analysis.csv',
                   help='Output table (default Analysis.csv)')
    p.add_argument('-c','--cores', type=int, default=1,
                   help='Parallel workers')
    p.add_argument('-l','--limit', type=int, default=None,
                   help='Process only N files (debug)')
    return vars(p.parse_args())

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def pivot_clean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(index='frame', columns='ID',
                       values=['centroidX','centroidY','frontX','frontY'])
          .sort_index(axis=1)
          .apply(pd.to_numeric, errors='coerce')
    )

def _find_brood_map(stem: str, brood_dir: str, brood_ext: str):
    ext_clean = brood_ext.lstrip('_-')
    hits = sorted(Path(brood_dir).glob(f"{stem}*{ext_clean}"))  # not recursive
    return hits[0] if hits else None

def processBrood(basename, oneLR, LR_name, brood_dir, brood_ext):
    stem = '_'.join(basename.split('_')[0:2]).replace('-', '_')
    map_path = _find_brood_map(stem, brood_dir, brood_ext)
    print(f"DEBUG  looking for brood map  {stem}  →  {map_path}")
    if map_path is None:
        print(f"⚠  Brood map not found for {basename}")
        return oneLR

    full = pd.read_csv(map_path)
    # split map to Left / Right
    THR = 2000
    if LR_name == 'Left':
        full = full[full['x'] < THR]
    elif LR_name == 'Right':
        full = full[full['x'] > THR]

    brood = full[full['label'] != 'Arena perimeter (polygon)']
    eggs  = brood[brood['radius'].isna()]
    allb  = brood.dropna(subset=['radius'])

    # ── FIX: reset index so processBroodFunctions expects 0…N-1
    brood = brood.reset_index(drop=True)
    eggs  = eggs.reset_index(drop=True)
    allb  = allb.reset_index(drop=True)

    dist1 = processBroodFunctions.distanceFromCentroid(oneLR, allb)
    dist2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    dist3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)
    return pd.concat([oneLR, dist1, dist2, dist3], axis=1)

# ══════════════════════════════════════════════════════════════════════════
#  Per-file processing
# ══════════════════════════════════════════════════════════════════════════
def file_to_analysis(fpath, opt, funcs):
    base = os.path.basename(fpath)
    # read clean tracking
    df = pd.read_csv(fpath, dtype={
        "frame":"int32", "ID":"int32",
        "centroidX":"float32","centroidY":"float32",
        "frontX":"float32","frontY":"float32"
    })

    workerID, Date, H, M, S, *_ = base.split("_")
    LR_match = re.search(r'_(Left|Right|Whole)_', base)
    LR = LR_match.group(1) if LR_match else 'Whole'

    oneLR = pivot_clean(df)

    if opt['brood']:
        oneLR = processBrood(base, oneLR, LR,
                             brood_dir  = opt['brood'],
                             brood_ext  = opt['broodExtension'])

    bee_ids = oneLR.columns.levels[1]
    if not len(bee_ids):
        print(f"⚠  No bee IDs in {base}")
        return None

    out = pd.DataFrame(index=bee_ids)
    # safe per-column metadata assignment  (fixes NumPy broadcast error)
    out['pi_ID'] = workerID
    out['bee_ID'] = bee_ids
    out['Date']   = Date
    out['Time']   = f"{H}-{M}-{S}"
    out['LR']     = LR

    # run metric functions
    for name, fn in funcs:
        try:
            res = fn(oneLR)
            if isinstance(res, (list, np.ndarray)):
                res = pd.Series(res, index=bee_ids)
            if isinstance(res, pd.Series):
                out[name] = res.reindex(bee_ids)
            else:
                out[name] = res
        except Exception as e:
            print(f"⚠  {name} failed on {base}: {e}")
            out[name] = np.nan
    return out

def file_wrapper(args):  # makes Pool.imap happy
    return file_to_analysis(*args)

# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    opt = parse_opt()

    # collect all metric functions
    funcs = [(n,f) for n,f in getmembers(baseFunctions)
             if isfunction(f) and f.__module__=='baseFunctions']
    if opt['brood']:
        funcs += [(n,f) for n,f in getmembers(broodFunctions)
                  if isfunction(f) and f.__module__=='broodFunctions']

    files = [str(p) for p in Path(opt['source']).rglob(f"*{opt['extension']}")]
    if opt['limit']:
        files = files[:opt['limit']]
    if not files:
        sys.exit("❌  No *_clean.csv files found")

    print(f"Processing {len(files)} files …")

    args_iter = [(fp, opt, funcs) for fp in files]
    if opt['cores'] > 1:
        with Pool(opt['cores']) as pool:
            results = list(tqdm(pool.imap_unordered(file_wrapper, args_iter),
                                total=len(files), desc="Files"))
    else:
        results = [file_wrapper(a) for a in tqdm(args_iter, desc="Files")]

    results = [r for r in results if r is not None]
    if not results:
        sys.exit("❌  No files processed successfully")

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(opt['outFile'], index=False)
    print(f"✅  Saved {opt['outFile']}  ({len(combined)} rows)")

# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
