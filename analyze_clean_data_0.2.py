#!/usr/bin/env python3
"""
analyze_clean_data_0.1.py
────────────────────────────────────────────────────────────────────────────
Successor of runMe13.py that starts from *_clean.csv files and now keeps
frame-level speed & activity inside each per-video pivot table.

Typical call
------------
python analyze_clean_data_0.1.py \
       -s  <project_root> \
       -b  <brood_map_dir> \
       --save-pivots \
       -o  Analysis.csv \
       -c  4
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
from aux import movement_metrics                # caches speed & activity

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s','--source', default='.',
                    help='Root searched recursively for *_clean.csv')
    ap.add_argument('-e','--extension', default='_clean.csv')
    ap.add_argument('-b','--brood', type=str, default=None,
                    help='Folder that holds brood maps')
    ap.add_argument('-x','--broodExtension', default='_nest_image.csv')
    ap.add_argument('-o','--outFile', default='Analysis.csv')
    ap.add_argument('-c','--cores', type=int, default=1,
                    help='Parallel workers (default 1 = serial)')
    ap.add_argument('-l','--limit', type=int, default=None,
                    help='Process only N files (debug)')
    ap.add_argument('--save-pivots', action='store_true',
                    help='Write *_pivot_enriched.feather per video')
    return vars(ap.parse_args())

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def pivot_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: MultiIndex (coord, ID) × frame."""
    return (df.pivot_table(index='frame', columns='ID',
                           values=['centroidX','centroidY','frontX','frontY'])
              .sort_index(axis=1)
              .apply(pd.to_numeric, errors='coerce'))

def _find_brood_map(stem: str, root: str, ext: str):
    """Recursive match  <stem>*<ext>  underneath *root*."""
    ext_clean = ext.lstrip('_-')
    hits = sorted(Path(root).rglob(f"{stem}*{ext_clean}"))
    return hits[0] if hits else None

def processBrood(basename, oneLR, LR, brood_dir, brood_ext):
    """Attach brood-distance matrices (if map present)."""
    stem = '_'.join(basename.split('_')[0:2]).replace('-', '_')
    mp = _find_brood_map(stem, brood_dir, brood_ext)
    if mp is None:
        return oneLR

    full = pd.read_csv(mp)
    THR = 2000
    if LR == 'Left':  full = full[full['x'] < THR]
    elif LR == 'Right': full = full[full['x'] > THR]

    brood = full[full['label']!='Arena perimeter (polygon)']
    eggs  = brood[brood['radius'].isna()]
    allb  = brood.dropna(subset=['radius'])
    for df in (brood, eggs, allb):
        df.reset_index(drop=True, inplace=True)

    d1 = processBroodFunctions.distanceFromCentroid(oneLR, allb)
    d2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    d3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)
    return pd.concat([oneLR, d1, d2, d3], axis=1)

# ══════════════════════════════════════════════════════════════════════════
#  Per-file job
# ══════════════════════════════════════════════════════════════════════════
def analyse_one(fpath, opt, funcs):
    base = os.path.basename(fpath)
    df = pd.read_csv(fpath, dtype={
        "frame":"int32","ID":"int32",
        "centroidX":"float32","centroidY":"float32",
        "frontX":"float32","frontY":"float32"})

    worker, Date, H, M, S, *_ = base.split("_")
    LRm = re.search(r'_(Left|Right|Whole)_', base)
    LR  = LRm.group(1) if LRm else 'Whole'

    pivot = pivot_clean(df)              # build wide table
    movement_metrics(pivot)              # adds speed + activity columns

    if opt['brood']:
        pivot = processBrood(base, pivot, LR,
                             brood_dir=opt['brood'],
                             brood_ext=opt['broodExtension'])

    bee_ids = pivot.columns.levels[1]
    if not len(bee_ids):
        return None

    out = pd.DataFrame(index=bee_ids)
    out['pi_ID'] = worker
    out['bee_ID'] = bee_ids
    out['Date']   = Date
    out['Time']   = f"{H}-{M}-{S}"
    out['LR']     = LR

    for name, fn in funcs:
        try:
            res = fn(pivot)
            if isinstance(res, (list, np.ndarray)):
                res = pd.Series(res, index=bee_ids)
            if isinstance(res, pd.Series):
                out[name] = res.reindex(bee_ids)
            else:
                out[name] = res
        except Exception as e:
            print(f"⚠ {name} failed on {base}: {e}")
            out[name] = np.nan

    if opt['save_pivots']:
        feather = Path(fpath).with_suffix("").with_name(
                      base.replace('_clean.csv', '_pivot_enriched.feather'))
        pivot.reset_index().to_feather(feather)

    return out

def job(arg): return analyse_one(*arg)

# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    opt = cli()

    funcs = [(n, f) for n, f in getmembers(baseFunctions)
             if isfunction(f) and f.__module__ == 'baseFunctions']
    if opt['brood']:
        funcs += [(n, f) for n, f in getmembers(broodFunctions)
                  if isfunction(f) and f.__module__ == 'broodFunctions']

    files = [str(p) for p in Path(opt['source']).rglob(f"*{opt['extension']}")]
    if opt['limit']:
        files = files[:opt['limit']]
    if not files:
        sys.exit("❌  No *_clean.csv files found.")

    it = [(fp, opt, funcs) for fp in files]
    if opt['cores'] > 1:
        with Pool(opt['cores']) as pool:
            results = list(tqdm(pool.imap_unordered(job, it),
                                total=len(files), desc="Files"))
    else:
        results = [job(a) for a in tqdm(it, desc="Files")]

    results = [r for r in results if r is not None]
    if not results:
        sys.exit("Nothing processed!")

    pd.concat(results, ignore_index=True).to_csv(opt['outFile'], index=False)
    print(f"✅  Saved {opt['outFile']}   ({len(results)} bees)")

if __name__ == "__main__":
    main()
