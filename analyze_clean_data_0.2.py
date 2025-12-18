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
from aux import mean_centroids_across_files
import pdb
from params import colony_number_position, Date_position, H_position, M_position, S_position

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
                           values=['centroidX','centroidY'])
              .sort_index(axis=1)
              .apply(pd.to_numeric, errors='coerce'))

def _find_brood_map(stem: str, root: str, ext: str):
    """Recursive match  <stem>*<ext>  underneath *root*."""
    #print("Brood info")
    #print(f"root: {root}")
    #print(f"stem: {stem}")
    #print(f"ext: {ext}")
    
    ext_clean = ext.lstrip('_-')

    #print(f"ext_clean: {ext_clean}")
    #print(Path(root).rglob(f"{stem}*"))
    #hits = sorted(Path(root).rglob(f"{stem}*")) #{ext_clean}"))
    #hits = []
    

    print(os.path.exists(root))
    for paths, dirs, files in os.walk(root):
 
        for path in paths:
            print(path)
            for file in files:

                lower_file = file.replace("-", "_")
                lower_stem = stem.replace("-", "_")
                print(f"file: {file}")
                print(f"stem: {stem}")
                if lower_stem in lower_file:
                    print("stem in file")
                else:
                    print("stem not in file")
                    continue
                if file.endswith(ext_clean):
                    path_to_brood_csv = os.path.join(root,file)
                    if os.path.exists(path_to_brood_csv):
                        print("Found brood file")
                        print(path_to_brood_csv)
                        return path_to_brood_csv



#    print(len(hits))
#    if not hits:
#        print("Error: Stem string for the brood map path couldnt be found")
#        return None
#    else:
#        for hit in hits:
#            if hit.endswith(ext):
#                return hit

def processBrood_test(basename, oneLR, LR, brood_dir, brood_ext):
    """Attach brood-distance matrices (if map present)."""

    # --- locate brood map file ---
    stem = '_'.join(basename.split('_')[0:2]).replace('-', '_')
    mp = _find_brood_map(stem, brood_dir, brood_ext)
    if mp is None:
        print("ERROR: Brood map could not be found")
        return oneLR

    full = pd.read_csv(mp)

    # --- Apply left/right filtering ---
    THR = 2000
    if LR == 'Left':
        full = full[full['x'] < THR]
    elif LR == 'Right':
        full = full[full['x'] > THR]

    # --- Remove arena perimeter, which should never be processed ---
    brood = full[full["label"] != "Arena perimeter (polygon)"].copy()

    # --- Partition brood objects cleanly ---
    circles = brood[
        brood["shape"].str.contains("circle", case=False) &
        brood["radius"].notna()
    ].copy()

    polygons = brood[
        brood["shape"].str.contains("polygon", case=False)
    ].copy()

    # Everything else (e.g. rectangles) appears only in centroid-based distances
    # They do *not* go into circle or polygon distance calculations.
    other = brood[
        ~brood.index.isin(circles.index) &
        ~brood.index.isin(polygons.index)
    ].copy()

    # --- Combine for centroid-based distances ---
    # Centroid-func can safely handle all brood objects
    all_for_centroid = pd.concat([circles, polygons, other], ignore_index=True)

    #print(f"[processBrood] circles:  {circles.shape}")
    #print(f"[processBrood] polygons: {polygons.shape}")
    #print(f"[processBrood] other:    {other.shape}")
    #print(f"[processBrood] centroid total: {all_for_centroid.shape}")

    # --- Call each distance method with correct subset ---
    d1 = processBroodFunctions.distanceFromCentroid_new(oneLR, all_for_centroid)
    #print(f"[processBrood] d1 (centroid) shape: {d1.shape}")

    d2 = processBroodFunctions.minimumDistanceCircle_new(circles, oneLR)
    #print(f"[processBrood] d2 (circle) shape:   {d2.shape}")

    d3 = processBroodFunctions.minimumDistancePolygon_new(oneLR, polygons)
    #print(f"[processBrood] d3 (polygon) shape:  {d3.shape}")

    # --- Combine all ---
    return pd.concat([oneLR, d1, d2, d3], axis=1)


def processBrood(basename, oneLR, LR, brood_dir, brood_ext):
    """Attach brood-distance matrices (if map present)."""
    stem = '_'.join(basename.split('_')[0:2]).replace('-', '_')
    #print(f"stem: {stem}")
    #print(f"brood_dir: {brood_dir}")
    #print(f"brood_ext: {brood_ext}")
    mp = _find_brood_map(stem, brood_dir, brood_ext)
    if mp is None:
        print("ERROR: Brood map could not be found")
        #raise TypeError
        return oneLR

    full = pd.read_csv(mp)
    THR = 2000
    if LR == 'Left':  full = full[full['x'] < THR]
    elif LR == 'Right': full = full[full['x'] > THR]

    brood = full[full['label']!='Arena perimeter (polygon)'].reset_index(drop=True)
    #eggs  = brood[brood['radius'].isna()]
    #allb  = brood.dropna(subset=['radius'])
    
    circles = brood[
        brood["shape"].str.contains("circle", case=False) 
        & brood["radius"].notna()
        ].reset_index(drop=True)
    
    polygons = brood[
    brood["shape"].str.contains("polygon", case=False)
    ].reset_index(drop=True)
    

    #print(f"allb shape: {allb.shape}")
    #for df in (brood, eggs, allb):
    #    df.reset_index(drop=True, inplace=True)

    d1 = processBroodFunctions.distanceFromCentroid(oneLR, brood)
    #print(f"d1 shape: {d1.shape}")
    d2 = processBroodFunctions.minimumDistanceCircle(circles, oneLR)
    #print(f"d2 shape: {d2.shape}")
    d3 = processBroodFunctions.minimumDistancePolygon(oneLR, polygons)
    #print(f"d3 shape: {d3.shape}")
    #pdb.set_trace()
    return pd.concat([oneLR, d1, d2, d3], axis=1)

# ══════════════════════════════════════════════════════════════════════════
#  Per-file job
# ══════════════════════════════════════════════════════════════════════════
def analyse_one(fpath, opt, funcs, social_center):
    base = os.path.basename(fpath)
    df = pd.read_csv(fpath, dtype={
        "frame":"int32","ID":"int32",
        "centroidX":"float32","centroidY":"float32",
        "frontX":"float32","frontY":"float32"})
    
    #Just for Augs 2021 dataset
    #col, workerdatehms = base.split("_", 1) #split just the first _
    #worker, DateHMS, *_ = workerdatehms.split("-", 1)
    #Date, HMS, W, C = DateHMS.split("_") #Ignore W and C
    #H, M, S = HMS.split("-")

    w = colony_number_position

    if len(w) == 2:
        worker = base[ w[0] : (w[1]+1) ]
    elif len(w) == 1:
        worker = base[ w[0] ]

    #worker = base[ w[0] : (w[1]+1) ]
    print(worker)
    d = Date_position
    Date = base[ d[0] : (d[1]+1) ]
    print(Date)
    h = H_position
    H = base[ h[0] : (h[1]+1) ]
    print(H)
    m = M_position
    M = base[ m[0] : (m[1]+1) ]
    print(M)
    s = S_position
    S = base[ s[0] : (s[1]+1)]
    print(S)

    LRm = re.search(r'_(Left|Right|Whole)_', base)
    LR  = LRm.group(1) if LRm else 'Whole'

    pivot = pivot_clean(df)              # build wide table
    #print(fpath)
    #print(pivot.head)
    movement_metrics(pivot)              # adds speed + activity columns

    if opt['brood']:
        pivot = processBrood_test(base, pivot, LR,
                             brood_dir=opt['brood'],
                             brood_ext=opt['broodExtension'])
        
        #dupes = pivot.columns[pivot.columns.duplicated()]
        #print("Duplicated column names:")
        #for d in dupes:
        #    print(d)

        #print("Before reset_index: ", pivot.columns.nlevels)
        #print("Are column names unique before reset?", pivot.columns.is_unique)
        #print("Are column names unique after reset?", pivot.reset_index().columns.is_unique)
        


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

        if name == "distSC":
            try:
                res = fn(pivot, social_center)
                if isinstance(res, (list, np.ndarray)):
                    res = pd.Series(res, index=bee_ids)
                if isinstance(res, pd.Series):
                    out[name] = res.reindex(bee_ids)
                else:
                    out[name] = res
            except Exception as e:
                print(f"⚠ {name} failed on {base}: {e}")
                out[name] = np.nan

        else:
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
        
        #raise Exception
        pivot.to_feather(feather)
        print(f"Feather file: {feather}")
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

    #print(Path(opt['source']))

    files = [str(p) for p in Path(opt['source']).rglob("*_Whole_clean.csv")] #f"*{opt['extension']}")]

    if opt['limit']:
        files = files[:opt['limit']]
    if not files:
        sys.exit("❌  No *_clean.csv files found.")

    example_file = files[0]
    example_filename = os.path.basename(example_file)
    print(f"This is an example filename from your data:\n{example_filename}")
    print(f"It is {len(example_filename)} characters long, with the following being the position of each character:")
    char_dict = {}
    for i in range(len(example_filename)):
            print(f"[ position {i:02d}: {example_filename[i]} ] ")   
    print("\n")
    try:
        if len(colony_number_position) == 2:
            filename_colony_number = example_filename[ colony_number_position[0] : (colony_number_position[1]+1) ]
        elif len(colony_number_position) == 1:
            filename_colony_number = example_filename[ colony_number_position[0] ]
        if len(filename_colony_number) == 0:
            filename_colony_number = "NONE"
        print(f"The characters representing the colony number are: { filename_colony_number}")

        filename_date = example_filename[ Date_position[0] : (Date_position[1]+1) ]
        if len(filename_date) == 0:
            filename_date = "NONE"
        print(f"The characters representing the date are: { filename_date }")

        filename_hours = example_filename[ H_position[0] : (H_position[1]+1) ]
        if len(filename_hours) == 0:
            filename_hours = "NONE"
        print(f"The characters representing the hour are: { filename_hours}")

        filename_minutes = example_filename[ M_position[0] : (M_position[1]+1) ]
        if len(filename_minutes) == 0:
            filename_minutes = "NONE"
        print(f"The characters representing the minute are: { filename_minutes }")

        filename_seconds = example_filename[ S_position[0] : (S_position[1]+1) ]
        if len(filename_seconds) == 0:
            filename_seconds = "NONE"
        print(f"The characters representing the seconds are: { filename_seconds }")

        yes_or_no = input("Are these correct? Enter Yes or No: ")
    except:
        print("ERROR! Issue occurred while trying to parse the colony number, date and time positions from your filename format. Ending the program. Please open the params.py file in the BuzzAnalysis folder and change the relevant variable values to reflect the correct positions in your filename. Bye for now")
    if yes_or_no == "No" or yes_or_no == "no":
        print("Ending the program. Please open the params.py file in the BuzzAnalysis folder and change the relevant variable values to reflect the correct positions in your filename. Bye for now!")
        return
    elif yes_or_no == "Yes" or yes_or_no == "yes":
        print("Perfect! Continuing on!")
    #:If not, end this program and make sure to change the values of the relevant variables in params.py")

    #files = []
    #for paths, dirs, files in os.walk('/media/august/Seagate Expansion Drive/Spring_2021_low_elevation/Colony_1 (tagged 06-07-21)/spring_2021_video/col_1-2021-06-10'):

    #    for path in paths:
    #        print(path)
    #        for file in files:
    #            print(file)
    #            if file.endswith("_Whole_clean.csv"):
    #                print(path)
    #                print(file)
    #               print(path,file)
    #                files.append(file)
    
    print("Calculating the social center from your data now...")
    mean_sc_dict = mean_centroids_across_files(files, chunksize=None)
    mean_x = mean_sc_dict["mean_centroidX"]
    mean_y = mean_sc_dict["mean_centroidY"]
    social_center = [mean_x, mean_y]

    it = [(fp, opt, funcs, social_center) for fp in files]
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
