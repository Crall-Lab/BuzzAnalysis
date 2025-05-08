#!/usr/bin/env python3
"""
Runs all functions in baseFunctions.py on a dataset.
If the brood flag is used, it will also run functions in broodFunctions.py.
Brood processing functions have been moved to processBroodFunctions.py.
This version implements parallel processing and uses dtype/usecols for faster CSV reads,
as well as tqdm for progress reporting.
"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu), editor August Easton-Calabria (eastoncalabr@wisc.edu)'
__version__ = '0.0.4'

# Set to True to enable cProfile profiling (for debugging and bottleneck analysis)
ENABLE_CPROFILE = False

# Standard library and third-party imports
import pandas as pd
import numpy as np
import os
import sys
import argparse
from inspect import getmembers, isfunction
import baseFunctions
import processBroodFunctions
import broodFunctions
import warnings
import shapely
import data_cleaning
from multiprocessing import Pool
from tqdm import tqdm  # For progress reporting

# Suppress runtime and future warnings for clarity
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    """
    Parse command-line options.
    Returns:
        args: Parsed command-line options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='testCSV',
                        help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='.csv',
                        help='Suffix that identifies tracking data files. Defaults to ".csv".')
    parser.add_argument('--brood', '-b', type=str, default=None,
                        help='Path to brood data (if running brood functions).')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv',
                        help='Suffix for brood data files. Defaults to "_nest_image.csv".')
    parser.add_argument('--whole', '-w', action='store_true',
                        help='Do not split frame into two when analyzing (process as a whole).')
    parser.add_argument('--bombus', '-z', action='store_true',
                        help='If set, use bombus-specific logic for data (e.g. MJPEG conversion).')
    parser.add_argument('--outFile', '-o', type=str, default='Analysis.csv',
                        help='Path to output file. Defaults to "Analysis.csv".')
    # Options added by August:
    parser.add_argument('--interpolate', '-i', action='store_true',
                        help='Enable interpolation of missing data.')
    parser.add_argument('--remove-jumps', '-rj', type=int, default=None,
                        help='Minimum number of pixels a tag can jump between frames.')
    parser.add_argument('--real-fps', '-rfps', type=float, default=None,
                        help='Frame rate used (required if interpolation is enabled).')
    parser.add_argument('--max-interpolation-seconds', '-mis', type=float, default=None,
                        help='Maximum number of seconds to interpolate between frames (required if interpolation is enabled).')
    parser.add_argument('--save-interpolation-data', '-sid', type=bool, default=False,
                        help='Save the interpolated data for debugging purposes.')
    parser.add_argument('--cores', '-c', type=int, default=1,
                        help='Number of CPU cores for parallel processing.')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the number of files to process (for testing purposes).')
    return parser.parse_args()

def restructure_tracking_data(rawOneLR, opt, interpolated_path_name):
    """
    Rearranges and (optionally) interpolates the tracking data.
    
    Parameters:
        rawOneLR (DataFrame): Raw tracking data for one LR group.
        opt (dict): Parsed command-line options.
        interpolated_path_name (str): Path to save interpolated data (if enabled).
    
    Returns:
        DataFrame: Pivoted DataFrame with MultiIndex columns ('centroidX', ID) and ('centroidY', ID) for each tag.
    """
    # Validate input DataFrame
    required_columns = ['frame', 'ID', 'centroidX', 'centroidY']
    if not all(col in rawOneLR.columns for col in required_columns):
        raise ValueError(f"Input DataFrame missing required columns: {required_columns}")

    # Remove duplicate tag readings in the same frame, keeping the one closest to the nearest available tag reading 
    df, return_val = data_cleaning.return_duplicate_bees(rawOneLR)
    df = data_cleaning.drop_duplicates_clean(df, return_val) 
    
    # Apply jump removal if enabled
    if type(opt['remove_jumps']) == int:
        df = data_cleaning.remove_jumps(df)
    
    # Check for and remove any remaining duplicates before interpolation
    duplicates = df.duplicated(subset=['frame', 'ID'], keep=False)
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate frame-ID pairs after jump removal in {interpolated_path_name}. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['frame', 'ID'], keep='first')
    
    # Interpolate if enabled
    if opt['interpolate']:
        if opt.get('real_fps') is None or opt.get('max_interpolation_seconds') is None:
            raise ValueError("Interpolation is enabled but --real-fps and --max-interpolation_seconds must be provided.")
        max_seconds_gap = opt['max_interpolation_seconds']
        actual_frames_per_second = opt['real_fps']
        interpolated = data_cleaning.interpolate(df, max_seconds_gap, actual_frames_per_second)
    else:
        interpolated = df
    
    # Validate ID values are integers
    non_integer_ids = interpolated[~interpolated['ID'].astype(str).str.isdigit()]['ID'].unique()
    if len(non_integer_ids) > 0:
        print(f"Warning: Non-integer IDs found in {interpolated_path_name}: {non_integer_ids}. Filtering out.")
        interpolated = interpolated[interpolated['ID'].astype(str).str.isdigit()]
        interpolated['ID'] = interpolated['ID'].astype(int)
    
    # Check for duplicates after interpolation and aggregate if necessary
    duplicates = interpolated.duplicated(subset=['frame', 'ID'], keep=False)
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate frame-ID pairs after interpolation in {interpolated_path_name}. Aggregating by mean.")
        interpolated = interpolated.groupby(['frame', 'ID'], as_index=False).agg({
            'centroidX': 'mean',
            'centroidY': 'mean'
        })
    
    # Save interpolated data if enabled
    if opt['save_interpolation_data'] and interpolated_path_name:
        interpolated.to_csv(interpolated_path_name, index=False)
    
    # Pivot the data to have frame as index and ID as columns
    try:
        xs = interpolated.pivot(index="frame", columns='ID', values=['centroidX', 'centroidY'])
    except ValueError as e:
        print(f"Error pivoting data for {interpolated_path_name}: {e}")
        raise ValueError("Pivot failed despite deduplication. Check input data for inconsistencies.")
    
    # Ensure numeric data
    xs = xs.apply(pd.to_numeric, errors='coerce')
    
    # Log columns for debugging
    print(f"oneLR columns for {interpolated_path_name}: {xs.columns.tolist()}")
    
    return xs

def processBrood(base, oneLR, name, ext, broodSource):
    """
    Processes brood data for a file.
    
    Parameters:
        base (str): Basename of the file.
        oneLR (DataFrame): Processed tracking data with MultiIndex columns ('centroidX', ID), ('centroidY', ID).
        name (str): Group name ('Left', 'Right', etc.).
        ext (str): Brood extension string (unused here).
        broodSource (str): Path to brood data.
    
    Returns:
        DataFrame: The tracking data augmented with brood distance metrics.
    """
    # Build the brood data file path using the basename and the brood extension.
    ext = '-nest_image.csv'
    print(ext)
    broodMapPath = os.path.join(broodSource, '_'.join(base.split('_')[0:2]).replace('-', '_') + ext)
    print(broodMapPath)
    
    # Read brood data
    if os.path.exists(broodMapPath):
        full = pd.read_csv(broodMapPath)
    else:
        print('Missing nest image data, did you mean to run brood functions?')
        return oneLR
    
    # Debug: Log brood data info
    print(f"Brood data shape for {base}, {name}: {full.shape}")
    print(f"Brood x range: min={full['x'].min()}, max={full['x'].max()}")
    print(f"Brood labels: {full['label'].unique().tolist()}")
    print(f"Radius NaN count: {full['radius'].isna().sum()}, Non-NaN count: {full['radius'].notna().sum()}")
    
    # Check for centroidX and centroidY in MultiIndex
    if 'centroidX' not in oneLR.columns.levels[0] or 'centroidY' not in oneLR.columns.levels[0]:
        raise ValueError(f"No centroidX or centroidY columns found in oneLR for {base}, {name}")
    
    # Use fixed splitting threshold
    threshold = 2000
    print(f"Splitting threshold (fixed) for {base}, {name}: {threshold}")
    
    # Split brood data based on LR values using fixed threshold
    if name == 'Left':
        fullLeft = full[full['x'] < threshold]
        full = fullLeft
        print(f"Left split size for {base}: {fullLeft.shape}")
        if fullLeft.shape[0] < 2:
            print(f"Warning: Left split has {fullLeft.shape[0]} rows for {base}, {name}. Using full brood data.")
            full = pd.read_csv(broodMapPath)  # Revert to full data
    elif name == "Right":
        fullRight = full[full['x'] > threshold]
        full = fullRight
        print(f"Right split size for {base}: {fullRight.shape}")
        if fullRight.shape[0] < 2:
            print(f"Warning: Right split has {fullRight.shape[0]} rows for {base}, {name}. Using full brood data.")
            full = pd.read_csv(broodMapPath)  # Revert to full data
    
    # Check if full is empty after splitting
    if full.empty:
        print(f"Warning: No valid brood data after splitting for {base}, {name}. Returning oneLR unchanged.")
        return oneLR
    
    brood = full[full['label'] != 'Arena perimeter (polygon)']
    eggs = brood[brood['radius'].isna()]
    allbrood = brood.dropna(axis=0, subset=['radius'])
    
    # Debug: Log brood processing intermediate steps
    print(f"Brood shape after filtering: {brood.shape}, Eggs shape: {eggs.shape}, Allbrood shape: {allbrood.shape}")
    print(f"Brood labels after filtering: {brood['label'].unique().tolist()}")
    if eggs.empty:
        print(f"Warning: No egg objects (radius NaN) found for {base}, {name}. Egg-related metrics may be NaN.")
    
    # For each unique brood object, compute its centroid and append to the brood data.
    for i in set(eggs['object index']):
        try:
            egg = eggs[eggs['object index'] == i].reset_index()
            x = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.x
            y = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.y
            eggRow = pd.Series([i, egg.label[0], 'polygon', x, y, np.nan])
            eggRow.index = allbrood.columns
            allbrood = pd.concat([allbrood.T, eggRow], axis=1).T
        except Exception as e:
            print(e)
            with open('Error.csv', 'a') as errorFile:
                try:
                    errorFile.write(base + ', object ' + str(i) + ': ' + egg['label'][0] + '\n')
                    errorFile.write(str(e) + '\n')
                except Exception as e2:
                    errorFile.write('Cannot read label for ' + base + '\n')
            # Remove problematic rows from brood data
            if brood[brood['object index'] == i].shape[1] > 0:
                brood[brood['object index'] == i] = np.nan
                brood = brood.dropna(axis=0)
            if eggs[eggs['object index'] == i].shape[1] > 0:
                eggs[eggs['object index'] == i] = np.nan
                eggs = eggs.dropna(axis=0)
            if allbrood[allbrood['object index'] == i].shape[1] > 0:
                allbrood[allbrood['object index'] == i] = np.nan
                allbrood = allbrood.dropna(axis=0)
            continue
    allbrood = allbrood.reset_index(drop=True)
    
    # Debug: Log final allbrood
    print(f"Final allbrood labels: {allbrood['label'].unique().tolist()}")
    
    # Compute distance metrics using functions in processBroodFunctions
    distDF = processBroodFunctions.distanceFromCentroid(oneLR, allbrood)
    distDF2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    distDF3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)
    return pd.concat([oneLR, distDF, distDF2, distDF3], axis=1)

def process_file(file_path, opt, funcs):
    """
    Process a single tracking file and return a DataFrame with analysis results.
    
    Steps:
      1. Read the CSV file using specified dtypes and usecols.
      2. Assign Left/Right (or Whole) labels.
      3. Restructure (and optionally interpolate) the tracking data.
      4. Optionally process brood data.
      5. Apply each analysis function from baseFunctions (and broodFunctions if enabled).
      6. Return a DataFrame including file metadata and analysis results.
    
    Returns:
        DataFrame: Analysis results for the file, or None if processing fails.
    """
    try:
        basename = os.path.basename(file_path)
        if opt['bombus']:
            if 'mjpeg' in basename and os.path.exists(file_path.replace(".mjpeg", opt['extension'])):
                v = file_path
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                dtype_spec = {
                    "filename": "str",
                    "colony number": "str",
                    "frame": "int64",
                    "ID": "int64",
                    "centroidX": "float64",
                    "centroidY": "float64",
                    "frontX": "float64",
                    "frontY": "float64"
                }
                usecols = ["filename", "colony number", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"]
                trackingResults = pd.read_csv(file_path.replace(".mjpeg", opt['extension']),
                                              dtype=dtype_spec, usecols=usecols)
            else:
                return None
        else:
            if opt['extension'] in basename:
                v = file_path
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                dtype_spec = {
                    "filename": "str",
                    "colony number": "str",
                    "frame": "int64",
                    "ID": "int64",
                    "centroidX": "float64",
                    "centroidY": "float64",
                    "frontX": "float64",
                    "frontY": "float64"
                }
                usecols = ["filename", "colony number", "frame", "ID", "centroidX", "centroidY", "frontX", "frontY"]
                trackingResults = pd.read_csv(v, dtype=dtype_spec, usecols=usecols)
            else:
                return None
    except Exception as e:
        print(f'Error reading file {basename}: {e}')
        return None

    # Assign the LR label: if whole is enabled, set to "Whole"; otherwise, assign based on centroidX vs. mean.
    if opt['whole']:
        trackingResults['LR'] = "Whole"
    else:
        trackingResults['LR'] = np.where(trackingResults['centroidX'] < 2000, "Left", "Right")

    # Process data for each LR group (e.g., Left and Right)
    fullAnalysis = pd.DataFrame()
    datasets = trackingResults.groupby('LR')
    for name, rawOneLR in datasets:
        # Build a filename for saving interpolated data (if interpolation is enabled)
        data_path_name = os.path.join(os.path.dirname(v), os.path.splitext(basename)[0])
        interpolated_path_name = data_path_name + '_' + name + '_interpolated.csv'
        # Get ordered bee IDs from oneLR columns
        oneLR = restructure_tracking_data(rawOneLR, opt, interpolated_path_name)
        # Extract bee IDs from MultiIndex level 1
        try:
            ordered_bee_ids = oneLR.columns.levels[1].tolist()
        except AttributeError:
            print(f"Error: oneLR columns are not MultiIndex for {basename}, {name}: {oneLR.columns.tolist()}")
            continue
        if not ordered_bee_ids:
            print(f"Warning: No valid bee IDs found in columns for {basename}, {name}")
            continue
        analysis = pd.DataFrame(index=ordered_bee_ids)
        analysis['LR'] = name
        analysis['ID'] = analysis.index
        if opt['brood']:
            oneLR = processBrood(basename, oneLR, name, opt['broodExtension'], opt['brood'])
        # Apply each analysis function from baseFunctions (and broodFunctions if enabled)
        for test in funcs:
            try:
                result = test[1](oneLR)
                if result is not None:
                    if isinstance(result, np.ndarray):
                        result = pd.Series(result, index=ordered_bee_ids)
                    elif isinstance(result, list):
                        result = pd.Series(result, index=ordered_bee_ids)
                    elif isinstance(result, pd.Series) and result.isna().all():
                        print(f"Warning: {test[0]} returned all NaN for {basename}")
                        result = np.nan
                    analysis[test[0]] = result
                else:
                    print(f"Warning: {test[0]} returned None for {basename}")
                    analysis[test[0]] = np.nan
            except Exception as e:
                print(f"Error running {test[0]}: {e}")
                analysis[test[0]] = np.nan
        fullAnalysis = pd.concat([fullAnalysis, analysis], axis=0)
    
    # Create a DataFrame for the file that includes metadata (workerID, Date, Time) and analysis results.
    oneVid = pd.DataFrame(index=fullAnalysis.index)
    oneVid['pi_ID'] = workerID
    oneVid['bee_ID'] = oneVid.index
    oneVid['Date'] = Date
    oneVid['Time'] = Time
    oneVid = pd.concat([oneVid, fullAnalysis], axis=1)
    return oneVid

def main():
    """
    Main function:
      1. Parses command-line options.
      2. Loads existing output if available.
      3. Retrieves analysis functions from baseFunctions (and broodFunctions if enabled).
      4. Walks the source directory to build a list of files to process.
      5. Optionally limits the number of files (for testing).
      6. Processes files either in parallel or serially, reporting progress with tqdm.
      7. Combines all results and writes them to the output CSV.
    """
    opt = vars(parse_opt())
    
    # Load existing output analysis file if it exists to add to it
    if os.path.exists(opt['outFile']):
        print('Found existing', opt['outFile'], 'and will add to it.')
        output = pd.read_csv(opt['outFile'])
    else:
        output = pd.DataFrame()

    # Get analysis functions from baseFunctions and (if enabled) broodFunctions.
    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if opt['brood']:
        funcs += [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    # Walk the source directory to build a list of files to process.
    file_list = []
    for root, dirs, files in os.walk(opt['source']):
        for f in files:
            full_path = os.path.join(root, f)
            if opt['bombus']:
                if 'mjpeg' in f and os.path.exists(full_path.replace(".mjpeg", opt['extension'])):
                    file_list.append(full_path)
            else:
                if opt['extension'] in f:
                    file_list.append(full_path)
    print("Found", len(file_list), "files to process.")

    # If a limit is set, restrict the number of files processed (useful for testing).
    if opt.get('limit') is not None:
        file_list = file_list[:opt['limit']]

    # Process files using multiprocessing if more than 1 core is specified, with tqdm progress reporting.
    if opt['cores'] > 1:
        print("Processing files in parallel using", opt['cores'], "cores...")
        with Pool(processes=opt['cores']) as pool:
            results = list(tqdm(
                            pool.imap_unordered(process_file_wrapper, [(file_path, opt, funcs) for file_path in file_list]),
                            total=len(file_list),
                            desc="Processing files"
                        ))
            successful_results = [r for r in results if r is not None]
            print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")
    else:
        print("Processing files serially...")
        results = [process_file(file_path, opt, funcs) for file_path in tqdm(file_list, desc="Processing files")]
        successful_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")

    # Filter out None results and combine all successful ones.
    results = [res for res in results if res is not None]
    if results:
        combined = pd.concat(results, ignore_index=True, axis=0)
        output = pd.concat([output, combined], ignore_index=True, axis=0)
    else:
        print("No files processed successfully.")
    
    # Write the final combined output to the specified CSV file.
    output.to_csv(opt['outFile'], index=False)
    print("All done! Analysis saved to", opt['outFile'])
    return 0

def process_file_wrapper(args):
    return process_file(*args)    

if __name__ == "__main__":
    # If profiling is enabled, run with cProfile; otherwise, run normally.
    if ENABLE_CPROFILE:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        status = 0  # default exit status
        try:
            profiler.enable()
            status = main()
        except KeyboardInterrupt:
            print("Execution interrupted by user!")
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs()
            print("\n--- Profiling Results (Top 20 by cumtime) ---")
            stats.sort_stats("cumtime").print_stats(20)
            print("\n--- Profiling Results (Top 20 by tottime) ---")
            stats.sort_stats("tottime").print_stats(20)
            sys.exit(status)
    else:
        start_time = pd.Timestamp.now()
        status = main()
        end_time = pd.Timestamp.now()
        print(f"Execution took {end_time - start_time}.")
        sys.exit(status)