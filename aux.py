from __future__ import annotations

#!/usr/bin/env python3

"""Collection of auxillary functions for baseFunctions.py and broodFunctions.py."""

__appname__ = 'baseFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'


#imports
import numpy as np
import pandas as pd
from scipy import spatial
from params import *
from pathlib import Path
from typing import Iterable, Union, Dict, Any, Optional

_behavior_frame_per_sec = frame_per_sec
_behavior_max_gap_seconds = globals().get("max_behavior_gap_seconds", 3)
_behavior_speed_cutoff = digital_noise_speed_cutoff


def configure_behavior_timing(frame_rate=None, max_gap_seconds=None, speed_cutoff=None):
    """Set run-level timing controls for behavior metrics."""
    global _behavior_frame_per_sec, _behavior_max_gap_seconds, _behavior_speed_cutoff

    if frame_rate is not None:
        frame_rate = float(frame_rate)
        if frame_rate <= 0:
            raise ValueError("frame_rate must be greater than zero")
        _behavior_frame_per_sec = frame_rate

    if max_gap_seconds is not None:
        max_gap_seconds = float(max_gap_seconds)
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be greater than zero")
        _behavior_max_gap_seconds = max_gap_seconds

    if speed_cutoff is not None:
        speed_cutoff = float(speed_cutoff)
        if speed_cutoff < 0:
            raise ValueError("speed_cutoff must be greater than or equal to zero")
        _behavior_speed_cutoff = speed_cutoff


#def nest_social_center(oneLR):
#    """Generic function to calculate the nest social center from standard pandas array with centroid coordinates."""
#    mean_x = np.nanmean(oneLR['centroidX'].to_numpy())
#    mean_y = np.nanmean(oneLR['centroidY'].to_numpy())
    
#    return (mean_x, mean_y)

PathLike = Union[str, Path]

def mean_centroids_across_files(
    csv_files: Iterable[PathLike],
    *,
    x_col: str = "centroidX",
    y_col: str = "centroidY",
    chunksize: Optional[int] = 250_000,
) -> Dict[str, Any]:
    """
    Compute the grand mean of centroidX and centroidY across *all rows* in a list of CSV files.

    - Ignores NaNs in centroid columns.
    - Streams files in chunks to avoid loading everything into memory (set chunksize=None to read whole files).
    - Returns means plus useful counts.

    Returns dict with:
      mean_centroidX, mean_centroidY, n_rows_used_x, n_rows_used_y, n_files, files_skipped
    """
    sum_x = 0.0
    sum_y = 0.0
    n_x = 0
    n_y = 0

    files_skipped = []
    csv_files = list(csv_files)

    for fp in csv_files:
        fp = Path(fp)
        if not fp.exists():
            files_skipped.append((str(fp), "missing"))
            continue

        try:
            if chunksize:
                reader = pd.read_csv(fp, usecols=lambda c: c in {x_col, y_col}, chunksize=chunksize)
                for chunk in reader:
                    if x_col in chunk:
                        x = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy()
                        mask = np.isfinite(x)
                        sum_x += float(x[mask].sum())
                        n_x += int(mask.sum())
                    if y_col in chunk:
                        y = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy()
                        mask = np.isfinite(y)
                        sum_y += float(y[mask].sum())
                        n_y += int(mask.sum())
            else:
                df = pd.read_csv(fp, usecols=lambda c: c in {x_col, y_col})
                if x_col in df:
                    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
                    mask = np.isfinite(x)
                    sum_x += float(x[mask].sum())
                    n_x += int(mask.sum())
                if y_col in df:
                    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
                    mask = np.isfinite(y)
                    sum_y += float(y[mask].sum())
                    n_y += int(mask.sum())

        except ValueError as e:
            # happens if neither x_col nor y_col exists in the CSV (usecols selection fails)
            files_skipped.append((str(fp), f"missing columns ({e})"))
        except Exception as e:
            files_skipped.append((str(fp), f"read error ({type(e).__name__}: {e})"))

    mean_x = (sum_x / n_x) if n_x else np.nan
    mean_y = (sum_y / n_y) if n_y else np.nan

    return {
        "mean_centroidX": mean_x,
        "mean_centroidY": mean_y,
        "n_rows_used_x": n_x,
        "n_rows_used_y": n_y,
        "n_files": len(csv_files),
        "files_skipped": files_skipped,
    }


def movement_metrics(oneLR, frame_rate=None, max_gap_seconds=None, speed_cutoff=None):
    """Return activity and per-frame speed, ignoring movements across long gaps."""
    if frame_rate is None:
        frame_rate = _behavior_frame_per_sec
    if max_gap_seconds is None:
        max_gap_seconds = _behavior_max_gap_seconds
    if speed_cutoff is None:
        speed_cutoff = _behavior_speed_cutoff

    frame_rate = float(frame_rate)
    max_gap_seconds = float(max_gap_seconds)
    speed_cutoff = float(speed_cutoff)
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than zero")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero")
    if speed_cutoff < 0:
        raise ValueError("speed_cutoff must be greater than or equal to zero")

    frame_values = pd.Series(oneLR.index, index=oneLR.index)
    frame_values = pd.to_numeric(frame_values, errors="coerce")
    frame_gap = frame_values.diff()
    max_frame_gap = frame_rate * max_gap_seconds
    valid_gap = (frame_gap > 0) & (frame_gap <= max_frame_gap)

    displacement = np.sqrt(oneLR['centroidX'].diff(axis=0)**2 + oneLR['centroidY'].diff(axis=0)**2)
    speed = displacement.div(frame_gap.replace(0, np.nan), axis=0)
    speed = speed.where(valid_gap, np.nan)
    act = speed > speed_cutoff
    act = 1*act
    act[np.isnan(speed)] = np.nan

    return act, speed

def extract_bee_locs(oneLR, fn):
    """Restructures coordinates of tag IDs through time."""
    #Get tracking data
    xs=oneLR['centroidX'].loc[fn]
    ys=oneLR['centroidY'].loc[fn]
    bee_locs = np.column_stack((xs,ys))
    
    return bee_locs

def interbee_distance_matrix(oneLR):
    """Calculates distance between bees through time"""
    frames = tuple(oneLR.index) #Get unique frames
    inter_bee_dist=[] #Create empty output array
    
    for fn in frames:
        bee_locs = extract_bee_locs(oneLR, fn)
        ib_dist = spatial.distance.pdist(bee_locs)
        ib_dist = spatial.distance.squareform(ib_dist)

        inter_bee_dist.append(ib_dist)

    return inter_bee_dist
