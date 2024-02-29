#!/usr/bin/env python3

"""Collection of auxillary functions for baseFunctions.py and broodFunctions.py."""

__appname__ = 'baseFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

#imports
import numpy as np
from scipy import spatial
import params

def nest_social_center(oneLR):
    """Generic function to calculate the nest social center from standard pandas array with centroid coordinates."""
    mean_x = np.nanmean(oneLR['centroidX'].to_numpy())
    mean_y = np.nanmean(oneLR['centroidY'].to_numpy())
    
    return (mean_x, mean_y)

def movement_metrics(oneLR):
    """Returns two pd.Series(), act is whether a bee is moving in a frame, speed is the speed of a bee in a frame."""
    speed = np.sqrt(oneLR['centroidX'].diff(axis=0)**2 + oneLR['centroidY'].diff(axis=0)**2)
    act = speed > params.digital_noise_speed_cutoff
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