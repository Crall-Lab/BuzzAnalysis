#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset. Do not add functions that require brood data here, please put those in broodFunctions.py instead."""

__appname__ = 'baseFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

#imports
import numpy as np
from aux import *
from params import *

#Tests (all tests should be vectorized)
def trackedFrames(oneLR):
    """Calculates number of frames where at least one tags is detected"""
    return np.sum(~np.isnan(oneLR['centroidX']))

def distSC(oneLR):
    """Given dataframe, return 1-D array containing average distance to social center."""
    sc = nest_social_center(oneLR)
    xd = oneLR['centroidX'] - sc[0]
    yd = oneLR['centroidY'] - sc[1]
    tot_d = np.sqrt(xd**2 + yd**2)
    mean_scd = np.nanmean(tot_d, axis=0)
    return mean_scd

def meanAct(oneLR):
    """Gives mean ratio of time spent moving"""
    act = movement_metrics(oneLR)[0]
    return np.nanmean(act, axis=0)

def meanSpeed(oneLR):
    """Gives mean moving speed"""
    act, speed = movement_metrics(oneLR)
    speed[act != 1] = np.nan #For moving speed matrix, replace all frames where bees are not detected as moving with nans
    return np.nanmean(speed, axis=0)

def meanIBD(oneLR):
    """Calculates mean distance to other bees in cm."""
    if len(oneLR.columns) == 2:
        print("No interactions possible, only one tag found in video.")
        return None
    ib_dists = interbee_distance_matrix(oneLR)
    ib_dist_mean = np.nanmean(ib_dists, axis=0)
    self_ind = ib_dist_mean == 0
    ib_dist_mean[self_ind] = np.nan
    ibd_mean = np.nanmean(ib_dist_mean, axis=1)
    return ibd_mean/pixels_per_cm

def totalInt(oneLR):
    """Calculates total number of interactions between bees in a video."""
    if len(oneLR.columns) == 2:
        print("No interactions possible, only one tag found in video.")
        return None
    ib_dists = interbee_distance_matrix(oneLR)
    for fn in range(len(ib_dists)):
        frame_dists= ib_dists[fn]
        frame_dists[frame_dists==0] = np.Inf
        int_mat = frame_dists < interaction_distance_cutoff
        int_mat = 1*int_mat

        if fn == 0:
            int_sums = int_mat
        else:
            int_sums = int_sums + int_mat
    
    return np.nansum(int_sums, axis=0)

def totalIntFrames(oneLR):
    """Calculates number of frames in a video where at least one interaction is detected."""
    if len(oneLR.columns) == 2:
        print("No interactions possible, only one tag found in video.")
        return None
    ib_dists = interbee_distance_matrix(oneLR)
    ib_dist_mean = np.nanmean(ib_dists, axis=0)
    self_ind = ib_dist_mean == 0
    ib_dist_mean[self_ind] = np.nan

    for fn in range(len(ib_dists)):
        frame_dists= ib_dists[fn]
        frame_dists[frame_dists==0] = np.Inf
        int_mat = frame_dists < interaction_distance_cutoff
        int_mat = 1*int_mat

        if fn == 0:
            int_sums = int_mat
        else:
            int_sums = int_sums + int_mat
    #Count frames where pairs of bees were both tracked (and thus potentially interacting)
    countable_frames = np.count_nonzero(~np.isnan(ib_dists), axis=0)
    countable_frames[self_ind] = 0

    return np.nansum(countable_frames, axis=0)

def meanX(oneLR):
    """Mean x-coordinate of bee."""
    return oneLR.centroidX.mean()

def meanY(oneLR):
    """Mean y-coordinate of bee."""
    return oneLR.centroidY.mean()

def varSpeed(oneLR):
    """Varience of speed of bee."""
    act, speed = movement_metrics(oneLR)
    return speed.var()

def medianMinDistToOthers(oneLR):
    """Median minimum distance to other bees in cm."""
    if len(oneLR.columns) == 2:
        print("No interactions possible, only one tag found in video.")
        return None
    ibm = np.array(interbee_distance_matrix(oneLR))
    beeN = np.arange(ibm.shape[1])
    ibm[:, beeN, beeN] = np.nan
    minDist = np.nanmin(ibm, axis=1)
    return np.nanmedian(minDist,axis=0)/pixels_per_cm
