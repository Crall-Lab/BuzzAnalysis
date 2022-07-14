#!/usr/bin/env python3 -u

"""Runs analysis that is independent of nest structure data."""

__appname__ = 'baseFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

#imports
import os
import sys
import pandas as pd
import numpy as np
from scipy import spatial
import params
import baseFunctions

#To run any test
def runTestLR(trackingResults, test):
    """Calculates average distance to social center of a hive."""
    metric = pd.Series(index = trackingResults.LR.unique(), dtype='float64')
    for lr in trackingResults.LR.unique():
        oneLR = trackingResults[trackingResults.LR == lr]
        metric[lr] = test(oneLR)
    return metric


#Called functions
def nest_social_center(oneLR):
    """Generic function to calculate the nest social center from standard pandas array with centroid coordinates."""
    mean_x = np.nanmean(oneLR['centroidX'].to_numpy())
    mean_y = np.nanmean(oneLR['centroidY'].to_numpy())
    return (mean_x, mean_y)

def movement_metrics(oneLR):
    """Returns two pd.Series(), act is whether a bee is moving in a frame, speed is the speed of a bee in a frame."""
    speed = pd.Series(dtype = "float64")
    for id in oneLR.ID.unique():
        oneID = oneLR[oneLR["ID"] == id]
        speedID = np.sqrt(oneID['centroidX'].diff()**2 + oneID['centroidY'].diff()**2)/oneID['frame'].diff() #to be discussed
        speed = pd.concat([speed, speedID])
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


#Tests
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
    moving_speed =speed
    moving_speed[act != 1] = np.nan #For moving speed matrix, replace all frames where bees are not detected as moving with nans
    return np.nanmean(moving_speed, axis=0)

def meanIBD(oneLR):
    """Calculates mean distance to other bees in cm."""
    ib_dists = interbee_distance_matrix(oneLR)
    ib_dist_mean = np.nanmean(ib_dists, axis=0)
    self_ind = ib_dist_mean == 0
    ib_dist_mean[self_ind] = np.nan
    ibd_mean = np.nanmean(ib_dist_mean, axis=1)
    return ibd_mean/params.pixels_per_cm

def totalInt(oneLR):
    """Calculates total number of interactions between bees in a video."""
    ib_dists = interbee_distance_matrix(oneLR)
    for fn in range(len(ib_dists)):
        frame_dists= ib_dists[fn]
        frame_dists[frame_dists==0] = np.Inf
        int_mat = frame_dists < params.interaction_distance_cutoff
        int_mat = 1*int_mat

        if fn == 0:
            int_sums = int_mat
        else:
            int_sums = int_sums + int_mat
    
    return np.nansum(int_sums, axis=0)

def totIntFrames(oneLR):
    """Calculates number of frames in a video where at least one interaction is detected."""
    ib_dists = interbee_distance_matrix(oneLR)
    ib_dist_mean = np.nanmean(ib_dists, axis=0)
    self_ind = ib_dist_mean == 0
    ib_dist_mean[self_ind] = np.nan

    for fn in range(len(ib_dists)):
        frame_dists= ib_dists[fn]
        frame_dists[frame_dists==0] = np.Inf
        int_mat = frame_dists < params.interaction_distance_cutoff
        int_mat = 1*int_mat

        if fn == 0:
            int_sums = int_mat
        else:
            int_sums = int_sums + int_mat
    #Count frames where pairs of bees were both tracked (and thus potentially interacting)
    countable_frames = np.count_nonzero(~np.isnan(ib_dists), axis=0)
    countable_frames[self_ind] = 0

    return np.nansum(countable_frames, axis=0)


def main(argv):
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    vids=[]
    for dir, subdir, files in os.walk(argv[1]):
        for f in files:
            if "mjpeg" in f:
                vids.append(os.path.join(dir,f))

    output = pd.DataFrame(columns = ['worker', 'Date', 'Time', 'ID']+argv[2:-1])
    for v in vids:
        workerID = v.split("worker")[1].split("-")[0]
        Date, Time = v.split("_")
        Date = Date.split("worker" + workerID + "-")[1]
        Time = Time.replace(".mjpeg", "").replace("-", ":")
        try:
            trackingResults = pd.read_csv(v.replace(".mjpeg", ".csv"))
        
        except Exception as e:
            print("Error: cannot read " + v.replace(".mjpeg", ".csv"))
            print(e)
            existingData = [workerID, Date, Time]
            addOn = ["" for i in range(len(argv[2:-1])+1)]
            row = existingData + addOn
            analysis = pd.DataFrame([row], columns = ['worker', 'Date', 'Time', 'ID']+argv[2:-1])
            analysis.worker = workerID
            analysis.Date = Date
            analysis.Time = Time
            output = pd.concat([output, analysis], ignore_index=True, axis = 0)
            continue
        
        if argv[-1] == "True":
            trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
            trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
            trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"
            LR = ["Left", "Right"]
        else:
            trackingResults['LR'] = "Whole"
            LR = ["Whole"]
        analysis = pd.DataFrame(index=LR, columns=['worker', 'Date', 'Time', 'ID']+argv[2:-1])
        analysis.worker = workerID
        analysis.Date = Date
        analysis.Time = Time
        analysis.ID = analysis.index
        for test in argv[2:-1]:
            try:
                analysis[test] = 0
                analysis[test] = runTestLR(trackingResults, getattr(baseFunctions, test))
            except Exception as e:
                print(test + " cannot be run on " + v.replace(".mjpeg", ".csv"))
                print(e)
                continue
    
        output = pd.concat([output, analysis], ignore_index=True, axis = 0)

    output.to_csv(path_or_buf = "Analysis.csv")
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)