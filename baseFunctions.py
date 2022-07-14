#!/usr/bin/env python3 -u

"""Runs analysis that is independent of nest structure data."""

__appname__ = 'baseFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import pandas as pd
import os
import sys
import numpy as np
import params #THIS IS NOT A PACKAGE?
import baseFunctions

def nest_social_center(oneID):
    """Generic function to calculate the nest social center from standard pandas array with centroid coordinates."""
    mean_x = np.nanmean(oneID['centroidX'].to_numpy())
    mean_y = np.nanmean(oneID['centroidY'].to_numpy())
    return (mean_x, mean_y)

def distSC_id(oneID):
    """Given dataframe, return 1-D array containing average distance to social center."""
    sc = nest_social_center(oneID)
    xd = oneID['centroidX'] - sc[0]
    yd = oneID['centroidY'] - sc[1]
    tot_d = np.sqrt(xd**2 + yd**2)
    mean_scd = np.nanmean(tot_d, axis=0)
    return mean_scd

def distSC(trackingResults):
    dist = pd.Series(index = trackingResults.LR.unique(), dtype='float64')
    for lr in trackingResults.LR.unique():
        oneID = trackingResults[trackingResults.LR == lr]
        dist[lr] = distSC_id(oneID)
    return dist



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
            trackingResults['LR'][trackingResults['LR']] = "Left"
            trackingResults['LR'][trackingResults['LR'] != "Left"] = "Right"
            LR = ["Left", "Right"]
        else:
            LR = ["Whole"]
        analysis = pd.DataFrame(index=LR, columns=['worker', 'Date', 'Time', 'ID']+argv[2:-1])
        analysis.worker = workerID
        analysis.Date = Date
        analysis.Time = Time
        analysis.ID = analysis.index
        for test in argv[2:-1]:
            try:
                analysis[test] = 0
                analysis[test] = getattr(baseFunctions, test)(trackingResults)
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