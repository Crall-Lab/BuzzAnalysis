#!/usr/bin/env python3 -u

"""Converts all old files into new format. All .csv files in folder are targets."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

import pandas as pd
import os
import sys
import numpy as np
import pandas as pd
import os
import params #THIS IS NOT A PACKAGE?
import baseFunctions

def nest_social_center(trackingResults):
    """Generic function to calculate the nest social center from standard pandas array with centroid coordinates."""
    mean_x = np.nanmean(trackingResults['centroidX'].to_numpy())
    mean_y = np.nanmean(trackingResults['centroidY'].to_numpy())
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
    dist = []
    for id in trackingResults.ID.unique():
        oneID = trackingResults[trackingResults.ID == id]
        dist.append(distSC_id(oneID))
    return dist

def main(argv):
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    vids=[]
    for dir, subdir, files in os.walk(argv[1]):
        for f in files:
            if "mjpeg" in f:
                vids.append(os.path.join(dir,f))

    output = pd.DataFrame(columns = ['worker', 'Date', 'Time', 'ID'].append(argv[2]))
    for v in vids:
        workerID = v.split("worker")[1].split("-")[0]
        Date, Time = v.split("_")
        Date = Date.split("worker" + workerID + "-")[1]
        Time = Time.replace(".mjpeg", "").replace("-", ":")
        try:
            trackingResults = pd.read_csv(v.replace(".mjpeg", ".csv"))
        
        except:
            existingData = [workerID, Date, Time]
            addOn = ["" for i in range(len(argv[2])+1)]
            row = existingData + addOn
            analysis = pd.DataFrame([row], columns = ['worker', 'Date', 'Time', 'ID']+argv[2])
            analysis.worker = workerID
            analysis.Date = Date
            analysis.Time = Time
            output = pd.concat([output, analysis], ignore_index=True, axis = 0)
            continue

        analysis = pd.DataFrame(index=trackingResults.ID.unique().tolist(), columns=['worker', 'Date', 'Time', 'ID']+argv[2])
        analysis.worker = workerID
        analysis.Date = Date
        analysis.Time = Time
        analysis.ID = analysis.index
        for test in argv[2]:
            try:
                analysis[test] = getattr(baseFunctions, test)(trackingResults)
            except:
                print(test + " cannot be run")
                continue
    
        output = pd.concat([output, analysis], ignore_index=True, axis = 0)

    output.to_csv(path_or_buf = "Analysis.csv")
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)