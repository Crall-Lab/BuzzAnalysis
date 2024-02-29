#!/usr/bin/env python3

"""Runs functions in baseFunctions.py on dataset. If the brood flag is used, will also run functions in broodFunctions.py"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

#imports
import pandas as pd
import numpy as np
import os
import sys
import baseFunctions
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(argv):
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    output = pd.DataFrame()
    for dir, subdir, files in os.walk(argv[1]):
        for f in files:
            if "mjpeg" in f and os.path.exists(os.path.join(dir,f).replace(".mjpeg", ".csv")):
                v = os.path.join(dir,f)
                base = os.path.basename(v)
                print('Analyzing: ' + base)
                workerID, Date, Time = base.split("_")
                Time = Time.replace(".mjpeg", "").replace("-", ":")
                try:
                    trackingResults = pd.read_csv(v.replace(".mjpeg", ".csv"))
                
                except Exception as e:
                    print(e)
                    print("Error: cannot read " + v.replace(".mjpeg", ".csv"))
                    existingData = [workerID, Date, Time]
                    addOn = ["" for i in range(len(argv[2:-1])+2)]
                    row = existingData + addOn
                    analysis = pd.DataFrame([row], columns = ['pi_ID', 'Date', 'Time', 'LR', 'ID']+argv[2:-1])
                    analysis.worker = workerID
                    analysis.Date = Date
                    analysis.Time = Time
                    output = pd.concat([output, analysis], ignore_index=True, axis = 0)
                    continue
                
                if argv[-1] == "True":
                    trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
                    trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
                    trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"
                else:
                    trackingResults['LR'] = "Whole"
                fullAnalysis = pd.DataFrame()
                datasets = trackingResults.groupby('LR')
                for name, rawOneLR in datasets:
                    analysis = pd.DataFrame(index = rawOneLR.ID.unique())
                    analysis['LR'] = name
                    analysis['ID'] = analysis.index
                    oneLR = restructure_tracking_data(rawOneLR)
                    for test in argv[2:-1]:
                        try:
                            analysis[test] = None
                            analysis[test] = getattr(baseFunctions, test)(oneLR)
                        except Exception as e:
                            print(test + " cannot be run on " + v.replace(".mjpeg", ".csv"))
                            analysis[test] = None
                            print(e)
                            break
                    fullAnalysis = pd.concat([fullAnalysis, analysis], axis = 0)
                
                oneVid =  pd.DataFrame(index = fullAnalysis.index)    
                oneVid['pi_ID'] = workerID
                oneVid['bee_ID'] = oneVid.index
                oneVid['Date'] = Date
                oneVid['Time'] = Time
                oneVid = pd.concat([oneVid, fullAnalysis], axis = 1)

                output = pd.concat([output, oneVid], ignore_index=True, axis = 0)
                print('Done!')

    output.to_csv(path_or_buf = "Analysis.csv")
    print("All done!")
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)