#!/usr/bin/env python3

"""Runs all functions in baseFunctions.py on dataset. If the brood flag is used, will also run functions in broodFunctions.py"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

#imports
import pandas as pd
import numpy as np
import os
import sys
import argparse
from inspect import getmembers, isfunction
import baseFunctions
import broodFunctions
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import shapely
import copy

def restructure_tracking_data(rawOneLR):
    """Take centroid data from aruco-tracking structured output, rearrange and interpolate missing data"""
    #Drop any duplicate rows
    rawOneLR = rawOneLR.drop_duplicates(subset=['ID', 'frame'])
    xs = rawOneLR.pivot(index="frame", columns='ID', values = ['centroidX', 'centroidY'])
    return xs.interpolate(method='linear', limit=2, axis='index', limit_direction='both')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='.', help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='_updated.csv', help='String at end of all data files from tracking. Defaults to "_updated.csv"')
    parser.add_argument('--brood', '-b', type=str, default=None, help='Provide path to brood data to run brood functions.')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv', help='String at end of all data files (must be CSVs) containing brood data. Defaults to "_nest_image.csv"')
    parser.add_argument('--whole', '-w', action='store_true', help='Do not split frame into two when analyzing.')
    parser.add_argument('--bombus', '-z', action='store_true', help='Data is from rig, run alternative search for data files.')
    return parser.parse_args()

def processBrood(base, oneLR, name, ext, broodSource):
    full = pd.read_csv(os.path.join(broodSource, '_'.join(base.split('_')[0:2]) + ext))
    brood = full[full['label'] != 'Arena perimeter (polygon)']
    
    #split
    if name == 'Left':
        brood = brood[brood['x'] < np.nanmean(brood['x'].to_numpy())]
    elif name == "Right":
        brood = brood[brood['x'] > np.nanmean(brood['x'].to_numpy())]

    #not eggs
    noteggs = brood[brood['radius'].isna()]
    circle = brood.dropna(axis =0)
    for i in set(noteggs['object index']):
        notegg = noteggs[noteggs['object index'] == i].reset_index()
        x = shapely.Polygon(notegg[['x', 'y']]).centroid.x
        y = shapely.Polygon(notegg[['x', 'y']]).centroid.y
        eggRow = pd.Series([i, notegg.label[0],'polygon',x,y,np.nan])
        eggRow.index = circle.columns
        circle = pd.concat([circle.T,eggRow],axis=1).T

    circle = circle.reset_index()
    oneM = np.moveaxis(oneLR.values.reshape(71, 2, 3), [0, 1], [1, 0])
    oneM = np.expand_dims(oneM, axis=3)
    oneMx = oneM[0,:,:,:]
    oneMy = oneM[1,:,:,:]
    circleMx = np.array(circle[['x']])
    circleMx =np.reshape(circleMx, circleMx.shape + (1,1))
    circleMx = np.moveaxis(circleMx, [0,1], [3, 0])
    circleMy = np.array(circle[['y']])
    circleMy =np.reshape(circleMy, circleMy.shape + (1,1))
    circleMy = np.moveaxis(circleMy, [0,1], [3, 0])

    distances = ((oneMx-circleMx)**2 + (oneMy-circleMy)**2)**0.5
    distances = np.squeeze(np.moveaxis(distances, [0,1,2,3], [3,0,1,2]))

    distDF = pd.DataFrame()
    for id in range(distances.shape[1]):
        newdist = pd.DataFrame(distances[:,id,:])
        newdist.index = oneLR.index
        newdist.columns = pd.MultiIndex.from_tuples([('distC_'+circle['label'][j], oneLR.columns[id][1]) for j in range(len(circle['label']))], names = ['metric', 'ID'])
        distDF = pd.concat([distDF, newdist], axis = 1)

    dist2DF = copy.deepcopy(distDF)
    

    #eggs
    

    #get distances and add to oneLR
    for i in set(brood['object index']):
        shape = brood[brood['object index'] == i].reset_index()
        if shape['shape'][0] == 'circle':
            broodLR = oneLR
        elif shape['shape'][0] == 'polygon':
            broodLR = oneLR
        else:
            continue

    return broodLR

def main():
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    opt = parse_opt()
    output = pd.DataFrame()
    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if vars(opt)['brood']:
        funcs = funcs + [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    for dir, subdir, files in os.walk(vars(opt)['source']):
        for f in files:
            if vars(opt)['bombus']:
                if "mjpeg" in f and os.path.exists(os.path.join(dir,f).replace(".mjpeg", vars(opt)['extension'])):
                    v = os.path.join(dir,f)
                    print('Analyzing: ' + f)
                    workerID, Date, Time = f.split("_")
                    Time = Time.replace(".mjpeg", "").replace("-", ":")
                    try:
                        trackingResults = pd.read_csv(v.replace(".mjpeg", vars(opt)['extension']))
                    
                    except Exception as e:
                        print(e)
                        print("Error: cannot read " + v.replace(".mjpeg", vars(opt)['extension']))
                        existingData = [workerID, Date, Time]
                        addOn = ["" for i in range(len(funcs)+2)]
                        row = existingData + addOn
                        analysis = pd.DataFrame([row], columns = ['pi_ID', 'Date', 'Time', 'LR', 'ID']+test)
                        analysis.worker = workerID
                        analysis.Date = Date
                        analysis.Time = Time
                        output = pd.concat([output, analysis], ignore_index=True, axis = 0)
                        continue
                else:
                    continue
            else:
                if vars(opt)['extension'] in f:
                    v = os.path.join(dir,f)
                    print('Analyzing: ' + f)
                    workerID, Date, Time = f.split("_")[0:3]
                    Time = Time.replace(vars(opt)['extension'], "").replace("-", ":")
                    try:
                        trackingResults = pd.read_csv(v)
                    
                    except Exception as e:
                        print(e)
                        print("Error: cannot read " + v)
                        existingData = [workerID, Date, Time]
                        addOn = ["" for i in range(len(funcs)+2)]
                        row = existingData + addOn
                        analysis = pd.DataFrame([row], columns = ['pi_ID', 'Date', 'Time', 'LR', 'ID']+test)
                        analysis.worker = workerID
                        analysis.Date = Date
                        analysis.Time = Time
                        output = pd.concat([output, analysis], ignore_index=True, axis = 0)
                        continue
                else:
                    continue
                
                if vars(opt)['whole']:
                    trackingResults['LR'] = "Whole"
                else:
                    trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
                    trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
                    trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"

                fullAnalysis = pd.DataFrame()
                datasets = trackingResults.groupby('LR')
                for name, rawOneLR in datasets:
                    analysis = pd.DataFrame(index = rawOneLR.ID.unique())
                    analysis['LR'] = name
                    analysis['ID'] = analysis.index
                    oneLR = restructure_tracking_data(rawOneLR) #one video of one colony
                    if vars(opt)['brood']:
                        oneLR = processBrood(f, oneLR)
                    for test in funcs:
                        try:
                            analysis[test[0]] = None
                            analysis[test[0]] = test[1](oneLR)
                        except Exception as e:
                            print(test + " cannot be run on " + v.replace(".mjpeg", vars(opt)['extension']))
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
    status = main()
    sys.exit(status)