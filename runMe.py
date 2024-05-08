#!/usr/bin/env python3

"""
Runs all functions in baseFunctions.py on dataset.
If the brood flag is used, will also run functions in broodFunctions.py
"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

# imports
import pandas as pd
import numpy as np
import os
import sys
import argparse
from inspect import getmembers, isfunction
import baseFunctions
import broodFunctions
import warnings
import shapely
import copy

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='testCSV', help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='.csv', help='String at end of all data files from tracking. Defaults to "_updated.csv"')
    parser.add_argument('--brood', '-b', type=str, default=None, help='Provide path to brood data to run brood functions.')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv', help='String at end of all data files (must be CSVs) containing brood data. Defaults to "_nest_image.csv"')
    parser.add_argument('--whole', '-w', action='store_true', help='Do not split frame into two when analyzing.')
    parser.add_argument('--bombus', '-z', action='store_true', help='Data is from rig, run alternative search for data files.')
    return parser.parse_args()

def restructure_tracking_data(rawOneLR):
    """Take centroid data from aruco-tracking structured output, rearrange and interpolate missing data"""
    # Drop any duplicate rows
    rawOneLR = rawOneLR.drop_duplicates(subset=['ID', 'frame'])
    xs = rawOneLR.pivot(index="frame", columns='ID', values=['centroidX', 'centroidY'])
    return xs.interpolate(method='linear', limit=2, axis='index', limit_direction='both')

def minDistance(A, B, P) : 
    # vector AB 
    AB = [B[0] - A[0], B[1] - A[1]]
 
    # vector BP
    BP = [P[0] - B[0], P[1] - B[1]]
 
    # vector AP 
    AP = [P[0] - A[0], P[1] - A[1]]
 
    # Calculating the dot product 
    AB_BP = AB[0] * BP[0] + AB[1] * BP[1] 
    AB_AP = AB[0] * AP[0] + AB[1] * AP[1]
 
    # Minimum distance from 
    # point E to the line segment 
    # Case 1 
    if (AB_BP > 0) :
 
        # Finding the magnitude 
        y = P[1] - B[1]; 
        x = P[0] - B[0]; 
        return (x * x + y * y)**0.5
 
    # Case 2 
    elif (AB_AP < 0):
        y = P[1] - A[1]
        x = P[0] - A[0] 
        return (x * x + y * y)*0.5
 
    # Case 3 
    else:
        # Finding the perpendicular distance 
        x1 = AB[0]; 
        y1 = AB[1]; 
        x2 = AP[0]; 
        y2 = AP[1]; 
        mod = (x1 * x1 + y1 * y1)**0.5
        return abs(x1 * y2 - y1 * x2) / mod


def distanceFromCentroid(oneLR, allbrood):
    if len(oneLR.columns) == 0 or len(allbrood.index) == 0:
        return pd.DataFrame()
    
    oneM = np.moveaxis(oneLR.values.reshape(oneLR.shape[0], 2, int(oneLR.shape[1]/2)), [0, 1], [1, 0])
    oneM = np.expand_dims(oneM, axis=3)
    oneMx = oneM[0,:,:,:]
    oneMy = oneM[1,:,:,:]
    allbroodx = np.array(allbrood[['x']])
    allbroodx =np.reshape(allbroodx, allbroodx.shape + (1,1))
    allbroodx = np.moveaxis(allbroodx, [0,1], [3, 0])
    allbroody = np.array(allbrood[['y']])
    allbroody =np.reshape(allbroody, allbroody.shape + (1,1))
    allbroody = np.moveaxis(allbroody, [0,1], [3, 0])

    distances = ((oneMx-allbroodx)**2 + (oneMy-allbroody)**2)**0.5
    distances = np.squeeze(np.moveaxis(distances, [0,1,2,3], [3,0,1,2]))

    
    if len(oneLR.columns) == 2:
        distances = np.expand_dims(distances, 1)
    if len(allbrood.index) == 1:
        distances = np.expand_dims(distances, 2)

    distDF = pd.DataFrame()
    for id in range(distances.shape[1]):
        newdist = pd.DataFrame(distances[:,id,:])
        newdist.index = oneLR.index
        newdist.columns = pd.MultiIndex.from_tuples([('distC_'+allbrood['label'][j]+'_'+str(allbrood['object index'][j]), oneLR.columns[id][1]) for j in range(len(allbrood['label']))], names = [None, 'ID'])
        distDF = pd.concat([distDF, newdist], axis = 1)
    return distDF

def minimumDistanceCircle(brood, oneLR):
    #distance to closet point: circle
    circle = brood.dropna(axis =0)
    circle = circle.reset_index()
    labels = ['distM_'+circle['label'][j]+'_'+str(circle['object index'][j]) for j in range(len(circle['label']))]
       
    if len(oneLR.columns) == 0 or len(labels) == 0:
        return pd.DataFrame()
    
    oneM = np.moveaxis(oneLR.values.reshape(oneLR.shape[0], 2, int(oneLR.shape[1]/2)), [0, 1], [1, 0])
    oneM = np.expand_dims(oneM, axis=3)
    oneMx = oneM[0,:,:,:]
    oneMy = oneM[1,:,:,:]
    circleX = np.array(circle[['x']])
    circleX =np.reshape(circleX, circleX.shape + (1,1))
    circleX = np.moveaxis(circleX, [0,1], [3, 0])
    circleY = np.array(circle[['y']])
    circleY =np.reshape(circleY, circleY.shape + (1,1))
    circleY = np.moveaxis(circleY, [0,1], [3, 0])
    circleR = np.array(circle[['radius']])
    circleR =np.reshape(circleR, circleR.shape + (1,1))
    circleR = np.moveaxis(circleR, [0,1], [3, 0])

    distances2 = ((oneMx-circleX)**2 + (oneMy-circleY)**2)**0.5 - circleR
    distances2 = np.squeeze(np.moveaxis(distances2, [0,1,2,3], [3,0,1,2]))
    
    if len(oneLR.columns) == 2:
        distances2 = np.expand_dims(distances2, 1)
    if len(labels) == 1:
        distances2 = np.expand_dims(distances2, 2)

    distDF2 = pd.DataFrame()
    for id in range(distances2.shape[1]):
        newdist = pd.DataFrame(distances2[:,id,:])
        newdist.index = oneLR.index
        identity = oneLR.columns[id][1]
        newdist.columns = pd.MultiIndex.from_tuples([(l, identity) for l in labels], names = [None, 'ID'])
        distDF2 = pd.concat([distDF2, newdist], axis = 1)
    return distDF2

def minimumDistancePolygon(oneLR, eggs):
    #distance to closet point: polygon
    eggs = eggs.reset_index()
    columns = list()
    for i in range(len(eggs.drop_duplicates('object index'))):
        for j in range(int(len(oneLR.columns)/2)):
            columns.append(('distM_'+eggs['label'][i]+'_'+str(eggs['object index'][i]), oneLR.columns[j][1]))
    if len(columns) == 0:
        return pd.DataFrame()
    
    distDF3 = pd.DataFrame(index = oneLR.index, columns =  pd.MultiIndex.from_tuples(columns))
    
    for i in range(distDF3.shape[0]):
        for j in range(distDF3.shape[1]):
            obj, bee = distDF3.columns[j]
            objID = obj.split('_')[-1]
            points = eggs[eggs['object index'] == int(objID)]
            
            blob = shapely.Polygon(np.array(points[['x','y']]))

            pt = shapely.Point((oneLR.loc[oneLR.index[i], ('centroidX', bee)], oneLR.loc[oneLR.index[i], ('centroidY', bee)]))

            if blob.contains(pt):
                distDF3.iloc[i,j] = 0
            else:
                ptDist = list()
                for p1 in range(len(points.index)):
                    for p2 in range(len(points.index)):
                        if p1 != p2:
                            A = [points.loc[p1, 'x'], points.loc[p1, 'y']]
                            B = [points.loc[p2, 'x'], points.loc[p2, 'y']]
                            P = [oneLR.loc[oneLR.index[i], ('centroidX', bee)], oneLR.loc[oneLR.index[i], ('centroidY', bee)]]
                            ptDist.append(minDistance(A, B, P))

                distDF3.iloc[i,j] = min(ptDist)
    return distDF3

def processBrood(base, oneLR, name, ext, broodSource):
    if os.path.exists(os.path.join(broodSource, '_'.join(base.split('_')[0:2]) + ext)):
        full = pd.read_csv(os.path.join(broodSource, '_'.join(base.split('_')[0:2]) + ext))
    else:
        print('Missing nest image data, did you mean to run brood functions?')
        return oneLR
    #split
    if name == 'Left':
        fullLeft = full[full['x'] < np.nanmean(oneLR['centroidX'].to_numpy())]
    elif name == "Right":
        fullRight = full[full['x'] > np.nanmean(oneLR['centroidY'].to_numpy())]

    #Don't care about areana
    brood = full[full['label'] != 'Arena perimeter (polygon)']

    #centroid: all
    eggs = brood[brood['radius'].isna()]
    allbrood = brood.dropna(axis =0)
    for i in set(eggs['object index']):
        try:
            egg = eggs[eggs['object index'] == i].reset_index()
            x = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.x
            y = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.y
            eggRow = pd.Series([i, egg.label[0],'polygon',x,y,np.nan])
            eggRow.index = allbrood.columns
            allbrood = pd.concat([allbrood.T,eggRow],axis=1).T
        except Exception as e:
            print(e)
            errorFile = open('Error.csv', 'a')
            try:
                errorFile.write(base + ', object ' + str(i) + ': ' + egg['label'][0] + '\n')
                errorFile.write(str(e) + '\n')
            except:
                errorFile.write('Cannot read label, printing entire list of objects' + '\n')
                errorFile.write(egg)
                errorFile.write(str(e) + '\n')
                                
                
            errorFile.close()
            if brood[brood['object index'] == i].shape[1] > 0:
                brood[brood['object index'] == i] = np.nan
                brood = brood.dropna(axis = 0)
            if eggs[eggs['object index'] == i].shape[1] > 0:
                eggs[eggs['object index'] == i] = np.nan
                eggs = eggs.dropna(axis = 0)
            if allbrood[allbrood['object index'] == i].shape[1] > 0:
                allbrood[allbrood['object index'] == i] = np.nan
                allbrood = allbrood.dropna(axis = 0)
            continue


    allbrood = allbrood.reset_index()

    distDF = distanceFromCentroid(oneLR, allbrood)
    distDF2 = minimumDistanceCircle(brood, oneLR)
    distDF3 = minimumDistancePolygon(oneLR, eggs)

    return pd.concat([oneLR, distDF, distDF2, distDF3], axis=1)

def main():
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    opt = vars(parse_opt())
    if os.path.exists('Analysis.csv'):
        print('I found a file named Analysis.csv in the current working directory and will be adding to the file.')
        output = pd.read_csv('Analysis.csv')
    else:
        output = pd.DataFrame()
    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if opt['brood']:
        funcs = funcs + [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    for dir, subdir, files in os.walk(opt['source']):
        for f in files:
            try:
                if opt['bombus']:
                    if 'mjpeg' in f and os.path.exists(os.path.join(dir, f).replace(".mjpeg", opt['extension'])):
                                v = os.path.join(dir, f)
                                print('Analyzing: ' + v)
                                workerID, Date, Time = f.split("_")
                                Time = Time.replace(".mjpeg", "").replace("-", ":")
                                trackingResults = pd.read_csv(v.replace(".mjpeg", opt['extension']))
                    else:
                        continue
                else:
                    if opt['extension'] in f:
                        v = os.path.join(dir, f)
                        print('Analyzing: ' + f)
                        workerID, Date, Time = f.split("_")[0:3]
                        Time = Time.replace(opt['extension'], "").replace("-", ":")
                        trackingResults = pd.read_csv(v)
                    else:
                        continue
            except Exception as e:
                print('Error reading file ' + f + ', skipping...')
                continue

            if opt['whole']:
                trackingResults['LR'] = "Whole"
            else:
                trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
                trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
                trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"

            fullAnalysis = pd.DataFrame()
            datasets = trackingResults.groupby('LR')
            for name, rawOneLR in datasets:
                analysis = pd.DataFrame(index=rawOneLR.ID.unique())
                analysis['LR'] = name
                analysis['ID'] = analysis.index
                oneLR = restructure_tracking_data(rawOneLR)  # one video of one colony
                if opt['brood']:
                    oneLR = processBrood(f, oneLR, name, opt['broodExtension'], opt['brood'])
                    oneLR.to_csv('oneLR.csv')
                for test in funcs:
                    #try:
                        analysis[test[0]] = None
                        analysis[test[0]] = test[1](oneLR)
                    #except Exception as e:
                    #    print(test[0] + " cannot be run on " + v.replace(".mjpeg", opt['extension']))
                    #    analysis[test] = None
                    #    print(e)
                    #    continue
                fullAnalysis = pd.concat([fullAnalysis, analysis], axis=0)

            oneVid = pd.DataFrame(index=fullAnalysis.index)
            oneVid['pi_ID'] = workerID
            oneVid['bee_ID'] = oneVid.index
            oneVid['Date'] = Date
            oneVid['Time'] = Time
            oneVid = pd.concat([oneVid, fullAnalysis], axis=1)

            output = pd.concat([output, oneVid], ignore_index=True, axis=0)
            output.to_csv(path_or_buf="Analysis.csv")
            print('Done!')
    
        output.to_csv(path_or_buf="Analysis.csv")
    print("All done!")
    return 0


if __name__ == "__main__":
    """Makes sure the "main" function is called from the command line"""
    status = main()
    sys.exit(status)
