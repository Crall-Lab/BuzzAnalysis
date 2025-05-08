#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset to process brood data before performing the functions within broodFunctions. Made a separate file by August Easton-Calabria in order to better implement parallel processing."""

__appname__ = 'processBroodFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import numpy as np
import pandas as pd
import shapely

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
    distDF2[distDF2 < 0] = 0
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
