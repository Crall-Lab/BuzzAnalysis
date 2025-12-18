#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset to process brood data before performing the functions within broodFunctions. Made a separate file by August Easton-Calabria in order to better implement parallel processing."""

__appname__ = 'processBroodFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon


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
    

def distanceFromCentroid_new(oneLR, allbrood):

    if len(oneLR.columns) == 0 or len(allbrood.index) == 0:
        return pd.DataFrame()

    n_frames = oneLR.shape[0]
    n_bees = oneLR.shape[1] // 2 #divide by two here because for each bee, there is an x and y column
    # --- compute distances the same way you do now ---
    oneM = oneLR.values.reshape(n_frames, n_bees, 2)

    #print("after first transformation, head and shape:")
    #print(oneM)
    #print(oneM.shape)

    #oneM = np.expand_dims(oneM, axis=3)

    oneMx = oneM[:,:,0]
    #oneMx = oneM[0,:,:,:]

    #print("oneMx")
    #print(oneMx)
    #print(oneMx.shape)

    oneMy = oneM[:,:,1]
    #oneMy = oneM[1,:,:,:]

    #print("oneMy")
    #print(oneMy)
    #print(oneMy.shape)

    allbroodx = allbrood['x'].to_numpy()
    allbroody = allbrood['y'].to_numpy()
    distances = np.sqrt((oneMx[:, :, None] - allbroodx[None, None, :])**2 + (oneMy[:, :, None] - allbroody[None, None, :])**2)

    # --- build columns once ---
    bee_ids = [bee for (feat, bee) in oneLR.columns if feat == "centroidX"]

    # Use vertex ID to make point names unique
    point_names = [
        f"distC_{allbrood.at[j,'label']}_{int(allbrood.at[j,'object index'])}_v{int(allbrood.at[j,'vertex ID'])}"
        for j in range(len(allbrood))
    ]

    cols = [(pname, bee) for pname in point_names for bee in bee_ids]
    out = pd.DataFrame(
        index=oneLR.index,
        columns=pd.MultiIndex.from_tuples(cols, names=[None, "ID"]),
        dtype="float32"
    )

    # distances is (frames, bees, points). We need columns ordered (points, bees).
    out.iloc[:, :] = np.moveaxis(distances, 2, 1).reshape(distances.shape[0], -1).astype("float32")

    return out


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
    #print(f"distances.shape: {distances.shape}")
    distances = np.moveaxis(distances, [0,1,2,3], [3,0,1,2])
    # now distances.shape = (1, 16, 55, 1)
    #force it to 3D (frames, bees, brood) without squeezing away size-1 dims
    distances = distances.reshape(distances.shape[0], distances.shape[1], -1)  # (frames=1, bees=16, brood=55)

    distDF = pd.DataFrame()
    for id in range(distances.shape[1]):

        newdist = pd.DataFrame(distances[:,id,:])
        newdist.index = oneLR.index
        #newdist.columns = pd.MultiIndex.from_tuples([('distC_'+allbrood['label'][j]+'_'+str(allbrood['object index'][j]), oneLR.columns[id][1]) for j in range(len(allbrood['label']))], names = [None, 'ID'])
        newdist.columns = pd.MultiIndex.from_tuples(
            [
                (
                    f"distC_{allbrood.at[j, 'label']}_{allbrood.at[j, 'object index']}_p{j:03d}",
                    oneLR.columns[id][1]
                )
                for j in range(len(allbrood))
            ],
            names=[None, 'ID']
        )
        
        distDF = pd.concat([distDF, newdist], axis = 1)
    return distDF


def minimumDistanceCircle_new(brood: pd.DataFrame, oneLR: pd.DataFrame) -> pd.DataFrame:
    """
    Minimum distance from each bee centroid to each circle boundary:
      max(0, distance_to_center - radius)

    Output columns: (f"distM_{label}_{object index}", bee_id)
    """
    if oneLR is None or len(oneLR.columns) == 0 or brood is None or len(brood.index) == 0:
        return pd.DataFrame()

    # Keep only circles with a radius (and assume 1 row per circle object)
    circle = brood.dropna(subset=["radius"]).reset_index(drop=True)
    if len(circle) == 0:
        return pd.DataFrame()

    # Bee IDs (robust): one per bee
    bee_ids = [bee for (feat, bee) in oneLR.columns if feat == "centroidX"]
    if len(bee_ids) == 0:
        return pd.DataFrame()

    # Extract centroid arrays once: (frames, bees)
    X = oneLR.loc[:, [("centroidX", b) for b in bee_ids]].to_numpy(dtype=float, copy=False)
    Y = oneLR.loc[:, [("centroidY", b) for b in bee_ids]].to_numpy(dtype=float, copy=False)

    # Circle params: (circles,)
    cx = circle["x"].to_numpy(dtype=float, copy=False)
    cy = circle["y"].to_numpy(dtype=float, copy=False)
    cr = circle["radius"].to_numpy(dtype=float, copy=False)

    # Distances: (frames, bees, circles)
    d_center = np.sqrt((X[:, :, None] - cx[None, None, :])**2 + (Y[:, :, None] - cy[None, None, :])**2)
    d = d_center - cr[None, None, :]
    d[d < 0] = 0
    d = d.astype("float32")

    # Column names: one per circle object, per bee
    obj_names = [
        f"distM_{circle.at[i, 'label']}_{int(circle.at[i, 'object index'])}"
        for i in range(len(circle))
    ]
    cols = [(name, bee) for name in obj_names for bee in bee_ids]

    out = pd.DataFrame(
        index=oneLR.index,
        columns=pd.MultiIndex.from_tuples(cols, names=[None, "ID"]),
        dtype="float32"
    )

    # Reorder (frames, bees, circles) -> (frames, circles, bees) to match obj-major cols
    out.iloc[:, :] = np.moveaxis(d, 2, 1).reshape(d.shape[0], -1)

    return out


def minimumDistanceCircle(brood, oneLR):
    #distance to closet point: circle
    circle = brood.dropna(subset=['radius'])
    circle = circle.reset_index(drop=True)
    #print(f"circle-shape: {circle.shape}")
    #labels = ['distM_'+circle['label'][j]+'_'+str(circle['object index'][j]) for j in range(len(circle['label']))]

    labels = [
        f"distM_{row.label}_{row['object index']}_c{i:03d}"
        for i, row in circle.iterrows()
    ]   
    if len(oneLR.columns) == 0 or len(labels) == 0:
        return pd.DataFrame()
    
    oneM = np.moveaxis(
        oneLR.values.reshape(oneLR.shape[0], 2, int(oneLR.shape[1]/2)), [0, 1], [1, 0])
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

    distances2 = np.moveaxis(distances2, [0,1,2,3], [3,0,1,2])
    distances2 = distances2.reshape(distances2.shape[0], distances2.shape[1], -1)
    #distances2 = np.squeeze(np.moveaxis(distances2, [0,1,2,3], [3,0,1,2]))
    
    #if len(oneLR.columns) == 2:
    #    distances2 = np.expand_dims(distances2, 1)
    #if len(labels) == 1:
    #    distances2 = np.expand_dims(distances2, 2)

    distDF2 = pd.DataFrame()
    
    for id in range(distances2.shape[1]):
        newdist = pd.DataFrame(distances2[:,id,:])
        newdist.index = oneLR.index
        identity = oneLR.columns[id][1]
        newdist.columns = pd.MultiIndex.from_tuples([(l, identity) for l in labels], names = [None, 'ID'])
        distDF2 = pd.concat([distDF2, newdist], axis = 1)
    distDF2[distDF2 < 0] = 0
    return distDF2


def _polygon_from_vertices(df_obj):
    # df_obj is eggs filtered to a single object index

    # 1) Prefer vertex-ID ordering if present
    if "vertex ID" in df_obj.columns and df_obj["vertex ID"].notna().all():
        pts = df_obj.sort_values("vertex ID")[["x", "y"]].to_numpy()
    else:
        pts = df_obj[["x", "y"]].to_numpy()

    poly = Polygon(pts)
    if poly.is_valid:
        return poly

    # 2) Fallback: angle-sort around centroid
    xy = df_obj[["x", "y"]].to_numpy()
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    ang = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    pts2 = xy[np.argsort(ang)]
    poly2 = Polygon(pts2)
    if poly2.is_valid:
        return poly2

    # 3) Last resort: buffer(0) repair
    return poly2.buffer(0)


def minimumDistancePolygon_new(oneLR: pd.DataFrame, eggs: pd.DataFrame) -> pd.DataFrame:
    if oneLR is None or len(oneLR.columns) == 0 or eggs is None or len(eggs) == 0:
        return pd.DataFrame()

    eggs = eggs.reset_index(drop=True)

    req = {"label", "object index", "x", "y"}
    missing = req - set(eggs.columns)
    if missing:
        raise ValueError(f"eggs missing required columns: {sorted(missing)}")

    bee_ids = [bee for (feat, bee) in oneLR.columns if feat == "centroidX"]
    if len(bee_ids) == 0:
        return pd.DataFrame()

    # Build polygons once per (label, object index)
    meta = eggs[["label", "object index"]].drop_duplicates()
    labels = meta["label"].astype(str).to_list()
    obj_indices = meta["object index"].astype(int).to_list()

    polys = []
    for lab, obj in zip(labels, obj_indices):
        df_obj = eggs[eggs["object index"] == obj]
        poly = _polygon_from_vertices(df_obj)
        polys.append(poly)

    polys = np.array(polys, dtype=object)  # (n_polys,)
    nP = len(polys)
    if nP == 0:
        return pd.DataFrame()

    frames = oneLR.index.to_numpy()
    nF = len(frames)

    # Output container
    cols = []
    for lab, obj in zip(labels, obj_indices):
        base = f"distM_{lab}_{obj}"
        for bee in bee_ids:
            cols.append((base, bee))
    out = pd.DataFrame(index=oneLR.index, columns=pd.MultiIndex.from_tuples(cols, names=[None, "ID"]), dtype="float32")

    # For each bee: compute all polygon distances to all frames in one broadcast
    blocks = []
    for bee in bee_ids:
        x = oneLR[("centroidX", bee)].to_numpy(dtype="float64", copy=False)
        y = oneLR[("centroidY", bee)].to_numpy(dtype="float64", copy=False)

        ok = np.isfinite(x) & np.isfinite(y)
        dist_full = np.full((nP, nF), np.nan, dtype="float64")

        if ok.any():
            pts_ok = shapely.points(x[ok], y[ok])  # (n_ok,)
            d_mat = shapely.distance(polys[:, None], pts_ok[None, :])  # (nP, n_ok)
            dist_full[:, ok] = d_mat

        # block columns for this bee, object-major to match `cols`
        blocks.append(dist_full.astype("float32"))

    # blocks is list of (nP, nF) per bee; we want (nF, nP*bees) object-major then bee
    # Currently each block is (nP, nF). Stack bees -> (bees, nP, nF) then reorder.
    stack = np.stack(blocks, axis=0)              # (nB, nP, nF)
    stack = np.moveaxis(stack, 2, 0)              # (nF, nB, nP)
    stack = np.moveaxis(stack, 2, 1)              # (nF, nP, nB)
    out.iloc[:, :] = stack.reshape(nF, -1)

    return out


def minimumDistancePolygon(oneLR, eggs):
    #distance to closet point: polygon
    eggs = eggs.reset_index()
    #print(eggs)
    #columns = list()
    #for i in range(len(eggs.drop_duplicates('object index'))):
    #    for j in range(int(len(oneLR.columns)/2)):
    #        columns.append(('distM_'+eggs['label'][i]+'_'+str(eggs['object index'][i]), oneLR.columns[j][1]))
    columns = []
    eggs = eggs.reset_index(drop=True)   # make row numbers stable

    for row_i, row in eggs.iterrows():
        obj_label = row['label']
        obj_index = row['object index']
        
        # UNIQUE name per vertex
        obj_name = f"distM_{obj_label}_{obj_index}_p{row_i:03d}"

        # Add one column for every bee
        for j in range(int(len(oneLR.columns)/2)):
            bee_id = oneLR.columns[j][1]
            #print(f"adding: {(obj_name, bee_id)}")
            columns.append((obj_name, bee_id))

    #print("Done adding columns")

    if len(columns) == 0:
        return pd.DataFrame()
    
    distDF3 = pd.DataFrame(index = oneLR.index, columns =  pd.MultiIndex.from_tuples(columns))
    #distDF3.to_csv('~/Desktop/2021 Data Analysis/distDF3_v1.csv')
    
    #print("DistDF3.shape:")
    #print(distDF3.shape)

    for i in range(distDF3.shape[0]):
        #print(i)
        for j in range(distDF3.shape[1]):
            #print(f"{i} out of {distDF3.shape[0]}")
            #print(f"{j} out of {distDF3.shape[1]}")
            #print(j)    
            obj, bee = distDF3.columns[j]
            #print(f"obj: {obj}")
            #print(f"bee: {bee}")

            # Remove the _p### suffix to get the core object name
            core, ext = obj.rsplit('_', 1)     # removes "_p000"
            objID = int(core.split('_')[-1]) # original object index

            #objID = obj.split('_')[-1]
            #print(f"objID: {objID}-{ext}  --- bee: {bee}")
            points = eggs[eggs['object index'] == int(objID)]
            #print(f"Number of points: {len(points)}")
            #print(points)
            
            blob = shapely.Polygon(np.array(points[['x','y']]))

            pt = shapely.Point((oneLR.loc[oneLR.index[i], ('centroidX', bee)], oneLR.loc[oneLR.index[i], ('centroidY', bee)]))

            if blob.contains(pt):
                distDF3.iloc[i,j] = 0
            else:
                ptDist = list()
                for p1 in points.index:
                    for p2 in points.index:
                        if p1 != p2:
                            A = [points.loc[p1, 'x'], points.loc[p1, 'y']]
                            B = [points.loc[p2, 'x'], points.loc[p2, 'y']]
                            P = [oneLR.loc[oneLR.index[i], ('centroidX', bee)], oneLR.loc[oneLR.index[i], ('centroidY', bee)]]
                            ptDist.append(minDistance(A, B, P))

                distDF3.iloc[i,j] = min(ptDist)
    return distDF3


def testing_new_functions():

    from io import StringIO

    brood_csv = """object index,label,label ID,vertex ID,shape,x,y,radius
    0,empty wax pots (circles),,1,circle,50.0,50.0,2.0
    1,Pupae (circles),,1,circle,100.0,150.0,10.0
    2,Larvae (circles),,1,circle,200.0,300.0,5.0
    3,Eggs perimeter (polygons),,1,polygon,20.0,10.0,nan
    3,Eggs perimeter (polygons),,2,polygon,30.0,10.0,nan
    3,Eggs perimeter (polygons),,3,polygon,30.0,20.0,nan
    3,Eggs perimeter (polygons),,4,polygon,20.0,20.0,nan
    4,Eggs (points),,1,point,25.0,15.0,nan
    5,Eggs (points),,1,point,20.0,15.0,nan
    """
    bee_csv = """filename,colony number,datetime,frame,ID,centroidX,centroidY,frontX,frontY,LR,in_frame_duplicate,og_duplicate,unresolvable_duplicate,flagged_as_jump,interpolated
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,0,1,50.0,51.0,635.5,2105.5,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,1,1,50.0,52.0,636.0,2105.5,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,2,1,52.0,52.0,635.5,2105.0,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,0,3,100.0,159.0,2054.5,2242.5,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,1,3,105.0,160.0,2055.0,2242.0,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,2,3,100.0,160.0,2055.5,2242.0,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,0,6,25.0,15.0,1376.0,2475.5,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,1,6,31.0,10.0,1422.0,2417.0,Whole,False,False,False,False,0
    col_46-2021-06-12_00-00-01,,10-2021-06-12_00-00-01,2,6,30.0,19.0,1463.5,2403.5,Whole,False,False,False,False,0
    """

    brood = pd.read_csv(StringIO(brood_csv))
    bee_long = pd.read_csv(StringIO(bee_csv))

    # Build oneLR: index=frame, columns MultiIndex (centroidX/centroidY, beeID)
    frames = sorted(bee_long["frame"].unique())
    bee_ids = sorted(bee_long["ID"].unique())

    pivotX = bee_long.pivot(index="frame", columns="ID", values="centroidX").reindex(frames)
    pivotY = bee_long.pivot(index="frame", columns="ID", values="centroidY").reindex(frames)

    cols = []
    arr = np.empty((len(frames), 2*len(bee_ids)), dtype=float)
    j = 0
    for b in bee_ids:
        cols.append(("centroidX", b)); arr[:, j] = pivotX[b].to_numpy(); j += 1
        cols.append(("centroidY", b)); arr[:, j] = pivotY[b].to_numpy(); j += 1

    oneLR = pd.DataFrame(arr, index=frames, columns=pd.MultiIndex.from_tuples(cols, names=[None, "ID"]))
    print(list(oneLR.columns))
    #Circles test
    circles = brood[brood["shape"].str.contains("circle", case=False) & brood["radius"].notna()].copy()
    d2 = minimumDistanceCircle_new(circles, oneLR)

    col = ("distM_empty wax pots (circles)_0", 1)  # (object name, bee ID)
    assert np.isclose(d2.loc[0, col], 0.0)
    assert np.isclose(d2.loc[2, col], np.sqrt(8) - 2.0, atol=1e-6)

    assert (d2.values >= 0).all(), "circle distances should never be negative"
    assert d2.index.equals(oneLR.index)
    print("✅ minimumDistanceCircle passes known-answer tests")

    #Distance from centroid test
    print(oneLR.head)
    print(oneLR.shape)
    d1 = distanceFromCentroid_new(oneLR, brood)

    col = ("distC_empty wax pots (circles)_0_v1", 1)
    assert np.isclose(d1.loc[0, col], 1.0)
    assert d1.index.equals(oneLR.index)
    print("✅ distanceFromCentroid_test passes basic checks")


    # polygons subset
    polygons = brood[brood["shape"].str.contains("polygon", case=False, na=False)].copy()

    d3 = minimumDistancePolygon_new(oneLR, polygons)

    # Column should exist: one polygon object (index 3) x each bee
    col6 = ("distM_Eggs perimeter (polygons)_3", 6)

    assert col6 in d3.columns, f"Missing expected column {col6}"
    assert np.isclose(d3.loc[0, col6], 0.0, atol=1e-6)  # inside
    assert np.isclose(d3.loc[1, col6], 1.0, atol=1e-6)  # (31,10) -> nearest is (30,10)
    assert np.isclose(d3.loc[2, col6], 0.0, atol=1e-6)  # on boundary

    # Optional: shape sanity (n_polys=1, n_bees should be 3 for IDs 1,3,6)
    expected_cols = 1 * len([bee for (feat, bee) in oneLR.columns if feat == "centroidX"])
    assert d3.shape == (len(oneLR.index), expected_cols)

    print("✅ minimumDistancePolygon_test passed known-answer tests")



if __name__ == "__main__":
    testing_new_functions()

