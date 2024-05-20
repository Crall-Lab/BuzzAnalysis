#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset if given brood data."""

__appname__ = 'broodFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import pandas as pd
import numpy as np
from params import *

def meanEggDistM(broodLR):
    """Mean distance to egg. Distance measured as distance to closest point in geometry."""
    egg = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0]]]

    if egg.shape[1] > 0:
        egg.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in egg.columns])
        out = egg.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out


def meanLarvaeDistM(broodLR):
    """Mean distance to larvae. Distance measured as distance to closest point in geometry."""
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]

    if larvae.shape[1] > 0:    
        larvae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in larvae.columns])
        out = larvae.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def meanPupaeDistM(broodLR):
    """Mean distance to pupae. Distance measured as distance to closest point in geometry."""
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]

    if pupae.shape[1] > 0:
        pupae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pupae.columns])
        out = pupae.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def meanWaxPotDistM(broodLR):
    """Mean distance to wax pots. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:
        wax.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in wax.columns])
        out = wax.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def meanNectarDistM(broodLR):
    """Mean distance to nectar source. Distance measured as distance to closest point in geometry."""
    nectar = broodLR[[col for col in broodLR.columns if 'distM_nectar' in col[0]]]
    if nectar.shape[1] > 0:    
        nectar.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in nectar.columns])
        out = nectar.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def meanPollenDistM(broodLR):
    """Mean distance to pollen. Distance measured as distance to closest point in geometry."""
    pollen = broodLR[[col for col in broodLR.columns if 'distM_pollen' in col[0]]]
    if pollen.shape[1] > 0:   
        pollen.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pollen.columns])
        out = pollen.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def meanBroodDistM(broodLR):
    """Mean distance to brood. Distance measured as distance to closest point in geometry."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in brood.columns])
        out = brood.distM.mean()
        return out.groupby(out.index).mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def medianClosestBroodDistM(broodLR):
    """Median distance to closest brood. Distance measured as distance to closest point in geometry."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = [('distM' + str(colname[1])) for colname in brood.columns]
        closest = brood.T.groupby(brood.T.index).min().T
        out = closest.median()
        out.index = [int(i.split('M')[1]) for i in out.index]
        return out
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def medianClosesWaxPotDistM(broodLR):
    """Median distance to closest wax pot. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        out = closest.median()
        out.index = [int(i.split('M')[1]) for i in out.index]
        return out
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def PropBroodTime(broodLR):
    """Proportion of time spent on brood, 'on' as defined by user."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = [('distM' + str(colname[1])) for colname in brood.columns]
        closest = brood.T.groupby(brood.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def PropPupaeTime(broodLR):
    """Proportion of time spent on pupae, 'on' as defined by user."""
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]
    if pupae.shape[1] > 0:   
        pupae.columns = [('distM' + str(colname[1])) for colname in pupae.columns]
        closest = pupae.T.groupby(pupae.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def PropLarvaeTime(broodLR):
    """Proportion of time spent on larvae, 'on' as defined by user."""
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]
    if larvae.shape[1] > 0:   
        larvae.columns = [('distM' + str(colname[1])) for colname in larvae.columns]
        closest = larvae.T.groupby(larvae.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out

def PropWaxPotTime(broodLR):
    """Proportion of time spent on wax pots, 'on' as defined by user."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out
    
def PropNectarTime(broodLR):
    """Proportion of time spent on nectar, 'on' as defined by user."""
    nectar = broodLR[[col for col in broodLR.columns if 'distM_Nectar' in col[0]]]
    if nectar.shape[1] > 0:   
        nectar.columns = [('distM' + str(colname[1])) for colname in nectar.columns]
        closest = nectar.T.groupby(nectar.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out    

def PropInactiveTime(broodLR):
    """Proportion of time spent away from next and food and not moving, 'on' and 'moving' as defined by user."""
    work = broodLR[[col for col in broodLR.columns if 'centroid' not in col[0]]]
    if work.shape[1] > 0:   
        work.columns = [('distM' + str(colname[1])) for colname in work.columns]
        closest = work.T.groupby(work.T.index).min().T
        working = closest < onDist
        working.columns = [int(i.split('M')[1]) for i in working.columns]

        speed = np.sqrt(broodLR['centroidX'].diff(axis=0)**2 + broodLR['centroidY'].diff(axis=0)**2)
        act = speed > digital_noise_speed_cutoff
        act = 1*act
        act[np.isnan(speed)] = np.nan

        out = ~(working|act) # na is false
        return out.mean()
    else:
        row = broodLR.centroidX.iloc[0]
        out = pd.Series(index = row.index)
        out.index.name = None
        return out
    