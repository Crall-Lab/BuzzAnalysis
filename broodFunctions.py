#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset if given brood data."""

__appname__ = 'broodFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import pandas as pd
import numpy as np
from params import *

def PropBroodTime(broodLR):
    """Proportion of time spent on brood, 'on' as defined by user, excluding non-detected frames."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    
    if brood.shape[1] > 0:   
        brood.columns = [('distM' + str(colname[1])) for colname in brood.columns]
        closest = brood.T.groupby(brood.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanEggDistM(broodLR):
    """Mean distance to egg. Distance measured as distance to closest point in geometry."""
    egg = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0]]]
    if egg.shape[1] > 0:
        egg.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in egg.columns])
        out = egg.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanLarvaeDistM(broodLR):
    """Mean distance to larvae. Distance measured as distance to closest point in geometry."""
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]
    if larvae.shape[1] > 0:    
        larvae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in larvae.columns])
        out = larvae.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanPupaeDistM(broodLR):
    """Mean distance to pupae. Distance measured as distance to closest point in geometry."""
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]
    if pupae.shape[1] > 0:
        pupae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pupae.columns])
        out = pupae.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanWaxPotDistM(broodLR):
    """Mean distance to wax pots. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:
        wax.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in wax.columns])
        out = wax.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        try:
            wax = broodLR[[col for col in broodLR.columns if 'distM_empty' in col[0] or 'distM_full' in col[0]]]
            if wax.shape[1] > 0:
                wax.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in wax.columns])
                out = wax.distM.mean()
                result = out.groupby(out.index).mean()
                return result
            else:
                raise Exception
        except:
            row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
            out = pd.Series(index=row.index, dtype=float)
            out[:] = np.nan
            out.index.name = None
            return out
    
def meanFullNectarPotDistM(broodLR):
    """Mean distance to full nectar pots. Distance measured as distance to closest point in geometry."""
    fullnp = broodLR[[col for col in broodLR.columns if 'distM_full' in col[0]]]
    if fullnp.shape[1] > 0:
        fullnp.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in fullnp.columns])
        out = fullnp.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out
    
def meanEmptyWaxPotDistM(broodLR):
    """Mean distance to full nectar pots. Distance measured as distance to closest point in geometry."""
    emptywp = broodLR[[col for col in broodLR.columns if 'distM_empty' in col[0]]]
    if emptywp.shape[1] > 0:
        emptywp.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in emptywp.columns])
        out = emptywp.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanNectarDistM(broodLR):
    """Mean distance to nectar source. Distance measured as distance to closest point in geometry."""
    nectar = broodLR[[col for col in broodLR.columns if 'distM_nectar' in col[0]]]
    if nectar.shape[1] > 0:    
        nectar.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in nectar.columns])
        out = nectar.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanPollenDistM(broodLR):
    """Mean distance to pollen. Distance measured as distance to closest point in geometry."""
    pollen = broodLR[[col for col in broodLR.columns if 'distM_pollen' in col[0]]]
    if pollen.shape[1] > 0:   
        pollen.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pollen.columns])
        out = pollen.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def meanBroodDistM(broodLR):
    """Mean distance to brood. Distance measured as distance to closest point in geometry."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in brood.columns])
        out = brood.distM.mean()
        result = out.groupby(out.index).mean()
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
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
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def medianClosestWaxPotDistM(broodLR):
    """Median distance to closest wax pot. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        out = closest.median()
        out.index = [int(i.split('M')[1]) for i in out.index]
        return out
    else:
        try:
            wax = broodLR[[col for col in broodLR.columns if 'distM_empty' in col[0] or 'distM_full' in col[0]]]
            if wax.shape[1] > 0:
                wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
                closest = wax.T.groupby(wax.T.index).min().T
                out = closest.median()
                out.index = [int(i.split('M')[1]) for i in out.index]
                return out
            else:
                raise Exception 
        except:
            row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
            out = pd.Series(index=row.index, dtype=float)
            out[:] = np.nan
            out.index.name = None
            return out

def PropPupaeTime(broodLR):
    """Proportion of time spent on pupae, 'on' as defined by user, excluding non-detected frames."""
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]
    
    if pupae.shape[1] > 0:   
        pupae.columns = [('distM' + str(colname[1])) for colname in pupae.columns]
        closest = pupae.T.groupby(pupae.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def PropLarvaeTime(broodLR):
    """Proportion of time spent on larvae, 'on' as defined by user, excluding non-detected frames."""
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]
    
    if larvae.shape[1] > 0:   
        larvae.columns = [('distM' + str(colname[1])) for colname in larvae.columns]
        closest = larvae.T.groupby(larvae.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out

def PropWaxPotTime(broodLR):
    """Proportion of time spent on wax pots, 'on' as defined by user, excluding non-detected frames."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0] or 'distM_empty' in col[0] or 'distM_full' in col[0]] ]
    
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out
    
def PropPollenTime(broodLR):
    """Proportion of time spent on pollen, 'on' as defined by user, excluding non-detected frames."""
    pollen = broodLR[[col for col in broodLR.columns if 'distM_pollen' in col[0]]]
    
    if pollen.shape[1] > 0:   
        pollen.columns = [('distM' + str(colname[1])) for colname in pollen.columns]
        closest = pollen.T.groupby(pollen.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out
    
def PropNectarTime(broodLR):
    """Proportion of time spent on nectar, 'on' as defined by user, excluding non-detected frames."""
    nectar = broodLR[[col for col in broodLR.columns if 'distM_nectar' in col[0]]]
    
    if nectar.shape[1] > 0:   
        nectar.columns = [('distM' + str(colname[1])) for colname in nectar.columns]
        closest = nectar.T.groupby(nectar.T.index).min().T
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out    

def PropInactiveTime(broodLR):
    """Proportion of time spent away from nest and food and not moving, 'on' and 'moving' as defined by user, excluding non-detected frames."""
    work = broodLR[[col for col in broodLR.columns if 'centroid' not in col[0]]]
    
    if work.shape[1] > 0:   
        work.columns = [('distM' + str(colname[1])) for colname in work.columns]
        closest = work.T.groupby(work.T.index).min().T
        working = closest < onDist
        working.columns = [int(i.split('M')[1]) for i in working.columns]
        
        speed = np.sqrt(broodLR['centroidX'].diff(axis=0)**2 + broodLR['centroidY'].diff(axis=0)**2)
        act = speed > digital_noise_speed_cutoff
        act = 1 * act
        act[np.isnan(speed)] = np.nan
        out = ~(working | act)
        
        result = pd.Series(index=working.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna() & act[bee].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            result[bee] = proportion
        
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        return out