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
    print("PropBroodTime: broodLR columns:", broodLR.columns.tolist())
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    print("PropBroodTime: brood columns:", brood.columns.tolist(), "shape:", brood.shape)
    
    if brood.shape[1] > 0:   
        brood.columns = [('distM' + str(colname[1])) for colname in brood.columns]
        closest = brood.T.groupby(brood.T.index).min().T
        print("PropBroodTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropBroodTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        print("PropBroodTime: out sample (closest < onDist):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames from closest
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropBroodTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropBroodTime: result:\n", result)
        if result.isna().all():
            print("PropBroodTime: Warning: All results are NaN, possibly due to no valid frames or no frames where closest < onDist")
        return result
    else:
        print("PropBroodTime: No brood columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropBroodTime: returning empty Series with index:", out.index.tolist())
        return out

def meanEggDistM(broodLR):
    """Mean distance to egg. Distance measured as distance to closest point in geometry."""
    egg = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0]]]
    if egg.shape[1] > 0:
        egg.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in egg.columns])
        out = egg.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanEggDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanEggDistM: No egg columns found, returning NaN Series")
        return out

def meanLarvaeDistM(broodLR):
    """Mean distance to larvae. Distance measured as distance to closest point in geometry."""
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]
    if larvae.shape[1] > 0:    
        larvae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in larvae.columns])
        out = larvae.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanLarvaeDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanLarvaeDistM: No larvae columns found, returning NaN Series")
        return out

def meanPupaeDistM(broodLR):
    """Mean distance to pupae. Distance measured as distance to closest point in geometry."""
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]
    if pupae.shape[1] > 0:
        pupae.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pupae.columns])
        out = pupae.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanPupaeDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanPupaeDistM: No pupae columns found, returning NaN Series")
        return out

def meanWaxPotDistM(broodLR):
    """Mean distance to wax pots. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:
        wax.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in wax.columns])
        out = wax.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanWaxPotDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanWaxPotDistM: No wax columns found, returning NaN Series")
        return out

def meanNectarDistM(broodLR):
    """Mean distance to nectar source. Distance measured as distance to closest point in geometry."""
    nectar = broodLR[[col for col in broodLR.columns if 'distM_nectar' in col[0]]]
    if nectar.shape[1] > 0:    
        nectar.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in nectar.columns])
        out = nectar.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanNectarDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanNectarDistM: No nectar columns found, returning NaN Series")
        return out

def meanPollenDistM(broodLR):
    """Mean distance to pollen. Distance measured as distance to closest point in geometry."""
    pollen = broodLR[[col for col in broodLR.columns if 'distM_pollen' in col[0]]]
    if pollen.shape[1] > 0:   
        pollen.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in pollen.columns])
        out = pollen.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanPollenDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanPollenDistM: No pollen columns found, returning NaN Series")
        return out

def meanBroodDistM(broodLR):
    """Mean distance to brood. Distance measured as distance to closest point in geometry."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = pd.MultiIndex.from_tuples([('distM', colname[1]) for colname in brood.columns])
        out = brood.distM.mean()
        result = out.groupby(out.index).mean()
        print("meanBroodDistM: result:\n", result)
        return result
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("meanBroodDistM: No brood columns found, returning NaN Series")
        return out

def medianClosestBroodDistM(broodLR):
    """Median distance to closest brood. Distance measured as distance to closest point in geometry."""
    brood = broodLR[[col for col in broodLR.columns if 'distM_Egg' in col[0] or 'distM_Larvae' in col[0] or 'distM_Pupae' in col[0]]]
    if brood.shape[1] > 0:   
        brood.columns = [('distM' + str(colname[1])) for colname in brood.columns]
        closest = brood.T.groupby(brood.T.index).min().T
        out = closest.median()
        out.index = [int(i.split('M')[1]) for i in out.index]
        print("medianClosestBroodDistM: result:\n", out)
        return out
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("medianClosestBroodDistM: No brood columns found, returning NaN Series")
        return out

def medianClosesWaxPotDistM(broodLR):
    """Median distance to closest wax pot. Distance measured as distance to closest point in geometry."""
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        out = closest.median()
        out.index = [int(i.split('M')[1]) for i in out.index]
        print("medianClosesWaxPotDistM: result:\n", out)
        return out
    else:
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("medianClosesWaxPotDistM: No wax columns found, returning NaN Series")
        return out

def PropPupaeTime(broodLR):
    """Proportion of time spent on pupae, 'on' as defined by user, excluding non-detected frames."""
    print(f"PropPupaeTime: Using onDist = {onDist}")
    print("PropPupaeTime: broodLR columns:", broodLR.columns.tolist())
    pupae = broodLR[[col for col in broodLR.columns if 'distM_Pupae' in col[0]]]
    print("PropPupaeTime: pupae columns:", pupae.columns.tolist(), "shape:", pupae.shape)
    
    if pupae.shape[1] > 0:   
        pupae.columns = [('distM' + str(colname[1])) for colname in pupae.columns]
        closest = pupae.T.groupby(pupae.T.index).min().T
        print("PropPupaeTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropPupaeTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        print("PropPupaeTime: out sample (closest < onDist):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames from closest
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropPupaeTime: Bee {bee} valid distances:\n", closest[f'distM{bee}'][valid_frames])
            if bee == 1:
                true_frames = out[bee] & valid_frames
                print(f"PropPupaeTime: Bee 1 sample True distances (first 5):\n", closest[f'distM{bee}'][true_frames].head())
            print(f"PropPupaeTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropPupaeTime: result:\n", result)
        if result.isna().all():
            print("PropPupaeTime: Warning: All results are NaN, possibly due to no valid frames or no frames where closest < onDist")
        return result
    else:
        print("PropPupaeTime: No pupae columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropPupaeTime: returning empty Series with index:", out.index.tolist())
        return out

def PropLarvaeTime(broodLR):
    """Proportion of time spent on larvae, 'on' as defined by user, excluding non-detected frames."""
    print(f"PropLarvaeTime: Using onDist = {onDist}")
    print("PropLarvaeTime: broodLR columns:", broodLR.columns.tolist())
    larvae = broodLR[[col for col in broodLR.columns if 'distM_Larvae' in col[0]]]
    print("PropLarvaeTime: larvae columns:", larvae.columns.tolist(), "shape:", larvae.shape)
    
    if larvae.shape[1] > 0:   
        larvae.columns = [('distM' + str(colname[1])) for colname in larvae.columns]
        closest = larvae.T.groupby(larvae.T.index).min().T
        print("PropLarvaeTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropLarvaeTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        print("PropLarvaeTime: out sample (closest < onDist):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames from closest
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropLarvaeTime: Bee {bee} valid distances:\n", closest[f'distM{bee}'][valid_frames])
            print(f"PropLarvaeTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropLarvaeTime: result:\n", result)
        if result.isna().all():
            print("PropLarvaeTime: Warning: All results are NaN, possibly due to no valid frames or no frames where closest < onDist")
        return result
    else:
        print("PropLarvaeTime: No larvae columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropLarvaeTime: returning empty Series with index:", out.index.tolist())
        return out

def PropWaxPotTime(broodLR):
    """Proportion of time spent on wax pots, 'on' as defined by user, excluding non-detected frames."""
    print(f"PropWaxPotTime: Using onDist = {onDist}")
    print("PropWaxPotTime: broodLR columns:", broodLR.columns.tolist())
    wax = broodLR[[col for col in broodLR.columns if 'distM_Wax' in col[0]]]
    print("PropWaxPotTime: wax columns:", wax.columns.tolist(), "shape:", wax.shape)
    
    if wax.shape[1] > 0:   
        wax.columns = [('distM' + str(colname[1])) for colname in wax.columns]
        closest = wax.T.groupby(wax.T.index).min().T
        print("PropWaxPotTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropWaxPotTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        print("PropWaxPotTime: out sample (closest < onDist):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames from closest
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropWaxPotTime: Bee {bee} valid distances:\n", closest[f'distM{bee}'][valid_frames])
            print(f"PropWaxPotTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropWaxPotTime: result:\n", result)
        if result.isna().all():
            print("PropWaxPotTime: Warning: All results are NaN, possibly due to no valid frames or no frames where closest < onDist")
        return result
    else:
        print("PropWaxPotTime: No wax columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropWaxPotTime: returning empty Series with index:", out.index.tolist())
        return out
    
def PropNectarTime(broodLR):
    """Proportion of time spent on nectar, 'on' as defined by user, excluding non-detected frames."""
    print(f"PropNectarTime: Using onDist = {onDist}")
    print("PropNectarTime: broodLR columns:", broodLR.columns.tolist())
    nectar = broodLR[[col for col in broodLR.columns if 'distM_nectar' in col[0]]]
    print("PropNectarTime: nectar columns:", nectar.columns.tolist(), "shape:", nectar.shape)
    
    if nectar.shape[1] > 0:   
        nectar.columns = [('distM' + str(colname[1])) for colname in nectar.columns]
        closest = nectar.T.groupby(nectar.T.index).min().T
        print("PropNectarTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropNectarTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        out = closest < onDist
        out.columns = [int(i.split('M')[1]) for i in out.columns]
        print("PropNectarTime: out sample (closest < onDist):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames from closest
        result = pd.Series(index=out.columns, dtype=float)
        for bee in out.columns:
            valid_frames = closest[f'distM{bee}'].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropNectarTime: Bee {bee} valid distances:\n", closest[f'distM{bee}'][valid_frames])
            print(f"PropNectarTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropNectarTime: result:\n", result)
        if result.isna().all():
            print("PropNectarTime: Warning: All results are NaN, possibly due to no valid frames or no frames where closest < onDist")
        return result
    else:
        print("PropNectarTime: No nectar columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropNectarTime: returning empty Series with index:", out.index.tolist())
        return out    

def PropInactiveTime(broodLR):
    """Proportion of time spent away from nest and food and not moving, 'on' and 'moving' as defined by user, excluding non-detected frames."""
    print(f"PropInactiveTime: Using onDist = {onDist}, digital_noise_speed_cutoff = {digital_noise_speed_cutoff}")
    print("PropInactiveTime: broodLR columns:", broodLR.columns.tolist())
    work = broodLR[[col for col in broodLR.columns if 'centroid' not in col[0]]]
    print("PropInactiveTime: work columns:", work.columns.tolist(), "shape:", work.shape)
    
    if work.shape[1] > 0:   
        work.columns = [('distM' + str(colname[1])) for colname in work.columns]
        closest = work.T.groupby(work.T.index).min().T
        print("PropInactiveTime: closest shape:", closest.shape, "NaN counts:", closest.isna().sum().to_dict())
        print("PropInactiveTime: closest sample:\n", closest.head())
        
        # Compute boolean mask for frames where closest < onDist
        working = closest < onDist
        working.columns = [int(i.split('M')[1]) for i in working.columns]
        print("PropInactiveTime: working sample (closest < onDist):\n", working.head())
        
        # Compute activity based on speed
        speed = np.sqrt(broodLR['centroidX'].diff(axis=0)**2 + broodLR['centroidY'].diff(axis=0)**2)
        act = speed > digital_noise_speed_cutoff
        act = 1 * act
        act[np.isnan(speed)] = np.nan
        print("PropInactiveTime: act sample:\n", act.head())
        
        # Compute inactive frames: not working and not active
        out = ~(working | act)
        print("PropInactiveTime: out sample (inactive):\n", out.head())
        
        # Compute proportion for each bee using only their non-NaN frames
        result = pd.Series(index=working.columns, dtype=float)
        for bee in working.columns:
            # Valid frames where both working and act are defined
            valid_frames = closest[f'distM{bee}'].notna() & act[bee].notna()
            valid_count = valid_frames.sum()
            true_count = out[bee][valid_frames].sum() if valid_count > 0 else 0
            proportion = true_count / valid_count if valid_count > 0 else np.nan
            print(f"PropInactiveTime: Bee {bee} valid frames count: {valid_count}")
            print(f"PropInactiveTime: Bee {bee}: {true_count} True frames / {valid_count} valid frames = {proportion}")
            result[bee] = proportion
        
        print("PropInactiveTime: result:\n", result)
        if result.isna().all():
            print("PropInactiveTime: Warning: All results are NaN, possibly due to no valid frames")
        return result
    else:
        print("PropInactiveTime: No work columns found")
        row = broodLR.centroidX.iloc[0] if 'centroidX' in broodLR else pd.Series(index=[])
        out = pd.Series(index=row.index, dtype=float)
        out[:] = np.nan
        out.index.name = None
        print("PropInactiveTime: returning empty Series with index:", out.index.tolist())
        return out