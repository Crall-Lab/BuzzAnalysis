#!/usr/bin/env python3

"""Collection of functions that runMe.py will run on provided dataset if given brood data."""

__appname__ = 'broodFunctions.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

import pandas as pd

def meanEggDistC(broodLR):
    egg = broodLR[[col for col in broodLR.columns if 'distC_Egg' in col[0]]]
    egg.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in egg.columns])
    each = egg.distC.mean()
    return each.groupby(each.index).mean()

def meanLarvaeDistC(broodLR):
    larvae = broodLR[[col for col in broodLR.columns if 'distC_Larvae' in col[0]]]
    larvae.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in larvae.columns])
    each = larvae.distC.mean()
    return each.groupby(each.index).mean()

def meanPupaeDistC(broodLR):
    pupae = broodLR[[col for col in broodLR.columns if 'distC_Pupae' in col[0]]]
    pupae.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in pupae.columns])
    each = pupae.distC.mean()
    return each.groupby(each.index).mean()

def meanWaxPotDistC(broodLR):
    wax = broodLR[[col for col in broodLR.columns if 'distC_Wax' in col[0]]]
    wax.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in wax.columns])
    each = wax.distC.mean()
    return each.groupby(each.index).mean()

def meanNectarDistC(broodLR):
    nectar = broodLR[[col for col in broodLR.columns if 'distC_nectar' in col[0]]]
    nectar.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in nectar.columns])
    each = nectar.distC.mean()
    return each.groupby(each.index).mean()

def meanPollenDistC(broodLR):
    pollen = broodLR[[col for col in broodLR.columns if 'distC_pollen' in col[0]]]
    pollen.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in pollen.columns])
    each = pollen.distC.mean()
    return each.groupby(each.index).mean()

def meanBroodDistC(broodLR):
    brood = broodLR[[col for col in broodLR.columns if 'distC_Egg' in col[0] or 'distC_Larvae' in col[0] or 'distC_Pupae' in col[0]]]
    brood.columns = pd.MultiIndex.from_tuples([('distC', colname[1]) for colname in brood.columns])
    each = brood.distC.mean()
    return each.groupby(each.index).mean()