# BuzzAnalysis
## Overview
Buzz analysis contains code for analysing data from Bumbleboxes.

<br><br>

## Requirements
- python3 (tested in Python 3.8.5)
- numpy
- pandas
- scipy
- shapely

python3 should come preinstalled in unix systems but if not, they can be installed with
```
sudo apt-get install python3
```

- pip (or pip3, if native python is not python 3)
- python modules: numpy, pandas, scipy

To install numpy, pandas, and scipy, you will need pip or pip3. To install the requirements:
```
sudo apt install pip3
pip3 install numpy pandas scipy shapely
```

<br><br>

## Running BuzzAnalysis
To run BuzzAnalysis, open up terminal and clone this repository. Then run:

```
cd BuzzAnalysis
python3 ./runMe.py
```

You can also run BuzzAnalysis with the following flags:
- --source, -s: Directory containing data. Defaults to current working directory.
- --extension, -e: String at end of all data files from tracking. Defaults to "_updated.csv".
- --brood, -b: If you want to run functions that work on brood data, provide path to brood data to run brood functions.
- --broodExtension, -x: String at end of all data files (must be CSVs) containing brood data. Defaults to "_nest_image.csv".
- --whole, -w: Do not split frame into two when analyzing.
- --bombus, -z: Data is from rig, run alternative search for data files.

<br>

As an example, to run the analysis on the sample data provided:
```
python3 ./runMe.py -s . -e '.csv' -z
```

The results of the analysis can then be found in *Analysis.csv* within the BuzzAnalysis folder (or current working directory). The results from each test called will occupy one column in *Analysis.csv*. Each row represents one bee in one video.

<br><br>

## Avaliable tests
### Base Functions
- trackedFrames: Gives number of frames a bee is found in
- distSC: Gives mean distance to social center of a hive (social center calculated as centroid of all tracked positions within the video)
- meanAct: Gives mean proportion of time spent moving of 
- meanSpeed: Gives mean moving speed
- meanIBD: Gives mean distance to other bees in cm
- totalInt: Gives total number of interactions based on proximity (between between focal bee and all other bees) within a video
- totalIntFrames: Gives total number of possible interactions between focal bee and all other bees in a video (i.e., total number of tracked frames for all other bees)
- meanX: Gives mean x-coordinate of bee.
- meanY: Gives mean y-coordiante of bee.
- varSpeed: Gives varience of speed of bee.
- medianMinDistToOthers: Gives median of minimum distance in each frame to other bees.

### Brood Functions
- meanEggDistM: Gives mean distance to eggs across frames (in cm).
- meanLarvaeDistM: Gives mean distance to larvae across frames.
- PropLarvaeTime: Gives proportion of time spent on larvae.
- meanPupaeDistM: Gives mean distance to pupae across frames.
- PropPupaeTime: Gives proportion of time spent on pupae.
- meanWaxPotDistM: Gives mean distance to wax pots across frames.
- medianClosesWaxPotDistM: Gives median of minimum distance in each frame to closest wax pot.
- PropWaxPotTime: Gives proportion of time spent on wax pots.
- meanNectarDistM: Gives mean distance to nectar source across frames.
- meanPollenDistM: Gives mean distance to pollen across frames.
- meanBroodDistM: Gives mean distance to brood across frames.
- medianClosestBroodDistM:Gives median of minimum distance in each frame to closest brood.
- PropBroodTime: Gives proportion of time spent on brood.
- PropInactiveTime: Gives proportion of time spent off nest and food, and not moving.

<br><br>

## Parameters
All avaliable tests except trackedFrames and distSC require user-provided parameters. These are stored in params.py, and should be chosen by the user for each experiment.

## Adding your own tests
To add your own custom analysis, simply add the relevant code into *baseFunctions.py* or *broodFunctions.py*.
Any parameters custom tests call can be added into *params.py*. They can then be called as params.*parameter*.

Each test must only return one column of data and each cell must correspond to one bee in one video.

<br><br>

## Maintainers
Acacia Tang --  [ttang53@wisc.edu](mailto:ttang53@wisc.edu)
