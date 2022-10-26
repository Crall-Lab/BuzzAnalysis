# BuzzAnalysis
## Overview
Buzz analysis contains code for analysing data from BombusBox rigs.

<br><br>

## Requirements
- python3 (tested in Python 3.8.5)
- numpy
- pandas
- scipy

python3 should come preinstalled in unix systems but if not, they can be installed with
```
sudo apt-get install python3
```

- pip (or pip3, if native python is not python 3)
- python modules: numpy, pandas, scipy

To install numpy, pandas, and scipy, you will need pip or pip3. To install the requirements:
```
sudo apt install pip3
pip3 install numpy pandas scipy
```

<br><br>

## Running BuzzAnalysis
To run BuzzAnalysis, open up terminal and clone this repository. Then run:

```
cd BuzzAnalysis
python3 ./baseFunctions.py [path to folder holding tracking results] [tests] [whether to analyse data on the left and right separately]
```

As an example, to run the analysis (distance to social center) on the sample data provided:
```
python3 ./baseFunctions.py . distSC meanAct meanSpeed True
```

The results of the analysis can then be found in *Analysis.csv* within the BuzzAnalysis folder. The results from each test called will occupy one cell in *Analysis.csv*.

<br><br>

## Avaliable tests
- trackedFrames: Gives number of frames a bee is found in
- distSC: Gives mean distance to social center of a hive
- meanAct: Gives mean ratio of time spent moving
- meanSpeed: Gives mean moving speed
- meanIBD: Gives mean distance to other bees in cm
- totalInt: Gives total number of interactions between bees in a video
- totalIntFrames: Gives number of frames in a video where at least one interaction is detected.

<br><br>

## Parameters
All avaliable tests except trackedFrames and distSC require user-provided parameters. These are stored in params.py, and should be chosen by the user for each experiment.

## Adding your own tests
To add your own custom analysis, simply add the relevant code into *baseFunctions.py*. You will then be able to call it from command line. For example:
```
python3 ./baseFunctions.py . customTest False
```

Any parameters custom tests call can be added into *params.py*. They can then be called as params.*parameter*.

<br><br>

## Maintainers
Acacia Tang --  [ttang53@wisc.edu](mailto:ttang53@wisc.edu)
