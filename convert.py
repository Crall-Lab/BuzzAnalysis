#!/usr/bin/env python3 -u

"""Converts all old files into new format. All .csv files in folder are targets."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

import pandas as pd
import os
import sys

def main(argv):
    """ Main entry point of the program """
    for dir, subdir, files in os.walk("."):
        for f in files:
            if "csv" in f:
                df = pd.read_csv(os.path.join(dir,f))
                df.drop(columns=['video path'])
                df.rename(columns={"bee ID": "ID", "frame number": "frame", "x1":"centroidX", "y1":"centroidY", "x2":"frontX", "y2":"frontY"})
                df["1cm"] = ""
                df["check"] = ""
 
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)
