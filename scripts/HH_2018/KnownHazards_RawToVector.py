# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS features that occur within CATZOC B, C, D, or U coverage and do not include BOOM or FISH HAVEN features.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm

# This script formats the RAW data. This script extracts raw feature data from the ENC viewer database, merges like-geometries and clips data to best scale ENC usage band. Data are then projected to North America
# Albers Equal Area Conic projection and exported as RAW data to a user-specified database.
# The user has an sde connection file to the ENC Direct Viewer Database
# The user has an sde connection file to the database where best usage band ENCs are stored

# Notes:
# The available features as of January 24, 2018 were used to generate this script, as described in the table below. Based on the data available at this time.

import os
import time

import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.known_hazards import *

# Define Variables

# Input Variables
haz = globals.Parameters("KnownHazards")
query = "(quapos IS NULL OR quapos = '10 ' OR quapos = '1  ') AND (catobs IS NULL OR catobs = '                         ' OR (catobs <> '10                       ' AND catobs <> '5                        ' AND catobs <> '8                        ')) AND (status IS NULL OR status <> '18   ')"

# RAW DATA
# @todo - output to Raw geodatabase
fns = extract_raw(haz)

# VECTOR DATA
# @todo - output to Vector geodatabase
# Remove boom and fish haven catobs features and only include features with unknown position*, surveyed, or precisely known
for geom in geoms_1:
    t = time.time()
    try:
        arcpy.Select_analysis(haz.raw_filename(geom), haz.vector_filename(geom), query)
        print(("Finished Select at %.1f secs" % (time.time() - t)))
    except arcgisscripting.ExecuteError:
        print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
    print(("Finished Copy at %.1f secs" % (time.time() - t)))
