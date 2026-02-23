import os
import math

import arcpy
import arcgisscripting
import traceback

# Auxiliary Data Generation
# EEZ data


def create(params):
    # Output Variables
    eez = params.ini["AUX"]["eez_buffered_ras"]
    eez_0 = params.ini["AUX"]["eez_0_ras"]
    eez_1 = params.ini["AUX"]["eez_1_ras"]

    # Create EEZ rasters with values set as 0 and 1 - Must Reclassify Twice to get to work correctly (?)
    eez_min_t = arcpy.GetRasterProperties_management(eez, "MINIMUM")
    eez_min = math.floor(float(eez_min_t.getOutput(0)))
    eez_max_t = arcpy.GetRasterProperties_management(eez, "MAXIMUM")
    eez_max = math.ceil(float(eez_max_t.getOutput(0)))

    r0_t = arcpy.sa.Reclassify(eez, "VALUE", "%s %s 0" % (eez_min, eez_max), "DATA")
    r0_t_fn = eez_0 + "t"
    r0_t.save(r0_t_fn)
    r0_t = arcpy.sa.Reclassify(r0_t_fn, "VALUE", "%s %s 0" % (eez_min, eez_max), "DATA")
    r0_t.save(eez_0)
    arcpy.Delete_management(r0_t_fn)

    r1_t = arcpy.sa.Reclassify(eez, "VALUE", "%s %s 1" % (eez_min, eez_max), "DATA")
    r1_t_fn = eez_1 + "t"
    r1_t.save(r1_t_fn)
    r1 = arcpy.sa.Reclassify(r1_t_fn, "VALUE", "%s %s 1" % (eez_min, eez_max), "DATA")
    r1.save(eez_1)
    arcpy.Delete_management(r1_t_fn)
