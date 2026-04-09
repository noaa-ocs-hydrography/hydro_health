import os
import time
import arcpy

from HSTB.ArcExt.NHSP import Functions
from HSTB.ArcExt.NHSP import globals
exec("from HSTB.ArcExt.NHSP.globals import print")

windswath_shp = "_windswath.shp"
str_c_line_dissolved = "c_line_dissolved"
list_dissolved = ["Serial_Num", "Basin", "Nature", "Season", "Num"]

wgs_sr = arcpy.SpatialReference(4326)  # wgs 1984


def merge_storm_swaths(datapath, output_path):
    list_windswath_file = []
    with globals.timer("Created a feature class containing merged windswath shapefiles") as t:
        for f in os.listdir(datapath):
            if f.endswith(windswath_shp):
                list_windswath_file.append(os.path.join(datapath, f))
        # print(list_windswath_file)
        output_tmp = output_path + "_tmp"
        arcpy.Merge_management(list_windswath_file, output_tmp)
        d2 = arcpy.Describe(output_tmp)
        print((output_tmp, "projection:", d2.spatialReference.projectionName))
        arcpy.Project_management(output_tmp, output_path, wgs_sr)
        print((output_path, "projection:", arcpy.Describe(output_path).spatialReference.projectionName))
        arcpy.Delete_management(output_tmp)
