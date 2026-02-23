# Currents

# Import modules
import os
import time
import traceback

import numpy
import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import Functions
from HSTB.ArcExt.NHSP import gridclass
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.globals import Parameters, timer
exec("from HSTB.ArcExt.NHSP.globals import print")
# from arcpy import env


aux = Parameters("AUX")

# FIGURE THIS OUT
# arcpy.env.workspace = r"H:\NHSP_2_0\Christy\CURRENTS"

# Variables
tin_field = "MEANCU"  # user defined
max_edge = 4000  # user defined
sandy_lt20m = aux["sandy_lt20m_ras"]
mbounds = aux["grid_extents_ras"]

# Output variables

# Set Environmental Parameters
r = (arcpy.sa.Raster(mbounds)).extent
arcpy.env.extent = arcpy.Extent(r.XMin, r.YMin, r.XMax, r.YMax)
arcpy.env.snapRaster = mbounds

# Project data, if projected in desired coordinate system, set projected variable to raw variable


def merge_east_west_data(curr):
    """combines east and west coast and puts result in vector_processed"""
    # Input Files
    curr_raw_e = curr.raw_filename("E")  # user defined
    curr_raw_w = curr.raw_filename("W")  # user defined
    curr_e_prj_fn = curr.working_filename("e_prj")
    curr_w_prj_fn = curr.working_filename("w_prj")
    curr_raw_e_prj = Functions.check_spatial_reference(curr_raw_e, curr_e_prj_fn, curr.projection_name, curr.projection_number)
    curr_raw_w_prj = Functions.check_spatial_reference(curr_raw_w, curr_w_prj_fn, curr.projection_name, curr.projection_number)

    # Merge East Coast and West Coast Current Data
    curr_vp = curr.vector_processed_filename()
    arcpy.Merge_management([curr_raw_e_prj, curr_raw_w_prj], curr_vp)
    # remove the projected data if it was made
    arcpy.Delete_management(curr_e_prj_fn)
    arcpy.Delete_management(curr_w_prj_fn)
    return curr_vp


def create_tin_raster(curr, curr_vp):
    # Create TIN of merged current data
    # Create Tin
    c_tin = curr.working_rastername("TIN")  # Delete
    c_tin_bounds = curr.working_rastername("Bounds")  # Delete
    c_raster = curr.raster_filename()
    arcpy.CreateTin_3d(c_tin, curr.projection_number, "'%s' %s Mass_Points <None>" % (curr_vp, tin_field), "Delaunay")
    # Delineate Tin Data Area
    arcpy.DelineateTinDataArea_3d(c_tin, max_edge, "ALL")
    # Export extents of delineated TIN
    arcpy.TinDomain_3d(c_tin, c_tin_bounds, "POLYGON")
    # Convert Tin to Raster
    arcpy.TinRaster_3d(c_tin, c_raster, "FLOAT", "LINEAR", "CELLSIZE %s" % (curr.cell_size), "1")
#     except arcpy.ExecuteError:
#         print(arcpy.GetMessages())
#     except Exception as err:
#         print(err)
    arcpy.Delete_management(c_tin)
    return c_raster


def classify_raster(curr, c_raster):
    # Classify Currents Raster
    # r = arcpy.sa.Reclassify(c_raster, "VALUE", "0 0.1 1;0.1 0.2 2;0.2 0.5 3;0.5 1 4;1 100 5;NODATA NODATA", "NODATA")
    c_raster_rct1 = curr.working_rastername("RC_t")  # Delete
    c_raster_rc = curr.raster_classified_filename()
    arcpy.gp.Reclassify_sa(c_raster, "VALUE", "0 0.1 1;0.1 0.2 2;0.2 0.5 3;0.5 1 4;1 100 5;NODATA NODATA", c_raster_rct1, "NODATA")

    # r.save(c_raster_rct1)
    # r2 = arcpy.sa.Reclassify(c_raster_rct1, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
    arcpy.gp.Reclassify_sa(c_raster_rct1, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", c_raster_rc, "NODATA")
    # r2.save(c_raster_rc)
    arcpy.Delete_management(c_raster_rct1)
    return c_raster_rc


def clip_to_shallow_sandy_area(curr, c_raster_rc):
    # Clip Current data to Depth (Z<20 m) and Sandy Bounds
    c_raster_final = curr.raster_final_filename()
    m = arcpy.sa.ExtractByMask(c_raster_rc, sandy_lt20m)
    m.save(c_raster_final)
    return c_raster_final


def execute(ini, grids=list(range(1, 14))):
    curr = Parameters("Curr", ini)
    curr_vp = merge_east_west_data(curr)
    c_raster = create_tin_raster(curr, curr_vp)
    c_raster_rc = classify_raster(curr, c_raster)
    c_raster_final = clip_to_shallow_sandy_area(curr, c_raster_rc)


def main():
    execute(globals.initial_values)
