# Density and Proximity to Groundings
# Groundings are defined as a grounding incident by a non-recreational or fishing vessel.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of grounding. Classify risk using the following table

# Risk Category                       1            2                3                4                  5
# Number of Groundings w/i 2 nm       0            0                1               2-4                 4+
# Distance to Hazard                >2 km       1 - 2 km        0.5 - 1 km      0.5 - 0.25 km       < 0.25 km

# This script assumes the following
# All input layers are projected into the correct and similar coordinate system
# All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os
import time
import arcpy
import arcgisscripting
from arcpy import env
from arcpy.sa import *
import numpy
import scipy
import scipy.signal
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP.globals import Parameters
from HSTB.ArcExt.NHSP import groundings

ground = Parameters("Groundings")
aux = Parameters("AUX")
inputs = Parameters("Input")

# FIX THIS
# Workspace
arcpy.env.workspace = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Working\Groundings"

# User-Defined Input Variables
ground_raw = os.path.join(ground.raw_dir, ground["csv_data"])  # User-Defined
x_coords = ground["x_coords"]  # User-Defined
y_coords = ground["y_coords"]  # User-Defined

# Parameters
eez_r = aux["eez_0_ras"]
grid_ws = inputs["grid_ws"]  # "r'H:\GRID\Grid.gdb'  # USER DEFINED

dis_fns = []
dens_fns = []
max_fns = []

# Output Variables
ground_dis_rc = ground.raster_classified_filename("dis")
ground_den_rc = ground.raster_classified_filename("den")
ground_rc_t = ground.working_filename("rc_t", False)
ground_rc = ground.raster_classified_filename()
ground_f = ground.raster_final_filename()

# VECTOR PROCESSED
# Read raw grounding data and extract only grounding incidents


ground_vp = groundings.create_vp(ground_raw, x_coords, y_coords, ground)
if not ground_vp:
    exit()
# Export Classified Distance and Density Rasters for all Grids (13 GRIDS TOTAL)
for igrid in range(2, 3):
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)
    dens_c_out = ground.working_filename("Grnd_%02d_denC" % igrid, False)
    dis_c_out = ground.working_filename("Grnd_%02d_disC" % igrid, False)
    dens_dis_max_out = ground.working_filename("Grnd_%02d_max" % igrid, False)

    # Read Grid and Extract XY Coordinates from Groundings File and Generate histogram of point groundings features based on extents of grid raster
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)
    h_den, h_dis3, h_dis5, h_dis7, h_dis9, h_den_f = grid.ComputePointDensity(ground_vp, [gridclass.circle3, gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle17])

    # Reclassify Numpy Array based on density of groundings and Export Classified Raster
    h_den_f[(h_den_f > 4)] = 5
    h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
    h_den_f[(h_den_f == 1)] = 2
    h_den_f[(h_den_f < 1)] = 1

    try:
        grid.ExportMatchingRaster(h_den_f, dens_c_out)
        print(("Finished Density Raster Classify Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    # Reclassify Point Groundings Distance Histograms and Export Classified Raster
    h_dis3[(h_dis3 != 0)] = 5
    h_dis5[(h_dis5 != 0)] = 4
    h_dis7[(h_dis7 != 0)] = 3
    h_dis9[(h_dis9 != 0)] = 2

    h_dis_max = numpy.array((h_dis9, h_dis7, h_dis5, h_dis3))
    h_dis_max_f = numpy.max(h_dis_max, 0)
    try:
        grid.ExportMatchingRaster(h_dis_max_f, dis_c_out)
        print(("Finished Distance Raster Classify Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    # Calculate Maximum Density or Distance and Export Raster
    h_den_dis_array = numpy.array((h_den_f, h_dis_max_f))
    h_den_dis_max = numpy.max(h_den_dis_array, 0)
    try:
        grid.ExportMatchingRaster(h_den_dis_max, dens_dis_max_out)
        print(("Finished Max Raster Classify Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Reclassified Maximum Distance and Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    # Create List of Distance, Density, and Max Rasters to be Combined and Deleted later
    dis_fns.append(dis_c_out)
    dens_fns.append(dens_c_out)
    max_fns.append(dens_dis_max_out)

# Combine Distance, Density, and Max Rasters
t_start = time.time()
print("Start Raster Mosaic")
arcpy.MosaicToNewRaster_management(dis_fns, os.path.split(ground_dis_rc)[0], os.path.split(ground_dis_rc)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(dens_fns, os.path.split(ground_den_rc)[0], os.path.split(ground_den_rc)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(max_fns, os.path.split(ground_rc_t)[0], os.path.split(ground_rc_t)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))

# Reclassify the mosaicked raster to remove zero values in no data areas
print("Start Raster Reclassification")
arcpy.gp.Reclassify_sa(ground_rc_t, "VALUE", "111 0;112 116 NODATA", ground_rc, "DATA")
print(("Finished Raster Reclassification at %.1f secs" % (time.time() - t_start)))

# Clip final raster to EEZ Bounds
t_start = time.time()
outExtractByMask = arcpy.sa.ExtractByMask(ground_rc, eez_r)
outExtractByMask.save(ground_f)
print(("Finished Raster Extraction at %.1f secs" % (time.time() - t_start)))

# Delete Unnecesary Files
t_start = time.time()
for fn in range(0, 13):
    print(("Deleting Grid %02d Intermediate Rasters" % (fn + 1)))
    arcpy.Delete_management(str(dis_fns[fn]))
    arcpy.Delete_management(str(dens_fns[fn]))
    arcpy.Delete_management(str(max_fns[fn]))
print(("Finished Process at %.1f secs" % (time.time() - t_start)))
arcpy.Delete_management(str(ground_rc_t))

# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution;
# Grid 1: 51 s
# Grid 2: 73 s
# Grid 3: 47 s
# Grid 4: 32 s
# Grid 5: 71 s
# Grid 6: 205 s
# Grid 7: 58 s
# Grid 8: 44 s
# Grid 9: 130 s
# Grid 10: 48 s
# Grid 11: 43 s
# Grid 12: 48 s
# Grid 13: 32 s
# Mosaic: 124 s
