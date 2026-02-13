# Density and Proximity to Active Captain Reports
# Active Captain Data found on GIS Server, here https://ocs-vs-appd7.nos.noaa/arcgis/services

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of active captain report.
# Note, active captain reports are being used as a proxy for recreational boater traffic.
# The proxy for recreational boater traffic is deduced from the active captain reports using the
# comment count (comment count + 1, since comment count = 0 if one report). Classify risk using the following table

# Risk Category                       1            2                3                4                  5
# Number of AC Reports w/i 5 nm       0            1               2-10               11-100             100+


# Import modules
import os
import time
import arcpy
import arcgisscripting
from arcpy import env
from arcpy.sa import *
import numpy
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP.globals import Parameters

# Define Parameters
ac = Parameters("ActiveCaptain")
aux = Parameters("AUX")
ais = Parameters("AIS")
inputs = Parameters("Input")

ac_raw = ac.raw_filename()

ac_vec = ac.vector_processed_filename()
ac_raster = ac.raster_filename()
ac_raster_classified_t = ac.raster_classified_filename("t")
ac_raster_classified = ac.raster_classified_filename()
ac_raster_final_t = ac.raster_final_filename("t")
ac_raster_final = ac.raster_final_filename()
eez_r = aux["eez_0_ras"]

grid_ws = inputs["grid_ws"]  # "r'H:\GRID\Grid.gdb'  # USER DEFINED

ais_rc_2016 = ais.raster_classified_filename("all_u")
ais_ac_rf_t = ais.working_rastername("ac")
ais_ac_rf = ais.raster_final_filename("ac")

# Hard-Coded Variables
den_fns = []  # delete
den_c_fns = []  # delete

# Project Data
ac_vp = Functions.check_spatial_reference(ac_raw, ac_vec, ac.projection_name, ac.projection_number)

# Add TrafficProxy Field and calculate as comment+1
field = "TrafficProxy"
field_exp = "!COMMENTCOUNT!+1"
arcpy.AddField_management(ac_vp, field, "DOUBLE")
arcpy.CalculateField_management(ac_vp, field, field_exp, "PYTHON_9.3")

# Export Classified Density Rasters for all Grids
for igrid in range(1, 14):  # Improve: Find all matching grid extents in the geodatabase
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)
    dens_out = ac.raster_filename("%02d_den" % igrid)
    dens_c_out = ac.raster_filename("%02d_denC" % igrid)

    # Read Grid and Extract XY Coordinates from Groundings File
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path)
    h_den, h_den_f = grid.ComputePointDensity(ac_vp, cell_densities=[gridclass.circle39], weights="TrafficProxy")

    try:
        grid.ExportMatchingRaster(h_den_f, dens_out)
        print(("Finished Density Raster Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    den_fns.append(dens_out)

    # Reclassify Point Active Captain Density Array. Copy convolved array to five separate arrays to be classified separately. Maximum of separate arrays will be exported
    # (prevents overwriting data with risk category)

    h_den_5 = numpy.copy(h_den_f)
    h_den_4 = numpy.copy(h_den_f)
    h_den_3 = numpy.copy(h_den_f)
    h_den_2 = numpy.copy(h_den_f)

    h_den_5[(h_den_5 <= 100)] = 1
    h_den_5[(h_den_5 > 100)] = 5

    h_den_4[(h_den_4 < 11)] = 1
    h_den_4[(h_den_4 > 100)] = 1
    h_den_4[(h_den_4 >= 11) & (h_den_4 <= 100)] = 4

    h_den_3[(h_den_3 < 2)] = 1
    h_den_3[(h_den_3 >= 11)] = 1
    h_den_3[(h_den_3 >= 2) & (h_den_3 < 11)] = 3

    h_den_2[(h_den_2 > 1)] = 99999
    h_den_2[(h_den_2 == 1)] = 2
    h_den_2[(h_den_2 == 99999)] = 1
    h_den_2[(h_den_2 == 0)] = 1

    # Max
    ac_cat = numpy.array((h_den_5, h_den_4, h_den_3, h_den_2))
    ac_cat_f = numpy.max(ac_cat, 0)

    try:
        grid.ExportMatchingRaster(ac_cat_f, dens_c_out)
        print(("Finished Density Raster Classify Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    den_c_fns.append(dens_c_out)
    print(("Finished Grid %02d at %.1f secs" % (igrid, time.time() - t_start)))

# Combine Density and Classified Density Rasters
t_start = time.time()
arcpy.MosaicToNewRaster_management(den_fns, ac.raster_dir, ac.raster, ac.projection_number, "32_BIT_SIGNED", str(ac.cell_size), "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(den_c_fns, ac.raster_classified_dir, os.path.basename(ac_raster_classified_t), ac.projection_number, "32_BIT_SIGNED", str(ac.cell_size), "1", "MAXIMUM", "FIRST")
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))

# Reclassify final raster to remove zero values in no data areas
r = arcpy.sa.Reclassify(ac_raster_classified_t, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
r.save(ac_raster_final_t)

# Clip final raster to EEZ Bounds
e = arcpy.sa.ExtractByMask(ac_raster_final_t, eez_r)
e.save(ac_raster_final)

''' Complete in Axiom_AIS_Mosaic Script

# Merge Active Captain with AIS Unique Count Raster
arcpy.MosaicToNewRaster_management([ac_raster_final, ais_rc_2016], os.path.split(ais_ac_rf_t)[0], os.path.split(ais_ac_rf_t)[1], ais.projection_number, "32_BIT_SIGNED", str(ais.cell_size), "1", "MAXIMUM", "FIRST")

# Clip Merged AIS/AC by EEZ
e = arcpy.sa.ExtractByMask(ais_ac_rf_t, eez_r)
e.save(ais_ac_rf)
'''

# Delete Unnecessary Files
t_start = time.time()
for fn in range(0, 13):
    print(("Deleting Grid %02d Intermediate Rasters" % (fn + 1)))
    arcpy.Delete_management(str(den_fns[fn]))
    arcpy.Delete_management(str(den_c_fns[fn]))
print(("Finished Process at %.1f secs" % (time.time() - t_start)))
arcpy.Delete_management(ac_raster_classified_t)
arcpy.Delete_management(ac_raster_final_t)

# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution) - second number is 2018 run
# Grid 1: 32 s - 42 s -
# Grid 2: 60 s - 66 s -
# Grid 3: 47 s - 50.5 s -
# Grid 4: 19 s - 17.4 s -
# Grid 5: 25 s - 76 s -
# Grid 6: 673 s -265 s -
# Grid 7: 96 s -45 s -
# Grid 8: 44 s - 22 s -
# Grid 9: 349 s - 164 s -
# Grid 10: 48 s - 24 s -
# Grid 11: 47 s - 21 s -
# Grid 12: 73 s - 29 s -
# Grid 13: 50 s - 20 s -
# Mosaic: 120 s -
