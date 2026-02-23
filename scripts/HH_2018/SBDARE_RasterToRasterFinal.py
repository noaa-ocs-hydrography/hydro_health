# SBDARE
# This script classifies the seabed type data for the likelihood term as follows:
# Soft Bottom = 2
# Hard Bottom = 5
# This script also creates a "soft bottom" for the changeability term by assuming all inshore areas
# included in the input seabed type data are sandy. The sandy area is then clipped by depths less than
# 20 m and Depths less than 40 m to create the limiting bounds for the current and hurricane changeability
# terms, respectively.

import arcpy
import os
from arcpy import sa

# Input Variables
shoreline = r'H:\Hydro_Health\2016\Raster_Auxiliary\shoreline'
sbdare = r'H:\Hydro_Health\2016\Raster\sbdare'
sbdare_hard = 1  # Classification in sbdare raster as hard bottom
sbdare_soft = 0  # Classification in sbdare raster as soft bottom
out_fldr = r'C:\Users\Christina.Fandel\Fandel\SBDARE'
output_prj = 102008  # USER DEFINED
cellSize = 500
depth_lt20m = r'H:\Hydro_Health\2016\Raster_Auxiliary\depth_lt20m'
depth_lt40m = r'H:\Hydro_Health\2016\Raster_Auxiliary\depth_lt40m'

# Output Variables
ship_damage_t = os.path.join(out_fldr, "ship_dam_ft")
ship_damage = os.path.join(out_fldr, "ship_dam_f")
shoreline_rc_t = os.path.join(out_fldr, "shoreline_rct")
shoreline_rc = os.path.join(out_fldr, "shoreline_rc")
shoreline_sbdare = os.path.join(out_fldr, "sbdare_sl")
shoreline_sandy = os.path.join(out_fldr, "sandy_sl")
sandy_lt20m = os.path.join(out_fldr, "sandy_lt20m")
sandy_lt40m = os.path.join(out_fldr, "sandy_lt40m")

# Likelihood:
# Reclassify seabed type data as 2 (soft bottom) or 5 (hard bottom)
sbdare_rc_remap = arcpy.sa.RemapRange([[sbdare_hard, sbdare_hard, 5], [sbdare_soft, sbdare_soft, 2], ["NODATA", "NODATA"]])
sbdare_rc = arcpy.sa.Reclassify(sbdare, "VALUE", sbdare_rc_remap, "NODATA")
sbdare_rc.save(ship_damage_t)
sbdare_rc2 = arcpy.sa.Reclassify(ship_damage_t, "VALUE", "0 NODATA;2 2 2;5 5 5", "NODATA")
sbdare_rc2.save(ship_damage)


# Changeability
# Reclassify shoreline data as soft bottom
sl_min = float(arcpy.GetRasterProperties_management(shoreline, "MINIMUM").getOutput(0))
sl_max = float(arcpy.GetRasterProperties_management(shoreline, "MAXIMUM").getOutput(0))
sl_remap = arcpy.sa.RemapRange([[sl_min, sl_max, 2], ["NODATA", "NODATA"]])
sl_rc = arcpy.sa.Reclassify(shoreline, "VALUE", sl_remap, "NODATA")
sl_rc.save(shoreline_rc_t)
sl_rc2 = arcpy.sa.Reclassify(shoreline_rc_t, "VALUE", "0 NODATA;2 2 2", "NODATA")
sl_rc2.save(shoreline_rc)

# Mosaic Classified Shoreline with Classified SBDARE
arcpy.MosaicToNewRaster_management([ship_damage, shoreline_rc], os.path.split(shoreline_sbdare)[0], os.path.split(shoreline_sbdare)[1], output_prj, "32_BIT_SIGNED", str(cellSize), "1", "MAXIMUM", "FIRST")

# Reclassify Classified Shoreline/SBDARE Raster as sandy areas only
sand_sl_rc = arcpy.sa.Reclassify(shoreline_sbdare, "VALUE", "5 NODATA;2 2 2", "NODATA")
sand_sl_rc.save(shoreline_sandy)

# Clip Shoreline/SBDARE Raster with Depths LT 20 m and Depths LT 40 m
sl_z20 = arcpy.sa.ExtractByMask(shoreline_sandy, depth_lt20m)
sl_z20.save(sandy_lt20m)
sl_z40 = arcpy.sa.ExtractByMask(shoreline_sandy, depth_lt40m)
sl_z40.save(sandy_lt40m)

# Delete Management
arcpy.Delete_management(ship_damage_t)
arcpy.Delete_management(shoreline_rc_t)
arcpy.Delete_management(shoreline_sbdare)
