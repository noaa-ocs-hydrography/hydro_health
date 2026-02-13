# This script assumes an input superseded catzoc layer and corresponding suvey year layer and outputs four files:
# 1. CATZOC < A2 Bounds
# 2. Survey Age
# 3. Survey Quality Likelihood Term
# 4. Initial Survey Score


# Import modules
import os
import time
import arcpy
import arcgisscripting
from arcpy import env
from arcpy.sa import *
import numpy
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
qual = Parameters("SurveyQuality")
aux = Parameters("AUX")
iss = Parameters("ISS")

# User-Specified Input
qual_r = qual.raster_makefilename()
qual_yr_r = qual.raster_makefilename("yr")
qual_od = qual.raster_makefilename("od")

# Define Parameters
final_survey_age_field = "survey_age"
final_survey_age_calc = "2018-!gridcode!"
final_catzoc_bcdu_calc = "gridcode>2"

eez_r = aux["eez_0_ras"]

# Output Variables
qual_yr_p = qual.working_makefilename("yr_poly")  # Delete
qual_age = qual.aux_makefilename("age", False)
qual_p = qual.working_makefilename("poly")  # Delete
qual_bcdu_p_temp = qual.working_makefilename("bcdu_p")  # Delete
qual_bcdu_p = qual.aux_makefilename("bcdu")
qual_bcdu_r = qual.aux_makefilename("bcdu", gdb=False)
qual_rc = qual.raster_classified_makefilename()
qual_rf = qual.raster_final_makefilename()
od_overlap = qual.working_makefilename("od_add", False)
iss_r = iss.raster_makefilename()
iss_rf_t = iss.working_rastername("t")
iss_rf = iss.raster_final_makefilename()


# Create Superseded CATZOC BCDU Raster
arcpy.RasterToPolygon_conversion(qual_r, qual_p, "NO_SIMPLIFY", "value")
arcpy.Select_analysis(qual_p, qual_bcdu_p_temp, final_catzoc_bcdu_calc)
arcpy.PolygonToRaster_conversion(qual_bcdu_p_temp, "gridcode", qual_bcdu_r, "MAXIMUM_COMBINED_AREA", "gridcode", qual.cell_size)
Functions.generate_polygon_bounds(qual_bcdu_p_temp, qual_bcdu_p)

# Create Survey Age Raster
arcpy.RasterToPolygon_conversion(qual_yr_r, qual_yr_p, "NO_SIMPLIFY", "value")
arcpy.AddField_management(qual_yr_p, final_survey_age_field, "double")
arcpy.CalculateField_management(qual_yr_p, final_survey_age_field, final_survey_age_calc)
arcpy.PolygonToRaster_conversion(qual_yr_p, final_survey_age_field, qual_age, "MAXIMUM_COMBINED_AREA", final_survey_age_field, qual.cell_size)

# Survey Quality Likelihood
# Reclassify CATZOC Raster using the following classification
# CATZOC
# A1         # 1
# A2         # 1
# B          # 3
# C          # 4
# D          # 5
# U          # 5

r = arcpy.sa.Reclassify(qual_r, "VALUE", "1 1;2 1;3 3;4 4;5 5;6 5", "DATA")
r.save(qual_rc)

# Clip Final CATZOC_RC to EEZ Bounds
e = arcpy.sa.ExtractByMask(qual_rc, eez_r)
e.save(qual_rf)


# Create Initial Score Raster
qual_r_ras = arcpy.sa.Raster(qual_r)
qual_od_ras = arcpy.sa.Raster(qual_od)

qual_r_od_add = qual_r_ras + qual_od_ras

# Reclassify object detection raster to only include values where OD raster overlaps with CATZOC = 1
#r = arcpy.sa.Reclassify(qual_r_od_add, "VALUE", "111 0;112 116 NODATA", "DATA")
# qual_r_od_add.save(od_overlap)
arcpy.gp.Reclassify_sa(qual_r_od_add, "VALUE", "111 0;112 116 NODATA", od_overlap, "DATA")

# Mosaic OD and Superseded CATZOC Raster based on minimum value
arcpy.MosaicToNewRaster_management([qual_r, od_overlap], os.path.split(iss_r)[0], os.path.split(iss_r)[1], iss.projection_number, "32_BIT_SIGNED", str(iss.cell_size), "1", "MINIMUM", "FIRST")

# Reclassify ISS Raster by ISS Score based on the following table:
# 110 = Object Detection
# 100 = A1/A2
# 80 = B
# 30 = C
# 0 = D/U
arcpy.gp.Reclassify_sa(iss_r, "VALUE", "0 110; 1 100; 2 100; 3 80; 4 30; 5 0; 6 0", iss_rf_t, "DATA")

# Clip ISS to EEZ
e = arcpy.sa.ExtractByMask(iss_rf_t, eez_r)
e.save(iss_rf)
