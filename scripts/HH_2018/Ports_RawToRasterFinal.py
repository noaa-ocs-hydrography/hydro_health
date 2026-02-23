# Ports
# This script assumes two input variables
# 1. USACE Principal Ports Layer (http://www.navigationdatacenter.us/data/datappor.htm)
# 2. Delineated ports bounds. This layer was assembled in summer 2016 by HSD intern and stored in H:\HH_2018\Raw\Raw.gdb\RAW_Ports_Bounds.
#    An alternate solution is to use MCD's rescheme band 5 ID = 9 coverage.

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
ports = Parameters("Ports")
aux = Parameters("AUX")

# User-specified inputs
usace_total_tonnage_field = "TOTAL"
raw_ports_prj_fn = ports.working_filename("Bounds_prj")
raw_ports_usace_prj_fn = ports.vector_filename("USACE_prj")


# Input Variables
raw_ports = ports.raw_filename("Bounds")
raw_ports_usace = ports.raw_filename("USACE")
eez_r = aux["eez_0_ras"]

# Output Variables
ports_v = ports.vector_filename()
ports_vp = ports.vector_processed_filename()
ports_r = ports.raster_filename()
ports_rc = ports.raster_classified_filename()
ports_rf_t = ports.working_rastername("rf_t")
ports_final = ports.raster_final_filename()
ports_usace_tonnage_populated = ports.working_filename("tonnage_populated")

# RAW DATA
# Project Raw Files, if necessary
raw_ports_prj = Functions.check_spatial_reference(raw_ports, raw_ports_prj_fn, ports.projection_name, ports.projection_number)
raw_ports_usace_prj = Functions.check_spatial_reference(raw_ports_usace, raw_ports_usace_prj_fn, ports.projection_name, ports.projection_number)

# THIS SHOULD BE VECTOR
# Spatially Join Ports Bounds with USACE Principal Ports layer.
# In port bounds where multiple USACE principal ports points exist, use the maximum value
fieldmappings = arcpy.FieldMappings()
total_tonnage = arcpy.FieldMap()
total_tonnage.addInputField(raw_ports_usace_prj, usace_total_tonnage_field)
total_tonnage.mergeRule = "Max"
fieldmappings.addFieldMap(total_tonnage)
arcpy.SpatialJoin_analysis(raw_ports_prj, raw_ports_usace_prj, ports_v, "JOIN_ONE_TO_ONE", "KEEP_ALL", fieldmappings, "INTERSECT", 2500, "METERS")

# THIS SHOULD BE VECTOR PROCESSED
# Populate ports where Total Tonnage = NULL with Tonnage = 1
query = usace_total_tonnage_field + " IS NULL"
ports_select_temp_layer = "ports_temp_lyr"
arcpy.MakeFeatureLayer_management(ports_v, ports_select_temp_layer)
arcpy.SelectLayerByAttribute_management(ports_select_temp_layer, "NEW SELECTION", query)
arcpy.CalculateField_management(ports_select_temp_layer, usace_total_tonnage_field, 1)
arcpy.SelectLayerByAttribute_management(ports_select_temp_layer, "CLEAR_SELECTION")
arcpy.FeatureClassToFeatureClass_conversion(ports_select_temp_layer, os.path.split(ports_vp)[0], os.path.split(ports_vp)[1])

# Convert to Raster
arcpy.PolygonToRaster_conversion(ports_vp, usace_total_tonnage_field, ports_r, "MAXIMUM_COMBINED_AREA", usace_total_tonnage_field, ports.cell_size)

# Reclassify Raster
r = arcpy.sa.Reclassify(ports_r, "VALUE", "0 0;0 100000 1;100000 1000000 2;1000000 10000000 3;10000000 100000000 4;100000000 100000000000000 5")
r.save(ports_rc)

arcpy.MosaicToNewRaster_management([ports_rc, eez_r], os.path.dirname(ports_rf_t), os.path.basename(ports_rf_t), ports.projection_number, "32_BIT_SIGNED", ports.cell_size, "1", "MAXIMUM", "FIRST")

# Clip to EEZ
e = arcpy.sa.ExtractByMask(ports_rf_t, eez_r)
e.save(ports_final)

# Delete Unnecessary Files
arcpy.Delete_management(ports_usace_tonnage_populated)
