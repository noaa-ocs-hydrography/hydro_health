# Reef Raw Data Extraction
# Reef data were defined as OBSTRN point and polygon features with CATOBS = 5 (fish haven) and
# SEAARE point and polygon features with CATSEA = 9 (reef)
# Sanctuary bounds were extracted from the Office of National Marine Sanctuaries feature service: http://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/National_Marine_Sanctuaries/FeatureServer

# Define Variables
import os
import arcpy
import time
import arcgisscripting
import math
from HSTB.ArcExt.NHSP import Functions
from HSTB.ArcExt.NHSP.globals import Parameters
from HSTB.ArcExt.NHSP.reef_sanctuary import *

# Input Variables

reefs = Parameters("Reefs")
sanc = Parameters("Sanctuary")
ais = Parameters("AIS")

features_1 = ['.obstrn', '.seaare']
geoms_1 = ['_point', '_polygon']

sanc_raw = sanc.gdb + r'\RAW_Sanc'  # @todo Make copy and output to Raw Geodatabase, if it does not already exist
query = "(catobs = '5                        ' OR catsea = '9                             ')"  # @todo There is likely a better way to program this, but the select analysis tool will not work unless all of the noted spaces are included
tank_buff_m = "4360;9260;18520;37040"
nontank_buff_m = "1852;3704"

ais_tanker = ais.raster_classified_filename("tank_1")

reef_tank_r = reefs.raster_filename("tk")
reef_tank_rc = reefs.raster_classified_filename("tk")
reef_tank_rf_t = reefs.working_rastername("tk_rft")
reef_tank_rf = reefs.raster_final_filename("tk")

reef_notank_r = reefs.raster_filename("ntk")
reef_notank_rc = reefs.raster_classified_filename("ntk")
reef_notank_rf = reefs.raster_final_filename("ntk")

# Definitions
# Raster Reclassification Values
# Tanker
t_rc = tank_buff_m.split(";")
tank_rc_val = "%s %s %s;%s %s %s;%s %s %s;%s %s %s" % (int(t_rc[0]), int(t_rc[0]), 5, int(t_rc[1]), int(t_rc[1]), 4, int(t_rc[2]), int(t_rc[2]), 3, int(t_rc[3]), int(t_rc[3]), 2)

# Non Tanker
nt_rc = nontank_buff_m.split(";")
ntank_rc_val = "%s %s %s; %s %s %s" % (int(nt_rc[0]), int(nt_rc[0]), 2, int(nt_rc[1]), int(nt_rc[1]), 1)

# RAW DATA
# Extract OBSTRN and SEARE Data
# Make list of raw files
fns_raw = extract_raw(reefs)
fns_raw.append(sanc_raw)

# Project Data
fns_prj = []
i = 0
for fn_raw in fns_raw:
    if os.path.split(fn_raw)[1][0:4] == reefs.ext:
        fn_prj = fn_raw + "_prj"
        fns_prj.append(fn_prj)
    else:
        fn_prj = sanc.vector_processed_fullpath  # @todo - output to Vector Processed geodatabase
    Functions.check_spatial_reference(reefs.gdb, fn_raw, fn_prj, reefs.projection_name, reefs.projection_number)
    i += 1

# VECTOR PROCESSED
# Select OBSTRN with CATOBS = 5 and SEAARE with CATSEA = 9
fns_reef_vp = []
i = 0
for fn_prj in fns_prj:
    fn_vp = os.path.join(reefs.gdb, reefs.ext + geoms_1[i] + "_VP")
    arcpy.Select_analysis(fn_prj, fn_vp, query)  # @todo - output to Vector Processed geodatabase
    i += 1
    fns_reef_vp.append(fn_vp)

# RASTER
# Tankers
# Reefs
reef_buff_v = []
reef_buff_r = []
i = 0
for fn_vp in fns_reef_vp:
    fn_buff = os.path.join(reefs.gdb, "temp_Buff" + "%d" % (i))
    fn_ras_t = os.path.join(out_fldr_1, "temp_r" + "%s" % (i))
    arcpy.MultipleRingBuffer_analysis(fn_vp, fn_buff, tank_buff_m, "Meters", "value", "ALL", "FULL")
    arcpy.PolygonToRaster_conversion(fn_buff, "value", fn_ras_t, "MAXIMUM_AREA", "value", 500)
    reef_buff_v.append(fn_buff)
    reef_buff_r.append(fn_ras_t)
    i += 1

# Sanctuaries
sanc_buff_v = os.path.join(reefs.gdb, "temp_Buff_sanc")
sanc_buff_r = os.path.join(out_fldr_1, "temp_r3")
arcpy.MultipleRingBuffer_analysis(sanc.vector_processed_fullpath, sanc_buff_v, tank_buff_m, "Meters", "value", "ALL", "FULL")
arcpy.PolygonToRaster_conversion(sanc_buff_v, "value", sanc_buff_r, "MAXIMUM_AREA", "value", 500)

# Join Tanker Reef and Sanctuary Files (Multi-Buffer and Raster)
reef_buff_v.append(sanc_buff_v)
reef_buff_r.append((sanc_buff_r))

# Mosaic to New Raster - Reef and Sanctuary Buffered Rasters
arcpy.MosaicToNewRaster_management(reef_buff_r, os.path.split(reef_tank_r)[0], os.path.split(reef_tank_r)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MINIMUM", "FIRST")

# Non Tankers
reef_buff_nt_v = []
reef_buff_nt_r = []
i = 0
for fn_vp in fns_reef_vp:
    fn_buff_nt = os.path.join(reefs.gdb, "temp_Buff_nt" + "%d" % (i))
    fn_ras_nt_t = os.path.join(out_fldr_1, "temp_r_nt" + "%s" % (i))
    arcpy.MultipleRingBuffer_analysis(fn_vp, fn_buff_nt, nontank_buff_m, "Meters", "value", "ALL", "FULL")
    arcpy.PolygonToRaster_conversion(fn_buff_nt, "value", fn_ras_nt_t, "MAXIMUM_AREA", "value", 500)
    reef_buff_nt_v.append(fn_buff_nt)
    reef_buff_nt_r.append(fn_ras_nt_t)
    i += 1

# Sanctuaries
sanc_buff_nt_v = os.path.join(reefs.gdb, "temp_Buff_nt_sanc")
sanc_buff_nt_r = os.path.join(out_fldr_1, "temp_nt_sanc")
arcpy.MultipleRingBuffer_analysis(sanc.vector_processed_fullpath, sanc_buff_nt_v, nontank_buff_m, "Meters", "value", "ALL", "FULL")
arcpy.PolygonToRaster_conversion(sanc_buff_nt_v, "value", sanc_buff_nt_r, "MAXIMUM_AREA", "value", 500)

# Join Tanker Reef and Sanctuary Files (Multi-Buffer and Raster)
reef_buff_nt_v.append(sanc_buff_nt_v)  # Delete
reef_buff_nt_r.append(sanc_buff_nt_r)

# Mosaic to New Raster - Reef and Sanctuary Buffered Rasters
arcpy.MosaicToNewRaster_management(reef_buff_nt_r, os.path.split(reef_notank_r)[0], os.path.split(reef_notank_r)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MINIMUM", "FIRST")

# RASTER CLASSIFIED
# Tanker
tank_rc_temp = arcpy.sa.Reclassify(reef_tank_r, "VALUE", tank_rc_val, "DATA")
arcpy.MosaicToNewRaster_management([tank_rc_temp, eez_1], os.path.split(reef_tank_rc)[0], os.path.split(reef_tank_rc)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")

# Non Tanker
ntank_rc_temp = arcpy.sa.Reclassify(reef_notank_r, "VALUE", ntank_rc_val, "DATA")
arcpy.MosaicToNewRaster_management([ntank_rc_temp, eez_0], os.path.split(reef_notank_rc)[0], os.path.split(reef_notank_rc)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")

# Raster Final
# Tanker
tank_rf_r = arcpy.sa.ExtractByMask(reef_tank_rc, eez_0)
tank_rf_r.save(reef_tank_rf_t)

# Clip Tanker to Extents of Tanker Traffic, such that, if no tanker (assume all tanker traffic captured by AIS, then consequence to tanker is 0)
ais_tanker_binary_ras = arcpy.sa.Raster(ais_tanker)
reef_tank_rf_t_ras = arcpy.sa.Raster(reef_tank_rf_t)

ais_reefs_tank = ais_tanker_binary_ras * reef_tank_rf_t_ras
ais_reefs_tank.save(reef_tank_rf + "5")

# Non Tanker
tank_rf_r = arcpy.sa.ExtractByMask(reef_notank_rc, eez_0)
tank_rf_r.save(reef_notank_rf)

# Delete Intermediate Files
del_files = reef_buff_v + reef_buff_nt_v + reef_buff_nt_r + reef_buff_r + fns_prj
for del_file in del_files:
    arcpy.Delete_management(del_file)


'''# This is still not working - but will work when called independently
# Create Buffers for Tankers - Note multiringBuffer would not work in for loop (Runtime error: Traceback (most recent call last): File "<string>", line 12, in <module> AttributeError: 'NoneType' object has no attribute 'write'"

for fn_vp in fns_vp:
    fn_buff = os.path.join(reefs.gdb, "temp_Buff"+"%d"%(i))
    fn_ras_t = os.path.join(out_fldr_1, "temp_r"+"%s"%(i))
    if os.path.split(fn_vp)[1][0:4] == reefs.ext:
        print("I am a reef")
        arcpy.MultipleRingBuffer_analysis (fn_vp, fn_buff, tank_buff_m, "Meters", "value", "ALL", "FULL")
        arcpy.PolygonToRaster_conversion(fn_buff, "value", fn_ras_t, "MAXIMUM_AREA", "value", 500)
    elif os.path.split(fn_vp)[1][0:4] == sanc.ext:
        print("I am a sanc")
        arcpy.MultipleRingBuffer_analysis (fn_vp, fn_buff, nontank_buff_m, "Meters", "value", "ALL", "FULL")
        arcpy.PolygonToRaster_conversion(fn_buff, "value", fn_ras_t, "MAXIMUM_AREA", "value", 500)
    else:
        print("Did not complete buffer analysis for file %s"%(fn_vp))
        print("I am not a reef or a sanc")
    fns_buff.append(fn_buff)
    fns_ras_t.append(fn_ras_t)
    i+=1
arcpy.MultipleRingBuffer_analysis (fn_vp, fn_buff, tank_buff_m, "Meters", "value", "ALL", "FULL")
    #Mosaic Tanker Buffered Raster to EEZ = 1
tank_r_list = fn_ras_t.append(eez_1)
arcpy.MosaicToNewRaster_management(tank_r_list, out_fldr_1, tank_ras, reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")
'''
