# CATZOC Disambiguation 
# This script completes the following:
    # Reprojects the input CATZOC and Shoreline-to-EEZ layers to WGS84 EASE
    # Extends the CATZOC layer coverage to the EEZ and replaces all non-surveyed areas with a CATZOC = 6 (undefined).
    # Removes overlapping CATZOC coverage to maintain lower CATZOC layer (e.g. overlap of 5 (CATZOC D) and 2 (CATZOC A1), 2 is maintained and 5 is deleted)
# Final Output is c_null_final (CATZOC_NULL_Superseded_Raster)

import arcpy, os, arcgisscripting
from arcpy import env

arcpy.env.OverWriteOutput=1

# Define Inputs, Outputs and Workspace
sl_eez = arcpy.GetParameter(0)
c = arcpy.GetParameter(1)
output_ws = str(arcpy.GetParameter(2))
c_prj = os.path.join(output_ws, 'CATZOC_WGS84EASE')
c_dissolve = os.path.join(output_ws, 'CATZOC_Dissolve_ALL')
null = os.path.join(output_ws, 'USA_Shoreline_to_EEZ_MINUS_CATZOC')
c_null_m = os.path.join(output_ws, 'CATZOC_NULL_JOIN_temp')
c_null_m_d = os.path.join(output_ws, 'CATZOC_NULL_JOIN_temp2')
c_null = os.path.join(output_ws, 'CATZOC_NULL_JOIN')
temp = "c_null_temp"
c_null_s = os.path.join(output_ws, 'CATZOC_NULL_Superseded')
c_null_final = os.path.join(output_ws, 'CATZOC_NULL_Superseded_Raster')
r_cell_size = arcpy.GetParameter(3)
#r_cell_size = 500

# Project CATZOC data to WGS84 EASE 
arcpy.Project_management('%s'%(c), '%s'%(c_prj), arcpy.SpatialReference(3975))

# Extend existing CATZOC coverage to model bounds (shoreline-to-eex layer)
# 1. Generate null layer that includes all unsurveyed areas out to the EEZ and Repair Geometry (eliminate self-intersections)

d_field = "dissolveNum"
arcpy.AddField_management(c_prj, d_field, "double")
arcpy.CalculateField_management(c_prj, d_field, 1)
arcpy.RepairGeometry_management('%s'%(c_prj))
arcpy.Dissolve_management(c_prj, c_dissolve, d_field)
arcpy.Erase_analysis(sl_eez, c_dissolve, null)

# 2. Add catzoc field to null layer and denote catzoc equal to 6 (undefined).
c_field = "catzoc"
arcpy.AddField_management(null, c_field, "double")
arcpy.CalculateField_management(null, c_field, 6)

# 3. Merge non-surveyed areas denoted with catzoc = 6 with original catzoc layer, Dissolve by CATZOC value, and Clip output to model bounds
arcpy.Merge_management([c_prj, null], c_null_m)
arcpy.Dissolve_management(c_null_m, c_null_m_d, c_field)
arcpy.Clip_analysis(c_null_m_d, sl_eez, c_null)

# Delete temporary files
tempfiles = [c_prj, c_dissolve, null, c_null_m, c_null_m_d]
for tempfile in tempfiles:
    arcpy.Delete_management('%s'%(tempfile))

#Eliminate overlapping CATZOC layers. Replace overlapping areas with higher catzoc value.
# 1. Generate two layers for each catzoc (catzoc = value and catzoc < value). Erase all (catzoc < value) from (catzoc = value)
arcpy.MakeFeatureLayer_management('%s'%(c_null), temp)

chars = ["", "lt"]
for count in range (1,7):
    for op, name in (['=',""], ['<', "lt"]):
        arcpy.SelectLayerByAttribute_management('%s'%(temp), "NEW_SELECTION",' "catzoc" %s %d '%(op,count))
        arcpy.CopyFeatures_management('%s'%(temp), 'catzoc_%s%d'%(name,count))
        arcpy.SelectLayerByAttribute_management('%s'%(temp), "CLEAR_SELECTION")    
        if name == 'lt':
           arcpy.Erase_analysis('catzoc_%d'%(count), 'catzoc_lt%d'%(count), 'catzoc_erase_%d'%(count))
           arcpy.Dissolve_management('catzoc_erase_%d'%(count), 'catzoc_erase_dissolve_%d'%(count), c_field)
            catzoc_num = 'catzoc_lt%d'%(count)
            for char in chars:
                arcpy.Delete_management('catzoc_%s%d'%(char, count))
                arcpy.Delete_management('catzoc_%s%d'%(char, count))
                
# 2. Merge all erased layers (highest catzoc value layers)
 arcpy.Merge_management(["catzoc_erase_dissolve_1", "catzoc_erase_dissolve_2", "catzoc_erase_dissolve_3", "catzoc_erase_dissolve_4", "catzoc_erase_dissolve_5", "catzoc_erase_dissolve_6"], c_null_s)

# 3. Delete unnecessary layers 
for count in range (1,7):
    arcpy.Delete_management('catzoc_erase_%d'%(count))
    arcpy.Delete_management('catzoc_erase_dissolve_%d'%(count))

# Generate raster of final catzoc layer
arcpy.PolygonToRaster_conversion(c_null_s, c_field, c_null_final, "MAXIMUM_COMBINED_AREA", c_field, r_cell_size)                



 