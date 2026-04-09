# Shoreline-to-EEZ
# This script completes the following:
    # Reprojects the user-selected shoreline and EEZ files and then generates a shoreline to EEZ file named USA_Shoreline_to_EEZ

import arcpy, os

# Define Inputs, Outputs and Workspace
sl_in = arcpy.GetParameter(0)
#arcpy.AddMessage(type(arcpy.GetParameter(0)))
eez_in = arcpy.GetParameter(1)
output_ws = str(arcpy.GetParameter(2))
sl_prj = os.path.join(ouput_ws, 'USA_Shoreline_WGS84EASE')
eez_prj = os.path.join(ouput_ws, 'USA_EEZ_WGS84EASE')
sl_eez = os.path.join(ouput_ws, 'USA_Shoreline_to_EEZ')

# Project Shoreline and EEZ data to WGS84 EASE
for input, output in zip([sl_in, eez_in], [sl_prj, eez_prj]):
    out_coord_sys = arcpy.SpatialReference(3975)
    arcpy.Project_management('%s'%(input), '%s'%(output), out_coord_sys)

# Genearate Shoreline to EEZ data layer
arcpy.Erase_analysis(eez_prj, sl_prj, sl_eez)

# Delete projected Shoreline and EEZ files from geodatabase
tempfiles = [sl_prj, eez_prj]
for tempfile in tempfiles:
    arcpy.Delete_management(tempfile)