# This script clips all rasters within a specified geodatabase or folder using an
# input polygon or raster (e.g. project/survey bounds) and exports a statistics
# table for each raster and a clipped raster for each survey/project within the
# input polygon or raster

# Import Modules
import os
import arcpy
from arcpy import sa
from HSTB.ArcExt.NHSP import Functions

# Import Variables
input_gdb = r"C:\Users\Christina.Fandel\Desktop\HydroHealth\Testing\Testing_Input.gdb"
arcpy.env.workspace = input_gdb
output_gdb = r"C:\Users\Christina.Fandel\Desktop\HydroHealth\Testing\Testing.gdb"
h_health = os.path.join(input_gdb, "Hydro_Health_Remap_Int_Final")

#h_risk = os.path.join(input_gdb, "Hydro_Risk_Remap_Class_Final")
#rasters = [h_health, h_risk]

rasters = [h_health]

#surveys = os.path.join(input_gdb, "Survey_Sample")
surveys = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\NHSP1\NHSP_Updated_Final.gdb\NHSP_2012_Final_FBC_C_Reindeer"
# field_id = "HNumber" # If multiple project or survey inputs, define unique identifier
field_id = "Priority"  # If multiple project or survey inputs, define unique identifier
des_sprj_name = "North_America_Albers_Equal_Area_Conic"  # USER DEFINED
output_prj = 102008  # USER DEFINED

# Extract all rasters within specified workspace. THIS WILL ONLY WORK IF
# ALL RASTERS HAVE AN ATTRIBUTE TABLE
raster_fns = arcpy.ListRasters("*Final", "ALL")  # CHANGE TO GET RID OF FINAL
rasters2 = []
for fn in raster_fns:
    rasters2.append(os.path.join(input_gdb, fn))

r_fns = []
# Check spatial reference of input shapefile and Project, if necessary
Functions.check_spatial_reference(input_gdb, surveys, des_sprj_name, output_prj)
if arcpy.Exists(os.path.join(input_gdb, surveys + "_" + des_sprj_name)):
    surveys_prj = os.path.join(input_gdb, surveys + "_" + des_sprj_name)
else:
    surveys_prj = surveys  # Data was not projected, already in desired projection

# Read individual surveys within survey file
surveys_prj_id = [row for row in arcpy.da.SearchCursor(surveys_prj, ["SHAPE@", field_id, ])]

# Define area fields and expressions to be added to raster
field_km2 = "Area_KM2"
field_km2_exp = "!Count!*.25"
field_snm = "Area_SNM"
field_snm_exp = "!Area_KM2!*0.291553"


#    try:
#        arcpy.SelectLayerByAttribute_management(temp, "NEW_SELECTION", "catobs IS NULL OR catobs = '                         ' OR catobs <> '8                        ' AND catobs <> '10                       ' AND catobs <> '5                        '")
#        print("Finished Select Layer by Attribute at %.1f secs"%(time.time()-t))
#    except arcgisscripting.ExecuteError:

#    try:
#        grid.ExportMatchingRaster(nh_array_sum, dens_c_out)
#        print("Finished Classified Summed Feature Export at %.1f secs"%(time.time()-t_start))
#    except RuntimeError:
#        print("Export Reclassified Sum Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

# Create individual project/survey rasters for each input raster. Calculate area (km2; snm)
# of each classified category
rasters = [gap, decay, desire]
for raster in rasters:
    # Verify raster values do not contains fraction values. Raster attribute tables do not support fractions.
    # If raster is comprised of integer values, but in a 32 bit/64 bit fraction form, will copy raster to 32 bit signed raster form
    # If raster is comprised of fraction values, a message will alert the user that the file has been skipped and needs to be manually reclassified
    raster_type = arcpy.GetRasterProperties_management(raster, "VALUETYPE")
    raster_max = arcpy.GetRasterProperties_management(raster, "MAXIMUM")
    if int(raster_type[0]) < 9:
        pass
    else:
        if float(raster_max[0]).is_integer():
            print("Copying Raster")
            raster_copy = raster + "_Copy32bitSigned"
            arcpy.CopyRaster_management(raster, raster_copy, "", "", "", "", "", "32_BIT_SIGNED")
            raster = raster_copy
        else:
            print("Raster contains fractions; must reclassify")
            continue
    for survey in surveys_prj_id:
        arcpy.BuildRasterAttributeTable_management(raster)
        r = arcpy.sa.ExtractByMask(raster, survey[0])
        for field, exp in ([field_km2, field_km2_exp], [field_snm, field_snm_exp]):
            arcpy.AddField_management(r, field, "DOUBLE")
            arcpy.CalculateField_management(r, field, exp, "PYTHON_9.3")

        r_out = os.path.join(output_gdb, "P%s" % survey[1] + "_" + os.path.split(raster)[1])
        r.save(r_out)

        r_fns.append(r_out)
    r_table_path = os.path.join(output_gdb, os.path.split(raster)[1] + "_" + os.path.split(surveys)[1] + "_table")
    r_table = arcpy.sa.ZonalStatisticsAsTable(surveys, field_id, raster, r_table_path, "DATA", "ALL")
