# Storms

import os
import time

import arcpy
import arcgisscripting

"""
from HSTB.ArcExt.NHSP import Functions
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.storms import *

# Define Parameters
storms = globals.Parameters("Storms")
aux = globals.Parameters("AUX")
"""

##### User Defined Variable: #####
yr = '2019' # Storm year of interest
i_dataset2_fldr = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Best_Track_WindSwath\2008_2019_windswath" # Folder with windswath shapefiles
i_dataset1_file = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\All_storms\Storm_lines_all.shp" # Shapefile with storm track polylines
r_clip_file = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Auxiliary\sandy_lt40m" # Raster of areas that have sandy bottom and depths less than 40m
#eez_file = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Auxiliary\eez_1c" # Raster of EEZ
#output_prj = arcpy.SpatialReference(4326) # Projection of strm tracks
output_prj = arcpy.SpatialReference(102008) # Desired projection of output


# Workspace
##arcpy.env.workspace = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Working\Storms"

#--For Test--#
arcpy.env.workspace = i_dataset2_fldr
GDB_Location = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms"
GDB_name = "HH_update_2019.gdb"
# function to create the GDB
if os.path.exists(os.path.join(GDB_Location, GDB_name)):
    gdb = os.path.join(GDB_Location, GDB_name)
else:
    arcpy.CreateFileGDB_management(GDB_Location, GDB_name)
    gdb = os.path.join(GDB_Location, GDB_name)
    

#arcpy.env.workspace = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms\Test_output.gdb"
#------------#


#--For Test--#
c_windswath_merged = gdb + "\\c_windswath_merged"
f_query = "RADII"
f_query_value = "34"
c_windswath_selected = gdb + "\\c_windswath_selected"
f_start_year = "START_YEAR"
f_start_date = "STARTDTG"
f_start_year_calc_express = "!STARTDTG!.replace(' ', '')[0:4]"
f_end_year = "END_YEAR"
f_end_date = "ENDDTG"
f_end_year_calc_express = "!ENDDTG!.replace(' ', '')[0:4]"
c_windswath_selected_yr = gdb + "\\c_windswath_selected_yr"


list_dissolved = ["SID", "BASIN", "NATURE", "SEASON", "NUMBER"]
c_line_dissolved = gdb + "\\c_line_dissolved"
c_line_d_selected = gdb + "\\c_line_d_selected"

c_windswath_selected_p = gdb + "\\c_windswath_selected_p"
c_line_dissolved_p_b_c = gdb + "\\c_line_dissolved_p_b_c"
c_line_dissolved_p = gdb + "\\c_line_dissolved_p"

f_dataset1_storm_year_num = "s_yr_num"
f_dataset2_storm_year_num = "s_yr_num"

f_dataset1_storm_year_num_calc_exp = "str( !Season! ) + str( !Number! )"
f_dataset2_storm_year_num_calc_exp = "str( !START_YEAR! ) + str(!STORMNUM! ).replace('.0', '')"

c_line_d_p_dissolved = gdb + "\\c_line_d_p_dissolved"
c_windswath_s_p_dissolved = gdb + "\\c_windswath_s_p_dissolved"
q_line_d_p_dissolved = "Shape_Area Is NULL"

c_line_selected = gdb + "\\c_line_selected"
xy_tol = "0.001 Meters"

c_line_selected_buf = gdb + "\\c_line_selected_buf"
distance_field = "80467.2 meters"

c_merged = gdb + "\\c_merged"
c_clip = gdb + "\\c_clip"
f_clip_value = "VALUE"
c_envelope = gdb + "\\c_envelope"
c_envelope_buf = gdb + "\\c_envelope_buf"
c_e_distance_field = "125.5 Kilometers"
#c_merged_env_buf = gdb + "\\c_merged_env_buf"
#c_env_buf_dissolved = gdb + "\\c_env_buf_dissolved"

c_merged_clipped = gdb + "\\c_merged_clipped"
f_dataset_storm_year_num = "s_yr_num"
c_merged_clipped_dissolved = gdb + "\\Storms_Final"

f_storm_year = "storm_year"
f_storm_year_calc_express = "!s_yr_num!.replace(' ', '')[0:4]"

f_storm_num = "storm_num"
f_storm_num_calc_express = "!s_yr_num!.replace(' ', '')[4:]"
#------------#


# #=============================  Start  ====================================
# Create the basic raw data -- after this is where to cut in with "test data" if desired

# Load, merge, and select NHC_BestTrackTSWind data

#--For Test--#
swaths = []
walk =arcpy.da.Walk(i_dataset2_fldr, datatype="FeatureClass", type="Polygon")

for dirpath, dirnames, filenames in walk:
    for filename in filenames:
        swaths.append(os.path.join(dirpath, filename))

#arcpy.Merge_management(swaths, c_windswath_merged)

wild = "*windswath*"
ft = "POLYGON"
swath = arcpy.ListFeatureClasses(wild_card = wild, feature_type = ft)
arcpy.Merge_management(swath, c_windswath_merged)


#------------#

#Line from original script:
#merge_storm_swaths(i_dataset2_fldr, c_windswath_merged)

# Dissolve Basin.NA.ibtracs_all_lines data
t_start = time.time()
arcpy.Dissolve_management(i_dataset1_file, c_line_dissolved, list_dissolved, "", "MULTI_PART", "DISSOLVE_LINES")
print(("Dissolved line feature class at %.1f secs" % (time.time() - t_start)))

## Here script should filter out storms we don't care about ##
s_query = "NATURE"
#s_query_value1 = "'DS'"
#s_query_value2 = "'SS'"
#s_query_value3 = "'TS'"
s_query_value = "'%S'"
y_query = "SEASON"
y_query_value = yr
#y_100_value = str(int(yr) - 100)

try:
    t_start = time.time()
    #where_clause = "(" + s_query + " = " + s_query_value1 + " OR " + s_query + " = " + s_query_value2 + " OR " + s_query + " = " + s_query_value3 + ") " + " AND " + y_query + " = " + y_query_value
    #where_clause = s_query + " LIKE " + s_query_value + " AND " + y_query + " <= " + y_query_value + " AND " + y_query + " >= " + y_100_value
    #where_clause = s_query + " LIKE " + s_query_value + " AND " + y_query + " <= " + y_query_value
    where_clause = s_query + " LIKE " + s_query_value + " AND " + y_query + " = " + y_query_value
    arcpy.Select_analysis(c_line_dissolved, c_line_d_selected, where_clause)
    print(("Filtered out storm polylines at %.1f secs" % (time.time() - t_start)))
except arcgisscripting.ExecuteError:
    print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
    exit()

# #==============================   End   ===================================


try:
    t_start = time.time()
    where_clause = f_query + " >= " + f_query_value ## Why filter out swaths with radii less than 34?
    arcpy.Select_analysis(c_windswath_merged, c_windswath_selected, where_clause)
    print(("Filtered out polygons at %.1f secs" % (time.time() - t_start)))
except arcgisscripting.ExecuteError:
    print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
    exit()

# Add column to include storm year
if not arcpy.ListFields(c_windswath_selected, f_start_year):
    t_start = time.time()
    arcpy.AddField_management(c_windswath_selected, f_start_year, "TEXT", field_length=4)
    print(("Added field START_YEAR at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_windswath_selected, f_start_year, f_start_year_calc_express, "PYTHON_9.3")
print(("Calculated START_YEAR field values at %.1f secs" % (time.time() - t_start)))
if not arcpy.ListFields(c_windswath_selected, f_end_year):
    t_start = time.time()
    arcpy.AddField_management(c_windswath_selected, f_end_year, "TEXT", field_length=4)
    print(("Added field END_YEAR at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_windswath_selected, f_end_year, f_end_year_calc_express, "PYTHON_9.3")
print(("Calculated END_YEAR field values at %.1f secs" % (time.time() - t_start)))


# Filter out polygons by storm year (yr):
try:
    t_start = time.time()
    where_clause = f_start_year + " = '" + yr + "'"
    arcpy.Select_analysis(c_windswath_selected, c_windswath_selected_yr, where_clause)
    print(("Filtered out polygons by year at %.1f secs" % (time.time() - t_start)))
except arcgisscripting.ExecuteError:
    print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
    exit()


# Convert raster to polygon
t_start = time.time()
arcpy.RasterToPolygon_conversion(r_clip_file, c_clip, "NO_SIMPLIFY", f_clip_value)
print(("Converted raster to polygon at %.1f secs" % (time.time() - t_start)))

# Get an envelope area for all features
t_start = time.time()
arcpy.MinimumBoundingGeometry_management(c_clip, c_envelope, "ENVELOPE", "ALL")
print(("Created an envelope at %.1f secs" % (time.time() - t_start)))

# Expand the envelope with distance_field
t_start = time.time()
#arcpy.Buffer_analysis(c_envelope, c_envelope_buf, distance_field, "OUTSIDE_ONLY", "ROUND") # Should buffer with "FULL" option instead of "OUTSIDE_ONLY"
arcpy.Buffer_analysis(c_envelope, c_envelope_buf, distance_field, "FULL", "ROUND")
# Merge and Dissolve are unnecessary if buffered was "FULL" instead of "OUTSIDE_ONLY"
#arcpy.Merge_management([c_envelope, c_envelope_buf], c_merged_env_buf) 
#arcpy.Dissolve_management(c_merged_env_buf, c_env_buf_dissolved, "OBJECTID", "", "MULTI_PART", "DISSOLVE_LINES") 
print(("Expanded the envelope at %.1f secs" % (time.time() - t_start)))

# Re-projection
# output_prj should have the projection code of 102008 (North_America_Albers_Equal_Area_Conic)
t_start = time.time()
arcpy.Project_management(c_windswath_selected_yr, c_windswath_selected_p, output_prj) 
print(("Re-projected the selected polygon feature class at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.Project_management(c_line_d_selected, c_line_dissolved_p_b_c, output_prj) ## input file should be c_line_d_selected if the storm natures have been filtered out
print(("Re-projected the dissolved line feature class at %.1f secs" % (time.time() - t_start)))

# Extract Dataset 1 within the extent
t_start = time.time()
#arcpy.Clip_analysis(c_line_dissolved_p_b_c, c_env_buf_dissolved, c_line_dissolved_p, xy_tol)
arcpy.Clip_analysis(c_line_dissolved_p_b_c, c_envelope_buf, c_line_dissolved_p, xy_tol)
print(("Extracted Dataset 1 within the extent at %.1f secs" % (time.time() - t_start)))

# Add fields and calculate field values
# Create unique ID that the storm swath layer and the storm path line file can share(s_yr_num). A combo of the strom year and the number
if not arcpy.ListFields(c_line_dissolved_p, f_dataset1_storm_year_num):
    t_start = time.time()
    arcpy.AddField_management(c_line_dissolved_p, f_dataset1_storm_year_num, "TEXT", field_length=10)
    print(("Added s_yr_num field for Dataset 1 at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_line_dissolved_p, f_dataset1_storm_year_num, f_dataset1_storm_year_num_calc_exp, "PYTHON_9.3")
print(("Calculated field values at %.1f secs" % (time.time() - t_start)))
if not arcpy.ListFields(c_windswath_selected_p, f_dataset2_storm_year_num):
    t_start = time.time()
    arcpy.AddField_management(c_windswath_selected_p, f_dataset2_storm_year_num, "TEXT", field_length=10)
    print(("Added s_yr_num field for Dataset 2 at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_windswath_selected_p, f_dataset2_storm_year_num, f_dataset2_storm_year_num_calc_exp, "PYTHON_9.3")
print(("Calculated field values at %.1f secs" % (time.time() - t_start)))

# Dissolve by storm_year_num for Dataset 1
# Assuming we filtered out storms we don't want (when Nature <> DS and <> SS) then dissolving by storm_year_num is ok
t_start = time.time()
arcpy.Dissolve_management(c_line_dissolved_p, c_line_d_p_dissolved, [f_dataset1_storm_year_num], "", "MULTI_PART", "DISSOLVE_LINES")
print(("Dissolved by s_yr_num for Dataset 1 at %.1f secs" % (time.time() - t_start)))
# Dissolve by storm_year_num for Dataset 2
t_start = time.time()
arcpy.Dissolve_management(c_windswath_selected_p, c_windswath_s_p_dissolved, [f_dataset2_storm_year_num], "", "MULTI_PART", "DISSOLVE_LINES")
print(("Dissolved by s_yr_num for Dataset 2 at %.1f secs" % (time.time() - t_start)))

t_start = time.time()
arcpy.JoinField_management(c_line_d_p_dissolved, f_dataset1_storm_year_num, c_windswath_s_p_dissolved, f_dataset2_storm_year_num)
print(("Joined fields at %.1f secs" % (time.time() - t_start)))

t_start = time.time()
where_clause = q_line_d_p_dissolved
arcpy.Select_analysis(c_line_d_p_dissolved, c_line_selected, where_clause)
print(("Deleted any storms that exist in Dataset 2 from Dataset 1 at %.1f secs" % (time.time() - t_start)))

# Buffer remaining storms by average buffered distance from Dataset 2
t_start = time.time()
arcpy.Buffer_analysis(c_line_selected, c_line_selected_buf, distance_field, "FULL", "ROUND") # Distance_field should be "80467 meters"
print(("Buffered remaining storms by average buffered distance from Dataset 2 at %.1f secs" % (time.time() - t_start)))

# Merge final Datasets 1 and 2
t_start = time.time()
arcpy.Merge_management([c_line_selected_buf, c_windswath_s_p_dissolved], c_merged)
print(("Merged final Datasets 1 and 2 at %.1f secs" % (time.time() - t_start)))

# Clip the merged dataset by Sandy LT 40 m
t_start = time.time()
arcpy.Clip_analysis(c_merged, c_clip, c_merged_clipped, xy_tol)
print(("Clipped the merged dataset by Sandy LT 40m at %.1f secs" % (time.time() - t_start)))

# Dissolve the clipped dataset
t_start = time.time()
arcpy.Dissolve_management(c_merged_clipped, c_merged_clipped_dissolved, [f_dataset_storm_year_num], "", "MULTI_PART", "DISSOLVE_LINES")
print(("Dissolved the clipped dataset at %.1f secs" % (time.time() - t_start)))

# Add field storm_year
if not arcpy.ListFields(c_merged_clipped_dissolved, f_storm_year):
    t_start = time.time()
    arcpy.AddField_management(c_merged_clipped_dissolved, f_storm_year, "TEXT", field_length=4)
    print(("Added field storm_year at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_merged_clipped_dissolved, f_storm_year, f_storm_year_calc_express, "PYTHON_9.3")
print(("Calculated storm_year field values at %.1f secs" % (time.time() - t_start)))

# Add field storm_num
if not arcpy.ListFields(c_merged_clipped_dissolved, f_storm_num):
    t_start = time.time()
    arcpy.AddField_management(c_merged_clipped_dissolved, f_storm_num, "TEXT", field_length=4)
    print(("Added field storm_num at %.1f secs" % (time.time() - t_start)))
t_start = time.time()
arcpy.CalculateField_management(c_merged_clipped_dissolved, f_storm_num, f_storm_num_calc_express, "PYTHON_9.3")
print(("Calculated storm_num field values at %.1f secs" % (time.time() - t_start)))

#===============================================================================
# # Clean temp gdb
# t_start = time.time()
# arcpy.env.workspace = temp_gdb
# print("Cleaning temp gdb")
# for fds in arcpy.ListDatasets('', 'feature') + ['']:
#     for fc in arcpy.ListFeatureClasses('', '', fds):
#         print("Deleting feature class " + str(fc))
#         arcpy.Delete_management(fc)
# print("Finished cleaning temp gdb at %.1f secs" % (time.time() - t_start))
#===============================================================================


# # Storm Raw to Vector for 2016 Run of Model#
# # User-defined Variables
# output_gdb = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\STORMS\Storms.gdb"
# s_raw = "Storms_RAW"
# #s_raw = "Storms_RAW_Test"
# des_sprj_name = "North_America_Albers_Equal_Area_Conic"
# des_sprj_code = 102008
#
# # Static Variables
# s_select = s_raw + "_select"
# s_dis = os.path.join(output_gdb, s_raw + "_dissolve")
# s_buff = os.path.join(output_gdb, s_raw + "_buffer")
# query = "Nature <> ' DS' AND Nature <> ' SS'"
# dis_fields = ["Serial_Num", "Basin", "Nature", "year"]
# buff_dist = "80467.2 Meters"
#
# # Project data, if necessary
# Functions.check_spatial_reference(output_gdb, s_raw, des_sprj_name, des_sprj_code)
#
# # Filter data to only include storms of nature ET, MX, NR, TS
# if arcpy.Exists(os.path.join(output_gdb, s_raw + "_" + des_sprj_name)):
#     Functions.Make_Selection(output_gdb, s_raw + "_" + des_sprj_name, query, s_select)
# else:
#     Functions.Make_Selection(output_gdb, s_raw, query, s_select)
#     print("NOPE!")
#
# # Dissolve storm data by specified dissolve fields (combines storm tracks)
# arcpy.Dissolve_management(s_select, s_dis, dis_fields)
#
# # Buffer data by 100 miles (80467 m). Diameter of hurricane is ~ 100 miles.
# arcpy.Buffer_analysis(s_dis, s_buff, buff_dist)
#
# # Delete buffer, dissolve, select, project once clips complete...
#
#
# # Must clip to sandy, less than 20 m water depths'''
# ===============================================================================
