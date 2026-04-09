# This script counts the number of storms since the last survey and the number of storms over the last 100 years.
# Storm Line File = Best track from NHC. Will be buffered by 67 nautical mile radius, the average radius of buffered dataset.
# Buffered Line File = Buffered best track from NHC representing maximum extent of tropical storm force winds


# Import Modules
import os
import time
import arcpy
import arcgisscripting
import pickle
from arcpy import env
from arcpy.sa import *
import numpy
import time

from . import gridclass 
from . import Functions

"""
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values, timer
storms = Parameters("Storms")
aux = Parameters("AUX")
qual = Parameters("SurveyQuality")
"""

## timer class from base.py:
class timer(object):
    def __init__(self, desc="Running time:", start_msg=""):
        self.lm = self.ts = time.time()
        self.desc = desc
        self.start_msg = start_msg

    def __enter__(self):
        if self.start_msg:
            print((self.start_msg))
        return self

    def __exit__(self, type, value, traceback):
        self.msg(self.desc)
        return False

    def msg(self, txt):
        print((txt, " %.2fsec (%.2f total)" % (time.time() - self.lm, time.time() - self.ts)))
        self.lm = time.time()
        
################################


# import importlib
# import HSTB.ArcExt.NHSP.globals
# importlib.reload(HSTB.ArcExt.NHSP.globals); Parameters = HSTB.ArcExt.NHSP.globals.Parameters; initial_values = HSTB.ArcExt.NHSP.globals.initial_values

# Check out extension licenses
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

"""
# User-Specified Variables
storms_vp_file = storms.vector_processed_filename()
storm_year_field = "Year"
storm_id_field = "storm_year_num"

# Input Variables
grid_ws = initial_values["Input"]["grid_ws"]
s_year = qual.raster_filename("yr")
sandy_lt40m = aux["sandy_lt40m_ras"]
pro_prj_gdb = storms.working_gdb

# Output Variables
storms_vp_file_prj = storms.working_filename("prj")

buffer_dist = 124084

storms_r = storms.raster_filename()
storms_100_r = storms.raster_filename("100")
storms_rc = storms.raster_classified_filename()
storms_100_rc = storms.raster_classified_filename("100")
storms_f = storms.raster_final_filename()
storms_100_f = storms.raster_final_filename("100")


# Check Spatial Reference of Vector Files
storms_vp_prj = Functions.check_spatial_reference(storms_vp_file, storms_vp_file_prj, storms.projection_name, storms.projection_number)

# Create Numpy Array of Year Data
year = gridclass.datagrid.FromRaster(s_year, nodata_to_value=0)

grid_b_fns = []  # Delete
storms_vp_gridclip_fns = []  # Delete
storms_since_survey_fns = []  # Merge, then Delete
storms_past_100yrs_fns = []  # Merge, then Delete

#################

tt = time.time()
start_grid = 11
end_grid = 14
start_cnt = 0
"""
####### User Input ######
since100_yr = 1918 # Year for storm count from past 100 years

# GDB with the Storms vector proecessed shapefile:
vp_gdb_location = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms"
vp_gdb_name = "HH_update_2018.gdb"
vp_gdb = os.path.join(vp_gdb_location, vp_gdb_name)
#vp_gdb = arcpy.CreateFileGDB_management(vp_gdb_location, vp_gdb_name)

# Create GDB for the raster outputs for the storm count for past 100 years and since last survey for each grid extent
storms_gdb_location = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms"
storms_gdb_name = "storms.gdb"
if os.path.exists(os.path.join(storms_gdb_location, storms_gdb_name)):
    storms_gdb = os.path.join(storms_gdb_location, storms_gdb_name)
else:
    arcpy.CreateFileGDB_management(storms_gdb_location, storms_gdb_name)
    storms_gdb = os.path.join(storms_gdb_location, storms_gdb_name)

# Create GDB for the temporary files
temp_gdb_location = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms"
temp_gdb_name = "temp.gdb"
if os.path.exists(os.path.join(temp_gdb_location, temp_gdb_name)):
    temp_gdb = os.path.join(temp_gdb_location, temp_gdb_name)
else:
    arcpy.CreateFileGDB_management(temp_gdb_location, temp_gdb_name)
    temp_gdb = os.path.join(temp_gdb_location, temp_gdb_name)

#Create GDB for final raster outputs of the mosaiced and reclassified storms for 100 years and since last survey
raster_final_gdb_location = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\Storms"
raster_final_gdb_name = "raster_final.gdb"
if os.path.exists(os.path.join(raster_final_gdb_location, raster_final_gdb_name)):
    raster_final_gdb = os.path.join(raster_final_gdb_location, raster_final_gdb_name)
else:
    arcpy.CreateFileGDB_management(raster_final_gdb_location, raster_final_gdb_name)
    raster_final_gdb = os.path.join(raster_final_gdb_location, raster_final_gdb_name)

# User-Specified Variables
##storms_vp_file = storms.vector_processed_filename()
storms_vp_prj = os.path.join(vp_gdb, "Storms_Final")
storm_year_field = "storm_year"
storm_id_field = "s_yr_num"

# Input Variables
arcpy.env.workspace = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\GRID\Grid.gdb" # GDB with grids

grids = os.path.join(r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\GRID", "Grid.gdb") #GDB with grids
s_year = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\qual_yr_2017" # Survey year raster
sandy_lt40m = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Auxiliary\sandy_lt40m" # Raster of areas that have sandy bottom and depths less than 40m
pro_prj_gdb = temp_gdb 


# Output Variables
#storms_vp_file_prj = storms.working_filename("prj")

buffer_dist = 124084

storms_r = os.path.join(storms_gdb, "storms_s") # storms since last survey
storms_100_r = os.path.join(storms_gdb, "storms_c") # storms from past 100 years
storms_rc = os.path.join(raster_final_gdb, "rc_storms_svy") # Raster with classified storms since last survey
storms_100_rc = os.path.join(raster_final_gdb, "rc_storms_100") # Raster with classified storms from past 100 years
storms_f = os.path.join(raster_final_gdb, "rf_storms_svy") # Final raster stroms since last survey 
storms_100_f = os.path.join(raster_final_gdb, "rf_storms_100") # Final raster storms from past 100 years

# Desired projection of output:
out_prj = arcpy.SpatialReference(102008) #North_America_Albers_Equal_Area_Conic 
# Desired cell size of output:
out_cell_size = 500 #500 meters

# Create Numpy Array of Year Data
year = gridclass.datagrid.FromRaster(s_year, nodata_to_value=0)

grid_b_fns = []  # Delete
storms_vp_gridclip_fns = []  # Delete
storms_since_survey_fns = []  # Merge, then Delete
storms_past_100yrs_fns = []  # Merge, then Delete

tt = time.time()
start_grid = 10
end_grid = 14
start_cnt = 0



restart = False  # change to restart = true if crashes and specify the file to use in the line below
restart_file = "C:\\Users\\Hilina.Tarekegn\\Documents\\HydroHealth\\Python_Files\\Storms\\Storms_Archive\\intermediate_dataXXXX.pickle"
if restart:
    picklefile = open(restart_file, "rb")
    start_grid = pickle.load(picklefile)
    successful_storms = pickle.load(picklefile)
    grid = pickle.load(picklefile)
    grid100 = pickle.load(picklefile)


for igrid in range(start_grid, end_grid):
    # if datetime.datetime(2016,12,8,22)<datetime.datetime.now():break
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grids, grid_name)
    model_bounds = os.path.join(grids, "Model_Extents_%02d" % igrid)

    #storms_line_gridclip = storms.working_filename("Stormsc_%02d" % igrid)
    #storms_vp_gridclip = storms.working_filename("%02d_t" % igrid)
    #storms_vp = storms.vector_processed_filename("%02d" % igrid)  # Merge and save, but delete intermediate files
    storms_line_gridclip = os.path.join(storms_gdb, "Stormsc_%02d" % igrid)
    storms_vp_gridclip = os.path.join(temp_gdb, "g_%02d_t" % igrid)
    storms_vp = os.path.join(vp_gdb, "%02d" % igrid)  # Merge and save, but delete intermediate files

    # Output Files
    #s_survey_ras = storms.working_filename("s_%02d" % igrid, gdb=False)
    #s_100_ras = storms.working_filename("c_%02d" % igrid, gdb=False)
    s_survey_ras = os.path.join(temp_gdb, "s_%02d" % igrid)
    s_100_ras = os.path.join(temp_gdb,"c_%02d" % igrid)


    # Buffer Grid Model Bounds to be used to clip line storm features
    #grid_b = storms.working_filename("Grid_%02d_Buff" % igrid)
    #arcpy.Buffer_analysis(model_bounds, grid_b, str(buffer_dist) + " meters")
    #print("Buffered  Grid %02d" % igrid)
    grid_b = os.path.join(temp_gdb,"Grid_%02d_Buff" % igrid)
    arcpy.Buffer_analysis(model_bounds, grid_b, str(buffer_dist) + " meters")
    print(("Buffered  Grid %02d" % igrid))
    

    # Clip line storm features and polygon storm features by buffered grid
    arcpy.Clip_analysis(storms_vp_prj, grid_b, storms_vp_gridclip)

    # Read Grid and Extract XY Coordinates from point known features File
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)

    grid.gridarray *= 0
    grid100 = grid.clip(grid)  # Copying Grid for 100 year storms

    s_cnt = 0
    successful_storms = []
    if arcpy.Exists(storms_vp_gridclip):
        storm_count_total = arcpy.GetCount_management(storms_vp_gridclip)
        print(("There are {} storms in grid {}".format(storm_count_total, igrid)))
        # Count number of storms over last 100 years and number of storms since last survey
        # Make Feature Layer
        try:
            input_temp = "temp"
            arcpy.MakeFeatureLayer_management(storms_vp_gridclip, input_temp)
        except arcgisscripting.ExecuteError:
            print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
        except RuntimeError:
            print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")

        for storm, storm_yr, sn in arcpy.da.SearchCursor(storms_vp_gridclip, ["SHAPE@", storm_year_field, storm_id_field]):
            # if datetime.datetime(2016,12,8,22)<datetime.datetime.now(): break
            if s_cnt in successful_storms:
                s_cnt += 1
                continue  # skip this storm (goes back to the for loop above)
            print(("Storm %d" % (s_cnt + 1)))
            print(("Storm {} of {}".format((s_cnt + 1), storm_count_total)))
            t = time.time()

            # Create raster of single storm
            query = "%s = '%s'" % (storm_id_field, sn)

            out_r = os.path.join(pro_prj_gdb, "in_memory")
            with timer("Select Storm and Convert to Raster") as ts:
                arcpy.SelectLayerByAttribute_management(input_temp, "NEW_SELECTION", query)
                try:
                    arcpy.PolygonToRaster_conversion(input_temp, storm_id_field, out_r, "", "", out_cell_size)

                    # Convert to Numpy Array
                    with timer("Convert Storms and Year to Numpy Array") as ts:
                        s1 = gridclass.datagrid.FromRaster(out_r, nodata_to_value=0)
    #                     s1 = gridclass.datagrid.FromRaster(out_r, nodata_to_value=None)
                        new_year = year.clip(s1)  # Clip year array by storm extents
                        final_year = new_year.clip(grid)  # Clip storm extents year array by grid (only get cells within grid)
                        del new_year
                        final_storm = s1.clip(final_year)
                        final_storm.gridarray[final_storm.gridarray == 255] = 0  # turn the 255 (empty cells from Arc) into zeros
                        final_storm.gridarray = numpy.array(final_storm.gridarray, numpy.int32)  # Change numpy array to integer
                        final_storm.gridarray *= int(storm_yr)  # now a matrix of zero or storm year

                    with timer("Create 100 Year Storms array") as ts:
                        # Insert storm into grid if more recent than last survey
                        # create a grid of if more recent than 100 years ago

                        if int(storm_yr) > int(since100_yr):
                            # print "storm year IS >1916", storm_yr
                            s_100 = final_storm.gridarray > 0  # boolean matrix which when added to another array will increment an integer or float value
                            # Insert storm into grid
                            xo, yo = grid100.ArrayIndicesFromXY([final_storm.minx, final_storm.miny])
                            xc, yc = s_100.shape
                            if xo + xc > grid100.gridarray.shape[0]:
                                xc -= 1
                            if yo + yc > grid100.gridarray.shape[1]:
                                yc -= 1
                            grid100.gridarray[xo:xo + xc, yo:yo + yc] += s_100[:xc, :yc]

                    # Calculate Number of Storms since survey
                    # Storm and year arrays are in the same geographic area, but may be slightly different shapes since the origins
                    # of the different input rasters may not be exactly equal. This comparison may be in error by one grid cell in
                    # x, y and the shapes may be different as well. Take overlap area in row and column space.
                    with timer("Create Storms Since Survey array") as ts:
                        max_r, max_c = numpy.min([final_storm.gridarray.shape, final_year.gridarray.shape], 0)
                        s_newer = final_storm.gridarray[:max_r, :max_c] > final_year.gridarray[:max_r, :max_c]  # boolean matrix which when added to another array will increment an integer or float value

                        # Insert storm into grid
                        xo, yo = grid.ArrayIndicesFromXY([final_storm.minx, final_storm.miny])
                        xc, yc = s_newer.shape
                        if xo + xc > grid.gridarray.shape[0]:
                            xc -= 1
                        if yo + yc > grid.gridarray.shape[1]:
                            yc -= 1
                        grid.gridarray[xo:xo + xc, yo:yo + yc] += s_newer[:xc, :yc]

                    # Clear Selection and Delete Temporary Storm Raster
                    with timer("Clear Select and Delete temp layer") as ts:
                        arcpy.SelectLayerByAttribute_management(input_temp, "CLEAR_SELECTION")
                        if arcpy.Exists(out_r):
                            arcpy.Delete_management(out_r)
                        if arcpy.Exists(os.path.join(pro_prj_gdb, os.path.split(out_r)[1])):
                            arcpy.Delete_management(os.path.join(pro_prj_gdb, os.path.split(out_r)[1]))
                        if arcpy.Exists(os.path.join(pro_prj_gdb, os.path.split(out_r)[1])):
                            arcpy.Delete_management(os.path.join(pro_prj_gdb, os.path.split(out_r)[1]))

                    successful_storms.append(s_cnt)
                    print(("Time for Storm = %.1f secs" % (time.time() - t)))
                    print(("Total Time for %d storms = " % (s_cnt + 1), time.time() - tt))

                except RuntimeError:
                    if arcpy.Exists(os.path.join(pro_prj_gdb, "ras_out")):
                        arcpy.Delete_management(os.path.join(pro_prj_gdb, "ras_out"))
                    if arcpy.Exists(os.path.join(pro_prj_gdb, os.path.split(out_r)[1])):
                        arcpy.Delete_management(os.path.join(pro_prj_gdb, os.path.split(out_r)[1]))
                    if arcpy.Exists(os.path.join(pro_prj_gdb, os.path.split(out_r)[1])):
                        arcpy.Delete_management(os.path.join(pro_prj_gdb, os.path.split(out_r)[1]))
                    print(("Serial Number = %s" % (sn)))
                    print(("Year = %s" % (storm_yr)))
                    print("COULD NOT CONVERT TO RASTER")
                    print("COULD NOT CONVERT TO RASTER")
                    print("COULD NOT CONVERT TO RASTER")
                    if arcpy.Exists(os.path.join(pro_prj_gdb, os.path.split(out_r)[1])):
                        print("in_memory file still exists")
            s_cnt += 1

            if s_cnt % 25 == 0:
                picklefile = open("C:\\Users\\Hilina.Tarekegn\\Documents\\HydroHealth\\Python_Files\\Storms\\Working\\Storms_intermediate_data%02d%04d.pickle" % (igrid, s_cnt), "wb")
    #             picklefile = open("C:\\Users\\Christina.Fandel\\Fandel\\Storms_Archive\\intermediate_data%02d%04d.pickle" % (igrid, s_cnt), "wb")
                pickle.dump(igrid, picklefile, 2)
                pickle.dump(successful_storms, picklefile, 2)
                pickle.dump(grid, picklefile, 2)
                pickle.dump(grid100, picklefile, 2)
    #    Export Storms Since Survey and Storms Over 100 Years Rasters
        with timer("Export Rasters") as ts:
            try:
                grid.ExportRaster(s_survey_ras)
                grid100.ExportRaster(s_100_ras)
                print(("Finished Max Raster Classify Export at %.1f secs" % (time.time() - t_start)))
            except RuntimeError:
                print("Export Reclassified Maximum Distance and Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

        # Create List Exported Rasters to be Combined and Deleted later
        storms_since_survey_fns.append(s_survey_ras)
        storms_past_100yrs_fns.append(s_100_ras)

        # Create List of Exported Rasters to be deleted
        grid_b_fns.append(grid_b)
    else:
        print(("No storms within grid {}, moving on to grid {}".format(igrid, igrid + 1)))

#####################
# Combine Distance, Density, and Max Rasters
t_start = time.time()
arcpy.MosaicToNewRaster_management(storms_since_survey_fns, os.path.dirname(storms_r), os.path.basename(storms_r), out_prj, "32_BIT_SIGNED", out_cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(storms_past_100yrs_fns, os.path.dirname(storms_100_r), os.path.basename(storms_100_r), out_prj, "32_BIT_SIGNED", out_cell_size, "1", "MAXIMUM", "FIRST")
#arcpy.Merge_management(storms_vp_fns, storms_vp_final)
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))
'''
# UPDATE ME
# Reclassify Final Rasters
t_start = time.time()
s_survey_max = (arcpy.sa.Raster(storms_r)).maximum + 21  # Maximum value begins at 20
s_100_max = (arcpy.sa.Raster(storms_100_r)).maximum + 51  # Maximum value begins at 50

s_survey_rc = arcpy.sa.Reclassify(storms_r, "VALUE", "0 1;1 4 2;5 10 3;11 20 4;21 %s 5;NODATA 0" % (s_survey_max), "NODATA")
s_survey_rc.save(storms_rc)
s_100_rc = arcpy.sa.Reclassify(storms_100_r, "VALUE", "0 5 1;6 10 2;11 25 3;26 50 4;51 %s 5;NODATA 0" % (s_100_max), "NODATA")
s_100_rc.save(storms_100_rc)
print("Finished Reclassify of final rasters at %.1f secs" % (time.time() - t_start))


# Clip Final Raster to Sandy and Depth < 40 m Bounds
t_start = time.time()
s_survey_clips = arcpy.sa.ExtractByMask(storms_rc, sandy_lt40m)
s_survey_clips.save(storms_f)

s_100_clips = arcpy.sa.ExtractByMask(storms_100_rc, sandy_lt40m)
s_100_clips.save(storms_100_f)
print("Finished Clip of final raster to Sandy and Depth < 40 at %.1f secs" % (time.time() - t_start))
'''

#################################################
# Because running through all extend grids at once can take too long and the script may crash
# you must rerun the script and change the start_grid and end_grid to do three to four grids at a time.
# This means that reclassifying and clipping the final raster for both storms since last survey and the
# storms since 100 years ago must be run after you finishing rerunning the script for all the grids.
# This means that you must comment out the following lines when rerunning the script. Each time you finish a run
# of the script, you must rename the output files in the final raster gdb output before rerunning so you don't write over
# them. Then once all grids are processed, you can un-comment the following lines, comment out the rest of the script, and
# run the follwing lines. This should mosaic all of the output grid storm count rasters to a new raster, then reclassify the counts,
# and clip the rasters to the sandy less than 40m boundary. 
arcpy.env.workspace = os.path.dirname(storms_r)
storms_output_list = []
walk =arcpy.da.Walk(os.path.dirname(storms_r), datatype="RasterDataset")

#path = os.path.dirname(storms_r)
for dirpath, dirnames, filenames in walk:
    for filename in filenames:
        storms_output_list.append(os.path.join(dirpath, filename))


wild = "storms_s"
#ft = "ALL"
#storms_since_survey = arcpy.ListFeatureClasses(wild_card = wild)
storms_since_survey = [i for i in storms_output_list if wild in i] 

wild = "storms_c"
#ft = "ALL"
#storms_past_100yrs = arcpy.ListFeatureClasses(wild_card = wild)
storms_past_100yrs = [i for i in storms_output_list if wild in i]

t_start = time.time()
arcpy.MosaicToNewRaster_management(storms_since_survey, os.path.dirname(storms_r), os.path.basename(storms_r), out_prj, "32_BIT_SIGNED", out_cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(storms_past_100yrs, os.path.dirname(storms_100_r), os.path.basename(storms_100_r), out_prj, "32_BIT_SIGNED", out_cell_size, "1", "MAXIMUM", "FIRST")
#arcpy.Merge_management(storms_vp_fns, storms_vp_final)
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))

# Reclassify Final Rasters
t_start = time.time()
s_survey_max = (arcpy.sa.Raster(storms_r)).maximum + 21  # Maximum value begins at 20
s_100_max = (arcpy.sa.Raster(storms_100_r)).maximum + 51  # Maximum value begins at 50

s_survey_rc = arcpy.sa.Reclassify(storms_r, "VALUE", "0 1;1 4 2;5 10 3;11 20 4;21 %s 5;NODATA 0" % (s_survey_max), "NODATA")
s_survey_rc.save(storms_rc)
s_100_rc = arcpy.sa.Reclassify(storms_100_r, "VALUE", "0 5 1;6 10 2;11 25 3;26 50 4;51 %s 5;NODATA 0" % (s_100_max), "NODATA")
s_100_rc.save(storms_100_rc)
print(("Finished Reclassify of final rasters at %.1f secs" % (time.time() - t_start)))


# Clip Final Raster to Sandy and Depth < 40 m Bounds
t_start = time.time()
s_survey_clips = arcpy.sa.ExtractByMask(storms_rc, sandy_lt40m)
s_survey_clips.save(storms_f)

s_100_clips = arcpy.sa.ExtractByMask(storms_100_rc, sandy_lt40m)
s_100_clips.save(storms_100_f)
print(("Finished Clip of final raster to Sandy and Depth < 40 at %.1f secs" % (time.time() - t_start)))
#################################################



# Delete Buffered Grid Model Bounds: Make list of these variables and then loop through to delete
# l = [grid_b, grid_br, s_clip, s_prj, s_select, s_survey_finalt, s_100_finalt] # remove grid_br

# for ll in l:
#     arcpy.Delete_management(ll)


# Timing for Storms (2018)
# Grid 1: 5 min
# Grid 2: 363 min (6.05 hrs)
# Grid 3: 510 min (8.5 hrs)
# Grid 4: 90 min
# Grid 5:
# Grid 6:
# Grid 7:
# Grid 8:
# Grid 9:
# Grid 10:
# Grid 11:
# Grid 12:
# Grid 13:
