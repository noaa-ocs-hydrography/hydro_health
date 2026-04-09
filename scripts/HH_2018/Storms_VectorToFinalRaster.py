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
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values, timer
storms = Parameters("Storms")
aux = Parameters("AUX")
qual = Parameters("SurveyQuality")

# import importlib
# import HSTB.ArcExt.NHSP.globals
# importlib.reload(HSTB.ArcExt.NHSP.globals); Parameters = HSTB.ArcExt.NHSP.globals.Parameters; initial_values = HSTB.ArcExt.NHSP.globals.initial_values

# Check out extension licenses
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

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

restart = False  # change to restart = true if crashes and specify the file to use in the line below
restart_file = "C:\\Users\\Christina.Fandel\\Fandel\\Storms_Archive\\intermediate_dataXXXX.pickle"
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
    grid_path = os.path.join(grid_ws, grid_name)
    model_bounds = os.path.join(grid_ws, "Model_Extents_%02d" % igrid)

    storms_line_gridclip = storms.working_filename("Stormsc_%02d" % igrid)
    storms_vp_gridclip = storms.working_filename("%02d_t" % igrid)
    storms_vp = storms.vector_processed_filename("%02d" % igrid)  # Merge and save, but delete intermediate files

    # Output Files
    s_survey_ras = storms.working_filename("s_%02d" % igrid, gdb=False)
    s_100_ras = storms.working_filename("c_%02d" % igrid, gdb=False)

    # Buffer Grid Model Bounds to be used to clip line storm features
    grid_b = storms.working_filename("Grid_%02d_Buff" % igrid)
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

        for storm, storm_yr, sn in arcpy.da.SearchCursor(storms_vp_gridclip, ["SHAPE@", storm_year_field, storm_id_field, ]):
            # if datetime.datetime(2016,12,8,22)<datetime.datetime.now(): break
            if s_cnt in successful_storms:
                s_cnt += 1
                continue  # skip this storm (goes back to the for loop above)
            print(("Storm %d" % (s_cnt)))
            print(("Storm {} of {}".format(s_cnt, storm_count_total)))
            t = time.time()

            # Create raster of single storm
            query = "%s = '%s'" % (storm_id_field, sn)

            out_r = os.path.join(pro_prj_gdb, "in_memory")
            with timer("Select Storm and Convert to Raster") as ts:
                arcpy.SelectLayerByAttribute_management(input_temp, "NEW_SELECTION", query)
                try:
                    arcpy.PolygonToRaster_conversion(input_temp, storm_id_field, out_r, "", "", storms.cell_size)

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

                        if storm_yr > 1918:
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
                    print(("Total Time for %d storms = " % (s_cnt), time.time() - tt))

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
                picklefile = open("C:\\Users\\Christina.Fandel\\Documents\\ArcGIS\\HH_2018\\Working\\Storms_intermediate_data%02d%04d.pickle" % (igrid, s_cnt), "wb")
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
arcpy.MosaicToNewRaster_management(storms_since_survey_fns, os.path.dirname(storms_r), os.path.basename(storms_r), storms.projection_number, "32_BIT_SIGNED", storms.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(storms_past_100yrs_fns, os.path.dirname(storms_100_r), os.path.basename(storms_100_r), storms.projection_number, "32_BIT_SIGNED", storms.cell_size, "1", "MAXIMUM", "FIRST")
#arcpy.Merge_management(storms_vp_fns, storms_vp_final)
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))

# UPDATE ME
# Reclassify Final Rasters
s_survey_max = (arcpy.sa.Raster(storms_r)).maximum + 21  # Maximum value begins at 20
s_100_max = (arcpy.sa.Raster(storms_100_r)).maximum + 51  # Maximum value begins at 50

s_survey_rc = arcpy.sa.Reclassify(storms_r, "VALUE", "0 0;1 4 1;5 10 2;11 20 3;21 %s 4;NODATA 0" % (s_survey_max), "NODATA")
s_survey_rc.save(storms_rc)
s_100_rc = arcpy.sa.Reclassify(storms_100_r, "VALUE", "0 5 0;6 10 1;11 25 2;26 50 3;51 %s 4;NODATA 0" % (s_100_max), "NODATA")
s_100_rc.save(storms_100_rc)

# Clip Final Raster to Sandy and Depth < 40 m Bounds
s_survey_clips = arcpy.sa.ExtractByMask(storms_rc, sandy_lt40m)
s_survey_clips.save(storms_f)

s_100_clips = arcpy.sa.ExtractByMask(storms_100_rc, sandy_lt40m)
s_100_clips.save(storms_100_f)

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
