# Storm Occurrence: total count and count since last survey

# Objective: Classify total storm count within each grid cell and total storm count within each grid cell since last survey. Reclassify accordingly

# TEMPORARY MUST EXAMINE DATA
# Consequence Category                             1            2                3                4                  5
# Number of Storms in last 100 yrs               0 - 10      11 - 25           26 - 75         76 - 150            150+ 
# Number of Storms Since last survey               0           1-4              5-10           11 - 20             20 +

# Save to C:\Program Files\ArcGIS\Pro\bin\Python\Lib\site-packages\ArcExt\NHSP

# This script assumes the following
    # All input layers are projected into the correct and similar coordinate system 

# Ask Barry
# Can we step through age raster? 

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
import HSTB.ArcExt.NHSP
from HSTB.ArcExt.NHSP import gridclass
arcpy.env.overwriteOutput = True

# Define Parameters # 2.5 GB
output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Storms\Storms.gdb"
#output_ws = r"H:\NHSP_2_0\Christy\FLORIDA_TEST_CASE\Storms.gdb"
#s_all = os.path.join(output_ws, 'FL_Storms_All_Sandy_LT40m_ShoreClip')
#s_all = os.path.join(output_ws, 'FL_Storms_All_Generalize10km')
#s_all = os.path.join(output_ws, 'FL_Storms_All_Generalize10km_1_378')
s_all = os.path.join(output_ws, 'FL_Storms_All_Generalize10km_379_678')
#s_all = os.path.join(output_ws, 'FL_Storms_All_Post1950_Sandy_LT40m_ShoreClip')
#s_100 = os.path.join(output_ws, 'FL_Storms_All_Post1916_LT40m_ShoreClip')
yr_path = os.path.join(output_ws, "FL_SurveyYear_500m_05042016")
yr = arcpy.Raster(yr_path)
#yr_pt_out = yr_path+"_Pts" #os.path.join(output_ws, "FL_SurveyYear_500m_05042016_Pts")
#intersect_out = os.path.join(output_ws, "FL_Storms_Year_Intersect")
#g = os.path.join(output_ws, "FL_Grid") # Necessary?
cellSize = 500
arcgrid = gridclass.grid(yr, cellSize)

## Convert survey year raster to point features and extract geometries as multipoint array (array of all point geometries)
#yr_pt = arcpy.RasterToPoint_conversion(yr, yr_pt_out, "Value") # 2.7 GB (during/after run) (1.97 quadrants) (AM 2.4 GB Quadrants)
#yr_list = [row for row in arcpy.da.SearchCursor(yr_pt,[ "SHAPE@", "grid_code"])] #4.76 GB (3.8 half) (2.6 quadrants) (during/after run) (AM 2.7 GB Quadrants)
#yr_list_geom = [row[0] for row in yr_list] #4.79 GB 
#yr_list_survey = [row[1] for row in yr_list] # 4.81 GB
##yr_multipt = arcpy.Multipoint(arcpy.Array([arcpy.Point(row.lastPoint.X, row.lastPoint.Y, 0) for row in yr_list_geom] )) #6.7 GB (4.7) (2.84 quadrants) (AM 2.9 GB Quadrants)

#yr_pt_geo = arcpy.da.SearchCursor(yr_pt,[ "SHAPE@"]) # Read geometries from year point feature, outputs individual geometries of each point 

# Convert Survey Year raster to Numpy array and change all no data to NaN
lenyr_array = numpy.rot90(arcpy.RasterToNumPyArray(yr, nodata_to_value = -99999), 3) #Currently all cells populated with year

# Read storm year from all storms file
#s_all_storm =  [row for row in arcpy.da.SearchCursor(s_all,[ "SHAPE@", "Season",])] #7.42 (with split) 6.73 quadrants (5.6 remote, but 0 free) (AM 3.6 GB, 580 MB free)

#storm, yr = s_all_storm[0]

#arcgrid.ArrayIndicesFromXY(lenyr_array)
s_100 = arcgrid.zeros()
s_survey = arcgrid.zeros()
tt = time.time()
s_cnt = 0
for storm, storm_yr in arcpy.da.SearchCursor(s_all,[ "SHAPE@", "Season",]):#storm is geometry; yr is storm year
    t = time.time()
    #affected_cells = storm.intersect(yr_multipt, 1) # 6.96 and 0 free memory (quadrants)
    ijs=arcgrid.ArrayIndicesFromXY(numpy.array([[storm.extent.XMin, storm.extent.YMin],[storm.extent.XMax, storm.extent.YMax]]))
    rmin = numpy.min(ijs[:,0])
    rmax = numpy.max(ijs[:,0])
    cmin = numpy.min(ijs[:,1])
    cmax = numpy.max(ijs[:,1])
    sliced_array = lenyr_array[rmin:rmax+1, cmin:cmax+1]# If min is less than zero, then zero; if max > max grid length, limit
    for (s_r,s_c), survey_year in numpy.ndenumerate(sliced_array):
        r=s_r+rmin # Reference sliced array back to full array 
        c=s_c+cmin
        x,y = arcgrid.XYFromIndices(numpy.array([r,c]))
        pt=arcpy.Point(float(x),float(y), 0)
        if storm.contains(pt):
            if storm_yr >= 1916:
                s_100[r,c] +=1
            if storm_yr >= survey_year:
                s_survey[r,c] +=1
    s_cnt+=1
    print(("Time for storm = ",time.time()-t))
    print(("Total Time for %d storms = "%(s_cnt),time.time()-tt))


arcgrid.ExportRaster(s_100, os.path.join(output_ws, "FL_Storms_379_678_Since1916"), rotate = True)
arcgrid.ExportRaster(s_survey, os.path.join(output_ws, "FL_Storms_379_678_SinceLastSurvey"), rotate = True)


#    if s_cnt>2:
#        break
    
#    for i, geom in enumerate(yr_list_geom):
#        if storm.contains(geom.lastPoint):
#            i,j = arcgrid.ArrayIndicesFromXY(numpy.array([geom.lastPoint.X, geom.lastPoint.Y]))
#            r, c = int(i), int(j)
#            s_100[r,c]+=1
    #arcpy.Intersect_analysis([yr_pt, storm], intersect_out)
        

# t = time.time()
# time.time() - t

## TESTING: Convert Polygon to Raster in Loop ##
import arcpy, os
output_gdb = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\STORMS\Storms.gdb"
storm_fn = "Testing_Storms"
storms = os.path.join(output_gdb, storm_fn)
out_fldr = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\STORMS\Testing"
yr_field = "Season"

for storm, storm_yr in arcpy.da.SearchCursor(storms,[ "SHAPE@", yr_field,]):#storm is geometry; yr is storm year
    t = time.time()
    i = 1
    arcpy.PolygonToRaster_conversion(storm, yr_field, storm_fn+"_raster", "", "", 500)
    print(("Finished Grid %02d at %.1f secs"%(i, time.time()-t_start)))
    i = i+1
    
    
