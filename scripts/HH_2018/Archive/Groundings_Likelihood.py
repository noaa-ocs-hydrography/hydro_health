# Density and Proximity to Groundings
# Groundings are defined as a grounding incident by a non-recreational or fishing vessel. 

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of grounding. Classify risk using the following table

# Risk Category                       1            2                3                4                  5
# Number of Groundings w/i 2 nm       0            0                1               2-4                 4+
# Distance to Hazard                >2 km       1 - 2 km        0.5 - 1 km      0.5 - 0.25 km       < 0.25 km

# This script assumes the following
    # All input layers are projected into the correct and similar coordinate system 
    # All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
import matplotlib, matplotlib.pylab
import skimage, skimage.draw
from HSTB.ArcExt.NHSP import gridclass

# Define Parameters
#output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Groundings\Groundings.gdb"
output_ws = r"H:\Christy\Groundings\Groundings.gdb"
#n = os.path.join(output_ws, 'FL_Groundings_NoRecFish')
n = os.path.join(output_ws, "Groundings_VECTOR_PROCESSED")

#g = gridclass.datagrid.FromRaster(r'H:\GRID\Grid.gdb\Grid_Model_Extents_01_Raster')



#g = os.path.join(output_ws, 'FL_Grid')
#cellSize = 500
#n_out = os.path.join(output_ws, 'FL_Groundings_Points_Max')

'''
# Identify spatial characteristics of input grid raster 
descData = arcpy.Describe(g)
#cellSize = descData.meanCellHeight
extent = descData.Extent
sr = descData.spatialReference
pnt = arcpy.Point(extent.XMin, extent.YMin) # Min x and y coordinate of grid raster; used to generate raster of Groundings
# Define extents of Groundings Histogram based on extents of grid raster
# Note: If Must keep origin at same grid cell node, set orig_x and orig_y as min coords of grid (bottom left)
minx = extent.XMin
maxx = extent.XMax
orig_x = int(minx/cellSize)*int(cellSize) # Desired grid cell size = 500 m
x_edges = range(orig_x, int(maxx+cellSize), int(cellSize))

miny = extent.YMin
maxy = extent.YMax
orig_y = int(miny/cellSize)*int(cellSize) # Desired grid cell size = 500 m
y_edges = range(orig_y, int(maxy+int(cellSize)), int(cellSize))
orig_new = arcpy.Point(orig_x, orig_y)
'''
# #################################### POINT FEATURES #################################### #

# Generate list and array of x and y coordinates for each Point Groundings 
#rows = [row[0] for row in arcpy.da.SearchCursor(n,[ "SHAPE@", "*",])]
#xys= numpy.array([(row.lastPoint.X, row.lastPoint.Y) for row in rows])
for igrid in range(1,14): #better to find all matching grid extents in the geodatabase
    lyr_name = GDB+"Grid_Model_Extents_%02d_Raster"%igrid
    grid = gridclass.datagrid.FromRaster(lyr_name)
    records = arcpy.da.FeatureClassToNumPyArray(n,["SHAPE@XY",], spatial_reference=grid.sr) #, explode_to_points=True)
    xys = records["SHAPE@XY"]
    # Generate histogram of Point Groundings Features based on extents of grid raster
    h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [grid.x_edges, grid.y_edges])
    h_den_f = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
    # Reclassify Numpy Array 
    h_den_f[(h_den_f > 4)] = 5
    h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
    h_den_f[(h_den_f == 1)] = 2
    h_den_f[(h_den_f < 1)] = 1   
    grid.ExportMatchingRaster(h_den_f, os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDensity9"))
# Reclassify Point Groundings Density Array  
#h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [10, 10]) # TEST
#h_den = scipy.array(h_den, scipy.int32) # TEST: Visual of h_den, can delete 
#h_den_f = numpy.array(h_den) # count of point occurrences in each cell

h_den_f = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
#h_den_f = scipy.array(h_den_f, scipy.int32) # TEST: Visual of h_den, can delete 
h_den_f[(h_den_f > 4)] = 5
h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
h_den_f[(h_den_f == 1)] = 2
h_den_f[(h_den_f < 1)] = 1

# Export Point Groundings Density Array
grid.ExportMatchingRaster(h_den_f, os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDensity9"))
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_den_f)), grid.origin, grid.cell_size_x, grid.cell_size_y)
#arcpy.DefineProjection_management(nav_raster, grid.sr)
#nav_raster.save(os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDensity9"))

# Generate Point Groundings Distance Histograms
h_dist3 = scipy.signal.convolve2d(h_den, numpy.ones((3,3), scipy.int32), 'same')
h_dist5 = scipy.signal.convolve2d(h_den, numpy.ones((5,5), scipy.int32), 'same')
h_dist7 = scipy.signal.convolve2d(h_den, numpy.ones((7,7), scipy.int32), 'same')
h_dist9 = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same')

# Reclassify Point Groundings Distance Histograms
h_dist3[(h_dist3 != 0)] = 5
h_dist5[(h_dist5 != 0)] = 4
h_dist7[(h_dist7 != 0)] = 3
h_dist9[(h_dist9 != 0)] = 2

# Export all distance point navigational hazard histograms
# Export Point Groundings Distance Array (3,3 = 500 m)
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist3)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDistance3"))

# Export Point Groundings Distance Array (5,5 = 1000 m)
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist5)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDistance5"))

# Export Point Groundings Distance Array (7,7 = 1500 m)
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist7)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDistance7"))

# Export Point Groundings Distance Array (9,9 = 2000 m)
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist9)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDistance9"))

# Export Max Point Groundings Distance Array
h_dist_max = numpy.array(( h_dist9, h_dist7, h_dist5, h_dist3))
h_dist_max_f =numpy.max(h_dist_max, 0)
grid.ExportMatchingRaster(h_den_f, os.path.join(output_ws, "FL_Groundings_Points_ClassifiedDensity9"))

# ################################## DEFINE FINAL ARRAY ################################## #

# Calculate Max value per grid cell as a function of all arrays (density, distance3, distance 5, distance 7 and distance 9) for each geometry type
pt_risk = numpy.array((h_den_f, h_dist9, h_dist7, h_dist5, h_dist3))
pt_risk_f = numpy.max(pt_risk, 0)

# Export max value raster per geometry type
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(pt_risk_f)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(n_out)
del grid #make sure the first array is destroyed before the second is loaded

###############################################################################################

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
import matplotlib, matplotlib.pylab
import skimage, skimage.draw
from HSTB.ArcExt.NHSP import gridclass

# Define Parameters
ground = r'H:\Christy\Groundings\Groundings.gdb\Groundings_VECTOR_PROCESSED'
grid_ws = r'H:\GRID\Grid.gdb'
out_fldr = r'H:\Christy\Groundings'
#n = os.path.join(output_ws, "Groundings_VECTOR_PROCESSED")

for igrid in range(1,14): #better to find all matching grid extents in the geodatabase
    grid_name = "Grid_Model_Extents_%02d_Raster"%igrid
    grid_path = os.path.join(grid_ws, grid_name)
    dens_out = os.path.join(out_fldr, grid_name+"_Density")
      
    grid = gridclass.datagrid.FromRaster(grid_path)
    records = arcpy.da.FeatureClassToNumPyArray(ground,["SHAPE@XY",], spatial_reference=grid.sr) #, explode_to_points=True)
    xys = records["SHAPE@XY"]
    
    # Generate histogram of point groundings features based on extents of grid raster
    h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [grid.x_edges, grid.y_edges])
    h_den_f = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
    
    # Reclassify Numpy Array based on density of groundings
    h_den_f[(h_den_f > 4)] = 5
    h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
    h_den_f[(h_den_f == 1)] = 2
    h_den_f[(h_den_f < 1)] = 1   
    grid.ExportMatchingRaster(h_den_f, dens_out)
    
    # Generate Point Groundings Distance Histograms
    h_dist3 = scipy.signal.convolve2d(h_den, numpy.ones((3,3), scipy.int32), 'same')
    h_dist5 = scipy.signal.convolve2d(h_den, numpy.ones((5,5), scipy.int32), 'same')
    h_dist7 = scipy.signal.convolve2d(h_den, numpy.ones((7,7), scipy.int32), 'same')
    h_dist9 = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same')

    # Reclassify Point Groundings Distance Histograms
    h_dist3[(h_dist3 != 0)] = 5
    h_dist5[(h_dist5 != 0)] = 4
    h_dist7[(h_dist7 != 0)] = 3
    h_dist9[(h_dist9 != 0)] = 2