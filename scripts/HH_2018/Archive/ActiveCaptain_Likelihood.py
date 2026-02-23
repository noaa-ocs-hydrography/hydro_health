# Density and Proximity to Active Captain Reports

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of active captain report. Note, active captain reports are being used as a proxy for recreational boater traffic. The proxy for
# recreational boater traffic is deduced from the active captain reports using the comment count (comment count + 1, since comment count = 0 if one report). Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of AC Reports w/i 5 nm       0            1               2-10               11-100             100+
# Distance to Hazard                >2 km       1 - 2 km        0.5 - 1 km      0.5 - 0.25 km       < 0.25 km

# This script assumes the following
    # All input layers are projected into the correct and similar coordinate system 
    # All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
import matplotlib, matplotlib.pylab
import skimage, skimage.draw

# Define Parameters
#output_ws = arcpy.GetParameter(0)
output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\ActiveCaptain\ActiveCaptain.gdb"
#n = arcpy.GetParameter(1)
n = os.path.join(output_ws, 'FL_ActiveCaptain_Raw')
#g = arcpy.GetParameter(2)
g = os.path.join(output_ws, 'FL_Grid')
#cellSize = arcpy.GetParameter(3)
cellSize = 500
n_out = os.path.join(output_ws, 'FL_ActiveCaptain_Points_Max')

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
x_edges = list(range(orig_x, int(maxx+cellSize), int(cellSize)))

miny = extent.YMin
maxy = extent.YMax
orig_y = int(miny/cellSize)*int(cellSize) # Desired grid cell size = 500 m
y_edges = list(range(orig_y, int(maxy+int(cellSize)), int(cellSize)))
orig_new = arcpy.Point(orig_x, orig_y)

# #################################### POINT FEATURES #################################### #

# Generate list and array of x and y coordinates for each Point Groundings 
rows = [row for row in arcpy.da.SearchCursor(n,[ "SHAPE@", "TrafficProxy",])]
xys_list=[]
dummy = [xys_list.extend([(row[0].lastPoint.X, row[0].lastPoint.Y) for i in range(row[1])]) for row in rows]
xys = numpy.array(xys_list)
# Generate histogram of Point Active Captain Features based on extents of grid raster
h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [x_edges,y_edges])

##alternatively can find the indices and increment the matrix at it's row/column
#h2_den = numpy.zeros()
#for row in rows:
#    r,c = row.lastPoint.X/whatever, row.lastPoint.Y/other
#    h2_den[r,c]+=row[1] #add the occurances

# Export Point Active Captain Density Array
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_den)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_AC_Points_ClassifiedDensity"))

# Reclassify Point Active Captain Density Array  
h_den_f = scipy.signal.convolve2d(h_den, numpy.ones((39,39), scipy.int32), 'same') # Count of point occurrences within 5 nm (~9.5 km) of grid cell 
#h_den_f = scipy.array(h_den_f, scipy.int32) # TEST: Visual of h_den, can delete 
h_den_f[(h_den_f > 100)] = 5
h_den_f[(h_den_f >= 11) & (h_den_f <= 100)] = 4
h_den_f[(h_den_f >= 2) & (h_den_f <= 10)] = 3
h_den_f[(h_den_f == 1)] = 2
h_den_f[(h_den_f < 1)] = 1

# Export Point Active Captain Density Array
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_den_f)), orig_new, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "FL_AC_Points_ClassifiedDensity39"))


