# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS feature

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm

# This scrip assumes the following
    # All input layers are projected into the correct and similar coordinate system (spit area if not?)
    # All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Ask Barry...
# cPickle function not available in arcGIS Pro, but scipy is. Remove cPickle
# Grid correct? Generate arbitrary (in spaced) grid based on x and y values of input and specified grid cell size -- >Fix grid cell projection (most likely need center or corner of cell and giving opposite)
# Is mirror and rotation upon import consistent? Yes. 
# Program differently for entire nation? Memory issue with grid as is

import os, numpy, matplotlib, matplotlib.pylab, scipy, scipy.signal, arcpy

output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Nav_Hazards\NAV_HAZARDS_BG.gdb"
n = os.path.join(output_ws, "FL_NavHazards_Pts")
l = os.path.join(output_ws, "FL_NavHazards_Lines")
p = os.path.join(output_ws, "FL_NavHazards_Polygons")
g = os.path.join(output_ws, "FL_Grid_Raster") 

# Identify spatial characteristics of input grid raster 
descData = arcpy.Describe(g)
cellSize = descData.meanCellHeight
extent = descData.Extent
sr = descData.spatialReference
pnt = arcpy.Point(extent.XMin, extent.YMin) # Min x and y coordinate of grid raster; used to generate raster of nav hazards

# Define extents of Navigation Hazards Histogram based on extents of grid raster
minx = extent.XMin
maxx = extent.XMax
orig_x = int(minx/cellSize)*int(cellSize) # Desired grid cell size = 500 m
x_edges = range(orig_x, int(maxx+cellSize), int(cellSize))

miny = extent.YMin
maxy = extent.YMax
orig_y = int(miny/cellSize)*int(cellSize) # Desired grid cell size = 500 m
y_edges = range(orig_y, int(maxy+int(cellSize)), int(cellSize))

# #################################### POINT FEATURES #################################### #

# Generate list and array of x and y coordinates for each Point Navigational Hazards Density Array
rows = [row[0] for row in arcpy.da.SearchCursor(n,[ "SHAPE@", "*",])]
xys= numpy.array([(row.lastPoint.X, row.lastPoint.Y) for row in rows])
# Generate histogram of Point Navigation Hazards Features based on extents of grid raster
h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [x_edges,y_edges])

# Reclassify Point Navigational Hazards Density Array  
#h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [10, 10]) # TEST
#h_den = scipy.array(h_den, scipy.int32) # TEST: Visual of h_den, can delete 

h_den_f = numpy.array(h_den)
h_den_f = scipy.array(h_den_f, scipy.int32) # TEST: Visual of h_den, can delete 
h_den_f[(h_den_f > 4)] = 5
h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
h_den_f[(h_den_f == 1)] = 3
h_den_f[(h_den_f < 1)] = 1

# Generate Point Navigational Hazards Distance Histograms
h_dist3 = scipy.signal.convolve2d(h_den, numpy.ones((3,3), scipy.int32), 'same')
h_dist5 = scipy.signal.convolve2d(h_den, numpy.ones((5,5), scipy.int32), 'same')
h_dist7 = scipy.signal.convolve2d(h_den, numpy.ones((7,7), scipy.int32), 'same')
h_dist9 = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same')

# Reclassify Point Navigational Hazards Distance Histograms
h_dist3[(h_dist3 != 0)] = 5
h_dist5[(h_dist5 != 0)] = 4
h_dist7[(h_dist7 != 0)] = 3
h_dist9[(h_dist9 != 0)] = 2

# ################################### POLYGON FEATURES ################################### #
# Generate enveloped polygon features (max bounds)
pe = arcpy.MinimumBoundingGeometry_management(p, os.path.join(output_ws, "FL_NavHazards_Polygons_Envelope"), "ENVELOPE", "NONE")

# Make point features of polygon (5 points per polygon, first and last the same)
pe_pts = arcpy.da.FeatureClassToNumPyArray(pe,["SHAPE@XY"], explode_to_points=True)
new_pts = numpy.array([pt[0] for pt in pe_pts]) # Generate array from points (if not, extra layer in data file and can not conduct vector math)
pe_reshape = new_pts.reshape((-1,5,2)) # Make each polygon a separate array within array. Specify two inputs, otherwise read as one layer (no math)
pe_grid_indices = (pe_reshape-numpy.array((orig_x, orig_y)))/numpy.array((cellSize, cellSize)) # Translating polygon enveloppe indices to grid indices

#def row_index(y): #deprecated
#    return (y-orig_y)/cell_size_y
#def col_index(x):
#    return (x-orig_x)/cell_size_x

p_den = numpy.zeros([len(x_edges),len(y_edges)]) #histogram for polygons that is the same size as the points histogram
for envelope in pe_grid_indices:
    r1,c1 = envelope[0]
    r2,c2 = envelope[2]
    minr, maxr = min((r1,r2)), max((r1,r2))
    minc, maxc = min((c1,c2)), max((c1,c2))
    p_den[minr:maxr+1, minc:maxc+1]+=1

nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_den)), pnt, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "Nav_Hazards_Polygons_Density"))

# ################################## DEFINE FINAL ARRAY ################################## #

# Calculate Max value per grid cell as a function of all arrays (density, distance3, distance 5, distance 7 and distance 9)
# then classify remaining grid cells (0) as 1
pt_risk = numpy.array((h_den_f, h_dist9, h_dist7, h_dist5, h_dist3))
pt_risk_f = numpy.max(pt_risk, 0)
ADD CODE TO MAKE ALL OTHER VALUES = 1 !!!!!!!!!!!!!!!!!!!!

# Generate, Define Projection, and Save Navigational Hazards Histogram as Raster
nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(pt_risk_f)), pnt, cellSize, cellSize)
arcpy.DefineProjection_management(nav_raster, sr)
nav_raster.save(os.path.join(output_ws, "Nav_Hazards_Classified"))


##rows_pe = [row[0] for row in arcpy.da.SearchCursor(pe,[ "SHAPE@", "*",])]
##xys_pe= numpy.array([(row.lastPoint.X, row.lastPoint.Y) for row in rows])


# #################################### PLOT FUNCTION #################################### #

# Plot histogram (if desired)
#h2 = numpy.log10(h+1)*10
#h3 = numpy.rot90(h2)
#fig = matplotlib.pylab.figure()
#plt = fig.add_subplot(111)
#plt.imshow(h3) #[:2000, :2000])
#matplotlib.pylab.show()


# ######################################## NOTES ######################################## #
'''
>>> max(xys[:,0])
1643258.8167999983
>>> min(xys[:,0])
651516.2322999984
>>> maxx=max(xys[:,0])
>>> minx=min(xys[:,0])
>>> minx/250
2606.0649291999935
>>> int(minx/500)
1303
>>> orig_x = int(minx/500)*500
>>> orig_x
651500
>>> x_edges = range(orig_x, maxx+500, 500)
Traceback (most recent call last):
  File "<interactive input>", line 1, in <module>
TypeError: range() integer end argument expected, got numpy.float64.
>>> x_edges = range(orig_x, int(maxx)+500, 500)
>>> min(x_edges), max(x_edges)
(651500, 1643500)

>>> h, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [10,10])
>>> hi = scipy.array(h, scipy.int32)
>>> hi
array([[   0,    0,    0,    0,    6,   98,   98, 1046,    2,    0],
       [   0,    0,    0,    0,    0,    2,   16,  389,    0,    0],
       [   0,    0,    0,    0,    0,    0,    5,  225,    0,    0],
       [   0,    0,    0,    0,    0,    6,  124,  130,    0,    0],
       [   0,    0,    0,    0,    0,    3,    8,   17,    0,    0],
       [   0,    0,    0,    1,   55,   17,   77,    3,    0,    0],
       [  17,    2,  136,  227,  513,   29,   97,  242,  207,   54],
       [ 739,   13,  187,  127,    0,   45,  172,   59,   80,    7],
       [ 243,   86,  175,   81,   94,   87,    1,    0,    6,    0],
       [   1,   28,   44,    0,    1,    0,    0,    0,    0,    0]])
>>> c=scipy.signal.convolve2d(h, numpy.ones((3,3)), 'same')
>>> c=scipy.signal.convolve2d(hi, numpy.ones((3,3)), 'same')
>>> c
>>> c=scipy.signal.convolve2d(hi, numpy.ones((3,3), scipy.int32), 'same')
>>> c
array([[   0,    0,    0,    6,  106,  220, 1649, 1551, 1437,    2],
       [   0,    0,    0,    6,  106,  225, 1879, 1781, 1662,    2],
       [   0,    0,    0,    0,    8,  153,  897,  889,  744,    0],
       [   0,    0,    0,    0,    9,  146,  518,  509,  372,    0],
       [   0,    0,    1,   56,   82,  290,  385,  359,  150,    0],
       [  19,  155,  366,  932,  845,  799,  493,  651,  523,  261],
       [ 771, 1094,  693, 1246, 1014, 1005,  741,  937,  652,  348],
       [1100, 1598, 1034, 1540, 1203, 1038,  732,  864,  655,  354],
       [1110, 1516,  741,  709,  435,  400,  364,  318,  152,   93],
       [ 358,  577,  414,  395,  263,  183,   88,    7,    6,    6]])
>>> pd=scipy.signal.convolve2d(hi, numpy.ones((2,2), scipy.int32), 'same')
>>> pd
array([[   0,    0,    0,    0,    6,  104,  196, 1144, 1048,    2],
       [   0,    0,    0,    0,    6,  106,  214, 1549, 1437,    2],
       [   0,    0,    0,    0,    0,    2,   23,  635,  614,    0],
       [   0,    0,    0,    0,    0,    6,  135,  484,  355,    0],
       [   0,    0,    0,    0,    0,    9,  141,  279,  147,    0],
       [   0,    0,    0,    1,   56,   75,  105,  105,   20,    0],
       [  17,   19,  138,  364,  796,  614,  220,  419,  452,  261],
       [ 756,  771,  338,  677,  867,  587,  343,  570,  588,  348],
       [ 982, 1081,  461,  570,  302,  226,  305,  232,  145,   93],
       [ 244,  358,  333,  300,  176,  182,   88,    1,    6,    6]])
>>> pd+c
array([[   0,    0,    0,    6,  112,  324, 1845, 2695, 2485,    4],
       [   0,    0,    0,    6,  112,  331, 2093, 3330, 3099,    4],
       [   0,    0,    0,    0,    8,  155,  920, 1524, 1358,    0],
       [   0,    0,    0,    0,    9,  152,  653,  993,  727,    0],
       [   0,    0,    1,   56,   82,  299,  526,  638,  297,    0],
       [  19,  155,  366,  933,  901,  874,  598,  756,  543,  261],
       [ 788, 1113,  831, 1610, 1810, 1619,  961, 1356, 1104,  609],
       [1856, 2369, 1372, 2217, 2070, 1625, 1075, 1434, 1243,  702],
       [2092, 2597, 1202, 1279,  737,  626,  669,  550,  297,  186],
       [ 602,  935,  747,  695,  439,  365,  176,    8,   12,   12]])
'''